#ifndef PDMRG_GPU_IMPL_H
#define PDMRG_GPU_IMPL_H

#include <rocsolver/rocsolver.h>
#include "accurate_svd.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <chrono>

#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "HIP error in " << __FILE__ << ":" << __LINE__ \
                      << " - " << hipGetErrorString(err) << std::endl; \
            throw std::runtime_error("HIP error"); \
        } \
    } while(0)

#define ROCBLAS_CHECK(call) \
    do { \
        rocblas_status status = call; \
        if (status != rocblas_status_success) { \
            std::cerr << "rocBLAS error in " << __FILE__ << ":" << __LINE__ \
                      << " - status " << status << std::endl; \
            throw std::runtime_error("rocBLAS error"); \
        } \
    } while(0)

// ============================================================================
// Constructor
// ============================================================================

template<typename Scalar>
PDMRGGPU<Scalar>::PDMRGGPU(int L, int d, int chi_max, int D_mpo, int n_segments, double tol)
    : L_(L), d_(d), chi_max_(chi_max), D_mpo_(D_mpo), tol_(tol), energy_(0.0),
      n_segments_(n_segments) {

    if (L < 2 * n_segments) {
        throw std::runtime_error("Need at least 2 sites per segment: L >= 2*n_segments");
    }

    // Bond dimensions (min-cut formula capped at chi_max)
    bond_dims_.resize(L + 1);
    bond_dims_[0] = 1;
    bond_dims_[L] = 1;
    for (int i = 1; i < L; i++) {
        double exact_dim = pow((double)d_, std::min(i, L - i));
        bond_dims_[i] = (exact_dim > chi_max_) ? chi_max_ : (int)exact_dim;
    }

    // Partition chain
    partition_chain();

    // Create streams and rocBLAS handles (one per segment)
    streams_.resize(n_segments_);
    handles_.resize(n_segments_);
    for (int k = 0; k < n_segments_; k++) {
        HIP_CHECK(hipStreamCreate(&streams_[k]));
        ROCBLAS_CHECK(rocblas_create_handle(&handles_[k]));
        ROCBLAS_CHECK(rocblas_set_stream(handles_[k], streams_[k]));
    }

    int dd = d_ * d_;

    // MPS tensors
    d_mps_tensors_.resize(L, nullptr);
    for (int i = 0; i < L; i++) {
        allocate_mps_tensor(i, chi_L(i), chi_R(i));
    }

    // MPO tensors
    d_mpo_tensors_.resize(L, nullptr);

    // W matrices for single-site env updates
    d_W_left_.resize(L, nullptr);
    d_W_right_.resize(L, nullptr);

    // Fused two-site MPO
    d_WW_.resize(L - 1, nullptr);

    // Environments
    d_L_envs_.resize(L + 1, nullptr);
    d_R_envs_.resize(L + 1, nullptr);
    L_env_alloc_chi_.resize(L + 1, 0);
    R_env_alloc_chi_.resize(L + 1, 0);

    for (int i = 0; i <= L; i++) {
        int chi_alloc = (i == 0 || i == L) ? 1 : chi_max_;
        int sz = chi_alloc * D_mpo_ * chi_alloc;
        HIP_CHECK(hipMalloc(&d_L_envs_[i], sz * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&d_R_envs_[i], sz * sizeof(Scalar)));
        HIP_CHECK(hipMemset(d_L_envs_[i], 0, sz * sizeof(Scalar)));
        HIP_CHECK(hipMemset(d_R_envs_[i], 0, sz * sizeof(Scalar)));
        L_env_alloc_chi_[i] = chi_alloc;
        R_env_alloc_chi_[i] = chi_alloc;
    }

    // Boundary V-matrices
    int n_boundaries = n_segments_ - 1;
    d_V_boundary_.resize(n_boundaries, nullptr);
    h_V_boundary_.resize(n_boundaries);
    for (int k = 0; k < n_boundaries; k++) {
        HIP_CHECK(hipMalloc(&d_V_boundary_[k], chi_max_ * sizeof(double)));
        h_V_boundary_[k].resize(chi_max_, 1.0);
    }

    // Allocate per-stream workspaces
    theta_size_max_ = chi_max_ * dd * chi_max_;
    max_lanczos_iter_ = std::min(100, theta_size_max_);
    use_cpu_svd_ = true;

    allocate_stream_workspaces();
}

// ============================================================================
// Per-stream workspace allocation
// ============================================================================

template<typename Scalar>
void PDMRGGPU<Scalar>::allocate_stream_workspaces() {
    int dd = d_ * d_;
    int t_max = D_mpo_ * dd * chi_max_ * chi_max_;
    int batch_max = D_mpo_ * dd;
    int svd_max_m = chi_max_ * d_;
    int svd_max_n = d_ * chi_max_;
    int svd_max_k = std::min(svd_max_m, svd_max_n);

    workspaces_.resize(n_segments_);

    for (int k = 0; k < n_segments_; k++) {
        auto& ws = workspaces_[k];

        // Contraction intermediates
        HIP_CHECK(hipMalloc(&ws.d_T1, t_max * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&ws.d_T2, t_max * sizeof(Scalar)));

        // Lanczos workspace
        HIP_CHECK(hipMalloc(&ws.d_theta, theta_size_max_ * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&ws.d_heff_result, theta_size_max_ * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&ws.d_lanczos_v, (size_t)max_lanczos_iter_ * theta_size_max_ * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&ws.d_ritz_coeffs, max_lanczos_iter_ * sizeof(Scalar)));

        // Batched GEMM pointer arrays
        HIP_CHECK(hipMalloc(&ws.d_batch_A, batch_max * sizeof(Scalar*)));
        HIP_CHECK(hipMalloc(&ws.d_batch_B, batch_max * sizeof(Scalar*)));
        HIP_CHECK(hipMalloc(&ws.d_batch_C, batch_max * sizeof(Scalar*)));

        // GPU SVD workspace
        HIP_CHECK(hipMalloc(&ws.d_svd_A, theta_size_max_ * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&ws.d_svd_U, (size_t)svd_max_m * svd_max_k * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&ws.d_svd_S, svd_max_k * sizeof(RealType)));
        HIP_CHECK(hipMalloc(&ws.d_svd_Vh, (size_t)svd_max_k * svd_max_n * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&ws.d_svd_E, svd_max_k * sizeof(RealType)));
        HIP_CHECK(hipMalloc(&ws.d_svd_info, sizeof(int)));
        HIP_CHECK(hipMalloc(&ws.d_svd_work, theta_size_max_ * sizeof(Scalar)));

        // CPU SVD workspace
        ws.h_svd_A.resize(theta_size_max_);
        ws.h_svd_U.resize((size_t)svd_max_m * svd_max_k);
        ws.h_svd_S.resize(svd_max_k);
        ws.h_svd_Vh.resize((size_t)svd_max_k * svd_max_n);
        ws.h_svd_tmp.resize(std::max((size_t)svd_max_m * svd_max_k, (size_t)svd_max_k * svd_max_n));
        ws.h_svd_rwork.resize(Traits::svd_rwork_size(svd_max_m, svd_max_n));

        // Query optimal LAPACK workspace
        {
            int m = svd_max_m, n = svd_max_n;
            int lwork_query = -1;
            Scalar work_opt;
            int info;
            const char jobu = 'S', jobvt = 'S';
            Traits::lapack_gesvd(&jobu, &jobvt, &m, &n, nullptr, &m, nullptr,
                    nullptr, &m, nullptr, &svd_max_k, &work_opt, &lwork_query,
                    ws.h_svd_rwork.empty() ? nullptr : ws.h_svd_rwork.data(), &info);
            int opt_size;
            if constexpr (Traits::is_complex) {
                opt_size = (int)Traits::real_part(work_opt) + 1;
            } else {
                opt_size = (int)work_opt + 1;
            }
            ws.h_svd_work.resize(opt_size);
        }
    }
}

// ============================================================================
// Destructor
// ============================================================================

template<typename Scalar>
PDMRGGPU<Scalar>::~PDMRGGPU() {
    free_gpu_resources();
}

template<typename Scalar>
void PDMRGGPU<Scalar>::free_gpu_resources() {
    for (auto ptr : d_mps_tensors_) if (ptr) hipFree(ptr);
    for (auto ptr : d_mpo_tensors_) if (ptr) hipFree(ptr);
    for (auto ptr : d_W_left_) if (ptr) hipFree(ptr);
    for (auto ptr : d_W_right_) if (ptr) hipFree(ptr);
    for (auto ptr : d_WW_) if (ptr) hipFree(ptr);
    for (auto ptr : d_L_envs_) if (ptr) hipFree(ptr);
    for (auto ptr : d_R_envs_) if (ptr) hipFree(ptr);
    for (auto ptr : d_V_boundary_) if (ptr) hipFree(ptr);

    for (auto& ws : workspaces_) {
        if (ws.d_theta) hipFree(ws.d_theta);
        if (ws.d_heff_result) hipFree(ws.d_heff_result);
        if (ws.d_T1) hipFree(ws.d_T1);
        if (ws.d_T2) hipFree(ws.d_T2);
        if (ws.d_lanczos_v) hipFree(ws.d_lanczos_v);
        if (ws.d_ritz_coeffs) hipFree(ws.d_ritz_coeffs);
        if (ws.d_batch_A) hipFree(ws.d_batch_A);
        if (ws.d_batch_B) hipFree(ws.d_batch_B);
        if (ws.d_batch_C) hipFree(ws.d_batch_C);
        if (ws.d_svd_A) hipFree(ws.d_svd_A);
        if (ws.d_svd_U) hipFree(ws.d_svd_U);
        if (ws.d_svd_S) hipFree(ws.d_svd_S);
        if (ws.d_svd_Vh) hipFree(ws.d_svd_Vh);
        if (ws.d_svd_E) hipFree(ws.d_svd_E);
        if (ws.d_svd_info) hipFree(ws.d_svd_info);
        if (ws.d_svd_work) hipFree(ws.d_svd_work);
    }

    for (auto& h : handles_) rocblas_destroy_handle(h);
    for (auto& s : streams_) hipStreamDestroy(s);
}

// ============================================================================
// Memory management
// ============================================================================

template<typename Scalar>
void PDMRGGPU<Scalar>::allocate_mps_tensor(int site, int cL, int cR) {
    if (d_mps_tensors_[site]) HIP_CHECK(hipFree(d_mps_tensors_[site]));
    HIP_CHECK(hipMalloc(&d_mps_tensors_[site], cL * d_ * cR * sizeof(Scalar)));
}

template<typename Scalar>
void PDMRGGPU<Scalar>::ensure_L_env_alloc(int idx, int chi) {
    if (chi > L_env_alloc_chi_[idx]) {
        if (d_L_envs_[idx]) HIP_CHECK(hipFree(d_L_envs_[idx]));
        int sz = chi * D_mpo_ * chi;
        HIP_CHECK(hipMalloc(&d_L_envs_[idx], sz * sizeof(Scalar)));
        L_env_alloc_chi_[idx] = chi;
    }
}

template<typename Scalar>
void PDMRGGPU<Scalar>::ensure_R_env_alloc(int idx, int chi) {
    if (chi > R_env_alloc_chi_[idx]) {
        if (d_R_envs_[idx]) HIP_CHECK(hipFree(d_R_envs_[idx]));
        int sz = chi * D_mpo_ * chi;
        HIP_CHECK(hipMalloc(&d_R_envs_[idx], sz * sizeof(Scalar)));
        R_env_alloc_chi_[idx] = chi;
    }
}

// ============================================================================
// Chain partitioning
// ============================================================================

template<typename Scalar>
void PDMRGGPU<Scalar>::partition_chain() {
    seg_first_.resize(n_segments_);
    seg_last_.resize(n_segments_);
    boundary_bonds_.resize(n_segments_ - 1);

    int base = L_ / n_segments_;
    int remainder = L_ % n_segments_;

    int pos = 0;
    for (int k = 0; k < n_segments_; k++) {
        seg_first_[k] = pos;
        int seg_len = base + (k < remainder ? 1 : 0);
        seg_last_[k] = pos + seg_len - 1;
        pos += seg_len;
    }

    for (int k = 0; k < n_segments_ - 1; k++) {
        boundary_bonds_[k] = seg_last_[k];
    }
}

// ============================================================================
// MPS initialization
// ============================================================================

template<typename Scalar>
void PDMRGGPU<Scalar>::initialize_mps_random(double scale) {
    for (int i = 0; i < L_; i++) {
        int size = chi_L(i) * d_ * chi_R(i);
        std::vector<Scalar> h_A(size);
        for (int j = 0; j < size; j++) {
            h_A[j] = Traits::scale_by_real(scale, Traits::random_val());
        }
        HIP_CHECK(hipMemcpy(d_mps_tensors_[i], h_A.data(),
                            size * sizeof(Scalar), hipMemcpyHostToDevice));
    }
}

// ============================================================================
// MPO setup and fused two-site MPO precomputation
// ============================================================================

template<typename Scalar>
void PDMRGGPU<Scalar>::set_mpo(const std::vector<Scalar*>& h_mpo_tensors) {
    int D = D_mpo_, d = d_;
    for (int i = 0; i < L_; i++) {
        int size = D * d * d * D;
        HIP_CHECK(hipMalloc(&d_mpo_tensors_[i], size * sizeof(Scalar)));
        HIP_CHECK(hipMemcpy(d_mpo_tensors_[i], h_mpo_tensors[i],
                            size * sizeof(Scalar), hipMemcpyHostToDevice));

        int wm_size = D * d * d * D;
        std::vector<Scalar> h_WL(wm_size, Traits::zero());
        std::vector<Scalar> h_WR(wm_size, Traits::zero());
        for (int w = 0; w < D; w++)
            for (int s = 0; s < d; s++)
                for (int sp = 0; sp < d; sp++)
                    for (int wp = 0; wp < D; wp++) {
                        Scalar val = h_mpo_tensors[i][w + s*D + sp*D*d + wp*D*d*d];
                        h_WL[(w*d+s) + (wp*d+sp) * D * d] = val;
                        h_WR[(wp*d+s) + (w*d+sp) * D * d] = val;
                    }
        HIP_CHECK(hipMalloc(&d_W_left_[i], wm_size * sizeof(Scalar)));
        HIP_CHECK(hipMemcpy(d_W_left_[i], h_WL.data(),
                            wm_size * sizeof(Scalar), hipMemcpyHostToDevice));
        HIP_CHECK(hipMalloc(&d_W_right_[i], wm_size * sizeof(Scalar)));
        HIP_CHECK(hipMemcpy(d_W_right_[i], h_WR.data(),
                            wm_size * sizeof(Scalar), hipMemcpyHostToDevice));
    }

    precompute_fused_mpo(h_mpo_tensors);
}

template<typename Scalar>
void PDMRGGPU<Scalar>::precompute_fused_mpo(const std::vector<Scalar*>& h_mpo_tensors) {
    int D = D_mpo_, d = d_;
    int dd = d * d;

    for (int bond = 0; bond < L_ - 1; bond++) {
        int ww_size = D * dd * dd * D;
        std::vector<Scalar> h_WW(ww_size, Traits::zero());

        const Scalar* WL = h_mpo_tensors[bond];
        const Scalar* WR = h_mpo_tensors[bond + 1];

        for (int w = 0; w < D; w++)
            for (int n = 0; n < D; n++)
                for (int s1 = 0; s1 < d; s1++)
                    for (int s2 = 0; s2 < d; s2++)
                        for (int s1p = 0; s1p < d; s1p++)
                            for (int s2p = 0; s2p < d; s2p++) {
                                Scalar val = Traits::zero();
                                for (int m = 0; m < D; m++) {
                                    Scalar wl = WL[w + s1*D + s1p*D*d + m*D*d*d];
                                    Scalar wr = WR[m + s2*D + s2p*D*d + n*D*d*d];
                                    if constexpr (Traits::is_complex) {
                                        val = hipCadd(val, hipCmul(wl, wr));
                                    } else {
                                        val += wl * wr;
                                    }
                                }
                                int row = w * dd + s1 * d + s2;
                                int col = n * dd + s1p * d + s2p;
                                h_WW[row + col * D * dd] = val;
                            }

        HIP_CHECK(hipMalloc(&d_WW_[bond], ww_size * sizeof(Scalar)));
        HIP_CHECK(hipMemcpy(d_WW_[bond], h_WW.data(),
                            ww_size * sizeof(Scalar), hipMemcpyHostToDevice));
    }
}

// ============================================================================
// Two-site theta formation (stream-aware)
// ============================================================================

template<typename Scalar>
void PDMRGGPU<Scalar>::form_theta_two_site(int site, int si) {
    int cL = chi_L(site);
    int chi_mid = bond_dims_[site + 1];
    int cR = chi_R(site + 1);
    Scalar one = Traits::one(), zero_val = Traits::zero();
    auto& ws = workspaces_[si];

    ROCBLAS_CHECK(Traits::gemm(handles_[si],
        rocblas_operation_none, rocblas_operation_none,
        cL * d_, d_ * cR, chi_mid,
        &one,
        d_mps_tensors_[site], cL * d_,
        d_mps_tensors_[site + 1], chi_mid,
        &zero_val,
        ws.d_theta, cL * d_));
}

// ============================================================================
// Two-site H_eff application (3-step with fused WW, stream-aware)
// ============================================================================

template<typename Scalar>
void PDMRGGPU<Scalar>::apply_heff_two_site(int site, const Scalar* d_theta_in,
                                            Scalar* d_result, int si) {
    int cL = chi_L(site);
    int cR = chi_R(site + 1);
    int D = D_mpo_, d = d_;
    int dd = d * d;
    Scalar one = Traits::one(), zero_val = Traits::zero();
    auto& ws = workspaces_[si];

    Scalar* L_env = d_L_envs_[site];
    Scalar* R_env = d_R_envs_[site + 2];
    Scalar* WW = d_WW_[site];
    Scalar* T1 = ws.d_T1;
    Scalar* T2 = ws.d_T2;

    // Step 1: Batched GEMM — L_env^T × theta
    {
        int batch_count = D * dd;
        std::vector<Scalar*> h_A(batch_count), h_B(batch_count), h_C(batch_count);
        for (int w = 0; w < D; w++)
            for (int s1 = 0; s1 < d; s1++)
                for (int s2 = 0; s2 < d; s2++) {
                    int ws_idx = w * dd + s1 * d + s2;
                    h_A[ws_idx] = L_env + w * cL;
                    h_B[ws_idx] = const_cast<Scalar*>(d_theta_in) + s1 * cL + s2 * cL * d;
                    h_C[ws_idx] = T1 + ws_idx * cL * cR;
                }
        HIP_CHECK(hipMemcpyAsync(ws.d_batch_A, h_A.data(), batch_count*sizeof(Scalar*), hipMemcpyHostToDevice, streams_[si]));
        HIP_CHECK(hipMemcpyAsync(ws.d_batch_B, h_B.data(), batch_count*sizeof(Scalar*), hipMemcpyHostToDevice, streams_[si]));
        HIP_CHECK(hipMemcpyAsync(ws.d_batch_C, h_C.data(), batch_count*sizeof(Scalar*), hipMemcpyHostToDevice, streams_[si]));
        ROCBLAS_CHECK(Traits::gemm_batched(handles_[si],
            Traits::op_t, rocblas_operation_none,
            cL, cR, cL,
            &one,
            (const Scalar**)ws.d_batch_A, cL * D,
            (const Scalar**)ws.d_batch_B, cL * dd,
            &zero_val,
            ws.d_batch_C, cL,
            batch_count));
    }

    // Step 2: Dense GEMM — T1 × WW
    ROCBLAS_CHECK(Traits::gemm(handles_[si],
        rocblas_operation_none, rocblas_operation_none,
        cL * cR, dd * D, D * dd,
        &one,
        T1, cL * cR,
        WW, D * dd,
        &zero_val,
        T2, cL * cR));

    // Step 3: Loop of GEMMs — T2 × R_env
    for (int s1p = 0; s1p < d; s1p++) {
        for (int s2p = 0; s2p < d; s2p++) {
            for (int n = 0; n < D; n++) {
                Scalar beta = (n == 0) ? Traits::zero() : Traits::one();
                int ws_out = n * dd + s1p * d + s2p;
                ROCBLAS_CHECK(Traits::gemm(handles_[si],
                    rocblas_operation_none, rocblas_operation_none,
                    cL, cR, cR,
                    &one,
                    T2 + ws_out * cL * cR, cL,
                    R_env + n * cR, cR * D,
                    &beta,
                    d_result + s1p * cL + s2p * cL * d, cL * dd));
            }
        }
    }
}

// ============================================================================
// Left environment update (stream-aware)
// ============================================================================

template<typename Scalar>
void PDMRGGPU<Scalar>::update_left_env(int site, int si) {
    int chi_in = bond_dims_[site];
    int chi_out = bond_dims_[site + 1];
    int D = D_mpo_, d = d_;
    Scalar one = Traits::one(), zero_val = Traits::zero();
    auto& ws = workspaces_[si];

    ensure_L_env_alloc(site + 1, chi_out);

    Scalar* L_env = d_L_envs_[site];
    Scalar* A = d_mps_tensors_[site];
    Scalar* W_mat = d_W_left_[site];
    Scalar* L_new = d_L_envs_[site + 1];
    Scalar* V = ws.d_T1;
    Scalar* U = ws.d_T2;

    // Step 1: V_ws[a',b] = L_w^T[a',a] * A_s[a,b]  (batched GEMM)
    {
        std::vector<Scalar*> h_A(D * d), h_B(D * d), h_C(D * d);
        for (int w = 0; w < D; w++)
            for (int s = 0; s < d; s++) {
                int ws_idx = w * d + s;
                h_A[ws_idx] = L_env + w * chi_in;
                h_B[ws_idx] = A + s * chi_in;
                h_C[ws_idx] = V + ws_idx * chi_in * chi_out;
            }
        HIP_CHECK(hipMemcpyAsync(ws.d_batch_A, h_A.data(), D*d*sizeof(Scalar*), hipMemcpyHostToDevice, streams_[si]));
        HIP_CHECK(hipMemcpyAsync(ws.d_batch_B, h_B.data(), D*d*sizeof(Scalar*), hipMemcpyHostToDevice, streams_[si]));
        HIP_CHECK(hipMemcpyAsync(ws.d_batch_C, h_C.data(), D*d*sizeof(Scalar*), hipMemcpyHostToDevice, streams_[si]));
        ROCBLAS_CHECK(Traits::gemm_batched(handles_[si],
            Traits::op_t, rocblas_operation_none,
            chi_in, chi_out, chi_in,
            &one,
            (const Scalar**)ws.d_batch_A, chi_in * D,
            (const Scalar**)ws.d_batch_B, chi_in * d,
            &zero_val,
            ws.d_batch_C, chi_in,
            D * d));
    }

    // Step 2: U = V * W_matrix
    ROCBLAS_CHECK(Traits::gemm(handles_[si],
        rocblas_operation_none, rocblas_operation_none,
        chi_in * chi_out, d * D, D * d,
        &one,
        V, chi_in * chi_out,
        W_mat, D * d,
        &zero_val,
        U, chi_in * chi_out));

    // Step 3: L_new_w'[b,b'] = sum_{a',s'} conj(U[a',ws',b])^H * A[a',s',b']
    for (int wp = 0; wp < D; wp++) {
        for (int sp = 0; sp < d; sp++) {
            Scalar beta = (sp == 0) ? Traits::zero() : Traits::one();
            int ws_out = wp * d + sp;
            ROCBLAS_CHECK(Traits::gemm(handles_[si],
                Traits::op_h, rocblas_operation_none,
                chi_out, chi_out, chi_in,
                &one,
                U + ws_out * chi_in * chi_out, chi_in,
                A + sp * chi_in, chi_in * d,
                &beta,
                L_new + wp * chi_out, chi_out * D));
        }
    }

    if constexpr (Traits::is_complex) {
        conjugate_inplace(L_new, chi_out * D * chi_out, streams_[si]);
    }
}

// ============================================================================
// Right environment update (stream-aware)
// ============================================================================

template<typename Scalar>
void PDMRGGPU<Scalar>::update_right_env(int site, int si) {
    int chi_in = bond_dims_[site + 1];
    int chi_out = bond_dims_[site];
    int D = D_mpo_, d = d_;
    Scalar one = Traits::one(), zero_val = Traits::zero();
    auto& ws = workspaces_[si];

    ensure_R_env_alloc(site, chi_out);

    Scalar* A = d_mps_tensors_[site];
    Scalar* R_env = d_R_envs_[site + 1];
    Scalar* W_mat = d_W_right_[site];
    Scalar* R_new = d_R_envs_[site];
    Scalar* V = ws.d_T1;
    Scalar* U = ws.d_T2;

    // Step 1: V_ws[a,b'] = A_s[a,b] * R_w'[b,b']  (batched GEMM)
    {
        std::vector<Scalar*> h_A(D * d), h_B(D * d), h_C(D * d);
        for (int wp = 0; wp < D; wp++)
            for (int s = 0; s < d; s++) {
                int ws_idx = wp * d + s;
                h_A[ws_idx] = A + s * chi_out;
                h_B[ws_idx] = R_env + wp * chi_in;
                h_C[ws_idx] = V + ws_idx * chi_out * chi_in;
            }
        HIP_CHECK(hipMemcpyAsync(ws.d_batch_A, h_A.data(), D*d*sizeof(Scalar*), hipMemcpyHostToDevice, streams_[si]));
        HIP_CHECK(hipMemcpyAsync(ws.d_batch_B, h_B.data(), D*d*sizeof(Scalar*), hipMemcpyHostToDevice, streams_[si]));
        HIP_CHECK(hipMemcpyAsync(ws.d_batch_C, h_C.data(), D*d*sizeof(Scalar*), hipMemcpyHostToDevice, streams_[si]));
        ROCBLAS_CHECK(Traits::gemm_batched(handles_[si],
            rocblas_operation_none, rocblas_operation_none,
            chi_out, chi_in, chi_in,
            &one,
            (const Scalar**)ws.d_batch_A, chi_out * d,
            (const Scalar**)ws.d_batch_B, chi_in * D,
            &zero_val,
            ws.d_batch_C, chi_out,
            D * d));
    }

    // Step 2: U = V * W_matrix
    ROCBLAS_CHECK(Traits::gemm(handles_[si],
        rocblas_operation_none, rocblas_operation_none,
        chi_out * chi_in, d * D, D * d,
        &one,
        V, chi_out * chi_in,
        W_mat, D * d,
        &zero_val,
        U, chi_out * chi_in));

    // Step 3: R_new_w[a,a'] = sum_s' U_ws'[a,b'] * A_s'^H[b',a']
    for (int w = 0; w < D; w++) {
        for (int sp = 0; sp < d; sp++) {
            Scalar beta = (sp == 0) ? Traits::zero() : Traits::one();
            int ws_out = w * d + sp;
            ROCBLAS_CHECK(Traits::gemm(handles_[si],
                rocblas_operation_none, Traits::op_h,
                chi_out, chi_out, chi_in,
                &one,
                U + ws_out * chi_out * chi_in, chi_out,
                A + sp * chi_out, chi_out * d,
                &beta,
                R_new + w * chi_out, chi_out * D));
        }
    }
}

// ============================================================================
// Environment building (uses stream 0)
// ============================================================================

template<typename Scalar>
void PDMRGGPU<Scalar>::build_initial_environments() {
    // L[0] = trivial left boundary
    {
        std::vector<Scalar> h_L(D_mpo_, Traits::zero());
        h_L[0] = Traits::one();
        HIP_CHECK(hipMemcpy(d_L_envs_[0], h_L.data(),
                            D_mpo_ * sizeof(Scalar), hipMemcpyHostToDevice));
    }

    // R[L] = trivial right boundary
    {
        std::vector<Scalar> h_R(D_mpo_, Traits::zero());
        h_R[D_mpo_ - 1] = Traits::one();
        HIP_CHECK(hipMemcpy(d_R_envs_[L_], h_R.data(),
                            D_mpo_ * sizeof(Scalar), hipMemcpyHostToDevice));
    }

    // Build all L environments left-to-right on stream 0
    for (int i = 0; i < L_; i++) {
        update_left_env(i, 0);
    }
    HIP_CHECK(hipStreamSynchronize(streams_[0]));

    // Build all R environments right-to-left on stream 0
    for (int i = L_ - 1; i >= 0; i--) {
        update_right_env(i, 0);
    }
    HIP_CHECK(hipStreamSynchronize(streams_[0]));
}

// ============================================================================
// Lanczos eigensolver (stream-aware)
// ============================================================================

template<typename Scalar>
double PDMRGGPU<Scalar>::lanczos_eigensolver(int site, Scalar* d_theta, int theta_size, int si) {
    int n = theta_size;
    int max_iter = std::min(max_lanczos_iter_, n);
    double tol_lanczos = 1e-12;
    double tol_eig_conv = 1e-12;
    auto& ws = workspaces_[si];

    Scalar* d_lanczos_v = ws.d_lanczos_v;

    std::vector<double> h_alpha(max_iter);
    std::vector<double> h_beta(max_iter);

    // v[0] = theta / ||theta||
    double norm;
    ROCBLAS_CHECK(Traits::nrm2(handles_[si], n, d_theta, 1, &norm));

    if (norm < 1e-14) {
        std::vector<Scalar> h_init(n);
        srand(42 + site);
        for (int i = 0; i < n; i++) h_init[i] = Traits::random_val();
        HIP_CHECK(hipMemcpy(d_theta, h_init.data(), n * sizeof(Scalar), hipMemcpyHostToDevice));
        ROCBLAS_CHECK(Traits::nrm2(handles_[si], n, d_theta, 1, &norm));
    }

    double inv_norm = 1.0 / norm;
    ROCBLAS_CHECK(Traits::scal_real(handles_[si], n, &inv_norm, d_theta, 1));
    HIP_CHECK(hipMemcpyAsync(d_lanczos_v, d_theta, n * sizeof(Scalar), hipMemcpyDeviceToDevice, streams_[si]));
    HIP_CHECK(hipStreamSynchronize(streams_[si]));

    double prev_energy = 1e30;
    int iter;
    for (iter = 0; iter < max_iter; iter++) {
        Scalar* d_vi = d_lanczos_v + (size_t)iter * n;

        // w = H|v_i>
        apply_heff_two_site(site, d_vi, ws.d_heff_result, si);

        // alpha_i = <v_i|w>
        Scalar alpha_result;
        ROCBLAS_CHECK(Traits::dot(handles_[si], n, d_vi, 1, ws.d_heff_result, 1, &alpha_result));
        double alpha_i = Traits::real_part(alpha_result);
        h_alpha[iter] = alpha_i;

        // w = w - alpha_i * v_i
        Scalar neg_alpha = Traits::make_scalar(-alpha_i);
        ROCBLAS_CHECK(Traits::axpy(handles_[si], n, &neg_alpha, d_vi, 1, ws.d_heff_result, 1));

        // w = w - beta_{i-1} * v_{i-1}
        if (iter > 0) {
            Scalar neg_beta = Traits::make_scalar(-h_beta[iter - 1]);
            Scalar* d_vim1 = d_lanczos_v + (size_t)(iter - 1) * n;
            ROCBLAS_CHECK(Traits::axpy(handles_[si], n, &neg_beta, d_vim1, 1, ws.d_heff_result, 1));
        }

        // Full reorthogonalization
        if (iter > 0) {
            Scalar one_val = Traits::one(), zero_sc = Traits::zero(), neg_one = Traits::neg(Traits::one());
            ROCBLAS_CHECK(Traits::gemv(handles_[si], Traits::op_h,
                n, iter + 1, &one_val,
                d_lanczos_v, n,
                ws.d_heff_result, 1,
                &zero_sc, ws.d_ritz_coeffs, 1));
            ROCBLAS_CHECK(Traits::gemv(handles_[si], rocblas_operation_none,
                n, iter + 1, &neg_one,
                d_lanczos_v, n,
                ws.d_ritz_coeffs, 1,
                &one_val, ws.d_heff_result, 1));
        } else {
            Scalar overlap;
            ROCBLAS_CHECK(Traits::dot(handles_[si], n, d_lanczos_v, 1, ws.d_heff_result, 1, &overlap));
            Scalar neg_overlap = Traits::neg(overlap);
            ROCBLAS_CHECK(Traits::axpy(handles_[si], n, &neg_overlap, d_lanczos_v, 1, ws.d_heff_result, 1));
        }

        // beta_i = ||w||
        double beta_i;
        ROCBLAS_CHECK(Traits::nrm2(handles_[si], n, ws.d_heff_result, 1, &beta_i));
        h_beta[iter] = beta_i;

        if (beta_i < tol_lanczos) {
            iter++;
            break;
        }

        // Convergence check every 3 iterations after iter >= 4
        if (iter >= 4 && iter % 3 == 0) {
            int ncheck = iter + 1;
            std::vector<double> h_D_chk(ncheck), h_E_chk(ncheck);
            std::copy(h_alpha.begin(), h_alpha.begin() + ncheck, h_D_chk.begin());
            for (int i = 0; i < ncheck - 1; i++) h_E_chk[i] = h_beta[i];
            h_E_chk[ncheck - 1] = 0.0;
            const char jobz_n = 'N';
            const int n_chk = ncheck;
            std::vector<double> h_work_chk(1);
            int info_chk = 0;
            dstev_(&jobz_n, &n_chk, h_D_chk.data(), h_E_chk.data(), nullptr, &n_chk, h_work_chk.data(), &info_chk);
            if (info_chk == 0) {
                double cur_energy = h_D_chk[0];
                if (std::abs(cur_energy - prev_energy) < tol_eig_conv) {
                    iter++;
                    break;
                }
                prev_energy = cur_energy;
            }
        }

        // v_{i+1} = w / beta_i
        if (iter + 1 < max_iter) {
            Scalar* d_vip1 = d_lanczos_v + (size_t)(iter + 1) * n;
            double scale = 1.0 / beta_i;
            HIP_CHECK(hipMemcpyAsync(d_vip1, ws.d_heff_result, n * sizeof(Scalar), hipMemcpyDeviceToDevice, streams_[si]));
            ROCBLAS_CHECK(Traits::scal_real(handles_[si], n, &scale, d_vip1, 1));
        }
    }

    int niter = iter;

    // Solve tridiagonal eigenvalue problem on CPU
    std::vector<double> h_D(niter), h_E(niter), h_Z(niter * niter);
    std::vector<double> h_work(std::max(1, 2*niter - 2));
    int lapack_info = 0;

    std::copy(h_alpha.begin(), h_alpha.begin() + niter, h_D.begin());
    for (int i = 0; i < niter - 1; i++) h_E[i] = h_beta[i];
    if (niter > 0) h_E[niter - 1] = 0.0;

    const char jobz = 'V';
    const int n_lapack = niter;
    const int ldz = niter;

    dstev_(&jobz, &n_lapack, h_D.data(), h_E.data(), h_Z.data(), &ldz, h_work.data(), &lapack_info);

    if (lapack_info != 0) {
        throw std::runtime_error("LAPACK dstev failed with info = " + std::to_string(lapack_info));
    }

    double energy = h_D[0];

    // Reconstruct ground state: |theta> = sum_i c[i] |v_i>
    std::vector<Scalar> h_ritz_scalar(niter);
    for (int i = 0; i < niter; i++) {
        h_ritz_scalar[i] = Traits::make_scalar(h_Z[i]);
    }
    HIP_CHECK(hipMemcpyAsync(ws.d_ritz_coeffs, h_ritz_scalar.data(),
              niter * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));

    Scalar one_sc = Traits::one(), zero_sc = Traits::zero();
    ROCBLAS_CHECK(Traits::gemv(
        handles_[si], rocblas_operation_none,
        n, niter, &one_sc,
        d_lanczos_v, n,
        ws.d_ritz_coeffs, 1,
        &zero_sc, d_theta, 1
    ));

    // Normalize
    ROCBLAS_CHECK(Traits::nrm2(handles_[si], n, d_theta, 1, &norm));
    inv_norm = 1.0 / norm;
    ROCBLAS_CHECK(Traits::scal_real(handles_[si], n, &inv_norm, d_theta, 1));

    return energy;
}

// ============================================================================
// SVD split (stream-aware)
// ============================================================================

template<typename Scalar>
void PDMRGGPU<Scalar>::svd_split(int site, Scalar* d_theta, char direction, int si) {
    int cL = chi_L(site);
    int cR = chi_R(site + 1);
    auto& ws = workspaces_[si];

    int m = cL * d_;
    int n_svd = d_ * cR;
    int full_k = std::min(m, n_svd);
    int k = std::min(full_k, chi_max_);

    Scalar* h_U_data;
    RealType* h_S_data;
    Scalar* h_Vh_data;

    if (use_cpu_svd_) {
        HIP_CHECK(hipMemcpyAsync(ws.h_svd_A.data(), d_theta, m * n_svd * sizeof(Scalar),
                                  hipMemcpyDeviceToHost, streams_[si]));
        HIP_CHECK(hipStreamSynchronize(streams_[si]));

        int lwork = (int)ws.h_svd_work.size();
        int info;
        const char jobu = 'S', jobvt = 'S';
        Traits::lapack_gesvd(&jobu, &jobvt, &m, &n_svd, ws.h_svd_A.data(), &m,
                ws.h_svd_S.data(), ws.h_svd_U.data(), &m, ws.h_svd_Vh.data(), &full_k,
                ws.h_svd_work.data(), &lwork,
                ws.h_svd_rwork.empty() ? nullptr : ws.h_svd_rwork.data(), &info);

        h_U_data = ws.h_svd_U.data();
        h_S_data = ws.h_svd_S.data();
        h_Vh_data = ws.h_svd_Vh.data();
    } else {
        HIP_CHECK(hipMemcpyAsync(ws.d_svd_A, d_theta, m * n_svd * sizeof(Scalar),
                                  hipMemcpyDeviceToDevice, streams_[si]));

        Traits::rocsolver_gesvd(handles_[si],
            rocblas_svect_singular, rocblas_svect_singular,
            m, n_svd,
            ws.d_svd_A, m,
            ws.d_svd_S,
            ws.d_svd_U, m,
            ws.d_svd_Vh, full_k,
            ws.d_svd_E,
            rocblas_outofplace,
            ws.d_svd_info);

        HIP_CHECK(hipMemcpy(ws.h_svd_U.data(), ws.d_svd_U, m * full_k * sizeof(Scalar), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(ws.h_svd_S.data(), ws.d_svd_S, full_k * sizeof(RealType), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(ws.h_svd_Vh.data(), ws.d_svd_Vh, full_k * n_svd * sizeof(Scalar), hipMemcpyDeviceToHost));

        h_U_data = ws.h_svd_U.data();
        h_S_data = ws.h_svd_S.data();
        h_Vh_data = ws.h_svd_Vh.data();
    }

    // Truncation
    int new_k = k;
    for (int i = 0; i < new_k; i++) {
        if (h_S_data[i] < 1e-14) { new_k = i; break; }
    }
    if (new_k == 0) new_k = 1;

    if (direction == 'R') {
        allocate_mps_tensor(site, cL, new_k);
        HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site], h_U_data,
                    m * new_k * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));

        for (int j = 0; j < n_svd; j++)
            for (int i = 0; i < new_k; i++)
                ws.h_svd_tmp[i + j * new_k] = Traits::scale_by_real(h_S_data[i], h_Vh_data[i + j * full_k]);

        allocate_mps_tensor(site + 1, new_k, cR);
        HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site + 1], ws.h_svd_tmp.data(),
                    new_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));

    } else {  // direction == 'L'
        for (int j = 0; j < new_k; j++)
            for (int i = 0; i < m; i++)
                ws.h_svd_tmp[i + j * m] = Traits::scale_by_real(h_S_data[j], h_U_data[i + j * m]);

        allocate_mps_tensor(site, cL, new_k);
        HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site], ws.h_svd_tmp.data(),
                    m * new_k * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));

        allocate_mps_tensor(site + 1, new_k, cR);
        if (new_k == full_k) {
            HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site + 1], h_Vh_data,
                        full_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));
        } else {
            for (int j = 0; j < n_svd; j++)
                for (int i = 0; i < new_k; i++)
                    ws.h_svd_tmp[i + j * new_k] = h_Vh_data[i + j * full_k];
            HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site + 1], ws.h_svd_tmp.data(),
                        new_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));
        }
    }

    bond_dims_[site + 1] = new_k;
}

// ============================================================================
// Bond optimization (stream-aware)
// ============================================================================

template<typename Scalar>
double PDMRGGPU<Scalar>::optimize_bond(int site, char direction, int si) {
    int cL = chi_L(site);
    int cR = chi_R(site + 1);
    int theta_size = cL * d_ * d_ * cR;
    auto& ws = workspaces_[si];

    form_theta_two_site(site, si);

    HIP_CHECK(hipStreamSynchronize(streams_[si]));
    double energy = lanczos_eigensolver(site, ws.d_theta, theta_size, si);
    HIP_CHECK(hipStreamSynchronize(streams_[si]));
    svd_split(site, ws.d_theta, direction, si);
    HIP_CHECK(hipStreamSynchronize(streams_[si]));

    return energy;
}

// ============================================================================
// Full-chain sweep methods (for warmup, use stream 0)
// ============================================================================

template<typename Scalar>
double PDMRGGPU<Scalar>::sweep_LR_full() {
    double energy = 0.0;
    for (int site = 0; site < L_ - 1; site++) {
        energy = optimize_bond(site, 'R', 0);
        update_left_env(site, 0);
        HIP_CHECK(hipStreamSynchronize(streams_[0]));
    }
    return energy;
}

template<typename Scalar>
double PDMRGGPU<Scalar>::sweep_RL_full() {
    double energy = 0.0;
    for (int site = L_ - 2; site >= 0; site--) {
        energy = optimize_bond(site, 'L', 0);
        update_right_env(site + 1, 0);
        HIP_CHECK(hipStreamSynchronize(streams_[0]));
    }
    return energy;
}

// ============================================================================
// Segment sweep methods (restricted range, per-segment stream)
// ============================================================================

template<typename Scalar>
void PDMRGGPU<Scalar>::segment_sweep_LR(int seg_idx) {
    int first = seg_first_[seg_idx];
    int last = seg_last_[seg_idx];
    int si = seg_idx;

    for (int site = first; site < last; site++) {
        optimize_bond(site, 'R', si);
        update_left_env(site, si);
    }
}

template<typename Scalar>
void PDMRGGPU<Scalar>::segment_sweep_RL(int seg_idx) {
    int first = seg_first_[seg_idx];
    int last = seg_last_[seg_idx];
    int si = seg_idx;

    for (int site = last - 1; site >= first; site--) {
        optimize_bond(site, 'L', si);
        update_right_env(site + 1, si);
    }
}

// ============================================================================
// Boundary V-matrix computation via SVD
// ============================================================================

template<typename Scalar>
void PDMRGGPU<Scalar>::compute_boundary_V(int boundary_idx) {
    int b = boundary_bonds_[boundary_idx];
    int cL = chi_L(b);
    int chi_mid = bond_dims_[b + 1];
    int cR = chi_R(b + 1);
    auto& ws = workspaces_[0];

    // Form theta = MPS[b] * MPS[b+1] on stream 0
    form_theta_two_site(b, 0);
    HIP_CHECK(hipStreamSynchronize(streams_[0]));

    // SVD theta: (cL*d, d*cR)
    int m = cL * d_;
    int n_svd = d_ * cR;
    int full_k = std::min(m, n_svd);
    int k = std::min(full_k, chi_max_);

    // Copy theta to host for accurate SVD
    HIP_CHECK(hipMemcpy(ws.h_svd_A.data(), ws.d_theta, m * n_svd * sizeof(Scalar), hipMemcpyDeviceToHost));

    // Accurate SVD (Stoudenmire Appendix): uniform relative accuracy for all
    // singular values, critical because V = 1/sigma amplifies errors in small sigma
    accurate_svd<Scalar>(m, n_svd, ws.h_svd_A.data(), m,
            ws.h_svd_U.data(), m, ws.h_svd_S.data(),
            ws.h_svd_Vh.data(), full_k, 1e-4);

    // Truncation
    int new_k = k;
    for (int i = 0; i < new_k; i++) {
        if (ws.h_svd_S[i] < 1e-14) { new_k = i; break; }
    }
    if (new_k == 0) new_k = 1;

    // V = 1/S (regularized)
    h_V_boundary_[boundary_idx].resize(new_k);
    for (int i = 0; i < new_k; i++) {
        h_V_boundary_[boundary_idx][i] = (ws.h_svd_S[i] > 1e-14) ? 1.0 / ws.h_svd_S[i] : 0.0;
    }
    HIP_CHECK(hipMemcpy(d_V_boundary_[boundary_idx], h_V_boundary_[boundary_idx].data(),
                        new_k * sizeof(double), hipMemcpyHostToDevice));

    // MPS[b] = U (left-canonical)
    allocate_mps_tensor(b, cL, new_k);
    HIP_CHECK(hipMemcpy(d_mps_tensors_[b], ws.h_svd_U.data(),
                        m * new_k * sizeof(Scalar), hipMemcpyHostToDevice));

    // MPS[b+1] = S * Vh
    for (int j = 0; j < n_svd; j++)
        for (int i = 0; i < new_k; i++)
            ws.h_svd_tmp[i + j * new_k] = Traits::scale_by_real(ws.h_svd_S[i], ws.h_svd_Vh[i + j * full_k]);

    allocate_mps_tensor(b + 1, new_k, cR);
    HIP_CHECK(hipMemcpy(d_mps_tensors_[b + 1], ws.h_svd_tmp.data(),
                        new_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice));

    bond_dims_[b + 1] = new_k;
}

// ============================================================================
// Boundary theta formation (standard, no V-weighting)
// ============================================================================

template<typename Scalar>
void PDMRGGPU<Scalar>::form_boundary_theta(int boundary_idx, Scalar* d_theta, int si) {
    int b = boundary_bonds_[boundary_idx];
    // Use standard theta formation: theta = MPS[b] * MPS[b+1]
    // V-weighting is not used because after local sweeps the boundary MPS
    // tensors have been modified, making the V-matrix stale.
    // Lanczos will find the ground state regardless of initial guess.
    form_theta_two_site(b, si);
    // form_theta_two_site writes to ws.d_theta which is the same as d_theta
}

// ============================================================================
// Boundary merge: optimize boundary bond
// ============================================================================

template<typename Scalar>
double PDMRGGPU<Scalar>::merge_boundary(int boundary_idx, int si) {
    int b = boundary_bonds_[boundary_idx];

    // Standard two-site optimization at the boundary bond
    double energy = optimize_bond(b, 'R', si);

    return energy;
}

// ============================================================================
// Rebuild environments at boundary after merge
// ============================================================================

template<typename Scalar>
void PDMRGGPU<Scalar>::rebuild_boundary_envs(int boundary_idx, int si) {
    int b = boundary_bonds_[boundary_idx];

    // L_env[b+1] from L_env[b] + new MPS[b]
    // (segment k+1 needs this as its left boundary)
    update_left_env(b, si);
    HIP_CHECK(hipStreamSynchronize(streams_[si]));

    // R_env[b+1] from R_env[b+2] + new MPS[b+1]
    // (segment k needs this as its right boundary)
    update_right_env(b + 1, si);
    HIP_CHECK(hipStreamSynchronize(streams_[si]));
}

// ============================================================================
// Boundary bond optimization with V-weighted initial guess (Stoudenmire Eq. 5)
// Uses accurate SVD for V = 1/sigma computation
// ============================================================================

template<typename Scalar>
double PDMRGGPU<Scalar>::optimize_boundary_bond(int boundary_idx, char direction, int si) {
    int b = boundary_bonds_[boundary_idx];
    int cL = chi_L(b);
    int chi_mid = bond_dims_[b + 1];
    int cR = chi_R(b + 1);
    int theta_size = cL * d_ * d_ * cR;
    auto& ws = workspaces_[si];
    Scalar one = Traits::one(), zero_val = Traits::zero();

    // Form V-weighted theta: theta = (MPS[b] · diag(V)) · MPS[b+1]
    // This is the Stoudenmire merge: Psi' = psi'_L · V · psi'_R (Eq. 5)
    // column_scale_real: T1[i,j] = V[j] * MPS[b][i,j]
    column_scale_real(ws.d_T1, d_mps_tensors_[b], d_V_boundary_[boundary_idx],
                       cL * d_, chi_mid, streams_[si]);

    // theta = T1 · MPS[b+1]
    ROCBLAS_CHECK(Traits::gemm(handles_[si],
        rocblas_operation_none, rocblas_operation_none,
        cL * d_, d_ * cR, chi_mid,
        &one,
        ws.d_T1, cL * d_,
        d_mps_tensors_[b + 1], chi_mid,
        &zero_val,
        ws.d_theta, cL * d_));

    HIP_CHECK(hipStreamSynchronize(streams_[si]));

    // Lanczos optimization at boundary bond
    double energy = lanczos_eigensolver(b, ws.d_theta, theta_size, si);
    HIP_CHECK(hipStreamSynchronize(streams_[si]));

    // Accurate SVD split (not standard svd_split — we need accurate singular
    // values for V = 1/sigma)
    int m = cL * d_;
    int n_svd = d_ * cR;
    int full_k = std::min(m, n_svd);
    int k = std::min(full_k, chi_max_);

    // Copy optimized theta to host
    HIP_CHECK(hipMemcpyAsync(ws.h_svd_A.data(), ws.d_theta,
                              m * n_svd * sizeof(Scalar), hipMemcpyDeviceToHost, streams_[si]));
    HIP_CHECK(hipStreamSynchronize(streams_[si]));

    // Accurate SVD (Stoudenmire Appendix)
    accurate_svd<Scalar>(m, n_svd, ws.h_svd_A.data(), m,
                         ws.h_svd_U.data(), m, ws.h_svd_S.data(),
                         ws.h_svd_Vh.data(), full_k, 1e-4);

    // Truncation
    int new_k = k;
    for (int i = 0; i < new_k; i++) {
        if (ws.h_svd_S[i] < 1e-14) { new_k = i; break; }
    }
    if (new_k == 0) new_k = 1;

    // Split into MPS tensors
    if (direction == 'R') {
        // MPS[b] = U, MPS[b+1] = S*Vh
        allocate_mps_tensor(b, cL, new_k);
        HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[b], ws.h_svd_U.data(),
                    m * new_k * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));

        for (int j = 0; j < n_svd; j++)
            for (int i = 0; i < new_k; i++)
                ws.h_svd_tmp[i + j * new_k] = Traits::scale_by_real(ws.h_svd_S[i], ws.h_svd_Vh[i + j * full_k]);

        allocate_mps_tensor(b + 1, new_k, cR);
        HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[b + 1], ws.h_svd_tmp.data(),
                    new_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));
    } else {
        // MPS[b] = U*S, MPS[b+1] = Vh
        for (int j = 0; j < new_k; j++)
            for (int i = 0; i < m; i++)
                ws.h_svd_tmp[i + j * m] = Traits::scale_by_real(ws.h_svd_S[j], ws.h_svd_U[i + j * m]);

        allocate_mps_tensor(b, cL, new_k);
        HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[b], ws.h_svd_tmp.data(),
                    m * new_k * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));

        allocate_mps_tensor(b + 1, new_k, cR);
        if (new_k == full_k) {
            HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[b + 1], ws.h_svd_Vh.data(),
                        full_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));
        } else {
            for (int j = 0; j < n_svd; j++)
                for (int i = 0; i < new_k; i++)
                    ws.h_svd_tmp[i + j * new_k] = ws.h_svd_Vh[i + j * full_k];
            HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[b + 1], ws.h_svd_tmp.data(),
                        new_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));
        }
    }

    bond_dims_[b + 1] = new_k;

    // Update V = 1/S from accurate singular values
    h_V_boundary_[boundary_idx].resize(new_k);
    for (int i = 0; i < new_k; i++) {
        h_V_boundary_[boundary_idx][i] = (ws.h_svd_S[i] > 1e-14) ? 1.0 / ws.h_svd_S[i] : 0.0;
    }
    HIP_CHECK(hipMemcpyAsync(d_V_boundary_[boundary_idx],
                              h_V_boundary_[boundary_idx].data(),
                              new_k * sizeof(double),
                              hipMemcpyHostToDevice, streams_[si]));

    HIP_CHECK(hipStreamSynchronize(streams_[si]));

    return energy;
}

// ============================================================================
// Boundary coupling sweep (replaces full-chain coupling sweep)
//
// Per Stoudenmire & White: rebuild environments, then optimize only at P-1
// boundary bonds with V-weighted initial guess. O(P) Lanczos calls + O(L)
// environment updates instead of O(L) Lanczos calls.
// ============================================================================

template<typename Scalar>
double PDMRGGPU<Scalar>::boundary_coupling_sweep() {
    int n_boundaries = n_segments_ - 1;

    if (n_boundaries == 0) {
        // Single segment — fall back to standard full-chain sweep
        build_initial_environments();
        sweep_LR_full();
        return sweep_RL_full();
    }

    double energy = 0.0;

    // Step 1: Rebuild all environments from current MPS state
    // After parallel segment sweeps, environments at boundaries are stale.
    // A full rebuild makes L_env[i] consistent with MPS[0..i-1] and
    // R_env[i] consistent with MPS[i..L-1] for all i.
    build_initial_environments();

    // Step 2: L→R boundary pass
    for (int k = 0; k < n_boundaries; k++) {
        int b = boundary_bonds_[k];
        energy = optimize_boundary_bond(k, 'R', 0);

        // Propagate L_env forward to next boundary (or end of chain).
        // After optimize_boundary_bond modified MPS[b] and MPS[b+1],
        // L_env[b+1] and beyond are stale. Rebuild incrementally.
        int end = (k + 1 < n_boundaries) ? boundary_bonds_[k + 1] - 1 : L_ - 2;
        for (int site = b; site <= end; site++) {
            update_left_env(site, 0);
        }
        HIP_CHECK(hipStreamSynchronize(streams_[0]));
    }

    // Step 3: Rebuild all R environments
    // The L→R pass modified MPS at each boundary, making R_env stale.
    for (int i = L_ - 1; i >= 0; i--) {
        update_right_env(i, 0);
    }
    HIP_CHECK(hipStreamSynchronize(streams_[0]));

    // Step 4: R→L boundary pass
    for (int k = n_boundaries - 1; k >= 0; k--) {
        int b = boundary_bonds_[k];
        energy = optimize_boundary_bond(k, 'L', 0);

        // Propagate R_env backward to previous boundary (or start of chain)
        int prev_end = (k > 0) ? boundary_bonds_[k - 1] + 1 : 0;
        for (int site = b + 1; site >= prev_end; site--) {
            update_right_env(site, 0);
        }
        HIP_CHECK(hipStreamSynchronize(streams_[0]));
    }

    return energy;
}

// ============================================================================
// Main algorithm
// ============================================================================

template<typename Scalar>
double PDMRGGPU<Scalar>::run(int n_outer_sweeps, int n_local_sweeps, int n_warmup) {
    const char* type_name = Traits::is_complex ? "complex128" : "float64";
    printf("=== Stream-Parallel DMRG (PDMRG-GPU, %s) ===\n", type_name);
    printf("L = %d, d = %d, chi_max = %d, D_mpo = %d, segments = %d\n",
           L_, d_, chi_max_, D_mpo_, n_segments_);
    printf("Warmup sweeps: %d, local sweeps/iter: %d, outer sweeps: %d\n\n",
           n_warmup, n_local_sweeps, n_outer_sweeps);

    // Print segment layout
    for (int k = 0; k < n_segments_; k++) {
        printf("  Segment %d: sites [%d, %d]\n", k, seg_first_[k], seg_last_[k]);
    }
    for (int k = 0; k < n_segments_ - 1; k++) {
        printf("  Boundary %d: bond at site %d\n", k, boundary_bonds_[k]);
    }
    printf("\n");

    auto t_start = std::chrono::high_resolution_clock::now();

    // === Phase 0: Build environments and warmup ===
    printf("Building initial environments...\n");
    build_initial_environments();

    auto t_envs = std::chrono::high_resolution_clock::now();
    printf("  Environment build: %.3f s\n", std::chrono::duration<double>(t_envs - t_start).count());

    // Warmup: full-chain dmrg2 sweeps on stream 0
    printf("Running %d warmup sweeps (full-chain dmrg2)...\n", n_warmup);
    double warmup_energy = 0.0;
    for (int sw = 0; sw < n_warmup; sw++) {
        auto t_sw = std::chrono::high_resolution_clock::now();
        double eLR = sweep_LR_full();
        double eRL = sweep_RL_full();
        warmup_energy = eRL;
        auto t_sw_end = std::chrono::high_resolution_clock::now();
        double sw_time = std::chrono::duration<double>(t_sw_end - t_sw).count();
        printf("  Warmup %d: E = %.12f, time = %.3f s\n", sw, warmup_energy, sw_time);
    }

    auto t_warmup_end = std::chrono::high_resolution_clock::now();
    printf("Warmup: %.3f s\n\n", std::chrono::duration<double>(t_warmup_end - t_start).count());

    // === Compute boundary V-matrices (Stoudenmire: V = Lambda^{-1}) ===
    if (n_segments_ > 1) {
        printf("Computing boundary V-matrices (accurate SVD)...\n");
        for (int k = 0; k < n_segments_ - 1; k++) {
            compute_boundary_V(k);
        }
        printf("  V-matrices computed for %d boundaries\n\n", n_segments_ - 1);
    }

    // === Main PDMRG loop ===
    double energy_prev = warmup_energy;
    energy_ = warmup_energy;

    for (int outer = 0; outer < n_outer_sweeps; outer++) {
        auto t_outer = std::chrono::high_resolution_clock::now();

        // === Phase 1: Independent segment sweeps (parallel streams) ===
        for (int local_sw = 0; local_sw < n_local_sweeps; local_sw++) {
            if (outer % 2 == 0) {
                // Even iteration: L→R then R→L
                for (int k = 0; k < n_segments_; k++) {
                    segment_sweep_LR(k);
                }
                for (int k = 0; k < n_segments_; k++) {
                    HIP_CHECK(hipStreamSynchronize(streams_[k]));
                }
                for (int k = 0; k < n_segments_; k++) {
                    segment_sweep_RL(k);
                }
                for (int k = 0; k < n_segments_; k++) {
                    HIP_CHECK(hipStreamSynchronize(streams_[k]));
                }
            } else {
                // Odd iteration: R→L then L→R
                for (int k = 0; k < n_segments_; k++) {
                    segment_sweep_RL(k);
                }
                for (int k = 0; k < n_segments_; k++) {
                    HIP_CHECK(hipStreamSynchronize(streams_[k]));
                }
                for (int k = 0; k < n_segments_; k++) {
                    segment_sweep_LR(k);
                }
                for (int k = 0; k < n_segments_; k++) {
                    HIP_CHECK(hipStreamSynchronize(streams_[k]));
                }
            }
        }

        // === Phase 2: Boundary coupling (Stoudenmire merge) ===
        // Replaces full-chain sweep with O(P) boundary optimizations
        energy_ = boundary_coupling_sweep();

        auto t_outer_end = std::chrono::high_resolution_clock::now();
        double outer_time = std::chrono::duration<double>(t_outer_end - t_outer).count();

        double dE = std::abs(energy_ - energy_prev);

        // Print bond dimensions
        printf("Outer %3d: E = %.12f, dE = %.2e, time = %.3f s  chi=[",
               outer, energy_, dE, outer_time);
        for (int i = 1; i < L_; i++) {
            printf("%d", bond_dims_[i]);
            if (i < L_ - 1) printf(",");
        }
        printf("]\n");

        if (dE < tol_ && outer > 0) {
            printf("Converged after %d outer iterations!\n", outer + 1);
            break;
        }

        energy_prev = energy_;
    }

    // === Polish phase: full-chain sweeps to converge to tight tolerance ===
    if (n_segments_ > 1) {
        int n_polish = std::min(2, n_outer_sweeps);
        printf("Polish sweeps (full-chain dmrg2, max %d)...\n", n_polish);
        build_initial_environments();
        for (int sw = 0; sw < n_polish; sw++) {
            auto t_sw = std::chrono::high_resolution_clock::now();
            double eLR = sweep_LR_full();
            double eRL = sweep_RL_full();
            auto t_sw_end = std::chrono::high_resolution_clock::now();
            double sw_time = std::chrono::duration<double>(t_sw_end - t_sw).count();
            double dE = std::abs(eRL - energy_);
            printf("  Polish %d: E = %.12f, dE = %.2e, time = %.3f s\n", sw, eRL, dE, sw_time);
            energy_ = eRL;
            if (dE < tol_) {
                printf("  Polish converged after %d sweeps\n", sw + 1);
                break;
            }
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(t_end - t_start).count();
    printf("\nTotal wall time: %.3f s\n", total_time);

    return energy_;
}

// ============================================================================
// Utility methods
// ============================================================================

template<typename Scalar>
void PDMRGGPU<Scalar>::get_mps(std::vector<std::vector<Scalar>>& h_mps) const {
    h_mps.resize(L_);
    for (int i = 0; i < L_; i++) {
        int size = chi_L(i) * d_ * chi_R(i);
        h_mps[i].resize(size);
        HIP_CHECK(hipMemcpy(h_mps[i].data(), d_mps_tensors_[i],
                            size * sizeof(Scalar), hipMemcpyDeviceToHost));
    }
}

#endif // PDMRG_GPU_IMPL_H
