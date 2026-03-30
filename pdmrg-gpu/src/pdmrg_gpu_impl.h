#ifndef PDMRG_GPU_IMPL_H
#define PDMRG_GPU_IMPL_H

#include <rocsolver/rocsolver.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <stdexcept>

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
    initialize_boundary_states();

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

    // Allocate per-stream workspaces
    theta_size_max_ = chi_max_ * dd * chi_max_;
    max_lanczos_iter_ = std::min(100, theta_size_max_);
    use_cpu_svd_ = false;
    use_rsvd_ = false;
    lanczos_use_1site_ = false;
    rsvd_oversampling_ = 20;

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

        // Batched GEMM pointer arrays (device)
        HIP_CHECK(hipMalloc(&ws.d_batch_A, batch_max * sizeof(Scalar*)));
        HIP_CHECK(hipMalloc(&ws.d_batch_B, batch_max * sizeof(Scalar*)));
        HIP_CHECK(hipMalloc(&ws.d_batch_C, batch_max * sizeof(Scalar*)));
        // Pinned host pointer arrays no longer needed — pointer setup done by GPU kernels
        ws.h_batch_A_pinned = nullptr;
        ws.h_batch_B_pinned = nullptr;
        ws.h_batch_C_pinned = nullptr;
        // Cached apply_heff A/C pointers (separate device arrays)
        HIP_CHECK(hipMalloc(&ws.d_heff_batch_A, batch_max * sizeof(Scalar*)));
        HIP_CHECK(hipMalloc(&ws.d_heff_batch_C, batch_max * sizeof(Scalar*)));
        ws.heff_cached_site = -1;

        // Lanczos device-pointer-mode scalars
        HIP_CHECK(hipMalloc(&ws.d_dot_result, sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&ws.d_nrm2_result, sizeof(RealType)));
        HIP_CHECK(hipMalloc(&ws.d_neg_alpha, sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&ws.d_neg_overlap, sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&ws.d_inv_nrm, sizeof(RealType)));
        HIP_CHECK(hipMalloc(&ws.d_alpha_dev, max_lanczos_iter_ * sizeof(RealType)));
        HIP_CHECK(hipMalloc(&ws.d_beta_dev, max_lanczos_iter_ * sizeof(RealType)));
        HIP_CHECK(hipMalloc(&ws.d_neg_beta_scalars, max_lanczos_iter_ * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&ws.d_const_one, sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&ws.d_const_zero, sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&ws.d_const_neg_one, sizeof(Scalar)));
        {
            Scalar h_one = Traits::one(), h_zero = Traits::zero();
            Scalar h_neg_one = Traits::neg(Traits::one());
            HIP_CHECK(hipMemcpy(ws.d_const_one, &h_one, sizeof(Scalar), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(ws.d_const_zero, &h_zero, sizeof(Scalar), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(ws.d_const_neg_one, &h_neg_one, sizeof(Scalar), hipMemcpyHostToDevice));
        }

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

        // Randomized truncated SVD workspace (GPU QR)
        {
            int rsvd_r = chi_max_ + rsvd_oversampling_;
            int rsvd_m = svd_max_m;  // chi*d
            int rsvd_n = svd_max_n;  // d*chi
            ws.d_rsvd_omega = nullptr;
            ws.d_rsvd_Y = nullptr;
            ws.d_rsvd_Q = nullptr;
            ws.d_rsvd_B = nullptr;
            ws.d_rsvd_ipiv = nullptr;
            ws.d_rsvd_U_full = nullptr;
            HIP_CHECK(hipMalloc(&ws.d_rsvd_omega,  (size_t)rsvd_n * rsvd_r * sizeof(Scalar)));
            HIP_CHECK(hipMalloc(&ws.d_rsvd_Y,     (size_t)rsvd_m * rsvd_r * sizeof(Scalar)));
            HIP_CHECK(hipMalloc(&ws.d_rsvd_Q,     (size_t)rsvd_m * rsvd_r * sizeof(Scalar)));
            HIP_CHECK(hipMalloc(&ws.d_rsvd_B,     (size_t)rsvd_r * rsvd_n * sizeof(Scalar)));
            HIP_CHECK(hipMalloc(&ws.d_rsvd_ipiv,  (size_t)rsvd_r * sizeof(Scalar)));
            HIP_CHECK(hipMalloc(&ws.d_rsvd_U_full,(size_t)rsvd_m * rsvd_r * sizeof(Scalar)));
            ws.h_rsvd_B.resize((size_t)rsvd_r * rsvd_n);
            ws.h_rsvd_U_small.resize((size_t)rsvd_r * rsvd_r);

            // Query SVD workspace for the smaller matrix (rsvd_r x rsvd_n)
            {
                int sm = rsvd_r, sn = rsvd_n;
                int sk = std::min(sm, sn);
                int svd_lwork_query = -1;
                Scalar svd_work_opt;
                int svd_info;
                const char jobu = 'S', jobvt = 'S';
                Traits::lapack_gesvd(&jobu, &jobvt, &sm, &sn, nullptr, &sm, nullptr,
                        nullptr, &sm, nullptr, &sk, &svd_work_opt, &svd_lwork_query,
                        ws.h_svd_rwork.empty() ? nullptr : ws.h_svd_rwork.data(), &svd_info);
                int svd_opt;
                if constexpr (Traits::is_complex) {
                    svd_opt = (int)Traits::real_part(svd_work_opt) + 1;
                } else {
                    svd_opt = (int)svd_work_opt + 1;
                }
                // Ensure h_svd_work is large enough for both full and reduced SVD
                if ((int)ws.h_svd_work.size() < svd_opt) {
                    ws.h_svd_work.resize(svd_opt);
                }
            }
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
        // h_batch_*_pinned no longer allocated (GPU kernel pointer setup)
        if (ws.d_heff_batch_A) hipFree(ws.d_heff_batch_A);
        if (ws.d_heff_batch_C) hipFree(ws.d_heff_batch_C);
        if (ws.d_dot_result) hipFree(ws.d_dot_result);
        if (ws.d_nrm2_result) hipFree(ws.d_nrm2_result);
        if (ws.d_neg_alpha) hipFree(ws.d_neg_alpha);
        if (ws.d_neg_overlap) hipFree(ws.d_neg_overlap);
        if (ws.d_inv_nrm) hipFree(ws.d_inv_nrm);
        if (ws.d_alpha_dev) hipFree(ws.d_alpha_dev);
        if (ws.d_beta_dev) hipFree(ws.d_beta_dev);
        if (ws.d_neg_beta_scalars) hipFree(ws.d_neg_beta_scalars);
        if (ws.d_const_one) hipFree(ws.d_const_one);
        if (ws.d_const_zero) hipFree(ws.d_const_zero);
        if (ws.d_const_neg_one) hipFree(ws.d_const_neg_one);
        if (ws.d_svd_A) hipFree(ws.d_svd_A);
        if (ws.d_svd_U) hipFree(ws.d_svd_U);
        if (ws.d_svd_S) hipFree(ws.d_svd_S);
        if (ws.d_svd_Vh) hipFree(ws.d_svd_Vh);
        if (ws.d_svd_E) hipFree(ws.d_svd_E);
        if (ws.d_svd_info) hipFree(ws.d_svd_info);
        if (ws.d_svd_work) hipFree(ws.d_svd_work);
        if (ws.d_rsvd_omega) hipFree(ws.d_rsvd_omega);
        if (ws.d_rsvd_Y) hipFree(ws.d_rsvd_Y);
        if (ws.d_rsvd_Q) hipFree(ws.d_rsvd_Q);
        if (ws.d_rsvd_B) hipFree(ws.d_rsvd_B);
        if (ws.d_rsvd_ipiv) hipFree(ws.d_rsvd_ipiv);
        if (ws.d_rsvd_U_full) hipFree(ws.d_rsvd_U_full);
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
// Initialize boundary V = ones (before any merge, V is identity)
// ============================================================================

template<typename Scalar>
void PDMRGGPU<Scalar>::initialize_boundary_states() {
    int n_boundaries = n_segments_ - 1;
    boundary_states_.resize(n_boundaries);
    for (int b = 0; b < n_boundaries; b++) {
        int bsite = boundary_bonds_[b];
        int chi = bond_dims_[bsite + 1];  // bond between bsite and bsite+1
        boundary_states_[b].chi = chi;
        boundary_states_[b].V.assign(chi, RealType(1.0));  // V = ones initially
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

        // Cache A and C pointers (constant for a given site) — GPU kernel, no DMA
        if (ws.heff_cached_site != site) {
            hipLaunchKernelGGL(setup_heff_A_ptrs<Scalar>, dim3(1), dim3(batch_count), 0, streams_[si],
                               ws.d_heff_batch_A, L_env, cL, dd, batch_count);
            hipLaunchKernelGGL(setup_heff_C_ptrs<Scalar>, dim3(1), dim3(batch_count), 0, streams_[si],
                               ws.d_heff_batch_C, T1, cL * cR, batch_count);
            ws.heff_cached_site = site;
        }

        // B pointers change per call (d_theta_in varies) — GPU kernel, no DMA race
        hipLaunchKernelGGL(setup_heff_B_ptrs<Scalar>, dim3(1), dim3(batch_count), 0, streams_[si],
                           ws.d_batch_B, const_cast<Scalar*>(d_theta_in), cL, d, dd, batch_count);

        ROCBLAS_CHECK(Traits::gemm_batched(handles_[si],
            Traits::op_t, rocblas_operation_none,
            cL, cR, cL,
            &one,
            (const Scalar**)ws.d_heff_batch_A, cL * D,
            (const Scalar**)ws.d_batch_B, cL * dd,
            &zero_val,
            ws.d_heff_batch_C, cL,
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

    // Step 3: Batched GEMMs — T2 × R_env (D batched calls of d² each)
    {
        int total = D * dd;
        Scalar* h_A3[256], *h_B3[256], *h_C3[256];
        for (int n = 0; n < D; n++) {
            for (int b = 0; b < dd; b++) {
                int idx = n * dd + b;
                int s1p = b / d, s2p = b % d;
                h_A3[idx] = T2 + (size_t)(n * dd + s1p * d + s2p) * cL * cR;
                h_B3[idx] = R_env + n * cR;
                h_C3[idx] = d_result + s1p * cL + s2p * cL * d;
            }
        }
        HIP_CHECK(hipMemcpyAsync(ws.d_batch_A, h_A3, total * sizeof(Scalar*), hipMemcpyHostToDevice, streams_[si]));
        HIP_CHECK(hipMemcpyAsync(ws.d_batch_B, h_B3, total * sizeof(Scalar*), hipMemcpyHostToDevice, streams_[si]));
        HIP_CHECK(hipMemcpyAsync(ws.d_batch_C, h_C3, total * sizeof(Scalar*), hipMemcpyHostToDevice, streams_[si]));

        ROCBLAS_CHECK(Traits::gemm_batched(handles_[si],
            rocblas_operation_none, rocblas_operation_none,
            cL, cR, cR, &one,
            (const Scalar**)ws.d_batch_A, cL,
            (const Scalar**)ws.d_batch_B, cR * D,
            &zero_val,
            ws.d_batch_C, cL * dd, dd));

        for (int n = 1; n < D; n++) {
            ROCBLAS_CHECK(Traits::gemm_batched(handles_[si],
                rocblas_operation_none, rocblas_operation_none,
                cL, cR, cR, &one,
                (const Scalar**)(ws.d_batch_A + n * dd), cL,
                (const Scalar**)(ws.d_batch_B + n * dd), cR * D,
                &one,
                (const Scalar**)(ws.d_batch_C + n * dd), cL * dd, dd));
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
        int batch_count = D * d;
        hipLaunchKernelGGL(setup_lenv_ptrs<Scalar>, dim3(1), dim3(batch_count), 0, streams_[si],
                           ws.d_batch_A, ws.d_batch_B, ws.d_batch_C,
                           L_env, A, V, chi_in, chi_out, d, batch_count);
        ROCBLAS_CHECK(Traits::gemm_batched(handles_[si],
            Traits::op_t, rocblas_operation_none,
            chi_in, chi_out, chi_in,
            &one,
            (const Scalar**)ws.d_batch_A, chi_in * D,
            (const Scalar**)ws.d_batch_B, chi_in * d,
            &zero_val,
            ws.d_batch_C, chi_in,
            batch_count));
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
        int batch_count = D * d;
        hipLaunchKernelGGL(setup_renv_ptrs<Scalar>, dim3(1), dim3(batch_count), 0, streams_[si],
                           ws.d_batch_A, ws.d_batch_B, ws.d_batch_C,
                           A, R_env, V, chi_in, chi_out, d, batch_count);
        ROCBLAS_CHECK(Traits::gemm_batched(handles_[si],
            rocblas_operation_none, rocblas_operation_none,
            chi_out, chi_in, chi_in,
            &one,
            (const Scalar**)ws.d_batch_A, chi_out * d,
            (const Scalar**)ws.d_batch_B, chi_in * D,
            &zero_val,
            ws.d_batch_C, chi_out,
            batch_count));
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

    // v[0] = theta / ||theta|| (host pointer mode for initial setup)
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

    double prev_energy = 1e30;
    int iter;
    int last_synced_iter = -1;

    for (iter = 0; iter < max_iter; iter++) {
        Scalar* d_vi = d_lanczos_v + (size_t)iter * n;

        // w = H|v_i> (apply_heff uses host pointer mode internally)
        if (lanczos_use_1site_)
            apply_heff_single_site(site, d_vi, ws.d_heff_result, si);
        else
            apply_heff_two_site(site, d_vi, ws.d_heff_result, si);

        // Switch to device pointer mode for scalar operations
        ROCBLAS_CHECK(rocblas_set_pointer_mode(handles_[si], rocblas_pointer_mode_device));

        // alpha_i = <v_i|w> → device
        ROCBLAS_CHECK(Traits::dot(handles_[si], n, d_vi, 1, ws.d_heff_result, 1, ws.d_dot_result));

        // Process alpha: store to d_alpha_dev[iter], compute d_neg_alpha
        hipLaunchKernelGGL(lanczos_process_alpha_kernel<Scalar>, dim3(1), dim3(1), 0, streams_[si],
                           ws.d_dot_result, ws.d_neg_alpha, ws.d_alpha_dev, iter);

        // w -= alpha_i * v_i (device pointer)
        ROCBLAS_CHECK(Traits::axpy(handles_[si], n, ws.d_neg_alpha, d_vi, 1, ws.d_heff_result, 1));

        // w -= beta_{i-1} * v_{i-1} (device pointer: pre-stored by previous iter)
        if (iter > 0) {
            ROCBLAS_CHECK(Traits::axpy(handles_[si], n,
                ws.d_neg_beta_scalars + (iter - 1),
                d_lanczos_v + (size_t)(iter - 1) * n, 1,
                ws.d_heff_result, 1));
        }

        // Full reorthogonalization (device pointer mode for gemv constants)
        if (iter > 0) {
            ROCBLAS_CHECK(Traits::gemv(handles_[si], Traits::op_h,
                n, iter + 1, ws.d_const_one,
                d_lanczos_v, n,
                ws.d_heff_result, 1,
                ws.d_const_zero, ws.d_ritz_coeffs, 1));
            ROCBLAS_CHECK(Traits::gemv(handles_[si], rocblas_operation_none,
                n, iter + 1, ws.d_const_neg_one,
                d_lanczos_v, n,
                ws.d_ritz_coeffs, 1,
                ws.d_const_one, ws.d_heff_result, 1));
        } else {
            ROCBLAS_CHECK(Traits::dot(handles_[si], n, d_lanczos_v, 1, ws.d_heff_result, 1, ws.d_dot_result));
            hipLaunchKernelGGL(negate_scalar_kernel<Scalar>, dim3(1), dim3(1), 0, streams_[si],
                               ws.d_dot_result, ws.d_neg_overlap);
            ROCBLAS_CHECK(Traits::axpy(handles_[si], n, ws.d_neg_overlap, d_lanczos_v, 1, ws.d_heff_result, 1));
        }

        // beta_i = ||w|| → device
        ROCBLAS_CHECK(Traits::nrm2(handles_[si], n, ws.d_heff_result, 1, ws.d_nrm2_result));

        // Process beta: store, compute 1/beta, store -beta as Scalar
        hipLaunchKernelGGL(lanczos_process_beta_kernel<Scalar>, dim3(1), dim3(1), 0, streams_[si],
                           ws.d_nrm2_result, ws.d_inv_nrm, ws.d_beta_dev, ws.d_neg_beta_scalars, iter);

        // v_{i+1} = w / beta_i (device pointer for scal)
        if (iter + 1 < max_iter) {
            Scalar* d_vip1 = d_lanczos_v + (size_t)(iter + 1) * n;
            HIP_CHECK(hipMemcpyAsync(d_vip1, ws.d_heff_result, n * sizeof(Scalar), hipMemcpyDeviceToDevice, streams_[si]));
            ROCBLAS_CHECK(Traits::scal_real(handles_[si], n, ws.d_inv_nrm, d_vip1, 1));
        }

        // Switch back to host pointer mode (needed by apply_heff next iteration)
        ROCBLAS_CHECK(rocblas_set_pointer_mode(handles_[si], rocblas_pointer_mode_host));

        // Convergence check every 3 iterations after iter >= 4
        // This is the ONLY sync point in the inner loop
        if (iter >= 4 && iter % 3 == 0) {
            HIP_CHECK(hipStreamSynchronize(streams_[si]));

            // Bulk copy alpha and beta from device to host
            int n_copy = iter + 1;
            HIP_CHECK(hipMemcpy(h_alpha.data(), ws.d_alpha_dev, n_copy * sizeof(double), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(h_beta.data(), ws.d_beta_dev, n_copy * sizeof(double), hipMemcpyDeviceToHost));
            last_synced_iter = iter;

            // Check if any beta was near zero (invariant subspace found)
            bool early_break = false;
            for (int j = 0; j <= iter; j++) {
                if (h_beta[j] < tol_lanczos) {
                    iter = j + 1;
                    early_break = true;
                    break;
                }
            }
            if (early_break) break;

            // Eigenvalue convergence check
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
    }

    int niter = iter;

    // Ensure stream is synchronized before reading results
    HIP_CHECK(hipStreamSynchronize(streams_[si]));

    // Copy any remaining alpha/beta values we haven't synced yet
    if (last_synced_iter < niter - 1) {
        HIP_CHECK(hipMemcpy(h_alpha.data(), ws.d_alpha_dev, niter * sizeof(double), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_beta.data(), ws.d_beta_dev, niter * sizeof(double), hipMemcpyDeviceToHost));
    }

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

    // Reconstruct ground state: |theta> = sum_i c[i] |v_i> (host pointer mode)
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

    Scalar* h_U_data = nullptr;
    RealType* h_S_data = nullptr;
    Scalar* h_Vh_data = nullptr;
    bool gpu_svd_path = false;

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

        // GPU SVD: U, S, Vh already on device. Only download S for truncation check.
        HIP_CHECK(hipMemcpy(ws.h_svd_S.data(), ws.d_svd_S, full_k * sizeof(RealType), hipMemcpyDeviceToHost));
        h_S_data = ws.h_svd_S.data();
        gpu_svd_path = true;
    }

    // Truncation
    int new_k = k;
    for (int i = 0; i < new_k; i++) {
        if (h_S_data[i] < 1e-14) { new_k = i; break; }
    }
    if (new_k == 0) new_k = 1;

    if (gpu_svd_path) {
        // GPU SVD path: U, S, Vh all on device already
        // d_svd_U (m × full_k, lda=m), d_svd_S (full_k), d_svd_Vh (full_k × n_svd, lda=full_k)

        if (direction == 'R') {
            // MPS[site] = U[:, :new_k]
            allocate_mps_tensor(site, cL, new_k);
            HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site], ws.d_svd_U,
                        (size_t)m * new_k * sizeof(Scalar), hipMemcpyDeviceToDevice, streams_[si]));
            // MPS[site+1] = diag(S) @ Vh[:new_k, :] — scale rows on GPU
            allocate_mps_tensor(site + 1, new_k, cR);
            scale_rows_by_real(ws.d_svd_Vh, full_k, ws.d_svd_S,
                               d_mps_tensors_[site + 1], new_k, new_k, n_svd, streams_[si]);
        } else {
            // MPS[site] = U[:, :new_k] @ diag(S) — scale columns on GPU
            allocate_mps_tensor(site, cL, new_k);
            scale_columns_by_real(ws.d_svd_U, m, ws.d_svd_S,
                                  d_mps_tensors_[site], m, m, new_k, streams_[si]);
            // MPS[site+1] = Vh[:new_k, :] — extract rows with stride when truncated
            allocate_mps_tensor(site + 1, new_k, cR);
            if (new_k == full_k) {
                HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site + 1], ws.d_svd_Vh,
                            (size_t)full_k * n_svd * sizeof(Scalar), hipMemcpyDeviceToDevice, streams_[si]));
            } else {
                // Vh is (full_k × n_svd) column-major with lda=full_k; extract first new_k rows
                HIP_CHECK(hipMemcpy2DAsync(
                    d_mps_tensors_[site + 1], new_k * sizeof(Scalar),      // dst, dpitch
                    ws.d_svd_Vh,              full_k * sizeof(Scalar),      // src, spitch
                    new_k * sizeof(Scalar),   n_svd,                        // width, height
                    hipMemcpyDeviceToDevice, streams_[si]));
            }
        }
    } else {
        // CPU SVD path: U, S, Vh on host. Upload S + raw factors, scale on GPU.
        HIP_CHECK(hipMemcpyAsync(ws.d_svd_S, h_S_data, new_k * sizeof(RealType),
                                  hipMemcpyHostToDevice, streams_[si]));

        if (direction == 'R') {
            // MPS[site] = U[:, :new_k]
            allocate_mps_tensor(site, cL, new_k);
            HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site], h_U_data,
                        m * new_k * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));

            // MPS[site+1] = diag(S) @ Vh[:new_k, :] — scale rows of Vh by S on GPU
            if (new_k == full_k) {
                HIP_CHECK(hipMemcpyAsync(ws.d_svd_work, h_Vh_data,
                            (size_t)full_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));
            } else {
                // Pack contiguous new_k rows from leading-dim full_k layout
                for (int j = 0; j < n_svd; j++)
                    for (int i = 0; i < new_k; i++)
                        ws.h_svd_tmp[i + j * new_k] = h_Vh_data[i + j * full_k];
                HIP_CHECK(hipMemcpyAsync(ws.d_svd_work, ws.h_svd_tmp.data(),
                            (size_t)new_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));
            }
            allocate_mps_tensor(site + 1, new_k, cR);
            int vh_ld = (new_k == full_k) ? full_k : new_k;
            scale_rows_by_real(ws.d_svd_work, vh_ld, ws.d_svd_S,
                               d_mps_tensors_[site + 1], new_k, new_k, n_svd, streams_[si]);

        } else {  // direction == 'L'
            // MPS[site] = U[:, :new_k] @ diag(S) — scale columns of U by S on GPU
            HIP_CHECK(hipMemcpyAsync(ws.d_svd_work, h_U_data,
                        (size_t)m * new_k * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));
            allocate_mps_tensor(site, cL, new_k);
            scale_columns_by_real(ws.d_svd_work, m, ws.d_svd_S,
                                  d_mps_tensors_[site], m, m, new_k, streams_[si]);

            // MPS[site+1] = Vh[:new_k, :]
            allocate_mps_tensor(site + 1, new_k, cR);
            if (new_k == full_k) {
                HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site + 1], h_Vh_data,
                            (size_t)full_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));
            } else {
                for (int j = 0; j < n_svd; j++)
                    for (int i = 0; i < new_k; i++)
                        ws.h_svd_tmp[i + j * new_k] = h_Vh_data[i + j * full_k];
                HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site + 1], ws.h_svd_tmp.data(),
                            (size_t)new_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));
            }
        }
    }

    bond_dims_[site + 1] = new_k;

    // Invalidate heff pointer cache (bond dims changed)
    ws.heff_cached_site = -1;
}

// ============================================================================
// Randomized truncated SVD split (Halko-Martinsson-Tropp, GPU QR)
// ============================================================================

template<typename Scalar>
void PDMRGGPU<Scalar>::rsvd_split(int site, Scalar* d_theta, char direction, int si) {
    int cL = chi_L(site);
    int cR = chi_R(site + 1);
    int m = cL * d_;
    int n_svd = d_ * cR;
    int full_k = std::min(m, n_svd);
    int k = std::min(full_k, chi_max_);
    auto& ws = workspaces_[si];

    // If matrix is small enough, fall back to full SVD (rSVD overhead not worth it)
    if (full_k <= k + rsvd_oversampling_ || m <= 2 * k) {
        svd_split(site, d_theta, direction, si);
        return;
    }

    int r = k + rsvd_oversampling_;  // projection rank

    // Step 1: Generate random Omega (n_svd x r) on CPU and upload
    {
        std::vector<Scalar> h_omega(n_svd * r);
        for (int i = 0; i < n_svd * r; i++) {
            h_omega[i] = Traits::random_val();
        }
        HIP_CHECK(hipMemcpyAsync(ws.d_rsvd_omega, h_omega.data(),
                            n_svd * r * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));
    }

    // Step 2: Y = theta @ Omega on GPU  (m x n_svd) @ (n_svd x r) -> (m x r)
    {
        Scalar one = Traits::one(), zero_val = Traits::zero();
        ROCBLAS_CHECK(Traits::gemm(handles_[si],
            rocblas_operation_none, rocblas_operation_none,
            m, r, n_svd, &one,
            d_theta, m,
            ws.d_rsvd_omega, n_svd,
            &zero_val,
            ws.d_rsvd_Y, m));
        HIP_CHECK(hipStreamSynchronize(streams_[si]));
    }

    // Step 3: QR factorization of Y on GPU -> Q (m x r) stays on device
    HIP_CHECK(hipMemcpyAsync(ws.d_rsvd_Q, ws.d_rsvd_Y, (size_t)m * r * sizeof(Scalar),
                              hipMemcpyDeviceToDevice, streams_[si]));
    ROCBLAS_CHECK(Traits::rocsolver_geqrf(handles_[si], m, r, ws.d_rsvd_Q, m, ws.d_rsvd_ipiv));
    ROCBLAS_CHECK(Traits::rocsolver_orgqr(handles_[si], m, r, r, ws.d_rsvd_Q, m, ws.d_rsvd_ipiv));
    HIP_CHECK(hipStreamSynchronize(streams_[si]));

    // Step 4: B = Q^H @ theta on GPU  (r x m) @ (m x n_svd) -> (r x n_svd)
    {
        Scalar one = Traits::one(), zero_val = Traits::zero();
        ROCBLAS_CHECK(Traits::gemm(handles_[si],
            Traits::op_h, rocblas_operation_none,
            r, n_svd, m, &one,
            ws.d_rsvd_Q, m,
            d_theta, m,
            &zero_val,
            ws.d_rsvd_B, r));
        HIP_CHECK(hipStreamSynchronize(streams_[si]));
    }

    // Step 5: Copy B to host, compute SVD of B (r x n_svd) — much smaller than (m x n_svd)
    HIP_CHECK(hipMemcpy(ws.h_rsvd_B.data(), ws.d_rsvd_B, r * n_svd * sizeof(Scalar), hipMemcpyDeviceToHost));

    int small_k = std::min(r, n_svd);
    {
        int lwork = (int)ws.h_svd_work.size();
        int info;
        const char jobu = 'S', jobvt = 'S';
        // U_small: (r x small_k), S: (small_k), Vh: (small_k x n_svd)
        Traits::lapack_gesvd(&jobu, &jobvt, &r, &n_svd, ws.h_rsvd_B.data(), &r, ws.h_svd_S.data(),
                ws.h_rsvd_U_small.data(), &r, ws.h_svd_Vh.data(), &small_k,
                ws.h_svd_work.data(), &lwork,
                ws.h_svd_rwork.empty() ? nullptr : ws.h_svd_rwork.data(), &info);
        if (info != 0) {
            svd_split(site, d_theta, direction, si);
            return;
        }
    }

    // Step 6: Upload U_small to GPU, compute U_full = Q @ U_small on GPU
    //   Q is (m x r) on device, U_small is (r x small_k) on host -> U_full (m x small_k)
    {
        // Upload U_small to device (reuse d_rsvd_B as temp — it's no longer needed)
        HIP_CHECK(hipMemcpyAsync(ws.d_rsvd_B, ws.h_rsvd_U_small.data(),
                            (size_t)r * small_k * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));
        Scalar one = Traits::one(), zero_val = Traits::zero();
        ROCBLAS_CHECK(Traits::gemm(handles_[si],
            rocblas_operation_none, rocblas_operation_none,
            m, small_k, r, &one,
            ws.d_rsvd_Q, m,
            ws.d_rsvd_B, r,
            &zero_val,
            ws.d_rsvd_U_full, m));
        HIP_CHECK(hipStreamSynchronize(streams_[si]));
    }

    // Now: U_full (m x small_k) is on GPU at ws.d_rsvd_U_full
    //      S (small_k) and Vh (small_k x n_svd) are on host
    RealType* h_S_data = ws.h_svd_S.data();
    Scalar* h_Vh_data = ws.h_svd_Vh.data();

    // Truncation
    int new_k = k;
    for (int i = 0; i < new_k; i++) {
        if (h_S_data[i] < 1e-14) { new_k = i; break; }
    }
    if (new_k == 0) new_k = 1;

    // Upload S to device for GPU-side scaling
    HIP_CHECK(hipMemcpyAsync(ws.d_svd_S, h_S_data, new_k * sizeof(RealType),
                              hipMemcpyHostToDevice, streams_[si]));

    if (direction == 'R') {
        // MPS[site] = U_full[:, :new_k]  (U_full already on GPU)
        allocate_mps_tensor(site, cL, new_k);
        if (new_k == small_k) {
            HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site], ws.d_rsvd_U_full,
                                (size_t)m * new_k * sizeof(Scalar), hipMemcpyDeviceToDevice, streams_[si]));
        } else {
            // Column subset: first new_k columns are contiguous in column-major
            HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site], ws.d_rsvd_U_full,
                                (size_t)m * new_k * sizeof(Scalar), hipMemcpyDeviceToDevice, streams_[si]));
        }

        // MPS[site+1] = diag(S) @ Vh[:new_k, :] — scale rows of Vh by S on GPU
        // Vh is on host with leading dim small_k, pack to new_k and upload
        if (new_k == small_k) {
            HIP_CHECK(hipMemcpyAsync(ws.d_svd_work, h_Vh_data,
                        (size_t)small_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));
        } else {
            for (int j = 0; j < n_svd; j++)
                for (int i = 0; i < new_k; i++)
                    ws.h_svd_tmp[i + j * new_k] = h_Vh_data[i + j * small_k];
            HIP_CHECK(hipMemcpyAsync(ws.d_svd_work, ws.h_svd_tmp.data(),
                        (size_t)new_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));
        }
        allocate_mps_tensor(site + 1, new_k, cR);
        int vh_ld = (new_k == small_k) ? small_k : new_k;
        scale_rows_by_real(ws.d_svd_work, vh_ld, ws.d_svd_S,
                           d_mps_tensors_[site + 1], new_k, new_k, n_svd, streams_[si]);

    } else {
        // MPS[site] = U_full[:, :new_k] @ diag(S) — scale columns on GPU
        // U_full already on GPU at d_rsvd_U_full
        allocate_mps_tensor(site, cL, new_k);
        scale_columns_by_real(ws.d_rsvd_U_full, m, ws.d_svd_S,
                              d_mps_tensors_[site], m, m, new_k, streams_[si]);

        // MPS[site+1] = Vh[:new_k, :]
        allocate_mps_tensor(site + 1, new_k, cR);
        if (new_k == small_k) {
            HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site + 1], h_Vh_data,
                                (size_t)small_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));
        } else {
            for (int j = 0; j < n_svd; j++)
                for (int i = 0; i < new_k; i++)
                    ws.h_svd_tmp[i + j * new_k] = h_Vh_data[i + j * small_k];
            HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site + 1], ws.h_svd_tmp.data(),
                                (size_t)new_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));
        }
    }

    bond_dims_[site + 1] = new_k;

    // Invalidate heff pointer cache (bond dims changed)
    ws.heff_cached_site = -1;
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
    // No sync needed: form_theta_two_site uses the same stream/handle,
    // rocBLAS operations are ordered within a stream.
    double energy = lanczos_eigensolver(site, ws.d_theta, theta_size, si);
    // Lanczos already syncs internally (nrm2/dot read host results).
    if (use_rsvd_)
        rsvd_split(site, ws.d_theta, direction, si);
    else
        svd_split(site, ws.d_theta, direction, si);
    // svd_split/rsvd_split syncs internally.

    return energy;
}

// ============================================================================
// Single-site apply_heff: H_eff|θ⟩ for one MPS tensor
// Same 3-step GEMM as dmrg-gpu: L_env × θ × W × R_env
// theta shape: (cL, d, cR), result shape: (cL, d, cR)
// ============================================================================

template<typename Scalar>
void PDMRGGPU<Scalar>::apply_heff_single_site(int site, const Scalar* d_theta_in,
                                               Scalar* d_result, int si) {
    int cL = chi_L(site);
    int cR = chi_R(site);
    int D = D_mpo_, d = d_;
    Scalar one = Traits::one(), zero_val = Traits::zero();
    auto& ws = workspaces_[si];

    Scalar* L_env = d_L_envs_[site];
    Scalar* R_env = d_R_envs_[site + 1];
    Scalar* W_mat = d_W_left_[site];
    Scalar* V = ws.d_T1;
    Scalar* U = ws.d_T2;

    // Step 1: V_ws[a',b] = L_w^T[a',a] * theta_s[a,b]  (D*d batched GEMMs)
    {
        int batch_count = D * d;
        Scalar* h_A[256], *h_B[256], *h_C[256]; // D*d ≤ 256 for practical MPOs
        for (int w = 0; w < D; w++)
            for (int s = 0; s < d; s++) {
                int ws = w * d + s;
                h_A[ws] = L_env + w * cL;
                h_B[ws] = const_cast<Scalar*>(d_theta_in) + s * cL;
                h_C[ws] = V + ws * cL * cR;
            }
        HIP_CHECK(hipMemcpyAsync(ws.d_batch_A, h_A, batch_count * sizeof(Scalar*),
                                  hipMemcpyHostToDevice, streams_[si]));
        HIP_CHECK(hipMemcpyAsync(ws.d_batch_B, h_B, batch_count * sizeof(Scalar*),
                                  hipMemcpyHostToDevice, streams_[si]));
        HIP_CHECK(hipMemcpyAsync(ws.d_batch_C, h_C, batch_count * sizeof(Scalar*),
                                  hipMemcpyHostToDevice, streams_[si]));
        ROCBLAS_CHECK(Traits::gemm_batched(handles_[si],
            Traits::op_t, rocblas_operation_none,
            cL, cR, cL,
            &one,
            (const Scalar**)ws.d_batch_A, cL * D,
            (const Scalar**)ws.d_batch_B, cL * d,
            &zero_val,
            ws.d_batch_C, cL,
            batch_count));
    }

    // Step 2: U = V * W_matrix  (single dense GEMM)
    ROCBLAS_CHECK(Traits::gemm(handles_[si],
        rocblas_operation_none, rocblas_operation_none,
        cL * cR, d * D, D * d,
        &one,
        V, cL * cR,
        W_mat, D * d,
        &zero_val,
        U, cL * cR));

    // Step 3: Batched GEMMs — contract R_env (D batched calls of d each)
    {
        int total = D * d;
        Scalar* h_A3[256], *h_B3[256], *h_C3[256];
        for (int wp = 0; wp < D; wp++) {
            for (int sp = 0; sp < d; sp++) {
                int idx = wp * d + sp;
                h_A3[idx] = U + (size_t)idx * cL * cR;
                h_B3[idx] = R_env + wp * cR;
                h_C3[idx] = d_result + sp * cL;
            }
        }
        HIP_CHECK(hipMemcpyAsync(ws.d_batch_A, h_A3, total * sizeof(Scalar*), hipMemcpyHostToDevice, streams_[si]));
        HIP_CHECK(hipMemcpyAsync(ws.d_batch_B, h_B3, total * sizeof(Scalar*), hipMemcpyHostToDevice, streams_[si]));
        HIP_CHECK(hipMemcpyAsync(ws.d_batch_C, h_C3, total * sizeof(Scalar*), hipMemcpyHostToDevice, streams_[si]));

        ROCBLAS_CHECK(Traits::gemm_batched(handles_[si],
            rocblas_operation_none, rocblas_operation_none,
            cL, cR, cR, &one,
            (const Scalar**)ws.d_batch_A, cL,
            (const Scalar**)ws.d_batch_B, cR * D,
            &zero_val,
            ws.d_batch_C, cL * d, d));

        for (int wp = 1; wp < D; wp++) {
            ROCBLAS_CHECK(Traits::gemm_batched(handles_[si],
                rocblas_operation_none, rocblas_operation_none,
                cL, cR, cR, &one,
                (const Scalar**)(ws.d_batch_A + wp * d), cL,
                (const Scalar**)(ws.d_batch_B + wp * d), cR * D,
                &one,
                (const Scalar**)(ws.d_batch_C + wp * d), cL * d, d));
        }
    }
}

// ============================================================================
// Single-site SVD split: decompose theta and shift canonical center
// Direction 'R': theta(cL*d, cR) → U=MPS[site], S*Vh absorbed into MPS[site+1]
// Direction 'L': theta(cL, d*cR) → Vh=MPS[site], U*S absorbed into MPS[site-1]
// ============================================================================

template<typename Scalar>
void PDMRGGPU<Scalar>::svd_split_single_site(int site, Scalar* d_theta, char direction, int si) {
    int cL = chi_L(site);
    int cR = chi_R(site);
    auto& ws = workspaces_[si];

    int m, n_svd;
    if (direction == 'R') { m = cL * d_; n_svd = cR; }
    else                  { m = cL;      n_svd = d_ * cR; }
    int full_k = std::min(m, n_svd);
    int k = std::min(full_k, chi_max_);

    Scalar* h_U_data = nullptr;
    RealType* h_S_data = nullptr;
    Scalar* h_Vh_data = nullptr;

    // CPU SVD (default path, faster for chi < 200)
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
        HIP_CHECK(hipMemcpy(ws.h_svd_S.data(), ws.d_svd_S, full_k * sizeof(RealType), hipMemcpyDeviceToHost));
        h_S_data = ws.h_svd_S.data();
    }

    // Truncation
    int new_k = k;
    for (int i = 0; i < new_k; i++) {
        if (h_S_data[i] < 1e-14) { new_k = i; break; }
    }
    if (new_k == 0) new_k = 1;

    Scalar one = Traits::one(), zero_val = Traits::zero();

    if (direction == 'R') {
        int new_chi_R = new_k;

        if (use_cpu_svd_) {
            // Upload U[:, :new_k] as MPS[site]
            allocate_mps_tensor(site, cL, new_chi_R);
            HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site], h_U_data,
                        m * new_k * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));

            // Compute S*Vh on CPU, then absorb into MPS[site+1] via GEMM
            for (int j = 0; j < n_svd; j++)
                for (int i = 0; i < new_k; i++)
                    ws.h_svd_tmp[i + j * new_k] = Traits::scale_by_real(h_S_data[i], h_Vh_data[i + j * full_k]);

            HIP_CHECK(hipMemcpyAsync(ws.d_svd_work, ws.h_svd_tmp.data(),
                        new_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));
        } else {
            // GPU path: U and Vh already on device
            allocate_mps_tensor(site, cL, new_chi_R);
            HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site], ws.d_svd_U,
                        (size_t)m * new_k * sizeof(Scalar), hipMemcpyDeviceToDevice, streams_[si]));
            // Scale rows of Vh by S → ws.d_svd_work
            scale_rows_by_real(ws.d_svd_Vh, full_k, ws.d_svd_S,
                               ws.d_svd_work, new_k, new_k, n_svd, streams_[si]);
        }

        // Absorb S*Vh into MPS[site+1]: new = (S*Vh) @ old_MPS[site+1]
        if (site + 1 < L_) {
            int next_cR = chi_R(site + 1);
            ROCBLAS_CHECK(Traits::gemm(handles_[si],
                rocblas_operation_none, rocblas_operation_none,
                new_k, d_ * next_cR, cR, &one,
                ws.d_svd_work, new_k,
                d_mps_tensors_[site + 1], cR, &zero_val,
                ws.d_T1, new_k));
            allocate_mps_tensor(site + 1, new_chi_R, next_cR);
            HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site + 1], ws.d_T1,
                        (size_t)new_k * d_ * next_cR * sizeof(Scalar),
                        hipMemcpyDeviceToDevice, streams_[si]));
        }
        bond_dims_[site + 1] = new_chi_R;

    } else {  // direction == 'L'
        int new_chi_L = new_k;

        if (use_cpu_svd_) {
            // Upload Vh[:new_k, :] as MPS[site]
            allocate_mps_tensor(site, new_chi_L, cR);
            if (new_k == full_k) {
                HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site], h_Vh_data,
                            full_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));
            } else {
                for (int j = 0; j < n_svd; j++)
                    for (int i = 0; i < new_chi_L; i++)
                        ws.h_svd_tmp[i + j * new_chi_L] = h_Vh_data[i + j * full_k];
                HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site], ws.h_svd_tmp.data(),
                            new_chi_L * n_svd * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));
            }

            // Compute U*S on CPU
            for (int j = 0; j < new_k; j++)
                for (int i = 0; i < m; i++)
                    ws.h_svd_tmp[i + j * m] = Traits::scale_by_real(h_S_data[j], h_U_data[i + j * m]);

            HIP_CHECK(hipMemcpyAsync(ws.d_svd_work, ws.h_svd_tmp.data(),
                        m * new_k * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));
        } else {
            // GPU path
            allocate_mps_tensor(site, new_chi_L, cR);
            if (new_k == full_k) {
                HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site], ws.d_svd_Vh,
                            (size_t)full_k * n_svd * sizeof(Scalar), hipMemcpyDeviceToDevice, streams_[si]));
            } else {
                HIP_CHECK(hipMemcpy2DAsync(
                    d_mps_tensors_[site], new_k * sizeof(Scalar),
                    ws.d_svd_Vh, full_k * sizeof(Scalar),
                    new_k * sizeof(Scalar), n_svd,
                    hipMemcpyDeviceToDevice, streams_[si]));
            }
            // Scale columns of U by S → ws.d_svd_work
            scale_columns_by_real(ws.d_svd_U, m, ws.d_svd_S,
                                  ws.d_svd_work, m, m, new_k, streams_[si]);
        }

        // Absorb U*S into MPS[site-1]: new = old_MPS[site-1] @ (U*S)
        if (site > 0) {
            int prev_cL = chi_L(site - 1);
            ROCBLAS_CHECK(Traits::gemm(handles_[si],
                rocblas_operation_none, rocblas_operation_none,
                prev_cL * d_, new_k, cL, &one,
                d_mps_tensors_[site - 1], prev_cL * d_,
                ws.d_svd_work, m, &zero_val,
                ws.d_T1, prev_cL * d_));
            allocate_mps_tensor(site - 1, prev_cL, new_chi_L);
            HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site - 1], ws.d_T1,
                        (size_t)prev_cL * d_ * new_k * sizeof(Scalar),
                        hipMemcpyDeviceToDevice, streams_[si]));
        }
        bond_dims_[site] = new_chi_L;
    }

    ws.heff_cached_site = -1;
}

// ============================================================================
// Single-site optimization: form theta → Lanczos → SVD split
// ============================================================================

template<typename Scalar>
double PDMRGGPU<Scalar>::optimize_site_single(int site, char direction, int si) {
    int cL = chi_L(site);
    int cR = chi_R(site);
    int theta_size = cL * d_ * cR;
    auto& ws = workspaces_[si];

    // form_theta: just copy MPS[site] to workspace
    HIP_CHECK(hipMemcpyAsync(ws.d_theta, d_mps_tensors_[site],
                              theta_size * sizeof(Scalar), hipMemcpyDeviceToDevice, streams_[si]));

    // Use single-site matvec in Lanczos
    lanczos_use_1site_ = true;
    double energy = lanczos_eigensolver(site, ws.d_theta, theta_size, si);
    lanczos_use_1site_ = false;

    svd_split_single_site(site, ws.d_theta, direction, si);

    return energy;
}

// ============================================================================
// Full-chain sweep methods (two-site, for main PDMRG sweeps)
// ============================================================================

template<typename Scalar>
double PDMRGGPU<Scalar>::sweep_LR_full() {
    double energy = 0.0;
    for (int site = 0; site < L_ - 1; site++) {
        energy = optimize_bond(site, 'R', 0);
        update_left_env(site, 0);
    }
    return energy;
}

template<typename Scalar>
double PDMRGGPU<Scalar>::sweep_RL_full() {
    double energy = 0.0;
    for (int site = L_ - 2; site >= 0; site--) {
        energy = optimize_bond(site, 'L', 0);
        update_right_env(site + 1, 0);
    }
    return energy;
}

// ============================================================================
// Full-chain single-site sweep methods (for warmup and polish)
// ============================================================================

template<typename Scalar>
double PDMRGGPU<Scalar>::sweep_LR_full_1site() {
    double energy = 0.0;
    for (int site = 0; site < L_ - 1; site++) {
        energy = optimize_site_single(site, 'R', 0);
        update_left_env(site, 0);
    }
    // Last site: optimize without SVD (endpoint)
    {
        int cL = chi_L(L_ - 1);
        int cR = chi_R(L_ - 1);
        int theta_size = cL * d_ * cR;
        auto& ws = workspaces_[0];
        HIP_CHECK(hipMemcpyAsync(ws.d_theta, d_mps_tensors_[L_ - 1],
                                  theta_size * sizeof(Scalar), hipMemcpyDeviceToDevice, streams_[0]));
        lanczos_use_1site_ = true;
        energy = lanczos_eigensolver(L_ - 1, ws.d_theta, theta_size, 0);
        lanczos_use_1site_ = false;
        HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[L_ - 1], ws.d_theta,
                                  theta_size * sizeof(Scalar), hipMemcpyDeviceToDevice, streams_[0]));
    }
    return energy;
}

template<typename Scalar>
double PDMRGGPU<Scalar>::sweep_RL_full_1site() {
    double energy = 0.0;
    for (int site = L_ - 1; site >= 1; site--) {
        energy = optimize_site_single(site, 'L', 0);
        update_right_env(site, 0);
    }
    // First site: optimize without SVD (endpoint)
    {
        int cL = chi_L(0);
        int cR = chi_R(0);
        int theta_size = cL * d_ * cR;
        auto& ws = workspaces_[0];
        HIP_CHECK(hipMemcpyAsync(ws.d_theta, d_mps_tensors_[0],
                                  theta_size * sizeof(Scalar), hipMemcpyDeviceToDevice, streams_[0]));
        lanczos_use_1site_ = true;
        energy = lanczos_eigensolver(0, ws.d_theta, theta_size, 0);
        lanczos_use_1site_ = false;
        HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[0], ws.d_theta,
                                  theta_size * sizeof(Scalar), hipMemcpyDeviceToDevice, streams_[0]));
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
// Form theta with V injection: θ = ψ_L · diag(V) · ψ_R  (Stoudenmire Eq. 5)
// V scales the columns of ψ_L (or equivalently the rows of ψ_R) at the
// shared boundary bond before contracting into a two-site tensor.
//
// ψ_L: (cL*d, chi_bond) col-major on GPU
// ψ_R: (chi_bond, d*cR) col-major on GPU
// V:   (chi_bond,) on host → uploaded to d_svd_S workspace
// Result: d_theta = (cL*d, d*cR) = ψ_L · diag(V) · ψ_R
// ============================================================================

template<typename Scalar>
void PDMRGGPU<Scalar>::form_theta_with_V(int site, int boundary_idx, int si) {
    int cL = chi_L(site);
    int chi_bond = bond_dims_[site + 1];
    int cR = chi_R(site + 1);
    auto& ws = workspaces_[si];
    auto& bs = boundary_states_[boundary_idx];

    // Upload V to device (reuse d_svd_S workspace)
    HIP_CHECK(hipMemcpyAsync(ws.d_svd_S, bs.V.data(),
                              chi_bond * sizeof(RealType),
                              hipMemcpyHostToDevice, streams_[si]));

    // Scale: T1 = diag(V) · ψ_R  (scale each row i of ψ_R by V[i])
    // ψ_R is (chi_bond × d*cR) col-major, so row i has stride 1, col stride chi_bond
    // Copy ψ_R to T1 then scale rows
    int psi_R_size = chi_bond * d_ * cR;
    HIP_CHECK(hipMemcpyAsync(ws.d_T1, d_mps_tensors_[site + 1],
                              psi_R_size * sizeof(Scalar),
                              hipMemcpyDeviceToDevice, streams_[si]));
    // Scale rows: T1[i, j] *= V[i] for all j
    // This is equivalent to dgam: scale_rows_by_real from svd_split
    scale_rows_by_real(ws.d_T1, chi_bond, ws.d_svd_S,
                       ws.d_T1, chi_bond, chi_bond, d_ * cR, streams_[si]);

    // Contract: theta = ψ_L · T1 = ψ_L · diag(V) · ψ_R
    Scalar one = Traits::one(), zero_val = Traits::zero();
    ROCBLAS_CHECK(Traits::gemm(handles_[si],
        rocblas_operation_none, rocblas_operation_none,
        cL * d_, d_ * cR, chi_bond,
        &one,
        d_mps_tensors_[site], cL * d_,
        ws.d_T1, chi_bond,
        &zero_val,
        ws.d_theta, cL * d_));
}

// ============================================================================
// Stoudenmire boundary merge+optimize (proper V = Λ⁻¹ coupling)
//
// For each boundary bond:
//   1. Form θ = ψ_L · diag(V) · ψ_R  (Eq. 5)
//   2. Optimize θ with Lanczos eigensolver
//   3. SVD split: θ → U · S · Vh
//   4. Store V_new = 1/clip(S, 1e-12) for next iteration
//   5. MPS[bsite] = U (left-canonical), MPS[bsite+1] = S·Vh
//   6. Update environments from canonical tensors
//
// parity: 0 = even-indexed boundaries, 1 = odd, -1 = all
// ============================================================================

template<typename Scalar>
double PDMRGGPU<Scalar>::merge_and_optimize_boundaries(int parity) {
    double energy = 0.0;
    int si = 0;  // boundary optimization uses stream 0

    for (int b = 0; b < (int)boundary_bonds_.size(); b++) {
        if (parity >= 0 && (b % 2) != parity) continue;

        int bsite = boundary_bonds_[b];
        int cL = chi_L(bsite);
        int cR = chi_R(bsite + 1);
        int theta_size = cL * d_ * d_ * cR;
        auto& ws = workspaces_[si];

        // Step 1: Form θ = ψ_L · diag(V) · ψ_R
        form_theta_with_V(bsite, b, si);

        // Step 2: Optimize θ with eigensolver
        energy = lanczos_eigensolver(bsite, ws.d_theta, theta_size, si);

        // Step 3: SVD split → direction 'R': MPS[bsite]=U, MPS[bsite+1]=S·Vh
        if (use_rsvd_)
            rsvd_split(bsite, ws.d_theta, 'R', si);
        else
            svd_split(bsite, ws.d_theta, 'R', si);

        // Step 4: Update V = 1/clip(S, 1e-12) for next iteration
        int new_chi = bond_dims_[bsite + 1];
        boundary_states_[b].chi = new_chi;
        boundary_states_[b].V.resize(new_chi);

        // S values are already in ws.h_svd_S after svd_split
        const RealType reg = RealType(1e-12);
        for (int i = 0; i < new_chi; i++) {
            RealType s_val = ws.h_svd_S[i];
            if (s_val < reg) s_val = reg;
            boundary_states_[b].V[i] = RealType(1.0) / s_val;
        }

        // Step 5: Update environments from canonical tensors
        // L_env from U (left-canonical) — correct
        update_left_env(bsite, si);

        // R_env needs right-canonical tensor (Vh, not S·Vh)
        // Temporarily store Vh in MPS[bsite+1], build R_env, then restore S·Vh
        // We need Vh which is in ws.d_svd_Vh (GPU) or ws.h_svd_Vh (CPU)
        // For simplicity: the R_env will be rebuilt during the next segment sweep
        // that sweeps RL through bsite+1, which is the standard pattern.
        update_right_env(bsite + 1, si);
    }
    return energy;
}

// ============================================================================
// Main algorithm
// ============================================================================

template<typename Scalar>
double PDMRGGPU<Scalar>::run(int n_outer_sweeps, int n_local_sweeps, int n_warmup) {
    build_initial_environments();

    // Timer starts AFTER env build — measures sweep-to-convergence only
    auto t_start = std::chrono::high_resolution_clock::now();

    // Warmup: single-site sweeps (cheaper eigsh: chi*d vs chi*d²)
    double warmup_energy = 0.0;
    double prev_warmup_energy = 1e30;
    for (int sw = 0; sw < n_warmup; sw++) {
        sweep_LR_full_1site();
        warmup_energy = sweep_RL_full_1site();
        double dE = std::abs(warmup_energy - prev_warmup_energy);
        prev_warmup_energy = warmup_energy;
        if (dE < tol_ && sw > 0) break;
    }

    // === Main PDMRG loop (Stoudenmire staggered sweeps) ===
    // Replaces O(L) full-chain coupling with O(P) boundary-only merge+optimize.
    // Staggered pattern: even segments LR / odd RL, then reverse.
    // This ensures fresh L_env + R_env at each boundary when it's optimized.
    // Re-initialize V = ones after warmup (bond dims may have changed)
    initialize_boundary_states();
    double energy_prev = warmup_energy;
    energy_ = warmup_energy;
    bool outer_converged = false;

    // Parallel sweep launcher: one CPU thread per segment, each with its own HIP stream
    auto parallel_sweep = [this](auto sweep_fn) {
        std::vector<std::thread> threads(n_segments_);
        for (int k = 0; k < n_segments_; k++) {
            threads[k] = std::thread([this, k, &sweep_fn]{ sweep_fn(this, k); });
        }
        for (auto& t : threads) t.join();
        // Sync all GPU streams — segment sweeps launch async GPU work that
        // must complete before boundary coupling reads their outputs on stream 0
        HIP_CHECK(hipDeviceSynchronize());
    };

    bool has_odd_boundaries = ((int)boundary_bonds_.size() > 1);

    for (int outer = 0; outer < n_outer_sweeps; outer++) {
        for (int local_sw = 0; local_sw < n_local_sweeps; local_sw++) {
            // Half-sweep 1: even segments LR, odd segments RL
            // After this, even-numbered boundaries have fresh L_env + R_env:
            //   even seg k swept LR → L_env[seg_last_[k]] fresh
            //   odd seg k+1 swept RL → R_env[seg_first_[k+1]+1] fresh
            parallel_sweep([](PDMRGGPU* self, int k) {
                if (k % 2 == 0) self->segment_sweep_LR(k);
                else             self->segment_sweep_RL(k);
            });

            // Merge+optimize at even boundaries
            if (boundary_bonds_.size() > 0) {
                energy_ = merge_and_optimize_boundaries(0);
            }

            // Half-sweep 2: even segments RL, odd segments LR
            // After this, odd-numbered boundaries have fresh environments
            parallel_sweep([](PDMRGGPU* self, int k) {
                if (k % 2 == 0) self->segment_sweep_RL(k);
                else             self->segment_sweep_LR(k);
            });

            // Merge+optimize at odd boundaries (if any exist)
            if (has_odd_boundaries) {
                energy_ = merge_and_optimize_boundaries(1);
            }
        }

        double dE = std::abs(energy_ - energy_prev);

        if (dE < tol_ && outer > 0) {
            printf("Converged after %d outer iterations!\n", outer + 1);
            outer_converged = true;
            break;
        }

        energy_prev = energy_;
    }

    // === Polish phase: full-chain sweeps to converge to tight tolerance ===
    // Always run polish when using multiple segments — parallel DMRG leaves
    // stale boundary environments that only full-chain sweeps can fix.
    if (n_segments_ > 1) {
        int n_polish = 10;
        build_initial_environments();
        for (int sw = 0; sw < n_polish; sw++) {
            sweep_LR_full_1site();
            double eRL = sweep_RL_full_1site();
            double dE = std::abs(eRL - energy_);
            energy_ = eRL;
            if (dE < tol_) {
                printf("Polish converged after %d sweeps\n", sw + 1);
                break;
            }
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(t_end - t_start).count();
    printf("Final energy: %.12f\n", energy_);
    printf("Total wall time: %.3f s\n", total_time);

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
