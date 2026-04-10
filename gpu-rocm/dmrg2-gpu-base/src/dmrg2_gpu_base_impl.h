#ifndef DMRG2_GPU_BASE_IMPL_H
#define DMRG2_GPU_BASE_IMPL_H

#include <rocsolver/rocsolver.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <chrono>
#include <stdexcept>
#include <string>

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
DMRG2GPUBase<Scalar>::DMRG2GPUBase(int L, int d, int chi_max, int D_mpo, double tol)
    : L_(L), d_(d), chi_max_(chi_max), D_mpo_(D_mpo), tol_(tol), energy_(0.0) {

    bond_dims_.resize(L + 1);
    bond_dims_[0] = 1;
    bond_dims_[L] = 1;
    for (int i = 1; i < L; i++) {
        double exact_dim = pow((double)d_, std::min(i, L - i));
        bond_dims_[i] = (exact_dim > chi_max_) ? chi_max_ : (int)exact_dim;
    }

    HIP_CHECK(hipStreamCreate(&stream_));
    ROCBLAS_CHECK(rocblas_create_handle(&rocblas_h_));
    ROCBLAS_CHECK(rocblas_set_stream(rocblas_h_, stream_));

    int dd = d_ * d_;

    // Contraction intermediates — two-site theta is d^2 times larger than single-site
    int t_max = D_mpo_ * dd * chi_max_ * chi_max_;
    HIP_CHECK(hipMalloc(&d_T1_, t_max * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_T2_, t_max * sizeof(Scalar)));
    // d_T3_ holds the per-call temporary WW tensor (D*d^2 x D*d^2)
    int ww_size = D_mpo_ * dd * dd * D_mpo_;
    HIP_CHECK(hipMalloc(&d_T3_, ww_size * sizeof(Scalar)));

    // MPS tensors
    d_mps_tensors_.resize(L, nullptr);
    for (int i = 0; i < L; i++) {
        allocate_mps_tensor(i, chi_L(i), chi_R(i));
    }

    d_mpo_tensors_.resize(L, nullptr);
    d_W_left_.resize(L, nullptr);
    d_W_right_.resize(L, nullptr);

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

    // Lanczos workspace
    theta_size_max_ = chi_max_ * dd * chi_max_;
    max_lanczos_iter_ = std::min(100, theta_size_max_);
    HIP_CHECK(hipMalloc(&d_theta_, theta_size_max_ * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_heff_result_, theta_size_max_ * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_lanczos_v_, (size_t)max_lanczos_iter_ * theta_size_max_ * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_ritz_coeffs_, max_lanczos_iter_ * sizeof(Scalar)));

    h_alpha_.resize(max_lanczos_iter_);
    h_beta_.resize(max_lanczos_iter_);
    h_steqr_work_.resize(std::max(1, 2 * max_lanczos_iter_));
    h_steqr_Z_.resize(max_lanczos_iter_ * max_lanczos_iter_);

    int svd_max_m = chi_max_ * d_;
    int svd_max_n = d_ * chi_max_;
    int svd_max_k = std::min(svd_max_m, svd_max_n);

    HIP_CHECK(hipMalloc(&d_svd_A_,    theta_size_max_ * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_svd_U_,    (size_t)svd_max_m * svd_max_k * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_svd_S_,    svd_max_k * sizeof(RealType)));
    HIP_CHECK(hipMalloc(&d_svd_Vh_,   (size_t)svd_max_k * svd_max_n * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_svd_E_,    svd_max_k * sizeof(RealType)));
    HIP_CHECK(hipMalloc(&d_svd_info_, sizeof(int)));

    h_svd_U_.resize((size_t)svd_max_m * svd_max_k);
    h_svd_S_.resize(svd_max_k);
    h_svd_Vh_.resize((size_t)svd_max_k * svd_max_n);
    h_svd_tmp_.resize(std::max((size_t)svd_max_m * svd_max_k, (size_t)svd_max_k * svd_max_n));
}

// ============================================================================
// Destructor
// ============================================================================

template<typename Scalar>
DMRG2GPUBase<Scalar>::~DMRG2GPUBase() {
    free_gpu_resources();
}

template<typename Scalar>
void DMRG2GPUBase<Scalar>::free_gpu_resources() {
    for (auto ptr : d_mps_tensors_) if (ptr) hipFree(ptr);
    for (auto ptr : d_mpo_tensors_) if (ptr) hipFree(ptr);
    for (auto ptr : d_W_left_) if (ptr) hipFree(ptr);
    for (auto ptr : d_W_right_) if (ptr) hipFree(ptr);
    for (auto ptr : d_L_envs_) if (ptr) hipFree(ptr);
    for (auto ptr : d_R_envs_) if (ptr) hipFree(ptr);

    if (d_theta_) hipFree(d_theta_);
    if (d_heff_result_) hipFree(d_heff_result_);
    if (d_lanczos_v_) hipFree(d_lanczos_v_);
    if (d_ritz_coeffs_) hipFree(d_ritz_coeffs_);
    if (d_T1_) hipFree(d_T1_);
    if (d_T2_) hipFree(d_T2_);
    if (d_T3_) hipFree(d_T3_);
    if (d_svd_A_) hipFree(d_svd_A_);
    if (d_svd_U_) hipFree(d_svd_U_);
    if (d_svd_S_) hipFree(d_svd_S_);
    if (d_svd_Vh_) hipFree(d_svd_Vh_);
    if (d_svd_E_) hipFree(d_svd_E_);
    if (d_svd_info_) hipFree(d_svd_info_);

    rocblas_destroy_handle(rocblas_h_);
    hipStreamDestroy(stream_);
}

// ============================================================================
// Memory management
// ============================================================================

template<typename Scalar>
void DMRG2GPUBase<Scalar>::allocate_mps_tensor(int site, int cL, int cR) {
    size_t max_sz = (size_t)chi_max_ * d_ * chi_max_ * sizeof(Scalar);
    if (!d_mps_tensors_[site]) {
        HIP_CHECK(hipMalloc(&d_mps_tensors_[site], max_sz));
    }
    (void)cL; (void)cR;
}

template<typename Scalar>
void DMRG2GPUBase<Scalar>::ensure_L_env_alloc(int idx, int chi) {
    if (chi > L_env_alloc_chi_[idx]) {
        if (d_L_envs_[idx]) HIP_CHECK(hipFree(d_L_envs_[idx]));
        int sz = chi * D_mpo_ * chi;
        HIP_CHECK(hipMalloc(&d_L_envs_[idx], sz * sizeof(Scalar)));
        L_env_alloc_chi_[idx] = chi;
    }
}

template<typename Scalar>
void DMRG2GPUBase<Scalar>::ensure_R_env_alloc(int idx, int chi) {
    if (chi > R_env_alloc_chi_[idx]) {
        if (d_R_envs_[idx]) HIP_CHECK(hipFree(d_R_envs_[idx]));
        int sz = chi * D_mpo_ * chi;
        HIP_CHECK(hipMalloc(&d_R_envs_[idx], sz * sizeof(Scalar)));
        R_env_alloc_chi_[idx] = chi;
    }
}

// ============================================================================
// MPS initialization
// ============================================================================

template<typename Scalar>
void DMRG2GPUBase<Scalar>::initialize_mps_random(double scale) {
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

template<typename Scalar>
void DMRG2GPUBase<Scalar>::set_mpo(const std::vector<Scalar*>& h_mpo_tensors) {
    int D = D_mpo_, d = d_;
    for (int i = 0; i < L_; i++) {
        int size = D * d * d * D;
        HIP_CHECK(hipMalloc(&d_mpo_tensors_[i], size * sizeof(Scalar)));
        HIP_CHECK(hipMemcpy(d_mpo_tensors_[i], h_mpo_tensors[i],
                            size * sizeof(Scalar), hipMemcpyHostToDevice));

        // Precompute W_left and W_right matrices (for single-site env updates AND
        // for building the per-call fused WW inside apply_heff_two_site).
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

    // NOTE: No precompute_fused_mpo here — the WW tensor used by apply_heff
    // is rebuilt from scratch on every Lanczos iteration (see below). This
    // is the "unfused" naive baseline; the optimized dmrg2-gpu caches WW
    // at setup time to amortize the cost across sweeps.
}

// ============================================================================
// Two-site theta formation
// ============================================================================

template<typename Scalar>
void DMRG2GPUBase<Scalar>::form_theta_two_site(int site) {
    int cL = chi_L(site);
    int chi_mid = bond_dims_[site + 1];
    int cR = chi_R(site + 1);
    Scalar one = Traits::one(), zero_val = Traits::zero();

    ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
        rocblas_operation_none, rocblas_operation_none,
        cL * d_, d_ * cR, chi_mid,
        &one,
        d_mps_tensors_[site], cL * d_,
        d_mps_tensors_[site + 1], chi_mid,
        &zero_val,
        d_theta_, cL * d_));
}

// ============================================================================
// Two-site H_eff — NAIVE: rebuild WW per call + unbatched 3-step GEMM
// ============================================================================

template<typename Scalar>
void DMRG2GPUBase<Scalar>::apply_heff_two_site(int site, const Scalar* d_theta_in, Scalar* d_result) {
    int cL = chi_L(site);
    int cR = chi_R(site + 1);
    int D = D_mpo_, d = d_;
    int dd = d * d;
    Scalar one = Traits::one(), zero_val = Traits::zero();

    Scalar* L_env = d_L_envs_[site];
    Scalar* R_env = d_R_envs_[site + 2];
    Scalar* T1 = d_T1_;
    Scalar* T2 = d_T2_;
    Scalar* WW = d_T3_;  // rebuilt every call — NO caching

    // ---------------------------------------------------------------
    // Step 0: Build the two-site fused MPO on the HOST from the raw
    // MPO tensors at site and site+1, then upload to GPU.
    // This is the cost the optimized implementation amortizes via
    // precompute_fused_mpo() at set_mpo() time.
    // ---------------------------------------------------------------
    {
        int ww_size = D * dd * dd * D;
        // Read raw MPOs back to host. Site1 = d_mpo_tensors_[site], Site2 = [site+1].
        std::vector<Scalar> h_WL_raw(D * d * d * D);
        std::vector<Scalar> h_WR_raw(D * d * d * D);
        HIP_CHECK(hipMemcpy(h_WL_raw.data(), d_mpo_tensors_[site],
                            D * d * d * D * sizeof(Scalar), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_WR_raw.data(), d_mpo_tensors_[site + 1],
                            D * d * d * D * sizeof(Scalar), hipMemcpyDeviceToHost));

        std::vector<Scalar> h_WW(ww_size, Traits::zero());
        for (int w = 0; w < D; w++)
          for (int n = 0; n < D; n++)
            for (int s1 = 0; s1 < d; s1++)
              for (int s2 = 0; s2 < d; s2++)
                for (int s1p = 0; s1p < d; s1p++)
                  for (int s2p = 0; s2p < d; s2p++) {
                      Scalar val = Traits::zero();
                      for (int m = 0; m < D; m++) {
                          Scalar wl = h_WL_raw[w + s1*D + s1p*D*d + m*D*d*d];
                          Scalar wr = h_WR_raw[m + s2*D + s2p*D*d + n*D*d*d];
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
        HIP_CHECK(hipMemcpy(WW, h_WW.data(),
                            ww_size * sizeof(Scalar), hipMemcpyHostToDevice));
    }

    // ---------------------------------------------------------------
    // Step 1: Contract L with theta over a.
    //   T1[w*dd+s1*d+s2, a', b] = L[w]^T[a', a] * theta[a, s1, s2, b]
    // Naive: for each (w, s1, s2), issue one unbatched single GEMM.
    // ---------------------------------------------------------------
    for (int w = 0; w < D; w++) {
        for (int s1 = 0; s1 < d; s1++) {
            for (int s2 = 0; s2 < d; s2++) {
                // theta[a, s1, s2, b] viewed as (cL, cR) with base at
                // theta + (s1 + s2*d)*cL and ldb = cL*dd.
                const Scalar* A_ptr = L_env + w * cL;
                const Scalar* B_ptr = d_theta_in + (s1 + s2 * d) * cL;
                Scalar* C_ptr = T1 + (w * dd + s1 * d + s2) * cL * cR;
                ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
                    Traits::op_t, rocblas_operation_none,
                    cL, cR, cL,
                    &one,
                    A_ptr, cL * D,
                    B_ptr, cL * dd,
                    &zero_val,
                    C_ptr, cL));
            }
        }
    }

    // ---------------------------------------------------------------
    // Step 2: Dense GEMM — absorb (per-call) WW.
    //   T2 = T1 @ WW, where T1 is (cL*cR, D*dd), WW is (D*dd, dd*D),
    //   T2 is (cL*cR, dd*D).
    // ---------------------------------------------------------------
    ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
        rocblas_operation_none, rocblas_operation_none,
        cL * cR, dd * D, D * dd,
        &one,
        T1, cL * cR,
        WW, D * dd,
        &zero_val,
        T2, cL * cR));

    // ---------------------------------------------------------------
    // Step 3: Contract R with T2 over b.
    //   result[s1', s2', a', b'] = sum_n T2[a', b, n*dd + s1'*d + s2']
    //                                    * R[n, b, b']
    // Naive: for each (n, s1', s2'), issue one unbatched single GEMM
    // with beta=0 on n=0 and beta=1 otherwise.
    // ---------------------------------------------------------------
    for (int n = 0; n < D; n++) {
        Scalar beta = (n == 0) ? Traits::zero() : Traits::one();
        for (int s1p = 0; s1p < d; s1p++) {
            for (int s2p = 0; s2p < d; s2p++) {
                const Scalar* A_ptr = T2 + (n * dd + s1p * d + s2p) * cL * cR;
                const Scalar* B_ptr = R_env + n * cR;
                Scalar* C_ptr = d_result + (s1p + s2p * d) * cL;  // col = s1p + s2p*d
                ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
                    rocblas_operation_none, rocblas_operation_none,
                    cL, cR, cR,
                    &one,
                    A_ptr, cL,
                    B_ptr, cR * D,
                    &beta,
                    C_ptr, cL * dd));
            }
        }
    }
}

// ============================================================================
// update_left_env — naive unbatched loops
// ============================================================================

template<typename Scalar>
void DMRG2GPUBase<Scalar>::update_left_env(int site) {
    int chi_in = bond_dims_[site];
    int chi_out = bond_dims_[site + 1];
    int D = D_mpo_, d = d_;
    Scalar one = Traits::one(), zero_val = Traits::zero();

    ensure_L_env_alloc(site + 1, chi_out);

    Scalar* L_env = d_L_envs_[site];
    Scalar* A = d_mps_tensors_[site];
    Scalar* W_mat = d_W_left_[site];
    Scalar* L_new = d_L_envs_[site + 1];
    Scalar* V = d_T1_;
    Scalar* U = d_T2_;

    for (int w = 0; w < D; w++) {
        for (int s = 0; s < d; s++) {
            const Scalar* A_ptr = L_env + w * chi_in;
            const Scalar* B_ptr = A + s * chi_in;
            Scalar* C_ptr = V + (w * d + s) * chi_in * chi_out;
            ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
                Traits::op_t, rocblas_operation_none,
                chi_in, chi_out, chi_in,
                &one,
                A_ptr, chi_in * D,
                B_ptr, chi_in * d,
                &zero_val,
                C_ptr, chi_in));
        }
    }

    ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
        rocblas_operation_none, rocblas_operation_none,
        chi_in * chi_out, d * D, D * d,
        &one,
        V, chi_in * chi_out,
        W_mat, D * d,
        &zero_val,
        U, chi_in * chi_out));

    for (int sp = 0; sp < d; sp++) {
        Scalar beta = (sp == 0) ? Traits::zero() : Traits::one();
        for (int w = 0; w < D; w++) {
            const Scalar* A_ptr = U + (w * d + sp) * chi_in * chi_out;
            const Scalar* B_ptr = A + sp * chi_in;
            Scalar* C_ptr = L_new + w * chi_out;
            ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
                Traits::op_h, rocblas_operation_none,
                chi_out, chi_out, chi_in,
                &one,
                A_ptr, chi_in,
                B_ptr, chi_in * d,
                &beta,
                C_ptr, chi_out * D));
        }
    }

    if constexpr (Traits::is_complex) {
        conjugate_inplace(L_new, chi_out * D * chi_out, stream_);
    }
}

// ============================================================================
// update_right_env — naive unbatched loops
// ============================================================================

template<typename Scalar>
void DMRG2GPUBase<Scalar>::update_right_env(int site) {
    int chi_in = bond_dims_[site + 1];
    int chi_out = bond_dims_[site];
    int D = D_mpo_, d = d_;
    Scalar one = Traits::one(), zero_val = Traits::zero();

    ensure_R_env_alloc(site, chi_out);

    Scalar* A = d_mps_tensors_[site];
    Scalar* R_env = d_R_envs_[site + 1];
    Scalar* W_mat = d_W_right_[site];
    Scalar* R_new = d_R_envs_[site];
    Scalar* V = d_T1_;
    Scalar* U = d_T2_;

    for (int w = 0; w < D; w++) {
        for (int s = 0; s < d; s++) {
            const Scalar* A_ptr = A + s * chi_out;
            const Scalar* B_ptr = R_env + w * chi_in;
            Scalar* C_ptr = V + (w * d + s) * chi_out * chi_in;
            ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
                rocblas_operation_none, rocblas_operation_none,
                chi_out, chi_in, chi_in,
                &one,
                A_ptr, chi_out * d,
                B_ptr, chi_in * D,
                &zero_val,
                C_ptr, chi_out));
        }
    }

    ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
        rocblas_operation_none, rocblas_operation_none,
        chi_out * chi_in, d * D, D * d,
        &one,
        V, chi_out * chi_in,
        W_mat, D * d,
        &zero_val,
        U, chi_out * chi_in));

    for (int sp = 0; sp < d; sp++) {
        Scalar beta = (sp == 0) ? Traits::zero() : Traits::one();
        for (int w = 0; w < D; w++) {
            const Scalar* A_ptr = U + (w * d + sp) * chi_out * chi_in;
            const Scalar* B_ptr = A + sp * chi_out;
            Scalar* C_ptr = R_new + w * chi_out;
            ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
                rocblas_operation_none, Traits::op_h,
                chi_out, chi_out, chi_in,
                &one,
                A_ptr, chi_out,
                B_ptr, chi_out * d,
                &beta,
                C_ptr, chi_out * D));
        }
    }
}

// ============================================================================
// Environment building
// ============================================================================

template<typename Scalar>
void DMRG2GPUBase<Scalar>::build_initial_environments() {
    {
        std::vector<Scalar> h_L(D_mpo_, Traits::zero());
        h_L[0] = Traits::one();
        HIP_CHECK(hipMemcpy(d_L_envs_[0], h_L.data(),
                            D_mpo_ * sizeof(Scalar), hipMemcpyHostToDevice));
    }
    {
        std::vector<Scalar> h_R(D_mpo_, Traits::zero());
        h_R[D_mpo_ - 1] = Traits::one();
        HIP_CHECK(hipMemcpy(d_R_envs_[L_], h_R.data(),
                            D_mpo_ * sizeof(Scalar), hipMemcpyHostToDevice));
    }

    for (int i = L_ - 1; i >= 0; i--) {
        update_right_env(i);
    }
}

// ============================================================================
// Lanczos (host-pointer mode + CPU LAPACK dstev) — operates on theta_size n
// ============================================================================

template<typename Scalar>
double DMRG2GPUBase<Scalar>::lanczos_eigensolver(int site, Scalar* d_theta, int theta_size) {
    int n = theta_size;
    int max_iter = std::min(max_lanczos_iter_, n);
    double tol_lanczos = 1e-12;

    Scalar* d_lanczos_v = d_lanczos_v_;

    double norm;
    ROCBLAS_CHECK(Traits::nrm2(rocblas_h_, n, d_theta, 1, &norm));

    if (norm < 1e-14) {
        std::vector<Scalar> h_init(n);
        srand(42 + site);
        for (int i = 0; i < n; i++) h_init[i] = Traits::random_val();
        HIP_CHECK(hipMemcpy(d_theta, h_init.data(), n * sizeof(Scalar), hipMemcpyHostToDevice));
        ROCBLAS_CHECK(Traits::nrm2(rocblas_h_, n, d_theta, 1, &norm));
    }

    double inv_norm = 1.0 / norm;
    ROCBLAS_CHECK(Traits::scal_real(rocblas_h_, n, &inv_norm, d_theta, 1));
    HIP_CHECK(hipMemcpy(d_lanczos_v, d_theta, n * sizeof(Scalar), hipMemcpyDeviceToDevice));

    int iter;
    for (iter = 0; iter < max_iter; iter++) {
        Scalar* d_vi = d_lanczos_v + (size_t)iter * n;

        apply_heff_two_site(site, d_vi, d_heff_result_);

        Scalar dot_result;
        ROCBLAS_CHECK(Traits::dot(rocblas_h_, n, d_vi, 1, d_heff_result_, 1, &dot_result));
        double alpha = Traits::real_part(dot_result);
        h_alpha_[iter] = alpha;

        Scalar neg_alpha = Traits::neg(Traits::make_scalar(alpha, 0.0));
        ROCBLAS_CHECK(Traits::axpy(rocblas_h_, n, &neg_alpha, d_vi, 1, d_heff_result_, 1));

        if (iter > 0) {
            Scalar neg_beta = Traits::neg(Traits::make_scalar(h_beta_[iter - 1], 0.0));
            ROCBLAS_CHECK(Traits::axpy(rocblas_h_, n, &neg_beta,
                d_lanczos_v + (size_t)(iter - 1) * n, 1,
                d_heff_result_, 1));
        }

        // Full reorthogonalization
        {
            std::vector<Scalar> h_coeffs(iter + 1);
            for (int j = 0; j <= iter; j++) {
                Scalar c;
                ROCBLAS_CHECK(Traits::dot(rocblas_h_, n,
                    d_lanczos_v + (size_t)j * n, 1,
                    d_heff_result_, 1, &c));
                h_coeffs[j] = Traits::neg(c);
            }
            for (int j = 0; j <= iter; j++) {
                ROCBLAS_CHECK(Traits::axpy(rocblas_h_, n, &h_coeffs[j],
                    d_lanczos_v + (size_t)j * n, 1,
                    d_heff_result_, 1));
            }
        }

        double beta;
        ROCBLAS_CHECK(Traits::nrm2(rocblas_h_, n, d_heff_result_, 1, &beta));
        h_beta_[iter] = beta;

        if (beta < tol_lanczos) { iter++; break; }

        if (iter + 1 < max_iter) {
            Scalar* d_vip1 = d_lanczos_v + (size_t)(iter + 1) * n;
            HIP_CHECK(hipMemcpy(d_vip1, d_heff_result_, n * sizeof(Scalar), hipMemcpyDeviceToDevice));
            double inv_beta = 1.0 / beta;
            ROCBLAS_CHECK(Traits::scal_real(rocblas_h_, n, &inv_beta, d_vip1, 1));
        }
    }

    int niter = iter;
    if (niter <= 0) niter = 1;

    // CPU LAPACK dstev
    std::vector<double> D(niter);
    std::vector<double> E(std::max(1, niter - 1));
    for (int j = 0; j < niter; j++) D[j] = h_alpha_[j];
    for (int j = 0; j < niter - 1; j++) E[j] = h_beta_[j];

    std::vector<double>& Z = h_steqr_Z_;
    if ((int)Z.size() < niter * niter) Z.resize(niter * niter);
    std::vector<double>& work = h_steqr_work_;
    if ((int)work.size() < std::max(1, 2 * niter - 2)) work.resize(std::max(1, 2 * niter - 2));

    const char jobz = 'V';
    const int ldz = niter;
    int info = 0;
    dstev_(&jobz, &niter, D.data(), E.data(), Z.data(), &ldz, work.data(), &info);
    if (info != 0) {
        throw std::runtime_error("dstev failed with info = " + std::to_string(info));
    }

    double energy = D[0];

    std::vector<Scalar> h_ritz(niter);
    for (int j = 0; j < niter; j++) h_ritz[j] = Traits::make_scalar(Z[j], 0.0);
    HIP_CHECK(hipMemcpy(d_ritz_coeffs_, h_ritz.data(),
                        niter * sizeof(Scalar), hipMemcpyHostToDevice));

    Scalar one = Traits::one(), zero_val = Traits::zero();
    ROCBLAS_CHECK(Traits::gemv(rocblas_h_, rocblas_operation_none,
        n, niter, &one,
        d_lanczos_v, n,
        d_ritz_coeffs_, 1,
        &zero_val, d_theta, 1));

    double theta_norm;
    ROCBLAS_CHECK(Traits::nrm2(rocblas_h_, n, d_theta, 1, &theta_norm));
    if (theta_norm > 0) {
        double inv_tn = 1.0 / theta_norm;
        ROCBLAS_CHECK(Traits::scal_real(rocblas_h_, n, &inv_tn, d_theta, 1));
    }

    return energy;
}

// ============================================================================
// SVD split — naive: rocsolver gesvd + host-side truncation
// ============================================================================

template<typename Scalar>
void DMRG2GPUBase<Scalar>::svd_split(int site, Scalar* d_theta, char direction) {
    int cL = chi_L(site);
    int cR = chi_R(site + 1);

    int m = cL * d_;
    int n_svd = d_ * cR;
    int full_k = std::min(m, n_svd);
    int k = std::min(full_k, chi_max_);

    HIP_CHECK(hipMemcpy(d_svd_A_, d_theta, m * n_svd * sizeof(Scalar), hipMemcpyDeviceToDevice));

    Traits::rocsolver_gesvd(rocblas_h_,
        rocblas_svect_singular, rocblas_svect_singular,
        m, n_svd,
        d_svd_A_, m,
        d_svd_S_,
        d_svd_U_, m,
        d_svd_Vh_, full_k,
        d_svd_E_,
        rocblas_outofplace,
        d_svd_info_);

    // Copy everything back to host for truncation and scaling
    HIP_CHECK(hipMemcpy(h_svd_S_.data(), d_svd_S_,
                        full_k * sizeof(RealType), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_svd_U_.data(), d_svd_U_,
                        (size_t)m * full_k * sizeof(Scalar), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_svd_Vh_.data(), d_svd_Vh_,
                        (size_t)full_k * n_svd * sizeof(Scalar), hipMemcpyDeviceToHost));
    HIP_CHECK(hipStreamSynchronize(stream_));

    int new_k = 0;
    for (int j = 0; j < k; j++) {
        if (h_svd_S_[j] > 1e-14) new_k++;
        else break;
    }
    if (new_k == 0) new_k = 1;

    if (direction == 'R') {
        // MPS[site] = U[:, :new_k]  (col slice, contiguous in col-major)
        allocate_mps_tensor(site, cL, new_k);
        HIP_CHECK(hipMemcpy(d_mps_tensors_[site], h_svd_U_.data(),
                            (size_t)m * new_k * sizeof(Scalar), hipMemcpyHostToDevice));

        // S*Vh on host → MPS[site+1]. Scale row j of Vh by S[j].
        // Source element (j, c) at Vh[c*full_k + j], dest at tmp[c*new_k + j].
        allocate_mps_tensor(site + 1, new_k, cR);
        for (int c = 0; c < n_svd; c++) {
            for (int j = 0; j < new_k; j++) {
                h_svd_tmp_[c * new_k + j] =
                    Traits::scale_by_real(h_svd_S_[j], h_svd_Vh_[c * full_k + j]);
            }
        }
        HIP_CHECK(hipMemcpy(d_mps_tensors_[site + 1], h_svd_tmp_.data(),
                            (size_t)new_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice));

    } else {  // direction == 'L'
        // U*S → MPS[site]. Scale col j of U by S[j].
        allocate_mps_tensor(site, cL, new_k);
        for (int j = 0; j < new_k; j++) {
            for (int r = 0; r < m; r++) {
                h_svd_tmp_[j * m + r] =
                    Traits::scale_by_real(h_svd_S_[j], h_svd_U_[j * m + r]);
            }
        }
        HIP_CHECK(hipMemcpy(d_mps_tensors_[site], h_svd_tmp_.data(),
                            (size_t)m * new_k * sizeof(Scalar), hipMemcpyHostToDevice));

        // MPS[site+1] = Vh[:new_k, :]. Repack col-major.
        allocate_mps_tensor(site + 1, new_k, cR);
        for (int c = 0; c < n_svd; c++) {
            for (int j = 0; j < new_k; j++) {
                h_svd_tmp_[c * new_k + j] = h_svd_Vh_[c * full_k + j];
            }
        }
        HIP_CHECK(hipMemcpy(d_mps_tensors_[site + 1], h_svd_tmp_.data(),
                            (size_t)new_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice));
    }

    bond_dims_[site + 1] = new_k;
}

// ============================================================================
// Bond optimization
// ============================================================================

template<typename Scalar>
double DMRG2GPUBase<Scalar>::optimize_bond(int site, char direction) {
    int cL = chi_L(site);
    int cR = chi_R(site + 1);
    int theta_size = cL * d_ * d_ * cR;

    form_theta_two_site(site);
    double energy = lanczos_eigensolver(site, d_theta_, theta_size);
    svd_split(site, d_theta_, direction);
    return energy;
}

// ============================================================================
// Sweep methods
// ============================================================================

template<typename Scalar>
double DMRG2GPUBase<Scalar>::sweep_left_to_right() {
    double energy = 0.0;
    for (int site = 0; site < L_ - 1; site++) {
        energy = optimize_bond(site, 'R');
        update_left_env(site);
    }
    return energy;
}

template<typename Scalar>
double DMRG2GPUBase<Scalar>::sweep_right_to_left() {
    double energy = 0.0;
    for (int site = L_ - 2; site >= 0; site--) {
        energy = optimize_bond(site, 'L');
        update_right_env(site + 1);
    }
    return energy;
}

// ============================================================================
// Main algorithm
// ============================================================================

template<typename Scalar>
double DMRG2GPUBase<Scalar>::run(int n_sweeps) {
    build_initial_environments();

    auto t_start = std::chrono::high_resolution_clock::now();
    double energy_prev = 0.0;

    for (int sweep = 0; sweep < n_sweeps; sweep++) {
        sweep_left_to_right();
        energy_ = sweep_right_to_left();

        double dE = std::abs(energy_ - energy_prev);
        if (dE < tol_ && sweep > 0) break;
        energy_prev = energy_;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(t_end - t_start).count();
    printf("Final energy: %.12f\n", energy_);
    printf("Total wall time: %.3f s\n", total_time);

    return energy_;
}

// ============================================================================
// Utility
// ============================================================================

template<typename Scalar>
void DMRG2GPUBase<Scalar>::get_mps(std::vector<std::vector<Scalar>>& h_mps) const {
    h_mps.resize(L_);
    for (int i = 0; i < L_; i++) {
        int size = chi_L(i) * d_ * chi_R(i);
        h_mps[i].resize(size);
        HIP_CHECK(hipMemcpy(h_mps[i].data(), d_mps_tensors_[i],
                            size * sizeof(Scalar), hipMemcpyDeviceToHost));
    }
}

#endif // DMRG2_GPU_BASE_IMPL_H
