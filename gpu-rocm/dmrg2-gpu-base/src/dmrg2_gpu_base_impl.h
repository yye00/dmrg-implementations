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

#include "../../common/hip_check.h"

// promote_double_to_complex now defined in common/scalar_traits.h
// (round-5 single-source-of-truth promotion).

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

    // Device-pointer-mode scratch (single scalars).
    HIP_CHECK(hipMalloc(&d_dot_result_,   sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_nrm2_result_,  sizeof(RealType)));
    HIP_CHECK(hipMalloc(&d_inv_nrm_,      sizeof(RealType)));
    HIP_CHECK(hipMalloc(&d_neg_alpha_,    sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_neg_overlap_,  sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_neg_beta_scalars_, max_lanczos_iter_ * sizeof(Scalar)));

    // Per-iteration alpha/beta arrays on device.
    HIP_CHECK(hipMalloc(&d_alpha_dev_, max_lanczos_iter_ * sizeof(RealType)));
    HIP_CHECK(hipMalloc(&d_beta_dev_,  max_lanczos_iter_ * sizeof(RealType)));

    // rocSOLVER dsteqr workspaces.
    HIP_CHECK(hipMalloc(&d_steqr_D_,    max_lanczos_iter_ * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_steqr_E_,    max_lanczos_iter_ * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_steqr_C_,    (size_t)max_lanczos_iter_ * max_lanczos_iter_ * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_steqr_info_, sizeof(rocblas_int)));

    int svd_max_m = chi_max_ * d_;
    int svd_max_n = d_ * chi_max_;
    int svd_max_k = std::min(svd_max_m, svd_max_n);

    HIP_CHECK(hipMalloc(&d_svd_A_,    theta_size_max_ * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_svd_U_,    (size_t)svd_max_m * svd_max_k * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_svd_S_,    svd_max_k * sizeof(RealType)));
    HIP_CHECK(hipMalloc(&d_svd_Vh_,   (size_t)svd_max_k * svd_max_n * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_svd_work_, std::max((size_t)svd_max_m * svd_max_k,
                                                (size_t)svd_max_k * svd_max_n) * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_svd_E_,    svd_max_k * sizeof(RealType)));
    HIP_CHECK(hipMalloc(&d_svd_info_, sizeof(int)));
    HIP_CHECK(hipMalloc(&d_svdj_residual_, sizeof(double)));
    HIP_CHECK(hipMalloc(&d_svdj_n_sweeps_, sizeof(rocblas_int)));

    h_svd_S_.resize(svd_max_k);

    // Two-site fused MPO storage. Allocated/populated by precompute_WW()
    // called from set_mpo() — one entry per bond (L_-1 bonds).
    d_WW_.resize(L_ - 1, nullptr);
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

    for (auto ptr : d_WW_) if (ptr) hipFree(ptr);

    if (d_theta_) hipFree(d_theta_);
    if (d_heff_result_) hipFree(d_heff_result_);
    if (d_lanczos_v_) hipFree(d_lanczos_v_);
    if (d_ritz_coeffs_) hipFree(d_ritz_coeffs_);
    if (d_T1_) hipFree(d_T1_);
    if (d_T2_) hipFree(d_T2_);
    if (d_dot_result_) hipFree(d_dot_result_);
    if (d_nrm2_result_) hipFree(d_nrm2_result_);
    if (d_inv_nrm_) hipFree(d_inv_nrm_);
    if (d_neg_alpha_) hipFree(d_neg_alpha_);
    if (d_neg_overlap_) hipFree(d_neg_overlap_);
    if (d_neg_beta_scalars_) hipFree(d_neg_beta_scalars_);
    if (d_alpha_dev_) hipFree(d_alpha_dev_);
    if (d_beta_dev_) hipFree(d_beta_dev_);
    if (d_steqr_D_) hipFree(d_steqr_D_);
    if (d_steqr_E_) hipFree(d_steqr_E_);
    if (d_steqr_C_) hipFree(d_steqr_C_);
    if (d_steqr_info_) hipFree(d_steqr_info_);
    if (d_svd_A_) hipFree(d_svd_A_);
    if (d_svd_U_) hipFree(d_svd_U_);
    if (d_svd_S_) hipFree(d_svd_S_);
    if (d_svd_Vh_) hipFree(d_svd_Vh_);
    if (d_svd_work_) hipFree(d_svd_work_);
    if (d_svd_E_) hipFree(d_svd_E_);
    if (d_svd_info_) hipFree(d_svd_info_);
    if (d_svdj_residual_) hipFree(d_svdj_residual_);
    if (d_svdj_n_sweeps_) hipFree(d_svdj_n_sweeps_);

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
        // Guard against double-call (round-7 M4): free previous MPO buffer.
        if (d_mpo_tensors_[i]) HIP_CHECK(hipFree(d_mpo_tensors_[i]));
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

    // Precompute the per-bond fused two-site MPO (WW) once at setup time.
    // The WW tensor for adjacent (site, site+1) is the contraction
    //   WW[w, s1, s2; n, s1', s2'] = sum_m W_L[w, s1, s1', m] * W_R[m, s2, s2', n]
    // built once from the host MPOs and uploaded to d_WW_[bond]. apply_heff
    // then uses the precomputed tensor — no per-iteration host roundtrip.
    // The natural first-pass GPU choice. The optimized DMRG2GPU does the
    // same precompute via precompute_fused_mpo() with D_PAD/SPARSE_MPO
    // optimizations on top.
    precompute_WW();
}

template<typename Scalar>
void DMRG2GPUBase<Scalar>::precompute_WW() {
    int D = D_mpo_, d = d_;
    int dd = d * d;
    int ww_size = D * dd * dd * D;

    // Read raw MPOs back to host once (small: D*d*d*D scalars per site).
    // This is at set_mpo() time — outside the timed sweep region.
    std::vector<Scalar> h_WL_raw((size_t)D * d * d * D);
    std::vector<Scalar> h_WR_raw((size_t)D * d * d * D);
    std::vector<Scalar> h_WW(ww_size);

    for (int site = 0; site < L_ - 1; site++) {
        HIP_CHECK(hipMemcpy(h_WL_raw.data(), d_mpo_tensors_[site],
                            (size_t)D * d * d * D * sizeof(Scalar),
                            hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_WR_raw.data(), d_mpo_tensors_[site + 1],
                            (size_t)D * d * d * D * sizeof(Scalar),
                            hipMemcpyDeviceToHost));

        std::fill(h_WW.begin(), h_WW.end(), Traits::zero());
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

        HIP_CHECK(hipMalloc(&d_WW_[site], ww_size * sizeof(Scalar)));
        HIP_CHECK(hipMemcpy(d_WW_[site], h_WW.data(),
                            ww_size * sizeof(Scalar), hipMemcpyHostToDevice));
    }
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
    Scalar* WW = d_WW_[site];   // precomputed once at set_mpo() time

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
// Lanczos eigensolver — fully on-device.
// See dmrg-gpu-base for the design rationale; this two-site variant uses
// the same device-pointer Lanczos pattern with rocsolver_dsteqr at the end.
// The only difference is that apply_heff is two-site (theta is cL*d * d*cR).
// ============================================================================

template<typename Scalar>
double DMRG2GPUBase<Scalar>::lanczos_eigensolver(int site, Scalar* d_theta, int theta_size) {
    int n = theta_size;
    int max_iter = std::min(max_lanczos_iter_, n);

    Scalar* d_lanczos_v = d_lanczos_v_;

    ROCBLAS_CHECK(rocblas_set_pointer_mode(rocblas_h_, rocblas_pointer_mode_device));

    // v[0] = theta / ||theta||  (entirely on device)
    ROCBLAS_CHECK(Traits::nrm2(rocblas_h_, n, d_theta, 1, d_nrm2_result_));
    hipLaunchKernelGGL(inv_real_kernel, dim3(1), dim3(1), 0, stream_,
                       d_nrm2_result_, d_inv_nrm_);
    ROCBLAS_CHECK(Traits::scal_real(rocblas_h_, n, d_inv_nrm_, d_theta, 1));
    HIP_CHECK(hipMemcpyAsync(d_lanczos_v, d_theta,
                             n * sizeof(Scalar), hipMemcpyDeviceToDevice, stream_));

    int iter;
    for (iter = 0; iter < max_iter; iter++) {
        Scalar* d_vi = d_lanczos_v + (size_t)iter * n;

        // w = H |v_i>
        apply_heff_two_site(site, d_vi, d_heff_result_);

        // alpha_i = Re <v_i | w>  (device-pointer dot)
        ROCBLAS_CHECK(Traits::dot(rocblas_h_, n, d_vi, 1, d_heff_result_, 1, d_dot_result_));
        hipLaunchKernelGGL(lanczos_process_alpha_kernel<Scalar>, dim3(1), dim3(1), 0, stream_,
                           d_dot_result_, d_neg_alpha_, d_alpha_dev_, iter);

        // w -= alpha_i * v_i
        ROCBLAS_CHECK(Traits::axpy(rocblas_h_, n, d_neg_alpha_, d_vi, 1, d_heff_result_, 1));

        // w -= beta_{i-1} * v_{i-1}
        if (iter > 0) {
            ROCBLAS_CHECK(Traits::axpy(rocblas_h_, n,
                d_neg_beta_scalars_ + (iter - 1),
                d_lanczos_v + (size_t)(iter - 1) * n, 1,
                d_heff_result_, 1));
        }

        // Full one-pass classical Gram-Schmidt reorthogonalization, on device.
        for (int j = 0; j <= iter; j++) {
            ROCBLAS_CHECK(Traits::dot(rocblas_h_, n,
                d_lanczos_v + (size_t)j * n, 1,
                d_heff_result_, 1, d_dot_result_));
            hipLaunchKernelGGL(negate_scalar_kernel<Scalar>, dim3(1), dim3(1), 0, stream_,
                               d_dot_result_, d_neg_overlap_);
            ROCBLAS_CHECK(Traits::axpy(rocblas_h_, n, d_neg_overlap_,
                d_lanczos_v + (size_t)j * n, 1,
                d_heff_result_, 1));
        }

        // beta_i = ||w||  (device-pointer nrm2)
        ROCBLAS_CHECK(Traits::nrm2(rocblas_h_, n, d_heff_result_, 1, d_nrm2_result_));
        hipLaunchKernelGGL(lanczos_process_beta_kernel<Scalar>, dim3(1), dim3(1), 0, stream_,
                           d_nrm2_result_, d_inv_nrm_, d_beta_dev_, d_neg_beta_scalars_, iter);

        // v_{i+1} = w / beta_i
        if (iter + 1 < max_iter) {
            Scalar* d_vip1 = d_lanczos_v + (size_t)(iter + 1) * n;
            HIP_CHECK(hipMemcpyAsync(d_vip1, d_heff_result_,
                                     n * sizeof(Scalar), hipMemcpyDeviceToDevice, stream_));
            ROCBLAS_CHECK(Traits::scal_real(rocblas_h_, n, d_inv_nrm_, d_vip1, 1));
        }
    }

    int niter = iter;
    if (niter <= 0) niter = 1;

    ROCBLAS_CHECK(rocblas_set_pointer_mode(rocblas_h_, rocblas_pointer_mode_host));

    // Tridiagonal eigensolve on device.
    HIP_CHECK(hipMemcpyAsync(d_steqr_D_, d_alpha_dev_,
                             niter * sizeof(double), hipMemcpyDeviceToDevice, stream_));
    HIP_CHECK(hipMemcpyAsync(d_steqr_E_, d_beta_dev_,
                             niter * sizeof(double), hipMemcpyDeviceToDevice, stream_));
    rocsolver_dsteqr(rocblas_h_, rocblas_evect_tridiagonal, niter,
                     d_steqr_D_, d_steqr_E_, d_steqr_C_, niter, d_steqr_info_);

    double energy;
    HIP_CHECK(hipMemcpy(&energy, d_steqr_D_, sizeof(double), hipMemcpyDeviceToHost));

    if constexpr (Traits::is_complex) {
        HIP_CHECK(hipMemsetAsync(d_ritz_coeffs_, 0,
                                 niter * sizeof(Scalar), stream_));
        hipLaunchKernelGGL(promote_double_to_complex, dim3((niter + 255) / 256), dim3(256), 0, stream_,
                           d_steqr_C_, (hipDoubleComplex*)d_ritz_coeffs_, niter);
    } else {
        HIP_CHECK(hipMemcpyAsync(d_ritz_coeffs_, d_steqr_C_,
                                 niter * sizeof(double), hipMemcpyDeviceToDevice, stream_));
    }

    Scalar one = Traits::one(), zero_val = Traits::zero();
    ROCBLAS_CHECK(Traits::gemv(rocblas_h_, rocblas_operation_none,
        n, niter, &one,
        d_lanczos_v, n,
        d_ritz_coeffs_, 1,
        &zero_val, d_theta, 1));

    ROCBLAS_CHECK(rocblas_set_pointer_mode(rocblas_h_, rocblas_pointer_mode_device));
    ROCBLAS_CHECK(Traits::nrm2(rocblas_h_, n, d_theta, 1, d_nrm2_result_));
    hipLaunchKernelGGL(inv_real_kernel, dim3(1), dim3(1), 0, stream_,
                       d_nrm2_result_, d_inv_nrm_);
    ROCBLAS_CHECK(Traits::scal_real(rocblas_h_, n, d_inv_nrm_, d_theta, 1));
    ROCBLAS_CHECK(rocblas_set_pointer_mode(rocblas_h_, rocblas_pointer_mode_host));

    return energy;
}

// ============================================================================
// SVD split — rocsolver_gesvd_auto + device-side truncation.
// See dmrg-gpu-base for the design rationale; this two-site variant always
// splits along (cL*d × d*cR), with both halves landing in adjacent MPS tensors.
// ============================================================================

template<typename Scalar>
void DMRG2GPUBase<Scalar>::svd_split(int site, Scalar* d_theta, char direction) {
    int cL = chi_L(site);
    int cR = chi_R(site + 1);

    int m = cL * d_;
    int n_svd = d_ * cR;
    int full_k = std::min(m, n_svd);
    int k = std::min(full_k, chi_max_);

    HIP_CHECK(hipMemcpyAsync(d_svd_A_, d_theta,
                             m * n_svd * sizeof(Scalar), hipMemcpyDeviceToDevice, stream_));

    Traits::rocsolver_gesvd_auto(rocblas_h_,
        rocblas_svect_singular, rocblas_svect_singular,
        m, n_svd,
        d_svd_A_, m,
        d_svd_S_,
        d_svd_U_, m,
        d_svd_Vh_, full_k,
        d_svd_E_,
        d_svdj_residual_, d_svdj_n_sweeps_,
        d_svd_info_);

    // Read back S only (full_k <= chi_max RealTypes) for truncation-rank decision.
    HIP_CHECK(hipMemcpy(h_svd_S_.data(), d_svd_S_,
                        full_k * sizeof(RealType), hipMemcpyDeviceToHost));

    int new_k = 0;
    for (int j = 0; j < k; j++) {
        if (h_svd_S_[j] > 1e-14) new_k++;
        else break;
    }
    if (new_k == 0) new_k = 1;

    if (direction == 'R') {
        // MPS[site] = U[:, :new_k]. First new_k columns of U; col-major contiguous.
        allocate_mps_tensor(site, cL, new_k);
        HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site], d_svd_U_,
                                 (size_t)m * new_k * sizeof(Scalar),
                                 hipMemcpyDeviceToDevice, stream_));

        // MPS[site+1] = S * Vh[:new_k, :]. Device kernel scales rows of Vh by S
        // and writes the (new_k × n_svd) result directly into MPS[site+1].
        allocate_mps_tensor(site + 1, new_k, cR);
        {
            int total = new_k * n_svd;
            int blocks = (total + 255) / 256;
            hipLaunchKernelGGL((scale_rows_by_diag_kernel<Scalar, RealType>),
                               dim3(blocks), dim3(256), 0, stream_,
                               d_svd_S_, d_svd_Vh_, full_k,
                               d_mps_tensors_[site + 1], new_k, new_k, n_svd);
        }

    } else {  // direction == 'L'
        // MPS[site] = U[:, :new_k] * S. Device kernel scales cols of U by S
        // and writes (m × new_k) into MPS[site].
        allocate_mps_tensor(site, cL, new_k);
        {
            int total = m * new_k;
            int blocks = (total + 255) / 256;
            hipLaunchKernelGGL((scale_cols_by_diag_kernel<Scalar, RealType>),
                               dim3(blocks), dim3(256), 0, stream_,
                               d_svd_S_, d_svd_U_, m,
                               d_mps_tensors_[site], m, m, new_k);
        }

        // MPS[site+1] = Vh[:new_k, :]. Row slice from (full_k × n_svd) col-major
        // with lda=full_k → (new_k × n_svd) with lda=new_k. extract_cols_kernel
        // does this generic sub-rectangle copy on device.
        allocate_mps_tensor(site + 1, new_k, cR);
        {
            int total = new_k * n_svd;
            int blocks = (total + 255) / 256;
            hipLaunchKernelGGL((extract_cols_kernel<Scalar>),
                               dim3(blocks), dim3(256), 0, stream_,
                               d_svd_Vh_, full_k,
                               d_mps_tensors_[site + 1], new_k,
                               new_k, n_svd);
        }
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
    // Sweep-only timer: starts AFTER MPS+MPO+env build, stops at convergence.
    // Matches DMRG2GPU::run() so the -gpu / -gpu-base wall-time comparison is
    // apples-to-apples. Env build is reported separately as a diagnostic line
    // but is NOT included in "Total wall time".
    auto t_env_start = std::chrono::high_resolution_clock::now();
    build_initial_environments();

    auto t_start = std::chrono::high_resolution_clock::now();
    double env_time = std::chrono::duration<double>(t_start - t_env_start).count();
    printf("  Environment build: %.3f s\n", env_time);

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
    printf("  env_build_sec: %.3f  timer_scope: sweep_only\n", env_time);

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
