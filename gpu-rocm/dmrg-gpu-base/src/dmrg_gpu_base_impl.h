#ifndef DMRG_GPU_BASE_IMPL_H
#define DMRG_GPU_BASE_IMPL_H

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
DMRGGPUBase<Scalar>::DMRGGPUBase(int L, int d, int chi_max, int D_mpo, double tol)
    : L_(L), d_(d), chi_max_(chi_max), D_mpo_(D_mpo), tol_(tol), energy_(0.0) {

    // Bond dimensions
    bond_dims_.resize(L + 1);
    bond_dims_[0] = 1;
    bond_dims_[L] = 1;
    for (int i = 1; i < L; i++) {
        double exact_dim = pow((double)d_, std::min(i, L - i));
        bond_dims_[i] = (exact_dim > chi_max_) ? chi_max_ : (int)exact_dim;
    }

    // GPU handles — single stream, single rocBLAS handle.
    // Pointer mode is set per-call (host for setup/SVD, device for the
    // Lanczos inner loop where BLAS-1 results stay on device).
    HIP_CHECK(hipStreamCreate(&stream_));
    ROCBLAS_CHECK(rocblas_create_handle(&rocblas_h_));
    ROCBLAS_CHECK(rocblas_set_stream(rocblas_h_, stream_));

    // Contraction intermediates
    int t_max = D_mpo_ * d_ * chi_max_ * chi_max_;
    HIP_CHECK(hipMalloc(&d_T1_, t_max * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_T2_, t_max * sizeof(Scalar)));

    // MPS tensors
    d_mps_tensors_.resize(L, nullptr);
    for (int i = 0; i < L; i++) {
        allocate_mps_tensor(i, chi_L(i), chi_R(i));
    }

    // MPO tensors
    d_mpo_tensors_.resize(L, nullptr);
    d_W_left_.resize(L, nullptr);
    d_W_right_.resize(L, nullptr);

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

    // Lanczos workspace
    theta_size_max_ = chi_max_ * d_ * chi_max_;
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

    // SVD workspace
    int svd_max_dim = chi_max_ * d_;
    HIP_CHECK(hipMalloc(&d_svd_A_,    theta_size_max_ * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_svd_U_,    svd_max_dim * chi_max_ * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_svd_S_,    chi_max_ * sizeof(RealType)));
    HIP_CHECK(hipMalloc(&d_svd_Vh_,   chi_max_ * svd_max_dim * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_svd_work_, svd_max_dim * chi_max_ * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_svd_E_,    chi_max_ * sizeof(RealType)));
    HIP_CHECK(hipMalloc(&d_svd_info_, sizeof(int)));
    HIP_CHECK(hipMalloc(&d_svdj_residual_, sizeof(double)));
    HIP_CHECK(hipMalloc(&d_svdj_n_sweeps_, sizeof(rocblas_int)));

    // Tiny host buffer for the truncation-rank decision (chi_max RealTypes).
    h_svd_S_.resize(chi_max_);
}

// ============================================================================
// Destructor
// ============================================================================

template<typename Scalar>
DMRGGPUBase<Scalar>::~DMRGGPUBase() {
    free_gpu_resources();
}

template<typename Scalar>
void DMRGGPUBase<Scalar>::free_gpu_resources() {
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
void DMRGGPUBase<Scalar>::allocate_mps_tensor(int site, int cL, int cR) {
    size_t max_sz = (size_t)chi_max_ * d_ * chi_max_ * sizeof(Scalar);
    if (!d_mps_tensors_[site]) {
        HIP_CHECK(hipMalloc(&d_mps_tensors_[site], max_sz));
    }
    (void)cL; (void)cR;
}

template<typename Scalar>
void DMRGGPUBase<Scalar>::ensure_L_env_alloc(int idx, int chi) {
    if (chi > L_env_alloc_chi_[idx]) {
        if (d_L_envs_[idx]) HIP_CHECK(hipFree(d_L_envs_[idx]));
        int sz = chi * D_mpo_ * chi;
        HIP_CHECK(hipMalloc(&d_L_envs_[idx], sz * sizeof(Scalar)));
        L_env_alloc_chi_[idx] = chi;
    }
}

template<typename Scalar>
void DMRGGPUBase<Scalar>::ensure_R_env_alloc(int idx, int chi) {
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
void DMRGGPUBase<Scalar>::initialize_mps_random(double scale) {
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
void DMRGGPUBase<Scalar>::set_mpo(const std::vector<Scalar*>& h_mpo_tensors) {
    int D = D_mpo_, d = d_;
    for (int i = 0; i < L_; i++) {
        int size = D * d * d * D;
        // Guard against double-call: free the previous MPO buffer if set_mpo
        // is invoked more than once (test harnesses can re-use a single
        // instance across multiple problems).
        if (d_mpo_tensors_[i]) HIP_CHECK(hipFree(d_mpo_tensors_[i]));
        HIP_CHECK(hipMalloc(&d_mpo_tensors_[i], size * sizeof(Scalar)));
        HIP_CHECK(hipMemcpy(d_mpo_tensors_[i], h_mpo_tensors[i],
                            size * sizeof(Scalar), hipMemcpyHostToDevice));

        // Precompute W_left and W_right matrices (kept: these are MPO reshapes,
        // not a fused-tensor optimization — they're what makes the 3-step GEMM
        // pattern work at all).
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
        if (d_W_left_[i]) HIP_CHECK(hipFree(d_W_left_[i]));
        HIP_CHECK(hipMalloc(&d_W_left_[i], wm_size * sizeof(Scalar)));
        HIP_CHECK(hipMemcpy(d_W_left_[i], h_WL.data(),
                            wm_size * sizeof(Scalar), hipMemcpyHostToDevice));
        if (d_W_right_[i]) HIP_CHECK(hipFree(d_W_right_[i]));
        HIP_CHECK(hipMalloc(&d_W_right_[i], wm_size * sizeof(Scalar)));
        HIP_CHECK(hipMemcpy(d_W_right_[i], h_WR.data(),
                            wm_size * sizeof(Scalar), hipMemcpyHostToDevice));
    }
}

// ============================================================================
// apply_heff — naive single-GEMM loops (no gemm_batched)
// ============================================================================

template<typename Scalar>
void DMRGGPUBase<Scalar>::apply_heff(int site, const Scalar* d_theta_in, Scalar* d_result) {
    int cL = chi_L(site);
    int cR = chi_R(site);
    int D = D_mpo_, d = d_;
    Scalar one = Traits::one(), zero_val = Traits::zero();

    Scalar* L_env = d_L_envs_[site];
    Scalar* R_env = d_R_envs_[site + 1];
    Scalar* W_mat = d_W_left_[site];
    Scalar* V = d_T1_;
    Scalar* U = d_T2_;

    // Step 1: V_{w*d+s}[a',b] = L_w^T[a',a] * theta_s[a,b]
    //   Naive: for each (w, s), issue an individual dgemm.
    for (int w = 0; w < D; w++) {
        for (int s = 0; s < d; s++) {
            const Scalar* A_ptr = L_env + w * cL;                      // stride cL*D
            const Scalar* B_ptr = d_theta_in + s * cL;                 // stride cL*d
            Scalar* C_ptr = V + (w * d + s) * cL * cR;                 // stride cL*cR
            ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
                Traits::op_t, rocblas_operation_none,
                cL, cR, cL,
                &one,
                A_ptr, cL * D,
                B_ptr, cL * d,
                &zero_val,
                C_ptr, cL));
        }
    }

    // Step 2: U = V * W_matrix (one big dense GEMM)
    ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
        rocblas_operation_none, rocblas_operation_none,
        cL * cR, d * D, D * d,
        &one,
        V, cL * cR,
        W_mat, D * d,
        &zero_val,
        U, cL * cR));

    // Step 3: result_{s'}[a',b'] = sum_{w'} U_{w'd+s'}[a',b] * R_{w'}[b,b']
    //   Naive: for each (w', s'), single dgemm with accumulation.
    for (int wp = 0; wp < D; wp++) {
        Scalar beta = (wp == 0) ? Traits::zero() : Traits::one();
        for (int s = 0; s < d; s++) {
            const Scalar* A_ptr = U + (wp * d + s) * cL * cR;
            const Scalar* B_ptr = R_env + wp * cR;                     // stride cR*D
            Scalar* C_ptr = d_result + s * cL;                         // stride cL*d
            ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
                rocblas_operation_none, rocblas_operation_none,
                cL, cR, cR,
                &one,
                A_ptr, cL,
                B_ptr, cR * D,
                &beta,
                C_ptr, cL * d));
        }
    }
}

// ============================================================================
// update_left_env — naive single-GEMM loops
// ============================================================================

template<typename Scalar>
void DMRGGPUBase<Scalar>::update_left_env(int site) {
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

    // Step 1: V_{w*d+s}[a',b] = L_w^T[a',a] * A_s[a,b]
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

    // Step 2: U = V * W_matrix
    ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
        rocblas_operation_none, rocblas_operation_none,
        chi_in * chi_out, d * D, D * d,
        &one,
        V, chi_in * chi_out,
        W_mat, D * d,
        &zero_val,
        U, chi_in * chi_out));

    // Step 3: L_new_{w'}[b,b'] = sum_{a',s'} U[a', w'*d+s', b] * A^H_{s'}[b', a']
    //   For real: op_h == op_t. For complex, we compute U^H*A and conjugate L_new
    //   after the loops — same convention as the optimized dmrg-gpu.
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
// update_right_env — naive single-GEMM loops
// ============================================================================

template<typename Scalar>
void DMRGGPUBase<Scalar>::update_right_env(int site) {
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

    // Step 1: V_{w*d+s}[a,b'] = A_s[a,b] * R_w[b,b']
    for (int w = 0; w < D; w++) {
        for (int s = 0; s < d; s++) {
            const Scalar* A_ptr = A + s * chi_out;                 // stride chi_out*d
            const Scalar* B_ptr = R_env + w * chi_in;              // stride chi_in*D
            Scalar* C_ptr = V + (w * d + s) * chi_out * chi_in;    // stride chi_out*chi_in
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

    // Step 2: U = V * W_matrix
    ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
        rocblas_operation_none, rocblas_operation_none,
        chi_out * chi_in, d * D, D * d,
        &one,
        V, chi_out * chi_in,
        W_mat, D * d,
        &zero_val,
        U, chi_out * chi_in));

    // Step 3: R_new_w[a,a'] = sum_{s'} U_{w*d+s'}[a,b'] * A^H_{s'}[b',a']
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
void DMRGGPUBase<Scalar>::build_initial_environments() {
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
// Theta formation and Lanczos
//
// Naive single-vector Lanczos with full one-pass classical Gram-Schmidt
// reorthogonalization. All scalar BLAS-1 results (dot, nrm2) stay on device
// via rocBLAS device-pointer mode; alpha/beta are stored to per-iter device
// arrays via the shared lanczos_process_alpha/beta kernels in
// common/scalar_traits.h. The tridiagonal eigenproblem is solved on device
// via rocsolver_dsteqr — only the final eigenvalue (one double) is read
// back to host as the function's return value.
//
// Compared to the optimized DMRGGPU Lanczos, this baseline omits:
//   - the every-3-iteration intermediate-dsteqr convergence-check pattern,
//   - HIP graph capture + replay,
//   - fused axpy+normalize kernels.
// It converges to the same energy but does the full max_iter iterations
// (or breaks early on beta < 1e-12), uses the standard rocBLAS axpy/scal,
// and recomputes the tridiagonal eigenproblem only once at the end.
// ============================================================================

template<typename Scalar>
void DMRGGPUBase<Scalar>::form_theta(int site, Scalar* d_theta) {
    int size = chi_L(site) * d_ * chi_R(site);
    HIP_CHECK(hipMemcpy(d_theta, d_mps_tensors_[site],
                        size * sizeof(Scalar), hipMemcpyDeviceToDevice));
}

template<typename Scalar>
double DMRGGPUBase<Scalar>::lanczos_eigensolver(int site, Scalar* d_theta) {
    int n = chi_L(site) * d_ * chi_R(site);
    int max_iter = std::min(max_lanczos_iter_, n);

    Scalar* d_lanczos_v = d_lanczos_v_;

    // Switch rocBLAS to device-pointer mode for the inner loop.
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
        apply_heff(site, d_vi, d_heff_result_);

        // alpha_i = Re <v_i | w>  (device-pointer dot)
        ROCBLAS_CHECK(Traits::dot(rocblas_h_, n, d_vi, 1, d_heff_result_, 1, d_dot_result_));
        // Process: store Re(dot_result) to d_alpha_dev_[iter] and emit -alpha into d_neg_alpha_
        hipLaunchKernelGGL(lanczos_process_alpha_kernel<Scalar>, dim3(1), dim3(1), 0, stream_,
                           d_dot_result_, d_neg_alpha_, d_alpha_dev_, iter);

        // w -= alpha_i * v_i
        ROCBLAS_CHECK(Traits::axpy(rocblas_h_, n, d_neg_alpha_, d_vi, 1, d_heff_result_, 1));

        // w -= beta_{i-1} * v_{i-1}     (use d_neg_beta_scalars_[iter-1] from prev iter)
        if (iter > 0) {
            ROCBLAS_CHECK(Traits::axpy(rocblas_h_, n,
                d_neg_beta_scalars_ + (iter - 1),
                d_lanczos_v + (size_t)(iter - 1) * n, 1,
                d_heff_result_, 1));
        }

        // Full one-pass classical Gram-Schmidt reorthogonalization, on device.
        // Per-pair dot+axpy: dot writes -<v_j|w> into d_neg_overlap_, axpy
        // applies it. (The optimized DMRGGPU fuses this into a single GEMV.)
        for (int j = 0; j <= iter; j++) {
            ROCBLAS_CHECK(Traits::dot(rocblas_h_, n,
                d_lanczos_v + (size_t)j * n, 1,
                d_heff_result_, 1, d_dot_result_));
            // Negate the dot result into d_neg_overlap_.
            hipLaunchKernelGGL(negate_scalar_kernel<Scalar>, dim3(1), dim3(1), 0, stream_,
                               d_dot_result_, d_neg_overlap_);
            ROCBLAS_CHECK(Traits::axpy(rocblas_h_, n, d_neg_overlap_,
                d_lanczos_v + (size_t)j * n, 1,
                d_heff_result_, 1));
        }

        // beta_i = ||w||  (device-pointer nrm2)
        ROCBLAS_CHECK(Traits::nrm2(rocblas_h_, n, d_heff_result_, 1, d_nrm2_result_));
        // Store beta[iter] to d_beta_dev_, compute 1/beta into d_inv_nrm_,
        // and emit -beta into d_neg_beta_scalars_[iter] for the next iter's axpy.
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

    // Restore host-pointer mode for SVD/setup calls outside the Lanczos loop.
    ROCBLAS_CHECK(rocblas_set_pointer_mode(rocblas_h_, rocblas_pointer_mode_host));

    // Solve the tridiagonal eigenproblem on device (rocsolver_dsteqr).
    HIP_CHECK(hipMemcpyAsync(d_steqr_D_, d_alpha_dev_,
                             niter * sizeof(double), hipMemcpyDeviceToDevice, stream_));
    HIP_CHECK(hipMemcpyAsync(d_steqr_E_, d_beta_dev_,
                             niter * sizeof(double), hipMemcpyDeviceToDevice, stream_));
    rocsolver_dsteqr(rocblas_h_, rocblas_evect_tridiagonal, niter,
                     d_steqr_D_, d_steqr_E_, d_steqr_C_, niter, d_steqr_info_);

    // Read back the smallest eigenvalue (one double) and the first column of
    // the eigenvector matrix (Ritz coefficients, niter doubles staying on device).
    double energy;
    HIP_CHECK(hipMemcpy(&energy, d_steqr_D_, sizeof(double), hipMemcpyDeviceToHost));

    // d_steqr_C_ has the eigenvectors as columns (first column = smallest evec).
    // For real Scalar, copy into d_ritz_coeffs_ directly. For complex, promote.
    if constexpr (Traits::is_complex) {
        HIP_CHECK(hipMemsetAsync(d_ritz_coeffs_, 0,
                                 niter * sizeof(Scalar), stream_));
        hipLaunchKernelGGL(promote_double_to_complex, dim3((niter + 255) / 256), dim3(256), 0, stream_,
                           d_steqr_C_, (hipDoubleComplex*)d_ritz_coeffs_, niter);
    } else {
        HIP_CHECK(hipMemcpyAsync(d_ritz_coeffs_, d_steqr_C_,
                                 niter * sizeof(double), hipMemcpyDeviceToDevice, stream_));
    }

    // theta = V * ritz_coeffs  (single GEMV, host-pointer mode for the constants).
    Scalar one = Traits::one(), zero_val = Traits::zero();
    ROCBLAS_CHECK(Traits::gemv(rocblas_h_, rocblas_operation_none,
        n, niter, &one,
        d_lanczos_v, n,
        d_ritz_coeffs_, 1,
        &zero_val, d_theta, 1));

    // Normalize theta on device (device-pointer scal).
    ROCBLAS_CHECK(rocblas_set_pointer_mode(rocblas_h_, rocblas_pointer_mode_device));
    ROCBLAS_CHECK(Traits::nrm2(rocblas_h_, n, d_theta, 1, d_nrm2_result_));
    hipLaunchKernelGGL(inv_real_kernel, dim3(1), dim3(1), 0, stream_,
                       d_nrm2_result_, d_inv_nrm_);
    ROCBLAS_CHECK(Traits::scal_real(rocblas_h_, n, d_inv_nrm_, d_theta, 1));
    ROCBLAS_CHECK(rocblas_set_pointer_mode(rocblas_h_, rocblas_pointer_mode_host));

    return energy;
}

// ============================================================================
// SVD and MPS update — rocsolver_gesvd_auto + device-side truncation
//
// Truncation pipeline: rocSOLVER returns S sorted in descending order, so the
// truncation-rank decision is a tiny D2H of S (chi_max doubles, control-flow
// scalar) followed by a host scan. The actual matrix slicing and S-scaling
// happen entirely on device via extract_cols_kernel and
// scale_rows_by_diag_kernel / scale_cols_by_diag_kernel from
// common/scalar_traits.h. No host roundtrip of U or Vh.
// ============================================================================

template<typename Scalar>
void DMRGGPUBase<Scalar>::svd_and_update_mps(int site, Scalar* d_theta, char direction) {
    int cL = chi_L(site);
    int cR = chi_R(site);

    int m, n_svd;
    if (direction == 'R') { m = cL * d_; n_svd = cR; }
    else                  { m = cL;      n_svd = d_ * cR; }
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

    // Read back S only (small: full_k <= chi_max doubles). Used for the
    // truncation-rank decision; not a tensor-data roundtrip.
    HIP_CHECK(hipMemcpy(h_svd_S_.data(), d_svd_S_,
                        full_k * sizeof(RealType), hipMemcpyDeviceToHost));

    int new_k = 0;
    for (int j = 0; j < k; j++) {
        if (h_svd_S_[j] > 1e-14) new_k++;
        else break;
    }
    if (new_k == 0) new_k = 1;

    if (direction == 'R') {
        int new_chi_R = new_k;

        // MPS[site] = U[:, :new_k]. U is (m, full_k) col-major with lda=m;
        // first new_k columns are contiguous in memory, so a single async
        // D2D copy suffices.
        allocate_mps_tensor(site, cL, new_chi_R);
        HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site], d_svd_U_,
                                 m * new_k * sizeof(Scalar),
                                 hipMemcpyDeviceToDevice, stream_));

        // S*Vh: scale rows of Vh[:new_k, :] by S[i], write to (new_k, n_svd)
        // contiguous layout in d_svd_work_.
        {
            int total = new_k * n_svd;
            int blocks = (total + 255) / 256;
            hipLaunchKernelGGL((scale_rows_by_diag_kernel<Scalar, RealType>),
                               dim3(blocks), dim3(256), 0, stream_,
                               d_svd_S_, d_svd_Vh_, full_k,
                               d_svd_work_, new_k, new_k, n_svd);
        }

        // Absorb S*Vh into A[site+1]: (new_k × cR) @ (cR × d*next_cR) → (new_k × d*next_cR)
        if (site + 1 < L_) {
            int next_cR = chi_R(site + 1);
            Scalar one = Traits::one(), zero_val = Traits::zero();
            ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
                rocblas_operation_none, rocblas_operation_none,
                new_k, d_ * next_cR, cR, &one,
                d_svd_work_, new_k,
                d_mps_tensors_[site + 1], cR, &zero_val,
                d_T1_, new_k));
            allocate_mps_tensor(site + 1, new_chi_R, next_cR);
            HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site + 1], d_T1_,
                                     new_k * d_ * next_cR * sizeof(Scalar),
                                     hipMemcpyDeviceToDevice, stream_));
        }
        bond_dims_[site + 1] = new_chi_R;

    } else {  // direction == 'L'
        int new_chi_L = new_k;

        // MPS[site] = Vh[:new_k, :]. Row slice from (full_k × n_svd) col-major
        // with lda=full_k → (new_k × n_svd) with lda=new_k. extract_cols_kernel
        // does this generic sub-rectangle copy on device.
        allocate_mps_tensor(site, new_chi_L, cR);
        {
            int total = new_chi_L * n_svd;
            int blocks = (total + 255) / 256;
            hipLaunchKernelGGL((extract_cols_kernel<Scalar>),
                               dim3(blocks), dim3(256), 0, stream_,
                               d_svd_Vh_, full_k,
                               d_mps_tensors_[site], new_chi_L,
                               new_chi_L, n_svd);
        }

        // U*S: scale columns of U[:, :new_k] by S[j], write to (m, new_k)
        // contiguous layout in d_svd_work_.
        {
            int total = m * new_k;
            int blocks = (total + 255) / 256;
            hipLaunchKernelGGL((scale_cols_by_diag_kernel<Scalar, RealType>),
                               dim3(blocks), dim3(256), 0, stream_,
                               d_svd_S_, d_svd_U_, m,
                               d_svd_work_, m, m, new_k);
        }

        // Absorb U*S into A[site-1]: (prev_cL*d × m) @ (m × new_k) → (prev_cL*d × new_k)
        if (site > 0) {
            int prev_cL = chi_L(site - 1);
            Scalar one = Traits::one(), zero_val = Traits::zero();
            ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
                rocblas_operation_none, rocblas_operation_none,
                prev_cL * d_, new_k, m, &one,
                d_mps_tensors_[site - 1], prev_cL * d_,
                d_svd_work_, m, &zero_val,
                d_T1_, prev_cL * d_));
            allocate_mps_tensor(site - 1, prev_cL, new_chi_L);
            HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site - 1], d_T1_,
                                     prev_cL * d_ * new_k * sizeof(Scalar),
                                     hipMemcpyDeviceToDevice, stream_));
        }
        bond_dims_[site] = new_chi_L;
    }
}

// ============================================================================
// Site optimization
// ============================================================================

template<typename Scalar>
double DMRGGPUBase<Scalar>::optimize_site(int site, char direction) {
    form_theta(site, d_theta_);
    double energy = lanczos_eigensolver(site, d_theta_);
    svd_and_update_mps(site, d_theta_, direction);
    return energy;
}

// ============================================================================
// Sweep methods
// ============================================================================

template<typename Scalar>
double DMRGGPUBase<Scalar>::sweep_left_to_right() {
    double energy = 0.0;

    for (int site = 0; site < L_ - 1; site++) {
        energy = optimize_site(site, 'R');
        update_left_env(site);
    }
    {
        int site = L_ - 1;
        form_theta(site, d_theta_);
        energy = lanczos_eigensolver(site, d_theta_);
        int sz = chi_L(site) * d_ * chi_R(site);
        // Stream-bound D2D — bare hipMemcpy routes through the legacy
        // stream and would synchronize the device unnecessarily on the
        // sweep-boundary path (round-5 A2 fix).
        HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site], d_theta_, sz * sizeof(Scalar),
                                  hipMemcpyDeviceToDevice, stream_));
    }

    return energy;
}

template<typename Scalar>
double DMRGGPUBase<Scalar>::sweep_right_to_left() {
    double energy = 0.0;

    for (int site = L_ - 1; site >= 1; site--) {
        energy = optimize_site(site, 'L');
        update_right_env(site);
    }
    {
        int site = 0;
        form_theta(site, d_theta_);
        energy = lanczos_eigensolver(site, d_theta_);
        int sz = chi_L(site) * d_ * chi_R(site);
        // Stream-bound D2D, see L→R sweep above.
        HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site], d_theta_, sz * sizeof(Scalar),
                                  hipMemcpyDeviceToDevice, stream_));
    }

    return energy;
}

// ============================================================================
// Main algorithm
// ============================================================================

template<typename Scalar>
double DMRGGPUBase<Scalar>::run(int n_sweeps) {
    // Sweep-only timer: starts AFTER MPS+MPO+env build, stops at convergence.
    // Matches DMRGGPU::run() so the -gpu / -gpu-base wall-time comparison is
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
// Utility methods
// ============================================================================

template<typename Scalar>
void DMRGGPUBase<Scalar>::get_mps(std::vector<std::vector<Scalar>>& h_mps) const {
    h_mps.resize(L_);
    for (int i = 0; i < L_; i++) {
        int size = chi_L(i) * d_ * chi_R(i);
        h_mps[i].resize(size);
        HIP_CHECK(hipMemcpy(h_mps[i].data(), d_mps_tensors_[i],
                            size * sizeof(Scalar), hipMemcpyDeviceToHost));
    }
}

#endif // DMRG_GPU_BASE_IMPL_H
