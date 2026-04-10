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

    // GPU handles — host pointer mode throughout (no device-pointer optimizations)
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

    // Host Lanczos tridiagonal workspace (CPU LAPACK dstev)
    h_alpha_.resize(max_lanczos_iter_);
    h_beta_.resize(max_lanczos_iter_);
    h_steqr_work_.resize(std::max(1, 2 * max_lanczos_iter_));
    h_steqr_Z_.resize(max_lanczos_iter_ * max_lanczos_iter_);

    // SVD workspace
    int svd_max_dim = chi_max_ * d_;
    HIP_CHECK(hipMalloc(&d_svd_A_,    theta_size_max_ * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_svd_U_,    svd_max_dim * chi_max_ * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_svd_S_,    chi_max_ * sizeof(RealType)));
    HIP_CHECK(hipMalloc(&d_svd_Vh_,   chi_max_ * svd_max_dim * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_svd_E_,    chi_max_ * sizeof(RealType)));
    HIP_CHECK(hipMalloc(&d_svd_info_, sizeof(int)));

    h_svd_U_.resize(svd_max_dim * chi_max_);
    h_svd_S_.resize(chi_max_);
    h_svd_Vh_.resize(chi_max_ * svd_max_dim);
    h_svd_tmp_.resize(svd_max_dim * chi_max_);
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
        HIP_CHECK(hipMalloc(&d_W_left_[i], wm_size * sizeof(Scalar)));
        HIP_CHECK(hipMemcpy(d_W_left_[i], h_WL.data(),
                            wm_size * sizeof(Scalar), hipMemcpyHostToDevice));
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
// Theta formation and Lanczos (host-pointer mode, CPU LAPACK dstev)
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
    double tol_lanczos = 1e-12;

    Scalar* d_lanczos_v = d_lanczos_v_;

    // v[0] = theta / ||theta||  (host pointer mode)
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

        // w = H |v_i>
        apply_heff(site, d_vi, d_heff_result_);

        // alpha_i = Re <v_i | w>  (host-pointer dot, sync back)
        Scalar dot_result;
        ROCBLAS_CHECK(Traits::dot(rocblas_h_, n, d_vi, 1, d_heff_result_, 1, &dot_result));
        double alpha = Traits::real_part(dot_result);
        h_alpha_[iter] = alpha;

        // w -= alpha_i * v_i
        Scalar neg_alpha = Traits::neg(Traits::make_scalar(alpha, 0.0));
        ROCBLAS_CHECK(Traits::axpy(rocblas_h_, n, &neg_alpha, d_vi, 1, d_heff_result_, 1));

        // w -= beta_{i-1} * v_{i-1}
        if (iter > 0) {
            Scalar neg_beta = Traits::neg(Traits::make_scalar(h_beta_[iter - 1], 0.0));
            ROCBLAS_CHECK(Traits::axpy(rocblas_h_, n, &neg_beta,
                d_lanczos_v + (size_t)(iter - 1) * n, 1,
                d_heff_result_, 1));
        }

        // Full reorthogonalization (two-pass Gram-Schmidt)
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

        // beta_i = ||w||
        double beta;
        ROCBLAS_CHECK(Traits::nrm2(rocblas_h_, n, d_heff_result_, 1, &beta));
        h_beta_[iter] = beta;

        if (beta < tol_lanczos) { iter++; break; }

        // v_{i+1} = w / beta_i
        if (iter + 1 < max_iter) {
            Scalar* d_vip1 = d_lanczos_v + (size_t)(iter + 1) * n;
            HIP_CHECK(hipMemcpy(d_vip1, d_heff_result_, n * sizeof(Scalar), hipMemcpyDeviceToDevice));
            double inv_beta = 1.0 / beta;
            ROCBLAS_CHECK(Traits::scal_real(rocblas_h_, n, &inv_beta, d_vip1, 1));
        }
    }

    int niter = iter;
    if (niter <= 0) niter = 1;

    // CPU LAPACK dstev to solve the tridiagonal eigenproblem
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

    // Ritz coefficients = first column of Z (smallest eigenvalue)
    std::vector<Scalar> h_ritz(niter);
    for (int j = 0; j < niter; j++) h_ritz[j] = Traits::make_scalar(Z[j], 0.0);
    HIP_CHECK(hipMemcpy(d_ritz_coeffs_, h_ritz.data(),
                        niter * sizeof(Scalar), hipMemcpyHostToDevice));

    // theta = V * ritz_coeffs  (single gemv in host pointer mode)
    Scalar one = Traits::one(), zero_val = Traits::zero();
    ROCBLAS_CHECK(Traits::gemv(rocblas_h_, rocblas_operation_none,
        n, niter, &one,
        d_lanczos_v, n,
        d_ritz_coeffs_, 1,
        &zero_val, d_theta, 1));

    // Normalize theta
    double theta_norm;
    ROCBLAS_CHECK(Traits::nrm2(rocblas_h_, n, d_theta, 1, &theta_norm));
    if (theta_norm > 0) {
        double inv_tn = 1.0 / theta_norm;
        ROCBLAS_CHECK(Traits::scal_real(rocblas_h_, n, &inv_tn, d_theta, 1));
    }

    return energy;
}

// ============================================================================
// SVD and MPS update — naive: rocsolver gesvd + host-side truncation
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

    // Copy U, S, Vh back to host for truncation and scaling
    HIP_CHECK(hipMemcpy(h_svd_S_.data(), d_svd_S_,
                        full_k * sizeof(RealType), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_svd_U_.data(), d_svd_U_,
                        m * full_k * sizeof(Scalar), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_svd_Vh_.data(), d_svd_Vh_,
                        full_k * n_svd * sizeof(Scalar), hipMemcpyDeviceToHost));
    HIP_CHECK(hipStreamSynchronize(stream_));

    // Host-side truncation: pick the largest new_k <= k singular values above 1e-14
    int new_k = 0;
    for (int j = 0; j < k; j++) {
        if (h_svd_S_[j] > 1e-14) new_k++;
        else break;
    }
    if (new_k == 0) new_k = 1;

    if (direction == 'R') {
        int new_chi_R = new_k;

        // MPS[site] = U[:, :new_k]  (column slice — col-major, contiguous)
        allocate_mps_tensor(site, cL, new_chi_R);
        HIP_CHECK(hipMemcpy(d_mps_tensors_[site], h_svd_U_.data(),
                            m * new_k * sizeof(Scalar), hipMemcpyHostToDevice));

        // S*Vh on host: scale row j of Vh[:new_k, :] by S[j], put into h_svd_tmp_
        // Col-major: element (j, c) at index c*full_k + j  (source) → c*new_k + j (dest)
        for (int c = 0; c < n_svd; c++) {
            for (int j = 0; j < new_k; j++) {
                h_svd_tmp_[c * new_k + j] =
                    Traits::scale_by_real(h_svd_S_[j], h_svd_Vh_[c * full_k + j]);
            }
        }
        HIP_CHECK(hipMemcpy(d_svd_A_, h_svd_tmp_.data(),
                            new_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice));

        // Absorb S*Vh into A[site+1]: new_chi_R × (d * next_cR)
        if (site + 1 < L_) {
            int next_cR = chi_R(site + 1);
            Scalar one = Traits::one(), zero_val = Traits::zero();
            ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
                rocblas_operation_none, rocblas_operation_none,
                new_k, d_ * next_cR, cR, &one,
                d_svd_A_, new_k,
                d_mps_tensors_[site + 1], cR, &zero_val,
                d_T1_, new_k));
            allocate_mps_tensor(site + 1, new_chi_R, next_cR);
            HIP_CHECK(hipMemcpy(d_mps_tensors_[site + 1], d_T1_,
                                new_k * d_ * next_cR * sizeof(Scalar),
                                hipMemcpyDeviceToDevice));
        }
        bond_dims_[site + 1] = new_chi_R;

    } else {  // direction == 'L'
        int new_chi_L = new_k;

        // MPS[site] = Vh[:new_k, :]  (row slice — need to repack col-major)
        // Source: h_svd_Vh_[c * full_k + j] for c in [0,n_svd), j in [0,new_k)
        // Dest:   d_mps_tensors_[site] with ldA = new_chi_L
        for (int c = 0; c < n_svd; c++) {
            for (int j = 0; j < new_k; j++) {
                h_svd_tmp_[c * new_chi_L + j] = h_svd_Vh_[c * full_k + j];
            }
        }
        allocate_mps_tensor(site, new_chi_L, cR);
        HIP_CHECK(hipMemcpy(d_mps_tensors_[site], h_svd_tmp_.data(),
                            new_chi_L * n_svd * sizeof(Scalar),
                            hipMemcpyHostToDevice));

        // U*S on host: scale column j of U[:, :new_k] by S[j]
        for (int j = 0; j < new_k; j++) {
            for (int r = 0; r < m; r++) {
                h_svd_tmp_[j * m + r] =
                    Traits::scale_by_real(h_svd_S_[j], h_svd_U_[j * m + r]);
            }
        }
        HIP_CHECK(hipMemcpy(d_svd_A_, h_svd_tmp_.data(),
                            m * new_k * sizeof(Scalar), hipMemcpyHostToDevice));

        // Absorb U*S into A[site-1]: (prev_cL * d) × new_k
        if (site > 0) {
            int prev_cL = chi_L(site - 1);
            Scalar one = Traits::one(), zero_val = Traits::zero();
            ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
                rocblas_operation_none, rocblas_operation_none,
                prev_cL * d_, new_k, m, &one,
                d_mps_tensors_[site - 1], prev_cL * d_,
                d_svd_A_, m, &zero_val,
                d_T1_, prev_cL * d_));
            allocate_mps_tensor(site - 1, prev_cL, new_chi_L);
            HIP_CHECK(hipMemcpy(d_mps_tensors_[site - 1], d_T1_,
                                prev_cL * d_ * new_k * sizeof(Scalar),
                                hipMemcpyDeviceToDevice));
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
        HIP_CHECK(hipMemcpy(d_mps_tensors_[site], d_theta_, sz * sizeof(Scalar),
                            hipMemcpyDeviceToDevice));
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
        HIP_CHECK(hipMemcpy(d_mps_tensors_[site], d_theta_, sz * sizeof(Scalar),
                            hipMemcpyDeviceToDevice));
    }

    return energy;
}

// ============================================================================
// Main algorithm
// ============================================================================

template<typename Scalar>
double DMRGGPUBase<Scalar>::run(int n_sweeps) {
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
