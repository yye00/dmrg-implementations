#ifndef DMRG_GPU_IMPL_H
#define DMRG_GPU_IMPL_H

#include <cusolverDn.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <chrono>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            throw std::runtime_error("CUDA error"); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error in " << __FILE__ << ":" << __LINE__ \
                      << " - status " << status << std::endl; \
            throw std::runtime_error("cuBLAS error"); \
        } \
    } while(0)

#define CUSOLVER_CHECK(call) \
    do { \
        cusolverStatus_t s = call; \
        if (s != CUSOLVER_STATUS_SUCCESS) { \
            std::cerr << "cuSOLVER error in " << __FILE__ << ":" << __LINE__ \
                      << " - status " << s << std::endl; \
            throw std::runtime_error("cuSOLVER error"); \
        } \
    } while(0)


// ============================================================================
// Constructor
// ============================================================================

template<typename Scalar>
DMRGGPU<Scalar>::DMRGGPU(int L, int d, int chi_max, int D_mpo, double tol)
    : L_(L), d_(d), chi_max_(chi_max), D_mpo_(D_mpo), tol_(tol), energy_(0.0) {

    // Bond dimensions
    bond_dims_.resize(L + 1);
    bond_dims_[0] = 1;
    bond_dims_[L] = 1;
    for (int i = 1; i < L; i++) {
        double exact_dim = pow((double)d_, std::min(i, L - i));
        bond_dims_[i] = (exact_dim > chi_max_) ? chi_max_ : (int)exact_dim;
    }

    // GPU handles
    CUDA_CHECK(cudaStreamCreate(&stream_));
    CUBLAS_CHECK(cublasCreate(&cublas_h_));
    CUBLAS_CHECK(cublasSetStream(cublas_h_, stream_));
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver_h_));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolver_h_, stream_));

    // Contraction intermediates
    int t_max = D_mpo_ * d_ * chi_max_ * chi_max_;
    CUDA_CHECK(cudaMalloc(&d_T1_, t_max * sizeof(Scalar)));
    CUDA_CHECK(cudaMalloc(&d_T2_, t_max * sizeof(Scalar)));

    // MPS tensors
    d_mps_tensors_.resize(L, nullptr);
    for (int i = 0; i < L; i++) {
        allocate_mps_tensor(i, chi_L(i), chi_R(i));
    }

    // MPO tensors
    d_mpo_tensors_.resize(L, nullptr);

    // W matrices (allocated in set_mpo)
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
        CUDA_CHECK(cudaMalloc(&d_L_envs_[i], sz * sizeof(Scalar)));
        CUDA_CHECK(cudaMalloc(&d_R_envs_[i], sz * sizeof(Scalar)));
        CUDA_CHECK(cudaMemset(d_L_envs_[i], 0, sz * sizeof(Scalar)));
        CUDA_CHECK(cudaMemset(d_R_envs_[i], 0, sz * sizeof(Scalar)));
        L_env_alloc_chi_[i] = chi_alloc;
        R_env_alloc_chi_[i] = chi_alloc;
    }

    // Lanczos workspace
    theta_size_max_ = chi_max_ * d_ * chi_max_;
    max_lanczos_iter_ = std::min(100, theta_size_max_);
    CUDA_CHECK(cudaMalloc(&d_theta_, theta_size_max_ * sizeof(Scalar)));
    CUDA_CHECK(cudaMalloc(&d_heff_result_, theta_size_max_ * sizeof(Scalar)));
    CUDA_CHECK(cudaMalloc(&d_lanczos_v_, (size_t)max_lanczos_iter_ * theta_size_max_ * sizeof(Scalar)));
    CUDA_CHECK(cudaMalloc(&d_ritz_coeffs_, max_lanczos_iter_ * sizeof(Scalar)));

    // Device scalars for sync-free Lanczos
    CUDA_CHECK(cudaMalloc(&d_dot_result_, sizeof(Scalar)));
    CUDA_CHECK(cudaMalloc(&d_nrm2_result_, sizeof(RealType)));
    CUDA_CHECK(cudaMalloc(&d_neg_alpha_, sizeof(Scalar)));
    CUDA_CHECK(cudaMalloc(&d_neg_overlap_, sizeof(Scalar)));
    CUDA_CHECK(cudaMalloc(&d_inv_nrm_, sizeof(RealType)));
    CUDA_CHECK(cudaMalloc(&d_alpha_dev_, max_lanczos_iter_ * sizeof(RealType)));
    CUDA_CHECK(cudaMalloc(&d_beta_dev_, max_lanczos_iter_ * sizeof(RealType)));
    CUDA_CHECK(cudaMalloc(&d_neg_beta_scalars_, max_lanczos_iter_ * sizeof(Scalar)));
    CUDA_CHECK(cudaMalloc(&d_const_one_, sizeof(Scalar)));
    CUDA_CHECK(cudaMalloc(&d_const_zero_, sizeof(Scalar)));
    CUDA_CHECK(cudaMalloc(&d_const_neg_one_, sizeof(Scalar)));
    {
        Scalar one = Traits::one(), zero = Traits::zero(), neg_one = Traits::neg(one);
        CUDA_CHECK(cudaMemcpy(d_const_one_, &one, sizeof(Scalar), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_const_zero_, &zero, sizeof(Scalar), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_const_neg_one_, &neg_one, sizeof(Scalar), cudaMemcpyHostToDevice));
    }

    // Batched GEMM pointer arrays
    int batch_max = D_mpo_ * d_;
    CUDA_CHECK(cudaMalloc(&d_batch_A_, batch_max * sizeof(Scalar*)));
    CUDA_CHECK(cudaMalloc(&d_batch_B_, batch_max * sizeof(Scalar*)));
    CUDA_CHECK(cudaMalloc(&d_batch_C_, batch_max * sizeof(Scalar*)));

    // SVD workspace - query cuSOLVER for workspace size
    int svd_max_dim = chi_max_ * d_;
    CUDA_CHECK(cudaMalloc(&d_svd_A_,    theta_size_max_ * sizeof(Scalar)));
    CUDA_CHECK(cudaMalloc(&d_svd_U_,    svd_max_dim * chi_max_ * sizeof(Scalar)));
    CUDA_CHECK(cudaMalloc(&d_svd_S_,    chi_max_ * sizeof(RealType)));
    CUDA_CHECK(cudaMalloc(&d_svd_Vh_,   chi_max_ * svd_max_dim * sizeof(Scalar)));
    CUDA_CHECK(cudaMalloc(&d_svd_info_, sizeof(int)));

    // Query SVD workspace size for max dimensions
    CUSOLVER_CHECK(Traits::cusolver_gesvd_bufferSize(cusolver_h_, svd_max_dim, svd_max_dim, &svd_lwork_));
    CUDA_CHECK(cudaMalloc(&d_svd_work_, svd_lwork_ * sizeof(Scalar)));

    // rwork for complex SVD
    int rwork_size = Traits::svd_rwork_size(svd_max_dim, svd_max_dim);
    if (rwork_size > 0) {
        CUDA_CHECK(cudaMalloc(&d_svd_rwork_, rwork_size * sizeof(RealType)));
    } else {
        d_svd_rwork_ = nullptr;
    }

    // Host workspace for SVD results (copied back from GPU)
    h_svd_U_.resize(svd_max_dim * chi_max_);
    h_svd_S_.resize(chi_max_);
    h_svd_Vh_.resize(chi_max_ * svd_max_dim);
    h_svd_tmp_.resize(svd_max_dim * chi_max_);
}

// ============================================================================
// Destructor
// ============================================================================

template<typename Scalar>
DMRGGPU<Scalar>::~DMRGGPU() {
    free_gpu_resources();
}

template<typename Scalar>
void DMRGGPU<Scalar>::free_gpu_resources() {
    for (auto ptr : d_mps_tensors_) if (ptr) cudaFree(ptr);
    for (auto ptr : d_mpo_tensors_) if (ptr) cudaFree(ptr);
    for (auto ptr : d_W_left_) if (ptr) cudaFree(ptr);
    for (auto ptr : d_W_right_) if (ptr) cudaFree(ptr);
    for (auto ptr : d_L_envs_) if (ptr) cudaFree(ptr);
    for (auto ptr : d_R_envs_) if (ptr) cudaFree(ptr);

    if (d_theta_) cudaFree(d_theta_);
    if (d_heff_result_) cudaFree(d_heff_result_);
    if (d_lanczos_v_) cudaFree(d_lanczos_v_);
    if (d_ritz_coeffs_) cudaFree(d_ritz_coeffs_);
    if (d_dot_result_) cudaFree(d_dot_result_);
    if (d_nrm2_result_) cudaFree(d_nrm2_result_);
    if (d_neg_alpha_) cudaFree(d_neg_alpha_);
    if (d_neg_overlap_) cudaFree(d_neg_overlap_);
    if (d_inv_nrm_) cudaFree(d_inv_nrm_);
    if (d_alpha_dev_) cudaFree(d_alpha_dev_);
    if (d_beta_dev_) cudaFree(d_beta_dev_);
    if (d_neg_beta_scalars_) cudaFree(d_neg_beta_scalars_);
    if (d_const_one_) cudaFree(d_const_one_);
    if (d_const_zero_) cudaFree(d_const_zero_);
    if (d_const_neg_one_) cudaFree(d_const_neg_one_);
    if (d_T1_) cudaFree(d_T1_);
    if (d_T2_) cudaFree(d_T2_);
    if (d_batch_A_) cudaFree(d_batch_A_);
    if (d_batch_B_) cudaFree(d_batch_B_);
    if (d_batch_C_) cudaFree(d_batch_C_);
    if (d_svd_A_) cudaFree(d_svd_A_);
    if (d_svd_U_) cudaFree(d_svd_U_);
    if (d_svd_S_) cudaFree(d_svd_S_);
    if (d_svd_Vh_) cudaFree(d_svd_Vh_);
    if (d_svd_work_) cudaFree(d_svd_work_);
    if (d_svd_rwork_) cudaFree(d_svd_rwork_);
    if (d_svd_info_) cudaFree(d_svd_info_);

    cusolverDnDestroy(cusolver_h_);
    cublasDestroy(cublas_h_);
    cudaStreamDestroy(stream_);
}

// ============================================================================
// Memory management
// ============================================================================

template<typename Scalar>
void DMRGGPU<Scalar>::allocate_mps_tensor(int site, int cL, int cR) {
    // Pre-allocate at chi_max to avoid cudaFree/cudaMalloc per bond (sync points)
    size_t needed = (size_t)cL * d_ * cR * sizeof(Scalar);
    size_t max_sz = (size_t)chi_max_ * d_ * chi_max_ * sizeof(Scalar);
    if (!d_mps_tensors_[site]) {
        CUDA_CHECK(cudaMalloc(&d_mps_tensors_[site], max_sz));
    }
    (void)needed;  // logical size tracked by bond_dims_
}

template<typename Scalar>
void DMRGGPU<Scalar>::ensure_L_env_alloc(int idx, int chi) {
    if (chi > L_env_alloc_chi_[idx]) {
        if (d_L_envs_[idx]) CUDA_CHECK(cudaFree(d_L_envs_[idx]));
        int sz = chi * D_mpo_ * chi;
        CUDA_CHECK(cudaMalloc(&d_L_envs_[idx], sz * sizeof(Scalar)));
        L_env_alloc_chi_[idx] = chi;
    }
}

template<typename Scalar>
void DMRGGPU<Scalar>::ensure_R_env_alloc(int idx, int chi) {
    if (chi > R_env_alloc_chi_[idx]) {
        if (d_R_envs_[idx]) CUDA_CHECK(cudaFree(d_R_envs_[idx]));
        int sz = chi * D_mpo_ * chi;
        CUDA_CHECK(cudaMalloc(&d_R_envs_[idx], sz * sizeof(Scalar)));
        R_env_alloc_chi_[idx] = chi;
    }
}

// ============================================================================
// MPS initialization
// ============================================================================

template<typename Scalar>
void DMRGGPU<Scalar>::initialize_mps_random(double scale) {
    for (int i = 0; i < L_; i++) {
        int size = chi_L(i) * d_ * chi_R(i);
        std::vector<Scalar> h_A(size);
        for (int j = 0; j < size; j++) {
            h_A[j] = Traits::scale_by_real(scale, Traits::random_val());
        }
        CUDA_CHECK(cudaMemcpy(d_mps_tensors_[i], h_A.data(),
                            size * sizeof(Scalar), cudaMemcpyHostToDevice));
    }
}

template<typename Scalar>
void DMRGGPU<Scalar>::initialize_mps_product() {
    for (int i = 0; i < L_; i++) {
        int cL = chi_L(i), cR = chi_R(i);
        int size = cL * d_ * cR;
        std::vector<Scalar> h_A(size, Traits::zero());
        int chi_min = std::min(cL, cR);
        for (int a = 0; a < chi_min; a++) {
            h_A[a + 0*cL + a*cL*d_] = Traits::one();
        }
        CUDA_CHECK(cudaMemcpy(d_mps_tensors_[i], h_A.data(),
                            size * sizeof(Scalar), cudaMemcpyHostToDevice));
    }
}

template<typename Scalar>
void DMRGGPU<Scalar>::initialize_mps_neel() {
    for (int i = 0; i < L_; i++) {
        int cL = chi_L(i), cR = chi_R(i);
        int size = cL * d_ * cR;
        std::vector<Scalar> h_A(size, Traits::zero());
        int spin = (i % 2 == 0) ? 0 : 1;
        int chi_min = std::min(cL, cR);
        for (int a = 0; a < chi_min; a++) {
            h_A[a + spin*cL + a*cL*d_] = Traits::one();
        }
        CUDA_CHECK(cudaMemcpy(d_mps_tensors_[i], h_A.data(),
                            size * sizeof(Scalar), cudaMemcpyHostToDevice));
    }
}

template<typename Scalar>
void DMRGGPU<Scalar>::set_mpo(const std::vector<Scalar*>& h_mpo_tensors) {
    int D = D_mpo_, d = d_;
    for (int i = 0; i < L_; i++) {
        int size = D * d * d * D;
        CUDA_CHECK(cudaMalloc(&d_mpo_tensors_[i], size * sizeof(Scalar)));
        CUDA_CHECK(cudaMemcpy(d_mpo_tensors_[i], h_mpo_tensors[i],
                            size * sizeof(Scalar), cudaMemcpyHostToDevice));

        // Precompute W_left and W_right matrices
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
        CUDA_CHECK(cudaMalloc(&d_W_left_[i], wm_size * sizeof(Scalar)));
        CUDA_CHECK(cudaMemcpy(d_W_left_[i], h_WL.data(),
                            wm_size * sizeof(Scalar), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_W_right_[i], wm_size * sizeof(Scalar)));
        CUDA_CHECK(cudaMemcpy(d_W_right_[i], h_WR.data(),
                            wm_size * sizeof(Scalar), cudaMemcpyHostToDevice));
    }
}

// ============================================================================
// GEMM-based tensor contractions
// ============================================================================

template<typename Scalar>
void DMRGGPU<Scalar>::apply_heff(int site, const Scalar* d_theta_in, Scalar* d_result) {
    int cL = chi_L(site);
    int cR = chi_R(site);
    int D = D_mpo_, d = d_;
    Scalar one = Traits::one(), zero_val = Traits::zero();

    Scalar* L_env = d_L_envs_[site];
    Scalar* R_env = d_R_envs_[site + 1];
    Scalar* W_mat = d_W_left_[site];
    Scalar* V = d_T1_;
    Scalar* U = d_T2_;

    // Step 1: V_ws[a',b] = L_w^T[a',a] * theta_s[a,b]  (batched GEMM)
    {
        Scalar* h_A[D * d], *h_B[D * d], *h_C[D * d];
        for (int w = 0; w < D; w++)
            for (int s = 0; s < d; s++) {
                int ws = w * d + s;
                h_A[ws] = L_env + w * cL;
                h_B[ws] = const_cast<Scalar*>(d_theta_in) + s * cL;
                h_C[ws] = V + ws * cL * cR;
            }
        CUDA_CHECK(cudaMemcpyAsync(d_batch_A_, h_A, D*d*sizeof(Scalar*), cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(d_batch_B_, h_B, D*d*sizeof(Scalar*), cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(d_batch_C_, h_C, D*d*sizeof(Scalar*), cudaMemcpyHostToDevice, stream_));
        CUBLAS_CHECK(Traits::gemm_batched(cublas_h_,
            Traits::op_t, CUBLAS_OP_N,
            cL, cR, cL,
            &one,
            (const Scalar**)d_batch_A_, cL * D,
            (const Scalar**)d_batch_B_, cL * d,
            &zero_val,
            d_batch_C_, cL,
            D * d));
    }

    // Step 2: U = V * W_matrix
    CUBLAS_CHECK(Traits::gemm(cublas_h_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        cL * cR, d * D, D * d,
        &one,
        V, cL * cR,
        W_mat, D * d,
        &zero_val,
        U, cL * cR));

    // Step 3: result_s'[a',b'] = sum_w' U_{w'd+s'}[a',b] * R_w'[b,b']  (batched GEMM)
    // Batch d GEMMs per wp (safe: different sp write to different C locations).
    // Cannot batch across wp since same sp/different wp accumulate into same C.
    {
        Scalar* h_A3[D * d], *h_B3[D * d], *h_C3[D * d];
        for (int wp = 0; wp < D; wp++) {
            Scalar beta = (wp == 0) ? Traits::zero() : Traits::one();
            for (int sp = 0; sp < d; sp++) {
                h_A3[sp] = U + (wp * d + sp) * cL * cR;
                h_B3[sp] = R_env + wp * cR;
                h_C3[sp] = d_result + sp * cL;
            }
            CUDA_CHECK(cudaMemcpyAsync(d_batch_A_, h_A3, d*sizeof(Scalar*), cudaMemcpyHostToDevice, stream_));
            CUDA_CHECK(cudaMemcpyAsync(d_batch_B_, h_B3, d*sizeof(Scalar*), cudaMemcpyHostToDevice, stream_));
            CUDA_CHECK(cudaMemcpyAsync(d_batch_C_, h_C3, d*sizeof(Scalar*), cudaMemcpyHostToDevice, stream_));
            CUBLAS_CHECK(Traits::gemm_batched(cublas_h_,
                CUBLAS_OP_N, CUBLAS_OP_N,
                cL, cR, cR,
                &one,
                (const Scalar**)d_batch_A_, cL,
                (const Scalar**)d_batch_B_, cR * D,
                &beta,
                d_batch_C_, cL * d,
                d));
        }
    }
}

// ============================================================================
// Left environment update
// ============================================================================

template<typename Scalar>
void DMRGGPU<Scalar>::update_left_env(int site) {
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

    // Step 1: V_ws[a',b] = L_w^T[a',a] * A_s[a,b]  (batched GEMM)
    {
        Scalar* h_A[D * d], *h_B[D * d], *h_C[D * d];
        for (int w = 0; w < D; w++)
            for (int s = 0; s < d; s++) {
                int ws = w * d + s;
                h_A[ws] = L_env + w * chi_in;
                h_B[ws] = A + s * chi_in;
                h_C[ws] = V + ws * chi_in * chi_out;
            }
        CUDA_CHECK(cudaMemcpyAsync(d_batch_A_, h_A, D*d*sizeof(Scalar*), cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(d_batch_B_, h_B, D*d*sizeof(Scalar*), cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(d_batch_C_, h_C, D*d*sizeof(Scalar*), cudaMemcpyHostToDevice, stream_));
        CUBLAS_CHECK(Traits::gemm_batched(cublas_h_,
            Traits::op_t, CUBLAS_OP_N,
            chi_in, chi_out, chi_in,
            &one,
            (const Scalar**)d_batch_A_, chi_in * D,
            (const Scalar**)d_batch_B_, chi_in * d,
            &zero_val,
            d_batch_C_, chi_in,
            D * d));
    }

    // Step 2: U = V * W_matrix
    CUBLAS_CHECK(Traits::gemm(cublas_h_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        chi_in * chi_out, d * D, D * d,
        &one,
        V, chi_in * chi_out,
        W_mat, D * d,
        &zero_val,
        U, chi_in * chi_out));

    // Step 3: L_new_w'[b,b'] = sum_{a',s'} U[a',ws',b] * conj(A[a',s',b'])  (batched)
    //
    // gemm(op_h, N, U, A) computes U^H * A = conj(desired result) for complex.
    // For real, U^T * A is correct directly.
    // For complex, we conjugate L_new after the loop.
    // Batch D GEMMs per sp (safe: different wp write to different C locations).
    {
        Scalar* h_A3[D * d], *h_B3[D * d], *h_C3[D * d];
        for (int sp = 0; sp < d; sp++) {
            Scalar beta = (sp == 0) ? Traits::zero() : Traits::one();
            for (int wp = 0; wp < D; wp++) {
                h_A3[wp] = U + (wp * d + sp) * chi_in * chi_out;
                h_B3[wp] = A + sp * chi_in;
                h_C3[wp] = L_new + wp * chi_out;
            }
            CUDA_CHECK(cudaMemcpyAsync(d_batch_A_, h_A3, D*sizeof(Scalar*), cudaMemcpyHostToDevice, stream_));
            CUDA_CHECK(cudaMemcpyAsync(d_batch_B_, h_B3, D*sizeof(Scalar*), cudaMemcpyHostToDevice, stream_));
            CUDA_CHECK(cudaMemcpyAsync(d_batch_C_, h_C3, D*sizeof(Scalar*), cudaMemcpyHostToDevice, stream_));
            CUBLAS_CHECK(Traits::gemm_batched(cublas_h_,
                Traits::op_h, CUBLAS_OP_N,
                chi_out, chi_out, chi_in,
                &one,
                (const Scalar**)d_batch_A_, chi_in,
                (const Scalar**)d_batch_B_, chi_in * d,
                &beta,
                d_batch_C_, chi_out * D,
                D));
        }
    }

    // For complex: L_new = conj(U^H * A) = U^T * conj(A), the correct bra contraction
    if constexpr (Traits::is_complex) {
        conjugate_inplace(L_new, chi_out * D * chi_out, stream_);
    }
}

// ============================================================================
// Right environment update
// ============================================================================

template<typename Scalar>
void DMRGGPU<Scalar>::update_right_env(int site) {
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

    // Step 1: V_ws[a,b'] = A_s[a,b] * R_w'[b,b']  (batched GEMM)
    {
        Scalar* h_A[D * d], *h_B[D * d], *h_C[D * d];
        for (int wp = 0; wp < D; wp++)
            for (int s = 0; s < d; s++) {
                int ws = wp * d + s;
                h_A[ws] = A + s * chi_out;
                h_B[ws] = R_env + wp * chi_in;
                h_C[ws] = V + ws * chi_out * chi_in;
            }
        CUDA_CHECK(cudaMemcpyAsync(d_batch_A_, h_A, D*d*sizeof(Scalar*), cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(d_batch_B_, h_B, D*d*sizeof(Scalar*), cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(d_batch_C_, h_C, D*d*sizeof(Scalar*), cudaMemcpyHostToDevice, stream_));
        CUBLAS_CHECK(Traits::gemm_batched(cublas_h_,
            CUBLAS_OP_N, CUBLAS_OP_N,
            chi_out, chi_in, chi_in,
            &one,
            (const Scalar**)d_batch_A_, chi_out * d,
            (const Scalar**)d_batch_B_, chi_in * D,
            &zero_val,
            d_batch_C_, chi_out,
            D * d));
    }

    // Step 2: U = V * W_matrix
    CUBLAS_CHECK(Traits::gemm(cublas_h_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        chi_out * chi_in, d * D, D * d,
        &one,
        V, chi_out * chi_in,
        W_mat, D * d,
        &zero_val,
        U, chi_out * chi_in));

    // Step 3: R_new_w[a,a'] = sum_s' U_ws'[a,b'] * A_s'^H[b',a']  (batched)
    // For complex: use conjugate transpose (op_h) on A for <bra|
    // Batch D GEMMs per sp (safe: different w write to different C locations).
    {
        Scalar* h_A3[D * d], *h_B3[D * d], *h_C3[D * d];
        for (int sp = 0; sp < d; sp++) {
            Scalar beta = (sp == 0) ? Traits::zero() : Traits::one();
            for (int w = 0; w < D; w++) {
                h_A3[w] = U + (w * d + sp) * chi_out * chi_in;
                h_B3[w] = A + sp * chi_out;
                h_C3[w] = R_new + w * chi_out;
            }
            CUDA_CHECK(cudaMemcpyAsync(d_batch_A_, h_A3, D*sizeof(Scalar*), cudaMemcpyHostToDevice, stream_));
            CUDA_CHECK(cudaMemcpyAsync(d_batch_B_, h_B3, D*sizeof(Scalar*), cudaMemcpyHostToDevice, stream_));
            CUDA_CHECK(cudaMemcpyAsync(d_batch_C_, h_C3, D*sizeof(Scalar*), cudaMemcpyHostToDevice, stream_));
            CUBLAS_CHECK(Traits::gemm_batched(cublas_h_,
                CUBLAS_OP_N, Traits::op_h,
                chi_out, chi_out, chi_in,
                &one,
                (const Scalar**)d_batch_A_, chi_out,
                (const Scalar**)d_batch_B_, chi_out * d,
                &beta,
                d_batch_C_, chi_out * D,
                D));
        }
    }
}

// ============================================================================
// Environment building
// ============================================================================

template<typename Scalar>
void DMRGGPU<Scalar>::build_initial_environments() {
    // L[0] = trivial left boundary: (1, D_mpo, 1), L[0][0,0,0] = 1
    {
        std::vector<Scalar> h_L(D_mpo_, Traits::zero());
        h_L[0] = Traits::one();
        CUDA_CHECK(cudaMemcpy(d_L_envs_[0], h_L.data(),
                            D_mpo_ * sizeof(Scalar), cudaMemcpyHostToDevice));
    }

    // R[L] = trivial right boundary: (1, D_mpo, 1), R[L][0,D-1,0] = 1
    {
        std::vector<Scalar> h_R(D_mpo_, Traits::zero());
        h_R[D_mpo_ - 1] = Traits::one();
        CUDA_CHECK(cudaMemcpy(d_R_envs_[L_], h_R.data(),
                            D_mpo_ * sizeof(Scalar), cudaMemcpyHostToDevice));
    }

    // Build all R environments from right to left
    for (int i = L_ - 1; i >= 0; i--) {
        update_right_env(i);
    }
}

// ============================================================================
// Theta formation and Lanczos
// ============================================================================

template<typename Scalar>
void DMRGGPU<Scalar>::form_theta(int site, Scalar* d_theta) {
    int size = chi_L(site) * d_ * chi_R(site);
    CUDA_CHECK(cudaMemcpy(d_theta, d_mps_tensors_[site],
                        size * sizeof(Scalar), cudaMemcpyDeviceToDevice));
}

template<typename Scalar>
double DMRGGPU<Scalar>::lanczos_eigensolver(int site, Scalar* d_theta) {
    int n = chi_L(site) * d_ * chi_R(site);
    int max_iter = std::min(max_lanczos_iter_, n);
    double tol_lanczos = 1e-12;
    double tol_eig_conv = 1e-12;

    Scalar* d_lanczos_v = d_lanczos_v_;
    std::vector<double> h_alpha(max_iter);
    std::vector<double> h_beta(max_iter);

    // v[0] = theta / ||theta|| (host pointer mode for initial setup)
    double norm;
    CUBLAS_CHECK(Traits::nrm2(cublas_h_, n, d_theta, 1, &norm));

    if (norm < 1e-14) {
        std::vector<Scalar> h_init(n);
        srand(42 + site);
        for (int i = 0; i < n; i++) h_init[i] = Traits::random_val();
        CUDA_CHECK(cudaMemcpy(d_theta, h_init.data(), n * sizeof(Scalar), cudaMemcpyHostToDevice));
        CUBLAS_CHECK(Traits::nrm2(cublas_h_, n, d_theta, 1, &norm));
    }

    double inv_norm = 1.0 / norm;
    CUBLAS_CHECK(Traits::scal_real(cublas_h_, n, &inv_norm, d_theta, 1));
    CUDA_CHECK(cudaMemcpy(d_lanczos_v, d_theta, n * sizeof(Scalar), cudaMemcpyDeviceToDevice));

    double prev_energy = 1e30;
    int iter;
    int last_synced_iter = -1;

    for (iter = 0; iter < max_iter; iter++) {
        Scalar* d_vi = d_lanczos_v + (size_t)iter * n;

        // w = H|v_i> (apply_heff uses host pointer mode internally)
        apply_heff(site, d_vi, d_heff_result_);

        // Switch to device pointer mode for scalar operations
        CUBLAS_CHECK(cublasSetPointerMode(cublas_h_, CUBLAS_POINTER_MODE_DEVICE));

        // alpha_i = <v_i|w> -> device
        CUBLAS_CHECK(Traits::dot(cublas_h_, n, d_vi, 1, d_heff_result_, 1, d_dot_result_));

        // Process alpha: store to d_alpha_dev_[iter], compute d_neg_alpha_
        lanczos_process_alpha_kernel<Scalar><<<1, 1, 0, stream_>>>(
            d_dot_result_, d_neg_alpha_, d_alpha_dev_, iter);

        // w -= alpha_i * v_i (device pointer)
        CUBLAS_CHECK(Traits::axpy(cublas_h_, n, d_neg_alpha_, d_vi, 1, d_heff_result_, 1));

        // w -= beta_{i-1} * v_{i-1} (device pointer)
        if (iter > 0) {
            CUBLAS_CHECK(Traits::axpy(cublas_h_, n,
                d_neg_beta_scalars_ + (iter - 1),
                d_lanczos_v + (size_t)(iter - 1) * n, 1,
                d_heff_result_, 1));
        }

        // Full reorthogonalization (device pointer mode for gemv constants)
        if (iter > 0) {
            CUBLAS_CHECK(Traits::gemv(cublas_h_, Traits::op_h,
                n, iter + 1, d_const_one_,
                d_lanczos_v, n,
                d_heff_result_, 1,
                d_const_zero_, d_ritz_coeffs_, 1));
            CUBLAS_CHECK(Traits::gemv(cublas_h_, CUBLAS_OP_N,
                n, iter + 1, d_const_neg_one_,
                d_lanczos_v, n,
                d_ritz_coeffs_, 1,
                d_const_one_, d_heff_result_, 1));
        } else {
            CUBLAS_CHECK(Traits::dot(cublas_h_, n, d_lanczos_v, 1, d_heff_result_, 1, d_dot_result_));
            negate_scalar_kernel<Scalar><<<1, 1, 0, stream_>>>(
                d_dot_result_, d_neg_overlap_);
            CUBLAS_CHECK(Traits::axpy(cublas_h_, n, d_neg_overlap_, d_lanczos_v, 1, d_heff_result_, 1));
        }

        // beta_i = ||w|| -> device
        CUBLAS_CHECK(Traits::nrm2(cublas_h_, n, d_heff_result_, 1, d_nrm2_result_));

        // Process beta: store, compute 1/beta, store -beta as Scalar
        lanczos_process_beta_kernel<Scalar><<<1, 1, 0, stream_>>>(
            d_nrm2_result_, d_inv_nrm_, d_beta_dev_, d_neg_beta_scalars_, iter);

        // v_{i+1} = w / beta_i (device pointer for scal)
        if (iter + 1 < max_iter) {
            Scalar* d_vip1 = d_lanczos_v + (size_t)(iter + 1) * n;
            CUDA_CHECK(cudaMemcpy(d_vip1, d_heff_result_, n * sizeof(Scalar), cudaMemcpyDeviceToDevice));
            CUBLAS_CHECK(Traits::scal_real(cublas_h_, n, d_inv_nrm_, d_vip1, 1));
        }

        // Switch back to host pointer mode
        CUBLAS_CHECK(cublasSetPointerMode(cublas_h_, CUBLAS_POINTER_MODE_HOST));

        // Convergence check every 3 iterations after iter >= 4
        if (iter >= 4 && iter % 3 == 0) {
            CUDA_CHECK(cudaStreamSynchronize(stream_));

            int n_copy = iter + 1;
            CUDA_CHECK(cudaMemcpy(h_alpha.data(), d_alpha_dev_, n_copy * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_beta.data(), d_beta_dev_, n_copy * sizeof(double), cudaMemcpyDeviceToHost));
            last_synced_iter = iter;

            bool early_break = false;
            for (int j = 0; j <= iter; j++) {
                if (h_beta[j] < tol_lanczos) {
                    iter = j + 1;
                    early_break = true;
                    break;
                }
            }
            if (early_break) break;

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

    CUDA_CHECK(cudaStreamSynchronize(stream_));

    if (last_synced_iter < niter - 1) {
        CUDA_CHECK(cudaMemcpy(h_alpha.data(), d_alpha_dev_, niter * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_beta.data(), d_beta_dev_, niter * sizeof(double), cudaMemcpyDeviceToHost));
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

    // Reconstruct ground state: |theta> = sum_i c[i] |v_i>
    std::vector<Scalar> h_ritz_scalar(niter);
    for (int i = 0; i < niter; i++) {
        h_ritz_scalar[i] = Traits::make_scalar(h_Z[i]);
    }
    CUDA_CHECK(cudaMemcpy(d_ritz_coeffs_, h_ritz_scalar.data(), niter * sizeof(Scalar), cudaMemcpyHostToDevice));

    // Use device pointer mode for finalization
    CUBLAS_CHECK(cublasSetPointerMode(cublas_h_, CUBLAS_POINTER_MODE_DEVICE));
    CUBLAS_CHECK(Traits::gemv(
        cublas_h_, CUBLAS_OP_N,
        n, niter, d_const_one_,
        d_lanczos_v, n,
        d_ritz_coeffs_, 1,
        d_const_zero_, d_theta, 1
    ));

    // Normalize theta (device pointer mode)
    CUBLAS_CHECK(Traits::nrm2(cublas_h_, n, d_theta, 1, d_nrm2_result_));
    invert_nrm_kernel<RealType><<<1, 1, 0, stream_>>>(
        d_nrm2_result_, d_inv_nrm_);
    CUBLAS_CHECK(Traits::scal_real(cublas_h_, n, d_inv_nrm_, d_theta, 1));
    CUBLAS_CHECK(cublasSetPointerMode(cublas_h_, CUBLAS_POINTER_MODE_HOST));

    return energy;
}

// ============================================================================
// SVD and MPS update
// ============================================================================

template<typename Scalar>
void DMRGGPU<Scalar>::svd_and_update_mps(int site, Scalar* d_theta, char direction) {
    int cL = chi_L(site);
    int cR = chi_R(site);

    int m, n_svd;
    if (direction == 'R') { m = cL * d_; n_svd = cR; }
    else                  { m = cL;      n_svd = d_ * cR; }
    int full_k = std::min(m, n_svd);
    int k = std::min(full_k, chi_max_);

    // GPU SVD via cuSOLVER gesvd
    CUDA_CHECK(cudaMemcpy(d_svd_A_, d_theta, m * n_svd * sizeof(Scalar), cudaMemcpyDeviceToDevice));

    // Query workspace size for this specific dimension
    int lwork;
    CUSOLVER_CHECK(Traits::cusolver_gesvd_bufferSize(cusolver_h_, m, n_svd, &lwork));
    // Use pre-allocated workspace if large enough, otherwise the pre-allocated svd_lwork_ should cover it

    int rwork_size = Traits::svd_rwork_size(m, n_svd);

    CUSOLVER_CHECK(Traits::cusolver_gesvd(cusolver_h_,
        'S', 'S',
        m, n_svd,
        d_svd_A_, m,
        d_svd_S_,
        d_svd_U_, m,
        d_svd_Vh_, full_k,
        d_svd_work_, svd_lwork_,
        d_svd_rwork_,
        d_svd_info_));

    // Truncation: find new_k on GPU, copy just 1 int back
    svd_truncate_kernel<RealType><<<1, 1, 0, stream_>>>(
        d_svd_S_, k, 1e-14, d_svd_info_);
    int new_k;
    CUDA_CHECK(cudaMemcpy(&new_k, d_svd_info_, sizeof(int), cudaMemcpyDeviceToHost));

    int threads = 256;

    if (direction == 'R') {
        int new_chi_R = new_k;

        // MPS[site] = U[:, :new_k]  (column slice, on GPU)
        allocate_mps_tensor(site, cL, new_chi_R);
        if (new_k == full_k) {
            CUDA_CHECK(cudaMemcpy(d_mps_tensors_[site], d_svd_U_,
                                m * new_k * sizeof(Scalar), cudaMemcpyDeviceToDevice));
        } else {
            int total = m * new_k;
            extract_cols_kernel<Scalar><<<(total+threads-1)/threads, threads, 0, stream_>>>(
                d_svd_U_, m, d_mps_tensors_[site], m, m, new_k);
        }

        // S*Vh -> d_svd_work_ (scale rows of Vh by S, on GPU)
        {
            int total = new_k * n_svd;
            scale_rows_by_diag_kernel<Scalar, RealType><<<(total+threads-1)/threads, threads, 0, stream_>>>(
                d_svd_S_, d_svd_Vh_, full_k, d_svd_work_, new_k, new_k, n_svd);
        }

        // Absorb S*Vh into A[site+1]
        if (site + 1 < L_) {
            int next_cR = chi_R(site + 1);
            Scalar one = Traits::one(), zero_val = Traits::zero();
            CUBLAS_CHECK(Traits::gemm(cublas_h_,
                CUBLAS_OP_N, CUBLAS_OP_N,
                new_k, d_ * next_cR, cR, &one,
                d_svd_work_, new_k,
                d_mps_tensors_[site + 1], cR, &zero_val,
                d_T1_, new_k));
            allocate_mps_tensor(site + 1, new_chi_R, next_cR);
            CUDA_CHECK(cudaMemcpy(d_mps_tensors_[site + 1], d_T1_,
                                new_k * d_ * next_cR * sizeof(Scalar), cudaMemcpyDeviceToDevice));
        }
        bond_dims_[site + 1] = new_chi_R;

    } else {  // direction == 'L'
        int new_chi_L = new_k;

        // MPS[site] = Vh[:new_k, :]  (row slice = column slice of Vh in col-major)
        allocate_mps_tensor(site, new_chi_L, cR);
        if (new_chi_L == full_k) {
            CUDA_CHECK(cudaMemcpy(d_mps_tensors_[site], d_svd_Vh_,
                                full_k * n_svd * sizeof(Scalar), cudaMemcpyDeviceToDevice));
        } else {
            int total = new_chi_L * n_svd;
            extract_cols_kernel<Scalar><<<(total+threads-1)/threads, threads, 0, stream_>>>(
                d_svd_Vh_, full_k, d_mps_tensors_[site], new_chi_L, new_chi_L, n_svd);
        }

        // U*S -> d_svd_work_ (scale columns of U by S, on GPU)
        {
            int total = m * new_k;
            scale_cols_by_diag_kernel<Scalar, RealType><<<(total+threads-1)/threads, threads, 0, stream_>>>(
                d_svd_S_, d_svd_U_, m, d_svd_work_, m, m, new_k);
        }

        // Absorb U*S into A[site-1]
        if (site > 0) {
            int prev_cL = chi_L(site - 1);
            Scalar one = Traits::one(), zero_val = Traits::zero();
            CUBLAS_CHECK(Traits::gemm(cublas_h_,
                CUBLAS_OP_N, CUBLAS_OP_N,
                prev_cL * d_, new_k, m, &one,
                d_mps_tensors_[site - 1], prev_cL * d_,
                d_svd_work_, m, &zero_val,
                d_T1_, prev_cL * d_));
            allocate_mps_tensor(site - 1, prev_cL, new_chi_L);
            CUDA_CHECK(cudaMemcpy(d_mps_tensors_[site - 1], d_T1_,
                                prev_cL * d_ * new_k * sizeof(Scalar), cudaMemcpyDeviceToDevice));
        }
        bond_dims_[site] = new_chi_L;
    }
}

// ============================================================================
// Site optimization
// ============================================================================

template<typename Scalar>
double DMRGGPU<Scalar>::optimize_site(int site, char direction) {
    form_theta(site, d_theta_);
    double energy = lanczos_eigensolver(site, d_theta_);
    svd_and_update_mps(site, d_theta_, direction);
    return energy;
}

// ============================================================================
// Sweep methods
// ============================================================================

template<typename Scalar>
double DMRGGPU<Scalar>::sweep_left_to_right() {
    double energy = 0.0;

    for (int site = 0; site < L_ - 1; site++) {
        energy = optimize_site(site, 'R');
        update_left_env(site);
    }
    // Optimize last site without SVD
    {
        int site = L_ - 1;
        form_theta(site, d_theta_);
        energy = lanczos_eigensolver(site, d_theta_);
        int sz = chi_L(site) * d_ * chi_R(site);
        CUDA_CHECK(cudaMemcpy(d_mps_tensors_[site], d_theta_, sz * sizeof(Scalar),
                            cudaMemcpyDeviceToDevice));
    }

    return energy;
}

template<typename Scalar>
double DMRGGPU<Scalar>::sweep_right_to_left() {
    double energy = 0.0;

    for (int site = L_ - 1; site >= 1; site--) {
        energy = optimize_site(site, 'L');
        update_right_env(site);
    }
    // Optimize first site without SVD
    {
        int site = 0;
        form_theta(site, d_theta_);
        energy = lanczos_eigensolver(site, d_theta_);
        int sz = chi_L(site) * d_ * chi_R(site);
        CUDA_CHECK(cudaMemcpy(d_mps_tensors_[site], d_theta_, sz * sizeof(Scalar),
                            cudaMemcpyDeviceToDevice));
    }

    return energy;
}

// ============================================================================
// Main algorithm
// ============================================================================

template<typename Scalar>
double DMRGGPU<Scalar>::run(int n_sweeps) {
    build_initial_environments();

    // Timer starts AFTER env build -- measures sweep-to-convergence only
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
void DMRGGPU<Scalar>::get_mps(std::vector<std::vector<Scalar>>& h_mps) const {
    h_mps.resize(L_);
    for (int i = 0; i < L_; i++) {
        int size = chi_L(i) * d_ * chi_R(i);
        h_mps[i].resize(size);
        CUDA_CHECK(cudaMemcpy(h_mps[i].data(), d_mps_tensors_[i],
                            size * sizeof(Scalar), cudaMemcpyDeviceToHost));
    }
}

#endif // DMRG_GPU_IMPL_H
