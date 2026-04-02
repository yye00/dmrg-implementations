#ifndef DMRG2_GPU_IMPL_H
#define DMRG2_GPU_IMPL_H

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
        cusolverStatus_t status = call; \
        if (status != CUSOLVER_STATUS_SUCCESS) { \
            std::cerr << "cuSOLVER error in " << __FILE__ << ":" << __LINE__ \
                      << " - status " << status << std::endl; \
            throw std::runtime_error("cuSOLVER error"); \
        } \
    } while(0)

// Profiling counters (reset per sweep pair)

// ============================================================================
// Constructor
// ============================================================================

template<typename Scalar>
DMRG2GPU<Scalar>::DMRG2GPU(int L, int d, int chi_max, int D_mpo, double tol)
    : L_(L), d_(d), chi_max_(chi_max), D_mpo_(D_mpo), tol_(tol), energy_(0.0) {

    // Bond dimensions (same as single-site: min-cut formula capped at chi_max)
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

    int dd = d_ * d_;  // d^2 for two-site

    // Contraction intermediates: D*d^2*chi_max^2
    int t_max = D_mpo_ * dd * chi_max_ * chi_max_;
    CUDA_CHECK(cudaMalloc(&d_T1_, t_max * sizeof(Scalar)));
    CUDA_CHECK(cudaMalloc(&d_T2_, t_max * sizeof(Scalar)));

    // MPS tensors
    d_mps_tensors_.resize(L, nullptr);
    for (int i = 0; i < L; i++) {
        allocate_mps_tensor(i, chi_L(i), chi_R(i));
    }

    // MPO tensors
    d_mpo_tensors_.resize(L, nullptr);

    // W matrices for single-site env updates (allocated in set_mpo)
    d_W_left_.resize(L, nullptr);
    d_W_right_.resize(L, nullptr);

    // Fused two-site MPO (allocated in set_mpo)
    d_WW_.resize(L - 1, nullptr);

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

    // Lanczos workspace: theta is d^2 times larger than single-site
    theta_size_max_ = chi_max_ * dd * chi_max_;
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

    // Batched GEMM pointer arrays: D*d^2 batches for two-site
    int batch_max = D_mpo_ * dd;
    CUDA_CHECK(cudaMalloc(&d_batch_A_, batch_max * sizeof(Scalar*)));
    CUDA_CHECK(cudaMalloc(&d_batch_B_, batch_max * sizeof(Scalar*)));
    CUDA_CHECK(cudaMalloc(&d_batch_C_, batch_max * sizeof(Scalar*)));

    // SVD workspace: theta reshaped as (chi_max*d, d*chi_max)
    int svd_max_m = chi_max_ * d_;
    int svd_max_n = d_ * chi_max_;
    int svd_max_k = std::min(svd_max_m, svd_max_n);  // = chi_max_ * d_

    CUDA_CHECK(cudaMalloc(&d_svd_A_,    theta_size_max_ * sizeof(Scalar)));
    CUDA_CHECK(cudaMalloc(&d_svd_U_,    (size_t)svd_max_m * svd_max_k * sizeof(Scalar)));
    CUDA_CHECK(cudaMalloc(&d_svd_S_,    svd_max_k * sizeof(RealType)));
    CUDA_CHECK(cudaMalloc(&d_svd_Vh_,   (size_t)svd_max_k * svd_max_n * sizeof(Scalar)));
    CUDA_CHECK(cudaMalloc(&d_svd_info_, sizeof(int)));

    // cuSOLVER SVD workspace query (for max dimensions)
    CUSOLVER_CHECK(Traits::cusolver_gesvd_bufferSize(cusolver_h_, svd_max_m, svd_max_n, &svd_lwork_));
    CUDA_CHECK(cudaMalloc(&d_svd_work_, svd_lwork_ * sizeof(Scalar)));

    // rwork for complex SVD
    int rwork_size = Traits::svd_rwork_size(svd_max_m, svd_max_n);
    if (rwork_size > 0) {
        CUDA_CHECK(cudaMalloc(&d_svd_rwork_, rwork_size * sizeof(RealType)));
    } else {
        d_svd_rwork_ = nullptr;
    }

    // CPU workspace (for receiving GPU SVD results and truncation/scaling)
    h_svd_U_.resize((size_t)svd_max_m * svd_max_k);
    h_svd_S_.resize(svd_max_k);
    h_svd_Vh_.resize((size_t)svd_max_k * svd_max_n);
    h_svd_tmp_.resize(std::max((size_t)svd_max_m * svd_max_k, (size_t)svd_max_k * svd_max_n));
}

// ============================================================================
// Destructor
// ============================================================================

template<typename Scalar>
DMRG2GPU<Scalar>::~DMRG2GPU() {
    free_gpu_resources();
}

template<typename Scalar>
void DMRG2GPU<Scalar>::free_gpu_resources() {
    for (auto ptr : d_mps_tensors_) if (ptr) cudaFree(ptr);
    for (auto ptr : d_mpo_tensors_) if (ptr) cudaFree(ptr);
    for (auto ptr : d_W_left_) if (ptr) cudaFree(ptr);
    for (auto ptr : d_W_right_) if (ptr) cudaFree(ptr);
    for (auto ptr : d_WW_) if (ptr) cudaFree(ptr);
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
    if (d_svd_rwork_) cudaFree(d_svd_rwork_);
    if (d_svd_info_) cudaFree(d_svd_info_);
    if (d_svd_work_) cudaFree(d_svd_work_);

    cusolverDnDestroy(cusolver_h_);
    cublasDestroy(cublas_h_);
    cudaStreamDestroy(stream_);
}

// ============================================================================
// Memory management
// ============================================================================

template<typename Scalar>
void DMRG2GPU<Scalar>::allocate_mps_tensor(int site, int cL, int cR) {
    size_t max_sz = (size_t)chi_max_ * d_ * chi_max_ * sizeof(Scalar);
    if (!d_mps_tensors_[site]) {
        CUDA_CHECK(cudaMalloc(&d_mps_tensors_[site], max_sz));
    }
    (void)cL; (void)cR;
}

template<typename Scalar>
void DMRG2GPU<Scalar>::ensure_L_env_alloc(int idx, int chi) {
    if (chi > L_env_alloc_chi_[idx]) {
        if (d_L_envs_[idx]) CUDA_CHECK(cudaFree(d_L_envs_[idx]));
        int sz = chi * D_mpo_ * chi;
        CUDA_CHECK(cudaMalloc(&d_L_envs_[idx], sz * sizeof(Scalar)));
        L_env_alloc_chi_[idx] = chi;
    }
}

template<typename Scalar>
void DMRG2GPU<Scalar>::ensure_R_env_alloc(int idx, int chi) {
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
void DMRG2GPU<Scalar>::initialize_mps_random(double scale) {
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
void DMRG2GPU<Scalar>::initialize_mps_product() {
    // Reset all interior bonds to 1 for product state
    for (int i = 1; i < L_; i++) bond_dims_[i] = 1;

    for (int i = 0; i < L_; i++) {
        int cL = chi_L(i), cR = chi_R(i);
        allocate_mps_tensor(i, cL, cR);
        int size = cL * d_ * cR;
        std::vector<Scalar> h_A(size, Traits::zero());
        // |0> state: only s=0, diagonal bonds
        int chi_min = std::min(cL, cR);
        for (int a = 0; a < chi_min; a++) {
            h_A[a + 0*cL + a*cL*d_] = Traits::one();
        }
        CUDA_CHECK(cudaMemcpy(d_mps_tensors_[i], h_A.data(),
                            size * sizeof(Scalar), cudaMemcpyHostToDevice));
    }
}

template<typename Scalar>
void DMRG2GPU<Scalar>::initialize_mps_neel() {
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

// ============================================================================
// MPO setup and fused two-site MPO precomputation
// ============================================================================

template<typename Scalar>
void DMRG2GPU<Scalar>::set_mpo(const std::vector<Scalar*>& h_mpo_tensors) {
    int D = D_mpo_, d = d_;
    for (int i = 0; i < L_; i++) {
        int size = D * d * d * D;
        CUDA_CHECK(cudaMalloc(&d_mpo_tensors_[i], size * sizeof(Scalar)));
        CUDA_CHECK(cudaMemcpy(d_mpo_tensors_[i], h_mpo_tensors[i],
                            size * sizeof(Scalar), cudaMemcpyHostToDevice));

        // Precompute W_left and W_right matrices (for single-site env updates)
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

    // Precompute fused two-site MPO
    precompute_fused_mpo(h_mpo_tensors);
}

template<typename Scalar>
void DMRG2GPU<Scalar>::precompute_fused_mpo(const std::vector<Scalar*>& h_mpo_tensors) {
    int D = D_mpo_, d = d_;
    int dd = d * d;

    for (int bond = 0; bond < L_ - 1; bond++) {
        // WW[row, col] where row = w*dd + s1*d + s2, col = n*dd + s1p*d + s2p
        // WW[(w*dd+s1*d+s2), (n*dd+s1p*d+s2p)] = sum_m W_L[w,s1,s1p,m] * W_R[m,s2,s2p,n]
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
                                    // W_L[w, s1, s1p, m] at w + s1*D + s1p*D*d + m*D*d*d
                                    Scalar wl = WL[w + s1*D + s1p*D*d + m*D*d*d];
                                    // W_R[m, s2, s2p, n] at m + s2*D + s2p*D*d + n*D*d*d
                                    Scalar wr = WR[m + s2*D + s2p*D*d + n*D*d*d];
                                    if constexpr (Traits::is_complex) {
                                        val = cuCadd(val, cuCmul(wl, wr));
                                    } else {
                                        val += wl * wr;
                                    }
                                }
                                int row = w * dd + s1 * d + s2;
                                int col = n * dd + s1p * d + s2p;
                                h_WW[row + col * D * dd] = val;  // column-major
                            }

        CUDA_CHECK(cudaMalloc(&d_WW_[bond], ww_size * sizeof(Scalar)));
        CUDA_CHECK(cudaMemcpy(d_WW_[bond], h_WW.data(),
                            ww_size * sizeof(Scalar), cudaMemcpyHostToDevice));
    }
}

// ============================================================================
// Two-site theta formation
// ============================================================================

template<typename Scalar>
void DMRG2GPU<Scalar>::form_theta_two_site(int site) {
    int cL = chi_L(site);
    int chi_mid = bond_dims_[site + 1];  // shared bond between site and site+1
    int cR = chi_R(site + 1);
    Scalar one = Traits::one(), zero_val = Traits::zero();

    // MPS[site]: (cL, d, chi_mid) as (cL*d, chi_mid) column-major
    // MPS[site+1]: (chi_mid, d, cR) as (chi_mid, d*cR) column-major
    // theta: (cL*d, d*cR) = (cL, d, d, cR) column-major
    CUBLAS_CHECK(Traits::gemm(cublas_h_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        cL * d_, d_ * cR, chi_mid,
        &one,
        d_mps_tensors_[site], cL * d_,
        d_mps_tensors_[site + 1], chi_mid,
        &zero_val,
        d_theta_, cL * d_));
}

// ============================================================================
// Two-site H_eff application (3-step with fused WW)
// ============================================================================

template<typename Scalar>
void DMRG2GPU<Scalar>::apply_heff_two_site(int site, const Scalar* d_theta_in, Scalar* d_result) {
    int cL = chi_L(site);
    int cR = chi_R(site + 1);
    int D = D_mpo_, d = d_;
    int dd = d * d;
    Scalar one = Traits::one(), zero_val = Traits::zero();

    Scalar* L_env = d_L_envs_[site];
    Scalar* R_env = d_R_envs_[site + 2];
    Scalar* WW = d_WW_[site];  // fused two-site MPO for bond (site, site+1)
    Scalar* T1 = d_T1_;
    Scalar* T2 = d_T2_;

    // ---------------------------------------------------------------
    // Step 1: Batched GEMM -- contract L_env with theta
    //   T1[ws, a', b] = L[w]^T[a', a] * theta[s1,s2][a, b]
    //   ws = w*dd + s1*d + s2, D*d^2 batches
    //   Each batch: (cL, cR) = op_t(L[w]) @ theta[s1,s2]
    //     L[w]: (cL, cL) with lda = cL*D
    //     theta[s1,s2]: (cL, cR) with ldb = cL*d^2
    //     T1[ws]: (cL, cR) with ldc = cL
    // ---------------------------------------------------------------
    {
        int batch_count = D * dd;
        std::vector<Scalar*> h_A(batch_count), h_B(batch_count), h_C(batch_count);
        for (int w = 0; w < D; w++)
            for (int s1 = 0; s1 < d; s1++)
                for (int s2 = 0; s2 < d; s2++) {
                    int ws = w * dd + s1 * d + s2;
                    h_A[ws] = L_env + w * cL;
                    h_B[ws] = const_cast<Scalar*>(d_theta_in) + s1 * cL + s2 * cL * d;
                    h_C[ws] = T1 + ws * cL * cR;
                }
        CUDA_CHECK(cudaMemcpyAsync(d_batch_A_, h_A.data(), batch_count*sizeof(Scalar*), cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(d_batch_B_, h_B.data(), batch_count*sizeof(Scalar*), cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(d_batch_C_, h_C.data(), batch_count*sizeof(Scalar*), cudaMemcpyHostToDevice, stream_));
        CUBLAS_CHECK(Traits::gemm_batched(cublas_h_,
            Traits::op_t, CUBLAS_OP_N,
            cL, cR, cL,
            &one,
            (const Scalar**)d_batch_A_, cL * D,      // lda: L_env column stride
            (const Scalar**)d_batch_B_, cL * dd,      // ldb: theta column stride (cL*d^2)
            &zero_val,
            d_batch_C_, cL,                            // ldc
            batch_count));
    }

    // ---------------------------------------------------------------
    // Step 2: Dense GEMM -- absorb fused WW
    //   T2 = T1 @ WW
    //   T1: (cL*cR, D*d^2)  WW: (D*d^2, d^2*D)  T2: (cL*cR, d^2*D)
    // ---------------------------------------------------------------
    CUBLAS_CHECK(Traits::gemm(cublas_h_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        cL * cR, dd * D, D * dd,
        &one,
        T1, cL * cR,
        WW, D * dd,
        &zero_val,
        T2, cL * cR));

    // ---------------------------------------------------------------
    // Step 3: Batched GEMMs -- contract R_env
    //   result[a', s1', s2', b'] = sum_{n} T2_col[a', b] * R[n][b, b']
    //   Accumulate over n (beta=0 for n=0, beta=1 for n>0)
    // ---------------------------------------------------------------
    // Batch dd GEMMs per n (safe: different s1p/s2p write to different C locations).
    // Cannot batch across n since same (s1p,s2p)/different n accumulate into same C.
    {
        std::vector<Scalar*> h_A3(D * dd), h_B3(D * dd), h_C3(D * dd);
        for (int n = 0; n < D; n++) {
            Scalar beta = (n == 0) ? Traits::zero() : Traits::one();
            int idx = 0;
            for (int s1p = 0; s1p < d; s1p++) {
                for (int s2p = 0; s2p < d; s2p++) {
                    int ws_out = n * dd + s1p * d + s2p;
                    h_A3[idx] = T2 + ws_out * cL * cR;
                    h_B3[idx] = R_env + n * cR;
                    h_C3[idx] = d_result + s1p * cL + s2p * cL * d;
                    idx++;
                }
            }
            CUDA_CHECK(cudaMemcpyAsync(d_batch_A_, h_A3.data(), dd*sizeof(Scalar*), cudaMemcpyHostToDevice, stream_));
            CUDA_CHECK(cudaMemcpyAsync(d_batch_B_, h_B3.data(), dd*sizeof(Scalar*), cudaMemcpyHostToDevice, stream_));
            CUDA_CHECK(cudaMemcpyAsync(d_batch_C_, h_C3.data(), dd*sizeof(Scalar*), cudaMemcpyHostToDevice, stream_));
            CUBLAS_CHECK(Traits::gemm_batched(cublas_h_,
                CUBLAS_OP_N, CUBLAS_OP_N,
                cL, cR, cR,
                &one,
                (const Scalar**)d_batch_A_, cL,
                (const Scalar**)d_batch_B_, cR * D,
                &beta,
                d_batch_C_, cL * dd,
                dd));
        }
    }
}

// ============================================================================
// Left environment update (identical to single-site)
// ============================================================================

template<typename Scalar>
void DMRG2GPU<Scalar>::update_left_env(int site) {
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
        std::vector<Scalar*> h_A(D * d), h_B(D * d), h_C(D * d);
        for (int w = 0; w < D; w++)
            for (int s = 0; s < d; s++) {
                int ws = w * d + s;
                h_A[ws] = L_env + w * chi_in;
                h_B[ws] = A + s * chi_in;
                h_C[ws] = V + ws * chi_in * chi_out;
            }
        CUDA_CHECK(cudaMemcpyAsync(d_batch_A_, h_A.data(), D*d*sizeof(Scalar*), cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(d_batch_B_, h_B.data(), D*d*sizeof(Scalar*), cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(d_batch_C_, h_C.data(), D*d*sizeof(Scalar*), cudaMemcpyHostToDevice, stream_));
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
    // Batch D GEMMs per sp (safe: different wp write to different C locations).
    {
        std::vector<Scalar*> h_A3(D * d), h_B3(D * d), h_C3(D * d);
        for (int sp = 0; sp < d; sp++) {
            Scalar beta = (sp == 0) ? Traits::zero() : Traits::one();
            for (int wp = 0; wp < D; wp++) {
                h_A3[wp] = U + (wp * d + sp) * chi_in * chi_out;
                h_B3[wp] = A + sp * chi_in;
                h_C3[wp] = L_new + wp * chi_out;
            }
            CUDA_CHECK(cudaMemcpyAsync(d_batch_A_, h_A3.data(), D*sizeof(Scalar*), cudaMemcpyHostToDevice, stream_));
            CUDA_CHECK(cudaMemcpyAsync(d_batch_B_, h_B3.data(), D*sizeof(Scalar*), cudaMemcpyHostToDevice, stream_));
            CUDA_CHECK(cudaMemcpyAsync(d_batch_C_, h_C3.data(), D*sizeof(Scalar*), cudaMemcpyHostToDevice, stream_));
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
// Right environment update (identical to single-site)
// ============================================================================

template<typename Scalar>
void DMRG2GPU<Scalar>::update_right_env(int site) {
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
        std::vector<Scalar*> h_A(D * d), h_B(D * d), h_C(D * d);
        for (int wp = 0; wp < D; wp++)
            for (int s = 0; s < d; s++) {
                int ws = wp * d + s;
                h_A[ws] = A + s * chi_out;
                h_B[ws] = R_env + wp * chi_in;
                h_C[ws] = V + ws * chi_out * chi_in;
            }
        CUDA_CHECK(cudaMemcpyAsync(d_batch_A_, h_A.data(), D*d*sizeof(Scalar*), cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(d_batch_B_, h_B.data(), D*d*sizeof(Scalar*), cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(d_batch_C_, h_C.data(), D*d*sizeof(Scalar*), cudaMemcpyHostToDevice, stream_));
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
    // Batch D GEMMs per sp (safe: different w write to different C locations).
    {
        std::vector<Scalar*> h_A3(D * d), h_B3(D * d), h_C3(D * d);
        for (int sp = 0; sp < d; sp++) {
            Scalar beta = (sp == 0) ? Traits::zero() : Traits::one();
            for (int w = 0; w < D; w++) {
                h_A3[w] = U + (w * d + sp) * chi_out * chi_in;
                h_B3[w] = A + sp * chi_out;
                h_C3[w] = R_new + w * chi_out;
            }
            CUDA_CHECK(cudaMemcpyAsync(d_batch_A_, h_A3.data(), D*sizeof(Scalar*), cudaMemcpyHostToDevice, stream_));
            CUDA_CHECK(cudaMemcpyAsync(d_batch_B_, h_B3.data(), D*sizeof(Scalar*), cudaMemcpyHostToDevice, stream_));
            CUDA_CHECK(cudaMemcpyAsync(d_batch_C_, h_C3.data(), D*sizeof(Scalar*), cudaMemcpyHostToDevice, stream_));
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
void DMRG2GPU<Scalar>::build_initial_environments() {
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
// Lanczos eigensolver (operates on theta of given size)
// ============================================================================

template<typename Scalar>
double DMRG2GPU<Scalar>::lanczos_eigensolver(int site, Scalar* d_theta, int theta_size) {
    int n = theta_size;
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
        apply_heff_two_site(site, d_vi, d_heff_result_);

        // Switch to device pointer mode for scalar operations
        CUBLAS_CHECK(cublasSetPointerMode(cublas_h_, CUBLAS_POINTER_MODE_DEVICE));

        // alpha_i = <v_i|w> -> device
        CUBLAS_CHECK(Traits::dot(cublas_h_, n, d_vi, 1, d_heff_result_, 1, d_dot_result_));

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

        // Full reorthogonalization (device pointer mode)
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

        lanczos_process_beta_kernel<Scalar><<<1, 1, 0, stream_>>>(
                           d_nrm2_result_, d_inv_nrm_, d_beta_dev_, d_neg_beta_scalars_, iter);

        // v_{i+1} = w / beta_i (device pointer)
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

    std::vector<Scalar> h_ritz_scalar(niter);
    for (int i = 0; i < niter; i++) {
        h_ritz_scalar[i] = Traits::make_scalar(h_Z[i]);
    }
    CUDA_CHECK(cudaMemcpy(d_ritz_coeffs_, h_ritz_scalar.data(), niter * sizeof(Scalar), cudaMemcpyHostToDevice));

    // Use device pointer mode for finalization to avoid implicit GPU syncs
    CUBLAS_CHECK(cublasSetPointerMode(cublas_h_, CUBLAS_POINTER_MODE_DEVICE));
    CUBLAS_CHECK(Traits::gemv(
        cublas_h_, CUBLAS_OP_N,
        n, niter, d_const_one_,
        d_lanczos_v, n,
        d_ritz_coeffs_, 1,
        d_const_zero_, d_theta, 1
    ));

    CUBLAS_CHECK(Traits::nrm2(cublas_h_, n, d_theta, 1, d_nrm2_result_));
    invert_nrm_kernel<RealType><<<1, 1, 0, 0>>>(
                       d_nrm2_result_, d_inv_nrm_);
    CUBLAS_CHECK(Traits::scal_real(cublas_h_, n, d_inv_nrm_, d_theta, 1));
    CUBLAS_CHECK(cublasSetPointerMode(cublas_h_, CUBLAS_POINTER_MODE_HOST));

    return energy;
}

// ============================================================================
// SVD split for two-site DMRG
// ============================================================================

template<typename Scalar>
void DMRG2GPU<Scalar>::svd_split(int site, Scalar* d_theta, char direction) {
    int cL = chi_L(site);
    int cR = chi_R(site + 1);

    // theta is (cL, d, d, cR) reshaped as (cL*d, d*cR) for SVD
    int m = cL * d_;
    int n_svd = d_ * cR;
    int full_k = std::min(m, n_svd);
    int k = std::min(full_k, chi_max_);

    // cuSOLVER in CUDA 13.0 requires m >= n for gesvd.
    // When m < n: SVD(A^T) = U_t S Vh_t, then A's U = Vh_t^T, A's Vh = U_t^T.
    bool transposed = (m < n_svd);
    int svd_m = transposed ? n_svd : m;
    int svd_n = transposed ? m : n_svd;
    int thr = 256;

    if (transposed) {
        int total_t = m * n_svd;
        transpose_kernel<Scalar><<<(total_t+thr-1)/thr, thr, 0, stream_>>>(
            d_theta, m, d_svd_A_, n_svd, m, n_svd);
    } else {
        CUDA_CHECK(cudaMemcpy(d_svd_A_, d_theta, m * n_svd * sizeof(Scalar), cudaMemcpyDeviceToDevice));
    }

    // Re-query workspace for actual SVD dimensions (may differ from max)
    int svd_lwork_actual = 0;
    CUSOLVER_CHECK(Traits::cusolver_gesvd_bufferSize(cusolver_h_, svd_m, svd_n, &svd_lwork_actual));
    int lwork_use = std::max(svd_lwork_, svd_lwork_actual);
    // If re-queried workspace exceeds pre-allocated, reallocate
    if (svd_lwork_actual > svd_lwork_) {
        cudaFree(d_svd_work_);
        svd_lwork_ = svd_lwork_actual;
        CUDA_CHECK(cudaMalloc(&d_svd_work_, svd_lwork_ * sizeof(Scalar)));
    }

    fprintf(stderr, "SVD: svd_m=%d svd_n=%d full_k=%d transposed=%d lwork=%d\n",
            svd_m, svd_n, full_k, transposed, lwork_use);

    cusolverStatus_t svd_status = Traits::cusolver_gesvd(cusolver_h_,
        'S', 'S',
        svd_m, svd_n,
        d_svd_A_, svd_m,
        d_svd_S_,
        d_svd_U_, svd_m,
        d_svd_Vh_, full_k,
        d_svd_work_, lwork_use,
        d_svd_rwork_,
        d_svd_info_);

    if (svd_status != CUSOLVER_STATUS_SUCCESS) {
        CUDA_CHECK(cudaDeviceSynchronize());
        int h_info = -999;
        cudaMemcpy(&h_info, d_svd_info_, sizeof(int), cudaMemcpyDeviceToHost);
        fprintf(stderr, "SVD FAILED: cusolver status=%d, devInfo=%d\n", svd_status, h_info);
        fprintf(stderr, "  original m=%d n=%d, svd_m=%d svd_n=%d full_k=%d\n", m, n_svd, svd_m, svd_n, full_k);
        exit(1);
    }

    if (transposed) {
        // d_svd_U_ = U_t (n_svd x full_k), d_svd_Vh_ = Vh_t (full_k x m)
        // actual_U (m x full_k) = Vh_t^T, actual_Vh (full_k x n_svd) = U_t^T
        int total_u = m * full_k;
        int total_vh = full_k * n_svd;

        // Transpose Vh_t (full_k x m) -> actual_U (m x full_k) into d_T1_
        transpose_kernel<Scalar><<<(total_u+thr-1)/thr, thr, 0, stream_>>>(
            d_svd_Vh_, full_k, d_T1_, m, full_k, m);

        // Transpose U_t (n_svd x full_k) -> actual_Vh (full_k x n_svd) into d_T2_
        transpose_kernel<Scalar><<<(total_vh+thr-1)/thr, thr, 0, stream_>>>(
            d_svd_U_, n_svd, d_T2_, full_k, n_svd, full_k);

        // Copy back to canonical locations
        CUDA_CHECK(cudaMemcpyAsync(d_svd_U_, d_T1_, total_u * sizeof(Scalar),
                                   cudaMemcpyDeviceToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(d_svd_Vh_, d_T2_, total_vh * sizeof(Scalar),
                                   cudaMemcpyDeviceToDevice, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    // Truncation: find new_k on GPU, copy just 1 int back
    svd_truncate_kernel<RealType><<<1, 1, 0, stream_>>>(
                       d_svd_S_, k, 1e-14, d_svd_info_);
    int new_k;
    CUDA_CHECK(cudaMemcpy(&new_k, d_svd_info_, sizeof(int), cudaMemcpyDeviceToHost));

    int threads = 256;

    if (direction == 'R') {
        // U -> MPS[site] (left-canonical), S*Vh -> MPS[site+1]

        // MPS[site] = U[:, :new_k] (column slice, on GPU)
        allocate_mps_tensor(site, cL, new_k);
        if (new_k == full_k) {
            CUDA_CHECK(cudaMemcpy(d_mps_tensors_[site], d_svd_U_,
                                m * new_k * sizeof(Scalar), cudaMemcpyDeviceToDevice));
        } else {
            int total = m * new_k;
            extract_cols_kernel<Scalar><<<(total+threads-1)/threads, threads, 0, stream_>>>(
                               d_svd_U_, m, d_mps_tensors_[site], m, m, new_k);
        }

        // S*Vh -> d_svd_work_ (scale rows of Vh by S, on GPU)
        allocate_mps_tensor(site + 1, new_k, cR);
        {
            int total = new_k * n_svd;
            scale_rows_by_diag_kernel<Scalar, RealType><<<(total+threads-1)/threads, threads, 0, stream_>>>(
                               d_svd_S_, d_svd_Vh_, full_k, d_mps_tensors_[site + 1], new_k, new_k, n_svd);
        }

    } else {  // direction == 'L'
        // U*S -> MPS[site], Vh -> MPS[site+1] (right-canonical)

        // U*S -> MPS[site] (scale columns of U by S, on GPU)
        allocate_mps_tensor(site, cL, new_k);
        {
            int total = m * new_k;
            scale_cols_by_diag_kernel<Scalar, RealType><<<(total+threads-1)/threads, threads, 0, stream_>>>(
                               d_svd_S_, d_svd_U_, m, d_mps_tensors_[site], m, m, new_k);
        }

        // MPS[site+1] = Vh[:new_k, :] (row slice, on GPU)
        allocate_mps_tensor(site + 1, new_k, cR);
        if (new_k == full_k) {
            CUDA_CHECK(cudaMemcpy(d_mps_tensors_[site + 1], d_svd_Vh_,
                                full_k * n_svd * sizeof(Scalar), cudaMemcpyDeviceToDevice));
        } else {
            int total = new_k * n_svd;
            extract_cols_kernel<Scalar><<<(total+threads-1)/threads, threads, 0, stream_>>>(
                               d_svd_Vh_, full_k, d_mps_tensors_[site + 1], new_k, new_k, n_svd);
        }
    }

    bond_dims_[site + 1] = new_k;
}

// ============================================================================
// Bond optimization (two-site)
// ============================================================================

template<typename Scalar>
double DMRG2GPU<Scalar>::optimize_bond(int site, char direction) {
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
double DMRG2GPU<Scalar>::sweep_left_to_right() {
    double energy = 0.0;

    for (int site = 0; site < L_ - 1; site++) {
        energy = optimize_bond(site, 'R');
        update_left_env(site);
    }

    return energy;
}

template<typename Scalar>
double DMRG2GPU<Scalar>::sweep_right_to_left() {
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
double DMRG2GPU<Scalar>::run(int n_sweeps) {
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
void DMRG2GPU<Scalar>::get_mps(std::vector<std::vector<Scalar>>& h_mps) const {
    h_mps.resize(L_);
    for (int i = 0; i < L_; i++) {
        int size = chi_L(i) * d_ * chi_R(i);
        h_mps[i].resize(size);
        CUDA_CHECK(cudaMemcpy(h_mps[i].data(), d_mps_tensors_[i],
                            size * sizeof(Scalar), cudaMemcpyDeviceToHost));
    }
}

#endif // DMRG2_GPU_IMPL_H
