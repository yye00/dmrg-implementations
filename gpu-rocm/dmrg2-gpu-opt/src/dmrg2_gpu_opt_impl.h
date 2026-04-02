#ifndef DMRG2_GPU_OPT_IMPL_H
#define DMRG2_GPU_OPT_IMPL_H

#include <rocsolver/rocsolver.h>
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

// Profiling counters (reset per sweep pair)
static double prof_davidson_ms = 0, prof_ns_ms = 0, prof_env_ms = 0;
static int prof_davidson_iters = 0, prof_ns_iters = 0, prof_site_count = 0;
static int prof_heff_calls = 0;

// ============================================================================
// Constructor
// ============================================================================

template<typename Scalar>
DMRG2GPUOpt<Scalar>::DMRG2GPUOpt(int L, int d, int chi_max, int D_mpo, double tol)
    : L_(L), d_(d), chi_max_(pad_mfma16(chi_max)), chi_max_user_(chi_max),
      D_mpo_(D_mpo), tol_(tol), energy_(0.0) {

    if (chi_max_ != chi_max_user_) {
        printf("[OPT] MFMA-16 padding: chi_max %d -> %d\n", chi_max_user_, chi_max_);
    }

    // Bond dimensions (same as single-site: min-cut formula capped at chi_max)
    bond_dims_.resize(L + 1);
    bond_dims_[0] = 1;
    bond_dims_[L] = 1;
    for (int i = 1; i < L; i++) {
        double exact_dim = pow((double)d_, std::min(i, L - i));
        bond_dims_[i] = (exact_dim > chi_max_user_) ? chi_max_user_ : (int)exact_dim;
    }

    // GPU handles
    HIP_CHECK(hipStreamCreate(&stream_));
    ROCBLAS_CHECK(rocblas_create_handle(&rocblas_h_));
    ROCBLAS_CHECK(rocblas_set_stream(rocblas_h_, stream_));

    int dd = d_ * d_;  // d^2 for two-site

    // Contraction intermediates: D*d^2*chi_max^2
    int t_max = D_mpo_ * dd * chi_max_ * chi_max_;
    HIP_CHECK(hipMalloc(&d_T1_, t_max * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_T2_, t_max * sizeof(Scalar)));

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
        HIP_CHECK(hipMalloc(&d_L_envs_[i], sz * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&d_R_envs_[i], sz * sizeof(Scalar)));
        HIP_CHECK(hipMemset(d_L_envs_[i], 0, sz * sizeof(Scalar)));
        HIP_CHECK(hipMemset(d_R_envs_[i], 0, sz * sizeof(Scalar)));
        L_env_alloc_chi_[i] = chi_alloc;
        R_env_alloc_chi_[i] = chi_alloc;
    }

    // Lanczos workspace (fallback): theta is d^2 times larger than single-site
    theta_size_max_ = chi_max_ * dd * chi_max_;
    max_lanczos_iter_ = std::min(100, theta_size_max_);
    HIP_CHECK(hipMalloc(&d_theta_, theta_size_max_ * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_heff_result_, theta_size_max_ * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_lanczos_v_, (size_t)max_lanczos_iter_ * theta_size_max_ * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_ritz_coeffs_, max_lanczos_iter_ * sizeof(Scalar)));

    // Batched GEMM pointer arrays: D*d^2 batches for two-site
    int batch_max = D_mpo_ * dd;
    HIP_CHECK(hipMalloc(&d_batch_A_, batch_max * sizeof(Scalar*)));
    HIP_CHECK(hipMalloc(&d_batch_B_, batch_max * sizeof(Scalar*)));
    HIP_CHECK(hipMalloc(&d_batch_C_, batch_max * sizeof(Scalar*)));

    // SVD workspace (reused as NS scratch)
    int svd_max_m = chi_max_ * d_;
    int svd_max_n = d_ * chi_max_;
    int svd_max_k = std::min(svd_max_m, svd_max_n);  // = chi_max_ * d_

    HIP_CHECK(hipMalloc(&d_svd_A_,    theta_size_max_ * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_svd_U_,    (size_t)svd_max_m * svd_max_k * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_svd_S_,    svd_max_k * sizeof(RealType)));
    HIP_CHECK(hipMalloc(&d_svd_Vh_,   (size_t)svd_max_k * svd_max_n * sizeof(Scalar)));

    // CPU SVD workspace (fallback)
    h_svd_A_.resize(theta_size_max_);
    h_svd_U_.resize((size_t)svd_max_m * svd_max_k);
    h_svd_S_.resize(svd_max_k);
    h_svd_Vh_.resize((size_t)svd_max_k * svd_max_n);
    h_svd_tmp_.resize(std::max((size_t)svd_max_m * svd_max_k, (size_t)svd_max_k * svd_max_n));
    h_svd_rwork_.resize(Traits::svd_rwork_size(svd_max_m, svd_max_n));

    // Query optimal LAPACK SVD workspace
    {
        int m = svd_max_m, n = svd_max_n;
        int lwork_query = -1;
        Scalar work_opt;
        int info;
        const char jobu = 'S', jobvt = 'S';
        Traits::lapack_gesvd(&jobu, &jobvt, &m, &n, nullptr, &m, nullptr,
                nullptr, &m, nullptr, &svd_max_k, &work_opt, &lwork_query,
                h_svd_rwork_.empty() ? nullptr : h_svd_rwork_.data(), &info);
        int opt_size;
        if constexpr (Traits::is_complex) {
            opt_size = (int)Traits::real_part(work_opt) + 1;
        } else {
            opt_size = (int)work_opt + 1;
        }
        h_svd_work_.resize(opt_size);
    }

    // Newton-Schulz workspace
    int ns_max = chi_max_ * d_;
    HIP_CHECK(hipMalloc(&d_ns_U_,     (size_t)ns_max * ns_max * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_ns_U_new_, (size_t)ns_max * ns_max * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_ns_gram_,  (size_t)ns_max * ns_max * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_ns_P_,     (size_t)ns_max * ns_max * sizeof(Scalar)));
    h_ns_PtP_.resize(ns_max * ns_max);
    h_ns_eigvals_.resize(ns_max);
    h_ns_syev_rwork_.resize(Traits::syev_rwork_size(ns_max));
    {
        int lwork_q = -1; Scalar work_opt; int info_q;
        const char jobz_q = 'V', uplo_q = 'U';
        Traits::lapack_syev(&jobz_q, &uplo_q, &ns_max, h_ns_PtP_.data(), &ns_max,
                            h_ns_eigvals_.data(), &work_opt, &lwork_q,
                            h_ns_syev_rwork_.empty() ? nullptr : h_ns_syev_rwork_.data(), &info_q);
        int opt_lwork;
        if constexpr (Traits::is_complex) opt_lwork = (int)Traits::real_part(work_opt) + 1;
        else opt_lwork = (int)work_opt + 1;
        h_ns_syev_work_.resize(opt_lwork);
    }

    // Block-Davidson workspace
    davidson_b_ = 4;
    davidson_max_sub_ = std::min(davidson_b_ * 8, theta_size_max_);
    HIP_CHECK(hipMalloc(&d_dav_V_,     (size_t)theta_size_max_ * davidson_max_sub_ * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_dav_AV_,    (size_t)theta_size_max_ * davidson_max_sub_ * sizeof(Scalar)));
    {
        // d_dav_work_ must hold both (dim, b) orthogonalization scratch and (k, k) projected H
        size_t dav_work_sz = std::max((size_t)theta_size_max_ * davidson_b_,
                                       (size_t)davidson_max_sub_ * davidson_max_sub_);
        HIP_CHECK(hipMalloc(&d_dav_work_,  dav_work_sz * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&d_dav_work2_, dav_work_sz * sizeof(Scalar)));
    }
    h_dav_H_proj_.resize(davidson_max_sub_ * davidson_max_sub_);
    h_dav_eigvals_.resize(davidson_max_sub_);
    h_dav_eigvecs_.resize(davidson_max_sub_ * davidson_max_sub_);
}

// ============================================================================
// Destructor
// ============================================================================

template<typename Scalar>
DMRG2GPUOpt<Scalar>::~DMRG2GPUOpt() {
    free_gpu_resources();
}

template<typename Scalar>
void DMRG2GPUOpt<Scalar>::free_gpu_resources() {
    for (auto ptr : d_mps_tensors_) if (ptr) hipFree(ptr);
    for (auto ptr : d_mpo_tensors_) if (ptr) hipFree(ptr);
    for (auto ptr : d_W_left_) if (ptr) hipFree(ptr);
    for (auto ptr : d_W_right_) if (ptr) hipFree(ptr);
    for (auto ptr : d_WW_) if (ptr) hipFree(ptr);
    for (auto ptr : d_L_envs_) if (ptr) hipFree(ptr);
    for (auto ptr : d_R_envs_) if (ptr) hipFree(ptr);

    if (d_theta_) hipFree(d_theta_);
    if (d_heff_result_) hipFree(d_heff_result_);
    if (d_lanczos_v_) hipFree(d_lanczos_v_);
    if (d_ritz_coeffs_) hipFree(d_ritz_coeffs_);
    if (d_T1_) hipFree(d_T1_);
    if (d_T2_) hipFree(d_T2_);
    if (d_batch_A_) hipFree(d_batch_A_);
    if (d_batch_B_) hipFree(d_batch_B_);
    if (d_batch_C_) hipFree(d_batch_C_);
    if (d_svd_A_) hipFree(d_svd_A_);
    if (d_svd_U_) hipFree(d_svd_U_);
    if (d_svd_S_) hipFree(d_svd_S_);
    if (d_svd_Vh_) hipFree(d_svd_Vh_);

    // Newton-Schulz workspace
    if (d_ns_U_) hipFree(d_ns_U_);
    if (d_ns_U_new_) hipFree(d_ns_U_new_);
    if (d_ns_gram_) hipFree(d_ns_gram_);
    if (d_ns_P_) hipFree(d_ns_P_);

    // Block-Davidson workspace
    if (d_dav_V_) hipFree(d_dav_V_);
    if (d_dav_AV_) hipFree(d_dav_AV_);
    if (d_dav_work_) hipFree(d_dav_work_);
    if (d_dav_work2_) hipFree(d_dav_work2_);

    rocblas_destroy_handle(rocblas_h_);
    hipStreamDestroy(stream_);
}

// ============================================================================
// Memory management
// ============================================================================

template<typename Scalar>
void DMRG2GPUOpt<Scalar>::allocate_mps_tensor(int site, int cL, int cR) {
    if (d_mps_tensors_[site]) HIP_CHECK(hipFree(d_mps_tensors_[site]));
    HIP_CHECK(hipMalloc(&d_mps_tensors_[site], cL * d_ * cR * sizeof(Scalar)));
}

template<typename Scalar>
void DMRG2GPUOpt<Scalar>::ensure_L_env_alloc(int idx, int chi) {
    if (chi > L_env_alloc_chi_[idx]) {
        if (d_L_envs_[idx]) HIP_CHECK(hipFree(d_L_envs_[idx]));
        int sz = chi * D_mpo_ * chi;
        HIP_CHECK(hipMalloc(&d_L_envs_[idx], sz * sizeof(Scalar)));
        L_env_alloc_chi_[idx] = chi;
    }
}

template<typename Scalar>
void DMRG2GPUOpt<Scalar>::ensure_R_env_alloc(int idx, int chi) {
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
void DMRG2GPUOpt<Scalar>::initialize_mps_random(double scale) {
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
void DMRG2GPUOpt<Scalar>::initialize_mps_product() {
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
        HIP_CHECK(hipMemcpy(d_mps_tensors_[i], h_A.data(),
                            size * sizeof(Scalar), hipMemcpyHostToDevice));
    }
}

template<typename Scalar>
void DMRG2GPUOpt<Scalar>::initialize_mps_neel() {
    for (int i = 0; i < L_; i++) {
        int cL = chi_L(i), cR = chi_R(i);
        int size = cL * d_ * cR;
        std::vector<Scalar> h_A(size, Traits::zero());
        int spin = (i % 2 == 0) ? 0 : 1;
        int chi_min = std::min(cL, cR);
        for (int a = 0; a < chi_min; a++) {
            h_A[a + spin*cL + a*cL*d_] = Traits::one();
        }
        HIP_CHECK(hipMemcpy(d_mps_tensors_[i], h_A.data(),
                            size * sizeof(Scalar), hipMemcpyHostToDevice));
    }
}

// ============================================================================
// MPO setup and fused two-site MPO precomputation
// ============================================================================

template<typename Scalar>
void DMRG2GPUOpt<Scalar>::set_mpo(const std::vector<Scalar*>& h_mpo_tensors) {
    int D = D_mpo_, d = d_;
    for (int i = 0; i < L_; i++) {
        int size = D * d * d * D;
        HIP_CHECK(hipMalloc(&d_mpo_tensors_[i], size * sizeof(Scalar)));
        HIP_CHECK(hipMemcpy(d_mpo_tensors_[i], h_mpo_tensors[i],
                            size * sizeof(Scalar), hipMemcpyHostToDevice));

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
        HIP_CHECK(hipMalloc(&d_W_left_[i], wm_size * sizeof(Scalar)));
        HIP_CHECK(hipMemcpy(d_W_left_[i], h_WL.data(),
                            wm_size * sizeof(Scalar), hipMemcpyHostToDevice));
        HIP_CHECK(hipMalloc(&d_W_right_[i], wm_size * sizeof(Scalar)));
        HIP_CHECK(hipMemcpy(d_W_right_[i], h_WR.data(),
                            wm_size * sizeof(Scalar), hipMemcpyHostToDevice));
    }

    // Precompute fused two-site MPO
    precompute_fused_mpo(h_mpo_tensors);
}

template<typename Scalar>
void DMRG2GPUOpt<Scalar>::precompute_fused_mpo(const std::vector<Scalar*>& h_mpo_tensors) {
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
// Two-site theta formation
// ============================================================================

template<typename Scalar>
void DMRG2GPUOpt<Scalar>::form_theta_two_site(int site) {
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
// Two-site H_eff application (3-step with fused WW)
// ============================================================================

template<typename Scalar>
void DMRG2GPUOpt<Scalar>::apply_heff_two_site(int site, const Scalar* d_theta_in, Scalar* d_result) {
    int cL = chi_L(site);
    int cR = chi_R(site + 1);
    int D = D_mpo_, d = d_;
    int dd = d * d;
    Scalar one = Traits::one(), zero_val = Traits::zero();

    Scalar* L_env = d_L_envs_[site];
    Scalar* R_env = d_R_envs_[site + 2];
    Scalar* WW = d_WW_[site];
    Scalar* T1 = d_T1_;
    Scalar* T2 = d_T2_;

    // Step 1: Batched GEMM — contract L_env with theta
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
        HIP_CHECK(hipMemcpyAsync(d_batch_A_, h_A.data(), batch_count*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
        HIP_CHECK(hipMemcpyAsync(d_batch_B_, h_B.data(), batch_count*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
        HIP_CHECK(hipMemcpyAsync(d_batch_C_, h_C.data(), batch_count*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
        ROCBLAS_CHECK(Traits::gemm_batched(rocblas_h_,
            Traits::op_t, rocblas_operation_none,
            cL, cR, cL,
            &one,
            (const Scalar**)d_batch_A_, cL * D,
            (const Scalar**)d_batch_B_, cL * dd,
            &zero_val,
            d_batch_C_, cL,
            batch_count));
    }

    // Step 2: Dense GEMM — absorb fused WW
    ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
        rocblas_operation_none, rocblas_operation_none,
        cL * cR, dd * D, D * dd,
        &one,
        T1, cL * cR,
        WW, D * dd,
        &zero_val,
        T2, cL * cR));

    // Step 3: Loop of GEMMs — contract R_env
    // Strided batched over s1p when d<=2 and chi>=16 (avoids cache contention at batch_count>2)
    if (cL >= 16 && cR >= 16 && d <= 2) {
        for (int s2p = 0; s2p < d; s2p++) {
            for (int n = 0; n < D; n++) {
                Scalar beta = (n == 0) ? Traits::zero() : Traits::one();
                // Batch over s1p: T2 columns at offsets n*dd+s1p*d+s2p are spaced by d in ws_out
                // A: T2 + (n*dd + 0*d + s2p)*cL*cR, strideA = d*cL*cR
                // B: R_env + n*cR, strideB = 0 (shared)
                // C: d_result + 0*cL + s2p*cL*d, strideC = cL (interleaved by s1p)
                ROCBLAS_CHECK(Traits::gemm_strided_batched(rocblas_h_,
                    rocblas_operation_none, rocblas_operation_none,
                    cL, cR, cR, &one,
                    T2 + (size_t)(n * dd + s2p) * cL * cR, cL, (rocblas_stride)(d * cL * cR),
                    R_env + n * cR, cR * D, (rocblas_stride)0,
                    &beta, d_result + s2p * cL * d, cL * dd, (rocblas_stride)cL, d));
            }
        }
    } else {
        for (int s1p = 0; s1p < d; s1p++) {
            for (int s2p = 0; s2p < d; s2p++) {
                for (int n = 0; n < D; n++) {
                    Scalar beta = (n == 0) ? Traits::zero() : Traits::one();
                    int ws_out = n * dd + s1p * d + s2p;
                    ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
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
}

// ============================================================================
// Left environment update
// ============================================================================

template<typename Scalar>
void DMRG2GPUOpt<Scalar>::update_left_env(int site) {
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
        HIP_CHECK(hipMemcpyAsync(d_batch_A_, h_A.data(), D*d*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
        HIP_CHECK(hipMemcpyAsync(d_batch_B_, h_B.data(), D*d*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
        HIP_CHECK(hipMemcpyAsync(d_batch_C_, h_C.data(), D*d*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
        ROCBLAS_CHECK(Traits::gemm_batched(rocblas_h_,
            Traits::op_t, rocblas_operation_none,
            chi_in, chi_out, chi_in,
            &one,
            (const Scalar**)d_batch_A_, chi_in * D,
            (const Scalar**)d_batch_B_, chi_in * d,
            &zero_val,
            d_batch_C_, chi_in,
            D * d));
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

    // Step 3: L_new_w'[b,b'] = sum_{a',s'} U[a',ws',b] * conj(A[a',s',b'])
    for (int wp = 0; wp < D; wp++) {
        for (int sp = 0; sp < d; sp++) {
            Scalar beta = (sp == 0) ? Traits::zero() : Traits::one();
            int ws_out = wp * d + sp;
            ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
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
        conjugate_inplace(L_new, chi_out * D * chi_out, stream_);
    }
}

// ============================================================================
// Right environment update
// ============================================================================

template<typename Scalar>
void DMRG2GPUOpt<Scalar>::update_right_env(int site) {
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
        HIP_CHECK(hipMemcpyAsync(d_batch_A_, h_A.data(), D*d*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
        HIP_CHECK(hipMemcpyAsync(d_batch_B_, h_B.data(), D*d*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
        HIP_CHECK(hipMemcpyAsync(d_batch_C_, h_C.data(), D*d*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
        ROCBLAS_CHECK(Traits::gemm_batched(rocblas_h_,
            rocblas_operation_none, rocblas_operation_none,
            chi_out, chi_in, chi_in,
            &one,
            (const Scalar**)d_batch_A_, chi_out * d,
            (const Scalar**)d_batch_B_, chi_in * D,
            &zero_val,
            d_batch_C_, chi_out,
            D * d));
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

    // Step 3: R_new_w[a,a'] = sum_s' U_ws'[a,b'] * A_s'^H[b',a']
    for (int w = 0; w < D; w++) {
        for (int sp = 0; sp < d; sp++) {
            Scalar beta = (sp == 0) ? Traits::zero() : Traits::one();
            int ws_out = w * d + sp;
            ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
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
// Environment building
// ============================================================================

template<typename Scalar>
void DMRG2GPUOpt<Scalar>::build_initial_environments() {
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

    // Build all R environments from right to left
    for (int i = L_ - 1; i >= 0; i--) {
        update_right_env(i);
    }
}

// ============================================================================
// Lanczos eigensolver (fallback for small systems / Davidson failure)
// ============================================================================

template<typename Scalar>
double DMRG2GPUOpt<Scalar>::lanczos_eigensolver(int site, Scalar* d_theta, int theta_size) {
    int n = theta_size;
    int max_iter = std::min(max_lanczos_iter_, n);
    double tol_lanczos = 1e-12;
    double tol_eig_conv = 1e-12;

    Scalar* d_lanczos_v = d_lanczos_v_;

    std::vector<double> h_alpha(max_iter);
    std::vector<double> h_beta(max_iter);

    // v[0] = theta / ||theta||
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

    double prev_energy = 1e30;
    int iter;
    for (iter = 0; iter < max_iter; iter++) {
        Scalar* d_vi = d_lanczos_v + (size_t)iter * n;

        // w = H|v_i>
        apply_heff_two_site(site, d_vi, d_heff_result_);

        // alpha_i = <v_i|w>
        Scalar alpha_result;
        ROCBLAS_CHECK(Traits::dot(rocblas_h_, n, d_vi, 1, d_heff_result_, 1, &alpha_result));
        double alpha_i = Traits::real_part(alpha_result);
        h_alpha[iter] = alpha_i;

        // w = w - alpha_i * v_i
        Scalar neg_alpha = Traits::make_scalar(-alpha_i);
        ROCBLAS_CHECK(Traits::axpy(rocblas_h_, n, &neg_alpha, d_vi, 1, d_heff_result_, 1));

        // w = w - beta_{i-1} * v_{i-1}
        if (iter > 0) {
            Scalar neg_beta = Traits::make_scalar(-h_beta[iter - 1]);
            Scalar* d_vim1 = d_lanczos_v + (size_t)(iter - 1) * n;
            ROCBLAS_CHECK(Traits::axpy(rocblas_h_, n, &neg_beta, d_vim1, 1, d_heff_result_, 1));
        }

        // Full reorthogonalization via gemv
        if (iter > 0) {
            Scalar one_val = Traits::one(), zero_sc = Traits::zero(), neg_one = Traits::neg(Traits::one());
            ROCBLAS_CHECK(Traits::gemv(rocblas_h_, Traits::op_h,
                n, iter + 1, &one_val,
                d_lanczos_v, n,
                d_heff_result_, 1,
                &zero_sc, d_ritz_coeffs_, 1));
            ROCBLAS_CHECK(Traits::gemv(rocblas_h_, rocblas_operation_none,
                n, iter + 1, &neg_one,
                d_lanczos_v, n,
                d_ritz_coeffs_, 1,
                &one_val, d_heff_result_, 1));
        } else {
            Scalar overlap;
            ROCBLAS_CHECK(Traits::dot(rocblas_h_, n, d_lanczos_v, 1, d_heff_result_, 1, &overlap));
            Scalar neg_overlap = Traits::neg(overlap);
            ROCBLAS_CHECK(Traits::axpy(rocblas_h_, n, &neg_overlap, d_lanczos_v, 1, d_heff_result_, 1));
        }

        // beta_i = ||w||
        double beta_i;
        ROCBLAS_CHECK(Traits::nrm2(rocblas_h_, n, d_heff_result_, 1, &beta_i));
        h_beta[iter] = beta_i;

        if (beta_i < tol_lanczos) {
            iter++;
            break;
        }

        // Eigenvalue convergence check (every 3 iterations after iter >= 4)
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
            HIP_CHECK(hipMemcpy(d_vip1, d_heff_result_, n * sizeof(Scalar), hipMemcpyDeviceToDevice));
            ROCBLAS_CHECK(Traits::scal_real(rocblas_h_, n, &scale, d_vip1, 1));
        }
    }

    int niter = iter;
    prof_davidson_iters += niter;
    prof_heff_calls += niter;

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
    HIP_CHECK(hipMemcpy(d_ritz_coeffs_, h_ritz_scalar.data(), niter * sizeof(Scalar), hipMemcpyHostToDevice));

    Scalar one_sc = Traits::one(), zero_sc = Traits::zero();
    ROCBLAS_CHECK(Traits::gemv(
        rocblas_h_, rocblas_operation_none,
        n, niter, &one_sc,
        d_lanczos_v, n,
        d_ritz_coeffs_, 1,
        &zero_sc, d_theta, 1
    ));

    // Normalize
    ROCBLAS_CHECK(Traits::nrm2(rocblas_h_, n, d_theta, 1, &norm));
    inv_norm = 1.0 / norm;
    ROCBLAS_CHECK(Traits::scal_real(rocblas_h_, n, &inv_norm, d_theta, 1));

    return energy;
}

// ============================================================================
// Newton-Schulz Left Polar Decomposition (tall/square, m >= n)
// A = U @ P, where U^H U = I_n, P = U^H A is PSD
// ============================================================================

template<typename Scalar>
void DMRG2GPUOpt<Scalar>::newton_schulz_left(
    Scalar* d_A, int m, int n,
    Scalar* d_U, Scalar* d_P,
    double tol, int* out_iters) {

    Scalar one = Traits::one(), zero_val = Traits::zero();
    Scalar half = Traits::make_scalar(0.5);

    // Compute ||A||_F
    RealType fro;
    HIP_CHECK(hipStreamSynchronize(stream_));
    ROCBLAS_CHECK(Traits::nrm2(rocblas_h_, m * n, d_A, 1, &fro));

    if (fro < 1e-300) {
        HIP_CHECK(hipMemsetAsync(d_U, 0, m * n * sizeof(Scalar), stream_));
        HIP_CHECK(hipMemsetAsync(d_P, 0, n * n * sizeof(Scalar), stream_));
        if (out_iters) *out_iters = 0;
        return;
    }

    // U = A / ||A||_F
    HIP_CHECK(hipMemcpyAsync(d_U, d_A, m * n * sizeof(Scalar), hipMemcpyDeviceToDevice, stream_));
    RealType inv_fro = 1.0 / fro;
    ROCBLAS_CHECK(Traits::scal_real(rocblas_h_, m * n, &inv_fro, d_U, 1));

    Scalar* d_gram = d_ns_gram_;    // (n, n)
    Scalar* d_U_new = d_ns_U_new_;  // (m, n)

    int total_iters = 0;
    for (int iter = 0; iter < 30; iter++) {
        // 1. gram = U^H @ U  -> (n, n)
        ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
            Traits::op_h, rocblas_operation_none,
            n, n, m, &one, d_U, m, d_U, m, &zero_val, d_gram, n));

        // 2. gram = 3I - gram  (in-place)
        launch_scaled_identity_minus(d_gram, n, 3.0, stream_);

        // 3. U_new = 0.5 * U @ gram  -> (m, n)
        ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
            rocblas_operation_none, rocblas_operation_none,
            m, n, n, &half, d_U, m, d_gram, n, &zero_val, d_U_new, m));

        total_iters = iter + 1;

        // Convergence check every 3 iterations
        if (iter >= 2 && iter % 3 == 0) {
            // diff = U_new - U -> compute in d_heff_result_ (scratch)
            HIP_CHECK(hipMemcpyAsync(d_heff_result_, d_U_new, m * n * sizeof(Scalar),
                                      hipMemcpyDeviceToDevice, stream_));
            Scalar neg_one = Traits::neg(Traits::one());
            ROCBLAS_CHECK(Traits::axpy(rocblas_h_, m * n, &neg_one, d_U, 1, d_heff_result_, 1));

            RealType diff_norm;
            HIP_CHECK(hipStreamSynchronize(stream_));
            ROCBLAS_CHECK(Traits::nrm2(rocblas_h_, m * n, d_heff_result_, 1, &diff_norm));

            // Swap U <-> U_new
            std::swap(d_U, d_U_new);

            if (diff_norm < tol) break;
        } else {
            // Swap without convergence check
            std::swap(d_U, d_U_new);
        }
    }

    if (out_iters) *out_iters = total_iters;

    // Ensure d_U is in d_ns_U_ (the expected output location)
    if (d_U != d_ns_U_) {
        HIP_CHECK(hipMemcpyAsync(d_ns_U_, d_U, m * n * sizeof(Scalar),
                                  hipMemcpyDeviceToDevice, stream_));
    }

    // P = U^H @ A  -> (n, m) x (m, n) -> (n, n)
    ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
        Traits::op_h, rocblas_operation_none,
        n, n, m, &one, d_ns_U_, m, d_A, m, &zero_val, d_P, n));
}

// ============================================================================
// Newton-Schulz bond split
// Uses Newton-Schulz + eigendecomposition of P^H P for truncation
// ============================================================================

template<typename Scalar>
void DMRG2GPUOpt<Scalar>::ns_split(int site, Scalar* d_theta, char direction) {
    int cL = chi_L(site);
    int cR = chi_R(site + 1);

    int m = cL * d_;
    int n_svd = d_ * cR;
    int k = std::min(m, n_svd);
    int max_k = std::min(k, chi_max_user_);

    // For very small systems, fall back to SVD
    if (k <= 4 || m < 2 || n_svd < 2) {
        svd_split_fallback(site, d_theta, direction);
        return;
    }

    if (m >= n_svd) {
        // Tall/square: left Newton-Schulz -> A = U_ns @ P
        int ns_iters = 0;
        newton_schulz_left(d_theta, m, n_svd, d_ns_U_, d_ns_P_, 1e-10, &ns_iters);
        prof_ns_iters += ns_iters;

        // If NS didn't converge well, fall back to SVD
        if (ns_iters >= 29) {
            svd_split_fallback(site, d_theta, direction);
            return;
        }

        HIP_CHECK(hipStreamSynchronize(stream_));

        // Verify ||U^H U - I||_F < tol to catch silent NS failures
        {
            Scalar one_v = Traits::one(), zero_v = Traits::zero();
            ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
                Traits::op_h, rocblas_operation_none,
                n_svd, n_svd, m, &one_v, d_ns_U_, m, d_ns_U_, m,
                &zero_v, d_ns_gram_, n_svd));

            std::vector<Scalar> h_UtU(n_svd * n_svd);
            HIP_CHECK(hipMemcpy(h_UtU.data(), d_ns_gram_,
                                n_svd * n_svd * sizeof(Scalar), hipMemcpyDeviceToHost));
            RealType frob2 = 0.0;
            for (int j = 0; j < n_svd; j++) {
                for (int i = 0; i < n_svd; i++) {
                    RealType re = Traits::real_part(h_UtU[i + j * n_svd]);
                    if (i == j) re -= 1.0;
                    frob2 += re * re;
                    if constexpr (Traits::is_complex) {
                        RealType im = hipCimag(h_UtU[i + j * n_svd]);
                        frob2 += im * im;
                    }
                }
            }
            if (std::sqrt(frob2) > 1e-10) {
                svd_split_fallback(site, d_theta, direction);
                return;
            }
        }

        // Eigendecompose P^H P -> eigenvalues sigma^2, eigenvectors V
        Scalar one = Traits::one(), zero_val = Traits::zero();
        Scalar* d_PtP = d_ns_gram_;  // reuse gram buffer for (n_svd, n_svd)
        ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
            Traits::op_h, rocblas_operation_none,
            n_svd, n_svd, n_svd, &one, d_ns_P_, n_svd, d_ns_P_, n_svd,
            &zero_val, d_PtP, n_svd));

        // Copy P^H P to host
        HIP_CHECK(hipMemcpy(h_ns_PtP_.data(), d_PtP, n_svd * n_svd * sizeof(Scalar),
                            hipMemcpyDeviceToHost));

        // Eigendecompose on CPU
        int info;
        const char jobz = 'V', uplo = 'U';
        int lwork = (int)h_ns_syev_work_.size();
        Traits::lapack_syev(&jobz, &uplo, &n_svd,
                h_ns_PtP_.data(), &n_svd,
                h_ns_eigvals_.data(),
                h_ns_syev_work_.data(), &lwork,
                h_ns_syev_rwork_.empty() ? nullptr : h_ns_syev_rwork_.data(),
                &info);

        if (info != 0) {
            svd_split_fallback(site, d_theta, direction);
            return;
        }

        // Eigenvalues are in ascending order. Singular values = sqrt(eigenvalues).
        // Reverse to get descending order.
        std::vector<RealType> sing_vals(n_svd);
        for (int i = 0; i < n_svd; i++) {
            RealType ev = h_ns_eigvals_[n_svd - 1 - i];
            sing_vals[i] = (ev > 0) ? std::sqrt(ev) : 0.0;
        }

        // Truncation
        int new_k = std::min(max_k, n_svd);
        for (int i = 0; i < new_k; i++) {
            if (sing_vals[i] < 1e-14) { new_k = i; break; }
        }
        if (new_k == 0) new_k = 1;

        // Build Vh_trunc (new_k x n_svd) on host: rows are reversed eigenvectors
        std::vector<Scalar> h_Vh_trunc(new_k * n_svd);
        for (int i = 0; i < new_k; i++) {
            int src_col = n_svd - 1 - i;
            for (int j = 0; j < n_svd; j++) {
                Scalar v_val = h_ns_PtP_[j + src_col * n_svd];
                if constexpr (Traits::is_complex) {
                    h_Vh_trunc[i + j * new_k] = make_hipDoubleComplex(hipCreal(v_val), -hipCimag(v_val));
                } else {
                    h_Vh_trunc[i + j * new_k] = v_val;
                }
            }
        }

        // Upload Vh_trunc to GPU (into d_svd_Vh_)
        HIP_CHECK(hipMemcpy(d_svd_Vh_, h_Vh_trunc.data(),
                            new_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice));

        // Compute U_p_trunc = P @ V_trunc @ diag(1/S) on GPU
        // First: upload V_trunc (n_svd x new_k) to GPU
        std::vector<Scalar> h_V_trunc(n_svd * new_k);
        for (int i = 0; i < new_k; i++) {
            int src_col = n_svd - 1 - i;
            for (int j = 0; j < n_svd; j++) {
                h_V_trunc[j + i * n_svd] = h_ns_PtP_[j + src_col * n_svd];
            }
        }
        HIP_CHECK(hipMemcpy(d_svd_U_, h_V_trunc.data(),
                            n_svd * new_k * sizeof(Scalar), hipMemcpyHostToDevice));

        // U_p = P @ V_trunc -> (n_svd, n_svd) x (n_svd, new_k) -> (n_svd, new_k)
        // Store in d_svd_A_ (scratch)
        ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
            rocblas_operation_none, rocblas_operation_none,
            n_svd, new_k, n_svd, &one, d_ns_P_, n_svd,
            d_svd_U_, n_svd, &zero_val, d_svd_A_, n_svd));

        // Scale columns by 1/S: U_p[:, i] /= S[i]
        for (int i = 0; i < new_k; i++) {
            if (sing_vals[i] > 1e-14) {
                RealType inv_s = 1.0 / sing_vals[i];
                ROCBLAS_CHECK(Traits::scal_real(rocblas_h_, n_svd, &inv_s,
                    d_svd_A_ + i * n_svd, 1));
            }
        }

        // U_full = U_ns @ U_p -> (m, n_svd) x (n_svd, new_k) -> (m, new_k)
        // Store in d_svd_U_ (reuse, large enough)
        ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
            rocblas_operation_none, rocblas_operation_none,
            m, new_k, n_svd, &one, d_ns_U_, m,
            d_svd_A_, n_svd, &zero_val, d_svd_U_, m));

        HIP_CHECK(hipStreamSynchronize(stream_));

        // Upload singular values to device for GPU-side scaling
        HIP_CHECK(hipMemcpyAsync(d_svd_S_, sing_vals.data(), new_k * sizeof(RealType),
                                  hipMemcpyHostToDevice, stream_));

        // Store MPS tensors -- scale on GPU
        if (direction == 'R') {
            // MPS[site] = U_full[:, :new_k] -> (cL*d, new_k) = (m, new_k)
            allocate_mps_tensor(site, cL, new_k);
            HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site], d_svd_U_,
                        m * new_k * sizeof(Scalar), hipMemcpyDeviceToDevice, stream_));

            // MPS[site+1] = diag(S) @ Vh[:new_k, :] -- scale rows on GPU
            allocate_mps_tensor(site + 1, new_k, cR);
            scale_rows_by_real(d_svd_Vh_, new_k, d_svd_S_,
                               d_mps_tensors_[site + 1], new_k, new_k, n_svd, stream_);
        } else {
            // MPS[site] = U_full @ diag(S) -- scale columns on GPU
            allocate_mps_tensor(site, cL, new_k);
            scale_columns_by_real(d_svd_U_, m, d_svd_S_,
                                  d_mps_tensors_[site], m, m, new_k, stream_);

            // MPS[site+1] = Vh[:new_k, :] -- upload from host
            allocate_mps_tensor(site + 1, new_k, cR);
            HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site + 1], h_Vh_trunc.data(),
                        new_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice, stream_));
        }

        bond_dims_[site + 1] = new_k;

    } else {
        // Wide case: fall back to SVD for simplicity
        svd_split_fallback(site, d_theta, direction);
        return;
    }
}

// ============================================================================
// SVD split fallback (CPU LAPACK only)
// ============================================================================

template<typename Scalar>
void DMRG2GPUOpt<Scalar>::svd_split_fallback(int site, Scalar* d_theta, char direction) {
    int cL = chi_L(site);
    int cR = chi_R(site + 1);

    int m = cL * d_;
    int n_svd = d_ * cR;
    int full_k = std::min(m, n_svd);
    int k = std::min(full_k, chi_max_user_);

    // CPU SVD path
    HIP_CHECK(hipMemcpy(h_svd_A_.data(), d_theta, m * n_svd * sizeof(Scalar), hipMemcpyDeviceToHost));

    int lwork = (int)h_svd_work_.size();
    int info;
    const char jobu = 'S', jobvt = 'S';
    Traits::lapack_gesvd(&jobu, &jobvt, &m, &n_svd, h_svd_A_.data(), &m, h_svd_S_.data(),
            h_svd_U_.data(), &m, h_svd_Vh_.data(), &full_k,
            h_svd_work_.data(), &lwork,
            h_svd_rwork_.empty() ? nullptr : h_svd_rwork_.data(), &info);

    if (info != 0) {
        throw std::runtime_error("svd_split_fallback: LAPACK gesvd failed, info=" + std::to_string(info));
    }

    Scalar* h_U_data = h_svd_U_.data();
    RealType* h_S_data = h_svd_S_.data();
    Scalar* h_Vh_data = h_svd_Vh_.data();

    // Truncation
    int new_k = k;
    for (int i = 0; i < new_k; i++) {
        if (h_S_data[i] < 1e-14) { new_k = i; break; }
    }
    if (new_k == 0) new_k = 1;

    if (direction == 'R') {
        // U -> MPS[site] (left-canonical), S*Vh -> MPS[site+1]
        allocate_mps_tensor(site, cL, new_k);
        HIP_CHECK(hipMemcpy(d_mps_tensors_[site], h_U_data,
                            m * new_k * sizeof(Scalar), hipMemcpyHostToDevice));

        // Compute S*Vh on CPU: (new_k, n_svd)
        for (int j = 0; j < n_svd; j++)
            for (int i = 0; i < new_k; i++)
                h_svd_tmp_[i + j * new_k] = Traits::scale_by_real(h_S_data[i], h_Vh_data[i + j * full_k]);

        allocate_mps_tensor(site + 1, new_k, cR);
        HIP_CHECK(hipMemcpy(d_mps_tensors_[site + 1], h_svd_tmp_.data(),
                            new_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice));

    } else {  // direction == 'L'
        // U*S -> MPS[site], Vh -> MPS[site+1] (right-canonical)
        for (int j = 0; j < new_k; j++)
            for (int i = 0; i < m; i++)
                h_svd_tmp_[i + j * m] = Traits::scale_by_real(h_S_data[j], h_U_data[i + j * m]);

        allocate_mps_tensor(site, cL, new_k);
        HIP_CHECK(hipMemcpy(d_mps_tensors_[site], h_svd_tmp_.data(),
                            m * new_k * sizeof(Scalar), hipMemcpyHostToDevice));

        allocate_mps_tensor(site + 1, new_k, cR);
        if (new_k == full_k) {
            HIP_CHECK(hipMemcpy(d_mps_tensors_[site + 1], h_Vh_data,
                                full_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice));
        } else {
            for (int j = 0; j < n_svd; j++)
                for (int i = 0; i < new_k; i++)
                    h_svd_tmp_[i + j * new_k] = h_Vh_data[i + j * full_k];
            HIP_CHECK(hipMemcpy(d_mps_tensors_[site + 1], h_svd_tmp_.data(),
                                new_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice));
        }
    }

    bond_dims_[site + 1] = new_k;
}

// ============================================================================
// Block-Davidson Eigensolver
// ============================================================================

template<typename Scalar>
double DMRG2GPUOpt<Scalar>::block_davidson_eigensolver(int site, Scalar* d_theta, int theta_size) {
    int dim = theta_size;
    int b = std::min(davidson_b_, dim);
    int max_sub = std::min(davidson_max_sub_, dim);
    int max_iter = 30;
    double tol_dav = 1e-10;

    Scalar one = Traits::one(), zero_val = Traits::zero();
    Scalar neg_one = Traits::neg(Traits::one());

    // For tiny systems, use Lanczos fallback
    if (dim <= 2 * b) {
        return lanczos_eigensolver(site, d_theta, theta_size);
    }

    // Initialize V: first column = theta/||theta||
    RealType norm;
    ROCBLAS_CHECK(Traits::nrm2(rocblas_h_, dim, d_theta, 1, &norm));
    if (norm < 1e-14) {
        srand(42 + site);
        std::vector<Scalar> h_init(dim);
        for (int i = 0; i < dim; i++) h_init[i] = Traits::random_val();
        HIP_CHECK(hipMemcpy(d_theta, h_init.data(), dim * sizeof(Scalar), hipMemcpyHostToDevice));
        ROCBLAS_CHECK(Traits::nrm2(rocblas_h_, dim, d_theta, 1, &norm));
    }

    Scalar* V = d_dav_V_;   // (dim, max_sub)
    Scalar* AV = d_dav_AV_; // (dim, max_sub)

    // V[:, 0] = theta / norm
    HIP_CHECK(hipMemcpyAsync(V, d_theta, dim * sizeof(Scalar), hipMemcpyDeviceToDevice, stream_));
    RealType inv_norm = 1.0 / norm;
    ROCBLAS_CHECK(Traits::scal_real(rocblas_h_, dim, &inv_norm, V, 1));

    // Fill remaining b-1 columns with random orthogonalized vectors
    srand(42 + site);
    for (int i = 1; i < b; i++) {
        std::vector<Scalar> h_v(dim);
        for (int j = 0; j < dim; j++) h_v[j] = Traits::random_val();
        HIP_CHECK(hipMemcpyAsync(V + (size_t)i * dim, h_v.data(), dim * sizeof(Scalar),
                                  hipMemcpyHostToDevice, stream_));

        // Orthogonalize against previous columns using CGS
        ROCBLAS_CHECK(Traits::gemv(rocblas_h_, Traits::op_h,
            dim, i, &one, V, dim, V + (size_t)i * dim, 1,
            &zero_val, d_dav_work_, 1));
        ROCBLAS_CHECK(Traits::gemv(rocblas_h_, rocblas_operation_none,
            dim, i, &neg_one, V, dim, d_dav_work_, 1,
            &one, V + (size_t)i * dim, 1));

        // Normalize
        RealType nrm_v;
        HIP_CHECK(hipStreamSynchronize(stream_));
        ROCBLAS_CHECK(Traits::nrm2(rocblas_h_, dim, V + (size_t)i * dim, 1, &nrm_v));
        if (nrm_v < 1e-14) {
            for (int j = 0; j < dim; j++) h_v[j] = Traits::random_val();
            HIP_CHECK(hipMemcpy(V + (size_t)i * dim, h_v.data(), dim * sizeof(Scalar), hipMemcpyHostToDevice));
            ROCBLAS_CHECK(Traits::nrm2(rocblas_h_, dim, V + (size_t)i * dim, 1, &nrm_v));
        }
        RealType inv_nrm_v = 1.0 / nrm_v;
        ROCBLAS_CHECK(Traits::scal_real(rocblas_h_, dim, &inv_nrm_v, V + (size_t)i * dim, 1));
    }

    // Compute AV[:, j] = H @ V[:, j] for j = 0..b-1
    for (int j = 0; j < b; j++) {
        apply_heff_two_site(site, V + (size_t)j * dim, AV + (size_t)j * dim);
    }

    double best_energy = 1e30;
    double energy_prev = 1e30;
    int k = b;  // current subspace size

    for (int iteration = 0; iteration < max_iter; iteration++) {
        HIP_CHECK(hipStreamSynchronize(stream_));

        // Rayleigh-Ritz: H_proj = V^H @ AV  -> (k, k)
        ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
            Traits::op_h, rocblas_operation_none,
            k, k, dim, &one, V, dim, AV, dim, &zero_val, d_dav_work_, k));

        // Copy H_proj to host
        HIP_CHECK(hipMemcpy(h_dav_H_proj_.data(), d_dav_work_,
                            k * k * sizeof(Scalar), hipMemcpyDeviceToHost));

        // Symmetrize on host: H_proj = 0.5 * (H_proj + H_proj^H)
        for (int i = 0; i < k; i++) {
            for (int j = i; j < k; j++) {
                Scalar hij = h_dav_H_proj_[i + j * k];
                Scalar hji = h_dav_H_proj_[j + i * k];
                Scalar sym;
                if constexpr (Traits::is_complex) {
                    sym = make_hipDoubleComplex(
                        0.5 * (hipCreal(hij) + hipCreal(hji)),
                        0.5 * (hipCimag(hij) - hipCimag(hji)));
                    h_dav_H_proj_[i + j * k] = sym;
                    h_dav_H_proj_[j + i * k] = make_hipDoubleComplex(hipCreal(sym), -hipCimag(sym));
                } else {
                    sym = 0.5 * (hij + hji);
                    h_dav_H_proj_[i + j * k] = sym;
                    h_dav_H_proj_[j + i * k] = sym;
                }
            }
        }

        // Eigendecompose H_proj on CPU
        std::copy(h_dav_H_proj_.begin(), h_dav_H_proj_.begin() + k * k,
                  h_dav_eigvecs_.begin());
        int info;
        const char jobz = 'V', uplo = 'U';
        int lwork = -1;
        Scalar work_opt;
        // Query workspace (need valid rwork for zheev_)
        std::vector<RealType> syev_rwork_q(std::max(1, Traits::syev_rwork_size(k)));
        Traits::lapack_syev(&jobz, &uplo, &k,
                h_dav_eigvecs_.data(), &k, h_dav_eigvals_.data(),
                &work_opt, &lwork,
                syev_rwork_q.empty() ? nullptr : syev_rwork_q.data(), &info);
        if constexpr (Traits::is_complex) {
            lwork = (int)Traits::real_part(work_opt) + 1;
        } else {
            lwork = (int)work_opt + 1;
        }
        std::vector<Scalar> syev_work(lwork);
        std::vector<RealType> syev_rwork(Traits::syev_rwork_size(k));
        Traits::lapack_syev(&jobz, &uplo, &k,
                h_dav_eigvecs_.data(), &k, h_dav_eigvals_.data(),
                syev_work.data(), &lwork,
                syev_rwork.empty() ? nullptr : syev_rwork.data(), &info);

        if (info != 0) {
            // Eigendecomp failed -- fall back to Lanczos
            return lanczos_eigensolver(site, d_theta, theta_size);
        }

        double energy = h_dav_eigvals_[0];  // lowest eigenvalue

        if (energy < best_energy) {
            best_energy = energy;
        }

        // Upload eigenvectors to GPU for Ritz vector computation
        HIP_CHECK(hipMemcpy(d_dav_work2_, h_dav_eigvecs_.data(),
                            k * k * sizeof(Scalar), hipMemcpyHostToDevice));

        // x0 = V @ eigvecs[:, 0] -> (dim, 1)
        ROCBLAS_CHECK(Traits::gemv(rocblas_h_, rocblas_operation_none,
            dim, k, &one, V, dim, d_dav_work2_, 1, &zero_val, d_theta, 1));

        // ax0 = AV @ eigvecs[:, 0] -> (dim, 1)
        ROCBLAS_CHECK(Traits::gemv(rocblas_h_, rocblas_operation_none,
            dim, k, &one, AV, dim, d_dav_work2_, 1, &zero_val, d_heff_result_, 1));

        // Residual: r = ax0 - energy * x0
        Scalar neg_energy = Traits::make_scalar(-energy);
        ROCBLAS_CHECK(Traits::axpy(rocblas_h_, dim, &neg_energy, d_theta, 1, d_heff_result_, 1));

        RealType res_norm;
        HIP_CHECK(hipStreamSynchronize(stream_));
        ROCBLAS_CHECK(Traits::nrm2(rocblas_h_, dim, d_heff_result_, 1, &res_norm));

        if (res_norm < tol_dav && std::abs(energy - energy_prev) < tol_dav) {
            // Converged: d_theta already has the ground state vector
            ROCBLAS_CHECK(Traits::nrm2(rocblas_h_, dim, d_theta, 1, &norm));
            inv_norm = 1.0 / norm;
            ROCBLAS_CHECK(Traits::scal_real(rocblas_h_, dim, &inv_norm, d_theta, 1));
            return energy;
        }
        energy_prev = energy;

        // Expand subspace with residual corrections
        int n_new = 0;
        for (int i = 0; i < std::min(b, k); i++) {
            Scalar* r_i = d_dav_work_ + (size_t)n_new * dim;

            // r_i = AV @ eigvecs[:, i] - eigvals[i] * V @ eigvecs[:, i]
            ROCBLAS_CHECK(Traits::gemv(rocblas_h_, rocblas_operation_none,
                dim, k, &one, V, dim, d_dav_work2_ + i * k, 1, &zero_val, r_i, 1));
            ROCBLAS_CHECK(Traits::gemv(rocblas_h_, rocblas_operation_none,
                dim, k, &one, AV, dim, d_dav_work2_ + i * k, 1, &zero_val, d_heff_result_, 1));
            Scalar neg_ei = Traits::make_scalar(-h_dav_eigvals_[i]);
            ROCBLAS_CHECK(Traits::scal(rocblas_h_, dim, &neg_ei, r_i, 1));
            ROCBLAS_CHECK(Traits::axpy(rocblas_h_, dim, &one, d_heff_result_, 1, r_i, 1));

            RealType ri_norm;
            HIP_CHECK(hipStreamSynchronize(stream_));
            ROCBLAS_CHECK(Traits::nrm2(rocblas_h_, dim, r_i, 1, &ri_norm));

            if (ri_norm > tol_dav * 0.01) {
                RealType inv_ri = 1.0 / ri_norm;
                ROCBLAS_CHECK(Traits::scal_real(rocblas_h_, dim, &inv_ri, r_i, 1));
                n_new++;
            }
        }

        if (n_new == 0) {
            ROCBLAS_CHECK(Traits::nrm2(rocblas_h_, dim, d_theta, 1, &norm));
            inv_norm = 1.0 / norm;
            ROCBLAS_CHECK(Traits::scal_real(rocblas_h_, dim, &inv_norm, d_theta, 1));
            return energy;
        }

        // Orthogonalize new vectors against V
        Scalar* W = d_dav_work_;

        // overlap = V^H @ W -> (k, n_new)
        ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
            Traits::op_h, rocblas_operation_none,
            k, n_new, dim, &one, V, dim, W, dim, &zero_val, d_dav_work2_, k));

        // W -= V @ overlap
        ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
            rocblas_operation_none, rocblas_operation_none,
            dim, n_new, k, &neg_one, V, dim, d_dav_work2_, k, &one, W, dim));

        // Orthogonalize new vectors among themselves (CGS within the block)
        int n_good = 0;
        for (int i = 0; i < n_new; i++) {
            Scalar* wi = W + (size_t)i * dim;

            for (int j = 0; j < n_good; j++) {
                Scalar* wj = W + (size_t)j * dim;
                Scalar overlap_val;
                ROCBLAS_CHECK(Traits::dot(rocblas_h_, dim, wj, 1, wi, 1, &overlap_val));
                Scalar neg_ov = Traits::neg(overlap_val);
                ROCBLAS_CHECK(Traits::axpy(rocblas_h_, dim, &neg_ov, wj, 1, wi, 1));
            }

            RealType wi_norm;
            HIP_CHECK(hipStreamSynchronize(stream_));
            ROCBLAS_CHECK(Traits::nrm2(rocblas_h_, dim, wi, 1, &wi_norm));

            if (wi_norm > 1e-14) {
                RealType inv_wi = 1.0 / wi_norm;
                ROCBLAS_CHECK(Traits::scal_real(rocblas_h_, dim, &inv_wi, wi, 1));
                if (n_good != i) {
                    HIP_CHECK(hipMemcpyAsync(W + (size_t)n_good * dim, wi, dim * sizeof(Scalar),
                                              hipMemcpyDeviceToDevice, stream_));
                }
                n_good++;
            }
        }

        if (n_good == 0) {
            ROCBLAS_CHECK(Traits::nrm2(rocblas_h_, dim, d_theta, 1, &norm));
            inv_norm = 1.0 / norm;
            ROCBLAS_CHECK(Traits::scal_real(rocblas_h_, dim, &inv_norm, d_theta, 1));
            return energy;
        }

        // Check if subspace would be too large -> restart
        if (k + n_good > max_sub) {
            int keep = std::min(b, k);

            // X_keep = V @ eigvecs[:, :keep] -> (dim, keep)
            HIP_CHECK(hipMemcpy(d_dav_work2_, h_dav_eigvecs_.data(),
                                k * k * sizeof(Scalar), hipMemcpyHostToDevice));
            ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
                rocblas_operation_none, rocblas_operation_none,
                dim, keep, k, &one, V, dim, d_dav_work2_, k,
                &zero_val, d_dav_work_, dim));

            // Copy X_keep to V[:, :keep]
            HIP_CHECK(hipMemcpyAsync(V, d_dav_work_, (size_t)dim * keep * sizeof(Scalar),
                                      hipMemcpyDeviceToDevice, stream_));

            // Re-orthogonalize V columns (MGS on GPU)
            for (int i = 0; i < keep; i++) {
                for (int j = 0; j < i; j++) {
                    Scalar ov;
                    ROCBLAS_CHECK(Traits::dot(rocblas_h_, dim,
                        V + (size_t)j * dim, 1, V + (size_t)i * dim, 1, &ov));
                    Scalar neg_ov = Traits::neg(ov);
                    ROCBLAS_CHECK(Traits::axpy(rocblas_h_, dim, &neg_ov,
                        V + (size_t)j * dim, 1, V + (size_t)i * dim, 1));
                }
                RealType vi_norm;
                HIP_CHECK(hipStreamSynchronize(stream_));
                ROCBLAS_CHECK(Traits::nrm2(rocblas_h_, dim, V + (size_t)i * dim, 1, &vi_norm));
                RealType inv_vi = 1.0 / vi_norm;
                ROCBLAS_CHECK(Traits::scal_real(rocblas_h_, dim, &inv_vi, V + (size_t)i * dim, 1));
            }

            // Recompute AV for kept vectors
            for (int j = 0; j < keep; j++) {
                apply_heff_two_site(site, V + (size_t)j * dim, AV + (size_t)j * dim);
            }

            k = keep;
            continue;
        }

        // Expand: append new vectors and compute their H-images
        for (int j = 0; j < n_good; j++) {
            HIP_CHECK(hipMemcpyAsync(V + (size_t)(k + j) * dim, W + (size_t)j * dim,
                                      dim * sizeof(Scalar), hipMemcpyDeviceToDevice, stream_));
            apply_heff_two_site(site, V + (size_t)(k + j) * dim, AV + (size_t)(k + j) * dim);
        }
        k += n_good;
    }

    // Didn't converge -- use best result
    ROCBLAS_CHECK(Traits::nrm2(rocblas_h_, dim, d_theta, 1, &norm));
    inv_norm = 1.0 / norm;
    ROCBLAS_CHECK(Traits::scal_real(rocblas_h_, dim, &inv_norm, d_theta, 1));
    return best_energy;
}

// ============================================================================
// Bond optimization (two-site)
// ============================================================================

template<typename Scalar>
double DMRG2GPUOpt<Scalar>::optimize_bond(int site, char direction) {
    form_theta_two_site(site);
    int cL = chi_L(site);
    int cR = chi_R(site + 1);
    int theta_size = cL * d_ * d_ * cR;

    auto t0 = std::chrono::high_resolution_clock::now();
    double energy = block_davidson_eigensolver(site, d_theta_, theta_size);
    auto t1 = std::chrono::high_resolution_clock::now();
    prof_davidson_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

    t0 = std::chrono::high_resolution_clock::now();
    HIP_CHECK(hipStreamSynchronize(stream_));
    ns_split(site, d_theta_, direction);
    t1 = std::chrono::high_resolution_clock::now();
    prof_ns_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

    prof_site_count++;
    return energy;
}

// ============================================================================
// Sweep methods
// ============================================================================

template<typename Scalar>
double DMRG2GPUOpt<Scalar>::sweep_left_to_right() {
    double energy = 0.0;

    for (int site = 0; site < L_ - 1; site++) {
        energy = optimize_bond(site, 'R');
        HIP_CHECK(hipStreamSynchronize(stream_));
        auto te0 = std::chrono::high_resolution_clock::now();
        update_left_env(site);
        HIP_CHECK(hipStreamSynchronize(stream_));
        auto te1 = std::chrono::high_resolution_clock::now();
        prof_env_ms += std::chrono::duration<double, std::milli>(te1 - te0).count();
    }

    return energy;
}

template<typename Scalar>
double DMRG2GPUOpt<Scalar>::sweep_right_to_left() {
    double energy = 0.0;

    for (int site = L_ - 2; site >= 0; site--) {
        energy = optimize_bond(site, 'L');
        HIP_CHECK(hipStreamSynchronize(stream_));
        auto te0 = std::chrono::high_resolution_clock::now();
        update_right_env(site + 1);
        HIP_CHECK(hipStreamSynchronize(stream_));
        auto te1 = std::chrono::high_resolution_clock::now();
        prof_env_ms += std::chrono::duration<double, std::milli>(te1 - te0).count();
    }

    return energy;
}

// ============================================================================
// Main algorithm
// ============================================================================

template<typename Scalar>
double DMRG2GPUOpt<Scalar>::run(int n_sweeps) {
    const char* type_name = Traits::is_complex ? "complex128" : "float64";
    printf("=== GPU-Native Two-Site DMRG-OPT (NS + Davidson + MFMA-16 pad + batched, %s) ===\n", type_name);
    if (chi_max_ != chi_max_user_)
        printf("L = %d, d = %d, chi_max = %d (padded from %d), D_mpo = %d\n",
               L_, d_, chi_max_, chi_max_user_, D_mpo_);
    else
        printf("L = %d, d = %d, chi_max = %d, D_mpo = %d\n", L_, d_, chi_max_, D_mpo_);
    printf("Running %d sweeps...\n\n", n_sweeps);

    auto t_start = std::chrono::high_resolution_clock::now();

    printf("Building initial environments...\n");
    build_initial_environments();

    auto t_envs = std::chrono::high_resolution_clock::now();
    double env_time = std::chrono::duration<double>(t_envs - t_start).count();
    printf("  Environment build: %.3f s\n\n", env_time);

    double energy_prev = 0.0;

    for (int sweep = 0; sweep < n_sweeps; sweep++) {
        prof_davidson_ms = prof_ns_ms = prof_env_ms = 0;
        prof_davidson_iters = prof_ns_iters = prof_site_count = prof_heff_calls = 0;

        auto t_sweep = std::chrono::high_resolution_clock::now();

        double energy_LR = sweep_left_to_right();
        double energy_RL = sweep_right_to_left();

        auto t_sweep_end = std::chrono::high_resolution_clock::now();
        double sweep_time = std::chrono::duration<double>(t_sweep_end - t_sweep).count();

        energy_ = energy_RL;
        double dE = std::abs(energy_ - energy_prev);

        // Print bond dimensions
        printf("Sweep %3d: E = %.12f, dE = %.2e, time = %.3f s  chi=[",
               sweep, energy_, dE, sweep_time);
        for (int i = 1; i < L_; i++) {
            printf("%d", bond_dims_[i]);
            if (i < L_ - 1) printf(",");
        }
        printf("]\n");

        double other_ms = sweep_time*1000.0 - prof_davidson_ms - prof_ns_ms - prof_env_ms;
        printf("  Profile: davidson=%.0fms (%d iters, %d heff) ns=%.0fms (%d iters) env=%.0fms other=%.0fms\n",
               prof_davidson_ms, prof_davidson_iters, prof_heff_calls,
               prof_ns_ms, prof_ns_iters, prof_env_ms, other_ms);

        if (dE < tol_ && sweep > 0) {
            printf("Converged after %d sweeps!\n", sweep + 1);
            break;
        }

        energy_prev = energy_;
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
void DMRG2GPUOpt<Scalar>::get_mps(std::vector<std::vector<Scalar>>& h_mps) const {
    h_mps.resize(L_);
    for (int i = 0; i < L_; i++) {
        int size = chi_L(i) * d_ * chi_R(i);
        h_mps[i].resize(size);
        HIP_CHECK(hipMemcpy(h_mps[i].data(), d_mps_tensors_[i],
                            size * sizeof(Scalar), hipMemcpyDeviceToHost));
    }
}

#endif // DMRG2_GPU_OPT_IMPL_H
