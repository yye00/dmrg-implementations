#ifndef DMRG_GPU_IMPL_H
#define DMRG_GPU_IMPL_H

#include <rocsolver/rocsolver.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <fstream>
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
static double prof_lanczos_ms = 0, prof_svd_ms = 0, prof_env_ms = 0;
static int prof_lanczos_iters = 0, prof_site_count = 0;
static int prof_heff_calls = 0;

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

    // Batched GEMM pointer arrays
    int batch_max = D_mpo_ * d_;
    HIP_CHECK(hipMalloc(&d_batch_A_, batch_max * sizeof(Scalar*)));
    HIP_CHECK(hipMalloc(&d_batch_B_, batch_max * sizeof(Scalar*)));
    HIP_CHECK(hipMalloc(&d_batch_C_, batch_max * sizeof(Scalar*)));

    // SVD workspace
    int svd_max_dim = chi_max_ * d_;
    HIP_CHECK(hipMalloc(&d_svd_A_,    theta_size_max_ * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_svd_U_,    svd_max_dim * chi_max_ * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_svd_S_,    chi_max_ * sizeof(RealType)));
    HIP_CHECK(hipMalloc(&d_svd_Vh_,   chi_max_ * svd_max_dim * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_svd_E_,    chi_max_ * sizeof(RealType)));
    HIP_CHECK(hipMalloc(&d_svd_info_, sizeof(int)));
    HIP_CHECK(hipMalloc(&d_svd_work_, theta_size_max_ * sizeof(Scalar)));

    // CPU SVD workspace
    use_cpu_svd_ = true;
    use_rsvd_ = false;
    rsvd_oversampling_ = 20;
    h_svd_A_.resize(theta_size_max_);
    h_svd_U_.resize(svd_max_dim * chi_max_);
    h_svd_S_.resize(chi_max_);
    h_svd_Vh_.resize(chi_max_ * svd_max_dim);
    h_svd_tmp_.resize(svd_max_dim * chi_max_);
    h_svd_rwork_.resize(Traits::svd_rwork_size(svd_max_dim, svd_max_dim));

    // Query optimal LAPACK workspace for ALL possible (m, n) combinations.
    // Rectangular matrices can require MORE workspace than square — querying only
    // for the square case (as done previously) caused silent SVD corruption on
    // OpenBLAS 0.3.20 for tall-skinny matrices (m = chi*d, n = chi).
    {
        int max_lwork = 0;
        const char jobu = 'S', jobvt = 'S';
        int dims[][2] = {
            {svd_max_dim, svd_max_dim},  // square
            {svd_max_dim, chi_max_},      // tall: m = chi*d, n = chi (direction R)
            {chi_max_, svd_max_dim},      // wide: m = chi, n = chi*d (direction L)
        };
        for (auto& dim : dims) {
            int qm = dim[0], qn = dim[1];
            int qk = std::min(qm, qn);
            int lwork_query = -1;
            Scalar work_opt;
            int info;
            Traits::lapack_gesvd(&jobu, &jobvt, &qm, &qn, nullptr, &qm, nullptr,
                    nullptr, &qm, nullptr, &qk, &work_opt, &lwork_query,
                    h_svd_rwork_.empty() ? nullptr : h_svd_rwork_.data(), &info);
            int opt_size;
            if constexpr (Traits::is_complex) {
                opt_size = (int)Traits::real_part(work_opt) + 1;
            } else {
                opt_size = (int)work_opt + 1;
            }
            if (opt_size > max_lwork) max_lwork = opt_size;
        }
        h_svd_work_.resize(max_lwork);
    }

    // Randomized truncated SVD workspace
    // For single-site: max m = chi_max*d, max n = chi_max*d (direction-dependent)
    d_rsvd_omega_ = nullptr;
    d_rsvd_Y_ = nullptr;
    d_rsvd_Q_ = nullptr;
    d_rsvd_B_ = nullptr;
    d_rsvd_ipiv_ = nullptr;
    d_rsvd_U_full_ = nullptr;
    {
        int rsvd_r = chi_max_ + rsvd_oversampling_;  // target rank + oversampling
        int rsvd_max_m = svd_max_dim;  // chi*d (max of m across both directions)
        int rsvd_max_n = svd_max_dim;  // chi*d (max of n across both directions)
        HIP_CHECK(hipMalloc(&d_rsvd_omega_, (size_t)rsvd_max_n * rsvd_r * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&d_rsvd_Y_,    (size_t)rsvd_max_m * rsvd_r * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&d_rsvd_Q_,    (size_t)rsvd_max_m * rsvd_r * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&d_rsvd_B_,    (size_t)rsvd_r * rsvd_max_n * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&d_rsvd_ipiv_, (size_t)rsvd_r * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&d_rsvd_U_full_, (size_t)rsvd_max_m * rsvd_r * sizeof(Scalar)));
        h_rsvd_B_.resize((size_t)rsvd_r * rsvd_max_n);
        h_rsvd_U_small_.resize((size_t)rsvd_r * rsvd_r);

        // Query SVD workspace for the smaller rSVD matrix (rsvd_r x rsvd_max_n)
        {
            int sm = rsvd_r, sn = rsvd_max_n;
            int sk = std::min(sm, sn);
            int svd_lwork_query = -1;
            Scalar svd_work_opt;
            int svd_info;
            const char jobu = 'S', jobvt = 'S';
            Traits::lapack_gesvd(&jobu, &jobvt, &sm, &sn, nullptr, &sm, nullptr,
                    nullptr, &sm, nullptr, &sk, &svd_work_opt, &svd_lwork_query,
                    h_svd_rwork_.empty() ? nullptr : h_svd_rwork_.data(), &svd_info);
            int svd_opt;
            if constexpr (Traits::is_complex) {
                svd_opt = (int)Traits::real_part(svd_work_opt) + 1;
            } else {
                svd_opt = (int)svd_work_opt + 1;
            }
            // Ensure h_svd_work_ is large enough for both full and reduced SVD
            if ((int)h_svd_work_.size() < svd_opt) {
                h_svd_work_.resize(svd_opt);
            }
        }
    }
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
    if (d_batch_A_) hipFree(d_batch_A_);
    if (d_batch_B_) hipFree(d_batch_B_);
    if (d_batch_C_) hipFree(d_batch_C_);
    if (d_svd_A_) hipFree(d_svd_A_);
    if (d_svd_U_) hipFree(d_svd_U_);
    if (d_svd_S_) hipFree(d_svd_S_);
    if (d_svd_Vh_) hipFree(d_svd_Vh_);
    if (d_svd_E_) hipFree(d_svd_E_);
    if (d_svd_info_) hipFree(d_svd_info_);
    if (d_svd_work_) hipFree(d_svd_work_);

    if (d_rsvd_omega_) hipFree(d_rsvd_omega_);
    if (d_rsvd_Y_) hipFree(d_rsvd_Y_);
    if (d_rsvd_Q_) hipFree(d_rsvd_Q_);
    if (d_rsvd_B_) hipFree(d_rsvd_B_);
    if (d_rsvd_ipiv_) hipFree(d_rsvd_ipiv_);
    if (d_rsvd_U_full_) hipFree(d_rsvd_U_full_);

    rocblas_destroy_handle(rocblas_h_);
    hipStreamDestroy(stream_);
}

// ============================================================================
// Memory management
// ============================================================================

template<typename Scalar>
void DMRGGPU<Scalar>::allocate_mps_tensor(int site, int cL, int cR) {
    if (d_mps_tensors_[site]) HIP_CHECK(hipFree(d_mps_tensors_[site]));
    HIP_CHECK(hipMalloc(&d_mps_tensors_[site], cL * d_ * cR * sizeof(Scalar)));
}

template<typename Scalar>
void DMRGGPU<Scalar>::ensure_L_env_alloc(int idx, int chi) {
    if (chi > L_env_alloc_chi_[idx]) {
        if (d_L_envs_[idx]) HIP_CHECK(hipFree(d_L_envs_[idx]));
        int sz = chi * D_mpo_ * chi;
        HIP_CHECK(hipMalloc(&d_L_envs_[idx], sz * sizeof(Scalar)));
        L_env_alloc_chi_[idx] = chi;
    }
}

template<typename Scalar>
void DMRGGPU<Scalar>::ensure_R_env_alloc(int idx, int chi) {
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
void DMRGGPU<Scalar>::initialize_mps_random(double scale) {
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
void DMRGGPU<Scalar>::initialize_mps_product() {
    for (int i = 0; i < L_; i++) {
        int cL = chi_L(i), cR = chi_R(i);
        int size = cL * d_ * cR;
        std::vector<Scalar> h_A(size, Traits::zero());
        int chi_min = std::min(cL, cR);
        for (int a = 0; a < chi_min; a++) {
            h_A[a + 0*cL + a*cL*d_] = Traits::one();
        }
        HIP_CHECK(hipMemcpy(d_mps_tensors_[i], h_A.data(),
                            size * sizeof(Scalar), hipMemcpyHostToDevice));
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
        HIP_CHECK(hipMemcpy(d_mps_tensors_[i], h_A.data(),
                            size * sizeof(Scalar), hipMemcpyHostToDevice));
    }
}

template<typename Scalar>
void DMRGGPU<Scalar>::set_mpo(const std::vector<Scalar*>& h_mpo_tensors) {
    int D = D_mpo_, d = d_;
    for (int i = 0; i < L_; i++) {
        int size = D * d * d * D;
        HIP_CHECK(hipMalloc(&d_mpo_tensors_[i], size * sizeof(Scalar)));
        HIP_CHECK(hipMemcpy(d_mpo_tensors_[i], h_mpo_tensors[i],
                            size * sizeof(Scalar), hipMemcpyHostToDevice));

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
        HIP_CHECK(hipMalloc(&d_W_left_[i], wm_size * sizeof(Scalar)));
        HIP_CHECK(hipMemcpy(d_W_left_[i], h_WL.data(),
                            wm_size * sizeof(Scalar), hipMemcpyHostToDevice));
        HIP_CHECK(hipMalloc(&d_W_right_[i], wm_size * sizeof(Scalar)));
        HIP_CHECK(hipMemcpy(d_W_right_[i], h_WR.data(),
                            wm_size * sizeof(Scalar), hipMemcpyHostToDevice));
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
        HIP_CHECK(hipMemcpyAsync(d_batch_A_, h_A, D*d*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
        HIP_CHECK(hipMemcpyAsync(d_batch_B_, h_B, D*d*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
        HIP_CHECK(hipMemcpyAsync(d_batch_C_, h_C, D*d*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
        ROCBLAS_CHECK(Traits::gemm_batched(rocblas_h_,
            Traits::op_t, rocblas_operation_none,
            cL, cR, cL,
            &one,
            (const Scalar**)d_batch_A_, cL * D,
            (const Scalar**)d_batch_B_, cL * d,
            &zero_val,
            d_batch_C_, cL,
            D * d));
    }

    // Step 2: U = V * W_matrix
    ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
        rocblas_operation_none, rocblas_operation_none,
        cL * cR, d * D, D * d,
        &one,
        V, cL * cR,
        W_mat, D * d,
        &zero_val,
        U, cL * cR));

    // Step 3: result_s'[a',b'] = sum_w' U_{w'd+s'}[a',b] * R_w'[b,b']
    for (int wp = 0; wp < D; wp++) {
        Scalar beta = (wp == 0) ? Traits::zero() : Traits::one();
        for (int sp = 0; sp < d; sp++) {
            int ws_out = wp * d + sp;
            ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
                rocblas_operation_none, rocblas_operation_none,
                cL, cR, cR,
                &one,
                U + ws_out * cL * cR, cL,
                R_env + wp * cR, cR * D,
                &beta,
                d_result + sp * cL, cL * d));
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
        HIP_CHECK(hipMemcpyAsync(d_batch_A_, h_A, D*d*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
        HIP_CHECK(hipMemcpyAsync(d_batch_B_, h_B, D*d*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
        HIP_CHECK(hipMemcpyAsync(d_batch_C_, h_C, D*d*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
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
    //
    // gemm(op_h, N, U, A) computes U^H * A = conj(desired result) for complex.
    // For real, U^T * A is correct directly.
    // For complex, we conjugate L_new after the loop.
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
        HIP_CHECK(hipMemcpyAsync(d_batch_A_, h_A, D*d*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
        HIP_CHECK(hipMemcpyAsync(d_batch_B_, h_B, D*d*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
        HIP_CHECK(hipMemcpyAsync(d_batch_C_, h_C, D*d*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
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
    // For complex: use conjugate transpose (op_h) on A for <bra|
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
void DMRGGPU<Scalar>::build_initial_environments() {
    // L[0] = trivial left boundary: (1, D_mpo, 1), L[0][0,0,0] = 1
    {
        std::vector<Scalar> h_L(D_mpo_, Traits::zero());
        h_L[0] = Traits::one();
        HIP_CHECK(hipMemcpy(d_L_envs_[0], h_L.data(),
                            D_mpo_ * sizeof(Scalar), hipMemcpyHostToDevice));
    }

    // R[L] = trivial right boundary: (1, D_mpo, 1), R[L][0,D-1,0] = 1
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
// Theta formation and Lanczos
// ============================================================================

template<typename Scalar>
void DMRGGPU<Scalar>::form_theta(int site, Scalar* d_theta) {
    int size = chi_L(site) * d_ * chi_R(site);
    HIP_CHECK(hipMemcpy(d_theta, d_mps_tensors_[site],
                        size * sizeof(Scalar), hipMemcpyDeviceToDevice));
}

template<typename Scalar>
double DMRGGPU<Scalar>::lanczos_eigensolver(int site, Scalar* d_theta) {
    int n = chi_L(site) * d_ * chi_R(site);
    int max_iter = std::min(max_lanczos_iter_, n);
    double tol_lanczos = 1e-12;
    double tol_eig_conv = 1e-12;

    Scalar* d_lanczos_v = d_lanczos_v_;

    // Alpha and beta are always real for Hermitian operators
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
        Scalar* d_vi = d_lanczos_v + iter * n;

        // w = H|v_i>
        apply_heff(site, d_vi, d_heff_result_);

        // alpha_i = <v_i|w> (real for Hermitian H)
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
            Scalar* d_vim1 = d_lanczos_v + (iter - 1) * n;
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
            Scalar* d_vip1 = d_lanczos_v + (iter + 1) * n;
            double scale = 1.0 / beta_i;
            HIP_CHECK(hipMemcpy(d_vip1, d_heff_result_, n * sizeof(Scalar), hipMemcpyDeviceToDevice));
            ROCBLAS_CHECK(Traits::scal_real(rocblas_h_, n, &scale, d_vip1, 1));
        }
    }

    int niter = iter;
    prof_lanczos_iters += niter;
    prof_heff_calls += niter;

    // Solve tridiagonal eigenvalue problem on CPU (always real)
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
    // Ritz coefficients are real (from dstev); convert to Scalar for gemv
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

    Scalar* h_U_data;
    RealType* h_S_data;
    Scalar* h_Vh_data;

    if (use_cpu_svd_) {
        // === CPU SVD path ===
        HIP_CHECK(hipMemcpy(h_svd_A_.data(), d_theta, m * n_svd * sizeof(Scalar), hipMemcpyDeviceToHost));

        int lwork = (int)h_svd_work_.size();
        int info;
        const char jobu = 'S', jobvt = 'S';
        Traits::lapack_gesvd(&jobu, &jobvt, &m, &n_svd, h_svd_A_.data(), &m, h_svd_S_.data(),
                h_svd_U_.data(), &m, h_svd_Vh_.data(), &full_k,
                h_svd_work_.data(), &lwork,
                h_svd_rwork_.empty() ? nullptr : h_svd_rwork_.data(), &info);

        h_U_data = h_svd_U_.data();
        h_S_data = h_svd_S_.data();
        h_Vh_data = h_svd_Vh_.data();
    } else {
        // === GPU SVD path ===
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

        HIP_CHECK(hipMemcpy(h_svd_U_.data(), d_svd_U_, m * full_k * sizeof(Scalar), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_svd_S_.data(), d_svd_S_, full_k * sizeof(RealType), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_svd_Vh_.data(), d_svd_Vh_, full_k * n_svd * sizeof(Scalar), hipMemcpyDeviceToHost));

        h_U_data = h_svd_U_.data();
        h_S_data = h_svd_S_.data();
        h_Vh_data = h_svd_Vh_.data();
    }

    // Truncation (CPU, tiny loop)
    int new_k = k;
    for (int i = 0; i < new_k; i++) {
        if (h_S_data[i] < 1e-14) { new_k = i; break; }
    }
    if (new_k == 0) new_k = 1;

    if (direction == 'R') {
        int new_chi_R = new_k;

        // Compute S*Vh on CPU
        for (int j = 0; j < n_svd; j++)
            for (int i = 0; i < new_k; i++)
                h_svd_tmp_[i + j * new_k] = Traits::scale_by_real(h_S_data[i], h_Vh_data[i + j * full_k]);

        // Upload U[:, :new_k]
        allocate_mps_tensor(site, cL, new_chi_R);
        HIP_CHECK(hipMemcpy(d_mps_tensors_[site], h_U_data,
                            m * new_chi_R * sizeof(Scalar), hipMemcpyHostToDevice));

        // Absorb S*Vh into A[site+1]
        if (site + 1 < L_) {
            HIP_CHECK(hipMemcpy(d_svd_work_, h_svd_tmp_.data(),
                                new_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice));
            int next_cR = chi_R(site + 1);
            Scalar one = Traits::one(), zero_val = Traits::zero();
            ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
                rocblas_operation_none, rocblas_operation_none,
                new_k, d_ * next_cR, cR, &one,
                d_svd_work_, new_k,
                d_mps_tensors_[site + 1], cR, &zero_val,
                d_T1_, new_k));
            allocate_mps_tensor(site + 1, new_chi_R, next_cR);
            HIP_CHECK(hipMemcpy(d_mps_tensors_[site + 1], d_T1_,
                                new_k * d_ * next_cR * sizeof(Scalar), hipMemcpyDeviceToDevice));
        }
        bond_dims_[site + 1] = new_chi_R;

    } else {  // direction == 'L'
        int new_chi_L = new_k;

        // Upload Vh[:new_k, :]
        allocate_mps_tensor(site, new_chi_L, cR);
        if (new_chi_L == full_k) {
            HIP_CHECK(hipMemcpy(d_mps_tensors_[site], h_Vh_data,
                                full_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice));
        } else {
            for (int j = 0; j < n_svd; j++)
                for (int i = 0; i < new_chi_L; i++)
                    h_svd_tmp_[i + j * new_chi_L] = h_Vh_data[i + j * full_k];
            HIP_CHECK(hipMemcpy(d_mps_tensors_[site], h_svd_tmp_.data(),
                                new_chi_L * n_svd * sizeof(Scalar), hipMemcpyHostToDevice));
        }

        // Compute U*S on CPU
        for (int j = 0; j < new_k; j++)
            for (int i = 0; i < m; i++)
                h_svd_tmp_[i + j * m] = Traits::scale_by_real(h_S_data[j], h_U_data[i + j * m]);

        // Absorb U*S into A[site-1]
        if (site > 0) {
            HIP_CHECK(hipMemcpy(d_svd_work_, h_svd_tmp_.data(),
                                m * new_k * sizeof(Scalar), hipMemcpyHostToDevice));
            int prev_cL = chi_L(site - 1);
            Scalar one = Traits::one(), zero_val = Traits::zero();
            ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
                rocblas_operation_none, rocblas_operation_none,
                prev_cL * d_, new_k, m, &one,
                d_mps_tensors_[site - 1], prev_cL * d_,
                d_svd_work_, m, &zero_val,
                d_T1_, prev_cL * d_));
            allocate_mps_tensor(site - 1, prev_cL, new_chi_L);
            HIP_CHECK(hipMemcpy(d_mps_tensors_[site - 1], d_T1_,
                                prev_cL * d_ * new_k * sizeof(Scalar), hipMemcpyDeviceToDevice));
        }
        bond_dims_[site] = new_chi_L;
    }
}

// ============================================================================
// Randomized truncated SVD (Halko-Martinsson-Tropp) and MPS update
// ============================================================================

template<typename Scalar>
void DMRGGPU<Scalar>::rsvd_and_update_mps(int site, Scalar* d_theta, char direction) {
    int cL = chi_L(site);
    int cR = chi_R(site);

    int m, n_svd;
    if (direction == 'R') { m = cL * d_; n_svd = cR; }
    else                  { m = cL;      n_svd = d_ * cR; }
    int full_k = std::min(m, n_svd);
    int k = std::min(full_k, chi_max_);

    // If matrix is small enough, fall back to full SVD (rSVD overhead not worth it)
    if (full_k <= k + rsvd_oversampling_ || m <= 2 * k) {
        svd_and_update_mps(site, d_theta, direction);
        return;
    }

    int r = k + rsvd_oversampling_;  // projection rank

    // Step 1: Generate random Omega (n_svd x r) on CPU and upload
    {
        std::vector<Scalar> h_omega(n_svd * r);
        for (int i = 0; i < n_svd * r; i++) {
            h_omega[i] = Traits::random_val();
        }
        HIP_CHECK(hipMemcpy(d_rsvd_omega_, h_omega.data(),
                            n_svd * r * sizeof(Scalar), hipMemcpyHostToDevice));
    }

    // Step 2: Y = theta @ Omega on GPU  (m x n_svd) @ (n_svd x r) -> (m x r)
    {
        Scalar one = Traits::one(), zero_val = Traits::zero();
        ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
            rocblas_operation_none, rocblas_operation_none,
            m, r, n_svd, &one,
            d_theta, m,
            d_rsvd_omega_, n_svd,
            &zero_val,
            d_rsvd_Y_, m));
        HIP_CHECK(hipStreamSynchronize(stream_));
    }

    // Step 3: QR factorization of Y on GPU -> Q (m x r) stays on device
    HIP_CHECK(hipMemcpy(d_rsvd_Q_, d_rsvd_Y_, (size_t)m * r * sizeof(Scalar), hipMemcpyDeviceToDevice));
    ROCBLAS_CHECK(Traits::rocsolver_geqrf(rocblas_h_, m, r, d_rsvd_Q_, m, d_rsvd_ipiv_));
    ROCBLAS_CHECK(Traits::rocsolver_orgqr(rocblas_h_, m, r, r, d_rsvd_Q_, m, d_rsvd_ipiv_));
    HIP_CHECK(hipStreamSynchronize(stream_));

    // Step 4: B = Q^H @ theta on GPU  (r x m) @ (m x n_svd) -> (r x n_svd)
    {
        Scalar one = Traits::one(), zero_val = Traits::zero();
        ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
            Traits::op_h, rocblas_operation_none,
            r, n_svd, m, &one,
            d_rsvd_Q_, m,
            d_theta, m,
            &zero_val,
            d_rsvd_B_, r));
        HIP_CHECK(hipStreamSynchronize(stream_));
    }

    // Step 5: Copy B to host, compute SVD of B (r x n_svd) -- much smaller than (m x n_svd)
    HIP_CHECK(hipMemcpy(h_rsvd_B_.data(), d_rsvd_B_, r * n_svd * sizeof(Scalar), hipMemcpyDeviceToHost));

    int small_k = std::min(r, n_svd);
    {
        int lwork = (int)h_svd_work_.size();
        int info;
        const char jobu = 'S', jobvt = 'S';
        // U_small: (r x small_k), S: (small_k), Vh: (small_k x n_svd)
        Traits::lapack_gesvd(&jobu, &jobvt, &r, &n_svd, h_rsvd_B_.data(), &r, h_svd_S_.data(),
                h_rsvd_U_small_.data(), &r, h_svd_Vh_.data(), &small_k,
                h_svd_work_.data(), &lwork,
                h_svd_rwork_.empty() ? nullptr : h_svd_rwork_.data(), &info);
        if (info != 0) {
            // Fall back to full SVD on failure
            svd_and_update_mps(site, d_theta, direction);
            return;
        }
    }

    // Step 6: Upload U_small to GPU, compute U_full = Q @ U_small on GPU
    //   Q is (m x r) on device, U_small is (r x small_k) on host -> U_full (m x small_k) on device
    {
        // Upload U_small to device (reuse d_rsvd_B_ as temp -- it's no longer needed)
        HIP_CHECK(hipMemcpy(d_rsvd_B_, h_rsvd_U_small_.data(),
                            (size_t)r * small_k * sizeof(Scalar), hipMemcpyHostToDevice));
        Scalar one = Traits::one(), zero_val = Traits::zero();
        ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
            rocblas_operation_none, rocblas_operation_none,
            m, small_k, r, &one,
            d_rsvd_Q_, m,
            d_rsvd_B_, r,
            &zero_val,
            d_rsvd_U_full_, m));
        HIP_CHECK(hipStreamSynchronize(stream_));
    }

    // Now: U_full (m x small_k) is on GPU at d_rsvd_U_full_
    //      S (small_k) and Vh (small_k x n_svd) are on host
    RealType* h_S_data = h_svd_S_.data();
    Scalar* h_Vh_data = h_svd_Vh_.data();

    // Truncation
    int new_k = k;
    for (int i = 0; i < new_k; i++) {
        if (h_S_data[i] < 1e-14) { new_k = i; break; }
    }
    if (new_k == 0) new_k = 1;

    if (direction == 'R') {
        int new_chi_R = new_k;

        // MPS[site] = U_full[:, :new_k] -- from GPU
        allocate_mps_tensor(site, cL, new_chi_R);
        if (new_k == small_k) {
            HIP_CHECK(hipMemcpy(d_mps_tensors_[site], d_rsvd_U_full_,
                                (size_t)m * new_k * sizeof(Scalar), hipMemcpyDeviceToDevice));
        } else {
            // Need to copy column subset -- download, extract, re-upload
            // Column-major: first new_k columns are contiguous if new_k == full leading dim,
            // but here lda = m and we want first new_k cols -> m*new_k elements from start
            HIP_CHECK(hipMemcpy(d_mps_tensors_[site], d_rsvd_U_full_,
                                (size_t)m * new_k * sizeof(Scalar), hipMemcpyDeviceToDevice));
        }

        // Compute S*Vh on CPU and absorb into A[site+1]
        for (int j = 0; j < n_svd; j++)
            for (int i = 0; i < new_k; i++)
                h_svd_tmp_[i + j * new_k] = Traits::scale_by_real(h_S_data[i], h_Vh_data[i + j * small_k]);

        if (site + 1 < L_) {
            HIP_CHECK(hipMemcpy(d_svd_work_, h_svd_tmp_.data(),
                                new_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice));
            int next_cR = chi_R(site + 1);
            Scalar one = Traits::one(), zero_val = Traits::zero();
            ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
                rocblas_operation_none, rocblas_operation_none,
                new_k, d_ * next_cR, cR, &one,
                d_svd_work_, new_k,
                d_mps_tensors_[site + 1], cR, &zero_val,
                d_T1_, new_k));
            allocate_mps_tensor(site + 1, new_chi_R, next_cR);
            HIP_CHECK(hipMemcpy(d_mps_tensors_[site + 1], d_T1_,
                                new_k * d_ * next_cR * sizeof(Scalar), hipMemcpyDeviceToDevice));
        }
        bond_dims_[site + 1] = new_chi_R;

    } else {  // direction == 'L'
        int new_chi_L = new_k;

        // MPS[site] = Vh[:new_k, :] (from host)
        allocate_mps_tensor(site, new_chi_L, cR);
        if (new_k == small_k) {
            HIP_CHECK(hipMemcpy(d_mps_tensors_[site], h_Vh_data,
                                (size_t)small_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice));
        } else {
            for (int j = 0; j < n_svd; j++)
                for (int i = 0; i < new_chi_L; i++)
                    h_svd_tmp_[i + j * new_chi_L] = h_Vh_data[i + j * small_k];
            HIP_CHECK(hipMemcpy(d_mps_tensors_[site], h_svd_tmp_.data(),
                                new_chi_L * n_svd * sizeof(Scalar), hipMemcpyHostToDevice));
        }

        // U_full[:, :new_k] @ diag(S[:new_k]) -- download from GPU, scale, absorb into A[site-1]
        HIP_CHECK(hipMemcpy(h_svd_U_.data(), d_rsvd_U_full_,
                            (size_t)m * small_k * sizeof(Scalar), hipMemcpyDeviceToHost));
        Scalar* h_U_data = h_svd_U_.data();
        for (int j = 0; j < new_k; j++)
            for (int i = 0; i < m; i++)
                h_svd_tmp_[i + j * m] = Traits::scale_by_real(h_S_data[j], h_U_data[i + j * m]);

        if (site > 0) {
            HIP_CHECK(hipMemcpy(d_svd_work_, h_svd_tmp_.data(),
                                m * new_k * sizeof(Scalar), hipMemcpyHostToDevice));
            int prev_cL = chi_L(site - 1);
            Scalar one = Traits::one(), zero_val = Traits::zero();
            ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
                rocblas_operation_none, rocblas_operation_none,
                prev_cL * d_, new_k, m, &one,
                d_mps_tensors_[site - 1], prev_cL * d_,
                d_svd_work_, m, &zero_val,
                d_T1_, prev_cL * d_));
            allocate_mps_tensor(site - 1, prev_cL, new_chi_L);
            HIP_CHECK(hipMemcpy(d_mps_tensors_[site - 1], d_T1_,
                                prev_cL * d_ * new_k * sizeof(Scalar), hipMemcpyDeviceToDevice));
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

    HIP_CHECK(hipStreamSynchronize(stream_));
    auto t0 = std::chrono::high_resolution_clock::now();
    double energy = lanczos_eigensolver(site, d_theta_);
    HIP_CHECK(hipStreamSynchronize(stream_));
    auto t1 = std::chrono::high_resolution_clock::now();
    if (use_rsvd_)
        rsvd_and_update_mps(site, d_theta_, direction);
    else
        svd_and_update_mps(site, d_theta_, direction);
    HIP_CHECK(hipStreamSynchronize(stream_));
    auto t2 = std::chrono::high_resolution_clock::now();

    prof_lanczos_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    prof_svd_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
    prof_site_count++;
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
        HIP_CHECK(hipStreamSynchronize(stream_));
        auto te0 = std::chrono::high_resolution_clock::now();
        update_left_env(site);
        HIP_CHECK(hipStreamSynchronize(stream_));
        auto te1 = std::chrono::high_resolution_clock::now();
        prof_env_ms += std::chrono::duration<double, std::milli>(te1 - te0).count();
    }
    // Optimize last site without SVD
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
double DMRGGPU<Scalar>::sweep_right_to_left() {
    double energy = 0.0;

    for (int site = L_ - 1; site >= 1; site--) {
        energy = optimize_site(site, 'L');
        HIP_CHECK(hipStreamSynchronize(stream_));
        auto te0 = std::chrono::high_resolution_clock::now();
        update_right_env(site);
        HIP_CHECK(hipStreamSynchronize(stream_));
        auto te1 = std::chrono::high_resolution_clock::now();
        prof_env_ms += std::chrono::duration<double, std::milli>(te1 - te0).count();
    }
    // Optimize first site without SVD
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
double DMRGGPU<Scalar>::run(int n_sweeps) {
    const char* type_name = Traits::is_complex ? "complex128" : "float64";
    printf("=== GPU-Native DMRG (rocBLAS GEMM, %s) ===\n", type_name);
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
        prof_lanczos_ms = prof_svd_ms = prof_env_ms = 0;
        prof_lanczos_iters = prof_site_count = prof_heff_calls = 0;

        auto t_sweep = std::chrono::high_resolution_clock::now();

        double energy_LR = sweep_left_to_right();
        double energy_RL = sweep_right_to_left();

        auto t_sweep_end = std::chrono::high_resolution_clock::now();
        double sweep_time = std::chrono::duration<double>(t_sweep_end - t_sweep).count();

        energy_ = energy_RL;
        double dE = std::abs(energy_ - energy_prev);

        double other_ms = sweep_time*1000.0 - prof_lanczos_ms - prof_svd_ms - prof_env_ms;
        printf("Sweep %3d: E = %.12f, dE = %.2e, time = %.3f s\n",
               sweep, energy_, dE, sweep_time);
        printf("  Profile: lanczos=%.0fms (%d iters, %d heff) svd=%.0fms env=%.0fms other=%.0fms\n",
               prof_lanczos_ms, prof_lanczos_iters, prof_heff_calls,
               prof_svd_ms, prof_env_ms, other_ms);

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
void DMRGGPU<Scalar>::load_mps_from_file(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open MPS file: " + filename);

    int L_file, d_file;
    file.read(reinterpret_cast<char*>(&L_file), sizeof(int));
    file.read(reinterpret_cast<char*>(&d_file), sizeof(int));
    if (L_file != L_ || d_file != d_)
        throw std::runtime_error("MPS file dimensions don't match");

    std::vector<int> bond_dims_file(L_ + 1);
    file.read(reinterpret_cast<char*>(bond_dims_file.data()), (L_ + 1) * sizeof(int));
    bond_dims_ = bond_dims_file;

    for (int i = 0; i < L_; i++) {
        allocate_mps_tensor(i, chi_L(i), chi_R(i));
        int size = chi_L(i) * d_ * chi_R(i);
        std::vector<Scalar> h_A(size);
        file.read(reinterpret_cast<char*>(h_A.data()), size * sizeof(Scalar));
        HIP_CHECK(hipMemcpy(d_mps_tensors_[i], h_A.data(),
                            size * sizeof(Scalar), hipMemcpyHostToDevice));
    }
    file.close();
}

template<typename Scalar>
void DMRGGPU<Scalar>::get_mps(std::vector<std::vector<Scalar>>& h_mps) const {
    h_mps.resize(L_);
    for (int i = 0; i < L_; i++) {
        int size = chi_L(i) * d_ * chi_R(i);
        h_mps[i].resize(size);
        HIP_CHECK(hipMemcpy(h_mps[i].data(), d_mps_tensors_[i],
                            size * sizeof(Scalar), hipMemcpyDeviceToHost));
    }
}

#endif // DMRG_GPU_IMPL_H
