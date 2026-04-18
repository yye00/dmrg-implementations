#ifndef DMRG_GPU_IMPL_H
#define DMRG_GPU_IMPL_H

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


// ============================================================================
// GPU kernels for batched GEMM pointer setup (eliminates CPU→GPU pointer copies)
// ============================================================================

// Step 1/update_env: A[w*d+s] = base_A + w*strideA, B[w*d+s] = base_B + s*strideB, C[w*d+s] = base_C + (w*d+s)*strideC
template<typename Scalar>
__global__ void setup_batch_ptrs_wd(Scalar** A, Scalar** B, Scalar** C,
                                     Scalar* base_A, Scalar* base_B, Scalar* base_C,
                                     int d, int strideA, int strideB, int strideC) {
    int idx = threadIdx.x;  // idx = w*d + s
    int w = idx / d, s = idx % d;
    A[idx] = base_A + w * strideA;
    B[idx] = base_B + s * strideB;
    C[idx] = base_C + idx * strideC;
}

// Step 3 (apply_heff): A[s] = base_A + (wp*d+s)*strideA, B[s] = base_B + wp*strideB, C[s] = base_C + s*strideC
template<typename Scalar>
__global__ void setup_batch_ptrs_step3(Scalar** A, Scalar** B, Scalar** C,
                                        Scalar* base_A, Scalar* base_B, Scalar* base_C,
                                        int wp, int d, int strideA, int strideB, int strideC) {
    int s = threadIdx.x;
    A[s] = base_A + (wp * d + s) * strideA;
    B[s] = base_B + wp * strideB;
    C[s] = base_C + s * strideC;
}

// Step 3 (R3-F1 full batched): single launch writes all D*d pointers.
// idx = wp*d + sp. Each wp writes to its own scratch slice inside T1; slices
// are summed afterwards by a single rocblas_gemv reduction.
//   A[idx] -> U[wp*d + sp]        (cL x cR block, lda = cL*cR)
//   B[idx] -> R[wp]               (cR x cR block, ldb = cR*D)
//   C[idx] -> scratch + wp*slice_stride + sp*strideC_tile
//             (cL x cR tile inside the (cL, d, cR) slice, ldc = cL*d)
template<typename Scalar>
__global__ void setup_batch_ptrs_step3_full(Scalar** A, Scalar** B, Scalar** C,
                                             Scalar* base_A, Scalar* base_B, Scalar* base_C_scratch,
                                             int d, int strideA, int strideB,
                                             int strideC_tile, int slice_stride) {
    int idx = threadIdx.x;  // wp*d + sp
    int wp = idx / d;
    int sp = idx % d;
    A[idx] = base_A + (wp * d + sp) * strideA;
    B[idx] = base_B + wp * strideB;
    C[idx] = base_C_scratch + wp * slice_stride + sp * strideC_tile;
}

// Step 3 (env update): A[w] = base_A + (w*d+sp)*strideA, B[w] = base_B + sp*strideB, C[w] = base_C + w*strideC
template<typename Scalar>
__global__ void setup_batch_ptrs_env3(Scalar** A, Scalar** B, Scalar** C,
                                       Scalar* base_A, Scalar* base_B, Scalar* base_C,
                                       int sp, int d, int strideA, int strideB, int strideC) {
    int w = threadIdx.x;
    A[w] = base_A + (w * d + sp) * strideA;
    B[w] = base_B + sp * strideB;
    C[w] = base_C + w * strideC;
}

// Lanczos: find first beta < tol, write index+1 to result (0 = none found)
static __global__ void lanczos_check_beta(const double* beta, int n, double tol, rocblas_int* result) {
    *result = 0;
    for (int j = 0; j < n; j++) {
        if (beta[j] < tol) { *result = j + 1; return; }
    }
}

// Promote double eigenvector to hipDoubleComplex (for Josephson Ritz coefficients)
static __global__ void promote_double_to_complex(const double* src, hipDoubleComplex* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = make_hipDoubleComplex(src[i], 0.0);
}

// ============================================================================
// Constructor
// ============================================================================

template<typename Scalar>
DMRGGPU<Scalar>::DMRGGPU(int L, int d, int chi_max, int D_mpo, double tol)
    : L_(L), d_(d), chi_max_(chi_max), D_mpo_(D_mpo), tol_(tol), energy_(0.0) {

    opts_.load_from_env();
    opts_.print(stderr);
    init_timers();

    // Bond dimensions
    bond_dims_.resize(L + 1);
    bond_dims_[0] = 1;
    bond_dims_[L] = 1;
    for (int i = 1; i < L; i++) {
        double exact_dim = pow((double)d_, std::min(i, L - i));
        bond_dims_[i] = (exact_dim > chi_max_) ? chi_max_ : (int)exact_dim;
    }

    // GPU handles (main + env stream for forward/backward-sweep pipelining)
    HIP_CHECK(hipStreamCreate(&stream_));
    HIP_CHECK(hipStreamCreate(&stream_env_));
    ROCBLAS_CHECK(rocblas_create_handle(&rocblas_h_));
    ROCBLAS_CHECK(rocblas_set_stream(rocblas_h_, stream_));
    ROCBLAS_CHECK(rocblas_create_handle(&rocblas_h_env_));
    ROCBLAS_CHECK(rocblas_set_stream(rocblas_h_env_, stream_env_));
    HIP_CHECK(hipEventCreateWithFlags(&event_canon_ready_, hipEventDisableTiming));
    HIP_CHECK(hipEventCreateWithFlags(&event_env_done_, hipEventDisableTiming));

    // Contraction intermediates — disjoint buffers per stream
    int t_max = D_mpo_ * d_ * chi_max_ * chi_max_;
    HIP_CHECK(hipMalloc(&d_T1_, t_max * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_T2_, t_max * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_T1_env_, t_max * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_T2_env_, t_max * sizeof(Scalar)));

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

    // Device scalars for sync-free Lanczos
    HIP_CHECK(hipMalloc(&d_dot_result_, sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_nrm2_result_, sizeof(RealType)));
    HIP_CHECK(hipMalloc(&d_neg_alpha_, sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_neg_overlap_, sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_inv_nrm_, sizeof(RealType)));
    HIP_CHECK(hipMalloc(&d_alpha_dev_, max_lanczos_iter_ * sizeof(RealType)));
    HIP_CHECK(hipMalloc(&d_beta_dev_, max_lanczos_iter_ * sizeof(RealType)));
    HIP_CHECK(hipMalloc(&d_neg_beta_scalars_, max_lanczos_iter_ * sizeof(Scalar)));

    // rocsolver tridiagonal eigensolver workspace
    HIP_CHECK(hipMalloc(&d_steqr_D_, max_lanczos_iter_ * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_steqr_E_, max_lanczos_iter_ * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_steqr_C_, max_lanczos_iter_ * max_lanczos_iter_ * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_steqr_info_, sizeof(rocblas_int)));

    HIP_CHECK(hipMalloc(&d_const_one_, sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_const_zero_, sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_const_neg_one_, sizeof(Scalar)));
    {
        Scalar one = Traits::one(), zero = Traits::zero(), neg_one = Traits::neg(one);
        HIP_CHECK(hipMemcpy(d_const_one_, &one, sizeof(Scalar), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_const_zero_, &zero, sizeof(Scalar), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_const_neg_one_, &neg_one, sizeof(Scalar), hipMemcpyHostToDevice));
    }

    // Batched GEMM pointer arrays (main + env — separate so env_update and
    // apply_heff/absorb can run concurrently without clobbering each other's
    // pointer tables).
    int batch_max = D_mpo_ * d_;
    HIP_CHECK(hipMalloc(&d_batch_A_, batch_max * sizeof(Scalar*)));
    HIP_CHECK(hipMalloc(&d_batch_B_, batch_max * sizeof(Scalar*)));
    HIP_CHECK(hipMalloc(&d_batch_C_, batch_max * sizeof(Scalar*)));
    HIP_CHECK(hipMalloc(&d_batch_A_env_, batch_max * sizeof(Scalar*)));
    HIP_CHECK(hipMalloc(&d_batch_B_env_, batch_max * sizeof(Scalar*)));
    HIP_CHECK(hipMalloc(&d_batch_C_env_, batch_max * sizeof(Scalar*)));

    // Length-D ones vector for Step-3 GEMV reduction (R3-F1)
    HIP_CHECK(hipMalloc(&d_ones_D_, D_mpo_ * sizeof(Scalar)));
    {
        std::vector<Scalar> h_ones(D_mpo_, Traits::one());
        HIP_CHECK(hipMemcpy(d_ones_D_, h_ones.data(),
                            D_mpo_ * sizeof(Scalar), hipMemcpyHostToDevice));
    }

    // SVD workspace
    int svd_max_dim = chi_max_ * d_;
    HIP_CHECK(hipMalloc(&d_svd_A_,    theta_size_max_ * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_svd_U_,    svd_max_dim * chi_max_ * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_svd_S_,    chi_max_ * sizeof(RealType)));
    HIP_CHECK(hipMalloc(&d_svd_Vh_,   chi_max_ * svd_max_dim * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_svd_E_,    chi_max_ * sizeof(RealType)));
    HIP_CHECK(hipMalloc(&d_svd_info_, sizeof(int)));
    HIP_CHECK(hipMalloc(&d_svd_work_, theta_size_max_ * sizeof(Scalar)));
    // R3-F2: gesvdj requires device pointers for residual / n_sweeps.
    HIP_CHECK(hipMalloc(&d_svdj_residual_, sizeof(double)));
    HIP_CHECK(hipMalloc(&d_svdj_n_sweeps_, sizeof(rocblas_int)));

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
    if (d_dot_result_) hipFree(d_dot_result_);
    if (d_nrm2_result_) hipFree(d_nrm2_result_);
    if (d_neg_alpha_) hipFree(d_neg_alpha_);
    if (d_neg_overlap_) hipFree(d_neg_overlap_);
    if (d_inv_nrm_) hipFree(d_inv_nrm_);
    if (d_alpha_dev_) hipFree(d_alpha_dev_);
    if (d_beta_dev_) hipFree(d_beta_dev_);
    if (d_neg_beta_scalars_) hipFree(d_neg_beta_scalars_);
    if (d_steqr_D_) hipFree(d_steqr_D_);
    if (d_steqr_E_) hipFree(d_steqr_E_);
    if (d_steqr_C_) hipFree(d_steqr_C_);
    if (d_steqr_info_) hipFree(d_steqr_info_);
    if (d_const_one_) hipFree(d_const_one_);
    if (d_const_zero_) hipFree(d_const_zero_);
    if (d_const_neg_one_) hipFree(d_const_neg_one_);
    if (d_T1_) hipFree(d_T1_);
    if (d_T2_) hipFree(d_T2_);
    if (d_T1_env_) hipFree(d_T1_env_);
    if (d_T2_env_) hipFree(d_T2_env_);
    if (d_ones_D_) hipFree(d_ones_D_);
    if (d_batch_A_) hipFree(d_batch_A_);
    if (d_batch_B_) hipFree(d_batch_B_);
    if (d_batch_C_) hipFree(d_batch_C_);
    if (d_batch_A_env_) hipFree(d_batch_A_env_);
    if (d_batch_B_env_) hipFree(d_batch_B_env_);
    if (d_batch_C_env_) hipFree(d_batch_C_env_);
    if (d_svd_A_) hipFree(d_svd_A_);
    if (d_svd_U_) hipFree(d_svd_U_);
    if (d_svd_S_) hipFree(d_svd_S_);
    if (d_svd_Vh_) hipFree(d_svd_Vh_);
    if (d_svd_E_) hipFree(d_svd_E_);
    if (d_svd_info_) hipFree(d_svd_info_);
    if (d_svd_work_) hipFree(d_svd_work_);
    if (d_svdj_residual_) hipFree(d_svdj_residual_);
    if (d_svdj_n_sweeps_) hipFree(d_svdj_n_sweeps_);

    rocblas_destroy_handle(rocblas_h_);
    rocblas_destroy_handle(rocblas_h_env_);
    hipEventDestroy(event_canon_ready_);
    hipEventDestroy(event_env_done_);
    hipStreamDestroy(stream_);
    hipStreamDestroy(stream_env_);
}

// ============================================================================
// Phase timers
// ============================================================================

template<typename Scalar>
void DMRGGPU<Scalar>::init_timers() {
    t_lanczos_.init("lanczos", opts_.profile);
    t_apply_heff_.init("apply_heff", opts_.profile);
    t_svd_.init("svd", opts_.profile);
    t_absorb_.init("absorb", opts_.profile);
    t_env_update_.init("env_update", opts_.profile);
}

template<typename Scalar>
void DMRGGPU<Scalar>::report_timers() {
    if (!opts_.profile) return;
    auto row = [](PhaseTimer& t) {
        double ms = t.total_ms();
        int c = t.calls();
        double per = c > 0 ? ms / c : 0.0;
        std::fprintf(stderr, "  %-12s: %10.2f ms   (%6d calls, %8.3f ms/call)\n",
                     t.name, ms, c, per);
    };
    std::fprintf(stderr, "== Phase timings ==\n");
    row(t_lanczos_);
    row(t_apply_heff_);
    row(t_svd_);
    row(t_absorb_);
    row(t_env_update_);
}

// ============================================================================
// Memory management
// ============================================================================

template<typename Scalar>
void DMRGGPU<Scalar>::allocate_mps_tensor(int site, int cL, int cR) {
    // Pre-allocate at chi_max to avoid hipFree/hipMalloc per bond (sync points)
    size_t needed = (size_t)cL * d_ * cR * sizeof(Scalar);
    size_t max_sz = (size_t)chi_max_ * d_ * chi_max_ * sizeof(Scalar);
    if (!d_mps_tensors_[site]) {
        HIP_CHECK(hipMalloc(&d_mps_tensors_[site], max_sz));
    }
    (void)needed;  // logical size tracked by bond_dims_
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
    t_apply_heff_.begin(stream_);
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
        hipLaunchKernelGGL(setup_batch_ptrs_wd<Scalar>, dim3(1), dim3(D*d), 0, stream_,
                           d_batch_A_, d_batch_B_, d_batch_C_,
                           L_env, const_cast<Scalar*>(d_theta_in), V,
                           d, cL, cL, cL * cR);
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

    // Step 3 (R3-F1 full-batched collapse): one batched GEMM over all D*d
    // tiles, writing to per-wp scratch slices in T1, then a GEMV reduction
    // summing the D slices into d_result.
    //
    // Old path: D sequential batches of d GEMMs (2*D launches) with same-C
    //   accumulation across wp.
    // New path: 1 setup kernel + 1 batched GEMM (batch = D*d) + 1 GEMV.
    //
    // T1 (V) is free here (consumed by Step 2 into U=T2) and its allocation
    //   (D_mpo * d * chi_max^2) exactly fits D slices of (cL, d, cR).
    {
        int batch_count = D * d;
        int slice_stride = cL * d * cR;   // per-wp slice size inside V scratch

        hipLaunchKernelGGL(setup_batch_ptrs_step3_full<Scalar>,
                           dim3(1), dim3(batch_count), 0, stream_,
                           d_batch_A_, d_batch_B_, d_batch_C_,
                           U, R_env, V,
                           d, cL * cR, cR, cL, slice_stride);

        ROCBLAS_CHECK(Traits::gemm_batched(rocblas_h_,
            rocblas_operation_none, rocblas_operation_none,
            cL, cR, cR,
            &one,
            (const Scalar**)d_batch_A_, cL,
            (const Scalar**)d_batch_B_, cR * D,
            &zero_val,
            d_batch_C_, cL * d,
            batch_count));

        // Reduce: d_result[slice_stride] = V[slice_stride x D] * ones_D
        ROCBLAS_CHECK(Traits::gemv(rocblas_h_,
            rocblas_operation_none,
            slice_stride, D,
            &one,
            V, slice_stride,
            d_ones_D_, 1,
            &zero_val,
            d_result, 1));
    }
    t_apply_heff_.end(stream_);
}

// ============================================================================
// Left environment update
// ============================================================================

template<typename Scalar>
void DMRGGPU<Scalar>::update_left_env(int site) {
    // Runs on stream_env_ (side stream) using env-dedicated scratch buffers so
    // it can overlap with the absorb(S*Vh) GEMM running on stream_. The caller
    // is responsible for making stream_env_ wait on event_canon_ready_ so that
    // MPS[site] = U has been written before this runs.
    t_env_update_.begin(stream_env_);
    int chi_in = bond_dims_[site];
    int chi_out = bond_dims_[site + 1];
    int D = D_mpo_, d = d_;
    Scalar one = Traits::one(), zero_val = Traits::zero();

    ensure_L_env_alloc(site + 1, chi_out);

    Scalar* L_env = d_L_envs_[site];
    Scalar* A = d_mps_tensors_[site];
    Scalar* W_mat = d_W_left_[site];
    Scalar* L_new = d_L_envs_[site + 1];
    Scalar* V = d_T1_env_;
    Scalar* U = d_T2_env_;

    // Step 1: V_ws[a',b] = L_w^T[a',a] * A_s[a,b]  (batched GEMM)
    {
        hipLaunchKernelGGL(setup_batch_ptrs_wd<Scalar>, dim3(1), dim3(D*d), 0, stream_env_,
                           d_batch_A_env_, d_batch_B_env_, d_batch_C_env_,
                           L_env, A, V,
                           d, chi_in, chi_in, chi_in * chi_out);
        ROCBLAS_CHECK(Traits::gemm_batched(rocblas_h_env_,
            Traits::op_t, rocblas_operation_none,
            chi_in, chi_out, chi_in,
            &one,
            (const Scalar**)d_batch_A_env_, chi_in * D,
            (const Scalar**)d_batch_B_env_, chi_in * d,
            &zero_val,
            d_batch_C_env_, chi_in,
            D * d));
    }

    // Step 2: U = V * W_matrix
    ROCBLAS_CHECK(Traits::gemm(rocblas_h_env_,
        rocblas_operation_none, rocblas_operation_none,
        chi_in * chi_out, d * D, D * d,
        &one,
        V, chi_in * chi_out,
        W_mat, D * d,
        &zero_val,
        U, chi_in * chi_out));

    // Step 3: L_new_w'[b,b'] = sum_{a',s'} U[a',ws',b] * conj(A[a',s',b'])  (batched)
    {
        for (int sp = 0; sp < d; sp++) {
            Scalar beta = (sp == 0) ? Traits::zero() : Traits::one();
            hipLaunchKernelGGL(setup_batch_ptrs_env3<Scalar>, dim3(1), dim3(D), 0, stream_env_,
                               d_batch_A_env_, d_batch_B_env_, d_batch_C_env_,
                               U, A, L_new,
                               sp, d, chi_in * chi_out, chi_in, chi_out);
            ROCBLAS_CHECK(Traits::gemm_batched(rocblas_h_env_,
                Traits::op_h, rocblas_operation_none,
                chi_out, chi_out, chi_in,
                &one,
                (const Scalar**)d_batch_A_env_, chi_in,
                (const Scalar**)d_batch_B_env_, chi_in * d,
                &beta,
                d_batch_C_env_, chi_out * D,
                D));
        }
    }

    if constexpr (Traits::is_complex) {
        conjugate_inplace(L_new, chi_out * D * chi_out, stream_env_);
    }
    t_env_update_.end(stream_env_);
}

// ============================================================================
// Right environment update
// ============================================================================

template<typename Scalar>
void DMRGGPU<Scalar>::update_right_env(int site) {
    // Runs on stream_env_ (see update_left_env).
    t_env_update_.begin(stream_env_);
    int chi_in = bond_dims_[site + 1];
    int chi_out = bond_dims_[site];
    int D = D_mpo_, d = d_;
    Scalar one = Traits::one(), zero_val = Traits::zero();

    ensure_R_env_alloc(site, chi_out);

    Scalar* A = d_mps_tensors_[site];
    Scalar* R_env = d_R_envs_[site + 1];
    Scalar* W_mat = d_W_right_[site];
    Scalar* R_new = d_R_envs_[site];
    Scalar* V = d_T1_env_;
    Scalar* U = d_T2_env_;

    // Step 1: V_ws[a,b'] = A_s[a,b] * R_w'[b,b']  (batched GEMM)
    // Note: A varies by s (inner dim) and B varies by w (outer dim) — opposite of
    // setup_batch_ptrs_wd convention, so swap output arrays and base/stride args.
    {
        hipLaunchKernelGGL(setup_batch_ptrs_wd<Scalar>, dim3(1), dim3(D*d), 0, stream_env_,
                           d_batch_B_env_, d_batch_A_env_, d_batch_C_env_,
                           R_env, A, V,
                           d, chi_in, chi_out, chi_out * chi_in);
        ROCBLAS_CHECK(Traits::gemm_batched(rocblas_h_env_,
            rocblas_operation_none, rocblas_operation_none,
            chi_out, chi_in, chi_in,
            &one,
            (const Scalar**)d_batch_A_env_, chi_out * d,
            (const Scalar**)d_batch_B_env_, chi_in * D,
            &zero_val,
            d_batch_C_env_, chi_out,
            D * d));
    }

    // Step 2: U = V * W_matrix
    ROCBLAS_CHECK(Traits::gemm(rocblas_h_env_,
        rocblas_operation_none, rocblas_operation_none,
        chi_out * chi_in, d * D, D * d,
        &one,
        V, chi_out * chi_in,
        W_mat, D * d,
        &zero_val,
        U, chi_out * chi_in));

    // Step 3: R_new_w[a,a'] = sum_s' U_ws'[a,b'] * A_s'^H[b',a']  (batched)
    {
        for (int sp = 0; sp < d; sp++) {
            Scalar beta = (sp == 0) ? Traits::zero() : Traits::one();
            hipLaunchKernelGGL(setup_batch_ptrs_env3<Scalar>, dim3(1), dim3(D), 0, stream_env_,
                               d_batch_A_env_, d_batch_B_env_, d_batch_C_env_,
                               U, A, R_new,
                               sp, d, chi_out * chi_in, chi_out, chi_out);
            ROCBLAS_CHECK(Traits::gemm_batched(rocblas_h_env_,
                rocblas_operation_none, Traits::op_h,
                chi_out, chi_out, chi_in,
                &one,
                (const Scalar**)d_batch_A_env_, chi_out,
                (const Scalar**)d_batch_B_env_, chi_out * d,
                &beta,
                d_batch_C_env_, chi_out * D,
                D));
        }
    }
    t_env_update_.end(stream_env_);
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

    // Build all R environments from right to left (runs on stream_env_)
    for (int i = L_ - 1; i >= 0; i--) {
        update_right_env(i);
    }
    // Ensure initial envs are visible to the first sweep on stream_.
    HIP_CHECK(hipStreamSynchronize(stream_env_));
}

// ============================================================================
// Theta formation and Lanczos
// ============================================================================

template<typename Scalar>
void DMRGGPU<Scalar>::form_theta(int site, Scalar* d_theta) {
    int size = chi_L(site) * d_ * chi_R(site);
    HIP_CHECK(hipMemcpyAsync(d_theta, d_mps_tensors_[site],
                             size * sizeof(Scalar), hipMemcpyDeviceToDevice, stream_));
}

template<typename Scalar>
double DMRGGPU<Scalar>::lanczos_eigensolver(int site, Scalar* d_theta) {
    t_lanczos_.begin(stream_);
    int n = chi_L(site) * d_ * chi_R(site);
    int max_iter = std::min(max_lanczos_iter_, n);
    double tol_lanczos = 1e-12;
    double tol_eig_conv = 1e-12;

    Scalar* d_lanczos_v = d_lanczos_v_;

    // v[0] = theta / ||theta|| (host pointer mode for initial setup)
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

        // w = H|v_i> (apply_heff uses host pointer mode internally)
        apply_heff(site, d_vi, d_heff_result_);

        // Switch to device pointer mode for scalar operations
        ROCBLAS_CHECK(rocblas_set_pointer_mode(rocblas_h_, rocblas_pointer_mode_device));

        // alpha_i = <v_i|w> → device
        ROCBLAS_CHECK(Traits::dot(rocblas_h_, n, d_vi, 1, d_heff_result_, 1, d_dot_result_));

        // Process alpha: store to d_alpha_dev_[iter], compute d_neg_alpha_
        hipLaunchKernelGGL(lanczos_process_alpha_kernel<Scalar>, dim3(1), dim3(1), 0, stream_,
                           d_dot_result_, d_neg_alpha_, d_alpha_dev_, iter);

        // w -= alpha_i * v_i (device pointer)
        ROCBLAS_CHECK(Traits::axpy(rocblas_h_, n, d_neg_alpha_, d_vi, 1, d_heff_result_, 1));

        // w -= beta_{i-1} * v_{i-1} (device pointer)
        if (iter > 0) {
            ROCBLAS_CHECK(Traits::axpy(rocblas_h_, n,
                d_neg_beta_scalars_ + (iter - 1),
                d_lanczos_v + (size_t)(iter - 1) * n, 1,
                d_heff_result_, 1));
        }

        // Full reorthogonalization (device pointer mode for gemv constants)
        if (iter > 0) {
            ROCBLAS_CHECK(Traits::gemv(rocblas_h_, Traits::op_h,
                n, iter + 1, d_const_one_,
                d_lanczos_v, n,
                d_heff_result_, 1,
                d_const_zero_, d_ritz_coeffs_, 1));
            ROCBLAS_CHECK(Traits::gemv(rocblas_h_, rocblas_operation_none,
                n, iter + 1, d_const_neg_one_,
                d_lanczos_v, n,
                d_ritz_coeffs_, 1,
                d_const_one_, d_heff_result_, 1));
        } else {
            ROCBLAS_CHECK(Traits::dot(rocblas_h_, n, d_lanczos_v, 1, d_heff_result_, 1, d_dot_result_));
            hipLaunchKernelGGL(negate_scalar_kernel<Scalar>, dim3(1), dim3(1), 0, stream_,
                               d_dot_result_, d_neg_overlap_);
            ROCBLAS_CHECK(Traits::axpy(rocblas_h_, n, d_neg_overlap_, d_lanczos_v, 1, d_heff_result_, 1));
        }

        // beta_i = ||w|| → device
        ROCBLAS_CHECK(Traits::nrm2(rocblas_h_, n, d_heff_result_, 1, d_nrm2_result_));

        // Process beta: store, compute 1/beta, store -beta as Scalar
        hipLaunchKernelGGL(lanczos_process_beta_kernel<Scalar>, dim3(1), dim3(1), 0, stream_,
                           d_nrm2_result_, d_inv_nrm_, d_beta_dev_, d_neg_beta_scalars_, iter);

        // v_{i+1} = w / beta_i (device pointer for scal)
        if (iter + 1 < max_iter) {
            Scalar* d_vip1 = d_lanczos_v + (size_t)(iter + 1) * n;
            HIP_CHECK(hipMemcpy(d_vip1, d_heff_result_, n * sizeof(Scalar), hipMemcpyDeviceToDevice));
            ROCBLAS_CHECK(Traits::scal_real(rocblas_h_, n, d_inv_nrm_, d_vip1, 1));
        }

        // Switch back to host pointer mode
        ROCBLAS_CHECK(rocblas_set_pointer_mode(rocblas_h_, rocblas_pointer_mode_host));

        // Convergence check every 3 iterations after iter >= 4.
        // LANCZOS_FIXED skips the check entirely → no mid-loop host syncs,
        // runs the full max_iter iterations. Cost: ~2-3× more apply_heff per
        // site if Lanczos converges early. Benefit: eliminates stream drain
        // + D2H readback on every third iter, which dominates at small n.
        if (!opts_.lanczos_fixed && iter >= 4 && iter % 3 == 0) {
            HIP_CHECK(hipStreamSynchronize(stream_));

            int ncheck = iter + 1;

            // Check for invariant subspace (beta < tol) on GPU
            hipLaunchKernelGGL(lanczos_check_beta, dim3(1), dim3(1), 0, stream_,
                               d_beta_dev_, ncheck, tol_lanczos, d_steqr_info_);
            rocblas_int h_beta_idx;
            HIP_CHECK(hipMemcpy(&h_beta_idx, d_steqr_info_, sizeof(rocblas_int), hipMemcpyDeviceToHost));
            if (h_beta_idx > 0) { iter = h_beta_idx; break; }
            // Copy alpha/beta to scratch buffers on device (steqr overwrites them)
            HIP_CHECK(hipMemcpyAsync(d_steqr_D_, d_alpha_dev_, ncheck * sizeof(double), hipMemcpyDeviceToDevice, stream_));
            HIP_CHECK(hipMemcpyAsync(d_steqr_E_, d_beta_dev_, ncheck * sizeof(double), hipMemcpyDeviceToDevice, stream_));
            // Eigenvalues only — no eigenvectors needed for convergence check
            rocsolver_dsteqr(rocblas_h_, rocblas_evect_none, ncheck,
                             d_steqr_D_, d_steqr_E_, nullptr, ncheck, d_steqr_info_);
            // Read smallest eigenvalue back
            rocblas_int h_info_chk;
            double cur_energy;
            HIP_CHECK(hipMemcpy(&h_info_chk, d_steqr_info_, sizeof(rocblas_int), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(&cur_energy, d_steqr_D_, sizeof(double), hipMemcpyDeviceToHost));
            if (h_info_chk == 0) {
                if (std::abs(cur_energy - prev_energy) < tol_eig_conv) {
                    iter++;
                    break;
                }
                prev_energy = cur_energy;
            }
        }
    }

    int niter = iter;

    HIP_CHECK(hipStreamSynchronize(stream_));

    // Solve tridiagonal eigenvalue problem on GPU via rocsolver
    HIP_CHECK(hipMemcpyAsync(d_steqr_D_, d_alpha_dev_, niter * sizeof(double), hipMemcpyDeviceToDevice, stream_));
    HIP_CHECK(hipMemcpyAsync(d_steqr_E_, d_beta_dev_, niter * sizeof(double), hipMemcpyDeviceToDevice, stream_));
    // C is initialized to identity by rocsolver when evect = tridiagonal
    rocsolver_dsteqr(rocblas_h_, rocblas_evect_tridiagonal, niter,
                     d_steqr_D_, d_steqr_E_, d_steqr_C_, niter, d_steqr_info_);

    rocblas_int h_steqr_info;
    HIP_CHECK(hipMemcpy(&h_steqr_info, d_steqr_info_, sizeof(rocblas_int), hipMemcpyDeviceToHost));
    if (h_steqr_info != 0) {
        throw std::runtime_error("rocsolver_dsteqr failed with info = " + std::to_string(h_steqr_info));
    }

    double energy;
    HIP_CHECK(hipMemcpy(&energy, d_steqr_D_, sizeof(double), hipMemcpyDeviceToHost));

    // Ritz coefficients = first column of eigenvector matrix (on device)
    if constexpr (std::is_same_v<Scalar, double>) {
        HIP_CHECK(hipMemcpyAsync(d_ritz_coeffs_, d_steqr_C_, niter * sizeof(double), hipMemcpyDeviceToDevice, stream_));
    } else {
        // Complex case: promote double eigenvectors to complex on GPU
        int blk = (niter + 63) / 64;
        hipLaunchKernelGGL(promote_double_to_complex, dim3(blk), dim3(64), 0, stream_,
                           d_steqr_C_, (hipDoubleComplex*)d_ritz_coeffs_, niter);
    }

    // Use device pointer mode for finalization
    ROCBLAS_CHECK(rocblas_set_pointer_mode(rocblas_h_, rocblas_pointer_mode_device));
    ROCBLAS_CHECK(Traits::gemv(
        rocblas_h_, rocblas_operation_none,
        n, niter, d_const_one_,
        d_lanczos_v, n,
        d_ritz_coeffs_, 1,
        d_const_zero_, d_theta, 1
    ));

    // Normalize theta (device pointer mode)
    ROCBLAS_CHECK(Traits::nrm2(rocblas_h_, n, d_theta, 1, d_nrm2_result_));
    hipLaunchKernelGGL(invert_nrm_kernel<RealType>, dim3(1), dim3(1), 0, stream_,
                       d_nrm2_result_, d_inv_nrm_);
    ROCBLAS_CHECK(Traits::scal_real(rocblas_h_, n, d_inv_nrm_, d_theta, 1));
    ROCBLAS_CHECK(rocblas_set_pointer_mode(rocblas_h_, rocblas_pointer_mode_host));

    t_lanczos_.end(stream_);
    return energy;
}

// ============================================================================
// SVD and MPS update
// ============================================================================

template<typename Scalar>
void DMRGGPU<Scalar>::svd_and_update_mps(int site, Scalar* d_theta, char direction) {
    t_svd_.begin(stream_);
    int cL = chi_L(site);
    int cR = chi_R(site);

    int m, n_svd;
    if (direction == 'R') { m = cL * d_; n_svd = cR; }
    else                  { m = cL;      n_svd = d_ * cR; }
    int full_k = std::min(m, n_svd);
    int k = std::min(full_k, chi_max_);

    // GPU SVD via size-gated dispatcher (R3-F2 + regression fix): Jacobi
    // for large / real; bidiagonal fallback for small complex shapes where
    // zgesvdj regresses. See docs/followups/r3_regression_analysis.md.
    HIP_CHECK(hipMemcpyAsync(d_svd_A_, d_theta, m * n_svd * sizeof(Scalar),
                             hipMemcpyDeviceToDevice, stream_));

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

    // Truncation: find new_k on GPU, copy just 1 int back.
    //
    // DEVICE_K skips the truncation kernel and the D2H readback, using k
    // (= min(chi_max, full_k)) directly. Consequences: (a) no stream drain
    // per site, and (b) near-zero singular values below 1e-14 are kept.
    // Those modes enter subsequent GEMMs multiplied by ~0 — no numerical
    // impact within the 1e-10 DMRG tolerance, just a small amount of wasted
    // compute on the tail columns.
    int new_k;
    if (opts_.device_k) {
        new_k = k;
    } else {
        hipLaunchKernelGGL(svd_truncate_kernel<RealType>, dim3(1), dim3(1), 0, stream_,
                           d_svd_S_, k, 1e-14, d_svd_info_);
        HIP_CHECK(hipMemcpy(&new_k, d_svd_info_, sizeof(int), hipMemcpyDeviceToHost));
    }

    int threads = 256;

    if (direction == 'R') {
        int new_chi_R = new_k;

        // MPS[site] = U[:, :new_k]  (column slice, on GPU)
        allocate_mps_tensor(site, cL, new_chi_R);
        if (new_k == full_k) {
            HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site], d_svd_U_,
                                     m * new_k * sizeof(Scalar), hipMemcpyDeviceToDevice, stream_));
        } else {
            int total = m * new_k;
            hipLaunchKernelGGL(extract_cols_kernel<Scalar>, dim3((total+threads-1)/threads), dim3(threads), 0, stream_,
                               d_svd_U_, m, d_mps_tensors_[site], m, m, new_k);
        }
        // MPS[site] = U is now queued on stream_. Signal the env stream so
        // update_left_env(site) can start concurrently with the absorb below.
        HIP_CHECK(hipEventRecord(event_canon_ready_, stream_));
        t_svd_.end(stream_);
        t_absorb_.begin(stream_);

        // S*Vh → d_svd_work_ (scale rows of Vh by S, on GPU)
        {
            int total = new_k * n_svd;
            hipLaunchKernelGGL((scale_rows_by_diag_kernel<Scalar, RealType>), dim3((total+threads-1)/threads), dim3(threads), 0, stream_,
                               d_svd_S_, d_svd_Vh_, full_k, d_svd_work_, new_k, new_k, n_svd);
        }

        // Absorb S*Vh into A[site+1]
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
        t_absorb_.end(stream_);

    } else {  // direction == 'L'
        int new_chi_L = new_k;

        // MPS[site] = Vh[:new_k, :]  (row slice = column slice of Vh in col-major)
        allocate_mps_tensor(site, new_chi_L, cR);
        if (new_chi_L == full_k) {
            HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site], d_svd_Vh_,
                                     full_k * n_svd * sizeof(Scalar), hipMemcpyDeviceToDevice, stream_));
        } else {
            int total = new_chi_L * n_svd;
            hipLaunchKernelGGL(extract_cols_kernel<Scalar>, dim3((total+threads-1)/threads), dim3(threads), 0, stream_,
                               d_svd_Vh_, full_k, d_mps_tensors_[site], new_chi_L, new_chi_L, n_svd);
        }
        // MPS[site] = Vh is now queued on stream_. Signal the env stream so
        // update_right_env(site) can start concurrently with the absorb below.
        HIP_CHECK(hipEventRecord(event_canon_ready_, stream_));
        t_svd_.end(stream_);
        t_absorb_.begin(stream_);

        // U*S → d_svd_work_ (scale columns of U by S, on GPU)
        {
            int total = m * new_k;
            hipLaunchKernelGGL((scale_cols_by_diag_kernel<Scalar, RealType>), dim3((total+threads-1)/threads), dim3(threads), 0, stream_,
                               d_svd_S_, d_svd_U_, m, d_svd_work_, m, m, new_k);
        }

        // Absorb U*S into A[site-1]
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
        t_absorb_.end(stream_);
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
    // Dual-stream pipeline:
    //   stream_     : Lanczos + SVD + absorb(S*Vh) into MPS[site+1]
    //   stream_env_ : update_left_env(site) → L[site+1]
    //
    // Each iteration: the SVD on stream_ records event_canon_ready_ the
    // moment MPS[site] = U is written. The env stream picks this up and runs
    // update_left_env concurrently with the absorb GEMM on stream_. Before
    // the next site's Lanczos (which reads L[site+1]) we make stream_ wait
    // on event_env_done_.
    double energy = 0.0;

    for (int site = 0; site < L_ - 1; site++) {
        if (env_update_pending_) {
            HIP_CHECK(hipStreamWaitEvent(stream_, event_env_done_, 0));
            env_update_pending_ = false;
        }

        energy = optimize_site(site, 'R');

        HIP_CHECK(hipStreamWaitEvent(stream_env_, event_canon_ready_, 0));
        update_left_env(site);
        HIP_CHECK(hipEventRecord(event_env_done_, stream_env_));
        env_update_pending_ = true;
    }
    if (env_update_pending_) {
        HIP_CHECK(hipStreamWaitEvent(stream_, event_env_done_, 0));
        env_update_pending_ = false;
    }
    // Optimize last site without SVD
    {
        int site = L_ - 1;
        form_theta(site, d_theta_);
        energy = lanczos_eigensolver(site, d_theta_);
        int sz = chi_L(site) * d_ * chi_R(site);
        HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site], d_theta_,
                                 sz * sizeof(Scalar), hipMemcpyDeviceToDevice, stream_));
    }

    return energy;
}

template<typename Scalar>
double DMRGGPU<Scalar>::sweep_right_to_left() {
    // Mirror of sweep_left_to_right: update_right_env(site) overlaps with the
    // absorb(U*S) into MPS[site-1] on stream_.
    double energy = 0.0;

    for (int site = L_ - 1; site >= 1; site--) {
        if (env_update_pending_) {
            HIP_CHECK(hipStreamWaitEvent(stream_, event_env_done_, 0));
            env_update_pending_ = false;
        }

        energy = optimize_site(site, 'L');

        HIP_CHECK(hipStreamWaitEvent(stream_env_, event_canon_ready_, 0));
        update_right_env(site);
        HIP_CHECK(hipEventRecord(event_env_done_, stream_env_));
        env_update_pending_ = true;
    }
    if (env_update_pending_) {
        HIP_CHECK(hipStreamWaitEvent(stream_, event_env_done_, 0));
        env_update_pending_ = false;
    }
    // Optimize first site without SVD
    {
        int site = 0;
        form_theta(site, d_theta_);
        energy = lanczos_eigensolver(site, d_theta_);
        int sz = chi_L(site) * d_ * chi_R(site);
        HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site], d_theta_,
                                 sz * sizeof(Scalar), hipMemcpyDeviceToDevice, stream_));
    }

    return energy;
}

// ============================================================================
// Main algorithm
// ============================================================================

template<typename Scalar>
double DMRGGPU<Scalar>::run(int n_sweeps) {
    build_initial_environments();

    // Timer starts AFTER env build — measures sweep-to-convergence only
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
    report_timers();

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
        HIP_CHECK(hipMemcpy(h_mps[i].data(), d_mps_tensors_[i],
                            size * sizeof(Scalar), hipMemcpyDeviceToHost));
    }
}

#endif // DMRG_GPU_IMPL_H
