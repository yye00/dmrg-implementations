#ifndef DMRG2_GPU_IMPL_H
#define DMRG2_GPU_IMPL_H

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

// GPU kernels for batched GEMM pointer setup (eliminates CPU→GPU pointer copies)

// Step 1 / env update step 1: A[w*d+s] = base_A + w*strideA, B[w*d+s] = base_B + s*strideB, C[w*d+s] = base_C + (w*d+s)*strideC
template<typename Scalar>
__global__ void setup_batch_ptrs_wd(Scalar** A, Scalar** B, Scalar** C,
                                     Scalar* base_A, Scalar* base_B, Scalar* base_C,
                                     int d, int strideA, int strideB, int strideC) {
    int idx = threadIdx.x;
    int w = idx / d, s = idx % d;
    A[idx] = base_A + w * strideA;
    B[idx] = base_B + s * strideB;
    C[idx] = base_C + idx * strideC;
}

// Two-site step 1: idx = w*dd + s1*d + s2, B uses transposed physical index (s1 + s2*d)
// A[idx] = base_A + (idx/dd)*strideA, B[idx] = base_B + (s1 + s2*d)*strideB, C[idx] = base_C + idx*strideC
template<typename Scalar>
__global__ void setup_batch_ptrs_wd_twosite(Scalar** A, Scalar** B, Scalar** C,
                                             Scalar* base_A, Scalar* base_B, Scalar* base_C,
                                             int d, int dd, int strideA, int strideB, int strideC) {
    int idx = threadIdx.x;
    int w = idx / dd;
    int ss = idx % dd;   // s1*d + s2
    int s1 = ss / d, s2 = ss % d;
    A[idx] = base_A + w * strideA;
    B[idx] = base_B + (s1 + s2 * d) * strideB;  // transposed physical index
    C[idx] = base_C + idx * strideC;
}

// Two-site step 3: per n iteration, dd batches. idx = s1p*d + s2p.
// A[idx] = base_A + (n*dd + idx)*strideA, B[idx] = base_B + n*strideB, C[idx] = base_C + (s1p + s2p*d)*strideC
template<typename Scalar>
__global__ void setup_batch_ptrs_step3_twosite(Scalar** A, Scalar** B, Scalar** C,
                                                Scalar* base_A, Scalar* base_B, Scalar* base_C,
                                                int n, int d, int dd, int strideA, int strideB, int strideC) {
    int idx = threadIdx.x;  // s1p*d + s2p
    int s1p = idx / d, s2p = idx % d;
    A[idx] = base_A + (n * dd + idx) * strideA;
    B[idx] = base_B + n * strideB;
    C[idx] = base_C + (s1p + s2p * d) * strideC;  // transposed physical index
}

// Two-site step 3 (full batched, R3-F1): one launch sets up D*dd batch pointers.
// idx = n*dd + (s1p*d + s2p). Each n writes to its own scratch slice; slices are
// later summed into the final result via a single rocblas_gemv reduction.
//   A[idx] -> T2[n*dd + ss]      (cL x cR block, lda = cL*cR)
//   B[idx] -> R_env[n]            (cR x cR block, ldb = cR*D)
//   C[idx] -> scratch + n*slice_stride + (s1p + s2p*d)*strideC_tile
//             (cL x cR tile inside the (cL, dd, cR) slice, ldc = cL*dd)
template<typename Scalar>
__global__ void setup_batch_ptrs_step3_twosite_full(Scalar** A, Scalar** B, Scalar** C,
                                                     Scalar* base_A, Scalar* base_B, Scalar* base_C_scratch,
                                                     int d, int dd, int strideA, int strideB,
                                                     int strideC_tile, int slice_stride) {
    int idx = threadIdx.x;  // n*dd + s1p*d + s2p
    int n  = idx / dd;
    int ss = idx % dd;      // s1p*d + s2p
    int s1p = ss / d, s2p = ss % d;
    A[idx] = base_A + (n * dd + ss) * strideA;
    B[idx] = base_B + n * strideB;
    C[idx] = base_C_scratch + n * slice_stride + (s1p + s2p * d) * strideC_tile;
}

// SPARSE_MPO (two-site): sparse variants of the Step 1 / Step 3-full setup
// kernels for two-site apply_heff. They index into a precomputed list of
// nonzero (w, s1, s2) or (n, s1p, s2p) pairs (packed as w*dd + s1*d + s2 or
// n*dd + s1p*d + s2p respectively).

// Sparse two-site Step 1:
//   A[idx] <- base_A + w*strideA   where (w, s1, s2) unpacked from nnz[idx]
//   B[idx] <- base_B + (s1 + s2*d)*strideB
//   C[idx] <- base_C + packed*strideC  (full-layout V slot, so V must be
//             zeroed beforehand)
template<typename Scalar>
__global__ void setup_batch_ptrs_wd_twosite_sparse(Scalar** A, Scalar** B, Scalar** C,
                                                    Scalar* base_A, Scalar* base_B, Scalar* base_C,
                                                    const int* nnz_wss, int d, int dd,
                                                    int strideA, int strideB, int strideC) {
    int idx = threadIdx.x;
    int packed = nnz_wss[idx];
    int w  = packed / dd;
    int ss = packed % dd;
    int s1 = ss / d, s2 = ss % d;
    A[idx] = base_A + w * strideA;
    B[idx] = base_B + (s1 + s2 * d) * strideB;
    C[idx] = base_C + packed * strideC;
}

// Sparse two-site Step 3-full:
//   A[idx] <- base_A + packed*strideA
//   B[idx] <- base_B + n*strideB
//   C[idx] <- base_C_scratch + n*slice_stride + (s1p + s2p*d)*strideC_tile
// The per-n scratch slices must be zeroed beforehand so the GEMV reduction
// over D doesn't pick up stale values from skipped n slices.
template<typename Scalar>
__global__ void setup_batch_ptrs_step3_twosite_full_sparse(Scalar** A, Scalar** B, Scalar** C,
                                                           Scalar* base_A, Scalar* base_B, Scalar* base_C_scratch,
                                                           const int* nnz_nss, int d, int dd,
                                                           int strideA, int strideB,
                                                           int strideC_tile, int slice_stride) {
    int idx = threadIdx.x;
    int packed = nnz_nss[idx];
    int n  = packed / dd;
    int ss = packed % dd;
    int s1p = ss / d, s2p = ss % d;
    A[idx] = base_A + packed * strideA;
    B[idx] = base_B + n * strideB;
    C[idx] = base_C_scratch + n * slice_stride + (s1p + s2p * d) * strideC_tile;
}

// Env update step 3 (left): A[w] = base_A + (w*d+sp)*strideA, B[w] = base_B + sp*strideB, C[w] = base_C + w*strideC
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

// Fused Lanczos update:  w := w + (-α)*v_i + [(-β_{im1})*v_{im1}]
template<typename Scalar>
__global__ void lanczos_fused_sub_kernel(
    Scalar* w, const Scalar* v_i, const Scalar* v_im1,
    const Scalar* d_neg_alpha, const Scalar* d_neg_beta_im1,
    int n, int has_prev) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    Scalar wi = w[idx];
    wi = scalar_add(wi, scalar_mul(*d_neg_alpha, v_i[idx]));
    if (has_prev) {
        wi = scalar_add(wi, scalar_mul(*d_neg_beta_im1, v_im1[idx]));
    }
    w[idx] = wi;
}

// Fused normalize-and-copy:  v_{i+1}[k] = w[k] * inv_beta
template<typename Scalar, typename RealType>
__global__ void lanczos_fused_norm_copy_kernel(
    Scalar* v_next, const Scalar* w, const RealType* d_inv_beta, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    v_next[idx] = ScalarTraits<Scalar>::scale_by_real(*d_inv_beta, w[idx]);
}

// ============================================================================
// Constructor
// ============================================================================

template<typename Scalar>
DMRG2GPU<Scalar>::DMRG2GPU(int L, int d, int chi_max, int D_mpo, double tol)
    : L_(L), d_(d), chi_max_(chi_max), D_mpo_(D_mpo), D_mpo_actual_(D_mpo),
      tol_(tol), energy_(0.0) {

    opts_.load_from_env();

    if (opts_.d_pad) {
        int padded = (D_mpo_ + 7) & ~7;
        if (padded != D_mpo_) {
            std::fprintf(stderr, "[D_PAD] D_mpo padded: %d -> %d\n", D_mpo_, padded);
            D_mpo_ = padded;
        }
    }

    opts_.print(stderr);
    init_timers();

    // Bond dimensions (same as single-site: min-cut formula capped at chi_max)
    bond_dims_.resize(L + 1);
    bond_dims_[0] = 1;
    bond_dims_[L] = 1;
    for (int i = 1; i < L; i++) {
        double exact_dim = pow((double)d_, std::min(i, L - i));
        bond_dims_[i] = (exact_dim > chi_max_) ? chi_max_ : (int)exact_dim;
    }

    // GPU handles — dual-stream
    HIP_CHECK(hipStreamCreate(&stream_));
    ROCBLAS_CHECK(rocblas_create_handle(&rocblas_h_));
    ROCBLAS_CHECK(rocblas_set_stream(rocblas_h_, stream_));
    HIP_CHECK(hipStreamCreate(&stream_env_));
    ROCBLAS_CHECK(rocblas_create_handle(&rocblas_h_env_));
    ROCBLAS_CHECK(rocblas_set_stream(rocblas_h_env_, stream_env_));
    HIP_CHECK(hipEventCreateWithFlags(&event_canon_ready_, hipEventDisableTiming));
    HIP_CHECK(hipEventCreateWithFlags(&event_env_done_, hipEventDisableTiming));

    int dd = d_ * d_;  // d^2 for two-site

    // Contraction intermediates: D*d^2*chi_max^2
    int t_max = D_mpo_ * dd * chi_max_ * chi_max_;
    HIP_CHECK(hipMalloc(&d_T1_, t_max * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_T2_, t_max * sizeof(Scalar)));

    // Env-stream scratch (independent of stream_'s T1/T2)
    HIP_CHECK(hipMalloc(&d_T1_env_, t_max * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_T2_env_, t_max * sizeof(Scalar)));
    int batch_max_env = D_mpo_ * d_;
    HIP_CHECK(hipMalloc(&d_batch_A_env_, batch_max_env * sizeof(Scalar*)));
    HIP_CHECK(hipMalloc(&d_batch_B_env_, batch_max_env * sizeof(Scalar*)));
    HIP_CHECK(hipMalloc(&d_batch_C_env_, batch_max_env * sizeof(Scalar*)));

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

    // Lanczos workspace: theta is d^2 times larger than single-site
    theta_size_max_ = chi_max_ * dd * chi_max_;
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

    // Batched GEMM pointer arrays: D*d^2 batches for two-site
    int batch_max = D_mpo_ * dd;
    HIP_CHECK(hipMalloc(&d_batch_A_, batch_max * sizeof(Scalar*)));
    HIP_CHECK(hipMalloc(&d_batch_B_, batch_max * sizeof(Scalar*)));
    HIP_CHECK(hipMalloc(&d_batch_C_, batch_max * sizeof(Scalar*)));

    // Length-D ones vector for Step-3 GEMV reduction (R3-F1)
    HIP_CHECK(hipMalloc(&d_ones_D_, D_mpo_ * sizeof(Scalar)));
    {
        std::vector<Scalar> h_ones(D_mpo_, Traits::one());
        HIP_CHECK(hipMemcpy(d_ones_D_, h_ones.data(),
                            D_mpo_ * sizeof(Scalar), hipMemcpyHostToDevice));
    }

    // SVD workspace: theta reshaped as (chi_max*d, d*chi_max)
    int svd_max_m = chi_max_ * d_;
    int svd_max_n = d_ * chi_max_;
    int svd_max_k = std::min(svd_max_m, svd_max_n);  // = chi_max_ * d_

    HIP_CHECK(hipMalloc(&d_svd_A_,    theta_size_max_ * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_svd_U_,    (size_t)svd_max_m * svd_max_k * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_svd_S_,    svd_max_k * sizeof(RealType)));
    HIP_CHECK(hipMalloc(&d_svd_Vh_,   (size_t)svd_max_k * svd_max_n * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_svd_E_,    svd_max_k * sizeof(RealType)));
    HIP_CHECK(hipMalloc(&d_svd_info_, sizeof(int)));
    HIP_CHECK(hipMalloc(&d_svd_work_, theta_size_max_ * sizeof(Scalar)));
    // R3-F2: rocsolver_dgesvdj requires device pointers for residual/n_sweeps.
    HIP_CHECK(hipMalloc(&d_svdj_residual_, sizeof(double)));
    HIP_CHECK(hipMalloc(&d_svdj_n_sweeps_, sizeof(rocblas_int)));

    // CPU workspace (for receiving GPU SVD results and truncation/scaling)
    h_svd_U_.resize((size_t)svd_max_m * svd_max_k);
    h_svd_S_.resize(svd_max_k);
    h_svd_Vh_.resize((size_t)svd_max_k * svd_max_n);
    h_svd_tmp_.resize(std::max((size_t)svd_max_m * svd_max_k, (size_t)svd_max_k * svd_max_n));

    // SPARSE_MPO nnz lists (populated in precompute_fused_mpo)
    d_WW_nnz_rows_.resize(L - 1, nullptr);
    d_WW_nnz_cols_.resize(L - 1, nullptr);
    ww_nnz_rows_count_.assign(L - 1, 0);
    ww_nnz_cols_count_.assign(L - 1, 0);

    if (opts_.lanczos_graph) {
        HIP_CHECK(hipMalloc(&d_heff_input_, (size_t)theta_size_max_ * sizeof(Scalar)));
    }

    if (opts_.rsvd) {
        int r_max = chi_max_ + RSVD_OVERSAMPLE_;
        HIP_CHECK(hipFree(d_svd_S_));
        HIP_CHECK(hipFree(d_svd_E_));
        HIP_CHECK(hipFree(d_svd_U_));
        HIP_CHECK(hipFree(d_svd_Vh_));
        HIP_CHECK(hipMalloc(&d_svd_S_,  r_max * sizeof(RealType)));
        HIP_CHECK(hipMalloc(&d_svd_E_,  r_max * sizeof(RealType)));
        HIP_CHECK(hipMalloc(&d_svd_U_,  (size_t)svd_max_m * r_max * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&d_svd_Vh_, (size_t)r_max * svd_max_n * sizeof(Scalar)));

        rsvd_r_max_ = r_max;
        HIP_CHECK(hipMalloc(&d_rsvd_omega_,   (size_t)svd_max_n * rsvd_r_max_ * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&d_rsvd_Y_,       (size_t)svd_max_m * rsvd_r_max_ * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&d_rsvd_tau_,     (size_t)rsvd_r_max_ * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&d_rsvd_B_,       (size_t)rsvd_r_max_ * svd_max_n * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&d_rsvd_U_small_, (size_t)rsvd_r_max_ * rsvd_r_max_ * sizeof(Scalar)));
    }
}

// ============================================================================
// Phase timers
// ============================================================================

template<typename Scalar>
void DMRG2GPU<Scalar>::init_timers() {
    t_lanczos_.init("lanczos", opts_.profile);
    t_apply_heff_.init("apply_heff", opts_.profile);
    t_svd_.init("svd", opts_.profile);
    t_absorb_.init("absorb", opts_.profile);
    t_env_update_.init("env_update", opts_.profile);
}

template<typename Scalar>
void DMRG2GPU<Scalar>::report_timers() {
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
// Destructor
// ============================================================================

template<typename Scalar>
DMRG2GPU<Scalar>::~DMRG2GPU() {
    free_gpu_resources();
}

template<typename Scalar>
void DMRG2GPU<Scalar>::free_gpu_resources() {
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
    if (d_batch_A_) hipFree(d_batch_A_);
    if (d_batch_B_) hipFree(d_batch_B_);
    if (d_batch_C_) hipFree(d_batch_C_);
    if (d_ones_D_) hipFree(d_ones_D_);
    if (d_svd_A_) hipFree(d_svd_A_);
    if (d_svd_U_) hipFree(d_svd_U_);
    if (d_svd_S_) hipFree(d_svd_S_);
    if (d_svd_Vh_) hipFree(d_svd_Vh_);
    if (d_svd_E_) hipFree(d_svd_E_);
    if (d_svd_info_) hipFree(d_svd_info_);
    if (d_svd_work_) hipFree(d_svd_work_);
    if (d_svdj_residual_) hipFree(d_svdj_residual_);
    if (d_svdj_n_sweeps_) hipFree(d_svdj_n_sweeps_);

    // Dual-stream env resources
    if (d_T1_env_) hipFree(d_T1_env_);
    if (d_T2_env_) hipFree(d_T2_env_);
    if (d_batch_A_env_) hipFree(d_batch_A_env_);
    if (d_batch_B_env_) hipFree(d_batch_B_env_);
    if (d_batch_C_env_) hipFree(d_batch_C_env_);

    // SPARSE_MPO nnz lists
    for (auto ptr : d_WW_nnz_rows_) if (ptr) hipFree(ptr);
    for (auto ptr : d_WW_nnz_cols_) if (ptr) hipFree(ptr);

    // RSVD workspace
    if (d_rsvd_omega_)   hipFree(d_rsvd_omega_);
    if (d_rsvd_Y_)       hipFree(d_rsvd_Y_);
    if (d_rsvd_tau_)     hipFree(d_rsvd_tau_);
    if (d_rsvd_B_)       hipFree(d_rsvd_B_);
    if (d_rsvd_U_small_) hipFree(d_rsvd_U_small_);

    // LANCZOS_GRAPH
    for (auto& kv : apply_heff_graph_cache_) hipGraphExecDestroy(kv.second);
    apply_heff_graph_cache_.clear();
    if (d_heff_input_) hipFree(d_heff_input_);

    rocblas_destroy_handle(rocblas_h_);
    hipStreamDestroy(stream_);
    rocblas_destroy_handle(rocblas_h_env_);
    hipStreamDestroy(stream_env_);
    hipEventDestroy(event_canon_ready_);
    hipEventDestroy(event_env_done_);
}

// ============================================================================
// Memory management
// ============================================================================

template<typename Scalar>
void DMRG2GPU<Scalar>::allocate_mps_tensor(int site, int cL, int cR) {
    size_t max_sz = (size_t)chi_max_ * d_ * chi_max_ * sizeof(Scalar);
    if (!d_mps_tensors_[site]) {
        HIP_CHECK(hipMalloc(&d_mps_tensors_[site], max_sz));
    }
    (void)cL; (void)cR;
}

template<typename Scalar>
void DMRG2GPU<Scalar>::ensure_L_env_alloc(int idx, int chi) {
    if (chi > L_env_alloc_chi_[idx]) {
        if (d_L_envs_[idx]) HIP_CHECK(hipFree(d_L_envs_[idx]));
        int sz = chi * D_mpo_ * chi;
        HIP_CHECK(hipMalloc(&d_L_envs_[idx], sz * sizeof(Scalar)));
        L_env_alloc_chi_[idx] = chi;
    }
}

template<typename Scalar>
void DMRG2GPU<Scalar>::ensure_R_env_alloc(int idx, int chi) {
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
void DMRG2GPU<Scalar>::initialize_mps_random(double scale) {
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
        HIP_CHECK(hipMemcpy(d_mps_tensors_[i], h_A.data(),
                            size * sizeof(Scalar), hipMemcpyHostToDevice));
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
        HIP_CHECK(hipMemcpy(d_mps_tensors_[i], h_A.data(),
                            size * sizeof(Scalar), hipMemcpyHostToDevice));
    }
}

// ============================================================================
// MPO setup and fused two-site MPO precomputation
// ============================================================================

template<typename Scalar>
void DMRG2GPU<Scalar>::set_mpo(const std::vector<Scalar*>& h_mpo_tensors) {
    int D_use = D_mpo_;
    int D_act = D_mpo_actual_;
    int d = d_;
    for (int i = 0; i < L_; i++) {
        int size_use = D_use * d * d * D_use;
        HIP_CHECK(hipMalloc(&d_mpo_tensors_[i], size_use * sizeof(Scalar)));
        HIP_CHECK(hipMemset(d_mpo_tensors_[i], 0, size_use * sizeof(Scalar)));

        std::vector<Scalar> h_WL(size_use, Traits::zero());
        std::vector<Scalar> h_WR(size_use, Traits::zero());

        if (D_use == D_act) {
            HIP_CHECK(hipMemcpy(d_mpo_tensors_[i], h_mpo_tensors[i],
                                size_use * sizeof(Scalar), hipMemcpyHostToDevice));
        } else {
            std::vector<Scalar> h_pad(size_use, Traits::zero());
            for (int wp = 0; wp < D_act; wp++)
                for (int sp = 0; sp < d; sp++)
                    for (int s = 0; s < d; s++)
                        for (int w = 0; w < D_act; w++)
                            h_pad[w + s*D_use + sp*D_use*d + wp*D_use*d*d] =
                                h_mpo_tensors[i][w + s*D_act + sp*D_act*d + wp*D_act*d*d];
            HIP_CHECK(hipMemcpy(d_mpo_tensors_[i], h_pad.data(),
                                size_use * sizeof(Scalar), hipMemcpyHostToDevice));
        }

        for (int w = 0; w < D_act; w++)
            for (int s = 0; s < d; s++)
                for (int sp = 0; sp < d; sp++)
                    for (int wp = 0; wp < D_act; wp++) {
                        Scalar val = h_mpo_tensors[i][w + s*D_act + sp*D_act*d + wp*D_act*d*d];
                        h_WL[(w*d+s) + (wp*d+sp) * D_use * d] = val;
                        h_WR[(wp*d+s) + (w*d+sp) * D_use * d] = val;
                    }
        HIP_CHECK(hipMalloc(&d_W_left_[i], size_use * sizeof(Scalar)));
        HIP_CHECK(hipMemcpy(d_W_left_[i], h_WL.data(),
                            size_use * sizeof(Scalar), hipMemcpyHostToDevice));
        HIP_CHECK(hipMalloc(&d_W_right_[i], size_use * sizeof(Scalar)));
        HIP_CHECK(hipMemcpy(d_W_right_[i], h_WR.data(),
                            size_use * sizeof(Scalar), hipMemcpyHostToDevice));
    }

    // Precompute fused two-site MPO
    precompute_fused_mpo(h_mpo_tensors);
}

template<typename Scalar>
void DMRG2GPU<Scalar>::precompute_fused_mpo(const std::vector<Scalar*>& h_mpo_tensors) {
    int D = D_mpo_, d = d_;        // padded (output layout)
    int D_act = D_mpo_actual_;     // actual (host MPO stride)
    int dd = d * d;

    for (int bond = 0; bond < L_ - 1; bond++) {
        // WW[row, col] where row = w*dd + s1*d + s2, col = n*dd + s1p*d + s2p
        // WW[(w*dd+s1*d+s2), (n*dd+s1p*d+s2p)] = sum_m W_L[w,s1,s1p,m] * W_R[m,s2,s2p,n]
        int ww_size = D * dd * dd * D;
        std::vector<Scalar> h_WW(ww_size, Traits::zero());

        const Scalar* WL = h_mpo_tensors[bond];
        const Scalar* WR = h_mpo_tensors[bond + 1];

        for (int w = 0; w < D_act; w++)
            for (int n = 0; n < D_act; n++)
                for (int s1 = 0; s1 < d; s1++)
                    for (int s2 = 0; s2 < d; s2++)
                        for (int s1p = 0; s1p < d; s1p++)
                            for (int s2p = 0; s2p < d; s2p++) {
                                Scalar val = Traits::zero();
                                for (int m = 0; m < D_act; m++) {
                                    // W_L[w, s1, s1p, m] at w + s1*D_act + s1p*D_act*d + m*D_act*d*d
                                    Scalar wl = WL[w + s1*D_act + s1p*D_act*d + m*D_act*d*d];
                                    // W_R[m, s2, s2p, n] at m + s2*D_act + s2p*D_act*d + n*D_act*d*d
                                    Scalar wr = WR[m + s2*D_act + s2p*D_act*d + n*D_act*d*d];
                                    if constexpr (Traits::is_complex) {
                                        val = hipCadd(val, hipCmul(wl, wr));
                                    } else {
                                        val += wl * wr;
                                    }
                                }
                                int row = w * dd + s1 * d + s2;
                                int col = n * dd + s1p * d + s2p;
                                h_WW[row + col * D * dd] = val;  // column-major, padded pitch
                            }

        HIP_CHECK(hipMalloc(&d_WW_[bond], ww_size * sizeof(Scalar)));
        HIP_CHECK(hipMemcpy(d_WW_[bond], h_WW.data(),
                            ww_size * sizeof(Scalar), hipMemcpyHostToDevice));

        // SPARSE_MPO: build nnz row/col index lists for WW[bond].
        // WW is (D*dd, dd*D) column-major. Row r = w*dd + s1*d + s2 is nonzero
        // if any col has |val| > 0. Col c = n*dd + s1p*d + s2p analogously.
        if (opts_.sparse_mpo) {
            int rows = D * dd;
            int cols = dd * D;
            std::vector<int> nnz_rows, nnz_cols;
            nnz_rows.reserve(rows);
            nnz_cols.reserve(cols);
            const double eps = 1e-14;
            for (int r = 0; r < rows; r++) {
                bool nz = false;
                for (int c = 0; c < cols && !nz; c++) {
                    if (scalar_abs(h_WW[r + (size_t)c * rows]) > eps) nz = true;
                }
                if (nz) nnz_rows.push_back(r);
            }
            for (int c = 0; c < cols; c++) {
                bool nz = false;
                for (int r = 0; r < rows && !nz; r++) {
                    if (scalar_abs(h_WW[r + (size_t)c * rows]) > eps) nz = true;
                }
                if (nz) nnz_cols.push_back(c);
            }
            ww_nnz_rows_count_[bond] = (int)nnz_rows.size();
            ww_nnz_cols_count_[bond] = (int)nnz_cols.size();
            if (!nnz_rows.empty()) {
                HIP_CHECK(hipMalloc(&d_WW_nnz_rows_[bond], nnz_rows.size() * sizeof(int)));
                HIP_CHECK(hipMemcpy(d_WW_nnz_rows_[bond], nnz_rows.data(),
                                    nnz_rows.size() * sizeof(int), hipMemcpyHostToDevice));
            }
            if (!nnz_cols.empty()) {
                HIP_CHECK(hipMalloc(&d_WW_nnz_cols_[bond], nnz_cols.size() * sizeof(int)));
                HIP_CHECK(hipMemcpy(d_WW_nnz_cols_[bond], nnz_cols.data(),
                                    nnz_cols.size() * sizeof(int), hipMemcpyHostToDevice));
            }
            if (bond == 0) {
                std::fprintf(stderr,
                    "[SPARSE_MPO] bond 0: WW shape (%d x %d), nnz rows=%d, nnz cols=%d (%.0f%% sparse)\n",
                    rows, cols, ww_nnz_rows_count_[bond], ww_nnz_cols_count_[bond],
                    100.0 * (1.0 - (double)(ww_nnz_rows_count_[bond] * ww_nnz_cols_count_[bond]) / (rows * cols)));
            }
        }
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
void DMRG2GPU<Scalar>::apply_heff_two_site(int site, const Scalar* d_theta_in, Scalar* d_result) {
    t_apply_heff_.begin(stream_);
    int cL = chi_L(site);
    int cR = chi_R(site + 1);
    int D = D_mpo_, d = d_;
    int dd = d * d;
    Scalar one = Traits::one(), zero_val = Traits::zero();

    // LANCZOS_GRAPH: stage theta into a fixed-address bounce buffer so every
    // graph replay reads from the same address regardless of which Lanczos v_i
    // the caller passes. On cache hit replay immediately; on miss run the body
    // under hipStreamBeginCapture, instantiate, cache, then launch.
    const Scalar* theta_src = d_theta_in;
    bool graph_capture_miss = false;
    if (opts_.lanczos_graph) {
        int n_theta = cL * dd * cR;
        HIP_CHECK(hipMemcpyAsync(d_heff_input_, d_theta_in,
                                 (size_t)n_theta * sizeof(Scalar),
                                 hipMemcpyDeviceToDevice, stream_));
        theta_src = d_heff_input_;

        uint64_t key = graph_key(site, cL, cR);
        auto it = apply_heff_graph_cache_.find(key);
        if (it != apply_heff_graph_cache_.end()) {
            HIP_CHECK(hipGraphLaunch(it->second, stream_));
            t_apply_heff_.end(stream_);
            return;
        }
        graph_capture_miss = true;
        HIP_CHECK(hipStreamBeginCapture(stream_, hipStreamCaptureModeGlobal));
    }

    Scalar* L_env = d_L_envs_[site];
    Scalar* R_env = d_R_envs_[site + 2];
    Scalar* WW = d_WW_[site];  // fused two-site MPO for bond (site, site+1)
    Scalar* T1 = d_T1_;
    Scalar* T2 = d_T2_;

    // ---------------------------------------------------------------
    // Step 1: Batched GEMM — contract L_env with theta
    //   T1[ws, a', b] = L[w]^T[a', a] * theta[s1,s2][a, b]
    //   ws = w*dd + s1*d + s2, D*d^2 batches
    //   Each batch: (cL, cR) = op_t(L[w]) @ theta[s1,s2]
    //     L[w]: (cL, cL) with lda = cL*D
    //     theta[s1,s2]: (cL, cR) with ldb = cL*d^2
    //     T1[ws]: (cL, cR) with ldc = cL
    // ---------------------------------------------------------------
    // SPARSE_MPO: compact the Step 1 / Step 3 batches to non-zero (w, s1, s2)
    // / (n, s1p, s2p) rows/cols of WW[site]. The setup kernels still write to
    // full-layout V slots so Step 2's dense GEMM consumes V unchanged; skipped
    // slots must be zero, enforced with hipMemsetAsync before Step 1 and again
    // before Step 3.
    const bool sparse_s1 = opts_.sparse_mpo
                         && ww_nnz_rows_count_[site] > 0
                         && ww_nnz_rows_count_[site] < D * dd;
    const bool sparse_s3 = opts_.sparse_mpo
                         && ww_nnz_cols_count_[site] > 0
                         && ww_nnz_cols_count_[site] < D * dd;

    {
        if (sparse_s1) {
            HIP_CHECK(hipMemsetAsync(T1, 0, (size_t)D * dd * cL * cR * sizeof(Scalar), stream_));
            int nnz = ww_nnz_rows_count_[site];
            hipLaunchKernelGGL(setup_batch_ptrs_wd_twosite_sparse<Scalar>,
                               dim3(1), dim3(nnz), 0, stream_,
                               d_batch_A_, d_batch_B_, d_batch_C_,
                               L_env, const_cast<Scalar*>(theta_src), T1,
                               d_WW_nnz_rows_[site],
                               d, dd, cL, cL, cL * cR);
            ROCBLAS_CHECK(Traits::gemm_batched(rocblas_h_,
                Traits::op_t, rocblas_operation_none,
                cL, cR, cL,
                &one,
                (const Scalar**)d_batch_A_, cL * D,
                (const Scalar**)d_batch_B_, cL * dd,
                &zero_val,
                d_batch_C_, cL,
                nnz));
        } else {
            int batch_count = D * dd;
            hipLaunchKernelGGL(setup_batch_ptrs_wd_twosite<Scalar>, dim3(1), dim3(batch_count), 0, stream_,
                               d_batch_A_, d_batch_B_, d_batch_C_,
                               L_env, const_cast<Scalar*>(theta_src), T1,
                               d, dd, cL, cL, cL * cR);
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
    }

    // ---------------------------------------------------------------
    // Step 2: Dense GEMM — absorb fused WW (unchanged; W_mat zero rows
    //   make the corresponding U columns zero without extra work)
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
    // Step 3 (R3-F1 full-batched collapse): per-(n,s1p,s2p) batched GEMM
    //   writing to per-n scratch slices in T1; GEMV reduces over D slices.
    // ---------------------------------------------------------------
    {
        int slice_stride = cL * dd * cR;
        if (sparse_s3) {
            HIP_CHECK(hipMemsetAsync(T1, 0, (size_t)D * slice_stride * sizeof(Scalar), stream_));
            int nnz = ww_nnz_cols_count_[site];
            hipLaunchKernelGGL(setup_batch_ptrs_step3_twosite_full_sparse<Scalar>,
                               dim3(1), dim3(nnz), 0, stream_,
                               d_batch_A_, d_batch_B_, d_batch_C_,
                               T2, R_env, T1,
                               d_WW_nnz_cols_[site],
                               d, dd, cL * cR, cR, cL, slice_stride);
            ROCBLAS_CHECK(Traits::gemm_batched(rocblas_h_,
                rocblas_operation_none, rocblas_operation_none,
                cL, cR, cR,
                &one,
                (const Scalar**)d_batch_A_, cL,
                (const Scalar**)d_batch_B_, cR * D,
                &zero_val,
                d_batch_C_, cL * dd,
                nnz));
        } else {
            int batch_count = D * dd;
            hipLaunchKernelGGL(setup_batch_ptrs_step3_twosite_full<Scalar>,
                               dim3(1), dim3(batch_count), 0, stream_,
                               d_batch_A_, d_batch_B_, d_batch_C_,
                               T2, R_env, T1,
                               d, dd, cL * cR, cR, cL, slice_stride);
            ROCBLAS_CHECK(Traits::gemm_batched(rocblas_h_,
                rocblas_operation_none, rocblas_operation_none,
                cL, cR, cR,
                &one,
                (const Scalar**)d_batch_A_, cL,
                (const Scalar**)d_batch_B_, cR * D,
                &zero_val,
                d_batch_C_, cL * dd,
                batch_count));
        }

        // Reduce: d_result[slice_stride] = T1[slice_stride x D] * ones_D
        ROCBLAS_CHECK(Traits::gemv(rocblas_h_,
            rocblas_operation_none,
            slice_stride, D,
            &one,
            T1, slice_stride,
            d_ones_D_, 1,
            &zero_val,
            d_result, 1));
    }

    if (graph_capture_miss) {
        hipGraph_t graph;
        HIP_CHECK(hipStreamEndCapture(stream_, &graph));
        hipGraphExec_t exec;
        HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        HIP_CHECK(hipGraphDestroy(graph));
        apply_heff_graph_cache_[graph_key(site, cL, cR)] = exec;
        HIP_CHECK(hipGraphLaunch(exec, stream_));
    }

    t_apply_heff_.end(stream_);
}

// ============================================================================
// Left environment update (identical to single-site)
// ============================================================================

template<typename Scalar>
void DMRG2GPU<Scalar>::update_left_env(int site) {
    t_env_update_.begin(stream_);
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
        hipLaunchKernelGGL(setup_batch_ptrs_wd<Scalar>, dim3(1), dim3(D*d), 0, stream_,
                           d_batch_A_, d_batch_B_, d_batch_C_,
                           L_env, A, V,
                           d, chi_in, chi_in, chi_in * chi_out);
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

    // Step 3: L_new_w'[b,b'] = sum_{a',s'} U[a',ws',b] * conj(A[a',s',b'])  (batched)
    // Batch D GEMMs per sp (safe: different wp write to different C locations).
    {
        for (int sp = 0; sp < d; sp++) {
            Scalar beta = (sp == 0) ? Traits::zero() : Traits::one();
            hipLaunchKernelGGL(setup_batch_ptrs_env3<Scalar>, dim3(1), dim3(D), 0, stream_,
                               d_batch_A_, d_batch_B_, d_batch_C_,
                               U, A, L_new,
                               sp, d, chi_in * chi_out, chi_in, chi_out);
            ROCBLAS_CHECK(Traits::gemm_batched(rocblas_h_,
                Traits::op_h, rocblas_operation_none,
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
    t_env_update_.end(stream_);
}

// ============================================================================
// Right environment update (identical to single-site)
// ============================================================================

template<typename Scalar>
void DMRG2GPU<Scalar>::update_right_env(int site) {
    t_env_update_.begin(stream_);
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
    // Note: A varies by s (inner dim) and B varies by w (outer dim) — opposite of
    // setup_batch_ptrs_wd convention, so swap output arrays and base/stride args.
    {
        hipLaunchKernelGGL(setup_batch_ptrs_wd<Scalar>, dim3(1), dim3(D*d), 0, stream_,
                           d_batch_B_, d_batch_A_, d_batch_C_,
                           R_env, A, V,
                           d, chi_in, chi_out, chi_out * chi_in);
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

    // Step 3: R_new_w[a,a'] = sum_s' U_ws'[a,b'] * A_s'^H[b',a']  (batched)
    // Batch D GEMMs per sp (safe: different w write to different C locations).
    {
        for (int sp = 0; sp < d; sp++) {
            Scalar beta = (sp == 0) ? Traits::zero() : Traits::one();
            hipLaunchKernelGGL(setup_batch_ptrs_env3<Scalar>, dim3(1), dim3(D), 0, stream_,
                               d_batch_A_, d_batch_B_, d_batch_C_,
                               U, A, R_new,
                               sp, d, chi_out * chi_in, chi_out, chi_out);
            ROCBLAS_CHECK(Traits::gemm_batched(rocblas_h_,
                rocblas_operation_none, Traits::op_h,
                chi_out, chi_out, chi_in,
                &one,
                (const Scalar**)d_batch_A_, chi_out,
                (const Scalar**)d_batch_B_, chi_out * d,
                &beta,
                d_batch_C_, chi_out * D,
                D));
        }
    }
    t_env_update_.end(stream_);
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
        HIP_CHECK(hipMemcpy(d_L_envs_[0], h_L.data(),
                            D_mpo_ * sizeof(Scalar), hipMemcpyHostToDevice));
    }

    // R[L] = trivial right boundary: (1, D_mpo, 1), R[L][0,D-1,0] = 1
    {
        std::vector<Scalar> h_R(D_mpo_, Traits::zero());
        h_R[D_mpo_actual_ - 1] = Traits::one();
        HIP_CHECK(hipMemcpy(d_R_envs_[L_], h_R.data(),
                            D_mpo_ * sizeof(Scalar), hipMemcpyHostToDevice));
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
    t_lanczos_.begin(stream_);
    int n = theta_size;
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
        apply_heff_two_site(site, d_vi, d_heff_result_);

        // Switch to device pointer mode for scalar operations
        ROCBLAS_CHECK(rocblas_set_pointer_mode(rocblas_h_, rocblas_pointer_mode_device));

        // alpha_i = <v_i|w> → device
        ROCBLAS_CHECK(Traits::dot(rocblas_h_, n, d_vi, 1, d_heff_result_, 1, d_dot_result_));

        hipLaunchKernelGGL(lanczos_process_alpha_kernel<Scalar>, dim3(1), dim3(1), 0, stream_,
                           d_dot_result_, d_neg_alpha_, d_alpha_dev_, iter);

        if (opts_.fuse_lanczos) {
            const Scalar* v_im1 = (iter > 0) ? (d_lanczos_v + (size_t)(iter - 1) * n) : nullptr;
            const Scalar* nb_im1 = (iter > 0) ? (d_neg_beta_scalars_ + (iter - 1)) : d_neg_alpha_;
            int block = 256, grid = (n + block - 1) / block;
            hipLaunchKernelGGL((lanczos_fused_sub_kernel<Scalar>),
                               dim3(grid), dim3(block), 0, stream_,
                               d_heff_result_, d_vi, v_im1,
                               d_neg_alpha_, nb_im1, n, iter > 0 ? 1 : 0);
        } else {
            ROCBLAS_CHECK(Traits::axpy(rocblas_h_, n, d_neg_alpha_, d_vi, 1, d_heff_result_, 1));
            if (iter > 0) {
                ROCBLAS_CHECK(Traits::axpy(rocblas_h_, n,
                    d_neg_beta_scalars_ + (iter - 1),
                    d_lanczos_v + (size_t)(iter - 1) * n, 1,
                    d_heff_result_, 1));
            }
        }

        // Full reorthogonalization (device pointer mode)
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

        hipLaunchKernelGGL(lanczos_process_beta_kernel<Scalar>, dim3(1), dim3(1), 0, stream_,
                           d_nrm2_result_, d_inv_nrm_, d_beta_dev_, d_neg_beta_scalars_, iter);

        // v_{i+1} = w / beta_i
        if (iter + 1 < max_iter) {
            Scalar* d_vip1 = d_lanczos_v + (size_t)(iter + 1) * n;
            if (opts_.fuse_lanczos) {
                int block = 256, grid = (n + block - 1) / block;
                hipLaunchKernelGGL((lanczos_fused_norm_copy_kernel<Scalar, RealType>),
                                   dim3(grid), dim3(block), 0, stream_,
                                   d_vip1, d_heff_result_, d_inv_nrm_, n);
            } else {
                HIP_CHECK(hipMemcpyAsync(d_vip1, d_heff_result_, n * sizeof(Scalar),
                                         hipMemcpyDeviceToDevice, stream_));
                ROCBLAS_CHECK(Traits::scal_real(rocblas_h_, n, d_inv_nrm_, d_vip1, 1));
            }
        }

        // Switch back to host pointer mode
        ROCBLAS_CHECK(rocblas_set_pointer_mode(rocblas_h_, rocblas_pointer_mode_host));

        // Convergence check every 3 iterations after iter >= 4
        if (iter >= 4 && iter % 3 == 0) {
            HIP_CHECK(hipStreamSynchronize(stream_));

            int ncheck = iter + 1;

            // Check for invariant subspace (beta < tol) on GPU
            hipLaunchKernelGGL(lanczos_check_beta, dim3(1), dim3(1), 0, stream_,
                               d_beta_dev_, ncheck, tol_lanczos, d_steqr_info_);
            rocblas_int h_beta_idx;
            HIP_CHECK(hipMemcpy(&h_beta_idx, d_steqr_info_, sizeof(rocblas_int), hipMemcpyDeviceToHost));
            if (h_beta_idx > 0) { iter = h_beta_idx; break; }

            HIP_CHECK(hipMemcpyAsync(d_steqr_D_, d_alpha_dev_, ncheck * sizeof(double), hipMemcpyDeviceToDevice, stream_));
            HIP_CHECK(hipMemcpyAsync(d_steqr_E_, d_beta_dev_, ncheck * sizeof(double), hipMemcpyDeviceToDevice, stream_));
            rocsolver_dsteqr(rocblas_h_, rocblas_evect_none, ncheck,
                             d_steqr_D_, d_steqr_E_, nullptr, ncheck, d_steqr_info_);
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
    rocsolver_dsteqr(rocblas_h_, rocblas_evect_tridiagonal, niter,
                     d_steqr_D_, d_steqr_E_, d_steqr_C_, niter, d_steqr_info_);

    rocblas_int h_steqr_info;
    HIP_CHECK(hipMemcpy(&h_steqr_info, d_steqr_info_, sizeof(rocblas_int), hipMemcpyDeviceToHost));
    if (h_steqr_info != 0) {
        throw std::runtime_error("rocsolver_dsteqr failed with info = " + std::to_string(h_steqr_info));
    }

    double energy;
    HIP_CHECK(hipMemcpy(&energy, d_steqr_D_, sizeof(double), hipMemcpyDeviceToHost));

    if constexpr (std::is_same_v<Scalar, double>) {
        HIP_CHECK(hipMemcpyAsync(d_ritz_coeffs_, d_steqr_C_, niter * sizeof(double), hipMemcpyDeviceToDevice, stream_));
    } else {
        // Complex case: promote double eigenvectors to complex on GPU
        int blk = (niter + 63) / 64;
        hipLaunchKernelGGL(promote_double_to_complex, dim3(blk), dim3(64), 0, stream_,
                           d_steqr_C_, (hipDoubleComplex*)d_ritz_coeffs_, niter);
    }

    // Use device pointer mode for finalization to avoid implicit GPU syncs
    ROCBLAS_CHECK(rocblas_set_pointer_mode(rocblas_h_, rocblas_pointer_mode_device));
    ROCBLAS_CHECK(Traits::gemv(
        rocblas_h_, rocblas_operation_none,
        n, niter, d_const_one_,
        d_lanczos_v, n,
        d_ritz_coeffs_, 1,
        d_const_zero_, d_theta, 1
    ));

    ROCBLAS_CHECK(Traits::nrm2(rocblas_h_, n, d_theta, 1, d_nrm2_result_));
    hipLaunchKernelGGL(invert_nrm_kernel<RealType>, dim3(1), dim3(1), 0, 0,
                       d_nrm2_result_, d_inv_nrm_);
    ROCBLAS_CHECK(Traits::scal_real(rocblas_h_, n, d_inv_nrm_, d_theta, 1));
    ROCBLAS_CHECK(rocblas_set_pointer_mode(rocblas_h_, rocblas_pointer_mode_host));

    t_lanczos_.end(stream_);
    return energy;
}

// ============================================================================
// SVD split for two-site DMRG
// ============================================================================

template<typename Scalar>
void DMRG2GPU<Scalar>::svd_split(int site, Scalar* d_theta, char direction) {
    t_svd_.begin(stream_);
    int cL = chi_L(site);
    int cR = chi_R(site + 1);

    // theta is (cL, d, d, cR) reshaped as (cL*d, d*cR) for SVD
    int m = cL * d_;
    int n_svd = d_ * cR;
    int full_k = std::min(m, n_svd);
    int k = std::min(full_k, chi_max_);

    // Choose between full SVD (rocsolver_gesvd_auto) and RSVD (Halko–
    // Martinsson–Tropp). RSVD is profitable when the matrix has many more
    // singular values than we want to keep. In 2-site DMRG,
    //   full_k = d * min(cL, cR)
    // which can comfortably exceed chi_max + oversample — unlike 1-site,
    // where full_k ≤ chi_max and RSVD is a net loss.
    int vh_lda = full_k;
    int svd_k  = full_k;
    bool used_rsvd = opts_.rsvd
                  && full_k > k + RSVD_OVERSAMPLE_
                  && m > 2 * k;

    if (!used_rsvd) {
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
    } else {
        int r_use = std::min({k + RSVD_OVERSAMPLE_, full_k, rsvd_r_max_});

        // Ω ∈ C^{n_svd x r_use}, fresh per call
        {
            std::vector<Scalar> h_omega((size_t)n_svd * r_use);
            for (size_t i = 0; i < h_omega.size(); i++) {
                h_omega[i] = Traits::random_val();
            }
            HIP_CHECK(hipMemcpyAsync(d_rsvd_omega_, h_omega.data(),
                h_omega.size() * sizeof(Scalar), hipMemcpyHostToDevice, stream_));
        }

        Scalar one = Traits::one(), zero_val = Traits::zero();

        // Y = A · Ω  → (m x r_use)
        ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
            rocblas_operation_none, rocblas_operation_none,
            m, r_use, n_svd, &one,
            d_theta, m,
            d_rsvd_omega_, n_svd,
            &zero_val,
            d_rsvd_Y_, m));

        // Q = qr(Y) in-place
        ROCBLAS_CHECK(Traits::rocsolver_geqrf(rocblas_h_, m, r_use,
            d_rsvd_Y_, m, d_rsvd_tau_));
        ROCBLAS_CHECK(Traits::rocsolver_orgqr(rocblas_h_, m, r_use, r_use,
            d_rsvd_Y_, m, d_rsvd_tau_));

        // B = Q^H · A  → (r_use x n_svd)
        ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
            Traits::op_h, rocblas_operation_none,
            r_use, n_svd, m, &one,
            d_rsvd_Y_, m,
            d_theta, m,
            &zero_val,
            d_rsvd_B_, r_use));

        // SVD(B) on GPU → U_small, S, Vh
        int b_k = std::min(r_use, n_svd);
        Traits::rocsolver_gesvd_auto(rocblas_h_,
            rocblas_svect_singular, rocblas_svect_singular,
            r_use, n_svd,
            d_rsvd_B_, r_use,
            d_svd_S_,
            d_rsvd_U_small_, r_use,
            d_svd_Vh_, b_k,
            d_svd_E_,
            d_svdj_residual_, d_svdj_n_sweeps_,
            d_svd_info_);

        // U = Q · U_small  → (m x b_k), written into d_svd_U_ with lda = m
        ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
            rocblas_operation_none, rocblas_operation_none,
            m, b_k, r_use, &one,
            d_rsvd_Y_, m,
            d_rsvd_U_small_, r_use,
            &zero_val,
            d_svd_U_, m));

        vh_lda = b_k;
        svd_k  = b_k;
    }

    // Cap truncation target at the number of singular pairs the SVD stage
    // actually produced (svd_k == full_k for full SVD; svd_k == b_k for RSVD).
    int k_target = std::min(k, svd_k);
    int new_k;
    if (opts_.device_k) {
        new_k = k_target;
    } else {
        hipLaunchKernelGGL(svd_truncate_kernel<RealType>, dim3(1), dim3(1), 0, stream_,
                           d_svd_S_, k_target, 1e-14, d_svd_info_);
        HIP_CHECK(hipMemcpy(&new_k, d_svd_info_, sizeof(int), hipMemcpyDeviceToHost));
    }

    int threads = 256;

    if (direction == 'R') {
        // U -> MPS[site] (left-canonical), S*Vh -> MPS[site+1]
        allocate_mps_tensor(site, cL, new_k);
        if (new_k == svd_k) {
            HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site], d_svd_U_,
                                     m * new_k * sizeof(Scalar),
                                     hipMemcpyDeviceToDevice, stream_));
        } else {
            int total = m * new_k;
            hipLaunchKernelGGL(extract_cols_kernel<Scalar>, dim3((total+threads-1)/threads), dim3(threads), 0, stream_,
                               d_svd_U_, m, d_mps_tensors_[site], m, m, new_k);
        }

        // S*Vh — Vh has leading dim vh_lda (full_k or b_k)
        allocate_mps_tensor(site + 1, new_k, cR);
        {
            int total = new_k * n_svd;
            hipLaunchKernelGGL((scale_rows_by_diag_kernel<Scalar, RealType>), dim3((total+threads-1)/threads), dim3(threads), 0, stream_,
                               d_svd_S_, d_svd_Vh_, vh_lda, d_mps_tensors_[site + 1], new_k, new_k, n_svd);
        }

    } else {  // direction == 'L'
        allocate_mps_tensor(site, cL, new_k);
        {
            int total = m * new_k;
            hipLaunchKernelGGL((scale_cols_by_diag_kernel<Scalar, RealType>), dim3((total+threads-1)/threads), dim3(threads), 0, stream_,
                               d_svd_S_, d_svd_U_, m, d_mps_tensors_[site], m, m, new_k);
        }

        allocate_mps_tensor(site + 1, new_k, cR);
        if (new_k == svd_k && vh_lda == new_k) {
            HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site + 1], d_svd_Vh_,
                                     (size_t)vh_lda * n_svd * sizeof(Scalar),
                                     hipMemcpyDeviceToDevice, stream_));
        } else {
            int total = new_k * n_svd;
            hipLaunchKernelGGL(extract_cols_kernel<Scalar>, dim3((total+threads-1)/threads), dim3(threads), 0, stream_,
                               d_svd_Vh_, vh_lda, d_mps_tensors_[site + 1], new_k, new_k, n_svd);
        }
    }

    bond_dims_[site + 1] = new_k;
    t_svd_.end(stream_);
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
    // Timer starts BEFORE env build — includes env build in total (timer_scope=include_env_build)
    auto t_start = std::chrono::high_resolution_clock::now();

    build_initial_environments();

    auto t_envs = std::chrono::high_resolution_clock::now();
    double env_time = std::chrono::duration<double>(t_envs - t_start).count();
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
    printf("  env_build_sec: %.3f  timer_scope: include_env_build\n", env_time);
    report_timers();

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
        HIP_CHECK(hipMemcpy(h_mps[i].data(), d_mps_tensors_[i],
                            size * sizeof(Scalar), hipMemcpyDeviceToHost));
    }
}

#endif // DMRG2_GPU_IMPL_H
