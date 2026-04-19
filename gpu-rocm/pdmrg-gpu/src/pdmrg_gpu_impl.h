#ifndef PDMRG_GPU_IMPL_H
#define PDMRG_GPU_IMPL_H

#include <rocsolver/rocsolver.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <stdexcept>
#include "accurate_svd.h"

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
// GPU kernels for batched GEMM pointer setup (eliminates CPU loops + H2D copies)
// ============================================================================

// apply_heff_single_site Step 3: per wp iteration, d pointers
// A[s] = base_U + (wp*d + s) * stride_A,  B[s] = R_env + wp * stride_B,  C[s] = result + s * stride_C
template<typename Scalar>
__global__ void setup_heff_ss_step3_ptrs(Scalar** A, Scalar** B, Scalar** C,
                                          Scalar* base_U, Scalar* base_R, Scalar* base_result,
                                          int wp, int d, int strideA, int strideB, int strideC) {
    int s = threadIdx.x;
    if (s < d) {
        A[s] = base_U + (wp * d + s) * strideA;
        B[s] = base_R + wp * strideB;
        C[s] = base_result + s * strideC;
    }
}

// apply_heff_single_site Step 3 (R3-F1 full-batched): single launch for all D*d tiles.
// idx = wp*d + sp. Each wp writes to its own scratch slice in V (free after Step 2);
// slices are reduced via a rocblas_gemv afterwards.
//   A[idx] -> U[wp*d + sp]  (cL x cR tile, lda = cL*cR)
//   B[idx] -> R[wp]          (cR x cR block, ldb = cR*D)
//   C[idx] -> scratch + wp*slice_stride + sp*strideC_tile  (ldc = cL*d)
template<typename Scalar>
__global__ void setup_heff_ss_step3_full_ptrs(Scalar** A, Scalar** B, Scalar** C,
                                               Scalar* base_U, Scalar* base_R, Scalar* base_C_scratch,
                                               int d, int strideA, int strideB,
                                               int strideC_tile, int slice_stride) {
    int idx = threadIdx.x;  // wp*d + sp
    int wp = idx / d;
    int sp = idx % d;
    A[idx] = base_U + (wp * d + sp) * strideA;
    B[idx] = base_R + wp * strideB;
    C[idx] = base_C_scratch + wp * slice_stride + sp * strideC_tile;
}

// apply_heff_two_site Step 3 (R3-F1 full-batched): single launch for all D*dd tiles.
// idx = n*dd + (s1p*d + s2p). Replaces the unbatched triple loop in
// apply_heff_two_site. Each n writes to its own scratch slice in T1; slices
// are reduced via a rocblas_gemv afterwards.
//   A[idx] -> T2[n*dd + ss]   (cL x cR tile, lda = cL*cR)
//   B[idx] -> R[n]             (cR x cR block, ldb = cR*D)
//   C[idx] -> scratch + n*slice_stride + (s1p + s2p*d)*strideC_tile (ldc = cL*dd)
template<typename Scalar>
__global__ void setup_heff_ts_step3_full_ptrs(Scalar** A, Scalar** B, Scalar** C,
                                               Scalar* base_T2, Scalar* base_R, Scalar* base_C_scratch,
                                               int d, int dd, int strideA, int strideB,
                                               int strideC_tile, int slice_stride) {
    int idx = threadIdx.x;  // n*dd + s1p*d + s2p
    int n  = idx / dd;
    int ss = idx % dd;
    int s1p = ss / d, s2p = ss % d;
    A[idx] = base_T2 + (n * dd + ss) * strideA;
    B[idx] = base_R + n * strideB;
    C[idx] = base_C_scratch + n * slice_stride + (s1p + s2p * d) * strideC_tile;
}

// SPARSE_MPO setup kernels (same layout as their dense counterparts but
// indexed via a precomputed list of nonzero packed (w*d+s) / (w*dd+ss) /
// (wp*d+sp) / (n*dd+s1p*d+s2p) positions). Skipped slots must be pre-zeroed
// by the caller (hipMemsetAsync on V / scratch) so Step 2's dense GEMM reads
// a correct layout and the Step-3 GEMV reduction doesn't pick up stale data.

// Sparse single-site Step 1: lays out L_env and theta pointers on nnz rows.
template<typename Scalar>
__global__ void setup_batch_ptrs_wd_sparse(Scalar** A, Scalar** B, Scalar** C,
                                           Scalar* base_A, Scalar* base_B, Scalar* base_C,
                                           const int* nnz_ws, int d,
                                           int strideA, int strideB, int strideC) {
    int idx = threadIdx.x;
    int ws = nnz_ws[idx];
    int w = ws / d;
    int s = ws % d;
    A[idx] = base_A + w * strideA;
    B[idx] = base_B + s * strideB;
    C[idx] = base_C + ws * strideC;
}

// Sparse single-site Step 3-full: indexes nnz (wp, sp) columns of W_left.
template<typename Scalar>
__global__ void setup_batch_ptrs_step3_full_sparse(Scalar** A, Scalar** B, Scalar** C,
                                                   Scalar* base_A, Scalar* base_B, Scalar* base_C_scratch,
                                                   const int* nnz_wpsp, int d,
                                                   int strideA, int strideB,
                                                   int strideC_tile, int slice_stride) {
    int idx = threadIdx.x;
    int wpsp = nnz_wpsp[idx];
    int wp = wpsp / d;
    int sp = wpsp % d;
    A[idx] = base_A + wpsp * strideA;
    B[idx] = base_B + wp * strideB;
    C[idx] = base_C_scratch + wp * slice_stride + sp * strideC_tile;
}

// Sparse two-site Step 1: A[idx] = L_env + w*strideA, B[idx] = theta offset
// for (s1, s2), C[idx] = T1 + packed*strideC. packed = w*dd + s1*d + s2.
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

// Sparse two-site Step 3-full: indexes nnz (n, s1p, s2p) columns of WW.
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

// update_left_env Step 3: per sp iteration, D pointers
// A[wp] = U + (wp*d + sp) * stride_A,  B[wp] = A_mps + sp * stride_B,  C[wp] = L_new + wp * stride_C
template<typename Scalar>
__global__ void setup_lenv_step3_ptrs(Scalar** dA, Scalar** dB, Scalar** dC,
                                       Scalar* base_U, Scalar* base_A, Scalar* base_L,
                                       int sp, int d, int D, int strideA, int strideB, int strideC) {
    int wp = threadIdx.x;
    if (wp < D) {
        dA[wp] = base_U + (wp * d + sp) * strideA;
        dB[wp] = base_A + sp * strideB;
        dC[wp] = base_L + wp * strideC;
    }
}

// update_right_env Step 3: per sp iteration, D pointers
// A[w] = U + (w*d + sp) * stride_A,  B[w] = A_mps + sp * stride_B,  C[w] = R_new + w * stride_C
template<typename Scalar>
__global__ void setup_renv_step3_ptrs(Scalar** dA, Scalar** dB, Scalar** dC,
                                       Scalar* base_U, Scalar* base_A, Scalar* base_R,
                                       int sp, int d, int D, int strideA, int strideB, int strideC) {
    int w = threadIdx.x;
    if (w < D) {
        dA[w] = base_U + (w * d + sp) * strideA;
        dB[w] = base_A + sp * strideB;
        dC[w] = base_R + w * strideC;
    }
}

// Lanczos: find first beta < tol, write index+1 to result (0 = none found)
static __global__ void lanczos_check_beta(const double* beta, int n, double tol, rocblas_int* result) {
    *result = 0;
    for (int j = 0; j < n; j++) {
        if (beta[j] < tol) { *result = j + 1; return; }
    }
}

// Fused Lanczos update:  w := w + (-α)*v_i + [(-β_{im1})*v_{im1}]
// d_neg_alpha and d_neg_beta_im1 are Scalar* on device (device-pointer mode).
// v_im1 may be nullptr when has_prev is false (iter==0 path).
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

// Promote double eigenvector to hipDoubleComplex (for Josephson Ritz coefficients)
static __global__ void promote_double_to_complex(const double* src, hipDoubleComplex* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = make_hipDoubleComplex(src[i], 0.0);
}

// ============================================================================
// Constructor
// ============================================================================

template<typename Scalar>
PDMRGGPU<Scalar>::PDMRGGPU(int L, int d, int chi_max, int D_mpo, int n_segments, double tol)
    : L_(L), d_(d), chi_max_(chi_max), D_mpo_(D_mpo), D_mpo_actual_(D_mpo),
      tol_(tol), energy_(0.0),
      n_segments_(n_segments) {

    opts_.load_from_env();

    // D_PAD: round MPO bond dim up to a multiple of 8 for MFMA-friendly
    // GEMM shapes. All allocations and internal GEMMs use the padded D;
    // the padded rows/cols of the W matrices are zero-filled in set_mpo
    // so they contribute nothing numerically. The R boundary still uses
    // D_mpo_actual_-1 (see build_initial_environments).
    if (opts_.d_pad) {
        int padded = (D_mpo_ + 7) & ~7;
        if (padded != D_mpo_) {
            std::fprintf(stderr, "[D_PAD] D_mpo padded: %d -> %d\n", D_mpo_, padded);
            D_mpo_ = padded;
        }
    }

    opts_.print(stderr);
    init_timers();

    if (L < 2 * n_segments) {
        throw std::runtime_error("Need at least 2 sites per segment: L >= 2*n_segments");
    }

    // Bond dimensions (min-cut formula capped at chi_max)
    bond_dims_.resize(L + 1);
    bond_dims_[0] = 1;
    bond_dims_[L] = 1;
    for (int i = 1; i < L; i++) {
        double exact_dim = pow((double)d_, std::min(i, L - i));
        bond_dims_[i] = (exact_dim > chi_max_) ? chi_max_ : (int)exact_dim;
    }

    // Partition chain
    partition_chain();
    initialize_boundary_states();

    // Create streams and rocBLAS handles (one per segment)
    streams_.resize(n_segments_);
    handles_.resize(n_segments_);
    for (int k = 0; k < n_segments_; k++) {
        HIP_CHECK(hipStreamCreate(&streams_[k]));
        ROCBLAS_CHECK(rocblas_create_handle(&handles_[k]));
        ROCBLAS_CHECK(rocblas_set_stream(handles_[k], streams_[k]));
    }

    int dd = d_ * d_;

    // MPS tensors
    d_mps_tensors_.resize(L, nullptr);
    for (int i = 0; i < L; i++) {
        allocate_mps_tensor(i, chi_L(i), chi_R(i));
    }

    // MPO tensors
    d_mpo_tensors_.resize(L, nullptr);

    // W matrices for single-site env updates
    d_W_left_.resize(L, nullptr);
    d_W_right_.resize(L, nullptr);

    // Fused two-site MPO
    d_WW_.resize(L - 1, nullptr);

    // SPARSE_MPO nnz lists (populated in set_mpo / precompute_fused_mpo when
    // opts_.sparse_mpo is on). Class-level, shared across all segments.
    d_WL_nnz_rows_.resize(L, nullptr);
    d_WL_nnz_cols_.resize(L, nullptr);
    wl_nnz_rows_count_.assign(L, 0);
    wl_nnz_cols_count_.assign(L, 0);
    d_WW_nnz_rows_.resize(L - 1, nullptr);
    d_WW_nnz_cols_.resize(L - 1, nullptr);
    ww_nnz_rows_count_.assign(L - 1, 0);
    ww_nnz_cols_count_.assign(L - 1, 0);

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

    // Allocate per-stream workspaces
    theta_size_max_ = chi_max_ * dd * chi_max_;
    max_lanczos_iter_ = std::min(100, theta_size_max_);
    use_cpu_svd_ = false;
    // RSVD: the DMRG_GPU_OPT_RSVD env var controls this (use_rsvd_ can still
    // be overridden at runtime via set_rsvd() if the caller prefers).
    use_rsvd_ = opts_.rsvd;
    lanczos_use_1site_ = false;
    rsvd_oversampling_ = 20;

    allocate_stream_workspaces();
}

// ============================================================================
// Per-stream workspace allocation
// ============================================================================

template<typename Scalar>
void PDMRGGPU<Scalar>::allocate_stream_workspaces() {
    int dd = d_ * d_;
    int t_max = D_mpo_ * dd * chi_max_ * chi_max_;
    int batch_max = D_mpo_ * dd;
    int svd_max_m = chi_max_ * d_;
    int svd_max_n = d_ * chi_max_;
    int svd_max_k = std::min(svd_max_m, svd_max_n);

    workspaces_.resize(n_segments_);

    for (int k = 0; k < n_segments_; k++) {
        auto& ws = workspaces_[k];

        // Contraction intermediates
        HIP_CHECK(hipMalloc(&ws.d_T1, t_max * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&ws.d_T2, t_max * sizeof(Scalar)));

        // Lanczos workspace
        HIP_CHECK(hipMalloc(&ws.d_theta, theta_size_max_ * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&ws.d_heff_result, theta_size_max_ * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&ws.d_lanczos_v, (size_t)max_lanczos_iter_ * theta_size_max_ * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&ws.d_ritz_coeffs, max_lanczos_iter_ * sizeof(Scalar)));

        // Batched GEMM pointer arrays (device)
        HIP_CHECK(hipMalloc(&ws.d_batch_A, batch_max * sizeof(Scalar*)));
        HIP_CHECK(hipMalloc(&ws.d_batch_B, batch_max * sizeof(Scalar*)));
        HIP_CHECK(hipMalloc(&ws.d_batch_C, batch_max * sizeof(Scalar*)));
        // Pinned host pointer arrays no longer needed — pointer setup done by GPU kernels
        ws.h_batch_A_pinned = nullptr;
        ws.h_batch_B_pinned = nullptr;
        ws.h_batch_C_pinned = nullptr;
        // Cached apply_heff A/C pointers (separate device arrays)
        HIP_CHECK(hipMalloc(&ws.d_heff_batch_A, batch_max * sizeof(Scalar*)));
        HIP_CHECK(hipMalloc(&ws.d_heff_batch_C, batch_max * sizeof(Scalar*)));
        ws.heff_cached_site = -1;

        // Lanczos device-pointer-mode scalars
        HIP_CHECK(hipMalloc(&ws.d_dot_result, sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&ws.d_nrm2_result, sizeof(RealType)));
        HIP_CHECK(hipMalloc(&ws.d_neg_alpha, sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&ws.d_neg_overlap, sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&ws.d_inv_nrm, sizeof(RealType)));
        HIP_CHECK(hipMalloc(&ws.d_alpha_dev, max_lanczos_iter_ * sizeof(RealType)));
        HIP_CHECK(hipMalloc(&ws.d_beta_dev, max_lanczos_iter_ * sizeof(RealType)));
        HIP_CHECK(hipMalloc(&ws.d_neg_beta_scalars, max_lanczos_iter_ * sizeof(Scalar)));

        // rocsolver tridiagonal eigensolver workspace
        HIP_CHECK(hipMalloc(&ws.d_steqr_D, max_lanczos_iter_ * sizeof(double)));
        HIP_CHECK(hipMalloc(&ws.d_steqr_E, max_lanczos_iter_ * sizeof(double)));
        HIP_CHECK(hipMalloc(&ws.d_steqr_C, max_lanczos_iter_ * max_lanczos_iter_ * sizeof(double)));
        HIP_CHECK(hipMalloc(&ws.d_steqr_info, sizeof(rocblas_int)));

        HIP_CHECK(hipMalloc(&ws.d_const_one, sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&ws.d_const_zero, sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&ws.d_const_neg_one, sizeof(Scalar)));
        {
            Scalar h_one = Traits::one(), h_zero = Traits::zero();
            Scalar h_neg_one = Traits::neg(Traits::one());
            HIP_CHECK(hipMemcpy(ws.d_const_one, &h_one, sizeof(Scalar), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(ws.d_const_zero, &h_zero, sizeof(Scalar), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(ws.d_const_neg_one, &h_neg_one, sizeof(Scalar), hipMemcpyHostToDevice));
        }

        // Length-D ones vector for Step-3 GEMV reduction (R3-F1)
        HIP_CHECK(hipMalloc(&ws.d_ones_D, D_mpo_ * sizeof(Scalar)));
        {
            std::vector<Scalar> h_ones(D_mpo_, Traits::one());
            HIP_CHECK(hipMemcpy(ws.d_ones_D, h_ones.data(),
                                D_mpo_ * sizeof(Scalar), hipMemcpyHostToDevice));
        }

        // GPU SVD workspace
        HIP_CHECK(hipMalloc(&ws.d_svd_A, theta_size_max_ * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&ws.d_svd_U, (size_t)svd_max_m * svd_max_k * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&ws.d_svd_S, svd_max_k * sizeof(RealType)));
        HIP_CHECK(hipMalloc(&ws.d_svd_Vh, (size_t)svd_max_k * svd_max_n * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&ws.d_svd_E, svd_max_k * sizeof(RealType)));
        HIP_CHECK(hipMalloc(&ws.d_svd_info, sizeof(int)));
        HIP_CHECK(hipMalloc(&ws.d_svd_work, theta_size_max_ * sizeof(Scalar)));
        // R3-F2: device scalars for gesvdj.
        HIP_CHECK(hipMalloc(&ws.d_svdj_residual, sizeof(double)));
        HIP_CHECK(hipMalloc(&ws.d_svdj_n_sweeps, sizeof(rocblas_int)));

        // CPU SVD workspace
        ws.h_svd_A.resize(theta_size_max_);
        ws.h_svd_U.resize((size_t)svd_max_m * svd_max_k);
        ws.h_svd_S.resize(svd_max_k);
        ws.h_svd_Vh.resize((size_t)svd_max_k * svd_max_n);
        ws.h_svd_tmp.resize(std::max((size_t)svd_max_m * svd_max_k, (size_t)svd_max_k * svd_max_n));
        ws.h_svd_rwork.resize(Traits::svd_rwork_size(svd_max_m, svd_max_n));

        // Query optimal LAPACK workspace
        {
            int m = svd_max_m, n = svd_max_n;
            int lwork_query = -1;
            Scalar work_opt;
            int info;
            const char jobu = 'S', jobvt = 'S';
            Traits::lapack_gesvd(&jobu, &jobvt, &m, &n, nullptr, &m, nullptr,
                    nullptr, &m, nullptr, &svd_max_k, &work_opt, &lwork_query,
                    ws.h_svd_rwork.empty() ? nullptr : ws.h_svd_rwork.data(), &info);
            int opt_size;
            if constexpr (Traits::is_complex) {
                opt_size = (int)Traits::real_part(work_opt) + 1;
            } else {
                opt_size = (int)work_opt + 1;
            }
            ws.h_svd_work.resize(opt_size);
        }

        // Randomized truncated SVD workspace (GPU QR)
        {
            int rsvd_r = chi_max_ + rsvd_oversampling_;
            int rsvd_m = svd_max_m;  // chi*d
            int rsvd_n = svd_max_n;  // d*chi
            ws.d_rsvd_omega = nullptr;
            ws.d_rsvd_Y = nullptr;
            ws.d_rsvd_Q = nullptr;
            ws.d_rsvd_B = nullptr;
            ws.d_rsvd_ipiv = nullptr;
            ws.d_rsvd_U_full = nullptr;
            HIP_CHECK(hipMalloc(&ws.d_rsvd_omega,  (size_t)rsvd_n * rsvd_r * sizeof(Scalar)));
            HIP_CHECK(hipMalloc(&ws.d_rsvd_Y,     (size_t)rsvd_m * rsvd_r * sizeof(Scalar)));
            HIP_CHECK(hipMalloc(&ws.d_rsvd_Q,     (size_t)rsvd_m * rsvd_r * sizeof(Scalar)));
            HIP_CHECK(hipMalloc(&ws.d_rsvd_B,     (size_t)rsvd_r * rsvd_n * sizeof(Scalar)));
            HIP_CHECK(hipMalloc(&ws.d_rsvd_ipiv,  (size_t)rsvd_r * sizeof(Scalar)));
            HIP_CHECK(hipMalloc(&ws.d_rsvd_U_full,(size_t)rsvd_m * rsvd_r * sizeof(Scalar)));
            ws.h_rsvd_B.resize((size_t)rsvd_r * rsvd_n);
            ws.h_rsvd_U_small.resize((size_t)rsvd_r * rsvd_r);

            // Pre-allocated Vh buffer for boundary merge R_env swap (avoids hot-path hipMalloc)
            // Max size = chi_max*d rows × d*chi_max cols = theta_size_max
            HIP_CHECK(hipMalloc(&ws.d_Vh_canonical, theta_size_max_ * sizeof(Scalar)));

            // LANCZOS_GRAPH: per-segment bounce buffer for captured apply_heff graphs.
            ws.d_heff_input = nullptr;
            if (opts_.lanczos_graph) {
                HIP_CHECK(hipMalloc(&ws.d_heff_input, (size_t)theta_size_max_ * sizeof(Scalar)));
            }

            // Query SVD workspace for the smaller matrix (rsvd_r x rsvd_n)
            {
                int sm = rsvd_r, sn = rsvd_n;
                int sk = std::min(sm, sn);
                int svd_lwork_query = -1;
                Scalar svd_work_opt;
                int svd_info;
                const char jobu = 'S', jobvt = 'S';
                Traits::lapack_gesvd(&jobu, &jobvt, &sm, &sn, nullptr, &sm, nullptr,
                        nullptr, &sm, nullptr, &sk, &svd_work_opt, &svd_lwork_query,
                        ws.h_svd_rwork.empty() ? nullptr : ws.h_svd_rwork.data(), &svd_info);
                int svd_opt;
                if constexpr (Traits::is_complex) {
                    svd_opt = (int)Traits::real_part(svd_work_opt) + 1;
                } else {
                    svd_opt = (int)svd_work_opt + 1;
                }
                // Ensure h_svd_work is large enough for both full and reduced SVD
                if ((int)ws.h_svd_work.size() < svd_opt) {
                    ws.h_svd_work.resize(svd_opt);
                }
            }
        }
    }
}

// ============================================================================
// Destructor
// ============================================================================

template<typename Scalar>
PDMRGGPU<Scalar>::~PDMRGGPU() {
    free_gpu_resources();
}

template<typename Scalar>
void PDMRGGPU<Scalar>::free_gpu_resources() {
    for (auto ptr : d_mps_tensors_) if (ptr) hipFree(ptr);
    for (auto ptr : d_mpo_tensors_) if (ptr) hipFree(ptr);
    for (auto ptr : d_W_left_) if (ptr) hipFree(ptr);
    for (auto ptr : d_W_right_) if (ptr) hipFree(ptr);
    for (auto ptr : d_WW_) if (ptr) hipFree(ptr);
    for (auto ptr : d_WL_nnz_rows_) if (ptr) hipFree(ptr);
    for (auto ptr : d_WL_nnz_cols_) if (ptr) hipFree(ptr);
    for (auto ptr : d_WW_nnz_rows_) if (ptr) hipFree(ptr);
    for (auto ptr : d_WW_nnz_cols_) if (ptr) hipFree(ptr);
    for (auto ptr : d_L_envs_) if (ptr) hipFree(ptr);
    for (auto ptr : d_R_envs_) if (ptr) hipFree(ptr);
    for (auto& ws : workspaces_) {
        if (ws.d_theta) hipFree(ws.d_theta);
        if (ws.d_heff_result) hipFree(ws.d_heff_result);
        if (ws.d_T1) hipFree(ws.d_T1);
        if (ws.d_T2) hipFree(ws.d_T2);
        if (ws.d_lanczos_v) hipFree(ws.d_lanczos_v);
        if (ws.d_ritz_coeffs) hipFree(ws.d_ritz_coeffs);
        if (ws.d_batch_A) hipFree(ws.d_batch_A);
        if (ws.d_batch_B) hipFree(ws.d_batch_B);
        if (ws.d_batch_C) hipFree(ws.d_batch_C);
        // h_batch_*_pinned no longer allocated (GPU kernel pointer setup)
        if (ws.d_heff_batch_A) hipFree(ws.d_heff_batch_A);
        if (ws.d_heff_batch_C) hipFree(ws.d_heff_batch_C);
        if (ws.d_dot_result) hipFree(ws.d_dot_result);
        if (ws.d_nrm2_result) hipFree(ws.d_nrm2_result);
        if (ws.d_neg_alpha) hipFree(ws.d_neg_alpha);
        if (ws.d_neg_overlap) hipFree(ws.d_neg_overlap);
        if (ws.d_inv_nrm) hipFree(ws.d_inv_nrm);
        if (ws.d_alpha_dev) hipFree(ws.d_alpha_dev);
        if (ws.d_beta_dev) hipFree(ws.d_beta_dev);
        if (ws.d_neg_beta_scalars) hipFree(ws.d_neg_beta_scalars);
        if (ws.d_steqr_D) hipFree(ws.d_steqr_D);
        if (ws.d_steqr_E) hipFree(ws.d_steqr_E);
        if (ws.d_steqr_C) hipFree(ws.d_steqr_C);
        if (ws.d_steqr_info) hipFree(ws.d_steqr_info);
        if (ws.d_const_one) hipFree(ws.d_const_one);
        if (ws.d_const_zero) hipFree(ws.d_const_zero);
        if (ws.d_const_neg_one) hipFree(ws.d_const_neg_one);
        if (ws.d_ones_D) hipFree(ws.d_ones_D);
        if (ws.d_svd_A) hipFree(ws.d_svd_A);
        if (ws.d_svd_U) hipFree(ws.d_svd_U);
        if (ws.d_svd_S) hipFree(ws.d_svd_S);
        if (ws.d_svd_Vh) hipFree(ws.d_svd_Vh);
        if (ws.d_svd_E) hipFree(ws.d_svd_E);
        if (ws.d_svd_info) hipFree(ws.d_svd_info);
        if (ws.d_svd_work) hipFree(ws.d_svd_work);
        if (ws.d_svdj_residual) hipFree(ws.d_svdj_residual);
        if (ws.d_svdj_n_sweeps) hipFree(ws.d_svdj_n_sweeps);
        if (ws.d_rsvd_omega) hipFree(ws.d_rsvd_omega);
        if (ws.d_rsvd_Y) hipFree(ws.d_rsvd_Y);
        if (ws.d_rsvd_Q) hipFree(ws.d_rsvd_Q);
        if (ws.d_rsvd_B) hipFree(ws.d_rsvd_B);
        if (ws.d_rsvd_ipiv) hipFree(ws.d_rsvd_ipiv);
        if (ws.d_rsvd_U_full) hipFree(ws.d_rsvd_U_full);
        if (ws.d_Vh_canonical) hipFree(ws.d_Vh_canonical);

        // LANCZOS_GRAPH: destroy cached graph execs and bounce buffer
        for (auto& kv : ws.apply_heff_graph_cache) {
            hipGraphExecDestroy(kv.second);
        }
        ws.apply_heff_graph_cache.clear();
        if (ws.d_heff_input) hipFree(ws.d_heff_input);
    }

    for (auto& h : handles_) rocblas_destroy_handle(h);
    for (auto& s : streams_) hipStreamDestroy(s);
}

// ============================================================================
// Memory management
// ============================================================================

template<typename Scalar>
void PDMRGGPU<Scalar>::allocate_mps_tensor(int site, int cL, int cR) {
    size_t max_sz = (size_t)chi_max_ * d_ * chi_max_ * sizeof(Scalar);
    if (!d_mps_tensors_[site]) {
        HIP_CHECK(hipMalloc(&d_mps_tensors_[site], max_sz));
    }
    (void)cL; (void)cR;
}

template<typename Scalar>
void PDMRGGPU<Scalar>::ensure_L_env_alloc(int idx, int chi) {
    if (chi > L_env_alloc_chi_[idx]) {
        if (d_L_envs_[idx]) HIP_CHECK(hipFree(d_L_envs_[idx]));
        int sz = chi * D_mpo_ * chi;
        HIP_CHECK(hipMalloc(&d_L_envs_[idx], sz * sizeof(Scalar)));
        L_env_alloc_chi_[idx] = chi;
    }
}

template<typename Scalar>
void PDMRGGPU<Scalar>::ensure_R_env_alloc(int idx, int chi) {
    if (chi > R_env_alloc_chi_[idx]) {
        if (d_R_envs_[idx]) HIP_CHECK(hipFree(d_R_envs_[idx]));
        int sz = chi * D_mpo_ * chi;
        HIP_CHECK(hipMalloc(&d_R_envs_[idx], sz * sizeof(Scalar)));
        R_env_alloc_chi_[idx] = chi;
    }
}

// ============================================================================
// Chain partitioning
// ============================================================================

template<typename Scalar>
void PDMRGGPU<Scalar>::partition_chain() {
    seg_first_.resize(n_segments_);
    seg_last_.resize(n_segments_);
    boundary_bonds_.resize(n_segments_ - 1);

    int base = L_ / n_segments_;
    int remainder = L_ % n_segments_;

    int pos = 0;
    for (int k = 0; k < n_segments_; k++) {
        seg_first_[k] = pos;
        int seg_len = base + (k < remainder ? 1 : 0);
        seg_last_[k] = pos + seg_len - 1;
        pos += seg_len;
    }

    for (int k = 0; k < n_segments_ - 1; k++) {
        boundary_bonds_[k] = seg_last_[k];
    }
}

// ============================================================================
// Initialize boundary V = ones (before any merge, V is identity)
// ============================================================================

template<typename Scalar>
void PDMRGGPU<Scalar>::initialize_boundary_states() {
    int n_boundaries = n_segments_ - 1;
    boundary_states_.resize(n_boundaries);
    for (int b = 0; b < n_boundaries; b++) {
        int bsite = boundary_bonds_[b];
        int chi = bond_dims_[bsite + 1];  // bond between bsite and bsite+1
        boundary_states_[b].chi = chi;
        boundary_states_[b].V.assign(chi, RealType(1.0));  // V = ones initially
    }
}

// ============================================================================
// MPS initialization
// ============================================================================

template<typename Scalar>
void PDMRGGPU<Scalar>::initialize_mps_random(double scale) {
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

// ============================================================================
// MPO setup and fused two-site MPO precomputation
// ============================================================================

template<typename Scalar>
void PDMRGGPU<Scalar>::set_mpo(const std::vector<Scalar*>& h_mpo_tensors) {
    // D_use: padded bond dim used for all internal buffers (D_PAD on).
    // D_act: user's MPO bond dim. User's h_mpo_tensors[i] is indexed with
    // stride D_act; we re-index into the D_use layout on the host side,
    // zero-padding any w/wp slot with index >= D_act.
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

        // SPARSE_MPO: build nnz row/col index lists for W_left (single-site).
        // Used by apply_heff_single_site (warmup / polish) to compact Step 1 / 3.
        if (opts_.sparse_mpo) {
            int rows = D_use * d;
            int cols = d * D_use;
            std::vector<int> nnz_rows, nnz_cols;
            nnz_rows.reserve(rows);
            nnz_cols.reserve(cols);
            const double eps = 1e-14;
            for (int r = 0; r < rows; r++) {
                bool nz = false;
                for (int c = 0; c < cols && !nz; c++) {
                    if (scalar_abs(h_WL[r + (size_t)c * rows]) > eps) nz = true;
                }
                if (nz) nnz_rows.push_back(r);
            }
            for (int c = 0; c < cols; c++) {
                bool nz = false;
                for (int r = 0; r < rows && !nz; r++) {
                    if (scalar_abs(h_WL[r + (size_t)c * rows]) > eps) nz = true;
                }
                if (nz) nnz_cols.push_back(c);
            }
            wl_nnz_rows_count_[i] = (int)nnz_rows.size();
            wl_nnz_cols_count_[i] = (int)nnz_cols.size();
            if (!nnz_rows.empty()) {
                HIP_CHECK(hipMalloc(&d_WL_nnz_rows_[i], nnz_rows.size() * sizeof(int)));
                HIP_CHECK(hipMemcpy(d_WL_nnz_rows_[i], nnz_rows.data(),
                                    nnz_rows.size() * sizeof(int), hipMemcpyHostToDevice));
            }
            if (!nnz_cols.empty()) {
                HIP_CHECK(hipMalloc(&d_WL_nnz_cols_[i], nnz_cols.size() * sizeof(int)));
                HIP_CHECK(hipMemcpy(d_WL_nnz_cols_[i], nnz_cols.data(),
                                    nnz_cols.size() * sizeof(int), hipMemcpyHostToDevice));
            }
            if (i == 0) {
                std::fprintf(stderr,
                    "[SPARSE_MPO] site 0: W shape (%d x %d), nnz rows=%d, nnz cols=%d (%.0f%% sparse)\n",
                    rows, cols, wl_nnz_rows_count_[i], wl_nnz_cols_count_[i],
                    100.0 * (1.0 - (double)(wl_nnz_rows_count_[i] * wl_nnz_cols_count_[i]) / (rows * cols)));
            }
        }
    }

    precompute_fused_mpo(h_mpo_tensors);
}

template<typename Scalar>
void PDMRGGPU<Scalar>::precompute_fused_mpo(const std::vector<Scalar*>& h_mpo_tensors) {
    // D_use/D_act: see set_mpo. User's h_mpo_tensors are indexed with
    // stride D_act; WW is allocated at padded size D_use*dd x dd*D_use
    // with padded (w,n) rows/cols zero.
    int D_use = D_mpo_;
    int D_act = D_mpo_actual_;
    int d = d_;
    int dd = d * d;

    for (int bond = 0; bond < L_ - 1; bond++) {
        int ww_size = D_use * dd * dd * D_use;
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
                                    Scalar wl = WL[w + s1*D_act + s1p*D_act*d + m*D_act*d*d];
                                    Scalar wr = WR[m + s2*D_act + s2p*D_act*d + n*D_act*d*d];
                                    if constexpr (Traits::is_complex) {
                                        val = hipCadd(val, hipCmul(wl, wr));
                                    } else {
                                        val += wl * wr;
                                    }
                                }
                                int row = w * dd + s1 * d + s2;
                                int col = n * dd + s1p * d + s2p;
                                h_WW[row + col * D_use * dd] = val;
                            }

        HIP_CHECK(hipMalloc(&d_WW_[bond], ww_size * sizeof(Scalar)));
        HIP_CHECK(hipMemcpy(d_WW_[bond], h_WW.data(),
                            ww_size * sizeof(Scalar), hipMemcpyHostToDevice));

        // SPARSE_MPO: build nnz row/col index lists for WW[bond] (two-site).
        if (opts_.sparse_mpo) {
            int rows = D_use * dd;
            int cols = dd * D_use;
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
// Two-site theta formation (stream-aware)
// ============================================================================

template<typename Scalar>
void PDMRGGPU<Scalar>::form_theta_two_site(int site, int si) {
    int cL = chi_L(site);
    int chi_mid = bond_dims_[site + 1];
    int cR = chi_R(site + 1);
    Scalar one = Traits::one(), zero_val = Traits::zero();
    auto& ws = workspaces_[si];

    ROCBLAS_CHECK(Traits::gemm(handles_[si],
        rocblas_operation_none, rocblas_operation_none,
        cL * d_, d_ * cR, chi_mid,
        &one,
        d_mps_tensors_[site], cL * d_,
        d_mps_tensors_[site + 1], chi_mid,
        &zero_val,
        ws.d_theta, cL * d_));
}

// ============================================================================
// Two-site H_eff application (3-step with fused WW, stream-aware)
// ============================================================================

template<typename Scalar>
void PDMRGGPU<Scalar>::apply_heff_two_site(int site, const Scalar* d_theta_in,
                                            Scalar* d_result, int si) {
    int cL = chi_L(site);
    int cR = chi_R(site + 1);
    int D = D_mpo_, d = d_;
    int dd = d * d;
    Scalar one = Traits::one(), zero_val = Traits::zero();
    auto& ws = workspaces_[si];

    // LANCZOS_GRAPH: stage caller's theta into a fixed-address per-segment
    // bounce buffer BEFORE any capture window, then either replay the cached
    // graph or capture a new one for this (site, cL, cR) shape. Each segment
    // has its own bounce buffer + cache since segments run on independent
    // streams. See dmrg_gpu_impl.h for the full rationale. Two-site theta has
    // shape (cL, d, d, cR).
    const Scalar* theta_src = d_theta_in;
    bool graph_capture_miss = false;
    if (opts_.lanczos_graph) {
        int n_theta = cL * dd * cR;
        HIP_CHECK(hipMemcpyAsync(ws.d_heff_input, d_theta_in,
                                 (size_t)n_theta * sizeof(Scalar),
                                 hipMemcpyDeviceToDevice, streams_[si]));
        theta_src = ws.d_heff_input;

        uint64_t key = graph_key(site, cL, cR);
        auto it = ws.apply_heff_graph_cache.find(key);
        if (it != ws.apply_heff_graph_cache.end()) {
            HIP_CHECK(hipGraphLaunch(it->second, streams_[si]));
            return;
        }
        graph_capture_miss = true;
        HIP_CHECK(hipStreamBeginCapture(streams_[si], hipStreamCaptureModeGlobal));
    }

    Scalar* L_env = d_L_envs_[site];
    Scalar* R_env = d_R_envs_[site + 2];
    Scalar* WW = d_WW_[site];
    Scalar* T1 = ws.d_T1;
    Scalar* T2 = ws.d_T2;

    // SPARSE_MPO: compact Step 1 / Step 3 batches to non-zero (w,s1,s2) /
    // (n,s1p,s2p) rows/cols of WW[site]. Step 2 reads T1 densely, so skipped
    // slots must be zero (hipMemsetAsync). For Step 3, per-n scratch slices
    // must be zero so the GEMV reduction doesn't include stale data.
    const bool sparse_s1 = opts_.sparse_mpo
                         && ww_nnz_rows_count_[site] > 0
                         && ww_nnz_rows_count_[site] < D * dd;
    const bool sparse_s3 = opts_.sparse_mpo
                         && ww_nnz_cols_count_[site] > 0
                         && ww_nnz_cols_count_[site] < D * dd;

    // Step 1: Batched GEMM — L_env^T × theta
    if (sparse_s1) {
        HIP_CHECK(hipMemsetAsync(T1, 0, (size_t)D * dd * cL * cR * sizeof(Scalar), streams_[si]));
        int nnz = ww_nnz_rows_count_[site];
        hipLaunchKernelGGL(setup_batch_ptrs_wd_twosite_sparse<Scalar>,
                           dim3(1), dim3(nnz), 0, streams_[si],
                           ws.d_batch_A, ws.d_batch_B, ws.d_batch_C,
                           L_env, const_cast<Scalar*>(theta_src), T1,
                           d_WW_nnz_rows_[site], d, dd, cL, cL, cL * cR);
        ROCBLAS_CHECK(Traits::gemm_batched(handles_[si],
            Traits::op_t, rocblas_operation_none,
            cL, cR, cL,
            &one,
            (const Scalar**)ws.d_batch_A, cL * D,
            (const Scalar**)ws.d_batch_B, cL * dd,
            &zero_val,
            ws.d_batch_C, cL,
            nnz));
        // Invalidate the dense A/C cache: next dense call must re-populate.
        ws.heff_cached_site = -1;
    } else {
        int batch_count = D * dd;

        // Cache A and C pointers (constant for a given site) — GPU kernel, no DMA
        if (ws.heff_cached_site != site) {
            hipLaunchKernelGGL(setup_heff_A_ptrs<Scalar>, dim3(1), dim3(batch_count), 0, streams_[si],
                               ws.d_heff_batch_A, L_env, cL, dd, batch_count);
            hipLaunchKernelGGL(setup_heff_C_ptrs<Scalar>, dim3(1), dim3(batch_count), 0, streams_[si],
                               ws.d_heff_batch_C, T1, cL * cR, batch_count);
            ws.heff_cached_site = site;
        }

        // B pointers change per call (d_theta_in varies) — GPU kernel, no DMA race
        hipLaunchKernelGGL(setup_heff_B_ptrs<Scalar>, dim3(1), dim3(batch_count), 0, streams_[si],
                           ws.d_batch_B, const_cast<Scalar*>(theta_src), cL, d, dd, batch_count);

        ROCBLAS_CHECK(Traits::gemm_batched(handles_[si],
            Traits::op_t, rocblas_operation_none,
            cL, cR, cL,
            &one,
            (const Scalar**)ws.d_heff_batch_A, cL * D,
            (const Scalar**)ws.d_batch_B, cL * dd,
            &zero_val,
            ws.d_heff_batch_C, cL,
            batch_count));
    }

    // Step 2: Dense GEMM — T1 × WW
    ROCBLAS_CHECK(Traits::gemm(handles_[si],
        rocblas_operation_none, rocblas_operation_none,
        cL * cR, dd * D, D * dd,
        &one,
        T1, cL * cR,
        WW, D * dd,
        &zero_val,
        T2, cL * cR));

    // Step 3 (R3-F1 full-batched collapse): replaces the unbatched triple loop
    // (dd*D = up to 36 separate GEMM launches per apply_heff call) with:
    //   1 setup kernel + 1 batched GEMM (batch = D*dd) + 1 GEMV reduction.
    //
    // Each n writes its cL x cR tile into its own per-n slice of T1 scratch
    // (T1 is free here, consumed by Step 2). Slices are then summed along
    // the D axis via rocblas_gemv with a length-D ones vector — one rocBLAS
    // call replaces the beta-accumulation across n.
    //
    // T1 allocation (D_mpo * dd * chi_max^2) fits D slices of (cL, dd, cR).
    {
        int slice_stride = cL * dd * cR;   // per-n slice size inside T1 scratch

        if (sparse_s3) {
            HIP_CHECK(hipMemsetAsync(T1, 0, (size_t)D * slice_stride * sizeof(Scalar), streams_[si]));
            int nnz = ww_nnz_cols_count_[site];
            hipLaunchKernelGGL(setup_batch_ptrs_step3_twosite_full_sparse<Scalar>,
                               dim3(1), dim3(nnz), 0, streams_[si],
                               ws.d_batch_A, ws.d_batch_B, ws.d_batch_C,
                               T2, R_env, T1,
                               d_WW_nnz_cols_[site], d, dd, cL * cR, cR, cL, slice_stride);
            ROCBLAS_CHECK(Traits::gemm_batched(handles_[si],
                rocblas_operation_none, rocblas_operation_none,
                cL, cR, cR,
                &one,
                (const Scalar**)ws.d_batch_A, cL,
                (const Scalar**)ws.d_batch_B, cR * D,
                &zero_val,
                ws.d_batch_C, cL * dd,
                nnz));
        } else {
            int batch_count = D * dd;
            hipLaunchKernelGGL(setup_heff_ts_step3_full_ptrs<Scalar>,
                               dim3(1), dim3(batch_count), 0, streams_[si],
                               ws.d_batch_A, ws.d_batch_B, ws.d_batch_C,
                               T2, R_env, T1,
                               d, dd, cL * cR, cR, cL, slice_stride);

            ROCBLAS_CHECK(Traits::gemm_batched(handles_[si],
                rocblas_operation_none, rocblas_operation_none,
                cL, cR, cR,
                &one,
                (const Scalar**)ws.d_batch_A, cL,
                (const Scalar**)ws.d_batch_B, cR * D,
                &zero_val,
                ws.d_batch_C, cL * dd,
                batch_count));
        }

        // Reduce D slices into d_result: d_result = T1[slice_stride x D] * ones_D
        ROCBLAS_CHECK(Traits::gemv(handles_[si],
            rocblas_operation_none,
            slice_stride, D,
            &one,
            T1, slice_stride,
            ws.d_ones_D, 1,
            &zero_val,
            d_result, 1));
    }

    if (graph_capture_miss) {
        hipGraph_t graph;
        HIP_CHECK(hipStreamEndCapture(streams_[si], &graph));
        hipGraphExec_t exec;
        HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        HIP_CHECK(hipGraphDestroy(graph));
        ws.apply_heff_graph_cache[graph_key(site, cL, cR)] = exec;
        HIP_CHECK(hipGraphLaunch(exec, streams_[si]));
    }
}

// ============================================================================
// Left environment update (stream-aware)
// ============================================================================

template<typename Scalar>
void PDMRGGPU<Scalar>::update_left_env(int site, int si) {
    int chi_in = bond_dims_[site];
    int chi_out = bond_dims_[site + 1];
    int D = D_mpo_, d = d_;
    Scalar one = Traits::one(), zero_val = Traits::zero();
    auto& ws = workspaces_[si];

    ensure_L_env_alloc(site + 1, chi_out);

    Scalar* L_env = d_L_envs_[site];
    Scalar* A = d_mps_tensors_[site];
    Scalar* W_mat = d_W_left_[site];
    Scalar* L_new = d_L_envs_[site + 1];
    Scalar* V = ws.d_T1;
    Scalar* U = ws.d_T2;

    // Step 1: V_ws[a',b] = L_w^T[a',a] * A_s[a,b]  (batched GEMM)
    {
        int batch_count = D * d;
        hipLaunchKernelGGL(setup_lenv_ptrs<Scalar>, dim3(1), dim3(batch_count), 0, streams_[si],
                           ws.d_batch_A, ws.d_batch_B, ws.d_batch_C,
                           L_env, A, V, chi_in, chi_out, d, batch_count);
        ROCBLAS_CHECK(Traits::gemm_batched(handles_[si],
            Traits::op_t, rocblas_operation_none,
            chi_in, chi_out, chi_in,
            &one,
            (const Scalar**)ws.d_batch_A, chi_in * D,
            (const Scalar**)ws.d_batch_B, chi_in * d,
            &zero_val,
            ws.d_batch_C, chi_in,
            batch_count));
    }

    // Step 2: U = V * W_matrix
    ROCBLAS_CHECK(Traits::gemm(handles_[si],
        rocblas_operation_none, rocblas_operation_none,
        chi_in * chi_out, d * D, D * d,
        &one,
        V, chi_in * chi_out,
        W_mat, D * d,
        &zero_val,
        U, chi_in * chi_out));

    // Step 3: L_new_w'[b,b'] = sum_{a',s'} conj(U[a',ws',b])^H * A[a',s',b']  (batched)
    // Batch D GEMMs per sp (safe: different wp write to different C locations).
    {
        for (int sp = 0; sp < d; sp++) {
            Scalar beta = (sp == 0) ? Traits::zero() : Traits::one();
            hipLaunchKernelGGL(setup_lenv_step3_ptrs<Scalar>, dim3(1), dim3(D), 0, streams_[si],
                               ws.d_batch_A, ws.d_batch_B, ws.d_batch_C,
                               U, A, L_new,
                               sp, d, D, chi_in * chi_out, chi_in, chi_out);
            ROCBLAS_CHECK(Traits::gemm_batched(handles_[si],
                Traits::op_h, rocblas_operation_none,
                chi_out, chi_out, chi_in,
                &one,
                (const Scalar**)ws.d_batch_A, chi_in,
                (const Scalar**)ws.d_batch_B, chi_in * d,
                &beta,
                ws.d_batch_C, chi_out * D,
                D));
        }
    }

    if constexpr (Traits::is_complex) {
        conjugate_inplace(L_new, chi_out * D * chi_out, streams_[si]);
    }
}

// ============================================================================
// Right environment update (stream-aware)
// ============================================================================

template<typename Scalar>
void PDMRGGPU<Scalar>::update_right_env(int site, int si) {
    int chi_in = bond_dims_[site + 1];
    int chi_out = bond_dims_[site];
    int D = D_mpo_, d = d_;
    Scalar one = Traits::one(), zero_val = Traits::zero();
    auto& ws = workspaces_[si];

    ensure_R_env_alloc(site, chi_out);

    Scalar* A = d_mps_tensors_[site];
    Scalar* R_env = d_R_envs_[site + 1];
    Scalar* W_mat = d_W_right_[site];
    Scalar* R_new = d_R_envs_[site];
    Scalar* V = ws.d_T1;
    Scalar* U = ws.d_T2;

    // Step 1: V_ws[a,b'] = A_s[a,b] * R_w'[b,b']  (batched GEMM)
    {
        int batch_count = D * d;
        hipLaunchKernelGGL(setup_renv_ptrs<Scalar>, dim3(1), dim3(batch_count), 0, streams_[si],
                           ws.d_batch_A, ws.d_batch_B, ws.d_batch_C,
                           A, R_env, V, chi_in, chi_out, d, batch_count);
        ROCBLAS_CHECK(Traits::gemm_batched(handles_[si],
            rocblas_operation_none, rocblas_operation_none,
            chi_out, chi_in, chi_in,
            &one,
            (const Scalar**)ws.d_batch_A, chi_out * d,
            (const Scalar**)ws.d_batch_B, chi_in * D,
            &zero_val,
            ws.d_batch_C, chi_out,
            batch_count));
    }

    // Step 2: U = V * W_matrix
    ROCBLAS_CHECK(Traits::gemm(handles_[si],
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
            hipLaunchKernelGGL(setup_renv_step3_ptrs<Scalar>, dim3(1), dim3(D), 0, streams_[si],
                               ws.d_batch_A, ws.d_batch_B, ws.d_batch_C,
                               U, A, R_new,
                               sp, d, D, chi_out * chi_in, chi_out, chi_out);
            ROCBLAS_CHECK(Traits::gemm_batched(handles_[si],
                rocblas_operation_none, Traits::op_h,
                chi_out, chi_out, chi_in,
                &one,
                (const Scalar**)ws.d_batch_A, chi_out,
                (const Scalar**)ws.d_batch_B, chi_out * d,
                &beta,
                ws.d_batch_C, chi_out * D,
                D));
        }
    }
}

// ============================================================================
// Environment building (uses stream 0)
// ============================================================================

template<typename Scalar>
void PDMRGGPU<Scalar>::build_initial_environments() {
    // L[0] = trivial left boundary
    {
        std::vector<Scalar> h_L(D_mpo_, Traits::zero());
        h_L[0] = Traits::one();
        HIP_CHECK(hipMemcpy(d_L_envs_[0], h_L.data(),
                            D_mpo_ * sizeof(Scalar), hipMemcpyHostToDevice));
    }

    // R[L] = trivial right boundary
    {
        // With D_PAD, D_mpo_ > D_mpo_actual_; the identity slot must be at
        // the unpadded index D_mpo_actual_ - 1 because the padded W rows past
        // that index are zero.
        std::vector<Scalar> h_R(D_mpo_, Traits::zero());
        h_R[D_mpo_actual_ - 1] = Traits::one();
        HIP_CHECK(hipMemcpy(d_R_envs_[L_], h_R.data(),
                            D_mpo_ * sizeof(Scalar), hipMemcpyHostToDevice));
    }

    // Build all L environments left-to-right on stream 0
    for (int i = 0; i < L_; i++) {
        update_left_env(i, 0);
    }
    HIP_CHECK(hipStreamSynchronize(streams_[0]));

    // Build all R environments right-to-left on stream 0
    for (int i = L_ - 1; i >= 0; i--) {
        update_right_env(i, 0);
    }
    HIP_CHECK(hipStreamSynchronize(streams_[0]));
}

// ============================================================================
// Lanczos eigensolver (stream-aware)
// ============================================================================

template<typename Scalar>
double PDMRGGPU<Scalar>::lanczos_eigensolver(int site, Scalar* d_theta, int theta_size, int si) {
    int n = theta_size;
    int max_iter = std::min(max_lanczos_iter_, n);
    double tol_lanczos = 1e-12;
    double tol_eig_conv = 1e-12;
    auto& ws = workspaces_[si];

    Scalar* d_lanczos_v = ws.d_lanczos_v;

    // v[0] = theta / ||theta|| — use device pointer mode to avoid implicit sync
    double norm;
    ROCBLAS_CHECK(rocblas_set_pointer_mode(handles_[si], rocblas_pointer_mode_device));
    ROCBLAS_CHECK(Traits::nrm2(handles_[si], n, d_theta, 1, ws.d_nrm2_result));
    ROCBLAS_CHECK(rocblas_set_pointer_mode(handles_[si], rocblas_pointer_mode_host));

    // Check norm on host (need the value for near-zero check)
    HIP_CHECK(hipMemcpy(&norm, ws.d_nrm2_result, sizeof(double), hipMemcpyDeviceToHost));
    if (norm < 1e-14) {
        std::vector<Scalar> h_init(n);
        srand(42 + site);
        for (int i = 0; i < n; i++) h_init[i] = Traits::random_val();
        HIP_CHECK(hipMemcpy(d_theta, h_init.data(), n * sizeof(Scalar), hipMemcpyHostToDevice));
        ROCBLAS_CHECK(Traits::nrm2(handles_[si], n, d_theta, 1, &norm));
    }

    // Normalize using device pointer mode
    ROCBLAS_CHECK(rocblas_set_pointer_mode(handles_[si], rocblas_pointer_mode_device));
    hipLaunchKernelGGL(inv_real_kernel, dim3(1), dim3(1), 0, streams_[si],
                       ws.d_nrm2_result, ws.d_inv_nrm);
    ROCBLAS_CHECK(Traits::scal_real(handles_[si], n, ws.d_inv_nrm, d_theta, 1));
    ROCBLAS_CHECK(rocblas_set_pointer_mode(handles_[si], rocblas_pointer_mode_host));
    HIP_CHECK(hipMemcpyAsync(d_lanczos_v, d_theta, n * sizeof(Scalar), hipMemcpyDeviceToDevice, streams_[si]));

    double prev_energy = 1e30;
    int iter;

    for (iter = 0; iter < max_iter; iter++) {
        Scalar* d_vi = d_lanczos_v + (size_t)iter * n;

        // w = H|v_i> (apply_heff uses host pointer mode internally)
        if (lanczos_use_1site_)
            apply_heff_single_site(site, d_vi, ws.d_heff_result, si);
        else
            apply_heff_two_site(site, d_vi, ws.d_heff_result, si);

        // Switch to device pointer mode for scalar operations
        ROCBLAS_CHECK(rocblas_set_pointer_mode(handles_[si], rocblas_pointer_mode_device));

        // alpha_i = <v_i|w> → device
        ROCBLAS_CHECK(Traits::dot(handles_[si], n, d_vi, 1, ws.d_heff_result, 1, ws.d_dot_result));

        // Process alpha: store to d_alpha_dev[iter], compute d_neg_alpha
        hipLaunchKernelGGL(lanczos_process_alpha_kernel<Scalar>, dim3(1), dim3(1), 0, streams_[si],
                           ws.d_dot_result, ws.d_neg_alpha, ws.d_alpha_dev, iter);

        if (opts_.fuse_lanczos) {
            // Fused: w += (-α)·v_i + (-β_{i-1})·v_{i-1} in one kernel pass
            const Scalar* v_im1 = (iter > 0) ? (d_lanczos_v + (size_t)(iter - 1) * n) : nullptr;
            const Scalar* nb_im1 = (iter > 0) ? (ws.d_neg_beta_scalars + (iter - 1)) : ws.d_neg_alpha;
            int block = 256;
            int grid = (n + block - 1) / block;
            hipLaunchKernelGGL((lanczos_fused_sub_kernel<Scalar>),
                               dim3(grid), dim3(block), 0, streams_[si],
                               ws.d_heff_result, d_vi, v_im1,
                               ws.d_neg_alpha, nb_im1,
                               n, iter > 0 ? 1 : 0);
        } else {
            // w -= alpha_i * v_i (device pointer)
            ROCBLAS_CHECK(Traits::axpy(handles_[si], n, ws.d_neg_alpha, d_vi, 1, ws.d_heff_result, 1));

            // w -= beta_{i-1} * v_{i-1} (device pointer: pre-stored by previous iter)
            if (iter > 0) {
                ROCBLAS_CHECK(Traits::axpy(handles_[si], n,
                    ws.d_neg_beta_scalars + (iter - 1),
                    d_lanczos_v + (size_t)(iter - 1) * n, 1,
                    ws.d_heff_result, 1));
            }
        }

        // Full reorthogonalization — "twice is enough" double CGS
        // Single-pass CGS accumulates orthogonality defect ~ ε·n_iter·sqrt(N).
        // At chi=128 (N=65536, n_iter=100) this reaches ~2.5e-12, right at our
        // eigenvalue tolerance. A second pass reduces defect to ~ε².
        if (iter > 0) {
            // First CGS pass
            ROCBLAS_CHECK(Traits::gemv(handles_[si], Traits::op_h,
                n, iter + 1, ws.d_const_one,
                d_lanczos_v, n,
                ws.d_heff_result, 1,
                ws.d_const_zero, ws.d_ritz_coeffs, 1));
            ROCBLAS_CHECK(Traits::gemv(handles_[si], rocblas_operation_none,
                n, iter + 1, ws.d_const_neg_one,
                d_lanczos_v, n,
                ws.d_ritz_coeffs, 1,
                ws.d_const_one, ws.d_heff_result, 1));
            // Second CGS pass (reduces defect from ~ε·κ to ~ε²)
            ROCBLAS_CHECK(Traits::gemv(handles_[si], Traits::op_h,
                n, iter + 1, ws.d_const_one,
                d_lanczos_v, n,
                ws.d_heff_result, 1,
                ws.d_const_zero, ws.d_ritz_coeffs, 1));
            ROCBLAS_CHECK(Traits::gemv(handles_[si], rocblas_operation_none,
                n, iter + 1, ws.d_const_neg_one,
                d_lanczos_v, n,
                ws.d_ritz_coeffs, 1,
                ws.d_const_one, ws.d_heff_result, 1));
        } else {
            ROCBLAS_CHECK(Traits::dot(handles_[si], n, d_lanczos_v, 1, ws.d_heff_result, 1, ws.d_dot_result));
            hipLaunchKernelGGL(negate_scalar_kernel<Scalar>, dim3(1), dim3(1), 0, streams_[si],
                               ws.d_dot_result, ws.d_neg_overlap);
            ROCBLAS_CHECK(Traits::axpy(handles_[si], n, ws.d_neg_overlap, d_lanczos_v, 1, ws.d_heff_result, 1));
        }

        // beta_i = ||w|| → device
        ROCBLAS_CHECK(Traits::nrm2(handles_[si], n, ws.d_heff_result, 1, ws.d_nrm2_result));

        // Process beta: store, compute 1/beta, store -beta as Scalar
        hipLaunchKernelGGL(lanczos_process_beta_kernel<Scalar>, dim3(1), dim3(1), 0, streams_[si],
                           ws.d_nrm2_result, ws.d_inv_nrm, ws.d_beta_dev, ws.d_neg_beta_scalars, iter);

        // v_{i+1} = w / beta_i (device pointer for scal)
        if (iter + 1 < max_iter) {
            Scalar* d_vip1 = d_lanczos_v + (size_t)(iter + 1) * n;
            if (opts_.fuse_lanczos) {
                // Fused normalize+copy: v_{i+1}[k] = w[k] * (1/β_i)
                int block = 256;
                int grid = (n + block - 1) / block;
                hipLaunchKernelGGL((lanczos_fused_norm_copy_kernel<Scalar, RealType>),
                                   dim3(grid), dim3(block), 0, streams_[si],
                                   d_vip1, ws.d_heff_result, ws.d_inv_nrm, n);
            } else {
                HIP_CHECK(hipMemcpyAsync(d_vip1, ws.d_heff_result, n * sizeof(Scalar), hipMemcpyDeviceToDevice, streams_[si]));
                ROCBLAS_CHECK(Traits::scal_real(handles_[si], n, ws.d_inv_nrm, d_vip1, 1));
            }
        }

        // Switch back to host pointer mode (needed by apply_heff next iteration)
        ROCBLAS_CHECK(rocblas_set_pointer_mode(handles_[si], rocblas_pointer_mode_host));

        // Convergence check every 3 iterations after iter >= 4
        // LANCZOS_FIXED skips the check entirely — no mid-loop host syncs.
        // This is the ONLY sync point in the inner loop
        if (!opts_.lanczos_fixed && iter >= 4 && iter % 3 == 0) {
            HIP_CHECK(hipStreamSynchronize(streams_[si]));

            int ncheck = iter + 1;

            // Check for invariant subspace (beta < tol) on GPU
            hipLaunchKernelGGL(lanczos_check_beta, dim3(1), dim3(1), 0, streams_[si],
                               ws.d_beta_dev, ncheck, tol_lanczos, ws.d_steqr_info);
            rocblas_int h_beta_idx;
            HIP_CHECK(hipMemcpy(&h_beta_idx, ws.d_steqr_info, sizeof(rocblas_int), hipMemcpyDeviceToHost));
            if (h_beta_idx > 0) { iter = h_beta_idx; break; }

            // Eigenvalue convergence check via rocsolver on GPU
            HIP_CHECK(hipMemcpyAsync(ws.d_steqr_D, ws.d_alpha_dev, ncheck * sizeof(double), hipMemcpyDeviceToDevice, streams_[si]));
            HIP_CHECK(hipMemcpyAsync(ws.d_steqr_E, ws.d_beta_dev, ncheck * sizeof(double), hipMemcpyDeviceToDevice, streams_[si]));
            rocsolver_dsteqr(handles_[si], rocblas_evect_none, ncheck,
                             ws.d_steqr_D, ws.d_steqr_E, nullptr, ncheck, ws.d_steqr_info);
            rocblas_int h_info_chk;
            double cur_energy;
            HIP_CHECK(hipMemcpy(&h_info_chk, ws.d_steqr_info, sizeof(rocblas_int), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(&cur_energy, ws.d_steqr_D, sizeof(double), hipMemcpyDeviceToHost));
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

    // Ensure stream is synchronized before reading results
    HIP_CHECK(hipStreamSynchronize(streams_[si]));

    // Solve tridiagonal eigenvalue problem on GPU via rocsolver
    HIP_CHECK(hipMemcpyAsync(ws.d_steqr_D, ws.d_alpha_dev, niter * sizeof(double), hipMemcpyDeviceToDevice, streams_[si]));
    HIP_CHECK(hipMemcpyAsync(ws.d_steqr_E, ws.d_beta_dev, niter * sizeof(double), hipMemcpyDeviceToDevice, streams_[si]));
    rocsolver_dsteqr(handles_[si], rocblas_evect_tridiagonal, niter,
                     ws.d_steqr_D, ws.d_steqr_E, ws.d_steqr_C, niter, ws.d_steqr_info);

    rocblas_int h_steqr_info;
    HIP_CHECK(hipMemcpy(&h_steqr_info, ws.d_steqr_info, sizeof(rocblas_int), hipMemcpyDeviceToHost));
    if (h_steqr_info != 0) {
        throw std::runtime_error("rocsolver_dsteqr failed with info = " + std::to_string(h_steqr_info));
    }

    double energy;
    HIP_CHECK(hipMemcpy(&energy, ws.d_steqr_D, sizeof(double), hipMemcpyDeviceToHost));

    // Ritz coefficients = first column of eigenvector matrix (on device)
    if constexpr (std::is_same_v<Scalar, double>) {
        HIP_CHECK(hipMemcpyAsync(ws.d_ritz_coeffs, ws.d_steqr_C, niter * sizeof(double), hipMemcpyDeviceToDevice, streams_[si]));
    } else {
        // Complex case: promote double eigenvectors to complex on GPU
        int blk = (niter + 63) / 64;
        hipLaunchKernelGGL(promote_double_to_complex, dim3(blk), dim3(64), 0, streams_[si],
                           ws.d_steqr_C, (hipDoubleComplex*)ws.d_ritz_coeffs, niter);
    }

    ROCBLAS_CHECK(rocblas_set_pointer_mode(handles_[si], rocblas_pointer_mode_device));
    ROCBLAS_CHECK(Traits::gemv(
        handles_[si], rocblas_operation_none,
        n, niter, ws.d_const_one,
        d_lanczos_v, n,
        ws.d_ritz_coeffs, 1,
        ws.d_const_zero, d_theta, 1
    ));

    // Normalize on device
    ROCBLAS_CHECK(Traits::nrm2(handles_[si], n, d_theta, 1, ws.d_nrm2_result));
    hipLaunchKernelGGL(inv_real_kernel, dim3(1), dim3(1), 0, streams_[si],
                       ws.d_nrm2_result, ws.d_inv_nrm);
    ROCBLAS_CHECK(Traits::scal_real(handles_[si], n, ws.d_inv_nrm, d_theta, 1));
    ROCBLAS_CHECK(rocblas_set_pointer_mode(handles_[si], rocblas_pointer_mode_host));

    return energy;
}

// ============================================================================
// SVD split (stream-aware)
// ============================================================================

template<typename Scalar>
void PDMRGGPU<Scalar>::svd_split(int site, Scalar* d_theta, char direction, int si) {
    int cL = chi_L(site);
    int cR = chi_R(site + 1);
    auto& ws = workspaces_[si];

    int m = cL * d_;
    int n_svd = d_ * cR;
    int full_k = std::min(m, n_svd);
    int k = std::min(full_k, chi_max_);

    Scalar* h_U_data = nullptr;
    RealType* h_S_data = nullptr;
    Scalar* h_Vh_data = nullptr;
    bool gpu_svd_path = false;

    if (use_cpu_svd_) {
        HIP_CHECK(hipMemcpyAsync(ws.h_svd_A.data(), d_theta, m * n_svd * sizeof(Scalar),
                                  hipMemcpyDeviceToHost, streams_[si]));
        HIP_CHECK(hipStreamSynchronize(streams_[si]));

        int lwork = (int)ws.h_svd_work.size();
        int info;
        const char jobu = 'S', jobvt = 'S';
        Traits::lapack_gesvd(&jobu, &jobvt, &m, &n_svd, ws.h_svd_A.data(), &m,
                ws.h_svd_S.data(), ws.h_svd_U.data(), &m, ws.h_svd_Vh.data(), &full_k,
                ws.h_svd_work.data(), &lwork,
                ws.h_svd_rwork.empty() ? nullptr : ws.h_svd_rwork.data(), &info);

        h_U_data = ws.h_svd_U.data();
        h_S_data = ws.h_svd_S.data();
        h_Vh_data = ws.h_svd_Vh.data();
    } else {
        HIP_CHECK(hipMemcpyAsync(ws.d_svd_A, d_theta, m * n_svd * sizeof(Scalar),
                                  hipMemcpyDeviceToDevice, streams_[si]));

        // R3-F2 + regression fix: size-gated dispatcher — gesvdj (Jacobi) for
        // large / real; bidiagonal gesvd fallback for small complex shapes
        // where zgesvdj regresses. See docs/followups/r3_regression_analysis.md.
        Traits::rocsolver_gesvd_auto(handles_[si],
            rocblas_svect_singular, rocblas_svect_singular,
            m, n_svd,
            ws.d_svd_A, m,
            ws.d_svd_S,
            ws.d_svd_U, m,
            ws.d_svd_Vh, full_k,
            ws.d_svd_E,
            ws.d_svdj_residual, ws.d_svdj_n_sweeps,
            ws.d_svd_info);

        gpu_svd_path = true;
    }

    // Truncation
    int new_k;
    if (opts_.device_k) {
        new_k = k;
    } else if (gpu_svd_path) {
        // GPU path: truncation on device, only copy 1 int back
        hipLaunchKernelGGL(svd_truncate_kernel<RealType>, dim3(1), dim3(1), 0, streams_[si],
                           ws.d_svd_S, k, 1e-14, ws.d_svd_info);
        HIP_CHECK(hipMemcpy(&new_k, ws.d_svd_info, sizeof(int), hipMemcpyDeviceToHost));
    } else {
        new_k = k;
        for (int i = 0; i < new_k; i++) {
            if (h_S_data[i] < 1e-14) { new_k = i; break; }
        }
        if (new_k == 0) new_k = 1;
    }

    if (gpu_svd_path) {
        // GPU SVD path: U, S, Vh all on device already
        // d_svd_U (m × full_k, lda=m), d_svd_S (full_k), d_svd_Vh (full_k × n_svd, lda=full_k)

        if (direction == 'R') {
            // MPS[site] = U[:, :new_k]
            allocate_mps_tensor(site, cL, new_k);
            HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site], ws.d_svd_U,
                        (size_t)m * new_k * sizeof(Scalar), hipMemcpyDeviceToDevice, streams_[si]));
            // MPS[site+1] = diag(S) @ Vh[:new_k, :] — scale rows on GPU
            allocate_mps_tensor(site + 1, new_k, cR);
            scale_rows_by_real(ws.d_svd_Vh, full_k, ws.d_svd_S,
                               d_mps_tensors_[site + 1], new_k, new_k, n_svd, streams_[si]);
        } else {
            // MPS[site] = U[:, :new_k] @ diag(S) — scale columns on GPU
            allocate_mps_tensor(site, cL, new_k);
            scale_columns_by_real(ws.d_svd_U, m, ws.d_svd_S,
                                  d_mps_tensors_[site], m, m, new_k, streams_[si]);
            // MPS[site+1] = Vh[:new_k, :] — extract rows with stride when truncated
            allocate_mps_tensor(site + 1, new_k, cR);
            if (new_k == full_k) {
                HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site + 1], ws.d_svd_Vh,
                            (size_t)full_k * n_svd * sizeof(Scalar), hipMemcpyDeviceToDevice, streams_[si]));
            } else {
                // Vh is (full_k × n_svd) column-major with lda=full_k; extract first new_k rows
                HIP_CHECK(hipMemcpy2DAsync(
                    d_mps_tensors_[site + 1], new_k * sizeof(Scalar),      // dst, dpitch
                    ws.d_svd_Vh,              full_k * sizeof(Scalar),      // src, spitch
                    new_k * sizeof(Scalar),   n_svd,                        // width, height
                    hipMemcpyDeviceToDevice, streams_[si]));
            }
        }
    } else {
        // CPU SVD path: U, S, Vh on host. Upload S + raw factors, scale on GPU.
        HIP_CHECK(hipMemcpyAsync(ws.d_svd_S, h_S_data, new_k * sizeof(RealType),
                                  hipMemcpyHostToDevice, streams_[si]));

        if (direction == 'R') {
            // MPS[site] = U[:, :new_k]
            allocate_mps_tensor(site, cL, new_k);
            HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site], h_U_data,
                        m * new_k * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));

            // MPS[site+1] = diag(S) @ Vh[:new_k, :] — scale rows of Vh by S on GPU
            if (new_k == full_k) {
                HIP_CHECK(hipMemcpyAsync(ws.d_svd_work, h_Vh_data,
                            (size_t)full_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));
            } else {
                // Pack contiguous new_k rows from leading-dim full_k layout
                for (int j = 0; j < n_svd; j++)
                    for (int i = 0; i < new_k; i++)
                        ws.h_svd_tmp[i + j * new_k] = h_Vh_data[i + j * full_k];
                HIP_CHECK(hipMemcpyAsync(ws.d_svd_work, ws.h_svd_tmp.data(),
                            (size_t)new_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));
            }
            allocate_mps_tensor(site + 1, new_k, cR);
            int vh_ld = (new_k == full_k) ? full_k : new_k;
            scale_rows_by_real(ws.d_svd_work, vh_ld, ws.d_svd_S,
                               d_mps_tensors_[site + 1], new_k, new_k, n_svd, streams_[si]);

        } else {  // direction == 'L'
            // MPS[site] = U[:, :new_k] @ diag(S) — scale columns of U by S on GPU
            HIP_CHECK(hipMemcpyAsync(ws.d_svd_work, h_U_data,
                        (size_t)m * new_k * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));
            allocate_mps_tensor(site, cL, new_k);
            scale_columns_by_real(ws.d_svd_work, m, ws.d_svd_S,
                                  d_mps_tensors_[site], m, m, new_k, streams_[si]);

            // MPS[site+1] = Vh[:new_k, :]
            allocate_mps_tensor(site + 1, new_k, cR);
            if (new_k == full_k) {
                HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site + 1], h_Vh_data,
                            (size_t)full_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));
            } else {
                for (int j = 0; j < n_svd; j++)
                    for (int i = 0; i < new_k; i++)
                        ws.h_svd_tmp[i + j * new_k] = h_Vh_data[i + j * full_k];
                HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site + 1], ws.h_svd_tmp.data(),
                            (size_t)new_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));
            }
        }
    }

    bond_dims_[site + 1] = new_k;

    // Invalidate heff pointer cache (bond dims changed)
    ws.heff_cached_site = -1;
}

// ============================================================================
// Randomized truncated SVD split (Halko-Martinsson-Tropp, GPU QR)
// ============================================================================

template<typename Scalar>
void PDMRGGPU<Scalar>::rsvd_split(int site, Scalar* d_theta, char direction, int si) {
    int cL = chi_L(site);
    int cR = chi_R(site + 1);
    int m = cL * d_;
    int n_svd = d_ * cR;
    int full_k = std::min(m, n_svd);
    int k = std::min(full_k, chi_max_);
    auto& ws = workspaces_[si];

    // If matrix is small enough, fall back to full SVD (rSVD overhead not worth it)
    if (full_k <= k + rsvd_oversampling_ || m <= 2 * k) {
        svd_split(site, d_theta, direction, si);
        return;
    }

    int r = k + rsvd_oversampling_;  // projection rank

    // Step 1: Generate random Omega (n_svd x r) on CPU and upload
    {
        std::vector<Scalar> h_omega(n_svd * r);
        for (int i = 0; i < n_svd * r; i++) {
            h_omega[i] = Traits::random_val();
        }
        HIP_CHECK(hipMemcpyAsync(ws.d_rsvd_omega, h_omega.data(),
                            n_svd * r * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));
    }

    // Step 2: Y = theta @ Omega on GPU  (m x n_svd) @ (n_svd x r) -> (m x r)
    {
        Scalar one = Traits::one(), zero_val = Traits::zero();
        ROCBLAS_CHECK(Traits::gemm(handles_[si],
            rocblas_operation_none, rocblas_operation_none,
            m, r, n_svd, &one,
            d_theta, m,
            ws.d_rsvd_omega, n_svd,
            &zero_val,
            ws.d_rsvd_Y, m));
        HIP_CHECK(hipStreamSynchronize(streams_[si]));
    }

    // Step 3: QR factorization of Y on GPU -> Q (m x r) stays on device
    HIP_CHECK(hipMemcpyAsync(ws.d_rsvd_Q, ws.d_rsvd_Y, (size_t)m * r * sizeof(Scalar),
                              hipMemcpyDeviceToDevice, streams_[si]));
    ROCBLAS_CHECK(Traits::rocsolver_geqrf(handles_[si], m, r, ws.d_rsvd_Q, m, ws.d_rsvd_ipiv));
    ROCBLAS_CHECK(Traits::rocsolver_orgqr(handles_[si], m, r, r, ws.d_rsvd_Q, m, ws.d_rsvd_ipiv));
    HIP_CHECK(hipStreamSynchronize(streams_[si]));

    // Step 4: B = Q^H @ theta on GPU  (r x m) @ (m x n_svd) -> (r x n_svd)
    {
        Scalar one = Traits::one(), zero_val = Traits::zero();
        ROCBLAS_CHECK(Traits::gemm(handles_[si],
            Traits::op_h, rocblas_operation_none,
            r, n_svd, m, &one,
            ws.d_rsvd_Q, m,
            d_theta, m,
            &zero_val,
            ws.d_rsvd_B, r));
        HIP_CHECK(hipStreamSynchronize(streams_[si]));
    }

    // Step 5: Copy B to host, compute SVD of B (r x n_svd) — much smaller than (m x n_svd)
    HIP_CHECK(hipMemcpy(ws.h_rsvd_B.data(), ws.d_rsvd_B, r * n_svd * sizeof(Scalar), hipMemcpyDeviceToHost));

    int small_k = std::min(r, n_svd);
    {
        int lwork = (int)ws.h_svd_work.size();
        int info;
        const char jobu = 'S', jobvt = 'S';
        // U_small: (r x small_k), S: (small_k), Vh: (small_k x n_svd)
        Traits::lapack_gesvd(&jobu, &jobvt, &r, &n_svd, ws.h_rsvd_B.data(), &r, ws.h_svd_S.data(),
                ws.h_rsvd_U_small.data(), &r, ws.h_svd_Vh.data(), &small_k,
                ws.h_svd_work.data(), &lwork,
                ws.h_svd_rwork.empty() ? nullptr : ws.h_svd_rwork.data(), &info);
        if (info != 0) {
            svd_split(site, d_theta, direction, si);
            return;
        }
    }

    // Step 6: Upload U_small to GPU, compute U_full = Q @ U_small on GPU
    //   Q is (m x r) on device, U_small is (r x small_k) on host -> U_full (m x small_k)
    {
        // Upload U_small to device (reuse d_rsvd_B as temp — it's no longer needed)
        HIP_CHECK(hipMemcpyAsync(ws.d_rsvd_B, ws.h_rsvd_U_small.data(),
                            (size_t)r * small_k * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));
        Scalar one = Traits::one(), zero_val = Traits::zero();
        ROCBLAS_CHECK(Traits::gemm(handles_[si],
            rocblas_operation_none, rocblas_operation_none,
            m, small_k, r, &one,
            ws.d_rsvd_Q, m,
            ws.d_rsvd_B, r,
            &zero_val,
            ws.d_rsvd_U_full, m));
        HIP_CHECK(hipStreamSynchronize(streams_[si]));
    }

    // Now: U_full (m x small_k) is on GPU at ws.d_rsvd_U_full
    //      S (small_k) and Vh (small_k x n_svd) are on host
    RealType* h_S_data = ws.h_svd_S.data();
    Scalar* h_Vh_data = ws.h_svd_Vh.data();

    // Truncation
    int new_k = k;
    for (int i = 0; i < new_k; i++) {
        if (h_S_data[i] < 1e-14) { new_k = i; break; }
    }
    if (new_k == 0) new_k = 1;

    // Upload S to device for GPU-side scaling
    HIP_CHECK(hipMemcpyAsync(ws.d_svd_S, h_S_data, new_k * sizeof(RealType),
                              hipMemcpyHostToDevice, streams_[si]));

    if (direction == 'R') {
        // MPS[site] = U_full[:, :new_k]  (U_full already on GPU)
        allocate_mps_tensor(site, cL, new_k);
        if (new_k == small_k) {
            HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site], ws.d_rsvd_U_full,
                                (size_t)m * new_k * sizeof(Scalar), hipMemcpyDeviceToDevice, streams_[si]));
        } else {
            // Column subset: first new_k columns are contiguous in column-major
            HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site], ws.d_rsvd_U_full,
                                (size_t)m * new_k * sizeof(Scalar), hipMemcpyDeviceToDevice, streams_[si]));
        }

        // MPS[site+1] = diag(S) @ Vh[:new_k, :] — scale rows of Vh by S on GPU
        // Vh is on host with leading dim small_k, pack to new_k and upload
        if (new_k == small_k) {
            HIP_CHECK(hipMemcpyAsync(ws.d_svd_work, h_Vh_data,
                        (size_t)small_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));
        } else {
            for (int j = 0; j < n_svd; j++)
                for (int i = 0; i < new_k; i++)
                    ws.h_svd_tmp[i + j * new_k] = h_Vh_data[i + j * small_k];
            HIP_CHECK(hipMemcpyAsync(ws.d_svd_work, ws.h_svd_tmp.data(),
                        (size_t)new_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));
        }
        allocate_mps_tensor(site + 1, new_k, cR);
        int vh_ld = (new_k == small_k) ? small_k : new_k;
        scale_rows_by_real(ws.d_svd_work, vh_ld, ws.d_svd_S,
                           d_mps_tensors_[site + 1], new_k, new_k, n_svd, streams_[si]);

    } else {
        // MPS[site] = U_full[:, :new_k] @ diag(S) — scale columns on GPU
        // U_full already on GPU at d_rsvd_U_full
        allocate_mps_tensor(site, cL, new_k);
        scale_columns_by_real(ws.d_rsvd_U_full, m, ws.d_svd_S,
                              d_mps_tensors_[site], m, m, new_k, streams_[si]);

        // MPS[site+1] = Vh[:new_k, :]
        allocate_mps_tensor(site + 1, new_k, cR);
        if (new_k == small_k) {
            HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site + 1], h_Vh_data,
                                (size_t)small_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));
        } else {
            for (int j = 0; j < n_svd; j++)
                for (int i = 0; i < new_k; i++)
                    ws.h_svd_tmp[i + j * new_k] = h_Vh_data[i + j * small_k];
            HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site + 1], ws.h_svd_tmp.data(),
                                (size_t)new_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));
        }
    }

    bond_dims_[site + 1] = new_k;

    // Invalidate heff pointer cache (bond dims changed)
    ws.heff_cached_site = -1;
}

// ============================================================================
// Bond optimization (stream-aware)
// ============================================================================

template<typename Scalar>
double PDMRGGPU<Scalar>::optimize_bond(int site, char direction, int si) {
    int cL = chi_L(site);
    int cR = chi_R(site + 1);
    int theta_size = cL * d_ * d_ * cR;
    auto& ws = workspaces_[si];

    form_theta_two_site(site, si);
    // No sync needed: form_theta_two_site uses the same stream/handle,
    // rocBLAS operations are ordered within a stream.
    double energy = lanczos_eigensolver(site, ws.d_theta, theta_size, si);
    // Lanczos already syncs internally (nrm2/dot read host results).
    if (use_rsvd_)
        rsvd_split(site, ws.d_theta, direction, si);
    else
        svd_split(site, ws.d_theta, direction, si);
    // svd_split/rsvd_split syncs internally.

    return energy;
}

// ============================================================================
// Single-site apply_heff: H_eff|θ⟩ for one MPS tensor
// Same 3-step GEMM as dmrg-gpu: L_env × θ × W × R_env
// theta shape: (cL, d, cR), result shape: (cL, d, cR)
// ============================================================================

template<typename Scalar>
void PDMRGGPU<Scalar>::apply_heff_single_site(int site, const Scalar* d_theta_in,
                                               Scalar* d_result, int si) {
    int cL = chi_L(site);
    int cR = chi_R(site);
    int D = D_mpo_, d = d_;
    Scalar one = Traits::one(), zero_val = Traits::zero();
    auto& ws = workspaces_[si];

    // LANCZOS_GRAPH: stage caller's theta into the per-segment fixed-address
    // bounce buffer BEFORE any capture window, then either replay the cached
    // graph or capture a new one for this (site, cL, cR) shape. Single-site
    // theta has shape (cL, d, cR). See dmrg_gpu_impl.h for full rationale.
    const Scalar* theta_src = d_theta_in;
    bool graph_capture_miss = false;
    if (opts_.lanczos_graph) {
        int n_theta = cL * d * cR;
        HIP_CHECK(hipMemcpyAsync(ws.d_heff_input, d_theta_in,
                                 (size_t)n_theta * sizeof(Scalar),
                                 hipMemcpyDeviceToDevice, streams_[si]));
        theta_src = ws.d_heff_input;

        uint64_t key = graph_key(site, cL, cR);
        auto it = ws.apply_heff_graph_cache.find(key);
        if (it != ws.apply_heff_graph_cache.end()) {
            HIP_CHECK(hipGraphLaunch(it->second, streams_[si]));
            return;
        }
        graph_capture_miss = true;
        HIP_CHECK(hipStreamBeginCapture(streams_[si], hipStreamCaptureModeGlobal));
    }

    Scalar* L_env = d_L_envs_[site];
    Scalar* R_env = d_R_envs_[site + 1];
    Scalar* W_mat = d_W_left_[site];
    Scalar* V = ws.d_T1;
    Scalar* U = ws.d_T2;

    // SPARSE_MPO: compact Step 1 / Step 3 batches to non-zero (w,s)/(wp,sp)
    // rows/cols of W_left. Step 1 writes into the full V layout (Step 2 reads
    // V densely) so skipped slots must be zero (hipMemsetAsync). Step 3 writes
    // into per-wp scratch slices; skipped slices must be zero so the GEMV
    // reduction over D doesn't pick up stale data.
    const bool sparse_s1 = opts_.sparse_mpo
                         && wl_nnz_rows_count_[site] > 0
                         && wl_nnz_rows_count_[site] < D * d;
    const bool sparse_s3 = opts_.sparse_mpo
                         && wl_nnz_cols_count_[site] > 0
                         && wl_nnz_cols_count_[site] < D * d;

    // Step 1: V_ws[a',b] = L_w^T[a',a] * theta_s[a,b]  (D*d batched GEMMs)
    if (sparse_s1) {
        HIP_CHECK(hipMemsetAsync(V, 0, (size_t)D * d * cL * cR * sizeof(Scalar), streams_[si]));
        int nnz = wl_nnz_rows_count_[site];
        hipLaunchKernelGGL(setup_batch_ptrs_wd_sparse<Scalar>,
                           dim3(1), dim3(nnz), 0, streams_[si],
                           ws.d_batch_A, ws.d_batch_B, ws.d_batch_C,
                           L_env, const_cast<Scalar*>(theta_src), V,
                           d_WL_nnz_rows_[site], d, cL, cL, cL * cR);
        ROCBLAS_CHECK(Traits::gemm_batched(handles_[si],
            Traits::op_t, rocblas_operation_none,
            cL, cR, cL,
            &one,
            (const Scalar**)ws.d_batch_A, cL * D,
            (const Scalar**)ws.d_batch_B, cL * d,
            &zero_val,
            ws.d_batch_C, cL,
            nnz));
    } else {
        int batch_count = D * d;
        hipLaunchKernelGGL(setup_lenv_ptrs<Scalar>, dim3(1), dim3(batch_count), 0, streams_[si],
                           ws.d_batch_A, ws.d_batch_B, ws.d_batch_C,
                           L_env, const_cast<Scalar*>(theta_src), V, cL, cR, d, batch_count);
        ROCBLAS_CHECK(Traits::gemm_batched(handles_[si],
            Traits::op_t, rocblas_operation_none,
            cL, cR, cL,
            &one,
            (const Scalar**)ws.d_batch_A, cL * D,
            (const Scalar**)ws.d_batch_B, cL * d,
            &zero_val,
            ws.d_batch_C, cL,
            batch_count));
    }

    // Step 2: U = V * W_matrix  (single dense GEMM)
    ROCBLAS_CHECK(Traits::gemm(handles_[si],
        rocblas_operation_none, rocblas_operation_none,
        cL * cR, d * D, D * d,
        &one,
        V, cL * cR,
        W_mat, D * d,
        &zero_val,
        U, cL * cR));

    // Step 3 (R3-F1 full-batched collapse): one batched GEMM over all D*d
    // tiles, writing to per-wp scratch slices in V (free after Step 2), then
    // one rocblas_gemv reduction summing the D slices into d_result.
    //
    // Old path: D sequential batches of d GEMMs (2*D launches) with same-C
    //   beta accumulation across wp.
    // New path: 1 setup kernel + 1 batched GEMM (batch = D*d) + 1 GEMV.
    {
        int slice_stride = cL * d * cR;   // per-wp slice size inside V scratch

        if (sparse_s3) {
            HIP_CHECK(hipMemsetAsync(V, 0, (size_t)D * slice_stride * sizeof(Scalar), streams_[si]));
            int nnz = wl_nnz_cols_count_[site];
            hipLaunchKernelGGL(setup_batch_ptrs_step3_full_sparse<Scalar>,
                               dim3(1), dim3(nnz), 0, streams_[si],
                               ws.d_batch_A, ws.d_batch_B, ws.d_batch_C,
                               U, R_env, V,
                               d_WL_nnz_cols_[site], d, cL * cR, cR, cL, slice_stride);
            ROCBLAS_CHECK(Traits::gemm_batched(handles_[si],
                rocblas_operation_none, rocblas_operation_none,
                cL, cR, cR,
                &one,
                (const Scalar**)ws.d_batch_A, cL,
                (const Scalar**)ws.d_batch_B, cR * D,
                &zero_val,
                ws.d_batch_C, cL * d,
                nnz));
        } else {
            int batch_count = D * d;
            hipLaunchKernelGGL(setup_heff_ss_step3_full_ptrs<Scalar>,
                               dim3(1), dim3(batch_count), 0, streams_[si],
                               ws.d_batch_A, ws.d_batch_B, ws.d_batch_C,
                               U, R_env, V,
                               d, cL * cR, cR, cL, slice_stride);

            ROCBLAS_CHECK(Traits::gemm_batched(handles_[si],
                rocblas_operation_none, rocblas_operation_none,
                cL, cR, cR,
                &one,
                (const Scalar**)ws.d_batch_A, cL,
                (const Scalar**)ws.d_batch_B, cR * D,
                &zero_val,
                ws.d_batch_C, cL * d,
                batch_count));
        }

        // Reduce D slices into d_result: d_result = V[slice_stride x D] * ones_D
        ROCBLAS_CHECK(Traits::gemv(handles_[si],
            rocblas_operation_none,
            slice_stride, D,
            &one,
            V, slice_stride,
            ws.d_ones_D, 1,
            &zero_val,
            d_result, 1));
    }

    if (graph_capture_miss) {
        hipGraph_t graph;
        HIP_CHECK(hipStreamEndCapture(streams_[si], &graph));
        hipGraphExec_t exec;
        HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        HIP_CHECK(hipGraphDestroy(graph));
        ws.apply_heff_graph_cache[graph_key(site, cL, cR)] = exec;
        HIP_CHECK(hipGraphLaunch(exec, streams_[si]));
    }
}

// ============================================================================
// Single-site SVD split: decompose theta and shift canonical center
// Direction 'R': theta(cL*d, cR) → U=MPS[site], S*Vh absorbed into MPS[site+1]
// Direction 'L': theta(cL, d*cR) → Vh=MPS[site], U*S absorbed into MPS[site-1]
// ============================================================================

template<typename Scalar>
void PDMRGGPU<Scalar>::svd_split_single_site(int site, Scalar* d_theta, char direction, int si) {
    int cL = chi_L(site);
    int cR = chi_R(site);
    auto& ws = workspaces_[si];

    int m, n_svd;
    if (direction == 'R') { m = cL * d_; n_svd = cR; }
    else                  { m = cL;      n_svd = d_ * cR; }
    int full_k = std::min(m, n_svd);
    int k = std::min(full_k, chi_max_);

    Scalar* h_U_data = nullptr;
    RealType* h_S_data = nullptr;
    Scalar* h_Vh_data = nullptr;

    // CPU SVD (default path, faster for chi < 200)
    if (use_cpu_svd_) {
        HIP_CHECK(hipMemcpyAsync(ws.h_svd_A.data(), d_theta, m * n_svd * sizeof(Scalar),
                                  hipMemcpyDeviceToHost, streams_[si]));
        HIP_CHECK(hipStreamSynchronize(streams_[si]));

        int lwork = (int)ws.h_svd_work.size();
        int info;
        const char jobu = 'S', jobvt = 'S';
        Traits::lapack_gesvd(&jobu, &jobvt, &m, &n_svd, ws.h_svd_A.data(), &m,
                ws.h_svd_S.data(), ws.h_svd_U.data(), &m, ws.h_svd_Vh.data(), &full_k,
                ws.h_svd_work.data(), &lwork,
                ws.h_svd_rwork.empty() ? nullptr : ws.h_svd_rwork.data(), &info);

        h_U_data = ws.h_svd_U.data();
        h_S_data = ws.h_svd_S.data();
        h_Vh_data = ws.h_svd_Vh.data();
    } else {
        HIP_CHECK(hipMemcpyAsync(ws.d_svd_A, d_theta, m * n_svd * sizeof(Scalar),
                                  hipMemcpyDeviceToDevice, streams_[si]));
        // R3-F2 + regression fix: size-gated dispatcher (see above call site).
        Traits::rocsolver_gesvd_auto(handles_[si],
            rocblas_svect_singular, rocblas_svect_singular,
            m, n_svd,
            ws.d_svd_A, m,
            ws.d_svd_S,
            ws.d_svd_U, m,
            ws.d_svd_Vh, full_k,
            ws.d_svd_E,
            ws.d_svdj_residual, ws.d_svdj_n_sweeps,
            ws.d_svd_info);
    }

    // Truncation
    int new_k;
    if (opts_.device_k) {
        new_k = k;
    } else if (!use_cpu_svd_) {
        // GPU path: truncation on device, only copy 1 int back
        hipLaunchKernelGGL(svd_truncate_kernel<RealType>, dim3(1), dim3(1), 0, streams_[si],
                           ws.d_svd_S, k, 1e-14, ws.d_svd_info);
        HIP_CHECK(hipMemcpy(&new_k, ws.d_svd_info, sizeof(int), hipMemcpyDeviceToHost));
    } else {
        h_S_data = ws.h_svd_S.data();
        new_k = k;
        for (int i = 0; i < new_k; i++) {
            if (h_S_data[i] < 1e-14) { new_k = i; break; }
        }
        if (new_k == 0) new_k = 1;
    }

    Scalar one = Traits::one(), zero_val = Traits::zero();

    if (direction == 'R') {
        int new_chi_R = new_k;

        if (use_cpu_svd_) {
            // Upload U[:, :new_k] as MPS[site]
            allocate_mps_tensor(site, cL, new_chi_R);
            HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site], h_U_data,
                        m * new_k * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));

            // Compute S*Vh on CPU, then absorb into MPS[site+1] via GEMM
            for (int j = 0; j < n_svd; j++)
                for (int i = 0; i < new_k; i++)
                    ws.h_svd_tmp[i + j * new_k] = Traits::scale_by_real(h_S_data[i], h_Vh_data[i + j * full_k]);

            HIP_CHECK(hipMemcpyAsync(ws.d_svd_work, ws.h_svd_tmp.data(),
                        new_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));
        } else {
            // GPU path: U and Vh already on device
            allocate_mps_tensor(site, cL, new_chi_R);
            HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site], ws.d_svd_U,
                        (size_t)m * new_k * sizeof(Scalar), hipMemcpyDeviceToDevice, streams_[si]));
            // Scale rows of Vh by S → ws.d_svd_work
            scale_rows_by_real(ws.d_svd_Vh, full_k, ws.d_svd_S,
                               ws.d_svd_work, new_k, new_k, n_svd, streams_[si]);
        }

        // Absorb S*Vh into MPS[site+1]: new = (S*Vh) @ old_MPS[site+1]
        if (site + 1 < L_) {
            int next_cR = chi_R(site + 1);
            ROCBLAS_CHECK(Traits::gemm(handles_[si],
                rocblas_operation_none, rocblas_operation_none,
                new_k, d_ * next_cR, cR, &one,
                ws.d_svd_work, new_k,
                d_mps_tensors_[site + 1], cR, &zero_val,
                ws.d_T1, new_k));
            allocate_mps_tensor(site + 1, new_chi_R, next_cR);
            HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site + 1], ws.d_T1,
                        (size_t)new_k * d_ * next_cR * sizeof(Scalar),
                        hipMemcpyDeviceToDevice, streams_[si]));
        }
        bond_dims_[site + 1] = new_chi_R;

    } else {  // direction == 'L'
        int new_chi_L = new_k;

        if (use_cpu_svd_) {
            // Upload Vh[:new_k, :] as MPS[site]
            allocate_mps_tensor(site, new_chi_L, cR);
            if (new_k == full_k) {
                HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site], h_Vh_data,
                            full_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));
            } else {
                for (int j = 0; j < n_svd; j++)
                    for (int i = 0; i < new_chi_L; i++)
                        ws.h_svd_tmp[i + j * new_chi_L] = h_Vh_data[i + j * full_k];
                HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site], ws.h_svd_tmp.data(),
                            new_chi_L * n_svd * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));
            }

            // Compute U*S on CPU
            for (int j = 0; j < new_k; j++)
                for (int i = 0; i < m; i++)
                    ws.h_svd_tmp[i + j * m] = Traits::scale_by_real(h_S_data[j], h_U_data[i + j * m]);

            HIP_CHECK(hipMemcpyAsync(ws.d_svd_work, ws.h_svd_tmp.data(),
                        m * new_k * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));
        } else {
            // GPU path
            allocate_mps_tensor(site, new_chi_L, cR);
            if (new_k == full_k) {
                HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site], ws.d_svd_Vh,
                            (size_t)full_k * n_svd * sizeof(Scalar), hipMemcpyDeviceToDevice, streams_[si]));
            } else {
                HIP_CHECK(hipMemcpy2DAsync(
                    d_mps_tensors_[site], new_k * sizeof(Scalar),
                    ws.d_svd_Vh, full_k * sizeof(Scalar),
                    new_k * sizeof(Scalar), n_svd,
                    hipMemcpyDeviceToDevice, streams_[si]));
            }
            // Scale columns of U by S → ws.d_svd_work
            scale_columns_by_real(ws.d_svd_U, m, ws.d_svd_S,
                                  ws.d_svd_work, m, m, new_k, streams_[si]);
        }

        // Absorb U*S into MPS[site-1]: new = old_MPS[site-1] @ (U*S)
        if (site > 0) {
            int prev_cL = chi_L(site - 1);
            ROCBLAS_CHECK(Traits::gemm(handles_[si],
                rocblas_operation_none, rocblas_operation_none,
                prev_cL * d_, new_k, cL, &one,
                d_mps_tensors_[site - 1], prev_cL * d_,
                ws.d_svd_work, m, &zero_val,
                ws.d_T1, prev_cL * d_));
            allocate_mps_tensor(site - 1, prev_cL, new_chi_L);
            HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site - 1], ws.d_T1,
                        (size_t)prev_cL * d_ * new_k * sizeof(Scalar),
                        hipMemcpyDeviceToDevice, streams_[si]));
        }
        bond_dims_[site] = new_chi_L;
    }

    ws.heff_cached_site = -1;
}

// ============================================================================
// Single-site optimization: form theta → Lanczos → SVD split
// ============================================================================

template<typename Scalar>
double PDMRGGPU<Scalar>::optimize_site_single(int site, char direction, int si) {
    int cL = chi_L(site);
    int cR = chi_R(site);
    int theta_size = cL * d_ * cR;
    auto& ws = workspaces_[si];

    // form_theta: just copy MPS[site] to workspace
    HIP_CHECK(hipMemcpyAsync(ws.d_theta, d_mps_tensors_[site],
                              theta_size * sizeof(Scalar), hipMemcpyDeviceToDevice, streams_[si]));

    // Use single-site matvec in Lanczos
    lanczos_use_1site_ = true;
    double energy = lanczos_eigensolver(site, ws.d_theta, theta_size, si);
    lanczos_use_1site_ = false;

    svd_split_single_site(site, ws.d_theta, direction, si);

    return energy;
}

// ============================================================================
// Full-chain sweep methods (two-site, for main PDMRG sweeps)
// ============================================================================

template<typename Scalar>
double PDMRGGPU<Scalar>::sweep_LR_full() {
    double energy = 0.0;
    for (int site = 0; site < L_ - 1; site++) {
        energy = optimize_bond(site, 'R', 0);
        update_left_env(site, 0);
    }
    return energy;
}

template<typename Scalar>
double PDMRGGPU<Scalar>::sweep_RL_full() {
    double energy = 0.0;
    for (int site = L_ - 2; site >= 0; site--) {
        energy = optimize_bond(site, 'L', 0);
        update_right_env(site + 1, 0);
    }
    return energy;
}

// ============================================================================
// Full-chain single-site sweep methods (for warmup and polish)
// ============================================================================

template<typename Scalar>
double PDMRGGPU<Scalar>::sweep_LR_full_1site() {
    double energy = 0.0;
    for (int site = 0; site < L_ - 1; site++) {
        energy = optimize_site_single(site, 'R', 0);
        update_left_env(site, 0);
    }
    // Last site: optimize without SVD (endpoint)
    {
        int cL = chi_L(L_ - 1);
        int cR = chi_R(L_ - 1);
        int theta_size = cL * d_ * cR;
        auto& ws = workspaces_[0];
        HIP_CHECK(hipMemcpyAsync(ws.d_theta, d_mps_tensors_[L_ - 1],
                                  theta_size * sizeof(Scalar), hipMemcpyDeviceToDevice, streams_[0]));
        lanczos_use_1site_ = true;
        energy = lanczos_eigensolver(L_ - 1, ws.d_theta, theta_size, 0);
        lanczos_use_1site_ = false;
        HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[L_ - 1], ws.d_theta,
                                  theta_size * sizeof(Scalar), hipMemcpyDeviceToDevice, streams_[0]));
    }
    return energy;
}

template<typename Scalar>
double PDMRGGPU<Scalar>::sweep_RL_full_1site() {
    double energy = 0.0;
    for (int site = L_ - 1; site >= 1; site--) {
        energy = optimize_site_single(site, 'L', 0);
        update_right_env(site, 0);
    }
    // First site: optimize without SVD (endpoint)
    {
        int cL = chi_L(0);
        int cR = chi_R(0);
        int theta_size = cL * d_ * cR;
        auto& ws = workspaces_[0];
        HIP_CHECK(hipMemcpyAsync(ws.d_theta, d_mps_tensors_[0],
                                  theta_size * sizeof(Scalar), hipMemcpyDeviceToDevice, streams_[0]));
        lanczos_use_1site_ = true;
        energy = lanczos_eigensolver(0, ws.d_theta, theta_size, 0);
        lanczos_use_1site_ = false;
        HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[0], ws.d_theta,
                                  theta_size * sizeof(Scalar), hipMemcpyDeviceToDevice, streams_[0]));
    }
    return energy;
}

// ============================================================================
// Segment sweep methods (restricted range, per-segment stream)
// ============================================================================

template<typename Scalar>
void PDMRGGPU<Scalar>::segment_sweep_LR(int seg_idx) {
    int first = seg_first_[seg_idx];
    int last = seg_last_[seg_idx];
    int si = seg_idx;

    for (int site = first; site < last; site++) {
        optimize_bond(site, 'R', si);
        update_left_env(site, si);
    }
}

template<typename Scalar>
void PDMRGGPU<Scalar>::segment_sweep_RL(int seg_idx) {
    int first = seg_first_[seg_idx];
    int last = seg_last_[seg_idx];
    int si = seg_idx;

    for (int site = last - 1; site >= first; site--) {
        optimize_bond(site, 'L', si);
        update_right_env(site + 1, si);
    }
}


// ============================================================================
// Form theta with V injection: θ = ψ_L · diag(V) · ψ_R  (Stoudenmire Eq. 5)
// V scales the columns of ψ_L (or equivalently the rows of ψ_R) at the
// shared boundary bond before contracting into a two-site tensor.
//
// ψ_L: (cL*d, chi_bond) col-major on GPU
// ψ_R: (chi_bond, d*cR) col-major on GPU
// V:   (chi_bond,) on host → uploaded to d_svd_S workspace
// Result: d_theta = (cL*d, d*cR) = ψ_L · diag(V) · ψ_R
// ============================================================================

template<typename Scalar>
void PDMRGGPU<Scalar>::form_theta_with_V(int site, int boundary_idx, int si) {
    int cL = chi_L(site);
    int chi_bond = bond_dims_[site + 1];
    int cR = chi_R(site + 1);
    auto& ws = workspaces_[si];
    auto& bs = boundary_states_[boundary_idx];

    // Upload V to device (reuse d_svd_S workspace)
    HIP_CHECK(hipMemcpyAsync(ws.d_svd_S, bs.V.data(),
                              chi_bond * sizeof(RealType),
                              hipMemcpyHostToDevice, streams_[si]));

    // Scale: T1 = diag(V) · ψ_R  (scale each row i of ψ_R by V[i])
    // ψ_R is (chi_bond × d*cR) col-major, so row i has stride 1, col stride chi_bond
    // Copy ψ_R to T1 then scale rows
    int psi_R_size = chi_bond * d_ * cR;
    HIP_CHECK(hipMemcpyAsync(ws.d_T1, d_mps_tensors_[site + 1],
                              psi_R_size * sizeof(Scalar),
                              hipMemcpyDeviceToDevice, streams_[si]));
    // Scale rows: T1[i, j] *= V[i] for all j
    // This is equivalent to dgam: scale_rows_by_real from svd_split
    scale_rows_by_real(ws.d_T1, chi_bond, ws.d_svd_S,
                       ws.d_T1, chi_bond, chi_bond, d_ * cR, streams_[si]);

    // Contract: theta = ψ_L · T1 = ψ_L · diag(V) · ψ_R
    Scalar one = Traits::one(), zero_val = Traits::zero();
    ROCBLAS_CHECK(Traits::gemm(handles_[si],
        rocblas_operation_none, rocblas_operation_none,
        cL * d_, d_ * cR, chi_bond,
        &one,
        d_mps_tensors_[site], cL * d_,
        ws.d_T1, chi_bond,
        &zero_val,
        ws.d_theta, cL * d_));
}

// ============================================================================
// Stoudenmire boundary merge+optimize (proper V = Λ⁻¹ coupling)
//
// For each boundary bond:
//   1. Form θ = ψ_L · diag(V) · ψ_R  (Eq. 5)
//   2. Optimize θ with Lanczos eigensolver
//   3. SVD split: θ → U · S · Vh
//   4. Store V_new = 1/clip(S, 1e-12) for next iteration
//   5. MPS[bsite] = U (left-canonical), MPS[bsite+1] = S·Vh
//   6. Update environments from canonical tensors
//
// parity: 0 = even-indexed boundaries, 1 = odd, -1 = all
// ============================================================================

template<typename Scalar>
double PDMRGGPU<Scalar>::merge_and_optimize_boundaries(int parity) {
    // Collect boundaries matching this parity
    std::vector<int> active_boundaries;
    for (int b = 0; b < (int)boundary_bonds_.size(); b++) {
        if (parity >= 0 && (b % 2) != parity) continue;
        active_boundaries.push_back(b);
    }
    if (active_boundaries.empty()) return energy_;

    // Same-parity boundaries are independent (staggered design) — run in parallel
    // Each boundary uses a different stream/workspace when possible
    int n_active = (int)active_boundaries.size();
    int n_avail_streams = (int)streams_.size();

    // For single boundary (n_segments=2), this is just serial on stream 0.
    // For multiple boundaries, parallelize across available streams.
    std::vector<double> energies(n_active, 0.0);

    auto optimize_boundary = [&](int idx) {
        int b = active_boundaries[idx];
        int si = idx % n_avail_streams;  // round-robin across streams
        int bsite = boundary_bonds_[b];
        int cL = chi_L(bsite);
        int cR = chi_R(bsite + 1);
        int theta_size = cL * d_ * d_ * cR;
        auto& ws = workspaces_[si];

        // Step 1: Form θ = ψ_L · diag(V) · ψ_R
        form_theta_with_V(bsite, b, si);

        // Step 2: Optimize θ with eigensolver
        energies[idx] = lanczos_eigensolver(bsite, ws.d_theta, theta_size, si);

        // Step 3: Accurate SVD split at boundary (Stoudenmire Appendix)
        // Standard SVD has poor relative accuracy for small singular values.
        // Since V = 1/S amplifies errors, we need the recursive accurate SVD
        // that achieves uniform relative accuracy for ALL singular values.
        int m = cL * d_;
        int n_svd = d_ * cR;
        int full_k = std::min(m, n_svd);
        int k = std::min(full_k, chi_max_);

        // Copy theta from GPU to host
        HIP_CHECK(hipMemcpyAsync(ws.h_svd_A.data(), ws.d_theta,
                                  (size_t)m * n_svd * sizeof(Scalar),
                                  hipMemcpyDeviceToHost, streams_[si]));
        HIP_CHECK(hipStreamSynchronize(streams_[si]));

        // Run accurate SVD on CPU (recursive refinement for small singular values)
        accurate_svd<Scalar>(m, n_svd,
                             ws.h_svd_A.data(), m,
                             ws.h_svd_U.data(), m,
                             ws.h_svd_S.data(),
                             ws.h_svd_Vh.data(), full_k);

        // Truncate to chi_max, dropping near-zero singular values
        int new_k = k;
        for (int i = 0; i < new_k; i++) {
            if (ws.h_svd_S[i] < 1e-14) { new_k = i; break; }
        }
        if (new_k == 0) new_k = 1;

        // Upload S to device (needed for scale_rows_by_real)
        HIP_CHECK(hipMemcpyAsync(ws.d_svd_S, ws.h_svd_S.data(),
                                  new_k * sizeof(RealType),
                                  hipMemcpyHostToDevice, streams_[si]));

        // MPS[bsite] = U[:, :new_k] (left-canonical)
        allocate_mps_tensor(bsite, cL, new_k);
        HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[bsite], ws.h_svd_U.data(),
                    (size_t)m * new_k * sizeof(Scalar),
                    hipMemcpyHostToDevice, streams_[si]));

        // MPS[bsite+1] = diag(S) @ Vh[:new_k, :] (absorbs singular values)
        if (new_k == full_k) {
            HIP_CHECK(hipMemcpyAsync(ws.d_svd_work, ws.h_svd_Vh.data(),
                        (size_t)full_k * n_svd * sizeof(Scalar),
                        hipMemcpyHostToDevice, streams_[si]));
        } else {
            // Pack contiguous new_k rows from leading-dim full_k layout
            for (int j = 0; j < n_svd; j++)
                for (int i = 0; i < new_k; i++)
                    ws.h_svd_tmp[i + j * new_k] = ws.h_svd_Vh[i + j * full_k];
            HIP_CHECK(hipMemcpyAsync(ws.d_svd_work, ws.h_svd_tmp.data(),
                        (size_t)new_k * n_svd * sizeof(Scalar),
                        hipMemcpyHostToDevice, streams_[si]));
        }
        allocate_mps_tensor(bsite + 1, new_k, cR);
        int vh_ld = (new_k == full_k) ? full_k : new_k;
        scale_rows_by_real(ws.d_svd_work, vh_ld, ws.d_svd_S,
                           d_mps_tensors_[bsite + 1], new_k, new_k, n_svd, streams_[si]);

        bond_dims_[bsite + 1] = new_k;
        ws.heff_cached_site = -1;

        // Step 4: Update V = 1/clip(S, 1e-12) for next iteration
        boundary_states_[b].chi = new_k;
        boundary_states_[b].V.resize(new_k);

        const RealType reg = RealType(1e-12);
        for (int i = 0; i < new_k; i++) {
            RealType s_val = ws.h_svd_S[i];
            if (s_val < reg) s_val = reg;
            boundary_states_[b].V[i] = RealType(1.0) / s_val;
        }

        // Step 5: Update environments from CANONICAL tensors
        // L_env from U (left-canonical) — already in MPS[bsite]
        update_left_env(bsite, si);

        // R_env must be built from Vh (right-canonical), NOT from S·Vh.
        // Using S·Vh gives norm = S² ≠ I, which breaks the standard eigenvalue
        // assumption (N_eff = I) in subsequent Lanczos eigensolves.
        // Temporarily swap Vh into MPS[bsite+1], build R_env, then restore S·Vh.

        // Save d_mps_tensors_[bsite+1] pointer (points to S·Vh)
        Scalar* d_SVh_tensor = d_mps_tensors_[bsite + 1];

        // Use pre-allocated Vh buffer (avoids hot-path hipMalloc/hipFree)
        size_t vh_size = (size_t)new_k * n_svd;

        // Upload Vh[:new_k, :] (without S) to the pre-allocated buffer
        if (new_k == full_k) {
            HIP_CHECK(hipMemcpyAsync(ws.d_Vh_canonical, ws.h_svd_Vh.data(),
                        vh_size * sizeof(Scalar),
                        hipMemcpyHostToDevice, streams_[si]));
        } else {
            // Sync before reusing h_svd_tmp — the earlier async H2D copy (S·Vh upload)
            // may still be reading from it.
            HIP_CHECK(hipStreamSynchronize(streams_[si]));
            for (int j = 0; j < n_svd; j++)
                for (int i = 0; i < new_k; i++)
                    ws.h_svd_tmp[i + j * new_k] = ws.h_svd_Vh[i + j * full_k];
            HIP_CHECK(hipMemcpyAsync(ws.d_Vh_canonical, ws.h_svd_tmp.data(),
                        vh_size * sizeof(Scalar),
                        hipMemcpyHostToDevice, streams_[si]));
        }

        // Swap Vh into MPS slot, build R_env, then restore S·Vh
        d_mps_tensors_[bsite + 1] = ws.d_Vh_canonical;
        update_right_env(bsite + 1, si);
        d_mps_tensors_[bsite + 1] = d_SVh_tensor;
        HIP_CHECK(hipStreamSynchronize(streams_[si]));
    };

    if (n_active == 1) {
        // Single boundary — no threading overhead
        optimize_boundary(0);
    } else {
        // Multiple boundaries — parallelize
        std::vector<std::thread> threads(n_active);
        for (int i = 0; i < n_active; i++) {
            threads[i] = std::thread(optimize_boundary, i);
        }
        for (auto& t : threads) t.join();
        // Sync all used streams
        for (int i = 0; i < std::min(n_active, n_avail_streams); i++) {
            HIP_CHECK(hipStreamSynchronize(streams_[i]));
        }
    }

    // Return the last boundary energy (same semantics as before)
    return energies.back();
}

// ============================================================================
// Main algorithm
// ============================================================================

template<typename Scalar>
double PDMRGGPU<Scalar>::run(int n_outer_sweeps, int n_local_sweeps, int n_warmup, int n_polish, int n_recal) {
    build_initial_environments();

    // Timer starts AFTER env build — measures sweep-to-convergence only
    auto t_start = std::chrono::high_resolution_clock::now();

    // Warmup: single-site sweeps (cheaper eigsh: chi*d vs chi*d²)
    double warmup_energy = 0.0;
    double prev_warmup_energy = 1e30;
    for (int sw = 0; sw < n_warmup; sw++) {
        sweep_LR_full_1site();
        warmup_energy = sweep_RL_full_1site();
        double dE = std::abs(warmup_energy - prev_warmup_energy);
        prev_warmup_energy = warmup_energy;
        if (dE < tol_ && sw > 0) break;
    }
    auto t_warmup = std::chrono::high_resolution_clock::now();
    double warmup_sec = std::chrono::duration<double>(t_warmup - t_start).count();
    if (n_warmup > 0)
        printf("[phase] warmup: %.3f s (%d sweep%s)\n", warmup_sec, n_warmup, n_warmup > 1 ? "s" : "");

    // === Main PDMRG loop (Stoudenmire staggered sweeps) ===
    // Replaces O(L) full-chain coupling with O(P) boundary-only merge+optimize.
    // Staggered pattern: even segments LR / odd RL, then reverse.
    // This ensures fresh L_env + R_env at each boundary when it's optimized.
    // Re-initialize V = ones after warmup (bond dims may have changed)
    initialize_boundary_states();
    double energy_prev = warmup_energy;
    energy_ = warmup_energy;
    bool outer_converged = false;

    // Parallel sweep launcher: one CPU thread per segment, each with its own HIP stream
    auto parallel_sweep = [this](auto sweep_fn) {
        std::vector<std::thread> threads(n_segments_);
        for (int k = 0; k < n_segments_; k++) {
            threads[k] = std::thread([this, k, &sweep_fn]{ sweep_fn(this, k); });
        }
        for (auto& t : threads) t.join();
        // Sync per-segment GPU streams — segment sweeps launch async GPU work that
        // must complete before boundary coupling reads their outputs on stream 0
        for (int s = 0; s < n_segments_; s++) {
            HIP_CHECK(hipStreamSynchronize(streams_[s]));
        }
    };

    bool has_odd_boundaries = ((int)boundary_bonds_.size() > 1);
    int actual_outer = 0;

    for (int outer = 0; outer < n_outer_sweeps; outer++) {
        actual_outer = outer + 1;
        double energy_before_local = energy_;
        for (int local_sw = 0; local_sw < n_local_sweeps; local_sw++) {
            // Half-sweep 1: even segments LR, odd segments RL
            // After this, even-numbered boundaries have fresh L_env + R_env:
            //   even seg k swept LR → L_env[seg_last_[k]] fresh
            //   odd seg k+1 swept RL → R_env[seg_first_[k+1]+1] fresh
            parallel_sweep([](PDMRGGPU* self, int k) {
                if (k % 2 == 0) self->segment_sweep_LR(k);
                else             self->segment_sweep_RL(k);
            });

            // Merge+optimize at even boundaries
            if (boundary_bonds_.size() > 0) {
                energy_ = merge_and_optimize_boundaries(0);
            }

            // Half-sweep 2: even segments RL, odd segments LR
            // After this, odd-numbered boundaries have fresh environments
            parallel_sweep([](PDMRGGPU* self, int k) {
                if (k % 2 == 0) self->segment_sweep_RL(k);
                else             self->segment_sweep_LR(k);
            });

            // Merge+optimize at odd boundaries (if any exist)
            if (has_odd_boundaries) {
                energy_ = merge_and_optimize_boundaries(1);
            }

            // Early exit within local sweeps: if energy converged mid-local, skip remaining
            double dE_local = std::abs(energy_ - energy_before_local);
            if (dE_local < tol_ && local_sw > 0) break;
            energy_before_local = energy_;
        }

        double dE = std::abs(energy_ - energy_prev);
        energy_prev = energy_;

        if (dE < tol_) {
            printf("Converged after %d outer iterations!\n", outer + 1);
            outer_converged = true;
            break;
        }

        // Recalibration: periodic serial full-chain two-site sweep to prevent
        // parallel segments from diverging into local minima.
        // Two-site sweep can adjust bond dims and converges faster than single-site.
        if (n_recal > 0 && ((outer + 1) % n_recal == 0) && outer + 1 < n_outer_sweeps) {
            build_initial_environments();
            sweep_LR_full();
            energy_ = sweep_RL_full();
            energy_prev = energy_;
            // Re-init boundary V from fresh canonical MPS
            initialize_boundary_states();
        }
    }
    auto t_parallel = std::chrono::high_resolution_clock::now();
    double parallel_sec = std::chrono::duration<double>(t_parallel - t_warmup).count();
    printf("[phase] parallel: %.3f s (%d outer × %d local sweeps)\n",
           parallel_sec, actual_outer, n_local_sweeps);

    // === Polish phase: single-site full-chain sweeps ===
    // Refines energy after parallel segment sweeps. Single-site only (cheaper,
    // and bond-dim adjustment is handled by recalibration sweeps).
    // Skipped when outer loop already converged or n_polish == 0.
    if (n_segments_ > 1 && n_polish > 0 && !outer_converged) {
        // Full env rebuild: after parallel segment sweeps + boundary merges,
        // environments across the chain are stale (built from old MPS tensors).
        // An incremental 2-site rebuild propagates this staleness — we need
        // a from-scratch walk from the trivial L_env[0] and R_env[L].
        build_initial_environments();

        for (int sw = 0; sw < n_polish; sw++) {
            sweep_LR_full_1site();
            double eRL = sweep_RL_full_1site();
            double dE = std::abs(eRL - energy_);
            energy_ = eRL;
            if (dE < tol_) {
                printf("Polish converged after %d sweeps\n", sw + 1);
                break;
            }
        }
    }
    auto t_polish = std::chrono::high_resolution_clock::now();
    double polish_sec = std::chrono::duration<double>(t_polish - t_parallel).count();
    if (n_polish > 0)
        printf("[phase] polish: %.3f s (%d sweep%s)\n", polish_sec, n_polish, n_polish > 1 ? "s" : "");

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(t_end - t_start).count();
    printf("Final energy: %.12f\n", energy_);
    printf("Total wall time: %.3f s (warmup=%.1f%% parallel=%.1f%% polish=%.1f%%)\n",
           total_time,
           total_time > 0 ? 100.0 * warmup_sec / total_time : 0.0,
           total_time > 0 ? 100.0 * parallel_sec / total_time : 0.0,
           total_time > 0 ? 100.0 * polish_sec / total_time : 0.0);
    report_timers();

    return energy_;
}

// ============================================================================
// Phase timers
// ============================================================================

template<typename Scalar>
void PDMRGGPU<Scalar>::init_timers() {
    t_lanczos_.init("lanczos", opts_.profile);
    t_apply_heff_.init("apply_heff", opts_.profile);
    t_svd_.init("svd", opts_.profile);
    t_absorb_.init("absorb", opts_.profile);
    t_env_update_.init("env_update", opts_.profile);
}

template<typename Scalar>
void PDMRGGPU<Scalar>::report_timers() {
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
// Utility methods
// ============================================================================

template<typename Scalar>
void PDMRGGPU<Scalar>::get_mps(std::vector<std::vector<Scalar>>& h_mps) const {
    h_mps.resize(L_);
    for (int i = 0; i < L_; i++) {
        int size = chi_L(i) * d_ * chi_R(i);
        h_mps[i].resize(size);
        HIP_CHECK(hipMemcpy(h_mps[i].data(), d_mps_tensors_[i],
                            size * sizeof(Scalar), hipMemcpyDeviceToHost));
    }
}

#endif // PDMRG_GPU_IMPL_H
