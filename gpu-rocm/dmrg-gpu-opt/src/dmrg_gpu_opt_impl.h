#ifndef DMRG_GPU_OPT_IMPL_H
#define DMRG_GPU_OPT_IMPL_H

#include <rocsolver/rocsolver.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <chrono>

#include "../../common/hip_check.h"

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

// promote_double_to_complex now defined in common/scalar_traits.h
// (round-5 single-source-of-truth promotion).

// Profiling counters (reset per sweep pair)
static double prof_davidson_ms = 0, prof_svd_ms = 0, prof_env_ms = 0;
static int prof_davidson_iters = 0, prof_site_count = 0;
static int prof_heff_calls = 0;

// ============================================================================
// Constructor
// ============================================================================

template<typename Scalar>
DMRGGPUOpt<Scalar>::DMRGGPUOpt(int L, int d, int chi_max, int D_mpo, double tol)
    : L_(L), d_(d), chi_max_(pad_mfma16(chi_max)), chi_max_user_(chi_max),
      D_mpo_(D_mpo), D_mpo_actual_(D_mpo), tol_(tol), energy_(0.0) {

    opts_.load_from_env();

    // LANCZOS_GRAPH safeguard: dmrg-gpu-opt uses Block-Davidson as its
    // eigensolver, not Lanczos. Block-Davidson calls apply_heff(site, V+j*dim,
    // AV+j*dim) with a variable output pointer per subspace column. HIP graph
    // capture burns the first-seen output address into the graph; replaying
    // writes to the stale address instead of AV+j*dim, Rayleigh-Ritz reads
    // garbage, the outer loop never converges, and the process hangs. Force
    // the flag off here; users that want captured apply_heff should use
    // dmrg-gpu (Lanczos, reuses one Hv buffer) or pdmrg-gpu (fixed bounce
    // buffer pattern). Print once so benchmark drivers can log the override.
    if (opts_.lanczos_graph) {
        std::fprintf(stderr,
            "[dmrg-gpu-opt] LANCZOS_GRAPH=1 is incompatible with Block-Davidson "
            "(variable output pointer per subspace column). Disabling.\n");
        opts_.lanczos_graph = false;
    }

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

    if (chi_max_ != chi_max_user_) {
        printf("[OPT] MFMA-16 padding: chi_max %d -> %d\n", chi_max_user_, chi_max_);
    }

    // Bond dimensions (using padded chi_max for MFMA alignment)
    bond_dims_.resize(L + 1);
    bond_dims_[0] = 1;
    bond_dims_[L] = 1;
    for (int i = 1; i < L; i++) {
        double exact_dim = pow((double)d_, std::min(i, L - i));
        bond_dims_[i] = (exact_dim > chi_max_user_) ? chi_max_user_ : (int)exact_dim;
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

    // SPARSE_MPO index lists (populated in set_mpo when opts_.sparse_mpo is on)
    d_WL_nnz_rows_.resize(L, nullptr);
    d_WL_nnz_cols_.resize(L, nullptr);
    wl_nnz_rows_count_.assign(L, 0);
    wl_nnz_cols_count_.assign(L, 0);
    h_WL_nnz_rows_.resize(L);
    h_WL_nnz_cols_.resize(L);

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

    // Lanczos workspace (fallback)
    theta_size_max_ = chi_max_ * d_ * chi_max_;
    max_lanczos_iter_ = std::min(100, theta_size_max_);
    HIP_CHECK(hipMalloc(&d_theta_, theta_size_max_ * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_heff_result_, theta_size_max_ * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_lanczos_v_, (size_t)max_lanczos_iter_ * theta_size_max_ * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_ritz_coeffs_, max_lanczos_iter_ * sizeof(Scalar)));

    // Device scalars for sync-free Lanczos (device pointer mode). Ported
    // from dmrg-gpu for the H10 device-pointer Lanczos path.
    HIP_CHECK(hipMalloc(&d_dot_result_, sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_nrm2_result_, sizeof(RealType)));
    HIP_CHECK(hipMalloc(&d_neg_alpha_, sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_neg_overlap_, sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_inv_nrm_, sizeof(RealType)));
    HIP_CHECK(hipMalloc(&d_alpha_dev_, max_lanczos_iter_ * sizeof(RealType)));
    HIP_CHECK(hipMalloc(&d_beta_dev_, max_lanczos_iter_ * sizeof(RealType)));
    HIP_CHECK(hipMalloc(&d_neg_beta_scalars_, max_lanczos_iter_ * sizeof(Scalar)));

    HIP_CHECK(hipMalloc(&d_const_one_, sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_const_zero_, sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_const_neg_one_, sizeof(Scalar)));
    {
        Scalar one = Traits::one(), zero = Traits::zero(), neg_one = Traits::neg(one);
        HIP_CHECK(hipMemcpy(d_const_one_, &one, sizeof(Scalar), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_const_zero_, &zero, sizeof(Scalar), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_const_neg_one_, &neg_one, sizeof(Scalar), hipMemcpyHostToDevice));
    }

    // Length-D ones vector for the R3-F1 full-batched-collapse Step-3 path
    HIP_CHECK(hipMalloc(&d_ones_D_, D_mpo_ * sizeof(Scalar)));
    {
        std::vector<Scalar> h_ones(D_mpo_, Traits::one());
        HIP_CHECK(hipMemcpy(d_ones_D_, h_ones.data(),
                            D_mpo_ * sizeof(Scalar), hipMemcpyHostToDevice));
    }

    // Batched GEMM pointer arrays (Step 1) — main + env stream variants so
    // env_update and apply_heff/absorb don't clobber each other's pointer tables.
    int batch_max = D_mpo_ * d_;
    HIP_CHECK(hipMalloc(&d_batch_A_, batch_max * sizeof(Scalar*)));
    HIP_CHECK(hipMalloc(&d_batch_B_, batch_max * sizeof(Scalar*)));
    HIP_CHECK(hipMalloc(&d_batch_C_, batch_max * sizeof(Scalar*)));
    HIP_CHECK(hipMalloc(&d_batch_A_env_, batch_max * sizeof(Scalar*)));
    HIP_CHECK(hipMalloc(&d_batch_B_env_, batch_max * sizeof(Scalar*)));
    HIP_CHECK(hipMalloc(&d_batch_C_env_, batch_max * sizeof(Scalar*)));


    // SVD workspace (on-device path is the new default; CPU path is opt-in)
    int svd_max_dim = chi_max_ * d_;
    int svd_max_k = chi_max_;
    HIP_CHECK(hipMalloc(&d_svd_A_,    theta_size_max_ * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_svd_U_,    (size_t)svd_max_dim * svd_max_k * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_svd_S_,    svd_max_k * sizeof(RealType)));
    HIP_CHECK(hipMalloc(&d_svd_Vh_,   (size_t)svd_max_k * svd_max_dim * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_svd_E_,    svd_max_k * sizeof(RealType)));
    HIP_CHECK(hipMalloc(&d_svd_info_, sizeof(int)));
    HIP_CHECK(hipMalloc(&d_svd_work_, theta_size_max_ * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_svdj_residual_, sizeof(double)));
    HIP_CHECK(hipMalloc(&d_svdj_n_sweeps_, sizeof(rocblas_int)));

    // rocsolver_dsteqr scratch — replaces 2× host LAPACK dstev_ in
    // lanczos_eigensolver. Lanczos α/β are kept host-resident in this
    // variant; the dsteqr path H2D's the small tridiagonal into d_steqr_D/E.
    HIP_CHECK(hipMalloc(&d_steqr_D_,    max_lanczos_iter_ * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_steqr_E_,    max_lanczos_iter_ * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_steqr_C_,    (size_t)max_lanczos_iter_ * max_lanczos_iter_ * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_steqr_info_, sizeof(rocblas_int)));

    // Randomized SVD workspace (Halko-Martinsson-Tropp). Allocated EAGERLY
    // regardless of use_rsvd_ default — round-5 A7 fix. Previously gated on
    // use_rsvd_ at construction, but set_rsvd(true) post-ctor (the J2 setter
    // surface) would have left buffers nullptr and crashed on first RSVD
    // call. Eager allocation costs ~4 MB at chi=256 (negligible vs the
    // multi-GB MPS) and lets the user toggle use_rsvd_ at any time safely.
    // Also resizes d_svd_S/E/U/Vh from chi_max to chi_max + OVERSAMPLE since
    // RSVD's small-SVD output can produce that many singular values.
    {
        int r_max = chi_max_ + RSVD_OVERSAMPLE_;
        if (d_svd_S_)  HIP_CHECK(hipFree(d_svd_S_));
        if (d_svd_E_)  HIP_CHECK(hipFree(d_svd_E_));
        if (d_svd_U_)  HIP_CHECK(hipFree(d_svd_U_));
        if (d_svd_Vh_) HIP_CHECK(hipFree(d_svd_Vh_));
        HIP_CHECK(hipMalloc(&d_svd_S_,  r_max * sizeof(RealType)));
        HIP_CHECK(hipMalloc(&d_svd_E_,  r_max * sizeof(RealType)));
        HIP_CHECK(hipMalloc(&d_svd_U_,  (size_t)svd_max_dim * r_max * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&d_svd_Vh_, (size_t)r_max * svd_max_dim * sizeof(Scalar)));

        rsvd_r_max_ = r_max;
        int n_max = svd_max_dim;
        int m_max = svd_max_dim;
        HIP_CHECK(hipMalloc(&d_rsvd_omega_,   (size_t)n_max * rsvd_r_max_ * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&d_rsvd_Y_,       (size_t)m_max * rsvd_r_max_ * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&d_rsvd_tau_,     (size_t)rsvd_r_max_ * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&d_rsvd_B_,       (size_t)rsvd_r_max_ * n_max * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&d_rsvd_U_small_, (size_t)rsvd_r_max_ * rsvd_r_max_ * sizeof(Scalar)));
    }

    // CPU SVD workspace
    h_svd_A_.resize(theta_size_max_);
    h_svd_U_.resize((size_t)svd_max_dim * chi_max_);
    h_svd_S_.resize(chi_max_);
    h_svd_Vh_.resize((size_t)chi_max_ * svd_max_dim);
    h_svd_tmp_.resize((size_t)svd_max_dim * chi_max_);
    h_svd_rwork_.resize(Traits::svd_rwork_size(svd_max_dim, svd_max_dim));

    // Query optimal LAPACK workspace for ALL possible (m, n) combinations
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

    // Block-Davidson workspace
    davidson_b_ = 4;
    davidson_max_sub_ = std::min(davidson_b_ * 8, theta_size_max_);
    HIP_CHECK(hipMalloc(&d_dav_V_,     (size_t)theta_size_max_ * davidson_max_sub_ * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_dav_AV_,    (size_t)theta_size_max_ * davidson_max_sub_ * sizeof(Scalar)));
    {
        size_t dav_work_sz = std::max((size_t)theta_size_max_ * davidson_b_,
                                       (size_t)davidson_max_sub_ * davidson_max_sub_);
        HIP_CHECK(hipMalloc(&d_dav_work_,  dav_work_sz * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&d_dav_work2_, dav_work_sz * sizeof(Scalar)));
    }
    // On-device Rayleigh-Ritz scratch — replaces the per-iteration
    // host-LAPACK syev roundtrip (round-7 C2). Eigenvalues written by
    // rocsolver_syevd to d_dav_eigvals_; tiny host mirror only for
    // energy/conv checks.
    HIP_CHECK(hipMalloc(&d_dav_eigvals_, davidson_max_sub_ * sizeof(RealType)));
    HIP_CHECK(hipMalloc(&d_dav_E_,       davidson_max_sub_ * sizeof(RealType)));
    HIP_CHECK(hipMalloc(&d_dav_info_,    sizeof(rocblas_int)));
    h_dav_eigvals_.resize(davidson_max_sub_);

    // LANCZOS_GRAPH: bounce buffer so captured apply_heff graphs read from a
    // fixed address regardless of which Lanczos v_i the caller passes in.
    if (opts_.lanczos_graph) {
        HIP_CHECK(hipMalloc(&d_heff_input_, (size_t)theta_size_max_ * sizeof(Scalar)));
    }
}

// ============================================================================
// Destructor
// ============================================================================

template<typename Scalar>
DMRGGPUOpt<Scalar>::~DMRGGPUOpt() {
    free_gpu_resources();
}

template<typename Scalar>
void DMRGGPUOpt<Scalar>::free_gpu_resources() {
    for (auto ptr : d_mps_tensors_) if (ptr) hipFree(ptr);
    for (auto ptr : d_mpo_tensors_) if (ptr) hipFree(ptr);
    for (auto ptr : d_W_left_) if (ptr) hipFree(ptr);
    for (auto ptr : d_W_right_) if (ptr) hipFree(ptr);
    for (auto ptr : d_WL_nnz_rows_) if (ptr) hipFree(ptr);
    for (auto ptr : d_WL_nnz_cols_) if (ptr) hipFree(ptr);
    for (auto ptr : d_L_envs_) if (ptr) hipFree(ptr);
    for (auto ptr : d_R_envs_) if (ptr) hipFree(ptr);

    if (d_theta_) hipFree(d_theta_);
    if (d_heff_result_) hipFree(d_heff_result_);
    if (d_lanczos_v_) hipFree(d_lanczos_v_);
    if (d_ritz_coeffs_) hipFree(d_ritz_coeffs_);
    if (d_T1_) hipFree(d_T1_);
    if (d_T2_) hipFree(d_T2_);
    if (d_T1_env_) hipFree(d_T1_env_);
    if (d_T2_env_) hipFree(d_T2_env_);
    if (d_dot_result_) hipFree(d_dot_result_);
    if (d_nrm2_result_) hipFree(d_nrm2_result_);
    if (d_neg_alpha_) hipFree(d_neg_alpha_);
    if (d_neg_overlap_) hipFree(d_neg_overlap_);
    if (d_inv_nrm_) hipFree(d_inv_nrm_);
    if (d_alpha_dev_) hipFree(d_alpha_dev_);
    if (d_beta_dev_) hipFree(d_beta_dev_);
    if (d_neg_beta_scalars_) hipFree(d_neg_beta_scalars_);
    if (d_const_one_) hipFree(d_const_one_);
    if (d_const_zero_) hipFree(d_const_zero_);
    if (d_const_neg_one_) hipFree(d_const_neg_one_);
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
    if (d_steqr_D_) hipFree(d_steqr_D_);
    if (d_steqr_E_) hipFree(d_steqr_E_);
    if (d_steqr_C_) hipFree(d_steqr_C_);
    if (d_steqr_info_) hipFree(d_steqr_info_);
    if (d_rsvd_omega_)   hipFree(d_rsvd_omega_);
    if (d_rsvd_Y_)       hipFree(d_rsvd_Y_);
    if (d_rsvd_tau_)     hipFree(d_rsvd_tau_);
    if (d_rsvd_B_)       hipFree(d_rsvd_B_);
    if (d_rsvd_U_small_) hipFree(d_rsvd_U_small_);

    // Block-Davidson workspace
    if (d_dav_V_) hipFree(d_dav_V_);
    if (d_dav_AV_) hipFree(d_dav_AV_);
    if (d_dav_work_) hipFree(d_dav_work_);
    if (d_dav_work2_) hipFree(d_dav_work2_);
    if (d_dav_eigvals_) hipFree(d_dav_eigvals_);
    if (d_dav_E_) hipFree(d_dav_E_);
    if (d_dav_info_) hipFree(d_dav_info_);

    // LANCZOS_GRAPH: destroy cached graph execs and bounce buffer
    for (auto& kv : apply_heff_graph_cache_) {
        hipGraphExecDestroy(kv.second);
    }
    apply_heff_graph_cache_.clear();
    if (d_heff_input_) hipFree(d_heff_input_);

    rocblas_destroy_handle(rocblas_h_);
    rocblas_destroy_handle(rocblas_h_env_);
    hipEventDestroy(event_canon_ready_);
    hipEventDestroy(event_env_done_);
    hipStreamDestroy(stream_);
    hipStreamDestroy(stream_env_);
}

// ============================================================================
// Memory management
// ============================================================================

template<typename Scalar>
void DMRGGPUOpt<Scalar>::allocate_mps_tensor(int site, int cL, int cR) {
    // Pre-allocate at chi_max to avoid hipFree+hipMalloc per bond (each
    // pair is a stream sync point). Logical size tracked by bond_dims_;
    // physical size is always chi_max·d·chi_max. Mirrors dmrg-gpu.
    (void)cL; (void)cR;
    if (!d_mps_tensors_[site]) {
        size_t max_sz = (size_t)chi_max_ * d_ * chi_max_ * sizeof(Scalar);
        HIP_CHECK(hipMalloc(&d_mps_tensors_[site], max_sz));
    }
}

template<typename Scalar>
void DMRGGPUOpt<Scalar>::ensure_L_env_alloc(int idx, int chi) {
    if (chi > L_env_alloc_chi_[idx]) {
        if (d_L_envs_[idx]) HIP_CHECK(hipFree(d_L_envs_[idx]));
        int sz = chi * D_mpo_ * chi;
        HIP_CHECK(hipMalloc(&d_L_envs_[idx], sz * sizeof(Scalar)));
        L_env_alloc_chi_[idx] = chi;
    }
}

template<typename Scalar>
void DMRGGPUOpt<Scalar>::ensure_R_env_alloc(int idx, int chi) {
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
void DMRGGPUOpt<Scalar>::initialize_mps_random(double scale) {
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
void DMRGGPUOpt<Scalar>::initialize_mps_product() {
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
void DMRGGPUOpt<Scalar>::initialize_mps_neel() {
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
void DMRGGPUOpt<Scalar>::set_mpo(const std::vector<Scalar*>& h_mpo_tensors) {
    // D_use: padded bond dim used for all internal buffers (D_PAD on).
    // D_act: user's MPO bond dim.
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

        // Precompute W_left and W_right matrices
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

        // SPARSE_MPO: build nonzero row/column index lists for W_left.
        // Row r = w*d+s nonzero iff any column has |val| > 0; col c = wp*d+sp
        // analogously. Padded rows/cols (w >= D_act) are always zero so they
        // get auto-excluded when the flag is on.
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
            h_WL_nnz_rows_[i] = std::move(nnz_rows);
            h_WL_nnz_cols_[i] = std::move(nnz_cols);
            if (i == 0) {
                std::fprintf(stderr,
                    "[SPARSE_MPO] site 0: W shape (%d x %d), nnz rows=%d, nnz cols=%d (%.0f%% sparse)\n",
                    rows, cols, wl_nnz_rows_count_[i], wl_nnz_cols_count_[i],
                    100.0 * (1.0 - (double)(wl_nnz_rows_count_[i] * wl_nnz_cols_count_[i]) / (rows * cols)));
            }
        }
    }
}

// ============================================================================
// GEMM-based tensor contractions
// ============================================================================

template<typename Scalar>
void DMRGGPUOpt<Scalar>::apply_heff(int site, const Scalar* d_theta_in, Scalar* d_result) {
    int cL = chi_L(site);
    int cR = chi_R(site);
    int D = D_mpo_, d = d_;
    Scalar one = Traits::one(), zero_val = Traits::zero();

    // LANCZOS_GRAPH: stage caller's theta into a fixed-address bounce buffer
    // BEFORE any capture window, then either replay the cached graph or
    // capture a new one for this (site, cL, cR) shape. See dmrg_gpu_impl.h
    // for the full rationale.
    const Scalar* theta_src = d_theta_in;
    bool graph_capture_miss = false;
    if (opts_.lanczos_graph) {
        int n_theta = cL * d * cR;
        HIP_CHECK(hipMemcpyAsync(d_heff_input_, d_theta_in,
                                 (size_t)n_theta * sizeof(Scalar),
                                 hipMemcpyDeviceToDevice, stream_));
        theta_src = d_heff_input_;

        uint64_t key = graph_key(site, cL, cR);
        auto it = apply_heff_graph_cache_.find(key);
        if (it != apply_heff_graph_cache_.end()) {
            HIP_CHECK(hipGraphLaunch(it->second, stream_));
            return;
        }
        graph_capture_miss = true;
        HIP_CHECK(hipStreamBeginCapture(stream_, hipStreamCaptureModeGlobal));
    }

    Scalar* L_env = d_L_envs_[site];
    Scalar* R_env = d_R_envs_[site + 1];
    Scalar* W_mat = d_W_left_[site];
    Scalar* V = d_T1_;
    Scalar* U = d_T2_;

    // SPARSE_MPO: compact Step 1 / Step 3 batches to non-zero (w,s)/(wp,sp)
    // rows/cols of W_left. Step 1 uses the full V layout (Step 2 reads V
    // densely), so skipped slots must be zero — enforced via hipMemsetAsync.
    // Step 3 accumulates into d_result; for sparse mode we zero d_result first
    // then use beta=1 for all nnz contributions.
    const bool sparse_s1 = opts_.sparse_mpo
                         && wl_nnz_rows_count_[site] > 0
                         && wl_nnz_rows_count_[site] < D * d;
    const bool sparse_s3 = opts_.sparse_mpo
                         && wl_nnz_cols_count_[site] > 0
                         && wl_nnz_cols_count_[site] < D * d;

    // Step 1: V_ws[a',b] = L_w^T[a',a] * theta_s[a,b]  (batched GEMM)
    {
        if (sparse_s1) {
            HIP_CHECK(hipMemsetAsync(V, 0, (size_t)D * d * cL * cR * sizeof(Scalar), stream_));
            int nnz = wl_nnz_rows_count_[site];
            const std::vector<int>& h_nnz = h_WL_nnz_rows_[site];
            std::vector<Scalar*> h_A(nnz), h_B(nnz), h_C(nnz);
            for (int idx = 0; idx < nnz; idx++) {
                int ws = h_nnz[idx];
                int w = ws / d, s = ws % d;
                h_A[idx] = L_env + w * cL;
                h_B[idx] = const_cast<Scalar*>(theta_src) + s * cL;
                h_C[idx] = V + ws * cL * cR;
            }
            HIP_CHECK(hipMemcpyAsync(d_batch_A_, h_A.data(), nnz*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
            HIP_CHECK(hipMemcpyAsync(d_batch_B_, h_B.data(), nnz*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
            HIP_CHECK(hipMemcpyAsync(d_batch_C_, h_C.data(), nnz*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
            ROCBLAS_CHECK(Traits::gemm_batched(rocblas_h_,
                Traits::op_t, rocblas_operation_none,
                cL, cR, cL,
                &one,
                (const Scalar**)d_batch_A_, cL * D,
                (const Scalar**)d_batch_B_, cL * d,
                &zero_val,
                d_batch_C_, cL,
                nnz));
        } else {
            std::vector<Scalar*> h_A(D * d), h_B(D * d), h_C(D * d);
            for (int w = 0; w < D; w++)
                for (int s = 0; s < d; s++) {
                    int ws = w * d + s;
                    h_A[ws] = L_env + w * cL;
                    h_B[ws] = const_cast<Scalar*>(theta_src) + s * cL;
                    h_C[ws] = V + ws * cL * cR;
                }
            HIP_CHECK(hipMemcpyAsync(d_batch_A_, h_A.data(), D*d*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
            HIP_CHECK(hipMemcpyAsync(d_batch_B_, h_B.data(), D*d*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
            HIP_CHECK(hipMemcpyAsync(d_batch_C_, h_C.data(), D*d*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
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
    // Use strided_batched GEMM only when: large chi AND d<=2
    // (d>=3 causes cache line contention from interleaved output layout)
    if (sparse_s3) {
        HIP_CHECK(hipMemsetAsync(d_result, 0, (size_t)cL * d * cR * sizeof(Scalar), stream_));
        int nnz = wl_nnz_cols_count_[site];
        const std::vector<int>& h_nnz = h_WL_nnz_cols_[site];
        Scalar beta_one = Traits::one();
        for (int idx = 0; idx < nnz; idx++) {
            int c = h_nnz[idx];
            int wp = c / d, sp = c % d;
            int ws_out = wp * d + sp;
            ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
                rocblas_operation_none, rocblas_operation_none,
                cL, cR, cR,
                &one,
                U + ws_out * cL * cR, cL,
                R_env + wp * cR, cR * D,
                &beta_one,
                d_result + sp * cL, cL * d));
        }
    } else if (cL >= 16 && cR >= 16 && d <= 2) {
        for (int wp = 0; wp < D; wp++) {
            Scalar beta = (wp == 0) ? Traits::zero() : Traits::one();
            ROCBLAS_CHECK(Traits::gemm_strided_batched(rocblas_h_,
                rocblas_operation_none, rocblas_operation_none,
                cL, cR, cR,
                &one,
                U + (size_t)wp * d * cL * cR, cL, (rocblas_stride)(cL * cR),
                R_env + wp * cR, cR * D, (rocblas_stride)0,
                &beta,
                d_result, cL * d, (rocblas_stride)cL,
                d));
        }
    } else {
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

    if (graph_capture_miss) {
        hipGraph_t graph;
        HIP_CHECK(hipStreamEndCapture(stream_, &graph));
        hipGraphExec_t exec;
        HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        HIP_CHECK(hipGraphDestroy(graph));
        apply_heff_graph_cache_[graph_key(site, cL, cR)] = exec;
        HIP_CHECK(hipGraphLaunch(exec, stream_));
    }
}

// ============================================================================
// Left environment update
// ============================================================================

template<typename Scalar>
void DMRGGPUOpt<Scalar>::update_left_env(int site) {
    // Runs on stream_env_ (side stream) using env-dedicated scratch buffers so
    // it can overlap with the absorb(S*Vh) GEMM on stream_. The caller is
    // responsible for making stream_env_ wait on event_canon_ready_ so that
    // MPS[site] = U has been written before this runs.
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
        std::vector<Scalar*> h_A(D * d), h_B(D * d), h_C(D * d);
        for (int w = 0; w < D; w++)
            for (int s = 0; s < d; s++) {
                int ws = w * d + s;
                h_A[ws] = L_env + w * chi_in;
                h_B[ws] = A + s * chi_in;
                h_C[ws] = V + ws * chi_in * chi_out;
            }
        HIP_CHECK(hipMemcpyAsync(d_batch_A_env_, h_A.data(), D*d*sizeof(Scalar*), hipMemcpyHostToDevice, stream_env_));
        HIP_CHECK(hipMemcpyAsync(d_batch_B_env_, h_B.data(), D*d*sizeof(Scalar*), hipMemcpyHostToDevice, stream_env_));
        HIP_CHECK(hipMemcpyAsync(d_batch_C_env_, h_C.data(), D*d*sizeof(Scalar*), hipMemcpyHostToDevice, stream_env_));
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

    // Step 3: L_new_w'[b,b'] = sum_{a',s'} U[a',ws',b] * conj(A[a',s',b'])
    // Batched over w' only when D<=2 and chi large (D>=3 causes cache contention)
    if (chi_in >= 16 && chi_out >= 16 && D <= 2) {
        for (int sp = 0; sp < d; sp++) {
            Scalar beta = (sp == 0) ? Traits::zero() : Traits::one();
            ROCBLAS_CHECK(Traits::gemm_strided_batched(rocblas_h_env_,
                Traits::op_h, rocblas_operation_none,
                chi_out, chi_out, chi_in,
                &one,
                U + (size_t)sp * chi_in * chi_out, chi_in,
                (rocblas_stride)((size_t)d * chi_in * chi_out),
                A + sp * chi_in, chi_in * d, (rocblas_stride)0,
                &beta,
                L_new, chi_out * D, (rocblas_stride)chi_out,
                D));
        }
    } else {
        for (int wp = 0; wp < D; wp++) {
            for (int sp = 0; sp < d; sp++) {
                Scalar beta = (sp == 0) ? Traits::zero() : Traits::one();
                int ws_out = wp * d + sp;
                ROCBLAS_CHECK(Traits::gemm(rocblas_h_env_,
                    Traits::op_h, rocblas_operation_none,
                    chi_out, chi_out, chi_in,
                    &one,
                    U + ws_out * chi_in * chi_out, chi_in,
                    A + sp * chi_in, chi_in * d,
                    &beta,
                    L_new + wp * chi_out, chi_out * D));
            }
        }
    }

    // For complex: L_new = conj(U^H * A) = U^T * conj(A), the correct bra contraction
    if constexpr (Traits::is_complex) {
        conjugate_inplace(L_new, chi_out * D * chi_out, stream_env_);
    }
}

// ============================================================================
// Right environment update
// ============================================================================

template<typename Scalar>
void DMRGGPUOpt<Scalar>::update_right_env(int site) {
    // Runs on stream_env_ (see update_left_env).
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
    {
        std::vector<Scalar*> h_A(D * d), h_B(D * d), h_C(D * d);
        for (int wp = 0; wp < D; wp++)
            for (int s = 0; s < d; s++) {
                int ws = wp * d + s;
                h_A[ws] = A + s * chi_out;
                h_B[ws] = R_env + wp * chi_in;
                h_C[ws] = V + ws * chi_out * chi_in;
            }
        HIP_CHECK(hipMemcpyAsync(d_batch_A_env_, h_A.data(), D*d*sizeof(Scalar*), hipMemcpyHostToDevice, stream_env_));
        HIP_CHECK(hipMemcpyAsync(d_batch_B_env_, h_B.data(), D*d*sizeof(Scalar*), hipMemcpyHostToDevice, stream_env_));
        HIP_CHECK(hipMemcpyAsync(d_batch_C_env_, h_C.data(), D*d*sizeof(Scalar*), hipMemcpyHostToDevice, stream_env_));
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

    // Step 3: R_new_w[a,a'] = sum_s' U_ws'[a,b'] * A_s'^H[b',a']
    // Batched over w only when D<=2 and chi large (D>=3 causes cache contention)
    if (chi_in >= 16 && chi_out >= 16 && D <= 2) {
        for (int sp = 0; sp < d; sp++) {
            Scalar beta = (sp == 0) ? Traits::zero() : Traits::one();
            ROCBLAS_CHECK(Traits::gemm_strided_batched(rocblas_h_env_,
                rocblas_operation_none, Traits::op_h,
                chi_out, chi_out, chi_in,
                &one,
                U + (size_t)sp * chi_out * chi_in, chi_out,
                (rocblas_stride)((size_t)d * chi_out * chi_in),
                A + sp * chi_out, chi_out * d, (rocblas_stride)0,
                &beta,
                R_new, chi_out * D, (rocblas_stride)chi_out,
                D));
        }
    } else {
        for (int w = 0; w < D; w++) {
            for (int sp = 0; sp < d; sp++) {
                Scalar beta = (sp == 0) ? Traits::zero() : Traits::one();
                int ws_out = w * d + sp;
                ROCBLAS_CHECK(Traits::gemm(rocblas_h_env_,
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
}

// ============================================================================
// Environment building
// ============================================================================

template<typename Scalar>
void DMRGGPUOpt<Scalar>::build_initial_environments() {
    // L[0] = trivial left boundary: (1, D_mpo, 1), L[0][0,0,0] = 1
    {
        std::vector<Scalar> h_L(D_mpo_, Traits::zero());
        h_L[0] = Traits::one();
        HIP_CHECK(hipMemcpy(d_L_envs_[0], h_L.data(),
                            D_mpo_ * sizeof(Scalar), hipMemcpyHostToDevice));
    }

    // R[L] = trivial right boundary: (1, D_mpo, 1), R[L][0,D_act-1,0] = 1.
    // With D_PAD, D_mpo_ > D_mpo_actual_; the identity slot must be at
    // the unpadded index D_mpo_actual_ - 1 because the padded W rows past
    // that index are zero.
    {
        std::vector<Scalar> h_R(D_mpo_, Traits::zero());
        h_R[D_mpo_actual_ - 1] = Traits::one();
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
// Theta formation and Lanczos (fallback)
// ============================================================================

template<typename Scalar>
void DMRGGPUOpt<Scalar>::form_theta(int site, Scalar* d_theta) {
    int size = chi_L(site) * d_ * chi_R(site);
    HIP_CHECK(hipMemcpy(d_theta, d_mps_tensors_[site],
                        size * sizeof(Scalar), hipMemcpyDeviceToDevice));
}

template<typename Scalar>
double DMRGGPUOpt<Scalar>::lanczos_eigensolver(int site, Scalar* d_theta) {
    int n = chi_L(site) * d_ * chi_R(site);
    int max_iter = std::min(max_lanczos_iter_, n);
    double tol_lanczos = 1e-12;
    double tol_eig_conv = 1e-12;

    Scalar* d_lanczos_v = d_lanczos_v_;

    std::vector<double> h_alpha(max_iter);
    std::vector<double> h_beta(max_iter);

    // FUSE_LANCZOS: small device scratches to stage host scalars
    // for the fused kernels (which expect device-pointer operands).
    Scalar*   d_neg_alpha_scr = nullptr;
    Scalar*   d_neg_beta_scr  = nullptr;
    RealType* d_inv_beta_scr  = nullptr;
    if (opts_.fuse_lanczos) {
        HIP_CHECK(hipMalloc(&d_neg_alpha_scr, sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&d_neg_beta_scr,  sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&d_inv_beta_scr,  sizeof(RealType)));
    }

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
        apply_heff(site, d_vi, d_heff_result_);

        // alpha_i = <v_i|w>
        Scalar alpha_result;
        ROCBLAS_CHECK(Traits::dot(rocblas_h_, n, d_vi, 1, d_heff_result_, 1, &alpha_result));
        double alpha_i = Traits::real_part(alpha_result);
        h_alpha[iter] = alpha_i;

        // w = w - alpha_i * v_i  [and - beta_{i-1} * v_{i-1}]
        Scalar neg_alpha = Traits::make_scalar(-alpha_i);
        if (opts_.fuse_lanczos) {
            // Fused: w += (-α)·v_i + (-β_{i-1})·v_{i-1} in one kernel pass.
            // Stage host scalars to tiny device scratches.
            HIP_CHECK(hipMemcpyAsync(d_neg_alpha_scr, &neg_alpha, sizeof(Scalar),
                                     hipMemcpyHostToDevice, stream_));
            const Scalar* v_im1 = nullptr;
            const Scalar* nb_im1 = d_neg_alpha_scr;  // unused when has_prev==0
            if (iter > 0) {
                Scalar neg_beta = Traits::make_scalar(-h_beta[iter - 1]);
                HIP_CHECK(hipMemcpyAsync(d_neg_beta_scr, &neg_beta, sizeof(Scalar),
                                         hipMemcpyHostToDevice, stream_));
                v_im1 = d_lanczos_v + (size_t)(iter - 1) * n;
                nb_im1 = d_neg_beta_scr;
            }
            int block = 256;
            int grid = (n + block - 1) / block;
            hipLaunchKernelGGL((lanczos_fused_sub_kernel<Scalar>),
                               dim3(grid), dim3(block), 0, stream_,
                               d_heff_result_, d_vi, v_im1,
                               d_neg_alpha_scr, nb_im1,
                               n, iter > 0 ? 1 : 0);
        } else {
            ROCBLAS_CHECK(Traits::axpy(rocblas_h_, n, &neg_alpha, d_vi, 1, d_heff_result_, 1));

            // w = w - beta_{i-1} * v_{i-1}
            if (iter > 0) {
                Scalar neg_beta = Traits::make_scalar(-h_beta[iter - 1]);
                Scalar* d_vim1 = d_lanczos_v + (size_t)(iter - 1) * n;
                ROCBLAS_CHECK(Traits::axpy(rocblas_h_, n, &neg_beta, d_vim1, 1, d_heff_result_, 1));
            }
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

        // Eigenvalue convergence check (every 3 iterations after iter >= 4).
        // Tridiagonal eigensolve runs on device via rocsolver_dsteqr — H2D
        // the small (≤max_iter doubles) host-resident α/β, solve, D2H one
        // double for the energy and one rocblas_int for info. Replaces the
        // host LAPACK dstev_ that introduced a per-3-iter PCIe stall and
        // dependency on host BLAS.
        if (iter >= 4 && iter % 3 == 0) {
            int ncheck = iter + 1;
            std::vector<double> h_E_chk(ncheck);
            for (int i = 0; i < ncheck - 1; i++) h_E_chk[i] = h_beta[i];
            h_E_chk[ncheck - 1] = 0.0;

            HIP_CHECK(hipMemcpyAsync(d_steqr_D_, h_alpha.data(),
                                      ncheck * sizeof(double),
                                      hipMemcpyHostToDevice, stream_));
            HIP_CHECK(hipMemcpyAsync(d_steqr_E_, h_E_chk.data(),
                                      ncheck * sizeof(double),
                                      hipMemcpyHostToDevice, stream_));
            rocsolver_dsteqr(rocblas_h_, rocblas_evect_none, ncheck,
                             d_steqr_D_, d_steqr_E_, nullptr, ncheck,
                             d_steqr_info_);
            rocblas_int h_info_chk;
            double cur_energy;
            HIP_CHECK(hipMemcpyAsync(&h_info_chk, d_steqr_info_, sizeof(rocblas_int),
                                      hipMemcpyDeviceToHost, stream_));
            HIP_CHECK(hipMemcpyAsync(&cur_energy, d_steqr_D_, sizeof(double),
                                      hipMemcpyDeviceToHost, stream_));
            HIP_CHECK(hipStreamSynchronize(stream_));
            if (h_info_chk == 0) {
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
            if (opts_.fuse_lanczos) {
                // Fused normalize+copy: v_{i+1}[k] = w[k] * (1/β_i)
                RealType inv_beta_rt = (RealType)scale;
                HIP_CHECK(hipMemcpyAsync(d_inv_beta_scr, &inv_beta_rt, sizeof(RealType),
                                         hipMemcpyHostToDevice, stream_));
                int block = 256;
                int grid = (n + block - 1) / block;
                hipLaunchKernelGGL((lanczos_fused_norm_copy_kernel<Scalar, RealType>),
                                   dim3(grid), dim3(block), 0, stream_,
                                   d_vip1, d_heff_result_, d_inv_beta_scr, n);
            } else {
                HIP_CHECK(hipMemcpy(d_vip1, d_heff_result_, n * sizeof(Scalar), hipMemcpyDeviceToDevice));
                ROCBLAS_CHECK(Traits::scal_real(rocblas_h_, n, &scale, d_vip1, 1));
            }
        }
    }

    // Free FUSE_LANCZOS scratches
    if (opts_.fuse_lanczos) {
        if (d_neg_alpha_scr) hipFree(d_neg_alpha_scr);
        if (d_neg_beta_scr)  hipFree(d_neg_beta_scr);
        if (d_inv_beta_scr)  hipFree(d_inv_beta_scr);
    }

    int niter = iter;
    prof_davidson_iters += niter;
    prof_heff_calls += niter;

    // Solve tridiagonal eigenvalue problem on device via rocsolver_dsteqr.
    // H2D the small (niter doubles) host α/β, run on-device, D2H one int +
    // one double for energy. Eigenvectors stay on device at d_steqr_C_;
    // the Ritz coefficient (first column) is moved straight into
    // d_ritz_coeffs_ with promote_double_to_complex when Scalar is complex.
    std::vector<double> h_E(niter);
    for (int i = 0; i < niter - 1; i++) h_E[i] = h_beta[i];
    if (niter > 0) h_E[niter - 1] = 0.0;

    HIP_CHECK(hipMemcpyAsync(d_steqr_D_, h_alpha.data(),
                              niter * sizeof(double),
                              hipMemcpyHostToDevice, stream_));
    HIP_CHECK(hipMemcpyAsync(d_steqr_E_, h_E.data(),
                              niter * sizeof(double),
                              hipMemcpyHostToDevice, stream_));
    rocsolver_dsteqr(rocblas_h_, rocblas_evect_tridiagonal, niter,
                     d_steqr_D_, d_steqr_E_, d_steqr_C_, niter,
                     d_steqr_info_);
    rocblas_int h_lapack_info;
    double energy;
    HIP_CHECK(hipMemcpyAsync(&h_lapack_info, d_steqr_info_, sizeof(rocblas_int),
                              hipMemcpyDeviceToHost, stream_));
    HIP_CHECK(hipMemcpyAsync(&energy, d_steqr_D_, sizeof(double),
                              hipMemcpyDeviceToHost, stream_));
    HIP_CHECK(hipStreamSynchronize(stream_));
    if (h_lapack_info != 0) {
        throw std::runtime_error("rocsolver_dsteqr failed with info = " + std::to_string(h_lapack_info));
    }

    // Ritz coefficients = first column of d_steqr_C. Promote double → complex
    // on device when Scalar is hipDoubleComplex (real Lanczos α/β produce
    // real eigenvectors, but d_ritz_coeffs is Scalar-typed).
    if constexpr (std::is_same_v<Scalar, double>) {
        HIP_CHECK(hipMemcpyAsync(d_ritz_coeffs_, d_steqr_C_,
                                  niter * sizeof(double),
                                  hipMemcpyDeviceToDevice, stream_));
    } else {
        int blk = (niter + 63) / 64;
        hipLaunchKernelGGL(promote_double_to_complex, dim3(blk), dim3(64), 0, stream_,
                           d_steqr_C_, (hipDoubleComplex*)d_ritz_coeffs_, niter);
    }

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
// SVD bond splitting — on-device default path (rocsolver_gesvd_auto + on-device
// truncation/scale/absorb), matching dmrg-gpu's svd_and_update_mps. Replaces
// the prior host-LAPACK svd_fallback that did per-sweep D2H of theta + host
// LAPACK gesvd + H2D of U/S/Vh — exactly the kind of per-sweep PCIe traffic
// that makes -opt slower than its sibling -gpu.
// ============================================================================

template<typename Scalar>
void DMRGGPUOpt<Scalar>::svd_fallback(int site, Scalar* d_theta, char direction) {
    int cL = chi_L(site);
    int cR = chi_R(site);

    int m, n_svd;
    if (direction == 'R') { m = cL * d_; n_svd = cR; }
    else                  { m = cL;      n_svd = d_ * cR; }
    int full_k = std::min(m, n_svd);
    int k = std::min(full_k, chi_max_user_);

    // Choose between full SVD (rocsolver_gesvd_auto), randomized SVD
    // (Halko-Martinsson-Tropp), and the use_cpu_svd_ opt-in CPU LAPACK path.
    // RSVD is profitable when full_k > k + p (oversample beats waste) and
    // m > 2k. vh_lda + svd_k track Vh's leading dim and the number of
    // singular pairs produced — full SVD gives full_k of each, RSVD gives
    // b_k = min(r_use, n_svd).
    int vh_lda = full_k;
    int svd_k  = full_k;
    bool used_rsvd = use_rsvd_
                  && !use_cpu_svd_
                  && full_k > k + RSVD_OVERSAMPLE_
                  && m > 2 * k;

    if (use_cpu_svd_) {
        // CPU LAPACK SVD — opt-in only (use_cpu_svd_ flag, off by default).
        // Round-5 A7 fix: previously the setter set_cpu_svd existed for J2
        // API parity but the flag was never read. Now wired: D2H theta,
        // host gesvd, H2D U/S/Vh into the on-device buffers so the existing
        // truncate+scale+absorb post-SVD logic works uniformly.
        HIP_CHECK(hipMemcpyAsync(h_svd_A_.data(), d_theta,
                                  m * n_svd * sizeof(Scalar),
                                  hipMemcpyDeviceToHost, stream_));
        HIP_CHECK(hipStreamSynchronize(stream_));
        int lwork = (int)h_svd_work_.size();
        int info;
        const char jobu = 'S', jobvt = 'S';
        Traits::lapack_gesvd(&jobu, &jobvt, &m, &n_svd, h_svd_A_.data(), &m,
                h_svd_S_.data(), h_svd_U_.data(), &m, h_svd_Vh_.data(), &full_k,
                h_svd_work_.data(), &lwork,
                h_svd_rwork_.empty() ? nullptr : h_svd_rwork_.data(), &info);
        if (info != 0) {
            throw std::runtime_error("svd_fallback (use_cpu_svd_): LAPACK gesvd info=" + std::to_string(info));
        }
        // Upload U / S / Vh back to the on-device buffers used by the
        // existing post-SVD truncate + scale + absorb logic.
        HIP_CHECK(hipMemcpyAsync(d_svd_S_, h_svd_S_.data(),
                                  full_k * sizeof(RealType),
                                  hipMemcpyHostToDevice, stream_));
        HIP_CHECK(hipMemcpyAsync(d_svd_U_, h_svd_U_.data(),
                                  (size_t)m * full_k * sizeof(Scalar),
                                  hipMemcpyHostToDevice, stream_));
        HIP_CHECK(hipMemcpyAsync(d_svd_Vh_, h_svd_Vh_.data(),
                                  (size_t)full_k * n_svd * sizeof(Scalar),
                                  hipMemcpyHostToDevice, stream_));
    } else if (!used_rsvd) {
        // GPU SVD via size-gated dispatcher (Jacobi for double, bidiagonal for
        // small complex). Keeps U/S/Vh device-resident throughout.
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
        // --- Randomized SVD (Halko–Martinsson–Tropp) ---
        // Approximates the leading k singular triplets of A ~ (m × n_svd)
        // with r = k + p oversampling. Mirrors dmrg-gpu's RSVD path
        // (round-5 J2 backport). Inner SVD of B stays on device.
        int r_use = std::min({k + RSVD_OVERSAMPLE_, full_k, rsvd_r_max_});

        // Ω ∈ C^{n_svd × r_use}, fresh Gaussian-ish per call.
        {
            std::vector<Scalar> h_omega((size_t)n_svd * r_use);
            for (size_t i = 0; i < h_omega.size(); i++) {
                h_omega[i] = Traits::random_val();
            }
            HIP_CHECK(hipMemcpyAsync(d_rsvd_omega_, h_omega.data(),
                h_omega.size() * sizeof(Scalar), hipMemcpyHostToDevice, stream_));
        }

        Scalar one = Traits::one(), zero_val = Traits::zero();

        // Y = A · Ω  —  (m × r_use)
        ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
            rocblas_operation_none, rocblas_operation_none,
            m, r_use, n_svd, &one,
            d_theta, m,
            d_rsvd_omega_, n_svd,
            &zero_val,
            d_rsvd_Y_, m));

        // QR(Y): geqrf overwrites Y with R+householders, orgqr reifies Q in place.
        ROCBLAS_CHECK(Traits::rocsolver_geqrf(rocblas_h_, m, r_use,
            d_rsvd_Y_, m, d_rsvd_tau_));
        ROCBLAS_CHECK(Traits::rocsolver_orgqr(rocblas_h_, m, r_use, r_use,
            d_rsvd_Y_, m, d_rsvd_tau_));

        // B = Q^H · A  —  (r_use × n_svd)
        ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
            Traits::op_h, rocblas_operation_none,
            r_use, n_svd, m, &one,
            d_rsvd_Y_, m,
            d_theta, m,
            &zero_val,
            d_rsvd_B_, r_use));

        // SVD(B) on GPU — small matrix, on-device inner SVD per the
        // round-4 reconciliation rule (no host-LAPACK roundtrip).
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

        // U = Q · U_small  —  (m × b_k), written into d_svd_U_.
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

    // Truncation: find new_k on device, copy 1 int back. svd_k caps the
    // search so RSVD's b_k limit is respected.
    int k_target = std::min(k, svd_k);
    int new_k;
    hipLaunchKernelGGL(svd_truncate_kernel<RealType>, dim3(1), dim3(1), 0, stream_,
                       d_svd_S_, k_target, 1e-14, d_svd_info_);
    HIP_CHECK(hipMemcpyAsync(&new_k, d_svd_info_, sizeof(int),
                              hipMemcpyDeviceToHost, stream_));
    HIP_CHECK(hipStreamSynchronize(stream_));

    int threads = 256;

    if (direction == 'R') {
        int new_chi_R = new_k;

        // MPS[site] = U[:, :new_k] — D2D when new_k == svd_k, else extract_cols.
        allocate_mps_tensor(site, cL, new_chi_R);
        if (new_k == svd_k) {
            HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site], d_svd_U_,
                                     (size_t)m * new_k * sizeof(Scalar),
                                     hipMemcpyDeviceToDevice, stream_));
        } else {
            int total = m * new_k;
            hipLaunchKernelGGL(extract_cols_kernel<Scalar>,
                               dim3((total + threads - 1) / threads), dim3(threads), 0, stream_,
                               d_svd_U_, m, d_mps_tensors_[site], m, m, new_k);
        }
        // MPS[site] = U is now queued on stream_. Signal the env stream so
        // update_left_env(site) can start concurrently with the absorb below
        // (round-6 J2 port of dmrg-gpu's dual-stream pipeline).
        HIP_CHECK(hipEventRecord(event_canon_ready_, stream_));

        // S*Vh → d_svd_work_ (scale rows of Vh by S on device).
        // Vh's leading dim is vh_lda (= full_k for full SVD, = b_k for RSVD).
        {
            int total = new_k * n_svd;
            hipLaunchKernelGGL((scale_rows_by_diag_kernel<Scalar, RealType>),
                               dim3((total + threads - 1) / threads), dim3(threads), 0, stream_,
                               d_svd_S_, d_svd_Vh_, vh_lda,
                               d_svd_work_, new_k, new_k, n_svd);
        }

        // Absorb S*Vh into A[site+1] via on-device gemm.
        if (site + 1 < L_) {
            int next_cR = chi_R(site + 1);
            Scalar one_v = Traits::one(), zero_v = Traits::zero();
            ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
                rocblas_operation_none, rocblas_operation_none,
                new_k, d_ * next_cR, cR, &one_v,
                d_svd_work_, new_k,
                d_mps_tensors_[site + 1], cR, &zero_v,
                d_T1_, new_k));
            allocate_mps_tensor(site + 1, new_chi_R, next_cR);
            HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site + 1], d_T1_,
                                     (size_t)new_k * d_ * next_cR * sizeof(Scalar),
                                     hipMemcpyDeviceToDevice, stream_));
        }
        bond_dims_[site + 1] = new_chi_R;

    } else {  // direction == 'L'
        int new_chi_L = new_k;

        // MPS[site] = Vh[:new_k, :] — extract first new_k rows from
        // (vh_lda × n_svd) col-major Vh on device. vh_lda == full_k for
        // full SVD, == b_k for RSVD.
        allocate_mps_tensor(site, new_chi_L, cR);
        if (new_chi_L == svd_k && vh_lda == new_chi_L) {
            HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site], d_svd_Vh_,
                                     (size_t)vh_lda * n_svd * sizeof(Scalar),
                                     hipMemcpyDeviceToDevice, stream_));
        } else {
            int total = new_chi_L * n_svd;
            hipLaunchKernelGGL(extract_cols_kernel<Scalar>,
                               dim3((total + threads - 1) / threads), dim3(threads), 0, stream_,
                               d_svd_Vh_, vh_lda, d_mps_tensors_[site], new_chi_L,
                               new_chi_L, n_svd);
        }
        // MPS[site] = Vh is now queued on stream_. Signal the env stream so
        // update_right_env(site) can start concurrently with the absorb below
        // (round-6 J2 port of dmrg-gpu's dual-stream pipeline).
        HIP_CHECK(hipEventRecord(event_canon_ready_, stream_));

        // U*S → d_svd_work_ (scale columns of U by S on device).
        {
            int total = m * new_k;
            hipLaunchKernelGGL((scale_cols_by_diag_kernel<Scalar, RealType>),
                               dim3((total + threads - 1) / threads), dim3(threads), 0, stream_,
                               d_svd_S_, d_svd_U_, m,
                               d_svd_work_, m, m, new_k);
        }

        // Absorb U*S into A[site-1] via on-device gemm.
        if (site > 0) {
            int prev_cL = chi_L(site - 1);
            Scalar one_v = Traits::one(), zero_v = Traits::zero();
            ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
                rocblas_operation_none, rocblas_operation_none,
                prev_cL * d_, new_k, m, &one_v,
                d_mps_tensors_[site - 1], prev_cL * d_,
                d_svd_work_, m, &zero_v,
                d_T1_, prev_cL * d_));
            allocate_mps_tensor(site - 1, prev_cL, new_chi_L);
            HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site - 1], d_T1_,
                                     (size_t)prev_cL * d_ * new_k * sizeof(Scalar),
                                     hipMemcpyDeviceToDevice, stream_));
        }
        bond_dims_[site] = new_chi_L;
    }
}

// ============================================================================
// Block-Davidson Eigensolver
// ============================================================================

template<typename Scalar>
double DMRGGPUOpt<Scalar>::block_davidson_eigensolver(int site, Scalar* d_theta) {
    int dim = chi_L(site) * d_ * chi_R(site);
    int b = std::min(davidson_b_, dim);
    int max_sub = std::min(davidson_max_sub_, dim);
    int max_iter = 30;
    double tol_dav = 1e-10;

    Scalar one = Traits::one(), zero_val = Traits::zero();
    Scalar neg_one = Traits::neg(Traits::one());

    // For tiny systems, use Lanczos fallback
    if (dim <= 2 * b) {
        return lanczos_eigensolver(site, d_theta);
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

        // Normalize. nrm2 in host-pointer mode (default) syncs internally
        // to return the result to host memory; explicit hipStreamSynchronize
        // before is redundant.
        RealType nrm_v;
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
        apply_heff(site, V + (size_t)j * dim, AV + (size_t)j * dim);
    }

    double best_energy = 1e30;
    double energy_prev = 1e30;
    int k = b;  // current subspace size

    for (int iteration = 0; iteration < max_iter; iteration++) {
        // Rayleigh-Ritz: H_proj = V^H @ AV  -> (k, k), written into
        // d_dav_work2_ which becomes the eigenvector buffer in-place after
        // rocsolver_syevd. (Round-7 C2: replaced the host-LAPACK roundtrip
        // path that did a per-iteration D2H + symmetrize + lapack_syev +
        // H2D — that violated the "no host roundtrips per sweep" rule on
        // the default Davidson code path.)
        ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
            Traits::op_h, rocblas_operation_none,
            k, k, dim, &one, V, dim, AV, dim, &zero_val, d_dav_work2_, k));

        // On-device Hermitian eigendecomposition. We pass uplo=upper so
        // syevd only reads the upper triangle (the projected matrix is
        // exactly Hermitian up to roundoff; the upper-triangle convention
        // is sufficient and avoids the explicit symmetrize kernel).
        ROCBLAS_CHECK(Traits::rocsolver_syevd(rocblas_h_,
                rocblas_evect_original, rocblas_fill_upper,
                k, d_dav_work2_, k, d_dav_eigvals_, d_dav_E_, d_dav_info_));

        // Single-double D2H of the lowest eigenvalue for energy /
        // convergence check. Required for control flow; everything else
        // stays on device.
        double energy;
        HIP_CHECK(hipMemcpyAsync(&energy, d_dav_eigvals_, sizeof(double),
                                  hipMemcpyDeviceToHost, stream_));
        rocblas_int h_dav_info;
        HIP_CHECK(hipMemcpyAsync(&h_dav_info, d_dav_info_, sizeof(rocblas_int),
                                  hipMemcpyDeviceToHost, stream_));
        HIP_CHECK(hipStreamSynchronize(stream_));
        if (h_dav_info != 0) {
            // Eigendecomp failed — fall back to Lanczos.
            return lanczos_eigensolver(site, d_theta);
        }

        if (energy < best_energy) {
            best_energy = energy;
        }
        // d_dav_work2_ now holds the (k×k) eigenvectors on device. We need
        // them alive both here and in the restart path below, so we do NOT
        // overwrite d_dav_work2_ — the overlap matrix later uses an offset
        // into d_dav_work_ instead.

        // D2H the lowest b eigenvalues for the residual computation below
        // (small copy; n_new_max ≤ b ≤ 4).
        int b_use = std::min(b, k);
        HIP_CHECK(hipMemcpyAsync(h_dav_eigvals_.data(), d_dav_eigvals_,
                                  b_use * sizeof(RealType),
                                  hipMemcpyDeviceToHost, stream_));
        HIP_CHECK(hipStreamSynchronize(stream_));

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
        for (int i = 0; i < b_use; i++) {
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

        // Orthogonalize new vectors against V. W = residuals at d_dav_work_;
        // overlap matrix lives at an offset PAST the residuals so it does
        // not overwrite d_dav_work2_ (which still holds the eigvecs we
        // need for the restart path below).
        Scalar* W = d_dav_work_;
        Scalar* overlap = d_dav_work_ + (size_t)n_new * dim;

        // overlap = V^H @ W -> (k, n_new)
        ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
            Traits::op_h, rocblas_operation_none,
            k, n_new, dim, &one, V, dim, W, dim, &zero_val, overlap, k));

        // W -= V @ overlap
        ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
            rocblas_operation_none, rocblas_operation_none,
            dim, n_new, k, &neg_one, V, dim, overlap, k, &one, W, dim));

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

            // X_keep = V @ eigvecs[:, :keep] -> (dim, keep). Eigvecs are
            // already in d_dav_work2_ from rocsolver_syevd above; we
            // preserved them by routing the overlap through an offset of
            // d_dav_work_ instead of overwriting d_dav_work2_.
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
                ROCBLAS_CHECK(Traits::nrm2(rocblas_h_, dim, V + (size_t)i * dim, 1, &vi_norm));
                RealType inv_vi = 1.0 / vi_norm;
                ROCBLAS_CHECK(Traits::scal_real(rocblas_h_, dim, &inv_vi, V + (size_t)i * dim, 1));
            }

            // Recompute AV for kept vectors
            for (int j = 0; j < keep; j++) {
                apply_heff(site, V + (size_t)j * dim, AV + (size_t)j * dim);
            }

            k = keep;
            continue;
        }

        // Expand: append new vectors and compute their H-images
        for (int j = 0; j < n_good; j++) {
            HIP_CHECK(hipMemcpyAsync(V + (size_t)(k + j) * dim, W + (size_t)j * dim,
                                      dim * sizeof(Scalar), hipMemcpyDeviceToDevice, stream_));
            apply_heff(site, V + (size_t)(k + j) * dim, AV + (size_t)(k + j) * dim);
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
// Site optimization
// ============================================================================

template<typename Scalar>
double DMRGGPUOpt<Scalar>::optimize_site(int site, char direction) {
    form_theta(site, d_theta_);

    // Profiling syncs gated by opts_.profile (round-6 fix). Default G1 path
    // (profile=false) skips these syncs and lets svd_fallback queue behind
    // the eigensolver work — significant speedup at the cost of inaccurate
    // per-phase timings, which we don't measure in production runs anyway.
    auto t0 = std::chrono::high_resolution_clock::now();
    double energy = use_davidson_ ? block_davidson_eigensolver(site, d_theta_)
                                  : lanczos_eigensolver(site, d_theta_);
    if (opts_.profile) {
        HIP_CHECK(hipStreamSynchronize(stream_));
        auto t1 = std::chrono::high_resolution_clock::now();
        prof_davidson_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        t0 = t1;
    }

    svd_fallback(site, d_theta_, direction);
    if (opts_.profile) {
        HIP_CHECK(hipStreamSynchronize(stream_));
        auto t1 = std::chrono::high_resolution_clock::now();
        prof_svd_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    prof_site_count++;
    return energy;
}

// ============================================================================
// Sweep methods
// ============================================================================

template<typename Scalar>
double DMRGGPUOpt<Scalar>::sweep_left_to_right() {
    // Dual-stream pipeline (round-6 J2 port from dmrg-gpu):
    //   stream_     : Lanczos/Davidson + SVD + absorb(S*Vh) into MPS[site+1]
    //   stream_env_ : update_left_env(site) → L[site+1]
    //
    // Each iteration: SVD on stream_ records event_canon_ready_ as soon as
    // MPS[site] = U is written. The env stream picks this up and runs
    // update_left_env concurrently with the absorb GEMM on stream_. Before
    // the next site's eigensolver (which reads L[site+1]) we make stream_
    // wait on event_env_done_.
    double energy = 0.0;

    for (int site = 0; site < L_ - 1; site++) {
        if (env_update_pending_) {
            HIP_CHECK(hipStreamWaitEvent(stream_, event_env_done_, 0));
            env_update_pending_ = false;
        }

        energy = optimize_site(site, 'R');

        // Env timing (gated by opts_.profile). When profiling is off, full
        // dual-stream overlap; when on, briefly serialize on stream_env_ so
        // the host-side timer is meaningful.
        auto te0 = std::chrono::high_resolution_clock::now();
        HIP_CHECK(hipStreamWaitEvent(stream_env_, event_canon_ready_, 0));
        update_left_env(site);
        HIP_CHECK(hipEventRecord(event_env_done_, stream_env_));
        env_update_pending_ = true;
        if (opts_.profile) {
            HIP_CHECK(hipStreamSynchronize(stream_env_));
            auto te1 = std::chrono::high_resolution_clock::now();
            prof_env_ms += std::chrono::duration<double, std::milli>(te1 - te0).count();
        }
    }
    if (env_update_pending_) {
        HIP_CHECK(hipStreamWaitEvent(stream_, event_env_done_, 0));
        env_update_pending_ = false;
    }
    // Optimize last site without SVD
    {
        int site = L_ - 1;
        form_theta(site, d_theta_);
        auto t0 = std::chrono::high_resolution_clock::now();
        energy = use_davidson_ ? block_davidson_eigensolver(site, d_theta_)
                               : lanczos_eigensolver(site, d_theta_);
        auto t1 = std::chrono::high_resolution_clock::now();
        prof_davidson_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        int sz = chi_L(site) * d_ * chi_R(site);
        HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site], d_theta_, sz * sizeof(Scalar),
                                 hipMemcpyDeviceToDevice, stream_));
    }

    return energy;
}

template<typename Scalar>
double DMRGGPUOpt<Scalar>::sweep_right_to_left() {
    // Mirror of sweep_left_to_right: update_right_env(site) overlaps with the
    // absorb(U*S) into MPS[site-1] on stream_.
    double energy = 0.0;

    for (int site = L_ - 1; site >= 1; site--) {
        if (env_update_pending_) {
            HIP_CHECK(hipStreamWaitEvent(stream_, event_env_done_, 0));
            env_update_pending_ = false;
        }

        energy = optimize_site(site, 'L');

        auto te0 = std::chrono::high_resolution_clock::now();
        HIP_CHECK(hipStreamWaitEvent(stream_env_, event_canon_ready_, 0));
        update_right_env(site);
        HIP_CHECK(hipEventRecord(event_env_done_, stream_env_));
        env_update_pending_ = true;
        if (opts_.profile) {
            HIP_CHECK(hipStreamSynchronize(stream_env_));
            auto te1 = std::chrono::high_resolution_clock::now();
            prof_env_ms += std::chrono::duration<double, std::milli>(te1 - te0).count();
        }
    }
    if (env_update_pending_) {
        HIP_CHECK(hipStreamWaitEvent(stream_, event_env_done_, 0));
        env_update_pending_ = false;
    }
    // Optimize first site without SVD
    {
        int site = 0;
        form_theta(site, d_theta_);
        auto t0 = std::chrono::high_resolution_clock::now();
        energy = use_davidson_ ? block_davidson_eigensolver(site, d_theta_)
                               : lanczos_eigensolver(site, d_theta_);
        auto t1 = std::chrono::high_resolution_clock::now();
        prof_davidson_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        int sz = chi_L(site) * d_ * chi_R(site);
        HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site], d_theta_, sz * sizeof(Scalar),
                                 hipMemcpyDeviceToDevice, stream_));
    }

    return energy;
}

// ============================================================================
// Main algorithm
// ============================================================================

template<typename Scalar>
double DMRGGPUOpt<Scalar>::run(int n_sweeps) {
    const char* type_name = Traits::is_complex ? "complex128" : "float64";
    printf("=== GPU-Native Single-Site DMRG-OPT (Davidson + MFMA-16 pad + batched, %s) ===\n", type_name);
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
        prof_davidson_ms = prof_svd_ms = prof_env_ms = 0;
        prof_davidson_iters = prof_site_count = prof_heff_calls = 0;

        auto t_sweep = std::chrono::high_resolution_clock::now();

        double energy_LR = sweep_left_to_right();
        double energy_RL = sweep_right_to_left();

        auto t_sweep_end = std::chrono::high_resolution_clock::now();
        double sweep_time = std::chrono::duration<double>(t_sweep_end - t_sweep).count();

        energy_ = energy_RL;
        double dE = std::abs(energy_ - energy_prev);

        double other_ms = sweep_time*1000.0 - prof_davidson_ms - prof_svd_ms - prof_env_ms;
        printf("Sweep %3d: E = %.12f, dE = %.2e, time = %.3f s\n",
               sweep, energy_, dE, sweep_time);
        printf("  Profile: davidson=%.0fms (%d iters, %d heff) svd=%.0fms env=%.0fms other=%.0fms\n",
               prof_davidson_ms, prof_davidson_iters, prof_heff_calls,
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
    report_timers();

    return energy_;
}

// ============================================================================
// Phase timers
// ============================================================================

template<typename Scalar>
void DMRGGPUOpt<Scalar>::init_timers() {
    t_lanczos_.init("lanczos", opts_.profile);
    t_apply_heff_.init("apply_heff", opts_.profile);
    t_svd_.init("svd", opts_.profile);
    t_absorb_.init("absorb", opts_.profile);
    t_env_update_.init("env_update", opts_.profile);
}

template<typename Scalar>
void DMRGGPUOpt<Scalar>::report_timers() {
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
void DMRGGPUOpt<Scalar>::get_mps(std::vector<std::vector<Scalar>>& h_mps) const {
    h_mps.resize(L_);
    for (int i = 0; i < L_; i++) {
        int size = chi_L(i) * d_ * chi_R(i);
        h_mps[i].resize(size);
        HIP_CHECK(hipMemcpy(h_mps[i].data(), d_mps_tensors_[i],
                            size * sizeof(Scalar), hipMemcpyDeviceToHost));
    }
}

#endif // DMRG_GPU_OPT_IMPL_H
