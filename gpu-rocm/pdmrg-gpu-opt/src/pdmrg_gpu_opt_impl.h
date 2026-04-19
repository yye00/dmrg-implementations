#ifndef PDMRG_GPU_OPT_IMPL_H
#define PDMRG_GPU_OPT_IMPL_H

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
PDMRGGPUOpt<Scalar>::PDMRGGPUOpt(int L, int d, int chi_max, int D_mpo, int n_segments, double tol)
    : L_(L), d_(d), chi_max_(pad_mfma16(chi_max)), chi_max_user_(chi_max),
      D_mpo_(D_mpo), tol_(tol), energy_(0.0),
      n_segments_(n_segments) {

    opts_.load_from_env();
    opts_.print(stderr);
    init_timers();

    if (chi_max_ != chi_max_user_) {
        printf("[OPT] MFMA-16 padding: chi_max %d -> %d\n", chi_max_user_, chi_max_);
    }

    if (L < 2 * n_segments) {
        throw std::runtime_error("Need at least 2 sites per segment: L >= 2*n_segments");
    }

    // Bond dimensions (min-cut formula capped at chi_max)
    bond_dims_.resize(L + 1);
    bond_dims_[0] = 1;
    bond_dims_[L] = 1;
    for (int i = 1; i < L; i++) {
        double exact_dim = pow((double)d_, std::min(i, L - i));
        bond_dims_[i] = (exact_dim > chi_max_user_) ? chi_max_user_ : (int)exact_dim;
    }

    partition_chain();
    initialize_boundary_states();

    // Create streams and rocBLAS handles
    streams_.resize(n_segments_);
    handles_.resize(n_segments_);
    for (int k = 0; k < n_segments_; k++) {
        HIP_CHECK(hipStreamCreate(&streams_[k]));
        ROCBLAS_CHECK(rocblas_create_handle(&handles_[k]));
        ROCBLAS_CHECK(rocblas_set_stream(handles_[k], streams_[k]));
    }

    // Worker stream pool: n_workers covers max(d*d, D_mpo) independent GEMMs
    n_workers_ = std::max(d * d, D_mpo);
    worker_streams_.resize(n_segments_);
    worker_handles_.resize(n_segments_);
    for (int k = 0; k < n_segments_; k++) {
        worker_streams_[k].resize(n_workers_);
        worker_handles_[k].resize(n_workers_);
        for (int w = 0; w < n_workers_; w++) {
            HIP_CHECK(hipStreamCreate(&worker_streams_[k][w]));
            ROCBLAS_CHECK(rocblas_create_handle(&worker_handles_[k][w]));
            ROCBLAS_CHECK(rocblas_set_stream(worker_handles_[k][w], worker_streams_[k][w]));
        }
    }
    worker_done_events_.resize(n_segments_);
    step_done_events_.resize(n_segments_);
    for (int k = 0; k < n_segments_; k++) {
        worker_done_events_[k].resize(n_workers_);
        for (int w = 0; w < n_workers_; w++) {
            HIP_CHECK(hipEventCreateWithFlags(&worker_done_events_[k][w], hipEventDisableTiming));
        }
        HIP_CHECK(hipEventCreateWithFlags(&step_done_events_[k], hipEventDisableTiming));
    }

    int dd = d_ * d_;

    // MPS tensors
    d_mps_tensors_.resize(L, nullptr);
    for (int i = 0; i < L; i++) {
        allocate_mps_tensor(i, chi_L(i), chi_R(i));
    }

    // MPO tensors
    d_mpo_tensors_.resize(L, nullptr);
    d_W_left_.resize(L, nullptr);
    d_W_right_.resize(L, nullptr);
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

    // Workspace parameters
    theta_size_max_ = chi_max_ * dd * chi_max_;
    max_lanczos_iter_ = std::min(100, theta_size_max_);
    davidson_b_ = 4;
    davidson_max_sub_ = std::min(davidson_b_ * 8, theta_size_max_);
    use_cpu_svd_ = false;
    use_ns_split_ = true;  // default: use Newton-Schulz for bond split
    use_davidson_ = false;  // default: use Lanczos (device-pointer-mode, 2-3 syncs/bond)
    use_rsvd_ = false;
    lanczos_use_1site_ = false;
    use_batched_sweep_ = false;  // cross-segment batching: slower for n_segments=2 due to BLAS-1 serialization
    use_chebyshev_ = false;       // Chebyshev-filtered subspace iteration eigensolver
    rsvd_oversampling_ = 20;

    allocate_stream_workspaces();
}

// ============================================================================
// Per-stream workspace allocation
// ============================================================================

template<typename Scalar>
void PDMRGGPUOpt<Scalar>::allocate_stream_workspaces() {
    int dd = d_ * d_;
    int t_max = D_mpo_ * dd * chi_max_ * chi_max_;
    int batch_max = D_mpo_ * dd;
    int svd_max_m = chi_max_ * d_;
    int svd_max_n = d_ * chi_max_;
    int svd_max_k = std::min(svd_max_m, svd_max_n);
    int max_ns_m = chi_max_ * d_;  // largest dimension for Newton-Schulz
    int max_ns_n = chi_max_ * d_;  // may equal m for square theta

    workspaces_.resize(n_segments_);

    for (int k = 0; k < n_segments_; k++) {
        auto& ws = workspaces_[k];

        // Contraction intermediates
        HIP_CHECK(hipMalloc(&ws.d_T1, t_max * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&ws.d_T2, t_max * sizeof(Scalar)));

        // Theta and heff result
        HIP_CHECK(hipMalloc(&ws.d_theta, theta_size_max_ * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&ws.d_heff_result, theta_size_max_ * sizeof(Scalar)));

        // Batched GEMM pointer arrays
        HIP_CHECK(hipMalloc(&ws.d_batch_A, batch_max * sizeof(Scalar*)));
        HIP_CHECK(hipMalloc(&ws.d_batch_B, batch_max * sizeof(Scalar*)));
        HIP_CHECK(hipMalloc(&ws.d_batch_C, batch_max * sizeof(Scalar*)));
        HIP_CHECK(hipMalloc(&ws.d_heff_batch_A, batch_max * sizeof(Scalar*)));
        HIP_CHECK(hipMalloc(&ws.d_heff_batch_C, batch_max * sizeof(Scalar*)));
        ws.heff_cached_site = -1;

        // === Block-Davidson workspace ===
        HIP_CHECK(hipMalloc(&ws.d_dav_V, (size_t)theta_size_max_ * davidson_max_sub_ * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&ws.d_dav_AV, (size_t)theta_size_max_ * davidson_max_sub_ * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&ws.d_dav_work, (size_t)theta_size_max_ * davidson_b_ * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&ws.d_dav_work2, (size_t)theta_size_max_ * davidson_b_ * sizeof(Scalar)));
        ws.h_dav_H_proj.resize(davidson_max_sub_ * davidson_max_sub_);
        ws.h_dav_eigvals.resize(davidson_max_sub_);
        ws.h_dav_eigvecs.resize(davidson_max_sub_ * davidson_max_sub_);
        ws.h_dav_syev_work.resize(Traits::syev_rwork_size(davidson_max_sub_));
        ws.h_dav_V_copy.resize((size_t)theta_size_max_ * davidson_b_);

        // === Lanczos fallback workspace ===
        HIP_CHECK(hipMalloc(&ws.d_lanczos_v, (size_t)max_lanczos_iter_ * theta_size_max_ * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&ws.d_ritz_coeffs, max_lanczos_iter_ * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&ws.d_dot_result, sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&ws.d_nrm2_result, sizeof(RealType)));
        HIP_CHECK(hipMalloc(&ws.d_neg_alpha, sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&ws.d_neg_overlap, sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&ws.d_inv_nrm, sizeof(RealType)));
        HIP_CHECK(hipMalloc(&ws.d_alpha_dev, max_lanczos_iter_ * sizeof(RealType)));
        HIP_CHECK(hipMalloc(&ws.d_beta_dev, max_lanczos_iter_ * sizeof(RealType)));
        HIP_CHECK(hipMalloc(&ws.d_neg_beta_scalars, max_lanczos_iter_ * sizeof(Scalar)));
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

        // === Newton-Schulz workspace ===
        HIP_CHECK(hipMalloc(&ws.d_ns_U, (size_t)max_ns_m * max_ns_n * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&ws.d_ns_U_new, (size_t)max_ns_m * max_ns_n * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&ws.d_ns_gram, (size_t)max_ns_n * max_ns_n * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&ws.d_ns_P, (size_t)max_ns_n * max_ns_n * sizeof(Scalar)));

        // === SVD workspace ===
        HIP_CHECK(hipMalloc(&ws.d_svd_A, theta_size_max_ * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&ws.d_svd_U, (size_t)svd_max_m * svd_max_k * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&ws.d_svd_S, svd_max_k * sizeof(RealType)));
        HIP_CHECK(hipMalloc(&ws.d_svd_Vh, (size_t)svd_max_k * svd_max_n * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&ws.d_svd_E, svd_max_k * sizeof(RealType)));
        HIP_CHECK(hipMalloc(&ws.d_svd_info, sizeof(int)));
        HIP_CHECK(hipMalloc(&ws.d_svd_work, theta_size_max_ * sizeof(Scalar)));

        ws.h_svd_A.resize(theta_size_max_);
        ws.h_svd_U.resize((size_t)svd_max_m * svd_max_k);
        ws.h_svd_S.resize(svd_max_k);
        ws.h_svd_Vh.resize((size_t)svd_max_k * svd_max_n);
        ws.h_svd_tmp.resize(std::max((size_t)svd_max_m * svd_max_k, (size_t)svd_max_k * svd_max_n));
        ws.h_svd_rwork.resize(Traits::svd_rwork_size(svd_max_m, svd_max_n));

        // LAPACK SVD workspace query
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

        // NS-split eigendecomp workspace
        int ns_k = svd_max_k;  // max P matrix dimension
        ws.h_ns_PtP.resize((size_t)ns_k * ns_k);
        ws.h_ns_eigvals.resize(ns_k);
        ws.h_ns_syev_rwork.resize(Traits::syev_rwork_size(ns_k));
        // Query optimal workspace for dsyev/zheev
        {
            int n_q = ns_k;
            int lwork_q = -1;
            Scalar work_opt;
            int info;
            const char jobz = 'V', uplo = 'U';
            Traits::lapack_syev(&jobz, &uplo, &n_q, nullptr, &n_q, nullptr,
                    &work_opt, &lwork_q,
                    ws.h_ns_syev_rwork.empty() ? nullptr : ws.h_ns_syev_rwork.data(), &info);
            int opt_size;
            if constexpr (Traits::is_complex) {
                opt_size = (int)Traits::real_part(work_opt) + 1;
            } else {
                opt_size = (int)work_opt + 1;
            }
            ws.h_ns_syev_work.resize(opt_size);
        }

        // === rSVD workspace (Halko-Martinsson-Tropp with GPU QR) ===
        {
            int rsvd_m = svd_max_m;
            int rsvd_n = svd_max_n;
            int rsvd_r = chi_max_ + rsvd_oversampling_;
            ws.d_rsvd_omega = nullptr;
            ws.d_rsvd_Y = nullptr;
            ws.d_rsvd_Q = nullptr;
            ws.d_rsvd_B = nullptr;
            ws.d_rsvd_ipiv = nullptr;
            ws.d_rsvd_U_full = nullptr;
            HIP_CHECK(hipMalloc(&ws.d_rsvd_omega, (size_t)rsvd_n * rsvd_r * sizeof(Scalar)));
            HIP_CHECK(hipMalloc(&ws.d_rsvd_Y,     (size_t)rsvd_m * rsvd_r * sizeof(Scalar)));
            HIP_CHECK(hipMalloc(&ws.d_rsvd_Q,     (size_t)rsvd_m * rsvd_r * sizeof(Scalar)));
            HIP_CHECK(hipMalloc(&ws.d_rsvd_B,     (size_t)rsvd_r * rsvd_n * sizeof(Scalar)));
            HIP_CHECK(hipMalloc(&ws.d_rsvd_ipiv,  (size_t)rsvd_r * sizeof(Scalar)));
            HIP_CHECK(hipMalloc(&ws.d_rsvd_U_full, (size_t)rsvd_m * rsvd_r * sizeof(Scalar)));
            ws.h_rsvd_B.resize((size_t)rsvd_r * rsvd_n);
            ws.h_rsvd_U_small.resize((size_t)rsvd_r * rsvd_r);
        }
    }

    // Cross-segment batched pointer arrays
    int xs_batch_max = n_segments_ * D_mpo_ * dd;
    HIP_CHECK(hipMalloc(&d_xs_batch_A_, xs_batch_max * sizeof(Scalar*)));
    HIP_CHECK(hipMalloc(&d_xs_batch_B_, xs_batch_max * sizeof(Scalar*)));
    HIP_CHECK(hipMalloc(&d_xs_batch_C_, xs_batch_max * sizeof(Scalar*)));
}

// ============================================================================
// Destructor
// ============================================================================

template<typename Scalar>
PDMRGGPUOpt<Scalar>::~PDMRGGPUOpt() {
    free_gpu_resources();
}

template<typename Scalar>
void PDMRGGPUOpt<Scalar>::free_gpu_resources() {
    for (auto ptr : d_mps_tensors_) if (ptr) hipFree(ptr);
    for (auto ptr : d_mpo_tensors_) if (ptr) hipFree(ptr);
    for (auto ptr : d_W_left_) if (ptr) hipFree(ptr);
    for (auto ptr : d_W_right_) if (ptr) hipFree(ptr);
    for (auto ptr : d_WW_) if (ptr) hipFree(ptr);
    for (auto ptr : d_L_envs_) if (ptr) hipFree(ptr);
    for (auto ptr : d_R_envs_) if (ptr) hipFree(ptr);
    for (auto& ws : workspaces_) {
        if (ws.d_theta) hipFree(ws.d_theta);
        if (ws.d_heff_result) hipFree(ws.d_heff_result);
        if (ws.d_T1) hipFree(ws.d_T1);
        if (ws.d_T2) hipFree(ws.d_T2);
        if (ws.d_batch_A) hipFree(ws.d_batch_A);
        if (ws.d_batch_B) hipFree(ws.d_batch_B);
        if (ws.d_batch_C) hipFree(ws.d_batch_C);
        if (ws.d_heff_batch_A) hipFree(ws.d_heff_batch_A);
        if (ws.d_heff_batch_C) hipFree(ws.d_heff_batch_C);
        // Block-Davidson
        if (ws.d_dav_V) hipFree(ws.d_dav_V);
        if (ws.d_dav_AV) hipFree(ws.d_dav_AV);
        if (ws.d_dav_work) hipFree(ws.d_dav_work);
        if (ws.d_dav_work2) hipFree(ws.d_dav_work2);
        // Lanczos
        if (ws.d_lanczos_v) hipFree(ws.d_lanczos_v);
        if (ws.d_ritz_coeffs) hipFree(ws.d_ritz_coeffs);
        if (ws.d_dot_result) hipFree(ws.d_dot_result);
        if (ws.d_nrm2_result) hipFree(ws.d_nrm2_result);
        if (ws.d_neg_alpha) hipFree(ws.d_neg_alpha);
        if (ws.d_neg_overlap) hipFree(ws.d_neg_overlap);
        if (ws.d_inv_nrm) hipFree(ws.d_inv_nrm);
        if (ws.d_alpha_dev) hipFree(ws.d_alpha_dev);
        if (ws.d_beta_dev) hipFree(ws.d_beta_dev);
        if (ws.d_neg_beta_scalars) hipFree(ws.d_neg_beta_scalars);
        if (ws.d_const_one) hipFree(ws.d_const_one);
        if (ws.d_const_zero) hipFree(ws.d_const_zero);
        if (ws.d_const_neg_one) hipFree(ws.d_const_neg_one);
        // Newton-Schulz
        if (ws.d_ns_U) hipFree(ws.d_ns_U);
        if (ws.d_ns_U_new) hipFree(ws.d_ns_U_new);
        if (ws.d_ns_gram) hipFree(ws.d_ns_gram);
        if (ws.d_ns_P) hipFree(ws.d_ns_P);
        // SVD
        if (ws.d_svd_A) hipFree(ws.d_svd_A);
        if (ws.d_svd_U) hipFree(ws.d_svd_U);
        if (ws.d_svd_S) hipFree(ws.d_svd_S);
        if (ws.d_svd_Vh) hipFree(ws.d_svd_Vh);
        if (ws.d_svd_E) hipFree(ws.d_svd_E);
        if (ws.d_svd_info) hipFree(ws.d_svd_info);
        if (ws.d_svd_work) hipFree(ws.d_svd_work);
        // rSVD
        if (ws.d_rsvd_omega) hipFree(ws.d_rsvd_omega);
        if (ws.d_rsvd_Y) hipFree(ws.d_rsvd_Y);
        if (ws.d_rsvd_Q) hipFree(ws.d_rsvd_Q);
        if (ws.d_rsvd_B) hipFree(ws.d_rsvd_B);
        if (ws.d_rsvd_ipiv) hipFree(ws.d_rsvd_ipiv);
        if (ws.d_rsvd_U_full) hipFree(ws.d_rsvd_U_full);
    }

    for (auto& wh : worker_handles_)
        for (auto& h : wh) rocblas_destroy_handle(h);
    for (auto& ws_vec : worker_streams_)
        for (auto& s : ws_vec) hipStreamDestroy(s);
    for (auto& ev_vec : worker_done_events_)
        for (auto& e : ev_vec) hipEventDestroy(e);
    for (auto& e : step_done_events_) hipEventDestroy(e);
    for (auto& h : handles_) rocblas_destroy_handle(h);
    for (auto& s : streams_) hipStreamDestroy(s);

    // Cross-segment batched arrays
    if (d_xs_batch_A_) hipFree(d_xs_batch_A_);
    if (d_xs_batch_B_) hipFree(d_xs_batch_B_);
    if (d_xs_batch_C_) hipFree(d_xs_batch_C_);
}

// ============================================================================
// Memory management
// ============================================================================

template<typename Scalar>
void PDMRGGPUOpt<Scalar>::allocate_mps_tensor(int site, int cL, int cR) {
    size_t max_sz = (size_t)chi_max_ * d_ * chi_max_ * sizeof(Scalar);
    if (!d_mps_tensors_[site]) {
        HIP_CHECK(hipMalloc(&d_mps_tensors_[site], max_sz));
    }
    (void)cL; (void)cR;
}

template<typename Scalar>
void PDMRGGPUOpt<Scalar>::ensure_L_env_alloc(int idx, int chi) {
    if (chi > L_env_alloc_chi_[idx]) {
        if (d_L_envs_[idx]) HIP_CHECK(hipFree(d_L_envs_[idx]));
        int sz = chi * D_mpo_ * chi;
        HIP_CHECK(hipMalloc(&d_L_envs_[idx], sz * sizeof(Scalar)));
        L_env_alloc_chi_[idx] = chi;
    }
}

template<typename Scalar>
void PDMRGGPUOpt<Scalar>::ensure_R_env_alloc(int idx, int chi) {
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
void PDMRGGPUOpt<Scalar>::partition_chain() {
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
void PDMRGGPUOpt<Scalar>::initialize_boundary_states() {
    int n_boundaries = n_segments_ - 1;
    boundary_states_.resize(n_boundaries);
    for (int b = 0; b < n_boundaries; b++) {
        int bsite = boundary_bonds_[b];
        int chi = bond_dims_[bsite + 1];
        boundary_states_[b].chi = chi;
        boundary_states_[b].V.assign(chi, RealType(1.0));
    }
}

// ============================================================================
// MPS initialization
// ============================================================================

template<typename Scalar>
void PDMRGGPUOpt<Scalar>::initialize_mps_random(double scale) {
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
void PDMRGGPUOpt<Scalar>::set_mpo(const std::vector<Scalar*>& h_mpo_tensors) {
    int D = D_mpo_, d = d_;
    for (int i = 0; i < L_; i++) {
        int size = D * d * d * D;
        HIP_CHECK(hipMalloc(&d_mpo_tensors_[i], size * sizeof(Scalar)));
        HIP_CHECK(hipMemcpy(d_mpo_tensors_[i], h_mpo_tensors[i],
                            size * sizeof(Scalar), hipMemcpyHostToDevice));

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

    precompute_fused_mpo(h_mpo_tensors);
}

template<typename Scalar>
void PDMRGGPUOpt<Scalar>::precompute_fused_mpo(const std::vector<Scalar*>& h_mpo_tensors) {
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
void PDMRGGPUOpt<Scalar>::form_theta_two_site(int site, int si) {
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
// Two-site H_eff application (3-step with fused WW)
// ============================================================================

template<typename Scalar>
void PDMRGGPUOpt<Scalar>::apply_heff_two_site(int site, const Scalar* d_theta_in,
                                             Scalar* d_result, int si) {
    int cL = chi_L(site);
    int cR = chi_R(site + 1);
    int D = D_mpo_, d = d_;
    int dd = d * d;
    Scalar one = Traits::one(), zero_val = Traits::zero();
    auto& ws = workspaces_[si];

    Scalar* L_env = d_L_envs_[site];
    Scalar* R_env = d_R_envs_[site + 2];
    Scalar* WW = d_WW_[site];
    Scalar* T1 = ws.d_T1;
    Scalar* T2 = ws.d_T2;

    // Step 1: Batched GEMM — L_env^T × theta
    {
        int batch_count = D * dd;

        if (ws.heff_cached_site != site) {
            hipLaunchKernelGGL(setup_heff_A_ptrs<Scalar>, dim3(1), dim3(batch_count), 0, streams_[si],
                               ws.d_heff_batch_A, L_env, cL, dd, batch_count);
            hipLaunchKernelGGL(setup_heff_C_ptrs<Scalar>, dim3(1), dim3(batch_count), 0, streams_[si],
                               ws.d_heff_batch_C, T1, cL * cR, batch_count);
            ws.heff_cached_site = site;
        }

        hipLaunchKernelGGL(setup_heff_B_ptrs<Scalar>, dim3(1), dim3(batch_count), 0, streams_[si],
                           ws.d_batch_B, const_cast<Scalar*>(d_theta_in), cL, d, dd, batch_count);

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

    // Step 3: Loop of GEMMs — T2 × R_env
    // Strided batched over s1p when d<=2 and chi>=16 (avoids cache contention at batch_count>2)
    if (cL >= 16 && cR >= 16 && d <= 2) {
        for (int s2p = 0; s2p < d; s2p++) {
            for (int n = 0; n < D; n++) {
                Scalar beta = (n == 0) ? Traits::zero() : Traits::one();
                ROCBLAS_CHECK(Traits::gemm_strided_batched(handles_[si],
                    rocblas_operation_none, rocblas_operation_none,
                    cL, cR, cR, &one,
                    T2 + (size_t)(n * dd + s2p) * cL * cR, cL, (rocblas_stride)(d * cL * cR),
                    R_env + (size_t)n * cR, cR * D, (rocblas_stride)0,
                    &beta, d_result + (size_t)s2p * cL * d, cL * dd, (rocblas_stride)cL, d));
            }
        }
    } else {
        for (int s1p = 0; s1p < d; s1p++) {
            for (int s2p = 0; s2p < d; s2p++) {
                for (int n = 0; n < D; n++) {
                    Scalar beta = (n == 0) ? Traits::zero() : Traits::one();
                    int ws_out = n * dd + s1p * d + s2p;
                    ROCBLAS_CHECK(Traits::gemm(handles_[si],
                        rocblas_operation_none, rocblas_operation_none,
                        cL, cR, cR,
                        &one,
                        T2 + (size_t)ws_out * cL * cR, cL,
                        R_env + (size_t)n * cR, cR * D,
                        &beta,
                        d_result + s1p * cL + (size_t)s2p * cL * d, cL * dd));
                }
            }
        }
    }
}

// ============================================================================
// Left environment update
// ============================================================================

template<typename Scalar>
void PDMRGGPUOpt<Scalar>::update_left_env(int site, int si) {
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

    ROCBLAS_CHECK(Traits::gemm(handles_[si],
        rocblas_operation_none, rocblas_operation_none,
        chi_in * chi_out, d * D, D * d,
        &one,
        V, chi_in * chi_out,
        W_mat, D * d,
        &zero_val,
        U, chi_in * chi_out));

    // Step 3: Independent GEMMs dispatched on worker streams — one worker per wp (MPO index)
    // Each worker loops over sp (accumulates with beta trick); workers write to distinct L_new slices
    {
        // Record event after Steps 1+2 complete on the main stream so workers can wait
        HIP_CHECK(hipEventRecord(step_done_events_[si], streams_[si]));

        int n_active = 0;
        for (int wp = 0; wp < D; wp++) {
            int wi = wp % n_workers_;
            auto& wh = worker_handles_[si][wi];
            auto& ws_stream = worker_streams_[si][wi];

            // Worker waits for Steps 1+2 to finish
            HIP_CHECK(hipStreamWaitEvent(ws_stream, step_done_events_[si], 0));

            for (int sp = 0; sp < d; sp++) {
                Scalar beta = (sp == 0) ? Traits::zero() : Traits::one();
                ROCBLAS_CHECK(Traits::gemm(wh,
                    Traits::op_h, rocblas_operation_none,
                    chi_out, chi_out, chi_in,
                    &one,
                    U + (size_t)(wp * d + sp) * chi_in * chi_out, chi_in,
                    A + (size_t)sp * chi_in,                        chi_in * d,
                    &beta,
                    L_new + (size_t)wp * chi_out, chi_out * D));
            }

            HIP_CHECK(hipEventRecord(worker_done_events_[si][wi], ws_stream));
            n_active = std::max(n_active, wi + 1);
        }
        // Main stream waits for all workers to finish
        for (int w = 0; w < n_active; w++) {
            HIP_CHECK(hipStreamWaitEvent(streams_[si], worker_done_events_[si][w], 0));
        }
    }

    if constexpr (Traits::is_complex) {
        conjugate_inplace(L_new, chi_out * D * chi_out, streams_[si]);
    }
}

// ============================================================================
// Right environment update
// ============================================================================

template<typename Scalar>
void PDMRGGPUOpt<Scalar>::update_right_env(int site, int si) {
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

    ROCBLAS_CHECK(Traits::gemm(handles_[si],
        rocblas_operation_none, rocblas_operation_none,
        chi_out * chi_in, d * D, D * d,
        &one,
        V, chi_out * chi_in,
        W_mat, D * d,
        &zero_val,
        U, chi_out * chi_in));

    // Step 3: Independent GEMMs dispatched on worker streams — one worker per w (MPO index)
    // Each worker loops over sp (accumulates with beta trick); workers write to distinct R_new slices
    {
        // Record event after Steps 1+2 complete on the main stream so workers can wait
        HIP_CHECK(hipEventRecord(step_done_events_[si], streams_[si]));

        int n_active = 0;
        for (int w = 0; w < D; w++) {
            int wi = w % n_workers_;
            auto& wh = worker_handles_[si][wi];
            auto& ws_stream = worker_streams_[si][wi];

            // Worker waits for Steps 1+2 to finish
            HIP_CHECK(hipStreamWaitEvent(ws_stream, step_done_events_[si], 0));

            for (int sp = 0; sp < d; sp++) {
                Scalar beta = (sp == 0) ? Traits::zero() : Traits::one();
                ROCBLAS_CHECK(Traits::gemm(wh,
                    rocblas_operation_none, Traits::op_h,
                    chi_out, chi_out, chi_in,
                    &one,
                    U + (size_t)(w * d + sp) * chi_out * chi_in, chi_out,
                    A + (size_t)sp * chi_out,                     chi_out * d,
                    &beta,
                    R_new + (size_t)w * chi_out, chi_out * D));
            }

            HIP_CHECK(hipEventRecord(worker_done_events_[si][wi], ws_stream));
            n_active = std::max(n_active, wi + 1);
        }
        // Main stream waits for all workers to finish
        for (int w = 0; w < n_active; w++) {
            HIP_CHECK(hipStreamWaitEvent(streams_[si], worker_done_events_[si][w], 0));
        }
    }
}

// ============================================================================
// Environment building
// ============================================================================

template<typename Scalar>
void PDMRGGPUOpt<Scalar>::build_initial_environments() {
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

    for (int i = 0; i < L_; i++) {
        update_left_env(i, 0);
    }
    HIP_CHECK(hipStreamSynchronize(streams_[0]));

    for (int i = L_ - 1; i >= 0; i--) {
        update_right_env(i, 0);
    }
    HIP_CHECK(hipStreamSynchronize(streams_[0]));
}

// ============================================================================
// Newton-Schulz Left Polar Decomposition (tall/square, m >= n)
// A = U @ P, where U^H U = I_n, P = U^H A is PSD
// ============================================================================

template<typename Scalar>
void PDMRGGPUOpt<Scalar>::newton_schulz_left(
    Scalar* d_A, int m, int n,
    Scalar* d_U, Scalar* d_P,
    int si, double tol, int* out_iters) {

    auto& ws = workspaces_[si];
    Scalar one = Traits::one(), zero_val = Traits::zero();
    Scalar half = Traits::make_scalar(0.5);

    // Compute ||A||_F
    RealType fro;
    HIP_CHECK(hipStreamSynchronize(streams_[si]));
    ROCBLAS_CHECK(Traits::nrm2(handles_[si], m * n, d_A, 1, &fro));

    if (fro < 1e-300) {
        HIP_CHECK(hipMemsetAsync(d_U, 0, m * n * sizeof(Scalar), streams_[si]));
        HIP_CHECK(hipMemsetAsync(d_P, 0, n * n * sizeof(Scalar), streams_[si]));
        if (out_iters) *out_iters = 0;
        return;
    }

    // U = A / ||A||_F
    HIP_CHECK(hipMemcpyAsync(d_U, d_A, m * n * sizeof(Scalar), hipMemcpyDeviceToDevice, streams_[si]));
    RealType inv_fro = 1.0 / fro;
    ROCBLAS_CHECK(Traits::scal_real(handles_[si], m * n, &inv_fro, d_U, 1));

    Scalar* d_gram = ws.d_ns_gram;    // (n, n)
    Scalar* d_U_new = ws.d_ns_U_new;  // (m, n)

    int total_iters = 0;
    for (int iter = 0; iter < 30; iter++) {
        // 1. gram = U^H @ U  → (n, n)
        ROCBLAS_CHECK(Traits::gemm(handles_[si],
            Traits::op_h, rocblas_operation_none,
            n, n, m, &one, d_U, m, d_U, m, &zero_val, d_gram, n));

        // 2. gram = 3I - gram  (in-place)
        launch_scaled_identity_minus(d_gram, n, 3.0, streams_[si]);

        // 3. U_new = 0.5 * U @ gram  → (m, n)
        ROCBLAS_CHECK(Traits::gemm(handles_[si],
            rocblas_operation_none, rocblas_operation_none,
            m, n, n, &half, d_U, m, d_gram, n, &zero_val, d_U_new, m));

        total_iters = iter + 1;

        // Convergence check every 3 iterations
        if (iter >= 2 && iter % 3 == 0) {
            // diff = U_new - U → compute in d_heff_result (scratch)
            HIP_CHECK(hipMemcpyAsync(ws.d_heff_result, d_U_new, m * n * sizeof(Scalar),
                                      hipMemcpyDeviceToDevice, streams_[si]));
            Scalar neg_one = Traits::neg(Traits::one());
            ROCBLAS_CHECK(Traits::axpy(handles_[si], m * n, &neg_one, d_U, 1, ws.d_heff_result, 1));

            RealType diff_norm;
            HIP_CHECK(hipStreamSynchronize(streams_[si]));
            ROCBLAS_CHECK(Traits::nrm2(handles_[si], m * n, ws.d_heff_result, 1, &diff_norm));

            // Swap U ← U_new
            std::swap(d_U, d_U_new);
            // Fix workspace pointer if we swapped
            if (d_U == ws.d_ns_U_new) {
                // d_U is now d_ns_U_new, d_U_new is d_ns_U
                // This is fine, just keep going with swapped pointers
            }

            if (diff_norm < tol) break;
        } else {
            // Swap without convergence check
            std::swap(d_U, d_U_new);
        }
    }

    if (out_iters) *out_iters = total_iters;

    // Ensure d_U is in ws.d_ns_U (the expected output location)
    // After swaps, d_U might be in d_ns_U_new
    if (d_U != ws.d_ns_U) {
        HIP_CHECK(hipMemcpyAsync(ws.d_ns_U, d_U, m * n * sizeof(Scalar),
                                  hipMemcpyDeviceToDevice, streams_[si]));
    }

    // P = U^H @ A  → (n, m) × (m, n) → (n, n)
    ROCBLAS_CHECK(Traits::gemm(handles_[si],
        Traits::op_h, rocblas_operation_none,
        n, n, m, &one, ws.d_ns_U, m, d_A, m, &zero_val, d_P, n));
}

// ============================================================================
// Newton-Schulz Right Polar Decomposition (wide, m < n)
// A = L @ Q, where Q Q^H = I_m, L = A @ Q^H is PSD
// ============================================================================

template<typename Scalar>
void PDMRGGPUOpt<Scalar>::newton_schulz_right(
    Scalar* d_A, int m, int n,
    Scalar* d_L, Scalar* d_Q,
    int si, double tol, int* out_iters) {

    auto& ws = workspaces_[si];
    Scalar one = Traits::one(), zero_val = Traits::zero();
    Scalar half = Traits::make_scalar(0.5);

    // Compute ||A||_F
    RealType fro;
    HIP_CHECK(hipStreamSynchronize(streams_[si]));
    ROCBLAS_CHECK(Traits::nrm2(handles_[si], m * n, d_A, 1, &fro));

    if (fro < 1e-300) {
        HIP_CHECK(hipMemsetAsync(d_Q, 0, m * n * sizeof(Scalar), streams_[si]));
        HIP_CHECK(hipMemsetAsync(d_L, 0, m * m * sizeof(Scalar), streams_[si]));
        if (out_iters) *out_iters = 0;
        return;
    }

    // Q = A / ||A||_F
    HIP_CHECK(hipMemcpyAsync(d_Q, d_A, m * n * sizeof(Scalar), hipMemcpyDeviceToDevice, streams_[si]));
    RealType inv_fro = 1.0 / fro;
    ROCBLAS_CHECK(Traits::scal_real(handles_[si], m * n, &inv_fro, d_Q, 1));

    Scalar* d_gram = ws.d_ns_gram;     // reuse for (m, m)
    Scalar* d_Q_new = ws.d_ns_U_new;   // reuse for (m, n)

    int total_iters = 0;
    for (int iter = 0; iter < 30; iter++) {
        // 1. gram = Q @ Q^H  → (m, m)
        ROCBLAS_CHECK(Traits::gemm(handles_[si],
            rocblas_operation_none, Traits::op_h,
            m, m, n, &one, d_Q, m, d_Q, m, &zero_val, d_gram, m));

        // 2. gram = 3I - gram  (in-place)
        launch_scaled_identity_minus(d_gram, m, 3.0, streams_[si]);

        // 3. Q_new = 0.5 * gram @ Q  → (m, n)
        ROCBLAS_CHECK(Traits::gemm(handles_[si],
            rocblas_operation_none, rocblas_operation_none,
            m, n, m, &half, d_gram, m, d_Q, m, &zero_val, d_Q_new, m));

        total_iters = iter + 1;

        if (iter >= 2 && iter % 3 == 0) {
            HIP_CHECK(hipMemcpyAsync(ws.d_heff_result, d_Q_new, m * n * sizeof(Scalar),
                                      hipMemcpyDeviceToDevice, streams_[si]));
            Scalar neg_one = Traits::neg(Traits::one());
            ROCBLAS_CHECK(Traits::axpy(handles_[si], m * n, &neg_one, d_Q, 1, ws.d_heff_result, 1));

            RealType diff_norm;
            HIP_CHECK(hipStreamSynchronize(streams_[si]));
            ROCBLAS_CHECK(Traits::nrm2(handles_[si], m * n, ws.d_heff_result, 1, &diff_norm));

            std::swap(d_Q, d_Q_new);
            if (diff_norm < tol) break;
        } else {
            std::swap(d_Q, d_Q_new);
        }
    }

    if (out_iters) *out_iters = total_iters;

    // Ensure d_Q is in the right place
    if (d_Q != ws.d_ns_U_new && d_Q != ws.d_ns_U) {
        // shouldn't happen, but safety
    }
    // Q might have been swapped. Copy to a known location for output.
    // We use d_ns_U_new as Q's home
    Scalar* final_Q = d_Q;

    // L = A @ Q^H  → (m, n) × (n, m) → (m, m)
    ROCBLAS_CHECK(Traits::gemm(handles_[si],
        rocblas_operation_none, Traits::op_h,
        m, m, n, &one, d_A, m, final_Q, m, &zero_val, d_L, m));

    // Copy Q to output location if needed
    if (final_Q != d_Q) {
        HIP_CHECK(hipMemcpyAsync(d_Q, final_Q, m * n * sizeof(Scalar),
                                  hipMemcpyDeviceToDevice, streams_[si]));
    }
}

// ============================================================================
// Newton-Schulz bond split (Option A)
// Uses Newton-Schulz + eigendecomposition of P^H P for truncation
// ============================================================================

template<typename Scalar>
void PDMRGGPUOpt<Scalar>::ns_split(int site, Scalar* d_theta, char direction, int si) {
    int cL = chi_L(site);
    int cR = chi_R(site + 1);
    auto& ws = workspaces_[si];

    int m = cL * d_;
    int n_svd = d_ * cR;
    int k = std::min(m, n_svd);
    int max_k = std::min(k, chi_max_user_);

    // For very small systems, fall back to SVD
    if (k <= 4 || m < 2 || n_svd < 2) {
        svd_split(site, d_theta, direction, si);
        return;
    }

    if (m >= n_svd) {
        // Tall/square: left Newton-Schulz → A = U_ns @ P
        // U_ns is (m, n_svd) isometric, P is (n_svd, n_svd)
        int ns_iters = 0;
        newton_schulz_left(d_theta, m, n_svd, ws.d_ns_U, ws.d_ns_P, si, 1e-10, &ns_iters);

        // If NS didn't converge well, fall back to SVD
        if (ns_iters >= 29) {
            svd_split(site, d_theta, direction, si);
            return;
        }

        HIP_CHECK(hipStreamSynchronize(streams_[si]));

        // Verify ||U^H U - I||_F < tol to catch silent NS failures.
        // Compute U^H U into d_ns_gram (n_svd × n_svd), then check diagonal/off-diagonal.
        {
            Scalar one_v = Traits::one(), zero_v = Traits::zero();
            ROCBLAS_CHECK(Traits::gemm(handles_[si],
                Traits::op_h, rocblas_operation_none,
                n_svd, n_svd, m, &one_v, ws.d_ns_U, m, ws.d_ns_U, m,
                &zero_v, ws.d_ns_gram, n_svd));

            // Copy to host and check ||UtU - I||_F
            std::vector<Scalar> h_UtU(n_svd * n_svd);
            HIP_CHECK(hipMemcpy(h_UtU.data(), ws.d_ns_gram,
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
                svd_split(site, d_theta, direction, si);
                return;
            }
        }

        // Eigendecompose P^H P → eigenvalues σ², eigenvectors V
        // P^H P on GPU: (n_svd, n_svd) × (n_svd, n_svd) → (n_svd, n_svd)
        Scalar one = Traits::one(), zero_val = Traits::zero();
        Scalar* d_PtP = ws.d_ns_gram;  // reuse gram buffer for (n_svd, n_svd)
        ROCBLAS_CHECK(Traits::gemm(handles_[si],
            Traits::op_h, rocblas_operation_none,
            n_svd, n_svd, n_svd, &one, ws.d_ns_P, n_svd, ws.d_ns_P, n_svd,
            &zero_val, d_PtP, n_svd));

        // Copy P^H P to host
        HIP_CHECK(hipMemcpy(ws.h_ns_PtP.data(), d_PtP, n_svd * n_svd * sizeof(Scalar),
                            hipMemcpyDeviceToHost));

        // Eigendecompose on CPU
        int info;
        const char jobz = 'V', uplo = 'U';
        int lwork = (int)ws.h_ns_syev_work.size();
        Traits::lapack_syev(&jobz, &uplo, &n_svd,
                ws.h_ns_PtP.data(), &n_svd,
                ws.h_ns_eigvals.data(),
                ws.h_ns_syev_work.data(), &lwork,
                ws.h_ns_syev_rwork.empty() ? nullptr : ws.h_ns_syev_rwork.data(),
                &info);

        if (info != 0) {
            // Eigendecomp failed, fall back to SVD
            svd_split(site, d_theta, direction, si);
            return;
        }

        // Eigenvalues are in ascending order. Singular values = sqrt(eigenvalues).
        // Reverse to get descending order.
        std::vector<RealType> sing_vals(n_svd);
        for (int i = 0; i < n_svd; i++) {
            RealType ev = ws.h_ns_eigvals[n_svd - 1 - i];
            sing_vals[i] = (ev > 0) ? std::sqrt(ev) : 0.0;
        }

        // Eigenvectors are columns of h_ns_PtP (overwritten by dsyev).
        // They correspond to ascending eigenvalues. We need descending order.
        // V_reversed[:, i] = V[:, n_svd-1-i]
        // Since P^H P = V Σ² V^H, and SVD of P gives P = U_p S Vh,
        // then P^H P = Vh^H S² Vh, so Vh = V^H (reversed).

        // Truncation
        int new_k = std::min(max_k, n_svd);
        for (int i = 0; i < new_k; i++) {
            if (sing_vals[i] < 1e-14) { new_k = i; break; }
        }
        if (new_k == 0) new_k = 1;

        // Build Vh_trunc (new_k × n_svd) on host: rows are reversed eigenvectors
        // V[:, n_svd-1-i] → row i of Vh
        std::vector<Scalar> h_Vh_trunc(new_k * n_svd);
        for (int i = 0; i < new_k; i++) {
            int src_col = n_svd - 1 - i;  // reversed index
            for (int j = 0; j < n_svd; j++) {
                // Vh[i, j] = conj(V[j, src_col])
                Scalar v_val = ws.h_ns_PtP[j + src_col * n_svd];
                if constexpr (Traits::is_complex) {
                    h_Vh_trunc[i + j * new_k] = make_hipDoubleComplex(hipCreal(v_val), -hipCimag(v_val));
                } else {
                    h_Vh_trunc[i + j * new_k] = v_val;
                }
            }
        }

        // Upload Vh_trunc to GPU (into d_svd_Vh)
        HIP_CHECK(hipMemcpy(ws.d_svd_Vh, h_Vh_trunc.data(),
                            new_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice));

        // Compute U_p_trunc = P @ V_trunc @ diag(1/S) on GPU
        // V_trunc[:, i] = eigenvector for i-th largest eigenvalue = column (n_svd-1-i)
        // First: upload V_trunc (n_svd × new_k) to GPU
        std::vector<Scalar> h_V_trunc(n_svd * new_k);
        for (int i = 0; i < new_k; i++) {
            int src_col = n_svd - 1 - i;
            for (int j = 0; j < n_svd; j++) {
                h_V_trunc[j + i * n_svd] = ws.h_ns_PtP[j + src_col * n_svd];
            }
        }
        HIP_CHECK(hipMemcpy(ws.d_svd_U, h_V_trunc.data(),
                            n_svd * new_k * sizeof(Scalar), hipMemcpyHostToDevice));

        // U_p = P @ V_trunc → (n_svd, n_svd) × (n_svd, new_k) → (n_svd, new_k)
        // Store in d_svd_work (reuse)
        ROCBLAS_CHECK(Traits::gemm(handles_[si],
            rocblas_operation_none, rocblas_operation_none,
            n_svd, new_k, n_svd, &one, ws.d_ns_P, n_svd,
            ws.d_svd_U, n_svd, &zero_val, ws.d_svd_work, n_svd));

        // Scale columns by 1/S: U_p[:, i] /= S[i]
        for (int i = 0; i < new_k; i++) {
            if (sing_vals[i] > 1e-14) {
                RealType inv_s = 1.0 / sing_vals[i];
                ROCBLAS_CHECK(Traits::scal_real(handles_[si], n_svd, &inv_s,
                    ws.d_svd_work + i * n_svd, 1));
            }
        }

        // U_full = U_ns @ U_p → (m, n_svd) × (n_svd, new_k) → (m, new_k)
        // Store in d_svd_A (reuse)
        ROCBLAS_CHECK(Traits::gemm(handles_[si],
            rocblas_operation_none, rocblas_operation_none,
            m, new_k, n_svd, &one, ws.d_ns_U, m,
            ws.d_svd_work, n_svd, &zero_val, ws.d_svd_A, m));

        HIP_CHECK(hipStreamSynchronize(streams_[si]));

        // Upload singular values to device for GPU-side scaling
        HIP_CHECK(hipMemcpyAsync(ws.d_svd_S, sing_vals.data(), new_k * sizeof(RealType),
                                  hipMemcpyHostToDevice, streams_[si]));

        // Store MPS tensors — scale on GPU
        if (direction == 'R') {
            // MPS[site] = U_full[:, :new_k] → (cL*d, new_k) = (m, new_k)
            allocate_mps_tensor(site, cL, new_k);
            HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site], ws.d_svd_A,
                        m * new_k * sizeof(Scalar), hipMemcpyDeviceToDevice, streams_[si]));

            // MPS[site+1] = diag(S) @ Vh[:new_k, :] — Vh is on device at d_svd_Vh, scale rows on GPU
            allocate_mps_tensor(site + 1, new_k, cR);
            scale_rows_by_real(ws.d_svd_Vh, new_k, ws.d_svd_S,
                               d_mps_tensors_[site + 1], new_k, new_k, n_svd, streams_[si]);
        } else {
            // MPS[site] = U_full @ diag(S) → scale columns of U_full on GPU
            allocate_mps_tensor(site, cL, new_k);
            scale_columns_by_real(ws.d_svd_A, m, ws.d_svd_S,
                                  d_mps_tensors_[site], m, m, new_k, streams_[si]);

            // MPS[site+1] = Vh[:new_k, :] — upload from host
            allocate_mps_tensor(site + 1, new_k, cR);
            HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site + 1], h_Vh_trunc.data(),
                        new_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));
        }

        bond_dims_[site + 1] = new_k;
        ws.heff_cached_site = -1;

    } else {
        // Wide case: right Newton-Schulz → A = L @ Q
        // Small matrix at boundaries — fall back to SVD for simplicity.
        svd_split(site, d_theta, direction, si);
        return;
    }
}

// ============================================================================
// Randomized truncated SVD split (Halko-Martinsson-Tropp with GPU QR)
// ============================================================================

template<typename Scalar>
void PDMRGGPUOpt<Scalar>::rsvd_split(int site, Scalar* d_theta, char direction, int si) {
    int cL = chi_L(site);
    int cR = chi_R(site + 1);
    auto& ws = workspaces_[si];

    int m = cL * d_;
    int n_svd = d_ * cR;
    int full_k = std::min(m, n_svd);
    int k = std::min(full_k, chi_max_user_);

    // If matrix is small enough, fall back to full SVD (rSVD overhead not worth it)
    if (full_k <= k + rsvd_oversampling_ || m <= 2 * k) {
        svd_split(site, d_theta, direction, si);
        return;
    }

    int r = k + rsvd_oversampling_;  // projection rank

    Scalar one = Traits::one(), zero_val = Traits::zero();

    // Step 1: Generate random Omega (n_svd x r) on CPU and upload to GPU
    {
        std::vector<Scalar> h_omega(n_svd * r);
        for (int i = 0; i < n_svd * r; i++) {
            h_omega[i] = Traits::random_val();
        }
        HIP_CHECK(hipMemcpyAsync(ws.d_rsvd_omega, h_omega.data(),
                                  n_svd * r * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));
    }

    // Step 2: Y = theta @ Omega on GPU  (m x n_svd) @ (n_svd x r) -> (m x r)
    ROCBLAS_CHECK(Traits::gemm(handles_[si],
        rocblas_operation_none, rocblas_operation_none,
        m, r, n_svd, &one,
        d_theta, m,
        ws.d_rsvd_omega, n_svd,
        &zero_val,
        ws.d_rsvd_Y, m));
    HIP_CHECK(hipStreamSynchronize(streams_[si]));

    // Step 3: QR factorization of Y on GPU -> Q (m x r) stays on device
    //   Copy Y -> Q buffer (geqrf overwrites input), then QR in-place
    HIP_CHECK(hipMemcpyAsync(ws.d_rsvd_Q, ws.d_rsvd_Y,
                              (size_t)m * r * sizeof(Scalar), hipMemcpyDeviceToDevice, streams_[si]));
    ROCBLAS_CHECK(Traits::rocsolver_geqrf(handles_[si], m, r, ws.d_rsvd_Q, m, ws.d_rsvd_ipiv));
    ROCBLAS_CHECK(Traits::rocsolver_orgqr(handles_[si], m, r, r, ws.d_rsvd_Q, m, ws.d_rsvd_ipiv));
    HIP_CHECK(hipStreamSynchronize(streams_[si]));

    // Step 4: B = Q^H @ theta on GPU  (r x m) @ (m x n_svd) -> (r x n_svd)
    ROCBLAS_CHECK(Traits::gemm(handles_[si],
        Traits::op_h, rocblas_operation_none,
        r, n_svd, m, &one,
        ws.d_rsvd_Q, m,
        d_theta, m,
        &zero_val,
        ws.d_rsvd_B, r));
    HIP_CHECK(hipStreamSynchronize(streams_[si]));

    // Step 5: Copy B to host, compute SVD of B (r x n_svd) -- much smaller than (m x n_svd)
    HIP_CHECK(hipMemcpy(ws.h_rsvd_B.data(), ws.d_rsvd_B,
                         r * n_svd * sizeof(Scalar), hipMemcpyDeviceToHost));

    int small_k = std::min(r, n_svd);
    {
        int lwork = (int)ws.h_svd_work.size();
        int info;
        const char jobu = 'S', jobvt = 'S';
        // U_small: (r x small_k), S: (small_k), Vh: (small_k x n_svd)
        Traits::lapack_gesvd(&jobu, &jobvt, &r, &n_svd, ws.h_rsvd_B.data(), &r,
                ws.h_svd_S.data(), ws.h_rsvd_U_small.data(), &r,
                ws.h_svd_Vh.data(), &small_k,
                ws.h_svd_work.data(), &lwork,
                ws.h_svd_rwork.empty() ? nullptr : ws.h_svd_rwork.data(), &info);
        if (info != 0) {
            // SVD of small B failed, fall back to full SVD
            svd_split(site, d_theta, direction, si);
            return;
        }
    }

    // Step 6: Upload U_small to GPU, compute U_full = Q @ U_small on GPU
    //   Q is (m x r) on device, U_small is (r x small_k) on host -> U_full (m x small_k) on device
    {
        // Upload U_small to device (reuse d_rsvd_B as temp -- it's no longer needed)
        HIP_CHECK(hipMemcpyAsync(ws.d_rsvd_B, ws.h_rsvd_U_small.data(),
                                  (size_t)r * small_k * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));
        ROCBLAS_CHECK(Traits::gemm(handles_[si],
            rocblas_operation_none, rocblas_operation_none,
            m, small_k, r, &one,
            ws.d_rsvd_Q, m,
            ws.d_rsvd_B, r,
            &zero_val,
            ws.d_rsvd_U_full, m));
        HIP_CHECK(hipStreamSynchronize(streams_[si]));
    }

    // Now: U_full (m x small_k) is on GPU at d_rsvd_U_full
    //      S (small_k) and Vh (small_k x n_svd) are on host
    RealType* h_S_data = ws.h_svd_S.data();

    // Truncation
    int new_k = k;
    for (int i = 0; i < new_k; i++) {
        if (h_S_data[i] < 1e-14) { new_k = i; break; }
    }
    if (new_k == 0) new_k = 1;

    // Upload S to device for GPU-side scaling
    HIP_CHECK(hipMemcpyAsync(ws.d_svd_S, h_S_data, small_k * sizeof(RealType),
                              hipMemcpyHostToDevice, streams_[si]));

    if (direction == 'R') {
        // MPS[site] = U_full[:, :new_k] -- D2D copy
        allocate_mps_tensor(site, cL, new_k);
        HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site], ws.d_rsvd_U_full,
                                  (size_t)m * new_k * sizeof(Scalar),
                                  hipMemcpyDeviceToDevice, streams_[si]));

        // MPS[site+1] = diag(S[:new_k]) @ Vh[:new_k, :] — upload Vh, scale rows on GPU
        HIP_CHECK(hipMemcpyAsync(ws.d_svd_Vh, ws.h_svd_Vh.data(),
                                  (size_t)small_k * n_svd * sizeof(Scalar),
                                  hipMemcpyHostToDevice, streams_[si]));
        allocate_mps_tensor(site + 1, new_k, cR);
        scale_rows_by_real(ws.d_svd_Vh, small_k, ws.d_svd_S,
                           d_mps_tensors_[site + 1], new_k, new_k, n_svd, streams_[si]);
    } else {
        // MPS[site] = U_full[:, :new_k] @ diag(S) — scale columns on GPU (U_full already on device)
        allocate_mps_tensor(site, cL, new_k);
        scale_columns_by_real(ws.d_rsvd_U_full, m, ws.d_svd_S,
                              d_mps_tensors_[site], m, m, new_k, streams_[si]);

        // MPS[site+1] = Vh[:new_k, :] — upload from host
        allocate_mps_tensor(site + 1, new_k, cR);
        if (new_k == small_k) {
            HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site + 1], ws.h_svd_Vh.data(),
                                      (size_t)small_k * n_svd * sizeof(Scalar),
                                      hipMemcpyHostToDevice, streams_[si]));
        } else {
            for (int j = 0; j < n_svd; j++)
                for (int i = 0; i < new_k; i++)
                    ws.h_svd_tmp[i + j * new_k] = ws.h_svd_Vh[i + j * small_k];
            HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site + 1], ws.h_svd_tmp.data(),
                                      (size_t)new_k * n_svd * sizeof(Scalar),
                                      hipMemcpyHostToDevice, streams_[si]));
        }
    }

    bond_dims_[site + 1] = new_k;
    ws.heff_cached_site = -1;
}

// ============================================================================
// Standard SVD split (fallback)
// ============================================================================

template<typename Scalar>
void PDMRGGPUOpt<Scalar>::svd_split(int site, Scalar* d_theta, char direction, int si) {
    int cL = chi_L(site);
    int cR = chi_R(site + 1);
    auto& ws = workspaces_[si];

    int m = cL * d_;
    int n_svd = d_ * cR;
    int full_k = std::min(m, n_svd);
    int k = std::min(full_k, chi_max_user_);

    RealType* h_S_data;
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

        h_S_data = ws.h_svd_S.data();
    } else {
        gpu_svd_path = true;
        HIP_CHECK(hipMemcpyAsync(ws.d_svd_A, d_theta, m * n_svd * sizeof(Scalar),
                                  hipMemcpyDeviceToDevice, streams_[si]));

        Traits::rocsolver_gesvd(handles_[si],
            rocblas_svect_singular, rocblas_svect_singular,
            m, n_svd,
            ws.d_svd_A, m,
            ws.d_svd_S,
            ws.d_svd_U, m,
            ws.d_svd_Vh, full_k,
            ws.d_svd_E,
            rocblas_outofplace,
            ws.d_svd_info);

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
        // GPU SVD: U, S, Vh all on device already — scale on GPU
        if (direction == 'R') {
            // MPS[site] = U[:, :new_k] — D2D copy
            allocate_mps_tensor(site, cL, new_k);
            HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site], ws.d_svd_U,
                        m * new_k * sizeof(Scalar), hipMemcpyDeviceToDevice, streams_[si]));

            // MPS[site+1] = diag(S) @ Vh[:new_k, :] — scale rows of Vh on GPU
            allocate_mps_tensor(site + 1, new_k, cR);
            scale_rows_by_real(ws.d_svd_Vh, full_k, ws.d_svd_S,
                               d_mps_tensors_[site + 1], new_k, new_k, n_svd, streams_[si]);
        } else {
            // MPS[site] = U[:, :new_k] @ diag(S) — scale columns of U on GPU
            allocate_mps_tensor(site, cL, new_k);
            scale_columns_by_real(ws.d_svd_U, m, ws.d_svd_S,
                                  d_mps_tensors_[site], m, m, new_k, streams_[si]);

            // MPS[site+1] = Vh[:new_k, :]
            allocate_mps_tensor(site + 1, new_k, cR);
            if (new_k == full_k) {
                HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site + 1], ws.d_svd_Vh,
                            (size_t)full_k * n_svd * sizeof(Scalar), hipMemcpyDeviceToDevice, streams_[si]));
            } else {
                // Need to extract first new_k rows from Vh (lda=full_k, want lda=new_k)
                for (int j = 0; j < n_svd; j++)
                    HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site + 1] + (size_t)j * new_k,
                                ws.d_svd_Vh + (size_t)j * full_k,
                                new_k * sizeof(Scalar), hipMemcpyDeviceToDevice, streams_[si]));
            }
        }
    } else {
        // CPU SVD: upload S to device, then upload U/Vh and scale on GPU
        HIP_CHECK(hipMemcpyAsync(ws.d_svd_S, h_S_data, full_k * sizeof(RealType),
                                  hipMemcpyHostToDevice, streams_[si]));

        if (direction == 'R') {
            // MPS[site] = U[:, :new_k] — upload directly
            allocate_mps_tensor(site, cL, new_k);
            HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site], ws.h_svd_U.data(),
                        m * new_k * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));

            // MPS[site+1] = diag(S) @ Vh[:new_k, :] — upload Vh to device, scale rows on GPU
            HIP_CHECK(hipMemcpyAsync(ws.d_svd_Vh, ws.h_svd_Vh.data(),
                        (size_t)full_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));
            allocate_mps_tensor(site + 1, new_k, cR);
            scale_rows_by_real(ws.d_svd_Vh, full_k, ws.d_svd_S,
                               d_mps_tensors_[site + 1], new_k, new_k, n_svd, streams_[si]);
        } else {
            // MPS[site] = U[:, :new_k] @ diag(S) — upload U, scale columns on GPU
            HIP_CHECK(hipMemcpyAsync(ws.d_svd_U, ws.h_svd_U.data(),
                        (size_t)m * new_k * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));
            allocate_mps_tensor(site, cL, new_k);
            scale_columns_by_real(ws.d_svd_U, m, ws.d_svd_S,
                                  d_mps_tensors_[site], m, m, new_k, streams_[si]);

            // MPS[site+1] = Vh[:new_k, :] — upload directly
            allocate_mps_tensor(site + 1, new_k, cR);
            if (new_k == full_k) {
                HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site + 1], ws.h_svd_Vh.data(),
                            (size_t)full_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));
            } else {
                for (int j = 0; j < n_svd; j++)
                    for (int i = 0; i < new_k; i++)
                        ws.h_svd_tmp[i + j * new_k] = ws.h_svd_Vh[i + j * full_k];
                HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site + 1], ws.h_svd_tmp.data(),
                            (size_t)new_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));
            }
        }
    }

    bond_dims_[site + 1] = new_k;
    ws.heff_cached_site = -1;
}

// ============================================================================
// Block-Davidson Eigensolver
// ============================================================================

template<typename Scalar>
double PDMRGGPUOpt<Scalar>::block_davidson_eigensolver(int site, Scalar* d_theta, int theta_size, int si) {
    int dim = theta_size;
    int b = std::min(davidson_b_, dim);
    int max_sub = std::min(davidson_max_sub_, dim);
    int max_iter = 30;
    double tol_dav = 1e-10;
    auto& ws = workspaces_[si];

    Scalar one = Traits::one(), zero_val = Traits::zero();
    Scalar neg_one = Traits::neg(Traits::one());

    // For tiny systems, use Lanczos fallback
    if (dim <= 2 * b) {
        return lanczos_eigensolver(site, d_theta, theta_size, si);
    }

    // Initialize V: first column = theta/||theta||
    RealType norm;
    ROCBLAS_CHECK(Traits::nrm2(handles_[si], dim, d_theta, 1, &norm));
    if (norm < 1e-14) {
        srand(42 + site);
        std::vector<Scalar> h_init(dim);
        for (int i = 0; i < dim; i++) h_init[i] = Traits::random_val();
        HIP_CHECK(hipMemcpy(d_theta, h_init.data(), dim * sizeof(Scalar), hipMemcpyHostToDevice));
        ROCBLAS_CHECK(Traits::nrm2(handles_[si], dim, d_theta, 1, &norm));
    }

    Scalar* V = ws.d_dav_V;   // (dim, max_sub)
    Scalar* AV = ws.d_dav_AV; // (dim, max_sub)

    // V[:, 0] = theta / norm
    HIP_CHECK(hipMemcpyAsync(V, d_theta, dim * sizeof(Scalar), hipMemcpyDeviceToDevice, streams_[si]));
    RealType inv_norm = 1.0 / norm;
    ROCBLAS_CHECK(Traits::scal_real(handles_[si], dim, &inv_norm, V, 1));

    // Fill remaining b-1 columns with random orthogonalized vectors
    srand(42 + site);
    for (int i = 1; i < b; i++) {
        std::vector<Scalar> h_v(dim);
        for (int j = 0; j < dim; j++) h_v[j] = Traits::random_val();
        HIP_CHECK(hipMemcpyAsync(V + (size_t)i * dim, h_v.data(), dim * sizeof(Scalar),
                                  hipMemcpyHostToDevice, streams_[si]));

        // Orthogonalize against previous columns using CGS
        // overlap = V[:, :i]^H @ V[:, i]
        ROCBLAS_CHECK(Traits::gemv(handles_[si], Traits::op_h,
            dim, i, &one, V, dim, V + (size_t)i * dim, 1,
            &zero_val, ws.d_dav_work, 1));
        // V[:, i] -= V[:, :i] @ overlap
        ROCBLAS_CHECK(Traits::gemv(handles_[si], rocblas_operation_none,
            dim, i, &neg_one, V, dim, ws.d_dav_work, 1,
            &one, V + (size_t)i * dim, 1));

        // Normalize
        RealType nrm_v;
        HIP_CHECK(hipStreamSynchronize(streams_[si]));
        ROCBLAS_CHECK(Traits::nrm2(handles_[si], dim, V + (size_t)i * dim, 1, &nrm_v));
        if (nrm_v < 1e-14) {
            // Generate another random vector
            for (int j = 0; j < dim; j++) h_v[j] = Traits::random_val();
            HIP_CHECK(hipMemcpy(V + (size_t)i * dim, h_v.data(), dim * sizeof(Scalar), hipMemcpyHostToDevice));
            ROCBLAS_CHECK(Traits::nrm2(handles_[si], dim, V + (size_t)i * dim, 1, &nrm_v));
        }
        RealType inv_nrm_v = 1.0 / nrm_v;
        ROCBLAS_CHECK(Traits::scal_real(handles_[si], dim, &inv_nrm_v, V + (size_t)i * dim, 1));
    }

    // Compute AV[:, j] = H @ V[:, j] for j = 0..b-1
    for (int j = 0; j < b; j++) {
        apply_heff_two_site(site, V + (size_t)j * dim, AV + (size_t)j * dim, si);
    }

    double best_energy = 1e30;
    double energy_prev = 1e30;
    int k = b;  // current subspace size

    for (int iteration = 0; iteration < max_iter; iteration++) {
        HIP_CHECK(hipStreamSynchronize(streams_[si]));

        // Rayleigh-Ritz: H_proj = V^H @ AV  → (k, k)
        ROCBLAS_CHECK(Traits::gemm(handles_[si],
            Traits::op_h, rocblas_operation_none,
            k, k, dim, &one, V, dim, AV, dim, &zero_val, ws.d_dav_work, k));

        // Copy H_proj to host
        HIP_CHECK(hipMemcpy(ws.h_dav_H_proj.data(), ws.d_dav_work,
                            k * k * sizeof(Scalar), hipMemcpyDeviceToHost));

        // Symmetrize on host: H_proj = 0.5 * (H_proj + H_proj^H)
        for (int i = 0; i < k; i++) {
            for (int j = i; j < k; j++) {
                Scalar hij = ws.h_dav_H_proj[i + j * k];
                Scalar hji = ws.h_dav_H_proj[j + i * k];
                Scalar sym;
                if constexpr (Traits::is_complex) {
                    sym = make_hipDoubleComplex(
                        0.5 * (hipCreal(hij) + hipCreal(hji)),
                        0.5 * (hipCimag(hij) - hipCimag(hji)));
                    ws.h_dav_H_proj[i + j * k] = sym;
                    ws.h_dav_H_proj[j + i * k] = make_hipDoubleComplex(hipCreal(sym), -hipCimag(sym));
                } else {
                    sym = 0.5 * (hij + hji);
                    ws.h_dav_H_proj[i + j * k] = sym;
                    ws.h_dav_H_proj[j + i * k] = sym;
                }
            }
        }

        // Eigendecompose H_proj on CPU
        std::copy(ws.h_dav_H_proj.begin(), ws.h_dav_H_proj.begin() + k * k,
                  ws.h_dav_eigvecs.begin());
        int info;
        const char jobz = 'V', uplo = 'U';
        int lwork = -1;
        Scalar work_opt;
        // Query workspace
        Traits::lapack_syev(&jobz, &uplo, &k,
                ws.h_dav_eigvecs.data(), &k, ws.h_dav_eigvals.data(),
                &work_opt, &lwork,
                ws.h_dav_syev_work.empty() ? nullptr : ws.h_dav_syev_work.data(), &info);
        if constexpr (Traits::is_complex) {
            lwork = (int)Traits::real_part(work_opt) + 1;
        } else {
            lwork = (int)work_opt + 1;
        }
        std::vector<Scalar> syev_work(lwork);
        std::vector<RealType> syev_rwork(Traits::syev_rwork_size(k));
        Traits::lapack_syev(&jobz, &uplo, &k,
                ws.h_dav_eigvecs.data(), &k, ws.h_dav_eigvals.data(),
                syev_work.data(), &lwork,
                syev_rwork.empty() ? nullptr : syev_rwork.data(), &info);

        if (info != 0) {
            // Eigendecomp failed — fall back to Lanczos
            return lanczos_eigensolver(site, d_theta, theta_size, si);
        }

        double energy = ws.h_dav_eigvals[0];  // lowest eigenvalue

        // Track best
        if (energy < best_energy) {
            best_energy = energy;
        }

        // Upload eigenvectors to GPU for Ritz vector computation
        HIP_CHECK(hipMemcpy(ws.d_dav_work2, ws.h_dav_eigvecs.data(),
                            k * k * sizeof(Scalar), hipMemcpyHostToDevice));

        // X = V @ eigvecs → use d_dav_work as scratch for X (dim, k)
        // But d_dav_work is being used for H_proj. Use d_heff_result as scratch for x0.
        // x0 = V @ eigvecs[:, 0] → (dim, 1)
        ROCBLAS_CHECK(Traits::gemv(handles_[si], rocblas_operation_none,
            dim, k, &one, V, dim, ws.d_dav_work2, 1, &zero_val, d_theta, 1));

        // ax0 = AV @ eigvecs[:, 0] → (dim, 1)
        ROCBLAS_CHECK(Traits::gemv(handles_[si], rocblas_operation_none,
            dim, k, &one, AV, dim, ws.d_dav_work2, 1, &zero_val, ws.d_heff_result, 1));

        // Residual: r = ax0 - energy * x0
        Scalar neg_energy = Traits::make_scalar(-energy);
        ROCBLAS_CHECK(Traits::axpy(handles_[si], dim, &neg_energy, d_theta, 1, ws.d_heff_result, 1));

        RealType res_norm;
        HIP_CHECK(hipStreamSynchronize(streams_[si]));
        ROCBLAS_CHECK(Traits::nrm2(handles_[si], dim, ws.d_heff_result, 1, &res_norm));

        if (res_norm < tol_dav && std::abs(energy - energy_prev) < tol_dav) {
            // Converged: d_theta already has the ground state vector
            // Normalize
            ROCBLAS_CHECK(Traits::nrm2(handles_[si], dim, d_theta, 1, &norm));
            inv_norm = 1.0 / norm;
            ROCBLAS_CHECK(Traits::scal_real(handles_[si], dim, &inv_norm, d_theta, 1));
            return energy;
        }
        energy_prev = energy;

        // Expand subspace with residual corrections
        int n_new = 0;
        // Compute residuals for lowest b Ritz pairs
        for (int i = 0; i < std::min(b, k); i++) {
            Scalar* r_i = ws.d_dav_work + (size_t)n_new * dim;

            // r_i = AV @ eigvecs[:, i] - eigvals[i] * V @ eigvecs[:, i]
            // Compute V @ eigvecs[:, i] → into r_i temporarily
            ROCBLAS_CHECK(Traits::gemv(handles_[si], rocblas_operation_none,
                dim, k, &one, V, dim, ws.d_dav_work2 + i * k, 1, &zero_val, r_i, 1));
            // Compute AV @ eigvecs[:, i] → into d_heff_result
            ROCBLAS_CHECK(Traits::gemv(handles_[si], rocblas_operation_none,
                dim, k, &one, AV, dim, ws.d_dav_work2 + i * k, 1, &zero_val, ws.d_heff_result, 1));
            // r_i = d_heff_result - eigvals[i] * r_i
            Scalar neg_ei = Traits::make_scalar(-ws.h_dav_eigvals[i]);
            ROCBLAS_CHECK(Traits::scal(handles_[si], dim, &neg_ei, r_i, 1));
            ROCBLAS_CHECK(Traits::axpy(handles_[si], dim, &one, ws.d_heff_result, 1, r_i, 1));

            // Check if residual is significant
            RealType ri_norm;
            HIP_CHECK(hipStreamSynchronize(streams_[si]));
            ROCBLAS_CHECK(Traits::nrm2(handles_[si], dim, r_i, 1, &ri_norm));

            if (ri_norm > tol_dav * 0.01) {
                // Normalize
                RealType inv_ri = 1.0 / ri_norm;
                ROCBLAS_CHECK(Traits::scal_real(handles_[si], dim, &inv_ri, r_i, 1));
                n_new++;
            }
        }

        if (n_new == 0) {
            // No significant residuals, converged
            ROCBLAS_CHECK(Traits::nrm2(handles_[si], dim, d_theta, 1, &norm));
            inv_norm = 1.0 / norm;
            ROCBLAS_CHECK(Traits::scal_real(handles_[si], dim, &inv_norm, d_theta, 1));
            return energy;
        }

        // Orthogonalize new vectors against V
        // W = d_dav_work[:, :n_new], V has k columns
        Scalar* W = ws.d_dav_work;

        // overlap = V^H @ W → (k, n_new)
        ROCBLAS_CHECK(Traits::gemm(handles_[si],
            Traits::op_h, rocblas_operation_none,
            k, n_new, dim, &one, V, dim, W, dim, &zero_val, ws.d_dav_work2, k));

        // W -= V @ overlap
        ROCBLAS_CHECK(Traits::gemm(handles_[si],
            rocblas_operation_none, rocblas_operation_none,
            dim, n_new, k, &neg_one, V, dim, ws.d_dav_work2, k, &one, W, dim));

        // Orthogonalize new vectors among themselves (CGS within the block)
        int n_good = 0;
        for (int i = 0; i < n_new; i++) {
            Scalar* wi = W + (size_t)i * dim;

            // Project out previously accepted new vectors
            for (int j = 0; j < n_good; j++) {
                Scalar* wj = W + (size_t)j * dim;
                Scalar overlap_val;
                ROCBLAS_CHECK(Traits::dot(handles_[si], dim, wj, 1, wi, 1, &overlap_val));
                Scalar neg_ov = Traits::neg(overlap_val);
                ROCBLAS_CHECK(Traits::axpy(handles_[si], dim, &neg_ov, wj, 1, wi, 1));
            }

            RealType wi_norm;
            HIP_CHECK(hipStreamSynchronize(streams_[si]));
            ROCBLAS_CHECK(Traits::nrm2(handles_[si], dim, wi, 1, &wi_norm));

            if (wi_norm > 1e-14) {
                RealType inv_wi = 1.0 / wi_norm;
                ROCBLAS_CHECK(Traits::scal_real(handles_[si], dim, &inv_wi, wi, 1));
                // Move to position n_good if different
                if (n_good != i) {
                    HIP_CHECK(hipMemcpyAsync(W + (size_t)n_good * dim, wi, dim * sizeof(Scalar),
                                              hipMemcpyDeviceToDevice, streams_[si]));
                }
                n_good++;
            }
        }

        if (n_good == 0) {
            ROCBLAS_CHECK(Traits::nrm2(handles_[si], dim, d_theta, 1, &norm));
            inv_norm = 1.0 / norm;
            ROCBLAS_CHECK(Traits::scal_real(handles_[si], dim, &inv_norm, d_theta, 1));
            return energy;
        }

        // Check if subspace would be too large → restart
        if (k + n_good > max_sub) {
            // Restart: keep best b Ritz vectors
            int keep = std::min(b, k);

            // X_keep = V @ eigvecs[:, :keep] → (dim, keep)
            HIP_CHECK(hipMemcpy(ws.d_dav_work2, ws.h_dav_eigvecs.data(),
                                k * k * sizeof(Scalar), hipMemcpyHostToDevice));
            ROCBLAS_CHECK(Traits::gemm(handles_[si],
                rocblas_operation_none, rocblas_operation_none,
                dim, keep, k, &one, V, dim, ws.d_dav_work2, k,
                &zero_val, ws.d_dav_work, dim));

            // Copy X_keep to V[:, :keep]
            HIP_CHECK(hipMemcpyAsync(V, ws.d_dav_work, (size_t)dim * keep * sizeof(Scalar),
                                      hipMemcpyDeviceToDevice, streams_[si]));

            // Re-orthogonalize V columns (MGS on GPU)
            for (int i = 0; i < keep; i++) {
                for (int j = 0; j < i; j++) {
                    Scalar ov;
                    ROCBLAS_CHECK(Traits::dot(handles_[si], dim,
                        V + (size_t)j * dim, 1, V + (size_t)i * dim, 1, &ov));
                    Scalar neg_ov = Traits::neg(ov);
                    ROCBLAS_CHECK(Traits::axpy(handles_[si], dim, &neg_ov,
                        V + (size_t)j * dim, 1, V + (size_t)i * dim, 1));
                }
                RealType vi_norm;
                HIP_CHECK(hipStreamSynchronize(streams_[si]));
                ROCBLAS_CHECK(Traits::nrm2(handles_[si], dim, V + (size_t)i * dim, 1, &vi_norm));
                RealType inv_vi = 1.0 / vi_norm;
                ROCBLAS_CHECK(Traits::scal_real(handles_[si], dim, &inv_vi, V + (size_t)i * dim, 1));
            }

            // Recompute AV for kept vectors
            for (int j = 0; j < keep; j++) {
                apply_heff_two_site(site, V + (size_t)j * dim, AV + (size_t)j * dim, si);
            }

            k = keep;
            continue;
        }

        // Expand: append new vectors and compute their H-images
        for (int j = 0; j < n_good; j++) {
            HIP_CHECK(hipMemcpyAsync(V + (size_t)(k + j) * dim, W + (size_t)j * dim,
                                      dim * sizeof(Scalar), hipMemcpyDeviceToDevice, streams_[si]));
            apply_heff_two_site(site, V + (size_t)(k + j) * dim, AV + (size_t)(k + j) * dim, si);
        }
        k += n_good;
    }

    // Didn't converge — use best result
    // d_theta should already have the best vector from the last Ritz computation
    ROCBLAS_CHECK(Traits::nrm2(handles_[si], dim, d_theta, 1, &norm));
    inv_norm = 1.0 / norm;
    ROCBLAS_CHECK(Traits::scal_real(handles_[si], dim, &inv_norm, d_theta, 1));
    return best_energy;
}

// ============================================================================
// Lanczos eigensolver (fallback)
// ============================================================================

template<typename Scalar>
double PDMRGGPUOpt<Scalar>::lanczos_eigensolver(int site, Scalar* d_theta, int theta_size, int si) {
    int n = theta_size;
    int max_iter = std::min(max_lanczos_iter_, n);
    double tol_lanczos = 1e-12;
    double tol_eig_conv = 1e-12;
    auto& ws = workspaces_[si];

    Scalar* d_lanczos_v = ws.d_lanczos_v;

    std::vector<double> h_alpha(max_iter);
    std::vector<double> h_beta(max_iter);

    double norm;
    ROCBLAS_CHECK(rocblas_set_pointer_mode(handles_[si], rocblas_pointer_mode_device));
    ROCBLAS_CHECK(Traits::nrm2(handles_[si], n, d_theta, 1, ws.d_nrm2_result));
    ROCBLAS_CHECK(rocblas_set_pointer_mode(handles_[si], rocblas_pointer_mode_host));

    HIP_CHECK(hipMemcpy(&norm, ws.d_nrm2_result, sizeof(double), hipMemcpyDeviceToHost));
    if (norm < 1e-14) {
        std::vector<Scalar> h_init(n);
        srand(42 + site);
        for (int i = 0; i < n; i++) h_init[i] = Traits::random_val();
        HIP_CHECK(hipMemcpy(d_theta, h_init.data(), n * sizeof(Scalar), hipMemcpyHostToDevice));
        ROCBLAS_CHECK(Traits::nrm2(handles_[si], n, d_theta, 1, &norm));
    }

    ROCBLAS_CHECK(rocblas_set_pointer_mode(handles_[si], rocblas_pointer_mode_device));
    hipLaunchKernelGGL(inv_real_kernel, dim3(1), dim3(1), 0, streams_[si],
                       ws.d_nrm2_result, ws.d_inv_nrm);
    ROCBLAS_CHECK(Traits::scal_real(handles_[si], n, ws.d_inv_nrm, d_theta, 1));
    ROCBLAS_CHECK(rocblas_set_pointer_mode(handles_[si], rocblas_pointer_mode_host));
    HIP_CHECK(hipMemcpyAsync(d_lanczos_v, d_theta, n * sizeof(Scalar), hipMemcpyDeviceToDevice, streams_[si]));

    double prev_energy = 1e30;
    int iter;
    int last_synced_iter = -1;

    for (iter = 0; iter < max_iter; iter++) {
        Scalar* d_vi = d_lanczos_v + (size_t)iter * n;

        if (lanczos_use_1site_)
            apply_heff_single_site(site, d_vi, ws.d_heff_result, si);
        else
            apply_heff_two_site(site, d_vi, ws.d_heff_result, si);

        ROCBLAS_CHECK(rocblas_set_pointer_mode(handles_[si], rocblas_pointer_mode_device));

        ROCBLAS_CHECK(Traits::dot(handles_[si], n, d_vi, 1, ws.d_heff_result, 1, ws.d_dot_result));

        hipLaunchKernelGGL(lanczos_process_alpha_kernel<Scalar>, dim3(1), dim3(1), 0, streams_[si],
                           ws.d_dot_result, ws.d_neg_alpha, ws.d_alpha_dev, iter);

        ROCBLAS_CHECK(Traits::axpy(handles_[si], n, ws.d_neg_alpha, d_vi, 1, ws.d_heff_result, 1));

        if (iter > 0) {
            ROCBLAS_CHECK(Traits::axpy(handles_[si], n,
                ws.d_neg_beta_scalars + (iter - 1),
                d_lanczos_v + (size_t)(iter - 1) * n, 1,
                ws.d_heff_result, 1));
        }

        if (iter > 0) {
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

        ROCBLAS_CHECK(Traits::nrm2(handles_[si], n, ws.d_heff_result, 1, ws.d_nrm2_result));

        hipLaunchKernelGGL(lanczos_process_beta_kernel<Scalar>, dim3(1), dim3(1), 0, streams_[si],
                           ws.d_nrm2_result, ws.d_inv_nrm, ws.d_beta_dev, ws.d_neg_beta_scalars, iter);

        if (iter + 1 < max_iter) {
            Scalar* d_vip1 = d_lanczos_v + (size_t)(iter + 1) * n;
            HIP_CHECK(hipMemcpyAsync(d_vip1, ws.d_heff_result, n * sizeof(Scalar), hipMemcpyDeviceToDevice, streams_[si]));
            ROCBLAS_CHECK(Traits::scal_real(handles_[si], n, ws.d_inv_nrm, d_vip1, 1));
        }

        ROCBLAS_CHECK(rocblas_set_pointer_mode(handles_[si], rocblas_pointer_mode_host));

        // LANCZOS_FIXED skips the check entirely — no mid-loop host syncs.
        if (!opts_.lanczos_fixed && iter >= 4 && iter % 3 == 0) {
            HIP_CHECK(hipStreamSynchronize(streams_[si]));
            int n_copy = iter + 1;
            HIP_CHECK(hipMemcpy(h_alpha.data(), ws.d_alpha_dev, n_copy * sizeof(double), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(h_beta.data(), ws.d_beta_dev, n_copy * sizeof(double), hipMemcpyDeviceToHost));
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
    HIP_CHECK(hipStreamSynchronize(streams_[si]));

    if (last_synced_iter < niter - 1) {
        HIP_CHECK(hipMemcpy(h_alpha.data(), ws.d_alpha_dev, niter * sizeof(double), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_beta.data(), ws.d_beta_dev, niter * sizeof(double), hipMemcpyDeviceToHost));
    }

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
        throw std::runtime_error("LAPACK dstev failed");
    }

    double energy = h_D[0];

    std::vector<Scalar> h_ritz_scalar(niter);
    for (int i = 0; i < niter; i++) {
        h_ritz_scalar[i] = Traits::make_scalar(h_Z[i]);
    }
    HIP_CHECK(hipMemcpyAsync(ws.d_ritz_coeffs, h_ritz_scalar.data(),
              niter * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));

    ROCBLAS_CHECK(rocblas_set_pointer_mode(handles_[si], rocblas_pointer_mode_device));
    ROCBLAS_CHECK(Traits::gemv(
        handles_[si], rocblas_operation_none,
        n, niter, ws.d_const_one,
        d_lanczos_v, n,
        ws.d_ritz_coeffs, 1,
        ws.d_const_zero, d_theta, 1
    ));

    ROCBLAS_CHECK(Traits::nrm2(handles_[si], n, d_theta, 1, ws.d_nrm2_result));
    hipLaunchKernelGGL(inv_real_kernel, dim3(1), dim3(1), 0, streams_[si],
                       ws.d_nrm2_result, ws.d_inv_nrm);
    ROCBLAS_CHECK(Traits::scal_real(handles_[si], n, ws.d_inv_nrm, d_theta, 1));
    ROCBLAS_CHECK(rocblas_set_pointer_mode(handles_[si], rocblas_pointer_mode_host));

    return energy;
}

// ============================================================================
// Chebyshev-Filtered Subspace Iteration Eigensolver
// For ground state: Chebyshev polynomial filter + Rayleigh quotient
// No orthogonalization during filtering — just apply_heff + BLAS-1
// ============================================================================

template<typename Scalar>
double PDMRGGPUOpt<Scalar>::chebyshev_eigensolver(int site, Scalar* d_theta, int theta_size, int si) {
    int n = theta_size;
    auto& ws = workspaces_[si];
    double tol_eig = 1e-12;
    int cheb_degree = 15;       // Chebyshev polynomial degree per outer iter
    int max_outer = 20;         // max outer iterations
    int bounds_lanczos = 10;    // Lanczos steps for spectral bounds

    // === Step 1: Estimate spectral bounds via truncated Lanczos ===
    // Run a few Lanczos steps to get rough [λ_min, λ_max]
    Scalar* d_lanczos_v = ws.d_lanczos_v;
    std::vector<double> h_alpha(bounds_lanczos);
    std::vector<double> h_beta(bounds_lanczos);

    // Normalize initial vector
    double init_norm;
    ROCBLAS_CHECK(Traits::nrm2(handles_[si], n, d_theta, 1, &init_norm));
    if (init_norm < 1e-14) {
        std::vector<Scalar> h_init(n);
        srand(42 + site);
        for (int i = 0; i < n; i++) h_init[i] = Traits::random_val();
        HIP_CHECK(hipMemcpy(d_theta, h_init.data(), n * sizeof(Scalar), hipMemcpyHostToDevice));
        ROCBLAS_CHECK(Traits::nrm2(handles_[si], n, d_theta, 1, &init_norm));
    }
    {
        Scalar inv = Traits::make_scalar(1.0 / init_norm);
        ROCBLAS_CHECK(Traits::scal(handles_[si], n, &inv, d_theta, 1));
    }
    HIP_CHECK(hipMemcpyAsync(d_lanczos_v, d_theta, n * sizeof(Scalar),
                              hipMemcpyDeviceToDevice, streams_[si]));

    // Short Lanczos for bounds estimation
    for (int iter = 0; iter < bounds_lanczos; iter++) {
        Scalar* d_vi = d_lanczos_v + (size_t)iter * n;

        if (lanczos_use_1site_)
            apply_heff_single_site(site, d_vi, ws.d_heff_result, si);
        else
            apply_heff_two_site(site, d_vi, ws.d_heff_result, si);

        // alpha = <v_i, H v_i>
        Scalar h_dot;
        ROCBLAS_CHECK(Traits::dot(handles_[si], n, d_vi, 1, ws.d_heff_result, 1, &h_dot));
        h_alpha[iter] = Traits::real_part(h_dot);

        // w -= alpha * v_i
        Scalar neg_alpha = Traits::make_scalar(-h_alpha[iter]);
        ROCBLAS_CHECK(Traits::axpy(handles_[si], n, &neg_alpha, d_vi, 1, ws.d_heff_result, 1));

        if (iter > 0) {
            Scalar neg_beta = Traits::make_scalar(-h_beta[iter - 1]);
            ROCBLAS_CHECK(Traits::axpy(handles_[si], n, &neg_beta,
                d_lanczos_v + (size_t)(iter - 1) * n, 1, ws.d_heff_result, 1));
        }

        // beta = ||w||
        ROCBLAS_CHECK(Traits::nrm2(handles_[si], n, ws.d_heff_result, 1, &h_beta[iter]));

        if (iter + 1 < bounds_lanczos && h_beta[iter] > 1e-14) {
            Scalar* d_vip1 = d_lanczos_v + (size_t)(iter + 1) * n;
            HIP_CHECK(hipMemcpyAsync(d_vip1, ws.d_heff_result, n * sizeof(Scalar),
                                      hipMemcpyDeviceToDevice, streams_[si]));
            Scalar inv_beta = Traits::make_scalar(1.0 / h_beta[iter]);
            ROCBLAS_CHECK(Traits::scal(handles_[si], n, &inv_beta, d_vip1, 1));
        }
    }

    HIP_CHECK(hipStreamSynchronize(streams_[si]));

    // Diagonalize tridiagonal matrix for spectral bounds
    {
        std::vector<double> D_chk(bounds_lanczos), E_chk(bounds_lanczos);
        std::copy(h_alpha.begin(), h_alpha.end(), D_chk.begin());
        for (int i = 0; i < bounds_lanczos - 1; i++) E_chk[i] = h_beta[i];
        E_chk[bounds_lanczos - 1] = 0.0;
        const char jobz = 'N';
        const int n_chk = bounds_lanczos;
        std::vector<double> work(1);
        int info = 0;
        dstev_(&jobz, &n_chk, D_chk.data(), E_chk.data(), nullptr, &n_chk, work.data(), &info);
        if (info != 0) {
            // Fallback to Lanczos if bounds estimation fails
            return lanczos_eigensolver(site, d_theta, theta_size, si);
        }
        // D_chk is now sorted eigenvalues
        double lambda_min = D_chk[0] - h_beta[bounds_lanczos - 1] - 0.1;  // conservative lower bound
        double lambda_max = D_chk[bounds_lanczos - 1];

        // === Step 2: Chebyshev-filtered iteration ===
        // Unwanted spectrum: [a, b] where a is slightly above ground state, b = λ_max
        // We use the 2nd eigenvalue estimate as 'a' (cutoff for unwanted)
        double lambda_1_est = D_chk[0];  // ground state estimate
        double gap_est = (bounds_lanczos > 1) ? (D_chk[1] - D_chk[0]) : 1.0;
        double a_unwanted = lambda_1_est + gap_est * 0.5;  // midpoint of gap
        double b_unwanted = lambda_max + 0.1;  // upper bound with margin

        // Map [a_unwanted, b_unwanted] → [-1, 1]
        double center = (b_unwanted + a_unwanted) / 2.0;
        double half_width = (b_unwanted - a_unwanted) / 2.0;
        if (half_width < 1e-10) half_width = 1.0;  // safety

        // Use d_lanczos_v slots for Chebyshev recurrence (apply_heff clobbers T1/T2/heff_result)
        // d_theta = current iterate, lanczos_v[0] = previous, lanczos_v[1] = next
        Scalar* d_yk = d_theta;
        Scalar* d_yk_prev = ws.d_lanczos_v;                     // slot 0
        Scalar* d_yk_next = ws.d_lanczos_v + (size_t)n;         // slot 1

        double energy = lambda_1_est;
        double prev_energy = 1e30;
        double scale = 2.0 / half_width;
        double shift = -2.0 * center / half_width;

        for (int outer = 0; outer < max_outer; outer++) {
            // Initialize: y_{-1} = d_theta (already normalized from init or previous outer)
            // Copy theta → d_yk_prev as T_0 = I (y_{-1})
            HIP_CHECK(hipMemcpyAsync(d_yk_prev, d_yk, n * sizeof(Scalar),
                                      hipMemcpyDeviceToDevice, streams_[si]));

            // y_0 = scale * H*x + shift * x  (T_1(φ(H))x = φ(H)x)
            if (lanczos_use_1site_)
                apply_heff_single_site(site, d_yk, ws.d_heff_result, si);
            else
                apply_heff_two_site(site, d_yk, ws.d_heff_result, si);

            // d_yk_next = scale * H*d_yk + shift * d_yk
            HIP_CHECK(hipMemcpyAsync(d_yk_next, ws.d_heff_result, n * sizeof(Scalar),
                                      hipMemcpyDeviceToDevice, streams_[si]));
            {
                Scalar s = Traits::make_scalar(scale);
                ROCBLAS_CHECK(Traits::scal(handles_[si], n, &s, d_yk_next, 1));
                Scalar sh = Traits::make_scalar(shift);
                ROCBLAS_CHECK(Traits::axpy(handles_[si], n, &sh, d_yk, 1, d_yk_next, 1));
            }

            // Swap: prev=theta(T_0), current=yk_next(T_1)
            // For the recurrence, we need: d_yk_prev = old d_yk, d_yk = d_yk_next
            // d_yk_prev already has old d_yk from the copy above
            // Now copy d_yk_next → d_yk
            HIP_CHECK(hipMemcpyAsync(d_yk, d_yk_next, n * sizeof(Scalar),
                                      hipMemcpyDeviceToDevice, streams_[si]));

            // Three-term recurrence: T_{k+1}(φ) = 2φ T_k - T_{k-1}
            for (int k = 1; k < cheb_degree; k++) {
                // H * d_yk → ws.d_heff_result
                if (lanczos_use_1site_)
                    apply_heff_single_site(site, d_yk, ws.d_heff_result, si);
                else
                    apply_heff_two_site(site, d_yk, ws.d_heff_result, si);

                // d_yk_next = 2*scale * H*d_yk + 2*shift * d_yk - d_yk_prev
                HIP_CHECK(hipMemcpyAsync(d_yk_next, ws.d_heff_result, n * sizeof(Scalar),
                                          hipMemcpyDeviceToDevice, streams_[si]));
                {
                    Scalar s2 = Traits::make_scalar(2.0 * scale);
                    ROCBLAS_CHECK(Traits::scal(handles_[si], n, &s2, d_yk_next, 1));
                    Scalar sh2 = Traits::make_scalar(2.0 * shift);
                    ROCBLAS_CHECK(Traits::axpy(handles_[si], n, &sh2, d_yk, 1, d_yk_next, 1));
                    Scalar neg_one = Traits::neg(Traits::one());
                    ROCBLAS_CHECK(Traits::axpy(handles_[si], n, &neg_one, d_yk_prev, 1, d_yk_next, 1));
                }

                // Rotate: prev ← current, current ← next
                // Use pointer swap (d_yk_prev = d_yk, d_yk = d_yk_next) via memcpy
                HIP_CHECK(hipMemcpyAsync(d_yk_prev, d_yk, n * sizeof(Scalar),
                                          hipMemcpyDeviceToDevice, streams_[si]));
                HIP_CHECK(hipMemcpyAsync(d_yk, d_yk_next, n * sizeof(Scalar),
                                          hipMemcpyDeviceToDevice, streams_[si]));
            }

            // Normalize filtered vector
            double nrm;
            ROCBLAS_CHECK(Traits::nrm2(handles_[si], n, d_yk, 1, &nrm));
            if (nrm > 1e-14) {
                Scalar inv_nrm = Traits::make_scalar(1.0 / nrm);
                ROCBLAS_CHECK(Traits::scal(handles_[si], n, &inv_nrm, d_yk, 1));
            }

            // Rayleigh quotient: E = <y|H|y>
            if (lanczos_use_1site_)
                apply_heff_single_site(site, d_yk, ws.d_heff_result, si);
            else
                apply_heff_two_site(site, d_yk, ws.d_heff_result, si);

            Scalar h_rq;
            ROCBLAS_CHECK(Traits::dot(handles_[si], n, d_yk, 1, ws.d_heff_result, 1, &h_rq));
            energy = Traits::real_part(h_rq);

            // Check convergence
            if (std::abs(energy - prev_energy) < tol_eig && outer > 0) {
                break;
            }
            prev_energy = energy;

            // Update cutoff for next iteration (tighten around ground state)
            if (outer == 0) {
                a_unwanted = energy + gap_est * 0.3;
                center = (b_unwanted + a_unwanted) / 2.0;
                half_width = (b_unwanted - a_unwanted) / 2.0;
                if (half_width < 1e-10) half_width = 1.0;
                scale = 2.0 / half_width;
                shift = -2.0 * center / half_width;
            }
        }

        // Ensure d_theta has the final eigenvector (it's d_yk which IS d_theta)
        // d_yk == d_theta by construction, so no copy needed

        return energy;
    }
}

// ============================================================================
// Bond optimization
// ============================================================================

template<typename Scalar>
double PDMRGGPUOpt<Scalar>::optimize_bond(int site, char direction, int si) {
    int cL = chi_L(site);
    int cR = chi_R(site + 1);
    int theta_size = cL * d_ * d_ * cR;
    auto& ws = workspaces_[si];

    form_theta_two_site(site, si);

    // Eigensolver: Chebyshev (BLAS-3 friendly), Davidson, or Lanczos (default)
    double energy;
    if (use_chebyshev_) {
        energy = chebyshev_eigensolver(site, ws.d_theta, theta_size, si);
    } else if (use_davidson_) {
        energy = block_davidson_eigensolver(site, ws.d_theta, theta_size, si);
    } else {
        energy = lanczos_eigensolver(site, ws.d_theta, theta_size, si);
    }

    // Bond split
    if (use_ns_split_) {
        ns_split(site, ws.d_theta, direction, si);
    } else if (use_rsvd_) {
        rsvd_split(site, ws.d_theta, direction, si);
    } else {
        svd_split(site, ws.d_theta, direction, si);
    }

    return energy;
}

// ============================================================================
// Canonicalization via Newton-Schulz
// ============================================================================

template<typename Scalar>
void PDMRGGPUOpt<Scalar>::left_canonize_site(int site, int si) {
    // MPS[site] has shape (chi_L, d, chi_R) stored as (chi_L*d, chi_R) in column-major
    int cL = chi_L(site);
    int cR = chi_R(site);
    int m = cL * d_;  // rows
    int n = cR;       // cols

    if (m < n) return;  // can't left-canonize if wide (boundary case)

    auto& ws = workspaces_[si];
    Scalar one = Traits::one(), zero_val = Traits::zero();

    // Newton-Schulz: MPS[site] = U @ P
    newton_schulz_left(d_mps_tensors_[site], m, n, ws.d_ns_U, ws.d_ns_P, si, 1e-10);
    HIP_CHECK(hipStreamSynchronize(streams_[si]));

    // MPS[site] = U (left-isometric)
    HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site], ws.d_ns_U, m * n * sizeof(Scalar),
                              hipMemcpyDeviceToDevice, streams_[si]));

    // Absorb P into MPS[site+1]: MPS[site+1] = P @ MPS[site+1]
    if (site + 1 < L_) {
        int cR_next = chi_R(site + 1);
        // P is (n, n) = (cR, cR), MPS[site+1] is (cR, d*cR_next)
        // new_MPS[site+1] = P @ MPS[site+1]: (cR, cR) × (cR, d*cR_next) → (cR, d*cR_next)
        ROCBLAS_CHECK(Traits::gemm(handles_[si],
            rocblas_operation_none, rocblas_operation_none,
            n, d_ * cR_next, n,
            &one, ws.d_ns_P, n,
            d_mps_tensors_[site + 1], n,
            &zero_val, ws.d_heff_result, n));

        HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site + 1], ws.d_heff_result,
                    n * d_ * cR_next * sizeof(Scalar), hipMemcpyDeviceToDevice, streams_[si]));
    }
}

template<typename Scalar>
void PDMRGGPUOpt<Scalar>::right_canonize_site(int site, int si) {
    // MPS[site] has shape (chi_L, d, chi_R) stored as (chi_L, d*chi_R) in column-major
    int cL = chi_L(site);
    int cR = chi_R(site);
    int m = cL;        // rows
    int n = d_ * cR;   // cols

    if (m >= n) return;  // can't right-canonize if tall (boundary case)

    auto& ws = workspaces_[si];
    Scalar one = Traits::one(), zero_val = Traits::zero();

    // Right Newton-Schulz: MPS[site] = L @ Q where Q Q^H = I_m
    // d_ns_P will hold L (m × m), d_ns_U will hold Q (m × n)
    newton_schulz_right(d_mps_tensors_[site], m, n, ws.d_ns_P, ws.d_ns_U, si, 1e-10);
    HIP_CHECK(hipStreamSynchronize(streams_[si]));

    // MPS[site] = Q (right-isometric)
    HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site], ws.d_ns_U, m * n * sizeof(Scalar),
                              hipMemcpyDeviceToDevice, streams_[si]));

    // Absorb L into MPS[site-1]: MPS[site-1] = MPS[site-1] @ L
    if (site > 0) {
        int cL_prev = chi_L(site - 1);
        // MPS[site-1] is (cL_prev*d, m) column-major, L is (m, m)
        // new_MPS[site-1] = MPS[site-1] @ L
        ROCBLAS_CHECK(Traits::gemm(handles_[si],
            rocblas_operation_none, rocblas_operation_none,
            cL_prev * d_, m, m,
            &one, d_mps_tensors_[site - 1], cL_prev * d_,
            ws.d_ns_P, m,
            &zero_val, ws.d_heff_result, cL_prev * d_));

        HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site - 1], ws.d_heff_result,
                    cL_prev * d_ * m * sizeof(Scalar), hipMemcpyDeviceToDevice, streams_[si]));
    }
}

template<typename Scalar>
void PDMRGGPUOpt<Scalar>::canonize_segment_right(int seg_idx) {
    int first = seg_first_[seg_idx];
    int last = seg_last_[seg_idx];
    int si = seg_idx;

    // Right-canonize from last site to first+1
    for (int j = last; j > first; j--) {
        right_canonize_site(j, si);
    }
    HIP_CHECK(hipStreamSynchronize(streams_[si]));
}

template<typename Scalar>
void PDMRGGPUOpt<Scalar>::canonize_segment_left(int seg_idx) {
    int first = seg_first_[seg_idx];
    int last = seg_last_[seg_idx];
    int si = seg_idx;

    // Left-canonize from first site to last-1
    for (int j = first; j < last; j++) {
        left_canonize_site(j, si);
    }
    HIP_CHECK(hipStreamSynchronize(streams_[si]));
}

// ============================================================================
// Single-site apply_heff: H|theta> for theta of size (cL*d, cR)
// Same 3-step GEMM pattern as two-site but with d instead of d²
// ============================================================================

template<typename Scalar>
void PDMRGGPUOpt<Scalar>::apply_heff_single_site(int site, const Scalar* d_theta_in,
                                                  Scalar* d_result, int si) {
    int cL = chi_L(site);
    int cR = chi_R(site);
    int D = D_mpo_, d = d_;
    Scalar one = Traits::one(), zero_val = Traits::zero();
    auto& ws = workspaces_[si];

    Scalar* L_env = d_L_envs_[site];
    Scalar* R_env = d_R_envs_[site + 1];
    Scalar* W_mat = d_W_left_[site];
    Scalar* V = ws.d_T1;
    Scalar* U = ws.d_T2;

    // Step 1: V_ws[a',b] = L_w^T[a',a] * theta_s[a,b]  (D*d batched GEMMs)
    {
        int batch_count = D * d;
        Scalar* h_A[256], *h_B[256], *h_C[256];
        for (int w = 0; w < D; w++)
            for (int s = 0; s < d; s++) {
                int ws_idx = w * d + s;
                h_A[ws_idx] = L_env + w * cL;
                h_B[ws_idx] = const_cast<Scalar*>(d_theta_in) + s * cL;
                h_C[ws_idx] = V + ws_idx * cL * cR;
            }
        HIP_CHECK(hipMemcpyAsync(ws.d_batch_A, h_A, batch_count * sizeof(Scalar*),
                                  hipMemcpyHostToDevice, streams_[si]));
        HIP_CHECK(hipMemcpyAsync(ws.d_batch_B, h_B, batch_count * sizeof(Scalar*),
                                  hipMemcpyHostToDevice, streams_[si]));
        HIP_CHECK(hipMemcpyAsync(ws.d_batch_C, h_C, batch_count * sizeof(Scalar*),
                                  hipMemcpyHostToDevice, streams_[si]));
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

    // Step 3: result_s'[a',b'] = sum_w' U_{w'd+s'}[a',b] * R_w'[b,b']
    // Strided batched over sp when d<=2 and chi>=16 (eliminates host pointer DMA)
    if (cL >= 16 && cR >= 16 && d <= 2) {
        for (int wp = 0; wp < D; wp++) {
            Scalar beta = (wp == 0) ? Traits::zero() : Traits::one();
            ROCBLAS_CHECK(Traits::gemm_strided_batched(handles_[si],
                rocblas_operation_none, rocblas_operation_none,
                cL, cR, cR, &one,
                U + (size_t)wp * d * cL * cR, cL, (rocblas_stride)(cL * cR),
                R_env + wp * cR, cR * D, (rocblas_stride)0,
                &beta, d_result, cL * d, (rocblas_stride)cL, d));
        }
    } else {
        Scalar* h_A3[256], *h_B3[256], *h_C3[256];
        for (int wp = 0; wp < D; wp++) {
            Scalar beta = (wp == 0) ? Traits::zero() : Traits::one();
            for (int sp = 0; sp < d; sp++) {
                h_A3[sp] = U + (wp * d + sp) * cL * cR;
                h_B3[sp] = R_env + wp * cR;
                h_C3[sp] = d_result + sp * cL;
            }
            HIP_CHECK(hipMemcpyAsync(ws.d_batch_A, h_A3, d*sizeof(Scalar*), hipMemcpyHostToDevice, streams_[si]));
            HIP_CHECK(hipMemcpyAsync(ws.d_batch_B, h_B3, d*sizeof(Scalar*), hipMemcpyHostToDevice, streams_[si]));
            HIP_CHECK(hipMemcpyAsync(ws.d_batch_C, h_C3, d*sizeof(Scalar*), hipMemcpyHostToDevice, streams_[si]));
            ROCBLAS_CHECK(Traits::gemm_batched(handles_[si],
                rocblas_operation_none, rocblas_operation_none,
                cL, cR, cR,
                &one,
                (const Scalar**)ws.d_batch_A, cL,
                (const Scalar**)ws.d_batch_B, cR * D,
                &beta,
                ws.d_batch_C, cL * d,
                d));
        }
    }
}

// ============================================================================
// Single-site SVD split: decompose theta and shift canonical center
// Direction 'R': theta(cL*d, cR) → U=MPS[site], S*Vh absorbed into MPS[site+1]
// Direction 'L': theta(cL, d*cR) → Vh=MPS[site], U*S absorbed into MPS[site-1]
// ============================================================================

template<typename Scalar>
void PDMRGGPUOpt<Scalar>::svd_split_single_site(int site, Scalar* d_theta, char direction, int si) {
    int cL = chi_L(site);
    int cR = chi_R(site);
    auto& ws = workspaces_[si];

    int m, n_svd;
    if (direction == 'R') { m = cL * d_; n_svd = cR; }
    else                  { m = cL;      n_svd = d_ * cR; }
    int full_k = std::min(m, n_svd);
    int k = std::min(full_k, chi_max_user_);

    Scalar* h_U_data = nullptr;
    RealType* h_S_data = nullptr;
    Scalar* h_Vh_data = nullptr;

    // CPU SVD path
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
        Traits::rocsolver_gesvd(handles_[si],
            rocblas_svect_singular, rocblas_svect_singular,
            m, n_svd,
            ws.d_svd_A, m,
            ws.d_svd_S,
            ws.d_svd_U, m,
            ws.d_svd_Vh, full_k,
            ws.d_svd_E,
            rocblas_outofplace,
            ws.d_svd_info);
    }

    // Truncation
    int new_k;
    if (opts_.device_k) {
        new_k = k;
    } else if (!use_cpu_svd_) {
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
            allocate_mps_tensor(site, cL, new_chi_R);
            HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site], h_U_data,
                        m * new_k * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));

            for (int j = 0; j < n_svd; j++)
                for (int i = 0; i < new_k; i++)
                    ws.h_svd_tmp[i + j * new_k] = Traits::scale_by_real(h_S_data[i], h_Vh_data[i + j * full_k]);

            HIP_CHECK(hipMemcpyAsync(ws.d_svd_work, ws.h_svd_tmp.data(),
                        new_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));
        } else {
            allocate_mps_tensor(site, cL, new_chi_R);
            HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site], ws.d_svd_U,
                        (size_t)m * new_k * sizeof(Scalar), hipMemcpyDeviceToDevice, streams_[si]));
            scale_rows_by_real(ws.d_svd_Vh, full_k, ws.d_svd_S,
                               ws.d_svd_work, new_k, new_k, n_svd, streams_[si]);
        }

        // Absorb S*Vh into MPS[site+1]
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

            for (int j = 0; j < new_k; j++)
                for (int i = 0; i < m; i++)
                    ws.h_svd_tmp[i + j * m] = Traits::scale_by_real(h_S_data[j], h_U_data[i + j * m]);

            HIP_CHECK(hipMemcpyAsync(ws.d_svd_work, ws.h_svd_tmp.data(),
                        m * new_k * sizeof(Scalar), hipMemcpyHostToDevice, streams_[si]));
        } else {
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
            scale_columns_by_real(ws.d_svd_U, m, ws.d_svd_S,
                                  ws.d_svd_work, m, m, new_k, streams_[si]);
        }

        // Absorb U*S into MPS[site-1]
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
double PDMRGGPUOpt<Scalar>::optimize_site_single(int site, char direction, int si) {
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
// Full-chain single-site sweep methods (for warmup and polish)
// ============================================================================

template<typename Scalar>
double PDMRGGPUOpt<Scalar>::sweep_LR_full_1site() {
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
double PDMRGGPUOpt<Scalar>::sweep_RL_full_1site() {
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
// Full-chain sweep methods (two-site, for main PDMRG sweeps, stream 0)
// ============================================================================

template<typename Scalar>
double PDMRGGPUOpt<Scalar>::sweep_LR_full() {
    double energy = 0.0;
    for (int site = 0; site < L_ - 1; site++) {
        energy = optimize_bond(site, 'R', 0);
        update_left_env(site, 0);
    }
    return energy;
}

template<typename Scalar>
double PDMRGGPUOpt<Scalar>::sweep_RL_full() {
    double energy = 0.0;
    for (int site = L_ - 2; site >= 0; site--) {
        energy = optimize_bond(site, 'L', 0);
        update_right_env(site + 1, 0);
    }
    return energy;
}

// ============================================================================
// Segment sweep methods (with Newton-Schulz pre-canonicalization)
// ============================================================================

template<typename Scalar>
void PDMRGGPUOpt<Scalar>::segment_sweep_LR(int seg_idx) {
    int first = seg_first_[seg_idx];
    int last = seg_last_[seg_idx];
    int si = seg_idx;

    // No pre-sweep canonicalization needed: the two-site optimize_bond → NS/SVD
    // split already produces left-canonical tensors during the LR sweep, and
    // update_left_env incrementally builds correct environments as we go.
    for (int site = first; site < last; site++) {
        optimize_bond(site, 'R', si);
        update_left_env(site, si);
    }
}

template<typename Scalar>
void PDMRGGPUOpt<Scalar>::segment_sweep_RL(int seg_idx) {
    int first = seg_first_[seg_idx];
    int last = seg_last_[seg_idx];
    int si = seg_idx;

    // No pre-sweep canonicalization needed: the two-site optimize_bond → NS/SVD
    // split already produces right-canonical tensors during the RL sweep, and
    // update_right_env incrementally builds correct environments as we go.
    for (int site = last - 1; site >= first; site--) {
        optimize_bond(site, 'L', si);
        update_right_env(site + 1, si);
    }
}

// ============================================================================
// Cross-segment batched apply_heff: process N segments' matvecs in one set
// of GEMM calls. All segments must have same (cL, cR) dimensions.
// Uses stream 0 for all work.
// ============================================================================

template<typename Scalar>
void PDMRGGPUOpt<Scalar>::batched_apply_heff_two_site(
    const int* sites, const int* seg_indices, int n_batch,
    const Scalar** d_thetas_in, Scalar** d_results) {

    // All segments assumed to have same dims (caller guarantees this)
    int cL = chi_L(sites[0]);
    int cR = chi_R(sites[0] + 1);
    int D = D_mpo_, d = d_;
    int dd = d * d;
    Scalar one = Traits::one(), zero_val = Traits::zero();

    // Step 1: Batched GEMM — L_env^T × theta, batch across all segments
    // batch_count = n_batch * D * dd
    {
        int per_seg_batch = D * dd;
        int total_batch = n_batch * per_seg_batch;

        // Build pointer arrays on host, copy to device
        std::vector<Scalar*> h_A(total_batch), h_B(total_batch), h_C(total_batch);
        for (int b = 0; b < n_batch; b++) {
            int site = sites[b];
            int si = seg_indices[b];
            Scalar* L_env = d_L_envs_[site];
            Scalar* T1 = workspaces_[si].d_T1;
            const Scalar* theta = d_thetas_in[b];

            for (int n = 0; n < D; n++) {
                for (int idx = 0; idx < dd; idx++) {
                    int k = b * per_seg_batch + n * dd + idx;
                    h_A[k] = L_env + (size_t)n * cL;          // L_env slice, stride cL*D
                    h_B[k] = const_cast<Scalar*>(theta) + (size_t)idx * cL;  // theta slice
                    h_C[k] = T1 + (size_t)(n * dd + idx) * cL * cR;         // T1 output
                }
            }
        }

        HIP_CHECK(hipMemcpyAsync(d_xs_batch_A_, h_A.data(), total_batch * sizeof(Scalar*),
                                  hipMemcpyHostToDevice, streams_[0]));
        HIP_CHECK(hipMemcpyAsync(d_xs_batch_B_, h_B.data(), total_batch * sizeof(Scalar*),
                                  hipMemcpyHostToDevice, streams_[0]));
        HIP_CHECK(hipMemcpyAsync(d_xs_batch_C_, h_C.data(), total_batch * sizeof(Scalar*),
                                  hipMemcpyHostToDevice, streams_[0]));

        ROCBLAS_CHECK(Traits::gemm_batched(handles_[0],
            Traits::op_t, rocblas_operation_none,
            cL, cR, cL,
            &one,
            (const Scalar**)d_xs_batch_A_, cL * D,
            (const Scalar**)d_xs_batch_B_, cL * dd,
            &zero_val,
            d_xs_batch_C_, cL,
            total_batch));
    }

    // Step 2: Dense GEMM per segment — T1 × WW (different WW per site)
    for (int b = 0; b < n_batch; b++) {
        int site = sites[b];
        int si = seg_indices[b];
        ROCBLAS_CHECK(Traits::gemm(handles_[0],
            rocblas_operation_none, rocblas_operation_none,
            cL * cR, dd * D, D * dd,
            &one,
            workspaces_[si].d_T1, cL * cR,
            d_WW_[site], D * dd,
            &zero_val,
            workspaces_[si].d_T2, cL * cR));
    }

    // Step 3: Per-segment — T2 × R_env (different R_env per site)
    for (int b = 0; b < n_batch; b++) {
        int site = sites[b];
        int si = seg_indices[b];
        Scalar* T2 = workspaces_[si].d_T2;
        Scalar* R_env = d_R_envs_[site + 2];

        if (cL >= 16 && cR >= 16 && d <= 2) {
            for (int s2p = 0; s2p < d; s2p++) {
                for (int n = 0; n < D; n++) {
                    Scalar beta = (n == 0) ? Traits::zero() : Traits::one();
                    ROCBLAS_CHECK(Traits::gemm_strided_batched(handles_[0],
                        rocblas_operation_none, rocblas_operation_none,
                        cL, cR, cR, &one,
                        T2 + (size_t)(n * dd + s2p) * cL * cR, cL, (rocblas_stride)(d * cL * cR),
                        R_env + (size_t)n * cR, cR * D, (rocblas_stride)0,
                        &beta, d_results[b] + (size_t)s2p * cL * d, cL * dd, (rocblas_stride)cL, d));
                }
            }
        } else {
            for (int s1p = 0; s1p < d; s1p++) {
                for (int s2p = 0; s2p < d; s2p++) {
                    for (int n = 0; n < D; n++) {
                        Scalar beta = (n == 0) ? Traits::zero() : Traits::one();
                        int ws_out = n * dd + s1p * d + s2p;
                        ROCBLAS_CHECK(Traits::gemm(handles_[0],
                            rocblas_operation_none, rocblas_operation_none,
                            cL, cR, cR,
                            &one,
                            T2 + (size_t)ws_out * cL * cR, cL,
                            R_env + (size_t)n * cR, cR * D,
                            &beta,
                            d_results[b] + s1p * cL + (size_t)s2p * cL * d, cL * dd));
                    }
                }
            }
        }
    }
}

// ============================================================================
// Batched Lanczos: lock-step across N segments, batched matvec when dims match
// All segments must have the same theta_size.
// ============================================================================

template<typename Scalar>
double PDMRGGPUOpt<Scalar>::batched_lanczos_eigensolver(
    const int* sites, const int* seg_indices, int n_batch,
    Scalar** d_thetas, int theta_size) {

    int n = theta_size;
    int max_iter = std::min(max_lanczos_iter_, n);
    double tol_lanczos = 1e-12;
    double tol_eig_conv = 1e-12;

    // Per-segment Lanczos state
    struct SegState {
        Scalar* d_lanczos_v;
        std::vector<double> h_alpha, h_beta;
        double prev_energy;
        bool converged;
        int niter;
    };
    std::vector<SegState> states(n_batch);
    for (int b = 0; b < n_batch; b++) {
        int si = seg_indices[b];
        states[b].d_lanczos_v = workspaces_[si].d_lanczos_v;
        states[b].h_alpha.resize(max_iter);
        states[b].h_beta.resize(max_iter);
        states[b].prev_energy = 1e30;
        states[b].converged = false;
        states[b].niter = max_iter;
    }

    // Normalize initial vectors (all on stream 0)
    for (int b = 0; b < n_batch; b++) {
        int si = seg_indices[b];
        auto& ws = workspaces_[si];
        double norm;
        ROCBLAS_CHECK(Traits::nrm2(handles_[0], n, d_thetas[b], 1, &norm));
        if (norm < 1e-14) {
            std::vector<Scalar> h_init(n);
            srand(42 + sites[b]);
            for (int i = 0; i < n; i++) h_init[i] = Traits::random_val();
            HIP_CHECK(hipMemcpy(d_thetas[b], h_init.data(), n * sizeof(Scalar), hipMemcpyHostToDevice));
            ROCBLAS_CHECK(Traits::nrm2(handles_[0], n, d_thetas[b], 1, &norm));
        }
        Scalar inv_norm = Traits::make_scalar(1.0 / norm);
        ROCBLAS_CHECK(Traits::scal(handles_[0], n, &inv_norm, d_thetas[b], 1));
        HIP_CHECK(hipMemcpyAsync(states[b].d_lanczos_v, d_thetas[b],
                                  n * sizeof(Scalar), hipMemcpyDeviceToDevice, streams_[0]));
    }
    HIP_CHECK(hipStreamSynchronize(streams_[0]));

    int global_iter;
    for (global_iter = 0; global_iter < max_iter; global_iter++) {

        // Check if all segments have converged
        bool all_converged = true;
        for (int b = 0; b < n_batch; b++) {
            if (!states[b].converged) { all_converged = false; break; }
        }
        if (all_converged) break;

        // Batched apply_heff: compute H|v_i> for all segments
        std::vector<const Scalar*> v_ptrs(n_batch);
        std::vector<Scalar*> hv_ptrs(n_batch);
        for (int b = 0; b < n_batch; b++) {
            int si = seg_indices[b];
            v_ptrs[b] = states[b].d_lanczos_v + (size_t)global_iter * n;
            hv_ptrs[b] = workspaces_[si].d_heff_result;
        }

        batched_apply_heff_two_site(sites, seg_indices, n_batch,
                                     v_ptrs.data(), hv_ptrs.data());

        // Per-segment Lanczos update (on stream 0 sequentially — these are cheap BLAS-1 ops)
        for (int b = 0; b < n_batch; b++) {
            if (states[b].converged) continue;

            int si = seg_indices[b];
            auto& ws = workspaces_[si];
            auto& st = states[b];
            Scalar* d_vi = st.d_lanczos_v + (size_t)global_iter * n;
            Scalar* d_w = ws.d_heff_result;

            // alpha = <v_i, H v_i>
            Scalar h_dot;
            ROCBLAS_CHECK(Traits::dot(handles_[0], n, d_vi, 1, d_w, 1, &h_dot));
            double alpha_val = Traits::real_part(h_dot);
            st.h_alpha[global_iter] = alpha_val;

            // w -= alpha * v_i
            Scalar neg_alpha = Traits::make_scalar(-alpha_val);
            ROCBLAS_CHECK(Traits::axpy(handles_[0], n, &neg_alpha, d_vi, 1, d_w, 1));

            // w -= beta_{i-1} * v_{i-1}
            if (global_iter > 0) {
                Scalar neg_beta = Traits::make_scalar(-st.h_beta[global_iter - 1]);
                Scalar* d_vim1 = st.d_lanczos_v + (size_t)(global_iter - 1) * n;
                ROCBLAS_CHECK(Traits::axpy(handles_[0], n, &neg_beta, d_vim1, 1, d_w, 1));
            }

            // Full reorthogonalization
            if (global_iter > 0) {
                Scalar h_one = Traits::one(), h_zero = Traits::zero(), h_neg_one = Traits::neg(Traits::one());
                // coeffs = V^H w
                ROCBLAS_CHECK(Traits::gemv(handles_[0], Traits::op_h,
                    n, global_iter + 1, &h_one,
                    st.d_lanczos_v, n,
                    d_w, 1,
                    &h_zero, ws.d_ritz_coeffs, 1));
                // w -= V * coeffs
                ROCBLAS_CHECK(Traits::gemv(handles_[0], rocblas_operation_none,
                    n, global_iter + 1, &h_neg_one,
                    st.d_lanczos_v, n,
                    ws.d_ritz_coeffs, 1,
                    &h_one, d_w, 1));
            } else {
                Scalar h_overlap;
                ROCBLAS_CHECK(Traits::dot(handles_[0], n, st.d_lanczos_v, 1, d_w, 1, &h_overlap));
                Scalar neg_overlap = Traits::neg(h_overlap);
                ROCBLAS_CHECK(Traits::axpy(handles_[0], n, &neg_overlap, st.d_lanczos_v, 1, d_w, 1));
            }

            // beta = ||w||
            double beta_val;
            ROCBLAS_CHECK(Traits::nrm2(handles_[0], n, d_w, 1, &beta_val));
            st.h_beta[global_iter] = beta_val;

            // v_{i+1} = w / beta
            if (global_iter + 1 < max_iter) {
                Scalar* d_vip1 = st.d_lanczos_v + (size_t)(global_iter + 1) * n;
                HIP_CHECK(hipMemcpyAsync(d_vip1, d_w, n * sizeof(Scalar),
                                          hipMemcpyDeviceToDevice, streams_[0]));
                if (beta_val > 1e-14) {
                    Scalar inv_beta = Traits::make_scalar(1.0 / beta_val);
                    ROCBLAS_CHECK(Traits::scal(handles_[0], n, &inv_beta, d_vip1, 1));
                }
            }
        }

        // Convergence check every 3 iterations
        if (global_iter >= 4 && global_iter % 3 == 0) {
            HIP_CHECK(hipStreamSynchronize(streams_[0]));

            for (int b = 0; b < n_batch; b++) {
                if (states[b].converged) continue;
                auto& st = states[b];
                int ncheck = global_iter + 1;

                // Check for small beta (invariant subspace)
                bool early_break = false;
                for (int j = 0; j <= global_iter; j++) {
                    if (st.h_beta[j] < tol_lanczos) {
                        st.niter = j + 1;
                        st.converged = true;
                        early_break = true;
                        break;
                    }
                }
                if (early_break) continue;

                // Tridiagonal eigenvalue check
                std::vector<double> h_D(ncheck), h_E(ncheck);
                std::copy(st.h_alpha.begin(), st.h_alpha.begin() + ncheck, h_D.begin());
                for (int i = 0; i < ncheck - 1; i++) h_E[i] = st.h_beta[i];
                h_E[ncheck - 1] = 0.0;
                const char jobz_n = 'N';
                const int n_chk = ncheck;
                std::vector<double> h_work_chk(1);
                int info_chk = 0;
                dstev_(&jobz_n, &n_chk, h_D.data(), h_E.data(), nullptr, &n_chk, h_work_chk.data(), &info_chk);
                if (info_chk == 0) {
                    double cur_energy = h_D[0];
                    if (std::abs(cur_energy - st.prev_energy) < tol_eig_conv) {
                        st.niter = global_iter + 1;
                        st.converged = true;
                    }
                    st.prev_energy = cur_energy;
                }
            }
        }
    }

    HIP_CHECK(hipStreamSynchronize(streams_[0]));

    // Extract Ritz vectors per segment
    double total_energy = 0.0;
    for (int b = 0; b < n_batch; b++) {
        int si = seg_indices[b];
        auto& ws = workspaces_[si];
        auto& st = states[b];
        int niter = st.converged ? st.niter : global_iter;
        if (niter < 1) niter = 1;

        std::vector<double> h_D(niter), h_E(niter), h_Z(niter * niter);
        std::vector<double> h_work(std::max(1, 2*niter - 2));
        int lapack_info = 0;

        std::copy(st.h_alpha.begin(), st.h_alpha.begin() + niter, h_D.begin());
        for (int i = 0; i < niter - 1; i++) h_E[i] = st.h_beta[i];
        if (niter > 0) h_E[niter - 1] = 0.0;

        const char jobz = 'V';
        const int n_lapack = niter;
        dstev_(&jobz, &n_lapack, h_D.data(), h_E.data(), h_Z.data(), &n_lapack,
               h_work.data(), &lapack_info);

        double energy = h_D[0];
        total_energy += energy;

        // Ritz vector: theta = V * z_0
        std::vector<Scalar> h_ritz(niter);
        for (int i = 0; i < niter; i++) h_ritz[i] = Traits::make_scalar(h_Z[i]);
        HIP_CHECK(hipMemcpyAsync(ws.d_ritz_coeffs, h_ritz.data(),
                  niter * sizeof(Scalar), hipMemcpyHostToDevice, streams_[0]));

        Scalar h_one = Traits::one(), h_zero = Traits::zero();
        ROCBLAS_CHECK(Traits::gemv(handles_[0], rocblas_operation_none,
            n, niter, &h_one,
            st.d_lanczos_v, n,
            ws.d_ritz_coeffs, 1,
            &h_zero, d_thetas[b], 1));

        // Normalize
        double nrm;
        ROCBLAS_CHECK(Traits::nrm2(handles_[0], n, d_thetas[b], 1, &nrm));
        if (nrm > 1e-14) {
            Scalar inv_nrm = Traits::make_scalar(1.0 / nrm);
            ROCBLAS_CHECK(Traits::scal(handles_[0], n, &inv_nrm, d_thetas[b], 1));
        }
    }

    return total_energy / n_batch;
}

// ============================================================================
// Batched segment sweep: lock-step across segments, batched GEMM in apply_heff
// Replaces parallel_sweep when use_batched_sweep_ is true
// ============================================================================

template<typename Scalar>
void PDMRGGPUOpt<Scalar>::batched_segment_sweep(bool even_LR) {
    // Determine sweep direction per segment:
    // even_LR=true: even segments go LR, odd go RL
    // even_LR=false: even segments go RL, odd go LR
    struct SegWork {
        int seg_idx;
        int first, last;
        bool is_LR;
        int sweep_len;  // number of two-site optimizations
    };
    std::vector<SegWork> work(n_segments_);
    int max_sweep_len = 0;

    for (int k = 0; k < n_segments_; k++) {
        work[k].seg_idx = k;
        work[k].first = seg_first_[k];
        work[k].last = seg_last_[k];
        bool even = (k % 2 == 0);
        work[k].is_LR = (even == even_LR);
        work[k].sweep_len = work[k].last - work[k].first;  // two-site: last-first bonds
        max_sweep_len = std::max(max_sweep_len, work[k].sweep_len);
    }

    // Process lock-step: at each relative position, optimize all active segments
    for (int step = 0; step < max_sweep_len; step++) {
        // Collect active segments at this step
        std::vector<int> active_sites;
        std::vector<int> active_segs;
        std::vector<char> active_dirs;

        for (int k = 0; k < n_segments_; k++) {
            if (step >= work[k].sweep_len) continue;
            int site;
            char dir;
            if (work[k].is_LR) {
                site = work[k].first + step;
                dir = 'R';
            } else {
                site = work[k].last - 1 - step;
                dir = 'L';
            }
            active_sites.push_back(site);
            active_segs.push_back(k);
            active_dirs.push_back(dir);
        }

        int n_active = (int)active_sites.size();
        if (n_active == 0) continue;

        // Check if all active segments have matching theta dimensions
        bool dims_match = true;
        int ref_cL = chi_L(active_sites[0]);
        int ref_cR = chi_R(active_sites[0] + 1);
        int theta_size = ref_cL * d_ * d_ * ref_cR;
        for (int b = 1; b < n_active; b++) {
            if (chi_L(active_sites[b]) != ref_cL || chi_R(active_sites[b] + 1) != ref_cR) {
                dims_match = false;
                break;
            }
        }

        if (dims_match && n_active > 1) {
            // === Batched path: form theta, batched Lanczos, split, env update ===

            // Form theta for all segments (each on its own stream)
            for (int b = 0; b < n_active; b++) {
                form_theta_two_site(active_sites[b], active_segs[b]);
            }
            // Sync all segment streams before batched Lanczos on stream 0
            for (int b = 0; b < n_active; b++) {
                HIP_CHECK(hipStreamSynchronize(streams_[active_segs[b]]));
            }

            // Batched Lanczos eigensolver (runs on stream 0)
            std::vector<Scalar*> d_thetas(n_active);
            for (int b = 0; b < n_active; b++) {
                d_thetas[b] = workspaces_[active_segs[b]].d_theta;
            }

            batched_lanczos_eigensolver(active_sites.data(), active_segs.data(),
                                         n_active, d_thetas.data(), theta_size);

            // Sync stream 0 before per-segment splits (they run on per-segment streams)
            HIP_CHECK(hipStreamSynchronize(streams_[0]));

            // Bond split per segment (each on its own stream for concurrency)
            for (int b = 0; b < n_active; b++) {
                int site = active_sites[b];
                int si = active_segs[b];
                char dir = active_dirs[b];
                auto& ws = workspaces_[si];

                if (use_ns_split_) {
                    ns_split(site, ws.d_theta, dir, si);
                } else if (use_rsvd_) {
                    rsvd_split(site, ws.d_theta, dir, si);
                } else {
                    svd_split(site, ws.d_theta, dir, si);
                }
            }

            // Sync all segment streams after splits before env updates
            for (int b = 0; b < n_active; b++) {
                HIP_CHECK(hipStreamSynchronize(streams_[active_segs[b]]));
            }

            // Environment updates — dispatch on per-segment streams for concurrency
            for (int b = 0; b < n_active; b++) {
                int site = active_sites[b];
                int si = active_segs[b];
                if (active_dirs[b] == 'R') {
                    update_left_env(site, si);
                } else {
                    update_right_env(site + 1, si);
                }
            }
            // Wait for all env updates
            for (int b = 0; b < n_active; b++) {
                HIP_CHECK(hipStreamSynchronize(streams_[active_segs[b]]));
            }

        } else {
            // === Fallback: per-segment processing (different dims or single segment) ===
            for (int b = 0; b < n_active; b++) {
                int site = active_sites[b];
                int si = active_segs[b];
                char dir = active_dirs[b];
                optimize_bond(site, dir, si);
                if (dir == 'R') {
                    update_left_env(site, si);
                } else {
                    update_right_env(site + 1, si);
                }
            }
        }
    }
}

// ============================================================================
// Form theta with V injection: θ = ψ_L · diag(V) · ψ_R  (Stoudenmire Eq. 5)
// ============================================================================

template<typename Scalar>
void PDMRGGPUOpt<Scalar>::form_theta_with_V(int site, int boundary_idx, int si) {
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
    int psi_R_size = chi_bond * d_ * cR;
    HIP_CHECK(hipMemcpyAsync(ws.d_T1, d_mps_tensors_[site + 1],
                              psi_R_size * sizeof(Scalar),
                              hipMemcpyDeviceToDevice, streams_[si]));
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
//   2. Optimize θ with eigensolver (Davidson or Lanczos)
//   3. SVD split: θ → U · S · Vh
//   4. Store V_new = 1/clip(S, 1e-12) for next iteration
//   5. MPS[bsite] = U (left-canonical), MPS[bsite+1] = S·Vh
//   6. Update environments from canonical tensors
//
// parity: 0 = even-indexed boundaries, 1 = odd, -1 = all
// ============================================================================

template<typename Scalar>
double PDMRGGPUOpt<Scalar>::merge_and_optimize_boundaries(int parity) {
    double energy = 0.0;
    int si = 0;

    for (int b = 0; b < (int)boundary_bonds_.size(); b++) {
        if (parity >= 0 && (b % 2) != parity) continue;

        int bsite = boundary_bonds_[b];
        int cL = chi_L(bsite);
        int cR = chi_R(bsite + 1);
        int theta_size = cL * d_ * d_ * cR;
        auto& ws = workspaces_[si];

        // Step 1: Form θ = ψ_L · diag(V) · ψ_R
        form_theta_with_V(bsite, b, si);

        // Step 2: Optimize θ with eigensolver
        if (use_davidson_) {
            energy = block_davidson_eigensolver(bsite, ws.d_theta, theta_size, si);
        } else {
            energy = lanczos_eigensolver(bsite, ws.d_theta, theta_size, si);
        }

        // Step 3: SVD split → direction 'R': MPS[bsite]=U, MPS[bsite+1]=S·Vh
        // Always use SVD at boundaries (not NS-split) for accurate V = 1/S
        if (use_rsvd_)
            rsvd_split(bsite, ws.d_theta, 'R', si);
        else
            svd_split(bsite, ws.d_theta, 'R', si);

        // Step 4: Update V = 1/clip(S, 1e-12) for next iteration
        int new_chi = bond_dims_[bsite + 1];
        boundary_states_[b].chi = new_chi;
        boundary_states_[b].V.resize(new_chi);

        const RealType reg = RealType(1e-12);
        for (int i = 0; i < new_chi; i++) {
            RealType s_val = ws.h_svd_S[i];
            if (s_val < reg) s_val = reg;
            boundary_states_[b].V[i] = RealType(1.0) / s_val;
        }

        // Step 5: Update environments
        update_left_env(bsite, si);
        update_right_env(bsite + 1, si);
    }
    return energy;
}

// ============================================================================
// Main algorithm
// ============================================================================

template<typename Scalar>
double PDMRGGPUOpt<Scalar>::run(int n_outer_sweeps, int n_local_sweeps, int n_warmup) {
    const char* type_name = Traits::is_complex ? "complex128" : "float64";
    printf("=== PDMRG-GPU-OPT (NS + MFMA-16 pad + %s, %s) ===\n",
           use_batched_sweep_ ? "cross-seg batched" : "per-seg streams", type_name);
    if (chi_max_ != chi_max_user_)
        printf("L = %d, d = %d, chi_max = %d (padded from %d), D_mpo = %d, segments = %d\n",
               L_, d_, chi_max_, chi_max_user_, D_mpo_, n_segments_);
    else
        printf("L = %d, d = %d, chi_max = %d, D_mpo = %d, segments = %d\n",
               L_, d_, chi_max_, D_mpo_, n_segments_);

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

    // Main PDMRG loop (Stoudenmire staggered sweeps)
    initialize_boundary_states();
    double energy_prev = warmup_energy;
    energy_ = warmup_energy;
    bool outer_converged = false;

    auto parallel_sweep = [this](auto sweep_fn) {
        std::vector<std::thread> threads(n_segments_);
        for (int k = 0; k < n_segments_; k++) {
            threads[k] = std::thread([this, k, &sweep_fn]{ sweep_fn(this, k); });
        }
        for (auto& t : threads) t.join();
        // Sync per-segment streams (not device-wide) before boundary coupling
        for (int s = 0; s < n_segments_; s++) {
            HIP_CHECK(hipStreamSynchronize(streams_[s]));
        }
    };

    bool has_odd_boundaries = ((int)boundary_bonds_.size() > 1);

    for (int outer = 0; outer < n_outer_sweeps; outer++) {
        for (int local_sw = 0; local_sw < n_local_sweeps; local_sw++) {
            if (use_batched_sweep_ && n_segments_ >= 2) {
                batched_segment_sweep(true);   // even=LR, odd=RL
            } else {
                parallel_sweep([](PDMRGGPUOpt* self, int k) {
                    if (k % 2 == 0) self->segment_sweep_LR(k);
                    else             self->segment_sweep_RL(k);
                });
            }

            if (boundary_bonds_.size() > 0) {
                energy_ = merge_and_optimize_boundaries(0);
            }

            if (use_batched_sweep_ && n_segments_ >= 2) {
                batched_segment_sweep(false);  // even=RL, odd=LR
            } else {
                parallel_sweep([](PDMRGGPUOpt* self, int k) {
                    if (k % 2 == 0) self->segment_sweep_RL(k);
                    else             self->segment_sweep_LR(k);
                });
            }

            if (has_odd_boundaries) {
                energy_ = merge_and_optimize_boundaries(1);
            }
        }

        double dE = std::abs(energy_ - energy_prev);
        if (dE < tol_ && outer > 0) {
            printf("Converged after %d outer iterations!\n", outer + 1);
            outer_converged = true;
            break;
        }
        energy_prev = energy_;
    }

    // Polish phase: full-chain sweeps to fix stale boundary environments
    if (n_segments_ > 1) {
        int n_polish = 10;
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

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(t_end - t_start).count();
    printf("Final energy: %.12f\n", energy_);
    printf("Total wall time: %.3f s\n", total_time);
    report_timers();

    return energy_;
}

// ============================================================================
// Phase timers
// ============================================================================

template<typename Scalar>
void PDMRGGPUOpt<Scalar>::init_timers() {
    t_lanczos_.init("lanczos", opts_.profile);
    t_apply_heff_.init("apply_heff", opts_.profile);
    t_svd_.init("svd", opts_.profile);
    t_absorb_.init("absorb", opts_.profile);
    t_env_update_.init("env_update", opts_.profile);
}

template<typename Scalar>
void PDMRGGPUOpt<Scalar>::report_timers() {
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
void PDMRGGPUOpt<Scalar>::get_mps(std::vector<std::vector<Scalar>>& h_mps) const {
    h_mps.resize(L_);
    for (int i = 0; i < L_; i++) {
        int size = chi_L(i) * d_ * chi_R(i);
        h_mps[i].resize(size);
        HIP_CHECK(hipMemcpy(h_mps[i].data(), d_mps_tensors_[i],
                            size * sizeof(Scalar), hipMemcpyDeviceToHost));
    }
}

#endif // PDMRG_GPU_OPT_IMPL_H
