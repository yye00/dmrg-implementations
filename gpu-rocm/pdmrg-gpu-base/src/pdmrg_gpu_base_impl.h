#ifndef PDMRG_GPU_BASE_IMPL_H
#define PDMRG_GPU_BASE_IMPL_H

#include <rocsolver/rocsolver.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <chrono>
#include <thread>
#include <stdexcept>
#include <string>

#include "../../common/hip_check.h"

// promote_double_to_complex now defined in common/scalar_traits.h
// (round-5 single-source-of-truth promotion).

// ============================================================================
// Constructor
// ============================================================================

template<typename Scalar>
PDMRGGPUBase<Scalar>::PDMRGGPUBase(int L, int d, int chi_max, int D_mpo,
                                   int n_segments, double tol)
    : L_(L), d_(d), chi_max_(chi_max), D_mpo_(D_mpo), tol_(tol), energy_(0.0),
      n_segments_(n_segments), lanczos_use_1site_(false) {

    if (L < 2 * n_segments) {
        throw std::runtime_error("Need at least 2 sites per segment: L >= 2*n_segments");
    }

    // Bond dimensions
    bond_dims_.resize(L + 1);
    bond_dims_[0] = 1;
    bond_dims_[L] = 1;
    for (int i = 1; i < L; i++) {
        double exact_dim = pow((double)d_, std::min(i, L - i));
        bond_dims_[i] = (exact_dim > chi_max_) ? chi_max_ : (int)exact_dim;
    }

    partition_chain();
    initialize_boundary_states();

    // Per-segment streams and handles. Non-blocking flag is REQUIRED (not an
    // optimization): under the default blocking-stream mode, any sync
    // hipMemcpy on the legacy stream serializes against all blocking streams,
    // which would defeat parallel boundary merges. The optimized PDMRGGPU
    // also uses hipStreamNonBlocking for the same reason.
    streams_.resize(n_segments_);
    handles_.resize(n_segments_);
    for (int k = 0; k < n_segments_; k++) {
        HIP_CHECK(hipStreamCreateWithFlags(&streams_[k], hipStreamNonBlocking));
        ROCBLAS_CHECK(rocblas_create_handle(&handles_[k]));
        ROCBLAS_CHECK(rocblas_set_stream(handles_[k], streams_[k]));
    }

    int dd = d_ * d_;

    // MPS tensors
    d_mps_tensors_.resize(L, nullptr);
    for (int i = 0; i < L; i++) {
        allocate_mps_tensor(i, chi_L(i), chi_R(i));
    }

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

    // Lanczos limits
    theta_size_max_ = chi_max_ * dd * chi_max_;
    max_lanczos_iter_ = std::min(100, theta_size_max_);

    // Per-bond fused MPO storage. Allocated/populated by precompute_WW()
    // called from set_mpo() — one entry per bond (L_ - 1 bonds).
    d_WW_.resize(L - 1, nullptr);

    allocate_stream_workspaces();
}

// ============================================================================
// Per-stream workspace allocation — fully on-device for the inner loop.
// ============================================================================

template<typename Scalar>
void PDMRGGPUBase<Scalar>::allocate_stream_workspaces() {
    int dd = d_ * d_;
    int t_max = D_mpo_ * dd * chi_max_ * chi_max_;
    int svd_max_m = chi_max_ * d_;
    int svd_max_n = d_ * chi_max_;
    int svd_max_k = std::min(svd_max_m, svd_max_n);
    int psi_R_max = chi_max_ * d_ * chi_max_;

    workspaces_.resize(n_segments_);

    for (int k = 0; k < n_segments_; k++) {
        auto& ws = workspaces_[k];

        HIP_CHECK(hipMalloc(&ws.d_T1, t_max * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&ws.d_T2, t_max * sizeof(Scalar)));

        HIP_CHECK(hipMalloc(&ws.d_theta, theta_size_max_ * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&ws.d_heff_result, theta_size_max_ * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&ws.d_lanczos_v, (size_t)max_lanczos_iter_ * theta_size_max_ * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&ws.d_ritz_coeffs, max_lanczos_iter_ * sizeof(Scalar)));

        // Device-pointer scratch (single scalars).
        HIP_CHECK(hipMalloc(&ws.d_dot_result,   sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&ws.d_nrm2_result,  sizeof(RealType)));
        HIP_CHECK(hipMalloc(&ws.d_inv_nrm,      sizeof(RealType)));
        HIP_CHECK(hipMalloc(&ws.d_neg_alpha,    sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&ws.d_neg_overlap,  sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&ws.d_neg_beta_scalars, max_lanczos_iter_ * sizeof(Scalar)));

        // Per-iteration alpha/beta arrays on device.
        HIP_CHECK(hipMalloc(&ws.d_alpha_dev, max_lanczos_iter_ * sizeof(RealType)));
        HIP_CHECK(hipMalloc(&ws.d_beta_dev,  max_lanczos_iter_ * sizeof(RealType)));

        // rocSOLVER dsteqr workspaces.
        HIP_CHECK(hipMalloc(&ws.d_steqr_D,    max_lanczos_iter_ * sizeof(double)));
        HIP_CHECK(hipMalloc(&ws.d_steqr_E,    max_lanczos_iter_ * sizeof(double)));
        HIP_CHECK(hipMalloc(&ws.d_steqr_C,    (size_t)max_lanczos_iter_ * max_lanczos_iter_ * sizeof(double)));
        HIP_CHECK(hipMalloc(&ws.d_steqr_info, sizeof(rocblas_int)));

        // SVD workspace
        HIP_CHECK(hipMalloc(&ws.d_svd_A,    theta_size_max_ * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&ws.d_svd_U,    (size_t)svd_max_m * svd_max_k * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&ws.d_svd_S,    svd_max_k * sizeof(RealType)));
        HIP_CHECK(hipMalloc(&ws.d_svd_Vh,   (size_t)svd_max_k * svd_max_n * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&ws.d_svd_work, std::max((size_t)svd_max_m * svd_max_k,
                                                      (size_t)svd_max_k * svd_max_n) * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&ws.d_svd_E,    svd_max_k * sizeof(RealType)));
        HIP_CHECK(hipMalloc(&ws.d_svd_info, sizeof(int)));
        // Round-8 fix (C-new1): boundary R_env must be built from canonical
        // Vh, not from S*Vh — pre-allocate the swap buffer.
        HIP_CHECK(hipMalloc(&ws.d_Vh_canonical, theta_size_max_ * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&ws.d_svdj_residual, sizeof(double)));
        HIP_CHECK(hipMalloc(&ws.d_svdj_n_sweeps, sizeof(rocblas_int)));

        // Pre-allocated scratch for form_theta_with_V.
        HIP_CHECK(hipMalloc(&ws.d_psi_R, psi_R_max * sizeof(Scalar)));

        // Tiny host buffer for truncation-rank decision.
        ws.h_svd_S.resize(svd_max_k);

        // GPU-native accurate SVD scratch — Stoudenmire is part of pdmrg's
        // algorithmic definition (per feedback_pdmrg_requires_stoudenmire memory).
        // Sized for the largest theta the boundary path will see.
        ws.asvd.allocate(chi_max_ * d_, d_ * chi_max_);
    }
}

// ============================================================================
// Destructor
// ============================================================================

template<typename Scalar>
PDMRGGPUBase<Scalar>::~PDMRGGPUBase() {
    free_gpu_resources();
}

template<typename Scalar>
void PDMRGGPUBase<Scalar>::free_gpu_resources() {
    for (auto ptr : d_mps_tensors_) if (ptr) hipFree(ptr);
    for (auto ptr : d_mpo_tensors_) if (ptr) hipFree(ptr);
    for (auto ptr : d_W_left_) if (ptr) hipFree(ptr);
    for (auto ptr : d_W_right_) if (ptr) hipFree(ptr);
    for (auto ptr : d_L_envs_) if (ptr) hipFree(ptr);
    for (auto ptr : d_R_envs_) if (ptr) hipFree(ptr);

    for (auto ptr : d_WW_) if (ptr) hipFree(ptr);

    for (auto& ws : workspaces_) {
        if (ws.d_T1) hipFree(ws.d_T1);
        if (ws.d_T2) hipFree(ws.d_T2);
        if (ws.d_theta) hipFree(ws.d_theta);
        if (ws.d_heff_result) hipFree(ws.d_heff_result);
        if (ws.d_lanczos_v) hipFree(ws.d_lanczos_v);
        if (ws.d_ritz_coeffs) hipFree(ws.d_ritz_coeffs);
        if (ws.d_dot_result) hipFree(ws.d_dot_result);
        if (ws.d_nrm2_result) hipFree(ws.d_nrm2_result);
        if (ws.d_inv_nrm) hipFree(ws.d_inv_nrm);
        if (ws.d_neg_alpha) hipFree(ws.d_neg_alpha);
        if (ws.d_neg_overlap) hipFree(ws.d_neg_overlap);
        if (ws.d_neg_beta_scalars) hipFree(ws.d_neg_beta_scalars);
        if (ws.d_alpha_dev) hipFree(ws.d_alpha_dev);
        if (ws.d_beta_dev) hipFree(ws.d_beta_dev);
        if (ws.d_steqr_D) hipFree(ws.d_steqr_D);
        if (ws.d_steqr_E) hipFree(ws.d_steqr_E);
        if (ws.d_steqr_C) hipFree(ws.d_steqr_C);
        if (ws.d_steqr_info) hipFree(ws.d_steqr_info);
        if (ws.d_svd_A) hipFree(ws.d_svd_A);
        if (ws.d_svd_U) hipFree(ws.d_svd_U);
        if (ws.d_svd_S) hipFree(ws.d_svd_S);
        if (ws.d_svd_Vh) hipFree(ws.d_svd_Vh);
        if (ws.d_svd_work) hipFree(ws.d_svd_work);
        if (ws.d_svd_E) hipFree(ws.d_svd_E);
        if (ws.d_svd_info) hipFree(ws.d_svd_info);
        if (ws.d_Vh_canonical) hipFree(ws.d_Vh_canonical);
        if (ws.d_svdj_residual) hipFree(ws.d_svdj_residual);
        if (ws.d_svdj_n_sweeps) hipFree(ws.d_svdj_n_sweeps);
        if (ws.d_psi_R) hipFree(ws.d_psi_R);
        ws.asvd.release();  // GPU-native accurate SVD scratch
    }

    for (auto& h : handles_) rocblas_destroy_handle(h);
    for (auto& s : streams_) hipStreamDestroy(s);
}

// ============================================================================
// Memory management
// ============================================================================

template<typename Scalar>
void PDMRGGPUBase<Scalar>::allocate_mps_tensor(int site, int cL, int cR) {
    size_t max_sz = (size_t)chi_max_ * d_ * chi_max_ * sizeof(Scalar);
    if (!d_mps_tensors_[site]) {
        HIP_CHECK(hipMalloc(&d_mps_tensors_[site], max_sz));
    }
    (void)cL; (void)cR;
}

template<typename Scalar>
void PDMRGGPUBase<Scalar>::ensure_L_env_alloc(int idx, int chi) {
    if (chi > L_env_alloc_chi_[idx]) {
        if (d_L_envs_[idx]) HIP_CHECK(hipFree(d_L_envs_[idx]));
        int sz = chi * D_mpo_ * chi;
        HIP_CHECK(hipMalloc(&d_L_envs_[idx], sz * sizeof(Scalar)));
        L_env_alloc_chi_[idx] = chi;
    }
}

template<typename Scalar>
void PDMRGGPUBase<Scalar>::ensure_R_env_alloc(int idx, int chi) {
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
void PDMRGGPUBase<Scalar>::partition_chain() {
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

template<typename Scalar>
void PDMRGGPUBase<Scalar>::initialize_boundary_states() {
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
void PDMRGGPUBase<Scalar>::initialize_mps_random(double scale) {
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
// MPO setup — W_left / W_right matrices only (NO fused WW precompute)
// ============================================================================

template<typename Scalar>
void PDMRGGPUBase<Scalar>::set_mpo(const std::vector<Scalar*>& h_mpo_tensors) {
    int D = D_mpo_, d = d_;
    for (int i = 0; i < L_; i++) {
        int size = D * d * d * D;
        // Guard against double-call (round-7 M4): free previous MPO buffer.
        if (d_mpo_tensors_[i]) HIP_CHECK(hipFree(d_mpo_tensors_[i]));
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
    // Precompute the per-bond fused two-site MPO (WW) once at setup time.
    // The natural first-pass GPU choice. Eliminates the per-iteration host
    // roundtrip that would otherwise dominate apply_heff_two_site.
    precompute_WW();
}

template<typename Scalar>
void PDMRGGPUBase<Scalar>::precompute_WW() {
    int D = D_mpo_, d = d_;
    int dd = d * d;
    int ww_size = D * dd * dd * D;

    std::vector<Scalar> h_WL_raw((size_t)D * d * d * D);
    std::vector<Scalar> h_WR_raw((size_t)D * d * d * D);
    std::vector<Scalar> h_WW(ww_size);

    for (int site = 0; site < L_ - 1; site++) {
        HIP_CHECK(hipMemcpy(h_WL_raw.data(), d_mpo_tensors_[site],
                            (size_t)D * d * d * D * sizeof(Scalar),
                            hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_WR_raw.data(), d_mpo_tensors_[site + 1],
                            (size_t)D * d * d * D * sizeof(Scalar),
                            hipMemcpyDeviceToHost));

        std::fill(h_WW.begin(), h_WW.end(), Traits::zero());
        for (int w = 0; w < D; w++)
          for (int n = 0; n < D; n++)
            for (int s1 = 0; s1 < d; s1++)
              for (int s2 = 0; s2 < d; s2++)
                for (int s1p = 0; s1p < d; s1p++)
                  for (int s2p = 0; s2p < d; s2p++) {
                      Scalar val = Traits::zero();
                      for (int m = 0; m < D; m++) {
                          Scalar wl = h_WL_raw[w + s1*D + s1p*D*d + m*D*d*d];
                          Scalar wr = h_WR_raw[m + s2*D + s2p*D*d + n*D*d*d];
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

        HIP_CHECK(hipMalloc(&d_WW_[site], ww_size * sizeof(Scalar)));
        HIP_CHECK(hipMemcpy(d_WW_[site], h_WW.data(),
                            ww_size * sizeof(Scalar), hipMemcpyHostToDevice));
    }
}

// ============================================================================
// Two-site theta formation
// ============================================================================

template<typename Scalar>
void PDMRGGPUBase<Scalar>::form_theta_two_site(int site, int si) {
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
// Two-site H_eff — NAIVE: rebuild WW on HOST per call + unbatched GEMM loops
// ============================================================================

template<typename Scalar>
void PDMRGGPUBase<Scalar>::apply_heff_two_site(int site, const Scalar* d_theta_in,
                                               Scalar* d_result, int si) {
    int cL = chi_L(site);
    int cR = chi_R(site + 1);
    int D = D_mpo_, d = d_;
    int dd = d * d;
    Scalar one = Traits::one(), zero_val = Traits::zero();
    auto& ws = workspaces_[si];

    Scalar* L_env = d_L_envs_[site];
    Scalar* R_env = d_R_envs_[site + 2];
    Scalar* T1 = ws.d_T1;
    Scalar* T2 = ws.d_T2;
    Scalar* WW = d_WW_[site];   // precomputed once at set_mpo() time

    // Step 1: Contract L with theta — unbatched loop of single GEMMs
    for (int w = 0; w < D; w++) {
        for (int s1 = 0; s1 < d; s1++) {
            for (int s2 = 0; s2 < d; s2++) {
                const Scalar* A_ptr = L_env + w * cL;
                const Scalar* B_ptr = d_theta_in + (s1 + s2 * d) * cL;
                Scalar* C_ptr = T1 + (w * dd + s1 * d + s2) * cL * cR;
                ROCBLAS_CHECK(Traits::gemm(handles_[si],
                    Traits::op_t, rocblas_operation_none,
                    cL, cR, cL,
                    &one,
                    A_ptr, cL * D,
                    B_ptr, cL * dd,
                    &zero_val,
                    C_ptr, cL));
            }
        }
    }

    // Step 2: Dense GEMM — absorb per-call WW
    ROCBLAS_CHECK(Traits::gemm(handles_[si],
        rocblas_operation_none, rocblas_operation_none,
        cL * cR, dd * D, D * dd,
        &one,
        T1, cL * cR,
        WW, D * dd,
        &zero_val,
        T2, cL * cR));

    // Step 3: Contract R with T2 — unbatched loop
    for (int n = 0; n < D; n++) {
        Scalar beta = (n == 0) ? Traits::zero() : Traits::one();
        for (int s1p = 0; s1p < d; s1p++) {
            for (int s2p = 0; s2p < d; s2p++) {
                const Scalar* A_ptr = T2 + (n * dd + s1p * d + s2p) * cL * cR;
                const Scalar* B_ptr = R_env + n * cR;
                Scalar* C_ptr = d_result + (s1p + s2p * d) * cL;
                ROCBLAS_CHECK(Traits::gemm(handles_[si],
                    rocblas_operation_none, rocblas_operation_none,
                    cL, cR, cR,
                    &one,
                    A_ptr, cL,
                    B_ptr, cR * D,
                    &beta,
                    C_ptr, cL * dd));
            }
        }
    }
}

// ============================================================================
// Single-site H_eff — naive unbatched loops, shares W_left and envs
// theta shape: (cL, d, cR), result shape: (cL, d, cR)
// ============================================================================

template<typename Scalar>
void PDMRGGPUBase<Scalar>::apply_heff_single_site(int site, const Scalar* d_theta_in,
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

    // Step 1: V[w*d+s, a', b] = L[w]^T[a', a] * theta[a, s, b]
    for (int w = 0; w < D; w++) {
        for (int s = 0; s < d; s++) {
            const Scalar* A_ptr = L_env + w * cL;
            const Scalar* B_ptr = d_theta_in + s * cL;
            Scalar* C_ptr = V + (w * d + s) * cL * cR;
            ROCBLAS_CHECK(Traits::gemm(handles_[si],
                Traits::op_t, rocblas_operation_none,
                cL, cR, cL,
                &one,
                A_ptr, cL * D,
                B_ptr, cL * d,
                &zero_val,
                C_ptr, cL));
        }
    }

    // Step 2: U = V * W_mat
    ROCBLAS_CHECK(Traits::gemm(handles_[si],
        rocblas_operation_none, rocblas_operation_none,
        cL * cR, d * D, D * d,
        &one,
        V, cL * cR,
        W_mat, D * d,
        &zero_val,
        U, cL * cR));

    // Step 3: result[s', a', b'] = sum_w' U[w'*d+s', a', b] * R[w'][b, b']
    for (int wp = 0; wp < D; wp++) {
        Scalar beta = (wp == 0) ? Traits::zero() : Traits::one();
        for (int sp = 0; sp < d; sp++) {
            const Scalar* A_ptr = U + (wp * d + sp) * cL * cR;
            const Scalar* B_ptr = R_env + wp * cR;
            Scalar* C_ptr = d_result + sp * cL;
            ROCBLAS_CHECK(Traits::gemm(handles_[si],
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
// update_left_env — naive unbatched loops
// ============================================================================

template<typename Scalar>
void PDMRGGPUBase<Scalar>::update_left_env(int site, int si) {
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

    for (int w = 0; w < D; w++) {
        for (int s = 0; s < d; s++) {
            const Scalar* A_ptr = L_env + w * chi_in;
            const Scalar* B_ptr = A + s * chi_in;
            Scalar* C_ptr = V + (w * d + s) * chi_in * chi_out;
            ROCBLAS_CHECK(Traits::gemm(handles_[si],
                Traits::op_t, rocblas_operation_none,
                chi_in, chi_out, chi_in,
                &one,
                A_ptr, chi_in * D,
                B_ptr, chi_in * d,
                &zero_val,
                C_ptr, chi_in));
        }
    }

    ROCBLAS_CHECK(Traits::gemm(handles_[si],
        rocblas_operation_none, rocblas_operation_none,
        chi_in * chi_out, d * D, D * d,
        &one,
        V, chi_in * chi_out,
        W_mat, D * d,
        &zero_val,
        U, chi_in * chi_out));

    for (int sp = 0; sp < d; sp++) {
        Scalar beta = (sp == 0) ? Traits::zero() : Traits::one();
        for (int w = 0; w < D; w++) {
            const Scalar* A_ptr = U + (w * d + sp) * chi_in * chi_out;
            const Scalar* B_ptr = A + sp * chi_in;
            Scalar* C_ptr = L_new + w * chi_out;
            ROCBLAS_CHECK(Traits::gemm(handles_[si],
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
        conjugate_inplace(L_new, chi_out * D * chi_out, streams_[si]);
    }
}

// ============================================================================
// update_right_env — naive unbatched loops
// ============================================================================

template<typename Scalar>
void PDMRGGPUBase<Scalar>::update_right_env(int site, int si) {
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

    for (int w = 0; w < D; w++) {
        for (int s = 0; s < d; s++) {
            const Scalar* A_ptr = A + s * chi_out;
            const Scalar* B_ptr = R_env + w * chi_in;
            Scalar* C_ptr = V + (w * d + s) * chi_out * chi_in;
            ROCBLAS_CHECK(Traits::gemm(handles_[si],
                rocblas_operation_none, rocblas_operation_none,
                chi_out, chi_in, chi_in,
                &one,
                A_ptr, chi_out * d,
                B_ptr, chi_in * D,
                &zero_val,
                C_ptr, chi_out));
        }
    }

    ROCBLAS_CHECK(Traits::gemm(handles_[si],
        rocblas_operation_none, rocblas_operation_none,
        chi_out * chi_in, d * D, D * d,
        &one,
        V, chi_out * chi_in,
        W_mat, D * d,
        &zero_val,
        U, chi_out * chi_in));

    for (int sp = 0; sp < d; sp++) {
        Scalar beta = (sp == 0) ? Traits::zero() : Traits::one();
        for (int w = 0; w < D; w++) {
            const Scalar* A_ptr = U + (w * d + sp) * chi_out * chi_in;
            const Scalar* B_ptr = A + sp * chi_out;
            Scalar* C_ptr = R_new + w * chi_out;
            ROCBLAS_CHECK(Traits::gemm(handles_[si],
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
// Environment building (stream 0)
// ============================================================================

template<typename Scalar>
void PDMRGGPUBase<Scalar>::build_initial_environments() {
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
// Lanczos eigensolver — fully on-device, per-stream.
// Same design as dmrg-gpu-base / dmrg2-gpu-base (device-pointer rocBLAS,
// rocsolver_dsteqr, shared lanczos_process kernels), but per-stream — each
// call uses workspaces_[si] and handles_[si] so concurrent segment sweeps
// do not interfere.
// ============================================================================

template<typename Scalar>
double PDMRGGPUBase<Scalar>::lanczos_eigensolver(int site, Scalar* d_theta,
                                                 int theta_size, int si) {
    int n = theta_size;
    int max_iter = std::min(max_lanczos_iter_, n);
    auto& ws = workspaces_[si];

    Scalar* d_lanczos_v = ws.d_lanczos_v;

    ROCBLAS_CHECK(rocblas_set_pointer_mode(handles_[si], rocblas_pointer_mode_device));

    // v[0] = theta / ||theta||
    ROCBLAS_CHECK(Traits::nrm2(handles_[si], n, d_theta, 1, ws.d_nrm2_result));
    hipLaunchKernelGGL(inv_real_kernel, dim3(1), dim3(1), 0, streams_[si],
                       ws.d_nrm2_result, ws.d_inv_nrm);
    ROCBLAS_CHECK(Traits::scal_real(handles_[si], n, ws.d_inv_nrm, d_theta, 1));
    HIP_CHECK(hipMemcpyAsync(d_lanczos_v, d_theta,
                             n * sizeof(Scalar), hipMemcpyDeviceToDevice, streams_[si]));

    int iter;
    for (iter = 0; iter < max_iter; iter++) {
        Scalar* d_vi = d_lanczos_v + (size_t)iter * n;

        if (lanczos_use_1site_)
            apply_heff_single_site(site, d_vi, ws.d_heff_result, si);
        else
            apply_heff_two_site(site, d_vi, ws.d_heff_result, si);

        // alpha_i = Re <v_i | w>
        ROCBLAS_CHECK(Traits::dot(handles_[si], n, d_vi, 1, ws.d_heff_result, 1, ws.d_dot_result));
        hipLaunchKernelGGL(lanczos_process_alpha_kernel<Scalar>, dim3(1), dim3(1), 0, streams_[si],
                           ws.d_dot_result, ws.d_neg_alpha, ws.d_alpha_dev, iter);

        // w -= alpha_i * v_i
        ROCBLAS_CHECK(Traits::axpy(handles_[si], n, ws.d_neg_alpha, d_vi, 1, ws.d_heff_result, 1));

        // w -= beta_{i-1} * v_{i-1}
        if (iter > 0) {
            ROCBLAS_CHECK(Traits::axpy(handles_[si], n,
                ws.d_neg_beta_scalars + (iter - 1),
                d_lanczos_v + (size_t)(iter - 1) * n, 1,
                ws.d_heff_result, 1));
        }

        // Full one-pass classical Gram-Schmidt reorthogonalization, on device.
        for (int j = 0; j <= iter; j++) {
            ROCBLAS_CHECK(Traits::dot(handles_[si], n,
                d_lanczos_v + (size_t)j * n, 1,
                ws.d_heff_result, 1, ws.d_dot_result));
            hipLaunchKernelGGL(negate_scalar_kernel<Scalar>, dim3(1), dim3(1), 0, streams_[si],
                               ws.d_dot_result, ws.d_neg_overlap);
            ROCBLAS_CHECK(Traits::axpy(handles_[si], n, ws.d_neg_overlap,
                d_lanczos_v + (size_t)j * n, 1,
                ws.d_heff_result, 1));
        }

        // beta_i = ||w||
        ROCBLAS_CHECK(Traits::nrm2(handles_[si], n, ws.d_heff_result, 1, ws.d_nrm2_result));
        hipLaunchKernelGGL(lanczos_process_beta_kernel<Scalar>, dim3(1), dim3(1), 0, streams_[si],
                           ws.d_nrm2_result, ws.d_inv_nrm, ws.d_beta_dev, ws.d_neg_beta_scalars, iter);

        // v_{i+1} = w / beta_i
        if (iter + 1 < max_iter) {
            Scalar* d_vip1 = d_lanczos_v + (size_t)(iter + 1) * n;
            HIP_CHECK(hipMemcpyAsync(d_vip1, ws.d_heff_result,
                                     n * sizeof(Scalar), hipMemcpyDeviceToDevice, streams_[si]));
            ROCBLAS_CHECK(Traits::scal_real(handles_[si], n, ws.d_inv_nrm, d_vip1, 1));
        }
    }

    int niter = iter;
    if (niter <= 0) niter = 1;

    ROCBLAS_CHECK(rocblas_set_pointer_mode(handles_[si], rocblas_pointer_mode_host));

    // Tridiagonal eigensolve on device.
    HIP_CHECK(hipMemcpyAsync(ws.d_steqr_D, ws.d_alpha_dev,
                             niter * sizeof(double), hipMemcpyDeviceToDevice, streams_[si]));
    HIP_CHECK(hipMemcpyAsync(ws.d_steqr_E, ws.d_beta_dev,
                             niter * sizeof(double), hipMemcpyDeviceToDevice, streams_[si]));
    rocsolver_dsteqr(handles_[si], rocblas_evect_tridiagonal, niter,
                     ws.d_steqr_D, ws.d_steqr_E, ws.d_steqr_C, niter, ws.d_steqr_info);

    // streams_[si] is non-blocking, so a bare hipMemcpy on the legacy stream
    // would NOT wait for rocsolver_dsteqr's enqueued work — it would race with
    // peer segments' in-flight kernels too. Use async + explicit sync so the
    // readback is correctly ordered against this stream's pending work and
    // doesn't depend on legacy-stream serialization.
    double energy;
    HIP_CHECK(hipMemcpyAsync(&energy, ws.d_steqr_D, sizeof(double),
                             hipMemcpyDeviceToHost, streams_[si]));
    HIP_CHECK(hipStreamSynchronize(streams_[si]));

    if constexpr (Traits::is_complex) {
        HIP_CHECK(hipMemsetAsync(ws.d_ritz_coeffs, 0,
                                 niter * sizeof(Scalar), streams_[si]));
        hipLaunchKernelGGL(promote_double_to_complex, dim3((niter + 255) / 256), dim3(256), 0, streams_[si],
                           ws.d_steqr_C, (hipDoubleComplex*)ws.d_ritz_coeffs, niter);
    } else {
        HIP_CHECK(hipMemcpyAsync(ws.d_ritz_coeffs, ws.d_steqr_C,
                                 niter * sizeof(double), hipMemcpyDeviceToDevice, streams_[si]));
    }

    Scalar one = Traits::one(), zero_val = Traits::zero();
    ROCBLAS_CHECK(Traits::gemv(handles_[si], rocblas_operation_none,
        n, niter, &one,
        d_lanczos_v, n,
        ws.d_ritz_coeffs, 1,
        &zero_val, d_theta, 1));

    ROCBLAS_CHECK(rocblas_set_pointer_mode(handles_[si], rocblas_pointer_mode_device));
    ROCBLAS_CHECK(Traits::nrm2(handles_[si], n, d_theta, 1, ws.d_nrm2_result));
    hipLaunchKernelGGL(inv_real_kernel, dim3(1), dim3(1), 0, streams_[si],
                       ws.d_nrm2_result, ws.d_inv_nrm);
    ROCBLAS_CHECK(Traits::scal_real(handles_[si], n, ws.d_inv_nrm, d_theta, 1));
    ROCBLAS_CHECK(rocblas_set_pointer_mode(handles_[si], rocblas_pointer_mode_host));

    return energy;
}

// ============================================================================
// Two-site SVD split — rocSOLVER gesvd + host-side truncation + host-side scaling
// ============================================================================

template<typename Scalar>
void PDMRGGPUBase<Scalar>::svd_split(int site, Scalar* d_theta, char direction, int si) {
    int cL = chi_L(site);
    int cR = chi_R(site + 1);
    auto& ws = workspaces_[si];

    int m = cL * d_;
    int n_svd = d_ * cR;
    int full_k = std::min(m, n_svd);
    int k = std::min(full_k, chi_max_);

    HIP_CHECK(hipMemcpyAsync(ws.d_svd_A, d_theta, m * n_svd * sizeof(Scalar),
                             hipMemcpyDeviceToDevice, streams_[si]));

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

    // Read back S only (control-flow scalar for truncation rank). Stream-
    // ordered against rocsolver_gesvd_auto's preceding work; bare hipMemcpy
    // on the legacy stream would NOT wait for non-blocking streams_[si].
    HIP_CHECK(hipMemcpyAsync(ws.h_svd_S.data(), ws.d_svd_S,
                             full_k * sizeof(RealType),
                             hipMemcpyDeviceToHost, streams_[si]));
    HIP_CHECK(hipStreamSynchronize(streams_[si]));

    int new_k = 0;
    for (int j = 0; j < k; j++) {
        if (ws.h_svd_S[j] > 1e-14) new_k++;
        else break;
    }
    if (new_k == 0) new_k = 1;

    if (direction == 'R') {
        // MPS[site] = U[:, :new_k]  (col-major contiguous, single async copy)
        allocate_mps_tensor(site, cL, new_k);
        HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site], ws.d_svd_U,
                                 (size_t)m * new_k * sizeof(Scalar),
                                 hipMemcpyDeviceToDevice, streams_[si]));

        // MPS[site+1] = S * Vh[:new_k, :]  via device kernel.
        allocate_mps_tensor(site + 1, new_k, cR);
        {
            int total = new_k * n_svd;
            int blocks = (total + 255) / 256;
            hipLaunchKernelGGL((scale_rows_by_diag_kernel<Scalar, RealType>),
                               dim3(blocks), dim3(256), 0, streams_[si],
                               ws.d_svd_S, ws.d_svd_Vh, full_k,
                               d_mps_tensors_[site + 1], new_k, new_k, n_svd);
        }
    } else {
        // MPS[site] = U[:, :new_k] * S  via device kernel.
        allocate_mps_tensor(site, cL, new_k);
        {
            int total = m * new_k;
            int blocks = (total + 255) / 256;
            hipLaunchKernelGGL((scale_cols_by_diag_kernel<Scalar, RealType>),
                               dim3(blocks), dim3(256), 0, streams_[si],
                               ws.d_svd_S, ws.d_svd_U, m,
                               d_mps_tensors_[site], m, m, new_k);
        }

        // MPS[site+1] = Vh[:new_k, :]  (row slice via extract_cols_kernel).
        allocate_mps_tensor(site + 1, new_k, cR);
        {
            int total = new_k * n_svd;
            int blocks = (total + 255) / 256;
            hipLaunchKernelGGL((extract_cols_kernel<Scalar>),
                               dim3(blocks), dim3(256), 0, streams_[si],
                               ws.d_svd_Vh, full_k,
                               d_mps_tensors_[site + 1], new_k,
                               new_k, n_svd);
        }
    }

    bond_dims_[site + 1] = new_k;
}

// ============================================================================
// Single-site SVD split — for warmup/polish
// Direction 'R': theta(cL*d, cR) → U=MPS[site], S*Vh absorbed into MPS[site+1]
// Direction 'L': theta(cL, d*cR) → Vh=MPS[site], U*S absorbed into MPS[site-1]
// ============================================================================

template<typename Scalar>
void PDMRGGPUBase<Scalar>::svd_split_single_site(int site, Scalar* d_theta,
                                                 char direction, int si) {
    int cL = chi_L(site);
    int cR = chi_R(site);
    auto& ws = workspaces_[si];

    int m, n_svd;
    if (direction == 'R') { m = cL * d_; n_svd = cR; }
    else                  { m = cL;      n_svd = d_ * cR; }
    int full_k = std::min(m, n_svd);
    int k = std::min(full_k, chi_max_);

    HIP_CHECK(hipMemcpyAsync(ws.d_svd_A, d_theta, m * n_svd * sizeof(Scalar),
                             hipMemcpyDeviceToDevice, streams_[si]));

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

    // Stream-ordered S readback (legacy hipMemcpy would race with non-blocking streams_[si]).
    HIP_CHECK(hipMemcpyAsync(ws.h_svd_S.data(), ws.d_svd_S,
                             full_k * sizeof(RealType),
                             hipMemcpyDeviceToHost, streams_[si]));
    HIP_CHECK(hipStreamSynchronize(streams_[si]));

    int new_k = 0;
    for (int j = 0; j < k; j++) {
        if (ws.h_svd_S[j] > 1e-14) new_k++;
        else break;
    }
    if (new_k == 0) new_k = 1;

    Scalar one = Traits::one(), zero_val = Traits::zero();

    if (direction == 'R') {
        int new_chi_R = new_k;
        allocate_mps_tensor(site, cL, new_chi_R);
        HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site], ws.d_svd_U,
                                 (size_t)m * new_k * sizeof(Scalar),
                                 hipMemcpyDeviceToDevice, streams_[si]));

        // Compute S*Vh on device into d_svd_work, then absorb into MPS[site+1].
        if (site + 1 < L_) {
            int total = new_k * n_svd;
            int blocks = (total + 255) / 256;
            hipLaunchKernelGGL((scale_rows_by_diag_kernel<Scalar, RealType>),
                               dim3(blocks), dim3(256), 0, streams_[si],
                               ws.d_svd_S, ws.d_svd_Vh, full_k,
                               ws.d_svd_work, new_k, new_k, n_svd);

            int next_cR = chi_R(site + 1);
            ROCBLAS_CHECK(Traits::gemm(handles_[si],
                rocblas_operation_none, rocblas_operation_none,
                new_k, d_ * next_cR, cR, &one,
                ws.d_svd_work, new_k,
                d_mps_tensors_[site + 1], cR, &zero_val,
                ws.d_T2, new_k));
            allocate_mps_tensor(site + 1, new_chi_R, next_cR);
            HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site + 1], ws.d_T2,
                                     (size_t)new_k * d_ * next_cR * sizeof(Scalar),
                                     hipMemcpyDeviceToDevice, streams_[si]));
        }
        bond_dims_[site + 1] = new_chi_R;

    } else {  // direction == 'L'
        int new_chi_L = new_k;
        allocate_mps_tensor(site, new_chi_L, cR);
        // Pack Vh[:new_k, :] into MPS[site] via device kernel.
        {
            int total = new_chi_L * n_svd;
            int blocks = (total + 255) / 256;
            hipLaunchKernelGGL((extract_cols_kernel<Scalar>),
                               dim3(blocks), dim3(256), 0, streams_[si],
                               ws.d_svd_Vh, full_k,
                               d_mps_tensors_[site], new_chi_L,
                               new_chi_L, n_svd);
        }

        // Compute U*S on device into d_svd_work, then absorb into MPS[site-1].
        if (site > 0) {
            {
                int total = m * new_k;
                int blocks = (total + 255) / 256;
                hipLaunchKernelGGL((scale_cols_by_diag_kernel<Scalar, RealType>),
                                   dim3(blocks), dim3(256), 0, streams_[si],
                                   ws.d_svd_S, ws.d_svd_U, m,
                                   ws.d_svd_work, m, m, new_k);
            }
            int prev_cL = chi_L(site - 1);
            ROCBLAS_CHECK(Traits::gemm(handles_[si],
                rocblas_operation_none, rocblas_operation_none,
                prev_cL * d_, new_k, cL, &one,
                d_mps_tensors_[site - 1], prev_cL * d_,
                ws.d_svd_work, m, &zero_val,
                ws.d_T2, prev_cL * d_));
            allocate_mps_tensor(site - 1, prev_cL, new_chi_L);
            HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site - 1], ws.d_T2,
                                     (size_t)prev_cL * d_ * new_k * sizeof(Scalar),
                                     hipMemcpyDeviceToDevice, streams_[si]));
        }
        bond_dims_[site] = new_chi_L;
    }
}

// ============================================================================
// Bond optimization (two-site)
// ============================================================================

template<typename Scalar>
double PDMRGGPUBase<Scalar>::optimize_bond(int site, char direction, int si) {
    int cL = chi_L(site);
    int cR = chi_R(site + 1);
    int theta_size = cL * d_ * d_ * cR;
    auto& ws = workspaces_[si];

    form_theta_two_site(site, si);
    double energy = lanczos_eigensolver(site, ws.d_theta, theta_size, si);
    svd_split(site, ws.d_theta, direction, si);
    return energy;
}

// ============================================================================
// Site optimization (single-site)
// ============================================================================

template<typename Scalar>
double PDMRGGPUBase<Scalar>::optimize_site_single(int site, char direction, int si) {
    int cL = chi_L(site);
    int cR = chi_R(site);
    int theta_size = cL * d_ * cR;
    auto& ws = workspaces_[si];

    HIP_CHECK(hipMemcpyAsync(ws.d_theta, d_mps_tensors_[site],
                             theta_size * sizeof(Scalar),
                             hipMemcpyDeviceToDevice, streams_[si]));

    lanczos_use_1site_ = true;
    double energy = lanczos_eigensolver(site, ws.d_theta, theta_size, si);
    lanczos_use_1site_ = false;

    svd_split_single_site(site, ws.d_theta, direction, si);
    return energy;
}

// ============================================================================
// Full-chain sweep methods (two-site)
// ============================================================================

// ============================================================================
// Full-chain single-site sweep methods (warmup and polish; per CLAUDE.md
// PDMRG rules these are the only correct full-chain sweeps in this variant —
// the dead two-site full-chain variants were removed as a foot-gun).
// ============================================================================

template<typename Scalar>
double PDMRGGPUBase<Scalar>::sweep_LR_full_1site() {
    double energy = 0.0;
    for (int site = 0; site < L_ - 1; site++) {
        energy = optimize_site_single(site, 'R', 0);
        update_left_env(site, 0);
    }
    // Last site: optimize without SVD
    {
        int cL = chi_L(L_ - 1);
        int cR = chi_R(L_ - 1);
        int theta_size = cL * d_ * cR;
        auto& ws = workspaces_[0];
        HIP_CHECK(hipMemcpyAsync(ws.d_theta, d_mps_tensors_[L_ - 1],
                                 theta_size * sizeof(Scalar),
                                 hipMemcpyDeviceToDevice, streams_[0]));
        lanczos_use_1site_ = true;
        energy = lanczos_eigensolver(L_ - 1, ws.d_theta, theta_size, 0);
        lanczos_use_1site_ = false;
        HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[L_ - 1], ws.d_theta,
                                 theta_size * sizeof(Scalar),
                                 hipMemcpyDeviceToDevice, streams_[0]));
    }
    return energy;
}

template<typename Scalar>
double PDMRGGPUBase<Scalar>::sweep_RL_full_1site() {
    double energy = 0.0;
    for (int site = L_ - 1; site >= 1; site--) {
        energy = optimize_site_single(site, 'L', 0);
        update_right_env(site, 0);
    }
    // First site: optimize without SVD
    {
        int cL = chi_L(0);
        int cR = chi_R(0);
        int theta_size = cL * d_ * cR;
        auto& ws = workspaces_[0];
        HIP_CHECK(hipMemcpyAsync(ws.d_theta, d_mps_tensors_[0],
                                 theta_size * sizeof(Scalar),
                                 hipMemcpyDeviceToDevice, streams_[0]));
        lanczos_use_1site_ = true;
        energy = lanczos_eigensolver(0, ws.d_theta, theta_size, 0);
        lanczos_use_1site_ = false;
        HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[0], ws.d_theta,
                                 theta_size * sizeof(Scalar),
                                 hipMemcpyDeviceToDevice, streams_[0]));
    }
    return energy;
}

// ============================================================================
// Segment sweep methods (restricted range, per-segment stream)
// ============================================================================

template<typename Scalar>
void PDMRGGPUBase<Scalar>::segment_sweep_LR(int seg_idx) {
    int first = seg_first_[seg_idx];
    int last = seg_last_[seg_idx];
    int si = seg_idx;

    for (int site = first; site < last; site++) {
        optimize_bond(site, 'R', si);
        update_left_env(site, si);
    }
}

template<typename Scalar>
void PDMRGGPUBase<Scalar>::segment_sweep_RL(int seg_idx) {
    int first = seg_first_[seg_idx];
    int last = seg_last_[seg_idx];
    int si = seg_idx;

    for (int site = last - 1; site >= first; site--) {
        optimize_bond(site, 'L', si);
        update_right_env(site + 1, si);
    }
}

// ============================================================================
// form_theta_with_V — θ = ψ_L · diag(V) · ψ_R  (Stoudenmire Eq. 5)
//
// All on device: upload V (small, chi_bond RealTypes) to ws.d_svd_S, use
// scale_rows_by_diag_kernel to scale rows of ψ_R into pre-allocated
// ws.d_psi_R, then GEMM with ψ_L.
// ============================================================================

template<typename Scalar>
void PDMRGGPUBase<Scalar>::form_theta_with_V(int site, int boundary_idx, int si) {
    int cL = chi_L(site);
    int chi_bond = bond_dims_[site + 1];
    int cR = chi_R(site + 1);
    auto& ws = workspaces_[si];
    auto& bs = boundary_states_[boundary_idx];

    // Upload V (host vector of length chi_bond) to ws.d_svd_S (sized chi_max).
    // Reusing d_svd_S as scratch is safe: the SVD inside the same
    // optimize_bond runs after this on the same stream and will overwrite it.
    HIP_CHECK(hipMemcpyAsync(ws.d_svd_S, bs.V.data(),
                             chi_bond * sizeof(RealType),
                             hipMemcpyHostToDevice, streams_[si]));

    // Scale rows of ψ_R[chi_bond × (d*cR)] by V[i], write to ws.d_psi_R.
    {
        int total = chi_bond * (d_ * cR);
        int blocks = (total + 255) / 256;
        hipLaunchKernelGGL((scale_rows_by_diag_kernel<Scalar, RealType>),
                           dim3(blocks), dim3(256), 0, streams_[si],
                           ws.d_svd_S, d_mps_tensors_[site + 1], chi_bond,
                           ws.d_psi_R, chi_bond, chi_bond, d_ * cR);
    }

    // Reuse ws.d_T1 as the destination for the final theta (GEMM target).

    Scalar one = Traits::one(), zero_val = Traits::zero();
    ROCBLAS_CHECK(Traits::gemm(handles_[si],
        rocblas_operation_none, rocblas_operation_none,
        cL * d_, d_ * cR, chi_bond,
        &one,
        d_mps_tensors_[site], cL * d_,
        ws.d_psi_R, chi_bond,
        &zero_val,
        ws.d_theta, cL * d_));
}

// ============================================================================
// Boundary merge+optimize (Stoudenmire V = Λ⁻¹)
// ============================================================================

template<typename Scalar>
double PDMRGGPUBase<Scalar>::merge_and_optimize_boundaries(int parity) {
    double energy = 0.0;
    int si = 0;  // boundary optimization uses stream 0

    for (int b = 0; b < (int)boundary_bonds_.size(); b++) {
        if (parity >= 0 && (b % 2) != parity) continue;

        int bsite = boundary_bonds_[b];
        int cL = chi_L(bsite);
        int cR = chi_R(bsite + 1);
        int theta_size = cL * d_ * d_ * cR;
        auto& ws = workspaces_[si];

        form_theta_with_V(bsite, b, si);
        energy = lanczos_eigensolver(bsite, ws.d_theta, theta_size, si);

        // Accurate SVD split (Stoudenmire recursive, GPU-native). Required
        // for every pdmrg variant — V = 1/clip(S, 1e-12) amplifies plain
        // gesvd's poor relative accuracy on small singular values, so the
        // recursive refinement is part of the algorithm itself, not an
        // optimization. The -base charter (no RSVD / no graph capture / no
        // sparse MPO / no fused kernels) does NOT exclude Stoudenmire.
        int m = cL * d_;
        int n_svd = d_ * cR;
        int full_k = std::min(m, n_svd);
        int k_max = std::min(full_k, chi_max_);

        accurate_svd_gpu<Scalar>(
            handles_[si], streams_[si],
            m, n_svd,
            ws.d_theta, m,
            ws.d_svd_U, m,
            ws.d_svd_S,
            ws.d_svd_Vh, full_k,
            ws.asvd);

        // Read back full_k singular values for truncation rank + boundary V.
        // Tiny D2H (≤ chi_max doubles) — same pattern as svd_split above.
        HIP_CHECK(hipMemcpyAsync(ws.h_svd_S.data(), ws.d_svd_S,
                                 full_k * sizeof(RealType),
                                 hipMemcpyDeviceToHost, streams_[si]));
        HIP_CHECK(hipStreamSynchronize(streams_[si]));

        int new_k = 0;
        for (int j = 0; j < k_max; j++) {
            if (ws.h_svd_S[j] > 1e-14) new_k++;
            else break;
        }
        if (new_k == 0) new_k = 1;

        // Direction='R': MPS[bsite] = U[:, :new_k], MPS[bsite+1] = S · Vh[:new_k, :].
        allocate_mps_tensor(bsite, cL, new_k);
        HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[bsite], ws.d_svd_U,
                                 (size_t)m * new_k * sizeof(Scalar),
                                 hipMemcpyDeviceToDevice, streams_[si]));

        allocate_mps_tensor(bsite + 1, new_k, cR);
        {
            int total = new_k * n_svd;
            int blocks = (total + 255) / 256;
            hipLaunchKernelGGL((scale_rows_by_diag_kernel<Scalar, RealType>),
                               dim3(blocks), dim3(256), 0, streams_[si],
                               ws.d_svd_S, ws.d_svd_Vh, full_k,
                               d_mps_tensors_[bsite + 1], new_k, new_k, n_svd);
        }

        bond_dims_[bsite + 1] = new_k;

        int new_chi = new_k;
        boundary_states_[b].chi = new_chi;
        boundary_states_[b].V.resize(new_chi);

        const RealType reg = RealType(1e-12);
        for (int i = 0; i < new_chi; i++) {
            RealType s_val = ws.h_svd_S[i];
            if (s_val < reg) s_val = reg;
            boundary_states_[b].V[i] = RealType(1.0) / s_val;
        }

        update_left_env(bsite, si);

        // Round-8 fix (C-new1): R_env must be built from canonical Vh, NOT
        // from S·Vh. Using S·Vh gives norm = S² ≠ I, breaking N_eff = I in
        // downstream Lanczos eigensolves. Mirrors pdmrg-gpu and pdmrg-gpu-opt.
        Scalar* d_SVh_tensor = d_mps_tensors_[bsite + 1];
        size_t vh_size = (size_t)new_k * n_svd;
        if (new_k == full_k) {
            HIP_CHECK(hipMemcpyAsync(ws.d_Vh_canonical, ws.d_svd_Vh,
                        vh_size * sizeof(Scalar),
                        hipMemcpyDeviceToDevice, streams_[si]));
        } else {
            HIP_CHECK(hipMemcpy2DAsync(
                ws.d_Vh_canonical,    new_k  * sizeof(Scalar),
                ws.d_svd_Vh,          full_k * sizeof(Scalar),
                new_k * sizeof(Scalar), n_svd,
                hipMemcpyDeviceToDevice, streams_[si]));
        }
        d_mps_tensors_[bsite + 1] = ws.d_Vh_canonical;
        update_right_env(bsite + 1, si);
        d_mps_tensors_[bsite + 1] = d_SVh_tensor;
        HIP_CHECK(hipStreamSynchronize(streams_[si]));
    }
    return energy;
}

// ============================================================================
// Main algorithm
// ============================================================================

template<typename Scalar>
double PDMRGGPUBase<Scalar>::run(int n_outer_sweeps, int n_local_sweeps,
                                 int n_warmup, int n_polish) {
    // Sweep-only timer: starts AFTER MPS+MPO+env build, stops at convergence.
    // Matches PDMRGGPU::run() so the -gpu / -gpu-base wall-time comparison is
    // apples-to-apples. Env build is reported separately as a diagnostic line.
    auto t_env_start = std::chrono::high_resolution_clock::now();
    build_initial_environments();

    auto t_start = std::chrono::high_resolution_clock::now();
    double env_time = std::chrono::duration<double>(t_start - t_env_start).count();
    printf("  Environment build: %.3f s\n", env_time);

    // Warmup: single-site sweeps (CLAUDE.md compliant; n_warmup configurable).
    double warmup_energy = 0.0;
    double prev_warmup_energy = 1e30;
    for (int sw = 0; sw < n_warmup; sw++) {
        sweep_LR_full_1site();
        warmup_energy = sweep_RL_full_1site();
        double dE = std::abs(warmup_energy - prev_warmup_energy);
        prev_warmup_energy = warmup_energy;
        if (dE < tol_ && sw > 0) break;
    }

    // Re-initialize V after warmup
    initialize_boundary_states();
    double energy_prev = warmup_energy;
    energy_ = warmup_energy;

    // Parallel sweep launcher: one CPU thread per segment with its own HIP stream.
    // Exception-safe: any exception escaping a thread function would call
    // std::terminate, abandoning peer threads' GPU work. We capture exceptions
    // via std::exception_ptr and rethrow after the per-stream sync barrier.
    auto parallel_sweep = [this](auto sweep_fn) {
        std::vector<std::thread> threads(n_segments_);
        std::vector<std::exception_ptr> exceptions(n_segments_, nullptr);
        for (int k = 0; k < n_segments_; k++) {
            threads[k] = std::thread([this, k, &sweep_fn, &exceptions]{
                try { sweep_fn(this, k); }
                catch (...) { exceptions[k] = std::current_exception(); }
            });
        }
        for (auto& t : threads) t.join();
        for (int s = 0; s < n_segments_; s++) {
            HIP_CHECK(hipStreamSynchronize(streams_[s]));
        }
        for (auto& e : exceptions) {
            if (e) std::rethrow_exception(e);
        }
    };

    bool has_odd_boundaries = ((int)boundary_bonds_.size() > 1);

    for (int outer = 0; outer < n_outer_sweeps; outer++) {
        for (int local_sw = 0; local_sw < n_local_sweeps; local_sw++) {
            // Half-sweep 1: even segments LR, odd segments RL
            parallel_sweep([](PDMRGGPUBase* self, int k) {
                if (k % 2 == 0) self->segment_sweep_LR(k);
                else             self->segment_sweep_RL(k);
            });

            if (boundary_bonds_.size() > 0) {
                energy_ = merge_and_optimize_boundaries(0);
            }

            // Half-sweep 2: even segments RL, odd segments LR
            parallel_sweep([](PDMRGGPUBase* self, int k) {
                if (k % 2 == 0) self->segment_sweep_RL(k);
                else             self->segment_sweep_LR(k);
            });

            if (has_odd_boundaries) {
                energy_ = merge_and_optimize_boundaries(1);
            }
        }

        double dE = std::abs(energy_ - energy_prev);
        if (dE < tol_ && outer > 0) {
            printf("Converged after %d outer iterations!\n", outer + 1);
            break;
        }
        energy_prev = energy_;
    }

    // Polish phase: SINGLE-SITE full-chain sweeps (CLAUDE.md compliant;
    // n_polish configurable, defaults to 0 so the caller is forced to be
    // explicit if polish is desired).
    if (n_segments_ > 1 && n_polish > 0) {
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
    printf("  env_build_sec: %.3f  timer_scope: sweep_only\n", env_time);

    return energy_;
}

// ============================================================================
// Utility
// ============================================================================

template<typename Scalar>
void PDMRGGPUBase<Scalar>::get_mps(std::vector<std::vector<Scalar>>& h_mps) const {
    h_mps.resize(L_);
    for (int i = 0; i < L_; i++) {
        int size = chi_L(i) * d_ * chi_R(i);
        h_mps[i].resize(size);
        HIP_CHECK(hipMemcpy(h_mps[i].data(), d_mps_tensors_[i],
                            size * sizeof(Scalar), hipMemcpyDeviceToHost));
    }
}

#endif // PDMRG_GPU_BASE_IMPL_H
