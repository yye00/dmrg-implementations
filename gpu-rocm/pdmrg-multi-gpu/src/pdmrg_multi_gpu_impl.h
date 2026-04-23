#ifndef PDMRG_MULTI_GPU_IMPL_H
#define PDMRG_MULTI_GPU_IMPL_H

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
// GPU kernels for batched GEMM pointer setup (eliminates CPU loops + H2D copies)
// ============================================================================

// apply_heff_single_site Step 3: per wp iteration, d pointers
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

// update_left_env Step 3: per sp iteration, D pointers
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

// Promote double eigenvector to hipDoubleComplex (for Josephson Ritz coefficients)
static __global__ void promote_double_to_complex(const double* src, hipDoubleComplex* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = make_hipDoubleComplex(src[i], 0.0);
}

// ============================================================================
// Multi-GPU data access helpers
// ============================================================================

template<typename Scalar>
int PDMRGMultiGPU<Scalar>::local_site(int global_site, int di) const {
    return global_site - devices_[di].seg_first;
}

template<typename Scalar>
int PDMRGMultiGPU<Scalar>::local_env_idx(int global_idx, int di) const {
    return global_idx - devices_[di].seg_first;
}

template<typename Scalar>
Scalar* PDMRGMultiGPU<Scalar>::get_mps(int site, int di) {
    return devices_[di].d_mps[local_site(site, di)];
}

template<typename Scalar>
Scalar* PDMRGMultiGPU<Scalar>::get_L_env(int idx, int di) {
    return devices_[di].d_L_envs[local_env_idx(idx, di)];
}

template<typename Scalar>
Scalar* PDMRGMultiGPU<Scalar>::get_R_env(int idx, int di) {
    return devices_[di].d_R_envs[local_env_idx(idx, di)];
}

template<typename Scalar>
Scalar* PDMRGMultiGPU<Scalar>::get_WW(int site, int di) {
    return devices_[di].d_WW[site];
}

template<typename Scalar>
Scalar* PDMRGMultiGPU<Scalar>::get_W_left(int site, int di) {
    return devices_[di].d_W_left[site];
}

template<typename Scalar>
Scalar* PDMRGMultiGPU<Scalar>::get_W_right(int site, int di) {
    return devices_[di].d_W_right[site];
}

// ============================================================================
// Constructor
// ============================================================================

template<typename Scalar>
PDMRGMultiGPU<Scalar>::PDMRGMultiGPU(int L, int d, int chi_max, int D_mpo, int n_devices, double tol)
    : L_(L), d_(d), chi_max_(chi_max), D_mpo_(D_mpo), tol_(tol), energy_(0.0),
      n_devices_(n_devices) {

    // Query available devices
    HIP_CHECK(hipGetDeviceCount(&n_available_devices_));
    if (n_available_devices_ <= 0) {
        throw std::runtime_error("No GPU devices found");
    }
    n_devices_ = std::min(n_devices, n_available_devices_);
    n_segments_ = n_devices_;

    if (L < 2 * n_segments_) {
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

    // Partition chain and build site-to-device mapping
    partition_chain();
    initialize_boundary_states();

    // Enable peer access between all device pairs
    setup_peer_access();

    // Workspace sizing
    int dd = d_ * d_;
    theta_size_max_ = chi_max_ * dd * chi_max_;
    max_lanczos_iter_ = std::min(100, theta_size_max_);
    use_cpu_svd_ = false;
    use_rsvd_ = false;
    lanczos_use_1site_ = false;
    rsvd_oversampling_ = 20;

    // Allocate per-device resources
    allocate_device_resources();

    // Allocate device 0 full-chain resources for warmup/polish
    HIP_CHECK(hipSetDevice(devices_[0].device_id));
    d0_mps_tensors_.resize(L, nullptr);
    d0_L_envs_.resize(L + 1, nullptr);
    d0_R_envs_.resize(L + 1, nullptr);
    d0_L_env_alloc_chi_.resize(L + 1, 0);
    d0_R_env_alloc_chi_.resize(L + 1, 0);

    size_t max_mps_sz = (size_t)chi_max_ * d_ * chi_max_ * sizeof(Scalar);
    for (int i = 0; i < L; i++) {
        HIP_CHECK(hipMalloc(&d0_mps_tensors_[i], max_mps_sz));
    }
    for (int i = 0; i <= L; i++) {
        int chi_alloc = (i == 0 || i == L) ? 1 : chi_max_;
        int sz = chi_alloc * D_mpo_ * chi_alloc;
        HIP_CHECK(hipMalloc(&d0_L_envs_[i], sz * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&d0_R_envs_[i], sz * sizeof(Scalar)));
        HIP_CHECK(hipMemset(d0_L_envs_[i], 0, sz * sizeof(Scalar)));
        HIP_CHECK(hipMemset(d0_R_envs_[i], 0, sz * sizeof(Scalar)));
        d0_L_env_alloc_chi_[i] = chi_alloc;
        d0_R_env_alloc_chi_[i] = chi_alloc;
    }
}

// ============================================================================
// Multi-GPU setup: peer access
// ============================================================================

template<typename Scalar>
void PDMRGMultiGPU<Scalar>::setup_peer_access() {
    peer_access_.resize(n_devices_ * n_devices_, false);
    for (int i = 0; i < n_devices_; i++) {
        for (int j = 0; j < n_devices_; j++) {
            if (i == j) { peer_access_[i * n_devices_ + j] = true; continue; }
            int can_access = 0;
            HIP_CHECK(hipDeviceCanAccessPeer(&can_access, i, j));
            if (can_access) {
                HIP_CHECK(hipSetDevice(i));
                hipError_t err = hipDeviceEnablePeerAccess(j, 0);
                if (err == hipSuccess || err == hipErrorPeerAccessAlreadyEnabled) {
                    peer_access_[i * n_devices_ + j] = true;
                }
            }
        }
    }
}

// ============================================================================
// Chain partitioning
// ============================================================================

template<typename Scalar>
void PDMRGMultiGPU<Scalar>::partition_chain() {
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

    // Build site-to-device mapping
    site_to_device_.resize(L_);
    for (int k = 0; k < n_segments_; k++) {
        for (int s = seg_first_[k]; s <= seg_last_[k]; s++) {
            site_to_device_[s] = k;
        }
    }
}

// ============================================================================
// Initialize boundary V = ones (before any merge, V is identity)
// ============================================================================

template<typename Scalar>
void PDMRGMultiGPU<Scalar>::initialize_boundary_states() {
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
// Per-device resource allocation
// ============================================================================

template<typename Scalar>
void PDMRGMultiGPU<Scalar>::allocate_device_resources() {
    int dd = d_ * d_;
    int t_max = D_mpo_ * dd * chi_max_ * chi_max_;
    int batch_max = D_mpo_ * dd;
    int svd_max_m = chi_max_ * d_;
    int svd_max_n = d_ * chi_max_;
    int svd_max_k = std::min(svd_max_m, svd_max_n);

    devices_.resize(n_devices_);

    for (int k = 0; k < n_devices_; k++) {
        auto& dev = devices_[k];
        dev.device_id = k;
        dev.seg_first = seg_first_[k];
        dev.seg_last = seg_last_[k];
        dev.seg_len = seg_last_[k] - seg_first_[k] + 1;

        HIP_CHECK(hipSetDevice(k));

        // Create stream and handle
        HIP_CHECK(hipStreamCreate(&dev.stream));
        ROCBLAS_CHECK(rocblas_create_handle(&dev.handle));
        ROCBLAS_CHECK(rocblas_set_stream(dev.handle, dev.stream));

        // MPS tensors (local indexing: 0..seg_len-1)
        size_t max_mps_sz = (size_t)chi_max_ * d_ * chi_max_ * sizeof(Scalar);
        dev.d_mps.resize(dev.seg_len, nullptr);
        for (int i = 0; i < dev.seg_len; i++) {
            HIP_CHECK(hipMalloc(&dev.d_mps[i], max_mps_sz));
        }

        // MPO tensors (replicated: full chain, global indexing)
        dev.d_mpo.resize(L_, nullptr);
        dev.d_W_left.resize(L_, nullptr);
        dev.d_W_right.resize(L_, nullptr);
        dev.d_WW.resize(L_ - 1, nullptr);
        // MPO will be populated in set_mpo()

        // Environment tensors (local indexing: 0..seg_len)
        // For segment [first, last], we need env indices first..last+1
        // That's seg_len+1 entries
        int n_envs = dev.seg_len + 1;
        dev.d_L_envs.resize(n_envs, nullptr);
        dev.d_R_envs.resize(n_envs, nullptr);
        dev.L_env_alloc_chi.resize(n_envs, 0);
        dev.R_env_alloc_chi.resize(n_envs, 0);

        for (int i = 0; i < n_envs; i++) {
            int global_idx = dev.seg_first + i;
            int chi_alloc = (global_idx == 0 || global_idx == L_) ? 1 : chi_max_;
            int sz = chi_alloc * D_mpo_ * chi_alloc;
            HIP_CHECK(hipMalloc(&dev.d_L_envs[i], sz * sizeof(Scalar)));
            HIP_CHECK(hipMalloc(&dev.d_R_envs[i], sz * sizeof(Scalar)));
            HIP_CHECK(hipMemset(dev.d_L_envs[i], 0, sz * sizeof(Scalar)));
            HIP_CHECK(hipMemset(dev.d_R_envs[i], 0, sz * sizeof(Scalar)));
            dev.L_env_alloc_chi[i] = chi_alloc;
            dev.R_env_alloc_chi[i] = chi_alloc;
        }

        // Workspace allocation
        HIP_CHECK(hipMalloc(&dev.d_T1, t_max * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&dev.d_T2, t_max * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&dev.d_theta, theta_size_max_ * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&dev.d_heff_result, theta_size_max_ * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&dev.d_lanczos_v, (size_t)max_lanczos_iter_ * theta_size_max_ * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&dev.d_ritz_coeffs, max_lanczos_iter_ * sizeof(Scalar)));

        // Batched GEMM pointer arrays
        HIP_CHECK(hipMalloc(&dev.d_batch_A, batch_max * sizeof(Scalar*)));
        HIP_CHECK(hipMalloc(&dev.d_batch_B, batch_max * sizeof(Scalar*)));
        HIP_CHECK(hipMalloc(&dev.d_batch_C, batch_max * sizeof(Scalar*)));
        dev.h_batch_A_pinned = nullptr;
        dev.h_batch_B_pinned = nullptr;
        dev.h_batch_C_pinned = nullptr;
        HIP_CHECK(hipMalloc(&dev.d_heff_batch_A, batch_max * sizeof(Scalar*)));
        HIP_CHECK(hipMalloc(&dev.d_heff_batch_C, batch_max * sizeof(Scalar*)));
        dev.heff_cached_site = -1;

        // Lanczos device-pointer-mode scalars
        HIP_CHECK(hipMalloc(&dev.d_dot_result, sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&dev.d_nrm2_result, sizeof(RealType)));
        HIP_CHECK(hipMalloc(&dev.d_neg_alpha, sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&dev.d_neg_overlap, sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&dev.d_inv_nrm, sizeof(RealType)));
        HIP_CHECK(hipMalloc(&dev.d_alpha_dev, max_lanczos_iter_ * sizeof(RealType)));
        HIP_CHECK(hipMalloc(&dev.d_beta_dev, max_lanczos_iter_ * sizeof(RealType)));
        HIP_CHECK(hipMalloc(&dev.d_neg_beta_scalars, max_lanczos_iter_ * sizeof(Scalar)));

        // rocsolver tridiagonal eigensolver workspace
        HIP_CHECK(hipMalloc(&dev.d_steqr_D, max_lanczos_iter_ * sizeof(double)));
        HIP_CHECK(hipMalloc(&dev.d_steqr_E, max_lanczos_iter_ * sizeof(double)));
        HIP_CHECK(hipMalloc(&dev.d_steqr_C, max_lanczos_iter_ * max_lanczos_iter_ * sizeof(double)));
        HIP_CHECK(hipMalloc(&dev.d_steqr_info, sizeof(rocblas_int)));

        HIP_CHECK(hipMalloc(&dev.d_const_one, sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&dev.d_const_zero, sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&dev.d_const_neg_one, sizeof(Scalar)));
        {
            Scalar h_one = Traits::one(), h_zero = Traits::zero();
            Scalar h_neg_one = Traits::neg(Traits::one());
            HIP_CHECK(hipMemcpy(dev.d_const_one, &h_one, sizeof(Scalar), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(dev.d_const_zero, &h_zero, sizeof(Scalar), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(dev.d_const_neg_one, &h_neg_one, sizeof(Scalar), hipMemcpyHostToDevice));
        }

        // GPU SVD workspace
        HIP_CHECK(hipMalloc(&dev.d_svd_A, theta_size_max_ * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&dev.d_svd_U, (size_t)svd_max_m * svd_max_k * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&dev.d_svd_S, svd_max_k * sizeof(RealType)));
        HIP_CHECK(hipMalloc(&dev.d_svd_Vh, (size_t)svd_max_k * svd_max_n * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&dev.d_svd_E, svd_max_k * sizeof(RealType)));
        HIP_CHECK(hipMalloc(&dev.d_svd_info, sizeof(int)));
        HIP_CHECK(hipMalloc(&dev.d_svd_work, theta_size_max_ * sizeof(Scalar)));

        // CPU SVD workspace
        dev.h_svd_A.resize(theta_size_max_);
        dev.h_svd_U.resize((size_t)svd_max_m * svd_max_k);
        dev.h_svd_S.resize(svd_max_k);
        dev.h_svd_Vh.resize((size_t)svd_max_k * svd_max_n);
        dev.h_svd_tmp.resize(std::max((size_t)svd_max_m * svd_max_k, (size_t)svd_max_k * svd_max_n));
        dev.h_svd_rwork.resize(Traits::svd_rwork_size(svd_max_m, svd_max_n));

        // Query optimal LAPACK workspace
        {
            int m = svd_max_m, n = svd_max_n;
            int lwork_query = -1;
            Scalar work_opt;
            int info;
            const char jobu = 'S', jobvt = 'S';
            Traits::lapack_gesvd(&jobu, &jobvt, &m, &n, nullptr, &m, nullptr,
                    nullptr, &m, nullptr, &svd_max_k, &work_opt, &lwork_query,
                    dev.h_svd_rwork.empty() ? nullptr : dev.h_svd_rwork.data(), &info);
            int opt_size;
            if constexpr (Traits::is_complex) {
                opt_size = (int)Traits::real_part(work_opt) + 1;
            } else {
                opt_size = (int)work_opt + 1;
            }
            dev.h_svd_work.resize(opt_size);
        }

        // Randomized truncated SVD workspace
        {
            int rsvd_r = chi_max_ + rsvd_oversampling_;
            int rsvd_m = svd_max_m;
            int rsvd_n = svd_max_n;
            dev.d_rsvd_omega = nullptr;
            dev.d_rsvd_Y = nullptr;
            dev.d_rsvd_Q = nullptr;
            dev.d_rsvd_B = nullptr;
            dev.d_rsvd_ipiv = nullptr;
            dev.d_rsvd_U_full = nullptr;
            HIP_CHECK(hipMalloc(&dev.d_rsvd_omega, (size_t)rsvd_n * rsvd_r * sizeof(Scalar)));
            HIP_CHECK(hipMalloc(&dev.d_rsvd_Y,    (size_t)rsvd_m * rsvd_r * sizeof(Scalar)));
            HIP_CHECK(hipMalloc(&dev.d_rsvd_Q,    (size_t)rsvd_m * rsvd_r * sizeof(Scalar)));
            HIP_CHECK(hipMalloc(&dev.d_rsvd_B,    (size_t)rsvd_r * rsvd_n * sizeof(Scalar)));
            HIP_CHECK(hipMalloc(&dev.d_rsvd_ipiv, (size_t)rsvd_r * sizeof(Scalar)));
            HIP_CHECK(hipMalloc(&dev.d_rsvd_U_full, (size_t)rsvd_m * rsvd_r * sizeof(Scalar)));
            dev.h_rsvd_B.resize((size_t)rsvd_r * rsvd_n);
            dev.h_rsvd_U_small.resize((size_t)rsvd_r * rsvd_r);

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
                        dev.h_svd_rwork.empty() ? nullptr : dev.h_svd_rwork.data(), &svd_info);
                int svd_opt;
                if constexpr (Traits::is_complex) {
                    svd_opt = (int)Traits::real_part(svd_work_opt) + 1;
                } else {
                    svd_opt = (int)svd_work_opt + 1;
                }
                if ((int)dev.h_svd_work.size() < svd_opt) {
                    dev.h_svd_work.resize(svd_opt);
                }
            }
        }

        // Staging buffer for boundary merge cross-device copies
        size_t boundary_staging_size = (size_t)chi_max_ * d_ * chi_max_ * sizeof(Scalar);
        HIP_CHECK(hipMalloc(&dev.d_boundary_staging, boundary_staging_size));
    }
}

// ============================================================================
// Destructor
// ============================================================================

template<typename Scalar>
PDMRGMultiGPU<Scalar>::~PDMRGMultiGPU() {
    free_gpu_resources();
}

template<typename Scalar>
void PDMRGMultiGPU<Scalar>::free_gpu_resources() {
    // Free device 0 full-chain resources
    HIP_CHECK(hipSetDevice(devices_.empty() ? 0 : devices_[0].device_id));
    for (auto ptr : d0_mps_tensors_) if (ptr) hipFree(ptr);
    for (auto ptr : d0_L_envs_) if (ptr) hipFree(ptr);
    for (auto ptr : d0_R_envs_) if (ptr) hipFree(ptr);
    d0_mps_tensors_.clear();
    d0_L_envs_.clear();
    d0_R_envs_.clear();

    // Free per-device resources
    for (int k = 0; k < (int)devices_.size(); k++) {
        auto& dev = devices_[k];
        HIP_CHECK(hipSetDevice(dev.device_id));

        for (auto ptr : dev.d_mps) if (ptr) hipFree(ptr);
        for (auto ptr : dev.d_mpo) if (ptr) hipFree(ptr);
        for (auto ptr : dev.d_W_left) if (ptr) hipFree(ptr);
        for (auto ptr : dev.d_W_right) if (ptr) hipFree(ptr);
        for (auto ptr : dev.d_WW) if (ptr) hipFree(ptr);
        for (auto ptr : dev.d_L_envs) if (ptr) hipFree(ptr);
        for (auto ptr : dev.d_R_envs) if (ptr) hipFree(ptr);

        if (dev.d_theta) hipFree(dev.d_theta);
        if (dev.d_heff_result) hipFree(dev.d_heff_result);
        if (dev.d_T1) hipFree(dev.d_T1);
        if (dev.d_T2) hipFree(dev.d_T2);
        if (dev.d_lanczos_v) hipFree(dev.d_lanczos_v);
        if (dev.d_ritz_coeffs) hipFree(dev.d_ritz_coeffs);
        if (dev.d_batch_A) hipFree(dev.d_batch_A);
        if (dev.d_batch_B) hipFree(dev.d_batch_B);
        if (dev.d_batch_C) hipFree(dev.d_batch_C);
        if (dev.d_heff_batch_A) hipFree(dev.d_heff_batch_A);
        if (dev.d_heff_batch_C) hipFree(dev.d_heff_batch_C);
        if (dev.d_dot_result) hipFree(dev.d_dot_result);
        if (dev.d_nrm2_result) hipFree(dev.d_nrm2_result);
        if (dev.d_neg_alpha) hipFree(dev.d_neg_alpha);
        if (dev.d_neg_overlap) hipFree(dev.d_neg_overlap);
        if (dev.d_inv_nrm) hipFree(dev.d_inv_nrm);
        if (dev.d_alpha_dev) hipFree(dev.d_alpha_dev);
        if (dev.d_beta_dev) hipFree(dev.d_beta_dev);
        if (dev.d_neg_beta_scalars) hipFree(dev.d_neg_beta_scalars);
        if (dev.d_steqr_D) hipFree(dev.d_steqr_D);
        if (dev.d_steqr_E) hipFree(dev.d_steqr_E);
        if (dev.d_steqr_C) hipFree(dev.d_steqr_C);
        if (dev.d_steqr_info) hipFree(dev.d_steqr_info);
        if (dev.d_const_one) hipFree(dev.d_const_one);
        if (dev.d_const_zero) hipFree(dev.d_const_zero);
        if (dev.d_const_neg_one) hipFree(dev.d_const_neg_one);
        if (dev.d_svd_A) hipFree(dev.d_svd_A);
        if (dev.d_svd_U) hipFree(dev.d_svd_U);
        if (dev.d_svd_S) hipFree(dev.d_svd_S);
        if (dev.d_svd_Vh) hipFree(dev.d_svd_Vh);
        if (dev.d_svd_E) hipFree(dev.d_svd_E);
        if (dev.d_svd_info) hipFree(dev.d_svd_info);
        if (dev.d_svd_work) hipFree(dev.d_svd_work);
        if (dev.d_rsvd_omega) hipFree(dev.d_rsvd_omega);
        if (dev.d_rsvd_Y) hipFree(dev.d_rsvd_Y);
        if (dev.d_rsvd_Q) hipFree(dev.d_rsvd_Q);
        if (dev.d_rsvd_B) hipFree(dev.d_rsvd_B);
        if (dev.d_rsvd_ipiv) hipFree(dev.d_rsvd_ipiv);
        if (dev.d_rsvd_U_full) hipFree(dev.d_rsvd_U_full);
        if (dev.d_boundary_staging) hipFree(dev.d_boundary_staging);

        rocblas_destroy_handle(dev.handle);
        hipStreamDestroy(dev.stream);
    }
}

// ============================================================================
// Memory management (device-aware)
// ============================================================================

template<typename Scalar>
void PDMRGMultiGPU<Scalar>::allocate_mps_tensor(int site, int cL, int cR, int di) {
    // MPS tensors are pre-allocated at max size; this is a no-op
    (void)site; (void)cL; (void)cR; (void)di;
}

template<typename Scalar>
void PDMRGMultiGPU<Scalar>::ensure_L_env_alloc(int site, int chi, int di) {
    int li = local_env_idx(site, di);
    if (chi > devices_[di].L_env_alloc_chi[li]) {
        HIP_CHECK(hipSetDevice(devices_[di].device_id));
        if (devices_[di].d_L_envs[li]) HIP_CHECK(hipFree(devices_[di].d_L_envs[li]));
        int sz = chi * D_mpo_ * chi;
        HIP_CHECK(hipMalloc(&devices_[di].d_L_envs[li], sz * sizeof(Scalar)));
        devices_[di].L_env_alloc_chi[li] = chi;
    }
}

template<typename Scalar>
void PDMRGMultiGPU<Scalar>::ensure_R_env_alloc(int site, int chi, int di) {
    int li = local_env_idx(site, di);
    if (chi > devices_[di].R_env_alloc_chi[li]) {
        HIP_CHECK(hipSetDevice(devices_[di].device_id));
        if (devices_[di].d_R_envs[li]) HIP_CHECK(hipFree(devices_[di].d_R_envs[li]));
        int sz = chi * D_mpo_ * chi;
        HIP_CHECK(hipMalloc(&devices_[di].d_R_envs[li], sz * sizeof(Scalar)));
        devices_[di].R_env_alloc_chi[li] = chi;
    }
}

// ============================================================================
// MPS initialization
// ============================================================================

template<typename Scalar>
void PDMRGMultiGPU<Scalar>::initialize_mps_random(double scale) {
    for (int i = 0; i < L_; i++) {
        int di = site_to_device_[i];
        int size = chi_L(i) * d_ * chi_R(i);
        std::vector<Scalar> h_A(size);
        for (int j = 0; j < size; j++) {
            h_A[j] = Traits::scale_by_real(scale, Traits::random_val());
        }
        HIP_CHECK(hipSetDevice(devices_[di].device_id));
        HIP_CHECK(hipMemcpy(get_mps(i, di), h_A.data(),
                            size * sizeof(Scalar), hipMemcpyHostToDevice));
    }
}

// ============================================================================
// MPO setup and fused two-site MPO precomputation
// ============================================================================

template<typename Scalar>
void PDMRGMultiGPU<Scalar>::set_mpo(const std::vector<Scalar*>& h_mpo_tensors) {
    int D = D_mpo_, d = d_;

    for (int i = 0; i < L_; i++) {
        int size = D * d * d * D;

        // Prepare W_left and W_right on host
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

        // Replicate MPO to all devices
        for (int k = 0; k < n_devices_; k++) {
            HIP_CHECK(hipSetDevice(devices_[k].device_id));
            HIP_CHECK(hipMalloc(&devices_[k].d_mpo[i], size * sizeof(Scalar)));
            HIP_CHECK(hipMemcpy(devices_[k].d_mpo[i], h_mpo_tensors[i],
                                size * sizeof(Scalar), hipMemcpyHostToDevice));
            HIP_CHECK(hipMalloc(&devices_[k].d_W_left[i], wm_size * sizeof(Scalar)));
            HIP_CHECK(hipMemcpy(devices_[k].d_W_left[i], h_WL.data(),
                                wm_size * sizeof(Scalar), hipMemcpyHostToDevice));
            HIP_CHECK(hipMalloc(&devices_[k].d_W_right[i], wm_size * sizeof(Scalar)));
            HIP_CHECK(hipMemcpy(devices_[k].d_W_right[i], h_WR.data(),
                                wm_size * sizeof(Scalar), hipMemcpyHostToDevice));
        }
    }

    precompute_fused_mpo(h_mpo_tensors);
}

template<typename Scalar>
void PDMRGMultiGPU<Scalar>::precompute_fused_mpo(const std::vector<Scalar*>& h_mpo_tensors) {
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

        // Replicate to all devices
        for (int k = 0; k < n_devices_; k++) {
            HIP_CHECK(hipSetDevice(devices_[k].device_id));
            HIP_CHECK(hipMalloc(&devices_[k].d_WW[bond], ww_size * sizeof(Scalar)));
            HIP_CHECK(hipMemcpy(devices_[k].d_WW[bond], h_WW.data(),
                                ww_size * sizeof(Scalar), hipMemcpyHostToDevice));
        }
    }
}

// ============================================================================
// Two-site theta formation (device-aware)
// ============================================================================

template<typename Scalar>
void PDMRGMultiGPU<Scalar>::form_theta_two_site(int site, int di) {
    int cL = chi_L(site);
    int chi_mid = bond_dims_[site + 1];
    int cR = chi_R(site + 1);
    Scalar one = Traits::one(), zero_val = Traits::zero();
    auto& dev = devices_[di];

    ROCBLAS_CHECK(Traits::gemm(dev.handle,
        rocblas_operation_none, rocblas_operation_none,
        cL * d_, d_ * cR, chi_mid,
        &one,
        get_mps(site, di), cL * d_,
        get_mps(site + 1, di), chi_mid,
        &zero_val,
        dev.d_theta, cL * d_));
}

// ============================================================================
// Two-site H_eff application (3-step with fused WW, device-aware)
// ============================================================================

template<typename Scalar>
void PDMRGMultiGPU<Scalar>::apply_heff_two_site(int site, const Scalar* d_theta_in,
                                                  Scalar* d_result, int di) {
    int cL = chi_L(site);
    int cR = chi_R(site + 1);
    int D = D_mpo_, d = d_;
    int dd = d * d;
    Scalar one = Traits::one(), zero_val = Traits::zero();
    auto& dev = devices_[di];

    Scalar* L_env = get_L_env(site, di);
    Scalar* R_env = get_R_env(site + 2, di);
    Scalar* WW = get_WW(site, di);
    Scalar* T1 = dev.d_T1;
    Scalar* T2 = dev.d_T2;

    // Step 1: Batched GEMM -- L_env^T x theta
    {
        int batch_count = D * dd;

        if (dev.heff_cached_site != site) {
            hipLaunchKernelGGL(setup_heff_A_ptrs<Scalar>, dim3(1), dim3(batch_count), 0, dev.stream,
                               dev.d_heff_batch_A, L_env, cL, dd, batch_count);
            hipLaunchKernelGGL(setup_heff_C_ptrs<Scalar>, dim3(1), dim3(batch_count), 0, dev.stream,
                               dev.d_heff_batch_C, T1, cL * cR, batch_count);
            dev.heff_cached_site = site;
        }

        hipLaunchKernelGGL(setup_heff_B_ptrs<Scalar>, dim3(1), dim3(batch_count), 0, dev.stream,
                           dev.d_batch_B, const_cast<Scalar*>(d_theta_in), cL, d, dd, batch_count);

        ROCBLAS_CHECK(Traits::gemm_batched(dev.handle,
            Traits::op_t, rocblas_operation_none,
            cL, cR, cL,
            &one,
            (const Scalar**)dev.d_heff_batch_A, cL * D,
            (const Scalar**)dev.d_batch_B, cL * dd,
            &zero_val,
            dev.d_heff_batch_C, cL,
            batch_count));
    }

    // Step 2: Dense GEMM -- T1 x WW
    ROCBLAS_CHECK(Traits::gemm(dev.handle,
        rocblas_operation_none, rocblas_operation_none,
        cL * cR, dd * D, D * dd,
        &one,
        T1, cL * cR,
        WW, D * dd,
        &zero_val,
        T2, cL * cR));

    // Step 3: Loop of GEMMs -- T2 x R_env
    for (int s1p = 0; s1p < d; s1p++) {
        for (int s2p = 0; s2p < d; s2p++) {
            for (int n = 0; n < D; n++) {
                Scalar beta = (n == 0) ? Traits::zero() : Traits::one();
                int ws_out = n * dd + s1p * d + s2p;
                ROCBLAS_CHECK(Traits::gemm(dev.handle,
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

// ============================================================================
// Left environment update (device-aware)
// ============================================================================

template<typename Scalar>
void PDMRGMultiGPU<Scalar>::update_left_env(int site, int di) {
    int chi_in = bond_dims_[site];
    int chi_out = bond_dims_[site + 1];
    int D = D_mpo_, d = d_;
    Scalar one = Traits::one(), zero_val = Traits::zero();
    auto& dev = devices_[di];

    ensure_L_env_alloc(site + 1, chi_out, di);

    Scalar* L_env = get_L_env(site, di);
    Scalar* A = get_mps(site, di);
    Scalar* W_mat = get_W_left(site, di);
    Scalar* L_new = get_L_env(site + 1, di);
    Scalar* V = dev.d_T1;
    Scalar* U = dev.d_T2;

    // Step 1: V_ws[a',b] = L_w^T[a',a] * A_s[a,b]  (batched GEMM)
    {
        int batch_count = D * d;
        hipLaunchKernelGGL(setup_lenv_ptrs<Scalar>, dim3(1), dim3(batch_count), 0, dev.stream,
                           dev.d_batch_A, dev.d_batch_B, dev.d_batch_C,
                           L_env, A, V, chi_in, chi_out, d, batch_count);
        ROCBLAS_CHECK(Traits::gemm_batched(dev.handle,
            Traits::op_t, rocblas_operation_none,
            chi_in, chi_out, chi_in,
            &one,
            (const Scalar**)dev.d_batch_A, chi_in * D,
            (const Scalar**)dev.d_batch_B, chi_in * d,
            &zero_val,
            dev.d_batch_C, chi_in,
            batch_count));
    }

    // Step 2: U = V * W_matrix
    ROCBLAS_CHECK(Traits::gemm(dev.handle,
        rocblas_operation_none, rocblas_operation_none,
        chi_in * chi_out, d * D, D * d,
        &one,
        V, chi_in * chi_out,
        W_mat, D * d,
        &zero_val,
        U, chi_in * chi_out));

    // Step 3: L_new_w'[b,b'] = sum_{a',s'} conj(U[a',ws',b])^H * A[a',s',b']  (batched)
    {
        for (int sp = 0; sp < d; sp++) {
            Scalar beta = (sp == 0) ? Traits::zero() : Traits::one();
            hipLaunchKernelGGL(setup_lenv_step3_ptrs<Scalar>, dim3(1), dim3(D), 0, dev.stream,
                               dev.d_batch_A, dev.d_batch_B, dev.d_batch_C,
                               U, A, L_new,
                               sp, d, D, chi_in * chi_out, chi_in, chi_out);
            ROCBLAS_CHECK(Traits::gemm_batched(dev.handle,
                Traits::op_h, rocblas_operation_none,
                chi_out, chi_out, chi_in,
                &one,
                (const Scalar**)dev.d_batch_A, chi_in,
                (const Scalar**)dev.d_batch_B, chi_in * d,
                &beta,
                dev.d_batch_C, chi_out * D,
                D));
        }
    }

    if constexpr (Traits::is_complex) {
        conjugate_inplace(L_new, chi_out * D * chi_out, dev.stream);
    }
}

// ============================================================================
// Right environment update (device-aware)
// ============================================================================

template<typename Scalar>
void PDMRGMultiGPU<Scalar>::update_right_env(int site, int di) {
    int chi_in = bond_dims_[site + 1];
    int chi_out = bond_dims_[site];
    int D = D_mpo_, d = d_;
    Scalar one = Traits::one(), zero_val = Traits::zero();
    auto& dev = devices_[di];

    ensure_R_env_alloc(site, chi_out, di);

    Scalar* A = get_mps(site, di);
    Scalar* R_env = get_R_env(site + 1, di);
    Scalar* W_mat = get_W_right(site, di);
    Scalar* R_new = get_R_env(site, di);
    Scalar* V = dev.d_T1;
    Scalar* U = dev.d_T2;

    // Step 1: V_ws[a,b'] = A_s[a,b] * R_w'[b,b']  (batched GEMM)
    {
        int batch_count = D * d;
        hipLaunchKernelGGL(setup_renv_ptrs<Scalar>, dim3(1), dim3(batch_count), 0, dev.stream,
                           dev.d_batch_A, dev.d_batch_B, dev.d_batch_C,
                           A, R_env, V, chi_in, chi_out, d, batch_count);
        ROCBLAS_CHECK(Traits::gemm_batched(dev.handle,
            rocblas_operation_none, rocblas_operation_none,
            chi_out, chi_in, chi_in,
            &one,
            (const Scalar**)dev.d_batch_A, chi_out * d,
            (const Scalar**)dev.d_batch_B, chi_in * D,
            &zero_val,
            dev.d_batch_C, chi_out,
            batch_count));
    }

    // Step 2: U = V * W_matrix
    ROCBLAS_CHECK(Traits::gemm(dev.handle,
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
            hipLaunchKernelGGL(setup_renv_step3_ptrs<Scalar>, dim3(1), dim3(D), 0, dev.stream,
                               dev.d_batch_A, dev.d_batch_B, dev.d_batch_C,
                               U, A, R_new,
                               sp, d, D, chi_out * chi_in, chi_out, chi_out);
            ROCBLAS_CHECK(Traits::gemm_batched(dev.handle,
                rocblas_operation_none, Traits::op_h,
                chi_out, chi_out, chi_in,
                &one,
                (const Scalar**)dev.d_batch_A, chi_out,
                (const Scalar**)dev.d_batch_B, chi_out * d,
                &beta,
                dev.d_batch_C, chi_out * D,
                D));
        }
    }
}

// ============================================================================
// Environment building (on device 0 using d0_ arrays, then scatter)
// ============================================================================

template<typename Scalar>
void PDMRGMultiGPU<Scalar>::build_initial_environments() {
    // Gather MPS to device 0 first
    gather_mps_to_device0();

    int di = 0;  // device 0
    auto& dev = devices_[di];
    HIP_CHECK(hipSetDevice(dev.device_id));

    // L[0] = trivial left boundary
    {
        std::vector<Scalar> h_L(D_mpo_, Traits::zero());
        h_L[0] = Traits::one();
        HIP_CHECK(hipMemcpy(d0_L_envs_[0], h_L.data(),
                            D_mpo_ * sizeof(Scalar), hipMemcpyHostToDevice));
    }

    // R[L] = trivial right boundary
    {
        std::vector<Scalar> h_R(D_mpo_, Traits::zero());
        h_R[D_mpo_ - 1] = Traits::one();
        HIP_CHECK(hipMemcpy(d0_R_envs_[L_], h_R.data(),
                            D_mpo_ * sizeof(Scalar), hipMemcpyHostToDevice));
    }

    // Build all L environments left-to-right on device 0 using d0_ arrays
    // We temporarily wire up the device 0 context to use d0_ arrays
    // by using a special path in update_left_env/update_right_env
    // Instead, we do explicit GEMM-based env builds using d0_ arrays directly.
    // For simplicity, we build envs by calling the device-aware methods
    // after setting up device 0 to hold the full chain.

    // Save original device 0 MPS/env pointers
    auto saved_mps = dev.d_mps;
    auto saved_L_envs = dev.d_L_envs;
    auto saved_R_envs = dev.d_R_envs;
    auto saved_L_alloc = dev.L_env_alloc_chi;
    auto saved_R_alloc = dev.R_env_alloc_chi;
    int saved_seg_first = dev.seg_first;
    int saved_seg_last = dev.seg_last;
    int saved_seg_len = dev.seg_len;

    // Temporarily set device 0 to cover full chain
    dev.d_mps = d0_mps_tensors_;
    dev.d_L_envs = d0_L_envs_;
    dev.d_R_envs = d0_R_envs_;
    dev.L_env_alloc_chi = d0_L_env_alloc_chi_;
    dev.R_env_alloc_chi = d0_R_env_alloc_chi_;
    dev.seg_first = 0;
    dev.seg_last = L_ - 1;
    dev.seg_len = L_;

    // Build all L environments left-to-right
    for (int i = 0; i < L_; i++) {
        update_left_env(i, di);
    }
    HIP_CHECK(hipStreamSynchronize(dev.stream));

    // Build all R environments right-to-left
    for (int i = L_ - 1; i >= 0; i--) {
        update_right_env(i, di);
    }
    HIP_CHECK(hipStreamSynchronize(dev.stream));

    // Copy back possibly reallocated env arrays
    d0_L_envs_ = dev.d_L_envs;
    d0_R_envs_ = dev.d_R_envs;
    d0_L_env_alloc_chi_ = dev.L_env_alloc_chi;
    d0_R_env_alloc_chi_ = dev.R_env_alloc_chi;

    // Restore device 0 segment info
    dev.d_mps = saved_mps;
    dev.d_L_envs = saved_L_envs;
    dev.d_R_envs = saved_R_envs;
    dev.L_env_alloc_chi = saved_L_alloc;
    dev.R_env_alloc_chi = saved_R_alloc;
    dev.seg_first = saved_seg_first;
    dev.seg_last = saved_seg_last;
    dev.seg_len = saved_seg_len;

    // Scatter environments from device 0 to all devices
    scatter_envs_from_device0();
}

// ============================================================================
// Cross-device transfers
// ============================================================================

template<typename Scalar>
void PDMRGMultiGPU<Scalar>::gather_mps_to_device0() {
    int d0_id = devices_[0].device_id;
    for (int k = 0; k < n_devices_; k++) {
        int src_id = devices_[k].device_id;
        for (int s = devices_[k].seg_first; s <= devices_[k].seg_last; s++) {
            int ls = local_site(s, k);
            int mps_size = chi_L(s) * d_ * chi_R(s) * sizeof(Scalar);
            if (k == 0) {
                // Same device: device-to-device copy
                HIP_CHECK(hipMemcpy(d0_mps_tensors_[s], devices_[k].d_mps[ls],
                                    mps_size, hipMemcpyDeviceToDevice));
            } else {
                HIP_CHECK(hipMemcpyPeer(d0_mps_tensors_[s], d0_id,
                                         devices_[k].d_mps[ls], src_id, mps_size));
            }
        }
    }
}

template<typename Scalar>
void PDMRGMultiGPU<Scalar>::scatter_mps_from_device0() {
    int d0_id = devices_[0].device_id;
    for (int k = 0; k < n_devices_; k++) {
        int dst_id = devices_[k].device_id;
        for (int s = devices_[k].seg_first; s <= devices_[k].seg_last; s++) {
            int ls = local_site(s, k);
            int mps_size = chi_L(s) * d_ * chi_R(s) * sizeof(Scalar);
            if (k == 0) {
                HIP_CHECK(hipMemcpy(devices_[k].d_mps[ls], d0_mps_tensors_[s],
                                    mps_size, hipMemcpyDeviceToDevice));
            } else {
                HIP_CHECK(hipMemcpyPeer(devices_[k].d_mps[ls], dst_id,
                                         d0_mps_tensors_[s], d0_id, mps_size));
            }
        }
    }
}

template<typename Scalar>
void PDMRGMultiGPU<Scalar>::gather_envs_to_device0() {
    int d0_id = devices_[0].device_id;
    for (int k = 0; k < n_devices_; k++) {
        int src_id = devices_[k].device_id;
        int n_envs = devices_[k].seg_len + 1;
        for (int li = 0; li < n_envs; li++) {
            int gi = devices_[k].seg_first + li;
            int chi = bond_dims_[gi];
            if (gi == 0) chi = 1;
            if (gi == L_) chi = 1;
            int env_size = chi * D_mpo_ * chi * sizeof(Scalar);
            if (k == 0) {
                HIP_CHECK(hipMemcpy(d0_L_envs_[gi], devices_[k].d_L_envs[li],
                                    env_size, hipMemcpyDeviceToDevice));
                HIP_CHECK(hipMemcpy(d0_R_envs_[gi], devices_[k].d_R_envs[li],
                                    env_size, hipMemcpyDeviceToDevice));
            } else {
                HIP_CHECK(hipMemcpyPeer(d0_L_envs_[gi], d0_id,
                                         devices_[k].d_L_envs[li], src_id, env_size));
                HIP_CHECK(hipMemcpyPeer(d0_R_envs_[gi], d0_id,
                                         devices_[k].d_R_envs[li], src_id, env_size));
            }
        }
    }
}

template<typename Scalar>
void PDMRGMultiGPU<Scalar>::scatter_envs_from_device0() {
    int d0_id = devices_[0].device_id;
    for (int k = 0; k < n_devices_; k++) {
        int dst_id = devices_[k].device_id;
        int n_envs = devices_[k].seg_len + 1;
        for (int li = 0; li < n_envs; li++) {
            int gi = devices_[k].seg_first + li;
            int chi = bond_dims_[gi];
            if (gi == 0) chi = 1;
            if (gi == L_) chi = 1;
            int env_size = chi * D_mpo_ * chi * sizeof(Scalar);
            if (k == 0) {
                HIP_CHECK(hipMemcpy(devices_[k].d_L_envs[li], d0_L_envs_[gi],
                                    env_size, hipMemcpyDeviceToDevice));
                HIP_CHECK(hipMemcpy(devices_[k].d_R_envs[li], d0_R_envs_[gi],
                                    env_size, hipMemcpyDeviceToDevice));
            } else {
                HIP_CHECK(hipMemcpyPeer(devices_[k].d_L_envs[li], dst_id,
                                         d0_L_envs_[gi], d0_id, env_size));
                HIP_CHECK(hipMemcpyPeer(devices_[k].d_R_envs[li], dst_id,
                                         d0_R_envs_[gi], d0_id, env_size));
            }
        }
    }
}

template<typename Scalar>
void PDMRGMultiGPU<Scalar>::copy_boundary_mps_to_device(int boundary_idx, int target_device) {
    // Copy MPS tensors at boundary bond to target device's staging buffer
    int bsite = boundary_bonds_[boundary_idx];
    int k_left = site_to_device_[bsite];
    int k_right = site_to_device_[bsite + 1];
    int d0_id = devices_[target_device].device_id;

    // Copy MPS[bsite] from owning device to target
    {
        int src_id = devices_[k_left].device_id;
        int mps_size = chi_L(bsite) * d_ * chi_R(bsite) * sizeof(Scalar);
        HIP_CHECK(hipMemcpyPeer(devices_[target_device].d_boundary_staging, d0_id,
                                 get_mps(bsite, k_left), src_id, mps_size));
    }
}

// ============================================================================
// Lanczos eigensolver (device-aware)
// ============================================================================

template<typename Scalar>
double PDMRGMultiGPU<Scalar>::lanczos_eigensolver(int site, Scalar* d_theta, int theta_size, int di) {
    int n = theta_size;
    int max_iter = std::min(max_lanczos_iter_, n);
    double tol_lanczos = 1e-12;
    double tol_eig_conv = 1e-12;
    auto& dev = devices_[di];

    Scalar* d_lanczos_v = dev.d_lanczos_v;

    // v[0] = theta / ||theta||
    double norm;
    ROCBLAS_CHECK(rocblas_set_pointer_mode(dev.handle, rocblas_pointer_mode_device));
    ROCBLAS_CHECK(Traits::nrm2(dev.handle, n, d_theta, 1, dev.d_nrm2_result));
    ROCBLAS_CHECK(rocblas_set_pointer_mode(dev.handle, rocblas_pointer_mode_host));

    HIP_CHECK(hipMemcpy(&norm, dev.d_nrm2_result, sizeof(double), hipMemcpyDeviceToHost));
    if (norm < 1e-14) {
        std::vector<Scalar> h_init(n);
        srand(42 + site);
        for (int i = 0; i < n; i++) h_init[i] = Traits::random_val();
        HIP_CHECK(hipMemcpy(d_theta, h_init.data(), n * sizeof(Scalar), hipMemcpyHostToDevice));
        ROCBLAS_CHECK(Traits::nrm2(dev.handle, n, d_theta, 1, &norm));
    }

    // Normalize using device pointer mode
    ROCBLAS_CHECK(rocblas_set_pointer_mode(dev.handle, rocblas_pointer_mode_device));
    hipLaunchKernelGGL(inv_real_kernel, dim3(1), dim3(1), 0, dev.stream,
                       dev.d_nrm2_result, dev.d_inv_nrm);
    ROCBLAS_CHECK(Traits::scal_real(dev.handle, n, dev.d_inv_nrm, d_theta, 1));
    ROCBLAS_CHECK(rocblas_set_pointer_mode(dev.handle, rocblas_pointer_mode_host));
    HIP_CHECK(hipMemcpyAsync(d_lanczos_v, d_theta, n * sizeof(Scalar), hipMemcpyDeviceToDevice, dev.stream));

    double prev_energy = 1e30;
    int iter;

    for (iter = 0; iter < max_iter; iter++) {
        Scalar* d_vi = d_lanczos_v + (size_t)iter * n;

        // w = H|v_i>
        if (lanczos_use_1site_)
            apply_heff_single_site(site, d_vi, dev.d_heff_result, di);
        else
            apply_heff_two_site(site, d_vi, dev.d_heff_result, di);

        ROCBLAS_CHECK(rocblas_set_pointer_mode(dev.handle, rocblas_pointer_mode_device));

        // alpha_i = <v_i|w>
        ROCBLAS_CHECK(Traits::dot(dev.handle, n, d_vi, 1, dev.d_heff_result, 1, dev.d_dot_result));

        hipLaunchKernelGGL(lanczos_process_alpha_kernel<Scalar>, dim3(1), dim3(1), 0, dev.stream,
                           dev.d_dot_result, dev.d_neg_alpha, dev.d_alpha_dev, iter);

        // w -= alpha_i * v_i
        ROCBLAS_CHECK(Traits::axpy(dev.handle, n, dev.d_neg_alpha, d_vi, 1, dev.d_heff_result, 1));

        // w -= beta_{i-1} * v_{i-1}
        if (iter > 0) {
            ROCBLAS_CHECK(Traits::axpy(dev.handle, n,
                dev.d_neg_beta_scalars + (iter - 1),
                d_lanczos_v + (size_t)(iter - 1) * n, 1,
                dev.d_heff_result, 1));
        }

        // Full reorthogonalization
        if (iter > 0) {
            ROCBLAS_CHECK(Traits::gemv(dev.handle, Traits::op_h,
                n, iter + 1, dev.d_const_one,
                d_lanczos_v, n,
                dev.d_heff_result, 1,
                dev.d_const_zero, dev.d_ritz_coeffs, 1));
            ROCBLAS_CHECK(Traits::gemv(dev.handle, rocblas_operation_none,
                n, iter + 1, dev.d_const_neg_one,
                d_lanczos_v, n,
                dev.d_ritz_coeffs, 1,
                dev.d_const_one, dev.d_heff_result, 1));
        } else {
            ROCBLAS_CHECK(Traits::dot(dev.handle, n, d_lanczos_v, 1, dev.d_heff_result, 1, dev.d_dot_result));
            hipLaunchKernelGGL(negate_scalar_kernel<Scalar>, dim3(1), dim3(1), 0, dev.stream,
                               dev.d_dot_result, dev.d_neg_overlap);
            ROCBLAS_CHECK(Traits::axpy(dev.handle, n, dev.d_neg_overlap, d_lanczos_v, 1, dev.d_heff_result, 1));
        }

        // beta_i = ||w||
        ROCBLAS_CHECK(Traits::nrm2(dev.handle, n, dev.d_heff_result, 1, dev.d_nrm2_result));

        hipLaunchKernelGGL(lanczos_process_beta_kernel<Scalar>, dim3(1), dim3(1), 0, dev.stream,
                           dev.d_nrm2_result, dev.d_inv_nrm, dev.d_beta_dev, dev.d_neg_beta_scalars, iter);

        // v_{i+1} = w / beta_i
        if (iter + 1 < max_iter) {
            Scalar* d_vip1 = d_lanczos_v + (size_t)(iter + 1) * n;
            HIP_CHECK(hipMemcpyAsync(d_vip1, dev.d_heff_result, n * sizeof(Scalar), hipMemcpyDeviceToDevice, dev.stream));
            ROCBLAS_CHECK(Traits::scal_real(dev.handle, n, dev.d_inv_nrm, d_vip1, 1));
        }

        ROCBLAS_CHECK(rocblas_set_pointer_mode(dev.handle, rocblas_pointer_mode_host));

        // Convergence check every 3 iterations after iter >= 4
        if (iter >= 4 && iter % 3 == 0) {
            HIP_CHECK(hipStreamSynchronize(dev.stream));

            int ncheck = iter + 1;

            hipLaunchKernelGGL(lanczos_check_beta, dim3(1), dim3(1), 0, dev.stream,
                               dev.d_beta_dev, ncheck, tol_lanczos, dev.d_steqr_info);
            rocblas_int h_beta_idx;
            HIP_CHECK(hipMemcpy(&h_beta_idx, dev.d_steqr_info, sizeof(rocblas_int), hipMemcpyDeviceToHost));
            if (h_beta_idx > 0) { iter = h_beta_idx; break; }

            HIP_CHECK(hipMemcpyAsync(dev.d_steqr_D, dev.d_alpha_dev, ncheck * sizeof(double), hipMemcpyDeviceToDevice, dev.stream));
            HIP_CHECK(hipMemcpyAsync(dev.d_steqr_E, dev.d_beta_dev, ncheck * sizeof(double), hipMemcpyDeviceToDevice, dev.stream));
            rocsolver_dsteqr(dev.handle, rocblas_evect_none, ncheck,
                             dev.d_steqr_D, dev.d_steqr_E, nullptr, ncheck, dev.d_steqr_info);
            rocblas_int h_info_chk;
            double cur_energy;
            HIP_CHECK(hipMemcpy(&h_info_chk, dev.d_steqr_info, sizeof(rocblas_int), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(&cur_energy, dev.d_steqr_D, sizeof(double), hipMemcpyDeviceToHost));
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

    HIP_CHECK(hipStreamSynchronize(dev.stream));

    // Solve tridiagonal eigenvalue problem on GPU
    HIP_CHECK(hipMemcpyAsync(dev.d_steqr_D, dev.d_alpha_dev, niter * sizeof(double), hipMemcpyDeviceToDevice, dev.stream));
    HIP_CHECK(hipMemcpyAsync(dev.d_steqr_E, dev.d_beta_dev, niter * sizeof(double), hipMemcpyDeviceToDevice, dev.stream));
    rocsolver_dsteqr(dev.handle, rocblas_evect_tridiagonal, niter,
                     dev.d_steqr_D, dev.d_steqr_E, dev.d_steqr_C, niter, dev.d_steqr_info);

    rocblas_int h_steqr_info;
    HIP_CHECK(hipMemcpy(&h_steqr_info, dev.d_steqr_info, sizeof(rocblas_int), hipMemcpyDeviceToHost));
    if (h_steqr_info != 0) {
        throw std::runtime_error("rocsolver_dsteqr failed with info = " + std::to_string(h_steqr_info));
    }

    double energy;
    HIP_CHECK(hipMemcpy(&energy, dev.d_steqr_D, sizeof(double), hipMemcpyDeviceToHost));

    // Ritz coefficients
    if constexpr (std::is_same_v<Scalar, double>) {
        HIP_CHECK(hipMemcpyAsync(dev.d_ritz_coeffs, dev.d_steqr_C, niter * sizeof(double), hipMemcpyDeviceToDevice, dev.stream));
    } else {
        int blk = (niter + 63) / 64;
        hipLaunchKernelGGL(promote_double_to_complex, dim3(blk), dim3(64), 0, dev.stream,
                           dev.d_steqr_C, (hipDoubleComplex*)dev.d_ritz_coeffs, niter);
    }

    ROCBLAS_CHECK(rocblas_set_pointer_mode(dev.handle, rocblas_pointer_mode_device));
    ROCBLAS_CHECK(Traits::gemv(
        dev.handle, rocblas_operation_none,
        n, niter, dev.d_const_one,
        d_lanczos_v, n,
        dev.d_ritz_coeffs, 1,
        dev.d_const_zero, d_theta, 1
    ));

    // Normalize
    ROCBLAS_CHECK(Traits::nrm2(dev.handle, n, d_theta, 1, dev.d_nrm2_result));
    hipLaunchKernelGGL(inv_real_kernel, dim3(1), dim3(1), 0, dev.stream,
                       dev.d_nrm2_result, dev.d_inv_nrm);
    ROCBLAS_CHECK(Traits::scal_real(dev.handle, n, dev.d_inv_nrm, d_theta, 1));
    ROCBLAS_CHECK(rocblas_set_pointer_mode(dev.handle, rocblas_pointer_mode_host));

    return energy;
}

// ============================================================================
// SVD split (device-aware)
// ============================================================================

template<typename Scalar>
void PDMRGMultiGPU<Scalar>::svd_split(int site, Scalar* d_theta, char direction, int di) {
    int cL = chi_L(site);
    int cR = chi_R(site + 1);
    auto& dev = devices_[di];

    int m = cL * d_;
    int n_svd = d_ * cR;
    int full_k = std::min(m, n_svd);
    int k = std::min(full_k, chi_max_);

    Scalar* h_U_data = nullptr;
    RealType* h_S_data = nullptr;
    Scalar* h_Vh_data = nullptr;
    bool gpu_svd_path = false;

    if (use_cpu_svd_) {
        HIP_CHECK(hipMemcpyAsync(dev.h_svd_A.data(), d_theta, m * n_svd * sizeof(Scalar),
                                  hipMemcpyDeviceToHost, dev.stream));
        HIP_CHECK(hipStreamSynchronize(dev.stream));

        int lwork = (int)dev.h_svd_work.size();
        int info;
        const char jobu = 'S', jobvt = 'S';
        Traits::lapack_gesvd(&jobu, &jobvt, &m, &n_svd, dev.h_svd_A.data(), &m,
                dev.h_svd_S.data(), dev.h_svd_U.data(), &m, dev.h_svd_Vh.data(), &full_k,
                dev.h_svd_work.data(), &lwork,
                dev.h_svd_rwork.empty() ? nullptr : dev.h_svd_rwork.data(), &info);

        h_U_data = dev.h_svd_U.data();
        h_S_data = dev.h_svd_S.data();
        h_Vh_data = dev.h_svd_Vh.data();
    } else {
        HIP_CHECK(hipMemcpyAsync(dev.d_svd_A, d_theta, m * n_svd * sizeof(Scalar),
                                  hipMemcpyDeviceToDevice, dev.stream));

        Traits::rocsolver_gesvd(dev.handle,
            rocblas_svect_singular, rocblas_svect_singular,
            m, n_svd,
            dev.d_svd_A, m,
            dev.d_svd_S,
            dev.d_svd_U, m,
            dev.d_svd_Vh, full_k,
            dev.d_svd_E,
            rocblas_outofplace,
            dev.d_svd_info);

        gpu_svd_path = true;
    }

    // Truncation
    int new_k;
    if (gpu_svd_path) {
        hipLaunchKernelGGL(svd_truncate_kernel<RealType>, dim3(1), dim3(1), 0, dev.stream,
                           dev.d_svd_S, k, 1e-14, dev.d_svd_info);
        HIP_CHECK(hipMemcpy(&new_k, dev.d_svd_info, sizeof(int), hipMemcpyDeviceToHost));
    } else {
        new_k = k;
        for (int i = 0; i < new_k; i++) {
            if (h_S_data[i] < 1e-14) { new_k = i; break; }
        }
        if (new_k == 0) new_k = 1;
    }

    if (gpu_svd_path) {
        if (direction == 'R') {
            allocate_mps_tensor(site, cL, new_k, di);
            HIP_CHECK(hipMemcpyAsync(get_mps(site, di), dev.d_svd_U,
                        (size_t)m * new_k * sizeof(Scalar), hipMemcpyDeviceToDevice, dev.stream));
            allocate_mps_tensor(site + 1, new_k, cR, di);
            scale_rows_by_real(dev.d_svd_Vh, full_k, dev.d_svd_S,
                               get_mps(site + 1, di), new_k, new_k, n_svd, dev.stream);
        } else {
            allocate_mps_tensor(site, cL, new_k, di);
            scale_columns_by_real(dev.d_svd_U, m, dev.d_svd_S,
                                  get_mps(site, di), m, m, new_k, dev.stream);
            allocate_mps_tensor(site + 1, new_k, cR, di);
            if (new_k == full_k) {
                HIP_CHECK(hipMemcpyAsync(get_mps(site + 1, di), dev.d_svd_Vh,
                            (size_t)full_k * n_svd * sizeof(Scalar), hipMemcpyDeviceToDevice, dev.stream));
            } else {
                HIP_CHECK(hipMemcpy2DAsync(
                    get_mps(site + 1, di), new_k * sizeof(Scalar),
                    dev.d_svd_Vh, full_k * sizeof(Scalar),
                    new_k * sizeof(Scalar), n_svd,
                    hipMemcpyDeviceToDevice, dev.stream));
            }
        }
    } else {
        // CPU SVD path
        HIP_CHECK(hipMemcpyAsync(dev.d_svd_S, h_S_data, new_k * sizeof(RealType),
                                  hipMemcpyHostToDevice, dev.stream));

        if (direction == 'R') {
            allocate_mps_tensor(site, cL, new_k, di);
            HIP_CHECK(hipMemcpyAsync(get_mps(site, di), h_U_data,
                        m * new_k * sizeof(Scalar), hipMemcpyHostToDevice, dev.stream));

            if (new_k == full_k) {
                HIP_CHECK(hipMemcpyAsync(dev.d_svd_work, h_Vh_data,
                            (size_t)full_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice, dev.stream));
            } else {
                for (int j = 0; j < n_svd; j++)
                    for (int i = 0; i < new_k; i++)
                        dev.h_svd_tmp[i + j * new_k] = h_Vh_data[i + j * full_k];
                HIP_CHECK(hipMemcpyAsync(dev.d_svd_work, dev.h_svd_tmp.data(),
                            (size_t)new_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice, dev.stream));
            }
            allocate_mps_tensor(site + 1, new_k, cR, di);
            int vh_ld = (new_k == full_k) ? full_k : new_k;
            scale_rows_by_real(dev.d_svd_work, vh_ld, dev.d_svd_S,
                               get_mps(site + 1, di), new_k, new_k, n_svd, dev.stream);

        } else {
            allocate_mps_tensor(site, cL, new_k, di);
            HIP_CHECK(hipMemcpyAsync(dev.d_svd_work, h_U_data,
                        (size_t)m * new_k * sizeof(Scalar), hipMemcpyHostToDevice, dev.stream));
            scale_columns_by_real(dev.d_svd_work, m, dev.d_svd_S,
                                  get_mps(site, di), m, m, new_k, dev.stream);

            allocate_mps_tensor(site + 1, new_k, cR, di);
            if (new_k == full_k) {
                HIP_CHECK(hipMemcpyAsync(get_mps(site + 1, di), h_Vh_data,
                            (size_t)full_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice, dev.stream));
            } else {
                for (int j = 0; j < n_svd; j++)
                    for (int i = 0; i < new_k; i++)
                        dev.h_svd_tmp[i + j * new_k] = h_Vh_data[i + j * full_k];
                HIP_CHECK(hipMemcpyAsync(get_mps(site + 1, di), dev.h_svd_tmp.data(),
                            (size_t)new_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice, dev.stream));
            }
        }
    }

    bond_dims_[site + 1] = new_k;
    dev.heff_cached_site = -1;
}

// ============================================================================
// Randomized truncated SVD split (Halko-Martinsson-Tropp, device-aware)
// ============================================================================

template<typename Scalar>
void PDMRGMultiGPU<Scalar>::rsvd_split(int site, Scalar* d_theta, char direction, int di) {
    int cL = chi_L(site);
    int cR = chi_R(site + 1);
    int m = cL * d_;
    int n_svd = d_ * cR;
    int full_k = std::min(m, n_svd);
    int k = std::min(full_k, chi_max_);
    auto& dev = devices_[di];

    if (full_k <= k + rsvd_oversampling_ || m <= 2 * k) {
        svd_split(site, d_theta, direction, di);
        return;
    }

    int r = k + rsvd_oversampling_;

    // Step 1: Generate random Omega
    {
        std::vector<Scalar> h_omega(n_svd * r);
        for (int i = 0; i < n_svd * r; i++) {
            h_omega[i] = Traits::random_val();
        }
        HIP_CHECK(hipMemcpyAsync(dev.d_rsvd_omega, h_omega.data(),
                            n_svd * r * sizeof(Scalar), hipMemcpyHostToDevice, dev.stream));
    }

    // Step 2: Y = theta @ Omega
    {
        Scalar one = Traits::one(), zero_val = Traits::zero();
        ROCBLAS_CHECK(Traits::gemm(dev.handle,
            rocblas_operation_none, rocblas_operation_none,
            m, r, n_svd, &one,
            d_theta, m,
            dev.d_rsvd_omega, n_svd,
            &zero_val,
            dev.d_rsvd_Y, m));
        HIP_CHECK(hipStreamSynchronize(dev.stream));
    }

    // Step 3: QR factorization
    HIP_CHECK(hipMemcpyAsync(dev.d_rsvd_Q, dev.d_rsvd_Y, (size_t)m * r * sizeof(Scalar),
                              hipMemcpyDeviceToDevice, dev.stream));
    ROCBLAS_CHECK(Traits::rocsolver_geqrf(dev.handle, m, r, dev.d_rsvd_Q, m, dev.d_rsvd_ipiv));
    ROCBLAS_CHECK(Traits::rocsolver_orgqr(dev.handle, m, r, r, dev.d_rsvd_Q, m, dev.d_rsvd_ipiv));
    HIP_CHECK(hipStreamSynchronize(dev.stream));

    // Step 4: B = Q^H @ theta
    {
        Scalar one = Traits::one(), zero_val = Traits::zero();
        ROCBLAS_CHECK(Traits::gemm(dev.handle,
            Traits::op_h, rocblas_operation_none,
            r, n_svd, m, &one,
            dev.d_rsvd_Q, m,
            d_theta, m,
            &zero_val,
            dev.d_rsvd_B, r));
        HIP_CHECK(hipStreamSynchronize(dev.stream));
    }

    // Step 5: SVD of B on CPU
    HIP_CHECK(hipMemcpy(dev.h_rsvd_B.data(), dev.d_rsvd_B, r * n_svd * sizeof(Scalar), hipMemcpyDeviceToHost));

    int small_k = std::min(r, n_svd);
    {
        int lwork = (int)dev.h_svd_work.size();
        int info;
        const char jobu = 'S', jobvt = 'S';
        Traits::lapack_gesvd(&jobu, &jobvt, &r, &n_svd, dev.h_rsvd_B.data(), &r, dev.h_svd_S.data(),
                dev.h_rsvd_U_small.data(), &r, dev.h_svd_Vh.data(), &small_k,
                dev.h_svd_work.data(), &lwork,
                dev.h_svd_rwork.empty() ? nullptr : dev.h_svd_rwork.data(), &info);
        if (info != 0) {
            svd_split(site, d_theta, direction, di);
            return;
        }
    }

    // Step 6: U_full = Q @ U_small
    {
        HIP_CHECK(hipMemcpyAsync(dev.d_rsvd_B, dev.h_rsvd_U_small.data(),
                            (size_t)r * small_k * sizeof(Scalar), hipMemcpyHostToDevice, dev.stream));
        Scalar one = Traits::one(), zero_val = Traits::zero();
        ROCBLAS_CHECK(Traits::gemm(dev.handle,
            rocblas_operation_none, rocblas_operation_none,
            m, small_k, r, &one,
            dev.d_rsvd_Q, m,
            dev.d_rsvd_B, r,
            &zero_val,
            dev.d_rsvd_U_full, m));
        HIP_CHECK(hipStreamSynchronize(dev.stream));
    }

    RealType* h_S_data = dev.h_svd_S.data();
    Scalar* h_Vh_data = dev.h_svd_Vh.data();

    // Truncation
    int new_k = k;
    for (int i = 0; i < new_k; i++) {
        if (h_S_data[i] < 1e-14) { new_k = i; break; }
    }
    if (new_k == 0) new_k = 1;

    HIP_CHECK(hipMemcpyAsync(dev.d_svd_S, h_S_data, new_k * sizeof(RealType),
                              hipMemcpyHostToDevice, dev.stream));

    if (direction == 'R') {
        allocate_mps_tensor(site, cL, new_k, di);
        HIP_CHECK(hipMemcpyAsync(get_mps(site, di), dev.d_rsvd_U_full,
                            (size_t)m * new_k * sizeof(Scalar), hipMemcpyDeviceToDevice, dev.stream));

        if (new_k == small_k) {
            HIP_CHECK(hipMemcpyAsync(dev.d_svd_work, h_Vh_data,
                        (size_t)small_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice, dev.stream));
        } else {
            for (int j = 0; j < n_svd; j++)
                for (int i = 0; i < new_k; i++)
                    dev.h_svd_tmp[i + j * new_k] = h_Vh_data[i + j * small_k];
            HIP_CHECK(hipMemcpyAsync(dev.d_svd_work, dev.h_svd_tmp.data(),
                        (size_t)new_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice, dev.stream));
        }
        allocate_mps_tensor(site + 1, new_k, cR, di);
        int vh_ld = (new_k == small_k) ? small_k : new_k;
        scale_rows_by_real(dev.d_svd_work, vh_ld, dev.d_svd_S,
                           get_mps(site + 1, di), new_k, new_k, n_svd, dev.stream);

    } else {
        allocate_mps_tensor(site, cL, new_k, di);
        scale_columns_by_real(dev.d_rsvd_U_full, m, dev.d_svd_S,
                              get_mps(site, di), m, m, new_k, dev.stream);

        allocate_mps_tensor(site + 1, new_k, cR, di);
        if (new_k == small_k) {
            HIP_CHECK(hipMemcpyAsync(get_mps(site + 1, di), h_Vh_data,
                                (size_t)small_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice, dev.stream));
        } else {
            for (int j = 0; j < n_svd; j++)
                for (int i = 0; i < new_k; i++)
                    dev.h_svd_tmp[i + j * new_k] = h_Vh_data[i + j * small_k];
            HIP_CHECK(hipMemcpyAsync(get_mps(site + 1, di), dev.h_svd_tmp.data(),
                                (size_t)new_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice, dev.stream));
        }
    }

    bond_dims_[site + 1] = new_k;
    dev.heff_cached_site = -1;
}

// ============================================================================
// Bond optimization (device-aware)
// ============================================================================

template<typename Scalar>
double PDMRGMultiGPU<Scalar>::optimize_bond(int site, char direction, int di) {
    int cL = chi_L(site);
    int cR = chi_R(site + 1);
    int theta_size = cL * d_ * d_ * cR;
    auto& dev = devices_[di];

    form_theta_two_site(site, di);
    double energy = lanczos_eigensolver(site, dev.d_theta, theta_size, di);
    if (use_rsvd_)
        rsvd_split(site, dev.d_theta, direction, di);
    else
        svd_split(site, dev.d_theta, direction, di);

    return energy;
}

// ============================================================================
// Single-site apply_heff (device-aware)
// ============================================================================

template<typename Scalar>
void PDMRGMultiGPU<Scalar>::apply_heff_single_site(int site, const Scalar* d_theta_in,
                                                     Scalar* d_result, int di) {
    int cL = chi_L(site);
    int cR = chi_R(site);
    int D = D_mpo_, d = d_;
    Scalar one = Traits::one(), zero_val = Traits::zero();
    auto& dev = devices_[di];

    Scalar* L_env = get_L_env(site, di);
    Scalar* R_env = get_R_env(site + 1, di);
    Scalar* W_mat = get_W_left(site, di);
    Scalar* V = dev.d_T1;
    Scalar* U = dev.d_T2;

    // Step 1: V_ws[a',b] = L_w^T[a',a] * theta_s[a,b]  (D*d batched GEMMs)
    {
        int batch_count = D * d;
        hipLaunchKernelGGL(setup_lenv_ptrs<Scalar>, dim3(1), dim3(batch_count), 0, dev.stream,
                           dev.d_batch_A, dev.d_batch_B, dev.d_batch_C,
                           L_env, const_cast<Scalar*>(d_theta_in), V, cL, cR, d, batch_count);
        ROCBLAS_CHECK(Traits::gemm_batched(dev.handle,
            Traits::op_t, rocblas_operation_none,
            cL, cR, cL,
            &one,
            (const Scalar**)dev.d_batch_A, cL * D,
            (const Scalar**)dev.d_batch_B, cL * d,
            &zero_val,
            dev.d_batch_C, cL,
            batch_count));
    }

    // Step 2: U = V * W_matrix
    ROCBLAS_CHECK(Traits::gemm(dev.handle,
        rocblas_operation_none, rocblas_operation_none,
        cL * cR, d * D, D * d,
        &one,
        V, cL * cR,
        W_mat, D * d,
        &zero_val,
        U, cL * cR));

    // Step 3: result_s'[a',b'] = sum_w' U_{w'd+s'}[a',b] * R_w'[b,b']
    {
        for (int wp = 0; wp < D; wp++) {
            Scalar beta = (wp == 0) ? Traits::zero() : Traits::one();
            hipLaunchKernelGGL(setup_heff_ss_step3_ptrs<Scalar>, dim3(1), dim3(d), 0, dev.stream,
                               dev.d_batch_A, dev.d_batch_B, dev.d_batch_C,
                               U, R_env, d_result,
                               wp, d, cL * cR, cR, cL);
            ROCBLAS_CHECK(Traits::gemm_batched(dev.handle,
                rocblas_operation_none, rocblas_operation_none,
                cL, cR, cR,
                &one,
                (const Scalar**)dev.d_batch_A, cL,
                (const Scalar**)dev.d_batch_B, cR * D,
                &beta,
                dev.d_batch_C, cL * d,
                d));
        }
    }
}

// ============================================================================
// Single-site SVD split (device-aware)
// ============================================================================

template<typename Scalar>
void PDMRGMultiGPU<Scalar>::svd_split_single_site(int site, Scalar* d_theta, char direction, int di) {
    int cL = chi_L(site);
    int cR = chi_R(site);
    auto& dev = devices_[di];

    int m, n_svd;
    if (direction == 'R') { m = cL * d_; n_svd = cR; }
    else                  { m = cL;      n_svd = d_ * cR; }
    int full_k = std::min(m, n_svd);
    int k = std::min(full_k, chi_max_);

    Scalar* h_U_data = nullptr;
    RealType* h_S_data = nullptr;
    Scalar* h_Vh_data = nullptr;

    if (use_cpu_svd_) {
        HIP_CHECK(hipMemcpyAsync(dev.h_svd_A.data(), d_theta, m * n_svd * sizeof(Scalar),
                                  hipMemcpyDeviceToHost, dev.stream));
        HIP_CHECK(hipStreamSynchronize(dev.stream));

        int lwork = (int)dev.h_svd_work.size();
        int info;
        const char jobu = 'S', jobvt = 'S';
        Traits::lapack_gesvd(&jobu, &jobvt, &m, &n_svd, dev.h_svd_A.data(), &m,
                dev.h_svd_S.data(), dev.h_svd_U.data(), &m, dev.h_svd_Vh.data(), &full_k,
                dev.h_svd_work.data(), &lwork,
                dev.h_svd_rwork.empty() ? nullptr : dev.h_svd_rwork.data(), &info);

        h_U_data = dev.h_svd_U.data();
        h_S_data = dev.h_svd_S.data();
        h_Vh_data = dev.h_svd_Vh.data();
    } else {
        HIP_CHECK(hipMemcpyAsync(dev.d_svd_A, d_theta, m * n_svd * sizeof(Scalar),
                                  hipMemcpyDeviceToDevice, dev.stream));
        Traits::rocsolver_gesvd(dev.handle,
            rocblas_svect_singular, rocblas_svect_singular,
            m, n_svd,
            dev.d_svd_A, m,
            dev.d_svd_S,
            dev.d_svd_U, m,
            dev.d_svd_Vh, full_k,
            dev.d_svd_E,
            rocblas_outofplace,
            dev.d_svd_info);
    }

    // Truncation
    int new_k;
    if (!use_cpu_svd_) {
        hipLaunchKernelGGL(svd_truncate_kernel<RealType>, dim3(1), dim3(1), 0, dev.stream,
                           dev.d_svd_S, k, 1e-14, dev.d_svd_info);
        HIP_CHECK(hipMemcpy(&new_k, dev.d_svd_info, sizeof(int), hipMemcpyDeviceToHost));
    } else {
        h_S_data = dev.h_svd_S.data();
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
            allocate_mps_tensor(site, cL, new_chi_R, di);
            HIP_CHECK(hipMemcpyAsync(get_mps(site, di), h_U_data,
                        m * new_k * sizeof(Scalar), hipMemcpyHostToDevice, dev.stream));

            for (int j = 0; j < n_svd; j++)
                for (int i = 0; i < new_k; i++)
                    dev.h_svd_tmp[i + j * new_k] = Traits::scale_by_real(h_S_data[i], h_Vh_data[i + j * full_k]);

            HIP_CHECK(hipMemcpyAsync(dev.d_svd_work, dev.h_svd_tmp.data(),
                        new_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice, dev.stream));
        } else {
            allocate_mps_tensor(site, cL, new_chi_R, di);
            HIP_CHECK(hipMemcpyAsync(get_mps(site, di), dev.d_svd_U,
                        (size_t)m * new_k * sizeof(Scalar), hipMemcpyDeviceToDevice, dev.stream));
            scale_rows_by_real(dev.d_svd_Vh, full_k, dev.d_svd_S,
                               dev.d_svd_work, new_k, new_k, n_svd, dev.stream);
        }

        // Absorb S*Vh into MPS[site+1]
        if (site + 1 < L_) {
            int next_cR = chi_R(site + 1);
            ROCBLAS_CHECK(Traits::gemm(dev.handle,
                rocblas_operation_none, rocblas_operation_none,
                new_k, d_ * next_cR, cR, &one,
                dev.d_svd_work, new_k,
                get_mps(site + 1, di), cR, &zero_val,
                dev.d_T1, new_k));
            allocate_mps_tensor(site + 1, new_chi_R, next_cR, di);
            HIP_CHECK(hipMemcpyAsync(get_mps(site + 1, di), dev.d_T1,
                        (size_t)new_k * d_ * next_cR * sizeof(Scalar),
                        hipMemcpyDeviceToDevice, dev.stream));
        }
        bond_dims_[site + 1] = new_chi_R;

    } else {  // direction == 'L'
        int new_chi_L = new_k;

        if (use_cpu_svd_) {
            allocate_mps_tensor(site, new_chi_L, cR, di);
            if (new_k == full_k) {
                HIP_CHECK(hipMemcpyAsync(get_mps(site, di), h_Vh_data,
                            full_k * n_svd * sizeof(Scalar), hipMemcpyHostToDevice, dev.stream));
            } else {
                for (int j = 0; j < n_svd; j++)
                    for (int i = 0; i < new_chi_L; i++)
                        dev.h_svd_tmp[i + j * new_chi_L] = h_Vh_data[i + j * full_k];
                HIP_CHECK(hipMemcpyAsync(get_mps(site, di), dev.h_svd_tmp.data(),
                            new_chi_L * n_svd * sizeof(Scalar), hipMemcpyHostToDevice, dev.stream));
            }

            for (int j = 0; j < new_k; j++)
                for (int i = 0; i < m; i++)
                    dev.h_svd_tmp[i + j * m] = Traits::scale_by_real(h_S_data[j], h_U_data[i + j * m]);

            HIP_CHECK(hipMemcpyAsync(dev.d_svd_work, dev.h_svd_tmp.data(),
                        m * new_k * sizeof(Scalar), hipMemcpyHostToDevice, dev.stream));
        } else {
            allocate_mps_tensor(site, new_chi_L, cR, di);
            if (new_k == full_k) {
                HIP_CHECK(hipMemcpyAsync(get_mps(site, di), dev.d_svd_Vh,
                            (size_t)full_k * n_svd * sizeof(Scalar), hipMemcpyDeviceToDevice, dev.stream));
            } else {
                HIP_CHECK(hipMemcpy2DAsync(
                    get_mps(site, di), new_k * sizeof(Scalar),
                    dev.d_svd_Vh, full_k * sizeof(Scalar),
                    new_k * sizeof(Scalar), n_svd,
                    hipMemcpyDeviceToDevice, dev.stream));
            }
            scale_columns_by_real(dev.d_svd_U, m, dev.d_svd_S,
                                  dev.d_svd_work, m, m, new_k, dev.stream);
        }

        // Absorb U*S into MPS[site-1]
        if (site > 0) {
            int prev_cL = chi_L(site - 1);
            ROCBLAS_CHECK(Traits::gemm(dev.handle,
                rocblas_operation_none, rocblas_operation_none,
                prev_cL * d_, new_k, cL, &one,
                get_mps(site - 1, di), prev_cL * d_,
                dev.d_svd_work, m, &zero_val,
                dev.d_T1, prev_cL * d_));
            allocate_mps_tensor(site - 1, prev_cL, new_chi_L, di);
            HIP_CHECK(hipMemcpyAsync(get_mps(site - 1, di), dev.d_T1,
                        (size_t)prev_cL * d_ * new_k * sizeof(Scalar),
                        hipMemcpyDeviceToDevice, dev.stream));
        }
        bond_dims_[site] = new_chi_L;
    }

    dev.heff_cached_site = -1;
}

// ============================================================================
// Single-site optimization (device-aware)
// ============================================================================

template<typename Scalar>
double PDMRGMultiGPU<Scalar>::optimize_site_single(int site, char direction, int di) {
    int cL = chi_L(site);
    int cR = chi_R(site);
    int theta_size = cL * d_ * cR;
    auto& dev = devices_[di];

    HIP_CHECK(hipMemcpyAsync(dev.d_theta, get_mps(site, di),
                              theta_size * sizeof(Scalar), hipMemcpyDeviceToDevice, dev.stream));

    lanczos_use_1site_ = true;
    double energy = lanczos_eigensolver(site, dev.d_theta, theta_size, di);
    lanczos_use_1site_ = false;

    svd_split_single_site(site, dev.d_theta, direction, di);

    return energy;
}

// ============================================================================
// Full-chain single-site sweep methods on device 0 (for warmup and polish)
// ============================================================================

template<typename Scalar>
double PDMRGMultiGPU<Scalar>::sweep_LR_full_1site() {
    int di = 0;
    auto& dev = devices_[di];
    HIP_CHECK(hipSetDevice(dev.device_id));

    // Temporarily set device 0 to cover full chain using d0_ arrays
    auto saved_mps = dev.d_mps;
    auto saved_L_envs = dev.d_L_envs;
    auto saved_R_envs = dev.d_R_envs;
    auto saved_L_alloc = dev.L_env_alloc_chi;
    auto saved_R_alloc = dev.R_env_alloc_chi;
    int saved_seg_first = dev.seg_first;
    int saved_seg_last = dev.seg_last;
    int saved_seg_len = dev.seg_len;

    dev.d_mps = d0_mps_tensors_;
    dev.d_L_envs = d0_L_envs_;
    dev.d_R_envs = d0_R_envs_;
    dev.L_env_alloc_chi = d0_L_env_alloc_chi_;
    dev.R_env_alloc_chi = d0_R_env_alloc_chi_;
    dev.seg_first = 0;
    dev.seg_last = L_ - 1;
    dev.seg_len = L_;

    double energy = 0.0;
    for (int site = 0; site < L_ - 1; site++) {
        energy = optimize_site_single(site, 'R', di);
        update_left_env(site, di);
    }
    // Last site: optimize without SVD (endpoint)
    {
        int cL = chi_L(L_ - 1);
        int cR = chi_R(L_ - 1);
        int theta_size = cL * d_ * cR;
        HIP_CHECK(hipMemcpyAsync(dev.d_theta, get_mps(L_ - 1, di),
                                  theta_size * sizeof(Scalar), hipMemcpyDeviceToDevice, dev.stream));
        lanczos_use_1site_ = true;
        energy = lanczos_eigensolver(L_ - 1, dev.d_theta, theta_size, di);
        lanczos_use_1site_ = false;
        HIP_CHECK(hipMemcpyAsync(get_mps(L_ - 1, di), dev.d_theta,
                                  theta_size * sizeof(Scalar), hipMemcpyDeviceToDevice, dev.stream));
    }

    // Copy back possibly reallocated arrays
    d0_L_envs_ = dev.d_L_envs;
    d0_R_envs_ = dev.d_R_envs;
    d0_L_env_alloc_chi_ = dev.L_env_alloc_chi;
    d0_R_env_alloc_chi_ = dev.R_env_alloc_chi;

    // Restore
    dev.d_mps = saved_mps;
    dev.d_L_envs = saved_L_envs;
    dev.d_R_envs = saved_R_envs;
    dev.L_env_alloc_chi = saved_L_alloc;
    dev.R_env_alloc_chi = saved_R_alloc;
    dev.seg_first = saved_seg_first;
    dev.seg_last = saved_seg_last;
    dev.seg_len = saved_seg_len;

    return energy;
}

template<typename Scalar>
double PDMRGMultiGPU<Scalar>::sweep_RL_full_1site() {
    int di = 0;
    auto& dev = devices_[di];
    HIP_CHECK(hipSetDevice(dev.device_id));

    auto saved_mps = dev.d_mps;
    auto saved_L_envs = dev.d_L_envs;
    auto saved_R_envs = dev.d_R_envs;
    auto saved_L_alloc = dev.L_env_alloc_chi;
    auto saved_R_alloc = dev.R_env_alloc_chi;
    int saved_seg_first = dev.seg_first;
    int saved_seg_last = dev.seg_last;
    int saved_seg_len = dev.seg_len;

    dev.d_mps = d0_mps_tensors_;
    dev.d_L_envs = d0_L_envs_;
    dev.d_R_envs = d0_R_envs_;
    dev.L_env_alloc_chi = d0_L_env_alloc_chi_;
    dev.R_env_alloc_chi = d0_R_env_alloc_chi_;
    dev.seg_first = 0;
    dev.seg_last = L_ - 1;
    dev.seg_len = L_;

    double energy = 0.0;
    for (int site = L_ - 1; site >= 1; site--) {
        energy = optimize_site_single(site, 'L', di);
        update_right_env(site, di);
    }
    // First site: optimize without SVD
    {
        int cL = chi_L(0);
        int cR = chi_R(0);
        int theta_size = cL * d_ * cR;
        HIP_CHECK(hipMemcpyAsync(dev.d_theta, get_mps(0, di),
                                  theta_size * sizeof(Scalar), hipMemcpyDeviceToDevice, dev.stream));
        lanczos_use_1site_ = true;
        energy = lanczos_eigensolver(0, dev.d_theta, theta_size, di);
        lanczos_use_1site_ = false;
        HIP_CHECK(hipMemcpyAsync(get_mps(0, di), dev.d_theta,
                                  theta_size * sizeof(Scalar), hipMemcpyDeviceToDevice, dev.stream));
    }

    d0_L_envs_ = dev.d_L_envs;
    d0_R_envs_ = dev.d_R_envs;
    d0_L_env_alloc_chi_ = dev.L_env_alloc_chi;
    d0_R_env_alloc_chi_ = dev.R_env_alloc_chi;

    dev.d_mps = saved_mps;
    dev.d_L_envs = saved_L_envs;
    dev.d_R_envs = saved_R_envs;
    dev.L_env_alloc_chi = saved_L_alloc;
    dev.R_env_alloc_chi = saved_R_alloc;
    dev.seg_first = saved_seg_first;
    dev.seg_last = saved_seg_last;
    dev.seg_len = saved_seg_len;

    return energy;
}

// ============================================================================
// Full-chain two-site sweep methods (for polish on device 0)
// ============================================================================

template<typename Scalar>
double PDMRGMultiGPU<Scalar>::sweep_LR_full() {
    int di = 0;
    auto& dev = devices_[di];
    HIP_CHECK(hipSetDevice(dev.device_id));

    auto saved_mps = dev.d_mps;
    auto saved_L_envs = dev.d_L_envs;
    auto saved_R_envs = dev.d_R_envs;
    auto saved_L_alloc = dev.L_env_alloc_chi;
    auto saved_R_alloc = dev.R_env_alloc_chi;
    int saved_seg_first = dev.seg_first;
    int saved_seg_last = dev.seg_last;
    int saved_seg_len = dev.seg_len;

    dev.d_mps = d0_mps_tensors_;
    dev.d_L_envs = d0_L_envs_;
    dev.d_R_envs = d0_R_envs_;
    dev.L_env_alloc_chi = d0_L_env_alloc_chi_;
    dev.R_env_alloc_chi = d0_R_env_alloc_chi_;
    dev.seg_first = 0;
    dev.seg_last = L_ - 1;
    dev.seg_len = L_;

    double energy = 0.0;
    for (int site = 0; site < L_ - 1; site++) {
        energy = optimize_bond(site, 'R', di);
        update_left_env(site, di);
    }

    d0_mps_tensors_ = dev.d_mps;
    d0_L_envs_ = dev.d_L_envs;
    d0_R_envs_ = dev.d_R_envs;
    d0_L_env_alloc_chi_ = dev.L_env_alloc_chi;
    d0_R_env_alloc_chi_ = dev.R_env_alloc_chi;

    dev.d_mps = saved_mps;
    dev.d_L_envs = saved_L_envs;
    dev.d_R_envs = saved_R_envs;
    dev.L_env_alloc_chi = saved_L_alloc;
    dev.R_env_alloc_chi = saved_R_alloc;
    dev.seg_first = saved_seg_first;
    dev.seg_last = saved_seg_last;
    dev.seg_len = saved_seg_len;

    return energy;
}

template<typename Scalar>
double PDMRGMultiGPU<Scalar>::sweep_RL_full() {
    int di = 0;
    auto& dev = devices_[di];
    HIP_CHECK(hipSetDevice(dev.device_id));

    auto saved_mps = dev.d_mps;
    auto saved_L_envs = dev.d_L_envs;
    auto saved_R_envs = dev.d_R_envs;
    auto saved_L_alloc = dev.L_env_alloc_chi;
    auto saved_R_alloc = dev.R_env_alloc_chi;
    int saved_seg_first = dev.seg_first;
    int saved_seg_last = dev.seg_last;
    int saved_seg_len = dev.seg_len;

    dev.d_mps = d0_mps_tensors_;
    dev.d_L_envs = d0_L_envs_;
    dev.d_R_envs = d0_R_envs_;
    dev.L_env_alloc_chi = d0_L_env_alloc_chi_;
    dev.R_env_alloc_chi = d0_R_env_alloc_chi_;
    dev.seg_first = 0;
    dev.seg_last = L_ - 1;
    dev.seg_len = L_;

    double energy = 0.0;
    for (int site = L_ - 2; site >= 0; site--) {
        energy = optimize_bond(site, 'L', di);
        update_right_env(site + 1, di);
    }

    d0_mps_tensors_ = dev.d_mps;
    d0_L_envs_ = dev.d_L_envs;
    d0_R_envs_ = dev.d_R_envs;
    d0_L_env_alloc_chi_ = dev.L_env_alloc_chi;
    d0_R_env_alloc_chi_ = dev.R_env_alloc_chi;

    dev.d_mps = saved_mps;
    dev.d_L_envs = saved_L_envs;
    dev.d_R_envs = saved_R_envs;
    dev.L_env_alloc_chi = saved_L_alloc;
    dev.R_env_alloc_chi = saved_R_alloc;
    dev.seg_first = saved_seg_first;
    dev.seg_last = saved_seg_last;
    dev.seg_len = saved_seg_len;

    return energy;
}

// ============================================================================
// Segment sweep methods (each segment on its own device)
// ============================================================================

template<typename Scalar>
void PDMRGMultiGPU<Scalar>::segment_sweep_LR(int seg_idx) {
    int di = seg_idx;
    int first = seg_first_[seg_idx];
    int last = seg_last_[seg_idx];

    HIP_CHECK(hipSetDevice(devices_[di].device_id));

    for (int site = first; site < last; site++) {
        optimize_bond(site, 'R', di);
        update_left_env(site, di);
    }
}

template<typename Scalar>
void PDMRGMultiGPU<Scalar>::segment_sweep_RL(int seg_idx) {
    int di = seg_idx;
    int first = seg_first_[seg_idx];
    int last = seg_last_[seg_idx];

    HIP_CHECK(hipSetDevice(devices_[di].device_id));

    for (int site = last - 1; site >= first; site--) {
        optimize_bond(site, 'L', di);
        update_right_env(site + 1, di);
    }
}

// ============================================================================
// Form theta with V injection (device-aware)
// ============================================================================

template<typename Scalar>
void PDMRGMultiGPU<Scalar>::form_theta_with_V(int site, int boundary_idx, int di) {
    int cL = chi_L(site);
    int chi_bond = bond_dims_[site + 1];
    int cR = chi_R(site + 1);
    auto& dev = devices_[di];
    auto& bs = boundary_states_[boundary_idx];

    // Upload V to device (reuse d_svd_S workspace)
    HIP_CHECK(hipMemcpyAsync(dev.d_svd_S, bs.V.data(),
                              chi_bond * sizeof(RealType),
                              hipMemcpyHostToDevice, dev.stream));

    // Scale: T1 = diag(V) * psi_R  (scale each row i of psi_R by V[i])
    int psi_R_size = chi_bond * d_ * cR;
    HIP_CHECK(hipMemcpyAsync(dev.d_T1, get_mps(site + 1, di),
                              psi_R_size * sizeof(Scalar),
                              hipMemcpyDeviceToDevice, dev.stream));
    scale_rows_by_real(dev.d_T1, chi_bond, dev.d_svd_S,
                       dev.d_T1, chi_bond, chi_bond, d_ * cR, dev.stream);

    // Contract: theta = psi_L * T1 = psi_L * diag(V) * psi_R
    Scalar one = Traits::one(), zero_val = Traits::zero();
    ROCBLAS_CHECK(Traits::gemm(dev.handle,
        rocblas_operation_none, rocblas_operation_none,
        cL * d_, d_ * cR, chi_bond,
        &one,
        get_mps(site, di), cL * d_,
        dev.d_T1, chi_bond,
        &zero_val,
        dev.d_theta, cL * d_));
}

// ============================================================================
// Boundary merge+optimize (runs on device 0 with cross-device transfers)
// ============================================================================

template<typename Scalar>
double PDMRGMultiGPU<Scalar>::merge_and_optimize_boundaries(int parity) {
    double energy = 0.0;
    int di = 0;  // boundary optimization uses device 0
    auto& dev = devices_[di];
    HIP_CHECK(hipSetDevice(dev.device_id));

    for (int b = 0; b < (int)boundary_bonds_.size(); b++) {
        if (parity >= 0 && (b % 2) != parity) continue;

        int bsite = boundary_bonds_[b];
        int k_left = site_to_device_[bsite];
        int k_right = site_to_device_[bsite + 1];
        int cL = chi_L(bsite);
        int cR = chi_R(bsite + 1);
        int theta_size = cL * d_ * d_ * cR;

        // === Copy boundary MPS tensors from owning devices to device 0 ===

        // Temporarily expand device 0's MPS/env to hold boundary sites
        // We use d0_mps_tensors_ which covers the full chain on device 0
        // Copy MPS[bsite] from device k_left to d0_mps_tensors_[bsite]
        {
            int mps_size = chi_L(bsite) * d_ * chi_R(bsite) * sizeof(Scalar);
            if (k_left == 0) {
                HIP_CHECK(hipMemcpy(d0_mps_tensors_[bsite], get_mps(bsite, k_left),
                                    mps_size, hipMemcpyDeviceToDevice));
            } else {
                HIP_CHECK(hipMemcpyPeer(d0_mps_tensors_[bsite], dev.device_id,
                                         get_mps(bsite, k_left), devices_[k_left].device_id, mps_size));
            }
        }

        // Copy MPS[bsite+1] from device k_right to d0_mps_tensors_[bsite+1]
        {
            int mps_size = chi_L(bsite + 1) * d_ * chi_R(bsite + 1) * sizeof(Scalar);
            if (k_right == 0) {
                HIP_CHECK(hipMemcpy(d0_mps_tensors_[bsite + 1], get_mps(bsite + 1, k_right),
                                    mps_size, hipMemcpyDeviceToDevice));
            } else {
                HIP_CHECK(hipMemcpyPeer(d0_mps_tensors_[bsite + 1], dev.device_id,
                                         get_mps(bsite + 1, k_right), devices_[k_right].device_id, mps_size));
            }
        }

        // Copy L_env[bsite] from device k_left to d0_L_envs_[bsite]
        {
            int chi = bond_dims_[bsite];
            if (bsite == 0) chi = 1;
            int env_size = chi * D_mpo_ * chi * sizeof(Scalar);
            int li = local_env_idx(bsite, k_left);
            if (k_left == 0) {
                HIP_CHECK(hipMemcpy(d0_L_envs_[bsite], devices_[k_left].d_L_envs[li],
                                    env_size, hipMemcpyDeviceToDevice));
            } else {
                HIP_CHECK(hipMemcpyPeer(d0_L_envs_[bsite], dev.device_id,
                                         devices_[k_left].d_L_envs[li], devices_[k_left].device_id, env_size));
            }
        }

        // Copy R_env[bsite+2] from device k_right to d0_R_envs_[bsite+2]
        {
            int chi = bond_dims_[bsite + 2];
            if (bsite + 2 == L_) chi = 1;
            int env_size = chi * D_mpo_ * chi * sizeof(Scalar);
            int li = local_env_idx(bsite + 2, k_right);
            if (k_right == 0) {
                HIP_CHECK(hipMemcpy(d0_R_envs_[bsite + 2], devices_[k_right].d_R_envs[li],
                                    env_size, hipMemcpyDeviceToDevice));
            } else {
                HIP_CHECK(hipMemcpyPeer(d0_R_envs_[bsite + 2], dev.device_id,
                                         devices_[k_right].d_R_envs[li], devices_[k_right].device_id, env_size));
            }
        }

        // === Now run boundary optimization on device 0 using d0_ arrays ===
        // Temporarily reconfigure device 0 to use d0_ arrays
        auto saved_mps = dev.d_mps;
        auto saved_L_envs = dev.d_L_envs;
        auto saved_R_envs = dev.d_R_envs;
        auto saved_L_alloc = dev.L_env_alloc_chi;
        auto saved_R_alloc = dev.R_env_alloc_chi;
        int saved_seg_first = dev.seg_first;
        int saved_seg_last = dev.seg_last;
        int saved_seg_len = dev.seg_len;

        dev.d_mps = d0_mps_tensors_;
        dev.d_L_envs = d0_L_envs_;
        dev.d_R_envs = d0_R_envs_;
        dev.L_env_alloc_chi = d0_L_env_alloc_chi_;
        dev.R_env_alloc_chi = d0_R_env_alloc_chi_;
        dev.seg_first = 0;
        dev.seg_last = L_ - 1;
        dev.seg_len = L_;

        // Step 1: Form theta = psi_L * diag(V) * psi_R
        form_theta_with_V(bsite, b, di);

        // Step 2: Optimize theta with eigensolver
        energy = lanczos_eigensolver(bsite, dev.d_theta, theta_size, di);

        // Step 3: SVD split -> direction 'R': MPS[bsite]=U, MPS[bsite+1]=S*Vh
        if (use_rsvd_)
            rsvd_split(bsite, dev.d_theta, 'R', di);
        else
            svd_split(bsite, dev.d_theta, 'R', di);

        // Step 4: Update V = 1/clip(S, 1e-12)
        int new_chi = bond_dims_[bsite + 1];
        boundary_states_[b].chi = new_chi;
        boundary_states_[b].V.resize(new_chi);

        const RealType reg = RealType(1e-12);
        for (int i = 0; i < new_chi; i++) {
            RealType s_val = dev.h_svd_S[i];
            if (s_val < reg) s_val = reg;
            boundary_states_[b].V[i] = RealType(1.0) / s_val;
        }

        // Step 5: Update environments from canonical tensors
        update_left_env(bsite, di);
        update_right_env(bsite + 1, di);

        HIP_CHECK(hipStreamSynchronize(dev.stream));

        // Copy back possibly reallocated arrays
        d0_L_envs_ = dev.d_L_envs;
        d0_R_envs_ = dev.d_R_envs;
        d0_L_env_alloc_chi_ = dev.L_env_alloc_chi;
        d0_R_env_alloc_chi_ = dev.R_env_alloc_chi;

        // Restore device 0 segment info
        dev.d_mps = saved_mps;
        dev.d_L_envs = saved_L_envs;
        dev.d_R_envs = saved_R_envs;
        dev.L_env_alloc_chi = saved_L_alloc;
        dev.R_env_alloc_chi = saved_R_alloc;
        dev.seg_first = saved_seg_first;
        dev.seg_last = saved_seg_last;
        dev.seg_len = saved_seg_len;

        // === Copy updated MPS and environments back to owning devices ===

        // Copy updated MPS[bsite] back to device k_left
        {
            int mps_size = chi_L(bsite) * d_ * chi_R(bsite) * sizeof(Scalar);
            if (k_left == 0) {
                HIP_CHECK(hipMemcpy(get_mps(bsite, k_left), d0_mps_tensors_[bsite],
                                    mps_size, hipMemcpyDeviceToDevice));
            } else {
                HIP_CHECK(hipMemcpyPeer(get_mps(bsite, k_left), devices_[k_left].device_id,
                                         d0_mps_tensors_[bsite], dev.device_id, mps_size));
            }
        }

        // Copy updated MPS[bsite+1] back to device k_right
        {
            int mps_size = chi_L(bsite + 1) * d_ * chi_R(bsite + 1) * sizeof(Scalar);
            if (k_right == 0) {
                HIP_CHECK(hipMemcpy(get_mps(bsite + 1, k_right), d0_mps_tensors_[bsite + 1],
                                    mps_size, hipMemcpyDeviceToDevice));
            } else {
                HIP_CHECK(hipMemcpyPeer(get_mps(bsite + 1, k_right), devices_[k_right].device_id,
                                         d0_mps_tensors_[bsite + 1], dev.device_id, mps_size));
            }
        }

        // Copy updated L_env[bsite+1] to device k_right
        {
            int chi = bond_dims_[bsite + 1];
            int env_size = chi * D_mpo_ * chi * sizeof(Scalar);
            int li_right = local_env_idx(bsite + 1, k_right);
            // Ensure target has enough space
            if (chi > devices_[k_right].L_env_alloc_chi[li_right]) {
                HIP_CHECK(hipSetDevice(devices_[k_right].device_id));
                if (devices_[k_right].d_L_envs[li_right]) HIP_CHECK(hipFree(devices_[k_right].d_L_envs[li_right]));
                HIP_CHECK(hipMalloc(&devices_[k_right].d_L_envs[li_right], env_size));
                devices_[k_right].L_env_alloc_chi[li_right] = chi;
                HIP_CHECK(hipSetDevice(dev.device_id));
            }
            if (k_right == 0) {
                HIP_CHECK(hipMemcpy(devices_[k_right].d_L_envs[li_right], d0_L_envs_[bsite + 1],
                                    env_size, hipMemcpyDeviceToDevice));
            } else {
                HIP_CHECK(hipMemcpyPeer(devices_[k_right].d_L_envs[li_right], devices_[k_right].device_id,
                                         d0_L_envs_[bsite + 1], dev.device_id, env_size));
            }
        }

        // Copy updated R_env[bsite+1] to device k_left
        {
            int chi = bond_dims_[bsite + 1];
            int env_size = chi * D_mpo_ * chi * sizeof(Scalar);
            int li_left = local_env_idx(bsite + 1, k_left);
            // Ensure target has enough space
            if (chi > devices_[k_left].R_env_alloc_chi[li_left]) {
                HIP_CHECK(hipSetDevice(devices_[k_left].device_id));
                if (devices_[k_left].d_R_envs[li_left]) HIP_CHECK(hipFree(devices_[k_left].d_R_envs[li_left]));
                HIP_CHECK(hipMalloc(&devices_[k_left].d_R_envs[li_left], env_size));
                devices_[k_left].R_env_alloc_chi[li_left] = chi;
                HIP_CHECK(hipSetDevice(dev.device_id));
            }
            if (k_left == 0) {
                HIP_CHECK(hipMemcpy(devices_[k_left].d_R_envs[li_left], d0_R_envs_[bsite + 1],
                                    env_size, hipMemcpyDeviceToDevice));
            } else {
                HIP_CHECK(hipMemcpyPeer(devices_[k_left].d_R_envs[li_left], devices_[k_left].device_id,
                                         d0_R_envs_[bsite + 1], dev.device_id, env_size));
            }
        }
    }
    return energy;
}

// ============================================================================
// Main algorithm
// ============================================================================

template<typename Scalar>
double PDMRGMultiGPU<Scalar>::run(int n_outer_sweeps, int n_local_sweeps, int n_warmup) {
    // Timer starts BEFORE env build — includes env build in total (timer_scope=include_env_build)
    auto t_start = std::chrono::high_resolution_clock::now();

    build_initial_environments();

    auto t_envs = std::chrono::high_resolution_clock::now();
    double env_time = std::chrono::duration<double>(t_envs - t_start).count();
    printf("  Environment build: %.3f s\n", env_time);

    // Warmup: single-site sweeps on device 0 (full chain gathered there)
    double warmup_energy = 0.0;
    double prev_warmup_energy = 1e30;
    // MPS is already gathered to device 0 by build_initial_environments
    for (int sw = 0; sw < n_warmup; sw++) {
        sweep_LR_full_1site();
        warmup_energy = sweep_RL_full_1site();
        double dE = std::abs(warmup_energy - prev_warmup_energy);
        prev_warmup_energy = warmup_energy;
        if (dE < tol_ && sw > 0) break;
    }

    // Scatter MPS from device 0 to all devices after warmup
    scatter_mps_from_device0();
    // Also scatter environments
    scatter_envs_from_device0();

    // === Main PDMRG loop (Stoudenmire staggered sweeps) ===
    initialize_boundary_states();
    double energy_prev = warmup_energy;
    energy_ = warmup_energy;
    bool outer_converged = false;

    // Parallel sweep launcher: one CPU thread per device
    auto parallel_sweep = [this](auto sweep_fn) {
        std::vector<std::thread> threads(n_segments_);
        for (int k = 0; k < n_segments_; k++) {
            threads[k] = std::thread([this, k, &sweep_fn]{
                HIP_CHECK(hipSetDevice(devices_[k].device_id));
                sweep_fn(this, k);
            });
        }
        for (auto& t : threads) t.join();
        // Sync all device streams
        for (int s = 0; s < n_segments_; s++) {
            HIP_CHECK(hipStreamSynchronize(devices_[s].stream));
        }
    };

    bool has_odd_boundaries = ((int)boundary_bonds_.size() > 1);

    for (int outer = 0; outer < n_outer_sweeps; outer++) {
        for (int local_sw = 0; local_sw < n_local_sweeps; local_sw++) {
            // Half-sweep 1: even segments LR, odd segments RL
            parallel_sweep([](PDMRGMultiGPU* self, int k) {
                if (k % 2 == 0) self->segment_sweep_LR(k);
                else             self->segment_sweep_RL(k);
            });

            // Merge+optimize at even boundaries
            if (boundary_bonds_.size() > 0) {
                energy_ = merge_and_optimize_boundaries(0);
            }

            // Half-sweep 2: even segments RL, odd segments LR
            parallel_sweep([](PDMRGMultiGPU* self, int k) {
                if (k % 2 == 0) self->segment_sweep_RL(k);
                else             self->segment_sweep_LR(k);
            });

            // Merge+optimize at odd boundaries
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

    // === Polish phase: two-site full-chain sweeps on device 0 ===
    // Two-site sweeps can re-optimize bond dimensions across segment
    // boundaries, escaping local minima that single-site polish cannot.
    if (n_segments_ > 1) {
        gather_mps_to_device0();

        int n_polish = 10;
        build_initial_environments();
        for (int sw = 0; sw < n_polish; sw++) {
            sweep_LR_full();
            double eRL = sweep_RL_full();
            double dE = std::abs(eRL - energy_);
            energy_ = eRL;
            if (dE < tol_) {
                printf("Polish converged after %d sweeps\n", sw + 1);
                break;
            }
        }

        scatter_mps_from_device0();
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(t_end - t_start).count();
    printf("Final energy: %.12f\n", energy_);
    printf("Total wall time: %.3f s (using %d GPUs)\n", total_time, n_devices_);
    printf("  env_build_sec: %.3f  timer_scope: include_env_build\n", env_time);

    return energy_;
}

// ============================================================================
// Utility methods
// ============================================================================

template<typename Scalar>
void PDMRGMultiGPU<Scalar>::get_mps(std::vector<std::vector<Scalar>>& h_mps) const {
    h_mps.resize(L_);
    for (int i = 0; i < L_; i++) {
        int di = site_to_device_[i];
        int size = chi_L(i) * d_ * chi_R(i);
        h_mps[i].resize(size);
        int ls = i - devices_[di].seg_first;
        HIP_CHECK(hipSetDevice(devices_[di].device_id));
        HIP_CHECK(hipMemcpy(h_mps[i].data(), devices_[di].d_mps[ls],
                            size * sizeof(Scalar), hipMemcpyDeviceToHost));
    }
}

#endif // PDMRG_MULTI_GPU_IMPL_H
