#include "boundary_merge_gpu.h"
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <vector>

// LAPACK dstev Fortran interface (CPU fallback for Lanczos eigensolver)
extern "C" {
    void dstev_(const char* jobz, const int* n, double* d, double* e,
                double* z, const int* ldz, double* work, int* info);
}

// Error checking macros
#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            fprintf(stderr, "HIP error in %s:%d - %s\n", __FILE__, __LINE__, \
                    hipGetErrorString(err)); \
            throw std::runtime_error("HIP error"); \
        } \
    } while(0)

#define ROCBLAS_CHECK(call) \
    do { \
        rocblas_status status = call; \
        if (status != rocblas_status_success) { \
            fprintf(stderr, "rocBLAS error in %s:%d - status %d\n", \
                    __FILE__, __LINE__, status); \
            throw std::runtime_error("rocBLAS error"); \
        } \
    } while(0)

#define ROCSOLVER_CHECK(call) \
    do { \
        rocblas_status status = call; \
        if (status != rocblas_status_success) { \
            fprintf(stderr, "rocSOLVER error in %s:%d - status %d\n", \
                    __FILE__, __LINE__, status); \
            throw std::runtime_error("rocSOLVER error"); \
        } \
    } while(0)

//==============================================================================
// GPU Kernels
//==============================================================================

/**
 * Kernel: Compute V * psi_right element-wise
 *
 * V_psi_right[c,s2,b] = V[c] * psi_right[c,s2,b]
 */
__global__ void kernel_multiply_v_psi(
    const double* d_V,
    const double* d_psi_right,
    double* d_V_psi_right,
    int chi_bond, int d, int chi_R
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = chi_bond * d * chi_R;

    if (idx < total) {
        // Column-major: psi_right[c, s2, b] is at index c + s2*chi_bond + b*chi_bond*d
        int c = idx % chi_bond;
        int rest = idx / chi_bond;
        int s2 = rest % d;
        int b = rest / d;

        d_V_psi_right[idx] = d_V[c] * d_psi_right[idx];
    }
}

/**
 * Kernel: Compute V = 1/S with regularization
 *
 * V[i] = 1 / max(S[i], regularization)
 */
__global__ void kernel_compute_v_from_s(
    const double* d_S,
    double* d_V,
    int k,
    double regularization
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < k) {
        double s = d_S[i];
        d_V[i] = 1.0 / fmax(s, regularization);
    }
}

/**
 * Kernel: Scale rows of matrix by vector
 *
 * A[i, j] *= S[i]  (for column-major storage)
 * In column-major: A[i, j] is at index i + j*lda
 */
__global__ void kernel_scale_rows(
    double* d_A,
    const double* d_S,
    int m, int n,
    int lda
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = m * n;

    if (idx < total) {
        int i = idx % m;
        int j = idx / m;
        d_A[i + j * lda] *= d_S[i];
    }
}

/**
 * Kernel: Compute norm of a vector
 */
__global__ void kernel_vector_norm_squared(
    const double* d_vec,
    double* d_result,
    int n
) {
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load and square
    sdata[tid] = (i < n) ? d_vec[i] * d_vec[i] : 0.0;
    __syncthreads();

    // Reduce in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block
    if (tid == 0) {
        atomicAdd(d_result, sdata[0]);
    }
}

//==============================================================================
// BoundaryMergeGPU Implementation
//==============================================================================

BoundaryMergeGPU::BoundaryMergeGPU(int max_bond, int max_iter, double tol)
    : max_bond_(max_bond), max_iter_(max_iter), tol_(tol),
      svd_(nullptr), heff_(nullptr),
      d_theta_(nullptr), d_theta_opt_(nullptr), d_M_(nullptr),
      d_H_theta_(nullptr), d_V_psi_right_(nullptr),
      d_lanczos_v_(nullptr), d_lanczos_alpha_(nullptr), d_lanczos_beta_(nullptr),
      workspace_size_(0),
      max_chi_L_(0), max_chi_R_(0), max_d_(0), max_D_mpo_(0),
      is_initialized_(false)
{
    // Create rocBLAS handle
    ROCBLAS_CHECK(rocblas_create_handle(&rocblas_h_));

    // Create AccurateSVD_GPU (exact SVD, no recursion)
    svd_ = new AccurateSVD_GPU(1e-4, 0);

    // OptimizedHeff will be created lazily when dimensions are known
}

BoundaryMergeGPU::~BoundaryMergeGPU() {
    free_workspace();

    if (svd_) delete svd_;
    if (heff_) delete heff_;

    rocblas_destroy_handle(rocblas_h_);
}

void BoundaryMergeGPU::allocate_workspace(int chi_L, int d, int chi_R, int D_mpo) {
    // Check if we need to reallocate
    if (is_initialized_ &&
        chi_L <= max_chi_L_ && chi_R <= max_chi_R_ &&
        d <= max_d_ && D_mpo <= max_D_mpo_) {
        return;  // Current workspace is sufficient
    }

    // Free old workspace if any
    free_workspace();

    // Update max dimensions
    max_chi_L_ = std::max(max_chi_L_, chi_L);
    max_chi_R_ = std::max(max_chi_R_, chi_R);
    max_d_ = std::max(max_d_, d);
    max_D_mpo_ = std::max(max_D_mpo_, D_mpo);

    // Allocate workspace for maximum dimensions
    size_t size_theta = max_chi_L_ * max_d_ * max_d_ * max_chi_R_;

    HIP_CHECK(hipMalloc(&d_theta_, size_theta * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_theta_opt_, size_theta * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_H_theta_, size_theta * sizeof(double)));

    // For SVD: M is (chi_L*d, d*chi_R)
    size_t size_M = max_chi_L_ * max_d_ * max_d_ * max_chi_R_;
    HIP_CHECK(hipMalloc(&d_M_, size_M * sizeof(double)));

    // For V * psi_right
    size_t size_V_psi = max_bond_ * max_d_ * max_chi_R_;
    HIP_CHECK(hipMalloc(&d_V_psi_right_, size_V_psi * sizeof(double)));

    // Lanczos workspace
    int lanczos_size = max_iter_ * size_theta;
    HIP_CHECK(hipMalloc(&d_lanczos_v_, lanczos_size * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_lanczos_alpha_, max_iter_ * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_lanczos_beta_, max_iter_ * sizeof(double)));

    is_initialized_ = true;
}

void BoundaryMergeGPU::free_workspace() {
    if (!is_initialized_) return;

    if (d_theta_) HIP_CHECK(hipFree(d_theta_));
    if (d_theta_opt_) HIP_CHECK(hipFree(d_theta_opt_));
    if (d_M_) HIP_CHECK(hipFree(d_M_));
    if (d_H_theta_) HIP_CHECK(hipFree(d_H_theta_));
    if (d_V_psi_right_) HIP_CHECK(hipFree(d_V_psi_right_));
    if (d_lanczos_v_) HIP_CHECK(hipFree(d_lanczos_v_));
    if (d_lanczos_alpha_) HIP_CHECK(hipFree(d_lanczos_alpha_));
    if (d_lanczos_beta_) HIP_CHECK(hipFree(d_lanczos_beta_));

    d_theta_ = nullptr;
    d_theta_opt_ = nullptr;
    d_M_ = nullptr;
    d_H_theta_ = nullptr;
    d_V_psi_right_ = nullptr;
    d_lanczos_v_ = nullptr;
    d_lanczos_alpha_ = nullptr;
    d_lanczos_beta_ = nullptr;

    is_initialized_ = false;
}

//==============================================================================
// Main Merge Function
//==============================================================================

void BoundaryMergeGPU::merge(BoundaryData* left, BoundaryData* right,
                             double& energy, double& trunc_err,
                             bool skip_optimization,
                             hipStream_t stream) {
    // Verify boundary data
    if (!left || !right) {
        throw std::runtime_error("BoundaryMergeGPU::merge: null boundary data");
    }

    if (!left->is_allocated || !right->is_allocated) {
        throw std::runtime_error("BoundaryMergeGPU::merge: boundary data not allocated");
    }

    // Extract dimensions
    int chi_L = left->chi_L;
    int chi_R = right->chi_R;
    int chi_bond = left->chi_bond;  // Should equal right->chi_bond
    int d = left->d;
    int D_mpo = left->D_mpo;

    if (chi_bond != right->chi_bond) {
        throw std::runtime_error("BoundaryMergeGPU::merge: chi_bond mismatch");
    }

    // Allocate workspace if needed
    allocate_workspace(chi_L, d, chi_R, D_mpo);

    // Step 1: Form theta = psi_left . diag(V) . psi_right
    form_theta_from_boundary(
        left->d_psi_left,
        right->d_psi_right,
        left->d_V,
        d_theta_,
        chi_L, d, chi_R, chi_bond,
        stream
    );

    // Step 2: Optimize with Lanczos (or skip)
    optimize_two_site_gpu(
        left->d_L_env,
        right->d_R_env,
        left->d_W_left,
        right->d_W_right,
        d_theta_,  // Input: initial theta, Output: optimized theta
        energy,
        chi_L, d, chi_R, D_mpo,
        skip_optimization,
        stream
    );

    // Step 3: Split with exact SVD
    int k_out;
    double* d_S_temp;
    double* d_A_left_temp;
    double* d_A_right_temp;

    HIP_CHECK(hipMalloc(&d_S_temp, max_bond_ * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_A_left_temp, chi_L * d * max_bond_ * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_A_right_temp, max_bond_ * d * chi_R * sizeof(double)));

    split_with_svd(
        d_theta_,  // Now contains optimized theta
        d_A_left_temp,   // Temporary buffer for A_left_new
        d_A_right_temp,  // Temporary buffer for A_right_new
        d_S_temp,
        trunc_err,
        k_out,
        chi_L, d, chi_R,
        stream
    );

    // Step 4: Compute V_new = 1/S (into temporary buffer)
    double* d_V_temp;
    HIP_CHECK(hipMalloc(&d_V_temp, k_out * sizeof(double)));

    compute_v_from_s(
        d_S_temp,
        d_V_temp,
        k_out,
        1e-12,  // Regularization
        stream
    );

    // Now reallocate boundary data with new bond dimension k_out
    // Free old allocations
    HIP_CHECK(hipFree(left->d_psi_left));
    HIP_CHECK(hipFree(right->d_psi_right));
    HIP_CHECK(hipFree(left->d_V));

    // Allocate new buffers with correct size
    HIP_CHECK(hipMalloc(&left->d_psi_left, chi_L * d * k_out * sizeof(double)));
    HIP_CHECK(hipMalloc(&right->d_psi_right, k_out * d * chi_R * sizeof(double)));
    HIP_CHECK(hipMalloc(&left->d_V, k_out * sizeof(double)));

    // Copy from temporary buffers
    HIP_CHECK(hipMemcpy(left->d_psi_left, d_A_left_temp,
                        chi_L * d * k_out * sizeof(double), hipMemcpyDeviceToDevice));
    HIP_CHECK(hipMemcpy(right->d_psi_right, d_A_right_temp,
                        k_out * d * chi_R * sizeof(double), hipMemcpyDeviceToDevice));
    HIP_CHECK(hipMemcpy(left->d_V, d_V_temp, k_out * sizeof(double), hipMemcpyDeviceToDevice));

    // Update chi_bond in boundary data
    left->chi_bond = k_out;
    right->chi_bond = k_out;

    // Free temporary buffers
    HIP_CHECK(hipFree(d_S_temp));
    HIP_CHECK(hipFree(d_A_left_temp));
    HIP_CHECK(hipFree(d_A_right_temp));
    HIP_CHECK(hipFree(d_V_temp));
}

//==============================================================================
// Step 1: Form theta from boundary
//==============================================================================

void BoundaryMergeGPU::form_theta_from_boundary(
    const double* d_psi_left,
    const double* d_psi_right,
    const double* d_V,
    double* d_theta,
    int chi_L, int d, int chi_R, int chi_bond,
    hipStream_t stream
) {
    // Step 1a: Compute V_psi_right[c,s2,b] = V[c] * psi_right[c,s2,b]
    int total = chi_bond * d * chi_R;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    kernel_multiply_v_psi<<<grid_size, block_size, 0, stream>>>(
        d_V, d_psi_right, d_V_psi_right_, chi_bond, d, chi_R
    );

    // Step 1b: Contract theta[a,s1,s2,b] = psi_left[a,s1,c] * V_psi_right[c,s2,b]
    // Using rocBLAS gemm:
    // psi_left: (chi_L, d, chi_bond) → reshape to (chi_L*d, chi_bond)
    // V_psi_right: (chi_bond, d, chi_R) → reshape to (chi_bond, d*chi_R)
    // result: (chi_L*d, d*chi_R) → reshape to (chi_L, d, d, chi_R)

    double alpha = 1.0;
    double beta = 0.0;

    // C = A * B where:
    // A = psi_left (chi_L*d, chi_bond)
    // B = V_psi_right (chi_bond, d*chi_R)
    // C = theta (chi_L*d, d*chi_R)

    ROCBLAS_CHECK(rocblas_dgemm(
        rocblas_h_,
        rocblas_operation_none,
        rocblas_operation_none,
        chi_L * d,     // m
        d * chi_R,     // n
        chi_bond,      // k
        &alpha,
        d_psi_left, chi_L * d,  // A, lda
        d_V_psi_right_, chi_bond,  // B, ldb
        &beta,
        d_theta, chi_L * d  // C, ldc
    ));
}

//==============================================================================
// Step 2: Optimize with Lanczos
//==============================================================================

void BoundaryMergeGPU::optimize_two_site_gpu(
    const double* d_L_env,
    const double* d_R_env,
    const double* d_W_left,
    const double* d_W_right,
    double* d_theta,
    double& energy,
    int chi_L, int d, int chi_R, int D_mpo,
    bool skip_optimization,
    hipStream_t stream
) {
    if (skip_optimization) {
        // Just compute energy: E = <theta|H|theta> / <theta|theta>

        // Apply H_eff
        apply_heff(d_L_env, d_R_env, d_W_left, d_W_right,
                   d_theta, d_H_theta_, chi_L, d, chi_R, D_mpo, stream);

        // Compute <theta|H|theta>
        int n = chi_L * d * d * chi_R;
        double h_theta_norm, theta_norm;

        ROCBLAS_CHECK(rocblas_ddot(rocblas_h_, n, d_theta, 1, d_H_theta_, 1, &h_theta_norm));
        ROCBLAS_CHECK(rocblas_ddot(rocblas_h_, n, d_theta, 1, d_theta, 1, &theta_norm));

        energy = h_theta_norm / theta_norm;

    } else {
        // Run Lanczos to optimize
        lanczos_eigensolver(d_L_env, d_R_env, d_W_left, d_W_right,
                           d_theta, energy, chi_L, d, chi_R, D_mpo, stream);
    }
}

void BoundaryMergeGPU::apply_heff(
    const double* d_L_env,
    const double* d_R_env,
    const double* d_W_left,
    const double* d_W_right,
    const double* d_theta,
    double* d_result,
    int chi_L, int d, int chi_R, int D_mpo,
    hipStream_t stream
) {
    // Apply H_eff = L_env * W_left * W_right * R_env to theta
    //
    // result[ap, s1p, s2p, bp] = sum_{a, s1, s2, b, w, wm, wr}
    //   L[a, w, ap] * W1[w, s1, s1p, wm] * W2[wm, s2, s2p, wr] * R[b, wr, bp] * theta[a, s1, s2, b]
    //
    // CPU contraction for now (can be GPU-optimized with hipTensor later)

    int psi_size = chi_L * d * d * chi_R;
    int L_size = chi_L * D_mpo * chi_L;
    int R_size = chi_R * D_mpo * chi_R;
    int W_size = D_mpo * d * d * D_mpo;

    // Copy to host
    std::vector<double> hL(L_size), hR(R_size), hW1(W_size), hW2(W_size), hTheta(psi_size);
    HIP_CHECK(hipMemcpy(hL.data(), d_L_env, L_size * sizeof(double), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hR.data(), d_R_env, R_size * sizeof(double), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hW1.data(), d_W_left, W_size * sizeof(double), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hW2.data(), d_W_right, W_size * sizeof(double), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hTheta.data(), d_theta, psi_size * sizeof(double), hipMemcpyDeviceToHost));

    // DEBUG: Check if environments are identity
    double L_norm = 0.0, R_norm = 0.0, W1_norm = 0.0;
    for (int i = 0; i < L_size; i++) L_norm += hL[i] * hL[i];
    for (int i = 0; i < R_size; i++) R_norm += hR[i] * hR[i];
    for (int i = 0; i < W_size; i++) W1_norm += hW1[i] * hW1[i];
    printf("DEBUG H_eff: ||L||=%.6f, ||R||=%.6f, ||W1||=%.6f, ||theta||=%.6f\n",
           sqrt(L_norm), sqrt(R_norm), sqrt(W1_norm),
           sqrt(hTheta[0]*hTheta[0] + hTheta[1]*hTheta[1] + hTheta[2]*hTheta[2]));

    std::vector<double> hResult(psi_size, 0.0);

    // Step 1: T1[w, ap, s1, s2, b] = sum_a L[a, w, ap] * theta[a, s1, s2, b]
    int T1_size = D_mpo * chi_L * d * d * chi_R;
    std::vector<double> hT1(T1_size, 0.0);
    for (int w = 0; w < D_mpo; w++) {
        for (int ap = 0; ap < chi_L; ap++) {
            for (int s1 = 0; s1 < d; s1++) {
                for (int s2 = 0; s2 < d; s2++) {
                    for (int b = 0; b < chi_R; b++) {
                        double sum = 0.0;
                        for (int a = 0; a < chi_L; a++) {
                            sum += hL[a + w*chi_L + ap*chi_L*D_mpo] * hTheta[a + s1*chi_L + s2*chi_L*d + b*chi_L*d*d];
                        }
                        hT1[w + ap*D_mpo + s1*D_mpo*chi_L + s2*D_mpo*chi_L*d + b*D_mpo*chi_L*d*d] = sum;
                    }
                }
            }
        }
    }

    // Step 2: T2[wm, ap, s1p, s2, b] = sum_{w,s1} W1[w, s1, s1p, wm] * T1[w, ap, s1, s2, b]
    int T2_size = D_mpo * chi_L * d * d * chi_R;
    std::vector<double> hT2(T2_size, 0.0);
    for (int wm = 0; wm < D_mpo; wm++) {
        for (int ap = 0; ap < chi_L; ap++) {
            for (int s1p = 0; s1p < d; s1p++) {
                for (int s2 = 0; s2 < d; s2++) {
                    for (int b = 0; b < chi_R; b++) {
                        double sum = 0.0;
                        for (int w = 0; w < D_mpo; w++) {
                            for (int s1 = 0; s1 < d; s1++) {
                                sum += hW1[w + s1*D_mpo + s1p*D_mpo*d + wm*D_mpo*d*d] *
                                       hT1[w + ap*D_mpo + s1*D_mpo*chi_L + s2*D_mpo*chi_L*d + b*D_mpo*chi_L*d*d];
                            }
                        }
                        hT2[wm + ap*D_mpo + s1p*D_mpo*chi_L + s2*D_mpo*chi_L*d + b*D_mpo*chi_L*d*d] = sum;
                    }
                }
            }
        }
    }

    // Step 3: T3[ap, s1p, s2p, wr, b] = sum_{wm,s2} W2[wm, s2, s2p, wr] * T2[wm, ap, s1p, s2, b]
    int T3_size = chi_L * d * d * D_mpo * chi_R;
    std::vector<double> hT3(T3_size, 0.0);
    for (int ap = 0; ap < chi_L; ap++) {
        for (int s1p = 0; s1p < d; s1p++) {
            for (int s2p = 0; s2p < d; s2p++) {
                for (int wr = 0; wr < D_mpo; wr++) {
                    for (int b = 0; b < chi_R; b++) {
                        double sum = 0.0;
                        for (int wm = 0; wm < D_mpo; wm++) {
                            for (int s2 = 0; s2 < d; s2++) {
                                sum += hW2[wm + s2*D_mpo + s2p*D_mpo*d + wr*D_mpo*d*d] *
                                       hT2[wm + ap*D_mpo + s1p*D_mpo*chi_L + s2*D_mpo*chi_L*d + b*D_mpo*chi_L*d*d];
                            }
                        }
                        hT3[ap + s1p*chi_L + s2p*chi_L*d + wr*chi_L*d*d + b*chi_L*d*d*D_mpo] = sum;
                    }
                }
            }
        }
    }

    // Step 4: result[ap, s1p, s2p, bp] = sum_{wr,b} R[b, wr, bp] * T3[ap, s1p, s2p, wr, b]
    for (int ap = 0; ap < chi_L; ap++) {
        for (int s1p = 0; s1p < d; s1p++) {
            for (int s2p = 0; s2p < d; s2p++) {
                for (int bp = 0; bp < chi_R; bp++) {
                    double sum = 0.0;
                    for (int wr = 0; wr < D_mpo; wr++) {
                        for (int b = 0; b < chi_R; b++) {
                            sum += hR[b + wr*chi_R + bp*chi_R*D_mpo] *
                                   hT3[ap + s1p*chi_L + s2p*chi_L*d + wr*chi_L*d*d + b*chi_L*d*d*D_mpo];
                        }
                    }
                    hResult[ap + s1p*chi_L + s2p*chi_L*d + bp*chi_L*d*d] = sum;
                }
            }
        }
    }

    // Copy back to device
    HIP_CHECK(hipMemcpy(d_result, hResult.data(), psi_size * sizeof(double), hipMemcpyHostToDevice));
}

void BoundaryMergeGPU::lanczos_eigensolver(
    const double* d_L_env,
    const double* d_R_env,
    const double* d_W_left,
    const double* d_W_right,
    double* d_theta,
    double& energy,
    int chi_L, int d, int chi_R, int D_mpo,
    hipStream_t stream
) {
    int n = chi_L * d * d * chi_R;

    // Allocate Lanczos vectors if needed
    if (!d_lanczos_v_) {
        HIP_CHECK(hipMalloc(&d_lanczos_v_, max_iter_ * n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_lanczos_alpha_, max_iter_ * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_lanczos_beta_, max_iter_ * sizeof(double)));
    }

    std::vector<double> h_alpha(max_iter_);
    std::vector<double> h_beta(max_iter_);

    // v[0] = theta / ||theta||
    double* d_v0 = d_lanczos_v_;
    double norm;
    ROCBLAS_CHECK(rocblas_dnrm2(rocblas_h_, n, d_theta, 1, &norm));

    double alpha_scale = 1.0 / norm;
    ROCBLAS_CHECK(rocblas_dscal(rocblas_h_, n, &alpha_scale, d_theta, 1));
    HIP_CHECK(hipMemcpy(d_v0, d_theta, n * sizeof(double), hipMemcpyDeviceToDevice));

    // Lanczos iteration
    int iter;
    for (iter = 0; iter < max_iter_; iter++) {
        double* d_vi = d_lanczos_v_ + iter * n;

        // w = H|v_i>
        double* d_w = d_H_theta_;  // Reuse workspace
        apply_heff(d_L_env, d_R_env, d_W_left, d_W_right,
                   d_vi, d_w, chi_L, d, chi_R, D_mpo, stream);

        // alpha_i = <v_i|w>
        double alpha_i;
        ROCBLAS_CHECK(rocblas_ddot(rocblas_h_, n, d_vi, 1, d_w, 1, &alpha_i));
        h_alpha[iter] = alpha_i;

        // w = w - alpha_i * v_i
        double neg_alpha = -alpha_i;
        ROCBLAS_CHECK(rocblas_daxpy(rocblas_h_, n, &neg_alpha, d_vi, 1, d_w, 1));

        // w = w - beta_{i-1} * v_{i-1}
        if (iter > 0) {
            double* d_vim1 = d_lanczos_v_ + (iter - 1) * n;
            double neg_beta = -h_beta[iter - 1];
            ROCBLAS_CHECK(rocblas_daxpy(rocblas_h_, n, &neg_beta, d_vim1, 1, d_w, 1));
        }

        // beta_i = ||w||
        double beta_i;
        ROCBLAS_CHECK(rocblas_dnrm2(rocblas_h_, n, d_w, 1, &beta_i));
        h_beta[iter] = beta_i;

        // Check convergence
        if (beta_i < tol_) {
            iter++;
            break;
        }

        // v_{i+1} = w / beta_i
        if (iter + 1 < max_iter_) {
            double* d_vip1 = d_lanczos_v_ + (iter + 1) * n;
            double scale = 1.0 / beta_i;
            HIP_CHECK(hipMemcpy(d_vip1, d_w, n * sizeof(double), hipMemcpyDeviceToDevice));
            ROCBLAS_CHECK(rocblas_dscal(rocblas_h_, n, &scale, d_vip1, 1));
        }
    }

    int niter = iter;

    // DEBUG: Print first few alpha and beta values
    printf("DEBUG Lanczos: niter=%d, alpha=[%.6f, %.6f, %.6f, ...], beta=[%.6f, %.6f, ...]\n",
           niter,
           h_alpha[0], (niter > 1 ? h_alpha[1] : 0.0), (niter > 2 ? h_alpha[2] : 0.0),
           (niter > 1 ? h_beta[0] : 0.0), (niter > 2 ? h_beta[1] : 0.0));

    // ==================================================================
    // CPU LAPACK fallback for tridiagonal eigenvalue problem
    // ==================================================================
    // NOTE: rocSOLVER dsteqr has a confirmed bug in ROCm 7.2 on gfx942 (MI300X)
    // returning incorrect eigenvalues [-1,0,1]. Using CPU LAPACK dstev instead.
    // The tridiagonal matrix is tiny (3-30 dimensions), so CPU overhead is
    // negligible (<0.01% of DMRG iteration time).
    //
    // This is the standard approach in production GPU Lanczos implementations.

    // Prepare CPU arrays (column-major for LAPACK)
    std::vector<double> h_D(niter);      // Diagonal (input), eigenvalues (output)
    std::vector<double> h_E(niter);      // Off-diagonal (input, destroyed)
    std::vector<double> h_Z(niter * niter);  // Eigenvectors (output, column-major)
    std::vector<double> h_work(std::max(1, 2*niter - 2));  // LAPACK workspace
    int lapack_info = 0;

    // Copy from Lanczos arrays (already on host)
    std::copy(h_alpha.begin(), h_alpha.begin() + niter, h_D.begin());
    std::copy(h_beta.begin(), h_beta.begin() + niter - 1, h_E.begin());
    if (niter > 0) h_E[niter-1] = 0.0;  // Last element unused by LAPACK

    // DEBUG: Print input to LAPACK
    printf("DEBUG LAPACK INPUT: niter=%d\n", niter);
    printf("  D (diagonal) = [");
    for (int i = 0; i < std::min(5, niter); i++) {
        printf("%.6f%s", h_D[i], (i < std::min(4, niter-1) ? ", " : ""));
    }
    if (niter > 5) printf(", ...");
    printf("]\n");
    printf("  E (off-diag) = [");
    for (int i = 0; i < std::min(5, niter-1); i++) {
        printf("%.6f%s", h_E[i], (i < std::min(4, niter-2) ? ", " : ""));
    }
    if (niter > 5) printf(", ...");
    printf("]\n");

    // Call LAPACK dstev: compute eigenvalues and eigenvectors
    const char jobz = 'V';  // Compute eigenvectors
    const int n_lapack = niter;
    const int ldz = niter;

    dstev_(&jobz, &n_lapack, h_D.data(), h_E.data(), h_Z.data(), &ldz, h_work.data(), &lapack_info);

    if (lapack_info != 0) {
        throw std::runtime_error("LAPACK dstev failed with info = " + std::to_string(lapack_info));
    }

    // Eigenvalues are now in h_D (sorted ascending)
    // Eigenvectors are in h_Z (column-major: first column = eigenvector for smallest eigenvalue)
    energy = h_D[0];  // Minimum eigenvalue (ground state energy)

    printf("DEBUG LAPACK dstev: niter=%d, eigenvalues = [", niter);
    for (int i = 0; i < std::min(5, niter); i++) {
        printf("%.10f%s", h_D[i], (i < std::min(4, niter-1) ? ", " : ""));
    }
    if (niter > 5) printf(", ...");
    printf("]\n");
    printf("DEBUG Ground state energy: %.12e\n", energy);

    // Copy Ritz coefficients (first column of h_Z) to GPU for reconstruction
    double* d_ritz_coeffs;
    HIP_CHECK(hipMalloc(&d_ritz_coeffs, niter * sizeof(double)));
    HIP_CHECK(hipMemcpy(d_ritz_coeffs, h_Z.data(), niter * sizeof(double),
                        hipMemcpyHostToDevice));

    // ==================================================================
    // Reconstruct ground state wavefunction on GPU
    // ==================================================================
    // |θ⟩ = Σ c[i] |v[i]⟩ where c[i] are Ritz coefficients (eigenvector)
    // This is a matrix-vector product: d_theta = d_lanczos_v_ * d_ritz_coeffs

    const double alpha_gemv = 1.0;
    const double beta_gemv = 0.0;

    ROCBLAS_CHECK(rocblas_dgemv(
        rocblas_h_,
        rocblas_operation_none,  // No transpose
        n,                        // Rows of V
        niter,                    // Cols of V
        &alpha_gemv,              // Scale factor
        d_lanczos_v_,            // Matrix V (n × niter, Lanczos vectors)
        n,                        // Leading dimension of V
        d_ritz_coeffs,           // Ritz coefficient vector (niter × 1)
        1,                        // Stride
        &beta_gemv,               // Scale for d_theta (0 = overwrite)
        d_theta,                  // Output: reconstructed wavefunction
        1                         // Stride
    ));

    // Normalize the reconstructed wavefunction
    double theta_norm;
    ROCBLAS_CHECK(rocblas_dnrm2(rocblas_h_, n, d_theta, 1, &theta_norm));
    double norm_scale = 1.0 / theta_norm;
    ROCBLAS_CHECK(rocblas_dscal(rocblas_h_, n, &norm_scale, d_theta, 1));

    // ==================================================================
    // Validation: Rayleigh quotient check
    // ==================================================================
    // Verify E_ground = <θ|H_eff|θ> matches λ_min to within tolerance
    apply_heff(d_L_env, d_R_env, d_W_left, d_W_right,
               d_theta, d_H_theta_, chi_L, d, chi_R, D_mpo, stream);

    double rayleigh;
    ROCBLAS_CHECK(rocblas_ddot(rocblas_h_, n, d_theta, 1, d_H_theta_, 1, &rayleigh));

    double eigenvalue_error = std::abs(rayleigh - energy);
    if (eigenvalue_error > 1e-10) {
        printf("WARNING: Rayleigh quotient mismatch: λ=%.12e, <θ|H|θ>=%.12e, err=%.2e\n",
               energy, rayleigh, eigenvalue_error);
    }

    // Free GPU memory
    HIP_CHECK(hipFree(d_ritz_coeffs));
}

//==============================================================================
// Step 3: Split with exact SVD
//==============================================================================

void BoundaryMergeGPU::split_with_svd(
    const double* d_theta_opt,
    double* d_A_left_new,
    double* d_A_right_new,
    double* d_S,
    double& trunc_err,
    int& k_out,
    int chi_L, int d, int chi_R,
    hipStream_t stream
) {
    // Reshape theta: (chi_L, d, d, chi_R) → M: (chi_L*d, d*chi_R)
    // For column-major, this is just a reinterpretation
    // Note: decompose() doesn't modify input, but signature requires non-const
    double* d_M_reshaped = const_cast<double*>(d_theta_opt);

    int m = chi_L * d;
    int n = d * chi_R;

    // Use AccurateSVD_GPU (exact SVD from Phase 1)
    AccurateSVDResult result = svd_->decompose(d_M_reshaped, m, n);

    // Truncate to max_bond
    int k = std::min(result.rank, max_bond_);
    k = std::max(k, 1);
    k_out = k;

    // Compute truncation error
    std::vector<double> h_S(result.rank);
    HIP_CHECK(hipMemcpy(h_S.data(), result.d_S, result.rank * sizeof(double),
                        hipMemcpyDeviceToHost));

    trunc_err = 0.0;
    for (int i = k; i < result.rank; i++) {
        trunc_err += h_S[i] * h_S[i];
    }

    // Copy singular values
    HIP_CHECK(hipMemcpy(d_S, result.d_S, k * sizeof(double),
                        hipMemcpyDeviceToDevice));

    // Form A_left = U[:, :k].reshape(chi_L, d, k)
    // U is (m, rank), we want first k columns
    // Just copy since U is already (m, result.rank) and we want (m, k)
    HIP_CHECK(hipMemcpy(d_A_left_new, result.d_U, m * k * sizeof(double),
                        hipMemcpyDeviceToDevice));

    // Form A_right = (S @ Vh)[:k, :].reshape(k, d, chi_R)
    // Vh is (result.rank, n), we want (k, n)
    // Multiply S into Vh: Vh[i, :] *= S[i]

    // Copy Vh first
    HIP_CHECK(hipMemcpy(d_A_right_new, result.d_Vh, k * n * sizeof(double),
                        hipMemcpyDeviceToDevice));

    // Scale rows by S using custom kernel
    int total = k * n;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    kernel_scale_rows<<<grid_size, block_size, 0, stream>>>(
        d_A_right_new, d_S, k, n, k  // lda = k for column-major (k, n)
    );
}

//==============================================================================
// Step 4: Compute V = 1/S
//==============================================================================

void BoundaryMergeGPU::compute_v_from_s(
    const double* d_S,
    double* d_V,
    int k,
    double regularization,
    hipStream_t stream
) {
    int block_size = 256;
    int grid_size = (k + block_size - 1) / block_size;

    kernel_compute_v_from_s<<<grid_size, block_size, 0, stream>>>(
        d_S, d_V, k, regularization
    );
}
