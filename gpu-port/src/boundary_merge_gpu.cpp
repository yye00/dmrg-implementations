#include "boundary_merge_gpu.h"
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <vector>

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
    // Create OptimizedHeff if needed
    if (!heff_ ||
        heff_->chi_L != chi_L || heff_->chi_R != chi_R ||
        heff_->d != d || heff_->D_mpo != D_mpo) {

        if (heff_) delete heff_;

        // Create hipTensor handle if needed
        static hiptensorHandle_t* ht_handle = nullptr;
        if (!ht_handle) {
            ht_handle = new hiptensorHandle_t;
            hiptensorCreate(ht_handle);
        }

        heff_ = new OptimizedHeff(chi_L, chi_R, d, D_mpo, ht_handle);
    }

    // Apply H_eff: result = L × W_left × theta × W_right × R
    heff_->apply(d_theta, d_result, d_L_env, d_R_env, d_W_left, d_W_right, stream);
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

    // Solve tridiagonal eigenvalue problem on CPU
    // T = tridiag(beta[:-1], alpha, beta[:-1])
    // For simplicity: use power method or just return Rayleigh quotient
    // Full implementation would use LAPACK dstev

    // For now: simple approach - lowest eigenvalue ≈ alpha[0] - 2*beta[0]
    // (valid for well-conditioned problems)
    energy = h_alpha[0];

    // Better: compute Rayleigh quotient with v[0]
    // E = <v0|H|v0>
    apply_heff(d_L_env, d_R_env, d_W_left, d_W_right,
               d_v0, d_H_theta_, chi_L, d, chi_R, D_mpo, stream);

    double e_rayleigh;
    ROCBLAS_CHECK(rocblas_ddot(rocblas_h_, n, d_v0, 1, d_H_theta_, 1, &e_rayleigh));
    energy = e_rayleigh;

    // Copy back normalized v0 to theta (ground state approximation)
    HIP_CHECK(hipMemcpy(d_theta, d_v0, n * sizeof(double), hipMemcpyDeviceToDevice));
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
