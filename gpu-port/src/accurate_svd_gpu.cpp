#include "accurate_svd_gpu.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <vector>

// Error checking macros
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

#define ROCSOLVER_CHECK(call) \
    do { \
        rocblas_status status = call; \
        if (status != rocblas_status_success) { \
            std::cerr << "rocSOLVER error in " << __FILE__ << ":" << __LINE__ \
                      << " - status " << status << std::endl; \
            throw std::runtime_error("rocSOLVER error"); \
        } \
    } while(0)

// ============================================================================
// GPU Kernels
// ============================================================================

__global__ void invert_with_clipping_kernel(
    const double* S,
    double* V,
    int k,
    double clip_min
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < k) {
        double s = S[i];
        V[i] = 1.0 / fmax(s, clip_min);
    }
}

void launch_invert_with_clipping(
    double* d_S,
    double* d_V,
    int k,
    double clip_min,
    hipStream_t stream
) {
    int block_size = 256;
    int num_blocks = (k + block_size - 1) / block_size;
    hipLaunchKernelGGL(
        invert_with_clipping_kernel,
        dim3(num_blocks), dim3(block_size),
        0, stream,
        d_S, d_V, k, clip_min
    );
    HIP_CHECK(hipGetLastError());
}

// ============================================================================
// AccurateSVD_GPU Implementation
// ============================================================================

AccurateSVD_GPU::AccurateSVD_GPU(double eps, int max_depth)
    : epsilon(eps), max_recursion_depth(max_depth)
{
    ROCBLAS_CHECK(rocblas_create_handle(&rocblas_h));
    // ROCm 7.2.0: rocsolver functions use rocblas_handle directly
}

AccurateSVD_GPU::~AccurateSVD_GPU() {
    rocblas_destroy_handle(rocblas_h);
}

int AccurateSVD_GPU::find_degradation_threshold(double* d_S, int k) {
    // Copy singular values to host
    std::vector<double> h_S(k);
    HIP_CHECK(hipMemcpy(h_S.data(), d_S, k * sizeof(double), hipMemcpyDeviceToHost));

    double sigma_max = h_S[0];
    if (sigma_max == 0.0) {
        return 0;  // All singular values are zero
    }

    // Find first index where S[p] / S[0] < epsilon
    for (int p = 0; p < k; p++) {
        if (h_S[p] / sigma_max < epsilon) {
            return p;
        }
    }

    return -1;  // No degradation found
}

AccurateSVDResult AccurateSVD_GPU::standard_svd(double* d_M, int m, int n) {
    int k = std::min(m, n);

    AccurateSVDResult result;
    result.m = m;
    result.n = n;
    result.rank = k;
    result.owns_memory = true;

    // Allocate output arrays
    HIP_CHECK(hipMalloc(&result.d_U, m * k * sizeof(double)));
    HIP_CHECK(hipMalloc(&result.d_S, k * sizeof(double)));
    HIP_CHECK(hipMalloc(&result.d_Vh, k * n * sizeof(double)));

    // Allocate E vector for superdiagonal elements (required but not used)
    double* d_E;
    HIP_CHECK(hipMalloc(&d_E, (k - 1) * sizeof(double)));

    // Allocate info on device
    if (!workspace.d_info) {
        HIP_CHECK(hipMalloc(&workspace.d_info, sizeof(int)));
    }

    // Debug: Check matrix before SVD
    std::vector<double> debug_M(std::min(9, m*n));
    HIP_CHECK(hipMemcpy(debug_M.data(), d_M, std::min(9, m*n) * sizeof(double), hipMemcpyDeviceToHost));
    std::cerr << "DEBUG: Matrix before SVD (first 9 elements): ";
    for (int i = 0; i < std::min(9, m*n); i++) {
        std::cerr << debug_M[i] << " ";
    }
    std::cerr << std::endl;

    // Compute SVD (ROCm 7.2.0 API - no separate buffer size query)
    // Note: rocsolver_dgesvd modifies the input matrix d_M
    std::cerr << "DEBUG: Calling rocsolver_dgesvd(m=" << m << ", n=" << n << ", lda=" << m << ", ldu=" << m << ", ldv=" << k << ")" << std::endl;

    ROCSOLVER_CHECK(rocsolver_dgesvd(
        rocblas_h,               // Note: use rocblas_handle, not rocsolver_handle
        rocblas_svect_singular,  // Compute U
        rocblas_svect_singular,  // Compute V (will be transposed to Vh)
        m, n,
        d_M, m,                  // Input matrix (will be destroyed)
        result.d_S,              // Singular values output
        result.d_U, m,           // Left singular vectors output
        result.d_Vh, k,          // Right singular vectors output (V^T stored as [k x n])
        d_E,                     // Superdiagonal elements (not used)
        rocblas_outofplace,      // Workspace mode
        workspace.d_info         // Info output
    ));

    // Check for errors
    int info;
    HIP_CHECK(hipMemcpy(&info, workspace.d_info, sizeof(int), hipMemcpyDeviceToHost));
    if (info != 0) {
        std::cerr << "rocsolver_dgesvd failed with info = " << info << std::endl;
        HIP_CHECK(hipFree(d_E));
        throw std::runtime_error("SVD computation failed");
    }

    HIP_CHECK(hipFree(d_E));

    return result;
}

AccurateSVDResult AccurateSVD_GPU::decompose_recursive(double* d_M, int m, int n, int depth) {
    // Base case 1: Maximum recursion depth reached
    if (depth >= max_recursion_depth) {
        return standard_svd(d_M, m, n);
    }

    // Base case 2: Matrix too small to refine
    int k = std::min(m, n);
    if (k < 2) {
        return standard_svd(d_M, m, n);
    }

    // CRITICAL FIX: Copy M before SVD because standard_svd destroys it
    // We'll need the original M for recursive refinement
    double* d_M_copy;
    HIP_CHECK(hipMalloc(&d_M_copy, m * n * sizeof(double)));
    HIP_CHECK(hipMemcpy(d_M_copy, d_M, m * n * sizeof(double), hipMemcpyDeviceToDevice));

    // Step 1: Compute standard SVD
    // NOTE: This consumes d_M (it gets overwritten)
    // Use d_M (parameter) for SVD, keep d_M_copy for later projection
    AccurateSVDResult result = standard_svd(d_M, m, n);

    // Step 2: Find degradation threshold
    int p = find_degradation_threshold(result.d_S, k);

    // Base case 3: No degradation found, return standard SVD
    if (p == -1 || p == k - 1) {
        HIP_CHECK(hipFree(d_M_copy));  // Clean up M_copy
        return result;
    }

    // Step 3: Recursive refinement needed
    int k_sub = k - p;  // Size of inaccurate subspace

    // Project original M onto inaccurate subspace: X = U[:, p:]^T @ M_copy @ Vh[p:, :]^T
    double* d_X;
    HIP_CHECK(hipMalloc(&d_X, k_sub * k_sub * sizeof(double)));

    // CRITICAL: Correct offsets for column-major storage
    // U is [m x k] with lda=m, so column p starts at offset p*m
    // Vh is [k x n] with ldv=k, so row p starts at offset p (NOT p*n!)
    double* d_U_sub = result.d_U + p * m;    // U[:, p:] column p in column-major
    double* d_Vh_sub = result.d_Vh + p;      // Vh[p:, :] row p in column-major [k x n]

    // First: T = U[:, p:]^T @ M_copy  (k_sub x n)
    double* d_T;
    HIP_CHECK(hipMalloc(&d_T, k_sub * n * sizeof(double)));

    double alpha = 1.0;
    double beta = 0.0;
    ROCBLAS_CHECK(rocblas_dgemm(
        rocblas_h,
        rocblas_operation_transpose,
        rocblas_operation_none,
        k_sub, n, m,
        &alpha,
        d_U_sub, m,              // U[:, p:] shape [m x k_sub], lda=m
        d_M_copy, m,             // M_copy shape [m x n], lda=m
        &beta,
        d_T, k_sub               // T shape [k_sub x n], lda=k_sub
    ));

    // Then: X = T @ Vh[p:, :]^T  (k_sub x k_sub)
    // Vh[p:, :] is already V^T, so we need (V^T)^T = V
    ROCBLAS_CHECK(rocblas_dgemm(
        rocblas_h,
        rocblas_operation_none,
        rocblas_operation_transpose,
        k_sub, k_sub, n,
        &alpha,
        d_T, k_sub,              // T shape [k_sub x n], lda=k_sub
        d_Vh_sub, k,             // Vh[p:, :] shape [k_sub x n], lda=k (from full Vh)
        &beta,
        d_X, k_sub               // X shape [k_sub x k_sub], lda=k_sub
    ));

    // Clean up temporaries
    HIP_CHECK(hipFree(d_T));

    // Step 4: Recursively compute accurate SVD of X
    AccurateSVDResult sub_result = decompose_recursive(d_X, k_sub, k_sub, depth + 1);

    // Step 5: Update U[:, p:] = U[:, p:] @ U_sub
    double* d_U_new;
    HIP_CHECK(hipMalloc(&d_U_new, m * k_sub * sizeof(double)));
    ROCBLAS_CHECK(rocblas_dgemm(
        rocblas_h,
        rocblas_operation_none,
        rocblas_operation_none,
        m, k_sub, k_sub,
        &alpha,
        d_U_sub, m,
        sub_result.d_U, k_sub,
        &beta,
        d_U_new, m
    ));
    HIP_CHECK(hipMemcpy(d_U_sub, d_U_new, m * k_sub * sizeof(double), hipMemcpyDeviceToDevice));
    HIP_CHECK(hipFree(d_U_new));

    // Step 6: Update Vh[p:, :] = Vh_sub @ Vh[p:, :]
    double* d_Vh_new;
    HIP_CHECK(hipMalloc(&d_Vh_new, k_sub * n * sizeof(double)));
    ROCBLAS_CHECK(rocblas_dgemm(
        rocblas_h,
        rocblas_operation_none,
        rocblas_operation_none,
        k_sub, n, k_sub,
        &alpha,
        sub_result.d_Vh, k_sub,
        d_Vh_sub, k,
        &beta,
        d_Vh_new, k_sub
    ));
    HIP_CHECK(hipMemcpy(d_Vh_sub, d_Vh_new, k_sub * n * sizeof(double), hipMemcpyDeviceToDevice));
    HIP_CHECK(hipFree(d_Vh_new));

    // Step 7: Update S[p:]
    HIP_CHECK(hipMemcpy(result.d_S + p, sub_result.d_S, k_sub * sizeof(double), hipMemcpyDeviceToDevice));

    // Clean up subspace result (we've copied the data out)
    // Note: sub_result destructor will free its memory

    // Clean up M_copy
    HIP_CHECK(hipFree(d_M_copy));

    return result;
}

AccurateSVDResult AccurateSVD_GPU::decompose(double* d_M, int m, int n) {
    // Copy input matrix to avoid modifying it
    double* d_M_copy;
    HIP_CHECK(hipMalloc(&d_M_copy, m * n * sizeof(double)));
    HIP_CHECK(hipMemcpy(d_M_copy, d_M, m * n * sizeof(double), hipMemcpyDeviceToDevice));

    // Perform recursive decomposition
    AccurateSVDResult result = decompose_recursive(d_M_copy, m, n, 0);

    // d_M_copy was consumed by decompose_recursive, no need to free

    return result;
}

AccurateSVDResult AccurateSVD_GPU::decompose_inplace(double* d_M, int m, int n) {
    return decompose_recursive(d_M, m, n, 0);
}

// ============================================================================
// Utility Functions
// ============================================================================

int compute_truncation_dim(
    double* d_S,
    int k,
    int max_bond_dim,
    double cutoff,
    double* h_trunc_error
) {
    // Copy singular values to host
    std::vector<double> h_S(k);
    HIP_CHECK(hipMemcpy(h_S.data(), d_S, k * sizeof(double), hipMemcpyDeviceToHost));

    // Compute sum of all squared singular values
    double total_weight = 0.0;
    for (int i = 0; i < k; i++) {
        total_weight += h_S[i] * h_S[i];
    }

    // Find truncation dimension
    int keep_dim = std::min(k, max_bond_dim);

    // Check cutoff criterion
    double kept_weight = 0.0;
    for (int i = 0; i < keep_dim; i++) {
        kept_weight += h_S[i] * h_S[i];
    }

    double discarded_weight = total_weight - kept_weight;
    double trunc_err = (total_weight > 0.0) ? discarded_weight / total_weight : 0.0;

    // If truncation error exceeds cutoff, try to keep more singular values
    if (trunc_err > cutoff && keep_dim < k) {
        // Find the dimension where truncation error <= cutoff
        for (int d = keep_dim + 1; d <= k; d++) {
            kept_weight += h_S[d - 1] * h_S[d - 1];
            discarded_weight = total_weight - kept_weight;
            trunc_err = (total_weight > 0.0) ? discarded_weight / total_weight : 0.0;

            if (trunc_err <= cutoff) {
                keep_dim = d;
                break;
            }
        }
    }

    if (h_trunc_error) {
        *h_trunc_error = trunc_err;
    }

    return keep_dim;
}
