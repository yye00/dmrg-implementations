#include "accurate_svd_gpu.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <vector>

// Error checking macros
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            throw std::runtime_error("CUDA error"); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error in " << __FILE__ << ":" << __LINE__ \
                      << " - status " << status << std::endl; \
            throw std::runtime_error("cuBLAS error"); \
        } \
    } while(0)

#define CUSOLVER_CHECK(call) \
    do { \
        cusolverStatus_t status = call; \
        if (status != CUSOLVER_STATUS_SUCCESS) { \
            std::cerr << "cuSOLVER error in " << __FILE__ << ":" << __LINE__ \
                      << " - status " << status << std::endl; \
            throw std::runtime_error("cuSOLVER error"); \
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
    cudaStream_t stream
) {
    int block_size = 256;
    int num_blocks = (k + block_size - 1) / block_size;
    invert_with_clipping_kernel<<<num_blocks, block_size, 0, stream>>>(
        d_S, d_V, k, clip_min
    );
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// AccurateSVD_GPU Implementation
// ============================================================================

AccurateSVD_GPU::AccurateSVD_GPU(double eps, int max_depth)
    : epsilon(eps), max_recursion_depth(0)  // TEMP: Disable recursion for debugging
{
    CUBLAS_CHECK(cublasCreate(&cublas_h));
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver_h));
    // Initialize workspace to zeros
    workspace.d_work = nullptr;
    workspace.size = 0;
    workspace.d_info = nullptr;
    workspace.d_rwork = nullptr;
}

AccurateSVD_GPU::~AccurateSVD_GPU() {
    cublasDestroy(cublas_h);
    cusolverDnDestroy(cusolver_h);
}

int AccurateSVD_GPU::find_degradation_threshold(double* d_S, int k) {
    // Copy singular values to host
    std::vector<double> h_S(k);
    CUDA_CHECK(cudaMemcpy(h_S.data(), d_S, k * sizeof(double), cudaMemcpyDeviceToHost));

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
    CUDA_CHECK(cudaMalloc(&result.d_U, m * k * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&result.d_S, k * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&result.d_Vh, k * n * sizeof(double)));

    // Allocate info on device
    if (!workspace.d_info) {
        CUDA_CHECK(cudaMalloc(&workspace.d_info, sizeof(int)));
    }

    // Query workspace size
    int lwork;
    CUSOLVER_CHECK(cusolverDnDgesvd_bufferSize(cusolver_h, m, n, &lwork));

    // Allocate workspace
    workspace.allocate(lwork * sizeof(double), 0);

    // Compute SVD using cuSOLVER
    CUSOLVER_CHECK(cusolverDnDgesvd(
        cusolver_h,
        'S', 'S',
        m, n,
        d_M, m,                  // Input matrix (will be destroyed)
        result.d_S,              // Singular values output
        result.d_U, m,           // Left singular vectors output
        result.d_Vh, k,          // Right singular vectors output (V^T stored as [k x n])
        (double*)workspace.d_work, lwork,
        nullptr,                 // rwork (not needed for real)
        workspace.d_info         // Info output
    ));

    // Check for errors
    int info;
    CUDA_CHECK(cudaMemcpy(&info, workspace.d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (info != 0) {
        std::cerr << "cusolverDnDgesvd failed with info = " << info << std::endl;
        throw std::runtime_error("SVD computation failed");
    }

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
    CUDA_CHECK(cudaMalloc(&d_M_copy, m * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_M_copy, d_M, m * n * sizeof(double), cudaMemcpyDeviceToDevice));

    // Step 1: Compute standard SVD
    // NOTE: This consumes d_M (it gets overwritten)
    // Use d_M (parameter) for SVD, keep d_M_copy for later projection
    AccurateSVDResult result = standard_svd(d_M, m, n);

    // Step 2: Find degradation threshold
    int p = find_degradation_threshold(result.d_S, k);

    // Base case 3: No degradation found, return standard SVD
    if (p == -1 || p == k - 1) {
        CUDA_CHECK(cudaFree(d_M_copy));  // Clean up M_copy
        return result;
    }

    // Step 3: Recursive refinement needed
    int k_sub = k - p;  // Size of inaccurate subspace

    // Project original M onto inaccurate subspace: X = U[:, p:]^T @ M_copy @ Vh[p:, :]^T
    double* d_X;
    CUDA_CHECK(cudaMalloc(&d_X, k_sub * k_sub * sizeof(double)));

    // CRITICAL: Correct offsets for column-major storage
    // U is [m x k] with lda=m, so column p starts at offset p*m
    // Vh is [k x n] with ldv=k, so row p starts at offset p (NOT p*n!)
    double* d_U_sub = result.d_U + p * m;    // U[:, p:] column p in column-major
    double* d_Vh_sub = result.d_Vh + p;      // Vh[p:, :] row p in column-major [k x n]

    // First: T = U[:, p:]^T @ M_copy  (k_sub x n)
    double* d_T;
    CUDA_CHECK(cudaMalloc(&d_T, k_sub * n * sizeof(double)));

    double alpha = 1.0;
    double beta = 0.0;
    CUBLAS_CHECK(cublasDgemm(
        cublas_h,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        k_sub, n, m,
        &alpha,
        d_U_sub, m,              // U[:, p:] shape [m x k_sub], lda=m
        d_M_copy, m,             // M_copy shape [m x n], lda=m
        &beta,
        d_T, k_sub               // T shape [k_sub x n], lda=k_sub
    ));

    // Then: X = T @ Vh[p:, :]^T  (k_sub x k_sub)
    // Vh[p:, :] is already V^T, so we need (V^T)^T = V
    CUBLAS_CHECK(cublasDgemm(
        cublas_h,
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        k_sub, k_sub, n,
        &alpha,
        d_T, k_sub,              // T shape [k_sub x n], lda=k_sub
        d_Vh_sub, k,             // Vh[p:, :] shape [k_sub x n], lda=k (from full Vh)
        &beta,
        d_X, k_sub               // X shape [k_sub x k_sub], lda=k_sub
    ));

    // Clean up temporaries
    CUDA_CHECK(cudaFree(d_T));

    // Step 4: Recursively compute accurate SVD of X
    AccurateSVDResult sub_result = decompose_recursive(d_X, k_sub, k_sub, depth + 1);

    // Step 5: Update U[:, p:] = U[:, p:] @ U_sub
    double* d_U_new;
    CUDA_CHECK(cudaMalloc(&d_U_new, m * k_sub * sizeof(double)));
    CUBLAS_CHECK(cublasDgemm(
        cublas_h,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        m, k_sub, k_sub,
        &alpha,
        d_U_sub, m,
        sub_result.d_U, k_sub,
        &beta,
        d_U_new, m
    ));
    CUDA_CHECK(cudaMemcpy(d_U_sub, d_U_new, m * k_sub * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaFree(d_U_new));

    // Step 6: Update Vh[p:, :] = Vh_sub @ Vh[p:, :]
    double* d_Vh_new;
    CUDA_CHECK(cudaMalloc(&d_Vh_new, k_sub * n * sizeof(double)));
    CUBLAS_CHECK(cublasDgemm(
        cublas_h,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        k_sub, n, k_sub,
        &alpha,
        sub_result.d_Vh, k_sub,
        d_Vh_sub, k,
        &beta,
        d_Vh_new, k_sub
    ));
    CUDA_CHECK(cudaMemcpy(d_Vh_sub, d_Vh_new, k_sub * n * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaFree(d_Vh_new));

    // Step 7: Update S[p:]
    CUDA_CHECK(cudaMemcpy(result.d_S + p, sub_result.d_S, k_sub * sizeof(double), cudaMemcpyDeviceToDevice));

    // Clean up subspace result (we've copied the data out)
    // Note: sub_result destructor will free its memory

    // Clean up M_copy
    CUDA_CHECK(cudaFree(d_M_copy));

    return result;
}

AccurateSVDResult AccurateSVD_GPU::decompose(double* d_M, int m, int n) {
    // Copy input matrix to avoid modifying it
    double* d_M_copy;
    CUDA_CHECK(cudaMalloc(&d_M_copy, m * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_M_copy, d_M, m * n * sizeof(double), cudaMemcpyDeviceToDevice));

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
    CUDA_CHECK(cudaMemcpy(h_S.data(), d_S, k * sizeof(double), cudaMemcpyDeviceToHost));

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
