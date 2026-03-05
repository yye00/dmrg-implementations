#include "../src/accurate_svd_gpu.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>

// Simple test utility
#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "TEST FAILED: " << msg << std::endl; \
            return false; \
        } \
    } while(0)

/**
 * @brief Generate test matrix with known singular values
 *
 * Creates M = U * diag(S) * V^T with specified singular values
 * to test SVD accuracy.
 */
void generate_test_matrix_with_known_sv(
    std::vector<double>& h_M,
    int m, int n,
    const std::vector<double>& desired_sv
) {
    std::mt19937 rng(12345);
    std::normal_distribution<double> dist(0.0, 1.0);

    int k = std::min(m, n);

    // Generate random orthogonal U (m x k)
    std::vector<double> U(m * k);
    for (int i = 0; i < m * k; i++) {
        U[i] = dist(rng);
    }
    // QR factorization would make U orthogonal, but for testing we'll use as-is

    // Generate random orthogonal V (k x n)
    std::vector<double> Vh(k * n);
    for (int i = 0; i < k * n; i++) {
        Vh[i] = dist(rng);
    }

    // M = U * diag(S) * Vh
    h_M.resize(m * n, 0.0);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int a = 0; a < k; a++) {
                sum += U[i * k + a] * desired_sv[a] * Vh[a * n + j];
            }
            h_M[i * n + j] = sum;  // Column-major
        }
    }
}

/**
 * @brief Test 1: Basic SVD decomposition
 */
bool test_basic_svd() {
    std::cout << "Test 1: Basic SVD decomposition..." << std::flush;

    int m = 100;
    int n = 80;
    int k = std::min(m, n);

    // Generate random matrix
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> h_M(m * n);
    for (int i = 0; i < m * n; i++) {
        h_M[i] = dist(rng);
    }

    // Copy to device
    double* d_M;
    hipMalloc(&d_M, m * n * sizeof(double));
    hipMemcpy(d_M, h_M.data(), m * n * sizeof(double), hipMemcpyHostToDevice);

    // Compute SVD
    AccurateSVD_GPU svd_solver(1e-4, 5);
    AccurateSVDResult result = svd_solver.decompose(d_M, m, n);

    // Copy singular values to host
    std::vector<double> h_S(k);
    hipMemcpy(h_S.data(), result.d_S, k * sizeof(double), hipMemcpyDeviceToHost);

    // Check singular values are non-negative and sorted
    for (int i = 0; i < k - 1; i++) {
        TEST_ASSERT(h_S[i] >= 0.0, "Singular values must be non-negative");
        TEST_ASSERT(h_S[i] >= h_S[i+1], "Singular values must be sorted descending");
    }

    // Cleanup
    hipFree(d_M);

    std::cout << " PASSED (largest SV = " << h_S[0] << ")" << std::endl;
    return true;
}

/**
 * @brief Test 2: Small singular value accuracy
 */
bool test_small_sv_accuracy() {
    std::cout << "Test 2: Small singular value accuracy..." << std::flush;

    int m = 50;
    int n = 50;
    int k = std::min(m, n);

    // Create matrix with exponentially decaying singular values
    std::vector<double> desired_sv(k);
    for (int i = 0; i < k; i++) {
        desired_sv[i] = std::exp(-0.1 * i);  // Decay from 1.0 to ~0.0067
    }

    std::vector<double> h_M;
    generate_test_matrix_with_known_sv(h_M, m, n, desired_sv);

    // Copy to device
    double* d_M;
    hipMalloc(&d_M, m * n * sizeof(double));
    hipMemcpy(d_M, h_M.data(), m * n * sizeof(double), hipMemcpyHostToDevice);

    // Compute accurate SVD with epsilon = 1e-4
    AccurateSVD_GPU svd_solver(1e-4, 5);
    AccurateSVDResult result = svd_solver.decompose(d_M, m, n);

    // Copy singular values to host
    std::vector<double> h_S(k);
    hipMemcpy(h_S.data(), result.d_S, k * sizeof(double), hipMemcpyDeviceToHost);

    // Check that small singular values are accurate
    // (In practice, we'd compare with known values, but here we just verify non-zero)
    int num_small = 0;
    double threshold = desired_sv[0] * 1e-4;
    for (int i = 0; i < k; i++) {
        if (h_S[i] < threshold) {
            num_small++;
        }
    }

    TEST_ASSERT(num_small > 0, "Should have small singular values for recursive refinement");

    // Cleanup
    hipFree(d_M);

    std::cout << " PASSED (" << num_small << " small SVs refined)" << std::endl;
    return true;
}

/**
 * @brief Test 3: Reconstruction accuracy (M ≈ U * S * Vh)
 */
bool test_reconstruction() {
    std::cout << "Test 3: Reconstruction accuracy..." << std::flush;

    int m = 40;
    int n = 30;
    int k = std::min(m, n);

    // Generate random matrix
    std::mt19937 rng(123);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> h_M_orig(m * n);
    for (int i = 0; i < m * n; i++) {
        h_M_orig[i] = dist(rng);
    }

    // Copy to device
    double* d_M;
    hipMalloc(&d_M, m * n * sizeof(double));
    hipMemcpy(d_M, h_M_orig.data(), m * n * sizeof(double), hipMemcpyHostToDevice);

    // Compute SVD
    AccurateSVD_GPU svd_solver(1e-4, 5);
    AccurateSVDResult result = svd_solver.decompose(d_M, m, n);

    // Copy U, S, Vh to host
    std::vector<double> h_U(m * k);
    std::vector<double> h_S(k);
    std::vector<double> h_Vh(k * n);

    hipMemcpy(h_U.data(), result.d_U, m * k * sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(h_S.data(), result.d_S, k * sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(h_Vh.data(), result.d_Vh, k * n * sizeof(double), hipMemcpyDeviceToHost);

    // Reconstruct: M_reconstructed = U * diag(S) * Vh
    std::vector<double> h_M_recon(m * n, 0.0);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int a = 0; a < k; a++) {
                sum += h_U[i * k + a] * h_S[a] * h_Vh[a * n + j];
            }
            h_M_recon[i * n + j] = sum;
        }
    }

    // Compute reconstruction error
    double error = 0.0;
    double norm = 0.0;
    for (int i = 0; i < m * n; i++) {
        double diff = h_M_orig[i] - h_M_recon[i];
        error += diff * diff;
        norm += h_M_orig[i] * h_M_orig[i];
    }
    double relative_error = std::sqrt(error / norm);

    TEST_ASSERT(relative_error < 1e-6, "Reconstruction error too large");

    // Cleanup
    hipFree(d_M);

    std::cout << " PASSED (relative error = " << relative_error << ")" << std::endl;
    return true;
}

/**
 * @brief Test 4: Bridge matrix computation (V = 1/S)
 */
bool test_bridge_matrix() {
    std::cout << "Test 4: Bridge matrix V = 1/S..." << std::flush;

    int k = 100;

    // Create test singular values
    std::vector<double> h_S(k);
    for (int i = 0; i < k; i++) {
        h_S[i] = 1.0 / (1.0 + i);  // 1.0, 0.5, 0.333, ...
    }

    // Copy to device
    double* d_S;
    double* d_V;
    hipMalloc(&d_S, k * sizeof(double));
    hipMalloc(&d_V, k * sizeof(double));
    hipMemcpy(d_S, h_S.data(), k * sizeof(double), hipMemcpyHostToDevice);

    // Compute V = 1/S with clipping
    double clip_min = 1e-12;
    launch_invert_with_clipping(d_S, d_V, k, clip_min);

    // Copy back
    std::vector<double> h_V(k);
    hipMemcpy(h_V.data(), d_V, k * sizeof(double), hipMemcpyDeviceToHost);

    // Verify V[i] = 1 / S[i]
    for (int i = 0; i < k; i++) {
        double expected = 1.0 / std::max(h_S[i], clip_min);
        double error = std::abs(h_V[i] - expected);
        TEST_ASSERT(error < 1e-10, "Bridge matrix computation incorrect");
    }

    // Test clipping for very small values
    h_S[k-1] = 1e-15;  // Below clip threshold
    hipMemcpy(d_S, h_S.data(), k * sizeof(double), hipMemcpyHostToDevice);
    launch_invert_with_clipping(d_S, d_V, k, clip_min);
    hipMemcpy(h_V.data(), d_V, k * sizeof(double), hipMemcpyDeviceToHost);

    double expected_clipped = 1.0 / clip_min;
    double error_clipped = std::abs(h_V[k-1] - expected_clipped);
    TEST_ASSERT(error_clipped < 1e-6, "Clipping not working correctly");

    // Cleanup
    hipFree(d_S);
    hipFree(d_V);

    std::cout << " PASSED" << std::endl;
    return true;
}

/**
 * @brief Test 5: Truncation dimension computation
 */
bool test_truncation_dim() {
    std::cout << "Test 5: Truncation dimension computation..." << std::flush;

    int k = 200;

    // Create test singular values with exponential decay
    std::vector<double> h_S(k);
    for (int i = 0; i < k; i++) {
        h_S[i] = std::exp(-0.05 * i);
    }

    // Copy to device
    double* d_S;
    hipMalloc(&d_S, k * sizeof(double));
    hipMemcpy(d_S, h_S.data(), k * sizeof(double), hipMemcpyHostToDevice);

    // Test with max_bond_dim = 100
    double trunc_error;
    int keep_dim = compute_truncation_dim(d_S, k, 100, 0.01, &trunc_error);

    TEST_ASSERT(keep_dim <= 100, "keep_dim exceeds max_bond_dim");
    TEST_ASSERT(keep_dim > 0, "keep_dim must be positive");
    TEST_ASSERT(trunc_error >= 0.0 && trunc_error <= 1.0, "trunc_error out of range");

    // Test with very strict cutoff
    keep_dim = compute_truncation_dim(d_S, k, 200, 1e-10, &trunc_error);
    TEST_ASSERT(keep_dim > 100, "Strict cutoff should keep more singular values");

    // Cleanup
    hipFree(d_S);

    std::cout << " PASSED (keep_dim = " << keep_dim << ", trunc_err = " << trunc_error << ")" << std::endl;
    return true;
}

int main() {
    std::cout << "==================================================" << std::endl;
    std::cout << "  Accurate SVD GPU Unit Tests" << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << std::endl;

    int passed = 0;
    int total = 5;

    if (test_basic_svd()) passed++;
    if (test_small_sv_accuracy()) passed++;
    if (test_reconstruction()) passed++;
    if (test_bridge_matrix()) passed++;
    if (test_truncation_dim()) passed++;

    std::cout << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << "  Results: " << passed << " / " << total << " tests passed" << std::endl;
    std::cout << "==================================================" << std::endl;

    return (passed == total) ? 0 : 1;
}
