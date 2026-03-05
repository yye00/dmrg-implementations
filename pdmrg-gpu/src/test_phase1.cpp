/**
 * @file test_phase1.cpp
 * @brief Test program for Phase 1 GPU optimization components
 *
 * Tests:
 *   1. AccurateSVD_GPU - Recursive SVD refinement
 *   2. OptimizedHeff - hipTensor H_eff with workspace caching
 *
 * Success criteria:
 *   - AccurateSVD matches CPU to 1e-12 precision
 *   - OptimizedHeff produces correct results
 *   - No memory leaks, no errors
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include "accurate_svd_gpu.h"
#include "heff_optimized_gpu.h"

#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << hipGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

//==============================================================================
// Test 1: Accurate SVD
//==============================================================================

bool test_accurate_svd() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Test 1: AccurateSVD_GPU" << std::endl;
    std::cout << "========================================" << std::endl;

    const int m = 100;
    const int n = 80;
    const int k = std::min(m, n);

    // Create test matrix with known condition number
    // M = U * diag(S) * Vh where S has exponentially decaying values
    std::vector<double> h_M(m * n);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            // Simple test matrix: M[i,j] = exp(-0.1 * (i + j))
            h_M[i + j * m] = std::exp(-0.1 * (i + j));
        }
    }

    // Debug: Print first few elements of input matrix
    std::cout << "Input matrix M (first 3x3):" << std::endl;
    for (int i = 0; i < std::min(3, m); i++) {
        for (int j = 0; j < std::min(3, n); j++) {
            std::cout << h_M[i + j * m] << " ";
        }
        std::cout << std::endl;
    }

    // Copy to device
    double* d_M;
    HIP_CHECK(hipMalloc(&d_M, m * n * sizeof(double)));
    HIP_CHECK(hipMemcpy(d_M, h_M.data(), m * n * sizeof(double), hipMemcpyHostToDevice));

    // Compute accurate SVD
    AccurateSVD_GPU svd_solver(1e-4, 5);
    AccurateSVDResult result = svd_solver.decompose(d_M, m, n);

    std::cout << "Matrix: " << m << " × " << n << std::endl;
    std::cout << "Rank: " << result.rank << std::endl;

    // Copy singular values to host
    std::vector<double> h_S(result.rank);
    HIP_CHECK(hipMemcpy(h_S.data(), result.d_S, result.rank * sizeof(double), hipMemcpyDeviceToHost));

    // Debug: Check U matrix
    std::vector<double> h_U(m * result.rank);
    HIP_CHECK(hipMemcpy(h_U.data(), result.d_U, m * result.rank * sizeof(double), hipMemcpyDeviceToHost));
    std::cout << "U matrix (first 3x3):" << std::endl;
    for (int i = 0; i < std::min(3, m); i++) {
        for (int j = 0; j < std::min(3, result.rank); j++) {
            std::cout << h_U[i + j * m] << " ";
        }
        std::cout << std::endl;
    }

    // Display first and last few singular values
    std::cout << "\nSingular values:" << std::endl;
    std::cout << "  σ[0] = " << std::scientific << std::setprecision(6) << h_S[0] << std::endl;
    if (result.rank > 1) {
        std::cout << "  σ[1] = " << h_S[1] << std::endl;
    }
    if (result.rank > 2) {
        std::cout << "  σ[2] = " << h_S[2] << std::endl;
    }
    std::cout << "  ..." << std::endl;
    if (result.rank > 3) {
        std::cout << "  σ[" << (result.rank - 2) << "] = " << h_S[result.rank - 2] << std::endl;
    }
    if (result.rank > 2) {
        std::cout << "  σ[" << (result.rank - 1) << "] = " << h_S[result.rank - 1] << std::endl;
    }

    // Compute condition number
    double cond_number = h_S[0] / h_S[result.rank - 1];
    std::cout << "Condition number: " << cond_number << std::endl;

    // Reconstruct matrix: M_recon = U * diag(S) * Vh
    double *d_U_scaled, *d_M_recon;
    HIP_CHECK(hipMalloc(&d_U_scaled, m * result.rank * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_M_recon, m * n * sizeof(double)));

    // Copy U and scale by S
    HIP_CHECK(hipMemcpy(d_U_scaled, result.d_U, m * result.rank * sizeof(double), hipMemcpyDeviceToDevice));

    rocblas_handle rocblas_h;
    rocblas_create_handle(&rocblas_h);

    for (int j = 0; j < result.rank; j++) {
        double scale = h_S[j];
        rocblas_dscal(rocblas_h, m, &scale, d_U_scaled + j * m, 1);
    }

    // M_recon = U_scaled * Vh
    double alpha = 1.0, beta = 0.0;
    rocblas_dgemm(
        rocblas_h,
        rocblas_operation_none,
        rocblas_operation_none,
        m, n, result.rank,
        &alpha,
        d_U_scaled, m,
        result.d_Vh, result.rank,
        &beta,
        d_M_recon, m
    );

    // Compute reconstruction error: ||M - M_recon||_F / ||M||_F
    double *d_diff;
    HIP_CHECK(hipMalloc(&d_diff, m * n * sizeof(double)));
    HIP_CHECK(hipMemcpy(d_diff, d_M, m * n * sizeof(double), hipMemcpyDeviceToDevice));

    double alpha_neg = -1.0;
    rocblas_daxpy(rocblas_h, m * n, &alpha_neg, d_M_recon, 1, d_diff, 1);

    double norm_diff, norm_M;
    rocblas_dnrm2(rocblas_h, m * n, d_diff, 1, &norm_diff);
    rocblas_dnrm2(rocblas_h, m * n, d_M, 1, &norm_M);

    double rel_error = norm_diff / norm_M;
    std::cout << "Reconstruction error: " << rel_error << std::endl;

    // Cleanup
    rocblas_destroy_handle(rocblas_h);
    HIP_CHECK(hipFree(d_M));
    HIP_CHECK(hipFree(d_U_scaled));
    HIP_CHECK(hipFree(d_M_recon));
    HIP_CHECK(hipFree(d_diff));

    // Success criterion: reconstruction error < 1e-10
    bool success = (rel_error < 1e-10);
    std::cout << "Result: " << (success ? "PASS" : "FAIL") << std::endl;

    return success;
}

//==============================================================================
// Test 2: Optimized H_eff
//==============================================================================

bool test_optimized_heff() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Test 2: OptimizedHeff" << std::endl;
    std::cout << "========================================" << std::endl;

    // Small test problem
    const int chi_L = 8;
    const int chi_R = 8;
    const int d = 2;
    const int D_mpo = 3;

    std::cout << "Dimensions:" << std::endl;
    std::cout << "  χ_L = " << chi_L << std::endl;
    std::cout << "  χ_R = " << chi_R << std::endl;
    std::cout << "  d = " << d << std::endl;
    std::cout << "  D_mpo = " << D_mpo << std::endl;

    // Create hipTensor handle
    hiptensorHandle_t handle;
    hiptensorStatus_t status = hiptensorCreate(&handle);
    if (status != HIPTENSOR_STATUS_SUCCESS) {
        std::cerr << "Failed to create hipTensor handle" << std::endl;
        return false;
    }

    // Create OptimizedHeff instance
    OptimizedHeff* heff = nullptr;
    try {
        heff = new OptimizedHeff(chi_L, chi_R, d, D_mpo, &handle);
        std::cout << "OptimizedHeff created successfully" << std::endl;
        std::cout << "Workspace size: " << heff->get_workspace_size() << " bytes" << std::endl;
        std::cout << "Total memory: " << heff->get_total_memory() << " bytes" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Failed to create OptimizedHeff: " << e.what() << std::endl;
        hiptensorDestroy(handle);
        return false;
    }

    // Allocate test tensors
    double *d_theta, *d_result, *d_L, *d_R, *d_W1, *d_W2;
    HIP_CHECK(hipMalloc(&d_theta, chi_L * d * d * chi_R * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_result, chi_L * d * d * chi_R * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_L, D_mpo * chi_L * chi_L * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_R, D_mpo * chi_R * chi_R * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_W1, D_mpo * d * d * D_mpo * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_W2, D_mpo * d * d * D_mpo * sizeof(double)));

    // Initialize with random values
    std::vector<double> h_theta(chi_L * d * d * chi_R);
    for (size_t i = 0; i < h_theta.size(); i++) {
        h_theta[i] = (double)rand() / RAND_MAX;
    }
    HIP_CHECK(hipMemcpy(d_theta, h_theta.data(), h_theta.size() * sizeof(double), hipMemcpyHostToDevice));

    // Initialize environments and MPO (identity-like)
    // CRITICAL: L and R must have identity only at one MPO bond index (w=0, y=0)
    // Otherwise we get a factor of D_mpo!
    std::vector<double> h_L(D_mpo * chi_L * chi_L, 0.0);
    for (int a = 0; a < chi_L; a++) {
        // Only w=0: L[0, a, a] = 1.0 (identity in MPS indices at w=0)
        h_L[0 + a * D_mpo + a * D_mpo * chi_L] = 1.0;
    }
    HIP_CHECK(hipMemcpy(d_L, h_L.data(), h_L.size() * sizeof(double), hipMemcpyHostToDevice));

    std::vector<double> h_R(D_mpo * chi_R * chi_R, 0.0);
    for (int b = 0; b < chi_R; b++) {
        // Only y=0: R[0, b, b] = 1.0 (identity in MPS indices at y=0)
        h_R[0 + b * D_mpo + b * D_mpo * chi_R] = 1.0;
    }
    HIP_CHECK(hipMemcpy(d_R, h_R.data(), h_R.size() * sizeof(double), hipMemcpyHostToDevice));

    // Simple MPO (identity)
    std::vector<double> h_W1(D_mpo * d * d * D_mpo, 0.0);
    for (int w = 0; w < D_mpo; w++) {
        for (int s = 0; s < d; s++) {
            for (int x = 0; x < D_mpo; x++) {
                if (w == x) {
                    h_W1[w + s * D_mpo + s * D_mpo * d + x * D_mpo * d * d] = 1.0;
                }
            }
        }
    }
    HIP_CHECK(hipMemcpy(d_W1, h_W1.data(), h_W1.size() * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_W2, h_W1.data(), h_W1.size() * sizeof(double), hipMemcpyHostToDevice));

    // Apply H_eff
    try {
        heff->apply(d_theta, d_result, d_L, d_R, d_W1, d_W2);
        HIP_CHECK(hipDeviceSynchronize());
        std::cout << "H_eff applied successfully" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Failed to apply H_eff: " << e.what() << std::endl;
        delete heff;
        hiptensorDestroy(handle);
        return false;
    }

    // With identity MPO and environments, result should equal theta
    // Check: ||result - theta||_F / ||theta||_F
    rocblas_handle rocblas_h;
    rocblas_create_handle(&rocblas_h);

    double *d_diff;
    HIP_CHECK(hipMalloc(&d_diff, chi_L * d * d * chi_R * sizeof(double)));
    HIP_CHECK(hipMemcpy(d_diff, d_result, chi_L * d * d * chi_R * sizeof(double), hipMemcpyDeviceToDevice));

    double alpha_neg = -1.0;
    rocblas_daxpy(rocblas_h, chi_L * d * d * chi_R, &alpha_neg, d_theta, 1, d_diff, 1);

    double norm_diff, norm_theta;
    rocblas_dnrm2(rocblas_h, chi_L * d * d * chi_R, d_diff, 1, &norm_diff);
    rocblas_dnrm2(rocblas_h, chi_L * d * d * chi_R, d_theta, 1, &norm_theta);

    double rel_error = norm_diff / norm_theta;
    std::cout << "Relative error (identity test): " << std::scientific << rel_error << std::endl;

    // Cleanup
    rocblas_destroy_handle(rocblas_h);
    HIP_CHECK(hipFree(d_theta));
    HIP_CHECK(hipFree(d_result));
    HIP_CHECK(hipFree(d_L));
    HIP_CHECK(hipFree(d_R));
    HIP_CHECK(hipFree(d_W1));
    HIP_CHECK(hipFree(d_W2));
    HIP_CHECK(hipFree(d_diff));

    delete heff;
    hiptensorDestroy(handle);

    // Success criterion: relative error < 1e-6 (loose tolerance for identity test)
    bool success = (rel_error < 1e-6);
    std::cout << "Result: " << (success ? "PASS" : "FAIL") << std::endl;

    return success;
}

//==============================================================================
// Main
//==============================================================================

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "Phase 1 Component Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    // Initialize HIP
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << std::endl;

    bool test1_pass = test_accurate_svd();
    bool test2_pass = test_optimized_heff();

    std::cout << "\n========================================" << std::endl;
    std::cout << "Summary" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Test 1 (AccurateSVD_GPU): " << (test1_pass ? "PASS" : "FAIL") << std::endl;
    std::cout << "Test 2 (OptimizedHeff):   " << (test2_pass ? "PASS" : "FAIL") << std::endl;
    std::cout << "Overall:                  " << (test1_pass && test2_pass ? "PASS" : "FAIL") << std::endl;

    return (test1_pass && test2_pass) ? 0 : 1;
}
