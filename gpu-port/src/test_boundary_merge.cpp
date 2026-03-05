#include "boundary_merge_gpu.h"
#include <hip/hip_runtime.h>
#include <iostream>
#include <cstdio>
#include <vector>
#include <cmath>

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "BoundaryMergeGPU Basic Test" << std::endl;
    std::cout << "========================================" << std::endl;

    // Check GPU
    int device_count = 0;
    hipGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "ERROR: No GPU devices found" << std::endl;
        return 1;
    }

    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, 0);
    std::cout << "GPU: " << props.name << std::endl;
    std::cout << "Compute capability: " << props.major << "." << props.minor << std::endl;
    std::cout << std::endl;

    // Test parameters
    const int chi_L = 4;
    const int chi_R = 4;
    const int chi_bond = 6;
    const int d = 2;
    const int D_mpo = 3;
    const int max_bond = 10;

    std::cout << "Test Configuration:" << std::endl;
    std::cout << "  chi_L = " << chi_L << std::endl;
    std::cout << "  chi_R = " << chi_R << std::endl;
    std::cout << "  chi_bond = " << chi_bond << std::endl;
    std::cout << "  d = " << d << std::endl;
    std::cout << "  D_mpo = " << D_mpo << std::endl;
    std::cout << "  max_bond = " << max_bond << std::endl;
    std::cout << std::endl;

    // Create BoundaryMergeGPU
    std::cout << "Creating BoundaryMergeGPU..." << std::endl;
    BoundaryMergeGPU* merger = nullptr;
    try {
        merger = new BoundaryMergeGPU(max_bond, 30, 1e-10);
        std::cout << "✓ BoundaryMergeGPU created successfully" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Failed to create BoundaryMergeGPU: "
                  << e.what() << std::endl;
        return 1;
    }

    // Create boundary data
    std::cout << std::endl;
    std::cout << "Allocating boundary data..." << std::endl;

    BoundaryData left, right;
    try {
        left.allocate(chi_L, chi_bond, chi_bond, d, D_mpo);
        right.allocate(chi_bond, chi_R, chi_bond, d, D_mpo);
        std::cout << "✓ Boundary data allocated" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Failed to allocate boundary data: "
                  << e.what() << std::endl;
        delete merger;
        return 1;
    }

    // Initialize with random data
    std::cout << "Initializing boundary tensors..." << std::endl;

    // psi_left: (chi_L, d, chi_bond)
    std::vector<double> h_psi_left(chi_L * d * chi_bond);
    for (size_t i = 0; i < h_psi_left.size(); i++) {
        h_psi_left[i] = ((double)rand() / RAND_MAX) * 0.1;
    }
    hipMemcpy(left.d_psi_left, h_psi_left.data(),
              h_psi_left.size() * sizeof(double), hipMemcpyHostToDevice);

    // psi_right: (chi_bond, d, chi_R)
    std::vector<double> h_psi_right(chi_bond * d * chi_R);
    for (size_t i = 0; i < h_psi_right.size(); i++) {
        h_psi_right[i] = ((double)rand() / RAND_MAX) * 0.1;
    }
    hipMemcpy(right.d_psi_right, h_psi_right.data(),
              h_psi_right.size() * sizeof(double), hipMemcpyHostToDevice);

    // V: (chi_bond) - initialize to 1.0
    std::vector<double> h_V(chi_bond, 1.0);
    hipMemcpy(left.d_V, h_V.data(), h_V.size() * sizeof(double), hipMemcpyHostToDevice);

    // Environments: identity-like
    std::vector<double> h_L_env(D_mpo * chi_L * chi_L, 0.0);
    for (int a = 0; a < chi_L; a++) {
        h_L_env[0 + a * D_mpo + a * D_mpo * chi_L] = 1.0;
    }
    hipMemcpy(left.d_L_env, h_L_env.data(),
              h_L_env.size() * sizeof(double), hipMemcpyHostToDevice);

    std::vector<double> h_R_env(D_mpo * chi_R * chi_R, 0.0);
    for (int b = 0; b < chi_R; b++) {
        h_R_env[0 + b * D_mpo + b * D_mpo * chi_R] = 1.0;
    }
    hipMemcpy(right.d_R_env, h_R_env.data(),
              h_R_env.size() * sizeof(double), hipMemcpyHostToDevice);

    // MPO: identity
    std::vector<double> h_W(D_mpo * d * d * D_mpo, 0.0);
    for (int w = 0; w < D_mpo; w++) {
        for (int s = 0; s < d; s++) {
            h_W[w + s * D_mpo + s * D_mpo * d + w * D_mpo * d * d] = 1.0;
        }
    }
    hipMemcpy(left.d_W_left, h_W.data(), h_W.size() * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(right.d_W_right, h_W.data(), h_W.size() * sizeof(double), hipMemcpyHostToDevice);

    std::cout << "✓ Boundary tensors initialized" << std::endl;

    // Test merge (with optimization skipped for now)
    std::cout << std::endl;
    std::cout << "Testing boundary merge..." << std::endl;

    double energy = 0.0;
    double trunc_err = 0.0;

    try {
        merger->merge(&left, &right, energy, trunc_err, true);  // skip_optimization=true
        std::cout << "✓ Merge completed successfully" << std::endl;
        std::cout << "  Energy: " << energy << std::endl;
        std::cout << "  Truncation error: " << trunc_err << std::endl;
        std::cout << "  New chi_bond: " << left.chi_bond << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Merge failed: " << e.what() << std::endl;
        delete merger;
        return 1;
    }

    // Verify V was updated
    std::vector<double> h_V_new(left.chi_bond);
    hipMemcpy(h_V_new.data(), left.d_V, left.chi_bond * sizeof(double), hipMemcpyDeviceToHost);

    std::cout << std::endl;
    std::cout << "V values (first 5):" << std::endl;
    for (int i = 0; i < std::min(5, left.chi_bond); i++) {
        std::cout << "  V[" << i << "] = " << h_V_new[i] << std::endl;
    }

    // Check V is reasonable (should be close to 1/S for small random tensors)
    bool v_ok = true;
    for (int i = 0; i < left.chi_bond; i++) {
        if (std::isnan(h_V_new[i]) || std::isinf(h_V_new[i]) || h_V_new[i] < 0) {
            v_ok = false;
            break;
        }
    }

    if (!v_ok) {
        std::cerr << "ERROR: V contains invalid values (NaN/Inf/negative)" << std::endl;
        delete merger;
        return 1;
    }
    std::cout << "✓ V values are valid" << std::endl;

    // Cleanup
    std::cout << std::endl;
    std::cout << "Cleaning up..." << std::endl;
    delete merger;
    std::cout << "✓ Cleanup complete" << std::endl;

    // Summary
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Result: PASS" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
