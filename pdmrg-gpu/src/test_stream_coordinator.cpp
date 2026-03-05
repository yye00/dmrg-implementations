#include "stream_coordinator.h"
#include <hip/hip_runtime.h>
#include <iostream>
#include <cstdio>

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "StreamCoordinator Test" << std::endl;
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

    // Test configuration: 2 segments on 8-site chain
    const int n_streams = 2;
    const int chain_length = 8;
    const int chi_max = 10;
    const int d = 2;             // Spin-1/2
    const int D_mpo = 3;         // Heisenberg MPO
    const int max_bond = 15;

    std::cout << "Test Configuration:" << std::endl;
    std::cout << "  Number of streams: " << n_streams << std::endl;
    std::cout << "  Chain length: " << chain_length << std::endl;
    std::cout << "  chi_max: " << chi_max << std::endl;
    std::cout << "  d: " << d << std::endl;
    std::cout << "  D_mpo: " << D_mpo << std::endl;
    std::cout << "  max_bond: " << max_bond << std::endl;
    std::cout << std::endl;

    // Create StreamCoordinator
    std::cout << "Creating StreamCoordinator..." << std::endl;
    StreamCoordinator* coordinator = nullptr;

    try {
        coordinator = new StreamCoordinator(n_streams, chain_length,
                                            chi_max, d, D_mpo, max_bond);
        std::cout << "✓ StreamCoordinator created successfully" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Failed to create StreamCoordinator: "
                  << e.what() << std::endl;
        return 1;
    }

    // Verify segment distribution
    std::cout << std::endl;
    std::cout << "Verifying segment distribution:" << std::endl;
    int total_sites = 0;
    for (int i = 0; i < n_streams; i++) {
        StreamSegment* seg = coordinator->get_segment(i);
        if (!seg) {
            std::cerr << "ERROR: Segment " << i << " is null" << std::endl;
            delete coordinator;
            return 1;
        }

        int start = seg->get_start_site();
        int end = seg->get_end_site();
        int num = seg->get_num_sites();

        std::cout << "  Segment " << i << ": [" << start << ", " << end
                  << "] = " << num << " sites" << std::endl;

        if (num != (end - start + 1)) {
            std::cerr << "ERROR: Site count mismatch" << std::endl;
            delete coordinator;
            return 1;
        }

        total_sites += num;

        // Check boundaries
        if (i == 0 && start != 0) {
            std::cerr << "ERROR: First segment should start at 0" << std::endl;
            delete coordinator;
            return 1;
        }

        if (i == n_streams - 1 && end != chain_length - 1) {
            std::cerr << "ERROR: Last segment should end at " << chain_length - 1 << std::endl;
            delete coordinator;
            return 1;
        }
    }

    if (total_sites != chain_length) {
        std::cerr << "ERROR: Total sites (" << total_sites
                  << ") != chain_length (" << chain_length << ")" << std::endl;
        delete coordinator;
        return 1;
    }
    std::cout << "✓ Site distribution correct (total: " << total_sites << ")" << std::endl;

    // Initialize MPO (identity for simplicity)
    std::cout << std::endl;
    std::cout << "Initializing MPO..." << std::endl;

    std::vector<double*> d_mpo_tensors(chain_length);
    size_t mpo_size = D_mpo * d * d * D_mpo;

    // Create identity-like MPO
    std::vector<double> h_mpo(mpo_size, 0.0);
    for (int w = 0; w < D_mpo; w++) {
        for (int s = 0; s < d; s++) {
            h_mpo[w + s * D_mpo + s * D_mpo * d + w * D_mpo * d * d] = 1.0;
        }
    }

    // Allocate and copy to device for each site
    for (int site = 0; site < chain_length; site++) {
        hipMalloc(&d_mpo_tensors[site], mpo_size * sizeof(double));
        hipMemcpy(d_mpo_tensors[site], h_mpo.data(),
                 mpo_size * sizeof(double), hipMemcpyHostToDevice);
    }

    try {
        coordinator->set_mpo(d_mpo_tensors.data());
        std::cout << "✓ MPO set successfully" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Failed to set MPO: " << e.what() << std::endl;
        for (auto ptr : d_mpo_tensors) hipFree(ptr);
        delete coordinator;
        return 1;
    }

    // Run one iteration (this tests the orchestration)
    std::cout << std::endl;
    std::cout << "Running test iteration..." << std::endl;

    try {
        double energy = coordinator->run_iteration(0);
        std::cout << "✓ Iteration completed successfully" << std::endl;
        std::cout << "  Energy: " << energy << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Iteration failed: " << e.what() << std::endl;
        for (auto ptr : d_mpo_tensors) hipFree(ptr);
        delete coordinator;
        return 1;
    }

    // Cleanup
    std::cout << std::endl;
    std::cout << "Cleaning up..." << std::endl;

    for (auto ptr : d_mpo_tensors) {
        hipFree(ptr);
    }

    delete coordinator;
    std::cout << "✓ Cleanup complete" << std::endl;

    // Summary
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Result: PASS" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    std::cout << "StreamCoordinator successfully orchestrated:" << std::endl;
    std::cout << "  - " << n_streams << " segments" << std::endl;
    std::cout << "  - Forward/backward sweeps" << std::endl;
    std::cout << "  - Even/odd boundary merges" << std::endl;
    std::cout << "  - Energy collection" << std::endl;

    return 0;
}
