#include "stream_coordinator.h"
#include "heisenberg_mpo_real.h"
#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>
#include <cmath>

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Test: Energy from Loaded MPS (No Sweeps)" << std::endl;
    std::cout << "========================================" << std::endl;

    int L = 8;
    int chi_max = 32;
    int d = 2;
    int D_mpo = 5;
    int n_streams = 2;
    int max_bond = 32;

    std::cout << "Parameters:" << std::endl;
    std::cout << "  L = " << L << std::endl;
    std::cout << "  chi_max = " << chi_max << std::endl;
    std::cout << "  n_streams = " << n_streams << std::endl;

    try {
        // Create coordinator
        StreamCoordinator coordinator(n_streams, L, chi_max, d, D_mpo, max_bond);

        // Build and set MPO
        std::cout << "\nBuilding Heisenberg MPO..." << std::endl;
        std::vector<double*> d_mpo_tensors = build_heisenberg_mpo_real_gpu(L);
        coordinator.set_mpo(d_mpo_tensors.data());

        // Load MPS from binary
        std::cout << "Loading MPS from binary..." << std::endl;
        if (!coordinator.load_mps_from_binary("/tmp/heisenberg_L8_mps_initial.bin")) {
            std::cerr << "ERROR: Failed to load MPS" << std::endl;
            return 1;
        }
        std::cout << "✓ MPS loaded" << std::endl;

        // Now compute energy WITHOUT doing any sweeps
        // Just rebuild environments and do a boundary merge
        std::cout << "\nComputing energy from loaded MPS..." << std::endl;
        
        // The energy is computed during boundary merges
        // Let's do one iteration but we need to call the internal methods
        // For now, let's just run one iteration and see what happens
        std::cout << "\nRunning one iteration to compute energy..." << std::endl;
        double energy = coordinator.run_iteration(0);
        
        std::cout << "\nEnergy from iteration: " << std::fixed << std::setprecision(10) << energy << std::endl;
        
        // Exact energy
        double E_exact = -3.374931816815;
        std::cout << "E_exact = " << std::fixed << std::setprecision(12) << E_exact << std::endl;
        std::cout << "|Error| = " << std::scientific << std::setprecision(3) << std::abs(energy - E_exact) << std::endl;

        // Cleanup
        for (int i = 0; i < L; i++) {
            hipFree(d_mpo_tensors[i]);
        }

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\n✓ Test complete" << std::endl;
    return 0;
}
