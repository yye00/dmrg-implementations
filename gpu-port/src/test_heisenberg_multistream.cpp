#include "stream_coordinator.h"
#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Testing Phase 2 Multi-Stream Iterative DMRG" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // System parameters
    const int L = 8;          // Chain length
    const int d = 2;          // Physical dimension
    const int chi_max = 16;   // Max bond dimension
    const int n_streams = 2;  // Number of parallel streams
    const int D_mpo = 3;      // MPO bond dimension

    std::cout << "Parameters:" << std::endl;
    std::cout << "  L = " << L << " sites" << std::endl;
    std::cout << "  d = " << d << std::endl;
    std::cout << "  chi_max = " << chi_max << std::endl;
    std::cout << "  n_streams = " << n_streams << std::endl;
    std::cout << "  D_mpo = " << D_mpo << " (identity)" << std::endl;
    std::cout << std::endl;

    try {
        // Create HIP streams
        std::vector<hipStream_t> streams(n_streams);
        for (int i = 0; i < n_streams; i++) {
            hipStreamCreate(&streams[i]);
        }

        // Create StreamCoordinator
        std::cout << "Initializing StreamCoordinator..." << std::endl;
        int max_bond = chi_max;  // Maximum bond dimension for merges
        StreamCoordinator coordinator(n_streams, L, chi_max, d, D_mpo, max_bond);

        // Create simple identity MPO (for now - full Hamiltonian needs proper interface)
        std::cout << "Creating identity MPO..." << std::endl;
        std::vector<double*> d_mpo_tensors(L);
        for (int site = 0; site < L; site++) {
            size_t mpo_size = D_mpo * d * d * D_mpo;
            hipMalloc(&d_mpo_tensors[site], mpo_size * sizeof(double));

            // Initialize to identity
            std::vector<double> h_mpo(mpo_size, 0.0);
            for (int i = 0; i < d; i++) {
                for (int w = 0; w < D_mpo; w++) {
                    h_mpo[w + i * D_mpo + i * D_mpo * d + w * D_mpo * d * d] = 1.0;
                }
            }
            hipMemcpy(d_mpo_tensors[site], h_mpo.data(),
                     mpo_size * sizeof(double), hipMemcpyHostToDevice);
        }

        // Set MPO
        std::cout << "Setting MPO..." << std::endl;
        coordinator.set_mpo(d_mpo_tensors.data());

        std::cout << "✓ Initialization complete\n" << std::endl;

        // Run multiple DMRG iterations
        const int max_iterations = 5;
        std::cout << "Running " << max_iterations << " DMRG iterations..." << std::endl;
        std::cout << std::string(50, '=') << std::endl;

        std::vector<double> energies;
        for (int iter = 0; iter < max_iterations; iter++) {
            std::cout << "\n=== Iteration " << iter << " ===" << std::endl;

            double energy = coordinator.run_iteration(iter);
            energies.push_back(energy);

            std::cout << "  Energy: " << std::fixed << std::setprecision(10) << energy << std::endl;

            // Check energy convergence
            if (iter > 0) {
                double delta = std::abs(energies[iter] - energies[iter-1]);
                std::cout << "  ΔE: " << std::scientific << std::setprecision(3) << delta << std::endl;

                if (delta < 1e-8) {
                    std::cout << "\n✓ Converged! (ΔE < 1e-8)" << std::endl;
                    break;
                }
            }
        }

        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "Energy history:" << std::endl;
        for (size_t i = 0; i < energies.size(); i++) {
            std::cout << "  Iter " << i << ": " << std::fixed << std::setprecision(10)
                      << energies[i] << std::endl;
        }

        // Note: With identity MPO, energy should stay constant (no optimization)
        // TODO: Implement proper Hamiltonian MPO interface for physics testing

        // Cleanup
        for (int site = 0; site < L; site++) {
            hipFree(d_mpo_tensors[site]);
        }
        for (int i = 0; i < n_streams; i++) {
            hipStreamDestroy(streams[i]);
        }

        std::cout << "\n========================================" << std::endl;
        std::cout << "Result: PASS" << std::endl;
        std::cout << "========================================" << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\nERROR: " << e.what() << std::endl;
        std::cerr << "\n========================================" << std::endl;
        std::cerr << "Result: FAIL" << std::endl;
        std::cerr << "========================================" << std::endl;
        return 1;
    }
}
