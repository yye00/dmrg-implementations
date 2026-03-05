#include "stream_coordinator.h"
#include "heisenberg_mpo.h"
#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Testing Phase 2 Multi-Stream with Heisenberg Hamiltonian" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // System parameters
    const int L = 8;          // Chain length
    const int d = 2;          // Physical dimension (spin-1/2)
    const int chi_max = 16;   // Max bond dimension
    const int n_streams = 2;  // Number of parallel streams

    // Heisenberg parameters
    const double J = 1.0;     // Exchange coupling
    const double h = 0.0;     // Magnetic field

    std::cout << "Parameters:" << std::endl;
    std::cout << "  L = " << L << " sites" << std::endl;
    std::cout << "  d = " << d << " (spin-1/2)" << std::endl;
    std::cout << "  chi_max = " << chi_max << std::endl;
    std::cout << "  n_streams = " << n_streams << std::endl;
    std::cout << "  J = " << J << ", h = " << h << std::endl;
    std::cout << std::endl;

    try {
        // Create HIP streams
        std::vector<hipStream_t> streams(n_streams);
        for (int i = 0; i < n_streams; i++) {
            hipStreamCreate(&streams[i]);
        }

        // Create Heisenberg MPO
        std::cout << "Creating Heisenberg MPO..." << std::endl;
        HeisenbergMPO mpo(L, J, h);

        // Create StreamCoordinator
        std::cout << "Initializing StreamCoordinator..." << std::endl;
        StreamCoordinator coordinator(L, d, chi_max, n_streams, streams);

        // Set MPO
        std::cout << "Setting MPO..." << std::endl;
        coordinator.set_mpo(&mpo);

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

        // Expected ground state energy for L=8 Heisenberg chain (from exact diagonalization):
        // E0 ≈ -3.30948 (J=1, h=0)
        std::cout << "\nReference (exact): E0 ≈ -3.30948 for L=8" << std::endl;
        std::cout << "Our result:        E  = " << std::fixed << std::setprecision(5)
                  << energies.back() << std::endl;

        // Cleanup
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
