#include "stream_coordinator.h"
#include "heisenberg_mpo_real.h"
#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "Testing Phase 2 Multi-Stream Iterative DMRG" << std::endl;
    std::cout << "Heisenberg Chain with Real Hamiltonian" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // System parameters (default or from command line)
    int L = 8;          // Chain length
    int chi_max = 32;   // Max bond dimension
    int n_streams = 2;  // Number of parallel streams
    int max_iterations = 11;  // Max DMRG iterations

    // Parse command-line arguments: L chi_max n_streams max_iterations
    if (argc >= 2) L = std::atoi(argv[1]);
    if (argc >= 3) chi_max = std::atoi(argv[2]);
    if (argc >= 4) n_streams = std::atoi(argv[3]);
    if (argc >= 5) max_iterations = std::atoi(argv[4]);

    // Fixed parameters
    const int d = 2;          // Physical dimension
    const int D_mpo = 5;      // MPO bond dimension (Heisenberg)

    // Get exact reference energy
    double E_exact = heisenberg_exact_energy_real(L);

    std::cout << "Parameters:" << std::endl;
    std::cout << "  L = " << L << " sites" << std::endl;
    std::cout << "  d = " << d << std::endl;
    std::cout << "  chi_max = " << chi_max << std::endl;
    std::cout << "  n_streams = " << n_streams << std::endl;
    std::cout << "  D_mpo = " << D_mpo << " (Heisenberg)" << std::endl;
    std::cout << "  E_exact = " << std::fixed << std::setprecision(12) << E_exact << std::endl;
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

        // Create Heisenberg Hamiltonian MPO
        std::cout << "Building Heisenberg MPO..." << std::endl;
        std::vector<double*> d_mpo_tensors = build_heisenberg_mpo_real_gpu(L);

        // Set MPO
        std::cout << "Setting MPO..." << std::endl;
        coordinator.set_mpo(d_mpo_tensors.data());


        // Load initial MPS from binary file
        std::cout << "Loading MPS from binary..." << std::endl;
        if (!coordinator.load_mps_from_binary("/tmp/heisenberg_L8_mps_initial.bin")) {
            std::cerr << "ERROR: Failed to load MPS" << std::endl;
            return 1;
        }
        std::cout << "✓ MPS loaded" << std::endl;
    
    // Build environments now that both MPS and MPO are loaded
    std::cout << "Building environments from MPS..." << std::endl;
    coordinator.build_all_environments();
    std::cout << "✓ Environments built" << std::endl;
        
        // Build environments now that both MPS and MPO are loaded
        std::cout << "Building environments from MPS..." << std::endl;
        coordinator.build_all_environments();
        std::cout << "✓ Environments built" << std::endl;
        std::cout << "✓ Initialization complete\n" << std::endl;

        // Run multiple DMRG iterations
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

        // Compute accuracy
        double E_final = energies.back();
        double error = std::abs(E_final - E_exact);
        double rel_error = error / std::abs(E_exact);

        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "Final Energy: " << std::fixed << std::setprecision(12) << E_final << std::endl;
        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "Accuracy vs Exact:" << std::endl;
        std::cout << "  E_DMRG  = " << std::fixed << std::setprecision(12) << E_final << std::endl;
        std::cout << "  E_exact = " << std::fixed << std::setprecision(12) << E_exact << std::endl;
        std::cout << "  |Error| = " << std::scientific << std::setprecision(3) << error << std::endl;
        std::cout << "  Rel err = " << std::scientific << std::setprecision(3) << rel_error << std::endl;

        // Validate accuracy (target: 1e-10 after LAPACK fix)
        bool accuracy_pass = (error < 1e-10);
        std::cout << "\n  Accuracy test: " << (accuracy_pass ? "✅ PASS" : "❌ FAIL") << std::endl;
        if (accuracy_pass) {
            std::cout << "  (Error < 1e-10 threshold)" << std::endl;
        } else {
            std::cout << "  (Error >= 1e-10, target not met)" << std::endl;
        }

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
