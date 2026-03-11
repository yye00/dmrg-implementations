#include "dmrg_gpu.h"
#include <iostream>
#include <map>
#include <cmath>
#include <vector>

// Heisenberg MPO construction - UPPER TRIANGULAR form
//
// MPO tensor layout: W[w, s, sp, wp] stored as w + s*D + sp*D*d + wp*D*d*d
// Physical index convention: W[w, s, sp, wp] = <sp|O_{w,wp}|s>
//
// MPO transfer matrix structure (D=5, upper triangular):
//   Row 0: [I, S+, S-, Sz, 0]      <- open channel: injects operators
//   Row 1: [0,  0,  0,  0, 0.5*S-] <- receives S+ from row 0, emits 0.5*S-
//   Row 2: [0,  0,  0,  0, 0.5*S+] <- receives S- from row 0, emits 0.5*S+
//   Row 3: [0,  0,  0,  0, Sz    ] <- receives Sz from row 0, emits Sz
//   Row 4: [0,  0,  0,  0, I     ] <- closed channel: accumulates energy
//
// Nearest-neighbor interactions via paths:
//   0 -> 1 -> 4: S+_i * 0.5*S-_j
//   0 -> 2 -> 4: S-_i * 0.5*S+_j
//   0 -> 3 -> 4: Sz_i * Sz_j
//   Total: 0.5*(S+S- + S-S+) + SzSz = Heisenberg
//
void build_heisenberg_mpo(int L, int D_mpo, std::vector<double*>& h_mpo_tensors) {
    // Operator arrays stored as O[sp*2+s] = <sp|O|s>
    double Sp[4] = {0, 1, 0, 0};  // S+ = |0><1|
    double Sm[4] = {0, 0, 1, 0};  // S- = |1><0|
    double Sz[4] = {0.5, 0, 0, -0.5};
    double Id[4] = {1, 0, 0, 1};

    for (int site = 0; site < L; site++) {
        int size = D_mpo * 2 * 2 * D_mpo;
        h_mpo_tensors[site] = new double[size]();

        if (site == 0) {
            // Left boundary: only row 0
            for (int s = 0; s < 2; s++) {
                for (int sp = 0; sp < 2; sp++) {
                    int idx = sp*2 + s;
                    h_mpo_tensors[site][0 + s*D_mpo + sp*D_mpo*2 + 0*D_mpo*2*2] = Id[idx];
                    h_mpo_tensors[site][0 + s*D_mpo + sp*D_mpo*2 + 1*D_mpo*2*2] = Sp[idx];
                    h_mpo_tensors[site][0 + s*D_mpo + sp*D_mpo*2 + 2*D_mpo*2*2] = Sm[idx];
                    h_mpo_tensors[site][0 + s*D_mpo + sp*D_mpo*2 + 3*D_mpo*2*2] = Sz[idx];
                }
            }
        } else if (site == L - 1) {
            // Right boundary: only column D-1
            for (int s = 0; s < 2; s++) {
                for (int sp = 0; sp < 2; sp++) {
                    int idx = sp*2 + s;
                    h_mpo_tensors[site][1 + s*D_mpo + sp*D_mpo*2 + 4*D_mpo*2*2] = 0.5*Sm[idx];
                    h_mpo_tensors[site][2 + s*D_mpo + sp*D_mpo*2 + 4*D_mpo*2*2] = 0.5*Sp[idx];
                    h_mpo_tensors[site][3 + s*D_mpo + sp*D_mpo*2 + 4*D_mpo*2*2] = Sz[idx];
                    h_mpo_tensors[site][4 + s*D_mpo + sp*D_mpo*2 + 4*D_mpo*2*2] = Id[idx];
                }
            }
        } else {
            // Bulk: upper triangular
            for (int s = 0; s < 2; s++) {
                for (int sp = 0; sp < 2; sp++) {
                    int idx = sp*2 + s;
                    // Row 0: identity and operator injection
                    h_mpo_tensors[site][0 + s*D_mpo + sp*D_mpo*2 + 0*D_mpo*2*2] = Id[idx];
                    h_mpo_tensors[site][0 + s*D_mpo + sp*D_mpo*2 + 1*D_mpo*2*2] = Sp[idx];
                    h_mpo_tensors[site][0 + s*D_mpo + sp*D_mpo*2 + 2*D_mpo*2*2] = Sm[idx];
                    h_mpo_tensors[site][0 + s*D_mpo + sp*D_mpo*2 + 3*D_mpo*2*2] = Sz[idx];
                    // Rows 1-3: operator completion (column D-1 only)
                    h_mpo_tensors[site][1 + s*D_mpo + sp*D_mpo*2 + 4*D_mpo*2*2] = 0.5*Sm[idx];
                    h_mpo_tensors[site][2 + s*D_mpo + sp*D_mpo*2 + 4*D_mpo*2*2] = 0.5*Sp[idx];
                    h_mpo_tensors[site][3 + s*D_mpo + sp*D_mpo*2 + 4*D_mpo*2*2] = Sz[idx];
                    // Row 4: identity propagation
                    h_mpo_tensors[site][4 + s*D_mpo + sp*D_mpo*2 + 4*D_mpo*2*2] = Id[idx];
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    int L = 8;
    int d = 2;
    int chi_max = 32;
    int D_mpo = 5;
    int n_sweeps = 30;

    if (argc > 1) L = std::atoi(argv[1]);
    if (argc > 2) chi_max = std::atoi(argv[2]);
    if (argc > 3) n_sweeps = std::atoi(argv[3]);

    printf("======================================\n");
    printf("Reference DMRG-GPU Test\n");
    printf("======================================\n");
    printf("Parameters:\n");
    printf("  L = %d (chain length)\n", L);
    printf("  d = %d (physical dim)\n", d);
    printf("  chi_max = %d (max bond dim)\n", chi_max);
    printf("  D_mpo = %d (MPO bond dim)\n", D_mpo);
    printf("  n_sweeps = %d\n", n_sweeps);
    printf("======================================\n\n");

    std::map<int, double> exact_energies = {
        {4, -1.616025403784},
        {8, -3.374932598688},
        {16, -7.0819438}
    };

    try {
        DMRGGPU dmrg(L, d, chi_max, D_mpo, 1e-12);

        printf("Initializing MPS (random state)...\n");
        dmrg.initialize_mps_random();

        printf("Building Heisenberg MPO (upper triangular)...\n");
        std::vector<double*> h_mpo_tensors(L);
        build_heisenberg_mpo(L, D_mpo, h_mpo_tensors);
        dmrg.set_mpo(h_mpo_tensors);

        double energy = dmrg.run(n_sweeps);

        printf("\n======================================\n");
        printf("RESULTS:\n");
        printf("======================================\n");
        printf("Final energy: %.12f\n", energy);

        if (exact_energies.count(L) > 0) {
            double exact = exact_energies[L];
            double error = std::abs(energy - exact);
            double rel_error = error / std::abs(exact) * 100.0;

            printf("Exact energy: %.12f\n", exact);
            printf("Absolute error: %.2e\n", error);
            printf("Relative error: %.6f%%\n", rel_error);

            if (error < 1e-10) {
                printf("\nSUCCESS: Accuracy < 1e-10\n");
            } else if (error < 1e-8) {
                printf("\nACCEPTABLE: Accuracy < 1e-8\n");
            } else {
                printf("\nFAILED: Error too large\n");
            }
        }
        printf("======================================\n");

        for (auto ptr : h_mpo_tensors) delete[] ptr;

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
