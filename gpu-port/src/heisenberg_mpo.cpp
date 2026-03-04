#include "heisenberg_mpo.hpp"
#include <cmath>
#include <iostream>

Tensor4D<Complex> build_heisenberg_mpo(int L) {
    // MPO tensor dimensions: [left_bond, phys_in, phys_out, right_bond]
    // Bond dimension = 3 for Heisenberg
    const int d = 2;  // physical dimension (spin-1/2)
    const int D_mpo = 3;  // MPO bond dimension

    Tensor4D<Complex> mpo(L);

    // Pauli matrices
    Tensor2D<Complex> sx = {{0, 1}, {1, 0}};
    Tensor2D<Complex> sy = {{0, Complex(0,-1)}, {Complex(0,1), 0}};
    Tensor2D<Complex> sz = {{1, 0}, {0, -1}};
    Tensor2D<Complex> eye = {{1, 0}, {0, 1}};

    for (int site = 0; site < L; site++) {
        int left_dim = (site == 0) ? 1 : D_mpo;
        int right_dim = (site == L-1) ? 1 : D_mpo;

        mpo[site].resize(left_dim);
        for (int a = 0; a < left_dim; a++) {
            mpo[site][a].resize(d);
            for (int i = 0; i < d; i++) {
                mpo[site][a][i].resize(d);
                for (int j = 0; j < d; j++) {
                    mpo[site][a][i][j].resize(right_dim, Complex(0.0));
                }
            }
        }

        // Fill MPO tensors (Heisenberg Hamiltonian: H = sum_i (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1}))
        if (site == 0) {
            // Left boundary: [1, d, d, 3]
            for (int i = 0; i < d; i++) {
                for (int j = 0; j < d; j++) {
                    mpo[site][0][i][j][0] = sx[i][j];  // X term
                    mpo[site][0][i][j][1] = sy[i][j];  // Y term
                    mpo[site][0][i][j][2] = sz[i][j];  // Z term
                }
            }
        } else if (site == L-1) {
            // Right boundary: [3, d, d, 1]
            for (int i = 0; i < d; i++) {
                for (int j = 0; j < d; j++) {
                    mpo[site][0][i][j][0] = sx[i][j];  // Receive X
                    mpo[site][1][i][j][0] = sy[i][j];  // Receive Y
                    mpo[site][2][i][j][0] = sz[i][j];  // Receive Z
                }
            }
        } else {
            // Bulk sites: [3, d, d, 3]
            for (int i = 0; i < d; i++) {
                for (int j = 0; j < d; j++) {
                    mpo[site][0][i][j][0] = eye[i][j];  // Identity
                    mpo[site][0][i][j][1] = sx[i][j];   // Create X
                    mpo[site][0][i][j][2] = sy[i][j];   // Create Y
                    mpo[site][1][i][j][0] = sz[i][j];   // Create Z
                    mpo[site][2][i][j][1] = sx[i][j];   // Receive X
                    mpo[site][2][i][j][2] = sy[i][j];   // Receive Y
                }
            }
        }
    }

    return mpo;
}

double heisenberg_exact_energy(int L) {
    // Exact ground state energies for small Heisenberg chains
    // E/L for infinite chain: -0.443147... (Bethe ansatz)
    // Exact values for small L from exact diagonalization

    if (L == 4) return -1.616025;
    if (L == 6) return -2.493341;
    if (L == 8) return -3.374932;
    if (L == 10) return -4.258060;
    if (L == 12) return -5.142091;  // Reference value for L=12
    if (L == 20) return -8.576302;
    if (L == 40) return -17.442682;

    // Approximate for other sizes (using infinite chain value)
    return L * (-0.443147);
}
