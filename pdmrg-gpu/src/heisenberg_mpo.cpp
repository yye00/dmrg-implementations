#include "heisenberg_mpo.hpp"
#include <cmath>
#include <iostream>

Tensor5D<Complex> build_heisenberg_mpo(int L) {
    // MPO tensor dimensions: [site][left_bond][phys_in][phys_out][right_bond]
    // Bond dimension = 5 for Heisenberg nearest-neighbor
    const int d = 2;  // physical dimension (spin-1/2)
    const int D_mpo = 5;  // MPO bond dimension

    Tensor5D<Complex> mpo(L);

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
            // Left boundary: [1, d, d, 5]
            // W[0] = [Sx, Sy, Sz, I, 0]
            for (int i = 0; i < d; i++) {
                for (int j = 0; j < d; j++) {
                    mpo[site][0][i][j][0] = sx[i][j];  // Send X
                    mpo[site][0][i][j][1] = sy[i][j];  // Send Y
                    mpo[site][0][i][j][2] = sz[i][j];  // Send Z
                    mpo[site][0][i][j][3] = eye[i][j]; // Send I
                    mpo[site][0][i][j][4] = Complex(0.0);
                }
            }
        } else if (site == L-1) {
            // Right boundary: [5, d, d, 1]
            // W[L-1] = [I; Sx; Sy; Sz; 0]^T
            for (int i = 0; i < d; i++) {
                for (int j = 0; j < d; j++) {
                    mpo[site][0][i][j][0] = eye[i][j];  // Receive I
                    mpo[site][1][i][j][0] = sx[i][j];   // Receive X
                    mpo[site][2][i][j][0] = sy[i][j];   // Receive Y
                    mpo[site][3][i][j][0] = sz[i][j];   // Receive Z
                    mpo[site][4][i][j][0] = Complex(0.0);
                }
            }
        } else {
            // Bulk sites: [5, d, d, 5]
            // W = [[I,  0,  0,  0,  0 ],
            //      [Sx, 0,  0,  0,  0 ],
            //      [Sy, 0,  0,  0,  0 ],
            //      [Sz, 0,  0,  0,  0 ],
            //      [0,  Sx, Sy, Sz, I ]]
            for (int i = 0; i < d; i++) {
                for (int j = 0; j < d; j++) {
                    mpo[site][0][i][j][0] = eye[i][j];  // (0,0): I
                    mpo[site][1][i][j][0] = sx[i][j];   // (1,0): Sx
                    mpo[site][2][i][j][0] = sy[i][j];   // (2,0): Sy
                    mpo[site][3][i][j][0] = sz[i][j];   // (3,0): Sz
                    mpo[site][4][i][j][1] = sx[i][j];   // (4,1): Sx
                    mpo[site][4][i][j][2] = sy[i][j];   // (4,2): Sy
                    mpo[site][4][i][j][3] = sz[i][j];   // (4,3): Sz
                    mpo[site][4][i][j][4] = eye[i][j];  // (4,4): I
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
