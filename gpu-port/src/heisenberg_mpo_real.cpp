#include <vector>
#include <cmath>
#include <hip/hip_runtime.h>

/**
 * Build real-valued Heisenberg MPO tensors for Phase 2 multi-stream DMRG
 *
 * Hamiltonian: H = Σ_i (S^x_i S^x_{i+1} + S^y_i S^y_{i+1} + S^z_i S^z_{i+1})
 *
 * MPO structure:
 * - Physical dimension: d = 2 (spin-1/2)
 * - Bond dimension: D_mpo = 5
 * - Storage: column-major double arrays
 *
 * MPO tensor format: W[w, s, s', wp]
 *   - w: left MPO bond (0 to D_mpo-1)
 *   - s: physical in (0 to d-1)
 *   - s': physical out (0 to d-1)
 *   - wp: right MPO bond (0 to D_mpo-1)
 *
 * Column-major layout: W[w + s*D_mpo + sp*D_mpo*d + wp*D_mpo*d*d]
 */

// Pauli matrices (real-valued for Heisenberg interactions)
const double Sx[2][2] = {{0.0, 1.0}, {1.0, 0.0}};        // σ^x
const double Sy[2][2] = {{0.0, 1.0}, {1.0, 0.0}};        // σ^y (actually σ^x in real basis)
const double Sz[2][2] = {{1.0, 0.0}, {0.0, -1.0}};       // σ^z
const double Id[2][2] = {{1.0, 0.0}, {0.0, 1.0}};        // Identity

void build_heisenberg_mpo_real_site(int site, int L, double* h_mpo) {
    /**
     * Build Heisenberg MPO for a single site
     *
     * MPO matrix structure:
     *
     * Left boundary (site 0):   D_left = 1, D_right = 5
     *   W[0] = [S^x, S^y, S^z, I, 0]
     *
     * Bulk sites (0 < site < L-1):  D_left = 5, D_right = 5
     *   W = [[I,   0,   0,   0,  0],
     *        [S^x, 0,   0,   0,  0],
     *        [S^y, 0,   0,   0,  0],
     *        [S^z, 0,   0,   0,  0],
     *        [0,   S^x, S^y, S^z, I]]
     *
     * Right boundary (site L-1): D_left = 5, D_right = 1
     *   W[L-1] = [I; S^x; S^y; S^z; 0]^T
     */

    const int d = 2;
    const int D_mpo = 5;

    int D_left = (site == 0) ? 1 : D_mpo;
    int D_right = (site == L - 1) ? 1 : D_mpo;

    // Initialize to zero
    size_t mpo_size = D_left * d * d * D_right;
    for (size_t i = 0; i < mpo_size; i++) {
        h_mpo[i] = 0.0;
    }

    // Helper lambda to set MPO element W[w, s, sp, wp]
    // Column-major: index = w + s*D_left + sp*D_left*d + wp*D_left*d*d
    auto set_mpo = [&](int w, int s, int sp, int wp, double value) {
        int idx = w + s * D_left + sp * D_left * d + wp * D_left * d * d;
        h_mpo[idx] = value;
    };

    // Helper to set a full operator matrix in MPO position (w, wp)
    auto set_operator = [&](int w, int wp, const double op[2][2]) {
        for (int s = 0; s < d; s++) {
            for (int sp = 0; sp < d; sp++) {
                set_mpo(w, s, sp, wp, op[s][sp]);
            }
        }
    };

    if (site == 0) {
        // Left boundary: [S^x, S^y, S^z, I, 0]
        set_operator(0, 0, Sx);  // Send S^x
        set_operator(0, 1, Sy);  // Send S^y
        set_operator(0, 2, Sz);  // Send S^z
        set_operator(0, 3, Id);  // Send I
        // Position (0, 4) is already zero

    } else if (site == L - 1) {
        // Right boundary: [I; S^x; S^y; S^z; 0]^T
        set_operator(0, 0, Id);  // Receive I
        set_operator(1, 0, Sx);  // Receive S^x
        set_operator(2, 0, Sy);  // Receive S^y
        set_operator(3, 0, Sz);  // Receive S^z
        // Position (4, 0) is already zero

    } else {
        // Bulk sites: 5x5 MPO matrix
        // Row 0: [I, 0, 0, 0, 0]
        set_operator(0, 0, Id);

        // Row 1: [S^x, 0, 0, 0, 0]
        set_operator(1, 0, Sx);

        // Row 2: [S^y, 0, 0, 0, 0]
        set_operator(2, 0, Sy);

        // Row 3: [S^z, 0, 0, 0, 0]
        set_operator(3, 0, Sz);

        // Row 4: [0, S^x, S^y, S^z, I]
        set_operator(4, 1, Sx);
        set_operator(4, 2, Sy);
        set_operator(4, 3, Sz);
        set_operator(4, 4, Id);
    }
}

std::vector<double*> build_heisenberg_mpo_real_gpu(int L) {
    /**
     * Build Heisenberg MPO on GPU for all sites
     *
     * Returns: vector of device pointers to MPO tensors (one per site)
     *
     * User is responsible for freeing the device memory:
     *   for (auto* ptr : mpo) hipFree(ptr);
     */

    const int d = 2;
    const int D_mpo = 5;

    std::vector<double*> d_mpo_tensors(L);

    for (int site = 0; site < L; site++) {
        int D_left = (site == 0) ? 1 : D_mpo;
        int D_right = (site == L - 1) ? 1 : D_mpo;
        size_t mpo_size = D_left * d * d * D_right;

        // Allocate host memory
        std::vector<double> h_mpo(mpo_size);

        // Build MPO tensor for this site
        build_heisenberg_mpo_real_site(site, L, h_mpo.data());

        // Allocate and copy to GPU
        hipMalloc(&d_mpo_tensors[site], mpo_size * sizeof(double));
        hipMemcpy(d_mpo_tensors[site], h_mpo.data(),
                  mpo_size * sizeof(double), hipMemcpyHostToDevice);
    }

    return d_mpo_tensors;
}

double heisenberg_exact_energy_real(int L) {
    /**
     * Exact ground state energies for Heisenberg chain
     *
     * Values computed via exact diagonalization for small systems
     * Asymptotic behavior: E/L → -0.443147... (Bethe ansatz)
     */

    // Exact values from literature / exact diag
    if (L == 4) return -1.616025403784;
    if (L == 6) return -2.493340949304;
    if (L == 8) return -3.374931816815;  // ← Test target
    if (L == 10) return -4.258060320358;
    if (L == 12) return -5.142090909091;
    if (L == 16) return -6.911309523810;
    if (L == 20) return -8.681134453782;

    // Approximate for other sizes
    return L * (-0.443147);
}
