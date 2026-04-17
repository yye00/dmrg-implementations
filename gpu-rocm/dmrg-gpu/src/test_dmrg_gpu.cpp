#include "dmrg_gpu.h"
#include "challenge_mpos.h"
#include <iostream>
#include <map>
#include <cmath>
#include <vector>
#include <cstring>
#include <string>

using Complex = hipDoubleComplex;

static inline Complex cplx(double re, double im = 0.0) {
    return make_hipDoubleComplex(re, im);
}

// ============================================================================
// Heisenberg MPO (real, D=5, upper triangular)
// ============================================================================
// W[w, s, sp, wp] stored as w + s*D + sp*D*d + wp*D*d*d
//
// Transfer matrix (D=5):
//   Row 0: [I, S+, S-, Sz, 0]
//   Row 1: [0,  0,  0,  0, 0.5*S-]
//   Row 2: [0,  0,  0,  0, 0.5*S+]
//   Row 3: [0,  0,  0,  0, Sz    ]
//   Row 4: [0,  0,  0,  0, I     ]
//
void build_heisenberg_mpo(int L, int D_mpo, std::vector<double*>& h_mpo_tensors) {
    double Sp[4] = {0, 1, 0, 0};
    double Sm[4] = {0, 0, 1, 0};
    double Sz[4] = {0.5, 0, 0, -0.5};
    double Id[4] = {1, 0, 0, 1};

    for (int site = 0; site < L; site++) {
        int size = D_mpo * 2 * 2 * D_mpo;
        h_mpo_tensors[site] = new double[size]();

        if (site == 0) {
            for (int s = 0; s < 2; s++)
                for (int sp = 0; sp < 2; sp++) {
                    int idx = sp*2 + s;
                    h_mpo_tensors[site][0 + s*D_mpo + sp*D_mpo*2 + 0*D_mpo*2*2] = Id[idx];
                    h_mpo_tensors[site][0 + s*D_mpo + sp*D_mpo*2 + 1*D_mpo*2*2] = Sp[idx];
                    h_mpo_tensors[site][0 + s*D_mpo + sp*D_mpo*2 + 2*D_mpo*2*2] = Sm[idx];
                    h_mpo_tensors[site][0 + s*D_mpo + sp*D_mpo*2 + 3*D_mpo*2*2] = Sz[idx];
                }
        } else if (site == L - 1) {
            for (int s = 0; s < 2; s++)
                for (int sp = 0; sp < 2; sp++) {
                    int idx = sp*2 + s;
                    h_mpo_tensors[site][1 + s*D_mpo + sp*D_mpo*2 + 4*D_mpo*2*2] = 0.5*Sm[idx];
                    h_mpo_tensors[site][2 + s*D_mpo + sp*D_mpo*2 + 4*D_mpo*2*2] = 0.5*Sp[idx];
                    h_mpo_tensors[site][3 + s*D_mpo + sp*D_mpo*2 + 4*D_mpo*2*2] = Sz[idx];
                    h_mpo_tensors[site][4 + s*D_mpo + sp*D_mpo*2 + 4*D_mpo*2*2] = Id[idx];
                }
        } else {
            for (int s = 0; s < 2; s++)
                for (int sp = 0; sp < 2; sp++) {
                    int idx = sp*2 + s;
                    h_mpo_tensors[site][0 + s*D_mpo + sp*D_mpo*2 + 0*D_mpo*2*2] = Id[idx];
                    h_mpo_tensors[site][0 + s*D_mpo + sp*D_mpo*2 + 1*D_mpo*2*2] = Sp[idx];
                    h_mpo_tensors[site][0 + s*D_mpo + sp*D_mpo*2 + 2*D_mpo*2*2] = Sm[idx];
                    h_mpo_tensors[site][0 + s*D_mpo + sp*D_mpo*2 + 3*D_mpo*2*2] = Sz[idx];
                    h_mpo_tensors[site][1 + s*D_mpo + sp*D_mpo*2 + 4*D_mpo*2*2] = 0.5*Sm[idx];
                    h_mpo_tensors[site][2 + s*D_mpo + sp*D_mpo*2 + 4*D_mpo*2*2] = 0.5*Sp[idx];
                    h_mpo_tensors[site][3 + s*D_mpo + sp*D_mpo*2 + 4*D_mpo*2*2] = Sz[idx];
                    h_mpo_tensors[site][4 + s*D_mpo + sp*D_mpo*2 + 4*D_mpo*2*2] = Id[idx];
                }
        }
    }
}

// ============================================================================
// Transverse-Field Ising Model MPO (real, D=3, upper triangular)
// ============================================================================
// H = -J sum_i sigma^z_i sigma^z_{i+1} - h sum_i sigma^x_i
//
// Transfer matrix (D=3):
//   Row 0: [I,    -J*Sz,   -h*Sx]
//   Row 1: [0,     0,       Sz   ]
//   Row 2: [0,     0,       I    ]
//
void build_tfim_mpo(int L, int D_mpo, double J, double h_field,
                    std::vector<double*>& h_mpo_tensors) {
    double Sx[4] = {0, 1, 1, 0};   // sigma_x (Pauli X)
    double Sz[4] = {1, 0, 0, -1};  // sigma_z (Pauli Z)
    double Id[4] = {1, 0, 0, 1};

    for (int site = 0; site < L; site++) {
        int d = 2;
        int size = D_mpo * d * d * D_mpo;
        h_mpo_tensors[site] = new double[size]();

        if (site == 0) {
            // Left boundary: only row 0 (D_L=0)
            for (int s = 0; s < d; s++)
                for (int sp = 0; sp < d; sp++) {
                    int idx = sp*d + s;
                    h_mpo_tensors[site][0 + s*D_mpo + sp*D_mpo*d + 0*D_mpo*d*d] = Id[idx];
                    h_mpo_tensors[site][0 + s*D_mpo + sp*D_mpo*d + 1*D_mpo*d*d] = -J * Sz[idx];
                    h_mpo_tensors[site][0 + s*D_mpo + sp*D_mpo*d + 2*D_mpo*d*d] = -h_field * Sx[idx];
                }
        } else if (site == L - 1) {
            // Right boundary: only column 2 (D_R=2)
            for (int s = 0; s < d; s++)
                for (int sp = 0; sp < d; sp++) {
                    int idx = sp*d + s;
                    h_mpo_tensors[site][0 + s*D_mpo + sp*D_mpo*d + 2*D_mpo*d*d] = -h_field * Sx[idx];
                    h_mpo_tensors[site][1 + s*D_mpo + sp*D_mpo*d + 2*D_mpo*d*d] = Sz[idx];
                    h_mpo_tensors[site][2 + s*D_mpo + sp*D_mpo*d + 2*D_mpo*d*d] = Id[idx];
                }
        } else {
            // Bulk: full transfer matrix
            for (int s = 0; s < d; s++)
                for (int sp = 0; sp < d; sp++) {
                    int idx = sp*d + s;
                    h_mpo_tensors[site][0 + s*D_mpo + sp*D_mpo*d + 0*D_mpo*d*d] = Id[idx];
                    h_mpo_tensors[site][0 + s*D_mpo + sp*D_mpo*d + 1*D_mpo*d*d] = -J * Sz[idx];
                    h_mpo_tensors[site][0 + s*D_mpo + sp*D_mpo*d + 2*D_mpo*d*d] = -h_field * Sx[idx];
                    h_mpo_tensors[site][1 + s*D_mpo + sp*D_mpo*d + 2*D_mpo*d*d] = Sz[idx];
                    h_mpo_tensors[site][2 + s*D_mpo + sp*D_mpo*d + 2*D_mpo*d*d] = Id[idx];
                }
        }
    }
}

// ============================================================================
// Josephson Junction MPO (complex128, D=4, UPPER triangular)
// ============================================================================
// H = E_C sum_i (n_i - n_g)^2 - (E_J/2) sum_<ij> [e^{i*phi_ext} phi+_i phi-_j + h.c.]
//
// Physical dim: d = 2*n_max + 1 (charge basis -n_max..+n_max)
// MPO bond dim: D_mpo = 4 (fixed for all sites, zero-padded boundaries)
//
// Operators:
//   exp_iphi:  e^{iphi}|n> = |n+1>, raises charge
//   exp_miphi: e^{-iphi}|n> = |n-1>, lowers charge
//   H_onsite:  diagonal, H_onsite[i,i] = E_C*(i-n_max)^2 - n_g*(i-n_max)
//
// Transfer matrix (D=4, upper triangular):
//   Row 0: [I,  a*phi+,  a'*phi-,  H_onsite]  <- open channel
//   Row 1: [0,  0,       0,        phi-    ]   <- receives phi+, emits phi-
//   Row 2: [0,  0,       0,        phi+    ]   <- receives phi-, emits phi+
//   Row 3: [0,  0,       0,        I       ]   <- closed channel
//
// where a = -E_J/2 * e^{i*phi_ext}, a' = conj(a)
//
// Coupling paths:
//   0 -> 1 -> 3: a*phi+_i * phi-_j  (= -EJ/2 * e^{iphi} * phi+_i * phi-_j)
//   0 -> 2 -> 3: a'*phi-_i * phi+_j (= -EJ/2 * e^{-iphi} * phi-_i * phi+_j)
//   On-site: (0,3) = H_onsite at each site
//
// Boundary convention (matches Heisenberg):
//   L[0, 0, 0] = 1 (selects row 0)
//   R[0, D-1, 0] = 1 (selects column D-1 = 3)
//
void build_josephson_mpo(int L, int d, int D_mpo,
                         double E_J, double E_C, double n_g, int n_max,
                         double phi_ext,
                         std::vector<Complex*>& h_mpo_tensors) {

    // Build operator matrices (d x d), stored as O[sp*d + s]
    std::vector<Complex> eye_op(d * d, cplx(0));
    std::vector<Complex> exp_iphi(d * d, cplx(0));
    std::vector<Complex> exp_miphi(d * d, cplx(0));
    std::vector<Complex> H_onsite(d * d, cplx(0));

    for (int i = 0; i < d; i++) {
        eye_op[i * d + i] = cplx(1.0);
        double charge = (double)(i - n_max);
        H_onsite[i * d + i] = cplx(E_C * charge * charge - n_g * charge);
    }
    for (int i = 0; i < d - 1; i++) {
        exp_iphi[(i + 1) * d + i] = cplx(1.0);   // |n+1><n|
        exp_miphi[i * d + (i + 1)] = cplx(1.0);   // |n><n+1| = |n-1><n|
    }

    // Coupling: alpha = -E_J/2 * e^{i*phi_ext}
    double cos_p = cos(phi_ext);
    double sin_p = sin(phi_ext);
    Complex alpha_coup = cplx(-E_J / 2.0 * cos_p, -E_J / 2.0 * sin_p);
    Complex alpha_conj = cplx(-E_J / 2.0 * cos_p,  E_J / 2.0 * sin_p);

    // Helper: scale operator by complex scalar
    auto scale_op = [&](Complex c, const std::vector<Complex>& op) {
        std::vector<Complex> result(d * d);
        for (int i = 0; i < d * d; i++) {
            result[i] = make_hipDoubleComplex(
                hipCreal(c) * hipCreal(op[i]) - hipCimag(c) * hipCimag(op[i]),
                hipCreal(c) * hipCimag(op[i]) + hipCimag(c) * hipCreal(op[i]));
        }
        return result;
    };

    auto alpha_exp_iphi = scale_op(alpha_coup, exp_iphi);
    auto alpha_exp_miphi = scale_op(alpha_conj, exp_miphi);

    // Helper: set W[w, s, sp, wp] = op[sp*d + s]
    // Memory layout: w + s*D_mpo + sp*D_mpo*d + wp*D_mpo*d*d
    auto set_block = [&](Complex* W, int w, int wp,
                         const std::vector<Complex>& op) {
        for (int s = 0; s < d; s++)
            for (int sp = 0; sp < d; sp++)
                W[w + s * D_mpo + sp * D_mpo * d + wp * D_mpo * d * d] = op[sp * d + s];
    };

    for (int site = 0; site < L; site++) {
        int size = D_mpo * d * d * D_mpo;
        h_mpo_tensors[site] = new Complex[size]();

        if (site == 0) {
            // Left boundary: only row 0 non-zero
            set_block(h_mpo_tensors[site], 0, 0, eye_op);             // (0,0): I
            set_block(h_mpo_tensors[site], 0, 1, alpha_exp_iphi);     // (0,1): a*phi+
            set_block(h_mpo_tensors[site], 0, 2, alpha_exp_miphi);    // (0,2): a'*phi-
            set_block(h_mpo_tensors[site], 0, 3, H_onsite);           // (0,3): H_onsite
        } else if (site == L - 1) {
            // Right boundary: only column D-1=3 non-zero
            set_block(h_mpo_tensors[site], 0, 3, H_onsite);           // (0,3): H_onsite
            set_block(h_mpo_tensors[site], 1, 3, exp_miphi);          // (1,3): phi-
            set_block(h_mpo_tensors[site], 2, 3, exp_iphi);           // (2,3): phi+
            set_block(h_mpo_tensors[site], 3, 3, eye_op);             // (3,3): I
        } else {
            // Bulk: upper triangular
            set_block(h_mpo_tensors[site], 0, 0, eye_op);             // (0,0): I
            set_block(h_mpo_tensors[site], 0, 1, alpha_exp_iphi);     // (0,1): a*phi+
            set_block(h_mpo_tensors[site], 0, 2, alpha_exp_miphi);    // (0,2): a'*phi-
            set_block(h_mpo_tensors[site], 0, 3, H_onsite);           // (0,3): H_onsite
            set_block(h_mpo_tensors[site], 1, 3, exp_miphi);          // (1,3): phi-
            set_block(h_mpo_tensors[site], 2, 3, exp_iphi);           // (2,3): phi+
            set_block(h_mpo_tensors[site], 3, 3, eye_op);             // (3,3): I
        }
    }
}

// ============================================================================
// Heisenberg test (real)
// ============================================================================
int test_heisenberg(int L, int chi_max, int n_sweeps, bool quiet) {
    int d = 2;
    int D_mpo = 5;

    if (!quiet) {
        printf("======================================\n");
        printf("Heisenberg DMRG-GPU Test (float64)\n");
        printf("======================================\n");
        printf("  L=%d, d=%d, chi_max=%d, D_mpo=%d, sweeps=%d\n", L, d, chi_max, D_mpo, n_sweeps);
        printf("======================================\n\n");
    }

    std::map<int, double> exact_energies = {
        {4, -1.616025403784},
        {8, -3.374932598688},
    };

    DMRGGPU<double> dmrg(L, d, chi_max, D_mpo, 1e-12);
    dmrg.set_quiet(quiet);
    dmrg.initialize_mps_random();

    std::vector<double*> h_mpo_tensors(L);
    build_heisenberg_mpo(L, D_mpo, h_mpo_tensors);
    dmrg.set_mpo(h_mpo_tensors);

    double energy = dmrg.run(n_sweeps);

    // Final energy already printed by run()

    int ret = 0;
    if (!quiet && exact_energies.count(L) > 0) {
        double exact = exact_energies[L];
        double error = std::abs(energy - exact);
        printf("Exact energy: %.12f\n", exact);
        printf("Absolute error: %.2e\n", error);
        if (error < 1e-10) printf("PASS\n");
        else { printf("FAIL\n"); ret = 1; }
    }

    for (auto ptr : h_mpo_tensors) delete[] ptr;
    return ret;
}

// ============================================================================
// TFIM test (real)
// ============================================================================
int test_tfim(int L, int chi_max, int n_sweeps, double J, double h_field, bool quiet) {
    int d = 2;
    int D_mpo = 3;

    if (!quiet) {
        printf("======================================\n");
        printf("TFIM DMRG-GPU Test (float64)\n");
        printf("======================================\n");
        printf("  L=%d, d=%d, chi_max=%d, D_mpo=%d, sweeps=%d\n", L, d, chi_max, D_mpo, n_sweeps);
        printf("  J=%.4f, h=%.4f\n", J, h_field);
        printf("======================================\n\n");
    }

    DMRGGPU<double> dmrg(L, d, chi_max, D_mpo, 1e-12);
    dmrg.set_quiet(quiet);
    dmrg.initialize_mps_random();

    std::vector<double*> h_mpo_tensors(L);
    build_tfim_mpo(L, D_mpo, J, h_field, h_mpo_tensors);
    dmrg.set_mpo(h_mpo_tensors);

    double energy = dmrg.run(n_sweeps);

    // Final energy already printed by run()
    if (!quiet) printf("Energy per site: %.12f\n", energy / L);

    for (auto ptr : h_mpo_tensors) delete[] ptr;
    return 0;
}

// ============================================================================
// Josephson Junction test (complex128)
// ============================================================================
int test_josephson(int L, int chi_max, int n_sweeps,
                   int n_max, double E_J, double E_C, double phi_ext, bool quiet) {
    int d = 2 * n_max + 1;
    int D_mpo = 4;

    if (!quiet) {
        printf("======================================\n");
        printf("Josephson Junction DMRG-GPU Test (complex128)\n");
        printf("======================================\n");
        printf("  L=%d, d=%d (n_max=%d), chi_max=%d, D_mpo=%d, sweeps=%d\n",
               L, d, n_max, chi_max, D_mpo, n_sweeps);
        printf("  E_J=%.2f, E_C=%.2f, phi_ext=pi/%.1f\n", E_J, E_C, M_PI / phi_ext);
        printf("======================================\n\n");
    }

    // Exact energies from direct diagonalization (n_max=1, E_J=1.0, E_C=0.5, phi_ext=pi/4)
    std::map<int, double> exact_energies;
    if (n_max == 1 && std::abs(E_J - 1.0) < 1e-10 && std::abs(E_C - 0.5) < 1e-10
        && std::abs(phi_ext - M_PI/4) < 1e-10) {
        exact_energies = {
            {4, -1.053346829927396},
            {6, -1.748843818181493},
        };
    }

    DMRGGPU<Complex> dmrg(L, d, chi_max, D_mpo, 1e-12);
    dmrg.set_quiet(quiet);
    dmrg.initialize_mps_random();

    std::vector<Complex*> h_mpo_tensors(L);
    build_josephson_mpo(L, d, D_mpo, E_J, E_C, 0.0, n_max, phi_ext, h_mpo_tensors);
    dmrg.set_mpo(h_mpo_tensors);

    double energy = dmrg.run(n_sweeps);

    // Final energy already printed by run()

    int ret = 0;
    if (!quiet && exact_energies.count(L) > 0) {
        double exact = exact_energies[L];
        double error = std::abs(energy - exact);
        printf("Exact energy: %.12f\n", exact);
        printf("Absolute error: %.2e\n", error);
        if (error < 1e-8) printf("PASS\n");
        else { printf("FAIL\n"); ret = 1; }
    }

    for (auto ptr : h_mpo_tensors) delete[] ptr;
    return ret;
}

// ============================================================================
// J1-J2 Heisenberg test (real, frustrated chain)
// ============================================================================
int test_j1j2(int L, int chi_max, int n_sweeps, double J1, double J2, bool quiet) {
    int d = 2;
    int D_mpo = 11;

    if (!quiet) {
        printf("======================================\n");
        printf("J1-J2 Heisenberg DMRG-GPU Test (float64)\n");
        printf("======================================\n");
        printf("  L=%d, d=%d, chi_max=%d, D_mpo=%d, sweeps=%d\n", L, d, chi_max, D_mpo, n_sweeps);
        printf("  J1=%.4f, J2=%.4f\n", J1, J2);
        printf("======================================\n\n");
    }

    std::map<int, double> exact_energies;
    if (std::abs(J1 - 1.0) < 1e-12 && std::abs(J2 - 0.5) < 1e-12) {
        exact_energies = {
            {4, -1.5},
            {6, -2.25},
            {8, -3.0},
        };
    }

    DMRGGPU<double> dmrg(L, d, chi_max, D_mpo, 1e-12);
    dmrg.set_quiet(quiet);
    dmrg.initialize_mps_random();

    std::vector<double*> h_mpo_tensors(L);
    challenge_mpos::build_j1j2_mpo(L, J1, J2, h_mpo_tensors);
    dmrg.set_mpo(h_mpo_tensors);

    double energy = dmrg.run(n_sweeps);

    int ret = 0;
    if (!quiet && exact_energies.count(L) > 0) {
        double exact = exact_energies[L];
        double error = std::abs(energy - exact);
        printf("Exact energy: %.12f\n", exact);
        printf("Absolute error: %.2e\n", error);
        if (error < 1e-8) printf("PASS\n");
        else { printf("FAIL\n"); ret = 1; }
    }

    for (auto ptr : h_mpo_tensors) delete[] ptr;
    return ret;
}

// ============================================================================
// Heisenberg 2-leg ladder test (real, d=4 supersite per rung)
// ============================================================================
int test_ladder(int L_rungs, int chi_max, int n_sweeps,
                double J_leg, double J_rung, bool quiet) {
    int d = 4;
    int D_mpo = 8;

    if (!quiet) {
        printf("======================================\n");
        printf("Heisenberg 2-leg Ladder DMRG-GPU Test (float64)\n");
        printf("======================================\n");
        printf("  L_rungs=%d (2*L_rungs=%d spins), d=%d, chi_max=%d, D_mpo=%d, sweeps=%d\n",
               L_rungs, 2*L_rungs, d, chi_max, D_mpo, n_sweeps);
        printf("  J_leg=%.4f, J_rung=%.4f\n", J_leg, J_rung);
        printf("======================================\n\n");
    }

    std::map<int, double> exact_energies;
    if (std::abs(J_leg - 1.0) < 1e-12 && std::abs(J_rung - 1.0) < 1e-12) {
        exact_energies = {
            {2, -2.0},
            {3, -3.1293852415718166},
            {4, -4.2930664566945656},
        };
    }

    DMRGGPU<double> dmrg(L_rungs, d, chi_max, D_mpo, 1e-12);
    dmrg.set_quiet(quiet);
    dmrg.initialize_mps_random();

    std::vector<double*> h_mpo_tensors(L_rungs);
    challenge_mpos::build_ladder_mpo(L_rungs, J_leg, J_rung, h_mpo_tensors);
    dmrg.set_mpo(h_mpo_tensors);

    double energy = dmrg.run(n_sweeps);

    int ret = 0;
    if (!quiet && exact_energies.count(L_rungs) > 0) {
        double exact = exact_energies[L_rungs];
        double error = std::abs(energy - exact);
        printf("Exact energy: %.12f\n", exact);
        printf("Absolute error: %.2e\n", error);
        if (error < 1e-8) printf("PASS\n");
        else { printf("FAIL\n"); ret = 1; }
    }

    for (auto ptr : h_mpo_tensors) delete[] ptr;
    return ret;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    // Defaults
    int L = 8;
    int chi_max = 32;
    int n_sweeps = 30;
    bool run_josephson = false;
    bool run_tfim = false;
    bool run_j1j2 = false;
    bool run_ladder = false;
    bool quiet = false;
    int n_max = 1;
    double E_J = 1.0, E_C = 0.5, phi_ext = M_PI / 4;
    double J_tfim = 1.0, h_field = 1.0;
    double J1 = 1.0, J2 = 0.5;
    double J_leg = 1.0, J_rung = 1.0;

    // Re-parse positional args more robustly
    {
        int pos = 0;
        for (int i = 1; i < argc; i++) {
            if (std::string(argv[i]) == "--josephson") { run_josephson = true; continue; }
            if (std::string(argv[i]) == "--tfim") { run_tfim = true; continue; }
            if (std::string(argv[i]) == "--j1j2") { run_j1j2 = true; continue; }
            if (std::string(argv[i]) == "--ladder") { run_ladder = true; continue; }
            if (std::string(argv[i]) == "--quiet") { quiet = true; continue; }
            if (std::string(argv[i]) == "--hfield" && i+1 < argc) { h_field = std::atof(argv[++i]); continue; }
            if (std::string(argv[i]) == "--nmax" && i+1 < argc) { n_max = std::atoi(argv[++i]); continue; }
            if (std::string(argv[i]) == "--ej" && i+1 < argc) { E_J = std::atof(argv[++i]); continue; }
            if (std::string(argv[i]) == "--ec" && i+1 < argc) { E_C = std::atof(argv[++i]); continue; }
            if (std::string(argv[i]) == "--phi" && i+1 < argc) { phi_ext = std::atof(argv[++i]); continue; }
            if (std::string(argv[i]) == "--j1" && i+1 < argc) { J1 = std::atof(argv[++i]); continue; }
            if (std::string(argv[i]) == "--j2" && i+1 < argc) { J2 = std::atof(argv[++i]); continue; }
            if (std::string(argv[i]) == "--jleg" && i+1 < argc) { J_leg = std::atof(argv[++i]); continue; }
            if (std::string(argv[i]) == "--jrung" && i+1 < argc) { J_rung = std::atof(argv[++i]); continue; }
            if (argv[i][0] == '-') {
                continue;
            }
            if (pos == 0) L = std::atoi(argv[i]);
            else if (pos == 1) chi_max = std::atoi(argv[i]);
            else if (pos == 2) n_sweeps = std::atoi(argv[i]);
            pos++;
        }
    }

    try {
        if (run_josephson) {
            return test_josephson(L, chi_max, n_sweeps, n_max, E_J, E_C, phi_ext, quiet);
        } else if (run_tfim) {
            return test_tfim(L, chi_max, n_sweeps, J_tfim, h_field, quiet);
        } else if (run_j1j2) {
            return test_j1j2(L, chi_max, n_sweeps, J1, J2, quiet);
        } else if (run_ladder) {
            return test_ladder(L, chi_max, n_sweeps, J_leg, J_rung, quiet);
        } else {
            return test_heisenberg(L, chi_max, n_sweeps, quiet);
        }
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
