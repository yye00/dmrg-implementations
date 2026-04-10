#include "pdmrg_gpu_base.h"
#include <iostream>
#include <map>
#include <cmath>
#include <vector>
#include <cstring>
#include <string>

// ============================================================================
// NAIVE BASELINE — pdmrg-gpu-base (stream-parallel, no optimizations)
// ============================================================================
// Problem-selection flags only:
//   (positional) L  chi_max  n_sweeps
//   --josephson     Run Josephson junction (complex128)
//   --tfim          Run TFIM
// No optimization knobs — all algorithmic defaults are hard-coded in the
// PDMRGGPUBase class constructor and run() method:
//   n_segments  = 2  (hard-coded in tests)
//   n_warmup    = 3  (single-site warmup sweeps)
//   n_local     = 2  (local sweeps per outer iteration)
//   polish      = 10 (two-site full-chain polish sweeps)

using Complex = hipDoubleComplex;

static inline Complex cplx(double re, double im = 0.0) {
    return make_hipDoubleComplex(re, im);
}

// ============================================================================
// Heisenberg MPO (real, D=5)
// ============================================================================
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
// TFIM MPO (real, D=3) — hard-coded J=1.0, h=1.0
// ============================================================================
void build_tfim_mpo(int L, int D_mpo, double J, double h_field,
                    std::vector<double*>& h_mpo_tensors) {
    double Sx[4] = {0, 1, 1, 0};
    double Sz[4] = {1, 0, 0, -1};
    double Id[4] = {1, 0, 0, 1};

    for (int site = 0; site < L; site++) {
        int d = 2;
        int size = D_mpo * d * d * D_mpo;
        h_mpo_tensors[site] = new double[size]();

        if (site == 0) {
            for (int s = 0; s < d; s++)
                for (int sp = 0; sp < d; sp++) {
                    int idx = sp*d + s;
                    h_mpo_tensors[site][0 + s*D_mpo + sp*D_mpo*d + 0*D_mpo*d*d] = Id[idx];
                    h_mpo_tensors[site][0 + s*D_mpo + sp*D_mpo*d + 1*D_mpo*d*d] = -J * Sz[idx];
                    h_mpo_tensors[site][0 + s*D_mpo + sp*D_mpo*d + 2*D_mpo*d*d] = -h_field * Sx[idx];
                }
        } else if (site == L - 1) {
            for (int s = 0; s < d; s++)
                for (int sp = 0; sp < d; sp++) {
                    int idx = sp*d + s;
                    h_mpo_tensors[site][0 + s*D_mpo + sp*D_mpo*d + 2*D_mpo*d*d] = -h_field * Sx[idx];
                    h_mpo_tensors[site][1 + s*D_mpo + sp*D_mpo*d + 2*D_mpo*d*d] = Sz[idx];
                    h_mpo_tensors[site][2 + s*D_mpo + sp*D_mpo*d + 2*D_mpo*d*d] = Id[idx];
                }
        } else {
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
// Josephson Junction MPO (complex128, D=4)
// ============================================================================
void build_josephson_mpo(int L, int d, int D_mpo,
                         double E_J, double E_C, double n_g, int n_max,
                         double phi_ext,
                         std::vector<Complex*>& h_mpo_tensors) {

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
        exp_iphi[(i + 1) * d + i] = cplx(1.0);
        exp_miphi[i * d + (i + 1)] = cplx(1.0);
    }

    double cos_p = cos(phi_ext);
    double sin_p = sin(phi_ext);
    Complex alpha_coup = cplx(-E_J / 2.0 * cos_p, -E_J / 2.0 * sin_p);
    Complex alpha_conj = cplx(-E_J / 2.0 * cos_p,  E_J / 2.0 * sin_p);

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
            set_block(h_mpo_tensors[site], 0, 0, eye_op);
            set_block(h_mpo_tensors[site], 0, 1, alpha_exp_iphi);
            set_block(h_mpo_tensors[site], 0, 2, alpha_exp_miphi);
            set_block(h_mpo_tensors[site], 0, 3, H_onsite);
        } else if (site == L - 1) {
            set_block(h_mpo_tensors[site], 0, 3, H_onsite);
            set_block(h_mpo_tensors[site], 1, 3, exp_miphi);
            set_block(h_mpo_tensors[site], 2, 3, exp_iphi);
            set_block(h_mpo_tensors[site], 3, 3, eye_op);
        } else {
            set_block(h_mpo_tensors[site], 0, 0, eye_op);
            set_block(h_mpo_tensors[site], 0, 1, alpha_exp_iphi);
            set_block(h_mpo_tensors[site], 0, 2, alpha_exp_miphi);
            set_block(h_mpo_tensors[site], 0, 3, H_onsite);
            set_block(h_mpo_tensors[site], 1, 3, exp_miphi);
            set_block(h_mpo_tensors[site], 2, 3, exp_iphi);
            set_block(h_mpo_tensors[site], 3, 3, eye_op);
        }
    }
}

// ============================================================================
// Heisenberg test
// ============================================================================
int test_heisenberg(int L, int chi_max, int n_sweeps) {
    int d = 2;
    int D_mpo = 5;
    const int n_segments = 2;  // hard-coded

    printf("======================================\n");
    printf("Heisenberg PDMRG-GPU-BASE Test (naive stream-parallel)\n");
    printf("======================================\n");
    printf("  L=%d, d=%d, chi_max=%d, D_mpo=%d, sweeps=%d, segments=%d\n",
           L, d, chi_max, D_mpo, n_sweeps, n_segments);
    printf("======================================\n\n");

    std::map<int, double> exact_energies = {
        {4, -1.616025403784},
        {8, -3.374932598688},
    };

    PDMRGGPUBase<double> dmrg(L, d, chi_max, D_mpo, n_segments, 1e-12);
    dmrg.initialize_mps_random();

    std::vector<double*> h_mpo_tensors(L);
    build_heisenberg_mpo(L, D_mpo, h_mpo_tensors);
    dmrg.set_mpo(h_mpo_tensors);

    double energy = dmrg.run(n_sweeps);

    int ret = 0;
    if (exact_energies.count(L) > 0) {
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
// TFIM test — hard-coded J=1.0, h=1.0 (critical point)
// ============================================================================
int test_tfim(int L, int chi_max, int n_sweeps) {
    int d = 2;
    int D_mpo = 3;
    const int n_segments = 2;  // hard-coded
    const double J_tfim = 1.0;
    const double h_field = 1.0;

    printf("======================================\n");
    printf("TFIM PDMRG-GPU-BASE Test (naive stream-parallel)\n");
    printf("======================================\n");
    printf("  L=%d, d=%d, chi_max=%d, D_mpo=%d, sweeps=%d, segments=%d\n",
           L, d, chi_max, D_mpo, n_sweeps, n_segments);
    printf("  J=%.4f, h=%.4f (hard-coded)\n", J_tfim, h_field);
    printf("======================================\n\n");

    PDMRGGPUBase<double> dmrg(L, d, chi_max, D_mpo, n_segments, 1e-12);
    dmrg.initialize_mps_random();

    std::vector<double*> h_mpo_tensors(L);
    build_tfim_mpo(L, D_mpo, J_tfim, h_field, h_mpo_tensors);
    dmrg.set_mpo(h_mpo_tensors);

    double energy = dmrg.run(n_sweeps);
    printf("Energy per site: %.12f\n", energy / L);

    for (auto ptr : h_mpo_tensors) delete[] ptr;
    return 0;
}

// ============================================================================
// Josephson Junction test — hard-coded n_max=2, E_J=1.0, E_C=0.5, phi_ext=pi/4
// ============================================================================
int test_josephson(int L, int chi_max, int n_sweeps) {
    const int n_max = 2;
    const double E_J = 1.0;
    const double E_C = 0.5;
    const double phi_ext = M_PI / 4;
    const int n_segments = 2;  // hard-coded

    int d = 2 * n_max + 1;
    int D_mpo = 4;

    printf("======================================\n");
    printf("Josephson Junction PDMRG-GPU-BASE Test (naive stream-parallel, complex128)\n");
    printf("======================================\n");
    printf("  L=%d, d=%d (n_max=%d), chi_max=%d, D_mpo=%d, sweeps=%d, segments=%d\n",
           L, d, n_max, chi_max, D_mpo, n_sweeps, n_segments);
    printf("  E_J=%.2f, E_C=%.2f, phi_ext=pi/4 (hard-coded)\n", E_J, E_C);
    printf("======================================\n\n");

    PDMRGGPUBase<Complex> dmrg(L, d, chi_max, D_mpo, n_segments, 1e-12);
    dmrg.initialize_mps_random();

    std::vector<Complex*> h_mpo_tensors(L);
    build_josephson_mpo(L, d, D_mpo, E_J, E_C, 0.0, n_max, phi_ext, h_mpo_tensors);
    dmrg.set_mpo(h_mpo_tensors);

    double energy = dmrg.run(n_sweeps);
    printf("Energy per site: %.12f\n", energy / L);

    for (auto ptr : h_mpo_tensors) delete[] ptr;
    return 0;
}

// ============================================================================
// Main — only problem-selection flags, no optimization toggles
// ============================================================================
int main(int argc, char** argv) {
    int L = 8;
    int chi_max = 32;
    int n_sweeps = 30;
    bool run_josephson = false;
    bool run_tfim = false;

    int pos = 0;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--josephson") { run_josephson = true; continue; }
        if (arg == "--tfim")      { run_tfim = true; continue; }
        if (arg == "--nmax" && i+1 < argc) { ++i; continue; }  // accepted+ignored for benchmark-runner compat
        if (arg[0] == '-') continue;
        if (pos == 0) L = std::atoi(argv[i]);
        else if (pos == 1) chi_max = std::atoi(argv[i]);
        else if (pos == 2) n_sweeps = std::atoi(argv[i]);
        pos++;
    }

    try {
        if (run_josephson) return test_josephson(L, chi_max, n_sweeps);
        if (run_tfim)      return test_tfim(L, chi_max, n_sweeps);
        return test_heisenberg(L, chi_max, n_sweeps);
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
