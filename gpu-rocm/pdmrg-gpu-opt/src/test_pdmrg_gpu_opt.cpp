#include "pdmrg_gpu_opt.h"
#include <iostream>
#include <map>
#include <cmath>
#include <vector>
#include <cstring>

using Complex = hipDoubleComplex;

static inline Complex cplx(double re, double im = 0.0) {
    return make_hipDoubleComplex(re, im);
}

// ============================================================================
// Heisenberg MPO (real, D=5, upper triangular)
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
// Josephson Junction MPO (complex128, D=4, upper triangular)
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
// Heisenberg test (real)
// ============================================================================
int test_heisenberg(int L, int chi_max, int n_outer, int n_segments,
                    int n_local, int n_warmup, int n_polish, bool gpu_svd,
                    bool davidson, bool rsvd, bool quiet, bool batched_sweep,
                    bool chebyshev) {
    int d = 2;
    int D_mpo = 5;

    if (!quiet) {
        printf("======================================\n");
        printf("PDMRG-GPU-OPT Heisenberg Test (float64)\n");
        printf("======================================\n");
        printf("  L=%d, d=%d, chi_max=%d, D_mpo=%d\n", L, d, chi_max, D_mpo);
        printf("  segments=%d, outer=%d, local=%d, warmup=%d\n",
               n_segments, n_outer, n_local, n_warmup);
        printf("  SVD: %s\n", gpu_svd ? "GPU (rocsolver)" : "CPU (LAPACK)");
        printf("  Bond split: %s\n", rsvd ? "rSVD" : "SVD");
        printf("  Eigensolver: %s\n", chebyshev ? "Chebyshev" : (davidson ? "Block-Davidson" : "Lanczos"));
        printf("======================================\n\n");
    }

    std::map<int, double> exact_energies = {
        {4, -1.616025403784},
        {8, -3.374932598688},
        {16, -6.911737145575},
        {32, -13.997315618007},
    };

    PDMRGGPUOpt<double> pdmrg(L, d, chi_max, D_mpo, n_segments, 1e-10);
    pdmrg.set_cpu_svd(!gpu_svd);
    pdmrg.set_use_davidson(davidson);
    pdmrg.set_rsvd(rsvd);
    pdmrg.set_use_batched_sweep(batched_sweep);
    pdmrg.set_use_chebyshev(chebyshev);
    pdmrg.set_quiet(quiet);
    pdmrg.initialize_mps_random();

    std::vector<double*> h_mpo_tensors(L);
    build_heisenberg_mpo(L, D_mpo, h_mpo_tensors);
    pdmrg.set_mpo(h_mpo_tensors);

    double energy = pdmrg.run(n_outer, n_local, n_warmup, n_polish);

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
int test_tfim(int L, int chi_max, int n_outer, int n_segments,
              int n_local, int n_warmup, int n_polish, bool gpu_svd,
              bool davidson, bool rsvd, double J, double h_field, bool quiet, bool batched_sweep,
              bool chebyshev) {
    int d = 2;
    int D_mpo = 3;

    if (!quiet) {
        printf("======================================\n");
        printf("PDMRG-GPU-OPT TFIM Test (float64)\n");
        printf("======================================\n");
        printf("  L=%d, d=%d, chi_max=%d, D_mpo=%d\n", L, d, chi_max, D_mpo);
        printf("  segments=%d, outer=%d, local=%d, warmup=%d\n",
               n_segments, n_outer, n_local, n_warmup);
        printf("  J=%.4f, h=%.4f\n", J, h_field);
        printf("  SVD: %s\n", gpu_svd ? "GPU (rocsolver)" : "CPU (LAPACK)");
        printf("  Bond split: %s\n", rsvd ? "rSVD" : "SVD");
        printf("  Eigensolver: %s\n", chebyshev ? "Chebyshev" : (davidson ? "Block-Davidson" : "Lanczos"));
        printf("======================================\n\n");
    }

    PDMRGGPUOpt<double> pdmrg(L, d, chi_max, D_mpo, n_segments, 1e-10);
    pdmrg.set_cpu_svd(!gpu_svd);
    pdmrg.set_use_davidson(davidson);
    pdmrg.set_rsvd(rsvd);
    pdmrg.set_use_batched_sweep(batched_sweep);
    pdmrg.set_use_chebyshev(chebyshev);
    pdmrg.set_quiet(quiet);
    pdmrg.initialize_mps_random();

    std::vector<double*> h_mpo_tensors(L);
    build_tfim_mpo(L, D_mpo, J, h_field, h_mpo_tensors);
    pdmrg.set_mpo(h_mpo_tensors);

    double energy = pdmrg.run(n_outer, n_local, n_warmup, n_polish);

    // Final energy already printed by run()
    if (!quiet) printf("Energy per site: %.12f\n", energy / L);

    for (auto ptr : h_mpo_tensors) delete[] ptr;
    return 0;
}

// ============================================================================
// Josephson Junction test (complex128)
// ============================================================================
int test_josephson(int L, int chi_max, int n_outer, int n_segments,
                   int n_local, int n_warmup, int n_polish, bool gpu_svd,
                   bool davidson, bool rsvd, int n_max, double E_J, double E_C, double phi_ext,
                   bool quiet, bool batched_sweep, bool chebyshev) {
    int d = 2 * n_max + 1;
    int D_mpo = 4;

    if (!quiet) {
        printf("======================================\n");
        printf("PDMRG-GPU-OPT Josephson Test (complex128)\n");
        printf("======================================\n");
        printf("  L=%d, d=%d (n_max=%d), chi_max=%d, D_mpo=%d\n",
               L, d, n_max, chi_max, D_mpo);
        printf("  segments=%d, outer=%d, local=%d, warmup=%d\n",
               n_segments, n_outer, n_local, n_warmup);
        printf("  E_J=%.2f, E_C=%.2f, phi_ext=pi/%.1f\n", E_J, E_C, M_PI / phi_ext);
        printf("  SVD: %s\n", gpu_svd ? "GPU (rocsolver)" : "CPU (LAPACK)");
        printf("  Bond split: %s\n", rsvd ? "rSVD" : "SVD");
        printf("  Eigensolver: %s\n", chebyshev ? "Chebyshev" : (davidson ? "Block-Davidson" : "Lanczos"));
        printf("======================================\n\n");
    }

    std::map<int, double> exact_energies;
    if (n_max == 1 && std::abs(E_J - 1.0) < 1e-10 && std::abs(E_C - 0.5) < 1e-10
        && std::abs(phi_ext - M_PI/4) < 1e-10) {
        exact_energies = {
            {4, -1.053346829927396},
            {6, -1.748843818181493},
        };
    }

    PDMRGGPUOpt<Complex> pdmrg(L, d, chi_max, D_mpo, n_segments, 1e-10);
    pdmrg.set_cpu_svd(!gpu_svd);
    pdmrg.set_use_davidson(davidson);
    pdmrg.set_rsvd(rsvd);
    pdmrg.set_use_batched_sweep(batched_sweep);
    pdmrg.set_use_chebyshev(chebyshev);
    pdmrg.set_quiet(quiet);
    pdmrg.initialize_mps_random();

    std::vector<Complex*> h_mpo_tensors(L);
    build_josephson_mpo(L, d, D_mpo, E_J, E_C, 0.0, n_max, phi_ext, h_mpo_tensors);
    pdmrg.set_mpo(h_mpo_tensors);

    double energy = pdmrg.run(n_outer, n_local, n_warmup, n_polish);

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
// Main
// ============================================================================
int main(int argc, char** argv) {
    int L = 8;
    int chi_max = 32;
    int n_outer = 20;
    int n_segments = 2;
    int n_local = 2;
    int n_warmup = 1;  // CLAUDE.md compliant: n_warmup <= 2
    int n_polish = 0;  // CLAUDE.md compliant: n_polish <= 2; 0 = skip polish phase
    bool gpu_svd = true;
    bool davidson = false;  // default: use Lanczos eigensolver
    bool rsvd = false;      // default: no randomized SVD
    bool quiet = false;
    bool batched_sweep = false;  // cross-segment batched: slower for 2 segments
    bool chebyshev = false;      // Chebyshev-filtered eigensolver
    bool run_josephson = false;
    bool run_tfim = false;
    int n_max = 1;
    double E_J = 1.0, E_C = 0.5, phi_ext = M_PI / 4;
    double J_tfim = 1.0, h_field = 1.0;

    // Parse args
    {
        int pos = 0;
        for (int i = 1; i < argc; i++) {
            if (std::string(argv[i]) == "--cpu-svd") { gpu_svd = false; continue; }
            if (std::string(argv[i]) == "--rsvd") { rsvd = true; continue; }
            if (std::string(argv[i]) == "--davidson") { davidson = true; continue; }
            if (std::string(argv[i]) == "--quiet") { quiet = true; continue; }
            if (std::string(argv[i]) == "--batched-sweep") { batched_sweep = true; continue; }
            if (std::string(argv[i]) == "--no-batched-sweep") { batched_sweep = false; continue; }
            if (std::string(argv[i]) == "--chebyshev") { chebyshev = true; continue; }
            if (std::string(argv[i]) == "--josephson") { run_josephson = true; continue; }
            if (std::string(argv[i]) == "--tfim") { run_tfim = true; continue; }
            if (std::string(argv[i]) == "--hfield" && i+1 < argc) { h_field = std::atof(argv[++i]); continue; }
            if (std::string(argv[i]) == "--segments" && i+1 < argc) { n_segments = std::atoi(argv[++i]); continue; }
            if (std::string(argv[i]) == "--local-sweeps" && i+1 < argc) { n_local = std::atoi(argv[++i]); continue; }
            if (std::string(argv[i]) == "--warmup" && i+1 < argc) { n_warmup = std::atoi(argv[++i]); continue; }
            if (std::string(argv[i]) == "--polish" && i+1 < argc) { n_polish = std::atoi(argv[++i]); continue; }
            if (std::string(argv[i]) == "--nmax" && i+1 < argc) { n_max = std::atoi(argv[++i]); continue; }
            if (std::string(argv[i]) == "--ej" && i+1 < argc) { E_J = std::atof(argv[++i]); continue; }
            if (std::string(argv[i]) == "--ec" && i+1 < argc) { E_C = std::atof(argv[++i]); continue; }
            if (std::string(argv[i]) == "--phi" && i+1 < argc) { phi_ext = std::atof(argv[++i]); continue; }
            if (argv[i][0] == '-') continue;
            if (pos == 0) L = std::atoi(argv[i]);
            else if (pos == 1) chi_max = std::atoi(argv[i]);
            else if (pos == 2) n_outer = std::atoi(argv[i]);
            pos++;
        }
    }

    try {
        if (run_josephson) {
            return test_josephson(L, chi_max, n_outer, n_segments, n_local, n_warmup, n_polish,
                                  gpu_svd, davidson, rsvd, n_max, E_J, E_C, phi_ext,
                                  quiet, batched_sweep, chebyshev);
        } else if (run_tfim) {
            return test_tfim(L, chi_max, n_outer, n_segments, n_local, n_warmup, n_polish,
                             gpu_svd, davidson, rsvd, J_tfim, h_field,
                             quiet, batched_sweep, chebyshev);
        } else {
            return test_heisenberg(L, chi_max, n_outer, n_segments, n_local, n_warmup, n_polish,
                                   gpu_svd, davidson, rsvd, quiet, batched_sweep,
                                   chebyshev);
        }
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
