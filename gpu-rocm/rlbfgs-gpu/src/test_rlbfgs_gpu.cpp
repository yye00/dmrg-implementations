#include "rlbfgs_gpu.h"
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
// TFIM MPO (real, D=3)
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
// Heisenberg test (real)
// ============================================================================
int test_heisenberg(int L, int chi_max, const RLBFGSGPU<double>::Config& cfg,
                    bool cpu_svd) {
    int d = 2;
    int D_mpo = 5;

    printf("======================================\n");
    printf("RLBFGS-GPU Heisenberg Test (float64)\n");
    printf("======================================\n");
    printf("  L=%d, d=%d, chi_max=%d, D_mpo=%d\n", L, d, chi_max, D_mpo);
    printf("  max_epochs=%d, warmup=%d, history=%d\n",
           cfg.max_epochs, cfg.n_warmup, cfg.history_size);
    printf("  c1=%.1e, alpha_init=%.2f, beta=%.2f, grad_tol=%.1e\n",
           cfg.c1, cfg.alpha_init, cfg.beta, cfg.grad_tol);
    printf("  SVD: %s\n", cpu_svd ? "CPU (LAPACK)" : "GPU (rocsolver)");
    printf("======================================\n\n");

    std::map<int, double> exact_energies = {
        {4, -1.616025403784},
        {8, -3.374932598688},
        {16, -6.911737145575},
        {32, -13.997315618007},
    };

    RLBFGSGPU<double> opt(L, d, chi_max, D_mpo, 1e-10);
    opt.set_cpu_svd(cpu_svd);
    opt.initialize_mps_random();

    std::vector<double*> h_mpo_tensors(L);
    build_heisenberg_mpo(L, D_mpo, h_mpo_tensors);
    opt.set_mpo(h_mpo_tensors);

    double energy = opt.run(cfg);

    int ret = 0;
    if (exact_energies.count(L) > 0) {
        double exact = exact_energies[L];
        double error = std::abs(energy - exact);
        printf("Exact energy: %.12f\n", exact);
        printf("Absolute error: %.2e\n", error);
        if (error < 1e-6) printf("PASS\n");
        else { printf("FAIL\n"); ret = 1; }
    }

    for (auto ptr : h_mpo_tensors) delete[] ptr;
    return ret;
}

// ============================================================================
// TFIM test (real)
// ============================================================================
int test_tfim(int L, int chi_max, const RLBFGSGPU<double>::Config& cfg,
              bool cpu_svd, double J, double h_field) {
    int d = 2;
    int D_mpo = 3;

    printf("======================================\n");
    printf("RLBFGS-GPU TFIM Test (float64)\n");
    printf("======================================\n");
    printf("  L=%d, d=%d, chi_max=%d, D_mpo=%d\n", L, d, chi_max, D_mpo);
    printf("  J=%.4f, h=%.4f\n", J, h_field);
    printf("  max_epochs=%d, warmup=%d, history=%d\n",
           cfg.max_epochs, cfg.n_warmup, cfg.history_size);
    printf("======================================\n\n");

    RLBFGSGPU<double> opt(L, d, chi_max, D_mpo, 1e-10);
    opt.set_cpu_svd(cpu_svd);
    opt.initialize_mps_random();

    std::vector<double*> h_mpo_tensors(L);
    build_tfim_mpo(L, D_mpo, J, h_field, h_mpo_tensors);
    opt.set_mpo(h_mpo_tensors);

    double energy = opt.run(cfg);
    printf("Energy per site: %.12f\n", energy / L);

    for (auto ptr : h_mpo_tensors) delete[] ptr;
    return 0;
}

// ============================================================================
// Josephson test (complex)
// ============================================================================
int test_josephson(int L, int chi_max, const RLBFGSGPU<Complex>::Config& cfg,
                   bool cpu_svd, int n_max, double E_J, double E_C, double phi_ext) {
    int d = 2 * n_max + 1;
    int D_mpo = 4;

    printf("======================================\n");
    printf("RLBFGS-GPU Josephson Test (complex128)\n");
    printf("======================================\n");
    printf("  L=%d, d=%d (n_max=%d), chi_max=%d, D_mpo=%d\n",
           L, d, n_max, chi_max, D_mpo);
    printf("  E_J=%.2f, E_C=%.2f, phi_ext=%.4f\n", E_J, E_C, phi_ext);
    printf("  max_epochs=%d, warmup=%d, history=%d\n",
           cfg.max_epochs, cfg.n_warmup, cfg.history_size);
    printf("======================================\n\n");

    std::map<int, double> exact_energies;
    if (n_max == 1 && std::abs(E_J - 1.0) < 1e-10 && std::abs(E_C - 0.5) < 1e-10
        && std::abs(phi_ext - M_PI/4) < 1e-10) {
        exact_energies = {
            {4, -1.053346829927396},
            {6, -1.748843818181493},
        };
    }

    RLBFGSGPU<Complex> opt(L, d, chi_max, D_mpo, 1e-10);
    opt.set_cpu_svd(cpu_svd);
    opt.initialize_mps_random();

    std::vector<Complex*> h_mpo_tensors(L);
    build_josephson_mpo(L, d, D_mpo, E_J, E_C, 0.0, n_max, phi_ext, h_mpo_tensors);
    opt.set_mpo(h_mpo_tensors);

    double energy = opt.run(cfg);

    int ret = 0;
    if (exact_energies.count(L) > 0) {
        double exact = exact_energies[L];
        double error = std::abs(energy - exact);
        printf("Exact energy: %.12f\n", exact);
        printf("Absolute error: %.2e\n", error);
        if (error < 1e-6) printf("PASS\n");
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
    int max_epochs = 200;
    int n_warmup = 0;
    int history_size = 10;
    double c1 = 1e-4;
    double alpha_init = 1.0;
    double ls_beta = 0.5;
    double grad_tol = 1e-10;
    double energy_tol = 1e-14;
    bool cpu_svd = true;
    bool quiet = false;
    int log_every = 1;

    bool run_josephson = false;
    bool run_tfim = false;
    int n_max = 1;
    double E_J = 1.0, E_C = 0.5, phi_ext = M_PI / 4;
    double J_tfim = 1.0, h_field = 1.0;

    {
        int pos = 0;
        for (int i = 1; i < argc; i++) {
            std::string a = argv[i];
            if (a == "--gpu-svd") { cpu_svd = false; continue; }
            if (a == "--cpu-svd") { cpu_svd = true; continue; }
            if (a == "--quiet") { quiet = true; continue; }
            if (a == "--josephson") { run_josephson = true; continue; }
            if (a == "--tfim") { run_tfim = true; continue; }
            if (a == "--warmup" && i+1 < argc) { n_warmup = std::atoi(argv[++i]); continue; }
            if (a == "--epochs" && i+1 < argc) { max_epochs = std::atoi(argv[++i]); continue; }
            if (a == "--history" && i+1 < argc) { history_size = std::atoi(argv[++i]); continue; }
            if (a == "--c1" && i+1 < argc) { c1 = std::atof(argv[++i]); continue; }
            if (a == "--alpha-init" && i+1 < argc) { alpha_init = std::atof(argv[++i]); continue; }
            if (a == "--ls-beta" && i+1 < argc) { ls_beta = std::atof(argv[++i]); continue; }
            if (a == "--grad-tol" && i+1 < argc) { grad_tol = std::atof(argv[++i]); continue; }
            if (a == "--energy-tol" && i+1 < argc) { energy_tol = std::atof(argv[++i]); continue; }
            if (a == "--log-every" && i+1 < argc) { log_every = std::atoi(argv[++i]); continue; }
            if (a == "--hfield" && i+1 < argc) { h_field = std::atof(argv[++i]); continue; }
            if (a == "--nmax" && i+1 < argc) { n_max = std::atoi(argv[++i]); continue; }
            if (a == "--ej" && i+1 < argc) { E_J = std::atof(argv[++i]); continue; }
            if (a == "--ec" && i+1 < argc) { E_C = std::atof(argv[++i]); continue; }
            if (a == "--phi" && i+1 < argc) { phi_ext = std::atof(argv[++i]); continue; }
            if (argv[i][0] == '-') continue;
            if (pos == 0) L = std::atoi(argv[i]);
            else if (pos == 1) chi_max = std::atoi(argv[i]);
            else if (pos == 2) max_epochs = std::atoi(argv[i]);
            pos++;
        }
    }

    // CLAUDE.md rule: warmup must be single-site and ≤ 2 sweeps
    if (n_warmup > 2) {
        fprintf(stderr, "ERROR: --warmup %d exceeds the max of 2 (CLAUDE.md rule)\n", n_warmup);
        return 1;
    }

    auto fill_cfg = [&](auto& cfg) {
        cfg.max_epochs = max_epochs;
        cfg.n_warmup = n_warmup;
        cfg.history_size = history_size;
        cfg.c1 = c1;
        cfg.alpha_init = alpha_init;
        cfg.beta = ls_beta;
        cfg.grad_tol = grad_tol;
        cfg.energy_tol = energy_tol;
        cfg.verbose = !quiet;
        cfg.log_every = log_every;
    };

    try {
        if (run_josephson) {
            RLBFGSGPU<Complex>::Config cfg;
            fill_cfg(cfg);
            return test_josephson(L, chi_max, cfg, cpu_svd, n_max, E_J, E_C, phi_ext);
        } else if (run_tfim) {
            RLBFGSGPU<double>::Config cfg;
            fill_cfg(cfg);
            return test_tfim(L, chi_max, cfg, cpu_svd, J_tfim, h_field);
        } else {
            RLBFGSGPU<double>::Config cfg;
            fill_cfg(cfg);
            return test_heisenberg(L, chi_max, cfg, cpu_svd);
        }
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
