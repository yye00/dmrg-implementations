#include "dmrg2_gpu.h"
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
int test_heisenberg(int L, int chi_max, int n_sweeps, bool gpu_svd, bool rsvd = false) {
    int d = 2;
    int D_mpo = 5;

    printf("======================================\n");
    printf("Two-Site Heisenberg DMRG-GPU Test (float64)\n");
    printf("======================================\n");
    printf("  L=%d, d=%d, chi_max=%d, D_mpo=%d, sweeps=%d\n", L, d, chi_max, D_mpo, n_sweeps);
    printf("  SVD: %s%s\n", gpu_svd ? "GPU (rocsolver)" : "CPU (LAPACK)",
           rsvd ? " + randomized truncation" : "");
    printf("======================================\n\n");

    std::map<int, double> exact_energies = {
        {4, -1.616025403784},
        {8, -3.374932598688},
    };

    DMRG2GPU<double> dmrg(L, d, chi_max, D_mpo, 1e-12);
    dmrg.set_cpu_svd(!gpu_svd);
    dmrg.set_rsvd(rsvd);
    dmrg.initialize_mps_random();

    std::vector<double*> h_mpo_tensors(L);
    build_heisenberg_mpo(L, D_mpo, h_mpo_tensors);
    dmrg.set_mpo(h_mpo_tensors);

    double energy = dmrg.run(n_sweeps);

    printf("\n--- RESULT ---\n");
    printf("Final energy: %.12f\n", energy);

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
// Josephson Junction test (complex128)
// ============================================================================
int test_josephson(int L, int chi_max, int n_sweeps, bool gpu_svd,
                   int n_max, double E_J, double E_C, double phi_ext) {
    int d = 2 * n_max + 1;
    int D_mpo = 4;

    printf("======================================\n");
    printf("Two-Site Josephson DMRG-GPU Test (complex128)\n");
    printf("======================================\n");
    printf("  L=%d, d=%d (n_max=%d), chi_max=%d, D_mpo=%d, sweeps=%d\n",
           L, d, n_max, chi_max, D_mpo, n_sweeps);
    printf("  E_J=%.2f, E_C=%.2f, phi_ext=pi/%.1f\n", E_J, E_C, M_PI / phi_ext);
    printf("  SVD: %s\n", gpu_svd ? "GPU (rocsolver)" : "CPU (LAPACK)");
    printf("======================================\n\n");

    std::map<int, double> exact_energies;
    if (n_max == 1 && std::abs(E_J - 1.0) < 1e-10 && std::abs(E_C - 0.5) < 1e-10
        && std::abs(phi_ext - M_PI/4) < 1e-10) {
        exact_energies = {
            {4, -1.053346829927396},
            {6, -1.748843818181493},
        };
    }

    DMRG2GPU<Complex> dmrg(L, d, chi_max, D_mpo, 1e-12);
    dmrg.set_cpu_svd(!gpu_svd);
    dmrg.initialize_mps_random();

    std::vector<Complex*> h_mpo_tensors(L);
    build_josephson_mpo(L, d, D_mpo, E_J, E_C, 0.0, n_max, phi_ext, h_mpo_tensors);
    dmrg.set_mpo(h_mpo_tensors);

    double energy = dmrg.run(n_sweeps);

    printf("\n--- RESULT ---\n");
    printf("Final energy: %.12f\n", energy);

    int ret = 0;
    if (exact_energies.count(L) > 0) {
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
// Main
// ============================================================================
int main(int argc, char** argv) {
    int L = 8;
    int chi_max = 32;
    int n_sweeps = 30;
    bool gpu_svd = false;
    bool rsvd = false;
    bool run_josephson = false;
    int n_max = 1;
    double E_J = 1.0, E_C = 0.5, phi_ext = M_PI / 4;

    // Parse args
    {
        int pos = 0;
        for (int i = 1; i < argc; i++) {
            if (std::string(argv[i]) == "--gpu-svd") { gpu_svd = true; continue; }
            if (std::string(argv[i]) == "--rsvd") { rsvd = true; continue; }
            if (std::string(argv[i]) == "--josephson") { run_josephson = true; continue; }
            if (std::string(argv[i]) == "--nmax" && i+1 < argc) { n_max = std::atoi(argv[++i]); continue; }
            if (std::string(argv[i]) == "--ej" && i+1 < argc) { E_J = std::atof(argv[++i]); continue; }
            if (std::string(argv[i]) == "--ec" && i+1 < argc) { E_C = std::atof(argv[++i]); continue; }
            if (std::string(argv[i]) == "--phi" && i+1 < argc) { phi_ext = std::atof(argv[++i]); continue; }
            if (argv[i][0] == '-') continue;
            if (pos == 0) L = std::atoi(argv[i]);
            else if (pos == 1) chi_max = std::atoi(argv[i]);
            else if (pos == 2) n_sweeps = std::atoi(argv[i]);
            pos++;
        }
    }

    try {
        if (run_josephson) {
            return test_josephson(L, chi_max, n_sweeps, gpu_svd, n_max, E_J, E_C, phi_ext);
        } else {
            return test_heisenberg(L, chi_max, n_sweeps, gpu_svd, rsvd);
        }
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
