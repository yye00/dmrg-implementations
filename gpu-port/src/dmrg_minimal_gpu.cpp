// Minimal GPU-Only DMRG for Heisenberg Chain - AMD MI300X
// Upload MPS at start → Full GPU computation → Download at end
// Simplified for nearest-neighbor Hamiltonians

#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <chrono>
#include <iomanip>

using Complex = hipDoubleComplex;

#define HIP_CHECK(call) { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << hipGetErrorString(err) << std::endl; \
        exit(1); \
    } \
}

inline Complex make_complex(double re, double im) {
    return hipDoubleComplex{re, im};
}

inline double get_real(const rocblas_double_complex& z) {
    return reinterpret_cast<const hipDoubleComplex*>(&z)->x;
}

// ============================================================================
// Minimal Power Iteration Eigensolver
// ============================================================================

class PowerIterationSolver {
private:
    rocblas_handle handle;
    int max_iter;
    double tol;

public:
    PowerIterationSolver(rocblas_handle h, int max_it = 30, double tolerance = 1e-12)
        : handle(h), max_iter(max_it), tol(tolerance) {}

    template<typename ApplyH>
    double solve(ApplyH apply_H, int dim, Complex* d_psi_inout) {
        Complex* d_Hpsi;
        HIP_CHECK(hipMalloc(&d_Hpsi, dim * sizeof(Complex)));

        double energy = 0.0;

        for (int iter = 0; iter < max_iter; iter++) {
            apply_H(d_psi_inout, d_Hpsi);

            // Compute energy = <psi|H|psi>
            Complex* d_Hpsi_orig;
            HIP_CHECK(hipMalloc(&d_Hpsi_orig, dim * sizeof(Complex)));
            apply_H(d_psi_inout, d_Hpsi_orig);

            rocblas_double_complex energy_z;
            rocblas_zdotc(handle, dim,
                         (rocblas_double_complex*)d_psi_inout, 1,
                         (rocblas_double_complex*)d_Hpsi_orig, 1,
                         &energy_z);
            HIP_CHECK(hipFree(d_Hpsi_orig));

            energy = get_real(energy_z);

            // Flip sign for power iteration: |psi> = -H|psi>
            Complex neg_one = make_complex(-1.0, 0.0);
            rocblas_zscal(handle, dim, (rocblas_double_complex*)&neg_one,
                         (rocblas_double_complex*)d_Hpsi, 1);

            // Normalize
            rocblas_double_complex norm_z;
            rocblas_zdotc(handle, dim,
                         (rocblas_double_complex*)d_Hpsi, 1,
                         (rocblas_double_complex*)d_Hpsi, 1,
                         &norm_z);

            double norm = std::sqrt(get_real(norm_z));
            Complex inv_norm = make_complex(1.0 / norm, 0.0);
            rocblas_zscal(handle, dim, (rocblas_double_complex*)&inv_norm,
                         (rocblas_double_complex*)d_Hpsi, 1);

            HIP_CHECK(hipMemcpy(d_psi_inout, d_Hpsi, dim * sizeof(Complex),
                               hipMemcpyDeviceToDevice));
        }

        HIP_CHECK(hipFree(d_Hpsi));
        return energy;
    }
};

// ============================================================================
// Minimal DMRG Engine - All GPU
// ============================================================================

class MinimalDMRG {
private:
    int L, d, max_D, n_sweeps;
    rocblas_handle rb_handle;

    std::vector<int> bond_dims;
    std::vector<Complex*> d_mps;

    double current_energy;

public:
    MinimalDMRG(int chain_length, int phys_dim, int max_bond, int sweeps)
        : L(chain_length), d(phys_dim), max_D(max_bond), n_sweeps(sweeps),
          current_energy(0.0) {

        rocblas_create_handle(&rb_handle);

        // Initialize bond dimensions
        bond_dims.resize(L + 1);
        bond_dims[0] = 1;
        bond_dims[L] = 1;
        for (int i = 1; i < L; i++) {
            bond_dims[i] = std::min(max_D, 1 << std::min(i, L - i));
        }

        // Initialize MPS on GPU with random data
        srand(42);
        d_mps.resize(L);
        for (int i = 0; i < L; i++) {
            int size = bond_dims[i] * d * bond_dims[i + 1];
            HIP_CHECK(hipMalloc(&d_mps[i], size * sizeof(Complex)));

            std::vector<Complex> h_mps(size);
            for (int j = 0; j < size; j++) {
                double r = (double)rand() / RAND_MAX - 0.5;
                double im = (double)rand() / RAND_MAX - 0.5;
                h_mps[j] = make_complex(r, im);
            }
            HIP_CHECK(hipMemcpy(d_mps[i], h_mps.data(), size * sizeof(Complex),
                               hipMemcpyHostToDevice));
        }
    }

    ~MinimalDMRG() {
        for (auto& p : d_mps) HIP_CHECK(hipFree(p));
        rocblas_destroy_handle(rb_handle);
    }

    double run() {
        auto t_start = std::chrono::high_resolution_clock::now();

        std::cout << "\n===========================================\n";
        std::cout << "Minimal GPU-Only DMRG - AMD MI300X\n";
        std::cout << "===========================================\n";
        std::cout << "Chain length: L = " << L << "\n";
        std::cout << "Max bond dim: D = " << max_D << "\n";
        std::cout << "Sweeps: " << n_sweeps << "\n";
        std::cout << "Expected E ≈ -5.142 (Heisenberg L=12)\n\n";

        for (int sweep = 0; sweep < n_sweeps; sweep++) {
            bool left_to_right = (sweep % 2 == 0);

            if (left_to_right) {
                for (int site = 0; site < L - 1; site++) {
                    optimize_bond(site);
                }
            } else {
                for (int site = L - 2; site >= 0; site--) {
                    optimize_bond(site);
                }
            }

            // Compute energy by summing bond energies
            current_energy = compute_energy_gpu();

            std::cout << "Sweep " << std::setw(2) << sweep
                      << " | E = " << std::fixed << std::setprecision(10) << current_energy
                      << " | E/site = " << (current_energy / L)
                      << "\n";
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double time_sec = std::chrono::duration<double>(t_end - t_start).count();

        std::cout << "\n===========================================\n";
        std::cout << "Final Energy: " << std::fixed << std::setprecision(12) << current_energy << "\n";
        std::cout << "Time: " << time_sec << " seconds\n";
        std::cout << "===========================================\n";

        return current_energy;
    }

private:
    void optimize_bond(int site) {
        int D_L = bond_dims[site];
        int D_M = bond_dims[site + 1];
        int D_R = bond_dims[site + 2];
        int psi_size = D_L * d * d * D_R;

        // Form 2-site wavefunction on GPU
        Complex* d_theta;
        HIP_CHECK(hipMalloc(&d_theta, psi_size * sizeof(Complex)));

        Complex alpha = make_complex(1.0, 0.0);
        Complex beta = make_complex(0.0, 0.0);

        // theta = MPS[site] * MPS[site+1]
        rocblas_zgemm(rb_handle, rocblas_operation_none, rocblas_operation_none,
                     d * D_R, D_L * d, D_M,
                     (rocblas_double_complex*)&alpha,
                     (rocblas_double_complex*)d_mps[site + 1], d * D_R,
                     (rocblas_double_complex*)d_mps[site], D_M,
                     (rocblas_double_complex*)&beta,
                     (rocblas_double_complex*)d_theta, d * D_R);

        // Optimize using power iteration with local Hamiltonian
        auto apply_H = [&](const Complex* d_in, Complex* d_out) {
            apply_local_heisenberg(d_in, d_out, D_L, D_R);
        };

        PowerIterationSolver solver(rb_handle, 30, 1e-12);
        solver.solve(apply_H, psi_size, d_theta);

        // SVD and update MPS
        svd_update(site, d_theta);

        HIP_CHECK(hipFree(d_theta));
    }

    void apply_local_heisenberg(const Complex* d_psi, Complex* d_Hpsi, int D_L, int D_R) {
        // Apply 2-site Heisenberg operator: H = Sx⊗Sx + Sy⊗Sy + Sz⊗Sz
        // Matrix in basis |↑↑⟩, |↑↓⟩, |↓↑⟩, |↓↓⟩
        std::vector<Complex> h_H(16, make_complex(0.0, 0.0));
        h_H[0] = make_complex(0.25, 0.0);     // |↑↑⟩→|↑↑⟩
        h_H[5] = make_complex(-0.25, 0.0);    // |↑↓⟩→|↑↓⟩
        h_H[6] = make_complex(0.5, 0.0);      // |↑↓⟩→|↓↑⟩
        h_H[9] = make_complex(0.5, 0.0);      // |↓↑⟩→|↑↓⟩
        h_H[10] = make_complex(-0.25, 0.0);   // |↓↑⟩→|↓↑⟩
        h_H[15] = make_complex(0.25, 0.0);    // |↓↓⟩→|↓↓⟩

        Complex* d_H;
        HIP_CHECK(hipMalloc(&d_H, 16 * sizeof(Complex)));
        HIP_CHECK(hipMemcpy(d_H, h_H.data(), 16 * sizeof(Complex), hipMemcpyHostToDevice));

        Complex alpha = make_complex(1.0, 0.0);
        Complex beta = make_complex(0.0, 0.0);

        // Apply batched matrix-vector product for each (a,b) configuration
        rocblas_zgemm_strided_batched(rb_handle,
            rocblas_operation_none, rocblas_operation_none,
            1, 4, 4,
            (rocblas_double_complex*)&alpha,
            (rocblas_double_complex*)d_psi, 1, 4,
            (rocblas_double_complex*)d_H, 4, 0,
            (rocblas_double_complex*)&beta,
            (rocblas_double_complex*)d_Hpsi, 1, 4,
            D_L * D_R);

        HIP_CHECK(hipFree(d_H));
    }

    void svd_update(int site, Complex* d_theta) {
        int D_L = bond_dims[site];
        int D_M = bond_dims[site + 1];
        int D_R = bond_dims[site + 2];

        // Reshape theta: (D_L, d, d, D_R) → (D_L*d, d*D_R) for SVD
        int m = D_L * d;
        int n = d * D_R;
        int k = std::min(m, n);

        Complex* d_U;
        Complex* d_Vt;
        double* d_S;
        double* d_E;
        int* d_info;

        HIP_CHECK(hipMalloc(&d_U, m * k * sizeof(Complex)));
        HIP_CHECK(hipMalloc(&d_Vt, k * n * sizeof(Complex)));
        HIP_CHECK(hipMalloc(&d_S, k * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_E, k * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_info, sizeof(int)));

        // Thin SVD: theta = U * S * Vt
        rocsolver_zgesvd(rb_handle,
                       rocblas_svect_singular,
                       rocblas_svect_singular,
                       m, n,
                       (rocblas_double_complex*)d_theta, m,
                       d_S,
                       (rocblas_double_complex*)d_U, m,
                       (rocblas_double_complex*)d_Vt, k,  // FIX: ldvt = k
                       d_E, rocblas_outofplace, d_info);

        // Keep bond dimension fixed at D_M
        int D_new = D_M;

        // Get singular values
        std::vector<double> h_S(std::min(D_new, k));
        HIP_CHECK(hipMemcpy(h_S.data(), d_S, std::min(D_new, k) * sizeof(double), hipMemcpyDeviceToHost));

        // Create new left tensor: (D_L, d, D_new) = reshape(U[:, :D_new] * sqrt(S))
        Complex* d_mps_new_left;
        int left_size = D_L * d * D_new;
        HIP_CHECK(hipMalloc(&d_mps_new_left, left_size * sizeof(Complex)));
        HIP_CHECK(hipMemset(d_mps_new_left, 0, left_size * sizeof(Complex)));

        int num_sv = std::min(D_new, k);
        for (int col = 0; col < num_sv; col++) {
            double sqrt_s = std::sqrt(std::max(h_S[col], 0.0));
            Complex scale = make_complex(sqrt_s, 0.0);
            rocblas_zcopy(rb_handle, m,
                         (rocblas_double_complex*)(d_U + col * m), 1,
                         (rocblas_double_complex*)(d_mps_new_left + col * m), 1);
            rocblas_zscal(rb_handle, m, (rocblas_double_complex*)&scale,
                         (rocblas_double_complex*)(d_mps_new_left + col * m), 1);
        }

        // Create new right tensor: (D_new, d, D_R) = reshape(sqrt(S) * Vt[:D_new, :])
        Complex* d_mps_new_right;
        int right_size = D_new * d * D_R;
        HIP_CHECK(hipMalloc(&d_mps_new_right, right_size * sizeof(Complex)));
        HIP_CHECK(hipMemset(d_mps_new_right, 0, right_size * sizeof(Complex)));

        for (int row = 0; row < num_sv; row++) {
            double sqrt_s = std::sqrt(std::max(h_S[row], 0.0));
            Complex scale = make_complex(sqrt_s, 0.0);
            rocblas_zcopy(rb_handle, n,
                         (rocblas_double_complex*)(d_Vt + row), k,
                         (rocblas_double_complex*)(d_mps_new_right + row), D_new);
            rocblas_zscal(rb_handle, n, (rocblas_double_complex*)&scale,
                         (rocblas_double_complex*)(d_mps_new_right + row), D_new);
        }

        // Replace MPS tensors
        Complex* old_left = d_mps[site];
        Complex* old_right = d_mps[site + 1];

        d_mps[site] = d_mps_new_left;
        d_mps[site + 1] = d_mps_new_right;

        HIP_CHECK(hipFree(old_left));
        HIP_CHECK(hipFree(old_right));

        // Cleanup SVD workspace
        HIP_CHECK(hipFree(d_U));
        HIP_CHECK(hipFree(d_Vt));
        HIP_CHECK(hipFree(d_S));
        HIP_CHECK(hipFree(d_E));
        HIP_CHECK(hipFree(d_info));
    }

    double compute_energy_gpu() {
        // Compute total energy by summing bond energies on GPU
        double total_energy = 0.0;

        for (int bond = 0; bond < L - 1; bond++) {
            int D_L = bond_dims[bond];
            int D_M = bond_dims[bond + 1];
            int D_R = bond_dims[bond + 2];
            int theta_size = D_L * d * d * D_R;

            // Form 2-site wavefunction
            Complex* d_theta;
            HIP_CHECK(hipMalloc(&d_theta, theta_size * sizeof(Complex)));

            Complex alpha = make_complex(1.0, 0.0);
            Complex beta = make_complex(0.0, 0.0);

            rocblas_zgemm(rb_handle, rocblas_operation_none, rocblas_operation_none,
                         d * D_R, D_L * d, D_M,
                         (rocblas_double_complex*)&alpha,
                         (rocblas_double_complex*)d_mps[bond + 1], d * D_R,
                         (rocblas_double_complex*)d_mps[bond], D_M,
                         (rocblas_double_complex*)&beta,
                         (rocblas_double_complex*)d_theta, d * D_R);

            // Apply Hamiltonian
            Complex* d_H_theta;
            HIP_CHECK(hipMalloc(&d_H_theta, theta_size * sizeof(Complex)));
            apply_local_heisenberg(d_theta, d_H_theta, D_L, D_R);

            // Compute <theta|H|theta>
            rocblas_double_complex bond_energy_z;
            rocblas_zdotc(rb_handle, theta_size,
                         (rocblas_double_complex*)d_theta, 1,
                         (rocblas_double_complex*)d_H_theta, 1,
                         &bond_energy_z);

            double bond_energy = get_real(bond_energy_z);

            // Normalize by <theta|theta>
            rocblas_double_complex norm_z;
            rocblas_zdotc(rb_handle, theta_size,
                         (rocblas_double_complex*)d_theta, 1,
                         (rocblas_double_complex*)d_theta, 1,
                         &norm_z);

            double norm = get_real(norm_z);
            bond_energy /= norm;

            total_energy += bond_energy;

            HIP_CHECK(hipFree(d_theta));
            HIP_CHECK(hipFree(d_H_theta));
        }

        return total_energy;
    }
};

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "====================================================\n";
    std::cout << "Minimal GPU-Only DMRG - AMD MI300X\n";
    std::cout << "====================================================\n\n";

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Memory: " << (prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0) << " GB\n\n";

    MinimalDMRG dmrg(12, 2, 100, 10);
    dmrg.run();

    return 0;
}
