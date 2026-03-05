// GPU DMRG implementation - supports both PDMRG and PDMRG2 variants
// Priority: 1. Accuracy (complex128), 2. Performance

#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include <hiptensor/hiptensor.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <chrono>

#include "gpu_memory.hpp"
#include "lanczos_eigensolver.hpp"
#include "block_davidson.hpp"
#include "svd_solver.hpp"
#include "dmrg_types.hpp"

using Complex = hipDoubleComplex;
using Clock = std::chrono::high_resolution_clock;

// DMRG algorithm variant
enum class DMRGVariant {
    PDMRG,   // Standard: Lanczos + StandardSVD + stream overlap
    PDMRG2   // Optimized: BlockDavidson + RandomizedSVD + GEMM-heavy
};

// DMRG GPU solver
class DMRG_GPU {
private:
    int L;                    // Chain length
    int max_bond;            // Maximum bond dimension
    int n_sweeps;            // Number of sweeps
    double tol;              // Convergence tolerance
    DMRGVariant variant;      // Algorithm variant

    // GPU resources
    StreamManager stream_mgr;
    hiptensorHandle_t hiptensor_handle;

    // Eigensolvers
    LanczosEigensolver* lanczos;
    BlockDavidson* davidson;

    // SVD solvers
    StandardSVD* std_svd;
    RandomizedSVD* rand_svd;

    // MPS tensors and MPO
    std::vector<GPUBuffer<Complex>> mps_tensors;  // A[i]: [D_left, d, D_right]
    std::vector<GPUBuffer<Complex>> mpo_tensors;  // W[i]: [DL, d_in, d_out, DR]

    // Environment tensors
    std::vector<GPUBuffer<Complex>> left_envs;    // L[i]: [DL_mps, DL_mpo]
    std::vector<GPUBuffer<Complex>> right_envs;   // R[i]: [DR_mpo, DR_mps]

    // Bond dimensions
    std::vector<int> bond_dims;

    // Timing statistics
    double time_contraction;
    double time_eigensolver;
    double time_svd;
    double time_total;

public:
    DMRG_GPU(int chain_length, int max_bond_dim, int sweeps,
             DMRGVariant var = DMRGVariant::PDMRG, double tolerance = 1e-12)
        : L(chain_length), max_bond(max_bond_dim), n_sweeps(sweeps),
          tol(tolerance), variant(var),
          stream_mgr(4),  // 4 streams for overlap
          time_contraction(0), time_eigensolver(0), time_svd(0), time_total(0) {

        // Create hipTensor handle
        hiptensorStatus_t status = hiptensorCreate(&hiptensor_handle);
        if (status != HIPTENSOR_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create hipTensor handle");
        }

        // Initialize eigensolvers based on variant
        if (variant == DMRGVariant::PDMRG) {
            lanczos = new LanczosEigensolver(50, tol);
            std_svd = new StandardSVD();
            davidson = nullptr;
            rand_svd = nullptr;
        } else {
            davidson = new BlockDavidson(4, 30, tol);
            rand_svd = new RandomizedSVD(10);
            lanczos = nullptr;
            std_svd = nullptr;
        }

        // Initialize MPS and bond dimensions
        bond_dims.resize(L + 1);
        bond_dims[0] = 1;
        bond_dims[L] = 1;
        for (int i = 1; i < L; i++) {
            bond_dims[i] = std::min(max_bond, 1 << std::min(i, L - i));
        }

        // Allocate MPS tensors
        mps_tensors.resize(L);
        for (int i = 0; i < L; i++) {
            int D_left = bond_dims[i];
            int d = 2;  // Spin-1/2
            int D_right = bond_dims[i + 1];
            mps_tensors[i].resize(D_left * d * D_right);
        }

        // Allocate environment tensors
        left_envs.resize(L + 1);
        right_envs.resize(L + 1);
    }

    ~DMRG_GPU() {
        hiptensorDestroy(hiptensor_handle);
        if (lanczos) delete lanczos;
        if (davidson) delete davidson;
        if (std_svd) delete std_svd;
        if (rand_svd) delete rand_svd;
    }

    // Run DMRG sweeps
    double run(const Tensor5D<std::complex<double>>& mpo_cpu);

private:
    // Initialize MPS to random state
    void initialize_mps();

    // Initialize MPO from CPU data
    void initialize_mpo(const Tensor5D<std::complex<double>>& mpo_cpu);

    // Initialize environments
    void initialize_environments();

    // Perform single DMRG sweep
    double perform_sweep(bool left_to_right);

    // Optimize single 2-site tensor
    double optimize_two_site(int site1, int site2, hipStream_t stream);

    // Build effective Hamiltonian H_eff
    void build_effective_hamiltonian(int site1, int site2,
                                     GPUBuffer<Complex>& d_H_eff,
                                     int& dim,
                                     hipStream_t stream);

    // Update left environment
    void update_left_environment(int site, hipStream_t stream);

    // Update right environment
    void update_right_environment(int site, hipStream_t stream);

    // Contract tensors using hipTensor
    void tensor_contract(const Complex* d_A, const int* dims_A, int ndim_A,
                        const Complex* d_B, const int* dims_B, int ndim_B,
                        Complex* d_C, const int* dims_C, int ndim_C,
                        const int* modes_A, const int* modes_B, const int* modes_C,
                        hipStream_t stream);

    // Print timing statistics
    void print_timing_stats() const;
};

void DMRG_GPU::initialize_mps() {
    // Initialize MPS to simple product state (all |0>)
    for (int i = 0; i < L; i++) {
        int D_left = bond_dims[i];
        int d = 2;
        int D_right = bond_dims[i + 1];

        std::vector<Complex> h_A(D_left * d * D_right);
        for (int a = 0; a < D_left; a++) {
            for (int s = 0; s < d; s++) {
                for (int b = 0; b < D_right; b++) {
                    int idx = a * d * D_right + s * D_right + b;
                    if (a == b && s == 0) {
                        h_A[idx] = make_complex(1.0, 0.0);
                    } else {
                        h_A[idx] = make_complex(0.0, 0.0);
                    }
                }
            }
        }
        mps_tensors[i].copy_from_host(h_A);
    }
}

void DMRG_GPU::initialize_mpo(const Tensor5D<std::complex<double>>& mpo_cpu) {
    mpo_tensors.resize(L);

    for (int i = 0; i < L; i++) {
        int DL = mpo_cpu[i].size();
        int d = mpo_cpu[i][0].size();
        int DR = (DL > 0 && d > 0) ? mpo_cpu[i][0][0][0].size() : 0;

        mpo_tensors[i].resize(DL * d * d * DR);

        std::vector<Complex> h_W(DL * d * d * DR);
        for (int a = 0; a < DL; a++) {
            for (int s1 = 0; s1 < d; s1++) {
                for (int s2 = 0; s2 < d; s2++) {
                    for (int b = 0; b < DR; b++) {
                        int idx = a * d * d * DR + s1 * d * DR + s2 * DR + b;
                        auto val = mpo_cpu[i][a][s1][s2][b];
                        h_W[idx] = to_hip_complex(val);
                    }
                }
            }
        }
        mpo_tensors[i].copy_from_host(h_W);
    }
}

void DMRG_GPU::initialize_environments() {
    // Initialize trivial boundary environments
    // Left: L[0] = identity
    left_envs[0].resize(1);
    std::vector<Complex> h_L0 = {make_complex(1.0, 0.0)};
    left_envs[0].copy_from_host(h_L0);

    // Right: R[L] = identity
    right_envs[L].resize(1);
    std::vector<Complex> h_RL = {make_complex(1.0, 0.0)};
    right_envs[L].copy_from_host(h_RL);

    // TODO: Build initial L and R environments by contracting from boundaries
    // For now, allocate with identity approximation
    for (int i = 1; i <= L; i++) {
        left_envs[i].resize(10);  // Placeholder
        right_envs[L - i].resize(10);
    }
}

double DMRG_GPU::run(const Tensor5D<std::complex<double>>& mpo_cpu) {
    auto t_start = Clock::now();

    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "DMRG-GPU (Variant: " << (variant == DMRGVariant::PDMRG ? "PDMRG" : "PDMRG2") << ")\n";
    std::cout << "========================================\n";
    std::cout << "L = " << L << ", max_bond = " << max_bond << ", sweeps = " << n_sweeps << "\n";
    std::cout << "Tolerance = " << tol << "\n\n";

    // Initialize
    initialize_mps();
    initialize_mpo(mpo_cpu);
    initialize_environments();

    double energy = 0.0;
    double prev_energy = 1e10;

    // Perform sweeps
    for (int sweep = 0; sweep < n_sweeps; sweep++) {
        bool left_to_right = (sweep % 2 == 0);

        energy = perform_sweep(left_to_right);

        double delta_E = std::abs(energy - prev_energy);
        std::cout << "Sweep " << std::setw(2) << sweep
                  << " | E = " << std::fixed << std::setprecision(12) << energy
                  << " | ΔE = " << std::scientific << std::setprecision(2) << delta_E
                  << "\n";

        if (delta_E < tol) {
            std::cout << "\nConverged!\n";
            break;
        }

        prev_energy = energy;
    }

    auto t_end = Clock::now();
    time_total = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "\nFinal energy: " << std::fixed << std::setprecision(12) << energy << "\n";
    print_timing_stats();

    return energy;
}

double DMRG_GPU::perform_sweep(bool left_to_right) {
    double energy = 0.0;

    if (left_to_right) {
        for (int i = 0; i < L - 1; i++) {
            hipStream_t stream = stream_mgr.get_stream(i % stream_mgr.num_streams());
            energy = optimize_two_site(i, i + 1, stream);
        }
    } else {
        for (int i = L - 2; i >= 0; i--) {
            hipStream_t stream = stream_mgr.get_stream(i % stream_mgr.num_streams());
            energy = optimize_two_site(i, i + 1, stream);
        }
    }

    stream_mgr.sync_all();
    return energy;
}

double DMRG_GPU::optimize_two_site(int site1, int site2, hipStream_t stream) {
    // Placeholder implementation
    // TODO: Complete the 2-site optimization
    //
    // Steps:
    //   1. Build effective Hamiltonian H_eff
    //   2. Solve eigenproblem using Lanczos or BlockDavidson
    //   3. Reshape eigenvector to 2-site tensor
    //   4. SVD to split tensor
    //   5. Update MPS tensors
    //   6. Update environments

    return 0.0;  // Placeholder energy
}

void DMRG_GPU::print_timing_stats() const {
    std::cout << "\n========================================\n";
    std::cout << "Timing Statistics\n";
    std::cout << "========================================\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Total time:        " << time_total << " s\n";
    std::cout << "Tensor contraction: " << time_contraction << " s ("
              << (time_contraction / time_total * 100) << "%)\n";
    std::cout << "Eigensolver:       " << time_eigensolver << " s ("
              << (time_eigensolver / time_total * 100) << "%)\n";
    std::cout << "SVD:               " << time_svd << " s ("
              << (time_svd / time_total * 100) << "%)\n";
    std::cout << "========================================\n";
}

// Main benchmark program
int main(int argc, char** argv) {
    std::cout << "DMRG-GPU Benchmark\n";
    std::cout << "==================\n\n";

    // Parse command line
    DMRGVariant variant = DMRGVariant::PDMRG;
    if (argc > 1 && std::string(argv[1]) == "--pdmrg2") {
        variant = DMRGVariant::PDMRG2;
    }

    // Parameters
    int L = 12;
    int max_bond = 100;
    int n_sweeps = 5;

    // Build Heisenberg MPO (using CPU function)
    // TODO: Link with build_heisenberg_mpo from heisenberg_mpo.cpp
    Tensor5D<std::complex<double>> mpo_placeholder(L);  // Placeholder

    // Run DMRG
    try {
        DMRG_GPU dmrg(L, max_bond, n_sweeps, variant);
        double energy = dmrg.run(mpo_placeholder);

        std::cout << "\nBenchmark complete!\n";
        std::cout << "Final energy: " << std::fixed << std::setprecision(12) << energy << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
