// PDMRG GPU Benchmark with Loaded MPS/MPO Data
// ==============================================
//
// This is the production benchmark executable for comparing GPU vs CPU DMRG.
//
// Features:
// 1. Loads initial MPS and MPO from binary files (same as CPU benchmarks)
// 2. Single-stream warm-up phase (configurable sweeps, default=3)
// 3. Multi-stream parallel DMRG phase
// 4. Reports final energy for comparison with CPU gold standard
//
// Usage:
//   ./pdmrg_benchmark_loaded <mps_file> <mpo_file> <chi_max> <max_sweeps> [warmup_sweeps] [num_streams]
//
// Example:
//   ./pdmrg_benchmark_loaded \
//       ../benchmarks/benchmark_data/heisenberg_L12_chi10_mps.bin \
//       ../benchmarks/benchmark_data/heisenberg_L12_mpo.bin \
//       100 20 3 1

#include "../include/mps_mpo_loader.hpp"
#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <complex>
#include <cmath>
#include <chrono>
#include <string>
#include <sstream>
#include <map>
#include <algorithm>

using Complex = std::complex<double>;

#define HIP_CHECK(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ \
                  << ": " << hipGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

// ============================================================================
// Timer utility
// ============================================================================
struct Timer {
    std::chrono::high_resolution_clock::time_point start;
    void tic() { start = std::chrono::high_resolution_clock::now(); }
    double toc() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(end - start).count();
    }
};

// ============================================================================
// GPU Tensor Structure (simplified - you'll adapt to your actual format)
// ============================================================================
struct GPUTensor3D {
    int D_left, d, D_right;
    hipDoubleComplex* data;
    size_t size;

    GPUTensor3D(int dl, int d_, int dr)
        : D_left(dl), d(d_), D_right(dr), size(dl * d_ * dr) {
        HIP_CHECK(hipMalloc(&data, size * sizeof(hipDoubleComplex)));
    }

    ~GPUTensor3D() {
        if (data) hipFree(data);
    }

    void copy_from_host(const std::vector<Complex>& host_data) {
        HIP_CHECK(hipMemcpy(data, host_data.data(),
                           size * sizeof(hipDoubleComplex),
                           hipMemcpyHostToDevice));
    }

    void copy_to_host(std::vector<Complex>& host_data) const {
        host_data.resize(size);
        HIP_CHECK(hipMemcpy(host_data.data(), data,
                           size * sizeof(hipDoubleComplex),
                           hipMemcpyDeviceToHost));
    }
};

struct GPUTensor4D {
    int D_mpo_left, d_bra, d_ket, D_mpo_right;
    hipDoubleComplex* data;
    size_t size;

    GPUTensor4D(int dl, int db, int dk, int dr)
        : D_mpo_left(dl), d_bra(db), d_ket(dk), D_mpo_right(dr),
          size(dl * db * dk * dr) {
        HIP_CHECK(hipMalloc(&data, size * sizeof(hipDoubleComplex)));
    }

    ~GPUTensor4D() {
        if (data) hipFree(data);
    }

    void copy_from_host(const std::vector<Complex>& host_data) {
        HIP_CHECK(hipMemcpy(data, host_data.data(),
                           size * sizeof(hipDoubleComplex),
                           hipMemcpyHostToDevice));
    }
};

// ============================================================================
// Convert loaded MPS/MPO to GPU format
// ============================================================================
std::vector<GPUTensor3D*> convert_mps_to_gpu(const std::vector<MPSTensor>& mps_host) {
    std::vector<GPUTensor3D*> mps_gpu;
    mps_gpu.reserve(mps_host.size());

    for (const auto& tensor : mps_host) {
        auto* gpu_tensor = new GPUTensor3D(
            tensor.D_left, tensor.d, tensor.D_right
        );
        gpu_tensor->copy_from_host(tensor.data);
        mps_gpu.push_back(gpu_tensor);
    }

    return mps_gpu;
}

std::vector<GPUTensor4D*> convert_mpo_to_gpu(const std::vector<MPOTensor>& mpo_host) {
    std::vector<GPUTensor4D*> mpo_gpu;
    mpo_gpu.reserve(mpo_host.size());

    for (const auto& tensor : mpo_host) {
        auto* gpu_tensor = new GPUTensor4D(
            tensor.D_mpo_left, tensor.d_bra, tensor.d_ket, tensor.D_mpo_right
        );
        gpu_tensor->copy_from_host(tensor.data);
        mpo_gpu.push_back(gpu_tensor);
    }

    return mpo_gpu;
}

void free_gpu_mps(std::vector<GPUTensor3D*>& mps) {
    for (auto* tensor : mps) {
        delete tensor;
    }
    mps.clear();
}

void free_gpu_mpo(std::vector<GPUTensor4D*>& mpo) {
    for (auto* tensor : mpo) {
        delete tensor;
    }
    mpo.clear();
}

// ============================================================================
// PLACEHOLDER: Single-stream warm-up phase
// ============================================================================
struct WarmupResult {
    double final_energy;
    int sweeps_completed;
    double wall_time_s;
};

WarmupResult run_single_stream_warmup(
    std::vector<GPUTensor3D*>& mps_gpu,
    const std::vector<GPUTensor4D*>& mpo_gpu,
    int warmup_sweeps
) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "  WARM-UP PHASE (Single Stream)\n";
    std::cout << std::string(80, '=') << "\n\n";

    std::cout << "Running " << warmup_sweeps << " warm-up sweeps (single stream)...\n";

    Timer timer;
    timer.tic();

    // TODO: Integrate with your actual PDMRG GPU implementation
    // This is where you would call your single-stream DMRG code
    //
    // For now, this is a placeholder that shows the interface:
    // - Takes GPU MPS and MPO tensors
    // - Runs specified number of sweeps with single stream
    // - Returns energy after warm-up
    //
    // Replace this with your actual implementation:
    //   double energy = run_pdmrg_single_stream(mps_gpu, mpo_gpu, warmup_sweeps, tol);

    std::cout << "⚠️  PLACEHOLDER: Actual PDMRG warm-up not yet integrated\n";
    std::cout << "    Replace with your single-stream DMRG implementation\n\n";

    // Placeholder: compute dummy energy
    double dummy_energy = -5.0;  // Will be replaced with actual DMRG

    double elapsed = timer.toc();

    std::cout << "Warm-up completed:\n";
    std::cout << "  Sweeps: " << warmup_sweeps << "\n";
    std::cout << "  Time:   " << std::fixed << std::setprecision(3) << elapsed << " s\n";
    std::cout << "  Energy: " << std::setprecision(10) << dummy_energy << " Ha\n\n";

    return WarmupResult{
        .final_energy = dummy_energy,
        .sweeps_completed = warmup_sweeps,
        .wall_time_s = elapsed
    };
}

// ============================================================================
// PLACEHOLDER: Multi-stream parallel DMRG
// ============================================================================
struct DMRGResult {
    double final_energy;
    int total_sweeps;
    double wall_time_s;
    bool converged;
};

DMRGResult run_multistream_dmrg(
    std::vector<GPUTensor3D*>& mps_gpu,
    const std::vector<GPUTensor4D*>& mpo_gpu,
    int chi_max,
    int max_sweeps,
    int num_streams,
    double tolerance = 1e-10
) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "  PARALLEL DMRG PHASE (" << num_streams << " Streams)\n";
    std::cout << std::string(80, '=') << "\n\n";

    std::cout << "Running multi-stream DMRG...\n";
    std::cout << "  Max bond dim: " << chi_max << "\n";
    std::cout << "  Max sweeps:   " << max_sweeps << "\n";
    std::cout << "  Num streams:  " << num_streams << "\n";
    std::cout << "  Tolerance:    " << tolerance << "\n\n";

    Timer timer;
    timer.tic();

    // TODO: Integrate with your actual multi-stream PDMRG GPU implementation
    // This is where you would call your parallel DMRG code
    //
    // Replace this with your actual implementation:
    //   double energy = run_pdmrg_multistream(mps_gpu, mpo_gpu, chi_max,
    //                                          max_sweeps, num_streams, tol);

    std::cout << "⚠️  PLACEHOLDER: Actual PDMRG multi-stream not yet integrated\n";
    std::cout << "    Replace with your multi-stream DMRG implementation\n\n";

    // Placeholder: compute dummy energy
    double dummy_energy = -5.142;  // Will be replaced with actual DMRG
    int dummy_sweeps = 5;
    bool dummy_converged = true;

    double elapsed = timer.toc();

    std::cout << "DMRG completed:\n";
    std::cout << "  Total sweeps: " << dummy_sweeps << "\n";
    std::cout << "  Time:         " << std::fixed << std::setprecision(3) << elapsed << " s\n";
    std::cout << "  Final energy: " << std::setprecision(12) << dummy_energy << " Ha\n";
    std::cout << "  Converged:    " << (dummy_converged ? "✓ Yes" : "✗ No") << "\n\n";

    return DMRGResult{
        .final_energy = dummy_energy,
        .total_sweeps = dummy_sweeps,
        .wall_time_s = elapsed,
        .converged = dummy_converged
    };
}

// ============================================================================
// Main benchmark function
// ============================================================================
int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <mps_file> <mpo_file> <chi_max> <max_sweeps> "
                  << "[warmup_sweeps=3] [num_streams=1]\n\n";
        std::cerr << "Example (Heisenberg small):\n";
        std::cerr << "  " << argv[0] << " \\\n";
        std::cerr << "    ../benchmarks/benchmark_data/heisenberg_L12_chi10_mps.bin \\\n";
        std::cerr << "    ../benchmarks/benchmark_data/heisenberg_L12_mpo.bin \\\n";
        std::cerr << "    100 20 3 1\n\n";
        std::cerr << "Example (Josephson small):\n";
        std::cerr << "  " << argv[0] << " \\\n";
        std::cerr << "    ../benchmarks/benchmark_data/josephson_L8_n2_chi10_mps.bin \\\n";
        std::cerr << "    ../benchmarks/benchmark_data/josephson_L8_n2_mpo.bin \\\n";
        std::cerr << "    50 20 3 1\n";
        return 1;
    }

    std::string mps_file = argv[1];
    std::string mpo_file = argv[2];
    int chi_max = std::stoi(argv[3]);
    int max_sweeps = std::stoi(argv[4]);
    int warmup_sweeps = (argc > 5) ? std::stoi(argv[5]) : 3;  // Default: 3
    int num_streams = (argc > 6) ? std::stoi(argv[6]) : 1;    // Default: 1

    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "  PDMRG GPU Benchmark with Loaded Data\n";
    std::cout << std::string(80, '=') << "\n\n";

    std::cout << "Configuration:\n";
    std::cout << "  MPS file:      " << mps_file << "\n";
    std::cout << "  MPO file:      " << mpo_file << "\n";
    std::cout << "  Max bond dim:  " << chi_max << "\n";
    std::cout << "  Max sweeps:    " << max_sweeps << "\n";
    std::cout << "  Warm-up:       " << warmup_sweeps << " sweeps (single stream)\n";
    std::cout << "  Parallel:      " << num_streams << " stream(s)\n\n";

    // ========================================================================
    // Phase 1: Load MPS and MPO from files
    // ========================================================================
    std::cout << std::string(80, '=') << "\n";
    std::cout << "  PHASE 1: Loading Initial State from Files\n";
    std::cout << std::string(80, '=') << "\n\n";

    std::vector<MPSTensor> mps_host;
    std::vector<MPOTensor> mpo_host;

    try {
        std::cout << "Loading MPS...\n";
        mps_host = MPSLoader::load(mps_file);

        std::cout << "\nLoading MPO...\n";
        mpo_host = MPOLoader::load(mpo_file);
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Error loading files: " << e.what() << "\n";
        return 1;
    }

    int L = mps_host.size();
    int d = mps_host[0].d;

    std::cout << "\n✓ Data loaded successfully\n";
    std::cout << "  Chain length:  L = " << L << "\n";
    std::cout << "  Physical dim:  d = " << d << "\n";

    // ========================================================================
    // Phase 2: Convert to GPU format
    // ========================================================================
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "  PHASE 2: Converting to GPU Format\n";
    std::cout << std::string(80, '=') << "\n\n";

    auto mps_gpu = convert_mps_to_gpu(mps_host);
    auto mpo_gpu = convert_mpo_to_gpu(mpo_host);

    std::cout << "✓ Tensors copied to GPU memory\n";
    std::cout << "  MPS tensors: " << mps_gpu.size() << "\n";
    std::cout << "  MPO tensors: " << mpo_gpu.size() << "\n";

    // ========================================================================
    // Phase 3: Single-stream warm-up
    // ========================================================================
    Timer total_timer;
    total_timer.tic();

    auto warmup_result = run_single_stream_warmup(mps_gpu, mpo_gpu, warmup_sweeps);

    // ========================================================================
    // Phase 4: Multi-stream parallel DMRG
    // ========================================================================
    auto dmrg_result = run_multistream_dmrg(
        mps_gpu, mpo_gpu, chi_max, max_sweeps, num_streams
    );

    double total_time = total_timer.toc();

    // ========================================================================
    // Phase 5: Report results
    // ========================================================================
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "  FINAL RESULTS\n";
    std::cout << std::string(80, '=') << "\n\n";

    std::cout << std::fixed;
    std::cout << "Warm-up phase:\n";
    std::cout << "  Sweeps:  " << warmup_result.sweeps_completed << "\n";
    std::cout << "  Time:    " << std::setprecision(3) << warmup_result.wall_time_s << " s\n";
    std::cout << "  Energy:  " << std::setprecision(12) << warmup_result.final_energy << " Ha\n\n";

    std::cout << "Main DMRG phase:\n";
    std::cout << "  Sweeps:  " << dmrg_result.total_sweeps << "\n";
    std::cout << "  Time:    " << std::setprecision(3) << dmrg_result.wall_time_s << " s\n";
    std::cout << "  Energy:  " << std::setprecision(12) << dmrg_result.final_energy << " Ha\n";
    std::cout << "  Status:  " << (dmrg_result.converged ? "✓ Converged" : "✗ Not converged") << "\n\n";

    std::cout << "Total time: " << std::setprecision(3) << total_time << " s\n\n";

    std::cout << std::string(80, '=') << "\n";
    std::cout << "  Compare with CPU Gold Standard\n";
    std::cout << std::string(80, '=') << "\n\n";

    // CPU gold standard energies
    std::map<std::string, double> gold_standard = {
        {"heisenberg_L12", -5.1420906328},
        {"heisenberg_L20", -8.6824733344},
        {"josephson_L8", -2.8438010431},
        {"josephson_L12", -4.5070608947}
    };

    // Try to identify which benchmark this is
    std::string benchmark_name = "unknown";
    if (mps_file.find("heisenberg_L12") != std::string::npos) {
        benchmark_name = "heisenberg_L12";
    } else if (mps_file.find("heisenberg_L20") != std::string::npos) {
        benchmark_name = "heisenberg_L20";
    } else if (mps_file.find("josephson_L8") != std::string::npos) {
        benchmark_name = "josephson_L8";
    } else if (mps_file.find("josephson_L12") != std::string::npos) {
        benchmark_name = "josephson_L12";
    }

    if (gold_standard.count(benchmark_name)) {
        double E_gold = gold_standard[benchmark_name];
        double error = std::abs(dmrg_result.final_energy - E_gold);

        std::cout << "Benchmark:    " << benchmark_name << "\n";
        std::cout << "CPU energy:   " << std::setprecision(12) << E_gold << " Ha\n";
        std::cout << "GPU energy:   " << std::setprecision(12) << dmrg_result.final_energy << " Ha\n";
        std::cout << "Error:        " << std::scientific << std::setprecision(2) << error << "\n";
        std::cout << "Tolerance:    1.00e-10\n";
        std::cout << "Status:       " << (error < 1e-10 ? "✅ PASS" : "❌ FAIL") << "\n\n";
    } else {
        std::cout << "⚠️  Unknown benchmark - cannot compare with gold standard\n\n";
    }

    // Cleanup
    free_gpu_mps(mps_gpu);
    free_gpu_mpo(mpo_gpu);

    std::cout << "✓ Benchmark complete\n\n";

    return 0;
}
