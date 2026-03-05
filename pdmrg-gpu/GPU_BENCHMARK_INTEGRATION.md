# GPU Benchmark Integration Guide

## Loading Serialized MPS/MPO Data into GPU DMRG

This guide shows how to integrate the serialized MPS/MPO data into your GPU DMRG implementation for fair comparison with CPU gold standard.

---

## Quick Start

### 1. Load Initial State and Hamiltonian

```cpp
#include "mps_mpo_loader.hpp"

// Load data files (same ones used in CPU benchmark)
auto mps_initial = MPSLoader::load("../benchmarks/benchmark_data/heisenberg_L12_chi10_mps.bin");
auto mpo = MPOLoader::load("../benchmarks/benchmark_data/heisenberg_L12_mpo.bin");

// Extract parameters
int L = mps_initial.size();          // Chain length
int d = mps_initial[0].d;            // Physical dimension
int chi_init = mps_initial[0].D_right;  // Initial bond dimension
```

### 2. Convert to Your GPU Data Structures

The loaded tensors are in `std::vector<std::complex<double>>` format. You'll need to convert them to your GPU tensor format:

```cpp
// Example: Convert MPS to your GPU format
for (int site = 0; site < L; ++site) {
    const auto& mps_tensor = mps_initial[site];

    // Copy to GPU
    // Your code here - example:
    // hip_malloc_and_copy(d_mps[site], mps_tensor.data.data(),
    //                     mps_tensor.total_elements() * sizeof(Complex));

    // Or convert to your tensor class:
    // MyGPUTensor gpu_tensor(mps_tensor.D_left, mps_tensor.d, mps_tensor.D_right);
    // gpu_tensor.copy_from_host(mps_tensor.data.data());
}

// Similar for MPO
for (int site = 0; site < L; ++site) {
    const auto& mpo_tensor = mpo[site];
    // Convert to your GPU MPO format
}
```

### 3. Run GPU DMRG

```cpp
// Run your GPU DMRG implementation
double final_energy = run_gpu_dmrg(
    gpu_mps,           // Initialized from loaded data
    gpu_mpo,          // Loaded Hamiltonian
    chi_max,          // Max bond dimension (e.g., 100)
    max_sweeps,       // Max sweeps (e.g., 20)
    tolerance         // Convergence tolerance (1e-10)
);
```

### 4. Compare Results

The CPU gold standard results will be in `cpu_gold_standard_results.json`. Compare:
- **Final energy**: Should match within ~1e-10
- **Convergence**: Number of sweeps should be similar
- **Performance**: GPU should be faster!

---

## Data Format Reference

### MPS Tensor Layout

Each MPS tensor has shape `(D_left, d, D_right)` and is stored in C-contiguous order:

```cpp
// Access element (i, j, k):
Complex value = mps_tensor.data[i * d * D_right + j * D_right + k];

// Or use the operator():
Complex value = mps_tensor(i, j, k);
```

**Boundary conditions:**
- First site: `D_left = 1`
- Last site: `D_right = 1`

### MPO Tensor Layout

Each MPO tensor has shape `(D_mpo_left, d, d, D_mpo_right)`:

```cpp
// Access element (i, j, k, l):
Complex value = mpo_tensor.data[i * d * d * D_mpo_right +
                                 j * d * D_mpo_right +
                                 k * D_mpo_right + l];

// Or use the operator():
Complex value = mpo_tensor(i, j, k, l);
```

**Index convention:**
- `i`: Left MPO bond
- `j`: Bra (output) physical index
- `k`: Ket (input) physical index
- `l`: Right MPO bond

---

## Integration Template

Here's a complete integration template:

```cpp
#include "mps_mpo_loader.hpp"
#include <iostream>
#include <chrono>

// Your GPU DMRG function signature
struct GPUDMRGResult {
    double energy;
    int sweeps;
    double wall_time_s;
    bool converged;
};

GPUDMRGResult your_gpu_dmrg_function(
    /* Your GPU tensors */,
    int chi_max,
    int max_sweeps,
    double tol
);

int main(int argc, char** argv) {
    // 1. Load data
    std::string mps_file = argv[1];
    std::string mpo_file = argv[2];

    auto mps_initial = MPSLoader::load(mps_file);
    auto mpo = MPOLoader::load(mpo_file);

    int L = mps_initial.size();
    int d = mps_initial[0].d;

    std::cout << "Loaded system: L=" << L << ", d=" << d << "\n";

    // 2. Convert to GPU format
    // TODO: Your conversion code here
    // YourGPUMPS gpu_mps = convert_to_gpu(mps_initial);
    // YourGPUMPO gpu_mpo = convert_to_gpu(mpo);

    // 3. Run GPU DMRG
    int chi_max = 100;      // Or from argv
    int max_sweeps = 20;    // Or from argv
    double tol = 1e-10;

    auto t_start = std::chrono::high_resolution_clock::now();

    GPUDMRGResult result = your_gpu_dmrg_function(
        /* gpu_mps, gpu_mpo, */ chi_max, max_sweeps, tol
    );

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    // 4. Report results
    std::cout << "\n=== GPU DMRG Results ===\n";
    std::cout << "Final energy: " << std::fixed << std::setprecision(12)
              << result.energy << " Ha\n";
    std::cout << "Sweeps:       " << result.sweeps << "\n";
    std::cout << "Wall time:    " << std::setprecision(3)
              << elapsed << " s\n";
    std::cout << "Converged:    " << (result.converged ? "✓" : "✗") << "\n";

    return 0;
}
```

---

## Test Cases Available

All with seed=42 for reproducibility:

### Heisenberg (d=2, real Hamiltonian)

| Case | L | χ_init | χ_max | Files |
|------|---|--------|-------|-------|
| Small | 12 | 10 | 100 | `heisenberg_L12_chi10_mps.bin`, `heisenberg_L12_mpo.bin` |
| Medium | 20 | 10 | 100 | `heisenberg_L20_chi10_mps.bin`, `heisenberg_L20_mpo.bin` |

### Josephson (d=5, complex Hamiltonian)

| Case | L | χ_init | χ_max | Files |
|------|---|--------|-------|-------|
| Small | 8 | 10 | 50 | `josephson_L8_n2_chi10_mps.bin`, `josephson_L8_n2_mpo.bin` |
| Medium | 12 | 10 | 50 | `josephson_L12_n2_chi10_mps.bin`, `josephson_L12_n2_mpo.bin` |

---

## Building

Add to your `CMakeLists.txt`:

```cmake
add_executable(gpu_benchmark_with_data
    src/gpu_benchmark_with_data.cpp
)

target_compile_options(gpu_benchmark_with_data PRIVATE -O3)

# Link GPU libraries as needed
target_link_libraries(gpu_benchmark_with_data
    hip::device
    roc::rocblas
    roc::rocsolver
    ${HIPTENSOR_LIBRARY}
    lapack
)
```

Then build:
```bash
cd pdmrg-gpu/build
cmake .. && make gpu_benchmark_with_data
```

---

## Running Benchmarks

```bash
# Run GPU benchmark
./gpu_benchmark_with_data \
    ../benchmarks/benchmark_data/heisenberg_L12_chi10_mps.bin \
    ../benchmarks/benchmark_data/heisenberg_L12_mpo.bin \
    100 20  # chi_max max_sweeps

# Compare with CPU gold standard
python3 ../benchmarks/compare_results.py \
    --cpu cpu_gold_standard_results.json \
    --gpu gpu_result_12_d2.json
```

---

## Validation Checklist

✅ **Data Loading**
- [ ] MPS tensors load correctly
- [ ] MPO tensors load correctly
- [ ] Tensor shapes match expected dimensions
- [ ] Complex values preserved accurately

✅ **Conversion**
- [ ] Host→GPU copy works
- [ ] Data layout matches your GPU code expectations
- [ ] Boundary conditions (D_left=1, D_right=1) handled

✅ **DMRG Execution**
- [ ] DMRG runs without errors
- [ ] Convergence within max_sweeps
- [ ] Energy decreases monotonically

✅ **Result Comparison**
- [ ] Final energy matches CPU within tolerance (~1e-10)
- [ ] Convergence behavior is similar
- [ ] GPU shows performance improvement

---

## Troubleshooting

**Q: Energies don't match**

Check:
- Data type (complex128 vs complex64)
- Index ordering (row-major C vs column-major Fortran)
- Initial state normalization
- Hamiltonian sign conventions

**Q: GPU code crashes on load**

Check:
- File paths are correct
- Sufficient GPU memory
- Tensor dimensions don't exceed GPU limits

**Q: Performance worse than expected**

Check:
- GPU memory transfers minimized
- Tensor contractions using hipTensor
- No unnecessary host↔device copies in inner loop

---

## Next Steps

1. **Integrate conversion**: Adapt the template to your GPU tensor format
2. **Test small case**: Start with Heisenberg L=12
3. **Validate energy**: Compare with CPU gold standard
4. **Scale up**: Test medium and large cases
5. **Optimize**: Profile and improve GPU performance

For questions, see `REPRODUCIBLE_BENCHMARKS.md` or `QUICKSTART.md`.
