# PDMRG Integration Guide
## Integrating Loaded MPS/MPO with PDMRG_GPU

**Date:** 2026-03-05
**Status:** Integration Ready

---

## Overview

This guide explains how to integrate the loaded MPS/MPO data from `pdmrg_benchmark_loaded.cpp` with the PDMRG_GPU implementation in `pdmrg_gpu.cpp`.

## Current State

### What Works ✅
- **MPS/MPO Loading**: Binary files load correctly into host memory
- **GPU Conversion**: Data converts to GPU format (hipDoubleComplex*)
- **Infrastructure**: Benchmark executable framework complete
- **Validation**: Gold standard energies available for comparison
- **LoadedMPO class**: Created in `include/loaded_mpo.hpp`

### What Needs Integration ⚠️
- Replace placeholder functions in `pdmrg_benchmark_loaded.cpp`:
  - `run_single_stream_warmup()` (line ~169)
  - `run_multistream_dmrg()` (line ~224)

---

## Integration Approaches

### Approach A: Refactor PDMRG_GPU into Headers (Recommended for Production)

**Pros:**
- Clean separation of interface and implementation
- Reusable across multiple executables
- Follows best practices

**Cons:**
- Requires significant refactoring of `pdmrg_gpu.cpp`
- Need to extract 6 classes into headers
- More work upfront

**Steps:**
1. Create headers for each class:
   - `include/hip_tensor_contractor.hpp`
   - `include/mpo_base.hpp`
   - `include/environments.hpp`
   - `include/lanczos_eigensolver.hpp`
   - `include/pdmrg_gpu.hpp`

2. Move implementations to source files
3. Update CMakeLists.txt to compile as library
4. Link against library in `pdmrg_benchmark_loaded`

**Estimated effort:** 2-4 hours

---

### Approach B: Inline Integration (Quick Prototype)

**Pros:**
- Fast to implement
- Self-contained executable
- Follows pattern of other executables in project

**Cons:**
- Code duplication
- Large file size (~2000+ lines)
- Harder to maintain

**Steps:**
1. Copy necessary classes from `pdmrg_gpu.cpp` to `pdmrg_benchmark_loaded.cpp`:
   - HipTensorContractor (lines 84-157)
   - MPOBase (lines 159-167)
   - Environments (lines 402-627)
   - LanczosEigensolver (lines 629-861)
   - PDMRG_GPU (lines 867-1439)

2. Modify PDMRG_GPU constructor to accept loaded MPS
3. Use LoadedMPO for MPO interface
4. Replace placeholder functions with PDMRG_GPU calls

**Estimated effort:** 1-2 hours

---

### Approach C: Hybrid (Recommended for MVP)

**Pros:**
- Balance of speed and maintainability
- Minimal refactoring
- Can evolve to Approach A later

**Cons:**
- Still some code duplication
- Two files to maintain temporarily

**Steps:**
1. Create `include/pdmrg_wrapper.hpp` with minimal interface
2. Create `src/pdmrg_wrapper.cpp` with adapter code
3. Update CMakeLists.txt to compile wrapper with benchmark
4. Use wrapper in `pdmrg_benchmark_loaded.cpp`

**Estimated effort:** 1-2 hours

---

## Detailed Implementation: Approach C (Hybrid)

### Step 1: Create Wrapper Interface

**File:** `include/pdmrg_wrapper.hpp`

```cpp
#pragma once

#include "mps_mpo_loader.hpp"
#include "loaded_mpo.hpp"
#include <vector>
#include <string>

// Simplified interface for running PDMRG with loaded data
namespace pdmrg {

struct DMRGResult {
    double final_energy;
    int total_sweeps;
    double wall_time_s;
    bool converged;
};

// Run single-stream DMRG (for warm-up phase)
DMRGResult run_single_stream(
    const std::vector<MPSTensor>& mps_host,
    const std::vector<MPOTensor>& mpo_host,
    int max_sweeps,
    double tolerance = 1e-12,
    bool verbose = false
);

// Run multi-stream parallel DMRG
DMRGResult run_multistream(
    const std::vector<MPSTensor>& mps_host,
    const std::vector<MPOTensor>& mpo_host,
    int chi_max,
    int max_sweeps,
    int num_streams,
    double tolerance = 1e-12,
    bool verbose = false
);

} // namespace pdmrg
```

### Step 2: Create Wrapper Implementation

**File:** `src/pdmrg_wrapper.cpp`

This file would include the necessary classes from `pdmrg_gpu.cpp` and implement the wrapper functions. Due to length, see `pdmrg_wrapper_template.cpp` for full implementation.

### Step 3: Update CMakeLists.txt

```cmake
# PDMRG Benchmark with Loaded Data
add_executable(pdmrg_benchmark_loaded
    src/pdmrg_benchmark_loaded.cpp
    src/pdmrg_wrapper.cpp  # Add wrapper
)
set_source_files_properties(src/pdmrg_benchmark_loaded.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(src/pdmrg_wrapper.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(pdmrg_benchmark_loaded PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=${GPU_TARGETS}>
    -O3
)
target_link_libraries(pdmrg_benchmark_loaded
    hip::device
    roc::rocblas
    roc::rocsolver
    ${HIPTENSOR_LIBRARY}
    lapack
)
```

### Step 4: Update pdmrg_benchmark_loaded.cpp

Replace placeholder functions:

```cpp
#include "pdmrg_wrapper.hpp"

WarmupResult run_single_stream_warmup(
    std::vector<GPUTensor3D*>& mps_gpu,
    const std::vector<GPUTensor4D*>& mpo_gpu,
    int warmup_sweeps
) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "  WARM-UP PHASE (Single Stream)\n";
    std::cout << std::string(80, '=') << "\n\n";

    Timer timer;
    timer.tic();

    // Convert GPU tensors back to host format for PDMRG
    // (PDMRG wrapper handles GPU allocation internally)
    std::vector<MPSTensor> mps_host;
    std::vector<MPOTensor> mpo_host;

    // Copy from GPU back to host (these are already loaded earlier)
    // ... conversion code ...

    // Run PDMRG single-stream
    auto result = pdmrg::run_single_stream(
        mps_host, mpo_host, warmup_sweeps, 1e-12, true
    );

    double elapsed = timer.toc();

    std::cout << "Warm-up completed:\n";
    std::cout << "  Sweeps: " << result.total_sweeps << "\n";
    std::cout << "  Time:   " << std::fixed << std::setprecision(3) << elapsed << " s\n";
    std::cout << "  Energy: " << std::setprecision(10) << result.final_energy << " Ha\n\n";

    return WarmupResult{
        .final_energy = result.final_energy,
        .sweeps_completed = result.total_sweeps,
        .wall_time_s = elapsed
    };
}

DMRGResult run_multistream_dmrg(
    std::vector<GPUTensor3D*>& mps_gpu,
    const std::vector<GPUTensor4D*>& mpo_gpu,
    int chi_max,
    int max_sweeps,
    int num_streams,
    double tolerance
) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "  PARALLEL DMRG PHASE (" << num_streams << " Streams)\n";
    std::cout << std::string(80, '=') << "\n\n";

    Timer timer;
    timer.tic();

    // Convert GPU tensors to host format
    std::vector<MPSTensor> mps_host;
    std::vector<MPOTensor> mpo_host;
    // ... conversion code ...

    // Run PDMRG multi-stream
    auto result = pdmrg::run_multistream(
        mps_host, mpo_host, chi_max, max_sweeps, num_streams, tolerance, true
    );

    double elapsed = timer.toc();

    std::cout << "DMRG completed:\n";
    std::cout << "  Total sweeps: " << result.total_sweeps << "\n";
    std::cout << "  Time:         " << std::fixed << std::setprecision(3) << elapsed << " s\n";
    std::cout << "  Final energy: " << std::setprecision(12) << result.final_energy << " Ha\n";
    std::cout << "  Converged:    " << (result.converged ? "✓ Yes" : "✗ No") << "\n\n";

    return DMRGResult{
        .final_energy = result.final_energy,
        .total_sweeps = result.total_sweeps,
        .wall_time_s = elapsed,
        .converged = result.converged
    };
}
```

---

## Data Flow

```
1. Load from files:
   MPSLoader::load() → std::vector<MPSTensor> (host)
   MPOLoader::load() → std::vector<MPOTensor> (host)

2. Pass to PDMRG wrapper:
   pdmrg::run_single_stream(mps_host, mpo_host, ...)

3. Inside wrapper:
   - Create LoadedMPO from mpo_host
   - Convert mps_host to GPU (hipMalloc + hipMemcpy)
   - Initialize PDMRG_GPU with loaded data
   - Run DMRG sweeps
   - Return results

4. Validate:
   |E_GPU - E_CPU_gold| < 1e-10
```

---

## Type Compatibility Matrix

| Source | Type | Target | Type | Compatible? |
|--------|------|--------|------|-------------|
| File (loader) | `std::vector<std::complex<double>>` | PDMRG_GPU | `Complex*` (hipDoubleComplex*) | ✅ Yes (via make_complex) |
| MPSTensor | `(D_left, d, D_right)` | PDMRG d_mps | `bond_dims[i] * d * bond_dims[i+1]` | ✅ Yes (same layout) |
| MPOTensor | `(D_mpo_left, d, d, D_mpo_right)` | MPOBase | `get_mpo(site)` device pointer | ✅ Yes (via LoadedMPO) |

---

## Testing Plan

### Phase 1: Verify Data Loading
```bash
# On f1a
cd ~/dmrg-implementations/pdmrg-gpu/build

# Test that LoadedMPO works
# (create simple test executable)
./test_loaded_mpo \
    ../../benchmarks/benchmark_data/heisenberg_L12_mpo.bin
```

### Phase 2: Test Single-Stream Warm-up
```bash
./pdmrg_benchmark_loaded \
    ../../benchmarks/benchmark_data/heisenberg_L12_chi10_mps.bin \
    ../../benchmarks/benchmark_data/heisenberg_L12_mpo.bin \
    100 3 3 1

# Expected:
# - Energy close to -5.142 Ha after 3 sweeps
# - No crashes or GPU errors
```

### Phase 3: Test Multi-Stream DMRG
```bash
./pdmrg_benchmark_loaded \
    ../../benchmarks/benchmark_data/heisenberg_L12_chi10_mps.bin \
    ../../benchmarks/benchmark_data/heisenberg_L12_mpo.bin \
    100 20 3 1

# Expected:
# - Converges to E = -5.1420906328 Ha
# - |Error| < 1e-10
# - Status: ✅ PASS
```

### Phase 4: Test Josephson
```bash
./pdmrg_benchmark_loaded \
    ../../benchmarks/benchmark_data/josephson_L8_n2_chi10_mps.bin \
    ../../benchmarks/benchmark_data/josephson_L8_n2_mpo.bin \
    50 20 3 1

# Expected:
# - Converges to E = -2.8438010431 Ha
# - |Error| < 1e-10
# - Status: ✅ PASS
```

---

## Key Code Modifications Needed

### In PDMRG_GPU constructor (if using Approach A/B):

**Current:**
```cpp
// Lines 925-940 in pdmrg_gpu.cpp
// Initialize MPS with random tensors
srand(42);
d_mps.resize(L);
for (int i = 0; i < L; i++) {
    int size = bond_dims[i] * d * bond_dims[i + 1];
    std::vector<Complex> h_mps(size);
    for (int j = 0; j < size; j++) {
        double re = (double)rand() / RAND_MAX - 0.5;
        double im = complex_model ? ((double)rand() / RAND_MAX - 0.5) : 0.0;
        h_mps[j] = make_complex(re, im);
    }
    HIP_CHECK(hipMalloc(&d_mps[i], size * sizeof(Complex)));
    HIP_CHECK(hipMemcpy(d_mps[i], h_mps.data(), size * sizeof(Complex),
                       hipMemcpyHostToDevice));
}
```

**Modified (to accept loaded MPS):**
```cpp
// NEW: Constructor overload for pre-loaded MPS
PDMRG_GPU(MPOBase* mpo_in,
          const std::vector<MPSTensor>& mps_loaded,  // NEW
          int max_bond, int sweeps, int num_streams,
          const std::string& model, bool debug = false)
{
    // ... initialization ...

    // Extract bond dimensions from loaded MPS
    bond_dims.resize(L + 1);
    for (int i = 0; i <= L; i++) {
        if (i == 0) bond_dims[i] = 1;
        else if (i == L) bond_dims[i] = 1;
        else bond_dims[i] = mps_loaded[i].D_left;
    }

    // Copy loaded MPS to GPU
    d_mps.resize(L);
    for (int i = 0; i < L; i++) {
        int size = bond_dims[i] * d * bond_dims[i + 1];
        std::vector<Complex> h_mps(size);

        // Convert std::complex<double> to hipDoubleComplex
        for (int j = 0; j < size; j++) {
            h_mps[j] = make_complex(
                mps_loaded[i].data[j].real(),
                mps_loaded[i].data[j].imag()
            );
        }

        HIP_CHECK(hipMalloc(&d_mps[i], size * sizeof(Complex)));
        HIP_CHECK(hipMemcpy(d_mps[i], h_mps.data(),
                           size * sizeof(Complex),
                           hipMemcpyHostToDevice));
    }

    // Continue with canonicalization and environment initialization
    right_canonicalize_mps();
    envs = new Environments(L, d, bond_dims, rb_handle);
    envs->initialize(d_mps, *mpo);
}
```

---

## Next Steps

1. **Choose approach:** Recommend Approach C (Hybrid) for MVP
2. **Create wrapper files:** `pdmrg_wrapper.hpp` and `pdmrg_wrapper.cpp`
3. **Update build system:** Modify CMakeLists.txt
4. **Test on f1a:** Build and run benchmarks
5. **Validate results:** Confirm |Error| < 1e-10
6. **Optimize (optional):** Profile and tune performance
7. **Refactor (future):** Move to Approach A for production

---

## Success Criteria

✅ **PASS Conditions:**
- Heisenberg L=12: |E_GPU - (-5.1420906328)| < 1e-10
- Josephson L=8: |E_GPU - (-2.8438010431)| < 1e-10
- No GPU memory errors or crashes
- Convergence within reasonable sweeps (±50% of CPU)

---

## Troubleshooting

### Bond Dimension Mismatch
- **Symptom:** "Bond dimension mismatch" error
- **Cause:** Loaded MPS has different bond dims than expected
- **Fix:** Extract bond_dims from loaded MPS, don't recompute

### Memory Errors
- **Symptom:** HIP_CHECK failures, segfaults
- **Cause:** Incorrect tensor size calculations
- **Fix:** Verify size = D_left * d * D_right for each site

### Wrong Energies
- **Symptom:** Energy far from gold standard (error > 1e-5)
- **Cause:** Wrong tensor layout or data type
- **Fix:** Verify make_complex conversion and C-contiguous layout

---

**Status:** Ready for implementation on f1a! 🚀

Choose Approach C and proceed with wrapper creation.
