# Reproducible CPU/GPU Benchmark Status

## Summary

**Objective**: Run identical benchmarks on CPU and GPU using the same exact MPS and MPO data.

**Status**: ✅ System complete and operational | ⏳ CPU benchmarks running

---

## Completed

### 1. Data Serialization System ✅

**Files**:
- `serialize_mps_mpo.py` - Generate and save MPS/MPO to binary files
- `load_mps_mpo.py` - Load MPS/MPO in Python
- `include/mps_mpo_loader.hpp` - Load MPS/MPO in C++
- `src/test_mps_mpo_loader.cpp` - C++ test program

**Binary Format**:
- Complex128 (16 bytes per element)
- C-contiguous (row-major) arrays
- Header: sites, bond dimensions, physical dimensions
- Compatible with Python (NumPy) and C++ (std::complex<double>)

### 2. Benchmark Data Generated ✅

**Seed**: 42 (for reproducibility)

**Heisenberg Model** (d=2, real):
- `heisenberg_L12_chi10_mps.bin` + `heisenberg_L12_mpo.bin`
- `heisenberg_L20_chi10_mps.bin` + `heisenberg_L20_mpo.bin`
- `heisenberg_L40_chi20_mps.bin` + `heisenberg_L40_mpo.bin`

**Josephson Junction** (d=5, complex):
- `josephson_L8_n2_chi10_mps.bin` + `josephson_L8_n2_mpo.bin`
- `josephson_L12_n2_chi10_mps.bin` + `josephson_L12_n2_mpo.bin`
- `josephson_L16_n2_chi20_mps.bin` + `josephson_L16_n2_mpo.bin`

**Total**: 12 files (~1.5 MB)
**Location**: `benchmark_data/`

### 3. Verification ✅

**Python Loader**:
```bash
python3 verify_loaders.py --data-dir benchmark_data
```
Result: ✅ All tensors load correctly

**C++ Loader**:
```bash
cd ../pdmrg-gpu
./test_mps_mpo_loader mps ../benchmarks/benchmark_data/heisenberg_L12_chi10_mps.bin
./test_mps_mpo_loader mpo ../benchmarks/benchmark_data/heisenberg_L12_mpo.bin
```
Result: ✅ All tensors load correctly, norms match Python

### 4. Documentation ✅

- `QUICKSTART.md` - Quick reference guide
- `REPRODUCIBLE_BENCHMARKS.md` - Full specification
- File format documentation
- Usage examples for Python and C++

---

## In Progress

### CPU Gold Standard Benchmarks ⏳

**Script**: `cpu_gpu_benchmark.py`
**Output**: `cpu_gold_standard_results.json`
**Log**: `cpu_benchmark_run.log`

**Test Cases**:
- Heisenberg: L=12, L=20, L=40 (DMRG1 + DMRG2)
- Josephson: L=8, L=12, L=16 (DMRG1 + DMRG2)

**Monitor**:
```bash
tail -f cpu_benchmark_run.log
```

**Estimated Time**: 10-15 minutes total

---

## Next Steps

### 1. When CPU Benchmark Completes

Check results:
```bash
cat cpu_gold_standard_results.json
```

Expected output:
- Ground state energies for all cases
- Wall-clock times
- Number of sweeps to convergence
- Memory usage

### 2. Integrate with GPU Benchmarks

**Option A: Update Existing GPU Code**

Add to your GPU DMRG executable:
```cpp
#include "mps_mpo_loader.hpp"

int main(int argc, char** argv) {
    std::string data_dir = "benchmark_data/";
    std::string case_name = "heisenberg_L12";

    // Load from files instead of generating random
    auto mps = MPSLoader::load(data_dir + case_name + "_chi10_mps.bin");
    auto mpo = MPOLoader::load(data_dir + case_name + "_mpo.bin");

    // Run your GPU DMRG
    run_dmrg_gpu(mps, mpo, chi_max, sweeps);

    return 0;
}
```

**Option B: Create New Benchmark Executable**

Create `src/reproducible_benchmark_gpu.cpp` that:
1. Loads all 6 test cases from `benchmark_data/`
2. Runs GPU DMRG on each
3. Saves results to `gpu_gold_standard_results.json`
4. Format matches CPU results for easy comparison

### 3. Compare Results

Create `compare_cpu_gpu_results.py`:
```python
import json

with open('cpu_gold_standard_results.json') as f:
    cpu = json.load(f)

with open('gpu_gold_standard_results.json') as f:
    gpu = json.load(f)

# Compare energies (should match to ~1e-10 or better)
# Compare times (GPU should be faster)
# Compare convergence (should be similar)
```

---

## Key Achievement

**Before**: CPU and GPU benchmarks used different random initial states and couldn't be directly compared.

**After**: Both use EXACT same MPS/MPO data → Fair comparison!

- ✅ Same initial quantum state
- ✅ Same Hamiltonian
- ✅ Reproducible results (seed=42)
- ✅ Cross-platform verification

---

## Files Created

```
benchmarks/
├── serialize_mps_mpo.py          # Generate data
├── load_mps_mpo.py               # Load data (Python)
├── verify_loaders.py             # Verification
├── QUICKSTART.md                 # Quick guide
├── REPRODUCIBLE_BENCHMARKS.md    # Full docs
├── STATUS.md                     # This file
├── cpu_benchmark_run.log         # Current run log
└── benchmark_data/               # Generated data
    ├── heisenberg_*.bin
    ├── josephson_*.bin
    └── *.json (metadata)

pdmrg-gpu/
├── include/
│   └── mps_mpo_loader.hpp        # Load data (C++)
├── src/
│   └── test_mps_mpo_loader.cpp   # C++ test
└── test_mps_mpo_loader           # Compiled test
```

---

## Questions?

See:
- `QUICKSTART.md` for usage examples
- `REPRODUCIBLE_BENCHMARKS.md` for technical details
- `benchmark_data/*.json` for data metadata

Monitor CPU benchmark:
```bash
tail -f cpu_benchmark_run.log
```

Check when done:
```bash
ls -lh cpu_gold_standard_results.json
```
