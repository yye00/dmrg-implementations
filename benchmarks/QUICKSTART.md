# Quick Start: Reproducible CPU/GPU Benchmarks

This guide shows you how to run the same exact benchmark on both CPU and GPU with identical MPS and MPO data.

## 1. Generate Benchmark Data (One-Time Setup)

```bash
cd benchmarks

# Generate all test data (Heisenberg + Josephson)
python3 serialize_mps_mpo.py --output-dir benchmark_data --seed 42 --all

# Or generate specific models:
python3 serialize_mps_mpo.py --output-dir benchmark_data --seed 42 --heisenberg
python3 serialize_mps_mpo.py --output-dir benchmark_data --seed 42 --josephson
```

This creates:
- `benchmark_data/*.bin` - Binary MPS/MPO files
- `benchmark_data/*.json` - Human-readable metadata
- `benchmark_data/index.json` - File index

## 2. Verify Data Loads Correctly

### Python Verification

```bash
# Verify all files
python3 verify_loaders.py --data-dir benchmark_data

# Or inspect individual files
python3 load_mps_mpo.py benchmark_data/heisenberg_L12_chi10_mps.bin --type mps
python3 load_mps_mpo.py benchmark_data/heisenberg_L12_mpo.bin --type mpo
```

### C++ Verification

```bash
cd ../pdmrg-gpu

# Compile test (if not already compiled)
g++ -std=c++17 -O2 -I./include src/test_mps_mpo_loader.cpp -o test_mps_mpo_loader

# Test loading
./test_mps_mpo_loader mps ../benchmarks/benchmark_data/heisenberg_L12_chi10_mps.bin
./test_mps_mpo_loader mpo ../benchmarks/benchmark_data/heisenberg_L12_mpo.bin
```

Compare the tensor norms printed by Python and C++ - they should match exactly!

## 3. Run CPU Benchmark with Loaded Data

### Option A: Use in Your Own Script

```python
from load_mps_mpo import load_mps_from_binary, load_mpo_from_binary
from load_mps_mpo import convert_to_quimb_mps, convert_to_quimb_mpo
import quimb.tensor as qtn

# Load data
mps_tensors, _ = load_mps_from_binary("benchmark_data/heisenberg_L12_chi10_mps.bin")
mpo_tensors, _ = load_mpo_from_binary("benchmark_data/heisenberg_L12_mpo.bin")

# Convert to quimb
mps = convert_to_quimb_mps(mps_tensors)
mpo = convert_to_quimb_mpo(mpo_tensors)

# Run DMRG
dmrg = qtn.DMRG2(mpo, bond_dims=100, cutoffs=1e-14)
dmrg.opts['initial_state'] = mps  # Use loaded initial state
dmrg.solve(max_sweeps=30, tol=1e-10, verbosity=1)

print(f"Final energy: {dmrg.energy}")
```

### Option B: Modify Existing Benchmark (TODO)

```bash
# This requires updating cpu_gpu_benchmark.py to accept --load-data flag
python3 cpu_gpu_benchmark.py --load-data benchmark_data --out results_cpu.json
```

## 4. Run GPU Benchmark with Loaded Data

### In Your C++ DMRG Code

```cpp
#include "mps_mpo_loader.hpp"

int main() {
    // Load data
    auto mps = MPSLoader::load("benchmark_data/heisenberg_L12_chi10_mps.bin");
    auto mpo = MPOLoader::load("benchmark_data/heisenberg_L12_mpo.bin");

    // Use loaded data in your DMRG implementation
    // ... your GPU DMRG code here ...

    return 0;
}
```

**Note:** You'll need to adapt the loaded tensors to your GPU DMRG code's data structures.

## 5. Compare Results

Once you have results from both CPU and GPU using the **same exact initial data**, you can compare:

- **Energy convergence**: Should reach the same final energy (within numerical precision)
- **Performance**: Wall-clock time, memory usage, GPU utilization
- **Scalability**: How GPU performance scales with problem size

## Files Generated

```
benchmark_data/
├── heisenberg_L12_chi10_mps.bin       # MPS binary
├── heisenberg_L12_chi10_mps.json      # MPS metadata
├── heisenberg_L12_mpo.bin             # MPO binary
├── heisenberg_L12_mpo.json            # MPO metadata
├── heisenberg_L20_chi10_mps.bin
├── heisenberg_L20_mpo.bin
├── heisenberg_L40_chi20_mps.bin
├── heisenberg_L40_mpo.bin
├── josephson_L8_n2_chi10_mps.bin
├── josephson_L8_n2_mpo.bin
├── josephson_L12_n2_chi10_mps.bin
├── josephson_L12_n2_mpo.bin
├── josephson_L16_n2_chi20_mps.bin
├── josephson_L16_n2_mpo.bin
└── index.json                         # File index
```

Total size: ~1-2 MB for all test cases

## Test Cases

### Heisenberg (d=2, real Hamiltonian)

- **Small**: L=12, χ=10 → D_max=100, 20 sweeps
- **Medium**: L=20, χ=10 → D_max=100, 30 sweeps
- **Large**: L=40, χ=20 → D_max=200, 40 sweeps

### Josephson (d=5, complex Hamiltonian)

- **Small**: L=8, n_max=2 (d=5), χ=10 → D_max=50, 20 sweeps
- **Medium**: L=12, n_max=2 (d=5), χ=10 → D_max=50, 30 sweeps
- **Large**: L=16, n_max=2 (d=5), χ=20 → D_max=100, 40 sweeps

## Key Files

**Python:**
- `serialize_mps_mpo.py` - Generate and save MPS/MPO
- `load_mps_mpo.py` - Load MPS/MPO from binary files
- `verify_loaders.py` - Verify data integrity
- `cpu_gpu_benchmark.py` - CPU benchmark (needs update for data loading)

**C++:**
- `include/mps_mpo_loader.hpp` - C++ loader header
- `src/test_mps_mpo_loader.cpp` - C++ loader test program

**Documentation:**
- `REPRODUCIBLE_BENCHMARKS.md` - Detailed specification
- `QUICKSTART.md` - This file

## Random Seed

The default seed is `42`. To use a different seed:

```bash
python3 serialize_mps_mpo.py --output-dir my_data --seed 99999 --all
```

This ensures **reproducibility**: Same seed → Same MPS/MPO → Same results

## Troubleshooting

**Problem**: "TypeError: Object of type int64 is not JSON serializable"

**Solution**: This is a numpy/JSON compatibility issue. The code handles it with `default=str` in `json.dump()`.

---

**Problem**: C++ loader shows different norms than Python

**Solution**: Check endianness and data types. Both should use:
- Little-endian byte order
- `complex128` (16 bytes per element)
- C-contiguous arrays

---

**Problem**: "File not found" when loading data

**Solution**: Use absolute paths or run from the correct directory:

```bash
# Python
python3 load_mps_mpo.py $(pwd)/benchmark_data/file.bin --type mps

# C++
./test_mps_mpo_loader mps $(pwd)/benchmark_data/file.bin
```

## Next Steps

To fully integrate this into your benchmarks:

1. ✅ **Data serialization** - DONE
2. ✅ **Data loading (Python + C++)** - DONE
3. ⬜ **Update CPU benchmark** to use `--load-data` flag
4. ⬜ **Update GPU executables** to use `MPSLoader` and `MPOLoader`
5. ⬜ **Create comparison script** for CPU vs GPU results

See `REPRODUCIBLE_BENCHMARKS.md` for full details.

---

**Happy Benchmarking!** 🚀
