# Reproducible CPU/GPU DMRG Benchmarks

This directory contains tools for running **exactly reproducible** benchmarks between CPU (Python/Quimb) and GPU (C++/HIP) DMRG implementations.

## Problem

When comparing CPU and GPU implementations, random initialization can introduce variability:
- Different random MPS initial states
- Different random number generators (Python vs C++)
- Makes it hard to compare "apples to apples"

## Solution

**Serialize the MPS and MPO to binary files** that both implementations can load:

1. **Generate once** (Python): Create MPS and MPO with fixed random seed
2. **Save to disk**: Binary files with complex128 data
3. **Load in both**: CPU (Python) and GPU (C++) load the same data
4. **Benchmark**: Run DMRG with identical initial conditions

This guarantees:
- ✅ Same initial MPS state
- ✅ Same Hamiltonian (MPO)
- ✅ Fair performance comparison
- ✅ Reproducibility across runs

---

## Quick Start

### Generate benchmark data and verify loaders

```bash
./run_reproducible_benchmark.sh --all
```

This will:
1. Generate MPS/MPO binary files for all test cases
2. Verify Python can load them
3. Verify C++ can load them
4. Run benchmarks (once implemented)

### Manual workflow

#### Step 1: Generate data

```bash
# Generate all test cases (Heisenberg + Josephson)
python serialize_mps_mpo.py --output-dir benchmark_data --seed 42 --all

# Or generate specific models
python serialize_mps_mpo.py --output-dir benchmark_data --seed 42 --heisenberg
python serialize_mps_mpo.py --output-dir benchmark_data --seed 42 --josephson
```

#### Step 2: Verify data (Python)

```bash
# Load and inspect MPS
python load_mps_mpo.py benchmark_data/heisenberg_L12_chi10_mps.bin --type mps

# Load and inspect MPO
python load_mps_mpo.py benchmark_data/heisenberg_L12_mpo.bin --type mpo
```

#### Step 3: Verify data (C++)

```bash
# Build C++ test
cd ../pdmrg-gpu/build
cmake .. && make test_mps_mpo_loader

# Load and inspect MPS
./test_mps_mpo_loader mps ../benchmarks/benchmark_data/heisenberg_L12_chi10_mps.bin

# Load and inspect MPO
./test_mps_mpo_loader mpo ../benchmarks/benchmark_data/heisenberg_L12_mpo.bin
```

#### Step 4: Run benchmarks with loaded data

**CPU (Python):**
```python
from load_mps_mpo import load_mps_from_binary, load_mpo_from_binary, convert_to_quimb_mps, convert_to_quimb_mpo

# Load data
mps_tensors, _ = load_mps_from_binary("benchmark_data/heisenberg_L12_chi10_mps.bin")
mpo_tensors, _ = load_mpo_from_binary("benchmark_data/heisenberg_L12_mpo.bin")

# Convert to quimb objects
mps = convert_to_quimb_mps(mps_tensors)
mpo = convert_to_quimb_mpo(mpo_tensors)

# Run DMRG
import quimb.tensor as qtn
dmrg = qtn.DMRG2(mpo, bond_dims=100, cutoffs=1e-14)
dmrg.opts['initial_state'] = mps
dmrg.solve(max_sweeps=30, tol=1e-10)
```

**GPU (C++):**
```cpp
#include "mps_mpo_loader.hpp"

// Load data
auto mps = MPSLoader::load("benchmark_data/heisenberg_L12_chi10_mps.bin");
auto mpo = MPOLoader::load("benchmark_data/heisenberg_L12_mpo.bin");

// Run DMRG with loaded data
run_dmrg_gpu(mps, mpo, chi_max=100, max_sweeps=30, tol=1e-10);
```

---

## File Format Specification

### Binary File Format

All binary files use:
- **Byte order**: Native (little-endian on x86-64)
- **Integer type**: `int64_t` (8 bytes)
- **Complex type**: `complex128` = `double` real + `double` imag (16 bytes)
- **Array layout**: C-contiguous (row-major)

### MPS File Format

```
Header:
  - num_sites: int64                    [1 value]
  - bond_dims: int64[num_sites + 1]     [left and right bonds]
  - phys_dims: int64[num_sites]         [physical dimension per site]

For each site i in 0..num_sites-1:
  - shape: int64[3]                     [D_left, d, D_right]
  - data: complex128[D_left * d * D_right]
```

**Indexing**: `data[i, j, k]` → `data[i * d * D_right + j * D_right + k]`

### MPO File Format

```
Header:
  - num_sites: int64                    [1 value]
  - mpo_bond_dims: int64[num_sites + 1] [left and right MPO bonds]
  - phys_dims: int64[num_sites]         [physical dimension per site]

For each site i in 0..num_sites-1:
  - shape: int64[4]                     [D_mpo_left, d_bra, d_ket, D_mpo_right]
  - data: complex128[D_mpo_left * d_bra * d_ket * D_mpo_right]
```

**Indexing**: `data[i, j, k, l]` → `data[i * d_bra * d_ket * D_mpo_right + j * d_ket * D_mpo_right + k * D_mpo_right + l]`

### Metadata Files (JSON)

Each `.bin` file has a corresponding `.json` file with human-readable metadata:

```json
{
  "model": {
    "model": "heisenberg",
    "L": 12,
    "d": 2,
    "chi_init": 10,
    "j": 1.0,
    "bz": 0.0,
    "cyclic": false,
    "seed": 42
  },
  "num_sites": 12,
  "shapes": [[1, 2, 10], [10, 2, 10], ...],
  "dtype": "complex128",
  "total_elements": 2400
}
```

---

## Test Cases

### Heisenberg Model (d=2, real Hamiltonian)

| Case | L | D_max | χ_init | Sweeps |
|------|---|-------|--------|--------|
| Small | 12 | 100 | 10 | 20 |
| Medium | 20 | 100 | 10 | 30 |
| Large | 40 | 200 | 20 | 40 |

### Josephson Junction (d=5, complex Hamiltonian)

| Case | L | D_max | n_max | χ_init | Sweeps |
|------|---|-------|-------|--------|--------|
| Small | 8 | 50 | 2 | 10 | 20 |
| Medium | 12 | 50 | 2 | 10 | 30 |
| Large | 16 | 100 | 2 | 20 | 40 |

---

## Files

### Python Tools

- `serialize_mps_mpo.py` - Generate and save MPS/MPO to binary files
- `load_mps_mpo.py` - Load MPS/MPO from binary files (Python)
- `cpu_gpu_benchmark.py` - CPU benchmark (to be updated for data loading)

### C++ Tools

- `include/mps_mpo_loader.hpp` - C++ header for loading binary files
- `src/test_mps_mpo_loader.cpp` - Test program for C++ loader

### Scripts

- `run_reproducible_benchmark.sh` - End-to-end workflow
- `benchmark_data/` - Output directory for binary files (created on first run)

---

## Implementation Notes

### Python → Binary

Uses NumPy's `tofile()` with `dtype=complex128`:
- Automatically handles endianness
- C-contiguous array layout
- Compatible with C++ `std::complex<double>`

### Binary → C++

Uses `std::ifstream` in binary mode:
- Reads `int64_t` headers
- Reads `std::complex<double>` arrays
- Validates shapes and dimensions

### Quimb Index Convention

Quimb MPO uses index order: `(bra, ket, bond_left, bond_right)`

Our convention: `(bond_left, bra, ket, bond_right)`

**Solution**: Transpose when converting:
```python
# Save: transpose from quimb to our convention
t_save = np.transpose(t_quimb, (2, 0, 1, 3))

# Load: transpose from our convention to quimb
t_quimb = np.transpose(t_load, (1, 2, 0, 3))
```

---

## Validation

To verify correctness:

1. **Round-trip test**: Save → Load → Check values match
2. **Cross-platform test**: Save (Python) → Load (C++) → Compare
3. **Benchmark consistency**: Same initial state → Same final energy

Example validation:
```bash
# Generate data
python serialize_mps_mpo.py --output-dir test_data --seed 12345 --heisenberg

# Verify in Python
python load_mps_mpo.py test_data/heisenberg_L12_chi10_mps.bin --type mps

# Verify in C++
./test_mps_mpo_loader mps test_data/heisenberg_L12_chi10_mps.bin

# Compare tensor norms, shapes, and first few elements
```

---

## Next Steps

To fully enable reproducible benchmarks:

1. ✅ **Data serialization** (Python) - DONE
2. ✅ **Data loading** (Python + C++) - DONE
3. ⬜ **Update CPU benchmark** to accept `--load-data` flag
4. ⬜ **Update GPU benchmark** to use `MPSLoader` and `MPOLoader`
5. ⬜ **Create comparison script** to analyze CPU vs GPU results
6. ⬜ **Add to CMakeLists.txt**: Build test_mps_mpo_loader by default

---

## FAQ

**Q: Why complex128 even for Heisenberg (real Hamiltonian)?**

A: For consistency and future-proofing. Complex numbers work for both real and complex Hamiltonians. Storage overhead is minimal (~2x), and it avoids maintaining separate real/complex code paths.

**Q: Can I use different random seeds?**

A: Yes! Use `--seed` flag:
```bash
python serialize_mps_mpo.py --seed 99999 --all
```

**Q: How much disk space do these files use?**

A: Approximate sizes:
- Heisenberg L=12, χ=10: ~4 KB (MPS) + ~20 KB (MPO)
- Heisenberg L=40, χ=20: ~80 KB (MPS) + ~200 KB (MPO)
- Josephson L=16, χ=20, d=5: ~200 KB (MPS) + ~500 KB (MPO)

Total for all test cases: < 5 MB

**Q: What if I need different initial bond dimensions?**

A: Edit `serialize_mps_mpo.py` to change `chi_init` values, or add command-line arguments.

---

## License

Same as parent project.
