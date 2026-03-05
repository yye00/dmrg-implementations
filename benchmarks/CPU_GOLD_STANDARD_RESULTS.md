# CPU Gold Standard Benchmark Results

Generated: 2026-03-05
Quimb v1.12.1 | NumPy v2.3.5
Random Seed: 42
Total Runtime: 18.0 minutes

## Summary

All CPU benchmarks completed successfully using Quimb DMRG1 (1-site) and DMRG2 (2-site) implementations. These results serve as the **gold standard** for validating GPU implementations.

## Heisenberg Model (d=2, real Hamiltonian)

### L=12, D_max=100, 20 sweeps

| Algorithm | Energy (Ha) | Time (s) | Sweeps | Memory (MB) |
|-----------|-------------|----------|--------|-------------|
| DMRG1     | -5.1420906328 | 9.70   | 3      | 17.3        |
| DMRG2     | -5.1420906328 | 4.34   | 3      | 226.0       |

**Status**: ✅ Converged
**Initial MPS**: `benchmark_data/heisenberg_L12_chi10_mps.bin`
**MPO**: `benchmark_data/heisenberg_L12_mpo.bin`

### L=20, D_max=100, 30 sweeps

| Algorithm | Energy (Ha) | Time (s) | Sweeps | Memory (MB) |
|-----------|-------------|----------|--------|-------------|
| DMRG1     | -8.6824733344 | 62.97  | 3      | 19.7        |
| DMRG2     | -8.6824733344 | 71.91  | 3      | 245.7       |

**Status**: ✅ Converged
**Initial MPS**: `benchmark_data/heisenberg_L20_chi10_mps.bin`
**MPO**: `benchmark_data/heisenberg_L20_mpo.bin`

## Josephson Junction Array (d=5, complex128 Hamiltonian)

Parameters: E_J=1.0, E_C=0.5, n_max=2, Φ_ext=π/4

### L=8, D_max=50, 20 sweeps

| Algorithm | Energy (Ha) | Time (s) | Sweeps | Memory (MB) |
|-----------|-------------|----------|--------|-------------|
| DMRG1     | -2.8438010431 | 76.18  | 5      | 7.1         |
| DMRG2     | -2.8438010431 | 145.97 | 4      | 31.6        |

**Status**: ✅ Converged
**Initial MPS**: `benchmark_data/josephson_L8_n2_chi10_mps.bin`
**MPO**: `benchmark_data/josephson_L8_n2_mpo.bin`

### L=12, D_max=50, 30 sweeps

| Algorithm | Energy (Ha) | Time (s) | Sweeps | Memory (MB) |
|-----------|-------------|----------|--------|-------------|
| DMRG1     | -4.5070608947 | 202.51 | 8      | 284.5       |
| DMRG2     | -4.5070608947 | 503.53 | 6      | 4.8         |

**Status**: ✅ Converged
**Initial MPS**: `benchmark_data/josephson_L12_n2_chi10_mps.bin`
**MPO**: `benchmark_data/josephson_L12_n2_mpo.bin`

## Key Observations

1. **Convergence**: Both DMRG1 and DMRG2 converge to identical energies (within numerical precision)
2. **Speed**: DMRG1 generally faster for these test cases
3. **Sweeps**: All cases converged quickly (3-8 sweeps)
4. **Memory**: DMRG2 uses more memory for smaller cases, less for larger cases

## GPU Validation Targets

When running GPU benchmarks with the same initial MPS/MPO data, the final energies should match these values within tolerance (~1e-10):

```
Heisenberg L=12:  -5.1420906328 Ha
Heisenberg L=20:  -8.6824733344 Ha
Josephson L=8:    -2.8438010431 Ha
Josephson L=12:   -4.5070608947 Ha
```

## Files Generated

- `cpu_gold_standard_results.json` - Machine-readable results
- `cpu_benchmark_run.log` - Full benchmark log
- `benchmark_data/*.bin` - Serialized MPS/MPO files (seed=42)

## Usage for GPU Benchmarks

### Load Initial Data

```cpp
#include "mps_mpo_loader.hpp"

// Load initial state and Hamiltonian
auto mps = MPSLoader::load("benchmark_data/heisenberg_L12_chi10_mps.bin");
auto mpo = MPOLoader::load("benchmark_data/heisenberg_L12_mpo.bin");

// Run GPU DMRG
auto result = run_dmrg_gpu(mps, mpo, chi_max=100, sweeps=20);

// Validate
double target_energy = -5.1420906328;
double error = std::abs(result.energy - target_energy);
if (error < 1e-9) {
    std::cout << "✓ GPU matches CPU gold standard!" << std::endl;
} else {
    std::cout << "✗ Energy mismatch: " << error << std::endl;
}
```

### Validation Criteria

- **Energy Error**: < 1e-9 Ha (excellent), < 1e-7 Ha (acceptable)
- **Convergence**: Should converge in similar number of sweeps
- **Performance**: Compare wall-clock time and memory usage

## Next Steps

1. Integrate `mps_mpo_loader.hpp` into GPU executables
2. Load initial MPS/MPO from binary files
3. Run GPU DMRG with loaded data
4. Compare final energies with these gold standard values
5. Analyze performance (speedup, memory usage)

---

**Important**: These benchmarks used `--skip-large` flag. For complete validation including L=40 Heisenberg and L=16 Josephson, rerun without this flag (will take ~30-40 minutes).
