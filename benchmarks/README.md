# DMRG Benchmark Suite

Unified benchmarking infrastructure for all DMRG implementations in this project.
Supports quick development validation, full timing benchmarks, and scaling studies
for publication.

## Quick Start

```bash
# List all registered implementations
./run.py list

# Quick correctness check (all implementations, small systems)
./run.py validate

# Validate just one implementation
./run.py validate --impl quimb-dmrg2

# Full timing benchmark
./run.py benchmark --model heisenberg --size medium

# Scaling study
./run.py scale --impl pdmrg --np 1,2,4,8 --threads 1
```

## Implementations

All DMRG implementations are registered in `lib/registry.py`. The benchmark
suite currently supports 10 implementations across CPU and GPU:

| Name | Type | `--np` | `--threads` | Description |
|------|------|--------|-------------|-------------|
| `quimb-dmrg1` | CPU serial | - | BLAS threads | Quimb DMRG single-site (reference) |
| `quimb-dmrg2` | CPU serial | - | BLAS threads | Quimb DMRG two-site (reference) |
| `pdmrg` | CPU parallel | MPI ranks | BLAS threads/rank | Parallel DMRG (numpy tensordot) |
| `pdmrg2` | CPU parallel | MPI ranks | BLAS threads/rank | Parallel DMRG two-site |
| `pdmrg-cotengra` | CPU parallel | MPI ranks | BLAS threads/rank | Parallel DMRG (cotengra paths) |
| `a2dmrg` | CPU parallel | MPI ranks | BLAS threads/rank | Additive two-level DMRG |
| `dmrg-gpu` | GPU serial | - | - | GPU single-site (HIP/rocBLAS) |
| `dmrg2-gpu` | GPU serial | - | - | GPU two-site (HIP/rocBLAS) |
| `pdmrg-gpu` | GPU parallel | HIP streams | - | GPU parallel DMRG |
| `pdmrg2-gpu` | GPU parallel | HIP streams | - | GPU parallel two-site DMRG |

### What `--np` means

- **CPU parallel implementations** (pdmrg, pdmrg2, pdmrg-cotengra, a2dmrg):
  `--np` sets the number of MPI ranks via `mpirun -np N`.

- **GPU parallel implementations** (pdmrg-gpu, pdmrg2-gpu):
  `--np` sets the number of HIP streams for concurrent segment sweeping.

- **Serial implementations** ignore `--np`.

### What `--threads` means

Sets `OPENBLAS_NUM_THREADS` (and equivalent vars) for CPU implementations.
Controls the number of BLAS threads used per MPI rank. GPU implementations
ignore this flag.

## Models

### Heisenberg Chain (d=2, real)

Antiferromagnetic spin-1/2 Heisenberg chain with open boundary conditions:

    H = J * sum_i (S_x^i S_x^{i+1} + S_y^i S_y^{i+1} + S_z^i S_z^{i+1})

Parameters: J=1.0, B_z=0.0, open boundaries.

### Josephson Junction Array (d=5, complex)

Capacitively coupled Josephson junction chain with external flux:

    H = -E_J/2 * sum_i (e^{i*phi_ext} e^{i*phi_i} e^{-i*phi_{i+1}} + h.c.)
        + E_C * sum_i n_i^2

Parameters: E_J=1.0, E_C=0.5, n_max=2 (d=2*n_max+1=5), phi_ext=pi/4.

The non-zero external flux breaks time-reversal symmetry, requiring complex128
arithmetic. The larger local Hilbert space (d=5 vs d=2) makes this a more
demanding benchmark than Heisenberg.

## Problem Sizes

| Size | Heisenberg | Josephson | Typical Runtime |
|------|-----------|-----------|-----------------|
| `small` | L=12, chi=20 | L=8, chi=20 | seconds |
| `medium` | L=20, chi=50 | L=12, chi=50 | minutes |
| `large` | L=40, chi=100 | L=16, chi=100 | tens of minutes |

Use `--size small` for quick validation, `--size medium` for standard
benchmarks, `--size large` for publication-quality scaling studies.

## CLI Reference

### `./run.py validate`

Quick correctness check. Runs each implementation on small systems and
compares energies against the gold standard (tolerance: 1e-10).

```bash
# All implementations, both models
./run.py validate

# Just GPU implementations
./run.py validate --impl dmrg-gpu,dmrg2-gpu,pdmrg-gpu,pdmrg2-gpu

# CPU parallel with specific np values
./run.py validate --impl pdmrg,a2dmrg --np 2,4 --threads 1

# Save results
./run.py validate --output results/validation.json
```

Exit code is 0 if all pass, 1 if any fail.

### `./run.py benchmark`

Full timing suite. Runs implementations across problem sizes and reports
wall-clock times.

```bash
# Everything (default: small + medium sizes)
./run.py benchmark

# Specific model and size
./run.py benchmark --model heisenberg --size medium,large

# GPU implementations only
./run.py benchmark --impl dmrg-gpu,dmrg2-gpu --size small,medium,large

# Parallel scaling
./run.py benchmark --impl pdmrg --np 2,4,8 --threads 1,4

# Save results for report generation
./run.py benchmark --output results/benchmark.json
```

### `./run.py scale`

Scaling study: systematically varies `--np` or `--threads` to measure
parallel efficiency.

```bash
# np scaling for all parallel implementations
./run.py scale --np 1,2,4,8

# Thread scaling for CPU implementations
./run.py scale --impl quimb-dmrg2,pdmrg --threads 1,2,4,8

# GPU stream scaling
./run.py scale --impl pdmrg-gpu --np 1,2,4,8 --size large

# Save results
./run.py scale --output results/scaling.json
```

### `./run.py report`

Generate a markdown report from saved results.

```bash
./run.py report --input results/benchmark.json
./run.py report --input results/benchmark.json --output REPORT.md
```

### `./run.py generate-data`

Regenerate the binary MPS/MPO test data files used for reproducible
benchmarks. Uses seed=42 for exact reproducibility.

```bash
./run.py generate-data
./run.py generate-data --seed 42
```

### `./run.py list`

Print all registered implementations with their capabilities.

## Directory Structure

```
benchmarks/
├── run.py                    # CLI entry point (start here)
├── README.md                 # This file
│
├── lib/                      # Shared infrastructure
│   ├── registry.py           # Implementation configs and size definitions
│   ├── dispatch.py           # Routes impl names to runner functions
│   ├── data_loader.py        # MPS/MPO binary file loading
│   ├── hardware.py           # CPU/core detection, MPI env, venv paths
│   ├── models.py             # Physics model builders (Josephson MPO)
│   ├── report.py             # Table formatting and report generation
│   └── runners/              # Per-implementation launch logic
│       ├── quimb_runner.py   # In-process quimb DMRG1/2
│       ├── pdmrg_runner.py   # MPI subprocess for pdmrg/pdmrg2/cotengra
│       ├── a2dmrg_runner.py  # MPI subprocess for a2dmrg
│       └── gpu_runner.py     # Subprocess for compiled GPU executables
│
├── data/                     # Binary MPS/MPO test data (seed=42)
│   ├── generate.py           # Data generation script
│   ├── verify.py             # Data integrity checker
│   └── *.bin, *.json         # Binary tensor files + metadata
│
├── validation/               # Correctness checking
│   ├── validate.py           # Validation runner
│   └── gold_standard.json    # CPU reference energies
│
├── performance/              # Timing and scaling
│   ├── benchmark.py          # Full timing suite
│   └── scaling.py            # np/thread scaling studies
│
└── results/                  # Output directory for benchmark runs
    └── .gitkeep
```

## Adding a New Implementation

1. **Add a registry entry** in `lib/registry.py`:

```python
IMPLEMENTATIONS["my-new-impl"] = {
    "type": "cpu-parallel",          # or cpu-serial, gpu-serial, gpu-parallel
    "description": "My new DMRG variant",
    "supports_threads": True,        # can use OPENBLAS_NUM_THREADS?
    "supports_np": True,             # can use mpirun -np / HIP streams?
    "runner": "pdmrg",              # which runner module to use
    "package": "my-new-impl",       # directory name in repo root
    "entry": "my_impl.dmrg",        # Python import path
    "function": "my_main",          # function to call
}
```

2. **Choose a runner** from `lib/runners/`:
   - `quimb` — in-process Python (direct import)
   - `pdmrg` — MPI-launched Python (mpirun subprocess)
   - `a2dmrg` — wrapper around pdmrg runner
   - `gpu` — compiled C++/HIP executable (subprocess)

   If none fit, create a new runner in `lib/runners/` with a `run()` function
   that returns `{"energy": float, "time": float, "success": bool}`.

3. **Test it**:
```bash
./run.py validate --impl my-new-impl
./run.py benchmark --impl my-new-impl --size small
```

## Data Format

Binary files store MPS and MPO tensors in complex128 with int64 headers.
The format is designed for cross-language compatibility (Python and C++).

### MPS Binary Format

```
Header:
  num_sites: int64
  bond_dims: int64[num_sites + 1]
  phys_dims: int64[num_sites]

Per site:
  shape: int64[3]                          # (D_left, d, D_right)
  data: complex128[D_left * d * D_right]   # C-contiguous, little-endian
```

### MPO Binary Format

```
Header:
  num_sites: int64
  mpo_bond_dims: int64[num_sites + 1]
  phys_dims: int64[num_sites]

Per site:
  shape: int64[4]                                # (D_left, d, d, D_right)
  data: complex128[D_left * d * d * D_right]     # C-contiguous, little-endian
```

## Gold Standard Reference Energies

CPU reference energies (quimb DMRG2) used for validation.
All implementations should match within 1e-10:

| Model | L | chi | Energy |
|-------|---|-----|--------|
| Heisenberg | 12 | 100 | -5.1420906328 |
| Heisenberg | 20 | 100 | -8.6824733344 |
| Josephson | 8 | 50 | -2.8438010431 |
| Josephson | 12 | 50 | -4.5070608947 |

Full results in `validation/gold_standard.json`.

## Literature References

- U. Schollwock, "The density-matrix renormalization group in the age of matrix product states," Ann. Phys. 326, 96-192 (2011).
- E. M. Stoudenmire and S. R. White, "Real-space parallel density matrix renormalization group," Phys. Rev. B 87, 155137 (2013).
- R. Fazio and H. van der Zant, "Quantum phase transitions and vortex dynamics in superconducting networks," Phys. Rep. 355, 235-334 (2001).
