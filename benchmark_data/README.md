# Static Benchmark Data

This directory contains pre-generated, serialized benchmark problems for DMRG implementations.

## Purpose

All DMRG implementations (quimb DMRG1, quimb DMRG2, PDMRG, PDMRG2, A2DMRG) load **identical** inputs from disk to ensure fair comparison and reproducibility.

## Directory Structure

```
benchmark_data/
├── heisenberg/
│   ├── L12_D20/
│   ├── L32_D20/
│   └── L48_D20/
└── josephson/
    ├── L20_D50_nmax2/
    ├── L24_D50_nmax2/
    ├── L28_D50_nmax2/
    └── L32_D50_nmax2/
```

Each benchmark case contains:
- `mpo.npz`: Serialized MPO (Matrix Product Operator)
- `initial_mps.npz`: Serialized initial MPS state
- `manifest.json`: Metadata (system parameters, dtype, gauge, etc.)
- `golden_results.json`: Reference results from quimb DMRG1/DMRG2

## Data Format

### MPO Format (`mpo.npz`)

NumPy archive containing:
- `tensors_0`, `tensors_1`, ..., `tensors_{L-1}`: MPO tensors as ndarrays
- `metadata`: JSON string with shape info and dtype

### MPS Format (`initial_mps.npz`)

NumPy archive containing:
- `tensors_0`, `tensors_1`, ..., `tensors_{L-1}`: MPS tensors as ndarrays
- `metadata`: JSON string with canonical form info

### Manifest Format (`manifest.json`)

```json
{
  "model": "heisenberg" | "josephson",
  "L": <int>,
  "bond_dim": <int>,
  "dtype": "float64" | "complex128",
  "parameters": {
    // Model-specific parameters
  },
  "initial_mps_gauge": "left" | "right" | "mixed",
  "generator_version": "1.0",
  "created_timestamp": "ISO-8601"
}
```

### Golden Results Format (`golden_results.json`)

```json
{
  "quimb_dmrg1": {
    "energy": <float>,
    "energy_per_site": <float>,
    "sweeps": <int>,
    "tolerance": <float>,
    "cutoff": <float>,
    "converged": <bool>,
    "wall_time": <float>,
    "threads": <int>
  },
  "quimb_dmrg2": {
    // Same structure
  }
}
```

## Loading Utilities

Use `benchmark_data_loader.py` to load data consistently across all implementations.

## Regeneration

To regenerate all static data:
```bash
python scripts/generate_benchmark_data.py --all
```

**Warning**: Regenerating data invalidates existing golden results. Use with caution.

## Validation Thresholds

- **Heisenberg (real)**: Machine precision agreement (ΔE < 1e-12 target, 1e-10 acceptance)
- **Josephson (complex)**: Machine precision agreement (ΔE < 1e-12 target, 1e-10 acceptance)

Any implementation failing these thresholds must document the reason explicitly.
