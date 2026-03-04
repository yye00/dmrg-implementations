# DMRG Implementations

A collection of high-performance Density Matrix Renormalization Group (DMRG) implementations for quantum many-body systems.

## Overview

This repository contains three parallel DMRG implementations optimized for different use cases:

- **PDMRG**: Parallel DMRG with MPI support for distributed computing
- **PDMRG2**: Next-generation implementation with GPU-optimized linear algebra kernels
- **A2DMRG**: Advanced adaptive DMRG with automatic truncation strategies

## Features

- **Parallel Computing**: MPI-based parallelization for efficient large-scale simulations
- **Multiple Backends**: Support for various tensor network backends (quimb, numpy)
- **Comprehensive Benchmarks**: Validated against exact solutions and published results
- **Complex Systems**: Support for both real and complex Hamiltonians (e.g., Josephson arrays)

## Benchmarks

See `BENCHMARK_MANIFEST.md` for detailed benchmark results including:
- Heisenberg model correctness tests (L=12, 48)
- Josephson junction arrays (complex128 validation)
- Performance scaling studies (np=1,2,4,8)

All implementations achieve machine precision agreement (ΔE < 1e-14) on correctness tests.

## Directory Structure

```
├── a2dmrg/          # A2DMRG implementation
├── pdmrg/           # Original PDMRG implementation
├── pdmrg2/          # GPU-optimized PDMRG2 implementation
├── benchmarks/      # Benchmark scripts and results
├── reports/         # Performance and analysis reports
└── debug/           # Debugging utilities
```

## Quick Start

### Requirements

- Python 3.13+ (tested with 3.13 and 3.14)
- MPI (OpenMPI or MPICH)
- NumPy, SciPy
- quimb (tensor network library)
- mpi4py (for parallel runs)

### Running Benchmarks

```bash
# Short correctness benchmark (L=12)
python benchmarks/heisenberg_benchmark.py

# Long benchmark (L=48, 50 sweeps)
python benchmarks/heisenberg_long_benchmark.py

# Josephson junction correctness test (complex128)
python a2dmrg/benchmarks/josephson_correctness_benchmark.py
```

**Note:** A2DMRG speedup benefits appear at larger system sizes (L > 20). At L=12, cotengra contraction overhead may dominate, making it slower than PDMRG in wall time.

### Running DMRG Simulations

Each implementation has its own virtual environment:

```bash
# PDMRG
pdmrg/venv/bin/python run_pdmrg_np1.py

# A2DMRG
a2dmrg/venv/bin/python run_a2dmrg_np1.py
```

## Development

See `pdmrg2_gpu.md` for details on the PDMRG2 CPU optimization plan (GEMM-optimized implementation).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributors

This project uses implementations and optimizations developed through various research efforts in quantum many-body physics.
