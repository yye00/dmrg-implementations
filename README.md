# DMRG Implementations

A collection of high-performance Density Matrix Renormalization Group (DMRG) implementations for quantum many-body systems.

## Overview

This repository contains **in-house MPI-parallel DMRG implementations** validated against **reference baselines** from the quimb tensor network library.

### In-House Implementations

- **PDMRG**: Parallel two-site DMRG with MPI domain decomposition and boundary merge protocol
- **A2DMRG**: Additive two-level DMRG with parallel warmup and coarse-space correction
- **PDMRG2**: ⚠️ **Specification only** — GEMM-optimized CPU implementation (see `pdmrg2_gpu.md`)
- **PDMRG-GPU**: 🔬 **Experimental** — GPU-accelerated implementation using hipTensor/rocBLAS (see `pdmrg-gpu/`)

### Reference Baselines (External)

- **quimb DMRG1**: Single-site serial DMRG (third-party, via quimb library)
- **quimb DMRG2**: Two-site serial DMRG (third-party, via quimb library)

**For detailed taxonomy, see [`IMPLEMENTATION_MATRIX.md`](IMPLEMENTATION_MATRIX.md)**

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
├── a2dmrg/                  # A2DMRG implementation
├── pdmrg/                   # PDMRG implementation
├── pdmrg-gpu/               # Experimental GPU implementation (C++/hipTensor)
├── benchmarks/              # Benchmark scripts and results
├── reports/                 # Performance and analysis reports
├── debug/                   # Debugging utilities
├── IMPLEMENTATION_MATRIX.md # Comprehensive implementation taxonomy
└── pdmrg2_gpu.md            # PDMRG2 CPU optimization specification
```

## Known Issues and Current Status

⚠️ **Important**: Both PDMRG and A2DMRG have documented correctness caveats that affect benchmark interpretation:

1. **np=1 Early Return Behavior**: When running with a single MPI process (`np=1`) and warmup enabled, both PDMRG and A2DMRG return the serial warmup energy **without executing the parallel algorithm**. This means "PDMRG np=1" and "A2DMRG np=1" benchmark results actually measure quimb DMRG2 performance.

2. **PDMRG Boundary Merge Optimization Disabled**: The boundary merge optimization path is permanently disabled due to an unresolved "H_eff spurious eigenvalue problem." Impact on performance is undocumented.

3. **A2DMRG Canonicalization Policy**: Canonicalization is intentionally skipped to preserve bond dimension structure required for coarse-space linear combination. This is a design decision, not a bug.

**See [`IMPLEMENTATION_MATRIX.md`](IMPLEMENTATION_MATRIX.md) for complete details and code line references.**

### Benchmark Metadata

All benchmark results should follow the metadata schema defined in [`BENCHMARK_METADATA_SCHEMA.md`](BENCHMARK_METADATA_SCHEMA.md), which requires explicit documentation of:
- Algorithm execution path (warmup vs. parallel algorithm)
- Early return behavior
- Disabled optimizations (e.g., skip_opt flag)
- Canonicalization status
- MPI configuration

**Current Status**: Benchmark scripts are being updated to output full metadata (Phase 2 of CPU audit in progress).

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

### PDMRG2 CPU Optimization Plan

See `pdmrg2_gpu.md` for the PDMRG2 specification. **Note**: Despite the filename, this document describes **CPU-level GEMM optimizations**, not GPU implementation. It prepares the mathematical architecture for potential future GPU porting by replacing memory-bandwidth-bound operations (BLAS-2) with compute-bound operations (BLAS-3).

Key planned changes:
- Block-Davidson (LOBPCG) local solver
- Newton-Schulz polar decomposition for gauge shifts
- Randomized SVD with Cholesky-QR2 for bond truncation
- Exact SVD retained for boundary merge (numerical stability)

**Implementation Status**: Specification complete, no code written yet.

### GPU Experimental Path

The `pdmrg-gpu/` directory contains experimental GPU implementations using hipTensor and rocBLAS. This is research code under active development and not production-ready.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributors

This project uses implementations and optimizations developed through various research efforts in quantum many-body physics.
