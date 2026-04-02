# DMRG Implementations

A collection of high-performance Density Matrix Renormalization Group (DMRG) implementations for quantum many-body systems.

## Overview

This repository contains **in-house MPI-parallel DMRG implementations** validated against **reference baselines** from the quimb tensor network library.

### In-House Implementations

- **PDMRG**: Parallel two-site DMRG with MPI domain decomposition and boundary merge protocol
- **A2DMRG**: Additive two-level DMRG with parallel warmup and coarse-space correction
- **PDMRG-OPT**: ⚠️ **Specification only** — GEMM-optimized CPU implementation (see `pdmrg_gpu_opt.md`)
- **PDMRG-GPU**: 🔬 **Experimental** — GPU-accelerated implementation using hipTensor/rocBLAS (see `gpu-rocm/pdmrg-gpu/`)
- **DMRG-GPU**: GPU-native single-site DMRG using rocBLAS/rocSOLVER on AMD MI300X (see `gpu-rocm/dmrg-gpu/`)

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
├── cpu/                         # CPU Python implementations
│   ├── pdmrg/                   #   Parallel two-site DMRG (MPI + numpy)
│   ├── pdmrg-cotengra/          #   PDMRG with cotengra contractions
│   ├── pdmrg-opt/               #   PDMRG with GEMM optimizations
│   └── a2dmrg/                  #   Additive two-level DMRG
├── gpu-rocm/                    # AMD MI300X GPU implementations (C++/HIP)
│   ├── dmrg-gpu/                #   Single-site DMRG (rocBLAS/rocSOLVER)
│   ├── dmrg-gpu-opt/            #   Optimized single-site DMRG
│   ├── dmrg2-gpu/               #   Two-site DMRG
│   ├── dmrg2-gpu-opt/           #   Optimized two-site DMRG
│   ├── pdmrg-gpu/               #   Parallel DMRG on GPU (hipTensor)
│   └── pdmrg-gpu-opt/           #   Optimized parallel DMRG on GPU
├── gpu-cuda/                    # NVIDIA H100 GPU implementations (planned)
│   ├── README.md                #   Porting status
│   └── (empty subdirs)          #   Mirrors gpu-rocm/ structure
├── benchmarks/                  # Benchmark scripts and results
│   ├── results/mi300x/          #   Raw validation results (MI300X)
│   ├── results/h100/            #   Raw validation results (H100, future)
│   ├── paper_results/mi300x/    #   Publication-grade results (MI300X)
│   └── paper_results/h100/      #   Publication-grade results (H100, future)
├── reports/                     # Timing reports
│   ├── mi300x/                  #   MI300X timing JSONs
│   └── h100/                    #   H100 timing JSONs (future)
├── docs/                        # GPU development prompts and references
├── paper/                       # CPC manuscript
├── IMPLEMENTATION_MATRIX.md     # Comprehensive implementation taxonomy
└── pdmrg_gpu_opt.md             # PDMRG-OPT CPU optimization specification
```

### Architecture Split

- **`cpu/`** -- Pure-Python implementations using numpy/scipy/quimb, parallelized with MPI.
- **`gpu-rocm/`** -- C++/HIP implementations targeting AMD MI300X via rocBLAS, rocSOLVER, and hipTensor. All current GPU benchmarks were run on MI300X.
- **`gpu-cuda/`** -- Planned CUDA ports targeting NVIDIA H100. Each subdirectory will be a fully independent codebase (no shared code with gpu-rocm/).
- **Benchmark results** are tagged by GPU architecture (`mi300x/`, `h100/`) under both `benchmarks/results/` and `benchmarks/paper_results/`.

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
python cpu/a2dmrg/benchmarks/josephson_correctness_benchmark.py
```

**Note:** A2DMRG speedup benefits appear at larger system sizes (L > 20). At L=12, cotengra contraction overhead may dominate, making it slower than PDMRG in wall time.

### Running DMRG Simulations

Each implementation has its own virtual environment:

```bash
# PDMRG
cpu/pdmrg/venv/bin/python run_pdmrg_np1.py

# A2DMRG
cpu/a2dmrg/venv/bin/python run_a2dmrg_np1.py
```

## Development

### PDMRG-OPT CPU Optimization Plan

See `pdmrg_gpu_opt.md` for the PDMRG-OPT specification. **Note**: Despite the filename, this document describes **CPU-level GEMM optimizations**, not GPU implementation. It prepares the mathematical architecture for potential future GPU porting by replacing memory-bandwidth-bound operations (BLAS-2) with compute-bound operations (BLAS-3).

Key planned changes:
- Block-Davidson (LOBPCG) local solver
- Newton-Schulz polar decomposition for gauge shifts
- Randomized SVD with Cholesky-QR2 for bond truncation
- Exact SVD retained for boundary merge (numerical stability)

**Implementation Status**: Specification complete, no code written yet.

### GPU-Native DMRG (`gpu-rocm/dmrg-gpu/`)

Single-site DMRG with all tensor contractions on GPU via rocBLAS `dgemm`. Targets AMD MI300X (gfx942) with ROCm 7.2+.

**Architecture**: All environment updates, H_eff applications, and Lanczos Krylov iterations run on GPU. Only the tridiagonal eigensolve (~100 elements) and SVD run on CPU (LAPACK). GPU SVD via rocsolver is available but slower for bond dimensions below ~250.

**Build & Run** (on MI300X host):
```bash
cd gpu-rocm/dmrg-gpu && mkdir build && cd build
cmake .. -DGPU_TARGETS=gfx942
make -j
./dmrg_gpu 32 64 30          # L=32, chi=64, 30 sweeps
./dmrg_gpu 32 64 30 --gpu-svd  # use GPU SVD instead of CPU
```

#### Performance (AMD MI300X, Heisenberg spin-1/2 chain)

All benchmarks: single-site DMRG, 5 sweeps to convergence, Heisenberg OBC.

| System | Bond dim | dmrg-gpu (CPU SVD) | dmrg-gpu (GPU SVD) | quimb DMRG1 (1 CPU) |
|--------|----------|-------------------|-------------------|---------------------|
| L=8    | 32       | 0.08s             | 0.3s              | 0.05s               |
| L=32   | 64       | 1.3s              | 4.5s              | 4.7s                |
| L=64   | 128      | 7.2s              | 18s               | ~40s                |
| L=128  | 128      | 15.8s             | ~45s              | ~80s                |

**Key findings**:
- GPU tensor contractions (dgemm) are fast even at small chi, but SVD dominates runtime (80-96%)
- rocsolver SVD has ~10ms launch overhead per call, making it 2-6x slower than LAPACK for chi < 200
- CPU LAPACK SVD is the default; use `--gpu-svd` flag to switch to rocsolver
- GPU SVD breaks even at chi ~250 and wins for chi > 300
- At chi=128+, dmrg-gpu with CPU SVD is 3-5x faster than single-threaded quimb

#### CPU vs GPU Work Distribution

| Component | Backend | Notes |
|-----------|---------|-------|
| Environment updates (L, R) | GPU (rocBLAS dgemm) | Batched over MPO indices |
| H_eff matvec (Lanczos) | GPU (rocBLAS dgemm) | Batched GEMM, ~10 iterations |
| Lanczos reorthogonalization | GPU (rocBLAS dgemv) | Full reorth via 2 dgemv calls |
| Tridiagonal eigensolve | CPU (LAPACK dstev) | ~100 elements, negligible cost |
| SVD truncation | CPU (LAPACK dgesvd) or GPU (rocsolver) | Toggleable; CPU default |
| MPS tensor reshape/update | GPU (rocBLAS dgemm) | S*Vh or U*S multiplication |

### GPU Experimental Path (legacy)

The `gpu-rocm/pdmrg-gpu/` directory contains experimental GPU implementations using hipTensor and rocBLAS. This is research code under active development and not production-ready.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributors

This project uses implementations and optimizations developed through various research efforts in quantum many-body physics.
