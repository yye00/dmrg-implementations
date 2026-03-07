# DMRG Implementation Matrix

This document provides a comprehensive taxonomy of all DMRG implementations in this repository, distinguishing **reference baselines** (third-party libraries) from **in-house implementations**.

## Implementation Overview Table

| Method Name | Ownership | Algorithm Family | Execution Model | Backend | Warmup Strategy | Current Status | Location |
|-------------|-----------|------------------|-----------------|---------|----------------|----------------|----------|
| **quimb DMRG1** | External (quimb) | Single-site DMRG | Serial | CPU (NumPy/SciPy) | N/A (serial algorithm) | ✅ Stable baseline | `quimb.tensor.tensor_dmrg.DMRG` |
| **quimb DMRG2** | External (quimb) | Two-site DMRG | Serial | CPU (NumPy/SciPy) | N/A (serial algorithm) | ✅ Stable baseline | `quimb.tensor.tensor_dmrg.DMRG2` |
| **PDMRG** | In-house | Parallel two-site DMRG | MPI parallel | CPU (NumPy/SciPy) | Serial warmup (quimb DMRG2) | ⚠️ Has known issues | `pdmrg/pdmrg/dmrg.py` |
| **A2DMRG** | In-house | Additive two-level DMRG | MPI parallel | CPU (NumPy/SciPy) | Parallel warmup (rank-local) | ⚠️ Has known issues | `a2dmrg/a2dmrg/dmrg.py` |
| **PDMRG2** | In-house (planned) | GEMM-optimized PDMRG | MPI parallel | CPU (BLAS-3) | TBD | 🚧 Specification only | `pdmrg2_gpu.md` (spec) |
| **PDMRG-GPU** | In-house (experimental) | GPU-accelerated PDMRG | MPI + GPU | GPU (hipTensor/rocBLAS) | TBD | 🔬 Experimental | `pdmrg-gpu/*.cpp` |

## Legend

- ✅ **Stable baseline**: Validated external reference implementation
- ⚠️ **Has known issues**: Working but with documented caveats
- 🚧 **Specification only**: Design document exists, no implementation yet
- 🔬 **Experimental**: Research code, not validated

---

## Detailed Implementation Descriptions

### Reference Baselines (External)

These are **third-party implementations** used for correctness validation.

#### quimb DMRG1
- **Purpose**: Single-site DMRG baseline for small-scale correctness tests
- **Algorithm**: Operates on one site at a time, iteratively optimizing each tensor
- **Usage**: Implicit in some benchmarks via quimb's default settings
- **Caveats**: Generally slower than two-site DMRG but numerically stable

#### quimb DMRG2
- **Purpose**: Two-site DMRG baseline, primary correctness reference
- **Algorithm**: Optimizes two adjacent sites simultaneously, then SVD-splits
- **Usage**:
  - Direct benchmarking against PDMRG and A2DMRG
  - Serial warmup initialization for PDMRG (when `warmup_sweeps > 0`)
- **Caveats**: Serial only, not suitable for large-scale simulations

---

### In-House CPU Implementations

These are **locally developed implementations** with MPI parallelization.

#### PDMRG (Parallel DMRG)

**Location**: `pdmrg/pdmrg/dmrg.py`

**Algorithm Overview**:
- Spatial domain decomposition: each MPI rank owns a contiguous spatial block
- Sweeps: left-to-right and right-to-left optimization passes
- Boundary merge: ranks exchange and merge boundary tensors using SVD-based protocol

**Key Features**:
- MPI-based parallelization for distributed-memory systems
- Serial warmup using quimb DMRG2 (when `warmup_sweeps > 0`)
- Two-site optimization with bond truncation

**Known Issues** (as of 2026-03-06):
1. **np=1 Early Return** (line 736-739):
   ```python
   if n_procs == 1 and not random_init_flag:
       if rank == 0 and verbose:
           print(f"np=1: returning serial-warmup energy {warmup_energy:.12f}")
       return warmup_energy, pmps
   ```
   - When running with `np=1` (single MPI process) and warmup enabled, the function returns the warmup energy **without running the PDMRG algorithm**
   - This means benchmarks labeled "PDMRG np=1" are actually measuring quimb DMRG2 performance
   - **Status**: Documented, requires fixing in Phase 3

2. **Boundary Merge Optimization Disabled** (line 807):
   ```python
   skip_opt = True  # Always skip until H_eff bug is fixed
   ```
   - The `boundary_merge` function has an optimization path (`skip_optimization=False`) that should accelerate convergence
   - This optimization is **permanently disabled** due to "H_eff spurious eigenvalue problem"
   - **Status**: Root cause unknown, requires investigation in Phase 3

**Warmup Behavior**:
- When `warmup_sweeps > 0`: uses quimb DMRG2 to initialize MPS before parallel sweeps
- When `warmup_sweeps = 0` and `random_init_flag = True`: initializes with random MPS

**Current Validation Status**:
- ✅ Correctness tests pass at L=12, L=48 (Heisenberg model)
- ✅ Complex128 validation passes (Josephson junction arrays)
- ⚠️ np=1 behavior is misleading
- ❓ Impact of disabled boundary merge optimization is undocumented

---

#### A2DMRG (Additive Two-Level DMRG)

**Location**: `a2dmrg/a2dmrg/dmrg.py`

**Algorithm Overview**:
- Two-level additive scheme: each rank performs local DMRG sweeps independently
- Coarse-space correction: linearly combines local MPS solutions
- No boundary merge protocol (fundamentally different from PDMRG)

**Key Features**:
- Parallel warmup: each rank initializes its local block independently
- No cross-rank communication during sweeps (only at coarse-space combination)
- Canonicalization **intentionally skipped** to preserve bond dimensions for linear combination

**Known Issues** (as of 2026-03-06):
1. **np=1 Early Return** (line 333-334):
   ```python
   if size == 1 and warmup_sweeps > 0 and initial_mps is None:
       return energy_prev, mps
   ```
   - Similar to PDMRG: when `np=1`, returns warmup result without running A2DMRG
   - **Status**: Documented, requires fixing in Phase 4

2. **Canonicalization Disabled** (line 348-356):
   ```python
   # NOTE: For A2DMRG, we do NOT left-canonicalize at each sweep!
   # Canonicalization can reduce bond dimensions, making MPS incompatible
   # for linear combination in the coarse-space method.
   ```
   - This is a **design decision**, not a bug
   - Rationale: preserving bond dimension structure is essential for A2DMRG's additive correction scheme
   - **Status**: Requires formal documentation in Phase 4

**Warmup Behavior**:
- When `warmup_sweeps > 0`: each rank performs local quimb DMRG2 warmup on its spatial block
- Parallel warmup means ranks do not share data during initialization

**Current Validation Status**:
- ✅ Correctness tests pass at small scales
- ⚠️ Performance benefits only appear at L > 20 (cotengra overhead at small L)
- ⚠️ np=1 behavior is misleading
- ❓ Canonicalization policy needs formal documentation

---

### In-House Implementations (Planned/Experimental)

#### PDMRG2 (GEMM-Optimized CPU Implementation)

**Location**: `pdmrg2_gpu.md` (specification document only)

**Status**: 🚧 **No implementation exists yet** — specification document describes a future CPU optimization phase

**Planned Algorithm Changes**:
1. **Block-Davidson Local Solver**: Replace single-vector Lanczos with block-Krylov LOBPCG (GEMM-heavy)
2. **Newton-Schulz Polar Decomposition**: Replace QR/SVD gauge shifts with iterative matrix multiplication
3. **Randomized SVD with Cholesky-QR2**: Replace exact SVD with rSVD for bond truncation (GEMM-heavy)
4. **Exception**: Boundary merge must still use exact SVD for numerical stability

**Clarification from pdmrg2_gpu.md**:
> "This document describes CPU-level optimizations using GEMM-heavy algorithms. It is NOT a GPU implementation roadmap, but rather prepares the mathematical architecture for potential future GPU porting."

**Objective**: Improve CPU cache efficiency by replacing memory-bandwidth-bound operations (BLAS-2) with compute-bound operations (BLAS-3)

**Implementation Status**: Specification complete, no code written yet

---

#### PDMRG-GPU (GPU-Accelerated Experimental)

**Location**: `pdmrg-gpu/*.cpp`

**Status**: 🔬 **Experimental research code** — under active development, not production-ready

**Technology Stack**:
- hipTensor API for tensor contractions
- rocBLAS for linear algebra primitives
- AMD ROCm ecosystem

**Files** (from preliminary survey):
- `pdmrg_gpu_with_loader.cpp`: Main GPU DMRG implementation
- `tensor_contractions_working.cpp`: Tensor contraction kernels
- Various test files for hipTensor validation

**Current Issues** (from earlier context):
- Validation failures and segmentation faults reported in previous work
- Requires dedicated GPU debugging phase (not part of current CPU audit)

**Phase 5 Task**: Map files and create `PDMRG2_BACKEND_MAP.md` **without modifying code**

---

## Benchmark Implications

### Current Benchmark Labeling Issues

1. **"PDMRG np=1" is misleading**:
   - Current behavior: returns quimb DMRG2 warmup result
   - Actual execution: PDMRG algorithm does not run
   - **Fix required**: Phase 2 must add metadata exposing early returns

2. **"A2DMRG np=1" is misleading**:
   - Same issue as PDMRG
   - **Fix required**: Phase 2 must add metadata exposing early returns

3. **Boundary merge optimization status undocumented**:
   - PDMRG always runs with `skip_opt=True`
   - Performance impact unknown
   - **Fix required**: Phase 3 must investigate and document

### Honest Benchmark Metadata (Phase 2 Goal)

All benchmark results must include:
- `warmup_used`: boolean
- `warmup_sweeps`: integer
- `early_return`: boolean (did the function return warmup result?)
- `skip_opt_flag`: boolean (for PDMRG)
- `canonicalization_enabled`: boolean (for A2DMRG)
- `actual_algorithm_executed`: string (e.g., "quimb DMRG2", "PDMRG parallel sweeps")

---

## Usage Guidelines

### When to Use Each Implementation

| Use Case | Recommended Implementation | Rationale |
|----------|---------------------------|-----------|
| **Correctness validation** | quimb DMRG2 | Stable, well-tested reference |
| **Serial production runs** | quimb DMRG2 | No MPI overhead, mature codebase |
| **Small parallel runs (L < 30)** | PDMRG | Validated, works despite known issues |
| **Large parallel runs (L > 30)** | A2DMRG or PDMRG | Both validated; A2DMRG may be faster at large L |
| **GPU acceleration** | None yet | PDMRG-GPU is experimental |
| **Future BLAS-3 optimization** | PDMRG2 | Specification complete, awaiting implementation |

### Running Each Implementation

See README.md for detailed usage instructions.

---

## Revision History

- **2026-03-06**: Initial taxonomy document created (Phase 1 of CPU audit)
  - Documented PDMRG np=1 early return issue
  - Documented A2DMRG np=1 early return issue
  - Documented PDMRG skip_opt flag status
  - Documented A2DMRG canonicalization policy
  - Clarified PDMRG2 status (specification only, not GPU implementation)
  - Identified PDMRG-GPU as experimental

---

## Next Steps (Remaining Audit Phases)

- **Phase 2**: Fix benchmark semantics and add metadata schema
- **Phase 3**: Investigate PDMRG issues, create `PDMRG_STATUS.md`
- **Phase 4**: Audit A2DMRG, create `A2DMRG_STATUS.md`
- **Phase 5**: Map GPU files, create `PDMRG2_BACKEND_MAP.md`
- **Phase 6**: Create honest validation suite with explicit metadata
