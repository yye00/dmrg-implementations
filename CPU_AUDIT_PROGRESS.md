# CPU Audit Progress Report

**Mission**: Six-phase CPU-side cleanup and documentation audit for DMRG implementations

**Status**: Phases 1-2 infrastructure complete; Phases 3-6 pending

**Last Updated**: 2026-03-06

---

## Phase 1: Implementation Taxonomy ✅ COMPLETE

### Deliverables

1. **`IMPLEMENTATION_MATRIX.md`** ✅
   - Comprehensive implementation taxonomy table
   - Distinguishes reference baselines (quimb DMRG1, DMRG2) from in-house implementations (PDMRG, A2DMRG, PDMRG2)
   - Documents PDMRG2 as "specification only" (not GPU implementation)
   - Identifies PDMRG-GPU as experimental research code
   - Includes detailed descriptions, known issues, and code line references

2. **`README.md` Updates** ✅
   - Clarified taxonomy: in-house vs. reference baselines
   - Reconciled PDMRG2 documentation contradiction (README claimed "GPU-optimized," but pdmrg2_gpu.md states "CPU optimization phase")
   - Added "Known Issues and Current Status" section with warnings about np=1 early returns, skip_opt flag, and canonicalization
   - Added directory structure with IMPLEMENTATION_MATRIX.md reference
   - Expanded Development section to clarify PDMRG2 CPU plan vs. experimental GPU path

### Key Findings Documented

1. **PDMRG Issues**:
   - np=1 early return (line 736-739): returns warmup energy without running PDMRG algorithm
   - skip_opt always True (line 807): boundary merge optimization permanently disabled due to H_eff bug
   - Impact on performance: undocumented

2. **A2DMRG Issues**:
   - np=1 early return (line 333-334): returns warmup energy without running A2DMRG algorithm
   - Canonicalization intentionally skipped (line 348-356): design decision for coarse-space compatibility

3. **Documentation Contradictions**:
   - README.md claimed PDMRG2 has "GPU-optimized linear algebra kernels"
   - pdmrg2_gpu.md states it's a "CPU optimization phase" preparing for future GPU porting
   - Resolution: PDMRG2 is specification-only, CPU GEMM optimization plan, not GPU implementation

---

## Phase 2: Benchmark Semantics ⚙️ IN PROGRESS

### Deliverables

1. **`BENCHMARK_METADATA_SCHEMA.md`** ✅
   - Defines required metadata fields for semantic honesty
   - Mandates explicit documentation of:
     - Early returns
     - Algorithm execution path
     - Warmup usage and method
     - Disabled optimizations (skip_opt)
     - Canonicalization status
     - MPI configuration
     - System parameters
     - Environment information
   - Provides examples of "before/after" benchmark outputs showing honest metadata

2. **`benchmarks/benchmark_utils.py`** ✅
   - Dynamic path detection (eliminates hard-coded absolute paths)
   - `get_repo_root()`, `get_implementation_paths()`, `get_venv_python()`
   - `get_mpi_env()` for portable MPI environment setup
   - `create_benchmark_metadata()` helper implementing full metadata schema
   - Can be used by all benchmark scripts going forward

3. **README.md Update** ✅
   - Added "Benchmark Metadata" subsection linking to BENCHMARK_METADATA_SCHEMA.md
   - Notes that benchmark scripts are being updated (Phase 2 in progress)

### Remaining Tasks (Phase 2)

❌ **Update benchmark scripts to use benchmark_utils.py**:
   - Replace hard-coded paths with `get_venv_python()`, `get_mpi_env()`
   - Use `create_benchmark_metadata()` for result output
   - Files to update:
     - `benchmarks/heisenberg_benchmark.py`
     - `benchmarks/heisenberg_long_benchmark.py`
     - `a2dmrg/benchmarks/josephson_correctness_benchmark.py`
     - Others as needed

❌ **Note**: Full metadata implementation requires modifying PDMRG and A2DMRG functions (Phases 3-4) to return metadata. For now, benchmark scripts can populate metadata based on input parameters.

### Hard-Coded Paths Identified

**Files with hard-coded paths**:
1. `benchmarks/heisenberg_benchmark.py`:
   - Lines 94, 140, 146-147, 174, 218, 250, 291, 464
2. `benchmarks/heisenberg_long_benchmark.py`:
   - Lines 84, 132, 138, 164, 212, 218, 244, 291, 297, 435
3. `a2dmrg/benchmarks/josephson_correctness_benchmark.py`:
   - Lines 79, 88, 112, 151, 184, 339

**Patterns**:
- Hard-coded `/home/captain/clawd/work/dmrg-implementations/...`
- Hard-coded `/usr/lib64/openmpi/bin` and `/usr/lib64/openmpi/lib`
- Hard-coded `/tmp/` for temp scripts (acceptable)
- Hard-coded output paths (should use `get_default_output_path()`)

---

## Phase 3: PDMRG Correctness ⏳ PENDING

### Planned Tasks

1. **Investigate skip_opt flag**:
   - Root cause analysis of "H_eff spurious eigenvalue problem"
   - Document findings in `PDMRG_STATUS.md`
   - Either fix the bug or document why it can't be fixed

2. **Fix np=1 early return**:
   - Option A: Remove early return, let PDMRG run with np=1
   - Option B: Keep early return but expose it clearly in metadata
   - Option C: Make it configurable via flag

3. **Create `PDMRG_STATUS.md`**:
   - Comprehensive status document for PDMRG
   - Algorithmic correctness assessment
   - Performance characterization
   - Known limitations and caveats

4. **Modify `pdmrg_main()` function**:
   - Return metadata tuple: `(energy, mps, metadata)`
   - Populate metadata with all required fields from schema

### Critical Files

- `pdmrg/pdmrg/dmrg.py` (lines 1-949)
- `pdmrg/pdmrg/mps/canonical.py` (boundary_merge function)

---

## Phase 4: A2DMRG Semantics ⏳ PENDING

### Planned Tasks

1. **Audit np=1 behavior**:
   - Decide whether to keep or remove early return
   - Ensure metadata exposes execution path

2. **Document canonicalization policy**:
   - Formalize the design decision in `A2DMRG_STATUS.md`
   - Explain why canonicalization is incompatible with additive correction

3. **Create `A2DMRG_STATUS.md`**:
   - Comprehensive status document for A2DMRG
   - Algorithmic correctness assessment
   - Performance scaling analysis (why L > 20 is needed)
   - Canonicalization policy rationale

4. **Modify `a2dmrg_main()` function**:
   - Return metadata tuple: `(energy, mps, metadata)`
   - Populate metadata with all required fields from schema

### Critical Files

- `a2dmrg/a2dmrg/dmrg.py` (lines 1-757)

---

## Phase 5: GPU Path Discovery ⏳ PENDING

### Planned Tasks

1. **Map GPU files** (read-only, no modifications):
   - Survey `pdmrg-gpu/*.cpp` files
   - Identify tensor contraction kernels
   - Identify hipTensor/rocBLAS usage patterns
   - Document MPI+GPU parallelization strategy

2. **Create `PDMRG2_BACKEND_MAP.md`**:
   - File inventory
   - Technology stack (hipTensor, rocBLAS, ROCm)
   - Architecture overview
   - Current validation status
   - Known issues from earlier GPU work (segfaults, validation failures)

### Critical Files

- `pdmrg-gpu/*.cpp`
- `pdmrg-gpu/*.h`

---

## Phase 6: Testing ⏳ PENDING

### Planned Tasks

1. **Create honest validation suite**:
   - New comprehensive benchmark comparing all methods
   - Explicit metadata for every run
   - Clear distinction between warmup and algorithm results
   - Document which code path was executed for each test

2. **Test outputs**:
   - Run all benchmarks with updated metadata
   - Generate honest comparison tables
   - Validate that metadata correctly reflects execution

3. **Unresolved risks list**:
   - Document any remaining issues that couldn't be resolved
   - Note any performance anomalies
   - Identify technical debt for future work

---

## Summary of Completed Work

### Files Created

1. ✅ `IMPLEMENTATION_MATRIX.md` - Comprehensive taxonomy (Phase 1)
2. ✅ `BENCHMARK_METADATA_SCHEMA.md` - Metadata specification (Phase 2)
3. ✅ `benchmarks/benchmark_utils.py` - Path and metadata helpers (Phase 2)
4. ✅ `CPU_AUDIT_PROGRESS.md` - This document

### Files Modified

1. ✅ `README.md` - Clarified taxonomy, added known issues, added benchmark metadata section

### Infrastructure Ready

- ✅ Implementation taxonomy is clear and documented
- ✅ Metadata schema is defined
- ✅ Utility functions are available for benchmarks
- ✅ Known issues are documented with code line references
- ✅ PDMRG2/GPU contradictions are resolved

### Remaining Work

- ❌ Update benchmark scripts to use benchmark_utils (Phase 2)
- ❌ Investigate and fix PDMRG issues (Phase 3)
- ❌ Document A2DMRG semantics (Phase 4)
- ❌ Map GPU files (Phase 5)
- ❌ Create honest validation suite (Phase 6)

---

## Next Steps

**Immediate** (Complete Phase 2):
1. Update `benchmarks/heisenberg_benchmark.py` to use `benchmark_utils.py`
2. Update `benchmarks/heisenberg_long_benchmark.py` to use `benchmark_utils.py`
3. Update `a2dmrg/benchmarks/josephson_correctness_benchmark.py` to use `benchmark_utils.py`

**Then** (Phase 3):
1. Deep dive into `pdmrg/pdmrg/dmrg.py`
2. Investigate H_eff spurious eigenvalue problem
3. Decide on np=1 early return strategy
4. Create PDMRG_STATUS.md

---

## Principles Maintained

✅ **Surgical changes only** - No refactoring, only documentation and path fixes
✅ **Semantic honesty** - All issues clearly documented
✅ **No degradation** - Existing working behavior preserved
✅ **Expose caveats** - Early returns, disabled optimizations documented
✅ **No premature optimization** - Not touching GPU code yet

---

**Document Status**: Phase 1 complete, Phase 2 infrastructure complete
**Next Action**: Update benchmark scripts to eliminate hard-coded paths and use metadata helpers
