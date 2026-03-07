# Benchmark Metadata Schema

## Purpose

This document defines the **required metadata fields** for all DMRG benchmark results. The goal is to ensure **semantic honesty**: benchmark outputs must explicitly expose all execution path decisions, early returns, disabled optimizations, and warmup strategies.

## Motivation

The current benchmark scripts have hidden semantics that make results misleading:

1. **np=1 Early Returns**: When PDMRG or A2DMRG run with `np=1`, they return the serial warmup energy without executing the parallel algorithm. Benchmark results labeled "PDMRG np=1" are actually measuring quimb DMRG2 performance.

2. **Disabled Optimizations**: PDMRG always runs with `skip_opt=True` due to an unresolved H_eff bug, but this is not exposed in benchmark outputs.

3. **Canonicalization Policy**: A2DMRG intentionally skips canonicalization, but benchmarks don't document this.

4. **Hard-Coded Paths**: Benchmarks contain hard-coded absolute paths that break portability.

## Required Metadata Fields

### Core Algorithm Execution Metadata

All benchmark results **must** include these fields in their JSON output:

```python
{
  "method": str,           # e.g., "PDMRG_np4", "A2DMRG_np2", "quimb_DMRG2"
  "energy": float,         # Ground state energy
  "time": float,           # Wall-clock time in seconds
  "success": bool,         # Whether the run succeeded

  # REQUIRED NEW FIELDS (Phase 2):
  "metadata": {
    # Execution path information
    "algorithm_executed": str,          # What actually ran? e.g., "quimb DMRG2 warmup (early return)", "PDMRG parallel sweeps", "A2DMRG additive"
    "early_return": bool,               # Did the function return early without running the main algorithm?
    "early_return_reason": str | null,  # Why? e.g., "np=1 with warmup enabled", null if no early return

    # Warmup information
    "warmup_used": bool,                # Was warmup initialization used?
    "warmup_sweeps": int,               # Number of warmup sweeps (0 if no warmup)
    "warmup_method": str | null,        # e.g., "quimb DMRG2 serial", "parallel rank-local", null if no warmup

    # Algorithm-specific flags
    "skip_opt": bool | null,            # PDMRG only: boundary merge optimization status (true = disabled)
    "canonicalization_enabled": bool | null,  # A2DMRG only: was canonicalization performed?
    "random_init": bool | null,         # Was random initialization used instead of warmup?

    # MPI configuration
    "np": int,                          # Number of MPI processes
    "mpi_used": bool,                   # Was MPI actually used?

    # Convergence information
    "converged": bool,                  # Did the solver converge within tolerance?
    "final_sweep": int | null,          # Final sweep number
    "max_sweeps": int,                  # Maximum allowed sweeps

    # System configuration
    "system_size": int,                 # L (chain length or number of sites)
    "bond_dim": int,                    # Bond dimension
    "dtype": str,                       # Data type, e.g., "float64", "complex128"
    "tolerance": float,                 # Solver convergence tolerance

    # Environment
    "hostname": str,                    # Machine where benchmark ran
    "timestamp": str,                   # ISO 8601 timestamp
    "python_version": str,              # e.g., "3.13.1"
    "numpy_version": str,               # NumPy version
    "quimb_version": str | null         # quimb version if used
  }
}
```

## Semantic Honesty Rules

### Rule 1: Explicit Early Returns

If a function returns early without executing its main algorithm (e.g., PDMRG np=1 returning warmup result), the metadata **must** set:

```python
"early_return": True
"early_return_reason": "np=1 with warmup enabled"
"algorithm_executed": "quimb DMRG2 warmup (early return)"
```

**Never label this as "PDMRG" or "A2DMRG" without clarifying the early return.**

### Rule 2: Expose Disabled Optimizations

If an optimization path is disabled (e.g., PDMRG `skip_opt=True`), the metadata **must** document it:

```python
"skip_opt": True  # PDMRG boundary merge optimization is disabled
```

### Rule 3: Document Canonicalization

For A2DMRG, the metadata **must** document whether canonicalization was performed:

```python
"canonicalization_enabled": False  # Intentionally disabled for coarse-space compatibility
```

### Rule 4: Distinguish Warmup from Algorithm

If warmup was used, the metadata **must** distinguish warmup energy from final energy:

- Report both `warmup_energy` and `final_energy` separately, OR
- Clearly indicate in `algorithm_executed` whether the reported energy is from warmup or the main algorithm

### Rule 5: Absolute Paths Forbidden

Benchmark scripts **must not** contain hard-coded absolute paths. Use:

- Relative paths from repository root
- Environment variables (e.g., `$DMRG_ROOT`)
- Path detection via `os.path.dirname(__file__)`

## Implementation Status (Phase 2 Goals)

### Current State (2026-03-06)

❌ **Benchmark scripts do NOT include required metadata**
❌ **Hard-coded paths present in all benchmark scripts**
❌ **np=1 early returns not exposed**
❌ **skip_opt status not exposed**
❌ **Canonicalization status not exposed**

### Target State (After Phase 2)

✅ All benchmark scripts output full metadata schema
✅ No hard-coded absolute paths
✅ Early returns clearly documented
✅ All execution path decisions exposed
✅ Benchmark results can be accurately interpreted

## Example: Honest PDMRG np=1 Result

### Before (Misleading)

```json
{
  "method": "PDMRG_np1",
  "energy": -5.373916515211431,
  "time": 0.42
}
```

**Problem**: This looks like PDMRG ran, but it's actually quimb DMRG2 warmup!

### After (Honest)

```json
{
  "method": "PDMRG_np1",
  "energy": -5.373916515211431,
  "time": 0.42,
  "metadata": {
    "algorithm_executed": "quimb DMRG2 warmup (early return)",
    "early_return": true,
    "early_return_reason": "np=1 with warmup enabled",
    "warmup_used": true,
    "warmup_sweeps": 5,
    "warmup_method": "quimb DMRG2 serial",
    "skip_opt": null,
    "canonicalization_enabled": null,
    "np": 1,
    "mpi_used": true,
    "system_size": 12,
    "bond_dim": 20,
    "dtype": "float64",
    "tolerance": 1e-10
  }
}
```

**Result**: Now it's clear that this is NOT a PDMRG parallel sweep result!

## Example: Honest PDMRG np=4 Result

```json
{
  "method": "PDMRG_np4",
  "energy": -5.373916515211431,
  "time": 1.83,
  "metadata": {
    "algorithm_executed": "PDMRG parallel sweeps",
    "early_return": false,
    "early_return_reason": null,
    "warmup_used": true,
    "warmup_sweeps": 5,
    "warmup_method": "quimb DMRG2 serial",
    "skip_opt": true,
    "canonicalization_enabled": null,
    "np": 4,
    "mpi_used": true,
    "converged": true,
    "final_sweep": 8,
    "max_sweeps": 30,
    "system_size": 12,
    "bond_dim": 20,
    "dtype": "float64",
    "tolerance": 1e-10
  }
}
```

**Key points**:
- `algorithm_executed` clearly states "PDMRG parallel sweeps"
- `skip_opt: true` exposes that boundary merge optimization is disabled
- `warmup_used: true` shows warmup was used for initialization

## Example: Honest A2DMRG np=2 Result

```json
{
  "method": "A2DMRG_np2",
  "energy": -5.373916515211431,
  "time": 2.15,
  "metadata": {
    "algorithm_executed": "A2DMRG additive two-level",
    "early_return": false,
    "early_return_reason": null,
    "warmup_used": true,
    "warmup_sweeps": 5,
    "warmup_method": "parallel rank-local quimb DMRG2",
    "skip_opt": null,
    "canonicalization_enabled": false,
    "np": 2,
    "mpi_used": true,
    "converged": true,
    "final_sweep": 30,
    "max_sweeps": 30,
    "system_size": 12,
    "bond_dim": 20,
    "dtype": "float64",
    "tolerance": 1e-10
  }
}
```

**Key points**:
- `canonicalization_enabled: false` documents the design decision
- `warmup_method` distinguishes parallel rank-local warmup from serial warmup

## Validation Checklist

Before marking Phase 2 complete, verify:

- [ ] All benchmark scripts output the full metadata schema
- [ ] No hard-coded absolute paths remain in benchmark scripts
- [ ] PDMRG np=1 results clearly indicate early return
- [ ] A2DMRG np=1 results clearly indicate early return
- [ ] PDMRG results expose skip_opt status
- [ ] A2DMRG results expose canonicalization status
- [ ] All results include system configuration (L, D, dtype, tol)
- [ ] All results include environment information (hostname, timestamp, versions)
- [ ] JSON schema is consistent across all benchmarks

## Integration with Implementation Functions

To support this metadata, the implementation functions must be modified to **return metadata alongside results**:

### PDMRG Signature Change

**Before:**
```python
def pdmrg_main(...) -> Tuple[float, MPS]:
    return energy, mps
```

**After (Phase 3):**
```python
def pdmrg_main(...) -> Tuple[float, MPS, Dict]:
    metadata = {
        "algorithm_executed": "PDMRG parallel sweeps",
        "early_return": False,
        "skip_opt": True,
        # ... etc
    }
    return energy, mps, metadata
```

### A2DMRG Signature Change

Similar modification needed for `a2dmrg_main`.

## References

- `IMPLEMENTATION_MATRIX.md` - Full implementation taxonomy
- `pdmrg/pdmrg/dmrg.py` lines 736-739 - np=1 early return code
- `a2dmrg/a2dmrg/dmrg.py` lines 333-334 - np=1 early return code
- Phase 3 (PDMRG_STATUS.md) - PDMRG-specific issues
- Phase 4 (A2DMRG_STATUS.md) - A2DMRG-specific issues

---

**Document Status**: ✅ Phase 2 metadata schema complete (2026-03-06)
**Next**: Update benchmark scripts to output this metadata
