# PDMRG Implementation Status Report

**Document Version**: 1.0
**Date**: 2026-03-06
**Phase**: 3 - CPU Audit

---

## Executive Summary

PDMRG (Parallel DMRG) is a functional MPI-parallel DMRG implementation with **two critical caveats** that affect result interpretation:

1. **Boundary Merge Optimization Disabled**: The optimization path (`skip_optimization=False`) is permanently disabled due to an unresolved H_eff spurious eigenvalue problem
2. **np=1 Early Return**: When running with a single MPI process and warmup enabled, PDMRG returns the serial warmup energy without executing the parallel algorithm

**Current Status**: ✅ Produces correct results for np > 1, but convergence may be suboptimal due to disabled optimization

---

## Algorithm Execution Paths

### Path 1: np=1 with Warmup (Serial Warmup Early Return)

**Code Location**: `pdmrg/pdmrg/dmrg.py` lines 730-739

**Execution**:
1. Serial warmup using quimb DMRG2 (lines 681-705)
2. **Early return** with warmup energy (line 739)
3. PDMRG parallel algorithm **never executes**

**Metadata**:
```python
{
    "algorithm_executed": "quimb DMRG2 warmup (early return)",
    "early_return": True,
    "early_return_reason": "np=1 with warmup enabled",
    "warmup_used": True,
    "pdmrg_sweeps_executed": False
}
```

**Rationale** (from code comment, line 730-734):
> "For np=1 with serial warmup: the warmup (quimb DMRG2 at tol=1e-12) is already optimal. The local_sweep path calls optimize_two_site, which shares the same H_eff eigensolver that produces spurious eigenvalues in certain gauge configurations (hence skip_opt=True for the multi-rank merge path). On a serial chain the additional sweeps therefore introduce numerical drift rather than improving accuracy."

**Implication**: Benchmarks labeled "PDMRG np=1" are actually measuring **quimb DMRG2 performance**, not PDMRG.

---

### Path 2: np=1 with Random Initialization (True Serial PDMRG)

**Code Location**: `pdmrg/pdmrg/dmrg.py` lines 753-772

**Execution**:
1. Random MPS initialization (line 714-719)
2. Standard DMRG sweeps using `local_sweep()` (line 758)
3. No boundary merge (single rank)

**Metadata**:
```python
{
    "algorithm_executed": "PDMRG serial sweeps",
    "early_return": False,
    "random_init": True,
    "warmup_used": False,
    "pdmrg_sweeps_executed": True
}
```

**Note**: This path uses the same H_eff eigensolver that has spurious eigenvalue issues (see Root Cause Analysis below).

---

### Path 3: np > 1 (True Parallel PDMRG)

**Code Location**: `pdmrg/pdmrg/dmrg.py` lines 774-830

**Execution**:
1. Serial or parallel warmup initialization (lines 681-726)
2. Parallel sweeps loop:
   - Local sweeps on each rank (line 785-789)
   - Recompute boundary V matrices (lines 796-803)
   - Boundary merge with **skip_optimization=True** (lines 808-823)
3. Convergence check via global energy (line 826-830)

**Metadata**:
```python
{
    "algorithm_executed": "PDMRG parallel sweeps",
    "early_return": False,
    "warmup_used": True,  # or False
    "skip_opt": True,      # ALWAYS TRUE
    "pdmrg_sweeps_executed": True,
    "boundary_optimization_enabled": False
}
```

**Current Limitation**: Boundary merge optimization is **always skipped** due to H_eff bug (see next section).

---

## Root Cause Analysis: H_eff Spurious Eigenvalue Problem

### Problem Statement

The boundary merge optimization (`skip_optimization=False`) produces spurious eigenvalues that cause convergence failures or incorrect energies.

### Code Location

- **Disabled**: `pdmrg/pdmrg/dmrg.py` line 807
  ```python
  skip_opt = True  # Always skip until H_eff bug is fixed
  ```

- **Comment**: Line 806
  ```python
  # Skip optimization due to spurious H_eff eigenvalues (TODO: fix H_eff bug)
  ```

- **Additional context**: Lines 732-734
  ```python
  # H_eff eigensolver that produces spurious eigenvalues in certain gauge configurations
  ```

### Technical Analysis

#### What Happens During Boundary Merge

When two ranks meet at a shared boundary bond, `merge_boundary_tensors()` performs:

1. **Form two-site wavefunction** (`pdmrg/pdmrg/parallel/merge.py` line 63-66):
   ```python
   V_psi_right = V[:, None, None] * psi_right
   theta = np.einsum('ija,akl->ijkl', psi_left, V_psi_right)
   ```

   Where `V = Lambda^-1` (inverse singular values) from `compute_v_from_svd()` (`pdmrg/pdmrg/numerics/accurate_svd.py` line 64-79):
   ```python
   def compute_v_from_svd(S, regularization=1e-12):
       return 1.0 / np.clip(S, regularization, None)
   ```

2. **Optimize theta** (line 75-78):
   ```python
   energy, theta_opt = optimize_two_site(
       L_env, R_env, W_left, W_right, theta,
       max_iter=max_iter, tol=tol
   )
   ```

   This calls scipy eigsh to find the ground state of H_eff (effective Hamiltonian).

3. **Split via SVD** (line 84-100)

#### Root Cause Hypothesis

**Primary Issue: V Matrix Conditioning**

When singular values `S` approach the truncation threshold:
- `S` becomes small (e.g., `S ~ 1e-10`)
- `V = 1/S` becomes large (e.g., `V ~ 1e10`)
- `theta = psi_left · diag(V) · psi_right` amplifies numerical errors

**Secondary Issue: Gauge Inconsistency**

The two ranks evolve their MPS tensors independently during local sweeps:
- Rank A's right boundary tensor is in some gauge
- Rank B's left boundary tensor is in some gauge
- These gauges may not be consistent
- The V matrix bridges them, but amplifies gauge mismatch

**Eigensolver Failure Mode**

When `theta` is poorly conditioned:
- H_eff = L_env · W_left · W_right · R_env has condition number issues
- scipy.sparse.linalg.eigsh may converge to a **spurious eigenvalue** instead of the true ground state
- The spurious eigenvalue is typically:
  - Significantly different from the expected energy
  - Not physically meaningful
  - Caused by numerical artifacts in the poorly-conditioned theta

#### Why skip_opt=True Works

When `skip_optimization=True` (`pdmrg/pdmrg/parallel/merge.py` line 69-73):
```python
# Just compute the energy without optimization
H_theta = apply_heff(L_env, R_env, W_left, W_right, theta)
energy = float(np.real(np.vdot(theta.ravel(), H_theta.ravel()) / np.vdot(theta.ravel(), theta.ravel())))
theta_opt = theta
```

- No eigensolver is invoked
- The current `theta` (formed from existing psi_left, V, psi_right) is used as-is
- Energy is computed via Rayleigh quotient
- No spurious eigenvalue risk

**Trade-off**: Convergence is likely slower because boundary tensors are not optimized, only swept.

---

## Impact Assessment

### What Works

✅ **Correctness for np > 1**: PDMRG produces correct ground state energies (validated at L=12, L=48 for Heisenberg model)

✅ **Complex dtype support**: Works correctly for complex128 (Josephson junction arrays)

✅ **MPI parallelization**: Domain decomposition and boundary exchange work correctly

✅ **Warmup initialization**: Serial warmup using quimb DMRG2 provides good initial state

### What Is Degraded

⚠️ **Convergence speed**: With `skip_opt=True`, boundary merge does not optimize, so convergence may require more sweeps

⚠️ **Semantic honesty**: np=1 early return is not exposed in current benchmark outputs

### What Is Broken

❌ **Boundary merge optimization**: `skip_optimization=False` produces spurious eigenvalues

❌ **np=1 PDMRG path**: Not a true PDMRG execution when warmup is used (early return)

### Are Current Multi-Rank Results Trustworthy?

**YES**, with caveats:

1. **Energy accuracy**: Results agree with quimb DMRG2 to within tolerance (1e-10)
2. **Correctness**: No evidence of wrong answers, only potentially suboptimal convergence
3. **Caveat**: Convergence might be faster with working boundary optimization, but current results are not incorrect

**Validation Evidence**:
- `benchmarks/heisenberg_benchmark.py`: PDMRG np=2,4,8 all pass (ΔE < 1e-10)
- `benchmarks/heisenberg_long_benchmark.py`: L=48 validation passes
- `a2dmrg/benchmarks/josephson_correctness_benchmark.py`: Complex128 validation passes

---

## Recommended Fixes

### Immediate Fix (Surgical)

**Goal**: Make execution path explicit without breaking anything

1. **Modify `pdmrg_main()` to return metadata** (see Implementation section below)

2. **Update benchmark scripts** to consume and report metadata

3. **Document skip_opt status** in all benchmark outputs

**Benefit**: Semantic honesty without code risk

---

### Short-Term Fix (Moderate Risk)

**Goal**: Attempt to fix the H_eff conditioning issue

**Approach A: Improved V Regularization**

Current:
```python
V = 1.0 / np.clip(S, 1e-12, None)
```

Proposed:
```python
# Use Tikhonov regularization: V_i = S_i / (S_i^2 + lambda^2)
# instead of V_i = 1 / max(S_i, lambda)
lambda_reg = 1e-10
V = S / (S**2 + lambda_reg**2)
```

**Rationale**: Tikhonov regularization smoothly handles small S without creating huge V values.

**Risk**: Changes boundary merge behavior, requires validation.

**Approach B: Gauge Synchronization**

Before boundary merge:
1. Canonize both boundary tensors to a consistent gauge (e.g., left-canonical)
2. Recompute V to ensure gauge consistency
3. Then perform merge

**Rationale**: Eliminates gauge mismatch as a source of conditioning issues.

**Risk**: Requires careful implementation of cross-rank gauge canonization.

---

### Long-Term Fix (Research)

**Goal**: Replace Lanczos eigensolver with more robust method

**Options**:
1. **Block Davidson** (LOBPCG): More stable for poorly-conditioned H_eff
2. **Preconditioned eigensolver**: Use approximate H_eff inverse as preconditioner
3. **Subspace iteration**: Less sensitive to conditioning

**Rationale**: If the eigensolver itself is more robust, the conditioning issues may be tolerable.

**Alignment**: This is part of the PDMRG2 specification (`pdmrg2_gpu.md`), which plans to replace Lanczos with LOBPCG for GEMM efficiency.

---

## Implementation: Metadata Return from pdmrg_main()

### Current Signature

```python
def pdmrg_main(L, mpo, *, bond_dim, max_sweeps, tol, comm,
               bond_dim_warmup=None, n_warmup_sweeps=0,
               dtype='float64', verbose=False) -> Tuple[float, ParallelMPS]:
    ...
    return energy, pmps
```

### Proposed Signature

```python
def pdmrg_main(L, mpo, *, bond_dim, max_sweeps, tol, comm,
               bond_dim_warmup=None, n_warmup_sweeps=0,
               dtype='float64', verbose=False,
               return_metadata=False) -> Union[
                   Tuple[float, ParallelMPS],
                   Tuple[float, ParallelMPS, Dict]
               ]:
    ...
    if return_metadata:
        metadata = {
            "algorithm_executed": algorithm_executed,
            "early_return": early_return,
            "early_return_reason": early_return_reason,
            "warmup_used": warmup_used,
            "warmup_sweeps": n_warmup_sweeps,
            "warmup_method": warmup_method,
            "skip_opt": skip_opt if n_procs > 1 else None,
            "random_init": random_init_flag,
            "np": n_procs,
            "converged": converged,
            "final_sweep": final_sweep,
            "max_sweeps": max_sweeps,
        }
        return energy, pmps, metadata
    else:
        return energy, pmps  # Backward compatible
```

### Implementation Plan

1. Add `return_metadata` parameter (default False for backward compatibility)
2. Track execution path variables throughout pdmrg_main():
   - `algorithm_executed` (string)
   - `early_return` (bool)
   - `early_return_reason` (string or None)
   - `warmup_used`, `warmup_method`, `random_init_flag`
   - `skip_opt` (for np > 1 path)
   - `converged`, `final_sweep`
3. At return points, construct metadata dict and return it if requested
4. Update benchmarks to pass `return_metadata=True` and consume metadata

---

## Diagnostic Tests

### Test 1: Skip-Opt Comparison

**File**: `pdmrg/tests/test_skip_opt_diagnostic.py` (created in Phase 3)

**Purpose**: Demonstrate the failure mode when `skip_optimization=False`

**Usage**:
```bash
# Safe path (skip_opt=True)
mpirun -np 2 python tests/test_skip_opt_diagnostic.py --skip-opt

# Trigger issue (skip_opt=False) - may fail
mpirun -np 2 python tests/test_skip_opt_diagnostic.py --no-skip-opt

# Run both for comparison
mpirun -np 2 python tests/test_skip_opt_diagnostic.py --both
```

**Expected Behavior**:
- `--skip-opt`: Success, energy matches warmup
- `--no-skip-opt`: May fail or produce significantly different energy

### Test 2: V Matrix Conditioning

**Purpose**: Measure V matrix condition number at boundaries

**Implementation** (TODO):
```python
def test_v_conditioning(L=20, bond_dim=20, n_procs=2):
    """
    Measure V = Lambda^-1 at rank boundaries after warmup.
    Report condition number and correlation with merge energy drift.
    """
    ...
```

---

## Current Workarounds

### 1. Skip-Opt Always Enabled

**File**: `pdmrg/pdmrg/dmrg.py` line 807

**Workaround**: `skip_opt = True` is hard-coded

**Status**: **Active** - this is the current safe path

**Impact**: Convergence may be slower, but results are correct

### 2. np=1 Early Return

**File**: `pdmrg/pdmrg/dmrg.py` lines 736-739

**Workaround**: Return warmup energy without running PDMRG

**Status**: **Active** - prevents numerical drift from local sweeps with same H_eff issue

**Impact**: "PDMRG np=1" benchmarks are actually quimb DMRG2

---

## Open Questions

1. **Quantify convergence degradation**: How many extra sweeps does skip_opt=True require compared to skip_opt=False (if it worked)?

2. **V matrix statistics**: What is the typical condition number of V at boundaries? Does it correlate with system size L or bond dimension?

3. **Gauge consistency**: Can we measure the gauge mismatch between ranks at boundaries?

4. **Alternative formulations**: Can we avoid V = Lambda^-1 entirely? (e.g., use a different MPS representation)

---

## References

### Code Files

- `pdmrg/pdmrg/dmrg.py`: Main PDMRG driver
  - Lines 730-739: np=1 early return
  - Lines 774-830: np > 1 parallel sweep loop
  - Line 807: skip_opt = True hard-coded
- `pdmrg/pdmrg/parallel/merge.py`: Boundary merge implementation
  - Lines 13-102: `merge_boundary_tensors()`
- `pdmrg/pdmrg/numerics/eigensolver.py`: Two-site optimization
  - Lines 13-57: `optimize_two_site()` (uses scipy eigsh)
- `pdmrg/pdmrg/numerics/effective_ham.py`: H_eff construction
  - Lines 21-81: `apply_heff()` (tensor contractions)
  - Lines 84-117: `build_heff_operator()` (LinearOperator wrapper)
- `pdmrg/pdmrg/numerics/accurate_svd.py`: SVD and V computation
  - Lines 64-79: `compute_v_from_svd()` (V = 1/S with regularization)

### Related Documents

- `IMPLEMENTATION_MATRIX.md`: Implementation taxonomy
- `BENCHMARK_METADATA_SCHEMA.md`: Required metadata fields
- `pdmrg2_gpu.md`: PDMRG2 specification (LOBPCG proposal)

---

## Revision History

- **2026-03-06**: Initial version (Phase 3 of CPU audit)
  - Root cause analysis of H_eff spurious eigenvalue problem
  - Documented all execution paths (np=1 warmup, np=1 random, np>1)
  - Proposed immediate, short-term, and long-term fixes
  - Created diagnostic test: `test_skip_opt_diagnostic.py`

---

## Next Steps (Phase 3 Deliverables)

- [x] Root-cause analysis documented
- [x] PDMRG_STATUS.md created
- [x] Diagnostic test created
- [ ] Implement metadata return from pdmrg_main()
- [ ] Update benchmarks to use metadata
- [ ] Decide on np=1 strategy
- [ ] Commit all changes to cpu-audit branch

**Status**: Phase 3 documentation complete; implementation changes pending
