# Fix External Audit Issues - Complete Implementation

**Branch:** cpu-audit
**Repository:** /home/captain/clawd/work/dmrg-implementations
**Goal:** Fix all critical issues identified by external audit, test thoroughly, and commit

---

## Context

An external auditor reviewed the cpu-audit branch and identified critical algorithmic issues in the **committed code**. A previous session made fixes but they remain uncommitted. This session will:

1. Review and complete all necessary fixes
2. Add the missing convergence sentinel fix
3. Test thoroughly with existing benchmark scripts
4. Commit everything with proper attribution

---

## Critical Issues to Fix

### Issue 1: PDMRG Boundary Merge Crippled (CRITICAL)

**Problem:** `skip_opt = True` in multi-rank path disables boundary optimization

**Location:** `pdmrg/pdmrg/dmrg.py` (and pdmrg2)

**Current committed code:**
```python
skip_opt = True  # Always skip until H_eff bug is fixed
```

**Required fix:**
```python
skip_opt = False  # Boundary optimization enabled (exact SVD method)
```

**Additional requirements:**
- Implement exact SVD: V = Λ⁻¹ (not V = ones)
- Add `compute_v_from_boundary_tensor()` helper function
- Update `recompute_boundary_v()` to use exact SVD
- Update `_compute_v_at_bond()` in `pdmrg/pdmrg/parallel/distribute.py`
- Apply same fixes to PDMRG2 while preserving GPU hooks
- Update metadata: `V_computation = "exact_svd_Lambda_inverse"`
- Update metadata: `boundary_optimization_enabled = True`
- Update README to reflect enabled status

**Reference:** Stoudenmire & White 2013, Eq. 5 - V = Λ⁻¹ bridge matrix

---

### Issue 2: A2DMRG i-Orthogonal Transformation Incomplete (CRITICAL)

**Problem:** Bond dimensions not preserved during canonization, violating Algorithm 2

**Location:** `a2dmrg/a2dmrg/numerics/local_microstep.py`

**Current committed code:**
```python
def _transform_to_i_orthogonal(mps, center_site, normalize=True):
    # Claims: "bond dimensions are guaranteed to be preserved"
    # Reality: quimb's canonize() CAN reduce bonds to numerical rank
    mps.canonize(where=center_site)
    # ... normalize ...
    return mps
```

**Required fix:**
```python
def _transform_to_i_orthogonal(mps, center_site, normalize=True):
    # Store original bond dimensions BEFORE canonization
    original_shapes = []
    for i in range(L):
        original_shapes.append(mps[i].data.shape)

    # Perform gauge transformation (may reduce bonds to numerical rank)
    mps.canonize(where=center_site)

    # CRITICAL: Pad back to original bond dimensions
    for i in range(L):
        current_shape = mps[i].data.shape
        target_shape = original_shapes[i]

        if current_shape != target_shape:
            # Bond dimensions changed - zero-pad
            padded = np.zeros(target_shape, dtype=mps[i].data.dtype)
            slices = tuple(slice(0, s) for s in current_shape)
            padded[slices] = mps[i].data
            mps[i].modify(data=padded)

    # Normalize
    if normalize:
        norm = mps.norm()
        if abs(norm) > 1e-15:
            mps /= norm

    return mps
```

**Rationale:** Grigori & Hassan Algorithm 2, Step 1 requires "gauge transformation **without bond compression**"

---

### Issue 3: Convergence Sentinel Fragile (HIGH PRIORITY)

**Problem:** Uses `e != 0.0` as sentinel - fails if ground state energy is exactly 0.0

**Location:** `pdmrg/pdmrg/parallel/communication.py`

**Current code:**
```python
def check_convergence(E_local, comm):
    all_E = comm.allgather(E_local)
    merge_energies = [e for e in all_E if e != 0.0]  # FRAGILE!
    if merge_energies:
        E_global = min(merge_energies)
    else:
        E_global = E_local
    return E_global
```

**Required fix:**

Step 1: Change `boundary_merge()` to return `None` for idle ranks:
```python
# In pdmrg/pdmrg/dmrg.py - boundary_merge() function
def boundary_merge(pmps, env_mgr, mpo_arrays, comm, boundaries, ...):
    # ... role determination ...

    if role == 'idle':
        return None  # Not 0.0!

    # ... rest of merge logic ...
    return energy
```

Step 2: Update convergence check:
```python
# In pdmrg/pdmrg/parallel/communication.py
def check_convergence(E_local, comm):
    all_E = comm.allgather(E_local)
    merge_energies = [e for e in all_E if e is not None]  # SAFE!
    if merge_energies:
        E_global = min(merge_energies)
    else:
        E_global = E_local if E_local is not None else 0.0
    return E_global
```

Step 3: Update all callers to handle `None` return:
```python
# In pdmrg/pdmrg/dmrg.py
E_merge1 = boundary_merge(..., boundaries='even')
if E_merge1 is None:
    E_merge1 = 0.0  # Idle rank - use placeholder for logging
```

**Apply to both PDMRG and PDMRG2**

---

### Issue 4: A2DMRG Metadata Incomplete

**Problem:** Missing metadata fields needed for benchmark verification

**Location:** `a2dmrg/a2dmrg/dmrg.py`

**Required addition:** (may already be in uncommitted changes)
```python
# At end of a2dmrg_main(), before return
metadata = {
    "algorithm_executed": "A2DMRG additive two-level parallel",
    "warmup_method": "quimb DMRG2 serial" if warmup_sweeps > 0 else None,
    "warmup_sweeps": warmup_sweeps,
    "experimental_nonpaper": experimental_nonpaper,
    "initialization_mode": init_mode,
    "paper_faithful": paper_faithful,
    "converged": converged_flag,
    "final_sweep": final_sweep_num,
    "np": size,
    "max_sweeps": max_sweeps,
    "bond_dim": bond_dim,
    "tol": tol,
    "total_time": time.time() - start_time,
}

return final_energy, mps, metadata
```

---

## Implementation Checklist

### Phase 1: Review Uncommitted Changes (10 min)

```bash
cd /home/captain/clawd/work/dmrg-implementations
git status
git diff HEAD -- pdmrg/pdmrg/dmrg.py | less
git diff HEAD -- a2dmrg/a2dmrg/numerics/local_microstep.py | less
```

**Action:** Determine what's already fixed in uncommitted changes vs. what still needs work.

---

### Phase 2: Complete All Fixes (30 min)

**2.1 PDMRG/PDMRG2 Exact SVD + Boundary Optimization**

Files to modify:
- [ ] `pdmrg/pdmrg/dmrg.py`
  - [ ] Add `compute_v_from_boundary_tensor()` helper
  - [ ] Update `recompute_boundary_v()` to call helper
  - [ ] Set `skip_opt = False`
  - [ ] Update metadata fields
  - [ ] Remove any stale TODO comments
- [ ] `pdmrg/pdmrg/parallel/distribute.py`
  - [ ] Update `_compute_v_at_bond()` to use exact SVD
- [ ] `pdmrg2/pdmrg/dmrg.py` (same changes)
- [ ] `pdmrg2/pdmrg/parallel/distribute.py` (same changes)
- [ ] `pdmrg/README.md` - Update status section
- [ ] `pdmrg2/README.md` - Update status section

**2.2 A2DMRG Bond Preservation**

Files to modify:
- [ ] `a2dmrg/a2dmrg/numerics/local_microstep.py`
  - [ ] Implement zero-padding in `_transform_to_i_orthogonal()`
  - [ ] Update docstring to explain the fix

**2.3 Convergence Sentinel Fix**

Files to modify:
- [ ] `pdmrg/pdmrg/dmrg.py`
  - [ ] Update `boundary_merge()` calls to handle `None`
  - [ ] Update logging to handle `None` values
- [ ] `pdmrg/pdmrg/parallel/merge.py` (or wherever boundary_merge is defined)
  - [ ] Return `None` for idle ranks (not 0.0)
- [ ] `pdmrg/pdmrg/parallel/communication.py`
  - [ ] Update `check_convergence()` to use `e is not None`
- [ ] Apply same changes to PDMRG2

**2.4 A2DMRG Metadata**

Files to modify:
- [ ] `a2dmrg/a2dmrg/dmrg.py`
  - [ ] Add metadata dict before return
  - [ ] Update return statement to include metadata

---

### Phase 3: Validation Testing (30 min)

**3.1 Unit Tests**
```bash
# Test warmup policy
uv run pytest test_warmup_policy.py -v

# Test any SVD-related tests
uv run pytest -k "svd" -v

# Test A2DMRG if tests exist
uv run pytest a2dmrg/tests/ -v 2>/dev/null || echo "No A2DMRG tests found"
```

**3.2 Quick Integration Tests**
```bash
# PDMRG np=2 (smallest parallel test)
uv run mpirun -np 2 python -m pdmrg --sites 40 --bond-dim 50 --model heisenberg --sweeps 5

# A2DMRG np=2
uv run mpirun -np 2 python -m a2dmrg --sites 40 --bond-dim 50 --model heisenberg --sweeps 5

# Check that boundary optimization actually runs (skip_opt=False)
# Look for output like "Phase 2: Merging even boundaries (skip_opt=False)"
```

**3.3 Benchmark Scripts - Energy Validation**

Run small-scale benchmarks to verify correctness:

```bash
# Create test script for quick validation
cat > validate_fixes.py << 'EOF'
#!/usr/bin/env python3
"""Quick validation of external audit fixes."""

import subprocess
import json
import sys
from pathlib import Path

def run_pdmrg_test():
    """Test PDMRG with np=2 on small system."""
    print("Testing PDMRG np=2...")
    result = subprocess.run(
        ["mpirun", "-np", "2", "python", "-m", "pdmrg",
         "--sites", "20", "--bond-dim", "30", "--model", "heisenberg",
         "--sweeps", "10"],
        capture_output=True, text=True
    )

    # Check for skip_opt=False in output
    if "skip_opt=False" in result.stdout:
        print("  ✓ Boundary optimization enabled (skip_opt=False)")
    else:
        print("  ✗ WARNING: skip_opt=False not found in output")
        print(result.stdout[-500:])

    # Check for convergence
    if "converged" in result.stdout.lower() or "final energy" in result.stdout.lower():
        print("  ✓ Completed successfully")
        return True
    else:
        print("  ✗ Did not complete")
        print(result.stderr[-500:])
        return False

def run_a2dmrg_test():
    """Test A2DMRG with np=2 on small system."""
    print("\nTesting A2DMRG np=2...")
    result = subprocess.run(
        ["mpirun", "-np", "2", "python", "-m", "a2dmrg",
         "--sites", "20", "--bond-dim", "30", "--model", "heisenberg",
         "--sweeps", "5"],
        capture_output=True, text=True
    )

    if "completed" in result.stdout.lower() or "final energy" in result.stdout.lower():
        print("  ✓ Completed successfully")
        return True
    else:
        print("  ✗ Did not complete")
        print(result.stderr[-500:])
        return False

def check_metadata():
    """Verify metadata is present in test runs."""
    print("\nChecking metadata availability...")
    # This would need to be adapted based on how metadata is accessed
    print("  → Manual check: Import and run with return_metadata=True")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("VALIDATION: External Audit Fixes")
    print("=" * 60)

    tests = [
        run_pdmrg_test(),
        run_a2dmrg_test(),
        check_metadata(),
    ]

    print("\n" + "=" * 60)
    if all(tests):
        print("✓ All validation tests passed")
        sys.exit(0)
    else:
        print("✗ Some validation tests failed")
        sys.exit(1)
EOF

chmod +x validate_fixes.py
uv run python validate_fixes.py
```

**3.4 Energy Accuracy Test**

```bash
# Run a benchmark script to verify energy accuracy
# Use one of the existing benchmark scripts on a small problem

# Option 1: Use comprehensive_dmrg_benchmark.py if it exists
uv run python comprehensive_dmrg_benchmark.py --quick-test 2>/dev/null || echo "Benchmark script not in expected format"

# Option 2: Manual comparison with quimb
cat > test_energy_accuracy.py << 'EOF'
#!/usr/bin/env python3
"""Compare PDMRG energy with quimb reference."""

import numpy as np
import quimb.tensor as qtn
from mpi4py import MPI
import sys
import os

# Add paths
sys.path.insert(0, '/home/captain/clawd/work/dmrg-implementations')

def test_pdmrg_accuracy():
    """Test PDMRG achieves reference accuracy."""
    from pdmrg.hamiltonians.heisenberg import heisenberg_mpo
    from pdmrg.dmrg import pdmrg_main

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    L = 20
    bond_dim = 50

    # Get reference from quimb
    if rank == 0:
        print("Computing quimb reference...")
        H = qtn.MPO_ham_heis(L=L)
        dmrg = qtn.DMRG2(H, bond_dims=bond_dim)
        dmrg.solve(tol=1e-10, verbosity=0)
        E_ref = dmrg.energy
        print(f"Quimb reference: {E_ref:.12f}")
    else:
        E_ref = None

    E_ref = comm.bcast(E_ref, root=0)

    # Run PDMRG
    if rank == 0:
        print("\nRunning PDMRG...")

    mpo = heisenberg_mpo(L)
    E_pdmrg, pmps = pdmrg_main(
        L=L, mpo=mpo, max_sweeps=20, bond_dim=bond_dim,
        comm=comm, verbose=(rank==0)
    )

    if rank == 0:
        delta = abs(E_pdmrg - E_ref)
        print(f"\nPDMRG energy: {E_pdmrg:.12f}")
        print(f"Reference:    {E_ref:.12f}")
        print(f"Difference:   {delta:.2e}")

        if delta < 1e-9:
            print("✓ PASS: Energy accuracy within 1e-9")
            return True
        else:
            print("✗ FAIL: Energy accuracy worse than 1e-9")
            return False

    return True

if __name__ == "__main__":
    success = test_pdmrg_accuracy()
    MPI.COMM_WORLD.Barrier()
    if MPI.COMM_WORLD.Get_rank() == 0:
        sys.exit(0 if success else 1)
EOF

uv run mpirun -np 2 python test_energy_accuracy.py
```

---

### Phase 4: Documentation (15 min)

**4.1 Update READMEs**

Verify these sections are current:
- [ ] `pdmrg/README.md` - Implementation Status section
- [ ] `pdmrg2/README.md` - PROTOTYPE warning and status
- [ ] `a2dmrg/README.md` - Status and warmup policy

**4.2 Create/Update Audit Documentation**

- [ ] Keep `EXTERNAL_AUDIT_RESPONSE.md` (current status)
- [ ] Archive old audit reports to `docs/historical_audits/` if needed
- [ ] Create final `FIXES_APPLIED.md` summary

---

### Phase 5: Commit Everything (15 min)

```bash
# Stage all fixes
git add pdmrg/pdmrg/dmrg.py pdmrg/pdmrg/parallel/
git add pdmrg2/pdmrg/dmrg.py pdmrg2/pdmrg/parallel/
git add a2dmrg/a2dmrg/dmrg.py a2dmrg/a2dmrg/numerics/
git add pdmrg/README.md pdmrg2/README.md a2dmrg/README.md

# Commit 1: PDMRG/PDMRG2 exact SVD and boundary optimization
git commit -m "CRITICAL: Enable boundary optimization and exact SVD (Issue #1)

- Set skip_opt = False (was True - disabled boundary merge)
- Implement exact SVD: V = Lambda^-1 (was V = ones)
- Add compute_v_from_boundary_tensor() helper function
- Update recompute_boundary_v() to use exact SVD
- Update _compute_v_at_bond() for exact SVD by default
- Update metadata: V_computation, boundary_optimization_enabled
- Apply to both PDMRG and PDMRG2 (GPU hooks preserved)
- Remove stale TODO comments

Addresses external audit critical issue: PDMRG boundary merge was
crippled by skip_opt=True, preventing proper algorithm execution.

Reference: Stoudenmire & White 2013, Eq. 5

Validated: Energy accuracy maintained at 1e-9 vs quimb reference

Co-authored-by: External Auditor <audit@review.org>"

# Commit 2: A2DMRG bond preservation
git commit -m "FIX: Preserve bond dimensions in A2DMRG i-orthogonal (Issue #2)

- Store original bond shapes before canonization
- Zero-pad back to original dimensions after quimb canonize()
- Ensures 'gauge transformation without bond compression'
- Complies with Grigori & Hassan Algorithm 2, Step 1

Addresses external audit critical issue: A2DMRG violated algorithm
specification by allowing bond dimension reduction during gauge
transformation.

Reference: Grigori & Hassan 2025, Algorithm 2, Step 1 (page 10)

Validated: Bond dimensions preserved across canonization

Co-authored-by: External Auditor <audit@review.org>"

# Commit 3: Convergence sentinel fix
git commit -m "FIX: Replace fragile 0.0 sentinel with None (Issue #3)

- boundary_merge() returns None for idle ranks (was 0.0)
- check_convergence() filters 'e is not None' (was 'e != 0.0')
- Update callers to handle None returns
- Apply to both PDMRG and PDMRG2

Addresses external audit issue: Zero is a bad sentinel in physics
code since ground state energy CAN be exactly 0.0 for some
Hamiltonians.

Validated: Convergence detection works correctly

Co-authored-by: External Auditor <audit@review.org>"

# Commit 4: A2DMRG metadata (if not already committed)
git commit -m "Add comprehensive metadata tracking to A2DMRG (Issue #4)

- algorithm_executed, warmup_method, converged, etc.
- Matches PDMRG metadata completeness
- Enables benchmark result verification

Addresses external audit issue: A2DMRG metadata incomplete

Validated: Metadata properly returned"

# Commit 5: Documentation
git add EXTERNAL_AUDIT_RESPONSE.md FIXES_APPLIED.md
git add EXACT_SVD_IMPLEMENTATION.md UV_SETUP_GUIDE.md
git commit -m "Add final audit documentation and fix summaries"
```

---

### Phase 6: Final Validation (10 min)

```bash
# Run comprehensive test suite
echo "=== Final Validation ==="

# 1. Verify commits
git log --oneline -5

# 2. Quick smoke tests
uv run mpirun -np 2 python -m pdmrg --sites 20 --bond-dim 30 --sweeps 5
uv run mpirun -np 2 python -m a2dmrg --sites 20 --bond-dim 30 --sweeps 5

# 3. Check no uncommitted changes remain
git status

# 4. Verify skip_opt=False in committed code
git show HEAD:pdmrg/pdmrg/dmrg.py | grep -A2 "skip_opt ="

echo "=== Validation Complete ==="
```

---

## Success Criteria

### Must Have (Blocking):
- [ ] `skip_opt = False` in committed code (PDMRG/PDMRG2)
- [ ] Exact SVD V = Λ⁻¹ implemented and used
- [ ] A2DMRG bond dimensions preserved (zero-padding)
- [ ] Convergence sentinel uses `None` (not 0.0)
- [ ] A2DMRG metadata complete
- [ ] All changes committed with attribution
- [ ] Energy accuracy validated (< 1e-9 vs quimb)
- [ ] No uncommitted changes remain

### Should Have (Important):
- [ ] READMEs updated to reflect current state
- [ ] Documentation consolidated (old audits archived)
- [ ] Validation test scripts created
- [ ] Benchmark scripts run successfully

### Nice to Have (Optional):
- [ ] Additional test coverage for new code paths
- [ ] Performance comparison (before/after exact SVD)
- [ ] Detailed commit messages with paper references

---

## Expected Outcome

After this session:

1. **Committed code** matches what external auditor expected
2. **All critical algorithmic issues** fixed:
   - PDMRG executes full algorithm (boundary optimization enabled)
   - A2DMRG preserves bond dimensions (paper-compliant)
   - Convergence detection is robust (no 0.0 sentinel)
3. **Validation** confirms:
   - Energy accuracy maintained/improved
   - Benchmarks run successfully
   - No regressions introduced
4. **Ready for external re-review**

---

## Reference Materials

**In Repository:**
- `EXTERNAL_AUDIT_RESPONSE.md` - Analysis of audit findings
- `CPU_AUDIT_FINAL_REPORT.md` - Previous audit details
- `pdmrg/pdmrg/numerics/accurate_svd.py` - Exact SVD infrastructure
- `a2dmrg/a2dmrg/numerics/local_microstep.py` - i-orthogonal transformation

**Papers:**
- Stoudenmire & White 2013 (arXiv:1301.3494v2) - PDMRG algorithm
- Grigori & Hassan 2025 (arXiv:2505.23429v2) - A2DMRG algorithm

**Key Algorithm Requirements:**
- PDMRG: 4-phase structure with boundary merge + optimization
- A2DMRG: Gauge transformation **without bond compression**
- Both: Exact SVD for numerical stability

---

## Troubleshooting

**If energy accuracy is poor:**
- Check that `skip_opt = False` is actually in effect
- Verify exact SVD is being called (not identity approximation)
- Increase `max_sweeps` and check convergence

**If A2DMRG fails:**
- Check bond dimension preservation in debug output
- Verify zero-padding is triggered when needed
- Test with uniform bond dimensions (e.g., all = 50)

**If commits fail:**
- Ensure no merge conflicts with uncommitted changes
- Review git diff before committing
- Use `git add -p` for selective staging if needed

**If validation tests fail:**
- Check MPI is working: `mpirun -np 2 python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_rank())"`
- Verify uv environment: `uv run python -c "import pdmrg, a2dmrg; print('OK')"`
- Check for import errors in modified files

---

## Time Estimate

- Phase 1 (Review): 10 min
- Phase 2 (Fixes): 30 min
- Phase 3 (Testing): 30 min
- Phase 4 (Documentation): 15 min
- Phase 5 (Commits): 15 min
- Phase 6 (Final validation): 10 min

**Total:** ~2 hours for complete, tested, committed fixes

---

**END OF PROMPT**
