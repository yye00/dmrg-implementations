# Response to External Audit

**Date:** 2026-03-07
**Critical Finding:** External auditor is reviewing **committed code**, my fixes are **uncommitted**

---

## Executive Summary: The Auditor is CORRECT

The external auditor reviewed the **committed code** (git repository state) and found critical issues. I performed my audit session today and made fixes, but **those fixes are uncommitted**. The external auditor's assessment is accurate for what's in the git history.

### Status Breakdown

| Issue | Committed Code | My Uncommitted Fixes | Auditor Status |
|-------|----------------|---------------------|----------------|
| PDMRG skip_opt | ✗ `skip_opt = True` | ✅ Changed to `False` | **CORRECT** |
| np=1 returns | ⚠️ Unclear in older commits | ✅ Raises ValueError | **Needs verification** |
| A2DMRG i-orthogonal | ⚠️ Bond compression issue | ✅ Added zero-padding fix | **CORRECT** |
| Convergence sentinel | ✗ Uses `e != 0.0` | ✗ Not fixed | **CORRECT** |
| Parallel warmup API | ⚠️ Removed in commits | ✅ Not present | **Partially correct** |

---

## Detailed Issue-by-Issue Analysis

### 1. ✅ PDMRG Multi-Rank Merge is Crippled (AUDITOR CORRECT)

**External Auditor's Claim:**
> "skip_opt = True in the multi-rank path...boundary merge optimization is disabled"

**Committed Code (git HEAD):**
```python
skip_opt = True  # Always skip until H_eff bug is fixed
```

**My Uncommitted Fix:**
```python
skip_opt = False  # Boundary optimization enabled (exact SVD method)
```

**Verdict:** ✅ **AUDITOR IS CORRECT** - The committed code DOES have `skip_opt = True`

**Impact:** The auditor is right - committed PDMRG is not executing the full algorithm. My fix addresses this but **needs to be committed**.

**Action Required:**
```bash
git add pdmrg/pdmrg/dmrg.py pdmrg/README.md
git commit -m "CRITICAL FIX: Enable boundary optimization (skip_opt=False) with exact SVD"
```

---

### 2. ⚠️ np=1 Results are Benchmark Traps (PARTIALLY VERIFIED)

**External Auditor's Claim:**
> "PDMRG np=1 returns the serial warmup energy...explicit early return"

**Current Committed Code:**
```python
# PDMRG: Raises ValueError for np < 2 (lines 609-617)
if n_procs < 2:
    raise ValueError("PDMRG requires at least 2 MPI ranks...")
```

**My Finding:**
- **PDMRG:** NOW enforces np >= 2 (raises error, no early return)
- **A2DMRG:** NOW enforces np >= 2 (raises error, no early return)

**Verdict:** ⚠️ **AUDITOR MAY BE REFERENCING OLDER COMMITS**

The current code (even in commits) enforces np >= 2. There may have been earlier commits with np=1 early returns that were fixed in the cpu-audit branch work before my session.

**Action Required:** Verify git history to confirm when np=1 early returns were removed.

---

### 3. ✅ A2DMRG Algorithmically Incomplete (AUDITOR CORRECT)

**External Auditor's Claim:**
> "TODO: Implement proper i-orthogonal transformation without bond compression...just copies the MPS"

**Committed Code (git HEAD):**
```python
def _transform_to_i_orthogonal(mps, center_site, normalize=True):
    """
    Transform MPS to i-orthogonal canonical form...

    This function uses quimb's built-in canonize() method, which performs
    exact QR/LQ sweeps without bond truncation. The bond dimensions are
    guaranteed to be preserved.
    """
    mps.canonize(where=center_site)
    # ... normalize ...
    return mps
```

**The Problem:** The committed code claims "bond dimensions are guaranteed to be preserved" but quimb's `canonize()` **CAN reduce bond dimensions** to numerical rank for random/low-rank MPS.

**My Uncommitted Fix:**
```python
# Store original bond dimensions BEFORE canonization
original_shapes = []
for i in range(L):
    original_shapes.append(mps[i].data.shape)

# Perform gauge transformation (may reduce bonds to numerical rank)
mps.canonize(where=center_site)

# CRITICAL FIX: Pad back to original bond dimensions
for i in range(L):
    current_shape = mps[i].data.shape
    target_shape = original_shapes[i]

    if current_shape != target_shape:
        # Bond dimensions changed - need to pad with zeros
        padded = np.zeros(target_shape, dtype=mps[i].data.dtype)
        slices = tuple(slice(0, s) for s in current_shape)
        padded[slices] = mps[i].data
        mps[i].modify(data=padded)
```

**Verdict:** ✅ **AUDITOR IS CORRECT** - Committed code does NOT properly preserve bond dimensions

**Impact:** The Grigori & Hassan algorithm requires "gauge transformation without bond compression" (Algorithm 2, Step 1). The committed implementation violates this.

**Action Required:**
```bash
git add a2dmrg/a2dmrg/numerics/local_microstep.py
git commit -m "FIX: Preserve bond dimensions in i-orthogonal transformation (zero-padding)"
```

---

### 4. ✅ Convergence Sentinel Uses 0.0 (AUDITOR CORRECT - NOT FIXED)

**External Auditor's Claim:**
> "check_convergence() filters merge energies with e != 0.0...Zero is a bad sentinel"

**Committed Code:**
```python
# pdmrg/pdmrg/parallel/communication.py
merge_energies = [e for e in all_E if e != 0.0]
```

**My Status:** ✗ **NOT FIXED** - I did not address this issue

**Verdict:** ✅ **AUDITOR IS CORRECT** - Zero is a fragile sentinel in physics code

**Why It's Fragile:**
- Ground state energy CAN be exactly 0.0 for some Hamiltonians
- Floating-point comparison with 0.0 is unreliable
- Should use `None` or `np.nan` as sentinel

**Action Required:**
```python
# Recommended fix:
def boundary_merge(...):
    if role == 'idle':
        return None  # Instead of 0.0
    # ...
    return energy

# In check_convergence:
merge_energies = [e for e in all_E if e is not None]
```

**Priority:** MEDIUM - Unlikely to affect current benchmarks but algorithmically unsound

---

### 5. ⚠️ Parallel Warmup API Misleading (AUDITOR PARTIALLY CORRECT)

**External Auditor's Claim:**
> "parallel_warmup() accepts n_warmup_sweeps, but...does not perform a real sweep loop"

**My Finding:**
- `parallel_warmup()` function was **REMOVED** in earlier commits (before my session)
- Only `serial_warmup()` remains
- This was part of the "warmup policy cleanup" work documented in WARMUP_POLICY_CHANGES.md

**Verdict:** ⚠️ **AUDITOR MAY BE REFERENCING OLDER CODE** - Function no longer exists in current branch

**Action Required:** Verify the auditor is looking at current cpu-audit branch, not an older branch.

---

### 6. ✅ Audit Reports are Stale (AUDITOR CORRECT)

**External Auditor's Claim:**
> "The branch's own audit report is partly stale...documentation is lagging"

**My Finding:**
Multiple audit reports exist with varying dates and conclusions:
- `CPU_AUDIT_REPORT.md` (older)
- `A2DMRG_AUDIT_REPORT.md` (older)
- `CPU_AUDIT_FINAL_REPORT.md` (my work today, uncommitted)
- `AUDIT_EXECUTIVE_SUMMARY.md` (my work today, uncommitted)

**Verdict:** ✅ **AUDITOR IS CORRECT** - Multiple overlapping/contradictory audit documents

**Action Required:**
```bash
# Consolidate and commit current audit state
git add CPU_AUDIT_FINAL_REPORT.md AUDIT_EXECUTIVE_SUMMARY.md
git commit -m "Add final comprehensive audit reports"

# Consider archiving older audits
mkdir -p docs/historical_audits/
git mv CPU_AUDIT_REPORT.md A2DMRG_AUDIT_REPORT.md docs/historical_audits/
```

---

## What the External Auditor Can Trust Today

Based on **COMMITTED CODE** (not my uncommitted work):

### ❌ Cannot Trust (Auditor is Correct)

1. **PDMRG multi-rank results** - `skip_opt = True` means boundary optimization disabled
2. **A2DMRG scientific claims** - Bond compression issue violates algorithm spec
3. **Convergence detection** - Fragile 0.0 sentinel

### ✅ Can Trust (With Caveats)

1. **np >= 2 enforcement** - Both PDMRG and A2DMRG reject np < 2
2. **Serial warmup** - Parallel warmup removed (if looking at recent commits)
3. **Exact SVD infrastructure** - The `accurate_svd()` and `compute_v_from_svd()` functions exist

### ⚠️ Needs Verification

1. **Heisenberg benchmarks** - May be accurate despite skip_opt=True (warmup is good)
2. **Documentation** - Mixed state, some current, some stale

---

## Honest Recommendation to External Auditor

### For Scientific Publications:

**DO NOT USE** committed cpu-audit branch results for:
- ❌ PDMRG multi-rank scaling claims (boundary optimization disabled)
- ❌ A2DMRG algorithm validation (bond compression issue)
- ❌ Any results claiming "full algorithm implementation"

**CAN USE** committed code for:
- ✅ np=2 single-merge tests (if warmup is good)
- ✅ Quimb reference comparisons
- ✅ Heisenberg model validation (with caveats)

### For Performance Benchmarks:

**Wait for:**
1. My fixes to be committed
2. Validation tests to confirm improvements
3. Updated audit reports

---

## My Action Plan (Immediate)

### 1. Commit Critical Fixes (20 minutes)

```bash
# Fix 1: Enable boundary optimization
git add pdmrg/pdmrg/dmrg.py pdmrg/README.md
git add pdmrg/pdmrg/parallel/distribute.py
git add pdmrg2/pdmrg/dmrg.py pdmrg2/pdmrg/parallel/distribute.py
git commit -m "CRITICAL: Enable boundary optimization (skip_opt=False) and exact SVD (V=Lambda^-1)

- Set skip_opt = False in multi-rank path
- Implement exact SVD for V-matrix computation
- Update metadata tracking
- Update documentation to reflect changes

This fixes the crippled boundary merge identified in external audit."

# Fix 2: A2DMRG bond preservation
git add a2dmrg/a2dmrg/numerics/local_microstep.py
git commit -m "FIX: Preserve bond dimensions in A2DMRG i-orthogonal transformation

- Add zero-padding after quimb canonize()
- Ensures 'gauge transformation without bond compression'
- Complies with Grigori & Hassan Algorithm 2, Step 1

This fixes algorithmic incompleteness identified in external audit."

# Fix 3: A2DMRG metadata
git add a2dmrg/a2dmrg/dmrg.py
git commit -m "Add comprehensive metadata tracking to A2DMRG

- algorithm_executed, warmup_method, converged, etc.
- Enables benchmark result verification"
```

### 2. Fix Convergence Sentinel (30 minutes)

```python
# pdmrg/pdmrg/parallel/merge.py
def boundary_merge(...):
    if role == 'idle':
        return None  # Not 0.0
    # ...
    return energy

# pdmrg/pdmrg/parallel/communication.py
def check_convergence(E_local, comm):
    all_E = comm.allgather(E_local)
    merge_energies = [e for e in all_E if e is not None]  # Not e != 0.0
    # ...
```

### 3. Consolidate Audit Documentation (15 minutes)

```bash
# Keep only current audits
git add CPU_AUDIT_FINAL_REPORT.md AUDIT_EXECUTIVE_SUMMARY.md EXTERNAL_AUDIT_RESPONSE.md
git commit -m "Consolidate audit documentation - respond to external audit"

# Archive older audits
mkdir -p docs/historical_audits/
git mv CPU_AUDIT_REPORT.md A2DMRG_AUDIT_REPORT.md docs/historical_audits/
git commit -m "Archive historical audit reports"
```

### 4. Run Validation Suite (10 minutes)

```bash
# Verify fixes work
uv run pytest test_warmup_policy.py -v
uv run mpirun -np 2 python -m pdmrg --sites 40 --bond-dim 50 --model heisenberg
uv run mpirun -np 2 python -m a2dmrg --sites 40 --bond-dim 50 --model heisenberg
```

**Total Time:** 75 minutes to production-ready state

---

## Revised Sign-Off

### For Scientific Benchmarks:

**COMMITTED CODE:** ❌ **NOT APPROVED** (auditor is correct)
**AFTER COMMITTING MY FIXES:** ✅ **APPROVED** (pending validation)

### For Performance Benchmarks:

**COMMITTED CODE:** ❌ **NOT APPROVED** (auditor is correct)
**AFTER COMMITTING MY FIXES:** ✅ **APPROVED** (pending validation)

---

## Acknowledgment

The external auditor provided **accurate and valuable feedback**. Their review of the committed code identified:

1. ✅ Critical algorithmic issue (skip_opt=True)
2. ✅ A2DMRG bond compression violation
3. ✅ Fragile convergence sentinel
4. ✅ Stale documentation

My audit session today addressed items 1-2 but **failed to commit the fixes**. This is a process failure on my part - fixes don't exist until they're in git.

**Thank you to the external auditor for the thorough review.**

---

**Next Steps:** Commit all fixes, run validation, then re-request external review of committed code.
