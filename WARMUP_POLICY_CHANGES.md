# Warmup Policy Changes - cpu-audit Branch (2026-03-07)

## Summary

Completed comprehensive warmup policy cleanup for algorithmic fidelity and benchmark hygiene:

1. **PDMRG & PDMRG2**: Removed parallel warmup entirely
2. **A2DMRG**: Changed default to paper-faithful mode (warmup_sweeps=0)
3. **All three**: Already enforce np>=2 (from previous audit work)

---

## Files Changed

### Core Implementation Files

**PDMRG** (`pdmrg/pdmrg/dmrg.py`):
- Removed `parallel_warmup()` function (77 lines)
- Removed `parallel_warmup_flag` parameter from `pdmrg_main()`
- Removed parallel warmup elif branch
- Removed `--parallel-warmup` CLI argument
- Updated metadata to remove parallel warmup references
- **Result:** Serial warmup only (rank 0 → scatter to all ranks)

**PDMRG2** (`pdmrg2/pdmrg/dmrg.py`):
- Same changes as PDMRG
- Removed `parallel_warmup()` function (78 lines)
- Removed all parallel warmup references
- **Result:** Serial warmup only (consistent with PDMRG)

**A2DMRG** (`a2dmrg/a2dmrg/dmrg.py`):
- Changed `warmup_sweeps` default: 2 → 0 (paper-faithful)
- Added `experimental_nonpaper` parameter (default: False)
- Added warmup bounds validation:
  - warmup_sweeps=0: Paper-faithful (default)
  - warmup_sweeps=1-2: Allowed (bounded experimental)
  - warmup_sweeps>2: Requires experimental_nonpaper=True
- Added metadata fields: `initialization_mode`, `paper_faithful_mode`, `experimental_nonpaper`
- Added validation error messages explaining paper-faithful policy

### Documentation Files

**PDMRG README** (`pdmrg/README.md`):
- Updated status section to note parallel warmup removal
- Explained serial warmup policy
- Clarified np>=2 requirement

**PDMRG2 README** (`pdmrg2/README.md`):
- Added recent changes section
- Noted parallel warmup removal
- Still marked as prototype (awaits PDMRG full validation)

**A2DMRG README** (`a2dmrg/README.md`):
- Completely rewrote initialization section
- Changed from "Critical: Warm-Up Requirement" to "Initialization Policy (Paper-Faithful Default)"
- Explained paper-faithful default (warmup_sweeps=0)
- Documented warmup bounds and experimental modes
- Updated status section to highlight new default

### Test Files

**Created** (`test_warmup_policy.py`):
- Verifies parallel_warmup_flag removed from PDMRG
- Verifies parallel_warmup_flag removed from PDMRG2
- Verifies A2DMRG warmup_sweeps default = 0
- Verifies A2DMRG experimental_nonpaper parameter exists
- Tests A2DMRG warmup bounds enforcement

---

## Behavior Changes

### PDMRG & PDMRG2

**Before:**
- `--parallel-warmup` CLI flag available
- `parallel_warmup_flag=True` → each rank initialized independently
- Mixed initialization strategies possible

**After:**
- `--parallel-warmup` CLI flag removed
- Only serial warmup: rank 0 runs quimb DMRG2, then scatters to all ranks
- Consistent initialization across all processors
- **Breaking change:** Any scripts using `--parallel-warmup` or `parallel_warmup_flag=True` will fail

### A2DMRG

**Before:**
- Default: `warmup_sweeps=2` (always ran serial warmup)
- No bounds on warmup_sweeps
- Behavior described as matching paper, but actually deviated

**After:**
- Default: `warmup_sweeps=0` (paper-faithful, no serial warmup)
- Hard cap: warmup_sweeps ∈ [0, 2] without experimental flag
- Warmup > 2 requires explicit `experimental_nonpaper=True`
- Metadata tracks paper_faithful_mode, initialization_mode
- **Breaking change:** Code expecting default warmup=2 will now get warmup=0
- **Migration:** Explicitly set `warmup_sweeps=2` if you want old behavior (marks run as experimental)

---

## Compatibility Breakages

### Scripts/Tests Affected

1. **Any code using `--parallel-warmup` CLI flag**
   - **Fix:** Remove the flag, use serial warmup only

2. **Any code calling with `parallel_warmup_flag=True`**
   - **Fix:** Remove the parameter, serial warmup is automatic

3. **A2DMRG code expecting default warmup=2**
   - **Fix:** Explicitly set `warmup_sweeps=2` if needed
   - **Note:** This marks the run as experimental (non-paper mode)

4. **Benchmark scripts testing np=1**
   - **Already fixed:** Previous audit enforced np>=2 everywhere

### Migration Examples

**PDMRG/PDMRG2 - Old code:**
```python
# OLD - will fail
energy, mps = pdmrg_main(L, mpo, parallel_warmup_flag=True, ...)
```

**PDMRG/PDMRG2 - New code:**
```python
# NEW - parallel_warmup_flag removed, serial warmup is default
energy, mps = pdmrg_main(L, mpo, ...)
```

**A2DMRG - Old code expecting warmup=2:**
```python
# OLD - relied on default warmup=2
energy, mps = a2dmrg_main(L, mpo, ...)  # Got warmup=2 implicitly
```

**A2DMRG - New code (two options):**
```python
# Option 1: Use paper-faithful default (recommended)
energy, mps = a2dmrg_main(L, mpo, ...)  # Gets warmup=0 (paper-faithful)

# Option 2: Explicitly request experimental warmup
energy, mps = a2dmrg_main(L, mpo, warmup_sweeps=2, ...)  # Experimental mode
```

---

## Acceptance Checklist

- [x] PDMRG has no parallel warmup mode
- [x] PDMRG2 has no parallel warmup mode
- [x] Both use serial warmup then distribute/scatter
- [x] PDMRG, PDMRG2, A2DMRG all reject np=1 (already done in previous audit)
- [x] A2DMRG defaults to warmup_sweeps=0
- [x] A2DMRG warmup is bounded and explicitly non-paper beyond default
- [x] Tests enforce all of the above
- [x] Help/docs/README are updated accordingly

---

## Verification Commands

```bash
# Test that parallel_warmup is gone
python3 test_warmup_policy.py

# Verify PDMRG no longer has parallel_warmup_flag
python3 -c "import sys; sys.path.insert(0, 'pdmrg'); from pdmrg.dmrg import pdmrg_main; import inspect; assert 'parallel_warmup_flag' not in inspect.signature(pdmrg_main).parameters"

# Verify A2DMRG default is warmup_sweeps=0
python3 -c "import sys; sys.path.insert(0, 'a2dmrg'); from a2dmrg.dmrg import a2dmrg_main; import inspect; assert inspect.signature(a2dmrg_main).parameters['warmup_sweeps'].default == 0"

# Run MPI test for A2DMRG bounds (requires np>=2)
mpirun -np 2 python3 -c "
import sys; sys.path.insert(0, 'a2dmrg')
from a2dmrg.dmrg import a2dmrg_main
import quimb.tensor as qtn
from mpi4py import MPI
mpo = qtn.MPO_ham_heis(4)
try:
    a2dmrg_main(4, mpo, warmup_sweeps=3, experimental_nonpaper=False, comm=MPI.COMM_WORLD, verbose=False)
    print('FAIL: Should have rejected warmup_sweeps=3')
except ValueError as e:
    if 'exceeds paper-faithful bound' in str(e):
        print('PASS: warmup_sweeps=3 correctly rejected')
"
```

---

## Metadata Changes

### A2DMRG Timing Reports

New metadata fields added to timing["meta"]:

```json
{
  "initialization_mode": "paper_default" | "experimental_warmup" | "provided_initial_mps",
  "paper_faithful_mode": true | false,
  "experimental_nonpaper": true | false,
  "warmup_sweeps": 0
}
```

**Values:**
- `initialization_mode="paper_default"` when `warmup_sweeps=0` and no `initial_mps`
- `initialization_mode="experimental_warmup"` when `warmup_sweeps>0`
- `initialization_mode="provided_initial_mps"` when `initial_mps` is provided
- `paper_faithful_mode=true` only when `warmup_sweeps=0` and no custom `initial_mps`

---

## Future Work

### Not Changed in This Task

These were out of scope per the user's directive:
- PDMRG V-matrix computation (still uses identity approximation)
- PDMRG boundary optimization (still disabled with skip_opt=True)
- PDMRG2 validation (still prototype-only)
- A2DMRG local-step mathematical fidelity

These are acknowledged limitations, documented in code and READMEs.

---

## Scientific Impact

**Before:**
- Parallel warmup was algorithmic shortcut (faster but non-rigorous)
- A2DMRG defaulted to serial warmup (engineering convenience, not paper-faithful)
- Easy to accidentally run non-paper modes without realizing it

**After:**
- All initialization is serial warmup + scatter (consistent, reproducible)
- A2DMRG defaults to paper-faithful random init (no serial warmup)
- Experimental/non-paper modes require explicit flags
- Metadata clearly marks paper-faithful vs experimental runs

**Result:** Benchmark hygiene improved. Scientific honesty enforced.
