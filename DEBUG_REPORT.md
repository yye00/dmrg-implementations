# DMRG Debugging Report

## Summary

Debug testing on Heisenberg model (L=12, D=20) reveals bugs in both PDMRG and A2DMRG.

## Reference Energies (quimb)

| Method | Energy | Time |
|--------|--------|------|
| quimb DMRG1 | -5.142090628199368 | 0.18s |
| quimb DMRG2 | -5.142090628178130 | 0.13s |

Reference: DMRG2 = **-5.142090628178130**

## PDMRG Results

| np | Energy | ΔE from ref | Status |
|----|--------|-------------|--------|
| 1 | -5.142090628178227 | 9.77e-14 | ✓ PASS |
| 2 | -5.142090629416210 | **1.24e-09** | ✗ FAIL |
| 4 | -5.142090628178191 | 6.13e-14 | ✓ PASS |

### PDMRG Bug Analysis (np=2)

**Root Cause:** V matrix at boundary becomes stale after local canonization.

**Details:**
1. Warmup produces perfect MPS (ΔE = 7e-15 from ref)
2. MPS is distributed to 2 ranks with V_right computed at bond 5↔6
3. `canonize_block` changes local MPS gauge (QR sweep)
4. V_right is NOT updated
5. `boundary_merge` uses stale V with new tensors
6. Reconstructed wavefunction theta = psi_left @ V @ psi_right is incorrect
7. Energy increases from -5.142090628178 to -5.142090629416

**Why np=4 works:** With 4 processors, the even boundary merges (0↔1, 2↔3) and odd merges (1↔2, 3↔4) happen in a pattern where the stale V error may cancel or be smaller.

**Fix Required:**
```python
# In boundary_merge or before it:
# Recompute V from current local tensors
V_new = recompute_v_at_boundary(pmps, neighbor_pmps)
```

## A2DMRG Results

| np | Energy | ΔE from ref | Status |
|----|--------|-------------|--------|
| 1 (1 sweep) | -5.142090596168226 | **3.20e-08** | ✗ FAIL |
| 1 (2+ sweeps) | CRASH | N/A | BUG |

### A2DMRG Bug Analysis

**Bug 1: Coarse-space approximation is inaccurate**
- Warmup energy: -5.142090596161
- Coarse-space energy: -5.142090544751 (HIGHER/worse!)
- After compression: -5.142090596168
- The coarse-space optimization produces a WORSE energy than the input

**Bug 2: Multi-sweep crashes due to bond dimension mismatch**
- Error: `Size of label 'd' for operand 4 (16) does not match previous terms (17)`
- Cause: After compression, MPS bond dimensions change
- Environments built in sweep N don't match MPS from sweep N-1
- Fix: Rebuild environments after each compression

**Bug 3: Finalization fails**
- MPS from A2DMRG has different structure than DMRG2 expects
- Simply copying tensor data doesn't work

## Recommendations

### PDMRG Fixes (Priority 1)

1. **Recompute V after canonize_block:**
   ```python
   def canonize_block(...):
       # ... existing code ...
       # After canonization, recompute V at boundaries
       if rank < n_procs - 1:
           pmps.V_right = compute_v_from_current_state(pmps, neighbor)
   ```

2. **Alternative: Update V during canonization:**
   Track how canonization transforms the boundary tensor and transform V accordingly.

### A2DMRG Fixes (Priority 2)

1. **Rebuild environments after each compression:**
   ```python
   # After Phase 5 compression
   environments = rebuild_all_environments(mps, mpo)
   ```

2. **Investigate coarse-space accuracy:**
   The coarse space should give energy ≤ input energy. If not, there may be bugs in:
   - `build_coarse_matrices`
   - `solve_coarse_eigenvalue_problem`
   - `form_linear_combination`

3. **Fix finalization:**
   Either skip finalization or properly convert MPS format.

## Test Commands

```bash
# PDMRG debug
mpirun -np 2 pdmrg/venv/bin/python debug/pdmrg_np2_debug.py

# A2DMRG debug  
mpirun -np 1 a2dmrg/venv/bin/python debug/a2dmrg_heisenberg_debug.py
```

## Files Modified During Debug

- `/debug/heisenberg_debug.py` - Reference quimb implementation
- `/debug/pdmrg_heisenberg_debug.py` - PDMRG test
- `/debug/pdmrg_np2_debug.py` - Detailed np=2 analysis
- `/debug/a2dmrg_heisenberg_debug.py` - A2DMRG test
- `/debug/reference_heisenberg.json` - Reference energies
