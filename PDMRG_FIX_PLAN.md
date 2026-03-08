# PDMRG Algorithmic Fix Plan

## Current State (BROKEN)

### Multi-rank loop structure (lines 781-831):
```python
for sweep in range(max_sweeps):
    # 1. QR sweep right (canonize_block) - NO OPTIMIZATION
    canonize_block(pmps, env_mgr, mpo_arrays, 'left')

    # 2. Merge even boundaries with skip_opt=True - NO OPTIMIZATION
    boundary_merge(..., skip_optimization=True)

    # 3. QR sweep left (canonize_block) - NO OPTIMIZATION
    canonize_block(pmps, env_mgr, mpo_arrays, 'right')

    # 4. Merge odd boundaries with skip_opt=True - NO OPTIMIZATION
    boundary_merge(..., skip_optimization=True)
```

**Problems:**
1. `canonize_block()` only does QR decomposition - NO energy optimization
2. `boundary_merge()` called with `skip_optimization=True` - NO energy optimization
3. **ZERO optimization happens**
4. `V` set to `np.ones()` instead of `V = Lambda^-1`

## Required Fix (per Stoudenmire & White 2013)

### Real-Space Parallel DMRG Algorithm:

Each processor independently optimizes its block, then merges at boundaries:

```python
for sweep in range(max_sweeps):
    # Phase 1: Independent local optimization within each block
    # All ranks sweep in opposite directions (staggered pattern)
    direction = 'right' if rank % 2 == 0 else 'left'
    E_local, _ = local_sweep(pmps, env_mgr, mpo_arrays, direction, bond_dim)

    # Phase 2: Merge at shared boundaries using V = Lambda^-1
    # Even boundaries: (0↔1), (2↔3), ...
    if rank is part of even boundary:
        V = compute_proper_V_from_SVD()  # V = 1/S, not np.ones()
        boundary_merge(..., V, skip_optimization=False)  # MUST optimize

    # Phase 3: Sweep in opposite direction
    direction = 'left' if rank % 2 == 0 else 'right'
    E_local, _ = local_sweep(pmps, env_mgr, mpo_arrays, direction, bond_dim)

    # Phase 4: Merge at odd boundaries
    # Odd boundaries: (1↔2), (3↔4), ...
    if rank is part of odd boundary:
        V = compute_proper_V_from_SVD()
        boundary_merge(..., V, skip_optimization=False)
```

## Implementation Changes Required

### 1. Add local sweeps to multi-rank loop
Replace `canonize_block()` calls with `local_sweep()` calls that actually optimize energy.

### 2. Fix V computation
Replace `recompute_boundary_v()` logic:
- **REMOVE**: `V = np.ones(chi_bond)` (identity)
- **ADD**: Proper V computation from SVD singular values
- **USE**: V = 1/S where S comes from the last SVD in the local sweep

### 3. Enable boundary optimization
- **REMOVE**: `skip_opt = True  # Always skip until H_eff bug is fixed`
- **TRY**: Set `skip_opt = False` and test
- **IF H_eff still broken**: Add TODO and make algorithm FAIL LOUDLY with error explaining the issue

### 4. Staggered sweep pattern
Implement proper staggered sweeps:
- Even ranks start sweeping right, odd ranks start sweeping left
- Ensures processors don't wait idly
- Maximizes parallel efficiency

## Verification Tests

After fix, verify:
1. ✓ np=1 raises ValueError
2. ✓ Local sweeps are called in multi-rank path
3. ✓ Energy decreases (or stays constant) with each sweep
4. ✓ V is NOT all-ones
5. ✓ Boundary optimization runs (skip_opt=False)
6. ✓ Final energy matches or improves warmup energy

## Fallback if H_eff Bug Can't Be Fixed Quickly

If boundary optimization still fails with skip_opt=False:

```python
# At line 803:
skip_opt = False  # Try to enable optimization
try:
    E_merge1 = boundary_merge(..., skip_optimization=skip_opt)
except Exception as e:
    if rank == 0:
        print("ERROR: Boundary optimization failed due to H_eff bug")
        print(f"Exception: {e}")
        print("PDMRG is algorithmically incomplete and should not be benchmarked.")
        print("TODO: Fix H_eff construction to enable boundary optimization.")
    raise RuntimeError("PDMRG boundary optimization broken - algorithm incomplete")
```

This ensures we FAIL LOUDLY rather than silently producing wrong results.

## Timeline

1. Add local sweeps: 30min
2. Fix V computation: 20min
3. Test skip_opt=False: 10min
4. If H_eff breaks, add loud failure: 10min
5. Test validation: 20min

**Total**: ~90min for core PDMRG fix
