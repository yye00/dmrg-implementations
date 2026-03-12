# PDMRG-GPU Boundary Merge Fix Prompt

## Goal

Fix the boundary merge in `pdmrg-gpu` so that segment sweeps provide actual parallel speedup. Currently, every outer iteration requires a **full-chain coupling sweep** after segment sweeps, which negates the parallelism. The merge_boundary approach that avoids full-chain sweeps produced incorrect energies (doubled or oscillating). Fix it.

## Current Architecture (working but slow)

```
for each outer iteration:
    1. Parallel segment sweeps on P HIP streams    (fast, O(L/P))
    2. build_initial_environments()                 (rebuilds ALL L+R envs from scratch)
    3. sweep_LR_full() + sweep_RL_full()           (full-chain sweep, O(L)) ← BOTTLENECK
    4. Report energy from step 3
Polish: 2 more full-chain sweeps
```

The full-chain sweep in step 3 dominates runtime and makes PDMRG **slower** than plain dmrg2-gpu.

## Benchmark Evidence (MI300X, Heisenberg OBC)

| L | chi | dmrg-gpu | dmrg2-gpu | pdmrg P=2 | pdmrg P=4 | pdmrg P=8 |
|---|-----|----------|-----------|-----------|-----------|-----------|
| 32 | 64 | **1.87s** | 2.14s | 19.57s | 20.92s | 17.53s |
| 32 | 128 | **2.74s** | 7.38s | 54.85s | 47.83s | 33.05s |
| 64 | 64 | **6.12s** | 8.82s | 58.75s | 61.55s | 58.52s |
| 64 | 128 | **13.97s** | 21.82s | 192.59s | 179.53s | 150.52s |

PDMRG is 10-14x slower than dmrg-gpu because the coupling sweep + polish + env rebuilds dominate.

## What Failed Before

### Attempt 1: merge_boundary with optimize_bond at boundaries

```cpp
// Phase 2: for each boundary k
double e = merge_boundary(k, 0);  // calls optimize_bond(boundary_site, 'R', 0)
update_left_env(boundary_bonds_[k], 0);
```

**Problem**: Energy was -27.99 (exactly 2x correct -13.997) on first iteration, then oscillated around -8.6 to -8.9 and never converged.

**Root cause analysis**:
- After parallel segment sweeps, environments at boundary sites are **stale** — they were built from pre-sweep MPS tensors
- `build_initial_environments()` was called before merges but it rebuilds from the CURRENT MPS which has been modified by segment sweeps that used frozen boundary environments
- The L_env[b] and R_env[b+2] used by optimize_bond at boundary b are internally consistent within their respective segments, but the environments encode partial traces built with **different frozen boundaries**, leading to double-counting or inconsistency in the Rayleigh quotient

### Attempt 2: V-matrix weighting on boundary theta

Applying V = 1/S matrices to weight the boundary theta before Lanczos. Failed because V-matrices become stale after local sweeps modify boundary MPS tensors.

## Target Architecture (what should work)

```
for each outer iteration:
    1. Parallel segment sweeps on P HIP streams
    2. Boundary merges (sequential or staggered) that correctly couple segments
    3. Report energy from merge step
    4. No full-chain sweep needed
```

This would make PDMRG O(L/P) per iteration instead of O(L).

## Key Insight

The boundary merge must produce correct environments for the next iteration. After segment k sweeps sites [first_k, last_k] and segment k+1 sweeps [first_{k+1}, last_{k+1}], the boundary bond at site b = last_k needs:

1. **L_env[b]** that is consistent with the current MPS[0..b-1] — built during segment k's L→R sweep
2. **R_env[b+2]** that is consistent with the current MPS[b+2..L-1] — built during segment k+1's R→L sweep

But these environments were built using frozen boundary conditions from the **previous** iteration. The issue is that L_env[b] was built assuming a specific R boundary (from the previous iteration), and R_env[b+2] was built assuming a specific L boundary (also from previous iteration).

## Possible Fix Strategies

### Strategy A: Rebuild boundary environments before merge

After all segment sweeps complete:
1. For each boundary k (left to right):
   - Rebuild L_env[b] by running update_left_env from the previous boundary (or site 0) up to site b-1
   - Rebuild R_env[b+2] by running update_right_env from the next boundary (or site L-1) down to site b+2
   - Now optimize_bond(b, 'R') with correct environments
   - After merge, update L_env[b+1] for the next boundary's merge

This is still O(L) total for env rebuilds, but amortized across boundaries.

### Strategy B: Only merge at boundaries, skip full-chain sweep

Instead of a full LR+RL sweep, do a **boundary-only sweep**:
1. After segment sweeps, rebuild all environments from scratch (one pass L→R, one R→L)
2. For each boundary bond: optimize_bond + SVD split
3. After all boundaries done, update just the boundary environments

The rebuild is O(L) but done once. The boundary optimizations are O(P) bonds total (much less than L bonds in a full sweep).

### Strategy C: Alternating boundary-focused sweeps

Do a short sweep that only covers 2-3 sites around each boundary instead of the full chain:
1. For boundary at site b: sweep sites [b-1, b, b+1, b+2] with proper environments
2. This requires correct L_env[b-1] and R_env[b+3], which can be built incrementally

### Strategy D: Reduce full-chain sweeps

Keep the current architecture but:
- Do full-chain coupling sweep only every N outer iterations instead of every iteration
- Use segment sweeps for N-1 iterations, then one coupling sweep
- This reduces the O(L) work by factor N

## Files to Modify

- `pdmrg-gpu/src/pdmrg_gpu_impl.h` — main implementation (~1260 lines)
  - `run()` method (line ~1123): main algorithm loop
  - `merge_boundary()` (line ~1090): boundary merge logic
  - `rebuild_boundary_envs()` (line ~1105): env rebuild at boundary
  - `build_initial_environments()` (line ~625): full env rebuild

- `pdmrg-gpu/src/pdmrg_gpu.h` — class declaration (147 lines)

## Existing Working Code to Reference

- `dmrg2-gpu/src/dmrg2_gpu_impl.h` — working two-site DMRG (all environment/GEMM patterns correct)
- The warmup sweeps in pdmrg-gpu (`sweep_LR_full`, `sweep_RL_full`) work correctly

## Correctness Tests

All must pass with error < 1e-10:
- Heisenberg L=8 chi=32 (2 segments): exact = -3.374932598688
- Heisenberg L=32 chi=64 (4 segments): exact = -13.997315618007
- Josephson L=6 chi=32 (2 segments): exact = -1.748843818181

## Success Criteria

1. **Correct**: All tests pass at 1e-10 tolerance
2. **Fast**: pdmrg P=4 should be faster than dmrg2-gpu for L=64 chi=128+
3. **Scaling**: More streams (P=8) should be faster than fewer (P=2) for large enough L

## Build & Test Commands

```bash
# Build on remote MI300X
ssh hotaisle@23.183.40.79
cd ~/dmrg-implementations/pdmrg-gpu/build
cmake .. -DGPU_TARGETS=gfx942 && make -j$(nproc)

# Test
./pdmrg_gpu 8 32 10 --segments 2 --warmup 3 --local-sweeps 2
./pdmrg_gpu 32 64 20 --segments 4 --warmup 3 --local-sweeps 2
./pdmrg_gpu 6 32 10 --segments 2 --warmup 3 --local-sweeps 2 --josephson

# Benchmark comparison
./pdmrg_gpu 64 128 10 --segments 4 --warmup 3 --local-sweeps 2
# vs
cd ~/dmrg-implementations/dmrg2-gpu/build && ./dmrg2_gpu 64 128 10
cd ~/dmrg-implementations/dmrg-gpu/build && ./dmrg_gpu 64 128 10
```
