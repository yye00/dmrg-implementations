# Fix PDMRG-OPT (CPU) and PDMRG-OPT-GPU Canonicalization Bugs

## Context

Benchmarking shows pdmrg-opt is 25x slower than pdmrg on small systems (L=8, chi=20, Josephson np=2: 46.4s vs 1.82s) and delivers worse accuracy (-2.8436 vs -2.8438). The root cause is three canonicalization bugs in pdmrg-opt that propagate stale/incorrect environments into the eigensolver. pdmrg-gpu-opt has analogous issues.

The reference (correct) implementation is pdmrg at `/home/captain/clawd/work/dmrg-implementations/pdmrg/pdmrg/dmrg.py`. The fixes below port its canonicalization discipline to pdmrg-opt and pdmrg-gpu-opt.

## Verification

Before and after each fix, run the benchmark suite to measure impact:

```bash
cd /home/captain/clawd/work/dmrg-implementations
python benchmarks/run.py validate --impl pdmrg,pdmrg-opt --model heisenberg,josephson --np 2
```

The pdmrg-opt Josephson energy should improve from -2.8436 toward -2.8438 (matching pdmrg), and timing should drop significantly.

For pdmrg-gpu-opt, compile and test on the remote MI300X (ssh hotaisle@23.183.40.75):

```bash
cd ~/dmrg-implementations/pdmrg-gpu-opt/build
cmake .. && make -j
./pdmrg_gpu_opt 8 20 10 --josephson --segments 2
```

---

## Fix 1: pdmrg-opt CPU — Return A_right_canonical from merge_boundary_tensors

**File**: `/home/captain/clawd/work/dmrg-implementations/pdmrg-opt/pdmrg/parallel/merge.py`

**Problem**: `merge_boundary_tensors` returns only 5 values. After SVD split `M = U @ diag(S) @ Vh`, it computes `A_right_new = (diag(S) @ Vh).reshape(k, d_R, chi_R)` and returns that. But when this tensor is used to build R_env, the norm matrix becomes S^2 instead of I, breaking the N_eff = I assumption in subsequent eigensolves.

**What pdmrg does correctly** (pdmrg/pdmrg/parallel/merge.py, line 97):
```python
A_right_canonical = Vh.reshape(k, d_R, chi_R)
return A_left_new, A_right_new, V_new, energy, trunc_err, A_right_canonical
```

**Fix**: Add `A_right_canonical = Vh.reshape(k, d_R, chi_R)` after the SVD truncation in pdmrg-opt's merge.py, and return it as a 6th value.

---

## Fix 2: pdmrg-opt CPU — Use A_right_canonical for R_env rebuild after merge

**File**: `/home/captain/clawd/work/dmrg-implementations/pdmrg-opt/pdmrg/dmrg.py`

**Problem**: In the `boundary_merge` function (around line 249), pdmrg-opt unpacks only 5 return values:
```python
A_left_new, A_right_new, V_new, energy, _ = merge_boundary_tensors(...)
```
Then at line 261-262, it builds R_env from `A_right_new` (which contains S*Vh, not right-canonical):
```python
env_mgr.R_envs[global_end] = update_right_env(R_env, A_right_new, mpo_arrays[right_global])
```

**What pdmrg does correctly** (pdmrg/pdmrg/dmrg.py, line 265):
```python
A_left_new, A_right_new, V_new, energy, _, A_right_canonical = merge_boundary_tensors(...)
...
env_mgr.R_envs[global_end] = update_right_env(R_env, A_right_canonical, mpo_arrays[right_global])
```

**Fix**: Unpack the 6th return value and use `A_right_canonical` instead of `A_right_new` for R_env construction.

---

## Fix 3: pdmrg-opt CPU — Canonicalize MPS before building environments in build_local_environments

**File**: `/home/captain/clawd/work/dmrg-implementations/pdmrg-opt/pdmrg/dmrg.py`

**Problem**: The `build_local_environments` function (around lines 290-347) builds L_envs and R_envs directly from the raw MPS arrays without first putting them in canonical form. L_env assumes left-canonical tensors (so that the norm matrix is I). R_env assumes right-canonical tensors. Using non-canonical tensors produces stale environments.

**What pdmrg does correctly** (pdmrg/pdmrg/dmrg.py, lines 313-366):
Before building L_envs, it left-canonicalizes the MPS (QR sweep left-to-right). Before building R_envs, it right-canonicalizes (LQ sweep right-to-left). This ensures norm = I at each step.

**Fix**: In pdmrg-opt's `build_local_environments`, add a left-canonicalization sweep (using Newton-Schulz polar, since that's pdmrg-opt's convention) before building L_envs, and a right-canonicalization sweep before building R_envs. The pdmrg version uses QR; pdmrg-opt should use Newton-Schulz polar for consistency, but QR would also work and be faster.

**Note**: pdmrg-opt already has `rebuild_boundary_r_env` and `rebuild_boundary_l_env` functions (lines 350-442) that are identical to pdmrg's. The issue is they're not called at the right time — specifically, `build_local_environments` doesn't canonicalize before env construction.

---

## Fix 4: pdmrg-gpu-opt — Rebuild boundary environments after segment canonicalization

**File**: `/home/captain/clawd/work/dmrg-implementations/pdmrg-gpu-opt/src/pdmrg_gpu_opt_impl.h`

**Problem**: In `segment_sweep_LR` (lines 1867-1886), after `canonize_segment_right(seg_idx)`, the code rebuilds R_envs via incremental `update_right_env`. But the L_envs at the segment boundaries were built from the PRE-canonicalization MPS and are now stale. Same issue in `segment_sweep_RL` with L_envs.

Current code:
```cpp
void PDMRGGPUOpt<Scalar>::segment_sweep_LR(int seg_idx) {
    int first = seg_first_[seg_idx];
    int last = seg_last_[seg_idx];
    int si = seg_idx;

    canonize_segment_right(seg_idx);  // MPS changed!

    // Only rebuilds R_envs, but L_env[first] is now stale
    for (int j = last; j > first; j--) {
        update_right_env(j, si);
    }
    // ... sweep ...
}
```

**Fix for segment_sweep_LR**: After `canonize_segment_right`, also rebuild L_env at the left boundary. The boundary L_env should be rebuilt from the LEFT neighbor's canonical tensors, but since we only control this segment, we need to rebuild L_env[first] using the segment's own left boundary MPS tensor (which is the orthogonality center after right-canonicalization). The simplest correct approach: before rebuilding R_envs, also rebuild L_env[first] from the segment boundary.

**Fix for segment_sweep_RL**: After `canonize_segment_left`, also rebuild R_env at the right boundary, same logic mirrored.

---

## Fix 5: pdmrg-gpu-opt — Canonicalize between double optimize_bond in merge_and_optimize_boundaries

**File**: `/home/captain/clawd/work/dmrg-implementations/pdmrg-gpu-opt/src/pdmrg_gpu_opt_impl.h`

**Problem**: In `merge_and_optimize_boundaries` (lines 1918-1936):
```cpp
energy = optimize_bond(bsite, 'R', si);   // After: MPS[bsite+1] = S*Vh (mixed form)
update_left_env(bsite, si);
energy = optimize_bond(bsite, 'L', si);   // Expects MPS[bsite+1] right-canonical, but it's S*Vh
update_right_env(bsite + 1, si);
```

After the first `optimize_bond(bsite, 'R')`, MPS[bsite+1] contains S*Vh — a mixed-canonical tensor where the singular values haven't been absorbed. The second `optimize_bond(bsite, 'L')` constructs theta from MPS[bsite] and MPS[bsite+1], and the stale S factors in MPS[bsite+1] bias the optimization.

**Fix**: Right-canonicalize MPS[bsite+1] before the second optimize_bond call. This can be done with a single Newton-Schulz polar decomposition (or QR) on MPS[bsite+1], absorbing the left factor into MPS[bsite]:

```cpp
energy = optimize_bond(bsite, 'R', si);
update_left_env(bsite, si);

// Canonicalize: right-canonicalize site bsite+1, absorb into bsite
right_canonize_site(bsite + 1, si);
// Rebuild R_env at bsite+1 with the now-canonical tensor
update_right_env(bsite + 1, si);

energy = optimize_bond(bsite, 'L', si);
update_right_env(bsite + 1, si);
```

---

## Fix 6: pdmrg-gpu-opt — Add Newton-Schulz convergence verification in ns_split

**File**: `/home/captain/clawd/work/dmrg-implementations/pdmrg-gpu-opt/src/pdmrg_gpu_opt_impl.h`

**Problem**: `ns_split` (lines 907-1091) only checks `if (ns_iters >= 29)` to decide whether to fall back to SVD. It never verifies that `||U^H U - I|| < tol` after Newton-Schulz iterations.

**Fix**: After the Newton-Schulz loop converges (or hits max iterations), compute `||U^H U - I||_F` and fall back to SVD if it exceeds a tolerance (e.g., 1e-10). This is a single GEMM + norm computation:

```cpp
// After NS loop:
// Compute UtU = U^H @ U (GEMM)
// Compute ||UtU - I||_F
// If > 1e-10, fall back to svd_split
```

This is lower priority than Fixes 1-5 but prevents silent accuracy degradation.

---

## Expected Impact

| Fix | Affects | Expected Improvement |
|-----|---------|---------------------|
| Fix 1+2 | pdmrg-opt CPU accuracy | Energy from -2.8436 → -2.8438 (matching pdmrg) |
| Fix 3 | pdmrg-opt CPU speed | Fewer eigensolver iterations per sweep, ~2-5x faster |
| Fix 1+2+3 combined | pdmrg-opt CPU overall | Should bring pdmrg-opt within 2-3x of pdmrg (down from 25x), remaining gap is Newton-Schulz vs QR overhead |
| Fix 4 | pdmrg-gpu-opt convergence | Fewer sweeps to converge, better boundary accuracy |
| Fix 5 | pdmrg-gpu-opt boundary quality | More accurate boundary optimization, better energy |
| Fix 6 | pdmrg-gpu-opt robustness | Prevents silent NS failures at large chi |

## Order of Operations

1. Fix 1 + Fix 2 together (merge.py return value + dmrg.py unpack) — they're a single logical change
2. Run benchmark, verify accuracy improvement
3. Fix 3 (build_local_environments canonicalization)
4. Run benchmark, verify speed improvement
5. Fix 4 + Fix 5 together (GPU boundary rebuild + merge canonicalization)
6. Compile and test on remote MI300X
7. Fix 6 (NS convergence check) — lowest priority

## Files to Modify

### pdmrg-opt CPU:
- `pdmrg-opt/pdmrg/parallel/merge.py` — Fix 1 (add A_right_canonical return)
- `pdmrg-opt/pdmrg/dmrg.py` — Fix 2 (use A_right_canonical for R_env) + Fix 3 (canonicalize in build_local_environments)

### pdmrg-gpu-opt:
- `pdmrg-gpu-opt/src/pdmrg_gpu_opt_impl.h` — Fix 4 (rebuild boundary envs after canonize) + Fix 5 (canonicalize between double optimize_bond) + Fix 6 (NS convergence check)

## Reference Implementation

The correct patterns are all in the pdmrg (original) implementation:
- `pdmrg/pdmrg/parallel/merge.py` — merge_boundary_tensors returning A_right_canonical
- `pdmrg/pdmrg/dmrg.py` — boundary_merge using A_right_canonical, build_local_environments with canonicalization, rebuild_boundary_r_env, rebuild_boundary_l_env
