# PDMRG Stoudenmire Fix: Correct Boundary Coupling

## Problem

Both `pdmrg-gpu` and `pdmrg2-gpu` implement the PDMRG outer loop incorrectly.
After parallel segment sweeps, they perform a **full-chain sequential LR+RL sweep**
as the coupling phase. This negates all parallelism gains — each outer iteration
costs O(L) for the coupling, identical to standard serial DMRG.

The correct algorithm from Stoudenmire & White (arXiv:1301.3494) uses a
lightweight **merge + optimize** step at each shared boundary bond only.

## Reference: Stoudenmire Algorithm (Section II, Figures 2 & 4)

### Two-node case (Figure 2)

```
(a) Distribute: copy shared bond matrix V to both sides
(b) Sweep:      each machine sweeps in parallel over its half
(c) Merge:      merge wavefunctions at shared bond using Ψ' = ψ'_L · V · ψ'_R
(d) Optimize:   run Lanczos/Davidson on the shared bond
```

### Key equations

The wavefunction at the shared bond (sites 3,4 in their notation):

```
Ψ^(α₂ s₃ s₄ α₄) = Σ_α₃ A^(α₂s₃)_α₃ · Λ^α₃ · V_α₃ · Λ^α₃ · B^(s₄α₄)_α₃    (Eq. 3)
```

where `V = Λ⁻¹` is inserted at the boundary. This defines transformed amplitudes:

```
ψ₃ = A · Λ · V_α₃         (left side's contribution)
ψ₄ = V_α₃ · Λ · B         (right side's contribution)
```

After both machines sweep in parallel, they return with updated `ψ'₃` and `ψ'₄`.
The merged wavefunction is:

```
Ψ' = ψ'₃ · V₃ · ψ'₄       (Eq. 5)
```

using the **original V** (not recomputed). This merged state is then optimized
with a few Lanczos/Davidson steps on the shared bond, producing a new SVD
which yields the updated Λ and V for the next iteration.

### n-node case (Figure 4)

Staggered sweep pattern:
```
(a) Odd nodes start at left end of block, even at right end
    All sweep to the other end of their block
(b) When a node reaches its block end, it waits for its neighbor
    Then they communicate (merge + optimize shared bond)
(c) Nodes sweep back to starting positions
(d) Communicate with their other neighbor
```

Key insight from paper: "If a node reaches the end of its block before its
neighbor arrives, it is better for the node to wait instead of immediately
beginning the next half sweep. Having an updated environment far outweighs
the loss in efficiency due to a node briefly remaining idle."

## What to Change

### Files to modify

1. `pdmrg-gpu/src/pdmrg_gpu.h` — add boundary state storage and merge methods
2. `pdmrg-gpu/src/pdmrg_gpu_impl.h` — implement merge+optimize, fix outer loop
3. `pdmrg2-gpu/src/pdmrg2_gpu.h` — same additions
4. `pdmrg2-gpu/src/pdmrg2_gpu_impl.h` — same fixes

### Step-by-step changes

#### 1. Add boundary state storage

At each boundary bond `b` (between segment k and k+1), store:

```cpp
// Per-boundary data
struct BoundaryState {
    Scalar* d_V;          // V matrix (chi × chi) on GPU — V = Λ⁻¹ from SVD
    Scalar* d_Lambda;     // Λ diagonal (singular values) stored as vector
    int chi;              // current bond dimension at this boundary
};
std::vector<BoundaryState> boundary_states_;  // [n_segments - 1]
```

#### 2. Implement distribute step

Before parallel sweeps begin, compute and distribute V = Λ⁻¹ at each boundary:

```
For each boundary bond b at site boundary_bonds_[b]:
  1. Contract MPS[site] and MPS[site+1] into theta (two-site tensor)
  2. SVD: theta = U · S · Vh
  3. Store Λ = diag(S) and V = Λ⁻¹ = diag(1/S)
  4. Store MPS[site] = U (left-canonical)
  5. Store MPS[site+1] = S·Vh (ready for right side)
```

**Critical**: Use accurate SVD (Stoudenmire Appendix) for computing V = Λ⁻¹.
Small singular values in Λ become large in Λ⁻¹, amplifying errors. The recursive
SVD algorithm provides uniform relative accuracy across all singular values.

For now, a simpler approach: truncate V = Λ⁻¹ by zeroing entries where
S[i] < epsilon (e.g. 1e-12), so 1/S[i] doesn't blow up.

#### 3. Modify segment sweep to use V-transformed tensors

Before each segment sweep, the boundary MPS tensors need to absorb the V matrix:

```
Left boundary of segment k (if k > 0):
  MPS[seg_first_[k]] = V_left · MPS[seg_first_[k]]
  (absorb V from the left boundary into the first tensor)

Right boundary of segment k (if k < n_segments - 1):
  MPS[seg_last_[k]] = MPS[seg_last_[k]] · V_right
  (absorb V from the right boundary into the last tensor)
```

This transforms to the "ψ" basis of Eq. (4). After the sweep, the segment
returns updated ψ' tensors.

#### 4. Implement merge step (Eq. 5)

After all parallel sweeps complete, at each boundary:

```
For each boundary bond b:
  site_L = boundary_bonds_[b]      (last site of left segment)
  site_R = boundary_bonds_[b] + 1  (first site of right segment)

  1. Undo V absorption:
     - ψ'_L already contains the swept result
     - ψ'_R already contains the swept result

  2. Form merged two-site tensor:
     theta = ψ'_L · V · ψ'_R       (Eq. 5)

     In MPS terms: contract MPS[site_L] × V × MPS[site_R]
     to get theta of shape (chi_L, d, d, chi_R)

  3. This theta is the initial state for optimization
```

#### 5. Implement boundary optimize step

At each boundary, optimize the merged state:

```
For each boundary bond b:
  1. Build/update left environment at site_L and right environment at site_R+1
     (these come from the segment sweeps — they should be current)

  2. Run eigensolver (Lanczos for pdmrg-gpu, Block-Davidson for pdmrg2-gpu)
     on the two-site Hamiltonian at (site_L, site_R) using theta as initial guess

  3. SVD the optimized theta:
     theta_opt = U · S · Vh

  4. Update MPS[site_L] = U, MPS[site_R] = S · Vh (or vice versa)

  5. Store new Λ = S and V = Λ⁻¹ for next iteration

  6. Update environments: L_env at site_L, R_env at site_R
```

#### 6. Fix the outer loop

Replace the current outer loop:

```cpp
// CURRENT (WRONG):
for each outer iteration:
    parallel_segment_sweeps()
    build_initial_environments()    // O(L) — wasteful
    sweep_LR_full()                 // O(L) — defeats parallelism
    energy_ = sweep_RL_full()       // O(L) — defeats parallelism

// CORRECT (Stoudenmire):
for each outer iteration:
    distribute_boundary_V()         // O(P) — distribute V matrices
    parallel_segment_sweeps()       // O(L/P) per node — the parallel part
    merge_and_optimize_boundaries() // O(P) — merge + optimize shared bonds
    energy_ = compute_energy()      // measure convergence
```

The energy can be measured from the last boundary optimization eigenvalue,
or by computing ⟨ψ|H|ψ⟩ at one boundary bond.

#### 7. Implement staggered sweep pattern (Figure 4, optional but recommended)

For n > 2 nodes, use the staggered pattern:
- Odd-indexed segments sweep LR first, even sweep RL first
- After reaching the end, wait and communicate with the neighbor that's arriving
- Then sweep back

This ensures that when two segments communicate, both have fresh environments
at their shared boundary.

Simple version (without explicit synchronization):
```cpp
// Half-sweep 1: all segments sweep in one direction
parallel_sweep([](Self* self, int k) {
    if (k % 2 == 0) self->segment_sweep_LR(k);
    else             self->segment_sweep_RL(k);
});

// Communicate at boundaries where neighbors just met
merge_and_optimize_boundaries();

// Half-sweep 2: sweep back
parallel_sweep([](Self* self, int k) {
    if (k % 2 == 0) self->segment_sweep_RL(k);
    else             self->segment_sweep_LR(k);
});

// Communicate at the other set of boundaries
merge_and_optimize_boundaries();
```

### Accurate SVD for V = Λ⁻¹ (Stoudenmire Appendix)

The paper describes a recursive SVD that provides uniform relative accuracy:

```
function accurate_svd(M, threshold):
    U, S, Vh = standard_svd(M)
    if S[0] / S[-1] < threshold:
        return U, S, Vh  // well-conditioned, standard SVD is fine

    // Split into well-conditioned and ill-conditioned parts
    k = find_split_point(S)  // where ratio jumps

    // Recursively SVD the small-singular-value block
    M_small = M - U[:,:k] · diag(S[:k]) · Vh[:k,:]
    U2, S2, Vh2 = accurate_svd(M_small, threshold)

    // Combine results
    return combine(U[:,:k], S[:k], Vh[:k,:], U2, S2, Vh2)
```

For the initial implementation, a simpler approach is acceptable:
- Use standard SVD
- When computing V = Λ⁻¹, cap: `V[i] = (S[i] > eps) ? 1/S[i] : 0`
- This loses accuracy for small singular values but avoids divergence

The accurate SVD can be added as a follow-up optimization.

## Environment handling at boundaries

A subtle but critical point: after parallel segment sweeps, the environments
at boundary sites need to be correct for the boundary optimize step.

- **Left environment at site_L**: This was the last L_env built during the
  left segment's sweep. It should be current if the segment swept LR
  (the sweep updates L_envs as it goes right).

- **Right environment at site_R+1**: This was built during the right segment's
  sweep. It should be current if the segment swept RL.

With the staggered pattern, after the first half-sweep:
- Even segments swept LR → their right boundary has a fresh L_env
- Odd segments swept RL → their left boundary has a fresh R_env
- At each boundary, one side has a fresh env → merge and optimize

## Correctness targets

Same as current tests — all must pass with error < 1e-10:

| Test | Energy |
|------|--------|
| Heisenberg L=8 chi=32 segments=2 | -3.374932598688 |
| Heisenberg L=32 chi=64 segments=4 | -13.997315618007 |
| Josephson L=6 chi=32 segments=2 | -1.748843818181493 |

## Performance targets

With correct Stoudenmire coupling, the outer loop cost should be:
- Parallel segment sweeps: O(L/P) wall time (P segments in parallel)
- Boundary merge+optimize: O(P) operations, each O(chi² · d² · D)
- Total per iteration: O(L/P + P) instead of current O(L/P + L)

Expected speedup vs serial DMRG for L=64 chi=128:
- 2 segments: ~1.5-1.8x
- 4 segments: ~2.5-3.5x
- 8 segments: ~4-6x

(Sub-ideal due to warmup cost and boundary convergence overhead)

## Build and test

```bash
# On remote MI300X:
ssh hotaisle@23.183.40.82
cd ~/dmrg-implementations && git pull

# Build pdmrg-gpu
cd pdmrg-gpu/build && cmake .. -DGPU_TARGETS=gfx942 && make -j$(nproc)
./pdmrg_gpu 8 32 20 --segments 2 --warmup 3
./pdmrg_gpu 32 64 20 --segments 4 --warmup 3

# Build pdmrg2-gpu
cd ../../pdmrg2-gpu/build && cmake .. -DGPU_TARGETS=gfx942 && make -j$(nproc)
./pdmrg2_gpu 8 32 20 --segments 2 --warmup 3
./pdmrg2_gpu 32 64 20 --segments 4 --warmup 3

# Scalability test
./pdmrg_gpu 64 128 20 --segments 2 --warmup 3
./pdmrg_gpu 64 128 20 --segments 4 --warmup 3
./pdmrg_gpu 64 128 20 --segments 8 --warmup 3
```

## Implementation order

1. **pdmrg-gpu first** — simpler (Lanczos + SVD), easier to debug
2. **Verify correctness** on all 3 test cases with segments=2
3. **Run scalability** tests for segments=2,4,8
4. **Port to pdmrg2-gpu** — same changes with Newton-Schulz + Davidson
5. **Compare performance** pdmrg-gpu vs pdmrg2-gpu vs serial baselines

## Key files for reference

- `pdmrg-gpu/src/pdmrg_gpu.h` — class declaration, StreamWorkspace struct
- `pdmrg-gpu/src/pdmrg_gpu_impl.h` — full implementation (~1270 lines)
- `pdmrg-gpu/src/scalar_traits.h` — type dispatch, LAPACK/rocBLAS wrappers
- `pdmrg2-gpu/src/pdmrg2_gpu.h` — class declaration with NS/Davidson additions
- `pdmrg2-gpu/src/pdmrg2_gpu_impl.h` — full implementation (~2090 lines)
- `pdmrg2-gpu/src/scalar_traits.h` — extended with syev, NS kernels
- Paper: https://arxiv.org/abs/1301.3494 (Stoudenmire & White, 2013)
- Benchmark results: `benchmarks/gpu_scalability_results.md`
