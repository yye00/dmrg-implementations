# PDMRG Stoudenmire Fix: Correct Boundary Coupling

## Problem

Both `pdmrg-gpu` and `pdmrg-gpu-opt` implement the PDMRG outer loop incorrectly.
After parallel segment sweeps, they perform a **full-chain sequential LR+RL sweep**
as the coupling phase. This negates all parallelism gains — each outer iteration
costs O(L) for the coupling, identical to standard serial DMRG.

The correct algorithm from Stoudenmire & White (arXiv:1301.3494) uses a
lightweight **merge + optimize** step at each shared boundary bond only.

## Lessons from GPU Optimization Attempts (2026-03-23)

We attempted two GPU-accelerated replacements for standard LAPACK SVD:

### Newton-Schulz Polar Decomposition + Block-Davidson Eigensolver

Replaced SVD with iterative polar decomposition (GPU GEMMs only) and replaced
Lanczos with Block-Davidson (BLAS-3 block subspace iteration).

**Results: 0% win rate across ALL opt variants vs their correct baselines.**

| Comparison | Win Rate | Notes |
|---|---|---|
| dmrg-gpu-opt vs **dmrg-gpu** | **0/24** (0%) | 2-7x slower |
| dmrg2-gpu-opt vs **dmrg2-gpu** | **0/26** (0%) | 1.5-6x slower |
| pdmrg-gpu-opt vs **dmrg-gpu** (best serial) | **0/23** (0%) | 0.07x-0.73x, always slower |

pdmrg-gpu-opt does beat pdmrg-gpu (87% win rate, up to 20x), but this is a
misleading comparison — pdmrg-gpu itself is far slower than serial dmrg-gpu
and dmrg2-gpu due to the broken coupling phase (the problem this prompt fixes).
The "opt" speedup just partially compensates for the PDMRG overhead; it never
catches up to straightforward serial GPU DMRG.

**Why the optimizations failed:**
- Newton-Schulz replaces one O(n³) GPU SVD call with ~12-16 O(n³) GPU GEMM
  iterations — pure overhead. rocsolver gesvd is already highly optimized.
- Block-Davidson has higher memory/compute per eigensolve than Lanczos.
- Neither optimization amortizes well in DMRG's sweep-site-by-site pattern.

**Newton-Schulz numerical instability at high chi:**
- chi<=50: Converges correctly but slower than GPU SVD (rocsolver gesvd)
- chi>=128: Diverges catastrophically (energies like -10²⁵) — condition number
  too high for Frobenius-norm-scaled iteration to converge in 30 iterations
- chi>=256 L>=64: Crashes/segfaults from corrupted MPS tensors
- SVD fallback exists (U^H U - I verification) but triggers too late — by the
  time one bad SVD replacement corrupts an MPS tensor, error compounds
  exponentially through subsequent sweep sites

**Implication for this fix:** Use standard GPU SVD (rocsolver) + Lanczos only.
Do NOT use Newton-Schulz or Block-Davidson. The Stoudenmire fix should be
implemented in `pdmrg-gpu` (standard SVD + Lanczos). The `-opt` variants
should be considered failed experiments.

### Benchmark result files

All results are committed to GitHub at the paths below:

- **GPU opt benchmark (192 configs, 3 models, MI300X):**
  `benchmarks/paper_results/bench_opt_results.csv` (192 rows)
  `benchmarks/paper_results/bench_opt_results.json`
- **Full 10-implementation comparison (737 configs):**
  `benchmarks/paper_results/summary.csv`
- **GPU 4-way scalability (pdmrg-gpu vs pdmrg-gpu-opt, segments=2,4):**
  `benchmarks/paper_results/gpu_4way_results.csv` (248 rows)
- **A2DMRG accuracy benchmarks:**
  `benchmarks/paper_results/a2dmrg_small_results.txt` (L=8-20, chi=20-50)
  `benchmarks/paper_results/a2dmrg_medium_results.txt` (L=32-64, chi=50-100)

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
3. `pdmrg-gpu-opt/src/pdmrg_gpu_opt.h` — same additions
4. `pdmrg-gpu-opt/src/pdmrg_gpu_opt_impl.h` — same fixes

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

  2. Run eigensolver (Lanczos for pdmrg-gpu, Block-Davidson for pdmrg-gpu-opt)
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
# On remote MI300X (passwordless SSH):
ssh hotaisle@23.183.40.75
cd ~/dmrg-implementations && git pull

# Build pdmrg-gpu
cd pdmrg-gpu/build && cmake .. -DGPU_TARGETS=gfx942 && make -j$(nproc)
./pdmrg_gpu 8 32 20 --segments 2 --warmup 3
./pdmrg_gpu 32 64 20 --segments 4 --warmup 3

# Build pdmrg-gpu-opt
cd ../../pdmrg-gpu-opt/build && cmake .. -DGPU_TARGETS=gfx942 && make -j$(nproc)
./pdmrg_gpu_opt 8 32 20 --segments 2 --warmup 3
./pdmrg_gpu_opt 32 64 20 --segments 4 --warmup 3

# Scalability test
./pdmrg_gpu 64 128 20 --segments 2 --warmup 3
./pdmrg_gpu 64 128 20 --segments 4 --warmup 3
./pdmrg_gpu 64 128 20 --segments 8 --warmup 3
```

**Note:** Remote VM IP may change between sessions — check CLAUDE.md for current IP.
Prerequisites on fresh VMs: `sudo apt-get install -y cmake liblapack-dev libopenblas-dev`

## Implementation order

1. **pdmrg-gpu only** — standard GPU SVD (rocsolver) + Lanczos eigensolver
2. **Verify correctness** on all 3 test cases with segments=2
3. **Run scalability** tests for segments=2,4,8
4. **Benchmark against dmrg-gpu and dmrg2-gpu** — the real competition
5. **Do NOT implement in -opt variants** — Newton-Schulz + Block-Davidson is a
   failed optimization (0% win rate vs standard GPU SVD in every context tested)

## Key files for reference

- `pdmrg-gpu/src/pdmrg_gpu.h` — class declaration, StreamWorkspace struct
- `pdmrg-gpu/src/pdmrg_gpu_impl.h` — full implementation (~1270 lines)
- `pdmrg-gpu/src/scalar_traits.h` — type dispatch, LAPACK/rocBLAS wrappers
- `pdmrg-gpu-opt/` — failed optimization experiment (0% win rate vs serial GPU, do not use)
- Paper: https://arxiv.org/abs/1301.3494 (Stoudenmire & White, 2013)
- Benchmark results: `benchmarks/paper_results/` (all CSV/JSON/TXT files)
