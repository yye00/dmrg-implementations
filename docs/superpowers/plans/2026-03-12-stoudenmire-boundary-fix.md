# Stoudenmire Boundary Coupling Fix Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the O(L) full-chain coupling sweep in pdmrg-gpu and pdmrg2-gpu with O(P) boundary-only merge+optimize per Stoudenmire & White (arXiv:1301.3494), using staggered sweep patterns so boundary environments are always fresh.

**Architecture:** The current outer loop does parallel segment sweeps (O(L/P)) then a full-chain serial sweep (O(L)) to couple segments, negating parallelism. The fix uses a staggered sweep pattern: even segments sweep LR while odd sweep RL, then boundary bonds where neighbors meet are optimized, then directions reverse for the second set of boundaries. This ensures fresh L_env and R_env at each boundary when it's optimized.

**Tech Stack:** HIP/ROCm, rocBLAS, C++ templates, std::thread parallelism, AMD MI300X GPU

**Remote build/test:** All compilation and testing on `ssh hotaisle@23.183.40.82` at `/home/hotaisle/dmrg-implementations`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `pdmrg-gpu/src/pdmrg_gpu.h` | Modify | Add `merge_and_optimize_boundaries()` declaration, remove `boundary_coupling_sweep` |
| `pdmrg-gpu/src/pdmrg_gpu_impl.h` | Modify | Implement `merge_and_optimize_boundaries()`, rewrite `run()` outer loop with staggered sweeps |
| `pdmrg2-gpu/src/pdmrg2_gpu.h` | Modify | Same header changes as pdmrg-gpu |
| `pdmrg2-gpu/src/pdmrg2_gpu_impl.h` | Modify | Same impl changes as pdmrg-gpu |

No new files needed — this is a focused refactor of the outer loop in both implementations.

---

## Background: Why Staggered Sweeps Are Required

After a segment does LR then RL:
- **L_envs are stale** (built during LR, but MPS changed during RL)
- **R_envs are fresh** (just built during RL)

For boundary optimization, we need BOTH L_env and R_env to be fresh. The staggered pattern solves this:

```
Half-sweep 1: even segments → LR,  odd segments → RL
  → Even segments build fresh L_envs at their right boundary
  → Odd segments build fresh R_envs at their left boundary
  → At even-numbered boundaries (between even seg k and odd seg k+1):
    L_env[seg_last_[k]] fresh ✓  R_env[seg_first_[k+1]+1] fresh ✓
  → Optimize EVEN boundaries

Half-sweep 2: even segments → RL,  odd segments → LR
  → Odd segments build fresh L_envs at their right boundary
  → Even segments build fresh R_envs at their left boundary
  → At odd-numbered boundaries (between odd seg k and even seg k+1):
    L_env[seg_last_[k]] fresh ✓  R_env[seg_first_[k+1]+1] fresh ✓
  → Optimize ODD boundaries
```

For 2 segments (1 boundary, even-indexed): only half-sweep 1 + even boundary coupling needed.
For 4+ segments: both half-sweeps + even and odd boundary coupling needed.

### Environment correctness after boundary coupling

Boundary coupling changes MPS[bsite] and MPS[bsite+1]. Key observations:
- L_env[bsite] is NOT invalidated (depends on MPS[bsite-1], not MPS[bsite])
- R_env[bsite+2] is NOT invalidated (depends on MPS[bsite+2], not MPS[bsite+1])
- The coupling updates L_env[bsite+1] and R_env[bsite+1] directly
- Segment sweeps after coupling use environments at their interior sites, which are unaffected

---

## Chunk 1: pdmrg-gpu Implementation

### Task 1: Add `merge_and_optimize_boundaries()` to pdmrg-gpu header

**Files:**
- Modify: `pdmrg-gpu/src/pdmrg_gpu.h:147` (replace `boundary_coupling_sweep` declaration)

- [ ] **Step 1: Edit the header file**

In `pdmrg-gpu/src/pdmrg_gpu.h`, replace line 147:
```cpp
    double boundary_coupling_sweep(int W = 4);  // B1: boundary-region only coupling
```
with:
```cpp
    double merge_and_optimize_boundaries(int parity = -1);  // Stoudenmire boundary coupling
```

- [ ] **Step 2: Verify the header compiles**

On remote:
```bash
ssh hotaisle@23.183.40.82
cd ~/dmrg-implementations && git pull
cd pdmrg-gpu/build && cmake .. -DGPU_TARGETS=gfx942 && make -j$(nproc) 2>&1 | tail -5
```
Expected: Compile error about missing `merge_and_optimize_boundaries` definition (since impl not written yet). That's OK — confirms header change is picked up.

---

### Task 2: Implement `merge_and_optimize_boundaries()` in pdmrg-gpu

**Files:**
- Modify: `pdmrg-gpu/src/pdmrg_gpu_impl.h:1068-1101` (replace `boundary_coupling_sweep` implementation)

- [ ] **Step 1: Replace `boundary_coupling_sweep` with `merge_and_optimize_boundaries`**

Replace the entire `boundary_coupling_sweep` method (lines 1068-1101) with:

```cpp
// ============================================================================
// Stoudenmire boundary merge+optimize (replaces full-chain coupling)
// Optimizes the two-site bond at each segment boundary.
// parity: 0 = even-indexed boundaries, 1 = odd, -1 = all
// Cost: O(P) bond optimizations instead of O(L) full-chain sweep.
// ============================================================================

template<typename Scalar>
double PDMRGGPU<Scalar>::merge_and_optimize_boundaries(int parity) {
    double energy = 0.0;
    int si = 0;  // boundary optimization uses stream 0

    for (int b = 0; b < (int)boundary_bonds_.size(); b++) {
        if (parity >= 0 && (b % 2) != parity) continue;

        int bsite = boundary_bonds_[b];

        // LR optimize: MPS[bsite] → left-canonical U, MPS[bsite+1] → S*Vh
        energy = optimize_bond(bsite, 'R', si);
        update_left_env(bsite, si);  // L_env[bsite+1] fresh

        // RL optimize: MPS[bsite] → U*S, MPS[bsite+1] → right-canonical Vh
        energy = optimize_bond(bsite, 'L', si);
        update_right_env(bsite + 1, si);  // R_env[bsite+1] fresh
    }
    return energy;
}
```

---

### Task 3: Rewrite pdmrg-gpu `run()` outer loop with staggered sweeps

**Files:**
- Modify: `pdmrg-gpu/src/pdmrg_gpu_impl.h:1163-1254` (the main PDMRG loop and polish phase)

- [ ] **Step 1: Replace the outer loop (lines 1163-1254)**

Replace everything from `// === Main PDMRG loop ===` (line 1163) through the end of `run()` (line 1254) with:

```cpp
    // === Main PDMRG loop (Stoudenmire staggered sweeps) ===
    double energy_prev = warmup_energy;
    energy_ = warmup_energy;
    bool outer_converged = false;

    auto parallel_sweep = [this](auto sweep_fn) {
        std::vector<std::thread> threads(n_segments_);
        for (int k = 0; k < n_segments_; k++) {
            threads[k] = std::thread([this, k, &sweep_fn]{ sweep_fn(this, k); });
        }
        for (auto& t : threads) t.join();
    };

    int n_boundaries = (int)boundary_bonds_.size();
    bool has_odd_boundaries = (n_boundaries > 1);

    for (int outer = 0; outer < n_outer_sweeps; outer++) {
        auto t_outer = std::chrono::high_resolution_clock::now();

        for (int local_sw = 0; local_sw < n_local_sweeps; local_sw++) {
            // Half-sweep 1: even segments LR, odd segments RL
            // After this, even-numbered boundaries have fresh L_env + R_env
            parallel_sweep([](PDMRGGPU* self, int k) {
                if (k % 2 == 0) self->segment_sweep_LR(k);
                else             self->segment_sweep_RL(k);
            });

            // Merge+optimize at even boundaries
            energy_ = merge_and_optimize_boundaries(0);

            // Half-sweep 2: even segments RL, odd segments LR
            // After this, odd-numbered boundaries have fresh L_env + R_env
            parallel_sweep([](PDMRGGPU* self, int k) {
                if (k % 2 == 0) self->segment_sweep_RL(k);
                else             self->segment_sweep_LR(k);
            });

            // Merge+optimize at odd boundaries (if any)
            if (has_odd_boundaries) {
                energy_ = merge_and_optimize_boundaries(1);
            }
        }

        auto t_outer_end = std::chrono::high_resolution_clock::now();
        double outer_time = std::chrono::duration<double>(t_outer_end - t_outer).count();
        double dE = std::abs(energy_ - energy_prev);

        // Print bond dimensions
        std::ostringstream chi_str;
        for (int i = 1; i < L_; i++) {
            chi_str << bond_dims_[i];
            if (i < L_ - 1) chi_str << ",";
        }
        std::cout << "Outer " << std::setw(3) << outer << ": E = " << std::setprecision(12)
                  << energy_ << ", dE = " << std::scientific << std::setprecision(2) << dE
                  << ", time = " << std::fixed << std::setprecision(3) << outer_time
                  << " s  chi=[" << chi_str.str() << "]" << std::endl;

        if (dE < tol_ && outer > 0) {
            std::cout << "Converged after " << outer + 1 << " outer iterations!" << std::endl;
            outer_converged = true;
            break;
        }

        energy_prev = energy_;
    }

    // === Polish phase: full-chain sweeps to converge to tight tolerance ===
    // B5: Skip polish if outer loop already converged (dE < tol)
    if (n_segments_ > 1 && !outer_converged) {
        int n_polish = 10;
        std::cout << "Polish sweeps (full-chain dmrg2, max " << n_polish << ")..." << std::endl;
        build_initial_environments();
        for (int sw = 0; sw < n_polish; sw++) {
            auto t_sw = std::chrono::high_resolution_clock::now();
            sweep_LR_full();
            double eRL = sweep_RL_full();
            auto t_sw_end = std::chrono::high_resolution_clock::now();
            double dE = std::abs(eRL - energy_);
            std::cout << "  Polish " << sw << ": E = " << std::fixed << std::setprecision(12)
                      << eRL << ", dE = " << std::scientific << std::setprecision(2) << dE
                      << ", time = " << std::fixed << std::setprecision(3)
                      << std::chrono::duration<double>(t_sw_end - t_sw).count()
                      << " s" << std::endl;
            energy_ = eRL;
            if (dE < tol_) {
                std::cout << "  Polish converged after " << sw + 1 << " sweeps" << std::endl;
                break;
            }
        }
    } else if (outer_converged) {
        std::cout << "Skipping polish (outer loop converged)" << std::endl;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << std::endl << "Total wall time: " << std::fixed << std::setprecision(3)
              << std::chrono::duration<double>(t_end - t_start).count() << " s" << std::endl;

    return energy_;
```

- [ ] **Step 2: Commit pdmrg-gpu changes**

```bash
git add pdmrg-gpu/src/pdmrg_gpu.h pdmrg-gpu/src/pdmrg_gpu_impl.h
git commit -m "feat(pdmrg-gpu): Stoudenmire boundary coupling with staggered sweeps

Replace O(L) full-chain coupling sweep with O(P) boundary-only
merge+optimize. Uses staggered sweep pattern (even LR/odd RL then
reverse) to ensure fresh environments at each boundary bond."
```

---

### Task 4: Build and test pdmrg-gpu on remote MI300X

**Files:** None (build/test only)

- [ ] **Step 1: Push changes and pull on remote**

```bash
git push origin main
ssh hotaisle@23.183.40.82 'cd ~/dmrg-implementations && git pull'
```

- [ ] **Step 2: Build pdmrg-gpu**

```bash
ssh hotaisle@23.183.40.82 'cd ~/dmrg-implementations/pdmrg-gpu/build && cmake .. -DGPU_TARGETS=gfx942 && make -j$(nproc) 2>&1'
```
Expected: Clean compile, no errors.

- [ ] **Step 3: Run Heisenberg L=8 chi=32 segments=2 test**

```bash
ssh hotaisle@23.183.40.82 'cd ~/dmrg-implementations/pdmrg-gpu/build && ./pdmrg_gpu 8 32 20 --segments 2 --warmup 3'
```
Expected: Energy = -3.374932598688 (error < 1e-10)

- [ ] **Step 4: Run Heisenberg L=32 chi=64 segments=4 test**

```bash
ssh hotaisle@23.183.40.82 'cd ~/dmrg-implementations/pdmrg-gpu/build && ./pdmrg_gpu 32 64 20 --segments 4 --warmup 3'
```
Expected: Energy = -13.997315618007 (error < 1e-10)

- [ ] **Step 5: Run Josephson L=6 chi=32 segments=2 test**

```bash
ssh hotaisle@23.183.40.82 'cd ~/dmrg-implementations/pdmrg-gpu/build && ./pdmrg_gpu 6 32 20 --segments 2 --warmup 3 --josephson'
```
Expected: Energy = -1.748843818181493 (error < 1e-10)

- [ ] **Step 6: If any test fails, debug and fix**

Common failure modes:
1. **Energy too high:** Boundary coupling not converging → increase n_outer_sweeps or n_local_sweeps
2. **NaN/inf:** Environment size mismatch at boundary → check `apply_heff_two_site` uses `d_L_envs_[site]` and `d_R_envs_[site+2]`
3. **Wrong energy:** Stale environments → verify staggered pattern ensures fresh envs at each boundary before optimization

---

## Chunk 2: pdmrg2-gpu Implementation

### Task 5: Add `merge_and_optimize_boundaries()` to pdmrg2-gpu header

**Files:**
- Modify: `pdmrg2-gpu/src/pdmrg2_gpu.h` (replace `boundary_coupling_sweep` declaration)

- [ ] **Step 1: Find and replace the declaration**

In `pdmrg2-gpu/src/pdmrg2_gpu.h`, replace:
```cpp
    double boundary_coupling_sweep(int W = 4);
```
with:
```cpp
    double merge_and_optimize_boundaries(int parity = -1);  // Stoudenmire boundary coupling
```

---

### Task 6: Implement Stoudenmire coupling in pdmrg2-gpu

**Files:**
- Modify: `pdmrg2-gpu/src/pdmrg2_gpu_impl.h` (replace `boundary_coupling_sweep`, rewrite `run()` outer loop)

- [ ] **Step 1: Replace `boundary_coupling_sweep` with `merge_and_optimize_boundaries`**

Find the `boundary_coupling_sweep` method and replace it entirely with:

```cpp
// ============================================================================
// Stoudenmire boundary merge+optimize (replaces full-chain coupling)
// parity: 0 = even-indexed boundaries, 1 = odd, -1 = all
// ============================================================================

template<typename Scalar>
double PDMRG2GPU<Scalar>::merge_and_optimize_boundaries(int parity) {
    double energy = 0.0;
    int si = 0;

    for (int b = 0; b < (int)boundary_bonds_.size(); b++) {
        if (parity >= 0 && (b % 2) != parity) continue;

        int bsite = boundary_bonds_[b];

        // LR optimize: left-canonicalize boundary
        energy = optimize_bond(bsite, 'R', si);
        update_left_env(bsite, si);

        // RL optimize: right-canonicalize boundary
        energy = optimize_bond(bsite, 'L', si);
        update_right_env(bsite + 1, si);
    }
    return energy;
}
```

- [ ] **Step 2: Rewrite the `run()` outer loop**

In the `run()` method, find the section starting with `// === Main PDMRG loop ===` and replace through the end of the method.

The new outer loop uses identical staggered sweep logic as pdmrg-gpu, but with `PDMRG2GPU` type in the parallel_sweep lambda:

```cpp
    // === Main PDMRG loop (Stoudenmire staggered sweeps) ===
    double energy_prev = warmup_energy;
    energy_ = warmup_energy;
    bool outer_converged = false;

    auto parallel_sweep = [this](auto sweep_fn) {
        std::vector<std::thread> threads(n_segments_);
        for (int k = 0; k < n_segments_; k++) {
            threads[k] = std::thread([this, k, &sweep_fn]{ sweep_fn(this, k); });
        }
        for (auto& t : threads) t.join();
    };

    int n_boundaries = (int)boundary_bonds_.size();
    bool has_odd_boundaries = (n_boundaries > 1);

    for (int outer = 0; outer < n_outer_sweeps; outer++) {
        auto t_outer = std::chrono::high_resolution_clock::now();

        for (int local_sw = 0; local_sw < n_local_sweeps; local_sw++) {
            // Half-sweep 1: even segments LR, odd segments RL
            parallel_sweep([](PDMRG2GPU* self, int k) {
                if (k % 2 == 0) self->segment_sweep_LR(k);
                else             self->segment_sweep_RL(k);
            });

            // Merge+optimize at even boundaries
            energy_ = merge_and_optimize_boundaries(0);

            // Half-sweep 2: even segments RL, odd segments LR
            parallel_sweep([](PDMRG2GPU* self, int k) {
                if (k % 2 == 0) self->segment_sweep_RL(k);
                else             self->segment_sweep_LR(k);
            });

            // Merge+optimize at odd boundaries (if any)
            if (has_odd_boundaries) {
                energy_ = merge_and_optimize_boundaries(1);
            }
        }

        auto t_outer_end = std::chrono::high_resolution_clock::now();
        double outer_time = std::chrono::duration<double>(t_outer_end - t_outer).count();
        double dE = std::abs(energy_ - energy_prev);

        std::ostringstream chi_str;
        for (int i = 1; i < L_; i++) {
            chi_str << bond_dims_[i];
            if (i < L_ - 1) chi_str << ",";
        }
        std::cout << "Outer " << std::setw(3) << outer << ": E = " << std::setprecision(12)
                  << energy_ << ", dE = " << std::scientific << std::setprecision(2) << dE
                  << ", time = " << std::fixed << std::setprecision(3) << outer_time
                  << " s  chi=[" << chi_str.str() << "]" << std::endl;

        if (dE < tol_ && outer > 0) {
            std::cout << "Converged after " << outer + 1 << " outer iterations!" << std::endl;
            outer_converged = true;
            break;
        }

        energy_prev = energy_;
    }
```

Keep the existing timing printout for parallel/coupling breakdown (adjust labels from "coup=" to "boundary=").

Keep the polish phase and total wall time printout from the existing code, unchanged.

- [ ] **Step 3: Commit pdmrg2-gpu changes**

```bash
git add pdmrg2-gpu/src/pdmrg2_gpu.h pdmrg2-gpu/src/pdmrg2_gpu_impl.h
git commit -m "feat(pdmrg2-gpu): Stoudenmire boundary coupling with staggered sweeps

Same fix as pdmrg-gpu: replace O(L) full-chain coupling with O(P)
boundary-only merge+optimize using staggered sweep pattern."
```

---

### Task 7: Build and test pdmrg2-gpu on remote MI300X

**Files:** None (build/test only)

- [ ] **Step 1: Push and pull**

```bash
git push origin main
ssh hotaisle@23.183.40.82 'cd ~/dmrg-implementations && git pull'
```

- [ ] **Step 2: Build pdmrg2-gpu**

```bash
ssh hotaisle@23.183.40.82 'cd ~/dmrg-implementations/pdmrg2-gpu/build && cmake .. -DGPU_TARGETS=gfx942 && make -j$(nproc) 2>&1'
```
Expected: Clean compile.

- [ ] **Step 3: Run Heisenberg L=8 chi=32 segments=2**

```bash
ssh hotaisle@23.183.40.82 'cd ~/dmrg-implementations/pdmrg2-gpu/build && ./pdmrg2_gpu 8 32 20 --segments 2 --warmup 3'
```
Expected: Energy = -3.374932598688 (error < 1e-10)

- [ ] **Step 4: Run Heisenberg L=32 chi=64 segments=4**

```bash
ssh hotaisle@23.183.40.82 'cd ~/dmrg-implementations/pdmrg2-gpu/build && ./pdmrg2_gpu 32 64 20 --segments 4 --warmup 3'
```
Expected: Energy = -13.997315618007 (error < 1e-10)

- [ ] **Step 5: Run Josephson L=6 chi=32 segments=2**

```bash
ssh hotaisle@23.183.40.82 'cd ~/dmrg-implementations/pdmrg2-gpu/build && ./pdmrg2_gpu 6 32 20 --segments 2 --warmup 3 --josephson'
```
Expected: Energy = -1.748843818181493 (error < 1e-10)

---

## Chunk 3: Scalability Benchmarks

### Task 8: Run scalability tests comparing old vs new coupling

**Files:** None (benchmarking only)

- [ ] **Step 1: Run pdmrg-gpu scalability series**

```bash
ssh hotaisle@23.183.40.82 'cd ~/dmrg-implementations/pdmrg-gpu/build && \
  echo "=== segments=2 ===" && ./pdmrg_gpu 64 128 20 --segments 2 --warmup 3 && \
  echo "=== segments=4 ===" && ./pdmrg_gpu 64 128 20 --segments 4 --warmup 3 && \
  echo "=== segments=8 ===" && ./pdmrg_gpu 64 128 20 --segments 8 --warmup 3'
```

Record: wall time, outer iterations to converge, parallel vs boundary coupling time breakdown.

- [ ] **Step 2: Run pdmrg2-gpu scalability series**

```bash
ssh hotaisle@23.183.40.82 'cd ~/dmrg-implementations/pdmrg2-gpu/build && \
  echo "=== segments=2 ===" && ./pdmrg2_gpu 64 128 20 --segments 2 --warmup 3 && \
  echo "=== segments=4 ===" && ./pdmrg2_gpu 64 128 20 --segments 4 --warmup 3 && \
  echo "=== segments=8 ===" && ./pdmrg2_gpu 64 128 20 --segments 8 --warmup 3'
```

- [ ] **Step 3: Compare results**

Expected performance targets:
- 2 segments: ~1.5-1.8x speedup vs serial (segments=1)
- 4 segments: ~2.5-3.5x speedup
- 8 segments: ~4-6x speedup

The key metric: coupling phase should now be negligible (O(P) boundary bonds) vs the old O(L) full-chain sweep.

---

## Key Implementation Notes

### What `apply_heff_two_site` needs at each boundary bond
- `d_L_envs_[bsite]` — left environment AT the boundary site (from left segment's last LR sweep)
- `d_R_envs_[bsite + 2]` — right environment PAST the boundary (from right segment's last RL sweep)
- `d_WW_[bsite]` — fused two-site MPO (precomputed, always valid)

### Difference between pdmrg-gpu and pdmrg2-gpu segment sweeps

**pdmrg-gpu segment sweeps** are minimal — just optimize+env_update in a loop. They rely on environments being pre-built (from warmup or previous iteration).

**pdmrg2-gpu segment sweeps** are self-contained — they pre-canonicalize the segment via Newton-Schulz and rebuild all environments before sweeping. This makes them more robust but costlier.

Both use the exact same `merge_and_optimize_boundaries()` implementation since `optimize_bond()` handles the eigensolver difference internally (Lanczos vs Block-Davidson).

### Why no V = Lambda^{-1} matrices

The Stoudenmire paper uses V matrices to decouple segments so boundary tensors can be modified independently. In our implementation, segments already operate on non-overlapping site ranges, so V matrices are unnecessary for correctness. They could improve convergence rate for poorly conditioned boundaries but add significant complexity. Can be added as a follow-up if convergence is slow.
