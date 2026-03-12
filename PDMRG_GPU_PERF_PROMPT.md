# PDMRG-GPU Performance Optimization Prompt

## Context

We have a working stream-parallel DMRG (PDMRG) implementation on AMD MI300X (gfx942, 304 CUs, 192GB HBM3, ROCm 7.2). It uses HIP/ROCm with rocBLAS for tensor contractions, Lanczos eigensolver, and CPU LAPACK SVD. The code is templated for `double` (real) and `hipDoubleComplex` (complex128).

**Current performance (Heisenberg OBC, 8 segments, CPU SVD):**

| System | dmrg2-gpu (serial) | pdmrg-gpu (parallel) | Speedup |
|--------|-----------|---------------|---------|
| L=32 chi=64 | 2.1s | 5.8s | 0.36× |
| L=64 chi=128 | 27.2s | 24.6s | 1.10× |
| L=64 chi=256 | 144.7s | 114.1s | 1.27× |
| L=128 chi=128 | 74.4s | 76.0s | 0.98× |

The goal is to maximize the speedup ratio, especially at chi≥128 where the GPU should dominate.

**Correctness requirements:** All optimizations must preserve accuracy to < 1e-10 on these tests:
- Heisenberg L=8 chi=32 segments=2: exact E = -3.374932598688
- Heisenberg L=32 chi=64 segments=4: exact E = -13.997315618007
- Josephson L=6 chi=32 segments=2: exact E = -1.748843818181493

## Repository Layout

```
pdmrg-gpu/
├── CMakeLists.txt           # Links hip::device, roc::rocblas, roc::rocsolver, LAPACK
└── src/
    ├── pdmrg_gpu.h          # Class declaration, StreamWorkspace struct
    ├── pdmrg_gpu_impl.h     # Full implementation (~1450 lines)
    ├── scalar_traits.h      # ScalarTraits<double/hipDoubleComplex>, rocBLAS dispatch, GPU kernels
    ├── accurate_svd.h       # Recursive accurate SVD (Stoudenmire Appendix), CPU BLAS
    └── test_pdmrg_gpu.cpp   # Test harness with Heisenberg + Josephson MPO builders
```

## Architecture

The algorithm has three phases:
1. **Warmup:** Full-chain two-site DMRG sweeps on stream 0 (identical to dmrg2-gpu)
2. **Outer loop:** Parallel segment sweeps (one `std::thread` per segment, each with its own HIP stream + rocBLAS handle) → full-chain coupling sweep on stream 0 (env rebuild + LR + RL sweep)
3. **Polish:** Full-chain sweeps on stream 0 until convergence

The hot path per bond optimization is: `form_theta_two_site` (1 GEMM) → `lanczos_eigensolver` (10-50 iterations of `apply_heff_two_site` + reductions) → `svd_split` (CPU LAPACK SVD + H2D upload).

Each `apply_heff_two_site` has 3 steps: batched GEMM (Step 1) → dense GEMM (Step 2) → loop of D×d² small GEMMs (Step 3).

## Remote MI300X Access

```bash
ssh hotaisle@23.183.40.82   # passwordless
cd ~/dmrg-implementations/pdmrg-gpu/build
cmake .. -DGPU_TARGETS=gfx942 && make -j$(nproc)
./pdmrg_gpu 32 64 3 --segments 4 --warmup 3       # Heisenberg L=32
./pdmrg_gpu 64 128 2 --segments 8 --warmup 1       # Heisenberg L=64
./pdmrg_gpu 6 32 5 --segments 2 --warmup 3 --josephson  # Josephson L=6
```

---

## Part A: Software / Systems Optimizations

These are optimizations to the GPU programming model, memory management, and dispatch patterns. They do not change the DMRG algorithm itself.

### A1. Eliminate Implicit Synchronization in Lanczos Inner Loop

**Problem:** rocBLAS defaults to `rocblas_pointer_mode_host`. Every `nrm2` and `dot` call writes results to a host pointer, forcing an implicit `hipStreamSynchronize`. The Lanczos loop has 2 blocking reductions per iteration (dot for alpha, nrm2 for beta), creating ~100 sync stalls per bond optimization.

**Current code** (`pdmrg_gpu_impl.h:709,746`):
```cpp
Scalar alpha_result;
ROCBLAS_CHECK(Traits::dot(handles_[si], n, d_vi, 1, ws.d_heff_result, 1, &alpha_result));
double alpha_i = Traits::real_part(alpha_result);
// ...
double beta_i;
ROCBLAS_CHECK(Traits::nrm2(handles_[si], n, ws.d_heff_result, 1, &beta_i));
```

**Fix:** Switch to `rocblas_pointer_mode_device`. Allocate device scalars for alpha/beta in `StreamWorkspace`. Use device-mode `dot`/`nrm2` that write to device memory (no sync). Only copy alpha/beta to host every N iterations for convergence checking (or use a device-side convergence kernel). The `scal` and `axpy` calls that consume alpha/beta must also use device pointers.

**Impact:** Eliminates ~100 sync stalls per bond optimization. Each stall flushes the GPU pipeline.

### A2. Pinned Host Memory for SVD Buffers

**Problem:** All host-side SVD buffers (`h_svd_A`, `h_svd_U`, `h_svd_S`, `h_svd_Vh`, `h_svd_tmp`, etc.) in `StreamWorkspace` are `std::vector` — pageable memory. `hipMemcpyAsync` to/from pageable memory silently degrades to synchronous on AMD hardware.

**Current code** (`pdmrg_gpu.h:98-100`, `pdmrg_gpu_impl.h:853`):
```cpp
struct StreamWorkspace {
    // ...
    std::vector<Scalar> h_svd_A, h_svd_U, h_svd_Vh, h_svd_work, h_svd_tmp;
    std::vector<RealType> h_svd_S;
};
```

**Fix:** Replace `std::vector` with `hipHostMalloc`-allocated pinned buffers (wrapped in a RAII class or managed manually). This enables truly asynchronous D2H/H2D transfers, allowing the GPU to continue processing while SVD data moves.

**Impact:** Eliminates sync stalls on every SVD-related transfer (2 per bond optimization in CPU SVD path).

### A3. Cache Batched GEMM Pointer Arrays

**Problem:** Every call to `apply_heff_two_site` (called 10-50× per Lanczos) allocates 3 `std::vector<Scalar*>` on the heap, fills them in a CPU loop, and uploads them via `hipMemcpyAsync` to device.

**Current code** (`pdmrg_gpu_impl.h:430-441`):
```cpp
std::vector<Scalar*> h_A(batch_count), h_B(batch_count), h_C(batch_count);
for (int w = 0; w < D; w++)
    for (int s1 = 0; s1 < d; s1++)
        for (int s2 = 0; s2 < d; s2++) { /* pointer arithmetic */ }
HIP_CHECK(hipMemcpyAsync(ws.d_batch_A, h_A.data(), ...));
```

**Fix:** Pre-compute pointer arrays once per bond optimization (when cL/cR are known), store in pinned host memory, upload once. For Step 1 of `apply_heff`, the `h_B` pointers reference `d_theta_in` which changes per Lanczos iteration, but the offsets are stride-constant — precompute base + offset pattern. The `h_A` and `h_C` pointers are completely static between SVD truncations.

Same issue exists in `update_left_env` (lines 513-515) and `update_right_env` (lines 589-591).

**Impact:** Eliminates thousands of heap allocations + H2D transfers per sweep.

### A4. Batch Step 3 GEMM Loops

**Problem:** Step 3 of `apply_heff_two_site` issues `d² × D` separate GEMM calls in a triple-nested CPU loop (e.g., 20 calls for Heisenberg, 36 for Josephson d=3). Similarly, `update_left_env` and `update_right_env` each issue `D × d` GEMMs per site.

**Current code** (`pdmrg_gpu_impl.h:464-479`):
```cpp
for (int s1p = 0; s1p < d; s1p++)
    for (int s2p = 0; s2p < d; s2p++)
        for (int n = 0; n < D; n++) {
            Scalar beta = (n == 0) ? zero : one;
            ROCBLAS_CHECK(Traits::gemm(handles_[si], ...));
        }
```

**Fix:** Replace with `rocblas_gemm_batched` or `rocblas_gemm_strided_batched`. The inner loop (over `n` or `sp`) accumulates into the same output slice — restructure by pre-summing the contributions or reformulating as a single larger GEMM with extended K dimension.

Alternative: Use hipTensor for the entire contraction. hipTensor supports gfx942, f64, cf64, and handles arbitrary tensor contractions via Einstein notation. This would replace all 3 steps of `apply_heff` with a single `hiptensorContraction` call. See: https://rocm.docs.amd.com/projects/hipTensor/en/docs-7.2.0/api-reference/api-reference.html

**Impact:** Reduces per-GEMM dispatch overhead and allows the GPU to optimize the full batch.

### A5. GPU-Side SVD Singular Value Scaling

**Problem:** After SVD, the singular value scaling (`S * Vh` or `U * S`) is done in a CPU double-loop on host memory, then uploaded to GPU.

**Current code** (`pdmrg_gpu_impl.h:904-906`):
```cpp
for (int j = 0; j < n_svd; j++)
    for (int i = 0; i < new_k; i++)
        ws.h_svd_tmp[i + j * new_k] = Traits::scale_by_real(h_S_data[i], h_Vh_data[i + j * full_k]);
```

**Fix:** Upload U, S, Vh to GPU separately, then use a simple HIP kernel (or `rocblas_dgmm` with diagonal S) to do the column/row scaling on device. Avoids the CPU loop and the round-trip upload of the result.

**Impact:** Eliminates O(chi² × d) CPU memory operations and one H2D transfer per bond optimization.

### A6. Remove Dead Code

The following methods/data are completely unused after the algorithm was changed from boundary-only coupling to full-chain coupling:

- `compute_boundary_V()` — dead
- `optimize_boundary_bond()` — dead
- `boundary_coupling_sweep()` — dead
- `form_boundary_theta()` — dead
- `merge_boundary()` — dead
- `rebuild_boundary_envs()` — dead
- `d_V_boundary_` + `h_V_boundary_` data members — dead (wastes GPU memory)
- `column_scale_real` kernel in `scalar_traits.h` — dead
- `accurate_svd.h` include and file — dead (only used by dead methods)

Remove all of these. Also remove declarations from `pdmrg_gpu.h`.

**Impact:** Code clarity, reduced binary size, freed GPU memory.

### A7. Reduce Synchronization in Full-Chain Sweeps

**Problem:** `sweep_LR_full` and `sweep_RL_full` call `hipStreamSynchronize(streams_[0])` after every single bond optimization (inherited from `optimize_bond`). This prevents pipeline overlap between the env update of bond i and the theta formation of bond i+1.

**Fix:** The `svd_split` CPU SVD path already forces a sync (D2H copy). But within the GPU SVD path, the operations are all on the same stream and naturally ordered. Remove the explicit syncs in `optimize_bond` (already partially done — verify no remaining unnecessary syncs).

---

## Part B: Numerical / Algorithmic Optimizations

These change the DMRG algorithm itself to reduce total work or improve convergence.

### B1. Reduce Coupling Sweep Cost

**Problem:** Each outer iteration does a full-chain coupling sweep (env rebuild + LR + RL sweep = ~3L bond optimizations + 2L env updates). This is the dominant serial bottleneck, taking ~5-13s per outer iteration at L=64 chi=128-256.

**Options to explore:**
1. **Boundary-region sweep instead of full-chain:** Only optimize bonds within ±W sites of each boundary (e.g., W=4). This is O(P×W) instead of O(L). Requires rebuilding environments incrementally only around the boundary regions.
2. **Single-direction coupling:** Do only LR (not LR+RL) for the coupling sweep, halving the cost. The segment sweeps already alternate direction.
3. **Skip coupling when converged:** If the energy change from the last outer iteration is small, skip the coupling sweep entirely.

**Impact:** Could halve or more the per-iteration cost, directly improving the speedup ratio.

### B2. Adaptive Warmup Count

**Problem:** Currently warmup count is a fixed parameter. For L=64 chi=128, 1 warmup sweep suffices (energy converges to ~1e-6). For larger chi or L, more may be needed.

**Fix:** Use convergence-based early exit in warmup (dE < threshold after first sweep → stop). The threshold can be less strict than the final tolerance since the outer loop + polish will converge further.

**Impact:** Saves 1-2 warmup sweeps in cases where they're unnecessary.

### B3. Reduce Lanczos Iterations via Better Initial Guess

**Problem:** After segment sweeps, the MPS at boundary sites has been re-optimized with stale environments. The coupling sweep's Lanczos starts from `theta = MPS[site] * MPS[site+1]`, which is already close to the ground state (the segment sweep just optimized it). But Lanczos doesn't know this and may still run 10-50 iterations.

**Fix:**
1. Tighten the Lanczos convergence tolerance adaptively: if the initial energy estimate is already close to the previous converged value, use a looser convergence threshold.
2. Reduce `max_lanczos_iter_` for the coupling sweep (the MPS is already close to optimal).

**Impact:** Fewer matvec evaluations per bond → faster sweeps.

### B4. Overlap Segment Sweeps with Environment Rebuild

**Problem:** Phase 1 (parallel segments) and Phase 2 (env rebuild + coupling) are strictly sequential. The environment rebuild in Phase 2 is serial on stream 0.

**Fix:** Pipeline the phases: while segment k+1 is still sweeping on stream k+1, start rebuilding environments for the portion of the chain covered by already-completed segment k on stream 0. This requires careful synchronization (events or barriers) but could overlap ~25-50% of the env rebuild with segment sweep tail.

**Impact:** Reduces the sequential gap between parallel and serial phases.

### B5. Skip Polish When Outer Loop Converges

**Problem:** The polish phase always runs (up to 10 sweeps) even if the outer loop already converged.

**Fix:** Track the full-chain energy from the coupling sweep. If dE < tol after the outer loop, skip polish entirely.

**Impact:** Saves 1-2 full-chain sweeps when the algorithm converges early.

---

## Implementation Order (Suggested)

**High-impact, moderate effort:**
1. A6 (dead code removal) — clean slate
2. A1 (device pointer mode) — biggest single-GPU speedup
3. A3 (cache pointer arrays) — eliminates thousands of allocations
4. B1 (boundary-region coupling) — biggest algorithmic speedup

**Medium-impact:**
5. A4 (batch Step 3 GEMMs) — reduces dispatch overhead
6. A5 (GPU-side S scaling) — eliminates CPU loop + transfer
7. A2 (pinned memory) — enables truly async transfers
8. B3 (adaptive Lanczos) — fewer matvec iterations

**Lower priority:**
9. B2 (adaptive warmup) — parameter tuning
10. B4 (overlap phases) — complex synchronization
11. B5 (skip polish) — small gain

## Build and Test

```bash
ssh hotaisle@23.183.40.82
cd ~/dmrg-implementations && git pull
cd pdmrg-gpu/build && cmake .. -DGPU_TARGETS=gfx942 && make -j$(nproc)

# Correctness (all must show PASS with error < 1e-10):
./pdmrg_gpu 8 32 5 --segments 2 --warmup 3
./pdmrg_gpu 32 64 3 --segments 4 --warmup 3
./pdmrg_gpu 6 32 5 --segments 2 --warmup 3 --josephson

# Performance benchmarks (compare to dmrg2-gpu):
./pdmrg_gpu 64 128 2 --segments 8 --warmup 1 --local-sweeps 1
./pdmrg_gpu 64 256 2 --segments 8 --warmup 1 --local-sweeps 1

# dmrg2-gpu baseline:
cd ~/dmrg-implementations/dmrg2-gpu/build
./dmrg2_gpu 64 128 5
./dmrg2_gpu 64 256 5
```

## Success Criteria

1. All correctness tests pass with error < 1e-10
2. PDMRG speedup over dmrg2-gpu at L=64 chi=128: target ≥ 1.5× (currently 1.10×)
3. PDMRG speedup over dmrg2-gpu at L=64 chi=256: target ≥ 2.0× (currently 1.27×)
4. No regressions at L=32 chi=64 (currently 5.8s, should not get worse)
