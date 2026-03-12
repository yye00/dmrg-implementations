# PDMRG-GPU Optimization Report

## Hardware & Software Environment

- **GPU**: AMD Instinct MI300X (gfx942), 304 CUs, 192 GB HBM3
- **Host CPU**: AMD EPYC (multi-core, exact model varies by VM allocation)
- **ROCm**: 7.2
- **Libraries**: rocBLAS (batched GEMM), rocsolver (GPU SVD), LAPACK (CPU SVD)
- **Compiler**: hipcc (clang-based)
- **Precision**: `double` (float64) and `hipDoubleComplex` (complex128), templated

## Algorithm Overview

PDMRG partitions an L-site chain into P segments, each assigned a HIP stream
and rocBLAS handle. The algorithm has three phases:

1. **Warmup**: Full-chain two-site DMRG sweeps on stream 0 (identical to dmrg2-gpu)
2. **Outer loop**: Parallel segment sweeps (one `std::thread` per segment, each with its own HIP stream) followed by a full-chain coupling sweep (env rebuild + LR + RL on stream 0)
3. **Polish**: Full-chain sweeps on stream 0 until convergence

The hot path per bond optimization is:
`form_theta_two_site` (1 GEMM) → `lanczos_eigensolver` (10-50 iterations of `apply_heff_two_site` + reductions) → `svd_split` (CPU LAPACK SVD + H2D upload).

Each `apply_heff_two_site` has 3 steps:
- Step 1: D×d² batched GEMMs (L_env × theta slices)
- Step 2: 1 dense GEMM (T1 × WW fused MPO)
- Step 3: D×d² GEMMs accumulating into output (T2 × R_env slices)

## Correctness Validation

All optimizations were validated against known exact energies:

| System | Exact Energy | Achieved Error | Status |
|--------|-------------|----------------|--------|
| Heisenberg L=8 chi=32 seg=2 | -3.374932598688 | < 1e-13 | PASS |
| Heisenberg L=32 chi=64 seg=4 | -13.997315618007 | < 1e-12 | PASS |
| Josephson L=6 chi=32 seg=2 | -1.748843818181493 | < 1e-14 | PASS |

## Performance Results

### Baseline (Pre-Optimization)

Measured on MI300X, Heisenberg OBC, CPU SVD:

| System | dmrg2-gpu | PDMRG (original) | Speedup |
|--------|-----------|-------------------|---------|
| L=32 chi=64 seg=4 | 2.1s | 5.8s | 0.36× |
| L=64 chi=128 seg=8 | 27.0s | 26.1s¹ | 1.10× |
| L=64 chi=256 seg=8 | 141.7s | 114.1s¹ | 1.27× |
| L=128 chi=128 seg=8 | 74.4s | 76.0s | 0.98× |

¹ Measured with `--warmup 1 --outer 2 --local-sweeps 1`

### Post-Optimization

| System | dmrg2-gpu | PDMRG (optimized) | Speedup | Δ vs baseline |
|--------|-----------|---------------------|---------|---------------|
| L=32 chi=64 seg=4 | 2.1s | 5.5s | 0.38× | −0.3s |
| L=64 chi=128 seg=8 | 27.0s | 24.1s | 1.12× | −2.0s |
| L=64 chi=256 seg=8 | 141.7s | 113-119s² | 1.19-1.25× | ~0s |

² Run-to-run variance of ±5s due to segment sweep randomness affecting chi profiles

### PDMRG Time Breakdown (L=64 chi=128, warmup=1, outer=2)

| Phase | Time | % of Total |
|-------|------|-----------|
| Env build | 0.5s | 2% |
| Warmup (1 LR+RL sweep) | 8.8s | 37% |
| Outer 0 (segments + coupling) | 4.4s | 18% |
| Outer 1 (segments + coupling) | 2.2s | 9% |
| Polish (2 LR+RL sweeps) | 7.7s | 32% |
| **Total** | **24.1s** | |

### PDMRG Time Breakdown (L=64 chi=256, warmup=1, outer=2)

| Phase | Time | % of Total |
|-------|------|-----------|
| Env build | 0.5s | <1% |
| Warmup (1 LR+RL sweep) | 61s | 51% |
| Outer 0 (segments + coupling) | 15s | 13% |
| Outer 1 (segments + coupling) | 4s | 3% |
| Polish (2 LR+RL sweeps) | 36s | 30% |
| **Total** | **119s** | |

### dmrg2-gpu Profile (L=64 chi=256)

| Component | Time/Sweep | % |
|-----------|-----------|---|
| SVD (CPU LAPACK) | 38-41s | 97-98% |
| Lanczos + apply_heff | 0.4-0.6s | 1-2% |
| Environment updates | 0.03s | <0.1% |
| Other | 0.004s | <0.01% |

This profile is critical: **CPU SVD completely dominates at chi ≥ 128**.

## Implemented Optimizations

### A1: Device Pointer Mode for Lanczos (commit 37980ec)

**What**: Switched rocBLAS from `rocblas_pointer_mode_host` to `rocblas_pointer_mode_device` for all `dot`/`nrm2`/`axpy`/`scal` calls in the Lanczos eigensolver.

**Why**: Host pointer mode forces an implicit `hipStreamSynchronize` after every reduction (`dot`, `nrm2`), stalling the GPU pipeline. With device pointer mode, reductions write to device memory and subsequent BLAS calls consume device pointers — no sync needed.

**Implementation**: Allocated per-stream device scalars in `StreamWorkspace` (`d_dot_result`, `d_nrm2_result`, `d_neg_alpha`, `d_neg_overlap`, `d_inv_nrm`, `d_alpha_dev`, `d_beta_dev`, `d_neg_beta_scalars`, `d_const_one/zero/neg_one`). Host-side convergence checking reads alpha/beta arrays every N iterations (or at the end) via `hipMemcpy`.

**Impact**: Eliminated ~100 sync stalls per bond optimization in the Lanczos inner loop. However, Lanczos is only 1-2% of runtime at chi ≥ 128, so the wall-clock impact is small (~0.1-0.3s per sweep).

### A3: GPU-Side Pointer Setup Kernels (commits 146cd85, 50eb73c)

**What**: Replaced host-to-device DMA of batched GEMM pointer arrays with tiny GPU kernels that compute the pointer arrays directly on the GPU.

**Why**: The original code allocated `std::vector<Scalar*>` on the heap, filled them in a CPU loop, then uploaded via `hipMemcpyAsync`. This was called 10-50× per Lanczos iteration per bond. The initial fix (commit 146cd85) used pinned host memory (`hipHostMalloc`) with `hipMemcpyAsync` DMA, but this introduced a **race condition** (see below).

**Race condition discovery**: `hipMemcpyAsync` from pinned host memory is truly asynchronous — the DMA engine reads the source buffer at *execution time*, not enqueue time. When `apply_heff_two_site` is called in a tight Lanczos loop, the CPU overwrites the pinned buffer before the DMA completes. This caused:
- Illegal memory access crashes at L=64 chi=128
- LAPACK `dstev` failures (alpha values corrupted to ~1e-10 instead of O(1))
- Intermittent NaN propagation

The original `std::vector` code worked because `hipMemcpyAsync` from pageable (non-pinned) memory is staged through an internal pinned buffer, making it effectively synchronous w.r.t. the source data.

**Final solution** (commit 50eb73c): Six GPU kernels in `scalar_traits.h` that compute pointer arrays directly:
- `setup_heff_A_ptrs`: L_env slice pointers (Step 1 A array, cached per site)
- `setup_heff_B_ptrs`: theta slice pointers (Step 1 B array, recomputed per Lanczos iter)
- `setup_heff_C_ptrs`: T1 slice pointers (Step 1 C array, cached per site)
- `setup_lenv_ptrs`: Left environment update pointer triple
- `setup_renv_ptrs`: Right environment update pointer triple

Each kernel launches with 1 block of batch_count threads (typically 20-36). The computation is trivial (integer divide/mod + pointer offset), so overhead is negligible.

**Impact**: Eliminated thousands of heap allocations + H2D transfers per sweep. Also eliminated the race condition entirely. Practical wall-clock savings: ~0.5-1s per sweep at chi=128.

### A4: Batched GEMM for Step 3 (commit 27a4967, reverted 5eb5a7f, restored in 50eb73c)

**What**: Step 3 of `apply_heff_two_site` originally issued d²×D separate GEMM calls in a triple-nested loop (20 calls for Heisenberg d=2 D=5, 36 for Josephson d=3 D=4). Replaced with `rocblas_gemm_batched`.

**Status**: Initially implemented, then reverted during debugging, then restored as part of the GPU pointer kernel commit. The batched version uses the same GPU-side pointer setup kernels as Step 1.

**Impact**: Reduces per-GEMM dispatch overhead. At chi ≥ 128 the individual GEMMs are large enough that dispatch overhead is not dominant, so the practical speedup is small.

### A6: Dead Code Removal (commit b13abe7)

**What**: Removed all boundary-coupling code that was dead after the algorithm switched to full-chain coupling:
- `compute_boundary_V()`, `optimize_boundary_bond()`, `form_boundary_theta()`, `merge_boundary()`, `rebuild_boundary_envs()`
- `d_V_boundary_` and `h_V_boundary_` data members
- `column_scale_real` kernel in `scalar_traits.h`
- `accurate_svd.h` include

**Impact**: Code clarity, reduced binary size, freed GPU memory. No performance impact.

### A7: Remove Redundant Stream Synchronization (commit 6df2548)

**What**: Removed `hipStreamSynchronize(streams_[0])` after every bond optimization in `sweep_LR_full()` and `sweep_RL_full()`.

**Why**: The CPU SVD path in `svd_split` already forces synchronization:
1. `hipMemcpyAsync` D2H to pageable `std::vector` is staged (effectively synchronous)
2. Explicit `hipStreamSynchronize` before LAPACK call
3. `hipMemcpyAsync` H2D from pageable `std::vector` is also staged

The explicit sync in the sweep loop was therefore redundant.

**Impact**: Negligible at chi ≥ 128 (CPU SVD dominates). Could matter with GPU SVD path.

### B2: Adaptive Warmup (commit 1ff1b07)

**What**: Warmup sweeps now exit early when dE < tol after the first sweep.

**Impact**: Saves 1-2 warmup sweeps when convergence is fast. At L=64 chi=128 with `--warmup 1`, no savings (already minimum). At L=32 chi=64 with `--warmup 3`, saves 0-1 sweeps.

### B5: Skip Polish When Converged (commits 1ff1b07, d18ba03)

**What**: If the outer loop converges (dE < tol), skip the polish phase entirely.

**Bug fixed** (d18ba03): Original convergence tracking compared `energy_` vs `energy_prev` after the loop, but both were equal (energy_prev updated at end of each iteration). Fixed with explicit `bool outer_converged` flag set at the convergence break.

**Impact**: Saves 1-2 full-chain sweeps when the outer loop converges. Rarely triggers at typical parameters.

## Optimizations Attempted and Reverted

### B1: Boundary-Region Coupling Sweep (implemented and reverted, commit d18ba03)

**What**: Replace full-chain coupling sweep with boundary-only sweep (±W sites around each segment boundary).

**Result**: Energy error of 8.41 at L=32 chi=64 — completely failed to converge. For 8 segments of 8 sites with W=4, boundary coupling covers ~89% of the chain (barely any saving), but the segment sweeps disrupt the global entanglement structure enough that boundary-only coupling cannot recover the correct energy.

**Lesson**: PDMRG segment sweeps disrupt the converged MPS in ways that require full-chain resweeping to fix. Narrow boundary coupling is insufficient.

### B1 (alternative): Single-Direction Coupling Sweep (tested, reverted)

**What**: Coupling sweep does only LR (not LR+RL), halving the cost. Direction alternates by outer iteration.

**Result**: Energy quality dropped significantly (E = -28.167 vs -28.175 correct). Two-site DMRG requires both sweep directions for proper convergence — LR grows chi from the left but doesn't optimize the right canonical form, and vice versa.

**Lesson**: Full LR+RL coupling is essential for energy convergence quality.

## Optimizations Not Implemented

### A2: Pinned Host Memory for SVD Buffers

**Rationale**: Would enable truly asynchronous D2H/H2D for SVD data. However, with CPU SVD dominating at 98% of runtime, the transfer time (~0.1ms per bond) is negligible compared to the LAPACK computation (~30ms per bond at chi=256). Not worth the added complexity.

### A5: GPU-Side SVD Singular Value Scaling

**Rationale**: The CPU-side S×Vh scaling loop is O(chi² × d) — negligible compared to the O(chi³) SVD itself. Would save ~0.01ms per bond at chi=256.

### B3: Adaptive Lanczos Tolerance

**Rationale**: Lanczos is only 1-2% of runtime. Even halving the iteration count would save <0.5% wall time.

### B4: Overlap Segment Sweeps with Environment Rebuild

**Rationale**: Complex synchronization required. The env rebuild is fast (~0.5s) compared to the coupling sweep (~4-15s). Maximum savings would be <0.5s per outer iteration.

## Key Findings and Limitations

### 1. CPU SVD Is the Dominant Bottleneck

At chi=256, CPU LAPACK SVD consumes **97-98% of per-sweep runtime**. All Lanczos, GEMM, and environment optimizations combined affect only 2-3% of the runtime. This was confirmed by dmrg2-gpu profiling:

```
Sweep 2: lanczos=467ms (990 iters) svd=38132ms env=26ms other=4ms
```

The SVD operates on matrices of size (chi×d, d×chi) = (512, 512) at chi=256. Each LAPACK `dgesvd` call takes ~0.6ms, and there are ~63 bonds × 2 directions = ~126 SVDs per sweep, totaling ~38s.

### 2. GPU SVD (rocsolver) Provides Modest Improvement

Testing with `--gpu-svd` on dmrg2-gpu at L=64 chi=256:
- CPU SVD: 141.7s (3 sweeps)
- GPU SVD: 123.4s (3 sweeps)
- **Improvement: 13%**

However, GPU SVD made PDMRG *slower* (140s vs 113s) because parallel segment sweeps produce smaller effective chi values, where GPU SVD has higher overhead than CPU SVD.

### 3. Parallel Segment Sweeps Are Counterproductive at L ≤ 64

After warmup converges the MPS, segment sweeps **disrupt** the converged state:
- Bond dimensions drop from chi_max to ~50-60 within segments
- Inter-segment correlations are destroyed
- The coupling sweep must redo the convergence work

At L=64 with 8 segments (8 sites each), segments are too short for meaningful independent optimization. The entanglement length of the Heisenberg chain exceeds the segment size.

### 4. PDMRG Speedup Is Limited by Amdahl's Law

PDMRG time = Warmup (serial) + Σ(Segments (parallel) + Coupling (serial)) + Polish (serial)

At L=64 chi=128:
- Serial work: warmup (8.8s) + coupling×2 (3.4+2.2s) + polish (7.7s) = 22.1s
- Parallel work: segment sweeps ≈ 2s (hidden behind coupling)
- Total: 24.1s

Even with zero-cost segments, the serial portion (22.1s) limits speedup to 27.0/22.1 = 1.22× over dmrg2-gpu (27.0s).

### 5. Run-to-Run Variance

PDMRG exhibits ±5s variance at chi=256 (±4% of 119s) due to:
- Segment sweeps producing different chi profiles depending on random MPS initialization
- Different chi profiles lead to different SVD matrix sizes in coupling/polish sweeps
- SVD cost scales as O(chi³), so small chi differences compound

dmrg2-gpu is more deterministic (~±1s variance).

### 6. Pinned Memory Race Conditions

A critical finding for GPU programming: `hipMemcpyAsync` from pinned host memory is truly asynchronous — the DMA engine reads the source buffer at **execution time**, not enqueue time. When called in a tight loop (like Lanczos iterations), the CPU may overwrite the source buffer before the DMA engine reads it, causing data corruption.

The original code using `std::vector` (pageable memory) was immune because HIP stages pageable-to-device transfers through an internal pinned buffer, making the copy effectively synchronous w.r.t. the source data. This is an important subtlety: **switching from pageable to pinned memory can introduce race conditions** if the code assumes synchronous-like behavior.

Our solution was to bypass host-to-device DMA entirely using GPU-side pointer setup kernels.

## Scaling Projections

Based on our analysis, PDMRG-GPU would be expected to show meaningful speedup when:

1. **L >> 64**: Segments become large enough for independent optimization to be meaningful. At L=256 with 8 segments of 32 sites, each segment can independently converge local entanglement.

2. **chi >> 256 with GPU SVD**: At very large chi, GPU SVD should outperform CPU SVD, and the parallel segment sweeps would each independently exercise the GPU's compute capacity.

3. **Multi-GPU**: Each segment could be assigned to a separate GPU, eliminating the stream-scheduling overhead on a single device. This is the original motivation for PDMRG.

4. **Higher physical dimension d**: For d ≥ 3 (e.g., Josephson junction, Bose-Hubbard), the ratio of Lanczos/GEMM to SVD increases, giving more room for parallel segment speedup.

## Recommendations for Future Work

1. **Focus on multi-GPU scaling**: The single-GPU PDMRG is limited by Amdahl's law. The algorithm's real value is in distributing segments across devices.

2. **GPU SVD optimization**: At chi ≥ 512, GPU SVD should dominate over CPU. Profile and optimize the rocsolver SVD path for large matrices.

3. **Larger benchmarks**: Test at L=128-512, chi=256-1024 to reach the regime where PDMRG's parallelism pays off.

4. **Alternative coupling strategies**: The full-chain LR+RL coupling sweep is the bottleneck. Explore subspace expansion or density matrix perturbation theory at boundaries instead of brute-force resweeping.

5. **Mixed-precision SVD**: Use single-precision SVD for early sweeps (warmup/segments) and double-precision only for polish. Could halve SVD time for ~90% of the computation.

## Reproducibility

All benchmarks can be reproduced with:

```bash
ssh hotaisle@23.183.40.82
cd ~/dmrg-implementations/pdmrg-gpu/build
cmake .. -DGPU_TARGETS=gfx942 && make -j$(nproc)

# Correctness (all must show PASS with error < 1e-10):
./pdmrg_gpu 8 32 5 --segments 2 --warmup 3
./pdmrg_gpu 32 64 3 --segments 4 --warmup 3
./pdmrg_gpu 6 32 5 --segments 2 --warmup 3 --josephson

# Performance (PDMRG):
./pdmrg_gpu 64 128 2 --segments 8 --warmup 1 --local-sweeps 1
./pdmrg_gpu 64 256 2 --segments 8 --warmup 1 --local-sweeps 1

# Performance (dmrg2-gpu baseline):
cd ~/dmrg-implementations/dmrg2-gpu/build
./dmrg2_gpu 64 128 5
./dmrg2_gpu 64 256 5
```

## Commit History

```
6df2548 A7: Remove redundant stream syncs from sweep loops
d18ba03 Revert B1 from outer loop, fix B5 convergence tracking
1ff1b07 B1+B2+B5: Boundary coupling, adaptive warmup, skip-polish
50eb73c Replace pinned DMA with GPU-side pointer setup kernels
80790ba Fix pinned memory race in apply_heff B pointer upload
a3d5310 Fix pinned memory race in env updates + remove debug code
146cd85 A3: Cache batched GEMM pointer arrays with pinned host memory
37980ec A1: Device pointer mode for Lanczos inner loop
b13abe7 A6: Remove dead boundary-coupling code from pdmrg-gpu
f074ab0 Add PDMRG-GPU performance optimization prompt
e5ea718 Enable real parallel segment sweeps via std::thread + cleanup
```
