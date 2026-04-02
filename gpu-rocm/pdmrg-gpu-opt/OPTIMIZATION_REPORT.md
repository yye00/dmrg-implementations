# PDMRG-GPU-OPT Optimization Report

## Overview

`pdmrg-gpu-opt` implements algorithmic replacements for the core DMRG linear algebra, targeting the MI300X's BLAS-3 (GEMM) throughput. The premise: CPU SVD consumes 97-98% of per-sweep runtime at chi ≥ 128 (`pdmrg-gpu/OPTIMIZATION_REPORT.md`), so replacing SVD and Lanczos with GEMM-heavy alternatives should yield large speedups.

This report documents what was attempted, what worked, and what failed.

## Hardware & Software

- **GPU**: AMD Instinct MI300X (gfx942), 304 CUs, 192 GB HBM3
- **ROCm**: 7.2, rocBLAS, rocsolver
- **Precision**: `double` (float64) and `hipDoubleComplex` (complex128), templated
- **Baseline**: `pdmrg-gpu` with CPU LAPACK SVD + Lanczos eigensolver

## Tier 1 Optimizations (Implemented, Shipped)

### 1. Newton-Schulz Polar Decomposition for Canonicalization

**What**: Replaced QR/SVD-based MPS canonicalization (`left_canonize_site`, `right_canonize_site`) with Newton-Schulz iterative polar decomposition. The iteration X_{k+1} = 0.5 * X_k * (3I - X_k^H X_k) converges to the unitary polar factor, using only GEMM operations.

**Why**: QR factorization is BLAS-2 dominated (Householder reflections). Newton-Schulz is pure BLAS-3 (three GEMMs per iteration, typically 5-10 iterations to converge).

**Result**: ✅ **Correct** — achieves 1e-10 accuracy on all test cases. The canonicalization works for both left (tall/square, A = UP) and right (wide, A = LQ) variants.

**Performance**: Mixed. Newton-Schulz canonicalization adds overhead for small chi because the iteration requires multiple GEMM calls per site, vs a single QR call. At chi ≥ 128, the GEMM calls are large enough to saturate the GPU and the per-iteration cost amortizes well, but the benefit over QR is modest because canonicalization is not the bottleneck (SVD split is).

### 2. Newton-Schulz for Bond Splitting (ns_split)

**What**: Replace SVD-based bond splitting (`svd_split`) with Newton-Schulz polar decomposition followed by eigendecomposition of the positive semidefinite factor P = U^H A. The singular values come from diag(sqrt(eigenvalues of P^H P)), and the right singular vectors from the eigenvectors of P^H P.

**Why**: SVD via LAPACK `dgesvd` is inherently sequential (bulge chase algorithm). Newton-Schulz + eigendecomp is GEMM-heavy and should saturate GPU compute.

**Result**: ✅ **Correct** but ❌ **slower than CPU SVD** for chi ≤ 256. The NS iteration (5-10 GEMMs) + host-side eigendecomp of the (chi × chi) P^H P matrix takes more total wall time than a single LAPACK SVD call. The eigendecomp itself is O(chi³) on CPU, same as SVD.

**Performance data** (Heisenberg L=32 chi=128 seg=2):
- CPU SVD split: 16.5s total
- NS split: ~20-25s total (Newton-Schulz iteration + eigendecomp overhead)

**Lesson**: Newton-Schulz polar decomposition replaces one O(n³) factorization (SVD) with another (polar iteration + eigendecomp), but with higher constant factors due to multiple GEMM passes. The win would require chi ≫ 256 where GPU GEMM throughput dominates over LAPACK's optimized CPU SVD.

### 3. Block-Davidson Eigensolver

**What**: Replace Lanczos with Block-Davidson for the two-site ground state optimization. Davidson maintains a subspace basis {v_1, ..., v_k}, projects H into the subspace, solves the small eigenproblem, and expands with a preconditioned residual.

**Why**: Lanczos is BLAS-1/2 dominated (matvec + dot products + axpy). Block-Davidson can use BLAS-3 operations for subspace projection when the block size > 1.

**Result**: ✅ **Correct** — converges to same energies as Lanczos. However, for DMRG's single ground state search, Davidson's overhead (larger subspace management, QR restarts, projected eigenproblem) exceeds its BLAS-3 benefits. Lanczos typically converges in 15-20 matvecs; Davidson uses comparable matvecs but with more bookkeeping.

**Performance**: Comparable to Lanczos (within ±10%) at all tested sizes. No significant speedup.

### 4. MFMA-16 Dimension Padding

**What**: Pad `chi_max` to the next multiple of 16 for MI300X MFMA FP64 tile alignment. All allocations use the padded dimension; actual bond dimensions remain unpadded.

**Why**: MI300X matrix cores operate on 16×16 FP64 tiles. Non-aligned dimensions cause partial tile utilization.

**Result**: ✅ **Implemented**. Small but consistent improvement (~5-10%) when chi is not already a multiple of 16.

### 5. Strided Batched Step-3 GEMMs

**What**: Replace the loop of D×d² individual GEMM calls in apply_heff Step 3 with `rocblas_gemm_strided_batched`, exploiting the regular stride pattern of the R_env slices.

**Result**: ✅ **Implemented**. Reduces kernel launch overhead for Step 3. Measurable improvement at small chi where launch overhead is significant relative to GEMM compute.

### 6. Per-Segment Worker Stream Pool

**What**: Each PDMRG segment gets a pool of worker HIP streams for dispatching independent GEMM groups concurrently within environment updates.

**Result**: ✅ **Implemented**. Modest improvement by overlapping independent small GEMMs within a single segment's work.

## Tier 2 Optimizations (Implemented, Failed)

### 7. Cross-Segment Batched GEMM Sweep (Item 4)

**What**: Replace the thread-per-segment parallel sweep with a lock-step sweep that batches GEMM calls across segments into single rocBLAS batched GEMM calls. Segments with matching (cL, cR) dimensions are grouped for a single batched dispatch.

**Why**: Expected to reduce kernel launch overhead by combining N segment GEMMs into one batched call.

**Result**: ❌ **Slower at all scales tested**. Comprehensively benchmarked across 19 configurations (L=20-100, chi=50-256, seg=2-16).

| Config | Baseline | Batched | Ratio |
|--------|----------|---------|-------|
| L=20 chi=50 seg=2 | 2.35s | 3.04s | 0.77× |
| L=32 chi=64 seg=2 | 9.70s | 94.30s | 0.10× |
| L=32 chi=128 seg=2 | 16.55s | 231.81s | 0.07× |
| L=32 chi=128 seg=4 | 15.37s | 25.56s | 0.60× |
| L=32 chi=256 seg=2 | 59.91s | 50.66s | **1.18×** |
| L=32 chi=256 seg=4 | 25.86s | 40.25s | 0.64× |
| L=64 chi=128 seg=2 | 72.23s | TIMEOUT | — |
| L=64 chi=128 seg=8 | TIMEOUT | 77.39s | — |
| L=64 chi=128 seg=16 | TIMEOUT | 56.51s | — |

**One exception**: chi=256 seg=2 showed 1.18× speedup — at very large bond dimension with few segments, the GEMM kernel sizes are large enough that batching pointer arrays reduces dispatch overhead meaningfully.

**Why it failed**:
1. Lock-step sweep serializes BLAS-1 operations (dot, nrm2, axpy) in Lanczos onto a single stream, losing the concurrent execution that thread-per-segment achieves
2. Segments often have mismatched bond dimensions, preventing effective batching
3. At high segment counts (seg=8+), the baseline thread-per-segment mode fails to converge within timeout while batched does — this is a *convergence behavior* difference (lock-step vs async), not a performance advantage

**Status**: Implemented, defaulted **OFF**. Available via `--batched-sweep` flag.

### 8. Chebyshev-Filtered Subspace Iteration (Item 5)

**What**: Replace Lanczos with Chebyshev polynomial filtering. Uses a 10-step Lanczos for spectral bounds [λ_min, λ_max], then applies a degree-15 Chebyshev polynomial filter via 3-term recurrence: T_{k+1}(φ(H))x = 2φ(H)T_k(φ(H))x - T_{k-1}(φ(H))x. The polynomial suppresses the unwanted spectrum, concentrating the filtered vector on the ground state. Energy extracted via Rayleigh quotient.

**Why**: Chebyshev filtering requires no orthogonalization during the polynomial application — just repeated apply_heff calls and BLAS-1 (scal, axpy). Expected to be sync-free and GPU-friendly.

**Result**: ✅ **Correct** (1e-13 accuracy) but ❌ **1.9-11× slower than Lanczos**.

| Config | Lanczos | Chebyshev | Ratio |
|--------|---------|-----------|-------|
| L=8 chi=32 seg=2 | 0.73s | 1.40s | 0.52× |
| L=20 chi=50 seg=2 | 5.27s | 59.15s | 0.09× |

**Why it failed**: Fundamental mismatch between algorithm and problem:
- Chebyshev filtering does degree × outer_iterations matvecs (up to 15 × 20 = 300)
- Lanczos converges in ~15 matvecs for a single ground state
- CheFSI is designed for computing *many eigenvalues simultaneously* (as in DFT), not a single ground state where Krylov methods are already optimal

**Status**: Implemented, defaulted **OFF**. Available via `--chebyshev` flag.

## Randomized SVD (rSVD)

**What**: Halko-Martinsson-Tropp randomized SVD. Project theta onto a random subspace (Y = theta @ Omega), QR factorize Y, project B = Q^H @ theta, SVD the small (r × n) matrix B.

**Result**: ✅ **Implemented** and available via `--rsvd`. Faster than full SVD when chi is large relative to the truncation rank, but the oversampling requirement (r = chi_max + p, p ≈ 10-20) means the savings are modest for typical DMRG bond dimensions.

## Summary: What Actually Helped

| Optimization | Category | Result | Impact |
|-------------|----------|--------|--------|
| Newton-Schulz canonicalization | Tier 1 | ✅ Works | Modest, not bottleneck |
| Newton-Schulz bond split | Tier 1 | ✅ Correct, ❌ Slower | Higher constant factors than LAPACK SVD |
| Block-Davidson eigensolver | Tier 1 | ✅ Works | ±10% vs Lanczos (no win) |
| MFMA-16 padding | Tier 1 | ✅ Works | 5-10% improvement |
| Strided batched Step-3 | Tier 1 | ✅ Works | Small launch overhead reduction |
| Worker stream pool | Tier 1 | ✅ Works | Modest overlap benefit |
| Cross-segment batched GEMM | Tier 2 | ❌ Slower | Only helps at chi=256 seg=2 |
| Chebyshev eigensolver | Tier 2 | ❌ Slower | Wrong algorithm for single eigenvalue |
| Randomized SVD | Tier 1 | ✅ Works | Modest for typical chi |

**Net result**: pdmrg-gpu-opt is comparable to pdmrg-gpu at large sizes (L=100 chi=128: 216s vs ~220s) but 1.2-2× slower at small sizes due to Newton-Schulz iteration overhead. The original Lanczos + CPU SVD approach in pdmrg-gpu remains the most efficient for the MI300X at chi ≤ 256.

## Key Lesson

The premise — that replacing BLAS-2 algorithms (Lanczos, Householder QR) with BLAS-3 algorithms (Newton-Schulz, Davidson, Chebyshev) would yield speedups on a GPU — is sound in theory but failed in practice for DMRG because:

1. **CPU SVD is surprisingly efficient** at chi ≤ 256. LAPACK's divide-and-conquer SVD is heavily optimized and difficult to beat with iterative GPU methods that require multiple passes.

2. **Lanczos is already near-optimal** for finding a single ground state. It converges in O(sqrt(condition number)) matvecs, and no polynomial method can do fundamentally better for a single eigenvalue.

3. **The BLAS-2 → BLAS-3 conversion adds overhead**. Newton-Schulz requires 5-10 GEMM iterations per decomposition vs 1 SVD call. The GEMM throughput advantage doesn't compensate until matrix sizes are very large.

4. **DMRG matrices are not large enough**. At chi=128, the matrices are 128×128 to 256×512 — these are "small" by GPU standards. The MI300X's MFMA throughput peaks at matrices ≥ 2048×2048.

The regime where these optimizations would shine is chi ≥ 1024 on multi-GPU systems — beyond our current benchmarking range.

## Benchmark Results

Full results (108 configurations) are in `benchmarks/paper_results/gpu_opt_bench.json`.

### Serial GPU Comparison (Heisenberg OBC)

| L | chi | dmrg-gpu | dmrg2-gpu | dmrg-gpu-opt | dmrg2-gpu-opt |
|---|-----|----------|-----------|--------------|---------------|
| 8 | 32 | 0.31s | 0.34s | 0.68s | 0.78s |
| 32 | 128 | 4.21s | 5.63s | 5.09s | 8.87s |
| 64 | 128 | 24.04s | 22.83s | 26.63s | 30.64s |
| 100 | 128 | 176.80s | 53.13s | 157.13s | 69.32s |

### Parallel GPU (pdmrg-gpu-opt, Heisenberg OBC)

| L | chi | seg=2 | seg=4 |
|---|-----|-------|-------|
| 8 | 32 | 0.31s | — |
| 20 | 50 | 2.54s | — |
| 32 | 128 | 16.46s | — |
| 64 | 128 | 49.38s | 78.11s |
| 100 | 128 | 216.46s | 196.52s |

## Reproducibility

```bash
ssh hotaisle@23.183.40.79
cd ~/dmrg-implementations/pdmrg-gpu-opt/build
cmake .. -DGPU_TARGETS=gfx942 && make -j$(nproc)

# Correctness:
./pdmrg_gpu_opt 8 32 20 --segments 2           # Heisenberg, expect PASS
./pdmrg_gpu_opt 8 32 20 --segments 2 --josephson --nmax 2  # Josephson

# Batched sweep comparison:
./pdmrg_gpu_opt 32 128 20 --segments 2                    # baseline
./pdmrg_gpu_opt 32 128 20 --segments 2 --batched-sweep     # batched

# Chebyshev comparison:
./pdmrg_gpu_opt 8 32 20 --segments 2                    # Lanczos (default)
./pdmrg_gpu_opt 8 32 20 --segments 2 --chebyshev        # Chebyshev
```

## Commit History

```
7197d2e feat(pdmrg-gpu-opt): Item 5 Chebyshev-filtered subspace iteration eigensolver
0359be3 feat(pdmrg-gpu-opt): Item 4 cross-segment batched GEMM sweep
172ec7a revert(dmrg-gpu): restore rocBLAS GEMM contractions, remove hiptensor
ae1c48f revert(dmrg-gpu): remove plan cache (hiptensor plan reuse segfaults)
```
(Earlier commits for Tier 1 optimizations in dmrg-gpu-opt/dmrg2-gpu-opt/pdmrg-gpu-opt)
