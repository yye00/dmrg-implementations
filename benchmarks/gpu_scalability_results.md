# GPU Scalability Benchmark Results

**Date**: 2026-03-13
**Hardware**: AMD Instinct MI300X (gfx942), ROCm 7.2
**Heisenberg OBC S=1/2 chain**

## Accuracy Summary

All configurations achieve error < 1e-10 (PASS threshold):

| Implementation | L=32 chi=64 | L=64 chi=128 |
|---|---|---|
| dmrg-gpu (1-site) | -13.997315618009 (2e-12) | -28.175424859649 |
| dmrg2-gpu (2-site) | -13.997315618006 (1e-13) | -28.175424859648 |
| pdmrg-gpu seg=2 | -13.997315618007 (4e-13) PASS | -28.175424859648 |
| pdmrg-gpu seg=4 | -13.997315618007 (4e-13) PASS | -28.175424859648 |
| pdmrg-gpu seg=8 | -13.997315618006 (5e-13) PASS | -28.175424859648 |
| pdmrg2-gpu seg=2 | -13.997315618006 (5e-13) PASS | -28.175424859648 |
| pdmrg2-gpu seg=4 | -13.997315618007 (5e-13) PASS | -28.175424859648 |
| pdmrg2-gpu seg=8 | (too slow, killed) | (too slow, killed) |

Exact reference (L=32): -13.997315618007

## Wall Time (seconds)

### L=32, chi=64, D_mpo=5

| Implementation | Segments | Wall Time | Notes |
|---|---|---|---|
| dmrg-gpu (1-site) | 1 | **2.0s** | 30 sweeps, Lanczos |
| dmrg2-gpu (2-site) | 1 | **2.1s** | 20 sweeps, Lanczos |
| pdmrg-gpu | 2 | 26.2s | 20 outer, Lanczos |
| pdmrg-gpu | 4 | 23.1s | 20 outer, Lanczos |
| pdmrg-gpu | 8 | 38.1s | 20 outer, Lanczos |
| pdmrg2-gpu | 2 | 88.7s | NS+Davidson, par=35s coup=45s |
| pdmrg2-gpu | 4 | 85.2s | NS+Davidson, par=28s coup=49s |
| pdmrg2-gpu | 8 | >600s | NS thread contention, killed |

### L=64, chi=128, D_mpo=5

| Implementation | Segments | Wall Time | Notes |
|---|---|---|---|
| dmrg-gpu (1-site) | 1 | **14.0s** | 30 sweeps, Lanczos |
| dmrg2-gpu (2-site) | 1 | **27.5s** | 20 sweeps, Lanczos |
| pdmrg-gpu | 2 | 243.0s | 20 outer, Lanczos |
| pdmrg-gpu | 4 | 174.2s | 20 outer, Lanczos |
| pdmrg-gpu | 8 | 125.0s | 20 outer, Lanczos |
| pdmrg2-gpu | 2 | 555.9s | NS+Davidson, par=232s coup=261s |
| pdmrg2-gpu | 4 | 504.4s | NS+Davidson, par=164s coup=276s |
| pdmrg2-gpu | 8 | >600s | NS thread contention, killed |

## Analysis

### 1. Single-stream baselines dominate

dmrg-gpu (single-site) and dmrg2-gpu are **dramatically faster** than any PDMRG variant:
- L=32: 2s vs 23-89s (12-44x slower)
- L=64: 14-28s vs 125-556s (5-40x slower)

The PDMRG overhead comes from:
- Warmup (full-chain DMRG2 sweeps to initialize): ~5s (L=32), ~50s (L=64)
- 20 outer iterations of parallel segments + full-chain coupling
- Polish sweeps to ensure accuracy

### 2. PDMRG architecture problem: coupling phase negates parallelism

The fundamental issue: each outer iteration does:
1. **Parallel segment sweeps** (threads, faster with more segments)
2. **Full-chain coupling sweep** (sequential LR+RL over entire chain)

The coupling phase costs O(L) per iteration — same as a standard DMRG sweep. This makes the total cost ≈ `n_outer × (par_time/P + coupling_time)`. Since coupling_time ≈ coupling_time_baseline, the parallel segments save only a fraction of the total.

For pdmrg-gpu L=64:
- 2 seg: 243s
- 4 seg: 174s (1.4x speedup)
- 8 seg: 125s (1.9x speedup)

The scaling is sub-linear because coupling dominates.

### 3. Newton-Schulz + Block-Davidson adds significant overhead

pdmrg2-gpu is **3-4x slower** than pdmrg-gpu at every segment count:
- L=32 seg=2: 89s vs 26s (3.4x)
- L=64 seg=4: 504s vs 174s (2.9x)

The Newton-Schulz pre-sweep canonicalization and Block-Davidson eigensolver are
more expensive per iteration than QR/Lanczos despite being BLAS-3. At these
bond dimensions (chi=64-128), the individual GEMM operations in Newton-Schulz
iterations (5-8 GEMMs per site) and Block-Davidson (subspace expansion + Ritz
projection) don't saturate the GPU.

### 4. Thread contention kills 8-segment pdmrg2-gpu

With 8 threads each calling synchronous rocBLAS GEMMs (Newton-Schulz iterations
require device synchronization), massive GPU contention occurs. The 8-segment
pdmrg2-gpu runs consume >100 CPU-minutes but produce <1 minute of wall-clock
progress.

### 5. Boundary coupling was insufficient

An attempt to replace full-chain coupling with boundary-only coupling (±4 sites)
caused catastrophic energy oscillation (E = -7 instead of -14 for L=32). The
parallel segment sweeps destroy inter-segment correlations too aggressively for
narrow boundary repairs.

## Conclusions

1. **For single-GPU work, standard single-stream DMRG is optimal.** The MI300X
   has enough compute bandwidth that a single-stream Lanczos+SVD sweep saturates
   the GPU well. Parallelizing across segments adds overhead without speedup.

2. **PDMRG may benefit from multi-GPU setups** where each segment maps to a
   separate GPU, avoiding the coupling bottleneck and GPU contention.

3. **Newton-Schulz + Block-Davidson do not improve performance** at chi=64-128
   on a single GPU. They add ~3x overhead compared to Lanczos+SVD. They may
   become competitive at much larger chi (>512) where BLAS-3 operations saturate
   the GPU better.

4. **pdmrg-gpu (Lanczos) shows mild scaling** with segments (1.4-1.9x for 4-8
   segments at L=64), but still 5-9x slower than single-stream dmrg-gpu.
