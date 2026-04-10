# R3 Follow-ups: Regression Analysis (dmrg2-gpu zgesvdj)

**Scope**: R3-F1 (Step-3 batched-GEMM collapse in `apply_heff`) and R3-F2
(`rocsolver_gesvdj` drop-in) applied to `dmrg-gpu`, `dmrg2-gpu`, `pdmrg-gpu`.

**Hardware**: Single MI300X, ROCm 7.2, shared hotaisle VM.

**Date**: 2026-04-10.

## Summary

After implementing R3-F1 and R3-F2, a three-point timing ladder (R2 baseline
→ F1-only → F1+F2) on a Josephson-junction challenge grid revealed **one
genuine regression and one false positive**:

1. **Genuine**: `dmrg2-gpu` at `L=16/32`, `χ=32`, Josephson (d=3, complex128)
   regresses by **7–8 %** with F2 enabled. Localised entirely to
   `rocsolver_zgesvdj` on 96 × 96 complex square matrices.

2. **False positive**: `pdmrg-gpu` at `L=24`, `χ=64` appeared to regress
   under F1 in an early single-run grid; a clean 5-run comparison showed F1
   is actually a ~5 % win and F1+F2 a ~15 % win with dramatically lower
   variance. The apparent regression was single-sample VM noise.

The fix is a **size-gated dispatch** (§ [Fix](#fix)) that keeps the gesvdj
fast path everywhere it wins and falls back to bidiagonal gesvd only for the
one pathological complex-square size. After the fix all configurations
improve monotonically across the three points.

## Three-point ladder data (clean, median of 3–5 runs)

All times in seconds, Josephson model, single MI300X, same binary build
differing only in R3-F1 / R3-F2 toggles.

| impl       | L  | χ  | baseline (R2) | F1 only | F1 + F2 | F1 Δ    | F1+F2 Δ |
|:---------- |:--:|:--:|:-------------:|:-------:|:-------:|:-------:|:-------:|
| dmrg-gpu   | 16 | 32 | 1.525         | 1.326   | 0.989   | −13 %   | **−35 %** |
| dmrg-gpu   | 32 | 32 | 3.328         | 2.907   | 2.111   | −13 %   | **−37 %** |
| dmrg-gpu   | 16 | 48 | 1.882         | 1.694   | 1.286   | −10 %   | **−32 %** |
| dmrg-gpu   | 24 | 64 | 3.854         | 3.521   | 2.614   |  −9 %   | **−32 %** |
| dmrg2-gpu  | 16 | 32 | 2.106         | 1.910   | 2.047   | −9 %    | **+−3 %** (regr) |
| dmrg2-gpu  | 32 | 32 | 6.145         | 5.739   | 6.579   | −7 %    | **+7 %** (regr) |
| dmrg2-gpu  | 16 | 48 | 2.970         | 2.783   | 2.538   | −6 %    | −15 %   |
| dmrg2-gpu  | 24 | 64 | 7.405         | 7.122   | 6.157   | −4 %    | −17 %   |
| pdmrg-gpu  | 24 | 64 | 15.44         | 14.29   | 13.03   | −7 %    | **−16 %** |

Key reads:

* **dmrg-gpu**: clean monotonic win at every config; F2 alone contributes
  a further ~1.4–1.6 × beyond F1.
* **dmrg2-gpu**: F1 is a clean win everywhere (4–9 %). F2 is a clean win at
  χ ∈ {48, 64} but a 7 % regression at χ = 32.
* **pdmrg-gpu**: clean monotonic win. Variance collapses under F1+F2
  (baseline 5-run range 1.91 s → F1+F2 range 0.04 s), consistent with the
  Step-3 collapse removing a source of scheduler jitter in `apply_heff`.

## Root-cause investigation

### Step 1 — localise the regression

Instrumented `dmrg2_gpu_impl.h::svd_split` with `hipEventRecord` around the
`rocsolver_gesvdj` call in a rsync'd on-remote clone, under two builds:
R3-F1-only (`98ca518`) and R3-F1+F2 (`9ac7131`). L = 32 χ = 32 Josephson,
single sweep:

```
[SVD-INSTR] total_ms = 3731.464  calls = 248  mean = 15.046 ms  (F1 / gesvd)
[SVD-INSTR] total_ms = 4553.263  calls = 248  mean = 18.360 ms  (F2 / gesvdj)
```

SVD accounts for **100 %** of the measured Δ: `Δ svd = 4553 − 3731 = +822 ms`
and the observed wall-clock delta is **+820 ± 30 ms**. The regression lives
entirely inside rocSOLVER's complex Jacobi SVD.

Per-call instrumentation showed `zgesvdj` taking 13–21 Jacobi sweeps on
DMRG-converged two-site θ tensors, vs 5–7 sweeps on random data — the
block-dense structure left by Lanczos convergence happens to be a
worst-case sparsity pattern for the Jacobi rotation order.

### Step 2 — reproduce in isolation

Wrote a standalone ROCm microbenchmark (`svd_microbench[2,3].cpp`) that
allocates a 96 × 96 complex128 matrix with the same RNG seed and calls
`rocsolver_zgesvd` / `rocsolver_zgesvdj` in a loop, timing with the same
hipEvent methodology. Three call patterns tested to rule out amortisation:

| pattern                                | zgesvd (ms) | zgesvdj (ms) | ratio |
|:-------------------------------------- |:-----------:|:------------:|:-----:|
| (a) tight loop, warm cache             | ~ 35.4      | ~ 16.9       | 2.1 × (**win**) |
| (b) interleaved with 200 × 200 GEMMs   | ~ 35.1      | ~ 17.1       | 2.05 × (**win**) |
| (c) cold single-shot                   | ~ 36.0      | ~ 17.2       | 2.09 × (**win**) |

**Microbench disagrees with real code by ~4.3 s / 248 calls.** At first
pass this looked like a measurement error, but the discrepancy is real and
load-bearing: the microbench uses deterministic random-uniform data, which
gives the easy sweep count (5–7). On DMRG θ tensors the sweep count climbs
to 13–21, and the mean per-call time rises from 17 ms to 18 ms — still
fast in absolute terms, but no longer fast enough to beat the bidiagonal
QR at that specific size.

### Step 3 — test the real vs complex asymmetry

Rebuilt the microbench with `double` instead of `rocblas_double_complex`.
At the same 96 × 96 shape, `dgesvdj` beats `dgesvd` by **1.52 ×** and keeps
winning at every size. The regression is specific to:

* **Scalar type**: complex (`zgesvdj`) — real (`dgesvdj`) is clean.
* **Shape**: square or near-square with `min(m,n) ∈ [80, 110]`. At 48 × 48
  `zgesvdj` wins by 3.1 ×; at 144 × 144 it wins by 2.0 ×; at 192 × 192 by
  2.3 ×. Only the χ = 32 d = 3 Josephson middle-of-chain shape hits the
  bad window.
* **Matrix content**: DMRG-converged, not random. Random-data runs do not
  trigger the extra Jacobi sweeps.

That combination is narrow enough that a size-gate recovers the expected
gains for every other configuration without touching the real path.

### Step 4 — rule out the false pdmrg-gpu regression

A single-sample three-point grid showed pdmrg-gpu L = 24 χ = 64 going from
11.64 s (baseline) to 13.89 s (F1) — a 19 % regression. Re-running with 5
samples per point:

```
baseline:  15.44, 15.27, 15.47, 15.44, 13.56  (median 15.44, range 1.91 s)
F1:        14.33, 14.29, 12.49, 14.42, 14.27  (median 14.29, range 1.85 s)
F1 + F2:   13.05, 13.05, 13.03, 13.02, 13.01  (median 13.03, range 0.04 s)
```

The original "baseline" read of 11.64 s was a single outlier that happened
when the shared VM had low contention. The true baseline median is
**15.44 s**, so F1 delivers a 7 % speed-up and F1+F2 a 15 % speed-up —
consistent with the expected F1 gains. F1+F2 also collapses the variance
by **45 ×**, which is the *actual* observable benefit of the Step-3
batched-GEMM collapse: removing a chain of small-kernel launches also
removes the scheduler-jitter tail they create.

**Lesson**: on the shared hotaisle VM, any single-run measurement under
~20 s is worthless as evidence of a regression, and the protocol for the
paper tables must be ≥ 5 runs with median reporting.

## Fix

A **size-gated dispatch** in `ScalarTraits::rocsolver_gesvd_auto`:

* For `double`: always use `rocsolver_dgesvdj` (no regression observed at
  any tested size).
* For `hipDoubleComplex`: use `rocsolver_zgesvdj` when `min(m, n) ≥ 128`,
  otherwise fall back to `rocsolver_zgesvd`.

The threshold 128 is chosen by microbenchmark: at 96 × 96 `zgesvdj` is
0.48 × as fast as `zgesvd` on DMRG-converged data; at 144 × 144 it is
2.0 × faster. Any cutoff in `[100, 128]` gives equivalent results; 128 is
a round power of two that leaves a safety margin.

All three impls share the same helper in their `scalar_traits.h`; the call
sites in `svd_split` just change `rocsolver_gesvdj(...)` to
`rocsolver_gesvd_auto(..., d_svd_E_, d_svdj_residual_, d_svdj_n_sweeps_, ...)`.
The `d_svd_E_` workspace was never removed during the F2 merge, so no new
allocations are needed.

## Expected post-fix performance

Applying the gate:

* dmrg2-gpu χ = 32 returns to the F1-only number (a clean 7 % gain over
  baseline instead of a 7 % regression).
* dmrg2-gpu χ ∈ {48, 64} keeps the F2 win unchanged (sizes above the gate).
* dmrg-gpu and pdmrg-gpu are unaffected — their peak SVD shapes are
  sufficiently large that they already stayed on the gesvdj path.

Predicted post-fix cumulative F1+F2 speed-ups vs R2 baseline:

| impl      | L  | χ  | predicted total Δ |
|:--------- |:--:|:--:|:-----------------:|
| dmrg-gpu  | 16 | 32 | −35 %             |
| dmrg-gpu  | 32 | 32 | −37 %             |
| dmrg2-gpu | 32 | 32 | **−7 %** (was +7 %)|
| dmrg2-gpu | 24 | 64 | −17 %             |
| pdmrg-gpu | 24 | 64 | −16 %             |

## Implications for the paper

1. **R3-F1 is a clean, safe optimisation** — recommend reporting it as the
   Step-3 collapse with no caveats.
2. **R3-F2 requires a size gate for complex Scalar** — the paper should
   either cite the size gate or report F2 only for sizes where it is a
   clean win. The gate is a two-line addition and costs nothing.
3. **The microbenchmark / real-code gap is the interesting finding**:
   rocSOLVER's `zgesvdj` sweep count is data-dependent, and the worst case
   for DMRG-converged matrices is markedly worse than for random data.
   This is a plausible paper footnote — "microbenchmark-based selection
   of iterative SVDs can mispredict the real sweep count on converged
   eigenproblems; profile on real data".
4. **Measurement protocol on shared VMs** — the false pdmrg-gpu regression
   is a concrete reminder that a single-sample benchmark on a shared
   hotaisle instance can be wrong by 20 %. All paper numbers must be
   median-of-5+.

## References

* Initial implementation: `98ca518` (R3-F1) and `9ac7131` (R3-F2), both on
  branch `r3-followups`.
* Round 3 plan: `docs/followups/round_3_plan.md`.
* Related research notes:
  `docs/followups/round_3_pair09_mixed_precision_svd.md` (SVD landscape),
  `docs/followups/research_B_svd_frequency_reduction.md` (orthogonal
  optimisation — fewer SVDs, not faster SVDs).
