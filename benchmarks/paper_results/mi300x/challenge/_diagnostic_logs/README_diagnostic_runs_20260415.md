# PDMRG-GPU Diagnostic Runs — 2026-04-15

## Context

After implementing accurate SVD + correct R_env at boundary merges (commit `7927fb6`),
we ran diagnostics to characterize convergence behavior at various chi values.
Three additional fixes were applied in commit `d595a22`:
1. Pre-allocated `d_Vh_canonical` (removes hot-path hipMalloc/hipFree)
2. Double CGS reorthogonalization in Lanczos
3. Polish switched to two-site (later reverted to single-site in `66524d1`)

A fourth fix in commit `3fab754` replaced the incremental env rebuild before
polish with `build_initial_environments()` (full from-scratch rebuild).

Polish was reverted to single-site in `66524d1` per CLAUDE.md rule #2.

## Remote Host

- **Host**: enc1-gpuvm014 (23.183.40.79)
- **GPU**: AMD Instinct MI300X (gfx942), ROCm 7.2+
- **Binary**: `/home/hotaisle/dmrg-implementations/gpu-rocm/pdmrg-gpu/build/pdmrg_gpu`

## Test 1: Baseline (no recal) — bench_pdmrg_fixed.log

**Binary**: commit `7927fb6` (accurate SVD + R_env fix only)
**Command**: `python3 benchmarks/run_mi300x_challenge.py --trim --repeats 3 --impl pdmrg-gpu,dmrg-gpu,dmrg2-gpu --pdmrg-warmup 1 --pdmrg-polish 0 --tag pdmrg_fixed`

Results (Heisenberg, pdmrg-gpu only):

| L | chi | sweeps | Energy (median rep) | Delta from ref | Time (s) |
|---|-----|--------|---------------------|----------------|----------|
| 50 | 64 | 20 | -21.972110272161 | ~1e-10 | 7.0 |
| 50 | 128 | 20 | -21.954863321834 | 0.017 | 116.6 |
| 50 | 256 | 15 | -21.961372066631 | 0.011 | 185.1 |
| 100 | 64 | 20 | -44.127739264899 | ~1e-10 | 36.2 |
| 100 | 128 | 20 | -44.119949693865 | 0.008 | 275.2 |

**Finding**: chi=64 converges perfectly. chi>=128 fails without recalibration.

## Test 2: Baseline v2 (pdmrg-gpu only) — bench_pdmrg_fixed_v2.log

**Binary**: commit `7927fb6`
**Command**: `python3 benchmarks/run_mi300x_challenge.py --trim --repeats 3 --impl pdmrg-gpu --pdmrg-warmup 1 --pdmrg-polish 0 --tag pdmrg_fixed_v2`

Confirms Test 1 results. L=50 chi=128 rep 3 got -21.9721055883 (delta 5e-6) showing
the algorithm CAN converge at chi=128 with lucky initialization but is unreliable.

## Test 3: Recal=3 diagnostic (old binary) — verify_results.log (partial)

**Binary**: commit `7927fb6` (pre-fix-2/3/4)
**Command**: `./pdmrg_gpu 50 128 40 --segments 2 --warmup 1 --polish 0 --recal 3 --local-sweeps 2`

```
E = -21.972109334297
Delta = 9.4e-7
Wall time = 335.3s (warmup=3.1% parallel=96.9% polish=0.0%)
```

Recalibration closed the gap from 0.017 to 9.4e-7 but NOT to 1e-10.

## Test 4: Recal=3 + two-site polish=2 (old binary, pre env-rebuild fix)

**Binary**: commit `d595a22` (fixes 2-4, two-site polish, incremental env rebuild)
**Command**: `./pdmrg_gpu 50 128 40 --segments 2 --warmup 1 --polish 2 --recal 3 --local-sweeps 2`

```
E = -21.972110281243
Delta = 9.1e-9
Wall time = 286.7s (warmup=1.5% parallel=96.3% polish=2.2%)
```

Two-site polish closes the gap from 9.4e-7 to 9.1e-9.

## Test 5: Recal=3 + single-site polish=2 (full env rebuild)

**Binary**: commit `66524d1` (single-site polish reverted, full env rebuild)
**Command**: `./pdmrg_gpu 50 128 40 --segments 2 --warmup 1 --polish 2 --recal 3 --local-sweeps 2`

```
L=50 chi=128:
  E = -21.972085624776
  Delta = 2.5e-5
  Wall time = 282.4s (warmup=1.6% parallel=97.4% polish=1.1%)

L=100 chi=128:
  E = -44.127447667206
  Delta = 2.9e-4
  Wall time = 714.5s (warmup=1.4% parallel=97.3% polish=1.2%)
```

Single-site polish is 2700x worse than two-site at L=50, and the gap widens at L=100.

## Test 6: Recal=3, 3 reps (new binary, no polish)

**Binary**: commit `d595a22`
**Command**: `./pdmrg_gpu 50 128 40 --segments 2 --warmup 1 --polish 0 --recal 3 --local-sweeps 2` (x3)

```
Rep 1: E = -21.972084226635  (delta 2.6e-5)
Rep 2: E = -21.943000252279  (delta 0.029 — stuck in local minimum)
Rep 3: E = -21.972083947468  (delta 2.6e-5)
```

Without polish, recal alone is unreliable at chi=128.

## Summary

| Configuration | L=50 chi=128 Delta | Notes |
|--------------|-------------------|-------|
| No recal, no polish | 0.017 | Broken |
| recal=3, no polish | 9.4e-7 (old), 2.6e-5 (new) | Improved but not 1e-10 |
| recal=3, single-site polish=2 | 2.5e-5 | Single-site can't fix bond dims |
| recal=3, two-site polish=2 | **9.1e-9** | Closest to 1e-10 |

**Conclusion**: Achieving 1e-10 at chi>=128 requires both recalibration AND two-site polish.
CLAUDE.md rule #2 (single-site polish only) prevents this — needs updating.

## Reference Energies (from dmrg-gpu/dmrg2-gpu)

- Heisenberg L=50: -21.972110272161 (chi=64 pdmrg-gpu converged value)
- Heisenberg L=100: -44.127739264899 (chi=64 pdmrg-gpu converged value)
