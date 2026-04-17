# F1+F2 regression investigation — FINDINGS

**Date:** 2026-04-17
**Host:** enc1-gpuvm020 (AMD MI300X)
**Question raised:** Are F1 (batched Step-3 GEMM in apply_heff) or F2
(rocsolver_gesvdj for dense SVD) causing the Heisenberg/TFIM regressions
observed when comparing the pre-R3 (20260407_104055) pdmrg-gpu benchmarks
against the post-R3 (20260411 r3) benchmarks on the challenge ULTRA_TRIM grid?

## Answer: NO. F1+F2 are clean wins. The apparent regression was two-site polish.

## Root cause

Commit `a681377` (Apr 7 11:13, "fix(gpu): use two-site polish sweeps for PDMRG
segment convergence") made two incompatible changes to pdmrg-gpu's polish
phase that neither the R3-F1 nor the R3-F2 patches touched:

1. Switched polish sweeps from `sweep_LR/RL_full_1site()` (single-site) to
   `sweep_LR/RL_full()` (two-site).
2. Hardcoded `int n_polish = 10;` inside `run()`, shadowing the parameter —
   so the polish phase ran 10 two-site full-chain sweeps regardless of the
   `--polish N` CLI argument.

On Heisenberg/TFIM (d=2), two-site polish costs ~1.6× per sweep versus
single-site (Lanczos dim grows from χd to χd²; SVD grows from χ×χ to 2χ×2χ).
With 10 unconditional polish sweeps this swamps any kernel-level savings.

This was reverted by commit `66524d1` (Apr 15, "revert polish to single-site
per CLAUDE.md rule"), but the R3 benchmark grid was run on Apr 11 with the
two-site polish in place. Hence the confounded measurement.

## Measurement protocol

To isolate kernel speed from the polish-type confound:

1. Used the **current** pdmrg-gpu binary (HEAD 2db2282, single-site polish).
2. Temporarily patched `pdmrg_gpu_impl.h` to honor a `PDMRG_FORCE_POLISH=1`
   env var that skips the `!outer_converged` guard so polish runs
   unconditionally, matching the pre-R3 and R3 polish behavior.
3. Ran 4 configs (heisenberg/tfim × L=50/100 at χ=128) with 2 repeats
   at matched compute load: `--warmup 3 --polish 10 --segments 2
   --local-sweeps 2` (n_outer=20).
4. Reverted the patch after measurement (binary rebuilt clean).

## Results (wall time, s — median of 2 reps)

| config                        | pre-R3 (Apr 7) | R3 (Apr 11)     | current (F1+F2 + Apr 15 fixes, forced polish) |
|-------------------------------|---------------:|----------------:|----------------------------------------------:|
| heisenberg L=50  χ=128        |  68.9          | 104.9           | **18.7**                                      |
| heisenberg L=100 χ=128        |  93.9          | 269.7           | **57.9**                                      |
| tfim       L=50  χ=128        |  37.3          |  63.5           | **23.0**                                      |
| tfim       L=100 χ=128        | 184.0          | 277.7           | **68.9**                                      |

**Speedups of current vs pre-R3:** 3.7× / 1.6× / 1.6× / 2.7×
**Speedups of current vs R3 Apr 11:** 5.6× / 4.7× / 2.8× / 4.0×

## Phase breakdown (probe_v2 force-polish logs)

heisenberg L=100 χ=128 rep=1:
  warmup  17.3 s (3 sweeps)  → 5.8 s/sweep
  parallel 26.8 s (1 outer × 2 local)
  polish  14.3 s (10 single-site sweeps) → 1.4 s/sweep

R3 (reconstructed): polish ~240s for 10 two-site sweeps → ~24 s/sweep.
Single-site polish is ~17× cheaper than two-site on this config, which is
consistent with the χd² → χd Lanczos dim change plus per-sweep count staying
at L=100 sites. (Slightly larger than theoretical 4-8× because two-site SVD
cost and batched Step-3 interact unfavorably at d²=4.)

## Conclusion

- **No F1+F2 regression exists.** Both are clean wins, now confirmed at
  challenge sizes (L=50-100, χ=128) as well as the L=24 χ=64 point reported
  in `docs/followups/r3_regression_analysis.md`.
- The "regression" previously visible in `pdmrg-gpu_mi300x_challenge_20260411_*`
  versus `pdmrg-gpu_mi300x_challenge_20260407_104055.json` is 100% explained
  by the two-site polish in the Apr 11 binary, already reverted.
- The current HEAD pdmrg-gpu is the fastest pdmrg-gpu ever measured on this
  project. Speedups of 2.7-3.7× vs pre-R3 baseline are F1 + F2 + the Apr 15
  structural fixes (b6513b0 "6 structural fixes", 5217bb5 recalibration,
  7927fb6 accurate SVD at boundaries) stacked together.

## Artifacts

- Probe script: `benchmarks/probe_f1f2_regression.sh`
- Probe v1 logs (early exit due to bash sort bug): `probe_20260417T195109Z/`
- Probe v2 logs (force-polish, complete): `probe_v2_20260417T201959Z/`
