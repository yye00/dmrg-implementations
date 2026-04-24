# Superseded / quarantined ablation JSONs

This directory holds ablation benchmark JSONs that are known to be corrupted
or otherwise unreliable.  Files here MUST NOT be used to populate paper tables.

## Contents

| Directory | Variant | Reason | Replacement |
|---|---|---|---|
| `20260420T181425Z_pdmrg-gpu_CRASH48of96/` | pdmrg-gpu | 48/96 SIGABRT crashes (rc=-6) contaminated median_wall_s | `../20260421T190910Z/pdmrg-gpu/results.json` |
| `20260421T004212Z_dmrg-gpu-opt_CRASH48of96/` | dmrg-gpu-opt | 48/96 SIGABRT crashes; **sole Table 6 dmrg-gpu-opt source; never re-run** | _pending clean rerun_ |

## Common root cause

Both files were produced from `git_commit c26f0bd` but with **different
binary sha256 values** (`0faa038…` and `98e1ab0…`), demonstrating the
binary-drift problem: ad-hoc rebuilds with mutated CMake flags during the
same benchmark session.

The original harness did not filter out failed reps (`returncode != 0`)
before computing `median_wall_s`, so crash-produced partial wall times
polluted the medians.

## Fix

`benchmarks/bench_dmrg_gpu_ablate.py` now (cluster H patch, 2026-04-24):
- Filters `walls[]` to `returncode==0 AND energy != None`.
- Emits `n_reps_attempted`, `n_reps_valid`, `data_quality` per group.
- Exits non-zero if any group has `n_reps_valid < ceil(reps/2)`.
- Accepts `--manifest` to enforce a pinned binary sha256 before any run.
