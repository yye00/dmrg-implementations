# R3 Benchmark Run — F1 + F2 + Size-Gate Validation

**Date launched:** 2026-04-11
**Goal:** measure the effect of the R3 follow-up optimizations (F1 Step-3 batched
GEMM collapse, F2 `rocsolver_gesvdj`, and the size-gated dispatch fix) against
the `-base` GPU baselines on paper-relevant challenge sizes.

## Grid

Ultra-trim — 18 configs per model, 3 models = 54 configs per impl.

| Model      | (L, χ) pairs                          |
|------------|---------------------------------------|
| heisenberg | (50,64) (50,128) (50,256) (100,64) (100,128) (100,256) |
| josephson  | (16,64) (16,128) (16,256) (32,64) (32,128) (32,256)    |
| tfim       | (50,64) (50,128) (50,256) (100,64) (100,128) (100,256) |

Josephson uses `nmax=2` ⇒ `d=5` (complex).
Repeats: **3**, output stores `times: [t1,t2,t3]` + `time_median` + `time_min`.

## Impls

Optimized (new, with F1+F2+size-gate), then baselines:

| Impl             | Purpose                                                  |
|------------------|----------------------------------------------------------|
| `dmrg-gpu`       | single-site DMRG, optimized                              |
| `dmrg2-gpu`      | two-site DMRG, optimized                                 |
| `pdmrg-gpu`      | parallel DMRG, optimized                                 |
| `dmrg-gpu-base`  | single-site baseline (no F1/F2)                          |
| `dmrg2-gpu-base` | two-site baseline (no F1/F2)                             |

**Skipped:** `pdmrg-gpu-base` — the optimized pdmrg-gpu pass alone already
dominates walltime; a `-base` run would blow the overnight window without
meaningfully changing the paper narrative. The F1/F2 delta for pdmrg-gpu is
already documented in `r3_regression_analysis.md` from the three-point ladder.

## Protocol

1. All jobs run inside the `r3_bench` tmux session on the MI300X host
   (`ssh hotaisle@23.183.40.84`).
2. Per-config warmup on the first repeat only.
3. After each `(impl, model)` block (15 total), wrapper commits + pushes to
   GitHub. A crash loses at most one in-flight config.
4. Old pre-fix result JSONs in `benchmarks/paper_results/mi300x/challenge/`
   from 20260407 are preserved untouched — new files carry `r3_<timestamp>`
   tags.

## Wall-time estimate (pre-run)

Derived from 20260407 full 44-config walltimes, scaled to 18/44 and multiplied
by 3 repeats:

| Impl             | Est. walltime |
|------------------|---------------|
| dmrg-gpu         | ~30 min       |
| dmrg2-gpu        | ~80 min       |
| pdmrg-gpu        | ~70 min       |
| dmrg-gpu-base    | ~85 min       |
| dmrg2-gpu-base   | ~230 min      |
| **Total**        | **~8-9 h**    |

Note: these are rough — F1+F2+size-gate is expected to cut optimized times by
10-20%, and the -base impls may run differently than expected. The wrapper
will report actuals block-by-block.

## Log

## Run started 20260411T115427Z

- Grid: ULTRA_TRIM (18 configs/model)
- Repeats: 3
- Impls: dmrg-gpu dmrg2-gpu pdmrg-gpu dmrg-gpu-base dmrg2-gpu-base
- Models: heisenberg josephson tfim

| impl | model | start | end | walltime | status |
|------|-------|-------|-----|----------|--------|
| dmrg-gpu | heisenberg | 20260411T115427Z | 12:01:08Z | 6m41s | OK |
