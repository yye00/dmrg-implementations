# Cluster F: 24 MI300X-hours rebench at N=10 + Wilson/bootstrap CIs + statistical_summary.py

Ground truth: docs/PATH_B_GROUND_TRUTH.md (locked at commit 6f45533)
Planner output SHA (source): ada0f3c4d68286d36.output
Date planned: 2026-04-23

---

Now I have enough to write a comprehensive plan. The aggregator computes median over ALL reps including failures (median_wall takes only successful ones, but reps_data still gets the failed entries — and crucially, the median is taken over `walls = [r["wall_s"] for r in reps_data if r["wall_s"] is not None]`, but rc=-6 may still produce a wall_s value via the regex match on partial output, which is the documented bug). Let me confirm and then write the plan.

---

# DEFECT CLUSTER F — Statistical Methodology Fix Plan

## 1. Per-cell N decision

Use Wald formula for required reps given measured per-variant CV and target relative CI half-width:

`N >= (1.96 * CV / target_relative_CI)^2`

Target: ±5% relative half-width on the per-cell median (matches a defensible 1.10× speedup claim). Measured CVs from current data:

| Variant | Worst-case CV | N for ±5% | N for ±10% | Recommended |
|---|---|---|---|---|
| pdmrg-gpu (LANCZOS_GRAPH) | <1% | 1 | 1 | **3** (floor) |
| pdmrg-gpu (other flags) | ~3% | 2 | 1 | **3** |
| pdmrg-gpu-opt | unknown, assume 5% | 4 | 1 | **5** |
| dmrg-gpu | ~10% (SHA-drift seen) | 16 | 4 | **10** |
| dmrg-gpu-opt | unknown (corrupted JSON) | — | — | **10** |
| dmrg2-gpu (RSVD) | 17% (bimodal) | 45 | 12 | **20** + bimodality probe |
| dmrg2-gpu (only_RSVD) | 33% | 170 | 43 | **30** + diagnose |
| dmrg2-gpu-opt | unknown, assume 10% | 16 | 4 | **10** |

For the bimodal RSVD cells: 20 reps + a Hartigan dip-test before reporting a single median; if bimodal at p<0.05, report both modes separately and investigate (likely auto-tune crossover in `rocsolver_gesvd_auto`).

Total cells = 6 variants × 2 problems × ~8 flag-configs = 96 cells. Weighted by per-variant N: **~1,150 binary invocations** for the full re-bench.

## 2. CI methodology spec

Add to a new `benchmarks/lib/stats.py`:

```python
import numpy as np
from scipy import stats

def wilson_ci(k, n, alpha=0.05):
    """Wilson score interval for proportion k/n. Returns (lo, hi)."""
    if n == 0: return (0.0, 1.0)
    z = stats.norm.ppf(1 - alpha/2)
    p = k / n
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return (centre - half, centre + half)

def paired_bootstrap_speedup(t_base, t_opt, n_resamples=10_000, seed=0):
    """Paired bootstrap over reps. Returns (median_speedup, lo, hi)."""
    rng = np.random.default_rng(seed)
    t_base = np.asarray(t_base); t_opt = np.asarray(t_opt)
    n = min(len(t_base), len(t_opt))
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_ratios = np.median(t_base[idx], axis=1) / np.median(t_opt[idx], axis=1)
    return (float(np.median(boot_ratios)),
            float(np.percentile(boot_ratios, 2.5)),
            float(np.percentile(boot_ratios, 97.5)))

def per_variant_win_threshold(cv, k=2.0):
    """Win threshold = 1 + k*CV. k=2 -> ~95% one-sided."""
    return 1.0 + k * cv
```

Replace the §5.5 fixed 1.05× threshold with per-variant noise-floor thresholds computed from baseline reps. For dmrg2-gpu RSVD that becomes **1.34×**, not 1.05×. For pdmrg-gpu LANCZOS_GRAPH it stays effectively 1.02×.

## 3. Crash filter spec

Two scrubs needed.

**(a) Aggregator at write time** — `bench_dmrg_gpu_ablate.py:217-221`. The current code stores `reps_data.append(res)` even when `returncode != 0`, and later filters only on `wall_s is not None`. But `wall_s` comes from regex-matching `Total wall time:` in stdout, which an aborted binary CAN print before SIGABRT. Fix:

```python
# bench_dmrg_gpu_ablate.py around line 219-221
if res["returncode"] != 0 or res["energy"] is None:
    print(f"    FAILED:\n{res['stderr_tail']}")
    res["wall_s"] = None              # <-- ADD: invalidate timing
    res["valid"] = False
else:
    res["valid"] = True
reps_data.append(res)
walls = [r["wall_s"] for r in reps_data
         if r.get("valid") and r["wall_s"] is not None]
```

**(b) Analysis-time filter** — every consumer of `results.json` must reject `rep` entries where `returncode != 0 OR energy is None OR wall_s is None`. Locations to patch:
- `benchmarks/run_mi300x_challenge.py:382-422` (already filters on `r["energy"]` truthy but doesn't check returncode)
- any plotting under `benchmarks/lib/` and `analysis/` (to be created)
- the new `analysis/statistical_summary.py` (item 6)

**(c) Quarantine the two known-bad JSONs:**
- `data/gpu_ablation/20260420T181425Z/pdmrg-gpu/results.json` → move to `data/gpu_ablation/_quarantine/`
- `data/gpu_ablation/20260421T004212Z/dmrg-gpu-opt/results.json` → quarantine + mandatory re-run

## 4. Re-bench matrix (MI300X hours at recommended N)

Wall-time per cell (sec) estimated from existing JSONs; rounded up; includes 10% overhead for warmup/teardown:

| Variant | Cells | N | Mean cell s | Hours |
|---|---|---|---|---|
| pdmrg-gpu | 16 | 3 | 60 | 0.8 |
| pdmrg-gpu-opt | 16 | 5 | 50 | 1.1 |
| dmrg-gpu | 16 | 10 | 20 | 0.9 |
| dmrg-gpu-opt | 16 | 10 | 25 | 1.1 |
| dmrg2-gpu | 16 | 20 | 100 | 8.9 |
| dmrg2-gpu-opt | 16 | 10 | 80 | 3.6 |
| **Total** | **96** | | | **~17 GPU-hours** |

Plus one full repeat after fixing crash filter and bimodality investigation: budget **24 MI300X-hours**. Single overnight on the persistent `test_remote` session.

## 5. Binary determinism / commit pinning protocol

Create `benchmarks/run_paper_rebench.sh` that enforces:

1. `git tag paper-rebench-$(date +%Y%m%d)` and `git rev-parse HEAD` recorded once at start.
2. `cd gpu-rocm/<variant> && make clean && make -j` for ALL six variants from that single commit; record `sha256sum` of every binary into `binaries.sha256`.
3. `rocm-smi --setperflevel high && rocm-smi --setsclk 7 && rocm-smi --setfan 100` (clock pin); snapshot `rocm-smi --showall > rocm_pre.json`.
4. Run all variants serially (no concurrent GPU jobs).
5. Snapshot `rocm-smi --showall > rocm_post.json`; abort report if SCLK delta > 5%.
6. Patch `bench_dmrg_gpu_ablate.py::run_one()` to always write `returncode` (already does, line 158 — just confirm consumers honor it) and add `binary_sha256`, `commit_sha`, `rocm_pre_sclk`, `rocm_post_sclk` into the per-rep record, not just the top-level payload.
7. After completion, `git tag paper-rebench-final` and amend the paper §D.A. data manifest to name THIS commit only.

## 6. `analysis/statistical_summary.py` spec

New file. Inputs: glob over `data/gpu_ablation/*/<variant>/results.json` (post-quarantine). Outputs: one CSV + one Markdown table per variant.

```
analysis/statistical_summary.py \
  --ablation-root data/gpu_ablation \
  --commit-pin <sha> \
  --out reports/mi300x/stats_<sha>/
```

Per-cell columns: `variant, problem, config, n_valid, n_failed, median_s, mean_s, cv, ci95_lo, ci95_hi, speedup_vs_baseline, speedup_ci_lo, speedup_ci_hi, win_threshold, is_win`.

Per-variant aggregate: win-rate `k/n` with Wilson CI; bimodality dip-test p-value for RSVD cells; flag-presence summary (drops cells whose flag is unwired per ground truth §"GpuOpts ablation flag wiring").

Refuses to run if any input JSON has `commit_sha` differing from `--commit-pin`.

## 7. Concrete text-change plan

| Paper location | Current claim | Restated claim |
|---|---|---|
| Abstract bullet "1.05× win threshold" | drop it | "per-variant noise-floor-calibrated threshold (Tab. F1)" |
| Highlights "100% of N=6 wins" | unconditional | "100% wins, Wilson 95% CI [61%, 100%]; N=6 underpowered, see Tab. F2" |
| §5.5 1.05× threshold paragraph | rewrite | enumerate per-variant thresholds: pdmrg-gpu 1.02×, dmrg2-gpu 1.34×, etc. |
| Table 1 (headline speedups) | point estimates | add bracketed 95% paired-bootstrap CI per cell |
| Table 2 (CPU vs GPU) | best-of-threads CPU | re-run at threads=1 only (matches code reality, ground truth §CPU); footnote that 1/2/4/8/12 sweep was never executed |
| Table 3 (ablation summary) | median ratios | median + CI; flag dmrg2-gpu RSVD as bimodal |
| Table 6 (per-flag) | corrupted dmrg-gpu-opt cells | rerun from quarantine; mark cells ineligible until N=10 |
| Table 7 | as-is | add CI columns; redact flags noted unwired in §"GpuOpts ablation" |
| Conclusion "consistent X× speedup" | drop "consistent" | "median X× (95% CI [a, b]) on Y of Z configs" |

Per-claim restatement template:

> **Old:** "Variant V achieves K× speedup over baseline."
> **New:** "Variant V achieves a median K× speedup (paired bootstrap 95% CI [Klo, Khi], N=n reps, n_failed=f) over baseline at problem P, commit SHA pinned to <sha>. Threshold for declaring a win was set to 1+2·CV_baseline = T×; cells where the bootstrap CI lower bound exceeds T are reported as wins."

## 8. Effort + dependencies

**Effort:** 0.5 day to land items 2/3/5/6 in code. ~1 night (24h) to re-bench. 1 day to regenerate paper tables and rewrite text per item 7. **Total: 3 calendar days.**

**Hard prereq for:**
- **Cluster D (data integrity / provenance)** — needs the commit-pin protocol and quarantine before any data can be re-published.
- **Cluster A (algorithm-claim audit)** — every speedup claim must carry a CI before being reattributed to its actual algorithm.
- **Cluster J (paper Tables 1-13 backing data)** — Tables 8/9/11/12/13 lacking JSONs cannot be rescued without item 5's rebench; Table 5's 53.13 vs 53.57 discrepancy resolves automatically once a single pinned commit produces all numbers.

**Blocked by:** nothing in F itself. F is the foundation — land it first, then unblock D, A, J in that order.
