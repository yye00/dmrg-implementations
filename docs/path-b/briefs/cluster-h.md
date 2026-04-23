# Cluster H: quarantine 2 corrupted JSONs, single-tag campaign, ~40 GPU-hours unified rebench

Ground truth: docs/PATH_B_GROUND_TRUTH.md (locked at commit 6f45533)
Planner output SHA (source): a0838f9b6d2837710.output
Date planned: 2026-04-23

---

Ground truth confirmed. Now I have everything needed for the response.

---

# DEFECT CLUSTER H — Disposition

## 1. Code fix for analyzer (`bench_dmrg_gpu_ablate.py`)

**Bug**: Line 220, `walls = [r["wall_s"] for r in reps_data if r["wall_s"] is not None]`. SIGABRT crashes (rc=-6) populate `wall_s` from the `Total wall time:` line printed before abort, or from `wall_wallclock_s`, so they pass the `is not None` filter and pollute the median.

**Spec the change** (one-liner, line 220):

```python
walls = [r["wall_s"] for r in reps_data
         if r.get("returncode") == 0
         and r.get("energy") is not None
         and r.get("wall_s") is not None]
```

Add a parallel hardening at line 217 — already prints "FAILED" but should also bump a `n_failed` counter and emit it into the per-group dict:

```python
results.append({
    ...,
    "n_reps_attempted": len(reps_data),
    "n_reps_valid": len(walls),
    "median_wall_s": median_wall,
})
```

So readers can spot crash-contaminated cells. Also add an **assertion** at end of `run_bench`: if any group has `n_reps_valid < ceil(reps/2)`, write `payload["data_quality"]="DEGRADED"` and exit non-zero.

**Downstream**: Any analysis script that reads `median_wall_s` (table-builders for paper Tables 6, 13) needs the same filter applied retroactively, or — preferably — needs to refuse to plot any group with `data_quality=="DEGRADED"`.

## 2. Corrupted JSON disposition

Both files confirmed: 48/96 reps have `returncode != 0`, **the JSONs do not set `partial=true`**, and median is contaminated.

| File | Action | Rationale |
|---|---|---|
| `data/gpu_ablation/20260420T181425Z/pdmrg-gpu/results.json` | **Move** to `data/gpu_ablation/superseded/20260420T181425Z_pdmrg-gpu_CRASH48of96/results.json` and add a `README.md` in that subdir | Superseded by `20260421T190910Z`; keep for forensics. |
| `data/gpu_ablation/20260421T004212Z/dmrg-gpu-opt/results.json` | **Move** to `superseded/` with the same naming, add `README.md` calling out it backs Table 6 | Never re-run; needed as evidence of provenance failure during paper rebuttal. |

**Commit the move** (do not delete). The superseded subdir should be added to a `data/gpu_ablation/superseded/README.md` that lists each file's reason, the binary SHA, the `git_commit` it was built from, and the count of crashed reps. Both crashed JSONs share commit `c26f0bd` but different binary SHAs (`0faa038344ae` vs `98e1ab093905`), which itself is evidence of the binary-drift problem (item 5).

## 3. dmrg-gpu-opt re-run

**Required** — the only dmrg-gpu-opt ablation JSON is the corrupted `20260421T004212Z`. Action:

- **First, wire the flags**: per ground-truth I-1, dmrg-gpu-opt declares `opts_.device_k` and `opts_.rsvd` but never reads them, and `lanczos_graph` is force-disabled (Block-Davidson incompat). Until those flags are wired, the ablation columns `only_DEVICE_K`, `only_RSVD`, `only_LANCZOS_GRAPH`, `no_DEVICE_K`, `no_RSVD`, `no_LANCZOS_GRAPH` are mathematically identical to baseline. Two paths:
  1. **Wire device_k + rsvd into the dmrg-gpu-opt apply_heff / SVD call sites** (preferred). Lanczos graph stays N/A because it is incompatible with Block-Davidson — explicitly mark that column "n/a (Block-Davidson)".
  2. If wiring is out of scope for the resubmit, **collapse the table**: drop the three columns and add a footnote.
- **Re-run config**: same `BENCH_CONFIGS` (josephson L32 chi128 sweeps=20; josephson L32 chi256 sweeps=15), `--reps 10` (bundles defect-cluster F, statistics, see item 8), `--model josephson`. 14 ablation labels × 2 problems × 10 reps × 2 (correctness gate adds ~6 small runs) ≈ 286 invocations.
- **Time estimate** on MI300X: dmrg-gpu-opt baseline ~16 s × 280 ≈ 78 min for the chi=128 row; the chi=256 row is SVD-bound and roughly 4× → ~5 hours; plus correctness gate (~5 min) and profile pass (~3 min). **Budget: 6 hours wall on MI300X for dmrg-gpu-opt alone.**

## 4. Empty regate file + diagnostic-log duplicates

- `data/gpu_ablation_regate/20260422T145338Z/pdmrg-gpu/results.json` — confirmed `partial=true`, `reps_ok=0`, `reps_crash=0`, no provenance. **Delete and commit the deletion.** No information to preserve. If the campaign is re-launched, a new timestamp dir will be created.
- `paper_results/mi300x/challenge/_diagnostic_logs/` — byte-identical duplicates of files already living in `paper_results/mi300x/challenge/`. **Delete the entire `_diagnostic_logs/` subdir** (commit the deletion). Keep the canonical copies under `paper_results/mi300x/challenge/`. Add `_diagnostic_logs/` to `.gitignore` to prevent recurrence — the 2026-04-15 README in there suggests this was an LLM-session staging area that escaped cleanup.

## 5. Binary-drift mitigation protocol

Root cause: `bench_dmrg_gpu_ablate.py` records `binary_info(...)` per-run, but the harness does not refuse to start if the binary differs from a pinned SHA. Two binary SHAs (`0faa038…` and `98e1ab0…`) appear within hours from the same `git_commit c26f0bd`, indicating ad-hoc rebuilds with mutated CMake flags (e.g., `-DROCM_ARCH`, `-DCMAKE_BUILD_TYPE`).

**Protocol**:
1. Every benchmark campaign starts with a **`git tag campaign-YYYYMMDD-N`** and a **clean rebuild from a clean checkout**.
2. Capture `sha256sum` of every binary into `benchmarks/lib/campaign_manifest.json` (one file per campaign tag).
3. Add a harness check in `bench_dmrg_gpu_ablate.py`: read `--manifest` arg, `binary_info(binary)["sha256"]` must equal the manifest entry, else **abort before any run**.
4. Persist `campaign_tag` and `expected_binary_sha` into every output JSON's top level (not buried in `binary_info`).

This is ~30 LOC in the harness plus a CLI flag.

## 6. Six-commit / single-tag re-run

Yes — for paper resubmission, **all 6 variants must be re-run from a single tagged commit on a single binary SHA per variant**. Aggregate estimate on MI300X for `--reps 10`:

| Variant | Per-rep ~chi128 | Per-rep ~chi256 | 14×2×10 reps total |
|---|---|---|---|
| dmrg-gpu | 18 s | 70 s | ~6.8 h |
| dmrg-gpu-opt | 16 s | 65 s | ~6.3 h |
| dmrg2-gpu | 22 s | 90 s | ~8.7 h |
| dmrg2-gpu-opt | 20 s | 80 s | ~7.8 h |
| pdmrg-gpu | 12 s | 50 s | ~4.8 h |
| pdmrg-gpu-opt | 14 s | 55 s | ~5.4 h |

**Total ≈ 40 wall hours on MI300X.** Practical: split across 2 calendar days on the persistent tmux session; checkpoint after each variant (the harness already writes `partial=true` JSONs each iter).

## 7. §D.A. (Data Availability) update

Paper currently names `paper_results/mi300x/results.json` and `gpu_opt_bench.json` — neither has provenance, and `bench_opt_results.json` (Table 5 source) is not named at all.

**§D.A. should point to**:
- `benchmarks/data/gpu_ablation/<campaign-tag>/{dmrg-gpu,dmrg2-gpu,pdmrg-gpu,dmrg-gpu-opt,dmrg2-gpu-opt,pdmrg-gpu-opt}/results.json` for Table 6.
- `benchmarks/paper_results/mi300x/challenge/<campaign-tag>/<variant>_*.json` for challenge tables.
- `benchmarks/data/bench_opt_results.json` (or its tagged successor) for Table 5 — also fix the 53.13 vs 53.57 number.
- Add a `MANIFEST.json` at repo root listing each table → backing-JSON mapping with sha256s; cite that in §D.A.

## 8. Effort + dependencies

- **Bundle with cluster F (statistics, N≥10 reruns)**: yes — both demand a single-tag rerun campaign with `--reps 10`. Do not run two passes.
- **Sequencing**:
  1. Land the `bench_dmrg_gpu_ablate.py` filter fix + `data_quality` gate (~1 h coding, 30 min review).
  2. Wire `device_k` + `rsvd` flags into dmrg-gpu-opt (and dmrg2-gpu-opt, pdmrg-gpu-opt for consistency) — **est. 4-6 h C++ work** if the call sites mirror dmrg-gpu; otherwise mark columns n/a.
  3. Move corrupted JSONs to `superseded/`, delete empty regate + `_diagnostic_logs/`, commit (~30 min).
  4. Cut `campaign-2026-04-XX` git tag, build all 6 binaries, write manifest (~1 h).
  5. Launch unified ablation campaign on MI300X (~40 h wall, mostly unattended).
  6. Regenerate Tables 5/6/13 from new JSONs; update §D.A. and MANIFEST.json (~3 h).

**Critical-path total: ~3 calendar days of MI300X time + ~2 dev-days.** Defect-cluster F (statistics) is satisfied for free by `--reps 10` in step 5.

Key files referenced: `/home/user/dmrg-implementations/benchmarks/bench_dmrg_gpu_ablate.py` (lines 217-228 for the filter fix), `/home/user/dmrg-implementations/benchmarks/data/gpu_ablation/20260420T181425Z/pdmrg-gpu/results.json`, `/home/user/dmrg-implementations/benchmarks/data/gpu_ablation/20260421T004212Z/dmrg-gpu-opt/results.json`, `/home/user/dmrg-implementations/benchmarks/data/gpu_ablation_regate/20260422T145338Z/pdmrg-gpu/results.json`, `/home/user/dmrg-implementations/benchmarks/paper_results/mi300x/challenge/_diagnostic_logs/`.
