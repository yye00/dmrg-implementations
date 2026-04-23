# Cluster D: pdmrg-gpu-opt --polish CLI fix + bundled reruns

Ground truth: docs/PATH_B_GROUND_TRUTH.md (locked at commit 6f45533)
Planner output SHA (source): af9663bd734391fc8.output
Date planned: 2026-04-23

---

I have enough to produce the plan. Let me write it now.

---

# DEFECT CLUSTER D — Plan

## 1. Code fix plan: make `n_polish` CLI-overridable in pdmrg-gpu-opt

Mirror the working pattern from pdmrg-gpu (which already plumbs `--polish`/`--recal` through `run()`).

**File:line edits (effort ≈ 2–3 hours including rebuild + smoke test):**

- `gpu-rocm/pdmrg-gpu-opt/src/pdmrg_gpu_opt.h:42` — change signature to:
  `double run(int n_outer_sweeps, int n_local_sweeps = 2, int n_warmup = 1, int n_polish = 0);`
  (also drop the non-compliant default 3 → 1).
- `gpu-rocm/pdmrg-gpu-opt/src/pdmrg_gpu_opt_impl.h:3305` — update definition signature to match.
- `pdmrg_gpu_opt_impl.h:3393` — delete `int n_polish = 10;`. Use the parameter directly.
- `pdmrg_gpu_opt_impl.h:3392` — change guard to `if (n_segments_ > 1 && n_polish > 0)` so `n_polish=0` cleanly skips polish (as pdmrg-gpu does at impl.h:2730).
- `gpu-rocm/pdmrg-gpu-opt/src/test_pdmrg_gpu_opt.cpp:371` — change `int n_warmup = 3;` to `int n_warmup = 1;`, add `int n_polish = 0;` next to it.
- `test_pdmrg_gpu_opt.cpp:400` (after `--warmup` parser) — add `if (std::string(argv[i]) == "--polish" && i+1 < argc) { n_polish = std::atoi(argv[++i]); continue; }`
- `test_pdmrg_gpu_opt.cpp:194,255,301` — add `int n_polish` param to the three `test_*` function signatures; thread through the `pdmrg.run(...)` call sites at lines 233, 288, 344.
- `test_pdmrg_gpu_opt.cpp:415,419,423` — pass `n_polish` to all three dispatch calls.
- `benchmarks/run_mi300x_challenge.py:160-171` — `_build_gpu_cmd` already forwards `--polish` for any pdmrg impl; no edit needed once the binary accepts the flag. Verify: pdmrg-gpu-opt is already in the pdmrg branch (line 162 substring match `"pdmrg"`).
- Rebuild: `cd gpu-rocm/pdmrg-gpu-opt/build && cmake --build . -j` on MI300X.
- Smoke test: `./pdmrg_gpu_opt 32 32 5 --warmup 1 --polish 2` and `--polish 0`.

A `--recal` flag is **not** required for pdmrg-gpu-opt because that variant has no recalibration phase (per ground truth). Document this gap in code comments.

## 2. Full rerun matrix on MI300X (CHALLENGE_SIZES, N=10)

Variants: `pdmrg-gpu`, `pdmrg-gpu-opt` (post-fix). Models: heisenberg (14 cells), josephson (15), tfim (15) = **44 cells/variant**. Compliant grid (warmup, polish): {(0,0), (1,0), (1,1), (1,2)} = 4 = matches `run_pdmrg_study.sh`. `local-sweeps=2` fixed; `recal=0` (off) for pdmrg-gpu (default behavior).

Total cells: 2 variants × 4 (w,p) × 44 sizes × 10 reps = **3,520 runs**.

Wall-time estimate (median per-run from existing JSONs ≈ 12 s, with chi=512 and L=500 cells reaching 60–180 s):
- 3-rep ULTRA_TRIM (54 cells × 4 variants × 1 impl) takes ≈ 30–45 min wall (per existing pdmrg_study logs).
- Scaling to full CHALLENGE × 10 reps × 2 impls ≈ ~12 hr per impl → **≈ 24 hr MI300X wall**.
- Add ≈4 hr buffer for chi=512 + L=500 long tail and warmup-pass amortization.
- Plan: **~30 hr exclusive MI300X**, single overnight + half-day.

## 3. Subset rerun (minimal defensible §5.3)

§5.3's Amdahl serial-fraction calculation needs the spread of `(serial = warmup+polish)` vs `(parallel = outer*segments*local)` time per cell. Minimum:

- 1 model: heisenberg (the paper's lead).
- 4 sizes: (L=100, chi=128), (L=100, chi=256), (L=200, chi=128), (L=200, chi=256). Same regime as the paper's figure.
- 1 variant: `pdmrg-gpu` (compliant defaults already match).
- 4 (w,p) cells. N=10 reps.
- = 4 × 4 × 10 = **160 runs ≈ 2–3 hr MI300X**.

This is enough to fit `T(p,w) = T_serial(w,p) + T_parallel` and back out the serial fraction. Add 1 size from josephson + 1 from tfim (40 more runs, +40 min) for cross-model robustness in the appendix.

## 4. Paper rows depending on non-compliant runs

From inventory: `paper_results/mi300x/results.json` rows with `polish_sweeps:3` or `polish_sweeps≥3` and `warmup:3` are pdmrg-gpu-opt artifacts (ground truth confirms hardcoded 10/3, output truncated by K-consecutive convergence at 3–9). Affected paper sites:

- **Table 6 (Fig 6) ablation** — pdmrg-gpu-opt rows. **Replace** with post-fix pdmrg-gpu-opt runs at (warmup=1, polish=2), or drop pdmrg-gpu-opt from the ablation pending fix.
- **§5.3 Amdahl analysis** — uses pdmrg-gpu (compliant defaults). Re-derive serial fraction from the run_pdmrg_study.sh data (already exists, see #5) rather than the headline `results.json` rows.
- **Appendix tables that aggregate pdmrg-gpu-opt timings** — flag rows with provenance hash != post-fix binary.
- **Per-row pdmrg-gpu cells in Table 6** — ground truth says "already used post-fix data", but cross-check the JSON commit hash matches a build with `n_polish=0` default.

## 5. Decision: replace headlines with `run_pdmrg_study.sh` data?

**Yes — replace, do not full-rerun for §5.3.** Rationale:

- `run_pdmrg_study.sh` already exists and IS compliant: explicit `--warmup`/`--polish`, ULTRA_TRIM grid (18 sizes × 3 models = 54 cells × 4 variants), 3 reps. Outputs in `benchmarks/paper_results/mi300x/challenge/pdmrg-gpu_mi300x_challenge_20260415_*pdmrg_w*` (already on disk, 4 tags: w0p0, w1p0, w1p1, w1p2).
- Re-deriving the Amdahl serial fraction from this is a few hours of analysis — not 30 GPU-hours.
- The full 30-hr CHALLENGE × N=10 rerun should still be done **only for pdmrg-gpu-opt after the §1 code fix** (it has zero compliant data on disk).
- Recommended action: (a) accept `run_pdmrg_study.sh` JSONs as the §5.3 source; (b) re-run pdmrg-gpu-opt's Table 6 / appendix cells post-fix at minimum cost (single (warmup=1, polish=2) on the chi=128/256 strip, ~2 hr wall, N=10).

Net new MI300X spend: **~5 hr** (pdmrg-gpu-opt rebuild + targeted rerun) instead of 30.

## 6. Concrete text-change plan

- **§5.3**: replace any pdmrg-gpu number sourced from `paper_results/mi300x/results.json` rows with `polish_sweeps:3` by the median over `pdmrg-gpu_*_pdmrg_w1p1*.json` and `*_pdmrg_w1p2*.json`. Add a sentence: "All pdmrg-gpu timings use n_warmup ≤ 1, n_polish ≤ 2 single-site sweeps (Methods §X)."
- **Methods**: state explicitly that warmup and polish sweeps are 1-site and capped at 2, citing the run_pdmrg_study.sh grid as canonical.
- **Table 6 / Fig 6**: re-render pdmrg-gpu-opt cells from the post-fix (warmup=1, polish=2) data. Footnote: "pdmrg-gpu-opt results regenerated 2026-04 after CLI exposure of `--polish`."
- **Appendix data manifest (§D.A)**: name the four `pdmrg_w*p*.json` files alongside `results.json`. Mark legacy rows with `polish_sweeps>2` as superseded.
- **Caption deltas**: anywhere "polish=10" was mentioned (none found), change to "polish ≤ 2". The Amdahl plot's serial/parallel decomposition labels should switch to (warmup+polish) ∈ {0,1,2,3}.

## 7. Cluster dependencies

- **Cluster F (statistics, N→10)**: tightly coupled. The reruns in #2 and the targeted rerun in #5 must use N=10 (currently `run_pdmrg_study.sh` is hardcoded N=3 — line `REPEATS="${REPEATS:-3}"`). Override env `REPEATS=10` or edit. Confidence intervals also block §5.3's Amdahl fit quality. Recommend: do the §1 code fix and the 5-hr targeted rerun **after** Cluster F's stats infrastructure (CI computation, percentile reporting) lands so re-renders use the new pipeline.
- **Cluster A (provenance gaps in `paper_results/mi300x/results.json`)**: any rerun MUST emit provenance records (binary SHA, git commit, GPU vendor). The 2026-04-21 binary SHA drift issue means we should pin the binary across the entire rerun batch.
- **Cluster B (data-path correctness, e.g. crash-counted-as-success rc=-6 JSONs)**: confirm rerun harness checks return code; the buggy `dmrg-gpu-opt` 20260421T004212Z run pattern should not recur on the pdmrg side.
- **Cluster E (Table 6 ablation re-renders)**: pdmrg-gpu-opt cells in Table 6 are the immediate consumer of the post-fix rerun. Sequence: D-fix → D-rerun → E-rerender.

**Critical-path order:** F (stats) → D-§1 (code fix) → D-§5 (targeted rerun w/ N=10) → §6 (text edits) → E (table re-render).
