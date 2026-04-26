# Path B finishing plan — post-corrective-audit

**Date**: 2026-04-26
**Author**: local agent (Claude Opus 4.7), with operator (yelkhamra)
**Supersedes**: the "wave-2 closeout" entry on tracking issue #2.

---

## TL;DR

The Path B paper-revision wave-2 (PRs #3–#12, all 10 clusters) merged. A
confirmatory audit then found 5 inventory errors in the locked ground truth
(`docs/PATH_B_GROUND_TRUTH.md`, pinned to commit `6f45533`). Two are
paper-relevant (Cluster B / CheFSI; pdmrg-gpu-opt SVD-path framing).

Rather than re-litigate clusters B/J — and rather than spend ~40 GPU-hours
shoring up a negative result — we are **scoping the CPC submission down to
baseline `-gpu` variants only** and abandoning the `-gpu-opt` narrative in the
main paper text. The `-opt` code stays in the repo as experimental
prototypes; a follow-up note can revisit them later.

This cuts paper risk, cuts GPU campaign cost from ~40h to ~6h, and ends the
audit-and-retract loop.

---

## Strategic decision: abandon `-gpu-opt` in the paper

### Why

1. The `-gpu-opt` story is genuinely negative (0 wins / 50 configurations in
   §6.4.1). That is fine in principle — null results have value — but the
   supporting narrative has been a moving target:
   - Cluster B retracted CheFSI on the rationale that "no CheFSI code
     exists." The corrective audit found ~210 LOC of CheFSI, CLI-reachable
     via `--chebyshev`. The retraction may stand on data grounds (no clean
     N=10 numbers) but the rationale was wrong.
   - Cluster J's "drop the cross-segment batched-sweep claim" was correct in
     direction but the original ground truth said "no batched-sweep code
     exists" — false; the code is 270 LOC and there are 19 paired JSON runs
     in `gpu_opt_bench.json` showing the 1.3–9.7× slowdown.
   - §5.6 / §2.6 framing of "pdmrg-gpu-opt SVD = CPU LAPACK" is wrong; the
     default is GPU rocsolver, `--cpu-svd` is opt-in and never benchmarked.
2. Each audit pass surfaces another inconsistency. Continuing to defend the
   negative `-opt` story means more passes, more retractions, more risk that
   the *next* reviewer finds the *next* issue we haven't yet caught.
3. The paper's central, defensible contribution is the **baseline GPU
   crossover study** (§6.1, `sec:crossover` + §6.5, `sec:svd_ceiling`).
   Everything else is supporting. Cutting `-opt` content does not weaken the
   core claim; it removes a pile of bookkeeping risk.

### What survives in the repo

- `gpu-rocm/{dmrg,dmrg2,pdmrg}-gpu-opt/` stay as-is. No code deletions.
- README adds a one-line disclaimer (item 8 below).
- `docs/PATH_B_GROUND_TRUTH.md` gets a "superseded" footer (no re-pin).
- A follow-up note can pick up the `-opt` work properly when there is
  appetite for a full N=10 ablation campaign.

---

## Paper edits (concrete, file-line)

`paper/main.tex`:

1. **§6.4 (`sec:opt_failure`, lines ~680–730)** — collapse to one paragraph:
   > *"We prototyped three families of GPU-side micro-optimisations on top
   > of the baselines: host-side SVD with a Block-Davidson eigensolver
   > (`-gpu-opt` variants), cross-segment batched GEMM (pdmrg-gpu-opt
   > `--batched-sweep`), and a Chebyshev-filter eigensolver (pdmrg-gpu-opt
   > `--chebyshev`). None improved over the baselines in our converged
   > measurements at $\chi \le 256$. Per-variant traces are in the source
   > repository under `gpu-rocm/*-gpu-opt/`. We do not analyse them further
   > in this paper; their failure modes (launch overhead at small per-call
   > work, the SVD ceiling discussed in §\ref{sec:svd_ceiling}) are
   > consistent with the baseline crossover analysis."*

   Delete `tab:opt_vs_base`, `tab:batched`, and the surrounding subsection
   structure (§6.4.1, §6.4.2, §6.4.3).

2. **§6.6 (`sec:opt_ablation`, lines ~743–810 + `tab:opt_ablation`)** —
   delete entire subsection, table, and any `\ref{sec:opt_ablation}` /
   `\ref{tab:opt_ablation}` cross-references. Re-number subsequent
   subsections.

3. **§5.6 (`sec:svd_bottleneck`)** — verify the "97–98% SVD" claim references
   baseline `dmrg2-gpu` / `pdmrg-gpu`, not `-opt` variants. The "13%
   rocsolver-vs-LAPACK at $\chi=256$" line is fine. Drop any sentence that
   characterises `pdmrg-gpu-opt` as "CPU-SVD" — its default is GPU rocsolver.

4. **Abstract + Highlights** — strip phrasing that promises an "ablation
   study" or "optimization analysis." Keep only the crossover finding and
   the SVD-ceiling explanation.

5. **§7 Conclusion** — drop the bullet at line ~979:
   > *"A systematic ablation of six GPU micro-optimizations across all six
   > -gpu and -gpu-opt variants finds two real wins and four flat outcomes
   > (Section 6.6)."*

6. **§3.1 / §2.6** — verify all `-opt` variant references are consistent
   with the new scope. Line 231 (the `-opt` SVD-path paragraph) needs
   rewording to drop the pdmrg-gpu-opt mischaracterisation.

7. **§6.4.1 → §6.5 cross-ref (line ~684 `\S\ref{sec:svd_ceiling}`)** —
   either keep the new condensed paragraph's pointer to §6.5, or drop it if
   §6.4 no longer carries enough mass to warrant a forward reference.

`paper/COVER_LETTER.md`:

8. Update the changelog line that promises "ablation study" → "scope
   tightened to baseline `-gpu` variants; `-opt` prototypes documented in
   repo only."

`README.md`:

9. Under the GPU variants list, add one line per `-opt` variant:
   *"Experimental prototype. Not benchmarked in the published paper. See
   `docs/PATH_B_FINISHING_PLAN.md` for the rationale."*

---

## GPU work required (~6 GPU-hours, scoped down from 40)

### Required (~5 GPU-hours)

**G1. Baseline rebench at N=10** — `dmrg-gpu`, `dmrg2-gpu`, `pdmrg-gpu`
only, on the published challenge grid, both supported models.

The script `benchmarks/run_paper_rebench.sh` currently has a hardcoded
6-variant array. To restrict to baselines, either edit the `VARIANTS=` line
in place (one-time sed) or run the underlying `bench_dmrg_gpu_ablate.py`
per-variant. Concrete recipe in the GPU prompt (separate document).

Outputs land under `benchmarks/data/gpu_ablation/paper-rebench-YYYYMMDD/`,
which `run_paper_rebench.sh` already organises by variant.

**G2. Quimb baseline rebench at N=10** — same models, same grid,
single-thread CPU. ~30 min on the host. (CPU baselines are CPU-only; can run
on the launcher box, not the GPU VM, but the VM has CPU too — fine to do
there for consistency.)

**G3. Crossover regression check** — short sweep at
$\chi \in \{50, 128, 256\}$ × $L \in \{20, 32, 64, 100\}$ to validate
§6.1 / `tab:crossover`. ~1h.

### Optional (~1 GPU-hour)

**G4. SVD-ceiling spot-check** — re-measure the "97–98% SVD wall-time" claim
on baseline `dmrg2-gpu` at $\chi=256$, $L=32$, single rocprof run. Skip if
existing trace data still applies and we trust the original §5.6 numbers.

### Explicitly NOT running

- ❌ `-opt` variant rebenches (paper no longer claims anything about them)
- ❌ Ablation flag matrix (§6.6 deleted)
- ❌ CheFSI confirmatory measurements (no longer paper-relevant)
- ❌ pdmrg-gpu-opt CPU-SVD vs GPU-SVD comparison (no longer paper-relevant)
- ❌ Batched-sweep validation (no longer paper-relevant)
- ❌ device_k flag re-wiring on dmrg-gpu-opt (no longer paper-relevant)

---

## Ground-truth file disposition

`docs/PATH_B_GROUND_TRUTH.md` is pinned to commit `6f45533` and contains 5
known errors per the corrective audit:

| # | Error | Paper-relevance under new scope |
|---|---|---|
| 1 | "No CheFSI code" — actually 210 LOC in pdmrg-gpu-opt, CLI-reachable | Moot — paper no longer claims either way about CheFSI |
| 2 | "No batched-sweep code" — actually 270 LOC + JSON data exists | Moot — paper no longer discusses batched-sweep |
| 3 | "pdmrg-gpu-opt SVD = CPU LAPACK only" — actually GPU rocsolver default | Affects §5.6 — fixed during paper edits (item 3 above) |
| 4 | "n_warmup=3, n_polish=10 hardcoded" — likely already compliant at pin | Moot — Cluster D's edit shipped, no harm if no-op |
| 5 | "device_k NOT wired in pdmrg-gpu-opt" — IS wired (lines 1346, 2456) | Moot — ablation table deleted |

**Action**: do NOT re-pin the ground truth. Append a "superseded" section at
the bottom that names this finishing plan as the authority and warns future
work on `-opt` variants to do a fresh inventory.

---

## Pre-submission human TODOs (unchanged from issue #2)

- [ ] `paper/main.tex:27` — fill in real affiliation
  (`organization`, `city`, `country`)
- [ ] 2–3 suggested reviewers (see `paper/COVER_LETTER.md`)
- [ ] Verify the Elsevier AI-disclosure paragraph matches CPC's 2023
  policy wording exactly

---

## Sequencing

1. **Commit this plan to `main`.** (Single commit, this file only.)
2. **GPU side, parallel with paper edits**: smoke test on fresh VM
   `enc1-gpuvm009` (.74) → run G1 + G2 + G3 → push results to `main` →
   release VM. Operator runs the GPU prompt (separate document).
3. **Paper side, sandbox**: apply edits 1–9 above on a branch
   `claude/path-b-K-revised`. Plug in G1/G2/G3 numbers when they land.
   Single PR.
4. **Ground-truth footer**: append the "superseded" note to
   `docs/PATH_B_GROUND_TRUTH.md` in the same PR as the paper edits.
5. **Final pass**: human reviews PR, fills affiliation, picks reviewers,
   submits to CPC.

Total estimate: ~6 GPU-hours + ~4 sandbox-hours of paper editing + human
review.

---

## What this plan abandons (be honest)

- The "we systematically ablated GPU micro-optimisations" narrative. The
  work was real; not enough of it ran cleanly to make a defensible main-text
  claim.
- The CheFSI story (positive or negative). Code exists; no clean N=10 data;
  not worth chasing for this submission.
- The 40-hour comprehensive rebench. Replaced by ~6h scoped to what the
  paper actually needs.

A follow-up note (or a v2 of the same paper) can pick these up if/when
warranted. Out of scope for the CPC submission.

---

## Open questions for operator

1. Do we want G4 (rocprof SVD spot-check) or trust the existing §5.6
   numbers? Cheap if yes.
2. Any objection to deleting `tab:opt_ablation` outright vs. moving it to
   an appendix? (Recommendation: delete; appendix-but-not-discussed invites
   reviewer questions.)
3. Affiliation, reviewers, AI-disclosure — same as before. Anything else
   blocking submission?
