# Cluster A: Drop Newton-Schulz GPU claims, rewrite §4.1/§5.4 to attribute to Block-Davidson

Ground truth: docs/PATH_B_GROUND_TRUTH.md (locked at commit 6f45533)
Planner output SHA (source): a5a55e816646761c6.output
Date planned: 2026-04-23

---

# Defect Cluster A — Plan: Newton-Schulz attribution + Block-Davidson reality

## 1. Decision tree

**Option (a): Re-add NS to GPU, rerun benchmarks, keep paper claims as-is**
- Pros: Preserves paper narrative; "NS diverges at chi>=128" becomes a real measurement; defends §4.1 in current form.
- Cons: NS exists only as CPU Python (`cpu/pdmrg-opt/pdmrg/numerics/linalg_utils.py`); requires porting to HIP/rocBLAS, integrating into one or both `-opt` variants, debugging convergence on MI300X, then re-running the full ablation grid. Likely 5-10 days of GPU dev + a full benchmark cycle. High risk of producing the same divergence the paper claims (because the paper's claim was never measured), which still requires rewrite.
- Effort: ~40-80 hours dev + ~10-20 MI300X-hours bench.

**Option (b): Drop NS claims entirely; present `-opt` as Block-Davidson-only**
- Pros: Matches reality of shipped code; no GPU dev needed; minimal new measurements (BD-only is what's already in the JSONs, modulo CLUSTER A's bundling problem); shortest path to a defensible paper.
- Cons: Loses §4.1 as a positive contribution; "0% win rate" attribution becomes "BD at b=4 hardcoded loses to Lanczos" — still an interesting null result but smaller story; reviewers who saw v1 will notice the disappearance.
- Effort: ~6-10 hours rewrite, 0 GPU dev. May need 1 small re-bench if BD-only must be isolated from the CPU-SVD confound (see §4 below).

**Option (c): Move NS to discussion as CPU-only / future-work**
- Pros: Preserves the polar-decomposition mathematical exposition (which is genuinely useful background); is honest that NS was never benchmarked on GPU; reuses §4.1 prose nearly verbatim.
- Cons: §5.4 results table loses NS column → must rebrand as BD-only anyway, so this option = (b) plus retention of §4.1 as a "design we did not pursue on GPU" subsection. Slightly more text to manage; risk reviewers ask "then why is it in the paper?".
- Effort: ~8-12 hours rewrite, 0 GPU dev.

## 2. Recommended path: **Option (c)**

Rationale: (b) is cleanest but throws away the polar-decomposition exposition that is mathematically correct and useful framing for SVD alternatives. (c) keeps that exposition, demotes it to "explored in CPU prototype, not deployed on GPU because [resource/scope]", and aligns the empirical sections with reality. (a) is rejected because it gambles dev time on producing data that may simply confirm the paper's unsourced claim — and any null result still requires the (b)/(c) rewrite.

The decisive factor: the GPU `-opt` variants already exist and have measurements; the differentiator is BD + CPU-SVD, NOT NS. The honest paper just says so.

## 3. Concrete text-change plan

**main.tex:38-47 (abstract).** Remove sentence claiming "Newton-Schulz diverges at chi >= 128 on GPU." Replace with: "We additionally evaluate a Block-Davidson (b=4) eigensolver variant against single-vector Lanczos on MI300X and find Block-Davidson loses across all tested (model, L, chi) cells, primarily due to a host-side SVD path required by the implementation." Drop NS mention from the abstract entirely.

**main.tex:75 (highlight bullet 4).** Rewrite from "NS, BD, Chebyshev all fail" to "Block-Davidson (b=4) fails to win in any tested cell on MI300X; CheFSI and Newton-Schulz are discussed as alternatives but not measured on GPU." Mirror this caveat in §1.

**main.tex:261-279 (§4.1 NS polar decomposition).** Retitle "Newton-Schulz polar decomposition (CPU prototype, not deployed on GPU)." Keep the math. Add a closing paragraph: "We implemented NS in our CPU Python prototype (`cpu/pdmrg-opt/numerics/linalg_utils.py`) but did not port to HIP/rocBLAS for this study. GPU evaluation of NS-based SVD truncation is left to future work." Remove any divergence-on-GPU claim.

**main.tex:677-705 (§5.4 algorithmic-variants).** This is the largest edit. Drop the "NS+BD bundle" framing. Recast as: "The `-opt` variants in this study differ from baselines in two respects: (i) Block-Davidson (b=4, hardcoded) replaces Lanczos as the local eigensolver; (ii) the SVD truncation is performed on the host via LAPACK rather than on-device via rocSOLVER. We do not isolate (i) from (ii) in the present measurements; see §6." Update Table caption and remove NS columns. Keep the 0% win-rate observation but attribute it to BD+CPU-SVD jointly.

**main.tex:946-947 (conclusion bullet).** Replace "NS, BD, and Chebyshev all underperform" with "Block-Davidson (b=4) with host-side SVD underperforms Lanczos with on-device SVD across all tested cells; isolation of the BD vs SVD-path contributions is left to future work."

**Global.** Search-and-replace "NS+BD" / "Newton-Schulz and Block-Davidson" / "NS-BD bundle" → "Block-Davidson (with host-side SVD)". Remove every occurrence of "NS diverges at chi=128/256." Add one footnote in §4 that CheFSI is also discussed-not-measured (this overlaps Cluster on Chebyshev — coordinate).

## 4. Block-Davidson re-measurement plan

The existing `-opt` JSONs measured BD bundled with CPU-SVD; they are sufficient for the joint claim ("BD+CPU-SVD loses 0/N"). They are NOT sufficient if we want to claim anything about BD in isolation.

**If the rewrite stays at the joint claim (recommended):** zero new runs needed beyond CLUSTER F's general re-bench. One JSON must be replaced regardless: `data/gpu_ablation/20260421T004212Z/dmrg-gpu-opt/results.json` (rc=-6 crash counted as success per ground truth). That's 1 variant × full grid.

**If we want BD-isolation (optional, strengthens story):** add a `-opt` build with BD + on-device rocSOLVER SVD (one-line change: swap `lapack_gesvd` → `rocsolver_gesvd_auto` in the `-opt` SVD dispatch). Re-run the ablation grid for `dmrg-gpu-opt` and `dmrg2-gpu-opt`:

| variant | model | L | chi | cells |
|---|---|---|---|---|
| dmrg-gpu-opt-BDonly | Heisenberg, TFIM | 32, 64 | 64, 128, 256 | 12 |
| dmrg2-gpu-opt-BDonly | Heisenberg, TFIM | 32, 64 | 64, 128, 256 | 12 |

= 24 cells × ~3 reps × ~2-15 min/cell ≈ **4-8 MI300X-hours**, plus the dmrg-gpu-opt re-run for the corrupted JSON (~1-2 hr). Total **~6-10 MI300X-hours**.

## 5. Effort estimate

- **Rewrite (sandbox-only):** 8-12 hours including Table 6 caption updates, figure relabels, bib adjustment (NS-related citations may need pruning).
- **Reruns (MI300X-required):** 1-2 hr if joint claim retained; 6-10 hr if BD-isolation added.

## 6. Dependencies

- **CLUSTER F (statistics / data integrity):** MUST run first or in parallel. The corrupted `20260421T004212Z` JSON is shared with this cluster; re-bench should be coordinated to avoid double-running. If CLUSTER F mandates n>=5 reps and CIs, do that pass once and cover both clusters.
- **CheFSI/Chebyshev cluster:** §4 footnote about "discussed-not-measured" should be unified with the CheFSI fix; same wording template.
- **Cluster on §2.6 SVD-on-CPU claim:** the rewrite of §5.4 references the CPU-SVD path — that section's correction (CPU-SVD true for `-opt` only) must land first to keep the §5.4 narrative consistent.
- **README cluster:** README §85 ("Boundary Merge Disabled" refuted) is independent; no blocker.

Recommendation: sequence as F → A (rewrite + minimal re-bench) → CheFSI cluster, with §2.6 correction batched into A.
