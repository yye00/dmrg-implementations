# Cluster C: Strict-1-site limitation disclosure + initial-state sensitivity experiment

Ground truth: docs/PATH_B_GROUND_TRUTH.md (locked at commit 6f45533)
Planner output SHA (source): a512a74787c0cfabc.output
Date planned: 2026-04-23

---

I have enough context to plan. The defect is clear: GPU 1-site DMRG is strict (no Hubig 2015 subspace expansion), which makes the paper's "5-10 sweeps" comparison unfair to 1-site. Let me produce the plan.

---

# Defect Cluster C — Plan: Strict 1-Site GPU DMRG (no subspace expansion)

## 1. Decision Tree

### Option (a) — Implement DMRG3S/noise term in `dmrg-gpu` and rerun
- **Pros**: Fixes the science. Lets us make the 1-vs-2-site comparison on equal footing. Hubig 2015 is the de facto standard; anything calling itself "single-site DMRG" in 2026 is expected to include it. Preserves the §5.2 narrative if 1-site still wins after expansion.
- **Cons**: Non-trivial C++/HIP work — needs density-matrix perturbation `ρ → ρ + α·Σ_a (H_a |ψ⟩)(H_a |ψ⟩)†` per bond, requires per-sweep noise schedule, requires re-tuning α (typical 1e-4 → 1e-8 over sweeps). Adds a kernel for the perturbation tensor and a new SVD on the augmented density matrix. Need MI300X for benchmarking. Risk: 1-site may *lose* to 2-site once 2-site keeps its current advantage on properly-converged runs — narrative may flip.
- **Effort**: ~5-8 engineering days for code + 1-2 days MI300X rerun of all 1-site cells in Tables affected.

### Option (b) — Keep strict 1-site, disclose limitation honestly
- **Pros**: Zero code work. Honest. The paper becomes a study of *strict* 1-site vs 2-site on GPU, which is still a publishable engineering result if framed correctly.
- **Cons**: The §5.2 recommendation "single-site dominates at chi >= 50 for large L" becomes indefensible without heavy caveats — most readers will (correctly) interpret "single-site DMRG" as Hubig 2015. Reviewer-bait.
- **Effort**: ~2 hours of text edits + 1 sensitivity experiment (see §5).

### Option (c) — Drop the 1-site recommendation entirely
- **Pros**: Removes the most contested claim. Paper becomes "2-site GPU DMRG study" — cleaner scope.
- **Cons**: Loses a headline result and ~half of §5.2. Wastes the dmrg-gpu / dmrg-gpu-opt code already written and benchmarked. Doesn't explain *why* we wrote two 1-site variants if we're not recommending them.
- **Effort**: ~1 day of text restructuring across §2.5, §5.2, abstract, conclusion.

## 2. Recommended Path: **(b) + a partial (a) deferred to v2**

Rationale: timeline-driven. (a) is the right scientific answer but adds a kernel + tuning work that risks slipping the paper. (b) is honest, fast, and the sensitivity experiment in §5 *strengthens* the disclosure with data. Frame the paper as "strict 1-site as a GPU-friendly baseline" and explicitly cite Hubig 2015 as the proper variant we did not implement. Add an "Implementation Roadmap" subsection naming DMRG3S as future work.

## 3. (a) Spec — deferred, but documented

Minimal Hubig-style perturbation per left-to-right sweep at site i:
```
# After local eigensolve gives |ψ_i⟩ with reduced density matrix ρ_L
# Compute perturbation tensor P from MPO blocks W_i and environment L_i
P[a,σ,b] = Σ_{a',σ'} L_i[a,a',l] · W_i[l,r,σ,σ'] · ψ_i[a',σ',b]
ρ_L_aug = ρ_L + α(sweep) · contract(P, P*, over [σ,b])
U,S,V = SVD(ρ_L_aug)         # truncate to chi
α(sweep) = α_0 · 10^(-sweep)  # α_0 ~ 1e-4
```
- New HIP kernels: perturbation contraction (~50 LOC), augmented density assembly (~30 LOC).
- Reuse existing rocsolver SVD path.
- **Coding effort**: 3-4 days code, 1-2 days α tuning, 1 day rerun.
- **Rerun matrix**: dmrg-gpu and dmrg-gpu-opt × {Heisenberg, Hubbard} × L∈{32,64,100} × chi∈{64,128,256}. ~24 cells, each 5-30 min on MI300X. Budget: 8 GPU-hours including warm-up.

## 4. (b) Text additions

Insert new paragraph in **§2.5** after the 1-site description:

> "The single-site implementation in this work is the *strict* variant: local eigensolve followed by exact SVD truncation, with no density-matrix perturbation. The Hubig-McCulloch-Schollwöck DMRG3S extension (Hubig 2015) restores the ability to grow Schmidt sectors at fixed bond dimension via a noise term α·Σ_a (H_a|ψ⟩)(H_a|ψ⟩)†; we do not implement it here. Consequently, our single-site results should be read as a lower bound on what an expansion-augmented single-site DMRG would achieve, and the convergence-sweep counts reported in §5.2 reflect this limitation."

Rewrite §5.2 recommendation (lines 546-602) from "single-site dominates at chi >= 50 for large L" to:

> "At chi >= 50 and large L, our strict single-site variant achieves lower wall-time per sweep than two-site, but requires more sweeps to reach the same energy tolerance (5-10 vs 2-3). The crossover wall-time favours single-site for our test set, but this advantage may shrink or invert under DMRG3S, which typically converges in a sweep count comparable to two-site. Practitioners with a DMRG3S-capable code should re-evaluate."

Add Hubig2015 to refs.bib (currently uncited per ground truth) and \cite it in both passages.

## 5. Initial-State Sensitivity Experiment

**Design**: Pick L=64, chi=128, Heisenberg. Generate 5 random initial MPS bond profiles (different RNG seeds, same chi). Run each through dmrg-gpu (strict 1-site) and record sweeps-to-convergence at fixed energy tolerance 1e-8. Also run all 5 through dmrg2-gpu (2-site) for control.

**Hypothesis**: If strict 1-site shows σ(sweeps)/μ(sweeps) > 0.3 across seeds, the "5-10 sweeps" claim is initialization-fragile and should be reported as a range with worst-case. 2-site should show σ < 0.1.

**Sandbox-doable?** No — requires MI300X (we want apples-to-apples wall-time too, and the kernels are HIP). 
**Effort**: 0.5 day to write seeded driver (modify existing run_mi300x_challenge.py to accept --seed and loop), 2 GPU-hours runtime (10 runs × ~10 min), 0.5 day to write up.

This is *cheap insurance*: it either confirms the §5.2 numbers are typical (good — disclosure stands) or shows they were lucky (then we report mean±std and the disclosure becomes mandatory).

## 6. Concrete Text-Change Plan (path b)

| File | Lines | Change |
|---|---|---|
| `main.tex` | 188-205 (§2.5) | Insert "strict variant" paragraph (above) after 1-site description; cite Hubig2015 |
| `main.tex` | 546-602 (§5.2) | Replace recommendation paragraph with hedged version (above); add subsection "Limitations of strict single-site" with sensitivity-experiment results |
| `main.tex` | abstract & conclusion | Replace "single-site dominates" → "strict single-site is competitive, pending DMRG3S evaluation" |
| `refs.bib` | new entry | Add Hubig2015 (PRB 91, 155115); reuse the already-listed-but-uncited entry from ground truth §"Reference reality" |
| `main.tex` | §6 future work | Add "DMRG3S subspace expansion on GPU" bullet |

Estimated total edit effort: 3-4 hours including bib and cross-refs.

## 7. Dependencies on Other Clusters

- **Cluster naming "strict 1-site"**: aligns with any cluster covering paper-vs-code algorithm-name mismatches (e.g. CheFSI / Newton-Schulz absence). Coordinate citation hygiene with the bibliography-cleanup cluster (Hubig2015, Schollwoeck2005, Lanczos1950 are all uncited per ground truth — fix in one pass).
- **Data-provenance cluster**: the sensitivity experiment must be run *after* the rocm-smi/provenance fix, so the new JSONs carry proper `provenance.gpu` fields. Otherwise we add more unprovenanced data.
- **CLAUDE.md compliance cluster**: sensitivity runs use dmrg-gpu (1-site, no warmup/polish concept), so no conflict — but if we ever extend to pdmrg-gpu for comparison, the n_warmup<=2 / n_polish<=2 rules apply.
- **Table-backing-JSON cluster**: any new 1-site cells we publish (option (a)) must land in a single dated benchmark dir with one git SHA, fixing the "6 different commits across ablation JSONs" issue noted in ground truth.

---

**Bottom line**: Recommend (b) + sensitivity experiment now, schedule (a) for v2 / arXiv revision. Total immediate effort: ~1.5 days (0.5 experiment + 0.5 GPU + 0.5 text). Risk: low. Honesty payoff: high.
