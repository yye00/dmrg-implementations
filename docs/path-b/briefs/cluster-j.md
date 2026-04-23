# Cluster J: Paper rewrite — contradictions, framing, prior art, bib hygiene

Ground truth: docs/PATH_B_GROUND_TRUTH.md (locked at commit 6f45533)
Planner output SHA (source): a3b5fbcdba2e10c06.output
Date planned: 2026-04-23

---

# Defect Cluster J — Paper Rewrite Plan

## 1. "Uniformly fails" rewrite (Conclusion bullet 4 + §5.4)

**Old (Conclusion, bullet 4):** "GPU DMRG uniformly fails to beat the CPU baseline at chi <= 256."

**New:** "Across the chi <= 256 regime, the *default* GPU configuration (Lanczos + on-device rocSOLVER gesvd) trails the single-thread quimb CPU baseline on a per-sweep basis for the Heisenberg and TFIM problem sizes we tested. Two ablations reverse that ordering: enabling RSVD truncation lowers GPU sweep time below the CPU baseline at chi=256 (§5.5, Table 10), and the LANCZOS_GRAPH capture path narrows the gap further on dmrg2-gpu (§5.5, Table 11). The remainder of this paper's claims about 'GPU losing' refer specifically to the un-ablated default, not to the optimized envelope."

**§5.4 paragraph rewrite:** Replace "the GPU is uniformly slower" with "the un-ablated GPU default is slower in 41/44 measured cells; the 3 cells where GPU wins at chi=256 are all RSVD- or graph-capture-enabled and are discussed in §5.5."

## 2. "93%" highlight rewrite

**Old:** "93% of CPU wins use 1 thread, demonstrating BLAS contention."

**New:** "Of the 14 cells in which the CPU baseline beats the default GPU configuration, 13 (93%) were measured at the single-thread setting that the harness actually exercises (`run_mi300x_challenge.py:382`). We do not have the multi-thread sweep data needed to support a BLAS-contention claim; the apparent 93% is an artifact of the single-thread harness rather than evidence about scaling. See limitations §6.5."

## 3. "Solution in search of" paragraph

**Old:** "GPU DMRG at chi <= 256 is a solution in search of a problem."

**New:** "At chi <= 256 the arithmetic intensity of a single MPS update is too low to amortize host-device transfer and kernel launch overhead in our default Lanczos path; on this hardware (single MI300X, ROCm 7.2) and at this bond dimension, an optimized single-thread CPU stack remains competitive. We do not generalize beyond this regime: §5.5 shows that algorithmic changes (RSVD, LANCZOS_GRAPH) already shift the crossover, and §6.4 outlines why we expect chi >= 1024 to favor the GPU. Whether smaller-chi GPU DMRG is *useful* depends on workload (e.g., parameter sweeps, time evolution) considerations outside this paper."

## 4. A2DMRG fair-framing rewrite (§5.3.2 + §6.3)

**§5.3.2 add:** "Grigori & Hassan (2024) targeted the *large-chi, deep-bulk* regime where the asymptotic FLOP advantage of randomized + recompressed contractions dominates. Our test points (Heisenberg L<=64, chi <= 512) sit at the small end of that regime; an unfavorable result here is consistent with their published crossover analysis and should not be read as a refutation of A2DMRG. We did not contact the original authors before publication; an acknowledgement and corrected re-evaluation will appear in a follow-up note (deferred)."

**§6.3 closing:** Strike "A2DMRG is not competitive." Replace with "Within our tested regime A2DMRG was not competitive; the regime where it was designed to win lies above our chi ceiling."

## 5. New §6.5 Limitations / Threats to Validity

Spec the subsection (target ~250 words) covering:
- **Hardware scope.** Single MI300X (gfx942, ROCm 7.2). No A100/H100/B200; no multi-GPU. CUDA port is scaffolding only.
- **CPU baseline caveats.** Vendor is Intel Xeon Platinum 8470 (not AMD EPYC); harness is hardcoded to `threads=1`; OpenBLAS provenance unverified (no build script, no LD_LIBRARY_PATH); cotengra is not actually invoked by the runner; quimb is run with flat bond_dims=chi rather than the idiomatic graduated schedule.
- **SVD path.** dgesvd (full) used everywhere; dgesdd (divide-and-conquer) untested. -opt variants force CPU host roundtrip; primary -gpu variants stay on device.
- **Algorithmic scope.** Strict 1-site DMRG; no DMRG3S, no noise, no subspace expansion (cf. Hubig 2015). Warmup/polish capped at <=2 per CLAUDE.md.
- **Provenance gaps.** Only 15/116 result JSONs carry full provenance; all GPU vendor fields read 'unknown' (rocm-smi parse failure); two ablation JSONs contain rc=-6 crash data.
- **No physics convergence diagnostics** (see §6.5.2 below).

## 6. "We expect chi >= 1024" softening (§6.4)

**Old:** "We expect GPU to win at chi >= 1024."

**New:** "Extrapolating sweep-time scaling from chi in {64, 128, 256, 512} (Fig. X), we *conjecture* a CPU/GPU crossover near chi ~ 1024 on the MI300X. We have not measured this regime: the conjecture rests on (i) the empirical exponent fit and (ii) the asymptotic chi^3 GEMM/SVD argument. It should not be cited as a measured result."

## 7. Physics convergence additions

Add §6.5.2 "Physics validation we did not perform":
- **TFIM:** central charge c via entanglement entropy fit S(L) = (c/6) log L at the critical point g=1; expected c=1/2. ~2 GPU-hours on MI300X (one chi=256 run per L in {32, 64, 96, 128}).
- **Heisenberg L=64 chi=256:** ⟨H²⟩-⟨H⟩² variance per site as convergence indicator; should be < 1e-6 at converged chi. Already computable from existing MPO contractions; ~0.5 GPU-hours.
- **Heisenberg achieved E/L vs Bethe ansatz:** E/L = 1/4 - ln 2 = -0.443147... in thermodynamic limit; report fractional error per (L, chi) cell. Free, just postprocess existing JSONs.

Total added MI300X time: ~3 GPU-hours.

## 8. Bibliography fix plan

- **Delete** `Liu1978` entry; remove the one citation in §2.4 (the LDL fact stands without it, or cite Higham 2002 *Accuracy and Stability* which is already standard).
- **Rename** bib key `Nakamura2013` → `Nakatsukasa2013`; global s/Nakamura2013/Nakatsukasa2013/ in main.tex (currently 2 cite sites).
- **Fill** Nakamura2020 (likely the 2020 *Phys. Rev. B* preprint — locate volume/pages or downgrade to arXiv-only entry with eprint field).
- **Fix** Nemes2014 DOI: strip ".surface" suffix; verify against CrossRef.
- **Add** entries: `HauschildPollmann2018` (TeNPy, SciPost Phys. Lect. Notes 5), `Hubig2015` (DMRG3S, PRB 91 155115), `Zhai2021` (already in bib but uncited — add cite), `Fishman2022` (ITensor, SciPost Phys. Codebases 4), `Ganahl2023` (GPU DMRG, PRX Quantum).
- **Drop** uncited dead weight: Schollwoeck2005, Kantian2019, Ren2021, Lanczos1950 (or cite them properly — Lanczos1950 belongs at the first mention of Lanczos in §2.3; Schollwoeck2005 belongs in §1).

## 9. Cotengra / CheFSI retraction wording (coordinate with P-B, P-E)

**Cotengra (§3.1):** Strike "with cotengra for path optimization." Replace with "quimb's default opt_einsum greedy path; cotengra is *not* invoked (we initially planned a cotengra comparison; it was not run for this paper)."

**CheFSI (§5.7):** Strike Table 14 and §5.7 entirely, OR replace with "CheFSI was *not* benchmarked for this paper. The implementation we cited (Zhou 2006) was not ported to GPU within our scope; the '1.9–11x slower' figure that appeared in pre-print v1 was an internal estimate and has been retracted." Mark the retraction in changelog.

## 10. CPU spec correction (bundle with P-E)

§3.1 line 386: "AMD EPYC processor" → "Intel Xeon Platinum 8470 (MI300X host) and 8480C (H100 host); both Sapphire Rapids, 52C/104T per socket."

§3.1 line 397: drop "thread counts of 1, 2, 4, 8, 12" (no such sweep exists). Replace with "single-thread quimb (`threads=1` hardcoded in `run_mi300x_challenge.py:382`)." Cross-link to limitations §6.5.

## 11. Prior-art additions

- **§1 (intro), 2nd paragraph**: cite ITensor (Fishman 2022), TeNPy (Hauschild-Pollmann 2018), Block2 (Zhai 2021) as the production CPU/GPU DMRG ecosystem this work builds on.
- **§2.5 (single-site limitations)**: cite Hubig 2015 for DMRG3S as the standard remedy we deliberately do not implement (link to limitations §6.5).
- **§6.4 (chi >= 1024 conjecture)**: cite Ganahl et al. 2023 (TensorNetwork-on-GPU) for an existing data point at large chi on NVIDIA hardware — independent confirmation that the crossover is real and architecture-portable.

## 12. Effort + dependencies

- **Sandbox-only (no compute):** items 1, 2, 3, 4, 6, 8, 9, 10, 11. Estimated 4–6 hours of careful LaTeX editing + bib hygiene + cross-reference pass. Single editor, no parallelism needed.
- **Item 5 (limitations):** ~1 hour write, but **depends on cluster F** (CPU-baseline audit confirming Intel Xeon + thread=1 + cotengra-not-used) and **cluster H** (provenance audit confirming 15/116 number and rc=-6 JSONs).
- **Item 7 (physics validation):** ~3 GPU-hours on MI300X + ~2 hours postprocessing. **Depends on cluster H** for re-run pipeline; can run in parallel with text edits. Decision point: include in v2 of paper, or defer to companion note.
- **Item 9 cotengra retraction depends on P-E** (confirming cotengra import grep returns empty); CheFSI retraction depends on **P-B** (confirming no chefsi*/chebyshev* source files).
- **Critical path:** items 1, 5, 8 must land before submission; item 7 is the only item that needs GPU time and is the schedule risk if v2 is to claim physics convergence.

Total cluster-J effort estimate: **~8 person-hours of editing + 3 GPU-hours**, blocked by clusters F and H landing first.
