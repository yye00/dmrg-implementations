# Paper TODO

Running list of items to revise / extend before submission to CPC,
and items deferred to the follow-up paper.

---

## DONE (this revision pass, commit pending)

- [x] CPC Highlights block (5 bullets, ≤85 chars each)
- [x] CPC Program Summary block (Program Title, repo, license, languages, libs,
      Nature of problem, Solution method, Additional comments)
- [x] CRediT authorship contribution statement
- [x] §4.6 GPU micro-optimization framework (ablation flags + methodology)
- [x] §5.5 Per-flag ablation across variants (Table + Figure 6 heatmap)
- [x] §6.4 updated with RSVD + LANCZOS_GRAPH wins
- [x] Conclusion bullet 5 added (micro-opt findings)
- [x] Cover letter draft (paper/COVER_LETTER.md)
- [x] Tables: ablation table transposed to fit page width (was 205pt overflow)
- [x] URL break support: [hyphens]{url} + [breaklinks=true]{hyperref}
- [x] §3.1 Hardware: explicit MI300X-only justification with H100 FP64 caveat
      (33.5 TFLOPS native vs 67 TFLOPS tensor; MI300X has 81.7 TFLOPS native;
      preliminary H100 data exists but cross-vendor comparison deferred)

## OPEN — must do before submission (author action)

- [ ] **Affiliation**: line 27 of main.tex — `organization={}, city={}, country={}`
      currently empty. Author must fill in.
- [ ] **Acknowledgments**: verify Hot Aisle credit + add any funding
      sources, grant numbers.
- [ ] **Suggested reviewers**: COVER_LETTER.md placeholder. Need 2-3
      DMRG/GPU-computing peers without recent collaborator conflicts.
      Candidates to consider: ITensor / TeNPy / Block2 / quimb teams,
      authors of recent GPU tensor-network papers.

## OPEN — should do before submission (mostly low-risk)

- [ ] **Single-thread CPU win claim verification**: §5.1 / Conclusion
      claim 93% of CPU wins are single-threaded. Re-verify against the
      raw `benchmarks/paper_results/mi300x/results.json` to confirm
      this is still true after the recent challenge re-runs (commit
      cfba1db).
- [ ] **A2DMRG warm-up table**: tab:a2dmrg_warmup is referenced in
      §6.3 (paradox discussion). Verify it's present and the numbers
      match the §5.3 prose.
- [ ] **§6.2 SVD ceiling — quantify GPU SVD crossover**: the paper says
      "13% improvement at χ=256 suggests crossover near χ ≈ 300--500".
      One χ=512 dmrg2-gpu point with rocsolver SVD vs LAPACK would
      either confirm this or correct the projection. Cheap to run.

## OPEN — would strengthen the paper (medium effort)

- [ ] **χ-scaling figure for dmrg2-gpu RSVD speedup**:
      χ ∈ {128, 256, 512, 1024} on a single Josephson config. One
      figure showing the RSVD win growing or saturating with χ would
      anchor the §5.5 finding and address the "but is this just for
      moderate χ?" reviewer concern. ~1 hour MI300X.

- [ ] **pdmrg-gpu-opt convergence regression note**: the polish
      single-site phase reaches ~1.3e-8 above pdmrg-gpu's deterministic
      energy due to missing recalibration phase. Document as a known
      limitation in §5 or §6, OR finish the recalibration port
      (commit pending in fix branch).

- [ ] **Memory-bandwidth analysis for FUSE_LANCZOS / DEVICE_K flatness**:
      currently the §5.5 prose says these are "bandwidth-dominated"
      without supporting numbers. A simple roofline analysis for one
      Lanczos iteration would make the claim quantitative.

- [ ] **Multi-GPU pdmrg-multi-gpu data**: §6.4 mentions multi-GPU as
      future work but the binary exists. Even one 4-GPU run on Heisenberg
      L=100 χ=128 would strengthen the multi-GPU paragraph.

## OPEN — deferred to follow-up paper (large scope, intentionally out of scope here)

- [ ] **Cross-vendor controlled comparison (MI300X vs H100)**: requires
      per-kernel profiling to disentangle native-FP64 vs tensor-FP64
      contributions on H100. Preliminary 18-config data exists but
      interpreting it correctly needs more work than this paper warrants.

- [ ] **Large-χ regime (χ ≥ 1024)**: the paper's "GPU wins at large χ"
      conclusion is asymptotic; we have not measured beyond χ=256.
      A separate paper targeting χ=1024-4096 with multi-GPU support
      would address the "is this only relevant at moderate χ?" critique.

- [ ] **Mixed-precision SVD**: §6.4 briefly mentions FP32-warmup +
      FP64-final-convergence as a possible win path. Out of scope here.

- [ ] **Real-time / TDVP analog**: this paper is ground-state DMRG only.
      The same MFMA/launch-overhead tradeoffs likely apply to
      time-evolution algorithms but should be tested separately.

## OPEN — cosmetic / nice-to-have

- [ ] Reduce remaining 14 overfull hboxes (all ≤25pt) — typesetting
      polish; not required for review acceptance.

- [ ] Decide whether to integrate or remove the orphan figures in
      `paper/figures/` (fig1-fig5 exist but are not currently
      referenced in the text). Current paper is 38 pages with one
      figure (the new ablation heatmap); some reviewers may expect
      more visual anchoring of the wall-time / crossover findings.

- [ ] **Consistent in-text "Fig.\ N" vs "Figure~\ref{}"** spelling.
      Currently mixed.

- [ ] Acknowledgments — sentence-level proofread.

---

## Tracking

| Pass | Date | Commits | Result |
|---|---|---|---|
| 1 | 2026-04-22 | `4047915` | §4.6 + §5.5 + Fig 6 added |
| 2 | 2026-04-22 | `bb8bcd0` | CPC front-matter + format fixes |
| 3 | 2026-04-23 | (this) | H100 FP64 caveat + this TODO file |
| 4 | TBD | TBD | Author affiliation + reviewers + (optional) χ-scaling |
| 5 | TBD | TBD | Submission |
