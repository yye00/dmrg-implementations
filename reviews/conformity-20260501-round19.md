# Full conformity review — round 19 (2026-05-01, HEAD cafd628)

Confidence re-run before MI300X G1 baseline allocation. Six fully
independent sub-reviews dispatched; each ran the defect-class registry
as pre-step (TOTAL HITS: 0 at code state 12d02c5 prior to fix).

## Charter proof — sub-review status

| Sub-review | Status | Findings |
|---|---|---|
| vertical-review-dmrg   | OK | 0 criticals, 0 highs |
| vertical-review-dmrg2  | OK | 0 criticals, 0 highs |
| vertical-review-pdmrg  | OK | 0 criticals, 0 highs (1 carry HIGH on opt-in path) |
| horizontal-review-base | OK | 0 criticals, 0 highs |
| horizontal-review-gpu  | OK | 0 criticals, **1 NEW HIGH** (H19) |
| horizontal-review-opt  | OK | 0 criticals, 0 highs (1 carry HIGH on opt-in path) |

**The R19 re-run paid for itself.** R18 missed H19 — a sibling
fix-propagation gap from R17's H4 multi-gpu Step 3 port. Only the
two-site path got collapsed; single-site (the DEFAULT warmup/polish
path per PDMRG-rules-2026-04-15) still had a per-wp host loop. Caught
by horizontal-review-gpu in R19. **Now fixed in `cafd628`.**

## CRITICALs (deduplicated)

**None.**

## HIGHs (deduplicated)

- **H19-multi-ss-step3 — FIXED in `cafd628`** (found by:
  horizontal-review-gpu). pdmrg-multi-gpu apply_heff_single_site
  Step 3 had a per-wp host loop (D launches per call, ~impl
  :1719-1737). Mirror of pdmrg-gpu :1992-2018 pattern: 1 setup
  kernel + 1 gemm_batched (D·d) + 1 gemv reduce with d_ones_D.
  Why R18 missed it: D13 registry pattern matched only bare
  `Traits::gemm(` inside the per-wp loop; the multi-gpu single-site
  loop wraps `gemm_batched(batch=d)`. **Widened D13 in `cafd628`**
  to also catch wrapped batched calls inside the same loop.

- **H-opt-batched-lanczos-host-mode** (carry, found by:
  horizontal-review-opt, vertical-review-pdmrg). Same finding as
  R18. `pdmrg_gpu_opt_impl.h:2870-2980` retains host-resident
  `h_alpha`/`h_beta`/etc. in `batched_lanczos_eigensolver`, gated
  on `set_use_batched_sweep(true)` — **default OFF**, never on G1
  baseline. Same shape applies to `chebyshev_eigensolver`
  (:2028, default OFF). Both opt-in. Defer until those campaigns.

## MEDIUMs (deduplicated)

- **M-multi-gpu-local-kernels-dup** (carry, found by:
  vertical-review-pdmrg, horizontal-review-gpu). pdmrg-multi-gpu
  has 4 file-local pointer-setup kernels (`setup_heff_ss_step3_ptrs`,
  `setup_heff_ss_step3_full_ptrs` [added in this round],
  `setup_lenv_step3_ptrs`, `setup_renv_step3_ptrs`) that are also
  defined locally in pdmrg-gpu. Different TUs → not ODR; consolidate
  into `common/batch_ptrs_kernels.h` in R20.
- **M-multi-gpu-precompute-fused-mpo-host** (carry from R18, found
  by: vertical-review-pdmrg). `pdmrg_multi_gpu_impl.h:700-730` host
  nested-loop + H2D for fused-MPO precompute. `set_mpo` time, not
  per-sweep — outside "no host roundtrips per sweep" rule.
- **M-opt-pdmrg-single-site-graph-comment-stale** (carry from R18,
  found by: horizontal-review-opt). `pdmrg_gpu_opt_impl.h:2302-2311`
  comment claims host-stack pointers; actual code uses device
  kernels. Either update or re-enable graph capture.
- **M1-opt-lanczos-init-D2H-sync** (carry, found by:
  vertical-review-dmrg). dmrg-gpu-opt explicit hipStreamSynchronize
  at lanczos init. Cosmetic, bounded cost.

## NITs (deduplicated)

- **N-base-header-overstates-pointer-mode** (carry, found by:
  vertical-review-dmrg).
- **N-pointer-mode-guard-return-discards** (carry, found by:
  horizontal-review-gpu).

## FALSE POSITIVES (cross-review verified)

- **D7+D8 host LAPACK gesvd** in pdmrg-{gpu,gpu-opt,multi-gpu} —
  filtered (use_cpu_svd_ opt-in branches + lwork=-1 workspace
  queries).
- **D13 -base apply_heff per-wp host loop** — registry whitelists
  -base by charter (naive single-GEMM IS the baseline tier).
- **pdmrg-gpu-base form_theta_with_V V upload + precompute_WW host
  build** — sibling-consistent set-once / boundary-state shuttle.

## SUMMARY VERDICT

- **Block GPU run / paper submission?** **NO.** 0 CRITICALs.
  H19 was found AND fixed in this round. The two carry HIGHs are on
  opt-in code paths (`set_use_batched_sweep(true)` /
  `set_use_chebyshev(true)`) — both default OFF, never on G1
  baseline. Benchmark drivers must NOT enable these toggles for the
  G1 campaign.

- **Top-3 actions before next major event:**
  1. **Task #120 — remote MI300X build verification.** This is the
     last gate; everything above is static review.
  2. R20 cleanup: consolidate the 4 local pdmrg-{gpu,multi-gpu}
     pointer-setup kernels into `common/batch_ptrs_kernels.h`
     (M-multi-gpu-local-kernels-dup).
  3. Update or remove the stale graph-capture comment in
     pdmrg-gpu-opt apply_heff_single_site (M-opt-pdmrg-…-stale).

- **What was checked vs. R18 baseline (`reviews/conformity-20260501-round18.md`):**
  - R18 declared READY with 0 CRITICALs / 1 carry HIGH.
  - R19 re-run **caught a NEW HIGH (H19)** that R18 missed because
    the D13 registry pattern was too narrow. R18's verdict was
    *correct under the registry as it stood*; the registry itself
    needed widening.
  - **Lesson learned:** the registry is only as good as its grep
    patterns. When a NEW failure mode is discovered, widen the
    pattern in the SAME commit as the fix so the next round's
    pre-step catches the class proactively. R19 commit `cafd628`
    does both.
  - Zero regressions: every R15→R18 fix is intact.

- **Methodology validation:** the registry pre-step caught 0 hits at
  HEAD before each sub-review, confirming the proactive sweep keeps
  pace with code edits. The R19H19 finding came from human-in-the-loop
  Technique-A/F deep audit — exactly the kind of finding the registry
  cannot catch alone, which is why we still run the 6 sub-reviews.
  This is the correct cost-of-overlap: registry catches the
  mechanical, sub-reviewers catch the structural-novel.

The codebase is **READY for the MI300X G1 baseline campaign** modulo
remote build verification. R19 confidence run delivered one
high-priority fix that R18 missed. Recommend no further code
changes until task #120 (remote build) and the G1 campaign.
