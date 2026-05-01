# Full conformity review — round 18 (2026-05-01, HEAD 12d02c5)

## Charter proof — sub-review status

| Sub-review | Status | Findings |
|---|---|---|
| vertical-review-dmrg   | OK | 0 criticals, 0 highs |
| vertical-review-dmrg2  | OK | 0 criticals, 0 highs |
| vertical-review-pdmrg  | OK | 0 criticals, 0 highs |
| horizontal-review-base | OK | 0 criticals, 0 highs |
| horizontal-review-gpu  | OK | 0 criticals, 0 highs |
| horizontal-review-opt  | OK | 0 criticals, 1 high (carry, opt-in OFF) |

All 6 sub-reviews completed techniques A–G. None FAILED.

Pre-step `bash .claude/scripts/defect-registry.sh` reported 0 hits across
14 defect classes (D1–D15) before each sub-review.

## CRITICALs (deduplicated)

**None.**

## HIGHs (deduplicated)

- **H-opt-batched-lanczos-host-mode** (carry from R17, found by:
  horizontal-review-opt). `pdmrg_gpu_opt_impl.h:2870-2980`
  `batched_lanczos_eigensolver` retains host-resident `h_alpha`/
  `h_beta`/`&h_dot`/`&beta_val`/`&inv_norm`. Only fires when
  `set_use_batched_sweep(true)` — **default OFF**, never on default
  G1 baseline path. Same fix shape as D12. Defer until batched-sweep
  campaign needs it.

## MEDIUMs (deduplicated)

- **M-multi-gpu-local-kernels-dup** (found by: vertical-review-pdmrg,
  horizontal-review-gpu). pdmrg-multi-gpu redefines 3 file-local
  kernels (`setup_heff_ss_step3_ptrs`, `setup_lenv_step3_ptrs`,
  `setup_renv_step3_ptrs`) that are also defined in pdmrg-gpu.
  Different translation units → not ODR. Consolidation candidate for
  `common/batch_ptrs_kernels.h`.
- **M-multi-gpu-precompute-fused-mpo-host** (found by:
  vertical-review-pdmrg). `pdmrg_multi_gpu_impl.h:700-730` runs a
  host nested-loop + H2D for fused-MPO precompute. Once-per-set_mpo,
  not per-sweep — outside the "no host roundtrips per sweep" rule —
  but warrants a charter decision vs. -gpu's on-device build.
- **M-opt-pdmrg-single-site-graph-comment-stale** (new, found by:
  horizontal-review-opt). `pdmrg_gpu_opt_impl.h:2302-2311` comment
  claims Step 1/3 use stack-allocated `h_A[256]`/`h_B[256]`/
  `h_C[256]` but actual code uses device kernels
  (`setup_batch_ptrs_wd[_sparse]`, `setup_batch_ptrs_step3`). The
  rationale for disabling LANCZOS_GRAPH for single-site no longer
  holds. Either update comment or re-enable graph capture.
- **M1-opt-lanczos-init-D2H-sync** (carry from R17, found by:
  vertical-review-dmrg). dmrg-gpu-opt has explicit
  `hipStreamSynchronize` at lanczos init vs. -gpu's cleaner
  host-pointer-mode pattern. Cosmetic.

## NITs (deduplicated)

- **N-pdmrg-opt-graph-comment** (same site as
  M-opt-pdmrg-single-site-graph-comment-stale) — purely textual.
- **N-base-header-overstates-pointer-mode** (carry, found by:
  vertical-review-dmrg). dmrg-gpu-base header says "device-pointer
  mode" but only BLAS-1 Lanczos ops use it; GEMMs use host-stack.
- **N-pointer-mode-guard-return-discards** (carry, found by:
  horizontal-review-gpu). `pointer_mode_guard.h` discards
  rocblas_status returns from set_pointer_mode in dtor.

## FALSE POSITIVES (cross-review verified)

- **D8 host LAPACK gesvd** in `pdmrg-{gpu,gpu-opt,multi-gpu}` —
  filtered by improved registry awk (recognizes lwork=-1 workspace
  queries + `use_cpu_svd_` opt-in branches).
- **D13 -base apply_heff per-wp host loop** — registry now skips
  -base by charter (naive single-GEMM IS the baseline tier).
- **form_theta_with_V V upload** in pdmrg-gpu-base — sibling-
  consistent boundary state shuttle, not on hot path.
- **precompute_WW host build** in pdmrg-gpu-base — set_mpo time,
  outside sweep.

## SUMMARY VERDICT

- **Block GPU run / paper submission?** **NO.** 0 CRITICALs.
  The single HIGH is a carry-over on an opt-in code path
  (`set_use_batched_sweep(true)`) that is OFF by default and never
  fires on the G1 baseline. All MEDIUMs are non-blocking (cosmetic,
  charter-question, or set-once init paths).
- **Top-3 actions before next major event:**
  1. Update the stale `apply_heff_single_site` graph-capture comment
     in pdmrg-gpu-opt (or re-enable graph capture once verified safe
     post device-kernel port).
  2. Optionally consolidate the 3 multi-gpu local kernels into
     `common/batch_ptrs_kernels.h` — reduces D6 surface across
     variants.
  3. Schedule remote MI300X build verification (task #120) — this
     is the next blocking step before any benchmark campaign.
- **What was checked vs. last conformity review (R15,
  `reviews/conformity-20260501-round15.md`):**
  - R15 had 1 CRITICAL (round-7 D12 propagation gap) + 5 HIGHs
    (PhaseTimer panel propagation, Davidson syev, sparse-MPO,
    Step 3 host-batch). All fixed by R17 commits 0efe96d, 54f2fcf,
    and 12d02c5. Zero regressions.
  - R16/R17 had 1 CRITICAL (CR-D1 Davidson buffer aliasing in
    pdmrg-gpu-opt — round-8 propagation gap, 9 rounds undetected).
    Closed by 54f2fcf. The defect-class registry's new D14 + D15
    entries codify both CR-D1 and the dead-PhaseTimer pattern that
    spawned R17H1; future rounds will catch them in pre-commit.

The codebase is **READY for the MI300X G1 baseline campaign** modulo
remote build verification. The defect-class registry methodology
shift (R16 → R17) has stabilized output: this is the first round
since R7 with zero CRITICALs AND zero blocking HIGHs across all 6
sub-reviewers.
