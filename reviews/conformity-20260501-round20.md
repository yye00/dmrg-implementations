# Full conformity review — round 20 (2026-05-01, HEAD f650466)

Second confidence re-run before MI300X G1 baseline allocation. Six
fully independent sub-reviewers re-audited from scratch. Each ran the
defect-class registry as pre-step (TOTAL HITS: 0).

## Charter proof — sub-review status

| Sub-review | Status | Findings |
|---|---|---|
| vertical-review-dmrg   | OK | 0 criticals, 0 highs |
| vertical-review-dmrg2  | OK | 0 criticals, 0 highs (10th consecutive clean round) |
| vertical-review-pdmrg  | OK | 0 criticals, 0 new highs |
| horizontal-review-base | OK | 0 criticals, 0 highs (cleanest tier) |
| horizontal-review-gpu  | OK | 0 criticals, 0 new highs |
| horizontal-review-opt  | OK | 0 criticals, 0 new highs |

All 6 sub-reviews completed techniques A–G. None FAILED.

## CRITICALs (deduplicated)

**None.**

## HIGHs (deduplicated)

- **H-opt-batched-lanczos-host-mode** (carry from R17/R18/R19, found
  by: horizontal-review-opt). Same finding. Opt-in
  `set_use_batched_sweep(true)` / `set_use_chebyshev(true)`, both
  default OFF. Never on G1 baseline.

## MEDIUMs (deduplicated)

- **M-multi-gpu-step3-old-kernel-dead** (NEW, found by:
  vertical-review-pdmrg). The R19H19 fix added
  `setup_heff_ss_step3_full_ptrs` to pdmrg_multi_gpu_impl.h but the
  old per-wp `setup_heff_ss_step3_ptrs` (impl :26-39) is now
  **never launched**. Quick cleanup; registry would catch it via a
  D-class for unused kernels, future R21 scope.
- **M-opt-pdmrg-ss-step3-not-r3f1** (NEW, found by:
  vertical-review-pdmrg). pdmrg-gpu-opt SS Step 3 d>2 fallback could
  adopt R3-F1 collapse for parity (graph-capture-disabled path so
  it's safe). Not blocking.
- **M-opt-pdmrg-graph-comment-stale** (carry from R18/R19, found by:
  horizontal-review-opt). `pdmrg_gpu_opt_impl.h:2302-2311` stale
  comment claims host-stack pointers; actual code uses device kernels.
- **M-multi-gpu-local-kernels-dup** (carry, NOT re-flagged in R20 —
  vertical-review-pdmrg notes the kernel addition for H19 widens the
  consolidation surface, R20 polish).
- **M1-opt-lanczos-init-D2H-sync** (carry, found by: vertical-review-dmrg).

## NITs (deduplicated)

- **N-d13-awk-function-scope** (NEW, found by: vertical-review-dmrg).
  D13 awk's `apply_heff` substring match exits state too early via
  header comments / other function references. Doesn't hide a real
  defect today, but a future genuine in-body wp-loop could escape
  detection. Tighten before R21.
- **N-r17-base-prose-drift** (NEW, found by: horizontal-review-opt).
  Minor doc drift on R3-F1 -opt commitment phrasing.
- **N-base-header-overstates-pointer-mode** (carry).
- **N-pointer-mode-guard-return-discards** (carry).

## VERIFIED FALSE POSITIVES (cataloged to prevent re-discovery)

- **dmrg-gpu-opt + dmrg2-gpu-opt apply_heff Step 3 per-wp loop**
  (cataloged by horizontal-review-opt). This is **INTENTIONAL** per
  commit `98ca518` (R3-F1, 2026-04-10) message:
  > "Targets only the plain -gpu implementations as directed;
  > -gpu-opt and -gpu-base variants are untouched."
  The -opt's strided-batched/per-wp design (commit `bd4d09c`,
  2026-03-31) was explicitly preserved. R11 review documented this
  pre-existing design choice as an INTENTIONAL ablation: per-wp loop
  with cache-contention diagnostic value.
- **D7+D8 host LAPACK gesvd** in pdmrg-{gpu,gpu-opt,multi-gpu} —
  filtered (use_cpu_svd_ opt-in / lwork=-1 workspace queries).
- **D13 -base apply_heff per-wp host loop** — registry whitelists
  -base by charter.
- **rlbfgs-gpu / radam-gpu** apply_heff per-wp loops with
  gemm_batched — out-of-charter variants, registry doesn't scan
  them. Informational only.

## R19H19 FIX VERIFICATION (triangulated)

Three independent sub-reviewers verified the R19H19 fix on
pdmrg-multi-gpu apply_heff_single_site Step 3 in `cafd628`:

| Audit dimension | vertical-pdmrg | horizontal-gpu | horizontal-opt |
|---|---|---|---|
| New kernel mirrors pdmrg-gpu :49-59         | ✓ | ✓ (byte-for-byte diff) | ✓ |
| dev.d_ones_D allocated/initialized/freed    | ✓ | ✓ (impl :370/:534)     | n/a |
| T1 sizing ≥ D·slice_stride headroom factor d| ✓ | ✓                      | n/a |
| t_apply_heff_.end fires after gemv reduce   | ✓ | ✓ (impl :1771)         | n/a |
| Two-site Step 3 didn't regress              | ✓ | ✓                      | n/a |
| Mathematically equivalent (gemv = sum-over-wp) | ✓ | ✓                   | n/a |
| Cross-sibling check (other apply_heff paths) | ✓ | ✓ (no per-wp remain) | ✓  |

The widened D13 registry detector was sanity-tested independently by
horizontal-review-gpu and confirmed to flag the original H19 pattern
without false-positives in the fixed code.

## SUMMARY VERDICT

- **Block GPU run / paper submission?** **NO.** 0 CRITICALs.
  The single carry HIGH is opt-in OFF-by-default. R19H19 fix
  triangulated clean.

- **Top-3 actions before next major event:**
  1. **Task #120 — remote MI300X build verification.** Static review
     is exhausted; everything else needs the toolchain.
  2. (Optional polish, R21 scope) Remove dead
     `setup_heff_ss_step3_ptrs` from pdmrg-multi-gpu (~5 lines).
  3. (Optional polish, R21 scope) Tighten D13 awk function-scope
     boundary detection.

- **What was checked vs. R19 baseline (`reviews/conformity-20260501-round19.md`):**
  - R19 had 1 NEW HIGH (H19) found AND fixed in same round.
  - R20 found 0 NEW HIGHs and confirmed the H19 fix is correct on 7
    audit dimensions across 3 independent sub-reviewers.
  - 2 new MEDIUMs (dead-kernel + Step 3 parity), neither blocking.
  - 1 verified false positive cataloged so R21 doesn't re-discover.
  - Zero regressions across all R15→R19 fixes.

- **Methodology validation:**
  - **R20 paid for itself in a different way than R19**: R19 found a
    real defect; R20 *confirmed* the absence of remaining defects of
    the same class via independent triangulation. Both are valuable;
    they differ in what they prove.
  - The dmrg2 family has gone **10 consecutive rounds without a
    finding**. This is a positive signal — not "we stopped looking,"
    but "the family has structurally stabilized." Compare to pdmrg
    family which is still surfacing polish-grade MEDIUMs.
  - The R20 false-positive catalog is a new artifact: documenting
    *why* something is intentional prevents the next round from
    flagging it as a regression.

The codebase is **READY for the MI300X G1 baseline campaign.**
Pre-allocation static review is exhausted. The only blocking gate is
remote MI300X build verification (task #120).

**Recommend: allocate the GPU window.** No further code changes
should land before task #120. If task #120 surfaces a build error,
fix and re-run R21 + registry; otherwise proceed directly to G1
baseline runs.
