# Full conformity review — round 15 (2026-05-01)

Post round-14 fix audit. Baseline = `reviews/conformity-20260501-round14.md`
at commit `f40140d` (after the 5deba6d round-14 commit).

## Charter proof — sub-review status

| Sub-review                | Status | Findings                              |
|---------------------------|--------|---------------------------------------|
| vertical-review-dmrg      | OK     | 0C / 2H / 0M                          |
| vertical-review-dmrg2     | OK     | 0C / 0H / 0M (5th consecutive clean)  |
| vertical-review-pdmrg     | OK     | 0C / 1H / 3M                          |
| horizontal-review-base    | OK     | 0C / 0H / 0M                          |
| horizontal-review-gpu     | OK     | 0C / 0H / 1M                          |
| horizontal-review-opt     | OK     | 0C / 2H / 1M (deduplicated to dmrg/dmrg2-opt panel-inert) |

All 6 sub-reviews completed all techniques.

**Headline**: round-15 surfaced **3 distinct HIGHs** plus several
MEDIUMs — most are technique-G axis-1 lonely-sibling propagation gaps.
Round-14's H-opt-pdmrg-phase-timer-prop fix was a textbook case of the
fix being applied to one -opt sibling but not the others. Round-15
catches the missed siblings (dmrg-gpu-opt + dmrg2-gpu-opt have the
panel surface but zero begin/end sites). Two HIGHs require larger
ports and are deferred to dedicated commits.

## CRITICALS — block GPU run / paper submission

(none)

## HIGHS (deduplicated)

- **[dmrg-gpu-opt + dmrg2-gpu-opt: PhaseTimer panel inert]**
  H1-opt-PhaseTimer-prop (found by: vertical-dmrg, horizontal-opt).
  Round-14 H-opt-pdmrg-phase-timer-prop fix was applied to
  pdmrg-gpu-opt only; dmrg-gpu-opt and dmrg2-gpu-opt have identical
  5-phase panel surfaces but ZERO `.begin/.end` sites in the body.
  M-skip-on-zero hides the symptom but doesn't repair instrumentation.
  **Fixed**: added `t_<phase>_.begin/end` at all 5 sites in both
  variants:
  - apply_heff (cache-hit early return + main path)
  - update_left_env / update_right_env
  - lanczos_eigensolver
  - svd_fallback / svd_split_fallback
  - block_davidson_eigensolver (uses shared t_lanczos_ name; multiple
    early-return paths each close the timer pair before returning;
    Lanczos fallback short-circuit avoids double-instrumentation).

- **[dmrg-gpu-opt: apply_heff/update_*_env] H2-opt-host-batch-pointer-
  roundtrip** (found by: vertical-dmrg). Per-call host construction of
  Scalar*[D*d] arrays + 3× hipMemcpyAsync H2D — direct CLAUDE.md "no
  host roundtrips per sweep" violation. -gpu sibling uses GPU
  `setup_batch_ptrs_*` kernels (defined in dmrg_gpu_impl.h but
  file-local). **DEFERRED** to a dedicated commit: requires moving 4-6
  templated kernels to `common/batch_ptrs_kernels.h` then refactoring
  the 3 dmrg-gpu-opt sites + verifying behavior. Risk-tradeoff:
  porting them along with the timer instrumentation in this commit
  doubles the change surface; a clean isolated port is safer.

- **[pdmrg-gpu-opt: block_davidson_eigensolver] H-opt-pdmrg-davidson-
  syev-host-LAPACK** (found by: vertical-pdmrg). Per-Davidson-iter
  host LAPACK syev call. Round-7 C2+H6 ported this on-device to
  dmrg-gpu-opt and dmrg2-gpu-opt via rocsolver_dsyevd; never
  propagated to pdmrg-gpu-opt. **DEFERRED** to a dedicated commit:
  requires careful workspace allocation and CGS sync correctness
  audit; not safe in the same commit as the timer changes.

## MEDIUMS (deduplicated)

- **[pdmrg-gpu: apply_heff_single_site] M-pdmrg-single-site-
  uninstrumented** (found by: horizontal-gpu). Same shape as
  apply_heff_two_site, including a cache-hit early-return at line
  1953. **Fixed**: added si==0 gated `.begin/.end` pair, including
  cache-hit path closure.

- **[pdmrg-gpu, pdmrg-gpu-opt: dead t_absorb_]** (found by:
  vertical-pdmrg). t_absorb_ instances allocated at ctor and reported
  but never `.begin/.end`'d. Round-14 multi-gpu fix added skip-on-
  zero to suppress noisy output; same applied in round-14 to all
  6 single-host variants. Confirming: skip-on-zero already in place
  per round-14 M-skip-on-zero-prop.

- **[pdmrg-multi-gpu: dead t_env_update_]** (found by: vertical-pdmrg).
  Same situation; skip-on-zero gates the noisy output. The timer
  exists for forward-instrumentation. Acceptable as documented.

- **[pdmrg-gpu-opt: per-iter Davidson CGS hipStreamSynchronize]**
  (found by: vertical-pdmrg). Folds into the Davidson syev port —
  same DEFERRED commit.

## NITS

(none new)

## FALSE POSITIVES (cross-review verified)

- **dmrg-gpu-opt cache-hit early return missing t_apply_heff_.end()**
  (round-14 H3 forward-looking note): no longer false-positive — H1
  propagation ABOVE adds the matching `.end()` at the cache-hit path.
- **dmrg2-gpu-opt similar** (same pattern verified fixed in this
  commit).
- **pdmrg-gpu-opt cache-hit early return**: round-14 fixed; verified
  intact.

## SUMMARY VERDICT

- **Block GPU run / paper submission?** **NO** for the panel-inert
  HIGH (closed in this commit) and the apply_heff_single_site MEDIUM.
  **YES, partially** for the two deferred HIGHs (H2-opt host-batch +
  Davidson-syev) — they violate the "no host roundtrips per sweep"
  rule. Recommend: do the dedicated ports in two follow-up commits
  before the next major event.

- **Top-3 actions before next major event:**
  1. **Commit-1 (this round)**: PhaseTimer instrumentation in
     dmrg-gpu-opt + dmrg2-gpu-opt + pdmrg-gpu single-site. Closed.
  2. **Commit-2 (deferred)**: Move setup_batch_ptrs kernels to
     `common/batch_ptrs_kernels.h`, refactor dmrg-gpu-opt's
     apply_heff + update_*_env to use them. Eliminates 3 host
     roundtrips per sweep on default code path.
  3. **Commit-3 (deferred)**: Port pdmrg-gpu-opt
     block_davidson_eigensolver host LAPACK syev →
     rocsolver_dsyevd. Eliminates per-Davidson-iter host roundtrip;
     also folds in the M-pdmrg-CGS-sync medium.

- **What was checked vs. round 14 (5deba6d):**
  All round-14 fixes preserved (H1-base scope rescope, H-pdmrg cache-
  hit-leak fix, H-opt-pdmrg-phase-timer-prop in pdmrg-gpu-opt,
  M-skip-on-zero across 6 variants, M-opt-pdmrg-pointer-mode,
  M-opt-svd_fallback-docstring, M-multi-n_recal-doc, dmrg2-gpu-base
  d_WW_ comment, dmrg-gpu set_quiet stub comment). Round-15 findings
  are net-new technique-G axis-1 propagation gaps — round-14 fixed
  one variant of each defect class but didn't sweep the siblings.

**Pattern recognition (round 15)**: round 14's fix-one-variant-don't-
sweep recurs. The PhaseTimer panel was wired in pdmrg-gpu (round 13),
pdmrg-gpu-opt (round 14), and now dmrg/dmrg2-gpu-opt (round 15) — but
each round only patched the variant the reviewer flagged, not the
known-identical siblings. Memory-resident technique-G checklist needs
a stronger axiom: when fixing a defect class in variant X, mechanically
search all sibling variants for the same defect class BEFORE the
commit, not after.

Round-15 disposition: 1 HIGH closed in this commit (panel propagation
to dmrg/dmrg2-gpu-opt). 2 HIGHs deferred to dedicated commits
(H2-opt host-batch port, Davidson syev port). Several MEDIUMs (timer
hygiene) closed alongside. Single-host + multi-gpu + all 3 tiers
clean for net-new defects in this round's commit; the two deferred
HIGHs remain known-deferred work blocking the next GPU run.
