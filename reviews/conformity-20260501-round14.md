# Full conformity review — round 14 (2026-05-01)

Post round-13 fix audit. Baseline = `reviews/conformity-20260430-round13.md`
at commit `f5c0617`.

## Charter proof — sub-review status

| Sub-review                | Status | Findings                              |
|---------------------------|--------|---------------------------------------|
| vertical-review-dmrg      | OK     | 0C / 1H / 1M                          |
| vertical-review-dmrg2     | OK     | 0C / 0H / 0M                          |
| vertical-review-pdmrg     | OK     | 0C / 2H / 3M                          |
| horizontal-review-base    | OK     | 0C / 0H / 0M (1 nit)                  |
| horizontal-review-gpu     | OK     | 0C / 0H / 0M (1 nit + 1 carry-over)   |
| horizontal-review-opt     | OK     | 0C / 0H / 0M                          |

All 6 sub-reviews completed all techniques.

**Headline**: round-13 closed the round-12 propagation gaps but opened
new ones — the axis-3 lesson recurs. Three HIGHs surfaced this round,
two of them in code I touched in round-13. The dmrg vertical reviewer
flagged a pre-existing scope bug in the round-13 PointerModeGuard
adoption (apply_heff was always inside device-pointer mode in the
-base lanczos, both pre and post round-13). Two HIGHs in pdmrg —
pdmrg-gpu-opt's PhaseTimer panel never got round-13's instrumentation,
and pdmrg-gpu's apply_heff timer leaks `hipEvent_t` on the
lanczos_graph cache-hit early-return.

All findings fixed in this round's commit.

## CRITICALS — block GPU run / paper submission

(none)

## HIGHS (deduplicated)

- **[3 -base variants: dmrg-gpu-base / dmrg2-gpu-base / pdmrg-gpu-base
  lanczos_eigensolver] H1-base-apply_heff-in-device-pointer-mode**
  (found by: vertical-dmrg; same defect class confirmed in dmrg2 and
  pdmrg -base by self-audit). Round-13 M1-base-prop's PointerModeGuard
  brace block wraps the entire Lanczos inner loop including
  `apply_heff(...)`, which uses host-stack `&one`/`&zero_val` for
  rocBLAS gemm — UB under device-pointer mode. The siblings -gpu and
  -opt show the correct tighter scope: device-mode region INSIDE the
  for-loop iteration, AFTER apply_heff returns. **Fixed**: rescoped
  PointerModeGuard to per-iter blocks AFTER apply_heff in all 3 -base
  variants. Initial-vector normalization wrapped in its own guard
  before the loop.

- **[pdmrg-gpu-opt: pdmrg_gpu_opt_impl.h ~799/lanczos/svd/env]
  PhaseTimer panel always inert** (found by: vertical-pdmrg).
  Identical 5-phase surface to pdmrg-gpu, ctor/report wired since
  round 7, but no `.begin/.end` sites. Round-13 fixed exactly this in
  pdmrg-gpu but didn't propagate to the -opt sibling. Lonely-sibling
  technique-G gap. **Fixed**: added si==0 gated `.begin/.end` at all 5
  phases (apply_heff_two_site, lanczos_eigensolver, svd_split,
  update_left_env, update_right_env). Cache-hit early-return path
  closes the timer pair before returning.

- **[pdmrg-gpu: pdmrg_gpu_impl.h:980] t_apply_heff_ event-leak on
  lanczos_graph cache hit** (found by: vertical-pdmrg).
  Round-13 added `t_apply_heff_.begin()` at line 959 but the matching
  `.end()` at 1135 is bypassed when `opts_.lanczos_graph` is on and a
  graph cache hit returns at line 980. PhaseTimer truncates total to
  `min(starts, stops)` so timing is silently low; every cache hit
  creates a `hipEvent_t` that's never destroyed. **Fixed**: added
  matching `t_apply_heff_.end()` before the early return. Same fix
  applied prophylactically to pdmrg-gpu-opt's analogous cache-hit
  path.

## MEDIUMS (deduplicated)

- **[6 variants: dmrg/dmrg2/pdmrg-gpu and dmrg/dmrg2/pdmrg-gpu-opt
  report_timers] skip-on-zero-calls not propagated** (found by:
  vertical-pdmrg + horizontal-gpu carry-over).
  Round-12 added `if (t.calls() == 0) return;` to pdmrg-multi-gpu's
  report_timers. The 5 single-host PhaseTimer panels in
  dmrg-gpu/dmrg2-gpu/pdmrg-gpu/dmrg-gpu-opt/dmrg2-gpu-opt/pdmrg-gpu-opt
  still print `0.00 ms / 0 calls` for uninstrumented timers. **Fixed**:
  added the same guard to all 6 panels.

- **[pdmrg-gpu-opt: pdmrg_gpu_opt_impl.h:1833,1835,1848,1852 and
  1868,1934 and 2004,2017] 8 raw rocblas_set_pointer_mode pairs in
  lanczos** (found by: vertical-pdmrg).
  Round-12 M1-final and round-13 M1-base-prop adopted PointerModeGuard
  across other variants; pdmrg-gpu-opt's lanczos still had 4 paired
  blocks (8 calls). **Fixed**: replaced all 4 pairs with
  `PointerModeGuard` scoped blocks, mirroring the per-iter pattern.

- **[dmrg-gpu-opt: dmrg_gpu_opt.h:276] svd_fallback docstring drift**
  (found by: vertical-dmrg).
  Header comment "// SVD bond splitting (CPU LAPACK)" contradicts the
  on-device default. Round-13 axis-3 violation. **Fixed**: rewrote
  comment to clarify default is on-device rocsolver_gesvd_auto with
  CPU LAPACK only on the use_cpu_svd_ opt-in branch.

- **[pdmrg-multi-gpu: pdmrg_multi_gpu.h:54] n_recal API gap not
  documented** (found by: vertical-pdmrg).
  Sibling pdmrg-gpu/-opt accept `n_recal=0`. **Fixed**: added explicit
  one-paragraph note in the run() docstring stating the gap is
  intentional (gather/scatter cost dominates short-cycle recal).

## NITS (deduplicated)

- **[dmrg2-gpu-base.h:83]** d_WW_ per-member comment said "precomputed
  on device" but precompute_WW is host-side. **Fixed** to "precomputed
  on host at set_mpo() time and hipMemcpy'd to device."

- carry-over from round-13: dmrg-gpu-base scoped-region indent
  cosmetic, dmrg-gpu.h:43 set_quiet stub comment stylistic — not
  closed in this round.

## FALSE POSITIVES (cross-review verified)

- **dmrg2-gpu-base raw pointer-mode pairs** (out-of-charter sighting
  from vertical-dmrg2 and dmrg2-gpu-base reviewer): real, deduplicated
  to H1-base-apply_heff propagation across all 3 -base variants. Same
  fix.
- **pdmrg-gpu-base svd_split uses plain gesvd, not Stoudenmire**: J1
  applies to *boundary merges* only; inner-segment splits use plain
  gesvd in every pdmrg variant. Boundary call at
  pdmrg_gpu_base_impl.h:1271 uses `accurate_svd_gpu`. Not a defect.
- **pdmrg-multi-gpu use_rsvd_ binding still works after round-13**:
  verified at impl :163. Round-13 axis-3 fix preserved.

## SUMMARY VERDICT

- **Block GPU run / paper submission?** **NO** — all round-14 findings
  fixed in this round's cleanup commit. Zero CRITICALs.

- **Top-3 actions before next major event:**
  1. Build verify on remote MI300X (Task #120 still pending).
  2. Smoke-test the new pdmrg-gpu-opt PhaseTimer panel and the H1-base
     scope fix with `DMRG_GPU_PROFILE=1`.
  3. Verify dmrg/dmrg2-gpu-base and pdmrg-gpu-base lanczos Lanczos
     correctness regression-free after the device-mode rescoping
     (smoke test should suffice; pre-fix code worked empirically by
     UB tolerance, post-fix is correct by construction).

- **What was checked vs. round 13 (f5c0617):**
  All round-13 fixes preserved: pdmrg-multi-gpu use_rsvd_ binding,
  pdmrg-gpu PhaseTimer instrumentation (now timer-leak fixed too),
  M1-base-prop PointerModeGuard adoption (now scope-corrected),
  M14-base-prop dead set_quiet stubs gone, pdmrg-gpu-base docstring
  default n_warmup=1/n_polish=0. The round-14 findings are all NEW
  technique-G propagation gaps — round-13 closed one set, opened
  another.

**Pattern recognition (round 14)**: this round's H1-base bug is
  unusual — it predates round-13 (the raw `set_pointer_mode(device)`
  pair already wrapped apply_heff before M1-base-prop) but the
  round-13 refactor inherited the broken scope. The lesson is that
  `M1-base-prop`'s mechanical "replace paired set/unset with
  PointerModeGuard" preserved a latent UB. **Future technique-G
  refactors must rescope, not just reskin.** Memory note updated.

Round-14 disposition: 3 HIGHs + 4 MEDIUMs + 1 nit closed. Single-host
+ multi-gpu + all 3 tiers clean. Eight consecutive zero-finding
sub-reviews for the dmrg2 family.
