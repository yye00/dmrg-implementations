# Full conformity review — round 13 (2026-04-30)

Post round-12 medium cleanup (`0b9fccf`) audit. Baseline =
`reviews/conformity-20260430-round12.md` at commit `8b7a68e`.

## Charter proof — sub-review status

| Sub-review                | Status | Findings                              |
|---------------------------|--------|---------------------------------------|
| vertical-review-dmrg      | OK     | 0C / 0H / 0M                          |
| vertical-review-dmrg2     | OK     | 0C / 0H / 0M (+ dmrg2-base sighting)  |
| vertical-review-pdmrg     | OK     | 0C / 1H / 2M                          |
| horizontal-review-base    | OK     | 0C / 0H / 2M                          |
| horizontal-review-gpu     | OK     | 0C / 2H                               |
| horizontal-review-opt     | OK     | 0C / 0H / 0M (1 nit)                  |

All 6 sub-reviews completed techniques A, B, C, E, F, G + self-audit.
Technique D again skipped (no ROCm headers locally).

**Headline**: round-13 surfaced a defect class I introduced in
`0b9fccf` (the round-12 medium cleanup): pdmrg-multi-gpu's docstring
claims `opts_.rsvd` is honoured, but I forgot to bind
`use_rsvd_ = opts_.rsvd`. Plus pdmrg-gpu's PhaseTimer panel was always
inert (init+report wired since round-7 but no `.begin/.end` sites
ever existed) — it became a lonely sibling once round-12 wired
pdmrg-multi-gpu's analogous timers. M1-base-prop and M14-base-prop
caught the round-12 consolidation never reaching the -base tier.

All findings fixed in this round (see commit `<round-13>`).

## CRITICALS — block GPU run / paper submission

(none)

## HIGHS (deduplicated, sorted by file)

- **[pdmrg-multi-gpu: pdmrg_multi_gpu_impl.h:151] use_rsvd_ never
  bound to opts_.rsvd** (found by: vertical-pdmrg AND horizontal-gpu
  — same defect, two reviewers)
  Round-12 0b9fccf docstring promises "only `rsvd` and `profile` are
  honoured here" but `use_rsvd_` is initialized to false and never
  assigned from `opts_.rsvd`. `DMRG_GPU_OPT_RSVD=1` env-var is a
  silent no-op. Same defect class as round-10 M-opt-rsvd-env in
  pdmrg-gpu-opt; technique-G miss when adding the docstring claim.
  **Fixed**: `use_rsvd_ = opts_.rsvd` after `opts_.load_from_env()`.

- **[pdmrg-gpu: pdmrg_gpu_impl.h around 947, 1142, 1219, 1325, 1575]
  PhaseTimer panel always inert** (found by: horizontal-gpu)
  init_timers + report_timers wired since round-7 H3, but no
  `.begin/.end` callsites in apply_heff_two_site / lanczos /
  svd_split / update_left_env / update_right_env. Output panel always
  `0.00 ms / 0 calls`. Round-12 wired analogous timers in
  pdmrg-multi-gpu, making pdmrg-gpu the lonely sibling. **Fixed**:
  added `if (si == 0) t_<phase>_.{begin,end}(streams_[si])` at all 5
  sites. Single-segment gating mirrors pdmrg-multi-gpu —
  `std::vector::push_back` is not thread-safe across parallel segments.

## MEDIUMS (deduplicated)

- **[3 -base variants: dmrg-gpu-base, dmrg2-gpu-base, pdmrg-gpu-base]
  M1-base-prop: PointerModeGuard not adopted** (found by:
  horizontal-base; same sighting from vertical-dmrg2 reviewer)
  Round-12 0b9fccf consolidated 3 inline guards into the shared
  `common/pointer_mode_guard.h::PointerModeGuard`. The -base tier was
  out of scope for that commit and still uses raw paired
  `set_pointer_mode(device)…(host)` calls. ROCBLAS_CHECK throws inside
  the block leak device-mode into the next caller. **Fixed**: 6 raw
  pointer-mode pairs across the 3 -base variants replaced with
  `PointerModeGuard` scoped blocks. include added.

- **[3 -base variants: dmrg-gpu-base.h, dmrg2-gpu-base.h,
  pdmrg-gpu-base.h] M14-base-prop: dead set_quiet stubs** (found by:
  horizontal-base)
  None of the -base test drivers call `.set_quiet(...)`. Round-12
  removed the same stubs from -opt headers using the same criterion.
  **Fixed**: dropped all 3.

- **[pdmrg-multi-gpu: pdmrg_multi_gpu.h:53] n_recal API gap** (found
  by: vertical-pdmrg)
  pdmrg-gpu and pdmrg-gpu-opt accept `int n_recal = 0` for periodic
  full-chain recalibration. multi-gpu drops it. Out-of-scope per the
  multi-device charter; accepted as documented gap. **Action**: noted
  in the docstring's GpuOpts disclaimer; no signature change.

- **[pdmrg-multi-gpu: pdmrg_multi_gpu.h] dead t_absorb_ /
  t_env_update_ timers** (found by: vertical-pdmrg)
  PhaseTimer instances init+reported but never `.begin/.end`'d. Output
  prints 0.00 ms / 0 calls. **Fixed**: report_timers now skips phases
  with `calls() == 0`. Forward-compat scaffolding preserved without
  noisy output.

## NITS (deduplicated)

- **[pdmrg-multi-gpu: pdmrg_multi_gpu.h:24-25]** docstring sub-sentence
  about `set_use_davidson` — pdmrg-multi-gpu has no Davidson path.
  **Fixed**: reworded to drop the misleading cross-reference.

- **[pdmrg-gpu-base.h:42]** "default 1 each" inconsistent with
  signature `n_warmup=1, n_polish=0`. **Fixed**.

- carry-over from round-12: 2 dmrg2 cosmetic comment nits
  (lanczos_use_1site_ atomicity, stale ws.d_T1 in form_theta_with_V).
  Cosmetic only; deferred.

## FALSE POSITIVES (cross-review verified)

- **dmrg2-gpu-base raw pointer-mode toggles** (out-of-charter sighting
  from vertical-dmrg2 reviewer): yes, real, but this is precisely
  M1-base-prop above — same defect tagged twice. Deduplicated to one
  fix.
- **pdmrg-gpu-base svd_split uses plain gesvd, not Stoudenmire**: J1
  applies to *boundary merges*, not inner-segment splits. Boundary
  call at pdmrg_gpu_base_impl.h:1267 uses accurate_svd_gpu. Not a
  defect.

## SUMMARY VERDICT

- **Block GPU run / paper submission?** **NO** — all round-13
  findings (2 HIGHs + 4 MEDIUMs + 2 nits) are fixed in this round's
  cleanup commit. Zero CRITICALs.

- **Top-3 actions before next major event:**
  1. Build verify on remote MI300X (Task #120).
  2. Run a smoke-test of the new pdmrg-gpu and pdmrg-multi-gpu
     PhaseTimer panels with `DMRG_GPU_PROFILE=1` to verify the timer
     output makes sense.
  3. Run a small ablation with `DMRG_GPU_OPT_RSVD=1` against
     pdmrg-multi-gpu to verify the new env-var binding actually
     enables RSVD in the multi-device path.

- **What was checked vs. round 12 (8b7a68e):**
  All round-12 fixes (PDMRG-rules-2026-04-15 lock in pdmrg-multi-gpu,
  pdmrg-gpu-opt ctor-time lanczos_graph gate, 8 raw pointer-mode
  toggles → PointerModeGuard, dead-buffer cleanup,
  initialize_mps_product / _neel, PhaseTimer panel scaffolding) ARE
  PRESERVED in this round. The round-13 findings are all
  technique-G propagation gaps:
  - I introduced a docstring promise (`opts_.rsvd` honoured) without
    the corresponding code binding — round-12 self-audit didn't catch
    this because the docstring was added in the same commit as the
    timer panel and the audit never re-read the resulting file
    against its own claim.
  - Round-12 cleanup scoped only -gpu/-opt; -base was out of scope,
    leaving M1-base-prop + M14-base-prop as obvious next-tier work.
  - pdmrg-gpu's PhaseTimer dead since round-7 — was a latent issue
    that became a HIGH only because round-12 made multi-gpu's panel
    work, exposing the lonely sibling.

**Pattern recognition (round 13)**: the round-12 lesson on "two
half-fix asymmetries" generalizes — a docstring promise IS a half-fix.
Adding "X is honoured" to a docstring without verifying X is wired
creates a false claim, which is functionally equivalent to a setter
without a ctor gate. Self-audit must, for any docstring promise added
in the same commit, grep the corresponding code path. Memory note
updated.

Round-13 disposition: all findings closed in `<round-13-fix-commit>`.
Single-host variants (dmrg, dmrg2, pdmrg-gpu, pdmrg-gpu-opt) AND
pdmrg-multi-gpu now clean. -base tier consolidation also closed
(round-12 pointer-mode RAII finally reaches all tiers).
