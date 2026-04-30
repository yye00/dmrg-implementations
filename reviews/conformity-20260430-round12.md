# Full conformity review — round 12 (2026-04-30)

Post-cleanup re-audit after commit `c3d3e50` (round-11 spotless dead-buffer
removal: -311/+2 across 11 files). Baseline = round-11 report
`reviews/conformity-20260430-round11.md` at commit `2d55d90`.

## Charter proof — sub-review status

| Sub-review                | Status | Findings                                       |
|---------------------------|--------|------------------------------------------------|
| vertical-review-dmrg      | OK     | 0C / 0H / 0M (net-new) + 1 nit + 3 carry-over  |
| vertical-review-dmrg2     | OK     | 0C / 0H / 0M + 2 nits                          |
| vertical-review-pdmrg     | OK     | 1C / 2H / 0M (all in pdmrg-multi-gpu)          |
| horizontal-review-base    | OK     | 0C / 0H / 0M                                   |
| horizontal-review-gpu     | OK     | 0C / 1H / 5M (all in pdmrg-multi-gpu)          |
| horizontal-review-opt     | OK     | 0C / 1H / 0M                                   |

All 6 sub-reviews completed all techniques (A, B, C, E, F, G + self-audit).
Technique D (clangd diagnostics) skipped on every run — no ROCm headers
locally; technique A subsumes the dead-symbol channel.

**Headline:** the `dmrg`, `dmrg2`, `pdmrg-gpu`, `pdmrg-gpu-opt` single-host
variants are CLEAN. All net-new findings concentrate in two siblings that
lagged on technique-G propagation: **`pdmrg-multi-gpu`** (1C + 2H + 5M)
and **`pdmrg-gpu-opt`** (1H, missing ctor gate that dmrg-gpu-opt and
dmrg2-gpu-opt have).

## CRITICALS (deduplicated, sorted by file)

- **[pdmrg-multi-gpu: pdmrg_multi_gpu.cpp + pdmrg_multi_gpu.h]
  PDMRG_RULES_LOCK_VIOLATION** (found by: vertical-pdmrg)
  The 2026-04-15 CLAUDE.md PDMRG-rules lock is binding for ALL pdmrg
  variants (pdmrg, pdmrg-gpu, pdmrg-gpu-opt, pdmrg-multi-gpu) but never
  propagated to pdmrg-multi-gpu. Specific violations:
  - default `n_warmup = 3` (rule: ≤ 2)
  - hard-coded `n_polish = 10` somewhere on the run path (rule: ≤ 2)
  - polish uses two-site `sweep_LR_full` / `sweep_RL_full` (rule:
    `_1site` only)
  - no `--polish N` CLI argument (rule: explicit pass-through required)
  This is the round-8 G-class failure mode (single-fix in primary, never
  propagated to lonely sibling) repeating for a project lock instead of
  a code defect.

## HIGHS (deduplicated)

- **[pdmrg-gpu-opt: src/pdmrg_gpu_opt_impl.h ctor]
  MISSING_CTOR_LANCZOS_GRAPH_GATE** (found by: horizontal-opt)
  bc3fcd0 added the symmetric `set_use_davidson` runtime setter to
  dmrg-gpu-opt (matching dmrg2-gpu-opt and pdmrg-gpu-opt). Both
  dmrg-gpu-opt and dmrg2-gpu-opt also gate at construction:
  `if (opts_.lanczos_graph && use_davidson_) { disable + warn }`.
  pdmrg-gpu-opt has the runtime setter but NOT the ctor-time gate, so a
  user setting `PDMRG_GPU_OPT_LANCZOS_GRAPH=1` with default Davidson
  (use_davidson_ = true per J2 lock) hits the variable-output-pointer
  vs graph-capture incompatibility that the setter's own comment warns
  about (Davidson calls `apply_heff_two_site` with `AV + j*dim`).
  Same defect class as M-opt-davidson-toggle round-11; one sibling
  still missing the ctor half of the fix.

- **[pdmrg-multi-gpu: src/pdmrg_multi_gpu.h]
  POINTER_MODE_RAW_TOGGLES** (found by: horizontal-gpu)
  H7-class defect: 8 raw `rocblas_set_pointer_mode(...HOST/DEVICE)`
  calls with no `PointerModeGuard` RAII wrapper. Round-7 H7 ported the
  guard into dmrg2-gpu and pdmrg-gpu; pdmrg-multi-gpu was missed.
  Exception thrown between the HOST→DEVICE→HOST bookend leaks
  device-pointer-mode into the next caller — silent corruption hazard
  on the next sweep boundary.

- **[pdmrg-multi-gpu: dead h_batch_*_pinned and h_rsvd_B / h_rsvd_U_small]
  ROUND_11_CLEANUP_NOT_PROPAGATED** (found by: vertical-pdmrg,
  horizontal-gpu — **strictest = HIGH**)
  c3d3e50 removed `h_batch_{A,B,C}_pinned`, `h_rsvd_B`, `h_rsvd_U_small`
  from pdmrg-gpu / pdmrg-gpu-opt. Same buffers persist in
  pdmrg-multi-gpu — declared, resized in ctor, never read, never freed.
  Mirrors the M3 / M-opt-rsvd-env defect class.

## MEDIUMS (deduplicated)

- **[pdmrg-multi-gpu: missing initialize_mps_product / initialize_mps_neel]**
  (found by: horizontal-gpu) — round-7 H3 added these to pdmrg-gpu;
  pdmrg-multi-gpu only exposes `initialize_mps_random`. API parity gap.

- **[pdmrg-multi-gpu: copy_boundary_mps_to_device + d_boundary_staging
  dead]** (found by: horizontal-gpu) — declared, allocated, freed; no
  reads on the live code path.

- **[pdmrg-multi-gpu: 5 of 7 GpuOpts toggles silently ignored]**
  (found by: horizontal-gpu) — header claims "Extends pdmrg-gpu" but
  the run path inspects only `opts_.use_davidson` and `opts_.rsvd`. The
  others (sparse_mpo, fused_mpo, lanczos_graph, profile, accurate_svd)
  are no-ops. False advertising of capabilities.

- **[pdmrg-multi-gpu: zero PhaseTimer instrumentation]**
  (found by: horizontal-gpu) — every other variant has the standard
  `t_lanczos_ / t_apply_heff_ / t_svd_ / t_absorb_ / t_env_update_`
  panel. Multi-GPU performance regressions cannot be diagnosed
  without these.

- **[carry-over from round-11: DmrgPointerModeGuard not migrated to
  shared header]** (found by: vertical-dmrg) — three -gpu/-opt variants
  still inline-define the guard. M1 consolidation candidate.

- **[carry-over: set_quiet no-op in -opt variants]** (found by:
  vertical-dmrg) — round-11 M14 removed the no-op declarations from
  -opt; one stub still in dmrg-gpu-opt.h. Cosmetic.

- **[carry-over: host-resident Lanczos α/β in -opt fallback]**
  (found by: vertical-dmrg) — known limitation, not regression.

## NITS (deduplicated)

- vertical-dmrg: 1 cosmetic comment alignment in dmrg-gpu-opt.h
- vertical-dmrg2: 2 (lanczos_use_1site_ atomicity comment, stale
  `ws.d_T1` comment in `form_theta_with_V`)
- horizontal-base: 0 net-new (carry-over: same 2 from dmrg2 as those
  files are shared)

## FALSE POSITIVES (cross-review verified)

- **[pdmrg-gpu-opt: d_dav_work_ aliasing different from dmrg-gpu-opt]**
  pdmrg-opt has its own residual+overlap layout that is correctly
  sized (round-9 H-new1-pdmrg-opt). horizontal-opt verified.
- **[h_svd_A_/U_/Vh_/work_/S_ in dmrg2-gpu-opt look dead]**
  These ARE alive in the use_cpu_svd_ fallback path (impl 1263-1283).
  Cleanup correctly removed only `h_svd_tmp_`.
- **[d_svd_work_ in dmrg-gpu looks dead]**
  Used at impl 1387, 1392, 1402, 1434, 1438, 1449 in svd_split.
  Cleanup correctly removed only the host h_svd_*.
- **[missing env_update_pending_ in pdmrg-gpu-opt]**
  Per-segment streams use a different sync pattern. N-A.

## SUMMARY VERDICT

- **Block GPU run / paper submission?**
  **YES** — 1 CRITICAL (pdmrg-multi-gpu PDMRG rules violation) and
  3 HIGHs that all map to two lonely-sibling propagation gaps.
- **Top-3 actions before next major event:**
  1. **Fix pdmrg-multi-gpu PDMRG rules**: default `n_warmup=1`,
     `n_polish=0`, single-site only, add `--warmup`/`--polish` CLI.
  2. **Fix pdmrg-gpu-opt ctor gate**: add the
     `if (opts_.lanczos_graph && use_davidson_) { disable + warn }`
     block at the same location dmrg-gpu-opt and dmrg2-gpu-opt have.
  3. **Propagate c3d3e50 + H7 to pdmrg-multi-gpu**: remove dead
     pinned-batch + h_rsvd_* + d_boundary_staging buffers; wrap raw
     pointer-mode toggles in PointerModeGuard.

- **What was checked vs. round 11 (2d55d90):**
  - Round-11 M-opt-davidson-toggle (bc3fcd0): preserved across all
    -opt siblings. **NEW finding**: pdmrg-gpu-opt was always missing
    the ctor half — round-11 caught only the runtime setter half.
  - Round-11 dead-buffer mediums (c3d3e50): RESOLVED in
    dmrg/dmrg2/pdmrg-gpu/pdmrg-gpu-opt. **NEW finding**:
    pdmrg-multi-gpu was never in scope of c3d3e50 and still has the
    same dead buffers.
  - All round-7 / round-8 / round-9 / round-10 fixes: PRESERVED
    across the variants they were applied to. The CRITICAL and 2 HIGHs
    in pdmrg-multi-gpu are technique-G propagation gaps that round-11
    self-audit did not catch (the audit only scoped the variants that
    c3d3e50 touched).

**Pattern recognition**: round 8 caught its first technique-G failure
(round-7 C6 fixed in pdmrg-gpu-opt but never propagated to
pdmrg-gpu-base). Round 12 catches the same failure mode for
**project-rules locks** (PDMRG-rules-2026-04-15) and for
**half-fixes** (M-opt-davidson-toggle ctor-vs-setter halves). The
self-audit checklist needs explicit "lock-propagation" and
"two-half-fix" line items, not just defect-class propagation.

Round-12 disposition: blocked on the 1 CRITICAL + 3 HIGHs above.
Fixing immediately, no need for round 13 if the fix scope stays inside
pdmrg-multi-gpu and the pdmrg-gpu-opt ctor.
