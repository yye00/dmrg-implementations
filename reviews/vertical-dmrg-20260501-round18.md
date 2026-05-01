# Vertical review — DMRG family (single-site) — round-18 — 2026-05-01

HEAD: `12d02c5`. Round-17 baseline: `0efe96d`. Watch list per charter:
`abd88b9`, `187fddf`, `8abb6e7`, `0efe96d`, `54f2fcf`, `12d02c5`. All
three tiers in scope. Defect-class registry sweep run first
(`bash .claude/scripts/defect-registry.sh`) — 0 hits across all 10
in-charter variants.

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | -base 22 / -gpu ~44 / -opt ~57 (incl. R16 D12 + R17 t_davidson_). All members have ≥1 non-ctor/dtor use. 0 dead. |
| B. Behavioral diff | DONE | Sweep functions across all three tiers structurally aligned (event_canon_ready_/event_env_done_ pair-up, env_update_pending_ guard, optimize_site signature). PhaseTimer pattern: -base has none (charter-correct), -gpu has 5 panels, -opt has 6 (adds `t_davidson_`). |
| C. Docstring verify | DONE | -base header L13-18 still overstates "device-pointer mode" for GEMM/GEMV (BLAS-3 actually uses host-stack `&one/&zero`); pre-existing NIT carried from R17. |
| D. clangd filter | N-A | No ROCm headers on host; A subsumes. |
| E. Absence-naming | FOLLOWED | All -base / -gpu / -gpu-opt expected features present. -opt 6/6 PhaseTimer panels now instrumented (R17 closed the absorb gap). |
| F. Workspace-aliasing | DONE | `d_dav_work_` :301-305 retains R8 CR-D1 sizing `max(theta·b + max_sub·b, max_sub²)`. R17 D12 buffers single-role. R17 H1's `t_absorb_` re-use of `t_svd_.end / t_absorb_.begin` boundary at :1384-1385 / :1437-1438 disjoint from absorb GEMM payload. 0 OVERRUN. |
| G. Sibling fix-propagation | DONE | 6 watch-list fixes traced, 6 propagated correctly, 0 MISSING. Detail in next section. |

## Regression-watch verification (round-17 fixes)

| Fix (commit) | Variant | File:line | Status |
|---|---|---|---|
| abd88b9 D5 host-batch-ptr propagation | dmrg-gpu | impl.h:600-616 (setup_batch_ptrs_wd) | OK |
| 187fddf D5 Step 3 fallback path | dmrg-gpu / -opt | dmrg-gpu fallback structurally aligned with -opt :718-755 | OK (not pdmrg-only as R17 noted) |
| 8abb6e7 D6 kernel duplication collapsed | dmrg-gpu / -opt | both `#include "../../common/batch_ptrs_kernels.h"` :13/:12 | OK |
| 0efe96d D12 Lanczos host-stack scalars | dmrg-gpu-opt | h:176-186 (11 members), impl :175-191 alloc, :348-358 free | OK |
| 0efe96d D12 baseline intact | dmrg-gpu | impl :119-127 (members), :100-122 (alloc) | OK |
| 0efe96d D12 immune | dmrg-gpu-base | impl :510-637 already in device-pointer pattern since R14 | IMMUNE |
| 54f2fcf CR-D1 / D14 prop | dmrg-gpu-opt | impl :301-307 (sizing), :1665-1671 (overlap-at-offset) | OK (this was the round-7 fix that didn't regress in -opt) |
| 54f2fcf CR-D1 / D14 prop | dmrg2-gpu-opt | per R8 baseline; out-of-scope for this vertical, sibling reviews verify | DEFERRED |
| 12d02c5 H1 t_absorb_ split | dmrg-gpu-opt | impl :1384-1385 / :1413, :1437-1438 / :1465 | OK |
| 12d02c5 H2 t_davidson_ panel | dmrg-gpu-opt | h:272 decl, impl :1492 begin, 5× .end at :1584/:1624/:1657/:1709/:1768 | OK |
| 12d02c5 H2 t_lanczos_ no longer wraps Davidson | dmrg-gpu-opt | impl :996/:1200 wrap only Lanczos; tiny-dim short-circuit at :1485-1487 returns BEFORE t_davidson_.begin | OK |
| 12d02c5 init/report includes t_davidson_ | dmrg-gpu-opt | impl :1982 init, :2002 report | OK |
| 12d02c5 sibling -gpu intentionally omits t_davidson_ | dmrg-gpu | h:193-197 only 5 panels (no Davidson) | INTENTIONAL (no Davidson) |
| 12d02c5 sibling -base intentionally omits all PhaseTimers | dmrg-gpu-base | h:53-148 charter-correct | INTENTIONAL |

## CRITICALS

None.

## HIGHS

None. The R17 H1-opt-absorb-timer gap (carried from R16) is **closed
this round** — `t_svd_.end / t_absorb_.begin / t_absorb_.end` now
present in both R-direction (impl :1384-1413) and L-direction
(impl :1437-1465) of `svd_fallback`. `t_davidson_` is now its own
panel and Lanczos timing is no longer overloaded. No new HIGHs found.

## MEDIUMS

- **M1-opt-lanczos-init-D2H-sync** [dmrg-gpu-opt: dmrg_gpu_opt_impl.h:1011-1017]. Carried from R17 unchanged. The D12 port of the initial-norm computation is in device-pointer mode and ends with `hipMemcpyAsync(&norm, d_nrm2_result_, ..., D2H) + hipStreamSynchronize`. Sibling **dmrg-gpu** :913-914 keeps rocBLAS in host-pointer mode for the one-shot init nrm2 (`Traits::nrm2(..., &norm)` direct host-stack), avoiding the explicit sync entirely. Once-per-Lanczos-call, not per-iter, so the "no host roundtrip per sweep" rule is technically satisfied (Davidson is the default; Lanczos init is on the fallback path). Cleaner pattern: match the -gpu sibling. Tagged MEDIUM, not HIGH, because (a) Lanczos is the fallback in -opt, not the hot path, and (b) the explicit sync is a bounded cost.

## NITS

- **dmrg-gpu-base docstring overgeneralization** [dmrg_gpu_base.h:13-18]. Pre-existing carry from R16/R17. The header reads "rocBLAS GEMM/GEMV/AXPY/DOT/NRM2 in device-pointer mode" — but apply_heff (impl :283/:286, :295/:298, :312, :347/:353, :362/:365) and env-update GEMMs use host-stack `&one/&zero`. Only Lanczos BLAS-1 ops (impl :519-523, 540-588, 629-633) are wrapped in `PointerModeGuard pm_guard(..., rocblas_pointer_mode_device)`. Either narrow the claim ("Lanczos BLAS-1 ops in device-pointer mode") or guard the GEMM scalars too. Cosmetic, no behavioral defect.
- **dmrg-gpu-opt header taxonomy paragraph** [dmrg_gpu_opt.h:13-50]. Solid, faithful to the R12 user lock. No action.

## FALSE POSITIVES VERIFIED

- **Registry D7/D8 (15 hits) all gated**. dmrg-gpu-opt :276 = init-time `lwork_query=-1` workspace-size query, no compute. :1250 = `use_cpu_svd_` opt-in branch (D2H + hipStreamSynchronize at :1246, then host gesvd, then H2D). dmrg-gpu has 0 lapack_ hits. dmrg-gpu-base has 0 lapack_ hits. All gated/legit.
- **Registry D9/D15 PhaseTimer "no .begin/.end"**. Sweep clean for the dmrg family this round; the R17-introduced sed-extract pattern correctly identifies 6 timers in -opt, all paired. R15's regate of `event_canon_ready_` (substring containing "t_canon_ready_" — false comment match) was already closed in R17.
- **Init-time host→device memcpys** [dmrg_gpu_opt_impl.h:188-190, :461, :476, :492, :515, :525, :540, :544, :577, :583, :961, :972]. All are constructor-time / `set_mpo` time / `build_initial_environments` time, NOT per-sweep. The "no host roundtrips per sweep" rule applies to the inner sweep loop, not initialization. Verified by call-site context.
- **Registry D2 (1 hit pdmrg-gpu-opt)**. Out-of-scope for dmrg vertical.
- **Registry D13 (3 hits, all `-base` siblings)**. Charter-allowed (naive single-GEMM is the -base baseline; D13 registry rule explicitly exempts -base).

## SUMMARY

R18 is the cleanest dmrg-family vertical to date: zero CRITICALs, zero
HIGHs, one carried MEDIUM (M1-opt-lanczos-init-D2H-sync) and one
carried NIT (dmrg-gpu-base header overgeneralization). The R17 commit
(12d02c5) cleanly closes the R15→R17 H1 absorb-timer gap with the
expected `t_svd_.end / t_absorb_.begin / t_absorb_.end` triple at both
sweep directions, and properly splits Davidson from Lanczos timing.
The R17-CRITICAL pdmrg-gpu-opt CR-D1 propagation (54f2fcf) is in the
sibling family but belongs to the same defect class as R8 CR-D1 in
dmrg-gpu-opt; R8's fix to the dmrg family did not regress in this
review. Defect registry shows 0 hits across all in-charter variants.
The dmrg family is in a stable, fully-instrumented, technique-G-clean
state for the next paper run.
