# Vertical review — DMRG family (single-site) — round-15 — 2026-05-01

HEAD: `f40140d` (round-14 nit — set_quiet stub comment uniformity).
Round-14 baseline: `5deba6d` (round-14 fixes — 3 highs + 4 mediums).

Scope: `gpu-rocm/dmrg-gpu-base/src/{dmrg_gpu_base.h, dmrg_gpu_base_impl.h}`,
`gpu-rocm/dmrg-gpu/src/{dmrg_gpu.h, dmrg_gpu_impl.h}`,
`gpu-rocm/dmrg-gpu-opt/src/{dmrg_gpu_opt.h, dmrg_gpu_opt_impl.h}`. All three
tiers in scope per charter (round-14 caught a HIGH in `-base` after round-13
mis-scoped).

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | -base 23 members, -gpu ~44, -opt ~46. 0 dead. `lanczos_graph_was_user_enabled_` only 1 impl-hit but legitimately used in ctor:78 + setter (header). |
| B. Behavioral diff | DONE | apply_heff: -gpu uses GPU `setup_batch_ptrs_*` kernels; -opt uses host-loop + 3× `hipMemcpyAsync` per call. PhaseTimer panel: -gpu instrumented, -opt has init/report but zero `.begin/.end` sites. |
| C. Docstring verify | DONE | -opt header :276 svd_fallback fix from round-14 intact. No new drift. |
| D. clangd filter | N-A | No ROCm headers on host; A subsumes dead-symbol case. |
| E. Absence-naming | FOLLOWED | -base/-gpu/-opt feature checklists; J2 holds for primitive-presence but breaks on PhaseTimer instrumentation. |
| F. Workspace-aliasing | DONE | -opt `d_dav_work_` (W + overlap concurrent, restart sequential, alloc=`max(b·θ + max_sub·b, max_sub²)`), `d_dav_work2_` (k×k or eigvecs, sequential), `d_T1_/T2_` env twins (disjoint streams), `d_svd_work_` -gpu/-opt/-base. 0 OVERRUN. |
| G. Sibling fix-propagation | DONE | Round-14 H1-base scope rescope verified intact in -base. Round-14 H2-pdmrg-opt PhaseTimer instrumentation NOT propagated to dmrg-gpu-opt. Round-14 H3 t_apply_heff cache-hit `.end` verified at -gpu impl:667. M-skip-on-zero verified at -gpu:430 + -opt:1985. |

A review with any technique SKIPPED that is not N-A is INVALID — none skipped.

## Regression-watch verification (commit-pinned)

| Watch item | Variant | File:line | Status |
|---|---|---|---|
| Round-14 H1-base apply_heff scope | dmrg-gpu-base | `dmrg_gpu_base_impl.h:519-524` (init norm guard), `:533-589` (per-iter guard at :540 AFTER apply_heff at :537) | OK |
| Round-14 M-skip-on-zero | dmrg-gpu | `dmrg_gpu_impl.h:430` | OK |
| Round-14 M-skip-on-zero | dmrg-gpu-opt | `dmrg_gpu_opt_impl.h:1985` | OK |
| Round-14 M-svd_fallback docstring | dmrg-gpu-opt | `dmrg_gpu_opt.h:276-279` | OK |
| Round-14 H3 cache-hit timer end | dmrg-gpu | `dmrg_gpu_impl.h:667` (`t_apply_heff_.end(stream_)` before early return) | OK |
| bc3fcd0 set_use_davidson symmetric | dmrg-gpu-opt | `dmrg_gpu_opt.h:89-97` | OK |
| bc3fcd0 ctor disable gate | dmrg-gpu-opt | `dmrg_gpu_opt_impl.h:73-79` | OK |
| M14-final dead set_quiet stub gone | dmrg-gpu-opt | not present | OK |
| Round-8 CR-D1 `d_dav_work_` sizing | dmrg-gpu-opt | `:281-287` `max(b·θ_max + max_sub·b, max_sub²)` | OK |
| f40140d set_quiet comment | dmrg-gpu | `dmrg_gpu.h:43` `"no-op"` | OK |
| c3d3e50 d_svd_work_ alive | dmrg-gpu | impl :1370,1375,1385,1417,1421,1432 | OK |

## CRITICALS

None.

## HIGHS

- **H1-opt-PhaseTimer-panel-inert** [dmrg-gpu-opt: dmrg_gpu_opt_impl.h:1973-1998]. The 5-phase PhaseTimer panel (`t_lanczos_`, `t_apply_heff_`, `t_svd_`, `t_absorb_`, `t_env_update_`) is init'd at :1973-1978 and report'd at :1982-1998, but ZERO `.begin/.end` sites exist in the body — `grep -n 't_lanczos_\|t_apply_heff_\|t_svd_\|t_absorb_\|t_env_update_'` returns only init+report rows. Round-14 H2 fixed this exact defect-class in pdmrg-gpu-opt; technique-G never traced the gap to dmrg-gpu-opt. Parallel host-side `prof_*` counters (:44-46) are wired and produce per-sweep stdout, but `report_timers()` under `DMRG_GPU_PROFILE=1` prints only the header (skip-on-zero suppresses everything). Lonely-sibling round-14 H2 recurrence. HIGH.

- **H2-opt-host-batch-pointer-roundtrip** [dmrg-gpu-opt: dmrg_gpu_opt_impl.h:624-655 (apply_heff), :766-776 (update_left_env), :860-870 (update_right_env)]. Every `apply_heff` builds Step-1 batched-GEMM pointer arrays as `std::vector<Scalar*> h_A(D*d), h_B(D*d), h_C(D*d)` on the host stack and issues 3× `hipMemcpyAsync(..., hipMemcpyHostToDevice)` to populate `d_batch_*`. Same pattern in `update_left_env` and `update_right_env` (3× H2D each). Per-Lanczos-iter / per-Davidson-iter / per-bond on default path. The -gpu sibling (impl :699-704, :715-718, :833-836) uses GPU `setup_batch_ptrs_wd_sparse<>` / `setup_batch_ptrs_wd<>` kernels — zero host roundtrip. Direct violation of CLAUDE.md "no host roundtrips per sweep" rule. The env H2D also races `event_canon_ready_`/`event_env_done_` pacing on the dual-stream pipeline. Pre-existing — prior reviewers scoped to recent edits and missed it. HIGH.

## MEDIUMS

None new.

## NITS

None new. Round-14's carry-over (dmrg-gpu-base scoped-region indent cosmetic) still open but cosmetic-only.

## FALSE POSITIVES VERIFIED

- **dmrg-gpu-opt no PointerModeGuard usage** — verified intentional. The -opt Lanczos and Davidson keep all rocBLAS calls in host-pointer mode (host-stack `&one`/`&neg_alpha`/scalar locals like `alpha_result`/`norm`). No device-pointer region exists, so no guard is needed. Not a defect.
- **dmrg-gpu-opt apply_heff cache-hit early return at :594 has no `t_apply_heff_.end()`** — the timer is never started in -opt (per H1 above), so there is no leak counterpart to round-14 H3. Once H1 is fixed, this site WILL need a matching `.end()` before the early return (forward-looking note for the H1 fix).
- **dmrg-gpu-opt RSVD path host Gaussian fill at :1302-1308** — opt-in path (use_rsvd_), once per site per sweep, fill cost dominated by H2D bandwidth. Acceptable per existing project consensus on the RSVD setup cost.

## SUMMARY

Two HIGHs, both pre-existing technique-G/B gaps. (1) dmrg-gpu-opt PhaseTimer
panel wired but uninstrumented — same defect-class round-14 fixed in
pdmrg-gpu-opt; sibling sweep never ran. (2) dmrg-gpu-opt apply_heff +
update_{left,right}_env build batched-GEMM pointer arrays on host and H2D
them every call; -gpu sibling uses GPU `setup_batch_ptrs_*` kernels. Direct
CLAUDE.md "no host roundtrips per sweep" violation on default path; also
races dual-stream event pacing. Round-14 fixes (H1-base scope,
M-skip-on-zero, M-svd_fallback docstring) all intact. Technique-F Davidson
sizing clean. No CRITICALs. Recommend fixing both as a propagation batch and
running a horizontal-opt diff of apply_heff/env-update H2D patterns across
all -opt siblings before next major event.
