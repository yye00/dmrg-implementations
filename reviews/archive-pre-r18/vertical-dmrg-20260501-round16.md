# Vertical review — DMRG family (single-site) — round-16 — 2026-05-01

HEAD: `187fddf`. Round-15 baseline: `f40140d`. Watch list per charter:
`5355c06` (round-15 follow-up — H2-opt + shared kernels), `abd88b9` /
`187fddf` (cross-family H2-opt propagation), `69da5b4` (PhaseTimer
instrumentation). All three tiers in scope.

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | -base 23 / -gpu ~44 / -opt ~46 members. 0 dead. `t_absorb_` declared+init+report but no begin/end pair — see H1. |
| B. Behavioral diff | DONE | apply_heff/update_*_env: -gpu and -opt now both use shared `setup_batch_ptrs_*` kernels. Lanczos: -opt host-pointer-mode dot/nrm2 vs -gpu/-base device-pointer mode + process_alpha/beta_kernel. |
| C. Docstring verify | DONE | -base header overstates "device-pointer mode" (apply_heff uses host scalars); pre-existing NIT. |
| D. clangd filter | N-A | No ROCm headers on host; A subsumes. |
| E. Absence-naming | FOLLOWED | -opt: 4 of 5 PhaseTimer phases instrumented; `t_absorb_` MISSING; -opt Lanczos device-pointer alpha/beta MISSING. |
| F. Workspace-aliasing | DONE | -opt `d_dav_work_` ctor :282-285 round-8 sizing intact. `d_T1_/T2_` env twins disjoint. 0 OVERRUN. |
| G. Sibling fix-propagation | DONE | 69da5b4 PhaseTimer landed for 4 of 5 timers — `t_absorb_` gap. H2-opt kernels intact at :634/:650/:768/:860. H1-base guard scope intact at :540. |

## Regression-watch verification

| Watch item | File:line | Status |
|---|---|---|
| 5355c06 H2-opt apply_heff Step 1 / update_left/right_env | -opt impl :634/:650/:768/:860 | OK |
| Shared `common/batch_ptrs_kernels.h` | 226 LOC, 12 templates | OK |
| dmrg-gpu uses shared header | dmrg_gpu_impl.h:13 | OK |
| Zero per-bond `std::vector<Scalar*> h_A` in -gpu/-opt | only init/MPS-load remain | OK |
| 69da5b4 PhaseTimer 5 sites in -opt | apply_heff/env/lanczos/svd/davidson begin+end balanced | PARTIAL (`t_absorb_` MISSING — H1) |
| Cache-hit closes `t_apply_heff_` | -opt :599 before return :600 | OK |
| Round-14 H1-base Lanczos guard scope | -base :540 inside loop AFTER apply_heff :537 | OK |
| f40140d set_quiet comment | dmrg_gpu.h:43 | OK |
| Round-8 CR-D1 `d_dav_work_` sizing | -opt :282-285 | OK |

## CRITICALS

None.

## HIGHS

- **H1-opt-absorb-timer-uninstrumented** [dmrg-gpu-opt: dmrg_gpu_opt_impl.h:1221, :1468, :1987, :2006]. The 69da5b4 instrumentation pass added begin/end for `t_lanczos_/t_apply_heff_/t_svd_/t_env_update_` plus Davidson, but **omitted `t_absorb_`**. Sibling -gpu splits SVD at the absorb boundary (impl :1268 `t_svd_.end` + :1269 `t_absorb_.begin` ... :1295/:1341 `t_absorb_.end`) so the dual-stream overlap (absorb on `stream_` vs env_update on `stream_env_`) is measurable separately. In -opt the absorb section runs from :1392-1416 (R) / :1442-1465 (L) under a single `t_svd_` window. `init_timers` :1987 and `report_timers` :2006 still wire `t_absorb_`, so under `DMRG_GPU_PROFILE=1` users get a "0 calls 0 ms" line — same false-quiet pattern round-15 H1 fixed for the other phases. Lonely-sibling propagation gap. HIGH.

- **H2-opt-lanczos-host-pointer-mode-roundtrip** [dmrg-gpu-opt: dmrg_gpu_opt_impl.h:1014, :1067, :1074]. `lanczos_eigensolver`'s per-iter loop reads scalars to host stack: `Traits::dot(..., &alpha_result)` :1014, `Traits::nrm2(..., &beta_i)` :1074, iter-0 reorth `&overlap` :1067 — every iteration the value comes back to host in default host-pointer mode. Sibling **-gpu** uses device-pointer mode + `lanczos_process_alpha_kernel/process_beta_kernel` writing to `d_alpha_dev_/d_beta_dev_` (impl :948-952, :1001-1002). Even **-base** does this correctly (impl :540 PointerModeGuard around per-iter ops; alpha → `d_alpha_dev_` :545; beta → `d_beta_dev_` :578). J2 contract violation: -opt is structurally inferior to BOTH -gpu and -base on Lanczos alpha/beta handling. CLAUDE.md "no host roundtrips per sweep" violation: per-iter dot+nrm2 sync. Default path is Davidson, but Lanczos fires on tiny systems (impl :1487 `dim <= 2*b`), on `set_use_davidson(false)`, and on Davidson syevd info!=0 fallback (:1588). Round-7 C2/H6 ported the same defect class for Davidson syev (host-LAPACK → on-device rocsolver_syevd) but never swept Lanczos. HIGH.

## MEDIUMS

None new.

## NITS

- **dmrg-gpu-opt unguarded `prof_davidson_ms` at boundary site** [impl :1855-1859, :1900-1905]. No `opts_.profile` guard (vs proper guard at :1790-1794). Host clock stamp only — cosmetic.
- **dmrg-gpu-base docstring overgeneralization** [dmrg_gpu_base.h:13-18]. "All linear algebra ... in device-pointer mode" — apply_heff GEMMs use host-stack `&one/&zero`. True for BLAS-1 Lanczos only. Pre-existing.

## FALSE POSITIVES VERIFIED

- **dmrg-gpu-opt `lanczos_graph_was_user_enabled_`** — live: ctor :73-79 + setter :89-97 (header). Impl-only grep undercounts.
- **dmrg-gpu-opt apply_heff cache-hit `t_apply_heff_.end` balance** — verified at :599 before `return` :600. Cache-miss path closes at :736. Both legs balanced.

## SUMMARY

Two HIGHs, both lonely-sibling propagation gaps. (1) Round-15 PhaseTimer
pass instrumented 4 of 5 phases in dmrg-gpu-opt; `t_absorb_` is init+report
but never measured — the one phase whose duration matters most for
dual-stream overlap. (2) dmrg-gpu-opt Lanczos leaks per-iter scalar
roundtrips via host-pointer-mode dot/nrm2; sibling -gpu and even -base
handle this on device. J2 violation; CLAUDE.md "no host roundtrips per
sweep" applies on Davidson-fallback paths. Round-15 fixes intact at the
level landed; both new HIGHs are next-layer propagation. Technique-F
sizing clean. No CRITICALs. Recommend bundling H1-opt absorb split with
the H2-opt-lanczos device-pointer port — same pattern as round-7 C2
(device alpha/beta scratches + process kernels + on-device dsteqr).
