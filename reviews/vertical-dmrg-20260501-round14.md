# Vertical review — DMRG family (single-site) — round-14 — 2026-05-01

HEAD: `f5c0617` (round-13 nit cleanup — orphan comment + h_svd_* docstring; touches dmrg2 only)
Round-13 baseline: `ee653f0` (round-13 fixes — 2 highs + 4 mediums, technique-G propagation)

Scope: `gpu-rocm/dmrg-gpu-base/src/{dmrg_gpu_base.h, dmrg_gpu_base_impl.h}` + `gpu-rocm/dmrg-gpu/src/{dmrg_gpu.h, dmrg_gpu_impl.h}` + `gpu-rocm/dmrg-gpu-opt/src/{dmrg_gpu_opt.h, dmrg_gpu_opt_impl.h}`. Per the charter (vertical-review-dmrg.md:14-16), all three tiers including `-base` are in scope. Round-13's report misread its own scope as "no -base ships"; this round audits all three.

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | dmrg-gpu-base (25 members), dmrg-gpu (~44), dmrg-gpu-opt (~45). 0 dead. |
| B. Behavioral diff | DONE | Lanczos device-pointer-mode region wrapping `apply_heff` differs between `-base` (wraps) and `-gpu`/`-opt` (excludes). 1 net-new defect surfaced. |
| C. Docstring verify | DONE | dmrg-gpu-opt header decl `svd_fallback` comment "(CPU LAPACK)" contradicts impl docstring at :1216 ("on-device default path"). 1 unverified header claim. Round-13 axis-3 lesson hit. |
| D. clangd filter | N-A | No ROCm headers on host; technique A subsumes dead-symbol case. |
| E. Absence-naming | FOLLOWED | -base, -gpu, -opt feature checklists run; J2 strict-superset invariant holds. 0 MISSING. |
| F. Workspace-aliasing | DONE | `d_dav_work_/2_` (opt), `d_T1_/T2_` + env twins (gpu/opt), `d_svd_work_` (base). Round-13 sizing intact. 0 OVERRUN. |
| G. Sibling fix-propagation | DONE | round-13 M1-base-prop guard adoption traced. 1 propagation gap: dmrg-gpu-base's brace-block scope wraps `apply_heff`, not aligned with `-gpu`/`-opt`'s tighter scope (apply_heff outside guard). |

A review with any technique SKIPPED that is not N-A is INVALID — none skipped.

## Regression-watch verification (commit-pinned)

| Watch item | Variant | File:line | Status |
|---|---|---|---|
| `ee653f0` M1-base-prop include | dmrg-gpu-base | `dmrg_gpu_base_impl.h:14` (`#include "../../common/pointer_mode_guard.h"`) | OK |
| `ee653f0` M1-base-prop guards | dmrg-gpu-base | `:521`, `:625` (`PointerModeGuard pm_guard(...)`) | OK |
| `ee653f0` M14-base-prop removal | dmrg-gpu-base | `dmrg_gpu_base.h` — no `set_quiet` member | OK |
| `bc3fcd0` symmetric davidson setter (carry) | dmrg-gpu-opt | `dmrg_gpu_opt.h:89-97` | OK |
| `bc3fcd0` ctor disable gate (carry) | dmrg-gpu-opt | `dmrg_gpu_opt_impl.h` ctor | OK |
| `c3d3e50` dead-buffer removal not regressed | dmrg-gpu/-opt | 0 reintroductions | OK |
| Round-8 CR-D1 `d_dav_work_` sizing | dmrg-gpu-opt | `:281-287` `dav_work_sz = max(b·θ_max + max_sub·b, max_sub²)` | OK |
| Davidson aliasing W/overlap | dmrg-gpu-opt | `:1660-1671` (W@0, overlap@n_new·dim) | OK |
| Dual-stream env-update overlap | dmrg-gpu | impl ~1472-1524 | OK |
| Dual-stream env-update overlap | dmrg-gpu-opt | impl ~1816-1886 | OK |
| `apply_heff` outside device-pointer-mode (sibling pattern) | dmrg-gpu | impl :1033 vs guard :1043 | OK |
| `apply_heff` outside device-pointer-mode (sibling pattern) | dmrg-gpu-opt | impl :1015 (no guard, host mode throughout) | OK |
| `apply_heff` outside device-pointer-mode (sibling pattern) | dmrg-gpu-base | impl :536 — INSIDE guard `:521-588` | **DEFECT** |

## CRITICALS

None.

## HIGHS

- **H1-base-apply_heff-in-device-pointer-mode** [dmrg-gpu-base: dmrg_gpu_base_impl.h:520-588]. Round-13 M1-base-prop wrapped `lanczos_eigensolver`'s inner loop in a `PointerModeGuard(...,device)` brace block at :521-:588. The block ENCLOSES `apply_heff(...)` at :536, which inside (impl :260-319) issues `Traits::gemm` with host-stack `&one`/`&zero_val` (lines 283, 286, 295, 298, 312, 315). With the handle in device-pointer mode, rocBLAS dereferences these as device pointers — UB. Pre-round-13 code had the same defect (raw `set_pointer_mode(device)` at function top); the guard refactor preserved the broken scope rather than tightening it. The correct sibling shape is `dmrg-gpu` (impl :1033 vs guard :1043) and `dmrg-gpu-opt` (impl :1015, host mode throughout) — scope the guard tightly around BLAS-1 ops only, exclude `apply_heff`. Technique-G propagation gap missed by round-13. (Same defect exists in dmrg2-gpu-base from the same M1-base-prop batch — flagged for vertical-dmrg2.) HIGH because the tests pass empirically (rocBLAS may tolerate stack-address-as-device-pointer on some MI300X paths) but this is UB on the default `-base` code path.

## MEDIUMS

- **M1-opt-svd_fallback-docstring-drift** [dmrg-gpu-opt: dmrg_gpu_opt.h:276]. Header decl comment "// SVD bond splitting (CPU LAPACK)" contradicts the impl docstring at `dmrg_gpu_opt_impl.h:1216` ("on-device default path") and the body at 1216-1349, which uses `rocsolver_gesvd_auto` by default and routes to host LAPACK only on the `use_cpu_svd_=true` opt-in branch (`:1247-1283`). Function name `svd_fallback` is itself stale. Round-13 axis-3 "docstring promise = half-fix" class.

## NITS

- `dmrg_gpu_base_impl.h:520-588` — the contents of the `PointerModeGuard` brace block are not re-indented (matching dmrg2-gpu-base round-13 style). Cosmetic.
- `dmrg_gpu.h:43` `set_quiet(bool){}` stub comment ("all output removed except final summary") still differs from sibling stubs ("no-op"). Carry-over from round-13.

## FALSE POSITIVES VERIFIED

- Low-grep-count members `d_lanczos_v_`, `d_T2_`, `d_const_neg_one_` in dmrg-gpu — verified aliased to local pointers inside hot-path functions.
- `hipStreamSynchronize` after `d_dav_eigvals_` D2H in Davidson and after dsteqr eigval D2H in Lanczos — required for control-flow (energy convergence + info status). Not `opts_.profile`-gated because branching needs the result.
- `h_svd_*` in dmrg-gpu-opt `:213-216` — used at `:235-267` (size query) and `:1253-1283` (use_cpu_svd_ opt-in). Live.
- `lanczos_graph_was_user_enabled_` round-trip in `set_use_davidson` — verified symmetric (round-11 `bc3fcd0` ctor disable gate intact).
- dmrg-gpu-opt's Lanczos has no PointerModeGuard at all because all BLAS-1 ops use host-pointer scalars (`alpha_result`, `&neg_alpha`, etc.). Intentional, immune to H1.

## SUMMARY

Round-14 surfaces **1 HIGH and 1 MEDIUM** — both catches that round-13's vertical-dmrg missed because that report self-scoped to "-gpu and -gpu-opt only" and skipped `dmrg-gpu-base/` (which IS in charter). H1 is a Technique-G sibling-propagation gap: round-13 M1-base-prop's `PointerModeGuard` brace in dmrg-gpu-base Lanczos wraps the whole inner loop, including `apply_heff` whose host-stack scalars then run under device-pointer mode (UB). Siblings `-gpu`/`-opt` show the correct tighter scope. M1-opt is the round-13 axis-3 docstring-promise pattern. Technique F re-verified `d_dav_work_` sizing intact. Technique A returns 0 dead across all three tiers. **Recommend H1 fix before the next GPU window**: rescope the brace block to exclude `apply_heff`, mirroring `dmrg_gpu_impl.h:1033-1043`.
