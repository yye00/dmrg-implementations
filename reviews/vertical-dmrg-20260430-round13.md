# Vertical review — DMRG family (single-site) — round-13 — 2026-04-30

HEAD: `0b9fccf` (round-12 medium cleanup — guard consolidation + multi-gpu polish)
Round-12 baseline: `8b7a68e` (round-12 fixes — 1 critical + 3 highs in lonely siblings)
Cleanup commits since baseline:
- `0b9fccf` — M1-final inline guard → shared `PointerModeGuard`; M14-final `set_quiet` stub removed in dmrg-gpu-opt; pdmrg-multi-gpu init/timer parity (out of scope here).

Scope: `gpu-rocm/dmrg-gpu/src/{dmrg_gpu.h, dmrg_gpu_impl.h}` + `gpu-rocm/dmrg-gpu-opt/src/{dmrg_gpu_opt.h, dmrg_gpu_opt_impl.h}`. Per charter, `dmrg-gpu` doubles as `-base` reference (no separate `-base` variant ships).

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | dmrg-gpu (44 members) and dmrg-gpu-opt (45 members) — **0 dead**. Round-11/12 cleanups intact; no member reintroduced. |
| B. Behavioral diff | DONE | dmrg-gpu Lanczos device-pointer-mode vs dmrg-gpu-opt Lanczos host-pointer-mode — intentional (Davidson default, Lanczos fallback in -opt). 0 net-new divergences. |
| C. Docstring verify | DONE | Headers (dmrg_gpu.h:12-94, dmrg_gpu_opt.h:14-50, 79-97, 141-149) all backed by code. 0 unverified. |
| D. clangd filter | N-A | No ROCm headers on host — A subsumes dead-symbol case. |
| E. Absence-naming | FOLLOWED | -gpu and -opt feature checklists run; J2 strict-superset invariant holds. 0 MISSING. |
| F. Workspace-aliasing | DONE | 4 shared scratches re-audited (`d_dav_work_`, `d_dav_work2_`, `d_T1_/T2_`, `d_svd_work_`). M1-final touched header includes only — no aliasing slices changed. **0 OVERRUN.** |
| G. Sibling fix-propagation | DONE | 2 net-new fixes (`0b9fccf` M1-final, `0b9fccf` M14-final) traced. Carry-over watch (`bc3fcd0`, `c3d3e50`) re-verified. **0 MISSING.** |

A review with any technique SKIPPED that is not N-A is INVALID — none skipped.

## Regression-watch verification (commit-pinned)

| Watch item | Variant | File:line | Status |
|---|---|---|---|
| `0b9fccf` M1-final include | dmrg-gpu | `dmrg_gpu_impl.h:12` (`#include "../../common/pointer_mode_guard.h"`) | OK |
| `0b9fccf` M1-final inline guard removed | dmrg-gpu | inline `DmrgPointerModeGuard` struct gone (was `:18-30` pre-cleanup) | OK |
| `0b9fccf` M1-final guard in use | dmrg-gpu | `dmrg_gpu_impl.h:1043, 1189` (`PointerModeGuard pm_guard(...)`) | OK |
| No inline guard struct anywhere | family-wide | `grep -r "struct.*PointerModeGuard" gpu-rocm/` returns only the canonical | OK |
| No stray `rocblas_set_pointer_mode` outside guard | dmrg family | 0 hits in dmrg-gpu/dmrg-gpu-opt | OK |
| `0b9fccf` M14-final removal | dmrg-gpu-opt | `dmrg_gpu_opt.h:75-101` — no `set_quiet` member | OK |
| dmrg-gpu-opt test driver does not call set_quiet | dmrg-gpu-opt | `test_dmrg_gpu_opt.cpp` 0 hits | OK |
| dmrg-gpu retains stub | dmrg-gpu | `dmrg_gpu.h:43` (intentional — `test_dmrg_gpu.cpp` calls it 6×) | OK |
| `bc3fcd0` symmetric setter (carry) | dmrg-gpu-opt | `dmrg_gpu_opt.h:89-97` | OK |
| `bc3fcd0` ctor disable gates on `use_davidson_` (carry) | dmrg-gpu-opt | `dmrg_gpu_opt_impl.h:73-79` | OK |
| `c3d3e50` dead-buffer removal not regressed | both | 0 reintroductions | OK |
| Round-8 CR-D1 dav_work sizing intact | dmrg-gpu-opt | `:281-287` `dav_work_sz = max(b·dim+max_sub·b, max_sub²)` | OK |
| Davidson aliasing W/overlap | dmrg-gpu-opt | `:1660-1671` (W@0, overlap@n_new·dim) | OK |
| Dual-stream env-update overlap | dmrg-gpu | `:1472-1488, 1508-1524` | OK |
| Dual-stream env-update overlap | dmrg-gpu-opt | `:1816-1840, 1864-1886` | OK |
| Round-7 C2 rocsolver_dsteqr/syevd | dmrg-gpu-opt | `:1568, syevd in Davidson loop` | OK |

## Technique F detail — `d_dav_work_` aliasing re-audit (post-`0b9fccf`)

`block_davidson_eigensolver` slices `d_dav_work_` into:
- Region 1 (residuals W) offset 0, size `n_new·dim ≤ b·theta_size_max_` — concurrent with…
- Region 2 (overlap) offset `n_new·dim`, size `k·n_new ≤ max_sub·b`. Both live during `:1664-1671`.
- Region 3 (restart X_keep) at offset 0, size `dim·keep ≤ b·theta_size_max_` — sequential (post-restart).

Required: `b·theta_size_max_ + max_sub·b`; allocation `:281-287` is `max(b·theta_size_max_ + max_sub·b, max_sub²)`. Strict-superset of required. **OK; no overrun.** `0b9fccf` did not touch Davidson code — confirmed by inspecting the diff (header guard include only, no impl-side allocation or aliasing edits).

## CRITICALS

None.

## HIGHS

None.

## MEDIUMS

None net-new.

### MEDIUMs — pre-existing carry-over

- M-carry: dmrg-gpu retains `set_quiet(bool){}` no-op stub at `dmrg_gpu.h:43` — kept because `test_dmrg_gpu.cpp` calls it (6 sites). Symmetric to `pdmrg-gpu-opt`'s retention. Not a defect, intentional API surface.
- M-carry: dmrg-gpu-opt Lanczos α/β kept host-resident at `:979-980`. Default code path is Block-Davidson; Lanczos is fallback-only.

## NITS

- dmrg_gpu.h:43 stub comment ("all output removed except final summary") differs from sibling stub comments ("no-op"). Cosmetic.

## FALSE POSITIVES VERIFIED

- Low-grep-count members `d_lanczos_v_`, `d_T2_`, `d_const_neg_one_` (dmrg-gpu only) — verified aliased to local pointers (`d_lanczos_v`, `U`/temp role) inside hot-path functions; live.
- `d_const_neg_one_` zero hits in dmrg-gpu-opt — variant doesn't declare it (host-pointer-mode Lanczos doesn't need device-side constants). Not dead, absent.
- `hipStreamSynchronize` after `d_dav_eigvals_` D2H in Davidson — required for control-flow (energy convergence + info status). Not `opts_.profile`-gated because branching needs the result.
- `rocblas_set_pointer_mode` zero hits in dmrg-gpu/dmrg-gpu-opt — guards do all toggles (technique C lock).
- CPU-SVD path `:1247-1280` is `use_cpu_svd_=true` opt-in; default is on-device.

## SUMMARY

Round-13 is **CLEAN** for vertical-review-dmrg. The two `0b9fccf` cleanups (M1-final guard consolidation, M14-final dead `set_quiet` stub) are both correctly contained: the dmrg-gpu inline guard struct is gone and replaced by the canonical `PointerModeGuard` from `common/pointer_mode_guard.h`, and the dmrg-gpu-opt `set_quiet` removal is consistent (its test driver doesn't call it). Technique A (symbol-usage scan) returns 0 dead members across both variants — three consecutive zero-net-new rounds for this sub-review. Technique F re-verified the `d_dav_work_` aliasing intact (round-8 CR-D1 sizing still strict-superset of required); the cleanup did not touch Davidson code paths. Technique G traced both fixes through siblings: M1-final propagated to all 3 inline guards (dmrg-gpu, accurate_svd_gpu, rlbfgs-gpu) with pdmrg-gpu-opt's reference updated; M14-final correctly retained in dmrg-gpu and pdmrg-* tiers where their test drivers still call set_quiet. No criticals, no highs, no net-new mediums.

**Verdict: ready for the GPU window with respect to the dmrg family vertical contract.**
