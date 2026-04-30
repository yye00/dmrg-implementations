# Vertical review — DMRG family (single-site) — round-12 — 2026-04-30

HEAD: `c3d3e50` (round-11 cleanup commit — spotless dead-buffer removal)
Round-11 baseline: `2d55d90` (round-11 conformity report)
Cleanup commits since baseline:
- `bc3fcd0` — M-opt-davidson-toggle in dmrg-gpu-opt (symmetric setter)
- `c3d3e50` — dead-buffer cleanup across -gpu / -gpu-opt (and dmrg2/pdmrg-opt siblings)

Scope: `gpu-rocm/dmrg-gpu/src/{.h,_impl.h}` + `gpu-rocm/dmrg-gpu-opt/src/{.h,_impl.h}`. Per charter, `dmrg-gpu` doubles as the `-base` reference (CPU-SVD opt-in path = the -base equivalent).

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | dmrg-gpu (44 members) and dmrg-gpu-opt (45 members) — **0 dead**. Round-11's MED-opt-dead-lanczos-scratches and MED-opt-dead-h_svd_tmp eliminated by `c3d3e50`. |
| B. Behavioral diff | DONE | dmrg-gpu Lanczos device-pointer vs dmrg-gpu-opt Lanczos host-pointer — intentional (Davidson default, Lanczos fallback). 0 net-new divergences. |
| C. Docstring verify | DONE | dmrg-gpu-opt header (lines 14-50) and dmrg-gpu header (12-94) — all claims backed by code. 0 unverified. |
| D. clangd filter | N-A | No ROCm headers on host — A subsumes dead-symbol case. |
| E. Absence-naming | FOLLOWED | -gpu and -opt feature checklists run; J2 strict-superset invariant holds. 0 MISSING. |
| F. Workspace-aliasing | DONE | 4 shared scratches re-audited (`d_dav_work_`, `d_dav_work2_`, `d_T1_/T2_`, `d_svd_work_`). Cleanup did not touch aliasing slices. **0 OVERRUN.** |
| G. Sibling fix-propagation | DONE | 2 net-new fixes (`bc3fcd0`, `c3d3e50`) traced. Davidson-toggle present in all 3 -opt variants; CPU-SVD-fallback buffers preserved. **0 MISSING.** |

A review with any technique SKIPPED that is not N-A is INVALID — none skipped.

## Regression-watch verification (commit-pinned)

| Watch item | Variant | File:line | Status |
|---|---|---|---|
| `bc3fcd0` symmetric setter | dmrg-gpu-opt | `dmrg_gpu_opt.h:89-97` | OK |
| `bc3fcd0` ctor disable gates on `use_davidson_` | dmrg-gpu-opt | `dmrg_gpu_opt_impl.h:73-79` | OK |
| Symmetric setter NO regression | dmrg2-gpu-opt | `dmrg2_gpu_opt.h:73-81` | OK |
| Symmetric setter NO regression | pdmrg-gpu-opt | `pdmrg_gpu_opt.h:61-72` | OK |
| `c3d3e50` 8 Lanczos scratches + 4 dev consts + h_svd_tmp_ removed | dmrg-gpu-opt | gone | OK |
| `c3d3e50` `h_svd_U_/Vh_/S_/tmp_` removed | dmrg-gpu | gone | OK |
| `d_svd_work_` still alive in dmrg-gpu | dmrg-gpu | `dmrg_gpu_impl.h:1387,1392,1438,1449` | OK |
| `h_svd_A_/U_/Vh_/work_/S_` alive in -opt CPU-SVD path | dmrg-gpu-opt | `:1253,1260-1275` | OK |
| Round-8 CR-D1 dav_work sizing intact | dmrg-gpu-opt | `:281-287` | OK |
| Dual-stream env-update overlap | dmrg-gpu | `:201-207, 1417-1430` | OK |
| Dual-stream env-update overlap | dmrg-gpu-opt | `:111-118, 1827-1838` | OK |
| Round-7 C2 rocsolver_dsteqr / syevd | dmrg-gpu-opt | `:1105-1107, 1566-1568` | OK |

## Technique F detail — `d_dav_work_` aliasing re-audit (post-c3d3e50)

`block_davidson_eigensolver` `:1660-1671` slices `d_dav_work_`: Region 1 (residuals W) offset 0, size `n_new·dim ≤ b·theta_size_max_`; Region 2 (overlap) offset `n_new·dim`, size `k·n_new ≤ max_sub·b` — concurrent in expansion. Region 3 (restart X_keep) at offset 0, size `dim·keep ≤ b·theta_size_max_` — sequential. Required = `b·theta_size_max_ + max_sub·b`; allocation `:281-287` is `max(b·theta_size_max_ + max_sub·b, max_sub²)`. **OK; no overrun.** `c3d3e50` did not touch Davidson code paths; removed Lanczos scratches were never aliased with `d_dav_work_`.

## CRITICALS

None.

## HIGHS

None.

## MEDIUMS

None net-new. The two round-11 MEDIUMs (MED-opt-dead-lanczos-scratches, MED-opt-dead-h_svd_tmp) were resolved by `c3d3e50`.

### MEDIUMs — pre-existing carry-over

- M-carry: `DmrgPointerModeGuard` is a struct local to `dmrg_gpu_impl.h:18-30`; not migrated to `common/pointer_mode_guard.h`. Round-9 deferral.
- M-carry: `set_quiet(bool)` no-op across all three tiers. API surface kept for parity with pdmrg variants.
- M-carry: dmrg-gpu-opt Lanczos α/β kept host-resident at `:979-980`. Default code path is Block-Davidson; Lanczos is fallback-only.

## NITS

- Comment block alignment between `dmrg_gpu_opt.h` line 142-149 (dual-stream description) and `dmrg_gpu.h` line 83-94 (sibling description) — minor wording differences. Cosmetic, deferred from earlier rounds.

## FALSE POSITIVES VERIFIED

- Low-grep-count members `d_lanczos_v_`, `d_dav_V_`, `d_dav_AV_`, `d_T2_` — all aliased to local pointers (`d_lanczos_v`, `V`, `AV`, `U`) and live through their functions. Not dead.
- `hipStreamSynchronize` after `d_dav_eigvals_` D2H `:1574-1579, 1596-1599` — required for control-flow (energy convergence + info status). Not `opts_.profile`-gated because branching needs the result.
- CPU-SVD path `:1247-1280` is `use_cpu_svd_=true` opt-in; default is on-device `:1281-1410`. `h_svd_*` correctly retained by `c3d3e50`.
- Per-call `hipMalloc/hipFree` of FUSE_LANCZOS scratches `:984-991, 1147-1149` — opt-in flag only, documented.

## SUMMARY

Round-12 is **CLEAN** for vertical-review-dmrg. Both round-11 MEDIUMs (dead Lanczos scratches + dead `h_svd_tmp_`) resolved by `c3d3e50`; the round-11 davidson-toggle gap closed by `bc3fcd0`. Technique A (symbol-usage scan) returns 0 dead members across both variants — first time this sub-review hits zero net-new across all techniques. Technique F re-verified the `d_dav_work_` aliasing intact (round-8 CR-D1 sizing still strict-superset of required). Technique G traced both fixes through siblings: Davidson-toggle pattern present in all 3 -opt variants; dead-buffer cleanup did not touch live code paths (CPU-SVD `h_svd_*` retention verified). No criticals, no highs, no net-new mediums. Two consecutive zero-net-new rounds across this sub-review now achieved (round-12 + projected round-13 if no new commits land).

**Verdict: ready for the GPU window with respect to the dmrg family vertical contract.**
