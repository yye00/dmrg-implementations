# Horizontal review — -gpu tier — round 13 (2026-04-30, post-cleanup)

HEAD: `0b9fccf` (round-12 medium cleanup). Diff vs round-12 baseline
(`8b7a68e`): `0b9fccf` consolidates pointer-mode guards and adds the
`PhaseTimer` panel + init API parity to pdmrg-multi-gpu.

## Charter

Review **-gpu tier** across dmrg-gpu, dmrg2-gpu, pdmrg-gpu, and pdmrg-multi-gpu.
Verify the round-12 fixes (M1-final guard consolidation, multi-gpu polish,
PDMRG-rules lock) landed; prior regression-watch list intact.

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | Single-host buffers live; multi-gpu purge of dead `h_batch_*_pinned`, `h_rsvd_B/U_small`, `d_boundary_staging`, `copy_boundary_mps_to_device` confirmed clean. `t_absorb_` / `t_env_update_` declared-but-unused in multi-gpu (intentional per commit). |
| B. Behavioral diff | DONE | dmrg-gpu / dmrg2-gpu share dual-stream env-update. multi-gpu's lanczos wraps all 4 pointer-mode toggles in `PointerModeGuard` (impl.h:1168/1183/1203/1314), matching pdmrg-gpu. Residual divergence: dmrg-gpu/dmrg2-gpu wire `t_apply_heff_.begin/end` and `t_lanczos_.begin/end`; pdmrg-gpu does not — see HIGH. |
| C. Docstring verification | DONE | `pdmrg_multi_gpu.h:24-31` claims `opts_.rsvd` honoured; impl reads only `use_rsvd_`. See HIGH. |
| D. clangd filter | N-A — no ROCm headers on host. |
| E. Absence-naming brief | FOLLOWED | All 11 features present in single-host triplet. multi-gpu intentionally limited per round-12 charter. |
| F. Workspace-aliasing audit | DONE | `0b9fccf` adds no new aliasing — only timers + init APIs. Surviving `d_svd_work_` (dmrg-gpu), per-device `d_rsvd_*` (multi-gpu) OK vs ctor sizes. |
| G. Sibling fix-propagation | DONE | All round-12 fixes traced four-way. 1 lonely-fix HIGH on pdmrg-gpu (dead timer panel relative to dmrg-gpu/dmrg2-gpu). |

A-G: all DONE or N-A. Review valid.

## Watch-list verification

| Fix | dmrg-gpu | dmrg2-gpu | pdmrg-gpu | pdmrg-multi-gpu |
|---|---|---|---|---|
| M1-final guard consolidation | `PointerModeGuard` (impl.h:1043,1189) | `PointerModeGuard` (impl.h:1124,1255) | `PointerModeGuard` (impl.h:1332,1356,1378,1543) | `PointerModeGuard` (impl.h:1168,1183,1203,1314) |
| Inline guards eliminated | `DmrgPointerModeGuard` gone | n/a | n/a | n/a — only comment hits in `pointer_mode_guard.h` |
| stream_env nonblocking (round-9 H1-ext) | impl.h:182,183 | impl.h:198,201 | per-segment :288 | per-device :288 |
| PointerModeGuard RAII (round-7 H7) | live | live | live | live (was MISSING, now fixed) |
| init_mps_product / init_mps_neel (round-7 H3) | h:35,36 | h:35,36 | h:36,37 | h:46,47 (was MISSING, now fixed) |
| set_mpo W-buffer guards (round-9 H-new1) | :539,572,576,611,617 | :561,591,595,643,675,681 | :743,773,777,808,814,869,899,905 | :655,659,663,710 |
| PDMRG-rules-2026-04-15 lock | n/a | n/a | run defaults `(2,2,1,0)`, polish via `_1site` | run defaults `(2,2,1,0)`, polish via `_1site` (was CRITICAL, now fixed) |
| c3d3e50 dead-buffer cleanup | clean | clean | clean | clean (was HIGH, now fixed) |
| M14-final dead set_quiet stubs | gone in dmrg-gpu-opt | gone in dmrg2-gpu-opt | n/a (still used in test) | retained no-op stub |
| pdmrg-gpu-opt lanczos_graph+davidson disable gate (round-11 leftover) | n/a | n/a | gate added at pdmrg_gpu_opt_impl.h:220 | n/a |

All entries from rounds 7/9/11/12 propagated. `0b9fccf` did not introduce any
regressions against the round-12 baseline.

## CRITICALS

None.

## HIGHS

### H13-multi-rsvd-doc — multi-gpu docstring claims `opts_.rsvd` honoured but env never wires `use_rsvd_`

**File**: `pdmrg_multi_gpu.h:24-31`. **Class**: technique C drift.

Round-12 docstring rewrite says *"only `rsvd` and (implicitly)
`lanczos_graph` are honoured"*. Impl inits `use_rsvd_=false`
(impl.h:151), calls `opts_.load_from_env()` (impl.h:159), then NEVER
reads `opts_.rsvd`. Only `set_rsvd(true)` at h:59 enables RSVD;
`DMRG_GPU_OPT_RSVD=1` is a silent no-op. The round-12 rewrite intended
to correct the previous "Extends pdmrg-gpu" overclaim and instead
introduced a fresh false claim.

**Fix**: one-liner at impl.h:159 —
`opts_.load_from_env(); use_rsvd_ = use_rsvd_ || opts_.rsvd;`. Or
revise the docstring to say "no `opts_*` flags are wired; use C++ setters."

### H13-pdmrg-phase-timer-dead — pdmrg-gpu PhaseTimer panel reports zero data

**File**: `pdmrg_gpu_impl.h:2800-2822` (init + report) — zero
`.begin/.end` callsites in the impl. **Class**: technique G (lonely
fix — multi-gpu polish raised the bar above its parent).

dmrg-gpu wires `t_apply_heff_.begin/end` (impl.h:638,666,802) and
`t_lanczos_.begin/end` (1002,1206); dmrg2-gpu wires analogues at
726,750,897,1090,1274. pdmrg-gpu's panel declares the same five
timers and calls `init_timers()` + `report_timers()` but never samples;
all rows print `total_ms=0  calls=0`. pdmrg-multi-gpu (`0b9fccf`) DID
wire timers at apply_heff_two_site / lanczos_eigensolver / svd_split
(di==0 only). pdmrg-gpu is now the lonely sibling.

**Fix**: mirror dmrg-gpu — wrap `apply_heff_*` body in
`t_apply_heff_.begin(stream_)...end(stream_)`; same for
`lanczos_eigensolver` and the `accurate_svd_gpu` boundary site.

## MEDIUMS

None new. (Round-12 deferred medium "5 of 7 GpuOpts toggles ignored in
pdmrg-multi-gpu" was addressed via the round-12 docstring rewrite —
but H13-multi-rsvd-doc above shows the rewrite itself introduced a
new technique-C drift.)

## NITS

### N13-pmg-error-discard — `PointerModeGuard` discards `rocblas_get/set_pointer_mode` return values

**File**: `gpu-rocm/common/pointer_mode_guard.h:16,17,22`. The
constructor and destructor call `rocblas_get_pointer_mode` /
`rocblas_set_pointer_mode` without checking the return status. The
destructor case is correct (cannot throw); the ctor case could
plausibly use `ROCBLAS_CHECK`. Cosmetic — these calls do not realistically
fail in practice.

## FALSE POSITIVES VERIFIED

- `t_absorb_` / `t_env_update_` declared in pdmrg-multi-gpu (h:210,211)
  but never `.begin/.end` called — verified intentional per `0b9fccf`
  commit message: *"t_absorb_/t_env_update_ timers exist for forward
  compatibility."* Same shape as the pdmrg-gpu pattern. Not a regression.
- pdmrg-multi-gpu `h_R[D_mpo_ - 1]` boundary identity slot at
  impl.h:983 — superficially looks like the round-9 D_PAD off-by-one,
  but `D_mpo_actual_` does not exist in pdmrg-multi-gpu (D_PAD
  documented as no-op per the round-12 charter rewrite). Since
  `D_mpo_ == D_mpo_actual_` always in this variant, the index is
  correct.
- pdmrg-multi-gpu `set_quiet(bool){}` no-op at h:60 — retained
  intentionally because the test driver still calls it; M14-final only
  removed the dead stubs from dmrg-gpu-opt / dmrg2-gpu-opt where the
  tests do not call it.

## SUMMARY

Round 13 surfaces 2 HIGHs, no CRITICALs. H13-pdmrg-phase-timer-dead:
pdmrg-gpu's PhaseTimer panel has been inert project-wide — `init_timers()`
and `report_timers()` fire but no samples are taken. Round-12's
multi-gpu polish wired its timers, making pdmrg-gpu the lonely sibling.
H13-multi-rsvd-doc: fresh docstring/code drift introduced by the same
`0b9fccf` rewrite — `DMRG_GPU_OPT_RSVD=1` is documented as honoured
but silently ignored. All round-12 fixes (M1-final guard consolidation,
multi-gpu polish, PDMRG-rules-2026-04-15 lock, set_mpo W-buffer guards,
nonblocking streams) verified clean four-way. Single-host -gpu tier
is paper-ready modulo the dead phase-timer wiring.

**Recommendation**: small pdmrg-gpu PhaseTimer wiring patch (~5
begin/end pairs); one-line `opts_.rsvd → use_rsvd_` patch in multi-gpu
or a docstring revision. Both sub-cluster-sized.
