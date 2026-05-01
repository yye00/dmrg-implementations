# Horizontal review — -gpu tier — round 14 (2026-05-01)

HEAD: `f5c0617` (round-13 nit cleanup, post-`ee653f0` round-13 fixes).
Diff vs round-13 baseline (`0b9fccf`): two small commits — round-13 HIGH+MEDIUM fixes
plus a one-comment cleanup in `dmrg2-gpu/src/dmrg2_gpu.h`.

## Charter

Review **-gpu tier** across `dmrg-gpu`, `dmrg2-gpu`, `pdmrg-gpu`, and
`pdmrg-multi-gpu`. Verify the round-13 fixes (PhaseTimer wiring in
pdmrg-gpu; `use_rsvd_ = opts_.rsvd` in pdmrg-multi-gpu;
report_timers skip-on-zero in multi-gpu; `set_use_davidson` cross-ref
removal) all landed; prior regression-watch list intact.

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | Ctor-and-dtor-only members tracked; pdmrg-gpu's `t_absorb_` remains intentionally declared-but-unused (forward-compat, matches multi-gpu). |
| B. Behavioral diff | DONE | dmrg-gpu / dmrg2-gpu share dual-stream env-update; pdmrg-gpu now wires `t_apply_heff_/t_env_update_/t_lanczos_/t_svd_` at single-segment gating (5 pairs). multi-gpu mirrors with `di == 0`. |
| C. Docstring verification | DONE | `pdmrg_multi_gpu.h:24` claim "only `rsvd` and `profile` honoured" now backed by `use_rsvd_=opts_.rsvd` (impl.h:163) and `init_timers()` reading `opts_.profile` (h:214-218). H13-multi-rsvd-doc closed. |
| D. clangd filter | N-A — no ROCm headers on host. |
| E. Absence-naming brief | FOLLOWED | All 11 features from the -gpu spec list present in single-host triplet. multi-gpu intentionally limited per round-12 charter, now backed by code. |
| F. Workspace-aliasing audit | DONE | `ee653f0` and `f5c0617` introduced no new aliasing — only timer instrumentation, a one-liner `use_rsvd_` binding, and a comment removal. No buffer-sizing risk. |
| G. Sibling fix-propagation | DONE | All round-13 fixes traced four-way (table below). 0 lonely-fix findings remain. |

A-G: all DONE or N-A. Review valid.

## Round-13 fix verification

| Fix | dmrg-gpu | dmrg2-gpu | pdmrg-gpu | pdmrg-multi-gpu |
|---|---|---|---|---|
| H13-pdmrg-phase-timer wiring | already wired (impl.h:638,802,1002,1206,1216,815) | already wired (726,897,1090,1274,1284) | **NEW (impl.h:959,1135,1149,1211,1225,1283,1330,1570,1583,1739)** | already wired (757,818,1165,1334,1347,1479) |
| H13-multi-rsvd-prop binding | n/a (reads opts_.rsvd directly :306,1233) | n/a (reads opts_.rsvd directly :328,1302) | n/a (set_rsvd writes opts_.rsvd; reads :1908) | **NEW: use_rsvd_ = opts_.rsvd (impl.h:163)** |
| report_timers skip-on-zero | unchanged (prints all 5) | unchanged (prints all 5) | unchanged (prints all 5) | **NEW: print_if_used (h:227-236)** |
| set_use_davidson cross-ref removed | n/a | n/a | n/a | docstring no longer mentions it (h:24-31) |
| f5c0617 orphan `// CPU workspace` | n/a | gone (was at h:154) | n/a | n/a |

## Regression-watch (round-9..round-12 baseline)

| Watch item | dmrg-gpu | dmrg2-gpu | pdmrg-gpu | multi-gpu |
|---|---|---|---|---|
| `stream_env_` nonblocking | :182,183 | :198,201 | per-seg :288 | per-dev :293 |
| `PointerModeGuard` sites | 2 | 2 | 4 | 4 |
| `init_mps_product/_neel` | h:35,36 | h:35,36 | h:36,37 | h:47,48 |
| set_mpo W-buffer guards | live | live | live | live |
| `d_svd_work_` alive | :1369-1431 | n/a | n/a | n/a |
| PDMRG-rules-2026-04-15 lock | n/a | n/a | run `(2,2,1,0,0)`; `_1site` | run `(2,2,1,0)`; `_1site` |
| D_PAD `D_mpo_actual_-1` slot | :976 | :1071 | :1306 | n/a |

All round-9..round-12 watch items intact. No regressions.

## Specific verification of round-13 brief items

1. `pdmrg-multi-gpu use_rsvd_` binding — at `pdmrg_multi_gpu_impl.h:163`,
   immediately after `opts_.load_from_env()`. ✅
2. `pdmrg-gpu PhaseTimer` 5 sites — 10 begin/end lines, balanced and
   `streams_[si]`-consistent: apply_heff_two_site 959/1135,
   update_left_env 1149/1211, update_right_env 1225/1283,
   lanczos_eigensolver 1330/1570, svd_split 1583/1739. ✅
3. `pdmrg-multi-gpu report_timers` skip — `print_if_used` lambda
   short-circuits when `t.calls() == 0` (h:228). ✅
4. `pdmrg-multi-gpu` docstring — line 25 reads "Lanczos-only (no
   Davidson path)" with no `set_use_davidson` cross-reference. ✅

## CRITICALS

None.

## HIGHS

None.

## MEDIUMS

None new. Round-13's 4 mediums (M1-base-prop, M14-base-prop, n_recal API
gap, multi-gpu dead timers report_timers skip) addressed in `ee653f0`.

## NITS

### N14-report-timers-skip-prop — pdmrg-gpu / dmrg-gpu / dmrg2-gpu `report_timers` could adopt skip-on-zero

**File**: `pdmrg_gpu_impl.h:2832-2836` (also `dmrg_gpu_impl.h`,
`dmrg2_gpu_impl.h`). **Class**: technique G (consistency).

`pdmrg-multi-gpu`'s round-13 polish added a `print_if_used` lambda that
suppresses uninstrumented phases (calls()==0). Single-host -gpu siblings
still print all 5 phases unconditionally — so when `t_absorb_` is
intentionally uninstrumented in pdmrg-gpu / dmrg2-gpu (forward-compat
scaffolding), the panel prints `absorb : 0.00 ms (0 calls)`. Cosmetic;
fix is a 4-line diff in each impl.h. Mirror multi-gpu's `print_if_used`.

### N13-pmg-error-discard (carry-over from round-13)

**File**: `gpu-rocm/common/pointer_mode_guard.h:16,17,22`. The
constructor and destructor call `rocblas_get_pointer_mode` /
`rocblas_set_pointer_mode` without checking return status. Destructor
case is correct (cannot throw); ctor case could plausibly use
`ROCBLAS_CHECK`. Cosmetic — these calls do not realistically fail.
Round-13 did not address this; carry forward.

## FALSE POSITIVES VERIFIED

- pdmrg-gpu `t_absorb_` declared (h:215) but not `.begin/.end` wired —
  intentional. Round-13 brief explicitly listed only 5 sites
  (apply_heff_two_site, update_left_env, update_right_env,
  lanczos_eigensolver, svd_split). `t_absorb_` is forward-compat
  scaffolding, mirroring multi-gpu's `t_absorb_/t_env_update_` pattern.
- dmrg2-gpu `t_absorb_` similarly declared but unwired — long-standing
  pattern (round-12 baseline noted it). Not a round-13 regression.
- pdmrg-multi-gpu `h_R[D_mpo_ - 1]` boundary identity slot (impl.h:983)
  — verified `D_PAD` is documented as no-op in multi-gpu (round-12
  charter rewrite); `D_mpo_ == D_mpo_actual_` always here, so the
  index is correct.
- Recalibration phase in pdmrg-gpu (`sweep_LR_full()` at impl.h:2755)
  uses two-site sweeps — opt-in via `n_recal > 0`, NOT warmup or
  polish. PDMRG-rules lock applies to warmup/polish only; recalibration
  is a separate phase. Compliant.

## SUMMARY

Round 14: 0 CRITICAL, 0 HIGH, 0 MEDIUM, 1 NEW NIT + 1 carry-over.
Round-13's two HIGHs verified fixed at the exact sites and counts:
10 begin/end lines balanced and `streams_[si]`-consistent in pdmrg-gpu;
`use_rsvd_ = opts_.rsvd` in multi-gpu's ctor; `report_timers`
skip-on-zero in multi-gpu; `set_use_davidson` cross-reference removed.
The round-13 axis-3 lesson (docstring promise = half-fix) is satisfied:
multi-gpu's GpuOpts scope claim now matches code in both directions
(rsvd binding present; profile read in `init_timers`). N14 suggests
propagating multi-gpu's `print_if_used` skip-on-zero pattern to the
single-host siblings — cosmetic only.

**Recommendation**: -gpu tier is paper-ready. No GPU-run blockers.
