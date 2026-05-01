# Horizontal review — -gpu tier — round 15 (2026-05-01)

HEAD: `f40140d` (round-14 nit on `set_quiet` stub comment).
Diff vs round-14 baseline (`5deba6d`): one comment-only commit. The
substantive round-14 fixes (timer-leak close + skip-on-zero rollout)
arrived inside `5deba6d` itself; round-15 verifies them.

## Charter

Review **-gpu tier** across `dmrg-gpu`, `dmrg2-gpu`, `pdmrg-gpu`, and
`pdmrg-multi-gpu`. Verify the round-14 fixes
(H-pdmrg-apply_heff-cache-hit-leak, M-skip-on-zero across the 4 siblings,
plus the f40140d cosmetic alignment) all landed; round-9..round-13
regression-watch list intact.

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | No new dead-infrastructure introduced; pdmrg-gpu `t_absorb_` remains intentional forward-compat (matches multi-gpu) — long-standing, not new. |
| B. Behavioral diff | DONE | Cache-hit early-return now balanced across all three single-host -gpu siblings (dmrg-gpu:667, dmrg2-gpu:751, pdmrg-gpu:983). multi-gpu has no graph cache → genuinely immune. |
| C. Docstring verification | DONE | f40140d aligns `set_quiet(bool)` comment to `// no-op` across four siblings (dmrg-gpu.h:43, dmrg2-gpu.h:44, pdmrg-gpu.h:53, pdmrg_multi_gpu.h:65). Round-14 N14 closed. |
| D. clangd filter | N-A — no ROCm headers on host. |
| E. Absence-naming brief | FOLLOWED | All 11 -gpu spec features still present in single-host triplet; multi-gpu intentionally limited per round-12 charter. |
| F. Workspace-aliasing audit | DONE | Round-14 deltas are 1-2 line timer/comment edits in 6 files. No buffer-sizing surface touched. Zero OVERRUN. |
| G. Sibling fix-propagation | DONE | Both round-14 fixes traced four-way (table below). 1 partial-instrumentation MEDIUM noted (apply_heff_single_site uninstrumented). |

A-G: all DONE or N-A. Review valid.

## Round-14 fix verification

| Fix | dmrg-gpu | dmrg2-gpu | pdmrg-gpu | pdmrg-multi-gpu |
|---|---|---|---|---|
| H-pdmrg-apply_heff cache-hit timer leak | already balanced (impl.h:667 end before :668 return) | already balanced (impl.h:751 end before :752 return) | **NEW: end before return at impl.h:983-984** | immune (no graph cache) |
| M-skip-on-zero in `report_timers` | **NEW: impl.h:430** | **NEW: impl.h:365** | **NEW: impl.h:2831** | already had it (h:232 `print_if_used`, round-13) |
| f40140d set_quiet stub uniformity | **NEW: dmrg_gpu.h:43 `// no-op`** | already `// no-op` (h:44) | already `// no-op` (h:53) | already `// no-op` (h:65) |

## Regression-watch (round-9..round-13)

| Watch item | dmrg-gpu | dmrg2-gpu | pdmrg-gpu | multi-gpu |
|---|---|---|---|---|
| `stream_env_` nonblocking | :182,183 | :198,201 | per-seg | per-dev :293 |
| `PointerModeGuard` sites | 2 | 2 | 4 | 4 |
| `init_mps_product/_neel` | h:35,36 | h:35,36 | h:36,37 | impl.h:596,613 |
| `set_mpo` W-buffer guards | live | live | live | live |
| `d_svd_work_` alive | :1369-1431 | n/a | n/a | n/a |
| PDMRG-rules lock | n/a | n/a | `_1site`, ≤2 | `_1site`, ≤2 |
| D_PAD `D_mpo_actual_-1` slot | :976 | :1071 | :1306 | n/a |
| use_rsvd_ = opts_.rsvd | direct read | direct read | direct read | impl.h:163 |
| PhaseTimer panel | intact | intact | 5 pairs + cache-hit end balanced | intact |

All intact. No regressions.

## Specific verification of round-14 brief items

1. pdmrg-gpu cache-hit `t_apply_heff_.end()` at impl.h:983 (before
   `return;` at 984), guarded by `if (si == 0)` matching begin at 959.
   Begin/end pairs across panel remain balanced (begins 959, 1153, 1229,
   1334, 1587; ends 983, 1139, 1215, 1287, 1574, 1743 — surplus end on
   cache-hit branch that itself returns).
2. `report_timers` skip-on-zero confirmed in all 4 siblings:
   `dmrg_gpu_impl.h:430`, `dmrg2_gpu_impl.h:365`, `pdmrg_gpu_impl.h:2831`,
   `pdmrg_multi_gpu.h:232` (round-13 `print_if_used` predates).
3. f40140d nit — `dmrg_gpu.h:43` `// no-op`, matches all siblings.

## CRITICALS

None.

## HIGHS

None.

## MEDIUMS

### M15-pdmrg-single-site-uninstrumented — `apply_heff_single_site` lacks PhaseTimer wrapping

**File**: `gpu-rocm/pdmrg-gpu/src/pdmrg_gpu_impl.h:1928-2010`. **Class**:
technique B (intra-variant sibling-pair diff between
`apply_heff_two_site` and `apply_heff_single_site`).

`apply_heff_two_site` was instrumented in round-13 (begin 959, end 1139,
plus the round-14 cache-hit end at 983). The sibling
`apply_heff_single_site` (impl.h:1928) has identical control-flow shape
including a `lanczos_graph` cache-hit early return at impl.h:1953 — but
no PhaseTimer instrumentation. Warmup / polish sweeps (driven by
`lanczos_use_1site_ = true` at impl.h:2287) therefore contribute zero
apply_heff calls/time to the panel. Per PDMRG-rules-2026-04-15, warmup
is up to 2 sweeps × L sites × Lanczos iters — non-trivial. Fix: add
`if (si == 0) t_apply_heff_.begin/.end` at entry, before the cache-hit
return, and at normal exit, mirroring 959/983/1139. No incorrectness,
just a measurement gap (round-13 campaign target missed on the
single-site path).

## NITS

### N13-pmg-error-discard (carry-over from round-13/14)

**File**: `gpu-rocm/common/pointer_mode_guard.h:16,17,22`. The constructor
and destructor call `rocblas_get_pointer_mode` / `rocblas_set_pointer_mode`
without checking return status. Cosmetic — these calls do not realistically
fail. Carry forward; round-14 did not address this.

## FALSE POSITIVES VERIFIED

- pdmrg-gpu `t_absorb_` declared but `.begin/.end`-unwired — intentional
  forward-compat scaffolding (matches multi-gpu). Round-15 skip-on-zero
  rollout mutes the dead-timer print noise.
- `apply_heff_single_site` cache-hit early return (impl.h:1953) does NOT
  need a `.end()` call alone (see M15). Correct fix is both begin+end.
- dmrg-gpu / dmrg2-gpu cache-hit early returns (impl.h:667, 751) already
  closed `t_apply_heff_` before `return;` — never affected by H-pdmrg.
- pdmrg-multi-gpu has no `apply_heff_graph_cache_` / `hipGraphLaunch` —
  genuinely immune to the cache-hit leak class.
- `if (si == 0)` guard on cache-hit end is consistent with begin at 959;
  PhaseTimer's std::vector is not thread-safe across parallel segments.

## SUMMARY

Round 15: 0 CRITICAL, 0 HIGH, 1 NEW MEDIUM (M15-single-site-
uninstrumented), 1 carry-over NIT. Round-14's two fixes verified at
exact sites: cache-hit early-return closes `t_apply_heff_` in pdmrg-gpu
(impl.h:983); skip-on-zero rolled out to dmrg-gpu:430, dmrg2-gpu:365,
pdmrg-gpu:2831; multi-gpu retains round-13 `print_if_used`. f40140d
brings `set_quiet` stub comments into uniformity. No regressions in
round-9..round-13 watch items.

The new M15 is technique-G in spirit: the round-13 PhaseTimer campaign
plus round-14 cache-hit fix should propagate to `apply_heff_single_site`
(warmup/polish path). Severity MEDIUM, not HIGH — under-reporting, not
incorrectness. Not a GPU-run blocker.

**Recommendation**: -gpu tier is paper-ready.
