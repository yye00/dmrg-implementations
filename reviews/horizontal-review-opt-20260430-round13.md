# Horizontal review — -opt tier — round 13 (2026-04-30, post-round-12-fixes)

## Charter

Review -opt tier (dmrg-gpu-opt, dmrg2-gpu-opt, pdmrg-gpu-opt). J2 contract:
each -opt is a strict superset of its -gpu sibling. J2 lock: Block-Davidson
default. Regression-watch since round-12 baseline `8b7a68e`: 0b9fccf
(M1-final guard rename + M14-final set_quiet removal) and 8b7a68e (round-12
four fixes incl. H-opt-pdmrg ctor-time `lanczos_graph` gate).

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | members of each -opt header counted vs impl; no DEAD members; round-12 cleanup left no orphans |
| B. Behavioral diff | DONE | three-way diff of `block_davidson_eigensolver`, `apply_heff(_two_site)`, sweep functions; dual-stream env-update overlap consistent in dmrg/dmrg2-opt; pdmrg uses per-segment streams (algorithm-specific, F.P.) |
| C. Docstring verification | DONE | setter comments / ctor comments cross-checked against code; ctor warning text accurately describes the failure mode in all 3 |
| D. clangd filter | N-A | clangd not invokable in sandbox |
| E. Absence-naming brief | FOLLOWED | required -opt features enumerated; all present (table below) |
| F. Workspace-aliasing audit | DONE | `d_dav_work_/work2_` traced in all 3; sizing OK; round-8 CR-D1 + round-9 H-new1 sizings preserved |
| G. Sibling fix-propagation | DONE | 8b7a68e (ctor gate), 0b9fccf (guard rename + stub removal), round-8 CR-D1, round-9 H-new1, round-7 C5+C6, round-10 M-opt-rsvd-env, round-11 M-opt-davidson-toggle all traced |

## J2 lock check

| Variant | `use_davidson_` default | Site |
|---|---|---|
| dmrg-gpu-opt | `true` (in-class init) | `dmrg_gpu_opt.h:251` |
| dmrg2-gpu-opt | `true` (in-class init) | `dmrg2_gpu_opt.h:227` |
| pdmrg-gpu-opt | `true` (ctor body) | `pdmrg_gpu_opt_impl.h:204` |

J2 lock holds in all three.

## Two-half-fix audit (round-13 new axis)

Round-12 split: **runtime setter** flipping `opts_.lanczos_graph`
symmetrically, plus **ctor-time gate** catching the env-var path
(`*_LANCZOS_GRAPH=1` via `opts_.load_from_env()`). Round-11 propagated
only the setter; ctor gate landed in dmrg/dmrg2-opt but missed
pdmrg-opt until 8b7a68e.

| Variant | Setter | Ctor gate | Tracking flag |
|---|---|---|---|
| dmrg-gpu-opt   | `dmrg_gpu_opt.h:89-97`   | `dmrg_gpu_opt_impl.h:73-79`    | `_was_user_enabled_` (h:255) |
| dmrg2-gpu-opt  | `dmrg2_gpu_opt.h:73-81`  | `dmrg2_gpu_opt_impl.h:69-75`   | (h:231) |
| pdmrg-gpu-opt  | `pdmrg_gpu_opt.h:61-72`  | `pdmrg_gpu_opt_impl.h:220-226` | (h:266) |

All three have BOTH halves. Ctor gate writes consistent warning shape,
disables `opts_.lanczos_graph`, sets the tracking flag so a later
`set_use_davidson(false)` re-enables capture symmetrically. Verified at
all six sites.

## CRITICALS — block GPU run / paper submission

(none)

## HIGHS — fix before next major event

(none)

## MEDIUMS — fix when convenient

(none)

## NITS — cosmetic

- The pdmrg-gpu-opt ctor gate prints a slightly different message string
  ("Disabling.") than the runtime setter ("disabling graph capture.").
  Cosmetic. `[pdmrg-gpu-opt: pdmrg_gpu_opt_impl.h:222 vs pdmrg_gpu_opt.h:65]`

## FALSE POSITIVES VERIFIED

- **`d_dav_work_` sizing in pdmrg-gpu-opt differs from dmrg/dmrg2-opt** —
  carried from round 12. pdmrg-gpu-opt aliases residuals in `d_dav_work`
  (`:1654, 1691`) and overlap in separate `d_dav_work2` (`:1696`), NOT
  offset-into. Round-9 sizing `max(theta*b, max_sub²)` correct for the
  separate-buffer pattern. Round-8 CR-D1 sizing
  `max(theta*b + max_sub*b, max_sub²)` preserved in dmrg-opt:281-284 and
  dmrg2-opt:265-268.

- **pdmrg-gpu-opt missing `env_update_pending_` / `stream_env_`** — not
  a J2 violation. pdmrg uses per-segment + worker streams instead of a
  side-channel env stream; algorithm-specific.

- **`set_quiet(bool){}` stub in pdmrg-gpu-opt only** —
  `[pdmrg_gpu_opt.h:76]`. Required by `test_pdmrg_gpu_opt.cpp:226, 281,
  337`. Removal from dmrg/dmrg2-opt headers verified.

- **Variant-specific guard names** — repo-wide grep across `gpu-rocm/`
  shows zero references to `AsvdPointerModeGuard`,
  `DmrgPointerModeGuard`, or `RlbfgsPointerModeGuard`. Only
  `PointerModeGuard` from `common/pointer_mode_guard.h` remains;
  pdmrg-gpu-opt's lone use at `pdmrg_gpu_opt_impl.h:2031` is canonical.

## Regression-watch summary

| Item | Status |
|---|---|
| 8b7a68e — pdmrg-gpu-opt ctor-time gate | **propagated**, impl:220-226 + comment 211-219 |
| 0b9fccf — M1-final guard consolidation | **complete**, zero variant-specific names anywhere in `gpu-rocm/` |
| 0b9fccf — M14-final set_quiet stub removal | **correct**, removed from dmrg/dmrg2-opt; retained in pdmrg-opt (test driver) |
| Round-8 CR-D1 — `d_dav_work_` overrun | preserved dmrg-opt:281-284, dmrg2-opt:265-268 |
| Round-9 H-new1 — pdmrg-gpu-opt sizing | preserved pdmrg-opt:274-276 |
| Round-7 C5+C6 — `n_recal` + `d_Vh_canonical` | present, h:48 + impl:3546, alloc:320, use:3425-3435 |
| Round-10 M-opt-rsvd-env | all 3 ctors: dmrg-opt:62, dmrg2-opt:60, pdmrg-opt:205 |
| Round-11 M-opt-davidson-toggle setter | dmrg-opt.h:89-97, dmrg2-opt.h:73-81, pdmrg-opt.h:61-72 |
| c3d3e50 — dead-buffer cleanup | clean; technique A scan: all members live |

## Self-audit

- Workspace aliasing F: traced d_dav_work, d_dav_work2 in all 3; sizing
  matches usage; no new aliasing introduced since round 12.
- Sibling propagation G: round-12 H-opt-pdmrg fix (the explicit two-half
  axis) verified propagated to all 3 variants. M1-final and M14-final
  sweeps verified. No lonely fixes detected.
- Regression watch: round-12 baseline + earlier fixes hold; no reverts.
- Verdict: READY (zero critical/high/medium findings).

## SUMMARY

Round-12's two-half-fix gap is closed: all three -opt variants now have
both the runtime `set_use_davidson` setter AND the ctor-time
`(opts_.lanczos_graph && use_davidson_)` gate, with consistent warning
text and tracking-flag wiring. The 0b9fccf cleanup landed cleanly:
zero variant-specific pointer-mode guard names remain, and the
asymmetric `set_quiet` stub retention in pdmrg-gpu-opt (test-driver
requirement) is documented. All earlier regression-watch items hold —
CR-D1, H-new1, C5+C6, M-opt-rsvd-env, M-opt-davidson-toggle. The single
NIT (minor wording drift between ctor and setter warning strings) does
not affect behavior. The -opt tier is in the cleanest state since the
review series began. Recommend no fixes before the next benchmark
window.
