# Horizontal review — -opt tier — round 15 (2026-04-30, post-5deba6d/f40140d)

## Charter

Review -opt tier (dmrg-gpu-opt, dmrg2-gpu-opt, pdmrg-gpu-opt). J2:
each -opt is a strict superset of its -gpu sibling. Regression-watch
since baseline `5deba6d` + nit `f40140d`. Round-14 landed four
-opt-relevant fixes: H-opt-pdmrg-phase-timer-prop, M-skip-on-zero-prop,
M-opt-pdmrg-pointer-mode, M-opt-svd_fallback-docstring.

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | 5 PhaseTimers DEAD in dmrg-opt and dmrg2-opt; t_absorb_ DEAD in pdmrg-opt |
| B. Behavioral diff | DONE | t_*_.{begin,end} count: dmrg-gpu 16 vs dmrg-opt 0; dmrg2-gpu 11 vs dmrg2-opt 0; pdmrg-opt 11 (round-14 fix) |
| C. Docstring verification | DONE | dmrg-opt.h:276-279 round-14 svd_fallback docstring matches impl 1224 |
| D. clangd filter | N-A | not invokable |
| E. Absence-naming brief | FOLLOWED | PhaseTimer instrumentation parity MISSING in 2 of 3 -opt |
| F. Workspace-aliasing audit | DONE | round-14 was instrumentation-only, no slicing changes; CR-D1 + H-new1 sizing preserved |
| G. Sibling fix-propagation | DONE | round-14 H-opt-pdmrg-phase-timer-prop is LONELY-SIBLING — same defect untouched in dmrg-opt and dmrg2-opt |

## Round-14 fix verification

| Fix | Site | Status |
|---|---|---|
| H-opt-pdmrg-phase-timer-prop | pdmrg-opt: 811/830/975 + 989/1064 + 1078/1149 + 1331/1479 + 1824/2020 | INTACT — 5 pairs balanced, cache-hit early-return at 830 closes timer |
| M-skip-on-zero-prop | dmrg-opt:1985, dmrg2-opt:1910, pdmrg-opt:3622 | INTACT |
| M-opt-pdmrg-pointer-mode | pdmrg-opt: 0 raw calls (was 8); 6 PointerModeGuard scopes 1834/1850/1869/2005/2049 | INTACT |
| M-opt-svd_fallback-docstring | dmrg-opt.h:276-279 | INTACT — no longer says "(CPU LAPACK)" |

Earlier baselines re-verified: J2 lock (dmrg-opt.h:251 / dmrg2-opt.h:229
/ pdmrg-opt impl:204), dual-half toggle, CR-D1 dmrg-opt:281-284 +
dmrg2-opt:265-268, H-new1 pdmrg-opt:274-276, C5+C6 (n_recal h:48 +
d_Vh_canonical impl:320/3443/3453), M-opt-rsvd-env impl:62/60/205,
c3d3e50 h_svd_* alive on use_cpu_svd_ branch.

## J2 contract — PhaseTimer instrumentation surface (technique B)

| Variant | t_lanczos_ | t_apply_heff_ | t_svd_ | t_absorb_ | t_env_update_ |
|---|---|---|---|---|---|
| dmrg-gpu-opt | DEAD | DEAD | DEAD | DEAD | DEAD |
| dmrg2-gpu-opt | DEAD | DEAD | DEAD | DEAD | DEAD |
| pdmrg-gpu-opt | live | live | live | DEAD | live |

The -gpu siblings have full instrumentation (dmrg-gpu 16 sites,
dmrg2-gpu 11 sites). Round-14 fix only propagated to pdmrg-gpu-opt.

## CRITICALS

(none)

## HIGHS

- **[dmrg-gpu-opt: dmrg_gpu_opt_impl.h:745, 842, 971, 1224, 1478]
  All five PhaseTimers dead — round-14 H-opt-pdmrg-phase-timer-prop
  not propagated.** dmrg-gpu has 16 begin/end sites in apply_heff
  (639/803), update_left_env (816/881), update_right_env
  (891/954), lanczos (1003/1207), svd (1217/1367 + 1414), absorb
  (1368/1394 + 1415/1440). dmrg-gpu-opt has the same hot-path
  functions and ZERO begin/end calls. Round-14 fixed this exact
  defect-class as HIGH in pdmrg-gpu-opt; this is the lonely sibling.
  M-skip-on-zero-prop hides the symptom (panel header alone with no
  rows) but does not repair instrumentation. Action: instrument all
  five timers; mind the `lanczos_graph` cache-hit early return at
  apply_heff:594 (close timer before `return;`, mirroring
  pdmrg-gpu-opt:830).

- **[dmrg2-gpu-opt: same defect-class as dmrg-gpu-opt]** dmrg2-gpu
  has 11 begin/end sites (apply_heff 727/898, env_update 911/978 +
  988/1052, lanczos 1091/1275, svd 1285/1450). dmrg2-gpu-opt has
  zero. Same fix needed; dmrg2's apply_heff also has a cache-hit
  early return at 751 that the propagation must close.

## MEDIUMS

- **[pdmrg-gpu-opt: pdmrg_gpu_opt.h:253] t_absorb_ declared, init'd
  at impl:3614, reported at impl:3633 — no begin/end sites
  anywhere.** Round-14 instrumented 4 of 5 timers; t_absorb_
  missed. pdmrg-gpu also doesn't instrument t_absorb_, so this is
  not strict J2 violation — dead infrastructure on both pdmrg
  variants. Either remove from both, or add begin/end around the
  absorb GEMM in svd_split.

## NITS

(round-13 wording-drift between pdmrg-opt ctor warning at impl:222-223
and setter at h:64-66 preserved, not regressed.)

## FALSE POSITIVES VERIFIED

- **set_quiet stub in pdmrg_gpu_opt.h:76** — by-design asymmetry;
  test driver test_pdmrg_gpu_opt.cpp:226/281/337 calls it. Carried
  forward.
- **`d_dav_work_` sizing differs across -opt** — CR-D1 form for
  dmrg/dmrg2-opt (concurrent), H-new1 form for pdmrg-opt
  (sequential). Both correct for their aliasing patterns.
- **pdmrg-opt absent `env_update_pending_` / `stream_env_`** —
  algorithm-specific (per-segment + worker streams). Not J2.

## Self-audit

- **F**: round-14 was instrumentation + docstring + guard-rename
  only — zero buffer slicing. d_dav_work/d_dav_work2 sizing matches
  usage in all 3.
- **G**: H-opt-pdmrg-phase-timer-prop propagated to pdmrg-opt but
  not to dmrg-opt/dmrg2-opt → two HIGH findings above. Other three
  round-14 fixes correctly scoped.
- **Regression watch**: zero references to legacy guard names
  anywhere in gpu-rocm/. Earlier fixes intact.
- **Verdict**: NOT READY. Two HIGHs (sibling propagation gaps);
  fix before next benchmark window.

## SUMMARY

Round 15 finds the round-14 H-opt-pdmrg-phase-timer-prop is a
textbook lonely-sibling fix: pdmrg-gpu-opt got 5 PhaseTimer pairs
instrumented (with cache-hit early-return handled),
M-skip-on-zero-prop landed in all 3 -opt, 8 raw pointer-mode calls
became 6 guards, and the dmrg-opt svd_fallback docstring was
corrected — all four round-14 fixes INTACT. But dmrg-gpu-opt and
dmrg2-gpu-opt still have ALL five timers declared+init+reported
with zero begin/end calls, while their -gpu siblings have 16 and
11 sites respectively. M-skip-on-zero-prop suppresses the symptom
(silent panel) but the fix's stated purpose — profiling parity
with -gpu — is unmet on 2 of 3 variants. Action: propagate the
H-opt-pdmrg-phase-timer-prop pattern to dmrg-gpu-opt and
dmrg2-gpu-opt at the hot-path sites listed in the HIGHs (mind
cache-hit early returns at apply_heff:594 and dmrg2 apply_heff:751).
Separate MEDIUM follow-up: t_absorb_ dead in pdmrg-opt within
round-14's own scope.
