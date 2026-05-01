# Horizontal review — -opt tier — round 14 (2026-05-01, post-f5c0617)

## Charter

Review the -opt tier (dmrg-gpu-opt, dmrg2-gpu-opt, pdmrg-gpu-opt). J2 contract:
each -opt is a strict superset of its -gpu sibling. Regression-watch since
round-13 baseline `ee653f0`: the only post-baseline commit is `f5c0617`
(round-13 nit cleanup — orphan comment in dmrg2-gpu.h + tightened
`h_svd_*` docstring in dmrg2-gpu-opt.h). Round-13 axis-3 lesson:
docstring promise = half-fix; verify the new docstring against the code it
describes.

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | members of all 3 -opt headers grep-counted; no DEAD members; round-12/13 cleanups left no orphans |
| B. Behavioral diff | DONE | `block_davidson_eigensolver`, `apply_heff(_two_site)`, sweep functions three-way diff; setter pattern uniform; pdmrg uses per-segment + worker streams (algorithm-specific F.P.) |
| C. Docstring verification | DONE | f5c0617 dmrg2-gpu-opt.h:189-194 docstring vs impl 1263-1285 — claim verified, every named buffer (h_svd_A_/U_/Vh_/work_/S_, plus h_svd_rwork_) is read on the use_cpu_svd_ branch |
| D. clangd filter | N-A | clangd not invokable in sandbox |
| E. Absence-naming brief | FOLLOWED | required -opt features enumerated below; all present |
| F. Workspace-aliasing audit | DONE | f5c0617 is docs-only — no buffer-sizing change; round-8 CR-D1 + round-9 H-new1 sizing preserved in all 3; re-verified |
| G. Sibling fix-propagation | DONE | f5c0617 itself is a documentation propagation (round-13 axis-3); 8b7a68e (ctor gate), 0b9fccf (guard rename + stub removal), CR-D1, H-new1, C5+C6, M-opt-rsvd-env, M-opt-davidson-toggle all traced |

## Round-13 axis-3 verification (the only new axis this round)

Round-13 lesson: docstring promise = half-fix. f5c0617 changed dmrg2-gpu-opt.h
189-194 from "legacy — only used by the init-time workspace query;
runtime SVD is fully on-device" to a docstring naming the use_cpu_svd_
opt-in fallback path at "impl ~1263-1283." I do not trust the docstring
text; I grep-checked the named line range:

| Member | Read site (use_cpu_svd_ branch) |
|---|---|
| `h_svd_A_` | `dmrg2_gpu_opt_impl.h:1263, 1270` (D2H + LAPACK input) |
| `h_svd_U_` | `dmrg2_gpu_opt_impl.h:1271, 1280` (LAPACK output + H2D) |
| `h_svd_Vh_` | `dmrg2_gpu_opt_impl.h:1271, 1283` (LAPACK output + H2D) |
| `h_svd_work_` | `dmrg2_gpu_opt_impl.h:1267, 1272` (lwork query + LAPACK scratch) |
| `h_svd_S_` | `dmrg2_gpu_opt_impl.h:1271, 1277` (LAPACK output + H2D) |
| `h_svd_rwork_` | `dmrg2_gpu_opt_impl.h:1273` (complex-only LAPACK rwork) |

The branch begins at `if (use_cpu_svd_) {` (impl:1259) and ends at
`} else if (!used_rsvd) {` (impl:1286). Every member named in the
docstring is read on this branch. The docstring is now consistent with
the code — round-13 axis-3 closed for dmrg2-gpu-opt.

The sibling dmrg-gpu-opt.h:213 has a tighter one-liner ("fallback — only
used when use_cpu_svd_ opt-in flag set") and impl:1247-1278 mirrors the
dmrg2 branch with the same six members read. Consistent.

## J2 lock check

| Variant | `use_davidson_` default | Site |
|---|---|---|
| dmrg-gpu-opt | `true` (in-class init) | `dmrg_gpu_opt.h:251` |
| dmrg2-gpu-opt | `true` (in-class init) | `dmrg2_gpu_opt.h:229` |
| pdmrg-gpu-opt | `true` (ctor body) | `pdmrg_gpu_opt_impl.h:204` |

## Two-half-fix audit

| Variant | Setter | Ctor gate | Tracking flag |
|---|---|---|---|
| dmrg-gpu-opt   | `dmrg_gpu_opt.h:89-97`   | `dmrg_gpu_opt_impl.h:73-79`    | `_was_user_enabled_` (h:255) |
| dmrg2-gpu-opt  | `dmrg2_gpu_opt.h:73-81`  | `dmrg2_gpu_opt_impl.h:69-75`   | (h:233) |
| pdmrg-gpu-opt  | `pdmrg_gpu_opt.h:61-72`  | `pdmrg_gpu_opt_impl.h:220-226` | (h:266) |

All three have BOTH halves with consistent `lanczos_graph_was_user_enabled_`
tracking. Symmetric round-trip: `set_use_davidson(false)` re-enables
`opts_.lanczos_graph` if and only if the ctor gate or an earlier
`set_use_davidson(true)` flipped it off. Verified at all six sites.

## Required -opt feature set (technique E)

| Feature | dmrg-opt | dmrg2-opt | pdmrg-opt |
|---|---|---|---|
| pad_mfma16 helper | present | present | present |
| chi_max_user_ | present | present | present |
| use_davidson_ default true | present | present | present |
| set_use_davidson + ctor gate | present | present | present |
| Block-Davidson + Lanczos fallback | present | present | present |
| Strided/batched Step-3 GEMMs | present | present | present |
| set_cpu_svd | present | present | present |
| set_rsvd | present | present | present |
| set_use_batched_sweep | N-A | N-A | present (live, impl:3508/3521) |
| set_use_chebyshev | N-A | N-A | present (live, impl:2265) |
| set_quiet stub | (removed M14) | (removed M14) | present (test-driver only) |
| use_rsvd_ = opts_.rsvd | impl:62 | impl:60 | impl:205 |
| n_recal + d_Vh_canonical | N-A | N-A | h:48 + impl:320/3425/3435 |
| Per-segment + worker streams | N-A | N-A | live (h:210, impl uses) |

All expected features present; nothing flagged DEAD by technique A.

## CRITICALS — block GPU run / paper submission

(none)

## HIGHS — fix before next major event

(none)

## MEDIUMS — fix when convenient

(none)

## NITS — cosmetic

(none — round-13's lone NIT was a wording-drift between pdmrg-gpu-opt
ctor and setter warning strings; preserved as-is and not regressed
this round.)

## FALSE POSITIVES VERIFIED

- **dmrg-gpu-opt.h:213 docstring is shorter than dmrg2-gpu-opt.h:189-194**
  — verification: dmrg-gpu-opt's one-line "fallback — only used when
  use_cpu_svd_ opt-in flag set" already names the gating flag; the
  expanded dmrg2 wording is for the dmrg2 nit lineage, not a defect class
  needing propagation. Both correctly describe their CPU-SVD branches.
  Not a finding.

- **`set_quiet(bool){}` stub asymmetry across -opt** — `pdmrg_gpu_opt.h:76`
  retains the no-op stub; dmrg/dmrg2-opt removed theirs in M14-final.
  `test_pdmrg_gpu_opt.cpp:226, 281, 337` calls it; the asymmetry is by
  design and documented. Preserved from round-13.

- **pdmrg-gpu-opt missing `env_update_pending_` / `stream_env_`** —
  algorithm-specific (per-segment + worker streams instead of side-channel
  env stream). Not a J2 violation.

- **`d_dav_work_` sizing differs** — dmrg-opt + dmrg2-opt use the round-8
  CR-D1 form `max(theta·b + max_sub·b, max_sub²)` (concurrent residual +
  overlap regions, sites 281-284 and 265-268). pdmrg-opt uses the round-9
  H-new1 form `max(theta·b, max_sub²)` (sequential, separate
  d_dav_work2_ at impl:1696). Both forms are correct for their respective
  aliasing patterns. Carried forward from round-13.

## Regression-watch summary

| Item | Status |
|---|---|
| f5c0617 — dmrg2-gpu-opt h_svd_* docstring | **consistent with impl 1263-1285** |
| 8b7a68e — pdmrg-gpu-opt ctor-time gate | preserved, impl:220-226 |
| 0b9fccf — M1-final guard consolidation | preserved; zero variant-specific guard names anywhere in `gpu-rocm/` (only doc references in `common/pointer_mode_guard.h`) |
| 0b9fccf — M14-final set_quiet stub removal | preserved; pdmrg-opt retains stub for test driver |
| Round-8 CR-D1 — `d_dav_work_` overrun | preserved dmrg-opt:281-284, dmrg2-opt:265-268 |
| Round-9 H-new1 — pdmrg-gpu-opt sizing | preserved pdmrg-opt:274-276 |
| Round-7 C5+C6 — `n_recal` + `d_Vh_canonical` | preserved, h:48 + impl:320/3425/3435 |
| Round-10 M-opt-rsvd-env | all 3 ctors: dmrg-opt:62, dmrg2-opt:60, pdmrg-opt:205 |
| Round-11 M-opt-davidson-toggle | all 3 setters, all 3 ctor gates |
| c3d3e50 — dead-buffer cleanup | clean; all surviving h_svd_* alive in CPU-SVD branch (verified line-by-line) |

## Self-audit

- Workspace aliasing F: f5c0617 is docs-only; no buffer slicing changed.
  Re-traced d_dav_work, d_dav_work2 in all 3 — sizing matches usage; no
  new aliasing introduced since round 13. CR-D1 form preserved in
  dmrg/dmrg2-opt; H-new1 form preserved in pdmrg-opt.
- Sibling propagation G: f5c0617 itself is a documentation propagation
  from round-13 axis-3 (the dmrg2-gpu-opt.h:189-194 docstring now matches
  the implemented behavior). dmrg-gpu-opt.h:213 was already correct.
  pdmrg-gpu-opt has no equivalent h_svd_* members (per-stream worker
  pool ws.h_svd_A etc., not class-level). No further propagation owed.
- Regression watch: round-13 baseline + earlier fixes hold; no reverts.
  Verified zero references to `AsvdPointerModeGuard`, `DmrgPointerModeGuard`,
  `RlbfgsPointerModeGuard` in any gpu-rocm code.
- Verdict: READY (zero critical/high/medium/nit findings).

## SUMMARY

Round 14 is a confirmation pass. The single post-baseline commit (f5c0617)
tightened a docstring on the dmrg2-gpu-opt h_svd_* members to name the
use_cpu_svd_ opt-in fallback path at impl ~1263-1283. Per the round-13
axis-3 lesson (docstring promise = half-fix), I grep-verified every
named buffer is actually read on that branch — six-for-six. All other
regression-watch items hold: J2 lock (use_davidson_ = true), dual-half
toggle in all three -opt variants with symmetric warning text, CR-D1 +
H-new1 Davidson workspace sizing, C5+C6 n_recal + d_Vh_canonical,
M-opt-rsvd-env binding, M1-final guard consolidation, M14-final stub
asymmetry. The -opt tier remains in the cleanest state of the review
series. Recommend no fixes before the next benchmark window.
