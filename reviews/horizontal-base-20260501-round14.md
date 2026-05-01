# Horizontal review — -base tier — round 14 (2026-05-01)

Scope: dmrg-gpu-base, dmrg2-gpu-base, pdmrg-gpu-base. Baseline:
round-13 horizontal-base report at ee653f0. Two commits since
(ee653f0 round-13 fixes; f5c0617 round-13 nit cleanup). f5c0617
touched only -gpu/-opt sites — no -base changes.

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | 0 unused private members across 3 variants |
| B. Behavioral diff | DONE | Lanczos-eigensolver structure now identical across the three (PointerModeGuard scope shape matches) |
| C. Docstring verification | DONE | 1 minor drift surfaced (NIT, dmrg2-gpu-base d_WW_ comment) |
| D. clangd filter | N-A | clangd not invokable on this host; A subsumes most |
| E. Absence-naming brief | FOLLOWED | -gpu-tier features absent from all three |
| F. Workspace-aliasing audit | DONE | 5 shared scratch buffers checked, 0 OVERRUN |
| G. Sibling fix-propagation | DONE | 3 round-13 fixes traced, 0 MISSING in -base |

## CRITICALS — block GPU run / paper submission

(none)

## HIGHS — fix before next major event

(none)

## MEDIUMS — fix when convenient

(none net-new for -base this round)

## NITS — cosmetic

- **dmrg2-gpu-base header comment drift on d_WW_**
  [dmrg2-gpu-base: dmrg2_gpu_base.h:83-86]. Per-member comment reads
  `"precomputed on device at set_mpo() time"`, but `precompute_WW`
  (dmrg2_gpu_base_impl.h:265-310) is host-side (6 nested C++ loops +
  one H2D upload). The class-level docstring at lines 28-34 is
  correct (round-7 H5 explicitly says "host-side in EVERY tier"); the
  per-member comment escaped that correction. Round-13 axis-3 lesson
  ("docstring promise = half-fix") applies. Reword to "precomputed on
  host at set_mpo() time, then uploaded once". Cosmetic only.

## FALSE POSITIVES VERIFIED

- "pdmrg-gpu-base svd_split uses plain gesvd_auto — J1 violation."
  Not a defect (carries from round-13): accurate_svd is mandatory at
  boundary merges only. Boundary call confirmed at impl.h:1271-1278.
- "h_WL/h_WR built per-call on host inside set_mpo." set_mpo is
  one-shot init, outside the timed sweep region — by design.

## Regression-watch verification

| Round | Fix | Status |
|---|---|---|
| 13 M1-base-prop | PointerModeGuard adoption in 3 -base variants | INTACT — `grep -rn "rocblas_set_pointer_mode" gpu-rocm/{dmrg,dmrg2,pdmrg}-gpu-base/` returns 0; PointerModeGuard scopes confirmed at dmrg_gpu_base_impl.h:521,625; dmrg2_gpu_base_impl.h:589,681; pdmrg_gpu_base_impl.h:745,846 |
| 13 M14-base-prop | dead set_quiet stubs removed | INTACT — `grep -n "set_quiet" gpu-rocm/{dmrg,dmrg2,pdmrg}-gpu-base/src/*.h` returns 0 hits |
| 13 NIT (pdmrg-base docstring) | "default 1 each" → explicit n_warmup=1, n_polish=0 | INTACT — pdmrg_gpu_base.h:42 reads `"defaults n_warmup=1, n_polish=0; caller must supply n_polish explicitly to enable the polish phase"` |
| 12 M-base-stoudenmire (J1) | accurate_svd_gpu boundary call in pdmrg-gpu-base | INTACT — pdmrg_gpu_base_impl.h:1271; AsvdScratch ws.asvd live at line 168/222/1278 |
| 12 M4-W set_mpo guards | free-then-realloc on d_W_left_/d_W_right_/d_WW_ | INTACT — dmrg_gpu_base_impl.h:245,249; dmrg2_gpu_base_impl.h:243,247; pdmrg_gpu_base_impl.h:343,347,398 |
| 9 MED-base-1 | dead d_svd_work_ in dmrg2-gpu-base removed | INTACT — `grep` returns 0 hits in dmrg2-gpu-base (header and impl) |
| 8 C-new1 | pdmrg-gpu-base R_env from canonical Vh | INTACT — pdmrg_gpu_base_impl.h:1325-1343, the d_Vh_canonical swap pattern matches pdmrg-gpu and pdmrg-gpu-opt |

## Technique-A & technique-F worksheet (audit evidence)

A — every private member of all three classes has ≥3 grep hits in
its impl.h (alloc + free + ≥1 use). pdmrg-gpu-base StreamWorkspace
counts: d_T1=7, d_T2=10, d_theta=17, d_svd_U=10, d_svd_Vh=12,
d_Vh_canonical=5, asvd=3, etc. No declared-but-unused infrastructure
remains.

F — shared scratch buffers checked:

| Buffer | Variant | Regions | Lifetimes | Required | Allocated | Verdict |
|---|---|---|---|---|---|---|
| d_T1_/d_T2_ | dmrg-gpu-base | apply_heff Step1/2 + update_*_env V/U | sequential within hot path | D·d·χ² | D·d·χ² (theta_size_max_) | OK |
| d_T1_/d_T2_ | dmrg2-gpu-base | apply_heff_two_site Step1/2 + update_*_env V/U | sequential | D·d²·χ² | D·d²·χ² | OK |
| d_svd_work | pdmrg-gpu-base ws | S·Vh OR U·S (mutex by direction) | sequential within svd_split | max(svd_max_m·k, k·svd_max_n) | same expression | OK |
| d_svd_work_ | dmrg-gpu-base | S·Vh OR U·S (mutex by direction) | sequential | svd_max_dim·χ_max | svd_max_dim·χ_max | OK |
| d_Vh_canonical | pdmrg-gpu-base ws | swap holder for canonical Vh in merge | local to boundary path | ≤ theta_size_max_ | theta_size_max_ | OK |

No Block-Davidson at -base ⇒ no `d_dav_work_` aliasing risk
(round-8 CR-D1 class is structurally absent from this tier).

G — round-13 fix-class propagation (the three flagged in the brief):
all three now present in every -base variant where applicable. M1
PointerModeGuard, M14 stub removal, and pdmrg-base docstring fix —
all verified above.

## SUMMARY

The -base tier is clean. Round-13's three propagated fixes
(PointerModeGuard adoption, set_quiet stub removal, pdmrg-base
docstring) verify INTACT in all three -base variants. Earlier-round
fixes (J1 Stoudenmire, M4-W set_mpo guards, round-9 dead
d_svd_work_, round-8 C-new1 canonical-Vh swap) remain in place.
Self-audit found one new minor docstring drift in
dmrg2-gpu-base.h:83 — NIT only. No CRITICALs, no HIGHs, no
MEDIUMs net-new, no charter violations, no -gpu-tier feature
creep. Verdict: READY.
