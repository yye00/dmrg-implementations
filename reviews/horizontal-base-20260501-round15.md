# Horizontal review — -base tier — round 15 (2026-05-01)

Scope: dmrg-gpu-base, dmrg2-gpu-base, pdmrg-gpu-base. Baseline:
round-14 horizontal-base report at f40140d. Two commits since
round-14 baseline (5deba6d):

- `5deba6d` — round-14 fixes: H1-base-apply_heff scope fix landed in
  all three -base variants (PointerModeGuard moved INSIDE per-iter
  loop body, AFTER `apply_heff*(...)` returns; initial-vector
  normalization in its own pre-loop guard).
- `f40140d` — round-14 nit cleanup, `dmrg-gpu.h:43` only (no -base
  files touched).

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | 0 unused private members across 3 variants |
| B. Behavioral diff | DONE | lanczos_eigensolver structurally identical across 3 -base variants (post-H1 scope shape) |
| C. Docstring verification | DONE | round-14 d_WW_ NIT confirmed FIXED at dmrg2_gpu_base.h:83-87; no new drift |
| D. clangd filter | N-A | clangd not invokable on this host; A subsumes the unused-private-field channel |
| E. Absence-naming brief | FOLLOWED | -gpu-tier features remain absent from all three |
| F. Workspace-aliasing audit | DONE | 5 shared scratch buffers checked, 0 OVERRUN; no buffer reslicing in 5deba6d |
| G. Sibling fix-propagation | DONE | 1 round-14 fix (H1-base-apply_heff) traced; PRESENT in all 3 -base variants |

## CRITICALS — block GPU run / paper submission

(none)

## HIGHS — fix before next major event

(none)

## MEDIUMS — fix when convenient

(none net-new for -base this round)

## NITS — cosmetic

(none net-new for -base this round)

## FALSE POSITIVES VERIFIED

- "-base references RSVD / GpuOpts / D_PAD / graph-capture":
  grep hits are ONLY inside docstrings stating what -base does NOT
  have (dmrg_gpu_base.h:23-26, pdmrg_gpu_base.h:25-28,
  dmrg2_gpu_base_impl.h:259, pdmrg_gpu_base_impl.h:1269). No code use.
- "pdmrg-gpu-base svd_split uses plain gesvd_auto — J1 violation":
  carries from rounds 13/14. accurate_svd is mandatory at boundary
  merges only; boundary call confirmed at pdmrg_gpu_base_impl.h:1276.

## Regression-watch verification

| Round | Fix | Status |
|---|---|---|
| 14 H1-base-apply_heff (round-14 brief) | PointerModeGuard inside per-iter loop body, after apply_heff returns | INTACT — dmrg_gpu_base_impl.h:518-524 (pre-loop nrm2/scal_real guard), :539-588 (per-iter device-mode guard wraps dot/axpy/reorth/nrm2/scal_real), :537 apply_heff in host mode; mirrors at dmrg2_gpu_base_impl.h:586-593 + :608-651 + :606; mirrors at pdmrg_gpu_base_impl.h:743-749 + :766-809 + :762/764 |
| 14 H1-base-apply_heff grep gate | `grep -rn "rocblas_set_pointer_mode" gpu-rocm/{dmrg,dmrg2,pdmrg}-gpu-base/src/` returns 0 hits | INTACT — exit code 1, no matches |
| 13 M1-base-prop | PointerModeGuard adoption in 3 -base variants | INTACT — every BLAS-1 region in lanczos uses PointerModeGuard, none use raw `rocblas_set_pointer_mode` |
| 13 M14-base-prop | dead set_quiet stubs removed | INTACT — `grep -n "set_quiet" gpu-rocm/{dmrg,dmrg2,pdmrg}-gpu-base/src/*.h` returns 0 hits |
| 13 NIT (pdmrg-base docstring) | "default 1 each" → explicit n_warmup=1, n_polish=0 | INTACT — pdmrg_gpu_base.h:42-43 (docstring) and pdmrg_gpu_base.h:64-65 (run signature) both read `n_warmup=1, n_polish=0` |
| 12 M-base-stoudenmire (J1) | accurate_svd_gpu boundary call in pdmrg-gpu-base | INTACT — pdmrg_gpu_base_impl.h:1276; AsvdScratch ws.asvd allocated/freed and live at use site :1283 |
| 12 M4-W set_mpo guards | free-then-realloc on d_W_left_/d_W_right_/d_WW_ | INTACT — dmrg_gpu_base_impl.h:245,249; dmrg2_gpu_base_impl.h:243,247,306; pdmrg_gpu_base_impl.h:343,347,398 |
| 14 NIT (dmrg2 d_WW_ comment) | "on host at set_mpo() time and hipMemcpy'd to device" | INTACT — dmrg2_gpu_base.h:83 reads as expected |
| 9 MED-base-1 | dead d_svd_work_ in dmrg2-gpu-base removed | INTACT — `grep` returns 0 hits |
| 8 C-new1 | pdmrg-gpu-base R_env from canonical Vh | INTACT — d_Vh_canonical swap pattern present |

## Technique-A & technique-F worksheet

A — every private member has ≥3 grep hits in impl.h. pdmrg-gpu-base
StreamWorkspace members all live (asvd, d_Vh_canonical, d_psi_R,
d_svdj_residual included). No declared-but-unused infrastructure.

F — shared scratch buffers (no buffer reslicing in 5deba6d — only
pointer-mode scope changed):

| Buffer | Variant | Regions | Lifetimes | Required | Allocated | Verdict |
|---|---|---|---|---|---|---|
| d_T1_/d_T2_ | dmrg-gpu-base | apply_heff Step1/2 + update_*_env V/U | sequential | D·d·χ² | theta_size_max_ | OK |
| d_T1_/d_T2_ | dmrg2-gpu-base | apply_heff_two_site Step1/2 + update_*_env V/U | sequential | D·d²·χ² | matched | OK |
| d_svd_work | pdmrg-gpu-base ws | S·Vh OR U·S (mutex by direction) | sequential within svd_split | max(svd_max_m·k, k·svd_max_n) | matched | OK |
| d_svd_work_ | dmrg-gpu-base | S·Vh OR U·S | sequential | svd_max_dim·χ_max | matched | OK |
| d_Vh_canonical | pdmrg-gpu-base ws | swap holder for canonical Vh in merge | local to boundary path | ≤ theta_size_max_ | theta_size_max_ | OK |

No Block-Davidson at -base ⇒ no `d_dav_work_` aliasing risk
(round-8 CR-D1 class is structurally absent from this tier).

G — round-14 H1-base-apply_heff fix-class propagation: present in
all three -base variants. dmrg-gpu-base and dmrg2-gpu-base each
have their explicit "Round-14 H1-base fix" comment at
dmrg_gpu_base_impl.h:528-531, dmrg2_gpu_base_impl.h:597-600,
pdmrg_gpu_base_impl.h:753-756. Pattern matches the brief exactly
(pre-loop guard for nrm2+inv_real+scal_real; for-loop body with
apply_heff in host mode, then per-iter device-mode guard wrapping
dot/axpy/gemv/nrm2/scal_real; rest of function in host mode).

## SUMMARY

The -base tier is clean. The round-14 H1-base-apply_heff scope fix
(PointerModeGuard inside the per-iter loop body, after apply_heff
returns; initial-vector normalization in its own pre-loop guard) is
verified PRESENT in all three -base variants with the expected
explicit comment tagging the round-14 fix. The `grep -rn
rocblas_set_pointer_mode` gate returns 0 hits across all three -base
src/ trees. All earlier-round fixes (J1 Stoudenmire, M4-W set_mpo
guards, M14 stub removal, M1 PointerModeGuard adoption, round-14
d_WW_ comment correction) remain INTACT. No new
declared-but-unused infrastructure, no -gpu-tier feature creep, no
buffer reslicing in 5deba6d (so workspace-aliasing audit is a
pass-through from round-14). No CRITICALs, no HIGHs, no MEDIUMs, no
NITs net-new this round. Verdict: READY.
