# Horizontal review — -base tier — round 13 (2026-04-30)

Scope: dmrg-gpu-base, dmrg2-gpu-base, pdmrg-gpu-base. Baseline:
round-12 horizontal-base report at 8b7a68e; only one new commit
(0b9fccf, round-12 medium cleanup) since.

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | 0 unused private members across 3 variants |
| B. Behavioral diff | DONE | No event/wait/graph divergence; all three on single stream (pdmrg per-segment) |
| C. Docstring verification | DONE | 0 unverified claims; pdmrg-gpu-base J1 claim backed by code (line 1267) |
| D. clangd filter | N-A | clangd not invokable on this host; A subsumes most |
| E. Absence-naming brief | FOLLOWED | -gpu-tier features absent from all three -base variants |
| F. Workspace-aliasing audit | DONE | 4 shared scratch buffers checked, 0 OVERRUN |
| G. Sibling fix-propagation | DONE | 2 round-12 fixes traced, 2 MISSING in all three -base variants |

## CRITICALS — block GPU run / paper submission

(none)

## HIGHS — fix before next major event

(none)

## MEDIUMS — fix when convenient

- **M1-base-prop: PointerModeGuard not adopted in any -base variant**
  [dmrg-gpu-base: dmrg_gpu_base_impl.h:517,586,622,627] /
  [dmrg2-gpu-base: dmrg2_gpu_base_impl.h:585,647,677,682] /
  [pdmrg-gpu-base: pdmrg_gpu_base_impl.h:741,805,842,847].
  Round-12 commit 0b9fccf consolidated 3 inline pointer-mode guards
  into `common/pointer_mode_guard.h::PointerModeGuard`. The brief's
  regression-watch asked: "Verify -base variants either use the
  shared guard or don't need one." All three -base variants still
  use raw paired `set_pointer_mode(device)…(host)` calls bracketing
  ROCBLAS_CHECK sequences. If any ROCBLAS_CHECK throws between the
  pair, the handle leaks device-mode into the next caller. Same
  defect class as M1-final, sibling-propagation gap. Fix: include
  `../../common/pointer_mode_guard.h` and replace each paired
  set_pointer_mode block with a `PointerModeGuard pm_guard(handle,
  rocblas_pointer_mode_device)` at the start of the device-mode
  region (handles_[si] for pdmrg-gpu-base).

- **M14-base-prop: dead `set_quiet(bool)` stub in all three -base
  headers** [dmrg-gpu-base: dmrg_gpu_base.h:52] / [dmrg2-gpu-base:
  dmrg2_gpu_base.h:63] / [pdmrg-gpu-base: pdmrg_gpu_base.h:72].
  Round-12 M14-final removed dead `set_quiet(bool){}` stubs from
  -opt headers on the criterion "test drivers don't call these."
  Same criterion now applies: none of test_dmrg_gpu_base.cpp,
  test_dmrg2_gpu_base.cpp, test_pdmrg_gpu_base.cpp call
  `.set_quiet(...)` (pdmrg-gpu-base test driver line 330 swallows
  `--quiet` as a documented argv no-op). Fix: drop the three stubs.

## NITS — cosmetic

- pdmrg-gpu-base.h:42 docstring reads "default 1 each" but the
  signature defaults to `n_warmup = 1, n_polish = 0`. The body of
  the docstring at line 58 correctly says "n_warmup=1, n_polish=0".
  The earlier "default 1 each" at line 42 should be "default
  n_warmup=1, n_polish=0" for consistency. CLAUDE.md compliance is
  intact — both defaults are ≤ 2 and both phases use single-site
  sweeps.

## FALSE POSITIVES VERIFIED

- "pdmrg-gpu-base svd_split (line 857) uses plain gesvd_auto — J1
  violation." Verified not a defect: accurate_svd_gpu is required
  only at *boundary merges* where V = Λ⁻¹. Inner-segment splits use
  plain gesvd in every pdmrg variant. Boundary call is at line 1267.
- "h_WL/h_WR built per-call on host inside set_mpo." Not a defect —
  set_mpo is one-shot init, outside the timed sweep region.

## Technique-A & technique-F worksheet (audit evidence)

A — every private member of all three classes has ≥3 grep hits in
its impl.h (alloc + free + ≥1 use). No declared-but-unused
infrastructure. dmrg2-gpu-base: round-9 dead `d_svd_work_` is no
longer present (verified by grep returning 0 hits in both header and
impl).

F — shared scratch buffers checked:

| Buffer | Variant | Regions | Lifetimes | Required | Allocated | Verdict |
|---|---|---|---|---|---|---|
| d_T1_/d_T2_ | dmrg-gpu-base | apply_heff Step1/2 + update_*_env V/U | sequential within hot path | D·d·χ² | D·d·χ² | OK |
| d_T1_/d_T2_ | dmrg2-gpu-base | apply_heff_two_site Step1/2 + update_*_env V/U | sequential | D·d²·χ² | D·d²·χ² | OK |
| d_svd_S | pdmrg-gpu-base ws | host V upload (form_theta_with_V) + SVD output | sequential within optimize_bond | χ_max | χ_max | OK (round-7 W3 reuse pattern, documented at line 1206) |
| d_psi_R | pdmrg-gpu-base ws | scaled-rows scratch in form_theta_with_V | local-only | χ_bond·d·χ_R ≤ χ_max·d·χ_max | χ_max·d·χ_max | OK |

No Block-Davidson at -base ⇒ no `d_dav_work_` aliasing risk
(round-8 CR-D1 class is structurally absent from this tier).

## SUMMARY

The -base tier is in good shape. Both M4-W and round-9 dead-buffer
fixes carry forward cleanly; J1 (pdmrg-gpu-base + accurate_svd_gpu
at boundary) and round-8 C-new1 canonical-Vh swap remain in place.
The round-13 work surfaced is purely round-12 sibling-propagation
gaps — Technique G fired correctly. Two MEDIUMs to back-port
PointerModeGuard and drop the three dead set_quiet stubs into the
-base variants will close the consolidation that round-12 began.
No CRITICALs, no HIGHs, no charter violations, no -gpu-tier feature
creep. Self-audit verdict: clean. Act on M1-base-prop first because
it is a latent correctness bug (handle pointer-mode leak on
ROCBLAS_CHECK throw); M14-base-prop is pure dead-code hygiene.
