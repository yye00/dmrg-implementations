# Horizontal review — `-gpu-opt` tier — round 20 (2026-05-03, HEAD `f650466`)

R20 confidence re-run before MI300X G1 baseline allocation. Code state
= `cafd628` (R19 H19 fix landed, `-multi-gpu` only). No `-opt` code
edited since R19. Defect-class registry pre-step: **TOTAL HITS: 0**.

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan       | DONE | No new dead members; R11/R17 carry-over `d_const_*`, `d_alpha_dev_`, `d_beta_dev_` now LIVE in D12 device-pointer Lanczos (all 3 -opt). |
| B. Behavioral diff         | DONE | 3-way: dmrg-opt, dmrg2-opt, pdmrg-opt. apply_heff Step 3 uses per-wp loop (intentional -opt design — see ruling). Davidson restart logic identical across 3 (CR-D1 fix-shape). pdmrg-opt extras gated. |
| C. Docstring verification  | DONE | One stale comment carry from R18 (`pdmrg_gpu_opt_impl.h:2302-2311`, single-site graph reason). M-opt-pdmrg-graph-comment-stale unchanged. |
| D. clangd filter           | N-A  | No ROCm headers on host. |
| E. Absence-naming brief    | FOLLOWED | All required -opt features present (pad_mfma16, chi_max_user_, Block-Davidson default + tiny-Lanczos fallback, public setters per spec, pdmrg-opt worker pool LIVE, accurate_svd_gpu in pdmrg-opt). |
| F. Workspace-aliasing audit| DONE | 3 shared buffers per variant; `d_dav_work` sizing OK across 3 -opt (CR-D1 R17 fix intact). No new aliasing introduced since R19. |
| G. Sibling fix-propagation | DONE | R19 H19 (cafd628 multi-gpu single-site Step 3 batched port) is `-multi-gpu` only, NOT in -opt scope. Verified -opt apply_heff_single_site Step 3 is intentionally per-wp/strided per 98ca518 commit message. CR-D1, D6, D12, t_davidson_, t_absorb_ propagation all intact. |

## CRITICALS — block GPU run / paper submission

**None.**

## HIGHs — fix before next major event

- **H-opt-batched-lanczos-host-mode** (CARRY from R18/R19, found by:
  this review). `pdmrg_gpu_opt_impl.h:2870-2900` (`batched_lanczos_eigensolver`)
  builds per-segment `std::vector<double> h_alpha, h_beta` host arrays,
  gated on `set_use_batched_sweep(true)` (default `false` at impl
  :194). Same shape applies to `chebyshev_eigensolver` host
  `h_alpha`/`h_beta` at impl :2028-2029, gated on `set_use_chebyshev(true)`
  (default `false` at impl :195). Both opt-in toggles. **Defer until
  those campaigns are on the menu — neither fires on G1 baseline.**

## MEDIUMs — fix when convenient

- **M-opt-pdmrg-single-site-graph-comment-stale** (CARRY from R18/R19,
  found by: this review). `pdmrg_gpu_opt_impl.h:2302-2311` comment
  block claims single-site Step 1/Step 3 use `Scalar* h_A[256]` stack
  arrays + `hipMemcpyAsync` H2D as the reason LANCZOS_GRAPH is
  disabled. Actual code at :2340/2356/2416 already uses device
  kernels (`setup_batch_ptrs_wd*`, `setup_batch_ptrs_step3`).
  Comment is the round-15 fix's own pre-fix narration that was never
  updated. Either drop the comment or re-enable graph capture and
  re-test the warmup hang on Josephson L=8.

## NITs — cosmetic

- **R17-base-review-claim-drift** (CARRY, this review). `reviews/horizontal-base-20260501-round17.md:52-57` states "R3-F1
  batched-collapse is an explicit `-gpu` / `-opt` optimization."
  Actual code: R3-F1 (commit `98ca518`, 2026-04-10) explicitly
  states "Targets only the plain -gpu implementations as directed;
  -gpu-opt and -gpu-base variants are untouched." The -opt tier
  retains its earlier strided-batched/per-wp Step 3 design (commit
  `bd4d09c`, 2026-03-31) by maintainer choice. The R17 base-review
  prose overstates the -opt commitment. Cosmetic — not actionable
  in code.

## FALSE POSITIVES VERIFIED

- **D13 candidate: pdmrg-gpu-opt apply_heff_single_site Step 3 per-wp
  loop with `gemm_strided_batched(d)`/`gemm_batched(d)` inside**
  (impl :2402, :2414). Structurally matches the cafd628 D13 widening
  pattern. Verified intentional: commit `98ca518` (R3-F1) message
  explicitly excludes -opt; sibling -gpu's R3-F1 collapse is a
  designer-directed -gpu-only optimization (see N-R17-base-review-claim-drift
  above). pdmrg-multi-gpu's R19 H19 fix targeted multi-gpu, not -opt.
  No defect.
- **D13 candidate: dmrg-gpu-opt apply_heff Step 3 per-wp loop**
  (impl :728, :741). Same ruling as above — intentional -opt design
  per `98ca518` commit; R11 review documented R3-F1 buffers as
  pre-existing design choice in -opt.
- **D13 candidate: dmrg2-gpu-opt apply_heff_two_site Step 3 nested
  per-(s2p,n) loops** (impl :779, :796). Same ruling.
- **D13 candidate: sparse Step 3 host-loop in all 3 -opt** (idx-loop
  with bare `Traits::gemm`). Gated by `opts_.sparse_mpo` (env-var
  `*_SPARSE_MPO=1`, default `false` at `gpu_opts.h:20`). R16 verified
  false positive; preserved here.
- **D7+D8 host LAPACK gesvd in single-site path of pdmrg-gpu-opt**
  (impl :2466, called via `use_cpu_svd_` opt-in branch — registry
  whitelists by `use_cpu_svd_` proximity).

## Self-audit

- **F**: Davidson buffer aliasing across 3 -opt verified.
  `d_dav_work` hosts residuals W (region `[0, n_new*dim)`, with
  `n_new ≤ b ≤ davidson_b_`) AND overlap matrix (region `[n_new*dim,
  n_new*dim + k*n_new)`, with `k ≤ davidson_max_sub_` and `n_new ≤
  davidson_b_`). Required total ≤ `theta_size_max_*davidson_b_ +
  davidson_max_sub_*davidson_b_`. Allocated total = `std::max(
  theta_size_max_*davidson_b_ + davidson_max_sub_*davidson_b_,
  davidson_max_sub_*davidson_max_sub_)`. **OK** in
  pdmrg-opt impl :266-269, dmrg-opt impl :301-304, dmrg2-opt impl
  :285-288. `d_dav_work2` separately allocated with same sizing —
  holds rocsolver_syevd eigenvectors needed by restart path; offset
  trick prevents the overlap GEMM from clobbering it. CR-D1 fix
  intact in all 3 -opt.
- **G**: cafd628 changed `pdmrg-multi-gpu/src/pdmrg_multi_gpu_impl.h`
  only (plus `defect-registry.sh`). Diff: 67/25. -opt scope
  untouched. Verified D13 widening pattern's intent (per-wp loop
  wrapping batched-`d` GEMM) does NOT apply to -opt — the -opt's
  per-wp Step 3 is the documented `bd4d09c` design choice that
  98ca518 explicitly preserved. PhaseTimer t_davidson_ +
  t_absorb_ R17 fixes intact: dmrg-opt + dmrg2-opt have
  t_absorb_.begin/.end pairs (live, M:1359/1391/1405 dmrg2 +
  M:1385/1413/1438/1465 dmrg); pdmrg-opt correctly omits
  t_absorb_ (intermixed with t_svd_ — header noted at
  `pdmrg_gpu_opt.h:260`).
- **Regression watch since R19**: every R15→R19 fix intact in -opt.
  `use_davidson_=true` at ctor in all 3 (+ ctor-time gate present
  for D11). `pad_mfma16` idempotent. accurate_svd_gpu live in
  pdmrg-opt (J1). PointerModeGuard on every Lanczos block in all 3.
  D6 shared kernels via `#include "../../common/batch_ptrs_kernels.h"`
  in all 3.
- **Verdict**: **READY**. 0 CRITICAL / 0 NEW HIGH / 1 carry HIGH on
  opt-in code path / 1 carry MEDIUM (cosmetic comment) / 1 NIT
  (R17 base-review prose drift, not -opt code).

## SUMMARY

R20 -opt review confirms no regression since R19. The R19 H19 fix
(`cafd628`) is correctly scoped to `pdmrg-multi-gpu` and does not
affect the -opt tier. The widened D13 pattern in the registry would
match -opt's per-wp Step 3 loops mechanically (intentional false
positive from the registry's perspective), but verified analysis of
commit `98ca518` proves -opt's Step 3 design is a deliberate
maintainer choice that 98ca518 explicitly preserved. The single
carry HIGH (`H-opt-batched-lanczos-host-mode`) and the chebyshev
counterpart remain on opt-in toggles (`set_use_batched_sweep(true)`
/ `set_use_chebyshev(true)`, both default `false`); G1 benchmark
drivers must NOT enable these. CR-D1 Davidson buffer sizing and
device-pointer Lanczos remain intact across all 3 -opt variants.
**No action items block the MI300X G1 baseline campaign.**
