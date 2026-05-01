# Vertical review — pdmrg family — 2026-05-01 (R19)

HEAD `bb809fb` (code state pinned at 12d02c5). Confidence re-run before
MI300X allocation. Independent re-audit of the four pdmrg variants;
prior baseline `reviews/conformity-20260501-round18.md`.

Pre-step `bash .claude/scripts/defect-registry.sh` reported
**TOTAL HITS: 0** across the 14 codified defect classes (D1–D15) —
clean before review.

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | Spot-checked changed members + opt-tier infra (worker_streams_, d_xs_batch_*, d_ones_D, d_dav_*, d0_* mirrors, peer_access_); all live. |
| B. Behavioral diff | DONE | base↔gpu↔opt↔multi compared on apply_heff_two_site Step 3, merge_and_optimize_boundaries, block_davidson_eigensolver, lanczos_eigensolver. |
| C. Docstring verification | DONE | -base J1 docstring verified (lines 30-33). multi-gpu opts_ scope claim verified (only rsvd+profile honoured). 1 stale comment carry. |
| D. clangd filter | N-A | No ROCm headers locally. Substituted by A + targeted greps. |
| E. Absence-naming brief | FOLLOWED | Tier-feature checklist evaluated for all four variants. |
| F. Workspace-aliasing audit | DONE | `d_dav_work` (CR-D1 fix), `d_T1` Step 1↔Step 3 reuse in multi-gpu, `t_max` ≥ slice_stride·D check. |
| G. Sibling fix-propagation | DONE | All 6 R15→R17 fix classes traced across 4 variants. 0 MISSING. |

## CRITICALs — block GPU run / paper submission

**None.**

### CR-D1 re-verification (special-focus, commit 54f2fcf)

**(a) ctor sizing.** `pdmrg_gpu_opt_impl.h:266-269`:
```
size_t dav_work_sz = std::max(
    (size_t)theta_size_max_ * davidson_b_
        + (size_t)davidson_max_sub_ * davidson_b_,
    (size_t)davidson_max_sub_ * davidson_max_sub_);
```
Matches spec exactly. `b·dim_max + max_sub·b` covers concurrent
residuals + overlap; `max_sub²` covers the projected-H region.

**(b) overlap GEMM target.** `pdmrg_gpu_opt_impl.h:1687-1693`:
`Scalar* W = ws.d_dav_work; Scalar* overlap = ws.d_dav_work +
n_new*dim;` — overlap GEMM (k×n_new) writes into `d_dav_work`, not
`d_dav_work2`. `d_dav_work2` still holds the rocsolver_syevd
eigenvectors needed by the restart path (line 1748). Confirmed.

**Sibling check.** pdmrg-multi-gpu has Lanczos only (no
block_davidson_eigensolver) — genuinely immune. pdmrg-gpu and
pdmrg-gpu-base also have no Davidson — immune.

## HIGHs — fix before next major event

- **H-opt-batched-lanczos-host-mode** (carry from R17/R18,
  opt-in OFF). `pdmrg_gpu_opt_impl.h:2858+` `batched_lanczos_eigensolver`
  retains host-resident `h_alpha`/`h_beta`/`&beta_val`/`&inv_norm`.
  Gated by `use_batched_sweep_` which is `false` by default
  (`pdmrg_gpu_opt_impl.h:194`). Never fires on G1 baseline. Defer.

## MEDIUMs — fix when convenient

- **M-multi-gpu-precompute-fused-mpo-host** (carry from R18).
  `pdmrg_multi_gpu_impl.h:701-720` runs a 6-deep host nested loop +
  H2D for `precompute_fused_mpo`. Once-per-`set_mpo()`, NOT per-
  sweep — outside the "no host roundtrips per sweep" charter — but
  still drifts from -gpu's on-device build. Charter decision pending.
- **M-opt-pdmrg-single-site-graph-comment-stale** (carry from R18).
  `pdmrg_gpu_opt_impl.h:2302-2311` claims Step 1/3 use stack
  `h_A[256]`/`h_B[256]`/`h_C[256]` to justify disabling
  LANCZOS_GRAPH for single-site. Verified at lines 2337-2370 and
  2384-2429 the actual code uses `setup_batch_ptrs_wd[_sparse]` and
  `setup_batch_ptrs_step3` device kernels. Comment is stale; either
  re-enable graph capture or rewrite the comment.
- **M-multi-gpu-local-kernels-dup** (carry from R18).
  `pdmrg_multi_gpu_impl.h:24-62` redefines 3 file-local kernels
  (`setup_heff_ss_step3_ptrs`, `setup_lenv_step3_ptrs`,
  `setup_renv_step3_ptrs`) that are also defined as file-local in
  pdmrg-gpu. Different translation units → not ODR. Consolidation
  candidate for `common/batch_ptrs_kernels.h`.

## NITs — cosmetic

None new.

## FALSE POSITIVES VERIFIED

- **multi-gpu Step 3 T1 reuse.** Verified T1 lifetime:
  Step 1 writes T1 (via `dev.d_heff_batch_C` aliasing T1 line 785).
  Step 2 READS T1 as source operand (line 808) and writes T2.
  Step 3 then writes T1 as scratch (line 824 `base_C_scratch=T1`).
  Sequential, no overlap. **Buffer size:** T1 allocated as
  `t_max = D·dd·chi_max² · sizeof(Scalar)` (line 274/334).
  Step 3 needs `slice_stride·D = cL·dd·cR·D` ≤ `chi_max²·dd·D`
  = `t_max`. OK.

- **multi-gpu d_ones_D.** Allocated `D_mpo_ * sizeof(Scalar)` and
  filled with `Traits::one()` in init (lines 352-357). Freed in
  destroy (line 516). gemv reduce uses it as inc=1 vector,
  size D — correct.

- **D5/D6/D12 propagation across pdmrg.**
  - D5 (host-batch ptrs): grep for `h_A[256]` / `h_B[256]` /
    `h_C[256]` in opt's apply_heff_two_site finds only the stale
    comment at line 2303; actual two-site path uses device kernels
    (lines 2768-2794 use `d_xs_batch_*` only). FIXED.
  - D6 (duplicate batch_ptrs kernels): pdmrg-gpu/-opt/-multi all
    `#include "../../common/batch_ptrs_kernels.h"`. -base does
    not need it (single-GEMM-per-pair tier). FIXED.
  - D12 (Lanczos device-pointer mode): pdmrg-gpu-opt
    parallel-segment Lanczos at lines 1814-1917 uses
    `workspaces_[si].d_alpha_dev` / `d_beta_dev` /
    `d_neg_alpha` / `d_inv_nrm` under
    `PointerModeGuard(handles_[si], rocblas_pointer_mode_device)`.
    FIXED on default path. (Carry-HIGH applies only to
    batched_lanczos_eigensolver, opt-in.)

- **PhaseTimer instrumentation R17 (commit 12d02c5).**
  - pdmrg-gpu-opt `apply_heff_single_site` wrapped in
    `t_apply_heff_.begin/.end` at lines 2300/2432. ✓
  - pdmrg-gpu-opt `t_davidson_` ends at all 5 return paths
    (lines 1592, 1628, 1678, 1734, 1795). ✓
  - pdmrg-multi-gpu `update_left_env` (line 860/950),
    `update_right_env` (line 968/1031), and
    `apply_heff_single_site` (line 1684/1738) all wrapped. ✓
  - `t_absorb_` removed from all three timed variants. ✓

- **J1 Stoudenmire lock all 4 variants.** Greppped
  `accurate_svd_gpu` in each `merge_and_optimize_boundaries`:
  - base impl:1276 ✓
  - gpu impl:2452 ✓
  - opt impl:3351 ✓
  - multi impl:2248 ✓
  All 4 headers `#include "../../common/accurate_svd_gpu.h"`.

- **Per-segment streams in -base.** `pdmrg_gpu_base_impl.h:52-57`
  creates `n_segments_` non-blocking streams + per-segment rocBLAS
  handles. -base correctly treats this as part of the algorithm,
  not an optimization (header docstring lines 14-21 confirm).

- **C-new1 / canonical-Vh swap.** All four variants allocate
  `d_Vh_canonical` (base impl:155, multi-gpu .h:164, gpu/opt
  workspaces). Sibling propagation closed in round-8.

## SUMMARY

**Verdict: READY for MI300X G1 baseline campaign.**

0 CRITICALs, 0 blocking HIGHs (the lone HIGH is on an opt-in code
path, `set_use_batched_sweep(true)`, default OFF — never fires on the
G1 baseline). All R17 changes verified:

1. **CR-D1 fix** in `pdmrg_gpu_opt_impl.h` is correctly sized AND
   the overlap GEMM writes to `d_dav_work` (not `d_dav_work2`),
   preserving the eigenvectors needed by the restart path. The
   sibling check is N-A (only opt has block-Davidson).

2. **pdmrg-multi-gpu Step 3 batched port** (largest R17 change)
   matches the dmrg2-gpu source pattern at impl:720-766 exactly:
   shared `setup_batch_ptrs_step3_twosite_full` kernel,
   `gemm_batched(cL × cR × cR, batch=D·dd)`, `gemv` reduce with
   `d_ones_D`. T1 reuse is sequentially safe (Step 2 reads T1
   before Step 3 overwrites). Buffer sizing OK (`t_max ≥
   slice_stride·D`).

3. **PhaseTimer panels** all instrumented with begin/end pairs,
   including all 5 davidson return paths in opt. `t_absorb_`
   removed cleanly.

4. **Defect-class registry** clean (TOTAL HITS: 0). All R15→R17
   defect-class fixes propagated correctly across the 4 pdmrg
   variants.

The 3 carry MEDIUMs (stale graph comment, multi-gpu host WW
precompute, file-local kernel duplication) are non-blocking. No
regressions detected vs R18 baseline.
