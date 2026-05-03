# Horizontal review — -gpu tier — round 20 (2026-05-01)

Confidence re-run #2 before MI300X G1 baseline. R19 H19 fix verification.

HEAD: f650466 (reviews-only); code state at cafd628.
Scope: dmrg-gpu, dmrg2-gpu, pdmrg-gpu, pdmrg-multi-gpu (4 -gpu variants).

## PRE-STEP — defect-registry sweep

`bash .claude/scripts/defect-registry.sh` -> **TOTAL HITS: 0**.

The widened D13 (R19M19, cafd628) now greps for per-wp host loops
wrapping bare `Traits::gemm(`, `Traits::gemm_batched(`, OR
`Traits::gemm_strided_batched(`. The prior R19H19 pattern (per-wp wrap
of gemm_batched(batch=d) in pdmrg-multi-gpu single-site Step 3) is the
exact case it now catches. With the fix in place, registry is clean.
No false-positives elsewhere.

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | d_ones_D used in 2 sites + ctor + dtor; no dead members from the H19 batch |
| B. Behavioral diff | DONE | apply_heff_single_site multi-gpu :1737-1770 vs pdmrg-gpu :1992-2018 — structurally identical |
| C. Docstring verification | DONE | new comment block in multi-gpu :1737-1742 cites pdmrg-gpu :1992-2018 source — verified accurate |
| D. clangd filter | N-A | not invokable on host (no ROCm headers) |
| E. Absence-naming brief | FOLLOWED | every -gpu feature checked across all 4 variants |
| F. Workspace-aliasing audit | DONE | dev.d_T1 (= V) audited as Step 3 scratch — no overrun |
| G. Sibling fix-propagation | DONE | R19H19 + R17H4 + non-blocking + Stoudenmire + PhaseTimer panels all verified |

## CRITICALs — block GPU run / paper submission

**None.**

## HIGHs — fix before next major event

**None new.** The R19H19 fix in pdmrg-multi-gpu is correctly applied
(see verification below).

Carry HIGH (still valid but opt-in only, deferred to that campaign):
- **H-opt-batched-lanczos-host-mode** [pdmrg-gpu-opt] — out of -gpu
  charter (lives in -opt tier), not relevant to this sub-review.

## R19H19 verification — pdmrg-multi-gpu single-site Step 3

### (a) Local kernel matches pdmrg-gpu :49-59 byte-for-byte

`diff` of `setup_heff_ss_step3_full_ptrs` between
`pdmrg_gpu_impl.h:49-59` and `pdmrg_multi_gpu_impl.h:43-53` returns
empty. Same threadIdx-based 1D-launch, same A/B/C strides
(`base_U + (wp*d+sp)*strideA`, `base_R + wp*strideB`,
`base_C_scratch + wp*slice_stride + sp*strideC_tile`).

The local-kernel duplication is pre-existing
**M-multi-gpu-local-kernels-dup** (carry MEDIUM — different TUs, not
ODR; consolidation deferred to a later round).

### (b) dev.d_ones_D allocated AND initialized to ones

`pdmrg_multi_gpu.h:134` declares `Scalar* d_ones_D` per-DeviceContext.
`pdmrg_multi_gpu_impl.h:370` allocates `D_mpo_ * sizeof(Scalar)`.
:371-375 builds `std::vector<Scalar> h_ones(D_mpo_, Traits::one())`
and `hipMemcpy`s host->device. Freed in dtor at :534. The same buffer
is reused by two-site Step 3 reduce at :860 (R17H4 path).

### (c) T1 size is sufficient

Required at single-site Step 3:
`D · slice_stride = D_mpo_ · cL · d · cR ≤ D_mpo_ · chi_max · d · chi_max`.

Allocated (`pdmrg_multi_gpu_impl.h:292,352`):
`t_max = D_mpo_ · dd · chi_max² = D_mpo_ · d² · chi_max²`.

Ratio: allocated / required = `d² / d = d` → headroom of factor `d`.
**OK**, no overrun. Same buffer is reused by two-site Step 3
(slice_stride = `cL · dd · cR` -> required = `D_mpo · dd · chi_max²`,
exact fit). Alias is sequential between Step 2 (`V` is read into `T2`,
then dead) and Step 3 (writes back to `V` per-wp slices); no
concurrent-region conflict.

### (d) t_apply_heff_.end fires after the new Step 3 block

Begin at :1702 `if (di == 0) t_apply_heff_.begin(dev.stream);`.
End at :1771 `if (di == 0) t_apply_heff_.end(dev.stream);` —
*after* the new gemv reduce closes at :1770. No timer leak. Same
guard pattern as two-site (:787 / :864).

### (e) Two-site Step 3 still uses R17-landed batched path

Verified `pdmrg_multi_gpu_impl.h:836-863` — `setup_batch_ptrs_step3_twosite_full`
(promoted to common, R16 D6) + `gemm_batched(batch=D·dd)` + `gemv` with
`d_ones_D`. No regression.

## Cross-family conformity

### D6 — common batch_ptrs_kernels.h, no shared-name duplicates

All 4 in-charter variants `#include "../../common/batch_ptrs_kernels.h"`
(dmrg-gpu :13, dmrg2-gpu :17, pdmrg-gpu :22, pdmrg-multi-gpu :18).
No duplicate `setup_batch_ptrs_*` definitions in any variant impl
file (greps confirm definitions live only in common/). Pre-existing
local kernels in pdmrg-multi-gpu (`setup_heff_ss_step3_*`,
`setup_lenv_step3_ptrs`, `setup_renv_step3_ptrs`) are file-local with
distinct names from the common ones — not D6 violations, but flagged
under M-multi-gpu-local-kernels-dup for consolidation.

### PhaseTimer panels

| variant | t_lanczos_ | t_apply_heff_ | t_svd_ | t_absorb_ | t_env_update_ | total |
|---|---|---|---|---|---|---|
| dmrg-gpu  | yes | yes | yes | yes | yes | 5 |
| dmrg2-gpu | yes | yes | yes | yes | yes | 5 |
| pdmrg-gpu | yes | yes | yes | (removed: intermixed) | yes | 4 |
| pdmrg-multi-gpu | yes | yes | yes | (removed: intermixed) | yes | 4 |

Total 18 declarations across .h files (matches grep count). The
absence of t_absorb_ in pdmrg variants is **intentional** — comments
on pdmrg_gpu.h:214-215 and pdmrg_multi_gpu.h:217-219 explain absorb
is intermixed with two-site SVD and cannot be cleanly separated.

In multi-gpu impl: 12 begin/end calls — paired across 6 timer
sites (lanczos init/teardown + apply_heff begin/end x2 (single+two)
+ svd + env_update). All paired.

### Non-blocking streams flag

All 4 variants create streams with `hipStreamNonBlocking` flag
(R10 H1-final): dmrg-gpu :83-84, dmrg2-gpu :79,82, pdmrg-gpu :228,
pdmrg-multi-gpu :312. **Pass.**

### Stoudenmire J1 lock (pdmrg only)

- pdmrg-gpu_impl.h:2452 `accurate_svd_gpu<Scalar>(...)` — present.
- pdmrg-multi-gpu_impl.h:2281 `accurate_svd_gpu<Scalar>(...)` —
  present.

J1 satisfied for both pdmrg variants.

## Technique G — new angle: any other apply_heff per-wp loops?

Per the brief, scan all -gpu apply_heff variants for residual per-wp
host loops with EITHER bare gemm OR wrapped gemm_batched.

Scan of the 6 in-charter apply_heff functions:

| function | location | per-wp loops in body? |
|---|---|---|
| dmrg-gpu apply_heff (single) | dmrg_gpu_impl.h:539 | NONE |
| dmrg2-gpu apply_heff_two_site | dmrg2_gpu_impl.h:607 | NONE |
| pdmrg-gpu apply_heff_two_site | pdmrg_gpu_impl.h:887 | NONE |
| pdmrg-gpu apply_heff_single_site | pdmrg_gpu_impl.h:1868 | NONE |
| pdmrg-multi-gpu apply_heff_two_site | pdmrg_multi_gpu_impl.h:779 | NONE |
| pdmrg-multi-gpu apply_heff_single_site | pdmrg_multi_gpu_impl.h:1695 | NONE (R19H19 fixed) |

**All 4 in-charter -gpu apply_heff paths are now fully batched.** No
remaining per-wp host loops in any apply_heff variant in scope.

**Out-of-charter sibling note (informational, NOT a finding for this
sub-review):** rlbfgs-gpu apply_heff (rlbfgs_gpu_impl.h:667) and
radam-gpu apply_heff (radam_gpu_impl.h:656) DO contain per-wp host
loops wrapping `gemm_batched`. Defect-registry whitelists these
variants (see `defect-registry.sh:16-28` — radam/rlbfgs explicitly
out of conformity charter). Filed as observation only; not blocking
for the G1 baseline (those variants are not benchmarked in G1).

## MEDIUMs — fix when convenient

- **M-multi-gpu-local-kernels-dup** (carry from R18/R19) —
  `pdmrg_multi_gpu_impl.h:25-79` has 4 file-local pointer-setup
  kernels (`setup_heff_ss_step3_ptrs`, `setup_heff_ss_step3_full_ptrs`,
  `setup_lenv_step3_ptrs`, `setup_renv_step3_ptrs`) duplicated from
  pdmrg-gpu. Different TUs -> not ODR violation. Consolidate to
  `common/batch_ptrs_kernels.h` once pdmrg-gpu's locals are also
  promoted.
- **N-pointer-mode-guard-return-discards** (carry from R19) — cosmetic.

## NITs

(None new this round — carry list per R19.)

## FALSE POSITIVES VERIFIED

- **D13 baseline -base apply_heff per-wp host loop** — registry
  whitelists -base by charter (naive single-GEMM IS the baseline tier).
- **Local kernel duplication is NOT D6** — D6 specifically targets
  shared-name kernels in the common header pattern; the multi-gpu
  local kernels have file-private names.

## SUMMARY

R19H19 (pdmrg-multi-gpu apply_heff_single_site Step 3 batched port)
is verified correct in all 5 audit dimensions: kernel-byte-identity
to pdmrg-gpu source, d_ones_D allocation+initialization+free,
T1-buffer headroom (factor d), timer pairing intact, two-site
sibling not regressed. The widened D13 detector now flags wrapped
gemm_batched inside per-wp loops AND returns 0 hits at HEAD (the fix
clears it and no false-positives elsewhere). All 6 in-charter
apply_heff variants across the 4 -gpu sources are now fully
batched — no per-wp host loops remain. Cross-family conformity
holds: common batch_ptrs include, PhaseTimer panels match the
documented 5/5/4/4 pattern (intentional t_absorb_ removal in pdmrg
variants), non-blocking streams everywhere, Stoudenmire J1 satisfied
in both pdmrg variants. **0 CRITICALs, 0 new HIGHs.** The -gpu tier
is READY for the MI300X G1 baseline campaign from a
horizontal-review-gpu standpoint.
