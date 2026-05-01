# Horizontal review — -gpu tier — round 18 (2026-05-01)

Charter: dmrg-gpu, dmrg2-gpu, pdmrg-gpu, pdmrg-multi-gpu. HEAD `12d02c5`
vs round-17 baseline `0efe96d`: 2 commits (`54f2fcf`, `12d02c5`).
`54f2fcf` is **-opt-only** (CR-D1 propagation to pdmrg-gpu-opt — Davidson
is an -opt-tier feature). Within -gpu charter only `12d02c5` lands deltas:
dmrg2-gpu (+4 lines, t_svd_/t_absorb_ split), pdmrg-gpu (panel doc + 2
lines removed), pdmrg-multi-gpu (substantial — Step 3 batched + 4 timer
panels wired + d_ones_D buffer added).

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | All 4 variants: every PhaseTimer member now has begin/end pairs (multi-gpu carries M17-* CLOSED). No dead concurrency primitives. |
| B. Behavioral diff | DONE | multi-gpu Step 3 ported to R3-F1 batched, pattern matches dmrg2-gpu :720-766 modulo sparse_s3 (out-of-scope) and graph capture (N-A in multi). H17 carry CLOSED. |
| C. Docstring verification | DONE | pdmrg-gpu.h:214 + pdmrg-multi-gpu.h:217 panel comments correctly document t_absorb_ removal ("intermixed in two-site SVD post-processing"). multi-gpu.h:134 d_ones_D comment names round + source. |
| D. clangd filter | N-A — no ROCm headers on host. |
| E. Absence-naming brief | FOLLOWED | All 4 variants now feature-complete vs the 11-item -gpu checklist. Multi-gpu still lacks D_PAD/sparse_mpo/lanczos_graph as documented (out-of-scope per pdmrg_multi_gpu.h:26). |
| F. Workspace-aliasing | DONE | 1 critical aliasing introduced this batch (multi-gpu T1 reuse Step1→Step3); verified safe — see below. d_ones_D sized D_mpo_·sizeof(Scalar) = single-vector reduce. No OVERRUN. |
| G. Sibling fix-propagation | DONE | 2 commits traced. CR-D1 (54f2fcf) is Davidson-only; -gpu tier is genuinely immune (zero block_davidson_ refs). H1 (t_absorb_ split) propagated through dmrg2-gpu correctly; H4 (multi-gpu Step 3) is a sibling-port FROM dmrg2-gpu, not a defect-class miss. 0 MISSING in siblings. |

A-G all DONE or N-A. Review valid.

## Regression-watch (round-17 → round-18)

| Watch item | dmrg-gpu | dmrg2-gpu | pdmrg-gpu | multi-gpu |
|---|---|---|---|---|
| `#include common/batch_ptrs_kernels.h` | yes :13 | yes :17 | yes :22 | yes :18 NEW |
| Local sparse `__global__` duplicates | 0 | 0 | 0 | 3 (pdmrg-specific ss_step3/lenv/renv — sibling of pdmrg-gpu, M18 carry) |
| `t_lanczos_` begin/end | 1/1 | 1/1 | 1/1 | 1/1 |
| `t_apply_heff_` begin/end | 1/2 | 1/2 | 2/4 | 2/2 NEW |
| `t_svd_` begin/end | 1/2 | 1/2 | 1/1 | 1/1 |
| `t_absorb_` begin/end | 2/2 | 2/1 (sym, R/L) | REMOVED | REMOVED |
| `t_env_update_` begin/end | 2/2 | 2/2 | 2/2 | 2/2 NEW |
| `D_mpo_actual_-1` boundary | :878 | :953 | :1250 | n/a |
| `init_mps_product/_neel` | h:35-36 | h:35-36 | h:36-37 | h:47-48 |
| `PointerModeGuard` use-sites | 2 | 2 | 4 | 4 |
| `d_ones_D` alloc/use/free | yes | yes | yes (per-stream) | yes (per-device) NEW |
| Lanczos device-pointer-mode | yes | yes | yes | yes |
| `accurate_svd_gpu` (J1) | n/a | n/a | yes :2452 | yes :2248 |
| Sparse-MPO compaction | yes | yes | yes | absent (out-of-scope) |

**Regression status**: zero. Round-17 closed all 3 multi-gpu MEDIUMs +
1 multi-gpu HIGH (H17-multi-apply_heff-step3-host-loop). dmrg2-gpu's
t_absorb_ end=1 begin=2 is a **balanced** pattern: the single end at
:1335 follows whichever R/L control-flow path fired (mutually exclusive
begins at :1297 / :1324). Verified by reading svd_split impl :1164-1336.

## CRITICALS

None.

## HIGHS

None.

## MEDIUMS

### M18-multi-pdmrg-kernel-duplication-with-pdmrg-gpu (carry from M17)

[`pdmrg-multi-gpu`: `pdmrg_multi_gpu_impl.h:25-61`].
`setup_heff_ss_step3_ptrs`, `setup_lenv_step3_ptrs`,
`setup_renv_step3_ptrs` are bytewise-identical to pdmrg-gpu impl.h:31-118
(modulo line numbers). Defect-class D6 axis-1 (sibling within family)
propagation gap. Should be promoted to common/batch_ptrs_kernels.h and
included from both pdmrg-gpu and pdmrg-multi-gpu, matching the round-16
pattern that consolidated the dmrg-gpu / dmrg2-gpu / pdmrg-gpu sparse
kernels. Cosmetic-class — no correctness or perf impact, but spec drift
risk over time.

## NITS

### N13-pmg-error-discard (carry-over)

[`common/pointer_mode_guard.h:16,17,22`] Discards return of
`rocblas_get/set_pointer_mode`. Cosmetic.

## FALSE POSITIVES VERIFIED

- **CR-D1 propagation to -gpu tier**: 54f2fcf fixes pdmrg-gpu-opt's
  block_davidson_eigensolver. Verified -gpu tier (dmrg-gpu, dmrg2-gpu,
  pdmrg-gpu, pdmrg-multi-gpu) has **zero** `block_davidson` /
  `d_dav_work` references — Davidson is an -opt-tier feature only.
  Genuinely immune per technique-G.
- **D5 host-batch ptr / Step 3 fallback**: abd88b9 + 187fddf are -opt
  fixes. Verified -gpu tier has zero `h_batch_*_pinned` allocations on
  hot path (one comment-only mention in pdmrg-gpu :494 documenting
  removal). No regression.
- **D6 kernel duplication**: 8abb6e7 consolidated into
  `common/batch_ptrs_kernels.h`. Verified all 4 variants `#include` it;
  only the 3 pdmrg-multi-gpu kernels at :25-61 remain locally (sibling
  of pdmrg-gpu, M18-carry — bytewise-identical, not divergent code).
- **D9/D15 PhaseTimer dmrg2-gpu t_absorb_**: split correctly per
  R/L control-flow; begin/end count is asymmetric (2/1) but balanced
  per call.
- **D9/D15 pdmrg-gpu t_absorb_ removal**: pdmrg_gpu.h:214-215 panel
  comment + pdmrg_gpu_impl.h init_timers/report_timers correctly
  prune the panel. Two-site svd_split intermixes scale + absorb GEMM
  with the trailing canonical-Vh swap; instrumenting an absorb sub-
  panel would require splitting that branch — abandoned by design.
- **dmrg2-gpu t_svd_ begin=1 end=2**: same R/L mutual-exclusive pattern
  as t_absorb_ — balanced per call.
- **Multi-gpu Step 3 T1 reuse**: T1 sized
  `t_max = D_mpo_ * dd * chi_max² ≥ D · cL · dd · cR = D · slice_stride`
  (impl :274,334). Step 1 Step1Result-in-T1 dies after Step 2's
  T1@WW→T2 read; Step 3 then writes per-n scratch of size
  D · slice_stride into T1. Sibling-check: dmrg2-gpu :91 sets
  `t_max = D_mpo_ * dd * chi_max² * sizeof(Scalar)` — same formula.
  Lifetime sequential (Step 1 dead before Step 3 writes), so size
  required = max(D·dd·cL·cR, D·cL·dd·cR) = D·dd·cL·cR. **OK.**
- **Multi-gpu d_ones_D allocation**: D_mpo_ Scalars per device, init'd
  once in allocate_device_resources (:352-356), freed in
  free_gpu_resources (:516). Mirrors dmrg-gpu :181 / dmrg2-gpu :177 /
  pdmrg-gpu :355 (per-stream). **OK.**

## SUMMARY

Round 18: **0 CRITICAL, 0 HIGH**, 1 MEDIUM (M17 sibling-kernel
duplication carries forward; cosmetic, no correctness or perf
impact), 1 NIT carry. The two commits since baseline (`54f2fcf`
CR-D1 + `12d02c5` H1-H4 panels) closed the round-17 backlog cleanly:
H1 split dmrg2-gpu t_absorb_ from t_svd_; H4 ported pdmrg-multi-gpu
Step 3 to the dmrg2-gpu R3-F1 batched pattern (kernel +
gemm_batched + gemv reduce, T1-as-scratch with verified sufficient
sizing). PhaseTimer panel discipline is now uniform across the four
variants — every declared timer has matched begin/end pairs, no
dead infrastructure remains. CR-D1 is genuinely Davidson-only;
-gpu tier verified immune by zero references. **Tier paper-ready
across all 4 variants.** Single remaining MEDIUM is cosmetic kernel
consolidation (pdmrg-multi-gpu's 3 local kernels mirror pdmrg-gpu's
exactly — bytewise duplication that should land in
`common/batch_ptrs_kernels.h`, mirroring the round-16 pattern). No
regression in any prior fix; D_PAD R-env slot, init_mps_*,
PointerModeGuard, non-blocking streams, accurate_svd_gpu (J1),
sparse-MPO compaction all intact at the same sites.
