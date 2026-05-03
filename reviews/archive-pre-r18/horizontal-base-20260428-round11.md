# Horizontal -base review — round 11 — 2026-04-28

HEAD: `1d44d89` (post round-10 conformity).
Scope: `gpu-rocm/{dmrg,dmrg2,pdmrg}-gpu-base/src/*` + `gpu-rocm/common/{scalar_traits.h,accurate_svd_gpu.h,hip_check.h}`.

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | 0 dead members across all three variants |
| B. Behavioral diff | DONE | 0 structural divergences classed as defects |
| C. Docstring verification | DONE | 0 unverified claims |
| D. clangd filter | N-A | clangd unavailable on host (no ROCm headers); A subsumes |
| E. Absence-naming brief | FOLLOWED | tier features absent (correctly), J1 present in pdmrg, single-site warmup/polish |
| F. Workspace-aliasing audit | DONE | 0 OVERRUN; all aliasings sequentially safe |
| G. Sibling fix-propagation | DONE | round-10 H10-multi-WW fix confirmed in -base; round-10 M-opt-rsvd-env immune (no GpuOpts in -base) |

## CRITICALS — block GPU run / paper submission

None.

## HIGHS — fix before next major event

None.

## MEDIUMS — fix when convenient

None net-new.

## NITS — cosmetic

- **N1 (carry-over from round-10, do not re-flag)**: `lanczos_use_1site_` is a
  non-atomic member shared across `parallel_sweep` threads in
  pdmrg-gpu-base. Currently safe (set/cleared only on stream 0 outside
  parallel regions; threads read `false`). Already documented; deferred.
- **N2 (net-new, micro-NIT)**:
  `gpu-rocm/pdmrg-gpu-base/src/pdmrg_gpu_base_impl.h:1222` — comment
  "Reuse ws.d_T1 as the destination for the final theta (GEMM target)"
  but the GEMM at line 1232 actually writes to `ws.d_theta`, not
  `ws.d_T1`. Stale comment from a prior factoring. Cosmetic only;
  no code change needed beyond updating the comment.

## FALSE POSITIVES VERIFIED

- HIP_CHECK / ROCBLAS_CHECK macro shadowing: SINGLE source of truth in
  `gpu-rocm/common/hip_check.h:13-31`. No local redefinitions in any
  -base impl.h. (Round-7 consolidation MEDIUM stays clean.)
- `scalar_traits.h` shadows in `dmrg-gpu-base/src/`, `dmrg2-gpu-base/src/`,
  `pdmrg-gpu-base/src/` are 2-line `#include "../../common/scalar_traits.h"`
  shims (verified by diff). No content duplication.
- All charter-forbidden features absent from impl: 0 hits for
  `hipStreamBeginCapture`, `GpuOpts`, `PhaseTimer`, `use_rsvd|RSVD_`,
  `sparse_mpo|d_w_idx|nonzero_terms_`, `D_PAD|D_mpo_actual_`,
  `block_davidson|d_dav_`, `gemm_batched|d_batch_`. (Mentions in headers
  are all in the "compared to -opt this baseline OMITS" doc block —
  verified.)
- Round-10 H1-final non-blocking streams: present in all three -base.
  `dmrg-gpu-base:38`, `dmrg2-gpu-base:34`, `pdmrg-gpu-base:54` — all use
  `hipStreamCreateWithFlags(..., hipStreamNonBlocking)`.
- Round-10 H10-multi-WW-leak class: pdmrg-gpu-base/dmrg2-gpu-base
  `precompute_WW` already has the `if (d_WW_[site]) hipFree(...)` guard
  before `hipMalloc(&d_WW_[site], ...)` (lines 305-306 + 397-398
  respectively). dmrg-gpu-base has no `d_WW_` (single-site, immune).
- Round-10 M-opt-rsvd-env class: -base has no `GpuOpts`/`opts_`/`use_rsvd_`
  — genuinely immune.
- Round-9 M4 + M4-W set_mpo guards: all three -base have the guard for
  `d_mpo_tensors_[i]`, `d_W_left_[i]`, `d_W_right_[i]`, plus `d_WW_[bond]`
  in dmrg2/pdmrg. Verified.
- Round-9 MED-base-1 dead `d_svd_work_` in dmrg2-gpu-base: removed.
  Verified by absence in header (no member declared) and impl (no
  hipMalloc/hipFree). dmrg2-gpu-base writes `S·Vh` directly into
  `d_mps_tensors_[site+1]` via the on-device kernel — no scratch needed.
  In contrast dmrg-gpu-base and pdmrg-gpu-base genuinely need `d_svd_work_`
  / `ws.d_svd_work` as a staging buffer for the subsequent absorb-GEMM
  into `MPS[site±1]`; both members hit ≥3 times in their impl files
  (live).
- Round-8 C-new1 canonical-Vh swap in pdmrg-gpu-base: verified at
  `pdmrg_gpu_base_impl.h:1321-1339`. `ws.d_Vh_canonical` allocated at
  line 154, freed at 217, and used to back the R_env build with norm = I
  rather than S². Mirrors pdmrg-gpu and pdmrg-gpu-opt.
- Round-7 H9 dead-sweep removal: `sweep_LR_full` / `sweep_RL_full`
  (two-site full-chain) absent from pdmrg-gpu-base header and impl.
  Only `sweep_LR_full_1site` / `sweep_RL_full_1site` (single-site,
  CLAUDE.md compliant) remain; warmup at line 1366 and polish at
  1439 both call the `_1site` variants.
- Round-7 H5 fused-WW docstring: `dmrg2_gpu_base.h:30-35` correctly
  documents WW precompute as host-side in EVERY tier.
- Round-7 M2 d_T3 removal: no `d_T3` symbol in any -base impl/header.
- J1 lock (Stoudenmire required for pdmrg): single live call site at
  `pdmrg_gpu_base_impl.h:1267` inside `merge_and_optimize_boundaries`,
  and `ws.asvd.allocate(...)` per-stream at line 167. AsvdScratch
  released in dtor at line 221. Header docstring lines 30-33 and 174-178
  call it out as mandatory.
- CLAUDE.md PDMRG rule compliance: `n_warmup=1`, `n_polish=0` defaults
  (header line 63-64); both `n_warmup`/`n_polish` configurable via
  `run()` parameters; warmup and polish loops use `_1site` sweeps
  exclusively. No two-site full-chain sweep helpers exist in this
  variant. Confirmed at impl lines 1365-1371 (warmup) and 1436-1448
  (polish).
- pdmrg-gpu-base form_theta_with_V scratch aliasing: `ws.d_svd_S` is
  reused to hold V (chi_bond RealTypes) before the subsequent SVD
  inside `merge_and_optimize_boundaries` overwrites it with singular
  values. Sequentially safe on a single stream. Sized for
  `svd_max_k = chi_max*d` which is ≥ chi_bond. OK. Comment at
  `pdmrg_gpu_base_impl.h:1206-1207` documents the intent explicitly.
- Symbol-usage scan: every private member of all three classes hits
  ≥3 times in its impl file (alloc + free + ≥1 use). The svd-output
  scalars (`d_svd_E_`, `d_svd_info_`, `d_svdj_residual_`,
  `d_svdj_n_sweeps_`) all 3-hit at alloc+free+gesvd_auto-arglist —
  verified live by `Traits::rocsolver_gesvd_auto` signature in
  `common/scalar_traits.h:178` and `:376`.

## SUMMARY

The -base tier is clean. Net-new findings this round: **zero CRITICAL,
zero HIGH, zero MEDIUM, one micro-NIT (stale comment in
form_theta_with_V).**

All seven techniques (A-G; D is N-A on this host) ran in full and
verified the round-7 / round-8 / round-9 / round-10 fixes intact. The
round-10 H10-multi-WW-leak fix-class explicitly checked at the d_WW
allocation site in both dmrg2-gpu-base and pdmrg-gpu-base (guard
present); dmrg-gpu-base has no d_WW (single-site, immune). The
round-10 M-opt-rsvd-env fix-class is structurally inapplicable to -base
(no `GpuOpts`/`use_rsvd_` member exists). The round-10 H1-final
non-blocking-stream propagation verified at every stream-creation site
(3/3).

This is the second consecutive clean horizontal-review-base (round-10
also returned net-new = 0 above the known `lanczos_use_1site_` NIT).
The trend from the round-10 summary table holds: methodology hardening
+ propagation gaps closed → rounds 9, 10, 11 produced 4, 2, 0
net-new substantive findings respectively.

Recommendation: **the -base tier is gating-clean for the GPU run.**
The N2 stale-comment NIT can be batched into the next post-G1 cleanup
sweep or fixed inline; it has no behavioral or correctness impact.
