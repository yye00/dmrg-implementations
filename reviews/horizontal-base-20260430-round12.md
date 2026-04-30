# Horizontal -base review — round 12 — 2026-04-30

HEAD: `a457fdc`. Round-11 baseline: `2d55d90`. Watch list: c3d3e50 (round-11
spotless cleanup), round-9 MED-base-1 (dead `d_svd_work_`), round-9 M4-W
(set_mpo guards on `d_W_left_/d_W_right_/d_WW_`), J1 lock (Stoudenmire in
pdmrg-gpu-base).

Scope: `gpu-rocm/{dmrg,dmrg2,pdmrg}-gpu-base/src/*` +
`gpu-rocm/common/{scalar_traits.h, accurate_svd_gpu.h}`. All three -base
variants exist as their own subdirectories.

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | 0 dead members across all three variants (all ≥3 hits = alloc + free + ≥1 use) |
| B. Behavioral diff | DONE | constructor, free, set_mpo, sweep, run patterns conform; 0 defects |
| C. Docstring verification | DONE | non-blocking-stream + Stoudenmire claims verified at hipStreamCreateWithFlags + accurate_svd_gpu call sites |
| D. clangd filter | N-A | clangd unavailable on host (no ROCm headers); A subsumes |
| E. Absence-naming brief | FOLLOWED | tier-forbidden features absent from impl; J1 present in pdmrg; single-site warmup/polish |
| F. Workspace-aliasing audit | DONE | no -base touched since round-11; pdmrg-gpu-base form_theta_with_V `d_svd_S` reuse for V re-verified sequentially safe |
| G. Sibling fix-propagation | DONE | c3d3e50 dead-buffer classes traced; -base genuinely immune (live use) |
| Self-audit | DONE | regression watch on 2d55d90 baseline + cleanup commit clean |

## CRITICALS

None.

## HIGHS

None.

## MEDIUMS

None net-new.

## NITS

- **N1 (carry-over rounds 10–11)**: `lanczos_use_1site_` is non-atomic and
  shared across `parallel_sweep` threads in pdmrg-gpu-base. Currently safe
  (set/cleared only on stream 0 outside parallel regions; threads read
  `false`). Documented; deferred. [pdmrg-gpu-base: pdmrg_gpu_base.h:184]
- **N2 (carry-over round-11, micro)**: comment "Reuse ws.d_T1 as the
  destination for the final theta (GEMM target)" but the GEMM at line 1232
  writes to `ws.d_theta`, not `ws.d_T1`. Stale comment.
  [pdmrg-gpu-base: pdmrg_gpu_base_impl.h:1222]

## FALSE POSITIVES VERIFIED

- **c3d3e50 cleanup propagation to -base (Technique G)**: the round-11
  cleanup removed 8 sync-free Lanczos scratches (`d_dot_result_`,
  `d_nrm2_result_`, `d_inv_nrm_`, `d_neg_alpha_`, `d_neg_overlap_`,
  `d_neg_beta_scalars_`, `d_alpha_dev_`, `d_beta_dev_`) from
  `dmrg-gpu-opt`. These same buffers are PRESENT in dmrg-gpu-base,
  dmrg2-gpu-base, and pdmrg-gpu-base (as `ws.*` for pdmrg). I verified each
  is genuinely live in -base — not dead-by-copy-paste. dmrg-gpu-base has
  6/8/9/5/6/6/5/5 hits respectively; all wire into the on-device sync-free
  Lanczos loop (`Traits::dot/nrm2/axpy/scal_real` calls in device-pointer
  mode + `kernel_alpha_*`/`kernel_beta_*` kernels). dmrg-gpu-opt switched
  to a different code path (local-scope `d_neg_alpha_scr` allocated inside
  the graph-captured Lanczos function) and so the persistent member became
  dead in -opt only. **-base is correct to keep them; not a defect.**
- **Dead host SVD buffers (h_svd_U_/Vh_/tmp_)**: removed from
  `dmrg-gpu`/`dmrg2-gpu`/`dmrg-gpu-opt`/`dmrg2-gpu-opt` by c3d3e50.
  Searched -base headers for these symbols — zero hits. -base never had
  them. Genuinely immune.
- **`d_const_one_/d_const_zero_/d_const_neg_one_/d_ones_D_`**: removed
  from `dmrg-gpu-opt` by c3d3e50. Searched all three -base headers — zero
  hits. -base never had them. Genuinely immune.
- **`h_dav_V_copy/h_rsvd_B/h_rsvd_U_small`**: removed from
  `pdmrg-gpu-opt`. Davidson and RSVD are charter-forbidden in -base; the
  buffers were structurally inapplicable.
- **Round-9 MED-base-1 (`d_svd_work_` in dmrg2-gpu-base)**: confirmed
  ABSENT from header (no member declared) and impl (no hipMalloc/hipFree).
  dmrg2-gpu-base writes `S·Vh` directly into `d_mps_tensors_[site+1]` via
  on-device kernel. dmrg-gpu-base and pdmrg-gpu-base genuinely need
  `d_svd_work_` / `ws.d_svd_work` (3+ hits each in impl) for the absorb-
  GEMM into adjacent MPS — present and live.
- **Round-9 M4-W set_mpo guards** for `d_W_left_/d_W_right_/d_WW_`:
  verified at `dmrg-gpu-base:225,244,248`, `dmrg2-gpu-base:224,242,246,305`,
  `pdmrg-gpu-base:326,342,346,397`. All three sets present.
- **Round-8 C-new1 canonical-Vh swap**: verified at
  `pdmrg-gpu-base:217` (free) + `:154` (alloc) + boundary R_env build
  uses `ws.d_Vh_canonical`. Mirrors pdmrg-gpu and pdmrg-gpu-opt.
- **J1 lock (Stoudenmire in pdmrg-gpu-base)**: single live call site at
  `pdmrg_gpu_base_impl.h:1267` inside `merge_and_optimize_boundaries`,
  with `ws.asvd.allocate(...)` per-stream at `:167` and free in dtor.
  Header `:9` includes `../../common/accurate_svd_gpu.h`; header `:177`
  documents J1 mandate. The c3d3e50 deletion of legacy
  `gpu-rocm/pdmrg-gpu/src/accurate_svd.h` (host-LAPACK Stoudenmire) does
  not affect pdmrg-gpu-base — pdmrg-gpu-base never included that file
  (verified by `git log --diff-filter=A`); it has always used the common
  GPU-native header.
- **Charter-forbidden tokens**
  (`hipStreamBeginCapture|GpuOpts|opts_\.|PhaseTimer|use_rsvd|RSVD_|sparse_mpo|d_w_idx|nonzero_terms_|D_PAD|D_mpo_actual_|block_davidson|d_dav_|gemm_batched|d_batch_`):
  zero impl-code hits in any -base. The two grep hits are comment-only —
  `dmrg_gpu_base_impl.h:256` ("no gemm_batched") and
  `dmrg2_gpu_base_impl.h:258` ("with D_PAD/SPARSE_MPO" describing the
  -opt sibling).
- **CLAUDE.md PDMRG rule compliance**: `n_warmup=1`, `n_polish=0`
  defaults at header `:62-64`; both configurable via `run()` parameters.
  Warmup at `:1365-1371` and polish at `:1438-1448` call the `_1site`
  helpers exclusively; no two-site full-chain sweep helpers exist in this
  variant (round-7 H9 fix intact). Defaults are within the ≤2 cap.
- **Non-blocking stream creation** present at `dmrg-gpu-base:38`,
  `dmrg2-gpu-base:34`, `pdmrg-gpu-base:54` (all 3/3).

## SUMMARY

The -base tier remains gating-clean for round 12. **Net-new findings:
zero CRITICAL / HIGH / MEDIUM / NIT.** All round-11 NITs (N1
`lanczos_use_1site_`, N2 stale `ws.d_T1` comment) carry over unchanged.
The c3d3e50 cleanup commit touched only `-gpu` and `-gpu-opt` variants —
no -base file was modified since the 2d55d90 baseline (`git diff
2d55d90..HEAD -- gpu-rocm/{dmrg,dmrg2,pdmrg}-gpu-base/ gpu-rocm/common/`
returns empty). Technique G correctly identified that the dead-buffer
classes removed in -opt are NOT applicable to -base: the persistent
sync-free Lanczos scratches are genuinely live in -base (the persistent
code path that -opt has migrated away from), and the dead host SVD
buffers / device constants never existed in -base in the first place.
The J1 lock and the deleted-legacy-`accurate_svd.h` cross-check are
clean — pdmrg-gpu-base routes through `common/accurate_svd_gpu.h` and
never referenced the deleted host-LAPACK header. This is the third
consecutive clean horizontal-review-base. **Recommendation: -base is
gating-clean for the next GPU run.**
