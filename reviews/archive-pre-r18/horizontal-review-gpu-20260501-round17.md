# Horizontal review — -gpu tier — round 17 (2026-05-01)

Charter: dmrg-gpu, dmrg2-gpu, pdmrg-gpu, pdmrg-multi-gpu. HEAD `0efe96d`
vs baseline `f40140d`: 6 commits (`69da5b4`, `5355c06`, `abd88b9`,
`187fddf`, `8abb6e7`, `0efe96d`). Within **-gpu** charter only D6
cleanup landed; multi-gpu had zero file deltas.

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | multi-gpu `t_env_update_` still 0 begin/end (M16 carry). `t_absorb_` unwired across all 4 -gpu (forward-compat — registry D9 confirms). |
| B. Behavioral diff | DONE | multi-gpu apply_heff Step 3 still 3-deep host loop (impl.h:802-817 two-site, :1687-1704 single-site) vs R3-F1 batched in single-host triplet (dmrg-gpu :643, dmrg2-gpu :717, pdmrg-gpu :1070). |
| C. Docstring verification | DONE | "round-16 D6 cleanup removed local duplicates" (pdmrg-gpu :88-90) — verified: `git diff f40140d..HEAD` shows 4 sparse kernels deleted; `#include "../../common/batch_ptrs_kernels.h"` added at :22. |
| D. clangd filter | N-A — no ROCm headers on host. |
| E. Absence-naming brief | FOLLOWED | Single-host triplet feature-complete. Multi-gpu still missing R3-F1 Step 3, `t_env_update_` wiring, single-site `t_apply_heff_` PhaseTimer (round-16 carries). |
| F. Workspace-aliasing | DONE | This batch's only -gpu-charter delta is removal of bytewise-identical kernel definitions (pdmrg-gpu impl.h: 71 lines net). Zero buffer sizing or aliasing impact. No OVERRUN. |
| G. Sibling fix-propagation | DONE | 6 fixes traced; D6 propagated correctly (dmrg/dmrg2/pdmrg-gpu all now `#include` shared header); D12 / H2-opt / PhaseTimer-opt are -opt-only fixes (axis-2 cross-tier propagation gap re-verified inappropriate at -gpu tier). 1 new finding: see H17. |

A-G all DONE or N-A. Review valid.

## Regression-watch (round-16 → round-17)

| Watch item | dmrg-gpu | dmrg2-gpu | pdmrg-gpu | multi-gpu |
|---|---|---|---|---|
| `#include common/batch_ptrs_kernels.h` | yes :13 (carry) | yes :17 (carry) | yes :22 NEW | no (M16 carry) |
| Local sparse `__global__` duplicates | 0 | 0 | 0 (down from 4) | 3 (un-promoted, M16 carry) |
| `t_apply_heff_` begin/end count | 3 | 3 | 6 (carry) | 2 (M16 carry) |
| `t_env_update_` begin/end count | 4 | 4 | 4 | 0 (M16 carry) |
| `t_absorb_` begin/end count | 0 | 0 | 0 | 0 (forward-compat) |
| `D_mpo_actual_-1` boundary | :878 | :953 | :1250 | n/a (no D_PAD) |
| `init_mps_product/_neel` | h:35-36 | h:35-36 | h:36-37 | h:47-48 |
| `PointerModeGuard` use-sites | 2 | 2 | 4 | 4 |
| `stream_env_` non-blocking flag | impl.h:84 | impl.h:82 | per-segment :228 | per-device :293 |
| `use_rsvd_ = opts_.rsvd` | direct | direct | direct | impl.h:163 |
| Sparse-MPO compaction | yes | yes | yes | absent (out-of-scope) |

**Regression status**: zero. Round-16 closed 1 of 4 MEDIUMs (M16-pdmrg-gpu-redundant-sparse-kernels via `8abb6e7`); 3 multi-gpu MEDIUMs deferred without regression. M15 close intact at pdmrg-gpu impl.h:1875/1894.

## CRITICALS

None.

## HIGHS

### H17-multi-apply_heff-step3-host-loop — multi-gpu Step 3 host loop (carry from H16)

[`pdmrg-multi-gpu`: `pdmrg_multi_gpu_impl.h:801-817` (two-site) +
`:1687-1704` (single-site)]. Technique B. Multi-gpu untouched this
batch. Single-host triplet uses **R3-F1 full-batched collapse** (1
setup kernel + 1 batched GEMM batch=D*dd + 1 GEMV reduction); multi-gpu
issues `D*d*d` separate `Traits::gemm` calls in 3-deep host loop on
`apply_heff_two_site`, plus `D` separate batched-GEMMs in host loop on
`apply_heff_single_site`. Challenge sizes (D=18, d=2 → up to 72
launches per Lanczos iter per site per device); direct CLAUDE.md "no
host roundtrips per sweep" violation. Not a single-device blocker;
must land before multi-gpu benchmark numbers. Mirror pdmrg-gpu
:1070 / :2031.

## MEDIUMS

### M17-multi-t_env_update-dead — `t_env_update_` declared/init'd but never wired (carry from M16)

[`pdmrg-multi-gpu`: `pdmrg_multi_gpu.h:216,222,240` vs impl.h: 0
begin/end]. Technique A. Sibling counts: 4/4/4/0. Round-13 panel
skip-on-zero hides absence. Same dead-infrastructure pattern as round-7
dmrg2-gpu dual-stream class. Add `begin/end` at `update_left_env`/
`update_right_env` entry/exit (4 lines).

### M17-multi-apply_heff_1s-uninstrumented — single-site no PhaseTimer (carry from M16)

[`pdmrg-multi-gpu`: `pdmrg_multi_gpu_impl.h:1645-1705`]. Technique B
mirror of round-15 M15 (closed for pdmrg-gpu in `69da5b4`, this batch
extended at :1875/1894). No graph cache → no cache-hit balance issue;
simple begin/end with `if (di == 0)` guard. Mirrors pdmrg-gpu :1875/1894.

### M17-multi-untouched-by-host-batch-promotion (carry from M16)

[`pdmrg-multi-gpu`: `pdmrg_multi_gpu_impl.h:25,38,51`].
`setup_heff_ss_step3_ptrs` mirrors common's `setup_batch_ptrs_step3`;
`setup_lenv_step3_ptrs`/`setup_renv_step3_ptrs` mirror
`setup_batch_ptrs_env3`. Defect-class D6 axis-1 propagation gap.
Consolidate during H17 fix.

## NITS

### N13-pmg-error-discard (carry-over)

[`common/pointer_mode_guard.h:16,17,22`] Discards return of
`rocblas_get/set_pointer_mode`. Cosmetic.

## FALSE POSITIVES VERIFIED

- Registry D8 `lapack_gesvd` hits in pdmrg-gpu impl.h:1547,2064 and
  multi-gpu :1367,1735 — verified gated by `if (use_cpu_svd_)` opt-in
  branch (default GPU SVD path is `accurate_svd_gpu`, J1 lock intact).
  Init-time / build-time hits in dmrg-gpu-opt and dmrg2-gpu-opt are out
  of scope for this charter.
- Registry D13 hits target `-base` tier only (out of scope).
- Registry D9 dead `t_absorb_` is forward-compat across all 4 -gpu
  variants (round-16 confirmation re-validated).
- pdmrg-gpu `t_apply_heff_` count 6 = (begin + cache-hit-end + normal-
  end) × 2 functions; balanced — verified by reading impl.h:540-704
  (two-site) and :1872-1894 (single-site).

## SUMMARY

Round 17: 0 CRITICAL, 1 HIGH (H16 carry — multi-gpu Step 3), 3 MEDIUM
(all multi-gpu carries), 1 NIT carry. Within -gpu charter only pdmrg-gpu
changed (`8abb6e7`: D6 removed 71 lines of duplicated sparse kernels +
added single-site PhaseTimer with cache-hit early-return). M16-pdmrg-
redundant-sparse-kernels closed at exact site. Multi-gpu retains 3
MEDIUMs and 1 HIGH unchanged. All round-9..16 watch items intact
(PointerModeGuard, init_mps, M4-W set_mpo guards, non-blocking streams,
D_PAD R-env slot, use_rsvd_); no regressions from the 6 commits.
Single-host triplet paper-ready. Multi-gpu has 1 HIGH performance
defect — not a single-device GPU-run blocker.
