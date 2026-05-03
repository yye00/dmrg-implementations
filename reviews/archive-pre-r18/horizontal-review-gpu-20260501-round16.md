# Horizontal review — -gpu tier — round 16 (2026-05-01)

HEAD: `187fddf`. vs round-15 `f40140d`: 4 commits — `69da5b4`
(PhaseTimer panel propagation + pdmrg-gpu single-site), `5355c06`
(shared `common/batch_ptrs_kernels.h` + pdmrg-gpu-opt syevd),
`abd88b9` (dmrg2-gpu-opt + pdmrg-gpu-opt host-batch elimination),
`187fddf` (pdmrg-gpu-opt Step 3 fallback).

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | multi-gpu `t_env_update_` declared/init'd/printed but 0 begin/end sites — DEAD vs siblings (4 each). |
| B. Behavioral diff | DONE | apply_heff Step 3 R3-F1 batched in dmrg/dmrg2/pdmrg-gpu (impl.h:643/717/1070); multi-gpu still 3-deep host loop with D*dd separate GEMMs (impl.h:802-817). |
| C. Docstring verification | DONE | "no host roundtrips per sweep" re-verified. Init-time host vectors only on default sweep path. |
| D. clangd filter | N-A — no ROCm headers on host. |
| E. Absence-naming brief | FOLLOWED | Single-host triplet feature-complete. Multi-gpu missing R3-F1 batched Step 3, `t_env_update_`/single-site `t_apply_heff_` instrumentation. |
| F. Workspace-aliasing | DONE | Round-15→16 deltas are 3-line PhaseTimer + 1-line `#include` adds. Zero OVERRUN. |
| G. Sibling fix-propagation | DONE | 4 fixes traced four-way; 1 NEW HIGH (R3-F1 multi-gpu absence), 1 NEW MEDIUM (pdmrg-gpu redundant sparse kernels), carries closed. |

A-G: all DONE or N-A. Review valid.

## Regression-watch (round-15 → round-16)

| Watch item | dmrg-gpu | dmrg2-gpu | pdmrg-gpu | multi-gpu |
|---|---|---|---|---|
| Local `__global__ setup_batch_ptrs_*` | 0 | 0 | 4 sparse-only | 3 (un-promoted) |
| Includes `common/batch_ptrs_kernels.h` | yes :13 | yes :17 | no | no |
| pdmrg-gpu single-site cache-hit close | n/a | n/a | impl.h:1935/1954/2090 NEW | n/a |
| `t_apply_heff_` begin/end counts | 3 | 3 | 6 | 2 |
| `t_env_update_` begin/end counts | 4 | 4 | 4 | 0 |
| `use_rsvd_ = opts_.rsvd` | direct | direct | direct | impl.h:163 |
| `D_mpo_actual_-1` boundary | :878 | :953 | :1310 | n/a |
| `init_mps_product/_neel` | h:35-36 | h:35-36 | h:36-37 | h:47-48 |
| `PointerModeGuard` use-sites | 2 | 2 | 4 | 4 |
| `_1site` warmup/polish lock | n/a | n/a | live | live |

`5355c06`/`abd88b9` correctly leave pdmrg-gpu/multi-gpu untouched
(host-batch defect was -opt-tier-only). Round-15 M15 closed at exact
cited line in `69da5b4`.

## CRITICALS

None.

## HIGHS

### H16-multi-apply_heff-step3-host-loop — multi-gpu Step 3 D*dd separate GEMMs

[`pdmrg-multi-gpu`: `pdmrg_multi_gpu_impl.h:801-817` (two-site) +
`:1687-1704` (single-site)]. Technique B (intra-tier sibling diff).
dmrg-gpu (:643), dmrg2-gpu (:717), and pdmrg-gpu (:1070) all use the
**R3-F1 full-batched collapse**: 1 setup kernel + 1 batched GEMM
(batch = D*dd) + 1 GEMV reduction. Multi-gpu still issues `D*dd`
separate `Traits::gemm` calls in a 3-deep host `for` loop on every
`apply_heff_two_site`, plus `D` separate batched GEMMs in a host loop on
`apply_heff_single_site`. For challenge sizes (D=18, d=2 → up to 72
launches per Lanczos iter per site per device) this is the dominant
launch-overhead. Round-12 scoped multi-gpu narrowly, but R3-F1 was in
dmrg-gpu before then — technique-G axis-3 cross-family propagation gap.
Direct CLAUDE.md "no host roundtrips per sweep" violation. Single fix
pattern (mirror pdmrg-gpu :1070 / :2031) lands both single- and two-site.

## MEDIUMS

### M16-multi-t_env_update-dead — multi-gpu `t_env_update_` declared but never wired

[`pdmrg-multi-gpu`: `pdmrg_multi_gpu.h:216,222,240` vs impl.h: 0
begin/end]. Technique A. Sibling counts: 4/4/4/0. Init'd and panel-
printed via `print_if_used`; round-13 skip-on-zero hides absence.
Same dead-infrastructure pattern as round-7 dmrg2-gpu dual-stream class.
Add `begin/end` at update_left_env / update_right_env entry/exit (4
lines).

### M16-multi-apply_heff_1s-uninstrumented — `apply_heff_single_site` no PhaseTimer

[`pdmrg-multi-gpu`: `pdmrg_multi_gpu_impl.h:1645-1705`]. Technique B
mirror of round-15 M15 (closed for pdmrg-gpu in `69da5b4`). No graph
cache → no cache-hit balance issue; simple begin/end with `if (di == 0)`
guard. Mirrors pdmrg-gpu :1935/2090.

### M16-pdmrg-gpu-redundant-sparse-kernels — pdmrg-gpu duplicates 4 common-header kernels

[`pdmrg-gpu`: `pdmrg_gpu_impl.h:88-150`]. Technique G axis-1.
`setup_batch_ptrs_wd_sparse`, `setup_batch_ptrs_step3_full_sparse`,
`setup_batch_ptrs_wd_twosite_sparse`, `setup_batch_ptrs_step3_twosite_full_sparse`
are **bytewise-identical** to `common/batch_ptrs_kernels.h:99-130,194-223`.
dmrg-gpu/dmrg2-gpu swept in `5355c06`/`abd88b9`; pdmrg-gpu missed. Risk:
silent ODR/sibling-drift if future fix lands in only one. Add `#include`,
delete the 4 locals; net negative LOC.

### M16-multi-untouched-by-host-batch-promotion — multi-gpu retains 3 un-promoted kernels

[`pdmrg-multi-gpu`: `pdmrg_multi_gpu_impl.h:25,38,51`].
`setup_heff_ss_step3_ptrs` mirrors common's `setup_batch_ptrs_step3`;
`setup_lenv_step3_ptrs`/`setup_renv_step3_ptrs` mirror `setup_batch_ptrs_env3`.
Axis-1 propagation gap. Consolidate during H16 fix.

## NITS

### N13-pmg-error-discard (carry-over)

[`common/pointer_mode_guard.h:16,17,22`] Discards return of
`rocblas_get/set_pointer_mode`. Cosmetic.

## FALSE POSITIVES VERIFIED

- pdmrg-gpu `t_absorb_` unwired — intentional forward-compat (matches
  multi-gpu); pdmrg has no separate absorb stage.
- multi-gpu `form_theta_with_V` (impl.h:2424) uploads `bs.V`
  (chi_bond ≤ chi_max RealType) per-merge-boundary, NOT per-Lanczos-iter
  or per-bond. Cadence ≪ per-sweep; not a defect.
- All `hipMemcpyHostToDevice` in pdmrg-gpu impl.h:1690-2235 are inside
  `if (use_cpu_svd_)` opt-in branches; default GPU SVD path clean.
- multi-gpu update_left/right_env `for sp` 2-iter outer loop matches
  dmrg-gpu impl.h:761-776 sibling pattern — uniform across all 4
  variants; intentional, not a defect.
- pdmrg-gpu `t_apply_heff_` count of 6 = (begin + cache-hit-end +
  normal-end) × 2 functions; balanced.

## SUMMARY

Round 16: 0 CRITICAL, 1 HIGH (single fix-pattern; both single- and
two-site multi-gpu apply_heff Step 3), 4 MEDIUM, 1 carry-over NIT. The
HIGH is technique-B + technique-G axis-3: round-15's host-batch
propagation campaign (`5355c06`, `abd88b9`) hit the -opt tier
exclusively; multi-gpu sibling carries the same defect class and was not
swept. Round-15 M15 closed at exact site (impl.h:1935/1954/2090).
All round-9..14 watch items intact; no regressions from the 4 commits.

**Recommendation**: -gpu tier single-host triplet (dmrg-gpu, dmrg2-gpu,
pdmrg-gpu) is paper-ready. Multi-gpu has 1 HIGH performance defect class
(R3-F1 missing on apply_heff Step 3 for both site counts) — should land
before multi-gpu benchmark numbers are published. Not a single-device
GPU-run blocker.
