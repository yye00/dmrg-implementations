# Vertical review — dmrg2 family — 2026-05-01 round-16

HEAD: post-`187fddf`. Baseline: round-15 at `f40140d` /
`reviews/vertical-dmrg2-20260501-round15.md`. Scope per round-16 brief:
`gpu-rocm/dmrg2-gpu-base/`, `gpu-rocm/dmrg2-gpu/`,
`gpu-rocm/dmrg2-gpu-opt/`. Four commits land between baseline and HEAD —
`69da5b4` (PhaseTimer panel propagation), `5355c06` (H2-opt host-batch
shared header), `abd88b9` (H2-opt sweep dmrg2-gpu-opt + pdmrg-gpu-opt),
`187fddf` (pdmrg-gpu-opt Step 3 fallback only — no dmrg2 touch).

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | only `lanczos_fused_*` and DMRG2 class members live; no dead members |
| B. Behavioral diff | DONE | base↔gpu↔opt deltas all intentional; `t_absorb_` init-only across whole family (pre-existing) |
| C. Docstring verification | DONE | `setup_batch_ptrs_*` shared-header claim verified; PhaseTimer 5-phase panel claim verified |
| D. clangd filter | N-A | no clangd locally; A subsumes |
| E. Absence-naming brief | FOLLOWED | -base, -gpu, -opt expected-feature checklists pass |
| F. Workspace-aliasing audit | DONE | 6 shared buffers re-checked; 0 OVERRUN. New `d_batch_*_env_` size still ≥ `D*d` (impl:179-184) |
| G. Sibling fix-propagation | DONE | 4 round-15→16 fixes traced; 0 MISSING in dmrg2 family |

## Regression-watch verification (round-15→round-16)

### 1. H2-opt host-batch propagation — INTACT

`grep "std::vector<Scalar\*> h_A" gpu-rocm/dmrg2-gpu*` returns **0 hits**.
`grep "hipLaunchKernelGGL.*setup_batch_ptrs" gpu-rocm/dmrg2-gpu-opt/`
shows the 4 expected sites:
- impl:681 `setup_batch_ptrs_wd_twosite_sparse` (apply_heff_two_site Step 1 sparse).
- impl:698 `setup_batch_ptrs_wd_twosite` (apply_heff_two_site Step 1 dense).
- impl:821 `setup_batch_ptrs_wd` (update_left_env Step 1).
- impl:894 `setup_batch_ptrs_sw` (update_right_env Step 1, new variant for
  swapped indexing).

dmrg2-gpu uses the shared header (`#include "../../common/batch_ptrs_kernels.h"`
at impl:17) and contains **zero local `__global__` setup_batch_ptrs_*
definitions** — only the 2 lanczos_fused kernels remain file-local
(impl:24, 40), which is intentional (lanczos-specific, not batch-ptrs).
**OK.**

### 2. dmrg2-gpu-opt PhaseTimer panel — INTACT

All 5 declared timers from `dmrg2_gpu_opt.h:234-238` have begin/end:
- `t_apply_heff_`: begin impl:631; cache-hit early-return end impl:650;
  main path end impl:791.
- `t_env_update_`: begin/end at impl:807/865 (update_left) and 879/934
  (update_right) — single timer covers both, matching the header
  comment.
- `t_lanczos_`: Lanczos main impl:976/1211; Davidson begin impl:1447
  AFTER the dim≤2b Lanczos-fallback short-circuit; Davidson early-return
  ends at impl:1530 (info!=0 fallback), 1562 (residual converged),
  1595 (n_new==0), 1646 (n_good==0); main-loop exit end impl:1704.
- `t_svd_`: begin impl:1225; end impl:1422. (Single end at function
  exit; `use_cpu_svd_` and RSVD branches share it.)
- `t_absorb_`: init-only at impl:1899 — **not** instrumented around
  the U·S / S·Vh absorb step. **Pre-existing across whole family**
  (dmrg2-gpu, dmrg-gpu-opt, pdmrg-gpu-opt all init-only). The
  `t.calls()==0` skip-row at impl:1907 means no spurious row in
  `report_timers`. Not a regression. Logged as round-15 awareness.

### 3. dmrg2-gpu-base lanczos PointerModeGuard scope — INTACT

Three PointerModeGuard scopes in `lanczos_eigensolver` (impl:588, 609,
686). The per-iter guard at impl:609 is opened **after**
`apply_heff_two_site(...)` at impl:606 and closed before the next iter's
apply_heff. apply_heff_two_site uses host-stack `&one`/`&zero_val` (5+
gemm sites at impl:324-541) — needs default host pointer mode. Round-14
H1-base fix verified intact. **OK.**

### 4. Shared header content (`common/batch_ptrs_kernels.h`)

12 templated kernels: 6 single-site (`_wd`, `_wd_sparse`, `_sw`, `_env3`,
`_step3`, `_step3_sparse`) + 6 two-site (`_wd_twosite`, `_wd_twosite_sparse`,
`_step3_twosite_full`, `_step3_twosite_full_sparse`,
`_wd_twosite_linear`, +1 already in dmrg-only). dmrg2-gpu uses 6 kernels
(includes `_env3`, `_step3_twosite_full`, `_step3_twosite_full_sparse`
at impl:725, 742, 839, 918), dmrg2-gpu-opt uses 4 (impl:681, 698, 821,
894). No symbol clashes; namespace via templated function = OK.

## Round-13/-12/-8 carry-forward (re-verified)

- **CR-D1 dav_work_sz** (-opt impl:265-268): unchanged. Inner-loop
  concurrent regions: residuals at offset 0 (impl:1570), overlap at
  `n_new·dim` (impl:1602-1603). Restart path (impl:1656-1664) uses
  sequential reuse of `d_dav_work_`. Allocated ≥ required. **OK.**
- **Round-6 dual-stream env-pipeline + direction-L MPS-write reorder**:
  -opt impl:1391/1421, -gpu impl:1414/1439. **INTACT.**
- **D_PAD precompute_fused_mpo OOB**: -opt impl:506-541 and -gpu
  impl:485-525 both bound inner loops by `D_act` while writing into
  padded `D_use*dd` stride. **INTACT.**
- **H1-base apply_heff scope** (round-14): impl:609 inner guard scoped
  AFTER apply_heff. **INTACT.**
- **Non-blocking streams**, **rocsolver_syevd in Davidson**,
  **M14-final `set_quiet`**: all intact.

## Sibling propagation cross-check (technique G)

- **H2-opt-host-batch** fixed in dmrg-gpu-opt (`5355c06`) — propagated to
  dmrg2-gpu-opt (`abd88b9`). dmrg2-gpu-base immune (single-stream, no
  batched-GEMM Step 1). **All siblings clean.**
- **PhaseTimer 5-phase panel** propagated to dmrg2-gpu-opt
  (`69da5b4`). dmrg2-gpu-base immune (no `opts_.profile`). **OK.**
- **H-opt-pdmrg-davidson-syev** (`5355c06`): pdmrg-only. dmrg2-gpu-opt
  was already on rocsolver_syevd (round-7 H6). Genuinely immune. **OK.**

## CRITICALS

None.

## HIGHS

None within charter.

## MEDIUMS

None within charter.

## NITS

None.

## FALSE POSITIVES VERIFIED

- **dmrg2-gpu-opt lanczos host h_alpha/h_beta + per-iter D2H of
  `&alpha_result` / `&beta_i`** at impl:984-985, 1024, 1084. Looks like
  a "no host roundtrips per sweep" violation. Verification: this is the
  `lanczos_eigensolver` codepath, identical to `dmrg-gpu-opt`
  sibling at impl:974-1075. The sibling `dmrg2-gpu` lanczos uses
  device-pointer-mode dot/nrm2 with `d_alpha_dev_`/`d_beta_dev_`
  (impl:1009-1012, 1052-1055). Pre-existing axis-5 sibling-divergence
  pattern, NOT introduced by any round-15→16 commit. Out of scope for
  round-16 regression watch. **Awareness only.**
- **dmrg2-gpu-opt RSVD per-call `std::vector<Scalar> h_omega` + host
  random fill** at impl:1292-1297. Same pattern in dmrg-gpu-opt:1299,
  dmrg-gpu:1165, dmrg2-gpu:1206. Family-wide pre-existing pattern.
  Out of scope.
- **dmrg2-gpu-opt `opts_.fuse_lanczos` per-call `hipMalloc`/`hipFree`
  of 3 small device scratches** at impl:993-995, 1151-1153. Mirrors
  dmrg-gpu-opt:983-985, 1083-1085. Off by default
  (`DMRG_GPU_OPT_FUSE_LANCZOS` env-var opt-in). Off the default code
  path. Pre-existing.

## SIBLING-PROPAGATION OBSERVATION (out-of-charter, awareness)

The `t_absorb_` init-only pattern (declared, init-ed, never has
begin/end called) is family-wide: dmrg-gpu-opt, dmrg2-gpu, dmrg2-gpu-opt,
pdmrg-gpu, pdmrg-gpu-opt all init without begin/end. Only dmrg-gpu wires
`t_absorb_` properly (impl:1269/1295/1316/1341). The `t.calls()==0`
skip-row in every `report_timers` makes this benign at runtime. Could be
addressed in a future "absorb instrumentation" sweep but is **not** a
regression and **not** a round-16 finding.

## SUMMARY

Round-16 returns **0 critical, 0 high, 0 medium, 0 nits** for the dmrg2
family across all three tiers. The four regression-watch items from the
round-15→16 commit cluster — H2-opt host-batch elimination via shared
`common/batch_ptrs_kernels.h` (3 hipMemcpyAsync H2D per Lanczos iter
removed at 4 sites), PhaseTimer 5-phase panel propagation in
dmrg2-gpu-opt, dmrg2-gpu shared-header inclusion with zero local
batch-ptr kernel defs, and dmrg2-gpu-base lanczos PointerModeGuard
scope — are all intact and verified at the cited file:line locations.
The round-13/-12/-8 carry-forward set (CR-D1 sizing, round-6 direction-L
reorder + dual-stream events, D_PAD fix, non-blocking streams, syevd in
Davidson) remains healthy. Three pre-existing CPU-on-non-default-or-
shared-pattern sites (lanczos host scalars, RSVD h_omega, fuse_lanczos
hipMalloc) are documented as false-positives — they are family-wide
patterns, not round-16 regressions, and the lanczos host-scalar one is
also present in the dmrg-gpu-opt sibling so this is a class-wide axis-5
candidate for a future dedicated sweep, not a round-16 finding. This is
the sixth consecutive zero-finding sub-review for the dmrg2 family
within charter. Block GPU run? **NO**, family is ready.

Self-audit: all seven techniques completed (D N-A); regression-watch
list explicitly traced for the four round-15→16 dmrg2-touching items
plus carry-forwards; -base brought into scope and verified; technique G
sibling propagation traced for both H2-opt and PhaseTimer fixes across
the dmrg2 tiers and confirmed clean. Verdict: **READY**.

(Length: ~790 words.)
