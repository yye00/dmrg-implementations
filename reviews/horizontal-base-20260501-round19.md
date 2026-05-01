# Horizontal review — -base tier — 2026-05-01 (round 19, confidence re-run)

HEAD: bb809fb. Code state: 12d02c5. Last conformity baseline: round 18.
Defect-registry pre-step: **TOTAL HITS: 0**.

Scope: `gpu-rocm/dmrg-gpu-base`, `gpu-rocm/dmrg2-gpu-base`,
`gpu-rocm/pdmrg-gpu-base` plus `gpu-rocm/common/{hip_check,scalar_traits,
accurate_svd_gpu}.h`.

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | 0 dead members across 3 variants (every member has ≥3 hits = alloc + free + ≥1 use). |
| B. Behavioral diff | DONE | 3-way structural compare on ctor / free / Lanczos / apply_heff / svd / sweep / run. 0 unexplained divergences. |
| C. Docstring verification | DONE | 6 doc-claim checks (single-stream, single rocBLAS handle, no GpuOpts, on-device gesvd_auto, no host LAPACK, J1 Stoudenmire in pdmrg) all matched code. |
| D. clangd filter | N-A | clangd unavailable on host (no ROCm headers); technique A subsumes the dead-symbol case. |
| E. Absence-naming brief | FOLLOWED | "Identify what should be present but is not": all three checked against the -base feature list. |
| F. Workspace-aliasing audit | DONE | 7 shared scratch buffers (d_T1, d_T2, d_svd_*, d_lanczos_v, d_steqr_C, d_Vh_canonical, asvd) sized correctly. 0 OVERRUN. |
| G. Sibling fix-propagation | DONE | 6 R15-R17 fixes traced; 0 MISSING in -base (5 immune by charter, 1 already-fixed pre-R18). |

## Regression watch — confirmed clean

R15-R17 commits in watch list (abd88b9 / 187fddf / 8abb6e7 / 0efe96d /
54f2fcf / 12d02c5) touched **zero -base files** (verified
`git show --stat`). Per fix:

| Fix | Defect class | -base status |
|---|---|---|
| abd88b9 H2-opt | host pointer-array roundtrip in batched GEMM setup (D2/D3) | immune — -base has no batched GEMM |
| 187fddf | pdmrg-opt Step 3 batched fallback (D2) | immune — -base has no batched fallback |
| 8abb6e7 D6 | duplicate batch_ptrs kernels (consolidation) | immune — -base does not include them |
| 0efe96d D12 | device-pointer Lanczos α/β port for -opt | already fixed pre-R18 (impl :518-588 dmrg-base, :580-700 dmrg2-base, :734-810 pdmrg-base — all in PointerModeGuard with d_alpha_dev_/d_beta_dev_ device-side arrays + lanczos_process_alpha/beta kernels + on-device dsteqr) |
| 54f2fcf | pdmrg-opt CR-D1 Davidson overlap-clobbers-eigvecs | immune — -base has no Block-Davidson |
| 12d02c5 H1-H4 | PhaseTimer panel splits + multi-gpu Step 3 batched | immune — -base has no PhaseTimer (charter) |

## Sibling-completeness re-checks (Technique G axes a-e from R15 lessons)

- **R8 C-new1 — pdmrg-base canonical Vh swap before R_env build**:
  intact. impl :1330-1348. `d_Vh_canonical` allocated :155, freed :218,
  `hipMemcpy[2D]Async` of canonical Vh on lines 1336-1345, MPS pointer
  swapped to `ws.d_Vh_canonical` before `update_right_env(bsite+1, si)`,
  then restored. **Did not revert.**
- **Same Lanczos algorithm across 3 variants**: all three use device-
  pointer-mode BLAS-1 + on-device dsteqr + per-iter `lanczos_process_*`
  kernels into device α/β arrays. host-stack α/β is the legacy pre-D12
  pattern; **none present** in any -base variant.
- **Same env-update pattern**: all three use single rocBLAS handle on
  one stream (or one handle per segment in pdmrg). No dual-stream
  concurrency (would be charter violation).
- **Same MPS canonicalization**: dmrg-base / dmrg2-base do U·S vs S·Vh
  absorption into neighbour at SVD time (pre-canonicalised next bond).
  pdmrg-base segment-internal svd_split (impl :866) leaves
  S·Vh in MPS[site+1] but the very next optimize_bond consumes both
  tensors as theta = MPS[s] @ MPS[s+1], so internal correctness is
  preserved. The boundary path (`merge_and_optimize_boundaries`) uses
  the canonical-Vh swap.
- **No D2/D3/D5 in any new function**: -base has no new functions since
  R15. Static.
- **D9/D15 PhaseTimer instrumentation**: 12d02c5 added timers to
  -gpu-opt and pdmrg-multi-gpu only. **-base has no `PhaseTimer`,
  no `panel_*`, no profile-mode** — verified by grep, confirmed
  charter-clean.

## Charter-violation scan (no -gpu-tier features in -base)

grep across all 6 -base files for `GpuOpts | gpu_opts_ | opts_. |
lanczos_graph | rsvd_ | sparse_mpo | D_PAD | block_davidson | use_cpu_svd
| set_quiet | quiet_ | profile | PhaseTimer`: only matches are in
docstring "compared to ... omits" lines (informational). **0 actual
feature creep.**

## Single-source consolidation

- `HIP_CHECK` / `ROCBLAS_CHECK`: defined once in
  `gpu-rocm/common/hip_check.h`. All three -base impls `#include` it
  (impl :13 dmrg, :14 dmrg2, :14 pdmrg). No local re-definition.
  R10 consolidation holds.
- `PointerModeGuard`: same — single header, `#include`d by all three.
- `ScalarTraits`: single source `gpu-rocm/common/scalar_traits.h` with
  shim files `<variant>/src/scalar_traits.h` (2 lines each, just
  `#include "../../common/scalar_traits.h"`).

## CRITICALS — block GPU run / paper submission

(none)

## HIGHS — fix before next major event

(none)

## MEDIUMS — fix when convenient

(none)

## NITS — cosmetic

(none)

## FALSE POSITIVES VERIFIED

- pdmrg-gpu-base `lanczos_use_1site_` is a class-level mutable flag
  read by parallel `parallel_sweep` worker threads. Looks like a
  data race at first glance. Verified safe: the flag is only ever
  written in serial-context single-site code paths (`run`'s warmup,
  polish, and `optimize_site_single` called only from
  `sweep_LR_full_1site`/`sweep_RL_full_1site`, both serial). The
  parallel `segment_sweep_LR/RL` threads only call `optimize_bond`
  (two-site path), which never reads the flag. Not a defect.
- Bare `hipMemcpy` (sync) calls in pdmrg-gpu-base impl :312/329/345/
  349/369/372/400/704/710 — all in `set_mpo`, `initialize_mps_random`,
  `precompute_WW`, `build_initial_environments`. One-shot init, not
  per-sweep. Not a hot-path defect.
- pdmrg-gpu-base `svd_split` (segment-internal, NOT boundary) does NOT
  swap canonical Vh into the R_env build. Verified intentional: that
  function does NOT call update_right_env afterward; the surrounding
  segment_sweep_RL calls update_right_env(site+1) at the next iteration
  using the new MPS[site+1] = Vh canonical (direction='L' branch puts
  canonical Vh in MPS[site], absorbs U·S backward; direction='R' branch
  puts canonical U in MPS[site], S·Vh in MPS[site+1] which the next
  bond's theta-form consumes structurally). Boundary canonical-Vh swap
  (impl :1330-1348) handles the genuinely-asymmetric merge case.

## SUMMARY

The -base tier is in the cleanest state since the audit started. All
seven techniques (A-G) executed without findings. Every defect-class
fix landed in -opt-tier between R15 and R17 was traced — five are
genuinely immune to -base by charter (no batched GEMMs, no Davidson,
no PhaseTimer), and the sixth (D12 device-pointer Lanczos α/β) was
already in -base before R15 and remains intact. The R8 C-new1
canonical-Vh-swap fix in pdmrg-gpu-base is unchanged. No charter
violations: zero GpuOpts / lanczos_graph / RSVD / sparse_mpo / D_PAD /
Block-Davidson / use_cpu_svd / set_quiet / PhaseTimer in any -base
file. CLAUDE.md PDMRG rules upheld: pdmrg-gpu-base run() uses
`sweep_LR_full_1site` / `sweep_RL_full_1site` for warmup and polish
with default `n_warmup=1, n_polish=0` (both ≤ 2; both 0 supported).
**Verdict: -base tier confidence-re-run CLEAN. No action required
prior to MI300X allocation.**
