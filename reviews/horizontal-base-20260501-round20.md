# Horizontal review — -base tier — 2026-05-03 (round 20, second confidence re-run)

HEAD: f650466 (R19 reports landed). Code state: cafd628. Last
conformity baseline: round 19. Defect-registry pre-step:
**TOTAL HITS: 0** (D7+D8 / D9 / D10 / D11 / D13 / D14 / D15 all clean
at HEAD; D13 explicitly skips -base per charter line 200).

Scope: `gpu-rocm/dmrg-gpu-base`, `gpu-rocm/dmrg2-gpu-base`,
`gpu-rocm/pdmrg-gpu-base` plus `gpu-rocm/common/{hip_check,
scalar_traits, accurate_svd_gpu, pointer_mode_guard}.h`.

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | Re-ran on all 3 -base headers vs impls. 0 dead members; every member has alloc + free + ≥1 use. `lanczos_use_1site_` (R19 false-positive) cross-verified again — 8 hits, ctor + 1 read + 3 write-pairs. |
| B. Behavioral diff | DONE | 3-way cross-algorithm structural compare on ctor / `free_gpu_resources` / `set_mpo` / `update_left_env` / `update_right_env` / `lanczos_eigensolver` / `apply_heff*` / SVD path / `run`. 0 unexplained divergences. |
| C. Docstring verification | DONE | 6 doc-claim spots verified: dmrg-base "no batched GEMM" (impl :257 confirms), dmrg2-base "host-side WW build at set_mpo time" (impl :265-311 + comment :253-262), pdmrg-base "uses Stoudenmire `accurate_svd_gpu` at boundaries" (header :30, impl :1276), pdmrg-base "non-blocking streams required" (impl :48-58), pdmrg-base CLAUDE.md "warmup/polish single-site" (impl :1119/1145, :1374/1448), all three "single rocBLAS handle on a single stream" (dmrg/dmrg2 impl :40/:36-37; pdmrg has one handle per segment, header docstring acknowledges). |
| D. clangd filter | N-A | clangd unavailable on host (no ROCm headers); A subsumes the dead-symbol case. |
| E. Absence-naming brief | FOLLOWED | All -base feature checklists honored: HIP_CHECK from common — yes (impl :13); ScalarTraits dispatch — yes (header :7-8 / :47-48); single stream + single rocBLAS handle — yes (dmrg/dmrg2: 1 handle, pdmrg: per-segment per algorithm). For pdmrg: per-segment streams + accurate_svd_gpu present (J1 lock). |
| F. Workspace-aliasing audit | DONE | 8 shared scratch buffers re-checked across 3 variants (`d_T1_/d_T2_`, `d_lanczos_v_`, `d_steqr_C_`, `d_svd_A_/U_/Vh_/work_`, `d_Vh_canonical`, `asvd`). 0 OVERRUN. No commit since R18 touched -base sizing. |
| G. Sibling fix-propagation | DONE | R17-R19 fix list traced; 0 MISSING in -base. cafd628 (only commit since R18 baseline that touches code) only modified pdmrg-multi-gpu. Verified by `git show --stat cafd628`: 0 -base files changed. |

A review with any technique SKIPPED that is not N-A is INVALID. None skipped.

## Regression watch — confirmed clean

Only one code-touching commit since R18: `cafd628` (R19 H19 fix +
D13 widening). `git show --stat cafd628` confirms zero -base files
modified — only `gpu-rocm/pdmrg-multi-gpu/src/pdmrg_multi_gpu_impl.h`
and `.claude/scripts/defect-registry.sh`. All R8-R17 fixes traced in
R19 horizontal-base review remain intact; no further drift to check.

| Carry watch item | -base status |
|---|---|
| M4-ext set_mpo guards (`if (d_mpo_tensors_[i]) HIP_CHECK(hipFree...)`) | dmrg-base impl :226, dmrg2-base impl :225, pdmrg-base impl :327 — all 3 present |
| C-new1 canonical-Vh swap before R_env build (R8 fix in pdmrg-base) | pdmrg-base impl :1330-1348 — `d_Vh_canonical` allocated :155, freed :218, swap-and-restore around `update_right_env(bsite+1)` intact |
| Stoudenmire `accurate_svd_gpu` in pdmrg-base (J1 lock) | pdmrg-base header :9 (#include), :30 (docstring), :177 (`AsvdScratch` in StreamWorkspace), impl :168 (allocate), :222 (release), :1276 (call site) — present and live |
| Dead-buffer cleanup (no `h_batch_*_pinned`, no `d_T3` dead) | grep across all 3 -base headers + impls → 0 hits. Clean. |
| PhaseTimer panels in -base | None expected (charter); grep for `PhaseTimer\|t_.*_\.begin\|panel_\|profile_mode` returned 0 hits. Clean. |
| D13 widening (cafd628) does not false-positive on -base | Verified: defect-registry.sh line 200 `[[ "$variant" == *-base ]] && continue;` — D13 explicitly skips -base by charter design. |

## Cross-family conformity tests (per brief)

- **`initialize_mps_random` signatures**: all 3 share
  `void initialize_mps_random(double scale = 0.1)` — header lines
  dmrg-base :45, dmrg2-base :56, pdmrg-base :57. (No
  `initialize_mps_product` / `initialize_mps_neel` in any -base —
  consistent absence; brief expects shared signatures across the 3
  but does not require non-random inits in -base.)
- **`build_initial_environments` pattern**: all 3 H2D the boundary
  L/R vectors then sweep one direction. dmrg-base sweeps R-to-L only
  (impl :478-481). dmrg2-base same (impl :567-570). pdmrg-base
  sweeps BOTH directions (impl :714-722) on `streams_[0]` because
  segment kernels need both envs initialized for any starting site.
  This divergence is **algorithmic, not a defect** — pdmrg's segment
  partitioning requires interior segments to start from non-edge
  envs.
- **apply_heff naming**: dmrg-base has `apply_heff` (single-site, impl
  :261). dmrg2-base has `apply_heff_two_site` (impl :339). pdmrg-base
  has both `apply_heff_two_site` (impl :432) and
  `apply_heff_single_site` (impl :503). Per CLAUDE.md PDMRG rules
  (single-site default for warmup/polish) the dual API is required.
  Naming convention consistent: two-site path uses `_two_site`
  suffix in both dmrg2-base and pdmrg-base.

## Charter-violation scan

`grep -nE 'GpuOpts|gpu_opts_|opts_\.|lanczos_graph|rsvd_|sparse_mpo|D_PAD|block_davidson|use_cpu_svd|set_quiet|PhaseTimer|begin_panel|end_panel|profile_mode'`
across all 6 -base files: only matches are docstring "compared to ...
omits" lines (informational). **0 actual feature creep.**
`gemm_batched` grep: only matches in docstring "where the optimized
variant uses gemm_batched" — no actual batched calls in any -base
impl.

## Single-source consolidation

- `HIP_CHECK` / `ROCBLAS_CHECK` defined once in
  `gpu-rocm/common/hip_check.h`, included by all 3 -base impls (`impl
  :13` dmrg, `:14` dmrg2, `:15` pdmrg). No local re-definition.
- `PointerModeGuard`: `gpu-rocm/common/pointer_mode_guard.h`,
  included identically in all 3.
- `ScalarTraits`: `gpu-rocm/common/scalar_traits.h`, with 2-line
  shim files at each variant's `src/scalar_traits.h`.

## CRITICALS — block GPU run / paper submission

(none)

## HIGHS — fix before next major event

(none)

## MEDIUMS — fix when convenient

(none)

## NITS — cosmetic

(none)

## FALSE POSITIVES VERIFIED

- **`lanczos_use_1site_` mutable class flag in pdmrg-gpu-base** —
  reads at impl :761; writes at :1100/:1102, :1134/:1136, :1160/
  :1162. All write sites are inside `optimize_site_single` and the
  serial last-site sections of `sweep_LR_full_1site` /
  `sweep_RL_full_1site`. The parallel `segment_sweep_LR/RL` workers
  (called from `parallel_sweep` :1391-1407) only invoke
  `optimize_bond` (two-site path) which calls `lanczos_eigensolver`
  with the flag's two-site default. No data race. Carry from R19;
  re-verified.
- **Bare sync `hipMemcpy` in pdmrg-gpu-base** at impl :312, :329,
  :345, :349, :369, :372, :400, :704, :710 — all in `set_mpo`,
  `initialize_mps_random`, `precompute_WW`,
  `build_initial_environments`. One-shot init, outside timed sweep
  region. Not a hot-path defect (per `feedback_no_host_roundtrips_per_sweep`
  — set-once is allowed).
- **pdmrg-gpu-base segment-internal `svd_split` (impl :866) does NOT
  swap canonical Vh** before the next `update_right_env`. Verified
  intentional: that function does not call `update_right_env`
  itself; the surrounding `segment_sweep_RL` calls it on the next
  iteration with the new tensor. Boundary case (the asymmetric
  merge) is the only place needing the swap, which is in
  `merge_and_optimize_boundaries` impl :1330-1348.
- **D13 detector and -base** — D13 widening in cafd628 added
  `gemm_batched` / `gemm_strided_batched` recognition inside per-wp
  loops; does NOT regress on -base because line 200 of
  `defect-registry.sh` skips `*-base` variants. -base IS
  charter-allowed to use per-wp loops with bare `Traits::gemm` — that
  is the entire point of the baseline tier.
- **pdmrg-gpu-base `precompute_WW` host nested-loop + H2D upload**
  (impl :265-311 / :359-403) — set_mpo() time, single bulk upload,
  outside the timed sweep region. Sibling-consistent with dmrg2-base.
  Not a per-sweep host roundtrip defect.

## SUMMARY

The -base tier remains in the cleanest state observed across this
review series. Second confidence re-run produced the same verdict as
R19: zero criticals, zero highs, zero mediums, zero nits. The single
post-R18 code commit (`cafd628`) touched only pdmrg-multi-gpu;
`git show --stat` confirmed no -base files modified. All seven
methodology techniques (A-G) executed in full. Every R8-R19 fix
traced in the regression watch is intact: M4-ext set_mpo guards in
all 3 variants, C-new1 canonical-Vh swap in pdmrg-base, Stoudenmire
J1 lock in pdmrg-base via `accurate_svd_gpu`, dead-buffer cleanups
holding (no `h_batch_*_pinned`, no `d_T3`). D13 widening in cafd628
correctly excludes -base by charter. CLAUDE.md PDMRG rules upheld:
pdmrg-gpu-base `run()` defaults `n_warmup=1, n_polish=0`, both
single-site (`sweep_LR_full_1site` / `sweep_RL_full_1site`), both
≤ 2, both 0 supported. **Verdict: -base tier R20 confidence-re-run
CLEAN. No action required prior to MI300X allocation.**
