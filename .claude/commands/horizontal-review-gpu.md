---
description: Horizontal review of -gpu tier — compares dmrg-gpu, dmrg2-gpu, pdmrg-gpu for paper-reference conformity, feature-set parity, and shared engineering style.
---

Run a horizontal review of the -gpu tier. Horizontal = different
algorithms, same tier. The audit checks that all three -gpu variants
have the SAME set of paper-reference features, engineered the same
way, with no silent omissions across siblings.

## Scope

Three variants:

- `gpu-rocm/dmrg-gpu/src/dmrg_gpu.h` + `dmrg_gpu_impl.h`
- `gpu-rocm/dmrg2-gpu/src/dmrg2_gpu.h` + `dmrg2_gpu_impl.h`
- `gpu-rocm/pdmrg-gpu/src/pdmrg_gpu.h` + `pdmrg_gpu_impl.h`

Plus: `gpu-rocm/common/scalar_traits.h`, `gpu-rocm/common/gpu_opts.h`,
`gpu-rocm/common/accurate_svd_gpu.h` (pdmrg only — J1).

## Methodology

Read `.claude/review-methodology.md` and follow techniques A through E
in full. Skipping a technique invalidates the review.

## Required -gpu feature set (technique E expectations)

All three -gpu variants MUST have ALL of the following. Anything
missing in one but present in the others is the exact failure mode
that bit us in dmrg2-gpu (dual-stream resources declared but never
wired). Apply technique A (symbol-usage scan) to every concurrency
primitive listed.

1. **`GpuOpts opts_`** — env-var-driven ablation flags:
   `device_k`, `lanczos_graph`, `rsvd`, `sparse_mpo`, `fuse_lanczos`,
   `d_pad`, `profile`. Identical surface across all three.
2. **`PhaseTimer`** instrumentation: `t_lanczos_`, `t_apply_heff_`,
   `t_svd_`, `t_absorb_`, `t_env_update_`, `init_timers()`,
   `report_timers()`.
3. **Dual-stream env-update overlap** (single-site / two-site only
   — pdmrg uses per-segment streams instead, also dual-handle):
   `stream_env_`, `rocblas_h_env_`, `event_canon_ready_`,
   `event_env_done_`, `env_update_pending_`, separate env scratch
   (`d_T1_env_`, `d_T2_env_`), separate env batch arrays
   (`d_batch_*_env_`). Verify that update_left/right_env actually
   USE these (technique A — count non-ctor/dtor refs).
4. **HIP-graph capture for Lanczos inner loop**:
   `apply_heff_graph_cache_` (or per-stream variant in pdmrg),
   `hipStreamBeginCapture` / `hipStreamEndCapture` /
   `hipGraphInstantiate` / `hipGraphLaunch`. Gated by
   `opts_.lanczos_graph`.
5. **Randomized SVD (RSVD)** workspace:
   `d_rsvd_omega_`, `d_rsvd_Y_`, `d_rsvd_tau_`, `d_rsvd_B_`,
   `d_rsvd_U_small_`, `RSVD_OVERSAMPLE_=10`. Code path active when
   `opts_.rsvd && full_k > k + OVERSAMPLE`. Inner SVD on device
   (rocsolver_gesvd_auto on the small B matrix).
6. **Sparse-MPO compaction**: per-site nnz lists
   (`d_W{L}_nnz_rows_/cols_` and counts). Two-site / pdmrg also
   need WW nnz lists. `apply_heff` Step 1 + Step 3 use the compacted
   batch when `opts_.sparse_mpo` is on.
7. **Batched GEMM** with cached pointer arrays
   (`d_batch_A_/B_/C_`). pdmrg also has pinned host pointer arrays.
8. **D_PAD MFMA-friendly padding** (`D_mpo_actual_` vs `D_mpo_`,
   `opts_.d_pad`). Boundary R-env identity slot at index
   `D_mpo_actual_ - 1` (NOT `D_mpo_ - 1`) — verify in
   `build_initial_environments`.
9. **On-device fused-MPO precompute** (dmrg2 / pdmrg):
   `precompute_fused_mpo` runs on the GPU. -base's host-build path
   is gone in -gpu.
10. **`accurate_svd_gpu` at segment-merge boundary** (pdmrg only,
    J1 lock).
11. **`initialize_mps_product`, `initialize_mps_neel`** — extra
    initializers added at -gpu tier.

## Hot-path functions for technique B (cross-algorithm structural diff)

Three-way diff:

- `apply_heff` (single-site) / `apply_heff_two_site` (dmrg2, pdmrg)
  — Step 1, Step 2, Step 3 sequence; sparse_s1 / sparse_s3 branches.
- `update_left_env` / `update_right_env` — batch pattern, stream
  membership.
- `sweep_left_to_right` / `sweep_right_to_left` (or
  `optimize_segment` / `merge_and_optimize_boundaries` for pdmrg)
  — count `hipEventRecord` and `hipStreamWaitEvent` sites in each.
  All three should have the same overlap pattern (modulo pdmrg's
  per-segment stream mechanism).

## Critical pitfalls to look for

- **A feature listed above present in two variants but missing in a
  third** = HIGH (this is the dmrg2-gpu dual-stream miss replayed).
- **Concurrency primitives declared but unused** (technique A
  flag) = HIGH.
- **Sparse-MPO sparse_s3 path missing** in one variant when the
  others have it = HIGH.
- **D_PAD R-env identity slot off-by-one** (writes to
  `D_mpo_-1` instead of `D_mpo_actual_-1`) = CRITICAL.
- **rocBLAS pointer-mode toggle without RAII / paired restore** =
  HIGH (round-6 rlbfgs-gpu finding — applies here too).
- **Bare `hipStreamSynchronize` in hot loop without `opts_.profile`
  guard** = HIGH (timing impact on the GPU run).

## Dispatch

Spawn an Agent (subagent_type: general-purpose) with this brief.
Have it read all three -gpu header/impl pairs plus the common
headers, run techniques A-E in full, and emit the standard Markdown
report from `.claude/review-methodology.md`. Report length budget:
≤ 1500 words (largest tier surface).
