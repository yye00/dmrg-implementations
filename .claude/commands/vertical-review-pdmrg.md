---
description: Vertical review of segment-parallel PDMRG family — compares pdmrg-gpu-base, pdmrg-gpu, pdmrg-gpu-opt for tier conformity, J1 Stoudenmire lock, and J2 superset relationship.
---

Run a vertical review of the segment-parallel PDMRG family. Vertical
= same algorithm across three tiers; the audit checks that each tier
correctly implements PDMRG AND that the tier invariants hold (J1
Stoudenmire is mandatory in EVERY tier including -base; J2: -base ⊂
-gpu ⊂ -gpu-opt).

## Scope

Three variants:

- `gpu-rocm/pdmrg-gpu-base/src/pdmrg_gpu_base.h` + `pdmrg_gpu_base_impl.h`
- `gpu-rocm/pdmrg-gpu/src/pdmrg_gpu.h` + `pdmrg_gpu_impl.h`
- `gpu-rocm/pdmrg-gpu-opt/src/pdmrg_gpu_opt.h` + `pdmrg_gpu_opt_impl.h`

Plus: `gpu-rocm/common/scalar_traits.h`, `gpu-rocm/common/gpu_opts.h`,
**`gpu-rocm/common/accurate_svd_gpu.h`** (J1 — required by all three).

## Methodology

Read `.claude/review-methodology.md` and follow techniques A through G
in full. Skipping a technique invalidates the review.

**Technique F (workspace-aliasing) is mandatory.** pdmrg-gpu-opt has
the largest set of shared scratch buffers (per-StreamWorkspace
multiples) — trace each buffer's regions and confirm sizing.

**Technique G (sibling fix-propagation) is mandatory and CRITICAL
for pdmrg.** The round-8 C-new1 finding was exactly this: round-7
fixed the canonical-Vh swap (C6) in -opt but the same defect-class
existed in -base and was missed. Read `reviews/conformity-*.md`,
identify each defect-class fix, and verify it propagated across all
three pdmrg tiers.

## Hot-path functions for technique B (behavioral diff)

For each pair (base↔gpu, gpu↔opt, base↔opt):

- `optimize_segment` / `batched_segment_sweep` (-opt only)
- `merge_and_optimize_boundaries` — Stoudenmire call site lives here
- `optimize_bond_two_site`
- `update_left_env_segment` / `update_right_env_segment`
- `svd_split_two_site`
- `apply_heff_two_site` (with WW)
- `precompute_fused_mpo`
- `lanczos_eigensolver`, `block_davidson_eigensolver` (-opt),
  `chebyshev_eigensolver` (-opt only)

## Tier-specific expected features (technique E)

**-base**:
- Per-segment streams (`streams_[seg]`) with per-segment rocBLAS
  handles. **Per-segment streams are NOT an optimization — they are
  the algorithm.** Required in every tier.
- `accurate_svd_gpu<Scalar>(...)` called at segment-merge boundary
  (J1 LOCK). The header docstring must NOT claim
  "uses plain rocsolver_gesvd_auto" — that was the round-6
  defect; verify it stays fixed.
- Per-stream `AsvdScratch` workspace.
- Fused WW precompute (host build then H2D acceptable in -base).
- HIP_CHECK + ScalarTraits.
- No GpuOpts, PhaseTimer, graph capture, RSVD, sparse-MPO, D_PAD,
  worker-stream pool, batched cross-segment, Chebyshev.
- Lanczos eigensolver only.

**-gpu**:
- All of -base PLUS:
- `accurate_svd_gpu` STAYS (J1).
- HIP-graph capture for Lanczos inner loop, keyed per-stream + per-
  shape (`apply_heff_graph_cache_`).
- GpuOpts + PhaseTimer.
- RSVD workspace + path with on-device inner SVD.
- Sparse-MPO with WW nnz lists, applied in single-site sparse_s1 and
  two-site sparse_s3 paths.
- Batched GEMM with cached pointer arrays AND pinned host pointer
  arrays.
- `precompute_fused_mpo` runs ON DEVICE.
- D_PAD with `D_mpo_actual_`.
- `set_cpu_svd`, `set_rsvd` setters.
- `n_recal` argument to `run()`.

**-gpu-opt** (must be a strict superset of -gpu):
- All of -gpu PLUS:
- `accurate_svd_gpu` STAYS (J1 — non-negotiable across all tiers).
- `pad_mfma16` + `chi_max_user_`.
- Block-Davidson default + Lanczos fallback + per-stream Davidson
  workspace.
- Worker-stream pool (`worker_streams_[seg][w]`,
  `worker_handles_`, `worker_done_events_`, `step_done_events_`).
- Cross-segment batched lock-step sweep (`batched_segment_sweep`,
  `batched_lanczos_eigensolver`, `batched_apply_heff_two_site`,
  `d_xs_batch_*` arrays, `use_batched_sweep_` flag + setter).
- Chebyshev-filtered subspace eigensolver (`chebyshev_eigensolver`,
  `use_chebyshev_` flag + setter).
- `set_use_davidson` setter (with auto-disable of `lanczos_graph`
  for graph-capture correctness).

## Critical pitfalls to look for

- ANY missing `accurate_svd_gpu` call in any tier = CRITICAL
  (J1 violation). Grep all three impls for `accurate_svd_gpu`;
  must appear at least once in each `merge_and_optimize_boundaries`
  call site.
- Header docstring claims contradicting J1 (e.g. "uses plain
  rocsolver_gesvd_auto" in -base) = HIGH.
- Per-segment streams missing or running on stream index 0 = HIGH
  (kills the algorithm's parallelism).
- Worker-stream pool dead infrastructure (declared but unused) =
  HIGH — apply technique A rigorously.
- `use_chebyshev_` or `use_batched_sweep_` declared but never read
  in the dispatch = HIGH.
- WW precompute on host instead of device in -gpu / -opt = HIGH.

## Dispatch

Spawn an Agent (subagent_type: general-purpose) with this brief.
Have it read all three headers + impls AND
`gpu-rocm/common/accurate_svd_gpu.h`, run techniques A-E in full,
and emit the standard Markdown report from
`.claude/review-methodology.md`. Report length budget: ≤ 1400 words
(pdmrg has more features, slightly larger budget).
