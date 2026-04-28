---
description: Vertical review of two-site DMRG family — compares dmrg2-gpu-base, dmrg2-gpu, dmrg2-gpu-opt for tier conformity and J2 superset relationship.
---

Run a vertical review of the two-site DMRG family. Vertical = same
algorithm across three tiers; the audit checks that each tier
correctly implements two-site DMRG AND that the tier invariants
hold (-base ⊂ -gpu ⊂ -gpu-opt per J2 contract).

## Scope

Three variants:

- `gpu-rocm/dmrg2-gpu-base/src/dmrg2_gpu_base.h` + `dmrg2_gpu_base_impl.h`
- `gpu-rocm/dmrg2-gpu/src/dmrg2_gpu.h` + `dmrg2_gpu_impl.h`
- `gpu-rocm/dmrg2-gpu-opt/src/dmrg2_gpu_opt.h` + `dmrg2_gpu_opt_impl.h`

Plus: `gpu-rocm/common/scalar_traits.h`, `gpu-rocm/common/gpu_opts.h` if
referenced.

## Methodology

Read `.claude/review-methodology.md` and follow techniques A through E
in full. Skipping a technique invalidates the review.

## Hot-path functions for technique B (behavioral diff)

For each pair (base↔gpu, gpu↔opt, base↔opt):

- `optimize_bond(site, direction)`
- `sweep_left_to_right` / `sweep_right_to_left`
- `update_left_env` / `update_right_env`
- `svd_split` (or `svd_split_fallback` in -opt)
- `apply_heff` (two-site, with fused MPO WW)
- `form_theta_two_site`
- `precompute_fused_mpo` (or `precompute_WW` in -base)
- `lanczos_eigensolver`, `block_davidson_eigensolver` (-opt only)
- `build_initial_environments`

## Tier-specific expected features (technique E)

**-base**:
- Single-stream, single rocBLAS handle.
- Two-site theta of shape (cL, d, d, cR) reshaped to (cL·d, d·cR).
- Fused MPO WW precomputed at set_mpo time (host-build then H2D).
- HIP_CHECK + ScalarTraits.
- No GpuOpts, PhaseTimer, graph capture, RSVD, sparse-MPO, D_PAD.
- Lanczos eigensolver only.
- svd_split: writes both MPS[site] and MPS[site+1] from full SVD.

**-gpu**:
- All of -base PLUS:
- Dual-stream env-update pipeline (`stream_env_`, `rocblas_h_env_`,
  `event_canon_ready_`, `event_env_done_`, `env_update_pending_`).
  In dmrg2-gpu pre-round-6 these were declared but unused — the
  round-6 commit wired them. Verify the wiring is intact.
- HIP-graph capture for Lanczos inner loop.
- GpuOpts + PhaseTimer.
- RSVD path with the full_k > k+OVERSAMPLE branch (RSVD is profitable
  in two-site because full_k = d·min(cL,cR) > chi_max + oversample,
  unlike single-site).
- Sparse-MPO with WW nnz lists (`d_WW_nnz_rows_/cols_`).
- Batched GEMM, D_PAD.
- `precompute_fused_mpo` runs ON DEVICE (vs -base's host build).
- svd_split direction L: MPS[site+1] = Vh write should be queued
  FIRST so event_canon_ready_ can be recorded before the U·S
  absorb (round-6 reorder — verify).

**-gpu-opt** (must be a strict superset of -gpu):
- All of -gpu PLUS:
- `pad_mfma16` helper, `chi_max_user_`.
- Block-Davidson default + Lanczos fallback.
- Strided/batched Step-3 GEMMs.
- Public setters: `set_cpu_svd`, `set_use_davidson`, `set_rsvd`.
- Dual-stream env-update pipeline AND the direction-L MPS write
  reorder (round-6 J2 port).

## Critical pitfalls to look for

- WW precompute on host instead of device in -gpu / -opt = HIGH.
- `precompute_WW` D_PAD out-of-bounds bug (caught in
  `lessons_gpu_ablation_20260420.md`) — verify the fix is intact.
- svd_split direction-L write order regression (U·S first, Vh
  second) defeats the dual-stream overlap = HIGH.
- Fused MPO WW dimension mismatch with sparse-MPO nnz lists =
  CRITICAL.
- See also the dmrg-vertical pitfall list — most apply identically.

## Dispatch

Spawn an Agent (subagent_type: general-purpose) with this brief.
Have it read all three headers + impls, run techniques A-E in full,
and emit the standard Markdown report from
`.claude/review-methodology.md`. Report length budget: ≤ 1200 words.
