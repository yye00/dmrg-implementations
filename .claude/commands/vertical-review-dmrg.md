---
description: Vertical review of single-site DMRG family — compares dmrg-gpu-base, dmrg-gpu, dmrg-gpu-opt for tier conformity and J2 superset relationship.
---

Run a vertical review of the single-site DMRG family. Vertical = same
algorithm across three tiers; the audit checks that each tier
correctly implements single-site DMRG AND that the tier invariants
hold (-base ⊂ -gpu ⊂ -gpu-opt per J2 contract).

## Scope

Three variants:

- `gpu-rocm/dmrg-gpu-base/src/dmrg_gpu_base.h` + `dmrg_gpu_base_impl.h`
- `gpu-rocm/dmrg-gpu/src/dmrg_gpu.h` + `dmrg_gpu_impl.h`
- `gpu-rocm/dmrg-gpu-opt/src/dmrg_gpu_opt.h` + `dmrg_gpu_opt_impl.h`

Plus: `gpu-rocm/common/scalar_traits.h`, `gpu-rocm/common/gpu_opts.h` if
referenced.

## Methodology

Read `.claude/review-methodology.md` and follow techniques A through E
in full. Skipping a technique invalidates the review.

## Hot-path functions for technique B (behavioral diff)

For each pair (base↔gpu, gpu↔opt, base↔opt):

- `optimize_site(site, direction)`
- `sweep_left_to_right` / `sweep_right_to_left`
- `update_left_env` / `update_right_env`
- `svd_fallback` (or `svd_and_update_mps` in -gpu)
- `apply_heff`
- `lanczos_eigensolver`, `block_davidson_eigensolver` (-opt only)
- `build_initial_environments`

## Tier-specific expected features (technique E)

**-base**:
- Single-stream, single rocBLAS handle.
- HIP_CHECK macro defined locally or via common/.
- ScalarTraits dispatch for double + hipDoubleComplex.
- No GpuOpts, no PhaseTimer, no env-var-driven flags.
- No graph capture, no RSVD, no sparse-MPO compaction.
- On-device SVD via rocsolver; no host LAPACK roundtrip per sweep.
- Lanczos eigensolver only; no Block-Davidson.

**-gpu**:
- All of -base PLUS:
- Dual-stream env-update pipeline (`stream_env_`, `rocblas_h_env_`,
  `event_canon_ready_`, `event_env_done_`, `env_update_pending_`).
- HIP-graph capture for Lanczos inner loop (gated by
  `opts_.lanczos_graph`).
- GpuOpts env-var surface (`opts_`, env vars
  `DMRG_GPU_*` or equivalent) and PhaseTimer instrumentation.
- RSVD workspace (`d_rsvd_omega_/Y_/B_/U_small_`,
  `RSVD_OVERSAMPLE_=10`) and code path.
- Sparse-MPO compaction (`d_W{L}_nnz_rows_/cols_`).
- Batched GEMM with cached pointer arrays (`d_batch_A/B/C_`).
- D_PAD MFMA-friendly padding (`D_mpo_actual_` vs `D_mpo_`).
- `initialize_mps_product` / `initialize_mps_neel` initializers.

**-gpu-opt** (must be a strict superset of -gpu):
- All of -gpu PLUS:
- `pad_mfma16` helper, `chi_max_user_` tracking pre-pad value.
- Block-Davidson eigensolver as default (`use_davidson_=true`,
  `set_use_davidson` setter), Lanczos fallback retained.
- Strided/batched Step-3 GEMMs (`d_batch3_A/B/C_`).
- Public-API setters: `set_cpu_svd`, `set_use_davidson`, `set_rsvd`.
- Round-6 dual-stream port: env-update overlap must also be present
  (J2: -opt is strict superset of -gpu).

## Critical pitfalls to look for

- Any -gpu feature missing from -gpu-opt = J2 violation (CRITICAL).
- Dead concurrency primitives (declared, allocated, freed, never
  used) = HIGH — caught by technique A.
- Docstring claim about a feature whose code is missing = HIGH —
  caught by technique C.
- `hipStreamSynchronize` in the hot loop without `opts_.profile`
  guard = MEDIUM (timing on the GPU run).
- Bare host LAPACK / host BLAS call per sweep on default code
  path = CRITICAL (CLAUDE.md "no host roundtrips per sweep" rule).

## Dispatch

Spawn an Agent (subagent_type: general-purpose) with this brief.
Have it read all three headers + impls, run techniques A-E in full,
and emit the standard Markdown report from
`.claude/review-methodology.md`. Report length budget: ≤ 1200 words.
