---
description: Horizontal review of -gpu-opt tier — compares dmrg-gpu-opt, dmrg2-gpu-opt, pdmrg-gpu-opt for opt-tier conformity, J2 superset relationship, and feature-set parity.
---

Run a horizontal review of the -gpu-opt tier. Horizontal = different
algorithms, same tier. The audit checks that all three -opt variants
have the SAME -opt feature set on top of -gpu, AND that each is a
strict superset of its -gpu sibling (J2 contract).

## Scope

Three variants:

- `gpu-rocm/dmrg-gpu-opt/src/dmrg_gpu_opt.h` + `dmrg_gpu_opt_impl.h`
- `gpu-rocm/dmrg2-gpu-opt/src/dmrg2_gpu_opt.h` + `dmrg2_gpu_opt_impl.h`
- `gpu-rocm/pdmrg-gpu-opt/src/pdmrg_gpu_opt.h` + `pdmrg_gpu_opt_impl.h`

Plus the corresponding -gpu siblings (for J2 superset check) and
`gpu-rocm/common/`.

## Methodology

Read `.claude/review-methodology.md` and follow techniques A through E
in full. Skipping a technique invalidates the review.

## J2 contract

For each (-gpu, -opt) sibling pair, -opt must contain every
non-cosmetic feature of -gpu. The audit must explicitly verify this
— it's the contract that bit us in round 6 (dmrg-gpu-opt was missing
the dual-stream env-update overlap from dmrg-gpu).

## Required -opt feature set (technique E expectations)

All three -opt variants MUST have:

1. **All -gpu features** (run technique B against the -gpu sibling
   first; any -gpu feature absent from -opt is a J2 CRITICAL).
2. **`pad_mfma16` helper** declared at file scope (static inline,
   `(x + 15) & ~15`).
3. **`chi_max_user_` member** alongside `chi_max_` (post-pad value)
   so the pre-pad bond dim is preserved.
4. **Block-Davidson eigensolver as default**:
   - `use_davidson_ = true` initialized at construction.
   - `set_use_davidson(bool)` public setter.
   - Lanczos eigensolver retained as fallback (`use_davidson_` =
     false branch).
   - `block_davidson_eigensolver` private method, with
     `d_dav_V_/AV_/work_/work2_` GPU workspace, `davidson_b_`,
     `davidson_max_sub_` config, host-side `h_dav_H_proj_`,
     `h_dav_eigvals_`, `h_dav_eigvecs_`.
   - Tiny-system fallback to Lanczos when `dim ≤ 2*b`.
5. **Strided/batched Step-3 GEMMs** (`d_batch3_A_/B_/C_` arrays)
   used inside `apply_heff` / `apply_heff_two_site`.
6. **Public-API setters**:
   - All -opt: `set_cpu_svd`, `set_use_davidson`, `set_rsvd`.
   - pdmrg-gpu-opt only: `set_use_batched_sweep`,
     `set_use_chebyshev` (reflecting the pdmrg-specific extras).
7. **pdmrg-gpu-opt only**:
   - Worker-stream pool: `n_workers_`, `worker_streams_[seg][w]`,
     `worker_handles_`, `worker_done_events_`,
     `step_done_events_`. Apply technique A — confirm these are
     ACTUALLY USED (not dead infrastructure).
   - Cross-segment batched lock-step sweep:
     `batched_segment_sweep`, `batched_lanczos_eigensolver`,
     `batched_apply_heff_two_site`, `d_xs_batch_*`,
     `use_batched_sweep_` flag + setter.
   - Chebyshev-filtered subspace eigensolver:
     `chebyshev_eigensolver`, `use_chebyshev_` flag + setter.
     Verify `use_chebyshev_` is actually consulted in the
     eigensolver dispatch.

## Hot-path functions for technique B (cross-algorithm structural diff)

Three-way diff:

- `optimize_site` / `optimize_bond` / `optimize_segment` — same
  profile-gated timing pattern? Same eigensolver dispatch (Davidson
  default, Lanczos fallback)?
- `sweep_left_to_right` / `sweep_right_to_left` — same dual-stream
  overlap pattern as -gpu sibling (J2)? Same `env_update_pending_`
  drain at start of loop, post-loop drain at end?
- `apply_heff` — same batched-Step-3 pattern across siblings?
- `block_davidson_eigensolver` — same restart logic, same
  Rayleigh-Ritz pattern? Tiny-system fallback present in all three?

## Critical pitfalls to look for

- **Any -gpu feature missing from the matching -opt** = CRITICAL
  (J2 violation).
- **`use_chebyshev_` declared but never read** in pdmrg-gpu-opt
  eigensolver dispatch = HIGH.
- **`use_batched_sweep_` declared but never read** = HIGH.
- **Block-Davidson workspace leaked or never freed** = HIGH.
- **MFMA padding inconsistent**: `chi_max_` used in alloc but
  `chi_max_user_` used in algorithm bond_dims initialization, or
  vice versa — verify the pad direction is consistent =
  CRITICAL if mismatched.
- **`pad_mfma16` not idempotent at zero-mod-16**: `(x + 15) & ~15`
  for x already a multiple of 16 should return x; verify.
- **Round-6 round-trip regressions**: dual-stream env-update
  overlap was added in round 6; any -opt now missing it again =
  CRITICAL.

## Dispatch

Spawn an Agent (subagent_type: general-purpose) with this brief.
Have it read all three -opt header/impl pairs AND all three -gpu
sibling header/impl pairs (for J2 check), run techniques A-E in
full, and emit the standard Markdown report from
`.claude/review-methodology.md`. Report length budget: ≤ 1500 words.
