---
description: Horizontal review of -base tier — compares dmrg-gpu-base, dmrg2-gpu-base, pdmrg-gpu-base for engineering-style conformity at the first-pass tier.
---

Run a horizontal review of the -base tier. Horizontal = different
algorithms (single-site / two-site / segment-parallel), same tier.
The audit checks engineering-style conformity, shared infrastructure
usage, and absence of -gpu-tier features (which would mean the -base
charter has been violated by feature creep).

## Scope

Three variants:

- `gpu-rocm/dmrg-gpu-base/src/dmrg_gpu_base.h` + `dmrg_gpu_base_impl.h`
- `gpu-rocm/dmrg2-gpu-base/src/dmrg2_gpu_base.h` + `dmrg2_gpu_base_impl.h`
- `gpu-rocm/pdmrg-gpu-base/src/pdmrg_gpu_base.h` + `pdmrg_gpu_base_impl.h`

Plus: `gpu-rocm/common/scalar_traits.h`,
`gpu-rocm/common/accurate_svd_gpu.h` (pdmrg only — J1).

## Methodology

Read `.claude/review-methodology.md` and follow techniques A through G
in full. Skipping a technique invalidates the review.

**Technique G (sibling fix-propagation) is the highest-leverage
technique for the -base tier.** When a defect is fixed in a -opt or
-gpu variant, the -base sibling almost never gets the same audit
unless this technique forces it. The round-8 C-new1 finding (canonical
Vh swap missing in pdmrg-gpu-base after C6 was fixed in -opt) was
exactly this miss.

## What "conformity" means at the -base tier

The three -base variants are first-pass GPU implementations of three
different algorithms. They should:

1. Share the SAME foundation:
   - HIP_CHECK macro definition (ideally one source in `common/`,
     not three copies — flag duplication as MEDIUM if found).
   - ScalarTraits dispatch for double + hipDoubleComplex.
   - Identical error-handling pattern around HIP / rocBLAS calls.
2. Use the SAME minimalist style:
   - Single stream, single rocBLAS handle (except pdmrg: per-segment
     streams which are algorithmic).
   - On-device SVD via `Traits::rocsolver_gesvd_auto`.
   - Lanczos eigensolver only (no Block-Davidson, no Chebyshev).
   - No GpuOpts, no PhaseTimer, no env-var-driven flags.
   - No HIP-graph capture, no RSVD, no sparse-MPO compaction, no
     D_PAD, no batched-Step-3.
3. NOT have -gpu-tier features. Each of the listed -gpu features
   appearing in any -base variant is a HIGH (charter violation).

## Algorithm-specific exceptions (technique E expectations)

**dmrg-gpu-base**: single-site theta of shape (cL, d, cR).
**dmrg2-gpu-base**: two-site theta of shape (cL, d, d, cR), fused WW
MPO precompute (host-build then H2D acceptable at -base).
**pdmrg-gpu-base**: per-segment streams (`streams_[seg]`) AND
`accurate_svd_gpu` call at segment-merge boundary (J1 — Stoudenmire
is part of pdmrg's algorithm, not an optimization).

## Hot-path functions for technique B (cross-algorithm structural diff)

Three-way diff on the engineering-style scaffolding:

- Constructor: stream/handle creation pattern, allocation order,
  error-checking style.
- `free_gpu_resources`: mirror of constructor; check frees match
  allocs in reverse order; check no leaks.
- HIP_CHECK / ROCBLAS_CHECK macro definitions: are they identical
  across the three? Or three slightly-different copies?
- `set_mpo`: H2D upload pattern, validation of input shapes.
- `update_left_env` / `update_right_env`: batched-vs-loop pattern
  consistency; same rocBLAS API surface; same scratch usage.
- Outer driver `run(...)`: same warmup/sweep/polish structure;
  CLAUDE.md compliance (single-site warmup/polish, n_warmup ≤ 2,
  n_polish ≤ 2, defaults configurable).

## Critical pitfalls to look for

- pdmrg-gpu-base WITHOUT `accurate_svd_gpu` = CRITICAL (J1).
- HIP_CHECK or ROCBLAS_CHECK redefined locally in three places when
  one in `common/` would do = MEDIUM (consolidation opportunity).
- A -base variant that has GpuOpts / lanczos_graph / RSVD /
  sparse_mpo / D_PAD / Block-Davidson = HIGH (charter violation —
  -base is the simplest correct port, advanced features belong at
  -gpu or -opt).
- Inconsistent error-handling (one variant throws, another aborts,
  another logs-and-continues) = MEDIUM.
- CLAUDE.md PDMRG rule violation: warmup or polish using two-site
  sweeps in pdmrg-gpu-base = CRITICAL.

## Dispatch

Spawn an Agent (subagent_type: general-purpose) with this brief.
Have it read all three -base header/impl pairs plus
`gpu-rocm/common/scalar_traits.h` and
`gpu-rocm/common/accurate_svd_gpu.h`, run techniques A-E in full, and
emit the standard Markdown report from
`.claude/review-methodology.md`. Report length budget: ≤ 1200 words.
