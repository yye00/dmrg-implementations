# Horizontal review — -gpu tier — 2026-04-28 (post round-7 / commit 69305f0)

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | Round-7 deletions complete (no leftover refs to `use_rsvd_`, `h_batch_*_pinned`, `nnz_rows_count_`, `nnz_cols_count_`). All declared concurrency primitives live. |
| B. Behavioral diff | DONE | 3-way structural diff complete on apply_heff / sweeps / env update. Pattern: dmrg-gpu + dmrg2-gpu use dual-stream pair (10 event/wait calls each); pdmrg-gpu uses per-segment streams (0 events; per-stream join syncs at boundary merge — by design). |
| C. Docstring verification | DONE | All header claims verified against impl. J1 lock (`accurate_svd_gpu` at boundary merge) intact at `pdmrg_gpu_impl.h:2482`. D_PAD identity slot at `D_mpo_actual_-1` correct in all three. |
| D. clangd filter | N-A | ROCm headers unavailable on host; technique A subsumes the dead-symbol path. |
| E. Absence-naming brief | FOLLOWED | Feature-checklist applied per variant; results below. |

## CRITICALS — block GPU run / paper submission

None. The four CRITICALS in the prior conformity report (C1-C6) all live in
the **-opt** tier and are out of scope for this -gpu review.

## HIGHS — fix before next major event

None new in -gpu tier post round-7. All prior HIGHS in this tier resolved:

- **H3 (FIXED)** — `initialize_mps_product()` / `initialize_mps_neel()`
  declared at `pdmrg_gpu.h:36-37` and defined at `pdmrg_gpu_impl.h:697,712`.
  Now sister-parity with dmrg-gpu / dmrg2-gpu.
- **H4 (FIXED)** — pdmrg-gpu `rsvd_split` no longer has the three
  unconditional `hipStreamSynchronize` calls at the original 1746/1754/1766
  sites. Single residual sync at `pdmrg_gpu_impl.h:1815` is required for
  the truncation-rank decision (D2H of `h_svd_S` followed by host loop) and
  is the same pattern used by `svd_split` at lines 1621/2128. Sibling
  variants (dmrg-gpu / dmrg2-gpu) have the equivalent sync in their RSVD
  paths.
- **H7 (FIXED)** — Pointer-mode RAII: `common/pointer_mode_guard.h`
  (lines 12-26) defines canonical `PointerModeGuard`. Used at
  `dmrg2_gpu_impl.h:1125, 1256` (Lanczos for-iter body + finalization
  block), and `pdmrg_gpu_impl.h:1323, 1347, 1369, 1534` (4 toggle-pair
  sites). No bare `set_pointer_mode` toggles remain in either file.

## MEDIUMS — fix when convenient

- **dmrg-gpu retains its own `DmrgPointerModeGuard`** at
  `dmrg_gpu_impl.h:18-30` instead of using the canonical
  `PointerModeGuard` from `common/pointer_mode_guard.h`. Functionally
  identical, used 7 sites in `dmrg_gpu_impl.h`. Round-7 promoted the
  pattern to common/ but only migrated dmrg2-gpu and pdmrg-gpu —
  dmrg-gpu still uses the local copy. Style/duplication finding,
  follow-on to M1.
- **dmrg2-gpu does not include `common/pointer_mode_guard.h` directly**
  — it pulls it in via the relative include at `dmrg2_gpu_impl.h:5`,
  which is correct. Verified.
- **M2 (residual)** — `d_T3_` scratch in dmrg2-gpu-base / pdmrg-gpu-base
  is the prior-round finding; this review's scope is -gpu where
  `d_T3` does not appear (verified absent — `grep -n "d_T3" gpu-rocm/dmrg2-gpu/src/dmrg2_gpu.h` is empty). Out of -gpu scope.

## NITS — cosmetic

- Leftover comment at `pdmrg_gpu_impl.h:554` mentioning `h_batch_*_pinned`
  even though the members are removed. Remove the stale reference.
- Per-segment join `hipStreamSynchronize` in pdmrg-gpu at
  `pdmrg_gpu_impl.h:2599, 2671` is correctly placed but the loop bound
  `std::min(n_active, n_avail_streams)` at 2598 differs from the
  unconditional `n_segments_` loop at 2670. Both are correct in their
  contexts (2598 is the boundary-merge join with `n_active <= P/2`;
  2670 is the parallel_sweep join with all segments) but the
  inconsistency reads odd at first glance — worth a comment.

## FALSE POSITIVES VERIFIED (prevent re-discovery)

- **dmrg2-gpu `stream_env_` dead-infrastructure replay** — re-counted
  60 references in `dmrg2_gpu_impl.h` (matches dmrg-gpu's 60). All live.
  The 2026-03 finding remains fixed.
- **D_PAD R-env identity slot off-by-one** — verified at
  `D_mpo_actual_ - 1` at `dmrg_gpu_impl.h:1000`, `dmrg2_gpu_impl.h:1072`,
  `pdmrg_gpu_impl.h:1287`. All correct.
- **pdmrg-gpu `apply_heff_graph_cache_` cross-segment reuse** — cache is
  PER-`StreamWorkspace` at `pdmrg_gpu.h:197` (inside the struct, not at
  class scope); key includes `(two_site, site, cL, cR)` via `graph_key`
  at `pdmrg_gpu.h:203-208`. Per-stream + per-shape requirement met.
- **`hipStreamNonBlocking` in pdmrg-gpu** — verified used (this is the
  -opt regression noted as H1 in the conformity report; the -gpu
  variant is correct).
- **J1 Stoudenmire lock** — `accurate_svd_gpu<Scalar>(...)` invoked at
  `pdmrg_gpu_impl.h:2482` inside `merge_and_optimize_boundaries`. The
  `AsvdScratch<Scalar> asvd` member is per-stream (declared at
  `pdmrg_gpu.h:184` inside `StreamWorkspace`) and allocated /
  released at `pdmrg_gpu_impl.h:488, 590`.
- **H5 (WW host precompute)** — out of -gpu-tier scope to fix
  (timing-only, not on hot path); docstring corrected in commit
  69305f0 per round-7 batch 7.
- **`hipStreamSynchronize` in pdmrg-gpu hot loop** — 20 sites total;
  audit shows every one is either (a) a host-scalar dependency
  (norm read for near-zero check, Lanczos β/eigenvalue convergence
  every 3 iters, truncation rank decision); or (b) a parallel-segment
  join (post-thread.join at lines 2599, 2671). None are gratuitous;
  none should be guarded by `opts_.profile`.

## Feature-set parity table (technique E)

| Feature | dmrg-gpu | dmrg2-gpu | pdmrg-gpu |
|---|---|---|---|
| `GpuOpts opts_` | ✓ (h:194) | ✓ (h:178) | ✓ (h:211) |
| `PhaseTimer` t_lanczos/apply_heff/svd/absorb/env_update | ✓ | ✓ | ✓ |
| Dual-stream env-update overlap | ✓ (60 refs) | ✓ (60 refs) | N/A — per-segment streams instead |
| Per-segment streams + per-stream rocblas handles | N/A | N/A | ✓ (`streams_[]`, `handles_[]`) |
| `apply_heff_graph_cache_` (Lanczos HIP graph) | ✓ class-scope | ✓ class-scope | ✓ per-StreamWorkspace |
| RSVD workspace + on-device inner SVD | ✓ | ✓ | ✓ |
| Sparse-MPO compaction (WL nnz) | ✓ wl_nnz_*_count_ | ✓ wl_nnz_*_count_ | ✓ wl_nnz_*_count_ |
| Sparse-MPO WW nnz (two-site) | N/A (single-site only) | ✓ ww_nnz_*_count_ | ✓ ww_nnz_*_count_ |
| Batched GEMM + GPU-kernel pointer setup | ✓ | ✓ | ✓ |
| D_PAD MFMA padding + identity slot at `D_mpo_actual_-1` | ✓ | ✓ | ✓ |
| On-device fused-MPO (WW) precompute | N/A | ✓ | ✓ |
| `accurate_svd_gpu` at segment boundary (J1) | N/A | N/A | ✓ (h:184; impl:2482) |
| `initialize_mps_random/product/neel` | ✓ | ✓ | ✓ (round-7 fix) |
| `PointerModeGuard` from common/ | local copy | ✓ | ✓ |
| `HIP_CHECK` from common/hip_check.h | ✓ (impl:11) | ✓ (impl:13) | ✓ (impl:21) |

All feature-checklist items present in all three variants where the
algorithm permits. No silent omissions. The two N/As for dmrg-gpu are
algorithmic (single-site has no two-site WW; pdmrg-only J1 lock).

## SUMMARY

Round-7 fixes for the -gpu tier are **complete and verified**. All four
named regressions (H3, H4, H7, M3 + M12 + M13 + M1) hold; no new HIGH or
CRITICAL findings introduced. Feature-set parity across the three -gpu
variants is intact, with the expected algorithmic asymmetries (dmrg-gpu
single-site only; pdmrg-gpu segment-stream pattern instead of dual-stream
pair) properly accounted for.

**Verdict for -gpu tier:** ready for the MI300X G1 window. The CRITICALS
that block G1 (C1-C6) live in the -opt tier and are tracked separately.

The one outstanding -gpu MEDIUM is dmrg-gpu's local
`DmrgPointerModeGuard` duplicating the now-canonical
`common/pointer_mode_guard.h::PointerModeGuard`. One-line fix, defer to
the post-G1 cleanup cycle.
