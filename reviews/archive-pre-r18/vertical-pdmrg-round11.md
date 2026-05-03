# Vertical review — pdmrg family — round 11

HEAD `1d44d89c3f90ce39f276669f368319d6cf35292f`. Gating round.

Scope: `gpu-rocm/pdmrg-gpu-{base,_,opt}/src/*` plus
`gpu-rocm/common/accurate_svd_gpu.h`.

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | 3 dead host buffers carried over (-opt) |
| B. Behavioral diff | DONE | base/gpu/opt update_left_env hits 7/7/11 (J2 superset preserved) |
| C. Docstring verification | DONE | -base "DOES use accurate_svd_gpu" matches code; no drift |
| D. clangd filter | N-A | clangd not invokable on host (no ROCm) |
| E. Absence-naming brief | FOLLOWED | tier feature checklist below |
| F. Workspace-aliasing audit | DONE | d_dav_work, d_WW, d_Vh_canonical, d_xs_batch_* checked, no OVERRUN |
| G. Sibling fix-propagation | DONE | round-10 fixes propagated; -gpu immune to M-opt-rsvd-env |

## Round-10 watch-list verification

| Watch item | Site | Status |
|---|---|---|
| M-opt-rsvd-env | `pdmrg_gpu_opt_impl.h:205` (`use_rsvd_ = opts_.rsvd;`) | FIXED, intact |
| H10-multi-WW guard (sibling check on -gpu/-opt own precompute_fused_mpo) | `pdmrg_gpu_impl.h:869`, `pdmrg_gpu_opt_impl.h:705` | guard in place pre-round-10 |
| Round-9 H-new1 d_dav_work sizing | `pdmrg_gpu_opt_impl.h:257-261` (`max(theta_size_max*b, max_sub*max_sub)`) | intact |
| Round-9 M4-W set_mpo guards (-base/-gpu/-opt) | base 326/342/346, gpu 743/773/777, opt 577/607/611 | all guarded |
| Round-8 C-new1 canonical-Vh swap (-base) | `pdmrg_gpu_base_impl.h:1321-1337` | intact |
| Round-7 C6 canonical-Vh swap (-gpu/-opt) | gpu 2549-2581, opt 3402-3421 | intact |
| J1 Stoudenmire lock (all 3 tiers) | base 1267, gpu 2491, opt 3353 | intact in `merge_and_optimize_boundaries` |

## Tier-feature checklist (technique E)

**-base**: per-segment streams (51-56) PRESENT, J1 accurate_svd_gpu PRESENT,
per-stream AsvdScratch PRESENT, fused WW (host build + H2D) PRESENT,
HIP_CHECK + ScalarTraits PRESENT, no GpuOpts/PhaseTimer/RSVD/D_PAD/sparse-MPO/
Chebyshev/worker-pool. Lanczos eigensolver only. **All match brief.**

**-gpu**: superset of -base PLUS J1 PRESENT (line 2491), HIP-graph cache
keyed per-shape (`apply_heff_graph_cache`) PRESENT, GpuOpts + PhaseTimer
PRESENT, RSVD workspace + path PRESENT, sparse-MPO nnz lists PRESENT,
batched GEMM with cached pointer arrays PRESENT, `precompute_fused_mpo`
on-device PRESENT, D_PAD with `D_mpo_actual_` PRESENT, `set_cpu_svd`/
`set_rsvd` PRESENT, `n_recal` PRESENT. **All match brief.**

**-gpu-opt**: superset of -gpu PLUS J1 PRESENT (line 3353), `pad_mfma16` +
`chi_max_user_` PRESENT, Block-Davidson default + Lanczos fallback PRESENT,
worker-stream pool PRESENT (worker_streams/handles/done_events live, 28 hits
in impl), cross-segment batched lock-step sweep PRESENT (`batched_segment_sweep`
dispatched at lines 3494/3507), Chebyshev PRESENT (dispatched at 2251),
`set_use_davidson` with lanczos_graph round-trip PRESENT,
`lanczos_graph_was_user_enabled_` toggle PRESENT. **All match brief.**

## Technique A — symbol-usage scan, pdmrg-gpu-opt

| Member | Total hits | Other hits | Verdict |
|---|---|---|---|
| `h_rsvd_B`         | 1 | 0 (resize only) | DEAD (carry-over) |
| `h_rsvd_U_small`   | 1 | 0 (resize only) | DEAD (carry-over) |
| `h_dav_V_copy`     | 1 | 0 (resize only) | DEAD (carry-over) |
| `worker_streams_`  | 28 (set) | many | live |
| `step_done_events_`| live | many | live |
| `d_xs_batch_A_/B_/C_` | 4 each | live | live |
| `use_chebyshev_`   | 2 | dispatched at 2251 | live |
| `use_batched_sweep_`| 5 | dispatched at 3494/3507 | live |
| `lanczos_graph_was_user_enabled_` | 3 | live in setter | live |
| `chebyshev_eigensolver` | 2 (decl + def) | reachable via use_chebyshev_ | live |

The three host buffers (h_rsvd_B/h_rsvd_U_small/h_dav_V_copy) remain dead —
unchanged from round-10 (MED-pdmrg-opt-{1,2,3}). Cosmetic post-G1 cleanup.

Also still in tree: `gpu-rocm/pdmrg-gpu/src/accurate_svd.h` — not included by
any source (verified with full-tree grep). Stale legacy host-LAPACK header,
superseded by `common/accurate_svd_gpu.h`. Carried over from round-10
deferred list.

## Technique F — workspace aliasing

| Buffer | Regions | Lifetime | Required | Allocated | Verdict |
|---|---|---|---|---|---|
| `d_dav_work` (-opt) | residuals W [0, b·dim), H_proj k×k, X = V·eigvecs | sequential | `max(theta_size_max·b, max_sub·max_sub)` | same expression at line 257 | OK |
| `d_dav_work2` (-opt) | overlap (k×n_new ≤ max_sub·b), restart eigvecs (max_sub·max_sub), x0 scratch | sequential | `max(theta_size_max·b, max_sub·max_sub)` (since b ≤ max_sub) | same | OK |
| `d_WW[bond]` (all 3 tiers) | per-bond fused MPO `D·dd·dd·D` | per-bond | guarded free + malloc at every `set_mpo` / `precompute_fused_mpo` call | guarded | OK |
| `d_Vh_canonical` (all 3 tiers) | full_k × n_svd canonical Vh | per-merge | `theta_size_max` Scalar | matches | OK |
| `d_xs_batch_A_/B_/C_` (-opt) | per-segment-batched GEMM pointer arrays | live across batched sweep | `n_segments · D_mpo · d · d` | line 327-context confirms sizing | OK |

No new OVERRUN class found.

## Technique G — recent fix propagation

Round-10 fix list:

| Fix | Variant | Status |
|---|---|---|
| H10-multi-WW-leak (`d_WW` guard in `precompute_fused_mpo`) | pdmrg-gpu-base | already had guard at impl 397 |
| same | pdmrg-gpu | already had guard at impl 869 |
| same | pdmrg-gpu-opt | already had guard at impl 705 |
| same | pdmrg-multi-gpu | FIXED at multi 683 (out of scope but verified per brief) |
| M-opt-rsvd-env (`use_rsvd_ = opts_.rsvd;`) | pdmrg-gpu-opt | FIXED at impl 205 |
| same | pdmrg-gpu-base | IMMUNE (no RSVD path; baseline) |
| same | pdmrg-gpu | IMMUNE — uses `opts_.rsvd` directly throughout, no `use_rsvd_` shadow member; setter at header 52 writes through opts_ |

No lonely-fix class detected.

## Technique B — paired-variant behavioral diff

Hot-path function counts (gemm/Memcpy/Event/Stream/HIP_CHECK calls):

- `update_left_env`: base=7, gpu=7, opt=11 → opt adds worker-pool dispatch
  (event record + n_workers stream-wait pairs). Intentional J2 superset.
- `merge_and_optimize_boundaries`: all three call `accurate_svd_gpu`, then
  copy first new_k rows of Vh into d_Vh_canonical, then assign to
  `d_mps_tensors_[bsite+1]`. Identical algorithmic shape across tiers — only
  differs in async/sync stream choreography (-opt uses streams_[si], -gpu
  same, -base uses single segment streams). No structural divergence.
- `apply_heff_two_site`: base uses dense WW × theta direct; gpu/opt have
  sparse_s3 fast path keyed on `opts_.sparse_mpo` plus dense fallback. J2
  superset.

No new behavioral defect.

## Technique C — docstring verification

- pdmrg-gpu-base header lines 30-33 explicitly says "The baseline DOES use
  the on-device Stoudenmire `accurate_svd_gpu`" — verified at impl 1267.
- "host-side compute then H2D" claim (line 29) for fused WW in -base —
  verified at impl 397-399 (host build, then `hipMemcpy` H2D).
- pdmrg-gpu header at line 12 includes accurate_svd_gpu.h — used at impl
  2491.
- pdmrg-gpu-opt header at line 13 includes the same — used at impl 3353.
- No remaining "uses plain rocsolver_gesvd_auto" claim that contradicts
  reality. Round-6 docstring drift stays fixed.

## CRITICALS — block GPU run / paper submission

None.

## HIGHS — fix before next major event

None net-new.

## MEDIUMS — fix when convenient

- **MED-pdmrg-opt-{1,2,3}** (carry-over): three dead host buffers in
  pdmrg-gpu-opt — `h_rsvd_B`, `h_rsvd_U_small`, `h_dav_V_copy`
  [pdmrg_gpu_opt.h:176/224/225, only resize() at impl 267/351/352]. Status
  unchanged from round-10. Cosmetic.
- **MED-pdmrg-stale-asvd-h** (carry-over): unused legacy host-LAPACK header
  `gpu-rocm/pdmrg-gpu/src/accurate_svd.h` not included anywhere. Should be
  deleted post-G1; functionally inert (no compilation impact).

## NITS — cosmetic

- **N-pdmrg-1site-foot-gun** (carry-over from round-10 N2): non-atomic
  `lanczos_use_1site_` member set on stream 0 outside parallel regions in
  -base, -gpu, -opt. Currently safe (warmup/polish are serial); promote to
  StreamWorkspace if any future refactor pulls single-site optimization
  into a per-segment path.

## FALSE POSITIVES VERIFIED

- pdmrg-gpu lacks `use_rsvd_ = opts_.rsvd;` line — looks like the same
  defect class as M-opt-rsvd-env, but pdmrg-gpu has no `use_rsvd_` member;
  the RSVD path is gated entirely on `opts_.rsvd` and the setter writes
  through. Confirmed immune.
- pdmrg-gpu-base lacks accurate_svd_gpu in two-site `svd_split` — by design
  (only the boundary merge is Stoudenmire; the interior two-site uses
  rocsolver_gesvd_auto plus clip, which is the standard pdmrg pattern).
  J1 lock applies to the boundary, where it is present.

## SUMMARY

**Verdict: clean.** Round 11 returns **zero net-new findings** for the
vertical pdmrg review. All round-7/-8/-9/-10 fixes verified intact at
their cited file:lines. J1 Stoudenmire lock holds in all three tiers.
J2 superset relationship holds at the structural level (each tier's
features fully superset the next-lower tier; no missing infrastructure
or dead-feature class beyond the three carry-over dead host buffers and
one carry-over stale `accurate_svd.h` file already noted in round-10).

The three dead host buffers (`h_rsvd_B`, `h_rsvd_U_small`, `h_dav_V_copy`)
and the stale `pdmrg-gpu/src/accurate_svd.h` did not get any traction
this round — they remain post-G1 cleanup items. They are carry-over
MEDIUMS, not net-new.

This is the second consecutive clean vertical-pdmrg review (round-10 also
returned zero net-new — the 3 mediums noted above were already
carry-overs at that point). Combined with the round-10 vertical-dmrg and
horizontal-gpu clean rounds, the diminishing-returns trend documented in
`reviews/conformity-20260428-round10.md` continues. Recommendation:
proceed with the round-11 sister sub-reviews to confirm the trend across
the other 5 commands.
