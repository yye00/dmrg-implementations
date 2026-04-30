# Horizontal review — -gpu tier — round 12 (2026-04-30, post-cleanup)

HEAD: `c3d3e50` (round-11 cleanup). Diff vs round-11 baseline
(`2d55d90`): single in-scope code commit `c3d3e50` removing dead host
buffers and the legacy `pdmrg-gpu/src/accurate_svd.h`.

## Charter

Review the **-gpu tier** across dmrg-gpu, dmrg2-gpu, pdmrg-gpu, plus
the multi-GPU sibling pdmrg-multi-gpu. Verify the round-11 cleanup
left no dead buffers and that the 5-item regression-watch list is
intact in every variant.

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A | DONE | dmrg-gpu private members audited (≥3 hits); cleanup-target `h_svd_*` gone in dmrg-gpu + dmrg2-gpu; `d_svd_work_` correctly preserved in dmrg-gpu (4 use-sites). 3 dead artifacts found in pdmrg-multi-gpu. |
| B | DONE | dmrg-gpu / dmrg2-gpu identical 6-event dual-stream pattern. pdmrg-gpu wraps every pointer-mode toggle in `PointerModeGuard`; pdmrg-multi-gpu uses raw toggles (8 raw calls in `lanczos_eigensolver`). |
| C | DONE | pdmrg-multi-gpu docstring "Extends pdmrg-gpu" overstates parity (5 GpuOpts toggles + 5 timers absent). |
| D | N-A — no ROCm headers on host; A subsumes. |
| E | FOLLOWED | -gpu feature checklist applied four-way; multi-gpu MISSING entries enumerated. |
| F | DONE | `d_svd_work_` (dmrg-gpu): hosts S·Vh `new_k * n_svd` OR U·S `m * new_k`, both ≤ `chi_max² · d`; allocated `theta_size_max_ = chi_max² · d²` ≥ required → OK. `c3d3e50` introduced no aliasing — only removed unread buffers. |
| G | DONE | 5 watch-list items verified four-way. 3 propagation gaps in pdmrg-multi-gpu: H7, H3, M3. |

A-G: all DONE or N-A. Review valid.

## Watch-list verification

| Fix | dmrg-gpu | dmrg2-gpu | pdmrg-gpu | pdmrg-multi-gpu |
|---|---|---|---|---|
| 1a. h_svd_*/d_svd_work cleanup | gone | gone (h_svd + d_svd_work) | n/a (live in StreamWorkspace) | n/a |
| 1b. d_svd_work_ survives in dmrg-gpu | live + F-OK (impl.h:313, 1387/1402/1438/1449) | — | — | — |
| 2. stream_env NonBlocking | impl.h:200,201 | impl.h:198,201 | per-seg :288 | per-dev :288 |
| 3. PointerModeGuard RAII | local 1061,1207 | common 1124,1255 | common 1332,1356,1378,1543 | **MISSING** — raw at :1149/1151/1163/1167/1182/1233/1293/1307 |
| 4. init_mps_product/neel | h:35,36 | h:35,36 | h:36,37 | **MISSING** — only init_mps_random |
| 5. set_mpo W-buf guards | :590,594,629,635 | :591,595,675,681 | :743,773,777,808,814,869,899,905 | :629,633,637,684 |

## CRITICALS

None.

## HIGHS

### H12-multi-pointer-mode — pdmrg-multi-gpu lanczos uses raw rocblas_set_pointer_mode without RAII

**File**: `pdmrg_multi_gpu_impl.h:1149,1151,1163,1167,1182,1233,1293,1307`.
**Defect class**: round-7 H7.

`lanczos_eigensolver` toggles pointer mode 4 times via bare
`rocblas_set_pointer_mode(...,device)` / `...,host)` pairs. Between
each pair are `apply_heff_two_site`, `apply_heff_single_site`,
`Traits::nrm2/dot/axpy/gemv`, and `ROCBLAS_CHECK` macros — any can
`return` early before the matching restore. pdmrg-gpu fixed this in
round-7 (`PointerModeGuard` at impl.h:1332,1356,1378,1543); the common
RAII guard at `gpu-rocm/common/pointer_mode_guard.h` exists exactly
for this purpose. pdmrg-multi-gpu was not touched by round-7
propagation — round-8 C-new1 lonely-fix pattern.

**Fix**: replace each toggle pair with a `PointerModeGuard` RAII
block. ~6 blocks per function.

## MEDIUMS

### M12-multi-init-api — pdmrg-multi-gpu missing init_mps_product / init_mps_neel

`pdmrg_multi_gpu.h:36` only exposes `initialize_mps_random`. Defect
class round-7 H3 (init API parity); dmrg-gpu/dmrg2-gpu/pdmrg-gpu all
have both at `:35,36` (header `:36,37` for pdmrg-gpu). Propagation to
multi-gpu missed.

### M12-multi-h_batch — dead h_batch_*_pinned in pdmrg-multi-gpu

`pdmrg_multi_gpu.h:113-115` declares 3 `Scalar**` members; impl.h:339-341
only sets them to nullptr. Defect class round-7 M3 (dead pinned host
pointer arrays).

### M12-multi-h_rsvd — dead h_rsvd_B / h_rsvd_U_small in pdmrg-multi-gpu

`pdmrg_multi_gpu.h:158-159` (comment claims "kept for fallback paths"
but no path reads them); `pdmrg_multi_gpu_impl.h:430-431` resize only.
Same class as round-10 MED-pdmrg-opt-{1,2}.

### M12-multi-boundary-staging — d_boundary_staging used by dead function

`copy_boundary_mps_to_device` (declared at .h:239, defined at
impl.h:1117-1131) has zero callers. Its only effect is a
`hipMemcpyPeer` into `d_boundary_staging` (alloc :461, free :540). Buffer
is written but never read. Either wire into
`merge_and_optimize_boundaries` or delete.

### M12-multi-feature-gap — pdmrg-multi-gpu silently ignores 5 of 7 GpuOpts toggles

`pdmrg_multi_gpu_impl.h:159` calls `opts_.load_from_env()` but only
`opts_.rsvd` is consulted on any path. `opts_.d_pad`, `sparse_mpo`,
`lanczos_graph`, `fuse_lanczos`, `device_k`, `profile` have zero
references. The boundary R-env identity slot at impl.h:955 writes
`h_R[D_mpo_ - 1]` — correct only because D_PAD is silently disabled,
not because of a `D_mpo_actual_` correction. Setting
`DMRG_GPU_OPT_D_PAD=1` is a silent no-op. Also no PhaseTimer
instrumentation (`t_lanczos_/t_apply_heff_/...`) — paper benchmarks
on multi-gpu cannot use the standard phase-timing report. Header
docstring "Extends pdmrg-gpu" overstates parity.

## NITS

(N11-gpu-h_svd-dead from round-11 was fixed by `c3d3e50` — closed.)

## FALSE POSITIVES VERIFIED

- pdmrg-gpu / pdmrg-multi-gpu retain `h_svd_*` on the **CPU-SVD
  fallback path** — live when `use_cpu_svd_=true`, off-by-default but
  reachable via `set_cpu_svd(true)`. Verified live at
  pdmrg_gpu_impl.h:1583-1597, 2096-2110; pdmrg_multi_gpu_impl.h:
  1333-1346, 1700-1714. Distinct from the round-11 NIT in single-host
  variants where buffers were never read on any path.
- dmrg-gpu still uses local `DmrgPointerModeGuard` (impl.h:18-30)
  instead of common `PointerModeGuard`. Round-10 deliberate carry-over
  for ABI continuity. Not a new finding.
- `c3d3e50` removed legacy `pdmrg-gpu/src/accurate_svd.h` (host LAPACK
  Stoudenmire). Confirmed superseded by `common/accurate_svd_gpu.h`
  (live at pdmrg_gpu.h:10) per 23f9d46. No stale includes remain.

## SUMMARY

Round 12 surfaces 1 HIGH + 5 MEDIUMs, all in pdmrg-multi-gpu. The
single-host variants (dmrg-gpu, dmrg2-gpu, pdmrg-gpu) are clean —
`c3d3e50` cleanup landed correctly, the surviving `d_svd_work_` in
dmrg-gpu is live and F-audit-OK, and all 5 watch-list items propagated
to all three. No regressions.

pdmrg-multi-gpu findings are all one shape: the round-7 H3 + H7 + M3
fixes and round-9 H1-ext / D_PAD / sparse / graph infrastructure never
propagated. The "Extends pdmrg-gpu" docstring claim is false on 5
GpuOpts toggles, 5 PhaseTimers, and the boundary-staging dead path —
canonical technique-G lonely-fix mode.

**Recommendation**: dispatch a pdmrg-multi-gpu propagation batch.
H12-multi-pointer-mode is the only HIGH; rest are MEDIUMs that can be
bundled. The single-host -gpu tier is paper-ready.
