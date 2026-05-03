# Horizontal review — `-gpu-opt` tier — 2026-04-28 (round 9, HEAD `85492c9`)

Scope: `gpu-rocm/{dmrg,dmrg2,pdmrg}-gpu-opt/src/*` plus -gpu siblings (J2 superset) plus `gpu-rocm/common/*`.

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | 0 newly DEAD; pre-existing `opts_.device_k` (gpu_opts.h:17) loaded but never read in dmrg-gpu-opt/dmrg2-gpu-opt impls — already filed M-round8. |
| B. Behavioral diff (-opt × -gpu, three-way -opt) | DONE | 0 J2 violations; round-6 dual-stream env-update overlap intact in dmrg-gpu-opt (impl:1856-1864) and dmrg2-gpu-opt (impl:1750-1779). |
| C. Docstring verification | DONE | -opt class-level claims (Block-Davidson default, RSVD, MFMA-16, dual-stream, sparse) all map to code; pdmrg-gpu-opt header line 36 "gemm_strided_batched fast path" verified (impl:734, 835, 929). |
| D. clangd filter | N-A | clangd not invokable on host (no ROCm headers); fell back to grep-based dead-symbol pass (technique A). |
| E. Absence-naming brief | FOLLOWED | -opt feature checklist run; no MISSING entries. |
| F. Workspace-aliasing audit | DONE | 5 shared scratch buffers traced per variant; CR-D1 fix verified in dmrg/dmrg2-opt; **1 new HIGH found in pdmrg-gpu-opt** (see below). |
| G. Sibling fix-propagation | DONE | 6 round-7/8 fix classes re-traced (M4-ext, C-new1, H1, H2, M5, hip_check); all propagated. |

## CRITICALS — block GPU run / paper submission

None.

## HIGHS — fix before next major event

### H-new1 (pdmrg-gpu-opt) — `ws.d_dav_work` undersized vs Rayleigh-Ritz H_proj region (technique F)

[`gpu-rocm/pdmrg-gpu-opt/src/pdmrg_gpu_opt_impl.h:251` (alloc), `:1525` (use)]

Per-stream allocation:
```
hipMalloc(&ws.d_dav_work, theta_size_max_ * davidson_b_ * sizeof(Scalar))
```
i.e. `b·dim_max` Scalars (`b = 4`). The block-Davidson body writes the projected Hamiltonian `H_proj = Vᴴ·AV` of shape `(k,k)` into the same buffer (impl:1525):
```
gemm(..., k, k, dim, ..., V, dim, AV, dim, &zero, ws.d_dav_work, k);
```
where `k ≤ max_sub = min(b·8, theta_size_max_) = min(32, theta_size_max_)`.

Required size = `max(b·dim_max, max_sub²)`. Allocated size = `b·dim_max` only. The H_proj region overruns whenever `max_sub² > b·dim_max`, i.e. `theta_size_max_ < max_sub²/b = 1024/4 = 256`.

The tiny-system Lanczos fallback at impl:1455 gates on `dim ≤ 2·b = 8`, so Davidson runs on systems with `dim` as small as 9. For `9 ≤ dim ≤ 31`, `theta_size_max_` (the per-segment `D_mpo·d²·χ²` upper bound) can plausibly be `< 256` only on toy fixtures, but the H_proj write of `k·k` Scalars where `k = max_sub = min(32, theta_size_max_)` then writes up to `k·k` Scalars into `b·dim_max` storage. When `theta_size_max_ < 256`, this overruns by `max_sub² − b·theta_size_max_` Scalars.

**Same defect class as round-8 CR-D1** in dmrg/dmrg2-gpu-opt — fixed there with:
```
dav_work_sz = max(theta_size_max_ * b + max_sub * b, max_sub * max_sub)
```
(dmrg-gpu-opt impl:305-308, dmrg2-gpu-opt impl:265-268).

In pdmrg-gpu-opt the inner loop is the host-syev path (H_proj copied D2H, eigvecs uploaded to `ws.d_dav_work2` at impl:1588, overlap matrix routed to `ws.d_dav_work2` at impl:1665 — different aliasing from dmrg/dmrg2-opt), so the formula differs slightly:
- `ws.d_dav_work` regions: H_proj `(k,k)` (impl:1525), residuals W `(b,dim)` (impl:1623, 1660), restart scratch `(dim,b)` (impl:1720). All sequential except H_proj overwritten by W in same iteration. **Required: `max(b·dim_max, max_sub²)`**.
- `ws.d_dav_work2` regions: eigvecs `(k,k)` (impl:1588) and overlap `(k,n_new)` (impl:1665). Sequential; required `max(max_sub², max_sub·b) = max_sub²`.

`ws.d_dav_work2` is similarly allocated to `b·dim_max` (impl:252), so it has the same potential undersize.

**Fix**: bump both allocations:
```
size_t dav_work_sz = std::max(
    (size_t)theta_size_max_ * davidson_b_,
    (size_t)davidson_max_sub_ * davidson_max_sub_);
hipMalloc(&ws.d_dav_work,  dav_work_sz * sizeof(Scalar));
hipMalloc(&ws.d_dav_work2, dav_work_sz * sizeof(Scalar));
```

**Severity**: HIGH (not CRITICAL). The benchmark/challenge sweeps run with `χ ≥ 16`, giving `theta_size_max_ ≫ 256` — H_proj fits comfortably. Heap corruption only triggers on test-fixture sizes (`L=4` Heisenberg type configurations) with `theta_size_max_ < 256`. Exactly the same "passes smoke tests, fails on real workloads" inverse: here it's "passes real workloads, may fail on smoke tests."

This is the technique F finding the brief specifically calls out: pdmrg-gpu-opt block_davidson is NOT yet ported to rocsolver_syevd, but its current host-syev aliasing still has a sizing inconsistency.

## MEDIUMS — fix when convenient

### M-r9-1 (carryover) — `opts_.device_k` declared, never read

[`gpu-rocm/common/gpu_opts.h:17, 32`]

Loaded in `from_env()` but no `-opt` impl reads it. Already in round-8 medium list. No new state — preserved here for traceability.

### M-r9-2 (carryover) — pdmrg-gpu-opt block_davidson host-syev path

[`pdmrg-gpu-opt/src/pdmrg_gpu_opt_impl.h:1559, 1570`]

Per-iteration host LAPACK syev — same defect class as round-7 C2/H6 (now fixed in dmrg-gpu-opt and dmrg2-gpu-opt via rocsolver_syevd). Round-7 deferred this for pdmrg-gpu-opt; H-new1 above arose precisely because the H6 sizing fix wasn't necessary here yet. Recommend porting to rocsolver_syevd post-G1 *and* doing the technique-F sizing audit again at the same time.

## NITS — cosmetic

- `pad_mfma16` still verbatim in three -opt headers (consolidation deferred).
- Stale comment "no longer allocated" patterns in pdmrg-gpu impl carried into -opt comments.

## FALSE POSITIVES VERIFIED

- **CR-D1 (dmrg/dmrg2-gpu-opt)**: Verified fixed at dmrg-gpu-opt impl:305-308 and dmrg2-gpu-opt impl:265-268. Aliasing logic re-traced: `d_dav_work_` hosts residuals W `[0, n_new·dim)` AND overlap `[n_new·dim, n_new·dim + k·n_new)` concurrently (impl:1692-1698 / 1603-1609); `dav_work_sz = max(b·dim + max_sub·b, max_sub²)` covers both regions plus the H_proj initial write to `d_dav_work2_`. OK.
- **C2/H6 syev port** in dmrg/dmrg2-gpu-opt: eigvecs survive in `d_dav_work2_` (rocsolver_syevd jobz=`evect_original` writes in place), restart path at impl:1747-1750 / 1657-1660 reuses them directly. No D2H+H2D roundtrip in inner loop. OK.
- **C3** (no unguarded `hipStreamSynchronize` in dmrg-gpu-opt block_davidson): all syncs after Davidson convergence/restart only; no per-iteration sync between gemm and syevd. OK.
- **C4 / C5 / C6** in pdmrg-gpu-opt: `use_davidson_=true` default (impl:206), `n_recal` arg in `run()` (impl:3417), `d_Vh_canonical` swap before R_env build (impl:3394-3404). All present.
- **H1** (pdmrg-gpu-opt): `streams_` and `worker_streams_` both use `hipStreamCreateWithFlags(NonBlocking)` (impl:125, 138). OK.
- **H2** (pdmrg-gpu-opt parallel_sweep): per-worker `std::exception_ptr` capture + post-join rethrow (impl:3449-3464). OK.
- **H8** (dmrg-gpu-opt no `d_batch3_*_`): zero hits in header and impl. Dead infrastructure removed. OK.
- **M4-ext** (set_mpo double-call guard): all three -opt free `d_mpo_tensors_[i]` before re-allocating (dmrg-opt impl:511, dmrg2-opt impl:461, pdmrg-opt impl:567). OK.
- **M5** (set_use_davidson symmetric): dmrg2-gpu-opt header:73, pdmrg-gpu-opt header:61 — both setters present. OK.
- **M7** (dmrg-gpu-opt allocate_mps_tensor pre-allocates at chi_max): verified in ctor body, preserved.
- **M8/M9** (device-pointer Lanczos scaffolding in dmrg-gpu-opt): present but unused — round-7 H-misleading-comment finding (round-8 high) is the standing item; defer.
- **M1** (HIP_CHECK consolidation): all three -opt impls `#include "../../common/hip_check.h"`. OK.
- **J2 superset** for round-6 dual-stream env-update overlap: dmrg-gpu-opt impl:1838, 1856-1864 mirrors dmrg-gpu; dmrg2-gpu-opt impl:1750-1779 mirrors dmrg2-gpu; pdmrg-gpu-opt uses per-segment streams instead (architecture-different — the dual-stream pattern doesn't apply, replaced by `streams_[seg]` overlap). Intentional.
- **pad_mfma16 idempotency**: `(0+15)&~15 = 0`, `(16+15)&~15 = 16`. Zero-mod-16 idempotent. OK.

## SUMMARY

One new HIGH (H-new1, pdmrg-gpu-opt `ws.d_dav_work` / `ws.d_dav_work2` undersized vs `max_sub²` for `theta_size_max_ < 256`). All round-7/8 critical fixes (CR-D1 in dmrg/dmrg2-opt; C2/H6 syev port; C4/C5/C6 in pdmrg-opt; H1/H2/H8; M4-ext set_mpo guard; M1/M5/M7) verified intact. No J2 violations; the dual-stream env-update overlap is correctly absent in pdmrg-gpu-opt (per-segment-stream design supersedes it) and present in dmrg/dmrg2-gpu-opt. The H-new1 finding is exactly the technique-F class the methodology was extended for in round 8: a sizing inconsistency that only materializes at small `theta_size_max_` (test fixtures), invisible at challenge-sized runs. Recommend a one-line ctor fix before any unit-test sweeps that include small `L`/`χ` configs; **does not block G1** (challenge sizes are immune by `theta_size_max_ ≫ 256`).
