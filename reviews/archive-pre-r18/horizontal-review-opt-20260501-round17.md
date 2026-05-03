# Horizontal review — -opt tier — round 17 (2026-05-01)

Baseline: `f40140d`. Last commit: `0efe96d` (D12 device-pointer-mode Lanczos for dmrg-gpu-opt + dmrg2-gpu-opt). Charter: dmrg-gpu-opt, dmrg2-gpu-opt, pdmrg-gpu-opt.

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | registry: t_absorb_ DEAD across all 3 (carried, M-opt-absorb-timer); use_chebyshev_/use_batched_sweep_ live |
| B. Behavioral diff | DONE | Lanczos: dmrg/dmrg2-opt now device-pointer; pdmrg-opt was already device-pointer in scalar lanczos; pdmrg-opt batched_lanczos still host-mode (not exercised by default) |
| C. Docstring verification | DONE | "α/β live on device" claim verified for all 3 single-vector lanczos paths |
| D. clangd filter | N-A | host has no ROCm headers; subsumed by A |
| E. Absence-naming brief | FOLLOWED | superset-of-gpu, MFMA padding, batched Step-3, public setter API, Chebyshev (pdmrg only) |
| F. Workspace-aliasing audit | DONE | **CRITICAL FINDING in pdmrg-gpu-opt block_davidson_eigensolver: d_dav_work2 region collision** |
| G. Sibling fix-propagation | DONE | **CR-D1 round-8 fix propagated to dmrg/dmrg2-opt but NOT pdmrg-gpu-opt; D12 round-17 fix correctly skipped pdmrg (already device-mode)** |

## Round-16 axis-5 lonely-sibling check (D12)

**pdmrg-gpu-opt's `lanczos_eigensolver` is already fully device-pointer-mode.**

Verified: `pdmrg_gpu_opt_impl.h:1786-2000` uses `PointerModeGuard pm_guard(handles_[si], rocblas_pointer_mode_device)` (lines 1802, 1818, 1837, 1973), `lanczos_process_alpha/beta_kernel` (lines 1841, 1886), `lanczos_check_beta` + `rocsolver_dsteqr` for convergence (lines 1915-1932), and α/β live in `ws.d_alpha_dev` / `ws.d_beta_dev`. The grep markers `&alpha_result`, `&beta_i`, host-`std::vector<double> h_alpha`, etc. are absent from `lanczos_eigensolver`. **The defect-class registry correctly identified this — D12 was right to skip pdmrg-gpu-opt's main Lanczos.**

Caveat (medium): `pdmrg_gpu_opt_impl.h:2838` `batched_lanczos_eigensolver` (used only when `use_batched_sweep_=true`, default OFF) STILL uses host-resident `h_alpha`/`h_beta`/`&h_dot`/`&beta_val`. Per the "no host roundtrips per sweep" 2026-04-27 lesson, this is a defect on the runtime-toggleable path. Not a regression — pre-existed D12 — but is now the only host-mode Lanczos remaining in -opt.

## CRITICALS — block GPU run / paper submission

- **CR-D1-pdmrg: Davidson `d_dav_work2` overlap matrix overwrites eigenvectors.** [`pdmrg-gpu-opt/src/pdmrg_gpu_opt_impl.h:1671-1736`]
  - Round-7 syev port (line 1564) leaves k×k eigenvectors in `ws.d_dav_work2`.
  - Residual loop (1635) places residuals in `ws.d_dav_work[0:n_new*dim)` ✓
  - Orthogonalization gemm (1675-1677) writes `overlap = V^H W` of shape (k, n_new) **into `ws.d_dav_work2`**, overwriting the eigenvectors.
  - Restart path (1722-1736) reads `ws.d_dav_work2` expecting eigenvectors → **garbage X_keep**, undefined behavior.
  - **Sibling propagation MISS (Technique G):** dmrg-gpu-opt:1657-1672 and dmrg2-gpu-opt:1578-1592 have the round-8 CR-D1 fix (`Scalar* overlap = d_dav_work_ + n_new*dim;`) which routes overlap PAST residuals so eigenvectors are preserved. Comment on dmrg-gpu-opt:1657-1660 explicitly says "so it does not overwrite d_dav_work2_". pdmrg-gpu-opt's identical defect-class fix never propagated.
  - Buffer is sized `max(theta_size_max·b, max_sub²)` (line 261-263) — does NOT include the `+ max_sub*b` term that dmrg/dmrg2-opt added in round 8. So even applying the offset-fix requires re-sizing `dav_work_sz` to `max(theta_size_max·b + max_sub·b, max_sub²)` to match siblings.
  - Severity: CRITICAL. Restart path is hit any time `k + n_good > max_sub`, i.e., every challenge-size run that doesn't converge in ≤8 iter — almost always.

## HIGHS — fix before next major event

- **H-opt-batched-lanczos-host-mode:** [`pdmrg-gpu-opt/src/pdmrg_gpu_opt_impl.h:2838-2972`] `batched_lanczos_eigensolver` keeps host-resident `h_alpha`, `h_beta`, `&h_dot`, `&beta_val`, `&inv_norm`. When `set_use_batched_sweep(true)` is set (public API), every Lanczos iter does a `Traits::dot/nrm2` host-mode roundtrip. Pre-existed D12 but is now the lonely sibling within pdmrg-gpu-opt. Same fix shape as D12: device-pointer mode + lanczos_process_alpha/beta_kernel.

## MEDIUMS — fix when convenient

- **M-opt-absorb-timer-dead** (carried): `t_absorb_` declared in all 3 -opt headers, never `.begin/.end`'d in any impl. Either wire it or drop it. Registry D9.
- **M-opt-pdmrg-svd-step-host:** [`pdmrg-gpu-opt/src/pdmrg_gpu_opt_impl.h:2284-2293`] single-site apply_heff documents host-stack `h_A[256]/h_B[256]/h_C[256]` and disabled lanczos_graph. Already flagged as deferred H2-opt. Plan: port `setup_heff_A/B/C_ptrs` device kernels (already in two-site) to single-site.

## NITS — cosmetic

- pdmrg-gpu-opt:1591 stale comment "the host h_dav_eigvecs upload is no longer needed" — `h_dav_eigvecs_` was removed but the comment phrasing implies it was just made redundant, not deleted.

## FALSE POSITIVES VERIFIED

- **Registry D8 hit at pdmrg-gpu-opt:325 `lapack_gesvd`:** ctor-time workspace query, not hot path — verified by reading `pdmrg_gpu_opt_impl.h:318-335`. Other D8 hits (1341, 2446) are gated behind `if (use_cpu_svd_)`. Default path: device SVD only.
- **D12 missing in pdmrg-gpu-opt main Lanczos:** verified ALREADY device-pointer-mode at lines 1802, 1818, 1837, 1973; α/β live in `ws.d_alpha_dev/d_beta_dev`; convergence via `lanczos_check_beta + rocsolver_dsteqr`. NOT a defect. Round-17 D12 fix correctly scoped.

## Regression-watch since `f40140d`

| Item | Status |
|---|---|
| Davidson J2-lock (`use_davidson_=true` ctor) all 3 | OK |
| `set_use_davidson` dual-half setter all 3 | OK |
| `use_rsvd_ = opts_.rsvd` ctor binding all 3 | OK |
| Shared `common/batch_ptrs_kernels.h` all 3 | OK |
| MFMA-16 `pad_mfma16` + `chi_max_user_` all 3 | OK |
| PhaseTimer panel (lanczos/apply_heff/svd/absorb/env_update) all 3 | OK |
| Round-15 H-opt-pdmrg-davidson-syev (`rocsolver_syevd` in pdmrg block_davidson) | OK [line 1569] |
| Round-8 CR-D1 `dav_work_sz += max_sub·b` | **MISS in pdmrg-gpu-opt** (see CR-D1-pdmrg) |

## SUMMARY

D12 device-pointer-mode Lanczos correctly applied to dmrg-gpu-opt and dmrg2-gpu-opt, and correctly skipped for pdmrg-gpu-opt's main `lanczos_eigensolver` which already had it from an earlier round — the registry's targeting was right. **However, technique G surfaced a HIGH-priority lonely-sibling defect**: the round-8 CR-D1 fix (Davidson `d_dav_work2` overlap-vs-eigenvector aliasing) was applied to dmrg-gpu-opt and dmrg2-gpu-opt but never propagated to pdmrg-gpu-opt, where the restart-path eigenvector buffer is overwritten by the orthogonalization GEMM, AND the buffer is undersized by the same `max_sub*b` term the siblings added. This is hit on every challenge-size run that triggers a Davidson restart. The pdmrg-gpu-opt `batched_lanczos_eigensolver` (default-off path) also retains host-pointer-mode BLAS-1 ops and would benefit from a D12-style port. Verdict: **NOT READY** for GPU run until CR-D1-pdmrg is fixed.
