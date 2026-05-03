# Vertical review — dmrg2 family — 2026-05-01 round-17

HEAD: post-`0efe96d`. Baseline: round-16 at `f40140d` /
`reviews/vertical-dmrg2-20260501-round16.md`. Scope: `gpu-rocm/dmrg2-gpu-base/`,
`gpu-rocm/dmrg2-gpu/`, `gpu-rocm/dmrg2-gpu-opt/`. Two commits land between
baseline and HEAD that touch dmrg2: `8abb6e7` (defect-class registry +
unrelated D6 fix elsewhere) and `0efe96d` (D12 — dmrg2-gpu-opt Lanczos ported
to device-pointer mode, mirroring dmrg-gpu-opt).

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | no dead members; D12 buffers all referenced in lanczos |
| B. Behavioral diff | DONE | -opt now matches -gpu sibling pattern in Lanczos main loop |
| C. Docstring verification | DONE | header comment "Round-16 D12 device-pointer-mode Lanczos buffers" matches impl |
| D. clangd filter | N-A | no clangd locally; A subsumes |
| E. Absence-naming brief | FOLLOWED | -base/-gpu/-opt expected-feature checklists pass |
| F. Workspace-aliasing audit | DONE | 11 new D12 buffers all dedicated singletons; 0 aliasing; CR-D1 dav_work_sz still ≥ required |
| G. Sibling fix-propagation | DONE | D12 traced: -opt fixed, -gpu was original model, -base immune (mode already device) |

## Regression-watch verification (round-16 → round-17)

### 1. D12 dmrg2-gpu-opt class members re-added — INTACT

`dmrg2_gpu_opt.h:154-164` declares all 11 D12 buffers:
`d_dot_result_`, `d_nrm2_result_`, `d_neg_alpha_`, `d_neg_overlap_`,
`d_inv_nrm_`, `d_alpha_dev_`, `d_beta_dev_`, `d_neg_beta_scalars_`,
`d_const_one_/zero_/neg_one_`. Allocations at impl:176-186; H2D const
inits at impl:189-191; frees at impl:330-340. **OK.**

### 2. Lanczos main loop wraps PointerModeGuard, device-pointer BLAS-1 — INTACT

`dmrg2_gpu_opt_impl.h:1052-1116` (per-iter): the entire α-compute /
fuse-or-axpy / reorthogonalization / β-compute / scal_real region is
under a single `PointerModeGuard pm_guard(rocblas_h_,
rocblas_pointer_mode_device)` (impl:1053). All BLAS-1 calls
(`Traits::dot`, `Traits::axpy`, `Traits::nrm2`, `Traits::scal_real`,
`Traits::gemv` reorth) read scalars from `d_dot_result_`,
`d_neg_alpha_`, `d_neg_overlap_`, `d_neg_beta_scalars_`, `d_inv_nrm_`,
`d_const_*`. Zero per-iter `&host_var` arguments inside the guard.
**OK.**

The initial v[0] = θ/||θ|| setup at impl:1019-1042 also runs in
device-pointer mode (three guarded scopes). One blocking
`hipStreamSynchronize` at impl:1026 only fires in the rare
`norm < 1e-14` re-init path.

### 3. Convergence check + final tridiagonal solve read α/β from device — INTACT

Convergence check (impl:1119-1143): reads `d_alpha_dev_`/`d_beta_dev_`
via `hipMemcpyAsync ...DeviceToDevice...` into `d_steqr_D_`/`E_`,
runs `rocsolver_dsteqr` on device, only the energy scalar comes back
to host. Final solve (impl:1153-1158): same D2D pattern, then
`rocsolver_dsteqr` with `evect_tridiagonal` on device. **OK.**

## Round-13/-12/-8 carry-forward (re-verified)

- **CR-D1 d_dav_work_sz** (-opt impl:265-268): unchanged. Block-Davidson
  inner-loop concurrent regions (residuals at offset 0, overlap matrix
  at `n_new·dim`) still bounded by allocated `dav_work_sz`. **OK.**
- **Round-6 dual-stream env-pipeline + direction-L MPS-write reorder**:
  -opt impl:1376→1387→1396 (Vh→event→absorb), -gpu impl:1312→1320→1326
  identical pattern. **INTACT.**
- **D_PAD precompute_fused_mpo OOB**: -opt impl:537-541 + -gpu inner
  loops bound by `D_act` writing into `D_use` stride. **INTACT.**
- **H1-base apply_heff scope** (round-14): -base impl:608 inner guard
  scoped after apply_heff_two_site. **INTACT.**
- **Round-15→16 carry-forwards** (H2-opt host-batch elimination via
  `common/batch_ptrs_kernels.h`, PhaseTimer 5-phase panel,
  shared-header inclusion): all intact and verified at the cited
  file:line locations from round-16.

## Sibling propagation cross-check (technique G)

| Fix | dmrg2-gpu-base | dmrg2-gpu | dmrg2-gpu-opt |
|---|---|---|---|
| D12 (device-pointer-mode Lanczos main loop) | immune (device-mode setup at impl:587-593, device-mode main loop already) | original model (impl:1004-1073, pre-existing) | NEWLY FIXED (impl:1052-1116) |

D12 is now consistent across all three tiers' Lanczos main loops.
0efe96d's port to -opt closes the round-16 false-positive flagged in
the prior review (host-stack `h_alpha[]`/`h_beta[]` per-iter D2H
elimination).

## CRITICALS

None.

## HIGHS

None within charter.

## MEDIUMS

None within charter.

## NITS

None.

## FALSE POSITIVES VERIFIED

- **dmrg2-gpu lanczos initial v[0] uses host `&norm`/`&inv_norm`**
  at impl:980-992 (implicit blocking D2H sync at first nrm2). Looks
  like a "no host roundtrips per sweep" violation. Verification:
  pre-existing pattern, NOT introduced by 0efe96d. dmrg-gpu (sibling
  single-site middle-tier) has the **identical** pattern at
  dmrg_gpu_impl.h:912-917. The D12 commit specifically targeted the
  -opt main-loop α/β arrays + final solve, not the
  initial-normalization preamble. Family-wide axis-5 candidate for a
  future dedicated sweep across both middle-tier variants;
  out-of-scope for round-17 regression watch. **Awareness only.**
- **dmrg2-gpu-opt RSVD per-call host `std::vector<Scalar> h_omega`
  + host random fill** at impl:1292-1297. Same pattern in
  dmrg-gpu-opt:1299, dmrg-gpu:1165, dmrg2-gpu:1206. Pre-existing
  family-wide pattern, opt-in only (`use_rsvd_`).
- **dmrg2-gpu-opt `opts_.fuse_lanczos` per-call hipMalloc/hipFree**
  at impl:993-995 (now after D12 port). Mirrors dmrg-gpu-opt
  pattern; off by default (`DMRG_GPU_OPT_FUSE_LANCZOS` env-var
  opt-in). Pre-existing.
- **D8 `lapack_gesvd` registry hits in dmrg2-gpu-opt** at impl:261
  (init-time workspace query, one-shot at construction) and
  impl:1236 (inside `if (use_cpu_svd_)` opt-in fallback). Both
  off-default-path; correctly registered as known false positives.
- **D9 `t_absorb_` declared without begin/end** at -opt impl:1878,
  1897. Init-only; `t.calls()==0` skip-row in `report_timers`
  benign. Family-wide pattern (dmrg-gpu-opt, dmrg2-gpu, dmrg2-gpu-opt,
  pdmrg-gpu, pdmrg-gpu-opt all init-only). Pre-existing.
- **D13 per-wp host loop in dmrg2-gpu-base apply_heff_two_site**
  at impl:357-375, 398-415 (Step 1 + Step 3 unbatched per-(w, s1, s2)
  rocBLAS GEMM loops). Per the -base charter (single-stream, naive
  first-pass tier), this is intentional; the optimization to batched
  GEMM lives in -gpu/-opt. Same pattern in dmrg-gpu-base,
  pdmrg-gpu-base. **Charter-conformant.**

## SUMMARY

Round-17 returns **0 critical, 0 high, 0 medium, 0 nits** for the
dmrg2 family across all three tiers. The single dmrg2-touching commit
in the round-16→17 window (`0efe96d`, D12 port to dmrg2-gpu-opt) is
correctly scoped: 11 device-side BLAS-1 / α-β / const-scalar buffers
re-added (all live, none dead per technique A), the Lanczos main loop
inner region is wrapped in PointerModeGuard with all BLAS-1 calls
reading device scalars, and the convergence check + final tridiagonal
solve consume α/β via D2D copy into `d_steqr_*`. Technique G traces
D12 cleanly across the three tiers: -opt fixed, -gpu was the original
device-pointer model (pre-existing main-loop pattern), -base genuinely
immune (its setup already uses device-mode). Workspace aliasing
(technique F) re-confirmed for the round-8 CR-D1 `dav_work_sz` site —
no regression. Round-15→16 carry-forwards (H2-opt host-batch shared
header, PhaseTimer 5-phase panel) remain intact. The
dmrg2-gpu lanczos initial-v[0] host-mode preamble is awareness-only
out-of-charter; it is family-wide across both middle-tier variants
(dmrg-gpu sibling has the same pattern), pre-existed this commit
window, and is a candidate for a future dedicated axis-5 sweep across
all middle-tier Lanczos preambles. This is the seventh consecutive
zero-finding sub-review for the dmrg2 family within charter. Block
GPU run? **NO**, family is ready.

Self-audit: all seven techniques completed (D N-A); regression-watch
list explicitly traced for the three round-17 D12 verification items
(class members re-added, main loop wraps PointerModeGuard,
convergence check + final solve read α/β from device); -base brought
into scope and verified; technique G sibling propagation traced for
D12 across all three dmrg2 tiers and confirmed clean.
Verdict: **READY**.

(Length: ~795 words.)
