# Horizontal review — -opt tier — round 18 (2026-05-01)

Baseline: `12d02c5` (HEAD). Last conformity: `reviews/conformity-20260501-round15.md`.
R17 baseline: `reviews/horizontal-review-opt-20260501-round17.md`. Charter:
`gpu-rocm/{dmrg-gpu-opt, dmrg2-gpu-opt, pdmrg-gpu-opt}`.

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | All 11 D12 device buffers live in dmrg/dmrg2-opt; `use_chebyshev_`, `use_batched_sweep_`, worker-pool members live in pdmrg-opt |
| B. Behavioral diff | DONE | Davidson tiny-fallback identical in all 3; dmrg/dmrg2-opt Lanczos now device-mode (matches pdmrg-opt main Lanczos); pdmrg `batched_lanczos_eigensolver` still host-mode (carried HIGH from R17) |
| C. Docstring verification | DONE | **Stale "stack-allocated h_A[256]/h_B[256]/h_C[256]" comment in pdmrg-opt single-site apply_heff** — code uses device kernels. New M finding |
| D. clangd filter | N-A | No ROCm headers locally; A subsumes |
| E. Absence-naming brief | FOLLOWED | pad_mfma16, chi_max_user_, J2 superset, public setter API, Chebyshev (pdmrg only) |
| F. Workspace-aliasing audit | DONE | 3 -opt `d_dav_work*` buffers verified; CR-D1-pdmrg fix from 54f2fcf reviewed in detail (sizing matches siblings) |
| G. Sibling fix-propagation | DONE | D5/D6/D12/CR-D1/PhaseTimer R17 fixes traced; R8 CR-D1 sizing not regressed in dmrg/dmrg2-opt |

Pre-step: `bash .claude/scripts/defect-registry.sh` returned 0 hits across all 14 tracked defect classes (D1–D15).

## Regression watch since R15 baseline

| Item (R17 commit / earlier round) | Status |
|---|---|
| D5 host-batch ptr propagation (`abd88b9`, `187fddf`) — dmrg2/pdmrg-opt | OK — no `h_batch_*_pinned` on hot path; sole H2D in pdmrg-opt is the random-vector init at line 1830 (legitimate cold path) |
| D6 shared `common/batch_ptrs_kernels.h` (`8abb6e7`) — all 3 -opt | OK — each impl includes the shared header; zero local `__global__ void setup_batch_ptrs_*` redefinitions; 13 kernels live only in `common/` |
| D12 Lanczos device-pointer mode (`0efe96d`) — dmrg/dmrg2-opt | OK — all 11 buffers (`d_dot_result_`, `d_nrm2_result_`, `d_neg_alpha_`, `d_neg_overlap_`, `d_inv_nrm_`, `d_alpha_dev_`, `d_beta_dev_`, `d_neg_beta_scalars_`, `d_const_one_`, `d_const_zero_`, `d_const_neg_one_`) declared in headers and referenced 4–11× per impl; main Lanczos loop wrapped in `PointerModeGuard(rocblas_h_, rocblas_pointer_mode_device)`; all 4 per-iter kernels (`lanczos_process_alpha_kernel`, `lanczos_process_beta_kernel`, `negate_scalar_kernel`, `invert_nrm_kernel`) launched |
| D12 pdmrg-opt main Lanczos already device-mode | OK — verified at `pdmrg_gpu_opt_impl.h:1820, 1836, 1855` (PointerModeGuard); ws.d_dot_result/d_neg_alpha/d_alpha_dev used; not regressed |
| CR-D1-pdmrg fix (`54f2fcf`) — overlap matrix at `ws.d_dav_work + n_new*dim`, eigvecs preserved in `ws.d_dav_work2` | OK — `pdmrg_gpu_opt_impl.h:1687-1693`; sizing `dav_work_sz = max(theta_max·b + max_sub·b, max_sub²)` matches siblings (lines 266-269) |
| CR-D1 in dmrg-gpu-opt + dmrg2-gpu-opt (R8) | OK — sizing unchanged at `dmrg_gpu_opt_impl.h:301-305` and `dmrg2_gpu_opt_impl.h:285-289`; offset overlap trick at `dmrg_gpu_opt_impl.h:1665-1666` and `dmrg2_gpu_opt_impl.h:1586-1587` |
| D9/D15 PhaseTimer panels (`12d02c5`) | OK — `t_davidson_` begin AFTER tiny-Lanczos fallback in all 3; `.end()` at every return path (5 sites in dmrg/dmrg2-opt, 5 in pdmrg-opt); `t_absorb_` instrumented in dmrg/dmrg2-opt (1384-1413, 1437-1465 / 1358-1390); `t_apply_heff_` instrumented in pdmrg-opt single-site (line 2300, 2432) and two-site (line 809, 828, 973); `t_absorb_` correctly REMOVED from pdmrg-opt header + init + report |
| pdmrg-gpu-opt parallel infra (worker streams, batched-segment-sweep, Chebyshev) | OK — `n_workers_`, `worker_streams_`, `worker_handles_`, `worker_done_events_`, `step_done_events_` all referenced ≥7× in impl; `use_chebyshev_` and `use_batched_sweep_` consulted at dispatch (1492/2269) and in print banner (3434, 3492, 3505); `chebyshev_eigensolver` reachable |
| J1 Stoudenmire lock (pdmrg-opt) | OK — `accurate_svd_gpu` invoked at `pdmrg_gpu_opt_impl.h:3351`; header includes `common/accurate_svd_gpu.h:13` |
| J2 superset of -gpu | OK — env_update_pending_/dual-stream events present in dmrg/dmrg2-opt (10 / 29-30); pdmrg-opt uses per-segment streams + 8 event/wait calls (intentional, different concurrency model — same as -gpu sibling); pad_mfma16 + chi_max_user_ in all 3; public setters parity with extras (set_use_batched_sweep, set_use_chebyshev) only in pdmrg-opt |

## Technique-F detail: pdmrg-opt block_davidson_eigensolver workspace

Brief asked for a deep audit of the post-CR-D1 fix sizing.

| Region | Offset | Max size (Scalars) | Lifetime |
|---|---|---|---|
| W (residuals) | `ws.d_dav_work + 0` | `n_new × dim` ≤ `b × theta_size_max` | live across orthogonalization GEMMs (1691, 1696) |
| overlap | `ws.d_dav_work + n_new*dim` | `k × n_new` ≤ `max_sub × b` | live concurrently with W in the same GEMM pair |
| restart X_keep | `ws.d_dav_work + 0` | `dim × keep` ≤ `theta_size_max × b` | restart path only; W+overlap dead by then |
| `ws.d_dav_work2` | offset 0 | `k × k` ≤ `max_sub × max_sub` (eigvecs) | live across both GEMM pair AND restart path |

Required `dav_work_sz` ≥ max(b·theta_max + max_sub·b, max_sub²).
Allocated (line 266-269) = `max(theta_size_max·davidson_b_ + davidson_max_sub_·davidson_b_, davidson_max_sub_²)` → **EXACT MATCH**. OK.
`davidson_max_sub_` is set to `min(davidson_b_*8, theta_size_max_)` at line 189 → `n_new ≤ b ≤ max_sub`, so the `n_new*dim` offset is bounded by `b*dim ≤ b*theta_size_max ≤ allocated`.
`ws.d_dav_work2` is sized identically and only ever holds the (k×k) eigvec matrix — never overlapped with W. OK.

The siblings dmrg-gpu-opt (1665) and dmrg2-gpu-opt (1586) carry the same offset trick with the same sizing; both have unchanged R8 fix (regression-watch row above).

## CRITICALS — block GPU run / paper submission

None.

## HIGHS — fix before next major event

- **H-opt-batched-lanczos-host-mode (carried R17)** — `pdmrg-gpu-opt/src/pdmrg_gpu_opt_impl.h:2870-2980` — `batched_lanczos_eigensolver` retains host-resident `h_alpha`, `h_beta`, `&h_dot`, `&beta_val`, `&inv_norm` and does host-mode `Traits::dot/nrm2` per iter. Triggered only by `set_use_batched_sweep(true)` (default OFF); any benchmark that exercises the public API for cross-segment batching will hit per-Lanczos-iter PCIe roundtrips. Same fix shape as D12 (PointerModeGuard + lanczos_process_*_kernel). Not regressed since R17 — explicitly retained as deferred. Per "no host roundtrips per sweep 2026-04-27" lesson, this is the only host-mode Lanczos remaining at -opt and stays HIGH until ported.

## MEDIUMS — fix when convenient

- **M-opt-pdmrg-single-site-graph-comment-stale** (NEW, technique C) — `pdmrg-gpu-opt/src/pdmrg_gpu_opt_impl.h:2302-2311` — comment block claims "Step 1/Step 3 else-branches (and the sparse path) use stack-allocated `Scalar* h_A[256], h_B[256], h_C[256]` plus `hipMemcpyAsync` H2D inside the capture window" and gives this as the reason LANCZOS_GRAPH is disabled for single-site. The actual code at lines 2340, 2356, 2416 already uses `setup_batch_ptrs_wd_sparse`, `setup_batch_ptrs_wd`, and `setup_batch_ptrs_step3` device kernels — the very fix the comment closes with ("Fix plan: port device-side ptr-setup kernels"). Either (a) update the comment because the host-stack pointers are gone (the warmup hang root cause may now be resolved and graph capture should be re-enabled and tested), or (b) document the new reason for keeping single-site uncaptured. As-is, this is a docstring-claim drift that conceals an enable-graph-caching opportunity. R17 listed this as deferred MEDIUM ("M-opt-pdmrg-svd-step-host"); this round it's a docstring/code drift that should at minimum be honest about the current code shape.

- **M-opt-pdmrg-stale-svd-port-comment** (NIT-grade) — `pdmrg-gpu-opt/src/pdmrg_gpu_opt_impl.h:1601-1602` — "the host h_dav_eigvecs upload is no longer needed" reads like a transitional note; `h_dav_eigvecs_` no longer exists in the header (was removed when rocsolver_syevd landed). Same NIT carried from R17.

## NITS — cosmetic

None new.

## FALSE POSITIVES VERIFIED

- **D5/D6/D12 grep on bare names** initially returned 0 hits because the header members carry trailing underscores (`d_dot_result_`, etc.). Re-running with the trailing underscore confirmed all 11 D12 buffers are live in both dmrg-gpu-opt and dmrg2-gpu-opt.
- **`hipMemcpyHostToDevice` at `pdmrg_gpu_opt_impl.h:1830`** flags as a host roundtrip but is the random-vector init for the ‖θ‖ < 1e-14 cold path, not an inner-loop H2D. Not a defect.
- **D8/D7 `lapack_gesvd` hits at pdmrg-opt:325 / 1341 / 2446** — registry returned zero hits, confirming the round-17 verification: ctor-time workspace query (line 325 region) and `if (use_cpu_svd_)` gates only.

## Cross-impl 3-way diff — block_davidson_eigensolver

Same Rayleigh-Ritz pattern in all 3:
- Random-init b columns w/ CGS orthogonalization;
- AV computation per column via apply_heff(_two_site)?;
- `Traits::gemm` H_proj = V^H · AV → `d_dav_work2` (size k×k);
- `rocsolver_syevd` in-place eigendecomp (R7 C2 / R15 H-opt-pdmrg-davidson-syev);
- D2H lowest eigenvalue + info for control flow only;
- residual loop: ri = AV·eigvec - λi·V·eigvec at `d_dav_work + n_new*dim`;
- orthogonalization with overlap PAST residuals (CR-D1 fix);
- restart when k+n_good > max_sub: keep best b Ritz vectors using preserved `d_dav_work2` eigvecs.

No structural divergences. pdmrg-opt has the additional `si == 0` gate on every PhaseTimer call (correct: `std::vector::push_back` not thread-safe across parallel segments).

## SUMMARY

Round 18 is clean at the -opt tier. The R17 CRITICAL (CR-D1-pdmrg, Davidson `d_dav_work2` overlap-clobber) was correctly fixed in `54f2fcf` with sizing that mirrors the R8 sibling fix, and a careful technique-F audit confirms no aliasing or sizing residue. The R17 D12 device-pointer-mode Lanczos port to dmrg/dmrg2-opt is in place and the pdmrg-opt main Lanczos is unregressed. R17 PhaseTimer panel propagation (t_davidson_, t_absorb_ split, t_apply_heff_ in pdmrg single-site, t_absorb_ removed from pdmrg-opt header) all verified. The defect-class registry returned 0 hits across all 14 tracked classes. One HIGH carries forward (H-opt-batched-lanczos-host-mode, default-OFF path) and one new MEDIUM surfaces (stale comment in pdmrg-opt single-site apply_heff describes host-stack pointers that the code no longer uses — masks an opportunity to re-enable LANCZOS_GRAPH for single-site warmup/polish). Verdict: **READY** for GPU run subject to the carried HIGH being either fixed or held off the public API for the run.

CRITICAL: 0
HIGH: 1 (carried)
MEDIUM: 1 new + 1 carried
NIT: 0 new
