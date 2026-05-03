# Vertical review — DMRG family (single-site) — round-17 — 2026-05-01

HEAD: `0efe96d`. Round-16 baseline: `f40140d`. Watch list per charter:
`8abb6e7` (defect-class registry + D6 fix), `0efe96d` (D12 port to
dmrg-gpu-opt + dmrg2-gpu-opt). All three tiers in scope. Registry
sweep run first (`.claude/scripts/defect-registry.sh`).

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | -base 23 / -gpu ~44 / -opt ~57 (+11 from D12 port). All re-added members ctor :175-191 + dtor :348-358 + ≥3 use sites in lanczos_eigensolver. 0 dead. |
| B. Behavioral diff | DONE | Lanczos: -gpu and -opt now structurally aligned (PointerModeGuard + process_alpha/beta + d_alpha_dev_/d_beta_dev_ on-device + dsteqr). H1-opt absorb-timer gap unchanged from round-16. |
| C. Docstring verify | DONE | -base header line 13 still overstates "device-pointer mode" — apply_heff/env GEMMs use host stack `&one/&zero`. Pre-existing NIT carried from round-16. |
| D. clangd filter | N-A | No ROCm headers on host; A subsumes. |
| E. Absence-naming | FOLLOWED | -opt: 4/5 PhaseTimer phases instrumented (`t_absorb_` STILL MISSING). All other -opt features present incl. D12 device-pointer Lanczos. |
| F. Workspace-aliasing | DONE | 11 new D12 buffers all single-role / disjoint lifetimes (d_dot_result_, d_nrm2_result_, d_neg_alpha_, d_neg_overlap_, d_inv_nrm_ are 1-elt scratch; d_alpha_dev_/d_beta_dev_/d_neg_beta_scalars_ are max_lanczos_iter_-sized; d_const_*_ are immutable constants). Round-8 CR-D1 `d_dav_work_` sizing :282-285 intact. 0 OVERRUN. |
| G. Sibling fix-propagation | DONE | D12 device-pointer-mode Lanczos: -base intact, -gpu intact, -opt NEW landed. D6 (duplicate batch_ptrs kernels): pdmrg-only, dmrg family unaffected (dmrg-gpu/-opt use shared `common/batch_ptrs_kernels.h` since round-15). |

## Regression-watch verification

| Watch item | File:line | Status |
|---|---|---|
| D12 dmrg-gpu-opt class members re-added | dmrg_gpu_opt.h:176-186 (11 members) | OK |
| D12 ctor allocations | dmrg_gpu_opt_impl.h:175-191 | OK |
| D12 dtor frees | dmrg_gpu_opt_impl.h:348-358 | OK |
| D12 PointerModeGuard wraps main loop | :1011, :1023, :1027, :1046 | OK |
| D12 process_alpha kernel call | :1051 (writes d_alpha_dev_[iter]) | OK |
| D12 process_beta kernel call | :1100 (writes d_beta_dev_[iter] + d_neg_beta_scalars_[iter]) | OK |
| D12 negate_scalar kernel calls | :1064, :1092 | OK |
| D12 invert_nrm kernel calls | :1028, :1195 | OK |
| D12 lanczos_check_beta + dsteqr convergence | :1127, :1137 | OK |
| D12 final tridiagonal solve reads device α/β | :1161-1163 (hipMemcpyAsync D→D from d_alpha_dev_/d_beta_dev_) | OK |
| D12 dmrg-gpu round-7 baseline intact | dmrg_gpu_impl.h:948-952, 1001-1002 | OK |
| D12 dmrg-gpu-base correct | dmrg_gpu_base_impl.h:519, 540, 545, 578, 595, 597, 629 | OK |
| Shared batch_ptrs kernel header | common/batch_ptrs_kernels.h | OK |
| dmrg-gpu uses shared header | dmrg_gpu_impl.h:13 | OK |
| dmrg-gpu-opt uses shared header | dmrg_gpu_opt_impl.h:12 | OK |
| Round-8 CR-D1 `d_dav_work_` sizing | dmrg_gpu_opt_impl.h:282-285 | OK |
| H2-opt apply_heff Step 1 / env updates | dmrg_gpu_opt_impl.h:634/650/768/860 | OK |
| Cache-hit closes `t_apply_heff_` | dmrg_gpu_opt_impl.h:599 before return :600 | OK |
| Round-14 H1-base Lanczos guard scope | dmrg_gpu_base_impl.h:540 inside loop | OK |
| set_quiet stubs/comments | dmrg_gpu.h:43 unchanged | OK |
| D11 ctor-time Davidson/lanczos_graph gate | dmrg_gpu_opt_impl.h:74-80 | OK |

## CRITICALS

None.

## HIGHS

- **H1-opt-absorb-timer-uninstrumented** [dmrg-gpu-opt: dmrg_gpu_opt_impl.h:1214, :1383, :1433, :1461, :1980, :1999]. Carried from round-16 H1. The 69da5b4 instrumentation pass added `t_lanczos_/t_apply_heff_/t_svd_/t_env_update_` + Davidson begin/end pairs but never split out `t_absorb_`. In the current SVD path, `t_svd_.begin(stream_)` :1214 covers gesvd, the post-SVD `event_canon_ready_` record, AND the absorb GEMM (`d_svd_work_=S*Vh` at :1387-1393, gemm at :1399-1404), closing only at :1461. Sibling **dmrg-gpu** correctly splits at impl :1267-1295 (R-direction) and :1314-1341 (L-direction) — `t_svd_.end` then `t_absorb_.begin` then `t_absorb_.end`. `init_timers` :1980 and `report_timers` :1999 still wire `t_absorb_`, so `DMRG_GPU_PROFILE=1` users see "absorb 0 calls 0 ms" — false-quiet identical to the round-15 H1 pattern. This is the phase whose duration matters most for measuring dual-stream overlap (absorb on `stream_` vs env_update on `stream_env_`). HIGH; bundle with round-16 H1 follow-up.

## MEDIUMS

- **M1-opt-lanczos-init-D2H-sync** [dmrg-gpu-opt: dmrg_gpu_opt_impl.h:1015-1017]. The new D12 port adds a `hipMemcpyAsync(&norm, d_nrm2_result_, ..., D2H) + hipStreamSynchronize` to host-sync the initial norm for the `norm < 1e-14` zero-vector guard. Once-per-Lanczos-call (not per-iter), so it does not violate the "no host roundtrip per sweep" rule on Davidson-default paths. Sibling **dmrg-gpu** :914 sidesteps the sync entirely by leaving rocblas in host-pointer mode for the init nrm2 (`Traits::nrm2(..., &norm)` direct host-pointer). Cleaner pattern; the -opt port could match it and drop the explicit hipStreamSynchronize. MEDIUM.

## NITS

- **dmrg-gpu-base docstring overgeneralization** [dmrg_gpu_base.h:13-18]. Pre-existing. "All linear algebra ... in device-pointer mode" — apply_heff GEMMs use host-stack `&one/&zero`. True for Lanczos BLAS-1 ops only.
- **dmrg-gpu-opt PhaseTimer fields `t_absorb_` declared but never measured** [dmrg_gpu_opt.h:274]. Captured under H1; tagged here for visibility in future Technique-A sweeps.

## FALSE POSITIVES VERIFIED

- **Registry D8 (15 hits in dmrg/dmrg2/pdmrg families) — all gated**. dmrg-gpu-opt :276 = init-time `lwork_query=-1` workspace-size query (no compute); :1250 = `use_cpu_svd_` opt-in branch (D2H + hipStreamSynchronize :1246, then host gesvd, then H2D). Both legitimate. Same shape in dmrg2-gpu-opt.
- **Registry D9 `t_canon_ready_` "no .begin/.end"** — false positive: substring match against the comment field on `event_canon_ready_` declarations (h:87, h:99). The corresponding event uses `hipEventRecord/WaitEvent`, not PhaseTimer.
- **Registry D13 (3 hits, dmrg/dmrg2/pdmrg `-base`)** — intentional naive single-GEMM loops per `-base` charter (docstring h:9-31). Host loops launch GEMMs; no host arithmetic. Charter-allowed.
- **Registry D2 (1 hit pdmrg-gpu-opt)** — comment-only reference, not a compute site (out-of-scope for dmrg family anyway).

## SUMMARY

D12 port to dmrg-gpu-opt landed cleanly: 11 device buffers re-added with correct ctor/dtor symmetry, PointerModeGuard wraps the per-iter loop, `lanczos_process_alpha/beta_kernel` write α/β into device arrays consumed directly by `rocsolver_dsteqr`. Sibling parity with dmrg-gpu round-7 H10 baseline restored. Workspace-aliasing audit clean (all new buffers are single-role or disjoint-lifetime). Defect registry sweep on HEAD shows zero new hits affecting the dmrg family — D6 was pdmrg-only, D7/D11/D12 zero hits, D8 hits all gated. **The one HIGH carries from round-16 unchanged**: dmrg-gpu-opt `t_absorb_` is declared/init/reported but never has `.begin/.end` calls, leaving `DMRG_GPU_PROFILE=1` users with a "0 ms" line for the phase that most needs measurement under dual-stream overlap. One MEDIUM flags an init-time D2H sync in the new D12 port that the -gpu sibling avoids by holding rocblas in host-pointer mode for the one-shot init nrm2. No CRITICALs.
