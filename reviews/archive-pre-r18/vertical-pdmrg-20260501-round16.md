# Vertical review — pdmrg family — round 16 — 2026-05-01

## Charter

Audit pdmrg-gpu-base, pdmrg-gpu, pdmrg-gpu-opt, pdmrg-multi-gpu for
J1 (Stoudenmire), J2 (-opt = Block-Davidson default + ctor gate),
PDMRG-rules-2026-04-15 lock, and the round-15 fix list (5355c06,
abd88b9, 187fddf, 69da5b4).

## Charter proof — techniques applied

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | new d_dav_eigvals/E/info live; t_absorb_ STILL DEAD in -gpu/-opt; t_env_update_ STILL DEAD in multi-gpu |
| B. Behavioral diff | DONE | apply_heff_single_site Step 1 in -opt diverges from -gpu sibling — see CRITICAL |
| C. Docstring verification | DONE | 187fddf commit msg contradicts code — see CRITICAL |
| D. clangd filter | SKIPPED | no ROCm headers in sandbox |
| E. Absence-naming brief | FOLLOWED | -opt PhaseTimer panel coverage gap on default Davidson path |
| F. Workspace-aliasing audit | DONE | new d_dav_* fields single-role/sized correctly; d_dav_work2 envelope intact at :274 |
| G. Sibling fix-propagation | DONE | 4 round-15 fixes traced; 3 lonely-sibling gaps |

## CRITICALS — block GPU run / paper submission

- **pdmrg-gpu-opt: apply_heff_single_site Step 1 STILL builds stack
  `Scalar* h_A[256], h_B[256], h_C[256]` + 3× hipMemcpyAsync H2D**
  `[pdmrg-gpu-opt: pdmrg_gpu_opt_impl.h:2332-2345 (sparse_s1),
  2357-2370 (dense)]` Commit 187fddf claimed "zero host pointer-
  array hipMemcpyAsync HostToDevice patterns remain across all
  in-charter -gpu/-opt variants on the default code path." Step 3
  fallback was patched but Step 1 of the same function (both
  branches) still builds host pointer arrays + 3× H2D every call.
  Single-site is the default warmup/polish path
  (PDMRG-rules-2026-04-15) so this fires every Lanczos/Davidson iter
  × every bond × every warmup/polish sweep. Direct violation of
  no-host-roundtrips-2026-04-27. Stale comment at :2297-2306 even
  narrates the defect and notes "pdmrg-gpu's single-site uses
  device-side setup kernels instead and is safe." Fix: port the
  `setup_lenv_ptrs` / `setup_batch_ptrs_wd_sparse` pattern from
  pdmrg_gpu_impl.h:2000-2002 / 1984-1988 — kernels already in shared
  `common/batch_ptrs_kernels.h`.

## HIGHS — fix before next major event

- **pdmrg-gpu-opt: block_davidson_eigensolver has NO PhaseTimer —
  default eigensolve silent in panel** `[pdmrg-gpu-opt:
  pdmrg_gpu_opt_impl.h:1493-1793]` 69da5b4 wired `t_lanczos_.begin/
  end` inside dmrg-gpu-opt::block_davidson and dmrg2-gpu-opt's (5
  early-return ends + final each). pdmrg-gpu-opt's has zero. With
  `use_davidson_=true` default, every optimize_bond:2268 dispatch
  is UNTIMED; panel only fires on Lanczos fallback at :1806/:2002.
  Lonely-sibling — same surface, fix shipped in two siblings same
  cycle.

- **pdmrg-gpu-opt: apply_heff_single_site lacks PhaseTimer**
  `[pdmrg-gpu-opt: pdmrg_gpu_opt_impl.h:2289-2442]` 69da5b4 added
  `t_apply_heff_.begin/end` to pdmrg-gpu's apply_heff_single_site
  (impl :1935 + cache-hit close :1954 + main close :2090) —
  explicitly the M-pdmrg-single-site-uninstrumented fix. Sibling
  pdmrg-gpu-opt got NO timer pair this round. Same lonely-sibling
  defect class.

- **pdmrg-multi-gpu: apply_heff_single_site lacks PhaseTimer**
  `[pdmrg-multi-gpu: pdmrg_multi_gpu_impl.h:1645-1705]` Same
  defect — round-15's fix to pdmrg-gpu was not propagated to
  multi-gpu. apply_heff_two_site at :757/:818 IS instrumented;
  the single-site twin is the outlier.

## MEDIUMS — fix when convenient

- **pdmrg-multi-gpu: blocking `hipMemcpy` for new_k D2H per SVD**
  `[pdmrg-multi-gpu: pdmrg_multi_gpu_impl.h:1398, 1763]` Both
  svd_split_two_site and svd_split_single_site call
  `HIP_CHECK(hipMemcpy(&new_k, dev.d_svd_info, sizeof(int),
  hipMemcpyDeviceToHost))` — synchronous, no stream context.
  Sibling -gpu/-opt use `hipMemcpyAsync(..., stream)` at
  pdmrg_gpu_impl.h:1646/2157 and pdmrg_gpu_opt_impl.h:1388/2506.
  Per-bond per-sweep host roundtrip; pre-existing not-yet-fixed.

- **pdmrg-gpu, pdmrg-gpu-opt: `t_absorb_` STILL DEAD** `[pdmrg-gpu:
  pdmrg_gpu_impl.h:215, 2824, 2845; pdmrg-gpu-opt: pdmrg_gpu_opt_
  impl.h:253, 3589, 3608]` Reconfirmed from round-15: declared,
  init'd, printed via skip-on-zero (silent) but no `.begin/.end`.
  Either wire at the absorb sites (svd's S·Vh-into-MPS) or drop.

- **pdmrg-multi-gpu: `t_env_update_` STILL DEAD** `[pdmrg-multi-
  gpu.h:216, 222, 240]` Reconfirmed: `update_left/right_env`
  (impl :826, :899) have zero begin/end. pdmrg-gpu wires the pair
  at :1153/:1215 and :1229/:1287 — multi-gpu omits.

- **pdmrg-gpu-opt: per-iter `hipStreamSynchronize` in Davidson CGS**
  `[pdmrg-gpu-opt: pdmrg_gpu_opt_impl.h:1713]` Reconfirmed from
  round-15. Before reading `wi_norm` via host-pointer-mode
  `Traits::nrm2`. 5355c06 closed the H_proj-level syev roundtrip
  but did not extend to CGS norm reads.

## NITS — cosmetic

- **pdmrg-gpu-opt: stale comment narrating the CRITICAL**
  `[pdmrg_gpu_opt_impl.h:2297-2306]` Names the exact host-stack
  defect and "Fix plan: port device-side ptr-setup kernels to this
  path." Remove when fixing the CRITICAL.

## FALSE POSITIVES VERIFIED

- **J1**: `accurate_svd_gpu<Scalar>` at base:1276, gpu:2512,
  opt:3360, multi:2214 — all four.

- **J2 + ctor gate**: `use_davidson_=true` at opt:205; ctor block
  at :221-227 disables `lanczos_graph` symmetrically with
  `set_use_davidson()`.

- **PDMRG-rules-2026-04-15**: defaults `n_warmup=1, n_polish=0` on
  all four `run()`; warmup/polish through `sweep_*_full_1site` only
  in all four; `--warmup`/`--polish` wired in all four CLIs.

- **5355c06**: `lapack_syev` gone from pdmrg-gpu-opt code (only
  `lapack_gesvd` at :339 init-time query + :1355/:2475 use_cpu_svd_
  opt-in remain). rocsolver_syevd at :1583, on-device d_dav_work2
  in-place. h_dav_H_proj/eigvecs/syev_work confirmed removed.

- **abd88b9 / 187fddf**: GPU kernel launches at opt:2787 (cross-seg
  Step 1) and opt:2426 (single-site Step 3 fallback); kernels in
  common/batch_ptrs_kernels.h. Note: 187fddf msg said "two_site
  Step 3" but diff was single-site — msg/code drift NIT.

- **69da5b4 PhaseTimer for pdmrg-gpu single-site**: gpu:1935
  begin, :1954 cache-hit, :2090 close — correct in -gpu.

- **F workspace aliasing**: `dav_work_sz = max(θ·b, max_sub²)` at
  opt:274-279 INTACT. New d_dav_eigvals/E/info single-role.

## SUMMARY

Round-15's four commits land cleanly at the macro level and J1/J2/
PDMRG-rules locks hold. One CRITICAL emerges from techniques B+C:
187fddf's msg asserts the host-pointer-array pattern is gone "across
all in-charter -gpu/-opt variants on the default code path," but the
patch only touched single-site Step 3 fallback — single-site Step 1
(both branches at :2332 and :2357) still builds `Scalar* h_A[256]`
on stack and fires 3× H2D every call, on the default warmup/polish
path. Three HIGHs are technique-G lonely-sibling gaps from 69da5b4:
pdmrg-gpu-opt's block_davidson and apply_heff_single_site, and
pdmrg-multi-gpu's apply_heff_single_site, all missing the PhaseTimer
pair that shipped to siblings the same commit cycle. Four MEDIUMs
(multi-gpu blocking new_k D2H; two dead PhaseTimers; Davidson CGS
sync) and one NIT round out. Recommend bundling the CRITICAL with
the three HIGHs — all in pdmrg-gpu-opt and pdmrg-multi-gpu.
