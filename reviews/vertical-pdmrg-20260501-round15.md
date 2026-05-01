# Vertical review — pdmrg family — round 15 — 2026-05-01

## Charter

Audit pdmrg-gpu-base, pdmrg-gpu, pdmrg-gpu-opt, pdmrg-multi-gpu for
tier conformity, J1 (Stoudenmire `accurate_svd_gpu`), J2 (-opt =
Block-Davidson default + ctor `lanczos_graph` gate), the
PDMRG-rules-2026-04-15 lock, and the round-14 fix list (5deba6d).

## Charter proof — techniques applied

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | t_absorb_ DEAD in -gpu and -opt; t_env_update_ DEAD in -multi-gpu; all other members live |
| B. Behavioral diff | DONE | apply_heff/lanczos/svd_split/update_*_env compared across 4 variants — see HIGHs |
| C. Docstring verification | DONE | -base J1 docstring intact; multi-gpu n_recal-doc gap closed (round-14) |
| D. clangd filter | SKIPPED | no ROCm headers in sandbox |
| E. Absence-naming brief | FOLLOWED | tier checklist applied per variant |
| F. Workspace-aliasing audit | DONE | d_dav_work envelope `max(b·θ_max, max_sub²)` intact at :274; d_T1/T2 D·dd·χ² intact |
| G. Sibling fix-propagation | DONE | 6 round-14 fixes traced INTACT; 1 long-deferred technique-G gap reconfirmed |

## CRITICALS — block GPU run / paper submission

(none)

## HIGHS — fix before next major event

- **pdmrg-gpu-opt: per-Davidson-iteration host LAPACK syev**
  `[pdmrg-gpu-opt: pdmrg_gpu_opt_impl.h:1572-1633, 1759]` Round-7
  C2/H6 (a9d24c3) ported `block_davidson_eigensolver` from host
  `lapack_syev` to on-device `rocsolver_syevd` in dmrg-gpu-opt
  (:1566) and dmrg2-gpu-opt (:1525) — but pdmrg-gpu-opt was deferred
  "post-G1" at round-8 and never re-addressed. The default
  `use_davidson_=true` path still does, per Davidson iteration:
  blocking D2H of H_proj at :1572, host `Traits::lapack_syev` at
  :1603/1614, blocking H2D of eigvecs at :1632 (and :1759 on restart).
  ~3 blocking PCIe roundtrips × ~5-10 iters × N_bonds = O(thousands)
  blocking syncs per sweep. Contradicts no-host-roundtrips-2026-04-27
  and -opt's own "Zero per-sweep host LAPACK on default" promise.
  Both sibling -opts already on-device; long-standing technique-G
  lonely-sibling.

## MEDIUMS — fix when convenient

- **pdmrg-gpu, pdmrg-gpu-opt: `t_absorb_` PhaseTimer is dead**
  `[pdmrg-gpu: pdmrg_gpu.h:215, pdmrg_gpu_impl.h:2821, 2842;
  pdmrg-gpu-opt: pdmrg_gpu_opt.h:253, pdmrg_gpu_opt_impl.h:3614,
  3633]` Declared, init'd in `init_timers()`, printed in
  `report_timers()` — but no `.begin/.end` site exists. Silent under
  round-14's skip-on-zero filter but pure dead infrastructure
  (technique A). Either wire a real measurement or drop the member.

- **pdmrg-multi-gpu: `t_env_update_` PhaseTimer is dead**
  `[pdmrg-multi-gpu: pdmrg_multi_gpu.h:216, 222]` Same defect class
  as the -gpu/-opt absorb timer: declared, init'd, printed via
  `print_if_used` at :231 (silent), but `update_left_env`/
  `update_right_env` (impl :826, :899) have no `.begin/.end`. Wire
  the pair (mirror pdmrg-gpu :1153/1215). Lonely-sibling —
  apply_heff/lanczos/svd timers ARE wired in multi-gpu (:757, :1165,
  :1347), env_update is the outlier.

- **pdmrg-gpu-opt: per-iteration `hipStreamSynchronize` in Davidson
  CGS** `[pdmrg-gpu-opt: pdmrg_gpu_opt_impl.h:1731]` Inner CGS does
  `hipStreamSynchronize` before reading `wi_norm` via host-pointer
  `Traits::nrm2`. n_new ≤ 4, but on default `use_davidson_=true`
  path. Same host-control-flow pattern sibling -opts removed via
  device-pointer mode + `rocsolver_syevd`. Folds into the HIGH —
  fixing H_proj host-LAPACK naturally restructures CGS norm reads.

## NITS — cosmetic

(none new)

## FALSE POSITIVES VERIFIED

- **J1 lock holds (technique C)**: `accurate_svd_gpu<Scalar>` at
  pdmrg_gpu_base_impl.h:1276, pdmrg_gpu_impl.h:2509,
  pdmrg_gpu_opt_impl.h:3385, pdmrg_multi_gpu_impl.h:2214 — all four
  variants. -base header docstring (pdmrg_gpu_base.h:30-33) still
  correctly identifies Stoudenmire as algorithmic, not
  optimization.

- **J2 lock + ctor gate (technique E)**: pdmrg_gpu_opt_impl.h:204
  `use_davidson_ = true`; ctor block at :220-226
  `if (opts_.lanczos_graph && use_davidson_)` disables
  `opts_.lanczos_graph` symmetrically with `set_use_davidson()`
  runtime guard.

- **PDMRG-rules-2026-04-15 lock**: defaults `n_warmup=1, n_polish=0`
  on all four `run()` signatures (base.h:62-65, gpu.h:41,
  gpu_opt.h:47-48, multi_gpu.h:58); zero supported in all loops;
  warmup/polish through `sweep_*_full_1site` only; CLI driver
  (`test_pdmrg_gpu_opt.cpp:401-402`) wires `--warmup`/`--polish`
  explicitly.

- **Round-14 H1-base apply_heff scope**: PointerModeGuard scope at
  pdmrg_gpu_base_impl.h:744-749 (init) and :766-809 (per-iter
  device-mode); apply_heff_* called in HOST mode. Pattern matches
  pdmrg-gpu.

- **Round-14 H-opt-phase-timer-prop**: pdmrg-gpu-opt 5 phases wired
  with si==0 gating — `t_apply_heff_` :811/975 + cache-hit close
  :830; `t_env_update_` :989/1064 + :1078/1149; `t_svd_` :1331/1479;
  `t_lanczos_` :1824/2020. 5 begins / 6 ends (extra end is cache-hit
  close, mirrors -gpu).

- **Round-14 H-pdmrg-cache-hit-leak**: pdmrg_gpu_impl.h:983 closes
  `t_apply_heff_` before cache-hit `return;` at :984.

- **Round-14 M-skip-on-zero-prop**: `if (t.calls()==0) return;` in
  pdmrg-gpu :2831, pdmrg-gpu-opt :3622, multi-gpu :231.

- **Round-14 M-opt-pointer-mode**: `grep
  rocblas_set_pointer_mode pdmrg_gpu_opt_impl.h` = 0 hits. RAII
  `PointerModeGuard` at :1834, :1850, :1869, :2005, :2049.

- **Round-14 M-multi-n_recal-doc**: pdmrg_multi_gpu.h:55-57.

- **Round-13 H13-multi-rsvd-prop**: `use_rsvd_ = opts_.rsvd;` at
  pdmrg_multi_gpu_impl.h:163 (after load_from_env at :158).

- **Davidson workspace aliasing (technique F)**: dav_work_sz =
  max(theta_size_max·b, max_sub²) at pdmrg_gpu_opt_impl.h:274-276 —
  round-9 H-new1 envelope intact. d_T1/d_T2 sized t_max =
  D·dd·χ² at :238/250 — covers all V/U/T1/T2 reuse roles.

- **pdmrg-gpu-base immune to round-14 #2/#3/#4/#5/#6**: no
  GpuOpts, PhaseTimer, lanczos_graph, n_recal, or pointer-mode RAII
  surface — only #1 applies and is intact.

- **`use_chebyshev_` and `use_batched_sweep_` live**:
  pdmrg_gpu_opt_impl.h:2283 (chebyshev dispatch in optimize_bond),
  :3526/3539 (batched dispatch in run loop).

## SUMMARY

Round-14's six fixes (H1-base apply_heff scope; H-opt-phase-timer
propagation across 5 sites; H-pdmrg cache-hit leak fix at :983;
M-skip-on-zero across 3 variants; M-opt pointer-mode RAII —
0 raw `rocblas_set_pointer_mode` left in pdmrg-gpu-opt; M-multi
n_recal-doc) all verify INTACT, and the J1/J2/PDMRG-rules locks
hold across all four variants. The only HIGH this round is a
technique-G long-standing lonely-sibling: pdmrg-gpu-opt's
`block_davidson_eigensolver` still does host LAPACK syev per
Davidson iteration on the default code path, even though
dmrg-gpu-opt and dmrg2-gpu-opt both got the on-device
`rocsolver_syevd` port at round-7 (a9d24c3, batch 5). It was
explicitly deferred "post-G1" in round-8/9 reviews; with the
no-host-roundtrips-2026-04-27 rule now formalized, the deferral
window has closed. Three mediums round out the report:
`t_absorb_` is dead in -gpu and -opt, `t_env_update_` is dead in
multi-gpu, and a per-iteration `hipStreamSynchronize` inside the
Davidson CGS is the same host-control-flow pattern that the syev
fix would naturally remove. Recommend bundling the three Davidson
mediums into one fix that mirrors round-7 a9d24c3.
