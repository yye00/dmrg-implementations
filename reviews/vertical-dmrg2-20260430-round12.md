# Vertical review — dmrg2 family — 2026-04-30 round-12 (post-cleanup)

HEAD: `a457fdc`. Baseline: round-11 at `2d55d90` /
`reviews/vertical-dmrg2-20260428-round11.md`. Scope this round per
brief: `gpu-rocm/dmrg2-gpu/` (-gpu reference) and
`gpu-rocm/dmrg2-gpu-opt/` (-opt). The -base tier is out of scope for
the round-12 charter; deltas since 2d55d90 in -base = none (`git log
2d55d90..HEAD -- gpu-rocm/dmrg2-gpu-base` empty).

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | -gpu: all surviving members live; -opt: all surviving members live |
| B. Behavioral diff | DONE | -gpu ↔ -opt: deltas all intentional (Davidson primary, RSVD/D_PAD/MFMA-16 padding, public setter API, sparse-MPO, env-stream batched ptrs); 30 vs 29 stream-event sites comparable |
| C. Docstring verification | DONE | dmrg2-gpu.h:82-93 (dual-stream pipeline), :149-152 (gesvdj device scalars), :165-172 (lanczos_graph cache) all match impl. dmrg2-gpu-opt.h:67-83 setters wired. `precompute_fused_mpo` host-build comment matches code at impl:506-546 (-opt) and impl:651+ (-gpu) |
| D. clangd filter | N-A | No clangd available locally |
| E. Absence-naming brief | FOLLOWED | -gpu and -opt expected-feature checklists pass; no MISSING items vs round-11 |
| F. Workspace-aliasing audit | DONE | 5 shared buffers checked across -opt (d_T1_, d_T2_, d_T1_env_, d_T2_env_, d_dav_work_/d_dav_work2_); 0 OVERRUN |
| G. Sibling fix-propagation | DONE | 2 fixes traced (bc3fcd0 setter pattern, c3d3e50 dead-buffer cleanup); 0 MISSING |

A-G all completed; D is N-A (no clangd locally — reverted to A for
dead-symbol coverage as methodology allows).

## Regression-watch verification (per round-12 brief)

### bc3fcd0 — M-opt-davidson-toggle setter pattern in dmrg-gpu-opt

Verify dmrg2-gpu-opt's symmetric setter from round-7 M5 is intact:

- **dmrg2-gpu-opt.h:73-81** — `set_use_davidson(bool use_dav)` toggles
  `use_davidson_`, disables `opts_.lanczos_graph` when Davidson goes
  ON (and sets `lanczos_graph_was_user_enabled_`), re-enables
  `opts_.lanczos_graph` when Davidson goes OFF iff the user had
  enabled it.
- **dmrg2-gpu-opt.h:232** — declaration
  `bool lanczos_graph_was_user_enabled_ = false;`.
- **dmrg2-gpu-opt_impl.h:74** — ctor sets
  `lanczos_graph_was_user_enabled_ = true` when ctor disables
  `opts_.lanczos_graph` on the Davidson-default path. Symmetric.
- **set_cpu_svd / set_rsvd** (dmrg2-gpu-opt.h:67, :82) present, simple
  flag setters (no graph-coupling needed).

**Verdict: INTACT.** Round-7 M5 still in force; no regression from
bc3fcd0 (which was the dmrg-gpu-opt back-port — does not touch the
dmrg2 family).

### c3d3e50 — Spotless dead-buffer cleanup

Per round-11 mediums (MED-d_svd_work-{gpu,opt}, MED-h_svd_buffers-gpu,
MED-h_svd_tmp-opt) the cleanup target list was: in dmrg2-gpu drop
`h_svd_U_`, `h_svd_Vh_`, `h_svd_S_`, `h_svd_tmp_`, and `d_svd_work_`;
in dmrg2-gpu-opt drop `h_svd_tmp_` and `d_svd_work_`.

- **dmrg2-gpu**: `grep h_svd_ dmrg2_gpu.h dmrg2_gpu_impl.h` → 0 hits;
  `grep d_svd_work_ … ` → 0 hits. Header (line 154) keeps the
  "CPU workspace" comment line as a delimiter; no member declarations
  remain. Surviving `d_svd_*` members (A_, U_, S_, Vh_, E_, info_,
  svdj_residual_, svdj_n_sweeps_) all live (impl:1300-1380 SVD path).
- **dmrg2-gpu-opt**: `grep h_svd_tmp_ …` → 0 hits;
  `grep d_svd_work_ …` → 0 hits. Surviving CPU-SVD-fallback host
  buffers `h_svd_A_`, `h_svd_U_`, `h_svd_Vh_`, `h_svd_work_`,
  `h_svd_S_`, `h_svd_rwork_` (header line 192-194) all live —
  resized at impl:228-232 and read in the `use_cpu_svd_` path at
  impl:1263-1283 (`hipMemcpyAsync(host)` → `Traits::lapack_gesvd` →
  `hipMemcpyAsync(device)`).

**Verdict: round-11 mediums fully cleared. No new dead infrastructure
introduced (`git diff 2d55d90..HEAD` is pure deletions: `-311/+2`).**

### Round-9/10/11 watchlist (carry-forward sanity)

- **dav_work_sz (CR-D1, round-8)**: -opt impl:265-269 sizes
  `dav_work_sz = max(theta_size_max·b + max_sub·b, max_sub·max_sub)`.
  Inner-loop residual+overlap aliasing at impl:1607-1608 fits;
  restart at impl:1663-1667 reuses sequentially. OK.
- **Round-6 dual-stream wire-up + direction-L reorder**: -gpu records
  `event_canon_ready_` at impl:1413 (R) and 1438 (L), waits at
  impl:1483/1495/1510/1522. -opt records at impl:1391/1421, waits at
  impl:1761/1782/1797/1815. Direction-L writes MPS[site+1]=Vh BEFORE
  the U·S absorb in both (-gpu impl:1428-1438; -opt impl:1408-1421).
  Intact.
- **D_PAD precompute_fused_mpo OOB fix**: -opt impl:515-541 loops
  bounded by `D_act` (`D_mpo_actual_`) while indexing into
  `h_WW[D_use*…]`. No write past `D_use*dd*dd*D_use`. Intact.
- **rocsolver_syevd port (round-7 H6)**: -opt impl:1527 calls
  `Traits::rocsolver_syevd` on `stream_`-bound `rocblas_h_`. Intact.
- **non-blocking stream flag (H1-ext-gpu)**: -gpu impl:198/201, -opt
  impl:107/108 both create `stream_`+`stream_env_` with
  `hipStreamNonBlocking`. Intact.

## CRITICALS

None.

## HIGHS

None.

## MEDIUMS

None. The round-11 medium-tier findings (4 dead-buffer mediums in
dmrg2 family) are all closed by c3d3e50.

## NITS

- `dmrg2-gpu/src/dmrg2_gpu.h:154` retains the orphan comment
  "// CPU workspace (for receiving GPU SVD results and truncation/
  scaling)" with no following member declarations after the cleanup.
  Cosmetic; defer or delete in next pass.
- `dmrg2-gpu-opt/src/dmrg2_gpu_opt.h:190-194` "CPU SVD workspace
  (legacy — only used by the init-time workspace query; runtime SVD
  is fully on-device)" comment slightly understates the role:
  these vectors ARE used at runtime in the `use_cpu_svd_` opt-in
  path (impl:1263-1283), not just at init. Cosmetic; comment is not
  wrong about default-path behavior.

## FALSE POSITIVES VERIFIED

- Lanczos sync-free scalars (`d_dot_result_`, `d_nrm2_result_`,
  `d_neg_alpha_`, `d_neg_overlap_`, `d_inv_nrm_`, `d_alpha_dev_`,
  `d_beta_dev_`, `d_neg_beta_scalars_`) and constants (`d_const_one_`,
  `d_const_zero_`, `d_const_neg_one_`, `d_ones_D_`) in dmrg2-gpu.h
  lines 115-140 were the cleanup target IN DMRG-GPU-OPT (not in the
  dmrg2 family). Verified live in dmrg2-gpu impl:1127-1186 (Lanczos
  inner loop) and :882 (Step-3 ones-D reduction). Not dead.
- `chi_max_user_`, `D_mpo_actual_`, `lanczos_graph_was_user_enabled_`
  in dmrg2-gpu-opt: live (D_PAD path, MFMA padding init message,
  setter logic). Same as round-11 false-positive list.

## SUMMARY

Round-12 returns **0 critical, 0 high, 0 medium, 2 nits** for the
dmrg2 family on the round-12 scope (-gpu and -opt). All round-11
deferred mediums are closed by `c3d3e50`. The bc3fcd0 setter-pattern
fix is a dmrg-gpu-opt back-port; the symmetric M5 pattern is intact
in dmrg2-gpu-opt with no regression. No new dead infrastructure was
introduced (cleanup is pure deletion: -311/+2 across the whole repo,
of which dmrg2 family is -17/+0). Workspace aliasing for
`d_dav_work_` / `d_dav_work2_` (the round-8 CR-D1 site) still satisfies
the residual+overlap concurrent-region budget. Round-11 was the last
non-clean round for dmrg2; round-12 is **clean** — recommend
declaring the second consecutive zero-finding sub-review for this
family. Block GPU run? **NO**, family is ready.

(Length: ~770 words.)
