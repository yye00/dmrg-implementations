# Horizontal review — -opt tier — round 16 (2026-05-01, post-187fddf)

## Charter

Review -opt tier (dmrg-gpu-opt, dmrg2-gpu-opt, pdmrg-gpu-opt). J2:
each -opt strict superset of -gpu sibling. Regression-watch since
`f40140d`. Round-16 closed two round-15 deferred HIGHs (H2-opt host
batch pointers, H-opt-pdmrg davidson syev) and propagated PhaseTimer
panel to dmrg/dmrg2-gpu-opt.

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol scan | DONE | t_absorb_ DEAD on all 3 (round-15 carry) |
| B. Behavioral diff | DONE | apply_heff_single_site (pdmrg-opt) diverges — host pointers on default warmup/polish |
| C. Docstring | DONE | impl:2297-2306 LANCZOS_GRAPH-disable note matches code |
| D. clangd | N-A | not invokable |
| E. Absence-naming | FOLLOWED | block_davidson PhaseTimer MISSING in pdmrg-opt |
| F. Workspace aliasing | DONE | d_dav_work/work2 + d_batch_*/*_env_ sizes match usage; round-15/16 didn't slice |
| G. Sibling fix-prop | DONE | H2-opt missed pdmrg-opt single-site Step 1; PhaseTimer-prop missed pdmrg-opt Davidson |

## Round-16 fix verification (since f40140d)

All four round-16 commits INTACT: H1-PhaseTimer-prop (69da5b4)
dmrg-opt 17 sites + dmrg2-opt 17 sites with early-returns covered;
H2-opt-host-batch (5355c06) dmrg-opt apply_heff/update_*_env using
shared kernels; H2-opt sweep (abd88b9) dmrg2-opt 4 sites + pdmrg-opt
cross-segment :2787; H2-opt pdmrg Step 3 fallback (187fddf) :2426;
H-opt-pdmrg-davidson-syev (5355c06) :1583 rocsolver_syevd +
d_dav_eigvals/E/info workspace fields. Earlier baselines all preserved:
use_davidson_=true J2 (3/3), M-opt-rsvd-env binding, CR-D1/H-new1
d_dav_work sizes, c3d3e50 h_svd_* gate.

Regression-watch grep `lapack_syev|std::vector<Scalar*> h_A|Scalar*
h_A[`: 4 hits — 2 comments (dmrg-opt:1558, pdmrg-opt:2298) plus
**2 LIVE host arrays at pdmrg-opt:2332 + 2357**, NOT init-time, NOT
use_cpu_svd_.

## CRITICALS

- **[pdmrg-gpu-opt: pdmrg_gpu_opt_impl.h:2332-2370]
  apply_heff_single_site Step 1 still builds `Scalar* h_A/B/C[256]`
  on host stack + 3× hipMemcpyAsync H2D per Lanczos iter — both
  sparse_s1 AND dense branches.** Single-site is default per PDMRG
  rules: lanczos_use_1site_=true set in sweep_LR_full_1site /
  sweep_RL_full_1site (impl:2659/2683); called from warmup :3461-3462
  and polish :3559-3560. H2-opt sweep (abd88b9 + 187fddf) patched
  apply_heff_two_site and cross-segment but missed single-site.
  Action: port to shared `setup_batch_ptrs_wd` (dense) +
  `setup_batch_ptrs_wd_sparse` (sparse), mirroring dmrg-gpu-opt
  apply_heff Step 1 at :650/634. Remove docstring caveat 2297-2306
  once fixed.

## HIGHS

- **[pdmrg-gpu-opt: impl:60-72] Duplicate template `__global__ void
  setup_batch_ptrs_wd_twosite_sparse`** — same signature as
  common/batch_ptrs_kernels.h:194-207 (included at :19). Other 4
  variants removed local copies in round-15. Compiles today because
  bodies byte-identical; future edit diverges silently or trips ODR.
  Action: delete impl:60-72.

- **[pdmrg-gpu-opt: block_davidson_eigensolver impl:1493-1792]
  zero `t_lanczos_.begin/end` sites.** Round-15 prop instrumented
  Davidson in dmrg-opt (begin :1494, 5 ends) and dmrg2-opt (begin
  :1447, 5 ends) but missed pdmrg-opt. Davidson is J2 default; on
  default path t_lanczos_ panel measures only tiny-fallback branch.
  Lonely-sibling per G. Action: add begin after :1499; matching end
  on every return path — tiny fallback :1506, norm-zero :1733,
  max-iter :1792.

## MEDIUMS

- **[all 3 -opt: t_absorb_ DEAD]** Round-15 carry. Init+report, no
  begin/end. -gpu siblings also uninstrumented. Either remove or add
  begin/end around absorb GEMMs.

- **[per-bond svd_truncate D2H + StreamSync, default path]**
  dmrg-opt:1366-1368, dmrg2-opt:1354, pdmrg-opt:2506/1387. When
  `device_k=false` (gpu_opts.h default), every SVD does 4-byte D2H
  + hipStreamSynchronize per bond. Structural; gated by
  `DMRG_GPU_OPT_DEVICE_K` env opt-in. Residual "host roundtrip per
  bond" violation per CLAUDE.md 2026-04-27. Recommend flipping
  default true or document as accepted debt.

## NITS

- **pdmrg-opt impl:2297-2306** LANCZOS_GRAPH-disabled docstring is
  accurate today; remove once CRITICAL fixed.
- **round-15 ctor/setter wording-drift** at pdmrg-opt :222-223 vs
  :64-66 unchanged.

## FALSE POSITIVES VERIFIED

- **dmrg-opt:1558 + pdmrg-opt:2298 `lapack_syev` / `Scalar* h_A[`
  in comments only** — not live code.
- **pdmrg-opt cross-segment v_ptrs/hv_ptrs/d_thetas at :2926/3212**
  — gated by `use_batched_sweep_` (default false at :208).
- **sparse Step 3 host-loop (dmrg-opt:684, dmrg2-opt:731,
  pdmrg-opt:925/2398)** — gated by `sparse_mpo` (default false).

## Self-audit

- **F**: 5355c06 added pdmrg-opt d_dav_eigvals/E/info, removed
  h_dav_H_proj/eigvecs/syev_work. d_dav_work/work2 sizes unchanged;
  H-new1 sequential-aliasing preserved. No overruns introduced.
- **G**: H2-opt propagated to dmrg-opt + dmrg2-opt + pdmrg-opt
  (cross-seg/Step 3 fallback). Two gaps: CRITICAL (single-site Step 1)
  and HIGH-1 (duplicate kernel). PhaseTimer-prop reached dmrg/dmrg2-opt
  Davidson; pdmrg-opt Davidson missed (HIGH-2).
- **Regression watch**: 3/3 use_davidson_=true; M-opt-rsvd-env live;
  pad_mfma16 idempotent; use_chebyshev_ live; no earlier reverts.
- **Verdict**: NOT READY. 1 CRITICAL on default warmup/polish path
  + 2 HIGHs. Fix before next benchmark window.

## SUMMARY

One CRITICAL + two HIGHs, all propagation gaps. CRITICAL: pdmrg-opt
`apply_heff_single_site` Step 1 still does host stack arrays + 3×
hipMemcpyAsync H2D per Lanczos iter — fires every warmup/polish iter
(single-site is default per PDMRG rules). H2-opt sweep handled two-
site, cross-segment, and Step 3 fallback but missed single-site
Step 1. HIGH-1: duplicate `__global__` template in pdmrg-opt impl:60
left over from shared-header migration; silent ODR risk. HIGH-2:
PhaseTimer-prop reached Davidson on dmrg/dmrg2-opt but missed
pdmrg-opt's block_davidson, so default-path t_lanczos_ panel is
empty. Carryovers: t_absorb_ dead all 3 (MEDIUM); per-bond
svd_truncate D2H is documented opt-in debt. Four round-16 fixes are
intact; gaps are propagation omissions, not regressions.
