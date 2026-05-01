# Vertical review — DMRG family (single-site) — round-19 — 2026-05-01

HEAD: `bb809fb` (R18 reports landed; code state at `12d02c5`).
Watch list per charter: `abd88b9`, `187fddf`, `8abb6e7`, `0efe96d`,
`54f2fcf`, `12d02c5`. All three tiers in scope.

This is an **independent confidence re-run** before MI300X allocation.
I conducted a fresh A-G audit without carrying R18's verdict.

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | -base 22 / -gpu ~44 / -opt ~57 members. Hit-counts: every member ≥3 (alloc + free + ≥1 use). 0 dead. |
| B. Behavioral diff | DONE | Sweep, optimize_site, svd, apply_heff, env-update, lanczos structurally aligned. -opt adds Davidson + MFMA-pad + RSVD per J2; -gpu has `opts_.device_k` ablation that -opt intentionally lacks (see false positives). |
| C. Docstring verify | DONE | -base header L13-18 still overstates "device-pointer mode" for GEMM/GEMV; carried NIT from R16/R17/R18. |
| D. clangd filter | N-A | No ROCm headers on host; A subsumes. |
| E. Absence-naming | FOLLOWED | All -base / -gpu / -gpu-opt expected features present. -opt 6/6 PhaseTimer panels instrumented. |
| F. Workspace-aliasing | DONE | 4 shared buffers checked across hot path; 0 OVERRUN. Detail below. |
| G. Sibling fix-propagation | DONE | 6 watch-list fixes traced; 0 MISSING in dmrg family. Cross-family CR-D1 propagation deferred to vertical-pdmrg / horizontal-opt. |

## Defect-class registry sweep

`bash .claude/scripts/defect-registry.sh` → **TOTAL HITS: 0** across all
14 D-classes (D1-D15, D6 collapsed). All 10 in-charter variants clean.

## Technique F detail (workspace-aliasing)

| Buffer | Regions | Lifetime | Required size | Allocated size | Verdict |
|---|---|---|---|---|---|
| `d_dav_work_` (-opt) | W (residuals) [0, n_new·dim) + overlap [n_new·dim, n_new·dim + k·n_new); restart-X_keep [0, dim·keep) | (1)+(2) concurrent in expand-path; (3) sequential w.r.t. (1)+(2) (restart only after (2) consumed) | max(b·theta + max_sub·b, theta·b) = b·theta + max_sub·b | max(b·theta + max_sub·b, max_sub²) — at chi=128,d=2 → 131200 Scalars (b·theta dominates) | OK |
| `d_dav_work2_` (-opt) | H_proj [0, k·k); eigvecs [0, k·k) (in-place after syevd) | sequential (overwritten) | max_sub² | max(b·theta + max_sub·b, max_sub²) (same alloc as work_) | OK (over-allocated; harmless) |
| `d_T1_/d_T2_` (-gpu, -opt) | apply_heff V/U; absorb scratch (d_T1_ only) | sequential (Davidson/Lanczos completes before absorb) | t_max = D·d·chi² | t_max | OK |
| `d_T1_env_/d_T2_env_` (-gpu, -opt) | env_update only | single-role | t_max | t_max | OK |
| `d_batch_*_` vs `d_batch_*_env_` (-gpu, -opt) | concurrent across stream_ vs stream_env_ | concurrent — disjoint buffers per stream | batch_max each | batch_max each | OK |

R8 CR-D1 sizing (`d_dav_work_` Davidson buffer overrun) verified intact
in dmrg-gpu-opt at impl :301-307. The R8 fix has not regressed across
R9-R18.

## Regression-watch verification (R15→R18 fixes)

| Fix (commit) | Variant | Site | Status |
|---|---|---|---|
| abd88b9 D5 host-batch-ptr propagation | dmrg-gpu | impl :600-616 (`setup_batch_ptrs_wd` w/ env stream) | OK |
| 187fddf D5 Step 3 fallback | dmrg-gpu / -opt | -opt :718-755 fallback aligned with -gpu pattern | OK |
| 8abb6e7 D6 kernel-dup collapsed | dmrg-gpu / -opt | both `#include "../../common/batch_ptrs_kernels.h"` | OK |
| 0efe96d D12 Lanczos device-ptr mode | dmrg-gpu-opt | h:176-186 (11 device scalars), impl :175-191 alloc, :996-1198 inner loop in device mode | OK |
| 0efe96d D12 baseline | dmrg-gpu | h:120-127 declarations, impl :143-167 alloc, :944-1021 device mode | OK |
| 0efe96d D12 immune | dmrg-gpu-base | impl :510-637 already device-pointer pattern (was R14 fix) | IMMUNE |
| 54f2fcf CR-D1/D14 propagation | dmrg-gpu-opt | impl :301-307 (sizing), :1665-1671 (overlap-at-offset) | OK |
| 54f2fcf CR-D1/D14 propagation | dmrg2-gpu-opt | out-of-scope vertical-dmrg; covered by vertical-dmrg2 / horizontal-opt | DEFERRED |
| 12d02c5 H1 t_absorb_ split | dmrg-gpu-opt | impl :1384-1413 (R-dir), :1437-1465 (L-dir) | OK |
| 12d02c5 H2 t_davidson_ panel | dmrg-gpu-opt | h:272 decl, impl :1492 begin, .end at :1584/1624/1657/1709/1768 (5 paths) | OK |
| 12d02c5 t_lanczos_ no longer wraps Davidson | dmrg-gpu-opt | impl :996/:1200 wrap only Lanczos; tiny-dim short-circuit at :1485-1487 returns BEFORE t_davidson_.begin | OK |
| 12d02c5 init/report includes t_davidson_ | dmrg-gpu-opt | impl :1982 init, :2002 report | OK |
| dmrg-gpu intentionally omits t_davidson_ | dmrg-gpu | h:193-197 5 panels (no Davidson) | INTENTIONAL |
| dmrg-gpu-base intentionally omits all PhaseTimers | dmrg-gpu-base | h:53-148 charter-correct | INTENTIONAL |

## CRITICALS

None.

## HIGHS

None.

## MEDIUMS

- **M1-opt-lanczos-init-D2H-sync** [dmrg-gpu-opt: dmrg_gpu_opt_impl.h:1011-1017].
  Carried unchanged from R18. The D12 device-pointer port of the
  initial-norm computation does `nrm2 → d_nrm2_result_` followed by
  `hipMemcpyAsync &norm, D2H + hipStreamSynchronize`. Sibling **dmrg-gpu**
  :914 keeps the one-shot init nrm2 in host-pointer mode
  (`Traits::nrm2(..., &norm)` direct host-stack), avoiding the explicit
  sync entirely. Once-per-Lanczos-call, on the fallback path (Davidson
  is the default in -opt). Cleaner pattern: match -gpu's host-pointer
  init, then enter the device-pointer guarded region for the per-iter
  loop. Tagged MEDIUM, not HIGH, because (a) Lanczos is the fallback
  here, not the hot path, and (b) the explicit sync is bounded cost.

## NITS

- **dmrg-gpu-base docstring overgeneralization** [dmrg_gpu_base.h:13-18].
  Carried from R16/R17/R18. The header reads "rocBLAS GEMM/GEMV/AXPY/DOT/
  NRM2 in device-pointer mode," but apply_heff (impl :273-318), env
  updates (impl :347-454), and the Lanczos-final GEMV (impl :620-625)
  use host-stack `&one/&zero`. Only Lanczos BLAS-1 ops (impl :518-524,
  :539-588, :629-633) are wrapped in `PointerModeGuard ...
  rocblas_pointer_mode_device`. Either narrow the claim ("Lanczos BLAS-1
  ops in device-pointer mode") or guard the GEMM scalars too. Cosmetic,
  no behavioral defect.

## FALSE POSITIVES VERIFIED

- **Init-time host→device memcpys** (const tables, W matrices, R[L]
  boundary). Constructor / `set_mpo` / `build_initial_environments`
  only — not per-sweep.
- **Registry D7/D8 (dmrg-gpu-opt :276 + :1250 lapack_gesvd)**. :276 is
  init-time `lwork_query=-1` workspace-size query. :1250 is
  `use_cpu_svd_` opt-in branch (off by default). Both gated. dmrg-gpu /
  dmrg-gpu-base have 0 lapack_ hits.
- **`opts_.device_k` in -gpu, absent in -opt**. Intentional: -opt's
  chi_max is MFMA-padded so the on-device truncation kernel produces
  the right shape unconditionally. J2 is feature presence, not
  ablation-flag parity.
- **t_apply_heff_.end inside hipStreamBeginCapture region**. Pre-existing
  design; matches the -gpu sibling. Slight measurement drift only.
- **Step 3 pattern divergence (-gpu D-batched + GEMV-reduce vs -opt
  strided/looped)**. Different ablations of the same algorithm; both
  correct.

## SUMMARY

R19 is a clean independent confirmation of R18: zero CRITICALs, zero
HIGHs, one carried MEDIUM (M1-opt-lanczos-init-D2H-sync) and one
carried NIT (dmrg-gpu-base header overgeneralization). The defect
registry sweeps all 14 classes to 0 hits across all 10 in-charter
variants. The R15→R18 watch list (six fix commits) is fully intact:
no fixes have regressed and no sibling propagation gaps reappeared.
The R8 CR-D1 Davidson buffer sizing in dmrg-gpu-opt remains correctly
sized (`b·theta + max_sub·b`) and the round-7 H6 syev-port aliasing
pattern is preserved. The dmrg family is **READY** for the MI300X GPU
allocation window, with the M1 carry as a tracked but non-blocking
cleanliness item.
