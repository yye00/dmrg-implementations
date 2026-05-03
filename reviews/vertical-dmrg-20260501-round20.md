# Vertical review — DMRG family (single-site) — round-20 — 2026-05-01

HEAD: `f650466` (R19 reports landed; code state at `cafd628`).
Watch list per charter: `cafd628`, `54f2fcf`, `0efe96d`, `8abb6e7`,
`187fddf`, `abd88b9`, `12d02c5`. All three tiers in scope.

This is a **second confidence re-run** before MI300X allocation —
R19 caught H19 (a propagation gap R18 missed); user wants extra
signal. I conducted a fresh A-G audit independently of R19's
verdict.

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | -base 22 / -gpu ~44 / -opt ~57 members. Every member ≥3 hits (alloc + free + ≥1 use). 0 dead. |
| B. Behavioral diff | DONE | apply_heff, env_update, lanczos, svd, sweep aligned across tiers; -opt-only Davidson; -opt Step 3 ablation divergence (intentional, see FP). |
| C. Docstring verify | DONE | -base header L13-18 still overstates "device-pointer mode" for GEMM/GEMV; carried NIT from R16-R19. |
| D. clangd filter | N-A | No ROCm headers on host; A subsumes. |
| E. Absence-naming | FOLLOWED | -base / -gpu / -opt expected features all present; -opt 6/6 PhaseTimer panels live. |
| F. Workspace-aliasing | DONE | 5 shared buffers checked; 0 OVERRUN. |
| G. Sibling fix-propagation | DONE | 7 watch-list fixes traced; 0 MISSING in dmrg family. R19 H19 (multi-gpu) examined for dmrg-family analogy — none present. |

## Defect-class registry sweep (pre-step)

`bash .claude/scripts/defect-registry.sh` → **TOTAL HITS: 0** across
all 14 D-classes (D1-D5, D6-collapsed, D7-D15) and all 10 in-charter
variants. Pre-step gate cleared.

## Technique F detail (workspace-aliasing)

| Buffer | Regions | Lifetime | Required size | Allocated size | Verdict |
|---|---|---|---|---|---|
| `d_dav_work_` (-opt, impl :305) | (1) W residuals [0, n_new·dim); (2) overlap [n_new·dim, n_new·dim + k·n_new); (3) restart-X_keep [0, dim·keep) | (1)+(2) concurrent (residual loop, impl :1632-1666); (3) sequential (post-syevd, :1718-1727) | max(b·theta + max_sub·b, theta·keep) | max(b·theta + max_sub·b, max_sub²) — at chi=128,d=2,b=4,max_sub=32: 1056 Scalars | OK |
| `d_dav_work2_` (-opt, impl :306) | H_proj [0, k·k); eigvecs [0, k·k) in-place after syevd | sequential (overwritten) | max_sub² | dav_work_sz (same alloc as work_) | OK (over-allocated, harmless) |
| `d_T1_/d_T2_` (-gpu, -opt) | apply_heff V/U slots; absorb scratch on T1 | sequential (eigensolver completes before absorb begin) | t_max = D·d·chi_max² | t_max | OK |
| `d_T1_env_/d_T2_env_` (-gpu, -opt) | env_update V/U only | single-role | t_max | t_max | OK |
| `d_batch_*_` vs `d_batch_*_env_` (-gpu, -opt) | concurrent across stream_ vs stream_env_ | concurrent — disjoint per-stream tables | batch_max each | batch_max each | OK |

R8 CR-D1 sizing (`b·theta + max_sub·b`) verified intact at -opt
impl :301-307. R7 H6 syev port aliasing pattern (`overlap = d_dav_work_
+ n_new*dim`) verified intact at impl :1666. **R8 CR-D1 fix has not
regressed across R9-R20.**

## Regression-watch verification (R15→R19)

| Fix (commit) | Variant | Site | Status |
|---|---|---|---|
| abd88b9 D5 host-batch-ptr → GPU kernels | dmrg-gpu | impl :600-616 | OK |
| 187fddf D5 Step 3 fallback align | dmrg-gpu / -opt | -opt :718-755 fallback aligned | OK |
| 8abb6e7 D6 kernel-dup collapsed | dmrg-gpu / -opt | both `#include "../../common/batch_ptrs_kernels.h"` (impl :12-13) | OK |
| 0efe96d D12 Lanczos device-ptr | dmrg-gpu-opt | h:176-186, impl :175-191 alloc, :996-1198 inner loop | OK |
| 0efe96d D12 baseline | dmrg-gpu | h:120-127, impl :143-167 alloc, :944-1021 device mode | OK |
| 0efe96d D12 immune | dmrg-gpu-base | impl :510-637 already device-ptr (R14 fix) | IMMUNE |
| 54f2fcf CR-D1/D14 Davidson sizing | dmrg-gpu-opt | impl :301-307 sizing, :1666-1671 overlap-at-offset | OK |
| 12d02c5 H1 t_absorb_ split | dmrg-gpu-opt | impl :1384-1413 (R-dir), :1437-1465 (L-dir) | OK |
| 12d02c5 H2 t_davidson_ panel | dmrg-gpu-opt | h:272 decl, impl :1492 begin, .end at :1584/:1624/:1657/:1709/:1768 (5 paths) | OK |
| 12d02c5 t_lanczos_ scoping | dmrg-gpu-opt | impl :996/:1200 wrap only Lanczos; tiny-dim short-circuit at :1485-1487 returns BEFORE t_davidson_.begin | OK |
| 12d02c5 init/report includes t_davidson_ | dmrg-gpu-opt | impl :1982 init, :2002 report | OK |
| **cafd628 H19** apply_heff_single_site Step 3 | pdmrg-multi-gpu (out-of-scope) | — | DEFERRED to vertical-pdmrg / horizontal-gpu |
| dmrg-gpu intentionally omits t_davidson_ | dmrg-gpu | h:193-197 5 panels (no Davidson) | INTENTIONAL |
| dmrg-gpu-base intentionally omits all PhaseTimers | dmrg-gpu-base | h:53-148 charter-correct | INTENTIONAL |

## Special technique-G focus: H19 sibling-propagation in dmrg family

The R19 H19 defect class was: **per-wp host loop wrapping a batched
gemm in apply_heff (default code path)**. R18 missed this in
pdmrg-multi-gpu apply_heff_single_site because D13's grep matched
only bare `Traits::gemm()`. R19 widened the awk to also flag
`gemm_batched`/`gemm_strided_batched` inside per-wp host loops.

**Audit for the dmrg family:**

- **dmrg-gpu apply_heff Step 3** (impl :645-692): single
  `setup_batch_ptrs_step3_full` kernel + single `gemm_batched(batch=D*d)`
  + `gemv` reduce over D using `d_ones_D_`. **Already R3-F1 collapsed.**
  No per-wp host loop. **IMMUNE to H19 class.**
- **dmrg-gpu-opt apply_heff Step 3** (impl :709-755): three branches.
  (a) sparse_s3 path uses host-side `h_WL_nnz_cols_[site]` loop with
  per-iter `Traits::gemm` — opt-in `opts_.sparse_mpo`, default OFF.
  (b) Strided-batched fast path (cL≥16, cR≥16, d≤2) does D launches of
  `gemm_strided_batched(batch=d)`. (c) Fallback (d≥3 or small chi) does
  D·d launches of `Traits::gemm`. The default-path divergence from -gpu
  (R3-F1 collapse) is documented at impl :707-708 as a deliberate
  cache-contention ablation.

  Per the registry awk, the wp-loops at -opt impl :728/:741 SHOULD have
  been flagged by D13 but were not — the awk's depth tracking over
  `apply_heff` triggers on header comments (h:66, :71) and other
  inner-function references, exits state too early. **This is a
  registry false negative**, but the divergence itself was explicitly
  classified INTENTIONAL by R19 (cache-contention rationale documented
  in code comment). The user-locked tier definition (header L17-30)
  permits algorithmic explorations in -opt that are not drop-in.

  **Recommendation**: tighten the D13 awk to scope by `template<...>
  void DMRGGPU*::apply_heff` rather than substring `apply_heff`, and
  re-evaluate the -opt wp-loops as a separate finding (currently
  classified intentional). Tracked as R20 NIT, not raised to a HIGH
  because (a) R19 explicitly evaluated and accepted the divergence and
  (b) the strided-batched path is still an order-of-magnitude faster
  than the fallback for the dominant (d=2, chi≥16) regime.

- **env_update Step 3** in both -gpu (impl :761-776) and -opt
  (impl :826-854) uses a per-sp host loop wrapping `gemm_batched(batch=D)`.
  Sibling-consistent across both. Could in principle be collapsed to
  a single batched call with d·D batches, but the per-sp accumulation
  pattern (beta=1 after first iter) is structurally different from
  apply_heff Step 3 (no GEMV reduce). Not raised.

## CRITICALS

None.

## HIGHS

None.

## MEDIUMS

- **M1-opt-lanczos-init-D2H-sync** [dmrg-gpu-opt:
  dmrg_gpu_opt_impl.h:1015-1017]. Carried unchanged from R18/R19. The
  D12 device-pointer port of init nrm2 does `→ d_nrm2_result_` then
  `hipMemcpyAsync &norm + hipStreamSynchronize`, while sibling
  **dmrg-gpu** :914 keeps init nrm2 in host-pointer mode
  (`Traits::nrm2(..., &norm)`), avoiding the explicit sync. Once-per-
  Lanczos-call on the fallback path (Davidson is default in -opt).
  Tagged MEDIUM not HIGH because (a) Lanczos is fallback here, not the
  hot path, and (b) the explicit sync is bounded one-shot cost.

## NITS

- **dmrg-gpu-base docstring overgeneralization** [dmrg_gpu_base.h:13-18].
  Carried from R16-R19. Header reads "rocBLAS GEMM/GEMV/AXPY/DOT/NRM2
  in device-pointer mode," but apply_heff (impl :273-318), env updates
  (impl :347-454), and the Lanczos-final GEMV (impl :620-625) use
  host-stack `&one/&zero`. Only Lanczos BLAS-1 ops (impl :518-524,
  :539-588, :629-633) are wrapped in `PointerModeGuard` device mode.
  Cosmetic, no behavioral defect.
- **R19M19 D13 awk function-scope tightening (suggested)**
  [.claude/scripts/defect-registry.sh:201-227]. The current awk pattern
  matches any line containing `apply_heff` (including comments, other
  function references, ctor/dtor mentions), then tracks `{`/`}` depth
  unreliably — exits the function body before reaching the actual
  Step 3 wp-loops in dmrg-gpu-opt. The dmrg-gpu-opt strided-batched
  path divergence is intentional (R19-classified), so the registry
  miss does not hide a defect; but a future genuine wp-loop defect in
  apply_heff body could escape detection. Suggest replacing substring
  match with a `template<...> void DMRG*::apply_heff` regex scoped to
  the actual function definition.

## FALSE POSITIVES VERIFIED

- **Init-time host→device memcpys** (const tables, W matrices, R/L
  boundary). Constructor / `set_mpo` / `build_initial_environments`
  only — not per-sweep.
- **dmrg-gpu-opt :276 + :1250 lapack_gesvd**. :276 is `lwork=-1`
  workspace-size query; :1250 is `use_cpu_svd_` opt-in. Both gated.
  dmrg-gpu / dmrg-gpu-base have 0 lapack_ hits.
- **`opts_.device_k` in -gpu, absent in -opt**. Intentional: -opt's
  chi_max is MFMA-padded so the on-device truncation kernel produces
  the right shape unconditionally.
- **t_apply_heff_.end inside hipStreamBeginCapture region**. Pre-existing
  design; matches -gpu sibling. Slight measurement drift only.
- **Step 3 pattern divergence (-gpu R3-F1 collapsed vs -opt
  strided/looped)**. Re-verified: the divergence is documented
  (impl :707-708 cache-contention rationale) and locked by the -opt
  tier definition (h:17-30). Both implementations are correct;
  performance ablation is the user-chosen design. Not the H19 class
  (which was a missed sibling port, not a deliberate divergence).
- **-opt sparse_s3 host-resident nnz loop** (impl :714-726). Off by
  default (`opts_.sparse_mpo`). When on, opts in for sparse-MPO
  workloads where the launch overhead amortizes against the row-skip
  savings.

## SUMMARY

R20 confirms R19: **0 CRITICAL, 0 HIGH** in the dmrg family. The
defect-class registry pre-step is clean (TOTAL HITS: 0 across all 14
classes / 10 variants). The R15→R19 watch list (7 fix commits) is
intact — every fix verified in place; no regressions. The R8 CR-D1
Davidson buffer overrun fix in dmrg-gpu-opt is correctly preserved
(`b·theta + max_sub·b` sizing at impl :301-307; overlap-at-offset
aliasing at impl :1666). All 6 -opt PhaseTimer panels are live
(init+begin+end+report). The R19 H19 sibling-propagation defect class
(per-wp host loop wrapping batched gemm in apply_heff) was
specifically audited in dmrg family: dmrg-gpu apply_heff is IMMUNE
(R3-F1 collapsed); dmrg-gpu-opt's per-wp pattern is documented
INTENTIONAL cache-contention ablation, not a missed propagation. One
M1 carry (Lanczos init D2H sync, fallback path only) and one NIT
carry (-base docstring overgeneralization). Plus one new R20-suggested
NIT (D13 awk function-scope tightening — does not hide a real defect
today, but the registry's current substring matching could miss a
future genuine in-body wp-loop). The dmrg family is **READY** for the
MI300X G1 baseline campaign with no blocking items.
