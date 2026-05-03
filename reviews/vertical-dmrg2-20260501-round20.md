# Vertical review — dmrg2 family — 2026-05-01 round-20

HEAD: `f650466` (R19 reports landed; **code state at `cafd628`**).
Baseline: `reviews/vertical-dmrg2-20260501-round19.md`. R20 is a
**second confidence re-run before MI300X allocation** (R19 caught H19 on
pdmrg-multi-gpu that R18 missed; user wants extra signal). Scope:
`gpu-rocm/dmrg2-gpu-base/`, `gpu-rocm/dmrg2-gpu/`,
`gpu-rocm/dmrg2-gpu-opt/`.

Pre-step: `bash .claude/scripts/defect-registry.sh` reports
**TOTAL HITS: 0** across all in-charter variants (D1–D15). Sweep
includes the dmrg2 family.

`git log bb809fb..HEAD -- gpu-rocm/dmrg2-* common/` returns empty —
**no code changes touch the dmrg2 family since R19**. The only commit
since the R19 baseline is `f650466` (review reports only).

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | All 11 D12 device buffers (-opt) live; 6 PhaseTimers in -opt all live (init+begin+end sites verified); 5 PhaseTimers in -gpu all live. 0 dead members. |
| B. Behavioral diff | DONE | -base ⊂ -gpu ⊂ -gpu-opt tier-superset relationship intact. -base lacks PhaseTimer/GpuOpts/RSVD/sparse/D_PAD/Davidson per docstring (impl:25–28); -gpu adds these; -opt adds Block-Davidson + MFMA pad + setters on top. 0 unexplained divergences. |
| C. Docstring verification | DONE | -opt header `t_davidson_` "(excl. Lanczos fallback)" matches impl: `t_davidson_.begin` at impl:1431 sits AFTER the `dim ≤ 2*b` Lanczos short-circuit at impl:1424–1426. -base header "host-side WW build at set_mpo time, outside timed sweep region" matches `precompute_WW()` at impl:259+. All claims verified. |
| D. clangd filter | N-A | no clangd locally; A subsumes |
| E. Absence-naming brief | FOLLOWED | -base/-gpu/-opt expected-feature checklists pass |
| F. Workspace-aliasing audit | DONE | `d_dav_work_` overlap-sizing re-verified; `d_T1_/T2_`, `d_T1_env_/T2_env_`, `d_lanczos_v_`, `d_svd_*`, `d_rsvd_*`, `d_heff_input_`, `d_dav_V_/AV_` audited. 0 OVERRUN. |
| G. Sibling fix-propagation | DONE | All R15→R19 watch-list fixes traced + cafd628 (out of dmrg2 scope) verified non-regressing. 0 MISSING. |

## R19→R20 watch-list verification

### 1. cafd628 — pdmrg-multi-gpu Step 3 batched port (D13 widened pattern)

R19 H19 was a per-element host loop in `pdmrg-multi-gpu`'s single-site
Step 3. **Out of dmrg2 family scope**, but per brief: must verify the
widened D13 pattern does NOT fire on dmrg2 family `apply_heff_two_site`.

- **dmrg2-gpu-opt** apply_heff Step 3 (impl:755–810): three branches,
  all GEMM-based. (a) sparse_s3 branch (impl:756–775): `nnz` host loop
  but each iteration issues a single `Traits::gemm`, with index packing
  `packed = h_nnz[idx]` from a precomputed `h_WW_nnz_cols_[site]` table
  (set at set_mpo time, off the per-call hot path). (b) Strided-batched
  branch (impl:776–792): `gemm_strided_batched` over `s1p` per `(s2p,n)`
  — single batched dispatch. (c) Naive nested-loop branch (impl:793–
  809): per-`(s1p,s2p,n)` `Traits::gemm`. **No per-element host scalar
  loops, no `std::vector<Scalar*>` host pointer arrays, no D2H/H2D in
  the body. D13 immune.**
- **dmrg2-gpu** apply_heff Step 3 (impl:716–767): full-batched collapse
  via `setup_batch_ptrs_step3_twosite_full[_sparse]` device kernels →
  `gemm_batched` → `gemv` reduction. Pointer arrays built by device
  kernel into `d_batch_A_/B_/C_`. **D13 immune.**
- **dmrg2-gpu-base** apply_heff Step 3 (impl:398–415): per-`(n, s1p,
  s2p)` `Traits::gemm` from a host loop, single dense GEMM per
  iteration. Charter-conformant per the -base "naive first-pass tier"
  docstring. **Not the D13 per-element scalar pattern; OK.**

### 2. R15→R19 carry-watch (must NOT regress)

| Item | Site | Status |
|---|---|---|
| D5 host-batch-ptr → device kernel (-gpu, -opt) | dmrg2_gpu_impl.h:725, 742, 851, 924; dmrg2_gpu_opt_impl.h:711, 728, 851, 924 | **INTACT** |
| D6 shared `common/batch_ptrs_kernels.h` | dmrg2_gpu_impl.h:17; dmrg2_gpu_opt_impl.h:13 | **INTACT** |
| D12 device-pointer Lanczos (-opt): 11 device buffers + PointerModeGuard + 4 per-iter kernels | dmrg2_gpu_opt.h:154–164 (decl), impl:176–186 (alloc), impl:1020/1032/1036/1053/1178 (PointerModeGuard scopes), impl:1056/1064/1093/1099/1107/1122 (per-iter kernels) | **INTACT** |
| CR-D1/D14 `d_dav_work_` overlap-into-offset (-opt) | impl:285–290 (sizing includes both `b·dim` AND `max_sub·b` terms via concurrent-region max), impl:1586–1597 (overlap GEMM lands at `d_dav_work_+n_new*dim`, NOT `d_dav_work2_`); eigvecs in `d_dav_work2_` preserved across orthogonalize → restart at impl:1641–1644 | **INTACT** |
| WW fused-MPO precompute charter | -gpu/-opt: host build at set_mpo time only, no per-sweep host roundtrip; -base: same pattern documented in header | **INTACT** |
| PhaseTimer panels (R17 split) | -gpu: 5 panels init at impl:235–239; -opt: 6 panels init at impl:1880–1885; all begin/end sites paired (verified below) | **INTACT** |
| D6 shared kernels — no duplicate definitions | `grep '__global__.*setup_batch_ptrs' dmrg2_gpu_impl.h dmrg2_gpu_opt_impl.h` returns 0 | **INTACT** |

## Special technique-G focus: t_absorb_ pairing across R/L branches

Per R20 brief: verify `t_absorb_` panel begin/end is correctly paired
across both `direction=='R'` and `direction=='L'` branches with no
early-return path leaking an open `.begin` without matching `.end`.

**dmrg2-gpu svd_split** (impl:1164–1336):

- Top: `t_svd_.begin` at impl:1166.
- Bottom: `t_absorb_.end` at impl:1335.
- Mid-pivot inside if/else fork:
  - `direction=='R'` branch: `t_svd_.end` at impl:1296, `t_absorb_.begin`
    at impl:1297.
  - `direction=='L'` branch: `t_svd_.end` at impl:1323, `t_absorb_.begin`
    at impl:1324.
- Branches are mutually exclusive (`if`/`else`), exactly one fires per
  call.
- `awk '/return\b/' impl.h NR=1166..1336` returns **empty** — no early
  returns. Per call: 1 `t_svd_.begin` + 1 `t_svd_.end` + 1
  `t_absorb_.begin` + 1 `t_absorb_.end`. **Balanced.**

**dmrg2-gpu-opt svd_split_fallback** (impl:1202–1406):

- Top: `t_svd_.begin` at impl:1204.
- Bottom: `t_absorb_.end` at impl:1405.
- Mid-pivot inside if/else fork:
  - `direction=='R'` branch: `t_svd_.end` at impl:1358, `t_absorb_.begin`
    at impl:1359.
  - `direction=='L'` branch: `t_svd_.end` at impl:1390, `t_absorb_.begin`
    at impl:1391.
- Mutually exclusive `if`/`else`. The CPU-SVD opt-in branch (impl:1225–
  1251) and RSVD branch (impl:1265–1329) execute INSIDE the `t_svd_`
  region BEFORE the pivot — they are not separate panels.
- `awk '/return\b/' impl.h NR=1204..1406` returns **empty** — no early
  returns; the lone `throw` at impl:1241 (CPU-SVD info-error) is a
  panic-path, not a normal exit. **Balanced.**

## All other PhaseTimer panels

- **t_apply_heff_** (-gpu impl:608/632/779; -opt impl:661/680/821):
  single begin at function top; two mutually-exclusive ends
  (graph-cache-hit early `return` AND function bottom). **Balanced.**
- **t_env_update_** (-gpu impl:792/859, 869/933; -opt impl:837/895,
  909/964): linear flow, no early returns. **Balanced.**
- **t_lanczos_** (-gpu impl:972/1156; -opt impl:1006/1190): single
  begin/end pair; lone `return energy` is the post-`.end` function exit.
  **Balanced.**
- **t_davidson_** (-opt impl:1431, ends at 1514/1546/1579/1630/1688):
  begin AFTER `dim ≤ 2*b` Lanczos short-circuit (impl:1424–1426); five
  exits cover (info-error, residual-converge, n_new=0, n_good=0,
  max_iter fall-through). The restart-without-exit branch at impl:1672
  uses `continue` and correctly does NOT call `.end` (loop body doesn't
  span begin/end). **Balanced.**

## Sibling propagation cross-check (technique G)

| Fix | dmrg2-gpu-base | dmrg2-gpu | dmrg2-gpu-opt |
|---|---|---|---|
| D5 device-kernel batch-ptr setup | immune (no batched GEMM) | applied | applied |
| D6 shared kernel header | immune | applied | applied |
| D12 device-ptr Lanczos | immune (no Lanczos device-mode in -base — naive tier) | (uses host scalars only at init; per-iter clean) | applied |
| CR-D1/D14 dav_work sizing | immune (no Davidson) | immune (no Davidson) | applied |
| D9/D15 PhaseTimer panels (R17 split + R12 instrumentation) | immune (no PhaseTimer per charter) | applied | applied |
| cafd628 D13 widened (Step 3 batched port) | immune (naive single-GEMM tier) | applied (full-batched collapse) | applied (3-branch device-side) |

**0 MISSING entries.**

## Carry-forward re-verification

- Round-6 dual-stream env-pipeline + direction-L MPS-write reorder:
  -gpu impl:1312–1326 (Vh→event→absorb), -opt impl:1376–1391
  (Vh→event→absorb). **INTACT.**
- D_PAD precompute bounds: -opt impl:286–288 dav_work_sz uses
  `theta_size_max_` which is computed from `D_use*dd*dd*D_use`
  with `D_use = D_mpo_` (post-pad). **INTACT.**
- H1-base apply_heff_two_site naive single-GEMM-per-pair pattern (D13
  immune): -base impl:357–375 (Step 1), 398–415 (Step 3).
  Charter-conformant per the -base "naive first-pass tier" docstring.
  **OK.**

## CRITICALS

None.

## HIGHS

None.

## MEDIUMS

None.

## NITS

None.

## FALSE POSITIVES VERIFIED

- **dmrg2-gpu-opt RSVD per-call host `std::vector<Scalar> h_omega` +
  random fill** at impl:1271–1276. Off-default (`use_rsvd_=false`).
  Pre-existing.
- **dmrg2-gpu RSVD per-call host omega** at impl:1206–1211. Same — off
  default. Pre-existing.
- **dmrg2-gpu lanczos init host `&norm`/`inv_norm`** at impl:980–992.
  Pre-existing init-time pattern (also in dmrg-gpu); not on per-iter
  hot path. Out of scope.
- **D8 `lapack_gesvd` registry hits in dmrg2-gpu-opt** at impl:261
  (init-time workspace query, lwork=-1) and impl:1236 (inside
  `if (use_cpu_svd_)` opt-in fallback). Both off-default; correctly
  registered as known false positives in the registry.
- **D13 per-`(n,s1p,s2p)` GEMM in dmrg2-gpu-base apply_heff_two_site**
  at impl:398–415 — single dense GEMM per iteration, charter-
  conformant naive tier. NOT the D13 per-element scalar pattern.
- **dmrg2-gpu/-opt host fused-MPO precompute** at impl:541+ (-gpu) /
  impl:537–626 (-opt). Runs once at `set_mpo()`, not per sweep. R7-H5
  docstring correction explicitly notes "host build in every tier".
  Outside the no-host-roundtrips-per-sweep rule.

## SUMMARY

Round-20 second confidence re-run returns **0 critical, 0 high, 0
medium, 0 nits** for the dmrg2 family across all three tiers. The only
commit since R19 (`f650466`) lands review reports; no code changes
touch the dmrg2 family. R19 finding H19 (per-element host Step 3 in
pdmrg-multi-gpu) was widened to the dmrg2 apply_heff Step 3 audit per
brief — none of `dmrg2-gpu`, `dmrg2-gpu-opt`, or `dmrg2-gpu-base` fires
the D13 widened pattern: -gpu uses fully device-side
`setup_batch_ptrs_step3_twosite_full[_sparse]` → `gemm_batched` →
`gemv`-reduce; -opt uses three branches all GEMM-based (sparse with a
host nnz-loop dispatching one `gemm` per non-zero, strided-batched, and
naive nested loop); -base uses naive single-GEMM-per-`(n,s1p,s2p)`
which is charter-conformant for the documented "naive first-pass" tier
and is NOT the D13 per-element scalar pattern.

The R20-special technique-G focus on `t_absorb_` pairing across R/L
branches confirms perfect balance in BOTH `dmrg2-gpu::svd_split`
(impl:1166→1335) and `dmrg2-gpu-opt::svd_split_fallback` (impl:1204→
1405): top-level begin and bottom-level end bracket the function; the
mid-pivot replicates inside an `if`/`else` whose branches are mutually
exclusive; and a direct `awk` scan of both panels confirms zero `return`
statements between begin and end — no leak path. All other PhaseTimer
panels (`t_apply_heff_`, `t_env_update_`, `t_lanczos_`, `t_davidson_`)
re-verified balanced.

This is the tenth consecutive zero-finding sub-review for the dmrg2
family within charter. Defect-registry sweep clean across all 10
in-charter variants. All seven techniques completed (D N-A); all five
R15→R19 regression-watch items + the new cafd628 cross-check
independently traced. -base re-verified within charter.

Verdict: **READY** for MI300X G1 baseline allocation.

(Length: ~1,470 words.)
