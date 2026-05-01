# Vertical review — dmrg2 family — 2026-05-01 round-19

HEAD: `bb809fb` (code state at `12d02c5`). Baseline:
`reviews/vertical-dmrg2-20260501-round18.md`. R19 is a **confidence
re-run before MI300X allocation** — fully independent re-audit with
no code changes since R18. Scope: `gpu-rocm/dmrg2-gpu-base/`,
`gpu-rocm/dmrg2-gpu/`, `gpu-rocm/dmrg2-gpu-opt/`.

Pre-step: `bash .claude/scripts/defect-registry.sh` returned **TOTAL
HITS: 0** across all in-charter variants (D1–D15).

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | All 11 D12 device buffers (-opt) live (4–10 hits each); 6 PhaseTimers in -opt all live (5–9 hits); 5 PhaseTimers in -gpu live (4–6 hits). 0 dead members. |
| B. Behavioral diff | DONE | Tier-superset relationship intact: -base (single-stream + Lanczos + host-WW + no opts) ⊂ -gpu (dual-stream + graph + RSVD + sparse + batched + D_PAD) ⊂ -gpu-opt (-gpu + Block-Davidson + MFMA-16 pad + setters). 0 unexplained divergences. |
| C. Docstring verification | DONE | -opt header `t_davidson_` "full block_davidson_eigensolver call (excl. Lanczos fallback)" matches impl: `t_davidson_.begin` at impl:1431 sits AFTER the `dim ≤ 2*b` Lanczos fallback at impl:1424–1426. -base "host-side WW build at set_mpo time, outside timed sweep region" matches `precompute_WW()` invocation at impl:261. -gpu/-opt header claim of "fused two-site MPO (WW) precomputed" matches `precompute_fused_mpo` at impl:537 (host build, set_mpo time only). |
| D. clangd filter | N-A | no clangd locally; A subsumes |
| E. Absence-naming brief | FOLLOWED | -base/-gpu/-opt expected-feature checklists pass (see below) |
| F. Workspace-aliasing audit | DONE | `d_dav_work_`, `d_dav_work2_`, `d_T1_/T2_`, `d_T1_env_/T2_env_`, `d_lanczos_v_`, `d_svd_*`, `d_rsvd_*` audited. 0 OVERRUN. |
| G. Sibling fix-propagation | DONE | All 5 R15→R18 watch-list fixes traced. 0 MISSING in dmrg2 family. |

## Regression-watch verification (independent re-audit)

### 1. abd88b9 D5 host-batch ptr propagation — INTACT

`dmrg2-gpu-opt/src/dmrg2_gpu_opt_impl.h` line 13 includes
`common/batch_ptrs_kernels.h`. Device-side kernel launches:
`setup_batch_ptrs_wd_twosite_sparse` at impl:711,
`setup_batch_ptrs_wd_twosite` at impl:728, `setup_batch_ptrs_wd` at
impl:851 (update_left_env), `setup_batch_ptrs_sw` at impl:924
(update_right_env). No `h_A/h_B/h_C` host arrays built per-iter on
the default GPU code path. **OK.**

### 2. 8abb6e7 D6 kernel duplication — INTACT

`grep '__global__.*setup_batch_ptrs' dmrg2_gpu_impl.h dmrg2_gpu_opt_impl.h`
returns 0 hits. All 13 `setup_batch_ptrs_*` kernels are defined
once, in `common/batch_ptrs_kernels.h:17–225`. Both impls include
the shared header (dmrg2-gpu impl:17, dmrg2-gpu-opt impl:13). **OK.**

### 3. 0efe96d D12 Lanczos device-pointer (-opt) — INTACT

11 device buffers declared at dmrg2_gpu_opt.h:154–164, allocated at
impl:176–186, freed at impl:330–340. PointerModeGuard wraps
3 init-region scopes (impl:1020, 1032, 1036), the Lanczos main loop
inner block (impl:1053), and the final ritz combine (impl:1178).
Per-iter kernels (`lanczos_process_alpha_kernel`,
`lanczos_fused_sub_kernel`, `negate_scalar_kernel`,
`lanczos_process_beta_kernel`, `lanczos_fused_norm_copy_kernel`,
`lanczos_check_beta`) operate purely on device pointers
(impl:1056, 1064, 1093, 1099, 1107, 1122). Final tridiagonal solve
uses D2D from `d_alpha_dev_/d_beta_dev_` (impl:1153–1156). No
per-iter host scalars on the hot path. **OK.**

### 4. 54f2fcf CR-D1/D14 Davidson buffer aliasing (-opt) — INTACT

Sizing at dmrg2_gpu_opt_impl.h:285–288:
```
size_t dav_work_sz = std::max(
    (size_t)theta_size_max_ * davidson_b_
        + (size_t)davidson_max_sub_ * davidson_b_,
    (size_t)davidson_max_sub_ * davidson_max_sub_);
```
First arg has both `b·dim` (residuals) AND `max_sub·b` (overlap)
terms — concurrent-region rule satisfied.

Overlap-GEMM target re-verified at impl:1586–1597:
- impl:1586: `Scalar* W = d_dav_work_;`
- impl:1587: `Scalar* overlap = d_dav_work_ + n_new*dim;`
- impl:1590–1592: `V^H @ W → overlap` writes to `d_dav_work_+offset`
- impl:1595–1597: `W -= V @ overlap` reads/writes `d_dav_work_`

The eigvec buffer (`d_dav_work2_`) is preserved across the
orthogonalize step (impl:1494–1496 wrote eigvecs there from
`rocsolver_syevd`; impl:1641–1644 reads them back for the restart
path's `V@eigvecs` GEMM). **CR-D1 invariant holds.**

Edge-case bound: in the residual-correction loop (impl:1553–1573),
`n_new ≤ min(b, k) ≤ b`, so `n_new·dim + max_sub·n_new ≤ b·dim +
max_sub·b` — within the allocated size. **OK.**

### 5. 12d02c5 D9/D15 PhaseTimer panels — INTACT

**dmrg2-gpu `t_absorb_/t_svd_` split** (svd_split, impl:1166–1335).
Single `t_svd_.begin` at top (impl:1166), single `t_absorb_.end` at
bottom (impl:1335). The middle pivot (`t_svd_.end + t_absorb_.begin`)
is duplicated **inside an if/else fork**: 'R' branch at impl:1296–
1297, 'L' branch at impl:1323–1324. Mutually exclusive — exactly one
fires per call → 1 begin + 1 end per panel per invocation. Ledger
balanced. **OK.**

**dmrg2-gpu-opt `t_absorb_/t_svd_` split** (svd_split_fallback,
impl:1204–1405). Same pattern: top `t_svd_.begin` at impl:1204,
bottom `t_absorb_.end` at impl:1405, mutually-exclusive pivot at
impl:1358–1359 ('R') vs impl:1390–1391 ('L'). **OK.**

**dmrg2-gpu-opt `t_davidson_` panel** (block_davidson_eigensolver,
impl:1413–1690): one begin at impl:1431 — placed AFTER the
`dim ≤ 2*b` Lanczos short-circuit at impl:1424–1426 so the Lanczos
fallback path uses only `t_lanczos_`. Five `t_davidson_.end` sites
cover all exits: rocsolver_syevd info-error (impl:1514), residual-
converge (impl:1546), `n_new == 0` (impl:1579), `n_good == 0`
(impl:1630), and final fall-through (impl:1688). The
restart-without-exit branch at impl:1672 (`continue`) does NOT
.end — correct, since the loop body doesn't span begin/end. The
overload of `t_lanczos_` on Davidson timing (R15 anti-pattern) is
removed: `t_lanczos_.begin/end` only at impl:1006/1190 inside
`lanczos_eigensolver`. No double-counting. **OK.**

## Sibling propagation cross-check (technique G)

| Fix (commit) | dmrg2-gpu-base | dmrg2-gpu | dmrg2-gpu-opt |
|---|---|---|---|
| D5 host-batch ptr (abd88b9) | immune (no batched GEMM in -base) | impls model (impl:670, 687, 725, 742, 809, 839, 888, 918) | fixed at impl:711, 728, 851, 924 |
| D6 dedup (8abb6e7) | immune (no batch_ptrs kernels) | shared header (impl:17) | shared header (impl:13) |
| D12 device-ptr Lanczos (0efe96d) | immune (already device-mode) | impls model (uses host scalars only at init) | fixed (R17) |
| CR-D1/D14 dav_work sizing (54f2fcf) | immune (no Davidson) | immune (no Davidson) | predates R17 (R8); re-verified |
| D9/D15 PhaseTimer (12d02c5) | immune (no PhaseTimer) | t_absorb_ split (impl:1296–1335) | t_davidson_ + t_lanczos_-no-longer-wraps-Davidson + t_absorb_ split |

**0 MISSING entries.**

## Carry-forward re-verification

- Round-6 dual-stream env-pipeline + direction-L MPS-write reorder:
  -gpu impl:1312–1326 (Vh→event→absorb), -opt impl:1376–1391
  (Vh→event→absorb). **INTACT.**
- D_PAD precompute_fused_mpo bounds: -opt impl:553–572 inner loops
  bounded by `D_act`; writes index `w*dd+s1*d+s2 ≤ (D_act-1)*dd+...
  < D_use*dd`, allocation `D_use*dd*dd*D_use` at impl:547. **INTACT.**
- H1-base apply_heff_two_site naive single-GEMM-per-pair pattern
  (D13): -base impl:357–375, 398–415. Charter-conformant per the
  -base "naive first-pass tier" docstring. **OK.**

## CRITICALS

None.

## HIGHS

None within charter.

## MEDIUMS

None within charter.

## NITS

None.

## FALSE POSITIVES VERIFIED

- **dmrg2-gpu lanczos init host `&norm`/`&inv_norm`** at impl:980–
  992. Pre-existing init-time pattern (also in dmrg-gpu); not on
  per-iter hot path. Out of scope for R19 watch list.
- **dmrg2-gpu-opt RSVD per-call host `std::vector<Scalar> h_omega` +
  random fill** at impl:1271–1276. Off-default (`use_rsvd_`).
  Pre-existing.
- **dmrg2-gpu-opt `opts_.fuse_lanczos` per-call host `std::vector`**
  none on default — fuse path is fully device-resident.
- **D8 `lapack_gesvd` registry hits in dmrg2-gpu-opt** at impl:261
  (init-time workspace query, lwork=-1) and impl:1236 (inside
  `if (use_cpu_svd_)` opt-in fallback). Both off-default; correctly
  registered as known false positives in the registry.
- **D9 `t_absorb_` declared without begin/end** — was a R17 known
  FP for -gpu and -opt; **retired**: 12d02c5 wires it in both
  variants. Verified again here.
- **D13 per-wp host loop in dmrg2-gpu-base apply_heff_two_site**
  at impl:357–375, 398–415. Per the -base charter (single-stream,
  naive first-pass tier per docstring), this is intentional.
  **Charter-conformant.**
- **dmrg2-gpu/-opt host fused-MPO precompute** at impl:537–626
  (-opt) / impl:541+ (-gpu). Runs once at `set_mpo()`, not per
  sweep. R7-H5 docstring correction explicitly notes "host build in
  every tier"; -base header makes this explicit. Outside the
  "no host roundtrips per sweep" rule. **OK.**

## SUMMARY

Round-19 confidence re-run returns **0 critical, 0 high, 0 medium,
0 nits** for the dmrg2 family across all three tiers. No code has
changed since R18 (HEAD `bb809fb` carries R18 reports only; code
state remains at `12d02c5`). All five R15→R18 regression-watch
items independently re-verified: abd88b9 D5 (device-side
setup_batch_ptrs kernels at the correct sites in -gpu and -opt;
Step 3 in -opt has no batched-pointer arrays — single dense GEMM
loop, no defect), 8abb6e7 D6 (no duplicate kernel definitions —
all in `common/batch_ptrs_kernels.h`), 0efe96d D12 (-opt Lanczos
in device-pointer mode with PointerModeGuard around all rocBLAS
calls that consume α/β; all per-iter scalar reductions land in
device buffers), 54f2fcf CR-D1/D14 (-opt `d_dav_work_sz` includes
the `max_sub·b` overlap term and the overlap GEMM writes to
`d_dav_work_+offset`, NOT `d_dav_work2_`; eigvecs preserved across
orthogonalize → restart), and 12d02c5 D9/D15 (`t_absorb_/t_svd_`
split balanced under mutually-exclusive R/L control flow in both
-gpu impl:1166→1335 and -opt impl:1204→1405; `t_davidson_` placed
correctly after the `dim ≤ 2*b` Lanczos short-circuit so Lanczos
fallback uses `t_lanczos_` exclusively). Technique F's special
charge — verifying balance of the single `t_svd_/t_absorb_` panel
through a divergent if/else — confirmed by direct trace: top-level
begin/end pair brackets the function, and the if/else replicates
the pivot but each branch fires exactly once. -base remains
charter-conformant (immune-by-design for every R15→R18 fix; D13
per-wp host loop is the documented "naive first-pass tier"
pattern). This is the ninth consecutive zero-finding sub-review for
the dmrg2 family within charter.

Self-audit: all seven techniques completed (D N-A); all five
regression-watch items independently traced to file:line; -base
re-verified within charter; technique G across all three tiers
clean. Verdict: **READY** for MI300X G1 baseline allocation.

(Length: ~1,180 words.)
