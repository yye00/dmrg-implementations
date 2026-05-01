# Vertical review — dmrg2 family — 2026-05-01 round-18

HEAD: `12d02c5`. Baseline: round-17 at `0efe96d` /
`reviews/vertical-dmrg2-20260501-round17.md`. Scope:
`gpu-rocm/dmrg2-gpu-base/`, `gpu-rocm/dmrg2-gpu/`,
`gpu-rocm/dmrg2-gpu-opt/`. One commit lands between baseline and HEAD
that touches dmrg2: `12d02c5` (round-17 HIGHs — H1/H4 PhaseTimer
panel propagation across -opt variants and dmrg2-gpu absorb-split).

Pre-step: `bash .claude/scripts/defect-registry.sh` returned **0 hits**
across all in-charter variants (D7/D8/D9/D10/D11/D13/D14/D15).

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | new `t_davidson_` member (-opt) referenced 11× outside ctor/dtor; no dead members |
| B. Behavioral diff | DONE | t_svd_/t_absorb_ split now consistent across -gpu and -opt; t_davidson_ exclusive of t_lanczos_ |
| C. Docstring verification | DONE | header comment "full block_davidson_eigensolver call (excl. Lanczos fallback)" matches impl: `t_davidson_.begin` lives at impl:1431 AFTER the `dim < 32` Lanczos short-circuit at impl:1424 |
| D. clangd filter | N-A | no clangd locally; A subsumes |
| E. Absence-naming brief | FOLLOWED | -base/-gpu/-opt expected-feature checklists pass |
| F. Workspace-aliasing audit | DONE | CR-D1 sibling check on `d_dav_work_` confirmed; size term + write-target both correct |
| G. Sibling fix-propagation | DONE | 5 round-17 fixes traced; 0 MISSING in dmrg2 family |

## Regression-watch verification (round-17 → round-18)

### 1. abd88b9 D5 host-batch ptr propagation — INTACT

`dmrg2-gpu-opt/src/dmrg2_gpu_opt_impl.h` includes
`common/batch_ptrs_kernels.h` at impl:13. Device-side
`setup_batch_ptrs_*` kernels invoked at:
- apply_heff_two_site Step 1 dense: `setup_batch_ptrs_wd_twosite`
  at impl:728
- apply_heff_two_site Step 1 sparse:
  `setup_batch_ptrs_wd_twosite_sparse` at impl:711
- update_left_env Step 1: `setup_batch_ptrs_wd` at impl:851
- update_right_env Step 1: `setup_batch_ptrs_sw` at impl:924

Step 2 is a single dense GEMM (no batched-pointer arrays — no defect).
Step 3 is a host-side rocBLAS GEMM loop (no per-iter pointer-array
H2D); the brief item 187fddf explicitly noted "no analog in
dmrg2-gpu-opt" for the Step 3 fallback path. **Verified.**

### 2. 8abb6e7 D6 kernel duplication — INTACT

`grep '__global__.*setup_batch_ptrs' dmrg2_gpu_impl.h
dmrg2_gpu_opt_impl.h` returns 0 hits — no duplicate local copies.
Both files include `common/batch_ptrs_kernels.h` (dmrg2-gpu impl:17,
dmrg2-gpu-opt impl:13). **OK.**

### 3. 0efe96d D12 Lanczos host-stack scalars (-opt) — INTACT

11 device buffers declared at `dmrg2_gpu_opt.h:154-164`, allocated at
impl:176-186, freed at impl:330-340. PointerModeGuard wraps the
Lanczos main loop at impl:1053. Per-iter kernels
(`lanczos_process_alpha_kernel`, `lanczos_fused_sub_kernel`,
`negate_scalar_kernel`, `lanczos_process_beta_kernel`,
`lanczos_fused_norm_copy_kernel`, `lanczos_check_beta`) launched at
impl:1056, 1064, 1093, 1099, 1107, 1122. Final tridiagonal solve uses
D2D copy from `d_alpha_dev_`/`d_beta_dev_` (impl:1153-1158). **OK.**

### 4. 54f2fcf CR-D1/D14 Davidson buffer aliasing (-opt sibling) — INTACT

`d_dav_work_sz` allocation at `dmrg2_gpu_opt_impl.h:285-288`:
```
size_t dav_work_sz = std::max(
    (size_t)theta_size_max_ * davidson_b_
        + (size_t)davidson_max_sub_ * davidson_b_,
    (size_t)davidson_max_sub_ * davidson_max_sub_);
```
Has both `b·dim` (residuals) AND `max_sub·b` (overlap) terms. **OK.**

Overlap GEMM target verification: at impl:1586-1597 the orthogonalize
phase declares `Scalar* W = d_dav_work_; Scalar* overlap = d_dav_work_
+ n_new*dim`. Both V^H@W (impl:1590-1592) and W -= V@overlap
(impl:1595-1597) write to/read from `d_dav_work_` (with offset for
overlap), NOT `d_dav_work2_`. Eigvecs from rocsolver_syevd remain in
`d_dav_work2_` for the restart path's V@eigvecs GEMM at impl:1641-1644.
The round-8 CR-D1 invariant (residuals + overlap concurrent in
`d_dav_work_`, eigvecs preserved in `d_dav_work2_`) **holds**.

### 5. 12d02c5 D9/D15 PhaseTimer panels — INTACT

**dmrg2-gpu t_absorb_ split** (impl:1296-1335): direction-R branch
records `event_canon_ready_` (impl:1295), ends t_svd_ (impl:1296),
begins t_absorb_ (impl:1297), then runs scale_rows_by_diag_kernel.
Direction-L branch mirrors at impl:1322-1324. Both branches funnel
into the single `t_absorb_.end(stream_)` at impl:1335. The ledger now
has 1 begin + 1 end per `svd_split` invocation regardless of
direction. **OK.**

**dmrg2-gpu-opt t_davidson_** (impl:1431, 1514, 1546, 1579, 1630,
1688): one begin AFTER the `dim < 32` Lanczos short-circuit (impl:1424
returns `lanczos_eigensolver(...)` — that path uses `t_lanczos_`
exclusively). Five end sites cover all early-return branches +
final fall-through. The previous round-15 t_lanczos_ overload of
the Davidson outer loop is removed: `t_lanczos_.begin/end` now
appears ONLY inside `lanczos_eigensolver` at impl:1006/1190. No
double-counting. Header docstring at dmrg2_gpu_opt.h:248 ("full
block_davidson_eigensolver call (excl. Lanczos fallback)") matches
the impl. **OK.**

**dmrg2-gpu-opt absorb split** (impl:1358-1405): same pattern as
dmrg2-gpu, both 'R' and 'L' branches end t_svd_ + begin t_absorb_
around the canonical-ready event. **OK.**

## Sibling propagation cross-check (technique G)

| Fix (commit) | dmrg2-gpu-base | dmrg2-gpu | dmrg2-gpu-opt |
|---|---|---|---|
| D5 host-batch ptr (abd88b9) | immune (no batched GEMM in -base) | original model (impl:725, 742) | NEWLY FIXED at impl:711, 728, 851, 924 |
| D6 dedup (8abb6e7) | immune (no batch_ptrs kernels) | shared header included (impl:17) | shared header included (impl:13) |
| D12 device-pointer Lanczos (0efe96d) | immune (already device-mode) | original model | fixed (round-17) |
| CR-D1/D14 dav_work sizing (54f2fcf) | immune (no Davidson) | immune (no Davidson) | predates round-17 (round-8); re-verified |
| D9/D15 PhaseTimer (12d02c5) | immune (no PhaseTimer) | t_absorb_ split (impl:1296-1335) | t_davidson_ added + t_lanczos_ no longer wraps Davidson + t_absorb_ split |

**0 MISSING entries.**

## Round-13/-12/-8 carry-forward (re-verified)

- Round-6 dual-stream env-pipeline + direction-L MPS-write reorder:
  -opt impl:1376→1387→1396 (Vh→event→absorb), -gpu impl:1312→1322→1326
  (Vh→event→absorb). **INTACT.**
- D_PAD precompute_fused_mpo OOB fix: -opt impl:537-541, -gpu inner
  loops bound by `D_act` writing into `D_use` stride. **INTACT.**
- H1-base apply_heff scope (round-14): -base impl:608 inner guard
  scoped after apply_heff_two_site. **INTACT.**
- H2-opt host-batch elimination via `common/batch_ptrs_kernels.h`
  (round-15+abd88b9): all 4 sites in -opt + 2 in -gpu wired. **INTACT.**

## CRITICALS

None.

## HIGHS

None within charter.

## MEDIUMS

None within charter.

## NITS

None.

## FALSE POSITIVES VERIFIED

- **dmrg2-gpu lanczos initial v[0] uses host `&norm`/`&inv_norm`**
  at impl:980-992. Pre-existing; family-wide pattern (also in
  dmrg-gpu impl:912-917). Out of scope for round-18 watch list.
- **dmrg2-gpu-opt RSVD per-call host `std::vector<Scalar> h_omega` +
  random fill** at impl:1292-1297. Off-default (`use_rsvd_`).
  Pre-existing.
- **dmrg2-gpu-opt `opts_.fuse_lanczos` per-call hipMalloc/hipFree**
  at impl:993-995. Off-default (`DMRG_GPU_OPT_FUSE_LANCZOS` env-var).
  Pre-existing.
- **D8 `lapack_gesvd` registry hits in dmrg2-gpu-opt** at impl:261
  (init-time workspace query, one-shot at construction) and
  impl:1236 (inside `if (use_cpu_svd_)` opt-in fallback). Both
  off-default-path; correctly registered as known false positives.
- **D9 `t_absorb_` declared without begin/end** — was a known FP in
  round-17 for -gpu and -opt; **now retired**: 12d02c5 wires
  t_absorb_ in both variants. dmrg-gpu sibling also wired in the
  same commit (out-of-charter for this review).
- **D13 per-wp host loop in dmrg2-gpu-base apply_heff_two_site**
  at impl:357-375, 398-415. Per the -base charter (single-stream,
  naive first-pass tier), this is intentional. **Charter-conformant.**

## SUMMARY

Round-18 returns **0 critical, 0 high, 0 medium, 0 nits** for the
dmrg2 family across all three tiers. The single dmrg2-touching
commit in the round-17→18 window (`12d02c5`, H1+H4 PhaseTimer panel
propagation + dmrg2-gpu absorb-split) is correctly scoped and
sibling-consistent: dmrg2-gpu now has a live `t_absorb_` panel
(split from `t_svd_` around `event_canon_ready_` in both 'R' and
'L' branches of svd_split), dmrg2-gpu-opt gains a dedicated
`t_davidson_` panel covering only the Davidson outer loop, and the
round-15 anti-pattern of overloading `t_lanczos_` with Davidson
timing is retired. All 5 regression-watch items pass: abd88b9 D5
(device-side setup_batch_ptrs kernels), 187fddf (immune for -opt
Step 3), 8abb6e7 D6 (no duplicate kernels), 0efe96d D12 (-opt
Lanczos device-pointer mode), 54f2fcf CR-D1/D14 (-opt
`d_dav_work_sz` has the max_sub·b term and overlap GEMM writes to
`d_dav_work_+offset`, not `d_dav_work2_`), and 12d02c5 itself
(`t_absorb_`/`t_davidson_` panels active per header docstring).
Technique G traces all five fixes cleanly across the three tiers
with 0 MISSING entries; -base remains charter-conformant
(immune-by-design for every fix above except the inherent
single-stream/naive-GEMM tier-baseline patterns, which are
intentional). Workspace-aliasing audit (technique F) re-confirmed
the round-8 CR-D1 invariant on dmrg2-gpu-opt: residuals + overlap
share `d_dav_work_` (concurrent), eigvecs preserved in
`d_dav_work2_`, allocation has both terms. This is the eighth
consecutive zero-finding sub-review for the dmrg2 family within
charter. Block GPU run? **NO**, family is ready.

Self-audit: all seven techniques completed (D N-A); regression-watch
list explicitly traced for all six round-17/18 verification items
(abd88b9, 187fddf, 8abb6e7, 0efe96d, 54f2fcf, 12d02c5); -base brought
into scope and verified; technique G sibling propagation traced
across all three dmrg2 tiers and confirmed clean for every recent
fix. Verdict: **READY**.

(Length: ~1,090 words.)
