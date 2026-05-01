# Horizontal review — -opt tier — round 19 (2026-05-01)

Baseline: `bb809fb` (HEAD); code state at `12d02c5`. Last conformity:
`reviews/conformity-20260501-round18.md`. R18 sub-baseline:
`reviews/horizontal-review-opt-20260501-round18.md`. Charter:
`gpu-rocm/{dmrg-gpu-opt, dmrg2-gpu-opt, pdmrg-gpu-opt}` + `gpu-rocm/common/`.

This is a **confidence re-run** before MI300X G1 allocation. R18 reported
0 CRITICALs and 1 carry HIGH on `set_use_batched_sweep(true)` (default OFF).
All R17 -opt landings (D5/D6/D12/CR-D1-pdmrg/PhaseTimer panels) are
re-traced with techniques A–G.

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | All 11 D12 device buffers live in dmrg/dmrg2-opt (64 / 58 references); pdmrg-opt worker-pool members (`n_workers_`, `worker_streams_`, `worker_handles_`, `worker_done_events_`, `step_done_events_`) referenced 33× in impl; `use_chebyshev_`/`use_batched_sweep_` consulted at dispatch |
| B. Behavioral diff | DONE | Davidson tiny-fallback + Rayleigh-Ritz identical in all 3; dmrg/dmrg2-opt main Lanczos in device pointer mode (matches pdmrg-opt main Lanczos); pdmrg-opt `batched_lanczos_eigensolver` (opt-in path) still host-mode (carried HIGH) |
| C. Docstring verification | DONE | Stale comment at `pdmrg_gpu_opt_impl.h:2302-2311` (M-opt-pdmrg-single-site-graph-comment-stale) carries forward — code uses device kernels but comment still claims `h_A[256]/h_B[256]/h_C[256]` stack pointers |
| D. clangd filter | N-A | No ROCm headers locally; A subsumes |
| E. Absence-naming brief | FOLLOWED | pad_mfma16 (3/3), chi_max_user_ (3/3), J2 superset (env_update_pending_ in dmrg/dmrg2-opt = 10/10; pdmrg-opt uses per-segment streams), public setter API (cpu_svd, use_davidson, rsvd in all; use_batched_sweep + use_chebyshev pdmrg-only), Chebyshev (pdmrg-only) all PRESENT |
| F. Workspace-aliasing audit | DONE | `d_dav_work*` sizing in all 3 -opt verified `EXACT MATCH` to required = max(theta_max·b + max_sub·b, max_sub²); `d_T1/d_T2` per-call sequential, sized at `t_max` |
| G. Sibling fix-propagation | DONE | D5/D6/D12/CR-D1 (R8 + R17)/PhaseTimer (R17) all traced; zero regressions |

Pre-step `bash .claude/scripts/defect-registry.sh` returned **TOTAL HITS: 0**
across all 14 tracked defect classes (D1–D15, D6 collapsed) at HEAD.

## Regression watch since R15 baseline

| Item (R17 commit / earlier round) | Status |
|---|---|
| `abd88b9` D5 host-batch ptr propagation — dmrg2/pdmrg-opt | OK — registry returns 0 D5 hits; sole H2D in pdmrg-opt is the cold random-vector init at line 1830 |
| `187fddf` D5 follow-up in pdmrg-opt Step 3 fallback | OK — not regressed; shared kernel path used |
| `8abb6e7` D6 shared `common/batch_ptrs_kernels.h` — all 3 -opt | OK — each impl includes shared header (`dmrg_gpu_opt_impl.h:12`, `dmrg2_gpu_opt_impl.h:13`, `pdmrg_gpu_opt_impl.h:19`); ZERO local `__global__ void setup_batch_ptrs_*` redefinitions in any of the three impls |
| `0efe96d` D12 Lanczos device pointer mode — dmrg/dmrg2-opt | OK — all 11 buffers (`d_dot_result_`, `d_nrm2_result_`, `d_neg_alpha_`, `d_neg_overlap_`, `d_inv_nrm_`, `d_alpha_dev_`, `d_beta_dev_`, `d_neg_beta_scalars_`, `d_const_one_`, `d_const_zero_`, `d_const_neg_one_`) live; PointerModeGuard wraps Lanczos main loop (`dmrg_gpu_opt_impl.h:1011-1188`, `dmrg2_gpu_opt_impl.h:1020-1178`); all 4 per-iter kernels (`lanczos_process_alpha_kernel`, `lanczos_process_beta_kernel`, `negate_scalar_kernel`, `invert_nrm_kernel`) launched |
| pdmrg-opt parallel-segment Lanczos | OK — uses `workspaces_[si].d_alpha_dev`/`d_dot_result`/`d_neg_alpha` pattern; PointerModeGuard at `pdmrg_gpu_opt_impl.h:1820, 1836, 1855, 1991`; not regressed |
| `54f2fcf` CR-D1-pdmrg fix — pdmrg-opt | OK — `dav_work_sz = max(theta_size_max_·davidson_b_ + davidson_max_sub_·davidson_b_, davidson_max_sub_²)` at lines 266-269; overlap matrix at `ws.d_dav_work + (size_t)n_new*dim` (line 1688), eigvecs preserved at `ws.d_dav_work2` |
| CR-D1 in dmrg-gpu-opt + dmrg2-gpu-opt (R8) | OK — same sizing formula at `dmrg_gpu_opt_impl.h:301-305` and `dmrg2_gpu_opt_impl.h:285-289`; offset overlap at `dmrg_gpu_opt_impl.h:1665-1666` and `dmrg2_gpu_opt_impl.h:1586-1587`; unchanged |
| `12d02c5` D9/D15 PhaseTimer panels | OK — `t_davidson_` panel begin/end on every return path in all 3 (5 sites in dmrg/dmrg2-opt at 1492/1584/1624/1657/1709/1768 and 1431/1514/1546/1579/1630/1688; 5 sites in pdmrg-opt at 1502/1592/1628/1678/1734/1795); `t_absorb_` instrumented in dmrg-opt (1385-1413, 1438-1465) and dmrg2-opt (1359, 1391-1405); `t_apply_heff_` instrumented in pdmrg-opt single-site (2300, 2432) and two-site (809, 828, 973); `t_absorb_` correctly REMOVED from pdmrg-opt header + impl (zero hits) |
| J1 Stoudenmire lock — pdmrg-opt | OK — `accurate_svd_gpu` invoked at `pdmrg_gpu_opt_impl.h:3351` |
| J2 superset of -gpu sibling | OK — env_update_pending_ counts equal between -gpu (10/10) and -opt (10/10) for dmrg/dmrg2; pad_mfma16 + chi_max_user_ in all 3; public-setter parity preserved; pdmrg-opt extras (set_use_batched_sweep, set_use_chebyshev) OFF by default at lines 194-195 |

## Technique-F detail: pdmrg-opt block_davidson_eigensolver post-CR-D1

Brief asked for re-verification of the post-`54f2fcf` aliasing.

| Region | Offset | Max size (Scalars) | Lifetime |
|---|---|---|---|
| W (residuals) | `ws.d_dav_work + 0` | `n_new × dim` ≤ `b × theta_size_max` | live across orthogonalization GEMMs (lines 1691, 1696) |
| overlap matrix | `ws.d_dav_work + n_new*dim` | `k × n_new` ≤ `max_sub × b` | live concurrently with W in same GEMM pair |
| restart X_keep | `ws.d_dav_work + 0` | `dim × keep` ≤ `theta_size_max × b` | restart path only; W+overlap dead by then |
| `ws.d_dav_work2` | offset 0 | `k × k` ≤ `max_sub × max_sub` (eigvecs) | live across both GEMM pair AND restart path |

Required `dav_work_sz` ≥ max(b·theta_max + max_sub·b, max_sub²).
Allocated (line 266-269) = same formula → **EXACT MATCH**. OK.
`davidson_max_sub_ = min(davidson_b_·8, theta_size_max_)` (line 189) ⇒
`n_new ≤ b ≤ max_sub`, so `n_new·dim ≤ b·theta_size_max ≤ allocated`. OK.
`ws.d_dav_work2` only ever holds the (k×k) eigvec matrix — never aliased
with W. OK.

Sibling sizing in dmrg-gpu-opt (lines 301-305, 1665-1666) and dmrg2-gpu-opt
(lines 285-289, 1586-1587) is identical and unchanged since R8.

## CRITICALS — block GPU run / paper submission

**None.**

## HIGHS — fix before next major event

- **H-opt-batched-lanczos-host-mode (carried R17/R18)** —
  `pdmrg-gpu-opt/src/pdmrg_gpu_opt_impl.h:2870-2980` —
  `batched_lanczos_eigensolver` retains host-resident `h_alpha`,
  `h_beta`, `&h_dot`, `&beta_val`, `&inv_norm` (per-iter `Traits::dot`,
  `Traits::nrm2` in host mode at lines 2940, 2979). Triggered only by
  `set_use_batched_sweep(true)`; default OFF (line 194). Same fix shape
  as D12 (PointerModeGuard + `lanczos_process_*_kernel`). Not regressed
  vs R18 — explicitly retained as deferred. **Not blocking for G1
  baseline** (default-OFF code path; benchmark scripts must not pass
  `--batched-sweep` or call `set_use_batched_sweep(true)`).

  Note: `chebyshev_eigensolver` at lines 2017-2028 also uses host
  `h_alpha`/`h_beta` for its truncated-Lanczos spectral-bounds estimator
  — but `use_chebyshev_` is also default OFF (line 195). Same opt-in
  shape as the batched HIGH; not separately escalated.

## MEDIUMS — fix when convenient

- **M-opt-pdmrg-single-site-graph-comment-stale (carried R18, NEW R18)** —
  `pdmrg-gpu-opt/src/pdmrg_gpu_opt_impl.h:2302-2311` — the comment block
  still claims "Step 1/Step 3 else-branches (and the sparse path) use
  stack-allocated `Scalar* h_A[256]`, `h_B[256]`, `h_C[256]` plus
  `hipMemcpyAsync` H2D inside the capture window" and gives this as the
  reason LANCZOS_GRAPH is disabled for single-site. Code at lines 2340,
  2356, 2416 uses `setup_batch_ptrs_wd_sparse`, `setup_batch_ptrs_wd`,
  `setup_batch_ptrs_step3` device kernels — the host-stack pointers
  are gone. Either (a) update the comment to reflect the device-kernel
  port and re-test single-site graph capture, or (b) document a
  different reason for keeping single-site uncaptured. As-is, this
  conceals an opportunity to re-enable graph caching for warmup/polish
  passes. Not blocking.

- **M-opt-pdmrg-stale-svd-port-comment (carried R17/R18, NIT-grade)** —
  `pdmrg-gpu-opt/src/pdmrg_gpu_opt_impl.h:1601-1602` — "the host
  h_dav_eigvecs upload is no longer needed" reads as a transitional
  note; `h_dav_eigvecs_` was removed from the header when
  `rocsolver_syevd` landed. Cosmetic; same NIT carried from R17.

## NITS — cosmetic

None new.

## FALSE POSITIVES VERIFIED

- **D5/D6/D12 grep on bare names** initially returns 0 hits because
  member fields carry trailing underscores. Grep with the trailing
  underscore confirms all D12 buffers are live (64 in dmrg-opt, 58 in
  dmrg2-opt) and the shared D6 header is included exactly once per
  impl with zero local kernel duplicates.
- **`hipMemcpyHostToDevice` at `pdmrg_gpu_opt_impl.h:1830`** flags as a
  host roundtrip but is the random-vector init for the ‖θ‖ < 1e-14
  cold path, not an inner-loop H2D. Not a defect.
- **D8/D7 `lapack_gesvd` hits in pdmrg-opt** (registry detection) —
  TOTAL HITS: 0 across the registry's awk filter, confirming all
  remaining occurrences are workspace queries (`lwork = -1`) and
  `if (use_cpu_svd_)` opt-in branches.
- **`chebyshev_eigensolver` host h_alpha/h_beta** (lines 2028-2105)
  — opt-in via `set_use_chebyshev(true)` only; default OFF (line 195);
  not on G1 default path. Folded into the H carry note rather than
  escalated separately.

## Cross-impl 3-way diff — block_davidson_eigensolver

Same Rayleigh-Ritz pattern in all 3 -opt variants:
- random init b columns w/ CGS orthogonalization;
- AV per column via `apply_heff(_two_site)`;
- `Traits::gemm` H_proj = V^H · AV → `d_dav_work2` (size k×k);
- `rocsolver_syevd` in-place eigendecomp;
- D2H lowest eigenvalue + info for control flow only;
- residual `r_i = AV·eigvec - λ_i·V·eigvec` at `d_dav_work + n_new*dim`;
- orthogonalization with overlap PAST the residual region (CR-D1 fix);
- restart when `k + n_good > max_sub`: keep best b Ritz vectors using
  preserved eigvecs in `d_dav_work2`.

No structural divergences. pdmrg-opt has the additional `if (si == 0)`
gate on every PhaseTimer call (correct: `std::vector::push_back` not
thread-safe across parallel segments).

## SUMMARY

Round 19 confirms R18 verdict at the -opt tier. The R17 CR-D1-pdmrg
sizing fix (`54f2fcf`) is intact and matches the R8 dmrg/dmrg2-opt
sibling sizing (technique F: EXACT MATCH on the `max(theta_max·b +
max_sub·b, max_sub²)` formula in all 3 variants). R17 D12
device-pointer-mode Lanczos port is fully wired with all 11 buffers
live in both dmrg-gpu-opt and dmrg2-gpu-opt; pdmrg-opt main Lanczos
unregressed. R17 PhaseTimer panel propagation (`t_davidson_`,
`t_absorb_` split, `t_apply_heff_` in pdmrg single-site, `t_absorb_`
removed from pdmrg-opt) verified across all return paths. The shared
D6 `common/batch_ptrs_kernels.h` is included exactly once per -opt
impl with zero local kernel duplicates. Pre-step defect registry:
TOTAL HITS 0 across D1–D15. One HIGH carries forward
(H-opt-batched-lanczos-host-mode at `pdmrg_gpu_opt_impl.h:2870-2980`,
opt-in `set_use_batched_sweep(true)` only, default OFF — **not
blocking for the G1 baseline campaign**) and one MEDIUM persists
(stale graph-disable comment in pdmrg-opt single-site apply_heff).

Verdict: **READY** for the MI300X G1 baseline campaign. Ensure
benchmark drivers do not exercise `set_use_batched_sweep(true)` or
`set_use_chebyshev(true)` until the host-mode Lanczos in those paths
is ported.

CRITICAL: 0
HIGH: 1 (carried, default-OFF opt-in)
MEDIUM: 2 (1 carried-from-R18, 1 NIT-grade carried-from-R17)
NIT: 0 new
