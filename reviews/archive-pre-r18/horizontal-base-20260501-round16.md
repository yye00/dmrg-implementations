# Horizontal review — -base tier — round 16 (2026-05-01)

Scope: dmrg-gpu-base, dmrg2-gpu-base, pdmrg-gpu-base. Baseline:
round-15 horizontal-base report at `f40140d`. Four commits since
(`69da5b4`, `5355c06`, `abd88b9`, `187fddf`) — **all -opt scope**.

`git diff --stat f40140d..HEAD -- gpu-rocm/{dmrg,dmrg2,pdmrg}-gpu-base/`
returns empty. Zero -base files touched. This review is a
regression-watch + re-audit of the CPU-on-hot-path question listed
in the round-16 brief.

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | Pass-through from r15 (no member changes); 0 dead. |
| B. Behavioral diff | DONE | lanczos shape identical across 3 -base; apply_heff / update_*_env are pure rocBLAS-loop, no host pointer-table builds. |
| C. Docstring verification | DONE | r15 d_WW_ comment intact; pdmrg-base "accurate_svd_gpu at boundary" claim still matches code (line 1276). |
| D. clangd filter | N-A | clangd not invokable; A subsumes the unused-private-field channel. |
| E. Absence-naming brief | FOLLOWED | -gpu-tier features (GpuOpts, RSVD, D_PAD, graph-capture, sparse-MPO, batched Step-3) remain absent. |
| F. Workspace-aliasing audit | DONE | 5 shared scratch buffers re-checked; no buffer reslicing this batch, 0 OVERRUN. |
| G. Sibling fix-propagation | DONE | r15 H2-opt host-batch elimination is OPT-tier only; -base structurally immune (never built host pointer tables). |

## CRITICALS / HIGHS / MEDIUMS / NITS

(none net-new for -base this round)

## FALSE POSITIVES VERIFIED

- "pdmrg-gpu-base apply_heff_two_site uses unbatched CPU `for` loops
  → CPU-on-hot-path" (pdmrg_gpu_base_impl.h:447-494; mirror in
  dmrg2_gpu_base_impl.h:357-415): CPU-side LOOP ORCHESTRATION over
  async rocBLAS GEMMs. No host data motion, no host pointer-table
  build, no HtoD copy of pointer data, no host LAPACK. Per -base
  charter. `grep -rEn "(LAPACKE_|cblas_|dgesvd_|dsyevd_|zheevd_)"`
  on 3 -base trees → 0 hits.

- "pdmrg-gpu-base precompute_WW does host nested 6-loop"
  (pdmrg_gpu_base_impl.h:359-403): called ONCE from `set_mpo`
  (init-time). Per-sweep cost is apply_heff_two_site reading
  device-resident `d_WW_[site]`. Comment at :352-354 is explicit.

## Regression-watch verification

| Round | Fix | Status |
|---|---|---|
| 14 H1-base apply_heff scope | PointerModeGuard inside per-iter loop body, after apply_heff returns; pre-loop guard for nrm2/scal_real | INTACT — dmrg_gpu_base_impl.h:519-524+:540-588; dmrg2_gpu_base_impl.h:588-593+:609-651; pdmrg_gpu_base_impl.h:744-749+:767-809 |
| 14 H1-base grep gate | `grep -rn rocblas_set_pointer_mode` on 3 -base trees | INTACT — 0 matches |
| 13 M1-base-prop | PointerModeGuard adoption | INTACT — 3 sites each in dmrg/dmrg2/pdmrg-gpu-base lanczos |
| 13 M14-base-prop | set_quiet stubs gone | INTACT — `grep set_quiet` returns 0 |
| 12 J1 (Stoudenmire) | accurate_svd_gpu at pdmrg-gpu-base boundary | INTACT — :1276; ws.asvd alloc :168, release :222 |
| 12 M4-W | free-then-realloc on d_W_left_/d_W_right_/d_WW_ | INTACT — dmrg-base :245,249; dmrg2-base :243,247,306; pdmrg-base :343,347,398 |
| 14 NIT (dmrg2 d_WW_ comment) | host-built then HtoD wording | INTACT — dmrg2_gpu_base.h:83 |
| 9 MED-base-1 | dead d_svd_work_ in dmrg2-gpu-base removed | INTACT — `grep` returns 0 |
| 8 C-new1 | pdmrg-gpu-base R_env from canonical Vh swap | INTACT — d_Vh_canonical alloc :155, swap pattern :1336-1346 |

## Technique-F worksheet

| Buffer | Variant | Lifetimes | Required | Alloc | Verdict |
|---|---|---|---|---|---|
| d_T1_/d_T2_ | dmrg-base | sequential | D·d·χ² | theta_size_max_ | OK |
| d_T1_/d_T2_ | dmrg2-base | sequential | D·d²·χ² | matched | OK |
| d_svd_work | pdmrg-base ws | sequential (mutex by direction) | max(m·k,k·n) | matched | OK |
| d_svd_work_ | dmrg-base | sequential | svd_max_dim·χ_max | matched | OK |
| d_Vh_canonical | pdmrg-base ws | local to boundary | ≤ theta_size_max_ | theta_size_max_ | OK |

No Block-Davidson at -base ⇒ round-8 CR-D1 class structurally
absent. Pass-through from r15 (no reslicing this batch).

## Technique-G — fix propagation since r15

| Recent fix | Defect class | -base sibling? |
|---|---|---|
| H2-opt host-batch elimination (`5355c06`, `abd88b9`, `187fddf`) | Host `std::vector<Scalar*>` ptr table built per call, then HtoD'd | **Structurally immune.** -base apply_heff and env-update use `for` orchestration around single-call `Traits::gemm`; never built host pointer tables. Verified: 0 hits for `std::vector<Scalar*>` constructions or HtoD ptr copies in non-init code. |
| PhaseTimer panel (`69da5b4`) | Scope reporting | **Immune by charter** — no PhaseTimer at -base. |

No fix-class missing in -base.

## SUMMARY

The -base tier is unchanged since round-15 (zero files touched
across the four post-r15 commits, all -opt scope). All ten
prior-round regression-watch items remain INTACT. The round-16
brief's explicit CPU-on-hot-path search returns no hits beyond
init-time `set_mpo` / `precompute_WW` / `initialize_mps_random` /
`build_initial_environments`, which is correct by design. The
naive `for`-loop GEMM orchestration in apply_heff and env-update
is CPU control flow over async device GEMMs — no host data motion,
no host BLAS/LAPACK, no per-call HtoD pointer copies. The H2-opt
host-batch defect class fixed in r15 follow-up is structurally
absent from -base. No CRITICALs, no HIGHs, no MEDIUMs, no NITs
net-new this round. **Verdict: READY.**
