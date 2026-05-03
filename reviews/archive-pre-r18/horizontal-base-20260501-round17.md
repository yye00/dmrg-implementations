# Horizontal review — -base tier — round 17 (2026-05-01)

Scope: `dmrg-gpu-base`, `dmrg2-gpu-base`, `pdmrg-gpu-base`. Baseline:
round-16 horizontal-base report at `f40140d`. Two commits since
(`8abb6e7` registry+D6, `0efe96d` D12 device-pointer Lanczos).

`git diff --stat f40140d..HEAD -- gpu-rocm/{dmrg,dmrg2,pdmrg}-gpu-base/`
returns empty. **Zero -base files touched in this batch.** This is a
regression-watch + an explicit charter ruling on the D13 registry hits.

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | Pass-through from r16 (no member changes); 0 dead. |
| B. Behavioral diff | DONE | apply_heff / update_*_env all 3 -base remain pure rocBLAS-loop scaffolding; no host pointer-table builds. |
| C. Docstring verification | DONE | "no RSVD / no GpuOpts / no D_PAD / no graph capture" claims in 3 -base headers all match code (greps return 0). |
| D. clangd filter | N-A | clangd not invokable; A subsumes the unused-private-field channel. |
| E. Absence-naming brief | FOLLOWED | -gpu-tier features absent: GpuOpts, RSVD, D_PAD, graph-capture, sparse-MPO, batched Step-3, lanczos_graph, Block-Davidson. |
| F. Workspace-aliasing audit | DONE | 5 shared scratch buffers; no buffer reslicing this batch (no -base edits). 0 OVERRUN. |
| G. Sibling fix-propagation | DONE | r17 D6+D12 fixes are -opt scope; -base structurally immune. |

## CRITICALS / HIGHS / MEDIUMS / NITS

(none net-new for -base this round)

## D13 charter ruling (brief's open question)

**Verdict: D13 hits in the -base tier are CHARTER-ACCEPTABLE.**

Per-`wp` `for`-loop scaffolding around `Traits::gemm` in Step 3 of
`apply_heff` (and the analogous Step 1 patterns in
`update_left_env` / `update_right_env`) is CPU control flow over
asynchronous, stream-resident rocBLAS calls — NOT a host roundtrip.

Locations:
- `dmrg_gpu_base_impl.h:303-318` (Step 3); :342-355 / :377-389 (env)
- `dmrg2_gpu_base_impl.h:238-...` (analogous fused-MPO path)
- `pdmrg_gpu_base_impl.h:545-560` (Step 3); :584-598 / :630-644 (env)

What "no host roundtrips per sweep" (memory entry 2026-04-27)
prohibits: per-sweep/per-bond host LAPACK or BLAS, host data motion,
HtoD-of-pointer-table per call. None of the three -base variants
exhibit any of these on the hot path:

- `grep -rEn "(LAPACKE_|cblas_|dgesvd_|dsyevd_|zheevd_|std::vector<Scalar\*>)"`
  on the three -base trees → 0 hits in non-test code.
- `hipMemcpy*HostToDevice` only appears in `set_mpo` /
  `precompute_WW` / init paths (correct by design, init-time).
- No `hipStreamSynchronize` inside the per-`wp` loop body.

The R3-F1 batched-collapse is an explicit `-gpu` / `-opt`
optimization; the -base header docstrings (each variant's
`*_base.h`) name the omissions: "no RSVD, no batched GEMM, no
GpuOpts, no D_PAD." Adopting batched Step-3 in -base would
contradict the charter and erase the cross-tier delta the paper
relies on for the ablation argument.

Recommendation: D13 stays a registry sentinel for -opt drift
(catch any -opt regression that re-introduces unbatched Step 3),
but is **not actionable at -base**. No defect.

## FALSE POSITIVES VERIFIED

- D13 "Step 3 per-element host loop in apply_heff" (registry hits
  in 3 -base): see ruling above. CPU loop orchestration only.
- "pdmrg-gpu-base apply_heff_two_site / precompute_WW unbatched
  CPU `for` loops" (carried from r16): same ruling. Init-time
  precompute is `set_mpo`-scoped; sweep-time scaffolding is
  CPU control flow over async device GEMMs.

## Regression-watch verification (since `f40140d`)

| Round | Fix | Status |
|---|---|---|
| 14 H1-base apply_heff scope | PointerModeGuard inside per-iter loop after apply_heff | INTACT — dmrg_gpu_base_impl.h:519+:540+:629; dmrg2_gpu_base_impl.h:588+:609+:686; pdmrg_gpu_base_impl.h:744+:767+:851 |
| 13 M1-base-prop | PointerModeGuard adopted in 3 -base lanczos | INTACT — 3 sites each (verified above) |
| 13 M14-base-prop | dead `set_quiet` stubs gone | INTACT — `grep set_quiet` on 3 -base trees → 0 |
| 12 J1 (Stoudenmire) | `accurate_svd_gpu` at pdmrg-gpu-base boundary | INTACT — :1276; ws.asvd alloc :168, release :222 |
| 12 M4-W | free-then-realloc on `d_W_left_/d_W_right_/d_WW_` | INTACT — dmrg-base :245+:249; dmrg2-base :243+:247+:306; pdmrg-base :343+:347+:398 |
| 8 C-new1 | pdmrg-gpu-base R_env from canonical Vh swap | INTACT — d_Vh_canonical alloc :155, swap pattern :1336-1346 |
| 9 MED-base-1 | dead `d_svd_work_` removed in dmrg2-gpu-base | INTACT — `grep` returns 0 |

## Technique-F worksheet

| Buffer | Variant | Lifetimes | Required | Alloc | Verdict |
|---|---|---|---|---|---|
| d_T1_/d_T2_ | dmrg-base | sequential | D·d·χ² | t_max ≥ theta_size_max | OK |
| d_T1_/d_T2_ | dmrg2-base | sequential | D·d²·χ² | matched | OK |
| ws.d_T1/d_T2 | pdmrg-base (per-segment) | sequential within segment | D·d·χ² | matched | OK |
| d_svd_A_/d_svd_work_ | dmrg-base | sequential | theta_size_max + svd_max_dim·χ_max | matched | OK |
| d_Vh_canonical | pdmrg-base | local to boundary | ≤ theta_size_max_ | theta_size_max_ | OK |

No Block-Davidson at -base ⇒ round-8 CR-D1 class (dav_work overrun)
structurally absent. No buffer reslicing this batch (no -base edits).

## Technique-G — fix propagation since r16

| Recent fix | Defect class | -base sibling? |
|---|---|---|
| `8abb6e7` D6 (duplicate batch_ptrs kernels) | Duplicate kernel definitions across -opt variants | **Structurally immune** — -base has no batch_ptrs kernels. |
| `0efe96d` D12 (Lanczos device-pointer mode) | Host-pointer mode dot/nrm2 in -opt Lanczos | **Already correct at -base** — round-14 H1-base scope fix put PointerModeGuard inside per-iter loop with rocblas_pointer_mode_device for all 3 -base lanczos. Verified: 9 PointerModeGuard sites across 3 -base, all `rocblas_pointer_mode_device`. |
| (registry only) D2 / D8 / D9 / D10 / D11 / D13 | various | -base hits enumerated; D2/D8/D9/D10/D11 have 0 -base hits; D13 -base hits ruled charter-acceptable above. |

No fix-class missing in -base.

## SUMMARY

The -base tier is unchanged since round-16 (zero files touched in
the two post-r16 commits, both -opt scope). All seven prior
regression-watch items remain INTACT. The brief's open question on
D13 is answered: per-`wp` CPU `for`-loop orchestration around async
rocBLAS GEMMs in -base apply_heff Step 3 (and analogous env-update
sites) is CPU control flow, not a host roundtrip — no host BLAS,
no host data motion, no per-call HtoD pointer table. The R3-F1
batched-collapse is an explicit -gpu/-opt feature whose absence
defines the -base charter. D13 stays a registry sentinel for -opt
drift but is **not actionable at -base**. No CRITICALs, no HIGHs,
no MEDIUMs, no NITs net-new this round. **Verdict: READY.**
