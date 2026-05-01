# Vertical review — pdmrg family — round 18 — 2026-05-01

HEAD: `12d02c5`. Baseline: `reviews/conformity-20260501-round15.md`.
Scope: `pdmrg-gpu-base`, `pdmrg-gpu`, `pdmrg-gpu-opt`, `pdmrg-multi-gpu`.

Pre-step: `bash .claude/scripts/defect-registry.sh` → 0 hits across all
in-charter variants (D5/D6/D7/D8/D9/D10/D11/D13/D14/D15).

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | t_absorb_ removed from -gpu/-gpu-opt/-multi headers + impls (R17). pdmrg-multi-gpu d_ones_D allocated/used/freed; live. |
| B. Behavioral diff | DONE | Step 3 multi-gpu now structurally matches dmrg2-gpu impl 720-766. Lanczos device-pointer mode parity across all 4 variants. |
| C. Docstring verification | DONE | -base J1 docstring still correct ("on-device Stoudenmire `accurate_svd_gpu` … J1 lock"). One stale comment in pdmrg-gpu-opt::apply_heff_single_site (R18-NIT-1). |
| D. clangd filter | N-A | clangd not invokable locally; technique A subsumes the dead-symbol case for this round. |
| E. Absence-naming brief | FOLLOWED | tier-checklist applied per command brief. |
| F. Workspace-aliasing audit | DONE | 4 buffers checked for R17 changes: pdmrg-gpu-opt `d_dav_work` (CR-D1 fix verified), pdmrg-multi-gpu `d_T1`/`d_T2`/`d_ones_D`. All OK. |
| G. Sibling fix-propagation | DONE | 5 R17 fix axes traced; all propagated correctly with one stale-comment NIT and one MEDIUM (multi-gpu/-base WW-precompute host loop, pre-existing). |

## Regression-watch list — every R17 fix verified

| Fix (commit) | Variant | Status | Evidence |
|---|---|---|---|
| `abd88b9` D5 host-batch ptr propagation | pdmrg-gpu-opt | fixed | impl: `setup_batch_ptrs_wd_twosite_linear` device-side at 2778; Step1/2/3 use kernels |
| `abd88b9` D5 host-batch ptr propagation | pdmrg-gpu | already fixed (round 16 D5/D6) | impl: `setup_batch_ptrs_wd_*` + `setup_heff_*` device-kernels |
| `abd88b9` D5 host-batch ptr propagation | pdmrg-gpu-base | immune | -base charter excludes batched-GEMM optimisations |
| `abd88b9` D5 host-batch ptr propagation | pdmrg-multi-gpu | already device-side | dev kernels at impl 26/39/52 + R17H4 batched Step 3 |
| `187fddf` D5 Step 3 fallback (chi<16, d>2) | pdmrg-gpu-opt | fixed | impl 2412-2429: device kernel `setup_batch_ptrs_step3` per wp, no host h_A/h_B/h_C |
| `8abb6e7` D6 kernel duplication | pdmrg-gpu / -opt | fixed | both `#include "../../common/batch_ptrs_kernels.h"` (impl 22 / 19) |
| `8abb6e7` D6 kernel duplication | pdmrg-multi-gpu | partial-fix; per-variant kernels remain | `setup_heff_ss_step3_ptrs`, `setup_lenv_step3_ptrs`, `setup_renv_step3_ptrs` are file-local in different TU than pdmrg-gpu's same-named ones; not an ODR violation but duplicates reusable logic. **Filed as R18-MED-1** below. |
| `0efe96d` D12 Lanczos host-stack | pdmrg-gpu-opt | already fixed (round 7) | impl 1855-1920: `ws.d_alpha_dev` / `ws.d_beta_dev` used inside `PointerModeGuard pm_guard(handles_[si], rocblas_pointer_mode_device)` |
| `0efe96d` D12 Lanczos host-stack | pdmrg-gpu | already fixed | impl 1338-1447: same pattern with `ws.d_alpha_dev` / `ws.d_beta_dev` |
| `0efe96d` D12 Lanczos host-stack | pdmrg-gpu-base | already fixed | impl 772-818 |
| `0efe96d` D12 Lanczos host-stack | pdmrg-multi-gpu | already fixed | impl 1245-1327 with `dev.d_alpha_dev` / `dev.d_beta_dev` |
| `54f2fcf` CR-D1 Davidson aliasing | pdmrg-gpu-opt | **fixed** | (a) impl 252-272: `dav_work_sz = max(theta_size_max_*davidson_b_ + davidson_max_sub_*davidson_b_, davidson_max_sub_²)` — adds the missing max_sub*b term; (b) impl 1675-1685: overlap GEMM writes to `ws.d_dav_work + n_new*dim`, not `ws.d_dav_work2` |
| `54f2fcf` CR-D1 Davidson aliasing | pdmrg-multi-gpu | **immune** | no `block_davidson` in pdmrg-multi-gpu — single Lanczos path only |
| `54f2fcf` CR-D1 Davidson aliasing | pdmrg-gpu / -base | immune | no `block_davidson` (Lanczos-only) |
| `12d02c5` t_davidson_ panel | pdmrg-gpu-opt | fixed | header 258 declares; impl 1502 begin (after tiny-dim Lanczos short-circuit), 5 return paths each end (1593, 1629, 1679, 1735, 1796); init/report at 3577/3596 |
| `12d02c5` apply_heff_single_site instrumented | pdmrg-gpu-opt | fixed | impl 2300 begin, 2432 end |
| `12d02c5` multi-gpu update_left/right_env + apply_heff_single_site instrumented | pdmrg-multi-gpu | fixed | impl 860/921 (LE), 935/992 (RE), 1684/1738 (SS apply_heff) |
| `12d02c5` multi-gpu Step 3 batched | pdmrg-multi-gpu | fixed | impl 813-845: `setup_batch_ptrs_step3_twosite_full` (shared kernel) + `gemm_batched` + `gemv` reduce — structurally matches dmrg2-gpu impl 720-766 |
| `12d02c5` t_absorb_ removed | pdmrg-gpu, pdmrg-gpu-opt, pdmrg-multi-gpu | fixed | grep `t_absorb_` across all 4 pdmrg variants → 0 hits |

All R17 fixes propagated correctly; no MISSING-in-sibling defect detected.

## Workspace-aliasing audit (technique F)

| Buffer | Variant | Regions | Lifetime | Required | Allocated | Verdict |
|---|---|---|---|---|---|---|
| `ws.d_dav_work` | pdmrg-gpu-opt | residuals W [0, n_new·dim) + overlap [n_new·dim, +max_sub·b); restart-mode reads `H_proj` (k²) sequentially | concurrent (W and overlap inside CGS); H_proj sequential later | max(theta·b + max_sub·b, max_sub²) | max(theta·b + max_sub·b, max_sub²) | OK (matches dmrg-gpu-opt sizing pattern) |
| `ws.d_dav_work2` | pdmrg-gpu-opt | Ritz eigvecs from `rocsolver_syevd` then V·eigvecs target | sequential within iter | max_sub² | max(theta·b + max_sub·b, max_sub²) ≥ max_sub² | OK (over-sized intentionally for symmetry with d_dav_work) |
| `dev.d_T1` | pdmrg-multi-gpu | Step 1 output (cL·cR·D·dd); Step 3 scratch (slice_stride·D = cL·dd·cR·D) | sequential (T1 dead after Step 2 read, reused as Step 3 scratch) | max(cL·cR·D·dd, slice_stride·D) = D·dd·chi_max² | t_max = D·dd·chi_max² | OK |
| `dev.d_T2` | pdmrg-multi-gpu | Step 2 output | single use | D·dd·chi_max² | t_max = D·dd·chi_max² | OK |
| `dev.d_ones_D` | pdmrg-multi-gpu | length-D vector of Traits::one(); read-only by Step 3 gemv | one-shot read | D | D_mpo_ | OK |

## CRITICALS

(none)

## HIGHS

(none)

## MEDIUMS

- **R18-MED-1: pdmrg-multi-gpu redefines 3 kernel names that exist in
  pdmrg-gpu** — `setup_heff_ss_step3_ptrs`, `setup_lenv_step3_ptrs`,
  `setup_renv_step3_ptrs` are defined file-local in
  `pdmrg-multi-gpu/src/pdmrg_multi_gpu_impl.h` (lines 26, 39, 52) AND
  in `pdmrg-gpu/src/pdmrg_gpu_impl.h` (lines 31, 95, 109). Different
  translation units, so technically not ODR-violating, but the
  signatures and bodies diverge slightly (multi-gpu's lenv/renv are
  multi-device-aware variants). The shared header
  `common/batch_ptrs_kernels.h` is the natural home, with optional
  parameterization for multi-device. Defect class is the same as the
  D6 round-16 finding. Not strictly hot-path-correctness; deferred
  consolidation, not a regression.

- **R18-MED-2: pdmrg-multi-gpu `precompute_fused_mpo` runs WW build
  on host then H2D** (impl 700-730). The 6-deep nested loop is on
  host with hipMemcpy upload. -gpu and -gpu-opt run WW precompute on
  device per round-9 lock (paper-reference parity). This is a known
  pre-existing pattern, NOT introduced by R17 — and it's executed
  ONCE per `set_mpo` (not per sweep), so it does not violate
  "no host roundtrips per sweep." Filed for the multi-gpu charter
  question: should multi-gpu inherit -gpu's on-device precompute?

## NITS

- **R18-NIT-1: pdmrg-gpu-opt apply_heff_single_site comment is stale**
  (impl 2302-2311). The comment says "Step 1/Step 3 else-branches
  (and the sparse path) use stack-allocated Scalar* h_A[256], h_B[256],
  h_C[256] plus hipMemcpyAsync H2D inside the capture window" — but
  round-16 / round-15 fixes replaced those with `setup_batch_ptrs_*`
  device kernels (verified at impl 2340-2370 for Step 1, 2384-2429
  for Step 3). The graph-capture-disabled rationale is now obsolete;
  graph caching could be re-enabled. Comment should be revised to
  reflect post-fix state. Cosmetic only; defect-registry-clean.

## FALSE POSITIVES VERIFIED

- **pdmrg-gpu-opt rsvd_split missing t_svd_ instrumentation** — same
  pattern in pdmrg-gpu's rsvd_split (no t_svd_ inside RSVD path).
  Sibling-consistent; not a propagation gap. RSVD path falls through
  to `svd_split` (which does have t_svd_) for small matrices; the
  unique RSVD path leaves a measurement hole but that's a sibling-
  matched design choice, not a defect surfaced by R17.

- **pdmrg-multi-gpu absorb-not-instrumented** (round-15 reported) —
  R17 explicitly removed `t_absorb_` from -gpu/-gpu-opt/-multi
  citing "intermixed with SVD post-processing across full GPU /
  RSVD / CPU fallback paths." Documented in headers; not a defect.

## SUMMARY

Round-18 vertical-pdmrg is **clean** — 0 CRITICAL, 0 HIGH, 2 MEDIUM,
1 NIT. All R17 changes verified propagated across all four pdmrg
variants on every applicable axis. The CR-D1 propagation (R17's
headline critical fix) is verified two ways: (a) the buffer sizing
formula in `allocate_stream_workspaces` matches dmrg-gpu-opt's; (b)
the overlap GEMM writes to `ws.d_dav_work + n_new*dim` (verified
at impl 1684 / 1690), not to `ws.d_dav_work2` which still holds
the eigenvectors needed by the restart path. pdmrg-multi-gpu was
never affected by CR-D1 (no block_davidson).

The pdmrg-multi-gpu Step 3 R17H4 port is correct and matches the
dmrg2-gpu reference: the kernel signature
`setup_batch_ptrs_step3_twosite_full(A, B, C, base_A, base_B,
base_C_scratch, d, dd, strideA=cL*cR, strideB=cR, strideC_tile=cL,
slice_stride=cL*dd*cR)` is invoked with the same parameters as
dmrg2-gpu's call site. `dev.d_ones_D` is allocated (length D,
filled with Traits::one() via std::vector + hipMemcpy at init time),
freed in `free_gpu_resources`, and used read-only by the gemv
reduce. T1 reuse for the per-n scratch is safe — T1 is dead after
Step 2 reads it. Sized at `t_max = D·dd·chi_max²`, which equals
`slice_stride·D` exactly.

Defect-registry sweep at HEAD = 0 hits across all 14 defect classes
in scope. The two MEDIUMs are deferred-consolidation work
(kernel-share / multi-gpu charter), not regressions.

**Verdict**: pdmrg family READY for next conformity-orchestrator
tick. No work required from this sub-review before the next major
event.
