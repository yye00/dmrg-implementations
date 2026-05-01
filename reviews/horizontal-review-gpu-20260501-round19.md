# Horizontal review — -gpu tier — round 19 (2026-05-01)

Confidence re-run before MI300X allocation. HEAD `bb809fb` (R18 reports
landed; code state at `12d02c5`). Baseline: round-18
`reviews/conformity-20260501-round18.md`. No code changes since R18 —
only the R18 reports were committed (`bb809fb`). This is a fresh A-G
pass on the -gpu tier code as it stands at `12d02c5`.

**Pre-step**: `bash .claude/scripts/defect-registry.sh` → **TOTAL HITS: 0** (clean).

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | All 4 variants: every PhaseTimer member has begin/end pairs (counts below); no dead concurrency primitives detected. |
| B. Behavioral diff | DONE | 1 NEW divergence found in apply_heff_single_site Step 3 (pdmrg-multi-gpu vs pdmrg-gpu/dmrg-gpu) — see HIGH H19-multi-ss-step3 below. |
| C. Docstring verification | DONE | Panel comments at pdmrg_gpu.h:214 + pdmrg_multi_gpu.h:217 correctly document t_absorb_ removal. multi-gpu.h:24 comment for kernel doc consistent. |
| D. clangd filter | N-A | No ROCm headers on host. |
| E. Absence-naming brief | FOLLOWED | All 4 variants feature-complete vs the 11-item -gpu checklist. Multi-gpu still genuinely-out-of-charter on D_PAD/sparse_mpo/lanczos_graph (header :26). |
| F. Workspace-aliasing | DONE | T1 reuse Step1→Step3 in multi-gpu apply_heff_two_site verified safe (Step 2 reads T1 before Step 3 writes; t_max ≥ D·slice_stride). d_ones_D sized D_mpo. No OVERRUN. |
| G. Sibling fix-propagation | DONE | R17 H4 (multi-gpu Step 3 batching) propagated only to two-site, NOT single-site — see H19-multi-ss-step3. CR-D1 / D6 / D12 propagation verified correct. |

A-G all DONE or N-A. Review valid.

## Regression-watch (R15 → R19)

| Watch item | dmrg-gpu | dmrg2-gpu | pdmrg-gpu | multi-gpu |
|---|---|---|---|---|
| `#include common/batch_ptrs_kernels.h` | :13 | :17 | :22 | :18 |
| Local `setup_batch_ptrs_*` __global__ | 0 | 0 | 0 | 0 |
| Local pdmrg sparse __global__ kernels | 0 | 0 | 5 | 3 (M18 carry — bytewise dup of pdmrg-gpu) |
| `t_lanczos_` begin/end | 1/1 | 1/1 | 1/1 | 1/1 |
| `t_apply_heff_` begin/end | 1/2 | 1/2 | 2/4 | 2/2 |
| `t_svd_` begin/end | 1/2 | 1/2 | 1/1 | 1/1 |
| `t_absorb_` begin/end | 2/2 | 2/1 (R/L sym) | 0/0 (REMOVED) | 0/0 (REMOVED) |
| `t_env_update_` begin/end | 2/2 | 2/2 | 2/2 | 2/2 |
| `D_mpo_actual_-1` boundary write | :878 | :953 | :1250 | n/a (no D_PAD) |
| `init_mps_product/_neel` | h:35-36 | h:35-36 | h:36-37 | h:47-48 |
| `PointerModeGuard` use-sites | 2 | 2 | 4 | 4 |
| `d_ones_D` alloc/use/free | 1/1+ref/1 | 1/1+ref/1 | per-stream | per-device :334-356 |
| Lanczos device-pointer (`d_alpha_dev` etc.) | 3 refs | 2 refs | 33 refs | 26 refs |
| `accurate_svd_gpu` (J1) hits | 0 | 0 | 7 (incl. :2452) | 4 (incl. :2248) |
| `lapack_gesvd` hot-path hits (non-init) | 0 | 0 | 2 (gated by use_cpu_svd_) | 2 (gated by use_cpu_svd_) |
| Sparse-MPO compaction | yes | yes | yes | absent (out-of-scope) |

**Regression status**: zero. All R15-R18 fixes verified intact. CR-D1
(54f2fcf) confirmed -gpu-tier-immune (zero `block_davidson` /
`d_dav_work` references in any of the 4 variants). D5 (-opt host-batch
removal) confirmed -gpu-tier never had `h_batch_*_pinned` on hot path.
D6 kernel consolidation: all 4 variants pull from
`common/batch_ptrs_kernels.h`; only pdmrg-specific `setup_heff_ss_*` /
`setup_lenv_*` / `setup_renv_*` remain locally and are NOT D6-class
(different name family). D12 device-pointer-mode Lanczos intact across
all 4. PhaseTimer panels balanced; pdmrg-gpu apply_heff begin=2 end=4
verified by inspection at impl :899 / :923 (early return) / :1079
(main return) / :1875 / :1894 / :2030 — two functions × (1 begin + 1
early-return-end + 1 main-end).

## CRITICALS

None.

## HIGHS

### H19-multi-ss-step3 — pdmrg-multi-gpu apply_heff_single_site Step 3 NOT batched (R17-H4 sibling-port gap)

[`pdmrg-multi-gpu`: `pdmrg_multi_gpu_impl.h:1719-1737`]

The R17 H4 commit `12d02c5` ported pdmrg-multi-gpu **two-site** Step 3
to the dmrg2-gpu R3-F1 batched-collapse pattern (one setup kernel +
one gemm_batched of size `D·dd` + one gemv reduce). The **single-site**
Step 3 in `apply_heff_single_site` was NOT updated and still uses the
per-`wp` host loop:

```cpp
for (int wp = 0; wp < D; wp++) {
    Scalar beta = (wp == 0) ? Traits::zero() : Traits::one();
    hipLaunchKernelGGL(setup_heff_ss_step3_ptrs<Scalar>, dim3(1), dim3(d), 0, dev.stream, ...);
    ROCBLAS_CHECK(Traits::gemm_batched(dev.handle, ..., d));   // batch=d, not D*d
}
```

This is **D launches × (kernel + gemm_batched-of-size-d) per
apply_heff_single_site call** vs. one launch in pdmrg-gpu
`apply_heff_single_site` (impl :1991-2018) — same defect class as the
two-site fix, just in the sibling function within the same file.

**Why this is HIGH (not MEDIUM):** apply_heff_single_site is the
**default warmup/polish path** per `PDMRG-rules-2026-04-15` (n_warmup=2,
n_polish=2 single-site mandatory). Every paper-config benchmark fires
this path D times per Lanczos iteration × Lanczos-iters × bonds × 4
sweeps. On D=24 (Heisenberg) that is ~24× extra dispatch overhead per
matvec, exactly the regression class round-17 H4 was supposed to fix.

**Sibling check (technique G)**:
- `dmrg-gpu` apply_heff impl :666-680 — fully batched (D·d). ✓
- `pdmrg-gpu` apply_heff_single_site impl :1991-2018 — fully batched (D·d). ✓
- `pdmrg-multi-gpu` apply_heff_two_site impl :813-845 — fully batched (D·dd). ✓ (R17 H4)
- `pdmrg-multi-gpu` apply_heff_single_site impl :1719-1737 — **per-wp host loop. MISSING.**

**Why D13 registry didn't catch it**: registry awk requires bare
`Traits::gemm(` inside the wp loop; here it's `Traits::gemm_batched(`
with batch=d. Registry pattern needs widening to flag "wp loop with
per-iter rocblas dispatch" regardless of batched/unbatched. Tracking
as a separate MEDIUM below.

**Workspace-sizing check (technique F) for the proposed fix**:
porting to D·d batched would write per-wp slices of size
`cL·cR·d·D = slice_stride·D` into d_T1 scratch (matching pdmrg-gpu's
single-site V-as-scratch pattern). T1 sized `D·d²·χ²`; slice_stride =
`cL·d·cR ≤ d·χ²`; `D · slice_stride ≤ D · d · χ² ≤ D · d² · χ² = t_max`.
**Sufficient sizing exists** for the fix.

## MEDIUMS

### M19-d13-pattern-too-narrow (registry hardening)

[`.claude/scripts/defect-registry.sh:D13`]

D13's awk filter requires `Traits::gemm(` (non-batched) inside the wp
loop. This misses the pdmrg-multi-gpu single-site case where the per-wp
loop wraps a `Traits::gemm_batched(handle, ..., d)` (batched over d only,
not over D). Suggest broadening D13 to flag ANY per-wp host loop
containing a rocBLAS dispatch — batched or not — so future
single-site/two-site asymmetries surface in the orchestrator pre-step.

### M18-multi-pdmrg-kernel-duplication-with-pdmrg-gpu (carry from R17)

[`pdmrg-multi-gpu`: `pdmrg_multi_gpu_impl.h:25-61`]

`setup_heff_ss_step3_ptrs`, `setup_lenv_step3_ptrs`,
`setup_renv_step3_ptrs` bytewise-identical to pdmrg-gpu impl :31-118
(modulo line numbers). D6 axis-1 (sibling within family) propagation
gap. Cosmetic-class — no correctness/perf impact, but should land in
`common/batch_ptrs_kernels.h` mirroring the R16 pattern.

## NITS

### N13-pmg-error-discard (carry-over)

[`common/pointer_mode_guard.h:16,17,22`]

Discards return of `rocblas_get/set_pointer_mode`. Cosmetic.

## FALSE POSITIVES VERIFIED

- **CR-D1 (54f2fcf) -gpu-tier propagation**: -gpu tier has zero
  `block_davidson` / `d_dav_work` references — Davidson is -opt-only.
  Genuinely immune per technique-G. ✓
- **D5/H2-opt host-batch (abd88b9 + 187fddf)**: -gpu tier has zero
  `h_batch_*_pinned` allocations on hot path; one comment-only
  reference at pdmrg-gpu :494 documenting historical removal. ✓
- **D6 (8abb6e7) kernel consolidation**: all 4 variants `#include
  common/batch_ptrs_kernels.h`; zero local `setup_batch_ptrs_*` defs.
  pdmrg-multi-gpu's 3 local kernels are pdmrg-specific names not in
  common — M18 carry, not D6 violation. ✓
- **D12 (0efe96d) Lanczos host-stack**: -gpu tier was already on
  device-pointer mode in earlier rounds; counts unchanged
  (3/2/33/26 refs to d_alpha_dev/d_beta_dev/d_neg_alpha/etc.). ✓
- **D9/D15 (12d02c5) PhaseTimer panels**: dmrg2-gpu t_absorb_
  begin=2 end=1 + t_svd_ begin=1 end=2 are R/L mutual-exclusive
  control flow — balanced per call (verified at impl :1297, :1324,
  :1335). pdmrg-gpu apply_heff begin=2 end=4 = two functions ×
  (1 begin + 1 early-return + 1 main-return). ✓
- **pdmrg/multi-gpu host LAPACK gesvd hits** (4 hot-path matches):
  all 4 are gated by `use_cpu_svd_` opt-in flag (verified at impl
  :1539, :2056, :1391, :1761) — not on default code path. ✓
- **multi-gpu apply_heff_two_site T1 reuse**: T1 sized
  `t_max = D·d²·χ² ≥ D·slice_stride`. Step 2 reads T1 → T2 (impl
  :803-811) BEFORE Step 3 writes T1 (impl :821-825). Lifetime
  sequential. **OK.** (Same verdict as R18.)
- **multi-gpu d_ones_D allocation** (impl :352-356): D_mpo_ Scalars
  per device; init'd once; freed in free_gpu_resources :516. ✓
- **multi-gpu update_left_env / update_right_env per-sp loop**
  (impl :898-916, :973-991): sibling-consistent with pdmrg-gpu
  :1131-1150, :1207+ (NOT a defect — env-update uses per-sp loop with
  D-batched GEMM in BOTH variants by design). ✓

## SUMMARY

**Round 19 confidence re-run**: 0 CRITICAL, **1 HIGH** (NEW —
H19-multi-ss-step3), 2 MEDIUMs (1 NEW M19-d13-pattern-too-narrow + 1
M18 carry), 1 NIT carry. Code state unchanged since R18 (bb809fb only
landed reports), but a fresh technique-G pass surfaced a sibling-port
gap that R18 missed: R17 commit `12d02c5` H4 ported the
pdmrg-multi-gpu two-site Step 3 to the dmrg2-gpu R3-F1 batched
pattern but left the **single-site** Step 3 on the per-wp host loop
(impl :1719-1737). The single-site path is the default warmup/polish
matvec per PDMRG rules and fires every benchmark; this is a perf-only
HIGH (not correctness), same defect-class as the two-site fix it
shipped alongside. Workspace sizing for the fix is verified
sufficient (T1 = D·d²·χ² ≥ D·slice_stride). All other R15-R18 fixes
verified intact: CR-D1 immunity, D6 consolidation, D12
device-pointer-mode Lanczos, D7/D8 use_cpu_svd_ gating, D_PAD R-env
slot, J1 accurate_svd_gpu (pdmrg only), PhaseTimer panel discipline,
d_ones_D allocation. **Recommend fixing H19-multi-ss-step3 BEFORE
GPU allocation** — single-site is the default path on every
challenge config, and the fix is mechanical (mirror pdmrg-gpu
:1991-2018 / dmrg-gpu :666-680 in pdmrg-multi-gpu apply_heff_single_site).

This R19 finding is a textbook lesson_round12_propagation_gaps
axis-(e): "fix-one-variant-don't-sweep-siblings — fixing variant X
means grep all sibling FUNCTIONS for the same defect class" — R17 H4
fixed multi-gpu apply_heff_two_site without grepping
apply_heff_single_site in the same file. Add to retrospective.
