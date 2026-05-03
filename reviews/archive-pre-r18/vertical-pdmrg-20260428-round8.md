# Vertical review — pdmrg family — 2026-04-28 (round-8 verification, post-69305f0)

Pre-G1 final gate. Verifies all round-7 fixes attributed to the pdmrg
cluster (C4, C5, C6, H1, H2, H3, H4, H7, M11, M12) are intact and runs
techniques A-E in full to surface any residual or newly-introduced
defects.

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | 0 dead members across all three variants on the critical set |
| B. Behavioral diff | DONE | merge_and_optimize_boundaries divergence found in -base (see C-new1) |
| C. Docstring verification | DONE | Stoudenmire J1 docstring correct in all 3 tiers; -base round-6 fix intact |
| D. clangd filter | N-A | ROCm headers absent on host; technique A subsumes |
| E. Absence-naming brief | FOLLOWED | Tier checklist below |

---

## Round-7 fix verification (in-scope cluster)

| ID | Variant | Claim | Evidence | Status |
|---|---|---|---|---|
| C4 | pdmrg-gpu-opt | `use_davidson_=true` default | `pdmrg_gpu_opt_impl.h:204` | INTACT |
| C5 | pdmrg-gpu-opt | `n_recal` in `run()` + recal loop | header `:43-48`; impl `:3415, :3513-3519` | INTACT |
| C6 | pdmrg-gpu-opt | `d_Vh_canonical` swap before `update_right_env(bsite+1)` | header `:211`; impl alloc `:294`, swap `:3392-3404` | INTACT |
| H1 | pdmrg-gpu-opt | streams created with `hipStreamNonBlocking` | impl `:125, :138` | INTACT |
| H2 | pdmrg-gpu-opt | `parallel_sweep` exception-safe via `std::exception_ptr` | impl `:3447-3469` | INTACT |
| H3 | pdmrg-gpu | `initialize_mps_product` / `_neel` present | header `:36-37`; impl `:697, :712` | INTACT |
| H4 | pdmrg-gpu | RSVD path syncs removed | impl `:1724-1815` (no syncs between gemm/qr/orgqr/gemm) | INTACT |
| H7 | pdmrg-gpu | `PointerModeGuard` RAII | impl `:1323, :1347, :1369, :1534` | INTACT |
| H7 | pdmrg-gpu-opt / -base | `PointerModeGuard` RAII | partial — see M-new1 | PARTIAL |
| M11 | pdmrg-gpu-opt | `set_use_davidson` symmetric round-trip | header `:61-72` | INTACT |
| M12 | pdmrg-gpu | `set_rsvd` writes `opts_.rsvd`; legacy `use_rsvd_` removed | header `:52`; impl `:343` | INTACT |
| H9 (round-7) | pdmrg-gpu-base | dead two-site `sweep_LR_full` / `sweep_RL_full` removed | header `:206-207` (only `_1site` declared); impl `:1103, :1129` (only `_1site` defined) | INTACT |

**All round-7 in-scope fixes for pdmrg verified intact.** No regressions vs the conformity-20260428 baseline on the listed defect IDs.

---

## CRITICALS — block GPU run / paper submission

### C-new1. pdmrg-gpu-base: `merge_and_optimize_boundaries` builds `R_env` from S·Vh, not from canonical Vh

`pdmrg_gpu_base_impl.h:1289-1313` allocates `MPS[bsite+1]` and writes
**S·Vh** into it via `scale_rows_by_diag_kernel`, then immediately calls
`update_left_env(bsite, si)` and `update_right_env(bsite+1, si)` on
that same tensor. The R-environment for bond `bsite+1` is therefore
built with norm = S² ≠ I, which breaks the `N_eff = I` assumption in
all subsequent Lanczos eigensolves on segments touching that boundary.

Both pdmrg-gpu (`pdmrg_gpu_impl.h:2540-2575`) and pdmrg-gpu-opt
(`pdmrg_gpu_opt_impl.h:3382-3404`) explicitly perform a swap-Vh,
build-R_env, restore-S·Vh dance with a pre-allocated `d_Vh_canonical`
buffer, and the comment block at pdmrg-gpu:2540-2542 names this exact
correctness consequence. The structural divergence in -base is the
same defect class as round-7 C6 — fixed in -opt that round but never
back-ported to -base.

Same charter argument as the J1 lock applies: this is part of pdmrg's
algorithmic correctness, not an optimization. The -base charter
("competent first-pass, correctness-equivalent to -gpu / -opt minus
the optimizations") is violated as long as this is missing.

`gpu-rocm/pdmrg-gpu-base/src/pdmrg_gpu_base_impl.h:1283-1313`

**Severity: CRITICAL.** Affects every benchmark that exercises
boundary-merge sweeps (i.e., every n_segments ≥ 2 run). Pre-G1
candidate fix: copy the swap pattern from `pdmrg_gpu_impl.h:2543-2575`
and add `d_Vh_canonical` to the -base StreamWorkspace.

---

## HIGHS

(none net-new in scope; previously-flagged H1-H4/H7/H9 all confirmed
INTACT above.)

---

## MEDIUMS

### M-new1. pdmrg-gpu-opt and pdmrg-gpu-base still toggle `rocblas_pointer_mode` with paired raw calls

H7 was applied to pdmrg-gpu in full (impl `:1323, :1347, :1369, :1534`
all use `PointerModeGuard`) but partially in pdmrg-gpu-opt and
pdmrg-gpu-base. Sites still using paired
`rocblas_set_pointer_mode(...,_device)` /
`rocblas_set_pointer_mode(...,_host)`:

- `pdmrg_gpu_opt_impl.h:1786/1788, 1801/1805, 1819/1885, 1955/1968` —
  inside `lanczos_eigensolver`. Lines 1998 already use
  `AsvdPointerModeGuard`, so the migration was started but not
  completed.
- `pdmrg_gpu_base_impl.h:734/798, 835/840` — inside Lanczos α/β scal
  loop.

If a `ROCBLAS_CHECK` between the two raw calls throws, device pointer
mode leaks into subsequent rocBLAS handles on the same stream. The
common `PointerModeGuard` (`gpu-rocm/common/pointer_mode_guard.h:12`)
is already available — replacement is a 2-line edit per site.

**Severity: MEDIUM** (same as the H7 origin defect; pdmrg-gpu got the
fix, two siblings only got it partially).

### M2 (carryover). `d_T3_` per-stream scratch declared but unused in pdmrg-gpu-base

Re-verified from conformity-20260428. Symbol absent from impl —
header carries only `d_T1`, `d_T2`. False positive: M2 actually
referred to dmrg2/pdmrg-gpu's `d_T3_`; in pdmrg-gpu-base the field
does not exist. Closing this carryover.

### M3 (carryover, FIXED). `h_batch_*_pinned` removed from pdmrg-gpu

`pdmrg_gpu_impl.h:554` carries the explanatory comment "h_batch_*_pinned
no longer allocated (GPU kernel pointer setup)". Header confirms
fields gone. Closing.

---

## NITS

- `pdmrg_gpu_opt.h:18-24` docstring orders points strangely — point 1
  ends mid-sentence, then a templating note interjects, then points 2
  and 3 follow. Cosmetic.
- `pdmrg_gpu_opt_impl.h:208` comment "Chebyshev-filtered subspace
  iteration eigensolver" appears at the flag-default site but the
  Chebyshev path's documentation lives at the function definition;
  cross-link missing.

---

## FALSE POSITIVES VERIFIED (preserved from conformity-20260428)

- **pdmrg-gpu-opt `worker_streams_` / `step_done_events_` /
  `worker_done_events_` / `use_batched_sweep_` / `use_chebyshev_`** —
  all live: 7, 7, 8, 5, 2 impl references respectively. Dispatched at
  `pdmrg_gpu_opt_impl.h:2232, 3475, 3488`. Round-6 verification holds
  post-round-7.
- **J1 Stoudenmire lock** — `accurate_svd_gpu` confirmed in all three
  tiers' `merge_and_optimize_boundaries`:
  `pdmrg_gpu_base_impl.h:1260`, `pdmrg_gpu_impl.h:2482`,
  `pdmrg_gpu_opt_impl.h:3334`. -base docstring claim corrected
  (round-6) and stable.
- **pdmrg-gpu-opt `block_davidson_eigensolver` host `lapack_syev` per
  iteration** — out of scope per orchestrator brief; deferred for
  post-G1 cycle. Not re-flagged.
- **pdmrg-gpu-opt `d_xs_batch_*_` cross-segment batched arrays** —
  4 references each. Allocated/used in `batched_segment_sweep`. Not
  dead.
- **`lanczos_graph_was_user_enabled_`** — 0 impl refs but 3 header
  refs (entirely consumed by inline `set_use_davidson`). Correct;
  not dead.

---

## Tier feature checklist (technique E)

| Feature | -base | -gpu | -opt |
|---|---|---|---|
| Per-segment streams (NonBlocking) | present | present | present (round-7 H1) |
| `accurate_svd_gpu` at boundary (J1) | present | present | present |
| Per-stream `AsvdScratch` | present | present | present |
| Vh-canonical swap before update_right_env | **MISSING (C-new1)** | present | present (round-7 C6) |
| Fused WW precompute | host (charter-allowed) | host (H5 docstring fix) | host (H5 docstring fix) |
| HIP-graph capture for Lanczos | absent (charter) | present | present |
| GpuOpts + PhaseTimer | absent (charter) | present | present |
| RSVD on-device inner SVD | absent (charter) | present (round-7 H4 syncs gone) | present |
| Sparse-MPO (single + two-site nnz lists) | absent (charter) | present | present |
| Batched GEMM with cached pointer arrays | absent (charter) | present | present |
| Pinned host pointer arrays | absent (charter, M3 cleaned) | absent (M3 cleaned) | absent |
| `precompute_fused_mpo` ON DEVICE | n/a | host (H5 known) | host (H5 known) |
| D_PAD with `D_mpo_actual_` | absent (charter) | present | present |
| `set_cpu_svd` setter | absent | present | present |
| `set_rsvd` setter (single source of truth) | absent | present (round-7 M12) | present |
| `n_recal` argument to `run()` | absent (charter; -base does not need it) | present | present (round-7 C5) |
| `pad_mfma16` + `chi_max_user_` | n/a | n/a | present |
| Block-Davidson default + Lanczos fallback | n/a | n/a | present (round-7 C4 default true) |
| Worker-stream pool (live) | n/a | n/a | present, live |
| `batched_segment_sweep` + flag | n/a | n/a | present, live |
| `chebyshev_eigensolver` + flag | n/a | n/a | present, live |
| `set_use_davidson` symmetric | n/a | n/a | present (round-7 M11) |
| `parallel_sweep` exception-safe | present (impl `:1391-1407`) | present (impl `:2645-2662`) | present (round-7 H2) |
| `PointerModeGuard` RAII at toggle sites | partial (M-new1) | full (round-7 H7) | partial (M-new1) |
| Two-site `sweep_LR_full` / `sweep_RL_full` removed from -base | YES (round-7 H9) | n/a | n/a |
| `initialize_mps_product/_neel` | n/a | present (round-7 H3) | n/a |

---

## SUMMARY

**Block GPU run? YES — 1 CRITICAL (new finding).**

All round-7 in-scope cluster fixes (C4, C5, C6, H1, H2, H3, H4, H7,
M11, M12, H9) are intact at commit 69305f0. The pdmrg variants are
otherwise in good shape — 0 dead infrastructure on the critical
member set, J1 Stoudenmire confirmed in all three tiers, J2 superset
contract restored after round-7.

The single new blocker is structural and uncovered by technique B
(pair-wise behavioral diff against pdmrg-gpu / pdmrg-gpu-opt):
**`pdmrg-gpu-base::merge_and_optimize_boundaries` skips the canonical
Vh swap before `update_right_env`** that pdmrg-gpu (round-6) and
pdmrg-gpu-opt (round-7 C6) both perform. R_env is built from S·Vh
instead of Vh; norm = S² ≠ I poisons the `N_eff = I` Lanczos
assumption on every subsequent boundary-touching segment sweep.

The fix mirrors the round-7 C6 pattern: declare `d_Vh_canonical` in
the -base `StreamWorkspace`, allocate at `theta_size_max_ * sizeof(Scalar)`,
copy `new_k` rows out of `d_svd_Vh` (handle the `new_k == full_k` and
`new_k < full_k` cases the same way pdmrg-gpu does at `:2557-2569`),
swap into `d_mps_tensors_[bsite+1]`, call `update_right_env`, then
restore the S·Vh pointer. About 25 lines of code.

### Top action before MI300X G1 window

**Fix C-new1 in pdmrg-gpu-base.** This is the only finding that affects
correctness on the default code path of an in-scope variant.

The two carryover MEDIUMs (M-new1 partial pointer-mode RAII migration
in -opt and -base; tier-checklist `PointerModeGuard` RAII column) are
not GPU-run blockers — paired raw toggles are correct under
no-exception conditions, and ROCBLAS_CHECK throws are unobserved in
benchmark runs. Schedule for the post-G1 cleanup pass.

The pdmrg-gpu-opt Block-Davidson host `lapack_syev` per-iteration
issue (analogous to dmrg-gpu-opt C2 / dmrg2-gpu-opt H6) remains
out-of-scope per the orchestrator brief and is not re-flagged here.
