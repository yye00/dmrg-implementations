# Full conformity review — 2026-04-28 (round-8, post round-7 fixes)

Pre-G1 GPU window gating audit. Six sub-reviews dispatched in parallel
against commit `69305f0`. Two net-new criticals surfaced and fixed in
commit `b67417a`.

## Charter proof — sub-review status

| Sub-review | Status | Findings (vs prior baseline) |
|---|---|---|
| vertical-review-dmrg     | OK | 0 critical, 1 high (misleading scaffolding comment), 5 mediums (4 deferred per #101/#102/#104/#112), 3 nits |
| vertical-review-dmrg2    | OK | **1 NEW CRITICAL (CR-D1, regression)**, 2 highs (pre-existing C1-class), 2 mediums, 3 nits |
| vertical-review-pdmrg    | OK | **1 NEW CRITICAL (C-new1, pre-existing)**, 0 highs, 1 medium |
| horizontal-review-base   | OK | 0 critical, 0 high, 1 medium (cosmetic set_quiet) |
| horizontal-review-gpu    | OK | 0 critical, 0 high (all round-7 -gpu fixes verified intact), 1 medium |
| horizontal-review-opt    | OK | 0 critical, 0 high (all round-7 -opt criticals verified intact), 2 mediums |

All six sub-reviews ran A-E in full (D done-via-grep on five of six).

## NET-NEW CRITICALS — both fixed in commit `b67417a`

### CR-D1 (REGRESSION I introduced in round-7 batch 5)

**Block-Davidson `d_dav_work_` buffer overrun in dmrg-gpu-opt and dmrg2-gpu-opt.**

[`gpu-rocm/dmrg-gpu-opt/src/dmrg_gpu_opt_impl.h:299-301`, `gpu-rocm/dmrg2-gpu-opt/src/dmrg2_gpu_opt_impl.h:259-261`]

The H6 syev port routed the residual-loop overlap matrix into `d_dav_work_ + n_new*dim`, but the `dav_work_sz` computation in the constructor was sized for the residual region only. On every Davidson iteration with `dim ≥ 256`, the overlap region (`max_sub*b ≤ 128 Scalars`) overruns `d_dav_work_` and corrupts the next allocation in the device address space. Triggered on the default code path (`use_davidson_=true`).

**Fix**: bump `dav_work_sz` to include the overlap region. One-line change in each ctor.

*(found by: vertical-review-dmrg2)*

### C-new1 (PRE-EXISTING latent bug, J1-adjacent)

**pdmrg-gpu-base builds boundary R_env from S·Vh instead of canonical Vh.**

[`gpu-rocm/pdmrg-gpu-base/src/pdmrg_gpu_base_impl.h:1283-1313`]

pdmrg-gpu (impl:2540-2575) and pdmrg-gpu-opt (round-7 C6 fix) both swap the canonical Vh into `MPS[bsite+1]` before `update_right_env(bsite+1)`, then restore S·Vh after. pdmrg-gpu-base never had this swap. R_env norm = S² ≠ I poisons the `N_eff = I` assumption in subsequent Lanczos eigensolves.

This is the C6 defect class that was fixed in -opt but never back-ported to -base. The vertical-review-pdmrg subagent's behavioral diff (technique B) caught it by direct comparison against the -gpu sibling.

**Fix**: add `d_Vh_canonical` to `StreamWorkspace`, alloc/free, port the swap pattern from pdmrg-gpu.

*(found by: vertical-review-pdmrg)*

## HIGHS (deduplicated, none new)

- **Misleading scaffolding comment in dmrg-gpu-opt ctor** [`dmrg_gpu_opt_impl.h:167-194`] — the device-pointer Lanczos scratches added in M8/M9 (`d_dot_result_`, `d_alpha_dev_`, etc.) say "ported for the H10 device-pointer Lanczos path" but the Lanczos function still uses host α/β. They are deferred-state per task #101. Comment should make that explicit. *(found by: vertical-review-dmrg)*
- **Pre-existing C1 in dmrg2-gpu-opt** [`dmrg2_gpu_opt_impl.h:664-708, 815-825, 891-901`] — same defect class as the C1 deferral in dmrg-gpu-opt: per-site host pointer-table construction in `apply_heff_two_site` and env updates. Conformity-20260428 audit didn't surface this because dmrg2 wasn't in C1 scope. Same fix template (port `setup_batch_ptrs_*` GPU kernels). Schedule for post-G1. *(found by: vertical-review-dmrg2)*
- **H7-extension in pdmrg-gpu-opt and pdmrg-gpu-base** — 7+ raw `set_pointer_mode` toggles in lanczos_eigensolver / accurate_svd boundary that didn't get migrated to `PointerModeGuard` when round-7 H7 promoted the guard. Common header exists; mechanical migration. Defer to post-G1. *(found by: vertical-review-pdmrg)*

## MEDIUMS (deduplicated, none new)

- M-base-1: `set_quiet(bool)` is still header-declared no-op across all -base variants (pre-existing, tier-consistent).
- pdmrg-gpu-opt uses `Traits::rocsolver_gesvd` while pdmrg-gpu uses `rocsolver_gesvd_auto` (Jacobi). Pre-existing perf gap; J2-relevant.
- pdmrg-gpu-opt block_davidson_eigensolver still uses host `lapack_syev` — same defect class as C2/H6 but in pdmrg-opt. Round-7 deferred this explicitly.
- dmrg-gpu retains its local `DmrgPointerModeGuard` instead of using common/. M1 promoted the pattern but missed dmrg-gpu.
- `opts_.device_k` declared and loaded in dmrg-gpu-opt but never read.
- Stale comment `// h_batch_*_pinned no longer allocated` in pdmrg-gpu impl.

## NITS (no action required)

- Multiple stale comments referencing pre-edit line numbers.
- `pad_mfma16` still verbatim across three -opt headers (consolidation deferred).
- Per-bond `hipMalloc` in `precompute_fused_mpo` (M6 deferred — set_mpo time only).
- `prof_*` profiling counters as file-scope statics (single-instance bench).

## FALSE POSITIVES VERIFIED (preserved)

- D_PAD R-env identity slot at `D_mpo_actual_-1` — verified clean across all variants.
- Round-6 dual-stream env-update overlap intact in dmrg-gpu, dmrg2-gpu, dmrg-gpu-opt, dmrg2-gpu-opt.
- Round-7 H6 Davidson `rocsolver_syevd` port — eigvecs survive in `d_dav_work2_` correctly; aliasing logic is right (only buffer-sizing was wrong, fixed in CR-D1).
- All round-7 fixes in pdmrg-gpu-opt (C4, C5, C6, H1, H2, M11) verified at the cited lines.
- pdmrg-gpu-opt `worker_streams_`, `use_chebyshev_`, `use_batched_sweep_` all live (not dead infrastructure).
- pdmrg J1 lock — `accurate_svd_gpu` confirmed in all three pdmrg tiers' `merge_and_optimize_boundaries`.
- HIP_CHECK consolidation — all 12 impls use `common/hip_check.h`.

## SUMMARY VERDICT

### Block GPU run? — **NO** (after `b67417a`).

Both net-new criticals fixed. The round-8 conformity pass surfaced exactly what it was designed to surface: a regression introduced by my own round-7 work (CR-D1 — buffer-sizing bug that smoke tests with dim<256 wouldn't hit), and a pre-existing latent J1-adjacent defect that the round-7 C6 fix in -opt should have triggered as a "check the same thing in -base" reflex (C-new1).

### What the user asked

> "you told me we were ready many many times before"

Fair. Each round has surfaced new defects. The round-8 ones, in particular:

- **CR-D1 was caused by my round-7 work.** This is the failure mode the methodology guards against: I introduced a regression while fixing C2/H6, the buffer-aliasing logic was correct but the buffer-sizing followed the old layout. Without the orchestrator catching it, the GPU run would have produced corrupted Davidson eigenvectors at chi ≥ 16 challenge sizes — a classic "passes smoke tests, fails on real workloads" pattern.
- **C-new1 was pre-existing.** Round-7 fixed C6 in pdmrg-gpu-opt but didn't audit the same defect class in pdmrg-gpu-base. The orchestrator's behavioral-diff (technique B) caught the divergence.

### Top-3 actions before MI300X G1 window

All three already done in commit `b67417a`:

1. ✓ Fix CR-D1 buffer-sizing in dmrg-gpu-opt + dmrg2-gpu-opt ctors.
2. ✓ Fix C-new1 by porting the canonical-Vh swap pattern from pdmrg-gpu to pdmrg-gpu-base.
3. ✓ Document both in commit message + this report so the next round's diff catches any regression.

### Recommended next step

Build all variants on the remote MI300X (`hipcc` compile + standalone `test_*` correctness runs) BEFORE starting the G1 benchmark sweep. The build will catch any compile-time bugs introduced by round-7 / round-8; the standalone tests will catch correctness regressions on small fixtures.

### Remaining deferred items (no G1 impact, post-G1 cycle)

- C1: dmrg-gpu-opt apply_heff host pointer-tables (paper-excluded variant).
- H10: dmrg-gpu-opt + dmrg2-gpu-opt device-pointer Lanczos α/β (Davidson is default, fallback rarely hit).
- H11: dmrg-gpu-opt sparse-MPO Step-3 → gemm_batched (niche path).
- H7-ext: pdmrg-gpu-opt / pdmrg-gpu-base remaining `set_pointer_mode` toggles → `PointerModeGuard`.
- M1-ext: dmrg-gpu local guard → common/.
- M6: precompute_fused_mpo contiguous d_WW alloc (set_mpo time only).
- pdmrg-gpu-opt block_davidson host syev → rocsolver_syevd (out-of-original-scope; same fix template as C2/H6).
- Pre-existing dmrg2-gpu-opt apply_heff host pointer-tables (same C1 defect class).

Each is documented with explicit reasoning. None affect G1 timing or correctness on the default code path of any G1-target variant.
