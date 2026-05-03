# Vertical review — DMRG family (single-site) — round-11 — 2026-04-28

HEAD: `1d44d89` (round-10 conformity report — round-10 fixes pushed in `4d8924d`)
Baseline: `cfd08c3` round-9, then `db7dcdf` round-10 self-audit, then `4d8924d` round-10 orchestrator fixes
Scope: `gpu-rocm/dmrg-gpu-{base,_,opt}/src/*` + `gpu-rocm/common/*`

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | 3 variants, ≥48 private members audited; **9 dead members in dmrg-gpu-opt** (Lanczos device-pointer scratches + `h_svd_tmp_`) — see CRITICAL/MEDIUM below. |
| B. Behavioral diff   | DONE | dmrg-gpu Lanczos uses device-pointer-mode (sync-free); dmrg-gpu-opt's Lanczos *fallback* regresses to host-pointer-mode + per-iteration host α/β. Documented as intentional in round-10 (FALSE POSITIVES); scaffolding for the device-pointer path is allocated but unused — defect class is "DEAD" not "behavior". 0 other divergences. |
| C. Docstring verify  | DONE | dmrg-gpu-opt header line 14-50 claims "fallback Lanczos retained" — code confirms function exists, is reachable when `set_use_davidson(false)`, and runs (host-pointer mode). Claim accurate. 7 other claims (dual-stream, graph capture, RSVD, sparse-MPO, D_PAD, MFMA-16, batched Step-3, on-device dsteqr in fallback) all backed by code. |
| D. clangd filter     | N-A  | No ROCm headers on host; technique A subsumes the dead-symbol case. |
| E. Absence-naming    | FOLLOWED | -base / -gpu / -opt expected-feature checklists run. Per-tier features all present per round-10. |
| F. Workspace-aliasing | DONE | `d_dav_work_` / `d_dav_work2_` re-verified at `:310-315` (CR-D1 sizing intact); `d_T1_/T2_` env-stream split intact `:121-124`; `d_svd_work_` lives. **0 OVERRUN.** |
| G. Sibling fix-propagation | DONE | Round-10 fixes (M-opt-rsvd-env, H10-multi-WW-leak) traced. M-opt-rsvd-env applied in all 3 -opt variants verified at `dmrg_gpu_opt_impl.h:62`, `dmrg2_gpu_opt_impl.h:60`, `pdmrg_gpu_opt_impl.h:205`. H10-multi-WW-leak is out-of-family (multi-gpu) and immune within dmrg-family. **0 MISSING.** |

A review with any technique SKIPPED that is not N-A is INVALID — none skipped.

## Regression-watch verification (commit-pinned)

| Watch item | Variant | File:line | Status |
|---|---|---|---|
| H-new1 (M4-W): W-buffer guards | dmrg-gpu-base | `dmrg_gpu_base_impl.h:225,244,248` | OK |
| H-new1 (M4-W): W-buffer guards | dmrg-gpu | `dmrg_gpu_impl.h:566,599,603` | OK |
| H-new1 (M4-W): W-buffer guards | dmrg-gpu-opt | `dmrg_gpu_opt_impl.h:516,547,551` | OK (shifted +5 by round-10 ctor edit; all guards present) |
| M4-W (sparse-nnz buffers) | dmrg-gpu | `dmrg_gpu_impl.h:638,644` | OK |
| M4-W (sparse-nnz buffers) | dmrg-gpu-opt | `dmrg_gpu_opt_impl.h:584,590` | OK |
| H1-ext-gpu: nonblocking stream / stream_env | dmrg-gpu | `dmrg_gpu_impl.h:200-201` | OK |
| H1-ext-gpu: nonblocking stream / stream_env | dmrg-gpu-opt | `dmrg_gpu_opt_impl.h:110-111` | OK |
| H1-final: nonblocking stream | dmrg-gpu-base | `dmrg_gpu_base_impl.h:38` | OK |
| Round-10 M-opt-rsvd-env: `use_rsvd_ = opts_.rsvd` | dmrg-gpu-opt | `dmrg_gpu_opt_impl.h:62` | OK |
| Round-8 CR-D1: dav_work_sz residual+overlap concurrent sizing | dmrg-gpu-opt | `dmrg_gpu_opt_impl.h:305-315` | OK (see F detail) |
| Round-7 C5: D_PAD R-env identity at `D_mpo_actual_-1` | dmrg-gpu-opt | `dmrg_gpu_opt_impl.h:982` | OK |
| Round-7 C2: rocsolver_dsteqr replaces host dstev (Lanczos fallback) | dmrg-gpu-opt | `:1146-1148` | OK |
| Round-7 C2: rocsolver_syevd in Davidson | dmrg-gpu-opt | `:1604` | OK |
| Round-6 dual-stream env-update overlap | dmrg-gpu | `:1500-1514, 1536-1548` | OK |
| Round-6 dual-stream env-update overlap | dmrg-gpu-opt | `:1857-1881, 1906-1924` | OK |

## Technique F detail — `d_dav_work_` aliasing audit (re-verified)

`block_davidson_eigensolver` routes two concurrent regions into `d_dav_work_`:
- Region 1 (residuals W): offset 0, size `n_new·dim ≤ b·dim`.
- Region 2 (overlap): offset `n_new·dim`, size `k·n_new ≤ max_sub·b`.
Lifetimes are concurrent in the inner expansion loop. The restart path reuses `d_dav_work_` as a sequential third region of size `dim·keep ≤ dim·b` (subsumed). Required total = `b·dim + max_sub·b`; allocation at `:310-315` is `max(b·theta_size_max_ + max_sub·b, max_sub²)`. Strict superset of required (since `theta_size_max_ ≥ dim`). **OK; no overrun.**

## Technique G detail — round-10 fix-propagation through dmrg-family

| Fix class | dmrg-gpu-base | dmrg-gpu | dmrg-gpu-opt |
|---|---|---|---|
| H10-multi-WW-leak (precompute_fused_mpo guard) | immune (no fused-MPO precompute, single-site) | immune (single-site) | immune (single-site) |
| M-opt-rsvd-env (`use_rsvd_ = opts_.rsvd`) | immune (no `use_rsvd_` member) | immune (`opts_.rsvd` read directly) | OK `:62` |
| All round-9 H-new1 / M4-W guards | OK | OK | OK |
| Round-8 CR-D1 dav_work sizing | immune (no Davidson) | immune (no Davidson) | OK |

**All round-10 fixes either landed where applicable or the variant is genuinely immune by tier contract. No lonely fixes.**

## CRITICALS

None.

## HIGHS

None.

## MEDIUMS — net-new (technique A)

- **MED-opt-dead-lanczos-scratches** [dmrg-gpu-opt: `dmrg_gpu_opt_impl.h:174-181` (alloc) and `:361-368` (free)]
  Eight device-pointer-mode Lanczos scratch buffers — `d_dot_result_`, `d_nrm2_result_`, `d_neg_alpha_`, `d_neg_overlap_`, `d_inv_nrm_`, `d_alpha_dev_`, `d_beta_dev_`, `d_neg_beta_scalars_` — are `hipMalloc`'d in the ctor and `hipFree`'d in the dtor with **zero reads or writes between**. The Lanczos *fallback* function at `:1012-1212` uses host-pointer-mode rocBLAS (`&norm`, `&inv_norm`, `&alpha_result`, `&beta_i`) and a **stack-local** `d_neg_alpha_scr` for the FUSE_LANCZOS path (`:1025`). Same defect class as round-9 MED-base-1 / MED-pdmrg-opt-{1,2,3} dead host buffers. Either wire the scaffolding into the fallback (port the sync-free dmrg-gpu Lanczos pattern) or delete the eight allocations. Cosmetic / wasted device memory; defer to post-G1.

- **MED-opt-dead-h_svd_tmp** [dmrg-gpu-opt: `dmrg_gpu_opt.h:217` (declare) and `dmrg_gpu_opt_impl.h:267` (resize)]
  `h_svd_tmp_` is declared as a `std::vector<Scalar>` member, resized in the ctor at `:267` to `svd_max_dim · chi_max_`, and **never read or written elsewhere**. Same dead-host-buffer class. Delete the declaration + resize.

These two findings are the same defect pattern that round-9 caught for pdmrg-gpu-opt and dmrg-gpu-base; round-10's vertical-dmrg sub-review claimed "0 dead" in technique A — that result was wrong. Round-11 catches the omission.

## MEDIUMS — pre-existing carry-over

- M-carry: `DmrgPointerModeGuard` is a struct local to `dmrg_gpu_impl.h:18-30`; not migrated to `common/pointer_mode_guard.h`. Round-9 deferral.
- M-carry: `set_quiet(bool)` no-op across all three tiers. API surface kept for parity with pdmrg variants.
- M-carry: dmrg-gpu-opt Lanczos α/β kept host-resident (`:1020-1021`). Default code path is Block-Davidson; this is fallback-only. Tridiagonal solve still on-device via rocsolver_dsteqr.

## NITS

- `dmrg_gpu_opt_impl.h:185` comment block alignment with -gpu sibling (`dmrg_gpu.h:78-94`). Cosmetic, deferred from round-10.

## FALSE POSITIVES VERIFIED

- `d_dav_V_` / `d_dav_AV_` show 3 hits each in -opt; verified live: aliased to `V`/`AV` pointers at `:1540-1541` and used throughout block_davidson via the alias names. Not dead.
- `d_const_one_` / `d_const_zero_` / `d_const_neg_one_` count 3 in -opt; live at `:1043,1296,1338,1393`.
- `lapack_gesvd` at -opt `:285` is a one-shot ctor workspace-size query; per-sweep `lapack_gesvd` at `:1301` is gated by `use_cpu_svd_` (default false). Default code path is on-device rocsolver gesvd.
- `hipStreamSynchronize` calls in -opt at `:1620, 1640, 1822, 1830` are intentional control-flow gates (D2H of single doubles/ints for Davidson convergence + info status). Without `opts_.profile` guard but result is required for branching — not a defect.
- The remaining `hipStreamSynchronize` calls in -gpu/-opt are all `opts_.profile`-gated or in once-per-`run()` paths.
- `chi_max_user_` is -opt-only (MFMA-16 padding); zero hits in -gpu impl is correct because the member doesn't exist there.

## SUMMARY

Round-11 finds **2 net-new MEDIUM** (dead Lanczos device-pointer scratches + dead `h_svd_tmp_` in dmrg-gpu-opt) — both same defect class as round-9 MED finds in pdmrg-gpu-opt that were caught and deferred. Round-10's vertical-dmrg sub-review missed these. **No criticals, no highs, no functional regressions.** All 9 round-10 regression-watch items verified intact at cited lines (5 lines shifted by the round-10 ctor edit, all content preserved).

This is **not a clean round** for vertical-review-dmrg: 2 net-new findings (cosmetic dead-buffer cleanup, both same class as known-deferred MEDs). **The defects do not block the GPU window** — they are wasted device/host memory in a fallback-only code path, identical to other dead-buffer MEDs already deferred to post-G1 cleanup. Two consecutive *zero-net-new* rounds across all six sub-reviews are not yet achieved by this sub-review at round-11.

Recommend: either (a) accept these as "same class as known-deferred dead-buffer MEDs" and treat round-11 as effectively clean for gating purposes, or (b) one-line cleanup commit removing the 9 dead allocations + dtor frees and the `h_svd_tmp_` declaration, then re-run round-12.
