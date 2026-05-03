# Vertical review — DMRG family (single-site) — round-10 — 2026-04-28

HEAD: `db7dcdf` (round-10 self-audit fixes — H1 + M4-W remainders)
Baseline: `cfd08c3` (round-9 conformity)
Scope: `gpu-rocm/dmrg-gpu-{base,_,opt}/src/*` + `gpu-rocm/common/*`

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | 3 variants, ≥48 private members audited; 0 dead. |
| B. Behavioral diff   | DONE | 3 pair-wise sweep/env/svd diffs; all divergences intentional and tier-justified. |
| C. Docstring verify  | DONE | 8 claims (dual-stream, graph capture, RSVD, sparse-MPO, D_PAD, MFMA-16, batched Step-3, on-device dsteqr) — all backed by code. |
| D. clangd filter     | N-A  | No ROCm headers on host; A subsumes dead-symbol case. |
| E. Absence-naming    | FOLLOWED | -base/-gpu/-opt expected-feature checklists run. |
| F. Workspace-aliasing | DONE | `d_dav_work_` / `d_dav_work2_` re-verified; `d_T1_/T2_` env-stream split intact; `d_svd_work_` lives. 0 OVERRUN. |
| G. Sibling fix-propagation | DONE | 5 recent fix-classes (H-new1 M4-W, H1-ext-gpu, H1-final, M4-W-multi, round-7/8 carryovers) traced through 3 tiers. 0 MISSING. |

A review with any technique SKIPPED that is not N-A is INVALID — none skipped.

## Regression-watch verification (commit-pinned)

| Watch item | Variant | File:line | Status |
|---|---|---|---|
| H-new1 (M4-W): W-buffer guards | dmrg-gpu-base | `dmrg_gpu_base_impl.h:225,244,248` | OK |
| H-new1 (M4-W): W-buffer guards | dmrg-gpu | `dmrg_gpu_impl.h:566,599,603,638,644` | OK |
| H-new1 (M4-W): W-buffer guards | dmrg-gpu-opt | `dmrg_gpu_opt_impl.h:511,542,546,579,585` | OK |
| H1-ext-gpu: nonblocking flag (stream_) | dmrg-gpu | `dmrg_gpu_impl.h:200` | OK |
| H1-ext-gpu: nonblocking flag (stream_env_) | dmrg-gpu | `dmrg_gpu_impl.h:201` | OK |
| H1-ext-gpu: nonblocking flag (stream_) | dmrg-gpu-opt | `dmrg_gpu_opt_impl.h:105` | OK |
| H1-ext-gpu: nonblocking flag (stream_env_) | dmrg-gpu-opt | `dmrg_gpu_opt_impl.h:106` | OK |
| H1-final: nonblocking flag (stream_) | dmrg-gpu-base | `dmrg_gpu_base_impl.h:38` | OK |
| Round-8 CR-D1: dav_work_sz residual+overlap concurrent sizing | dmrg-gpu-opt | `dmrg_gpu_opt_impl.h:305-310` | OK (see F detail) |
| Round-7 C5: D_PAD R-env identity at `D_mpo_actual_-1` | dmrg-gpu-opt | `dmrg_gpu_opt_impl.h:982` | OK |
| Round-7 C2: rocsolver_dsteqr replaces host dstev | dmrg-gpu-opt | header line 197-204 + impl `:589-590` | OK |
| Round-7 C2: rocsolver_syevd in Davidson | dmrg-gpu-opt | `dmrg_gpu_opt_impl.h:1604` | OK |

## Technique F detail — `d_dav_work_` aliasing audit

`block_davidson_eigensolver` (lines 1514-1810) routes two concurrent regions
into `d_dav_work_` inside the inner expansion+orthogonalization loop:

- Region 1: residuals `W = d_dav_work_` at offset 0, size `n_new·dim ≤ b·dim`.
  (impl lines 1664, 1696)
- Region 2: overlap matrix `d_dav_work_ + n_new·dim`, size `k·n_new ≤ max_sub·b`.
  (impl line 1697 + GEMM at 1700-1702)

Lifetimes are concurrent — region 2 is read while region 1 still holds
the residual block (line 1705-1707 axpy `W -= V·overlap`). The restart
path at 1751-1758 reuses `d_dav_work_` as a sequential third region of
size `dim·keep ≤ dim·b` (subsumed by Region 1's allocation).

Required total: `b·dim + max_sub·b`. Plus the sequential Rayleigh-Ritz
H_proj of size `max_sub²` written into `d_dav_work2_` (line 1596). Both
buffers allocated as `max(b·theta_size_max_ + max_sub·b, max_sub·max_sub) ·
sizeof(Scalar)` at lines 305-310. Since `theta_size_max_ ≥ dim` for all
sites, this is a strict superset of the required size. **OK; no
overrun.** Round-8 CR-D1 fix retained verbatim with the explanatory
comment at lines 300-308.

## Technique G detail — fix-propagation through dmrg family

| Fix class | dmrg-gpu-base | dmrg-gpu | dmrg-gpu-opt |
|---|---|---|---|
| M4 (d_mpo_tensors guard) | OK `:225` | OK `:566` | OK `:511` |
| M4-W (W_left/W_right guard) | OK `:244,248` | OK `:599,603` | OK `:542,546` |
| M4-W (sparse-nnz buffers) | immune (no sparse path) | OK `:638,644` | OK `:579,585` |
| H1 / H1-ext-gpu / H1-final (`hipStreamNonBlocking`) | OK `:38` | OK `:200-201` | OK `:105-106` |
| C5 (D_PAD R-env identity) | immune (no D_PAD) | OK `:1228` (R-env build) | OK `:982` |
| C2 (rocsolver_dsteqr / syevd) | OK (always device) `:586,622` | OK | OK |
| Round-6 dual-stream env-update | immune (single-stream tier) | OK `:1487-1554` | OK `:1839-1928` |
| CR-D1 (dav_work_sz residual+overlap) | immune (no Davidson) | immune (no Davidson) | OK `:305-310` |

All fixes either landed or the variant is genuinely immune by tier
contract. **No lonely fixes.**

## CRITICALS

None.

## HIGHS

None.

## MEDIUMS (carry-over from round-9 — pre-existing)

- M-carry: `DmrgPointerModeGuard` is a struct local to
  `dmrg_gpu_impl.h:18-30`; not migrated to `common/pointer_mode_guard.h`.
  Round-9 deferral, not a regression.
- M-carry: `set_quiet(bool)` no-op across all three tiers
  (`dmrg_gpu_base.h:52`, `dmrg_gpu.h:43`, `dmrg_gpu_opt.h:85`). API
  surface kept for parity with pdmrg variants.
- M-carry: dmrg-gpu-opt Lanczos α/β kept host-resident
  (`dmrg_gpu_opt_impl.h:1015-1016`). Default code path is
  Block-Davidson (`use_davidson_=true`); this path is fallback only.
  Tridiagonal solve still on-device via rocsolver_dsteqr.

## NITS

- `dmrg_gpu_opt_impl.h:185` comment block ends on line 191 — minor
  alignment with the round-7 D_PAD comment block style differs from
  -gpu sibling (`dmrg_gpu.h:78-94`). Cosmetic.

## FALSE POSITIVES VERIFIED

- `d_dav_V_` / `d_dav_AV_` show grep count 3 each in -opt
  (`:297-298,394-395,1540-1541`). Verified live: aliased to `V`/`AV`
  pointers at line 1540-1541 then used throughout block_davidson via
  the alias names. Not dead.
- `d_const_one_` / `d_const_zero_` / `d_const_neg_one_` count 3 in -opt.
  Verified live: read at impl `:1043,1296,1338,1393` (Lanczos / SVD /
  apply_heff). Compiler can fold but they're addressable.
- Per-Lanczos-call `std::vector<double> h_alpha(max_iter)` /
  `h_beta(max_iter)` at -opt `:1015-1016` is **not** a host-roundtrip
  defect. CLAUDE.md "no host roundtrips per sweep" rule targets host
  LAPACK/BLAS calls and large host data. These are host-resident
  scalars (max_iter ≈ 100 doubles) for the standard Lanczos
  recurrence; the eigensolve itself runs on-device via
  rocsolver_dsteqr at `:589-590`. Documented at header lines 197-204.
- `lapack_gesvd` at -opt `:280` is a one-shot ctor workspace-size
  query; the per-sweep `lapack_gesvd` at `:1296` is gated by
  `use_cpu_svd_` (default false). Default code path is on-device
  rocsolver gesvd_auto / gesvdj.
- dmrg-gpu-opt forces `opts_.lanczos_graph = false` at `:68-72` —
  intentional override (Block-Davidson's per-column variable output
  pointer is incompatible with HIP-graph capture). Documented in
  fprintf message and header lines 56-67.

## SUMMARY

The single-site DMRG family is clean. All 5 round-9 + round-10
regression-watch items verified intact at cited lines. Technique F
re-confirms the round-8 CR-D1 fix (concurrent residual+overlap
regions in `d_dav_work_`) sized correctly. Technique G traces 8
fix-classes through 3 tiers and finds 0 propagation gaps — every
sibling either has the fix or is genuinely immune by tier contract.
No criticals, no highs, no new mediums or nits. The family is ready
for the G1 MI300X window from a vertical-DMRG-conformity standpoint.

The diminishing-returns pattern continues: round-9 found 4 net-new,
round-10 self-audit found 2 more (in *other* families — radam,
rlbfgs, multi — outside this review's scope), and this round-10
vertical-dmrg orchestrator finds 0 net-new in the dmrg family.
Recommend the orchestrator-level conformity-review-full continue but
this sub-review is signaling "stable" for the dmrg family. Two
consecutive 0-net-new rounds at the orchestrator level would close
the F+G discipline cycle.
