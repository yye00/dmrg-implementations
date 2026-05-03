# Vertical review — dmrg2 family — 2026-04-28 round-11

HEAD: `1d44d89`. Scope: `gpu-rocm/dmrg2-gpu-{base,_,opt}/src/*` plus
`gpu-rocm/common/*`. Round-11 is a gating round before G1.

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | -base 25 members all live; -gpu 4 host SVD buffers + 1 device buffer dead; -opt 1 device buffer + 1 host buffer dead |
| B. Behavioral diff | DONE | -base ↔ -gpu ↔ -opt diffs all intentional (Lanczos vs Davidson, sparse-MPO, RSVD, dual-stream, D_PAD, lanczos_graph) |
| C. Docstring verification | DONE | All claims verified — round-7 H5 correction (host-side WW precompute in all tiers) intact in -base.h; -gpu and -opt headers accurate; round-6 dual-stream docstring matches code |
| D. clangd filter | N-A | No clangd available locally |
| E. Absence-naming brief | FOLLOWED | Per-tier expected-features checklist used; no MISSING items |
| F. Workspace-aliasing audit | DONE | 6 shared buffers checked (d_T1_, d_T2_, d_T1_env_, d_T2_env_, d_dav_work_, d_dav_work2_); 0 OVERRUN |
| G. Sibling fix-propagation | DONE | 5 round-10/9 fixes traced; **round-9 MED-base-1 (dead d_svd_work_) propagation MISSING in -gpu and -opt** |

A-G all completed.

## Watch-list verification

- **M-opt-rsvd-env (round-10)**: `dmrg2-gpu-opt` ctor at impl:60 has
  `use_rsvd_ = opts_.rsvd;` immediately after `opts_.load_from_env()`.
  Confirmed intact.
- **M4-W (round-9 incl. d_WW_)**: `set_mpo` in -base impl:222-247 and
  -opt impl:454-501 use the `if (ptr) hipFree(ptr); hipMalloc(&ptr,...)`
  guard for d_mpo_tensors_, d_W_left_, d_W_right_. `precompute_fused_mpo`
  in -gpu impl:651-693 and -opt impl:546-587 has the matching guard for
  d_WW_, d_WW_nnz_rows_, d_WW_nnz_cols_. All intact.
- **H1-ext-gpu (non-blocking flag)**: -gpu impl:198,201 and -opt
  impl:107,108 both create stream_ + stream_env_ with
  `hipStreamNonBlocking`. -base impl:34 single stream non-blocking.
  Intact.
- **dav_work_sz (CR-D1, round-8)**: -opt impl:267-272 sizes
  `dav_work_sz = max(theta_size_max·b + max_sub·b, max_sub·max_sub)`.
  Inner loop residual+overlap aliasing at impl:1610-1611 fits
  (region B at offset n_new·dim, size k·n_new ≤ max_sub·b). Restart
  X_keep at impl:1664-1670 reuses d_dav_work_ sequentially with the
  prior overlap region — fits (dim·keep ≤ theta_size_max·b). OK.
- **rocsolver_syevd port (round-7 H6)**: -opt impl:1528-1530 calls
  `Traits::rocsolver_syevd` on stream_-bound rocblas_h_. Intact.
- **Round-6 dual-stream wire-up**: -gpu records event_canon_ready_ at
  impl:1421 (R) and 1446 (L), waits at impl:1497, 1524. -opt records
  at impl:1394 and 1424; waits at impl:1774 and 1807. All intact.
- **Round-6 direction-L svd_split write-order reorder**: -gpu impl:1431-1454
  and -opt impl:1406-1435 both write MPS[site+1] = Vh BEFORE the U·S
  absorb into MPS[site], with hipEventRecord between them. Intact.

All watch-list items hold.

## CRITICALS

None.

## HIGHS

None.

## MEDIUMS

### MED-d_svd_work-gpu (technique G — round-9 propagation gap)

`gpu-rocm/dmrg2-gpu/src/dmrg2_gpu.h:149` declares `Scalar* d_svd_work_;`
and impl:314 hipMallocs it `theta_size_max_ * sizeof(Scalar)`; impl:436
frees. Zero other references. The two-site `svd_split` writes
S·Vh / U·S directly into adjacent MPS tensors via
`scale_rows_by_diag_kernel` / `scale_cols_by_diag_kernel`, never touching
this scratch. Same dead-infrastructure class as round-9 MED-base-1
(dmrg2-gpu-base) — the round-9 fix was applied to -base only and never
propagated. ~80MB of GPU memory wasted at challenge-size theta. Defect
class identifier: same as MED-base-1.

### MED-d_svd_work-opt (technique G — round-9 propagation gap)

`gpu-rocm/dmrg2-gpu-opt/src/dmrg2_gpu_opt.h:169` + impl:196 + impl:329.
Same pattern as -gpu: alloc / free, zero use. The opt's
`svd_split_fallback` `use_cpu_svd_` path uses host `h_svd_*`; the
on-device path uses `d_svd_U_`/`d_svd_S_`/`d_svd_Vh_` directly. The
RSVD path uses `d_rsvd_*`. `d_svd_work_` is never read or written
between alloc and free. Same defect class as MED-d_svd_work-gpu.

### MED-h_svd_buffers-gpu (technique A — dead host scaffolding)

`gpu-rocm/dmrg2-gpu/src/dmrg2_gpu.h:156-157` declares `h_svd_U_`,
`h_svd_Vh_`, `h_svd_tmp_`, `h_svd_S_` (vectors). Impl:320-323 resizes
them in the ctor. Zero subsequent references. -gpu has no
`use_cpu_svd_` opt-in path (that's -opt-tier only), so these were
intended for the legacy host SVD that was retired when on-device
`rocsolver_gesvd_auto` landed. Dead host scaffolding (~few hundred MB
of host RAM at challenge sizes). Same defect class as round-7
dmrg2-gpu dead-stream and round-9 MED-base-1. Cosmetic — defer.

### MED-h_svd_tmp-opt (technique A — dead host scaffolding)

`gpu-rocm/dmrg2-gpu-opt/src/dmrg2_gpu_opt.h:193` includes `h_svd_tmp_`
in the `std::vector<Scalar>` group. Impl:233 resizes; never read or
written subsequently. Distinct from `h_svd_A_`/`h_svd_U_`/`h_svd_S_`/
`h_svd_Vh_`/`h_svd_work_` which ARE used by the `use_cpu_svd_` path
(impl:1266-1286). The leftover `h_svd_tmp_` was a workspace for a
truncate-then-copy step that was inlined when on-device
`extract_cols_kernel` / `scale_*_by_diag_kernel` replaced the host
truncation. Cosmetic — defer.

## NITS

- Pre-existing carry-overs noted in round-10 report:
  `set_quiet(bool) {}` no-ops in all three headers (intentional for
  API surface parity); cosmetic comment improvements suggested in
  round-10 on `dav_work_sz` documentation already applied.
- `dmrg2-gpu/src/dmrg2_gpu.h:159-167` lists d_rsvd_omega_ etc. as
  "(allocated only when opts_.rsvd is on)" but the round-5 A8 fix
  made allocation eager in -opt; the comment in -gpu still claims
  gated allocation — verify against current impl. (Confirmed: -gpu
  impl:413-422 IS gated on `opts_.rsvd`, comment matches code.)

## FALSE POSITIVES VERIFIED

- `lanczos_graph_was_user_enabled_` only 1 hit in -opt impl. Verified
  the setter `set_use_davidson` in -opt header reads it (line 78). Live.
- `d_T1_env_` / `d_T2_env_` in -opt only 4 hits each; verified used in
  update_left_env / update_right_env on stream_env_ (impl:838-880,
  945-985 in -gpu; matching offsets in -opt). Live.
- `D_mpo_actual_` 7 hits in -opt; used by D_PAD path for host-MPO
  indexing (impl:457, 514, etc.). Live.
- `chi_max_user_` 7 hits; used in MFMA-16 padding init message and
  chi_max gating. Live.
- The round-10 H10-multi-WW-leak fix is in pdmrg-multi-gpu (out of
  scope for this vertical); no propagation needed to dmrg2 family.
- The round-10 M-opt-rsvd-env fix verified at impl:60.
- `precompute_fused_mpo` is host-build + H2D in all tiers per the
  round-7 H5 docstring correction. Verified.

## SUMMARY

Round-11 returns **0 critical, 0 high, 4 medium, 0 nit** for the
dmrg2 family. All four mediums are dead-code propagation gaps from
the round-9 MED-base-1 fix that landed only in dmrg2-gpu-base —
sibling -gpu and -opt still carry the dead `d_svd_work_` and host
SVD scaffolding. **Technique G caught these (lonely-fix pattern),
which is exactly what G exists for.**

The findings do NOT block G1: dead host/device scratch wastes memory
but does not affect numerical correctness or default-path performance
on challenge sizes (challenge-size theta_size_max is ~chi_max²·d² —
the wasted `d_svd_work_` is theta_size_max Scalars per binary, ~80MB
real / 160MB complex at chi=512, irrelevant against the ~16GB total
GPU footprint of the env tensors).

The pattern is a methodology success: round-9's fix in -base did not
propagate, round-10 did not catch it (dmrg2-vertical was clean +
nit-only in round-10), but round-11's mandatory technique-G sweep
flagged it. Mediums can be cleaned up post-G1 in a single batched
PR alongside the round-10 deferred dead-buffer findings in
pdmrg-gpu-opt.

**Verdict on round-11 cleanliness for the dmrg2 family**: 4 fresh
mediums, no criticals or highs. Block GPU run? **NO** — mediums are
non-impacting dead infrastructure. **NOT a clean round** in the
"two consecutive zero-finding sub-reviews" sense; recommend a
6th-finding follow-up commit applying the round-9 MED-base-1 fix
to siblings dmrg2-gpu and dmrg2-gpu-opt before declaring G1-ready.
