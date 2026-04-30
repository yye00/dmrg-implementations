# Vertical review — dmrg2 family — 2026-04-30 round-13

HEAD: `a457fdc`. Baseline: round-12 at `8b7a68e` /
`reviews/vertical-dmrg2-20260430-round12.md`. Scope per brief:
`gpu-rocm/dmrg2-gpu/` (-gpu reference) and
`gpu-rocm/dmrg2-gpu-opt/` (-opt). The -base tier is out of scope this
round; only one commit (`0b9fccf`) lands between baseline and HEAD
and it is a small medium-cleanup pass.

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | no new dead members; round-12 cleanup persists |
| B. Behavioral diff | DONE | -gpu ↔ -opt deltas all intentional (Davidson primary, RSVD/D_PAD/MFMA-16, sparse-MPO, env-stream batched ptrs); ≈ same as round-12 |
| C. Docstring verification | DONE | dmrg2-gpu-opt.h:13-35 (algorithm differences), :64-66 (J2 setter parity), :122-128 (dual-stream pipeline) all match impl |
| D. clangd filter | N-A | no clangd available locally; A subsumes |
| E. Absence-naming brief | FOLLOWED | -gpu and -opt expected-feature checklists pass |
| F. Workspace-aliasing audit | DONE | 5 shared buffers re-checked (d_T1_, d_T2_, d_T1_env_, d_T2_env_, d_dav_work_/d_dav_work2_); 0 OVERRUN |
| G. Sibling fix-propagation | DONE | 2 fixes traced (M1-final guard consolidation, M14-final stub removal); 0 MISSING within charter scope, 1 surfaced for -base awareness |

## Regression-watch verification (round-13 brief)

### M1-final — guard consolidation

The 0b9fccf commit promoted three inline pointer-mode guard structs
(`DmrgPointerModeGuard`, `AsvdPointerModeGuard`,
`RlbfgsPointerModeGuard`) to a single shared
`common/pointer_mode_guard.h` (`PointerModeGuard`). The dmrg2 family
never had its own `Dmrg2PointerModeGuard` struct, so this commit
should leave dmrg2 functionally untouched and the file diff confirms:

- **dmrg2-gpu**: `dmrg2_gpu_impl.h:5` includes
  `../../common/pointer_mode_guard.h`. `dmrg2_gpu_impl.h:1124, 1255`
  use `PointerModeGuard` already (no inline struct, no rename
  needed). `grep -nE 'rocblas_set_pointer_mode|struct.*Guard'
  dmrg2_gpu*.h` → 0 hits outside the include. INTACT.
- **dmrg2-gpu-opt**: no rocblas_set_pointer_mode calls anywhere in
  `dmrg2_gpu_opt_impl.h` (the variant uses pointer_mode_device only
  via Lanczos / Davidson, but those run inside an outer caller that
  manages mode in -gpu's pdmrg-multi callers; -opt itself is mode-
  neutral). INTACT — no guard needed.

**Verdict: M1-final is a no-op for the dmrg2 charter; no regression.**

### M14-final — dead set_quiet stub removal

Confirm the `void set_quiet(bool) {}` no-op is gone from
`dmrg2-gpu-opt.h`:

- **dmrg2-gpu-opt.h**: `grep -n set_quiet` → no hits. The diff at
  `dmrg2_gpu_opt.h:82` cleanly removes the stub line; the surrounding
  `set_rsvd` / `chi_L` / `chi_R` block is preserved. **GONE — fix
  intact.**

The dmrg2-gpu and dmrg2-gpu-base headers still declare `set_quiet`
(dmrg2_gpu.h:44, dmrg2_gpu_base.h:63) — verified deliberate: their
test drivers (`test_dmrg2_gpu.cpp:214,257,301,348,395,448`) actively
call `dmrg.set_quiet(quiet)` six times each. Removing those would
break the build, matching the M14-final commit-message guidance
("Test drivers don't call these — only pdmrg-gpu-opt needs the stub
[for -opt]"). Inverted for dmrg2: -opt's test driver does NOT call
it, so -opt's stub was dead; -gpu/-base's stubs are alive.

### Round-12 watchlist (carry-forward)

- **M-opt-davidson-toggle (bc3fcd0 dual-half fix)**: dmrg2-gpu-opt.h:73-81
  `set_use_davidson` body unchanged (round-12 verified); ctor at
  dmrg2-gpu-opt_impl.h:74 still sets
  `lanczos_graph_was_user_enabled_ = true` when ctor disables
  `opts_.lanczos_graph` for the Davidson-default path. INTACT.
- **c3d3e50 dead-buffer cleanup**: round-12 confirmed
  `h_svd_*` survivors in dmrg2-gpu-opt.h:191-194 are alive in the
  CPU-SVD fallback path at impl:1263-1283 (`use_cpu_svd_` opt-in
  branch). No round-13 changes touched this code. INTACT.
- **dav_work_sz CR-D1 sizing (round-8)**: dmrg2-gpu-opt_impl.h:259-269
  still computes `dav_work_sz = max(theta_size_max·b + max_sub·b,
  max_sub·max_sub)`. Inner-loop residuals at `d_dav_work_` (line
  1607) plus overlap at `d_dav_work_ + n_new·dim` (line 1608) fit;
  restart at line 1663-1667 reuses sequentially. OK.
- **Round-6 dual-stream env-pipeline + direction-L MPS-write reorder**:
  -gpu records `event_canon_ready_` at impl:1413/1438; -opt at
  impl:1391/1421. Direction-L writes MPS[site+1]=Vh BEFORE U·S
  absorb in both. INTACT (round-12 line numbers match).
- **D_PAD precompute_fused_mpo OOB fix**: -opt impl:515-541 loops
  bounded by `D_act` (`D_mpo_actual_`); no write past
  `D_use·dd·dd·D_use`. INTACT.
- **rocsolver_syevd (round-7 H6) + non-blocking stream flag
  (H1-ext-gpu)**: both intact (no diff hunks touched).

## CRITICALS

None.

## HIGHS

None.

## MEDIUMS

None within charter (-gpu, -opt). Round-12 had 0 mediums in the
dmrg2 family; round-13 introduces none.

## NITS

- Round-12 nit `dmrg2_gpu.h:154` (orphan "// CPU workspace" comment
  with no following member declarations) is unchanged — defer to
  next pass or delete in a tidy-up commit. Cosmetic only.
- Round-12 nit `dmrg2_gpu_opt.h:189-194` "CPU SVD workspace (legacy
  — only used by the init-time workspace query; runtime SVD is
  fully on-device)" comment slightly understates: these vectors are
  also read by `use_cpu_svd_` opt-in path at impl:1263-1283, not
  just init. Cosmetic; default-path claim is correct.

## FALSE POSITIVES VERIFIED

- `set_quiet` stubs in dmrg2-gpu.h:44 and dmrg2-gpu-base.h:63 look
  like the M14-final cleanup missed two siblings — verified alive:
  test_dmrg2_gpu.cpp calls `dmrg.set_quiet(quiet)` 6 times. The
  M14-final brief explicitly retains stubs whose test driver calls
  them. Not dead, not a defect.

## SIBLING-PROPAGATION OBSERVATION (out-of-charter, awareness)

`dmrg2-gpu-base/src/dmrg2_gpu_base_impl.h:585, 647, 677, 682` still
uses inline `rocblas_set_pointer_mode(device)` / `(host)` pairs
without the shared `PointerModeGuard`. The M1-final commit
consolidated -gpu / accurate_svd / rlbfgs siblings to the shared
guard but did not touch any -base file. dmrg2-gpu-base is
out-of-scope this round (round-13 brief explicitly: "-gpu and -opt").
Flagged here for the round-14 horizontal-base reviewer to
adjudicate — could be intentional (the -base tier deliberately
keeps inline pattern as an "engineering style" baseline) or a
genuine missed propagation. NOT counted against this round's
finding tally; mentioned for technique-G transparency.

## SUMMARY

Round-13 returns **0 critical, 0 high, 0 medium, 2 nits** for the
dmrg2 family (-gpu, -opt). Both M1-final and M14-final regression-
watch items verify clean: M1-final is a no-op for dmrg2 (no
inline guard struct existed in dmrg2 to consolidate; the shared
include at impl:5 was already in place pre-round-13). M14-final
correctly removed the dead `set_quiet` stub from dmrg2-gpu-opt.h
while preserving the live stubs in -gpu/-base whose test drivers
still call them. All round-12 carry-forward watchlist items
(davidson-toggle, c3d3e50 cleanup, dav_work CR-D1, dual-stream
env-pipeline, D_PAD OOB, rocsolver_syevd, non-blocking streams)
are intact. This is the third consecutive zero-finding sub-review
for the dmrg2 family — recommend the orchestrator continue
declaring it stable. Block GPU run? **NO**, family is ready.

Self-audit: techniques A–G all completed (D N-A); regression-watch
list explicitly traced; -base out-of-scope finding flagged for
horizontal-base reviewer; 2 nits carry over from round-12 with no
new defects introduced. Verdict: READY.

(Length: ~770 words.)
