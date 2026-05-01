# Vertical review — dmrg2 family — 2026-05-01 round-14

HEAD: `f5c0617`. Baseline: round-13 at `a457fdc` /
`reviews/vertical-dmrg2-20260430-round13.md`. Scope per brief:
`gpu-rocm/dmrg2-gpu/` and `gpu-rocm/dmrg2-gpu-opt/`. Only one commit
(`f5c0617`) lands between baseline and HEAD — a pure-cosmetic round-13
nit cleanup that touches two header files.

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | no member additions/deletions; nothing to re-grep |
| B. Behavioral diff | DONE | no impl changes since round-13; -gpu↔-opt deltas all intentional and unchanged |
| C. Docstring verification | DONE | the new f5c0617 h_svd_* docstring claim verified against impl 1263-1285 |
| D. clangd filter | N-A | no clangd available locally; A subsumes |
| E. Absence-naming brief | FOLLOWED | -gpu and -opt expected-feature checklists pass identically to round-13 |
| F. Workspace-aliasing audit | DONE | 5 shared buffers re-checked; 0 OVERRUN; CR-D1 sizing math re-derived (see below) |
| G. Sibling fix-propagation | DONE | 1 round-13 fix traced (f5c0617 nit pair); both halves landed; 0 MISSING |

## Regression-watch verification (round-14 brief)

### f5c0617 — orphan comment removal in dmrg2-gpu

`dmrg2_gpu.h` line 154 area: the orphan `// CPU workspace (for
receiving GPU SVD results and truncation/scaling)` comment is **GONE**.
File now reads: line 152 closes the SVD device-scalar block, blank
line 153, then line 154 = `// RSVD workspace (allocated only when
opts_.rsvd is on)`. Round-13 nit cleared.

### f5c0617 — h_svd_* docstring tightening in dmrg2-gpu-opt

`dmrg2_gpu_opt.h:189-194` now reads:
```
// CPU SVD workspace — used by the use_cpu_svd_ opt-in fallback path
// (impl ~1263-1283) and by the init-time LAPACK workspace query. The
// default GPU code path bypasses these; they only allocate/touch when
// use_cpu_svd_ is set or chi_max changes during init.
```
Verified against impl: line 1259 enters `if (use_cpu_svd_) { ... }`
which reads `h_svd_A_`, `h_svd_S_`, `h_svd_U_`, `h_svd_Vh_`,
`h_svd_work_`, and conditionally `h_svd_rwork_`. The block extends
1263-1285 (the docstring's "~1263-1283" range straddles correctly with
the tilde — final memcpy lands at 1283-1285). Workspace-query
allocation site verified at impl ~225 (initial `resize` of these
vectors based on LAPACK lwork query). The new docstring is **accurate**
and a strict improvement over the prior "legacy / init-time only"
phrasing that round-13 flagged. Both halves of f5c0617 land cleanly.

### Round-13 axis-3 lesson — docstring promise = half-fix

The round-14 brief explicitly cautioned against docstring-only fixes.
In this case the f5c0617 commit **only** touches docstring/comment text
(no code change), which is intentional: the runtime behavior was
already correct (use_cpu_svd_ path was alive in round-12 and round-13),
the docstring was lagging. The fix is a docstring/code reconciliation,
not a half-fix. Verified: the *code* the docstring describes already
existed (impl 1259-1285 at round-12) and remains unchanged.

### Round-13 carry-forward (still holds)

- **M-opt-davidson-toggle** (bc3fcd0 dual-half): dmrg2-gpu-opt.h:73-81
  setter intact; ctor at impl:74 still sets
  `lanczos_graph_was_user_enabled_ = true` symmetrically. INTACT.
- **M1-final guard consolidation** (0b9fccf): dmrg2-gpu uses shared
  `PointerModeGuard` at impl:1124, 1255 (include at impl:5).
  dmrg2-gpu-opt: zero pointer-mode toggles in -opt impl (mode-neutral
  inner loop). INTACT.
- **M14-final dead set_quiet stub removal**: dmrg2-gpu-opt.h has no
  set_quiet; dmrg2-gpu.h:44 keeps the stub (test driver calls it 6×).
  INTACT. Note: round-13's claim that dmrg2-gpu-base.h:63 also has a
  set_quiet stub was stale — the base header has no such stub now and
  its test driver has no set_quiet calls; correctly absent.
- **CR-D1 dav_work_sz** (round-8): impl:265-268 still computes
  `max(theta_size_max·b + max_sub·b, max_sub·max_sub)` and allocates
  both `d_dav_work_` and `d_dav_work2_` at this size. Inner loop
  uses concurrent residuals at offset 0 and overlap at offset
  `n_new·dim` (impl:1607-1608); restart path at 1659-1667 reuses
  sequentially. Allocation ≥ required. **OK**.
- **Round-6 dual-stream env-pipeline + direction-L MPS-write reorder**:
  -gpu impl:1413, 1438 / -opt impl:1391, 1421 record
  `event_canon_ready_` immediately after MPS write; direction-L writes
  MPS[site+1]=Vh BEFORE U·S absorb in both. INTACT.
- **D_PAD precompute_fused_mpo OOB**: -opt impl:511-541 loops bounded
  by `D_act = D_mpo_actual_`; no write past `D_use·dd·dd·D_use`. INTACT.
- **rocsolver_syevd (round-7 H6)** + **non-blocking stream flag
  (H1-ext-gpu)**: both intact (no diff hunks touched).

## CRITICALS

None.

## HIGHS

None.

## MEDIUMS

None within charter (-gpu, -opt). Round-13 had 0 mediums; round-14
introduces none.

## NITS

None new. The two round-13 carry-over nits (orphan comment in
dmrg2_gpu.h:154 and stale h_svd_* docstring in dmrg2_gpu_opt.h:189-194)
were **resolved** by f5c0617. The dmrg2 family is now nit-clean within
charter scope.

## FALSE POSITIVES VERIFIED

- The new docstring's "impl ~1263-1283" range looks one line short of
  the actual block (real range is 1263-1285 including the final
  hipMemcpyAsync of `h_svd_Vh_`). The tilde explicitly signals
  approximation, and reading the block confirms the intent. Not a
  defect; cosmetic-only and the brief explicitly endorsed the wording.

## SIBLING-PROPAGATION OBSERVATION (out-of-charter, awareness)

Round-13's flag for dmrg2-gpu-base still holding inline
`rocblas_set_pointer_mode` calls (`dmrg2_gpu_base_impl.h:585, 647, 677,
682`) without the shared `PointerModeGuard` is **unchanged** by
f5c0617 (which only touched -gpu/-opt header comments). Out of scope
for round-14 vertical-dmrg2; defer to horizontal-base reviewer per
round-13 transparency note. Not counted against this round's tally.

## SUMMARY

Round-14 returns **0 critical, 0 high, 0 medium, 0 nits** for the
dmrg2 family (-gpu, -opt). The single commit since the round-13
baseline (`f5c0617`) is a clean, audit-driven cosmetic pair: orphan
comment removed from dmrg2_gpu.h:154, stale h_svd_* docstring tightened
in dmrg2_gpu_opt.h:189-194 to correctly describe the use_cpu_svd_
opt-in fallback path it gates. The new docstring claim verifies
against impl:1259-1285 — the block straddles 1263-1283 with the tilde
absorbing the 2-line tail. The round-13 axis-3 lesson ("docstring
promise = half-fix") is not violated here because the *code* path was
already correct in round-12; this commit is a docstring/code
reconciliation, not a behavior change. All seven round-13 carry-forward
watchlist items remain intact: davidson-toggle, M1-final guard
consolidation, M14-final dead-stub cleanup, CR-D1 dav_work_sz,
round-6 direction-L reorder, D_PAD OOB fix, rocsolver_syevd, and
non-blocking streams. This is the fourth consecutive zero-finding
sub-review for the dmrg2 family; the orchestrator can continue
declaring it stable. Block GPU run? **NO**, family is ready.

Self-audit: techniques A–G all completed (D N-A); regression-watch
list explicitly traced for f5c0617 plus the round-13 carry-forward
items; -base out-of-scope observation flagged for horizontal-base
reviewer; the "docstring promise = half-fix" axis-3 lesson explicitly
addressed by verifying the *code* (not just the new docstring) at
impl:1259-1285. Verdict: READY.

(Length: ~770 words.)
