# Horizontal review — -gpu tier — round 11 (2026-04-28)

HEAD: `1d44d89` (gating round, post-round-10).

Scope: `gpu-rocm/{dmrg,dmrg2,pdmrg}-gpu/src/*` plus
`gpu-rocm/common/{scalar_traits,gpu_opts,accurate_svd_gpu,hip_check,pointer_mode_guard}.h`.

Diff vs round-10 baseline (`db7dcdf`): **zero code changes in scope.**
The only post-round-10 commit is `1d44d89`, which adds
`reviews/conformity-20260428-round10.md` (docs only). Round-11 thus
re-audits the same code that round-10 horizontal-review-gpu cleared
with 0 net-new findings, plus the 2026-04-28 fixes from round-10
(`4d8924d`).

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | 4 dead host buffers per variant in dmrg-gpu + dmrg2-gpu (`h_svd_U_/Vh_/S_/tmp_`), allocated in ctor, never read. Same defect class as round-9 MED-base-1 / round-10 MED-pdmrg-opt. |
| B. Behavioral diff | DONE | Sibling sweep functions identical: dmrg-gpu and dmrg2-gpu each have 6 `hipEventRecord` + `hipStreamWaitEvent` sites in sweep + update functions, with identical ordering. pdmrg-gpu's per-segment stream model is the documented intentional divergence. |
| C. Docstring verification | DONE | All three header docstrings match implementation. dmrg-gpu's "Dual-stream pipeline" comment in sweep_left_to_right matches the event/wait wiring. pdmrg-gpu's "P segments, each assigned to a HIP stream" matches `streams_[k]`. |
| D. clangd filter | N-A (still no ROCm headers on host — falls back to A). |
| E. Absence-naming brief | FOLLOWED | All 11 expected -gpu features present in all 3 variants (modulo intentional algorithmic asymmetries). |
| F. Workspace-aliasing audit | DONE | `d_T1_`/`d_T2_`/`d_T1_env_`/`d_T2_env_` sized at `t_max = D_mpo_*d*chi*chi` (1-site) / `D_mpo_*dd*chi*chi` (2-site). Env scratch is dedicated (separate alloc), not aliased with main scratch — no concurrent-region overrun risk. Block-Davidson is -opt-only (out of scope here). |
| G. Sibling fix-propagation | DONE | Round-9 + round-10 fixes traced. All -gpu siblings have the relevant guards. No `MISSING` entries. |

A-G: all DONE or N-A with rationale. Review valid.

## Watch-list verification (round-9 + round-10 fixes)

| Fix class | Source variant | dmrg-gpu | dmrg2-gpu | pdmrg-gpu | Status |
|---|---|---|---|---|---|
| M4-W: `set_mpo` `if (d_X) hipFree` guards | round-9 self-audit | impl.h:566/599/603 + sparse rows/cols | impl.h:569/599/603 | impl.h:743/773/777 + WW at 869, sparse rows/cols at 808/814/899/905 | **all present** |
| H10-multi-WW-leak: WW guard inside `precompute_fused_mpo` | round-10 (multi-gpu) | N/A (single-site, no WW) | impl.h:651 `if (d_WW_[bond]) hipFree` | impl.h:869 `if (d_WW_[bond]) hipFree` | **immune (dmrg-gpu) / fixed (dmrg2,pdmrg)** |
| H1-ext-gpu: `hipStreamNonBlocking` for stream_ AND stream_env_ | round-9 self-audit | impl.h:200,201 | impl.h:198,201 | impl.h:288 (per-segment) | **all present** |
| H4: RSVD pointer-mode + sync gating | round-7 | active | active | (per-StreamWorkspace) | **intact** |
| H7: `PointerModeGuard` RAII pairs | round-7 | local `DmrgPointerModeGuard` carry-over (1070, 1216) | common `PointerModeGuard` (1132, 1263) | common `PointerModeGuard` (1332, 1356, 1378, 1543) | **intact** |
| M3: `h_batch_*_pinned` removal | round-7 | only stale comment at impl.h:554 (pdmrg) | gone | comment-only | **intact** |
| H3: init methods (`initialize_mps_product`, `initialize_mps_neel`) | round-5 | header :2 hits | header :2 hits | header :2 hits | **intact** |
| M12: `set_rsvd` setter unification | round-7 | done | done | done | **intact** |
| M13: nnz rename | round-7 | `_nnz_rows_/_cols_` consistent | same | same (incl. WW variant) | **intact** |
| D_PAD R-env identity slot | round-4 | impl.h:1003 `D_mpo_actual_-1` | impl.h:1079 | impl.h:1296 | **intact** |
| accurate_svd_gpu (J1 lock) | round-5 | N/A (no boundaries) | N/A | impl.h:4 hits | **intact** |
| Dual-stream env-update overlap | round-6 | 6 event/wait sites in sweep_LR | 6 sites | per-segment streams | **intact** |

Round-10's pre-existing carry-overs (local `DmrgPointerModeGuard`,
stale `h_batch_*_pinned` comment, `set_quiet` no-op, deferred H7-ext)
are unchanged; they were explicitly deferred to post-G1 cleanup, not
defects to re-flag here.

## CRITICALS

None.

## HIGHS

None.

## MEDIUMS

None net-new vs round-10. (The carry-overs above remain deferred per
round-10 SUMMARY.)

## NITS — net-new

### N11-gpu-h_svd-dead — `h_svd_*` host scratch dead in dmrg-gpu + dmrg2-gpu

**Files**:
- `gpu-rocm/dmrg-gpu/src/dmrg_gpu.h:164-165` — declares
  `h_svd_U_, h_svd_Vh_, h_svd_tmp_, h_svd_S_`.
- `gpu-rocm/dmrg-gpu/src/dmrg_gpu_impl.h:319,323-325` — `resize()` the
  four vectors in the ctor.
- `gpu-rocm/dmrg2-gpu/src/dmrg2_gpu.h:156-157` — same four vectors.
- `gpu-rocm/dmrg2-gpu/src/dmrg2_gpu_impl.h:320-323` — `resize()` only.

These four host vectors are sized in the constructor and then never
read or written by any code path. The actual SVD runs entirely on
device through the `d_svd_*` buffer family
(`d_svd_A_/U_/S_/Vh_/E_/info_/work_` allocated at impl.h:307-313 in
dmrg-gpu and impl.h:307-314 in dmrg2-gpu). pdmrg-gpu's `ws.h_svd_*`
counterparts ARE live (used at `pdmrg_gpu_impl.h:451` for the
host-fallback path inside `accurate_svd_gpu`), so this is specifically
a leftover-from-removed-code-path artifact in the two non-pdmrg
single-host -gpu variants, not a tier-wide pattern.

**Severity**: NIT. No correctness or performance impact — the
host-allocated vectors cost about
`(svd_max_dim * chi_max_) * 3 * sizeof(double)` host RAM per
instance. At chi=512, L=200, that's ~1 MB per variant — trivial.

**Defect class**: same dead-infrastructure class as round-9
MED-base-1 (pdmrg-gpu-base h_svd) and round-10 MED-pdmrg-opt-{1,2,3}
(`h_rsvd_B`, `h_rsvd_U_small`, `h_dav_V_copy`). Caught by technique A.
Surfaced now, not earlier, because previous A scans focused on
device buffers and concurrency primitives; this round added host-side
`h_*_` members to the systematic A-scan.

**Recommendation**: defer to the same post-G1 cleanup batch as the
other dead host buffers. Removing the four declarations + their
`resize()` lines is a 6-line patch. Do not block G1 on this.

## FALSE POSITIVES VERIFIED

- `pdmrg_gpu_impl.h` shows 0 hits for `d_rsvd_omega_` and `d_batch_A_`
  with trailing underscore — initial scan looked dead. Verified
  intentional: pdmrg-gpu uses per-`StreamWorkspace` (segment-local)
  fields named `d_rsvd_omega` / `d_batch_A` (no trailing underscore),
  declared in `pdmrg_gpu.h:116-199`. 60 hits in the impl. Not a
  defect.
- dmrg-gpu still uses the local `DmrgPointerModeGuard` instead of the
  common `PointerModeGuard`. Round-10 noted this as a deliberate
  carry-over (the local guard was kept for ABI continuity with
  pre-round-7 dmrg-gpu downstreams). Not a new finding.
- `h_batch_*_pinned` no-longer-allocated comment at
  `pdmrg_gpu_impl.h:554` — round-10 already noted as cosmetic
  carry-over.

## SUMMARY

**Round 11 is net-clean of HIGH/MEDIUM findings on the -gpu tier.**

One NIT (`h_svd_*` dead host scratch in dmrg-gpu + dmrg2-gpu) was
surfaced by an expanded technique-A scan covering host-side `h_*_`
members in addition to device buffers. The defect class is identical
to round-9 MED-base-1 and round-10 MED-pdmrg-opt-{1,2,3}: declared,
resized, never used. Costs ~1 MB host RAM. No correctness or
performance impact. Defer with the existing post-G1 cleanup batch.

**The watch list is fully verified intact.** All 12 round-7 / round-9
/ round-10 fix classes propagated to every -gpu sibling that needs
them. No regressions.

**Verdict on the round-10 question** ("verify horizontal-review-gpu
still returns 0 net-new findings at round 11"):

- 0 CRITICAL, 0 HIGH, 0 MEDIUM net-new — confirmed.
- 1 NIT net-new (dead host scratch). Cosmetic. Same class as
  pre-existing deferred items.

If the gate is "0 critical/high/medium net-new for two consecutive
rounds" the -gpu tier passes. If the gate is "0 findings of any
severity," the NIT can be closed in 6 lines of edits before G1 — but
it does not block GPU run / paper submission on its own.

**Recommendation**: -gpu tier is G1-ready. Bundle N11-gpu-h_svd-dead
into the same post-G1 cleanup commit that covers the other deferred
dead-host-buffer items.
