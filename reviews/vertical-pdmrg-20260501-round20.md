# Vertical review — segment-parallel PDMRG family — round 20 (2026-05-01)

HEAD: f650466 (R19 reports landed; code state cafd628). Confidence
re-run #2 before MI300X G1 baseline allocation. Pre-step
`bash .claude/scripts/defect-registry.sh` → **TOTAL HITS: 0**.

Scope: pdmrg-gpu-base, pdmrg-gpu, pdmrg-gpu-opt, **pdmrg-multi-gpu**
(R19H19 lives here).

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | 1 dead kernel found (multi-gpu) |
| B. Behavioral diff | DONE | Step-3 SS pattern: gpu/multi-gpu collapsed; opt fallback NOT collapsed |
| C. Docstring verification | DONE | J1 lock holds in all 4 variants; pdmrg-gpu-opt :2302-2311 still partly stale (R18 carry) |
| D. clangd filter | N-A | local clangd lacks ROCm headers; technique A subsumes |
| E. Absence-naming brief | FOLLOWED | per-segment streams, GpuOpts, RSVD, sparse, batched, Davidson, Chebyshev all present per tier |
| F. Workspace-aliasing audit | DONE | T1 in multi-gpu SS Step 3 sized correctly (verified below) |
| G. Sibling fix-propagation | DONE | R19H19 fix verified on multi-gpu; opt SS Step 3 NOT R3-F1-collapsed (intentional per file comment) |

---

## R19H19 fix verification — pdmrg-multi-gpu apply_heff_single_site Step 3

The brief lists 7 verification items. Each is checked against
`pdmrg_multi_gpu_impl.h` at HEAD code state cafd628.

### (1) T1 (= V) reused as scratch — dead post-Step 2 → OK
- impl :1707 `Scalar* V = dev.d_T1;`
- Step 1 (:1710-1725) writes V; Step 2 (:1727-1735) reads V into U.
- After Step 2 V is dead — port reuses it as Step-3 per-wp scratch
  (:1737-1769), then reduces with gemv into d_result. Same lifetime
  as pdmrg-gpu :1971-2018.

### (2) T1 sizing → OK
- Allocation at :352, sized
  `t_max = D_mpo · dd · chi_max² = D · d² · chi_max²` (impl :292).
- Step-3 demand: `D · slice_stride = D · cL · d · cR`.
- Required ≤ allocated iff `cL·d·cR ≤ d²·chi_max²` iff
  `cL·cR ≤ d·chi_max²`. Since cL,cR ≤ chi_max and d ≥ 1, this is
  always true (`cL·cR ≤ chi_max² ≤ d·chi_max²`).
- **Verdict: OK by a wide margin (factor of d).** No overrun risk.

### (3) d_ones_D — per-device, allocated/freed in init/destroy → OK
- Allocation: :370-374 inside `allocate_device_resources()` per
  device, populated to all-ones via H2D.
- Free: :534 inside the destructor's per-device loop.
- Used in two-site Step-3 reduce (:860, R17H4) and now in
  single-site Step-3 reduce (:1767, R19H19). Liveness verified.

### (4) Mathematical equivalence — old vs new → OK
- Old code (per-wp loop with beta accumulation): for each wp,
  Y_wp[a',b'] = sum_b U_{wp·d+sp}[a',b] · R_wp[b,b']; result
  accumulates Y across wp via beta=1 after wp=0.
- New code: writes Y_wp directly into V[wp·slice_stride] for ALL
  wp∈[0,D) and sp∈[0,d) via 1 batched GEMM (batch = D·d, beta=0),
  then reduces with `gemv(slice_stride, D, V, d_ones_D)` — i.e.
  `d_result = sum_wp V[wp·slice_stride : (wp+1)·slice_stride]`.
- The two are algebraically identical: gemv with all-ones is exactly
  the cross-wp sum. The per-sp tile interleaving inside slice
  (:1750 `(wp · slice_stride) + (sp · cL)`, ldc = cL·d) matches the
  old `d_result + sp·cL` layout (cL·d stride). **Equivalent.**

### (5) New kernel `setup_heff_ss_step3_full_ptrs` mirrors pdmrg-gpu
   :49-59 byte-for-byte → OK
- multi-gpu :43-59 vs pdmrg-gpu :42-59: signatures, body, and index
  arithmetic identical. Only the leading comment block differs (the
  multi-gpu copy notes the round-19 origin and points to the R18
  consolidation medium). Confirmed.

### (6) `t_apply_heff_.end` placement → OK
- :1702 `.begin(dev.stream)` at function entry.
- :1771 `.end(dev.stream)` AFTER the Step-3 closing brace
  (:1770) and BEFORE the function return (:1772). Single
  begin/end pair per call, balanced.

### (7) Sibling cross-check — does pdmrg-gpu-opt SS Step 3 also use
   the R3-F1 collapse? → **NO, intentionally.**
- pdmrg-gpu-opt :2293-2433 retains a 3-branch Step-3 dispatch
  (sparse / strided-batched at d≤2 / per-wp host loop fallback).
  No `d_ones_D`, no gemv reduce.
- The file-internal comment at :2302-2311 explains: opt's
  apply_heff_single_site disables LANCZOS_GRAPH because Step
  1/Step 3 else-branches historically used stack-allocated host
  pointer arrays + hipMemcpyAsync inside the capture window. The
  round-16 fix replaced Step 1 host pointers with a kernel; the
  Step-3 fallback (:2414-2429) also now uses a kernel (single
  `setup_batch_ptrs_step3` per wp), so the original rationale for
  not collapsing is partly stale (this is the R18 carry MEDIUM
  M-opt-pdmrg-single-site-graph-comment-stale).
- Decision matrix:
  - pdmrg-gpu (no Davidson, has graph capture) → R3-F1 collapse +
    graph capture, both shipped.
  - pdmrg-multi-gpu (no graph capture) → R3-F1 collapse just shipped
    in cafd628.
  - pdmrg-gpu-opt (Davidson default, graph capture disabled in SS
    by file-internal comment) → still has D launches in Step-3
    fallback. **Could be collapsed for free; deferred to R20+ as
    polish.** Not a CRITICAL/HIGH because pdmrg-gpu-opt's SS path
    is equivalent in correctness and the strided-batched fast path
    (:2401-2410) covers d≤2 (Heisenberg/TFIM challenge sizes); the
    fallback fires only on d>2 (e.g. Josephson d=4 ladders).

**R19H19 verification: PASS on all 7 items.** Fix is correct,
mathematically equivalent, sized, and timed.

---

## J1 lock — Stoudenmire `accurate_svd_gpu` in EVERY pdmrg variant

| Variant | Call site | Verdict |
|---|---|---|
| pdmrg-gpu-base   | impl :1276 (`merge_and_optimize_boundaries`)        | OK |
| pdmrg-gpu        | impl :2452 (`merge_and_optimize_boundaries`)        | OK |
| pdmrg-gpu-opt    | impl :3351 (`merge_and_optimize_boundaries`)        | OK |
| pdmrg-multi-gpu  | impl :2281 (`merge_and_optimize_boundaries`)        | OK |

J1 holds across all 4 tiers. No regressions vs R19 baseline.

## J2 superset — pdmrg-base ⊂ -gpu ⊂ -gpu-opt

Spot-checked headers: per-segment streams (all 3), GpuOpts (-gpu,
-opt), `set_cpu_svd`/`set_rsvd` (-gpu, -opt), `set_use_davidson`/
`set_use_batched_sweep`/`set_use_chebyshev` + `pad_mfma16` +
`worker_streams_` + `d_xs_batch_*` (-opt only). Holds.

## Carry-watch from R15→R19

- **CR-D1 / D14 (Davidson buffer overrun in pdmrg-gpu-opt)** —
  fix at :255-271 still intact:
  `dav_work_sz = max(b·dim_max + max_sub·b, max_sub²)`. The
  `+ max_sub·b` term that round-17 added is still there. **Not
  reverted.**
- **PhaseTimer t_davidson_ panel** — pdmrg-gpu-opt:
  `init` :3578, `row(t_davidson_)` :3597, `.begin` :1502,
  `.end` ×5 sites (:1592, :1628, :1678, :1734, :1795). Balanced.
  Other pdmrg variants don't have Davidson — correctly absent.
- **t_absorb_ removal** — confirmed REMOVED in all 4 pdmrg variants
  (zero hits). Two-site SVD post-processing no longer panel-counted
  separately, as planned.

---

## CRITICALs — block GPU run / paper submission

**None.**

## HIGHs — fix before next major event

- **H-opt-batched-lanczos-host-mode** (carry from R18/R19, NOT
  blocking G1). `pdmrg_gpu_opt_impl.h:2870-2980` —
  `batched_lanczos_eigensolver` retains host-resident `h_alpha`,
  `h_beta`, per-iter `Traits::dot/nrm2/axpy` with host scalars,
  and explicit `hipStreamSynchronize` (:2904, :2996). Gated on
  `set_use_batched_sweep(true)` → **default OFF** (impl :194).
  Same shape applies to `chebyshev_eigensolver` (:2028, default
  OFF). Defer until those campaigns turn the toggles on.

## MEDIUMs — fix when convenient

- **M-multi-gpu-step3-old-kernel-dead** (NEW this round, technique-A).
  After the R19H19 port, the OLD `setup_heff_ss_step3_ptrs` kernel
  at multi-gpu impl :26-39 is declared but never launched (only the
  new `_full_ptrs` at :49-59 is referenced at :1746). Safe to delete
  in R20 cleanup or fold into the M-multi-gpu-local-kernels-dup
  consolidation into `common/batch_ptrs_kernels.h`.
- **M-opt-pdmrg-ss-step3-not-r3f1** (NEW — promoted from technique-G
  observation in (7) above). pdmrg-gpu-opt SS Step 3 d>2 fallback
  could be R3-F1-collapsed for free (sibling pattern is shipped in
  -gpu and -multi-gpu). Cost: 1 setup kernel + 1 batched GEMM + 1
  gemv vs current D launches. Safe inside the
  graph-capture-disabled SS context. Polish, not blocking.
- **M-multi-gpu-local-kernels-dup** (carry R18). 4 file-local
  pointer-setup kernels duplicated across pdmrg-gpu and multi-gpu.
- **M-multi-gpu-precompute-fused-mpo-host** (carry R18). `set_mpo`
  time, not per-sweep — outside no-host-roundtrips rule.
- **M-opt-pdmrg-single-site-graph-comment-stale** (carry R18).
  pdmrg_gpu_opt_impl.h :2302-2311 comment partly contradicts
  current code (Step 1 + Step 3 fallback now use kernels).

## NITs — cosmetic

None new this round.

## FALSE POSITIVES VERIFIED

- **D7+D8 host LAPACK** flagged in pdmrg-{gpu, gpu-opt, multi-gpu}
  by registry — all are `use_cpu_svd_=true` opt-in branches plus
  `lwork=-1` workspace-query stubs. Not on default code path.
- **D13 -base apply_heff per-wp host loop** — registry whitelists
  -base: naive single-GEMM IS the baseline tier.

## SUMMARY

R19H19 fix is correct on all 7 verification axes (T1 reuse, T1
sizing margin = factor of d, d_ones_D liveness, mathematical
equivalence, byte-for-byte kernel mirror, balanced timer, sibling
rationale). J1 lock holds in all 4 pdmrg variants. J2 superset
holds. CR-D1 fix in -opt still intact (no regression). One new
MEDIUM (dead kernel left behind by R19H19 port — trivial cleanup).
One promoted MEDIUM (opt SS d>2 fallback could adopt R3-F1 collapse
for parity). Carry HIGH on opt-in batched-lanczos / chebyshev paths
remains deferred — both default OFF for G1.

**Verdict: 0 CRITICALs, 0 NEW HIGHs.** READY for MI300X G1
baseline. The R20 confidence re-run added no new blockers; the two
new MEDIUMs are polish items for the next code cycle.
