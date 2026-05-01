# Vertical review — dmrg2 family — 2026-05-01 round-15

HEAD: `f40140d`. Baseline: round-14 at `5deba6d` /
`reviews/vertical-dmrg2-20260501-round14.md`. Scope per round-15 brief:
the full dmrg2 family — `gpu-rocm/dmrg2-gpu-base/`, `gpu-rocm/dmrg2-gpu/`,
`gpu-rocm/dmrg2-gpu-opt/`. Two commits land between baseline and HEAD:
`5deba6d` (round-14 dmrg2-touching fixes) and `f40140d` (dmrg-gpu-only
nit; no dmrg2 changes).

## Charter proof

| Technique | Status | Notes |
|---|---|---|
| A. Symbol-usage scan | DONE | dmrg2-gpu-base 28 members audited, all live; dmrg2-gpu/-opt critical members all live |
| B. Behavioral diff | DONE | base↔gpu↔opt deltas all intentional and unchanged |
| C. Docstring verification | DONE | dmrg2-gpu-base.h:83-84 `d_WW_` docstring fix verified against host-side `precompute_WW` at impl:265 |
| D. clangd filter | N-A | no clangd locally; A subsumes |
| E. Absence-naming brief | FOLLOWED | -base, -gpu, -opt expected-feature checklists pass |
| F. Workspace-aliasing audit | DONE | 5 shared buffers re-checked; 0 OVERRUN |
| G. Sibling fix-propagation | DONE | 3 round-14 fixes traced; 0 MISSING in dmrg2 family |

## Regression-watch verification (round-14 baseline)

### 1. H1-base-apply_heff scope fix in dmrg2-gpu-base — INTACT

`dmrg2_gpu_base_impl.h` `lanczos_eigensolver` (impl:579-694) now mirrors
the dmrg2-gpu sibling pattern exactly:
- Pre-loop normalization (impl:587-593) wraps `nrm2 / inv_real / scal_real`
  in its own `PointerModeGuard` block; closed before the H2D `v[0]` copy.
- Per-iter device-mode guard (impl:608-651) is scoped INSIDE the for-loop
  body AFTER `apply_heff_two_site(...)` returns at impl:606. The block
  encloses only the `dot / alpha / axpy / reorth-CGS / nrm2 / scal_real`
  BLAS-1 ops; destroys at the closing brace before the next iter's
  `apply_heff_two_site`.
- Final post-loop normalization (impl:685-691) is in its own guard block.

`apply_heff_two_site` (impl:339-417) uses host-stack `&one`/`&zero_val`
in 5+ rocBLAS gemm sites — requires HOST pointer mode. The new scope
pattern guarantees this. Verified `rocblas_create_handle` at impl:36
leaves the handle in default host mode; the only `rocblas_set_pointer_mode`
calls now go through `PointerModeGuard` (zero raw calls, was 4 in
round-13 baseline). **OK**.

### 2. M-skip-on-zero-prop in dmrg2-gpu and dmrg2-gpu-opt — INTACT

`dmrg2_gpu_impl.h:365` and `dmrg2_gpu_opt_impl.h:1910` both have
`if (t.calls() == 0) return;` at the top of the row lambda. Phases that
weren't exercised (e.g., `t_env_update_` when `n_sweeps==0`, or the
RSVD branch when `use_rsvd_=false`) skip cleanly. **OK**.

### 3. dmrg2-gpu-base.h:83-84 d_WW_ docstring — INTACT

Line 83-84 now reads `"precomputed on host at set_mpo() time and
hipMemcpy'd to device. (Class-level docstring documents this.)"`.
Cross-checked against the host build at impl:265 (`precompute_WW`)
and the H2D upload at the end of that function — accurate. **OK**.

## Round-13/-12 carry-forward (re-verified)

- **CR-D1 dav_work_sz** (-opt impl:265-268): still
  `max(theta_size_max·b + max_sub·b, max_sub·max_sub)`. Inner-loop
  concurrent regions: residuals at offset 0 (impl:1576), overlap at
  `n_new·dim` (impl:1607-1608). Restart path (1659-1667) uses sequential
  reuse. Allocated ≥ required. **OK**.
- **Round-6 dual-stream env-pipeline + direction-L MPS-write reorder**:
  -gpu impl:1414/1439, -opt impl:1391/1421 record `event_canon_ready_`
  immediately after the MPS write; direction-L writes MPS[site+1]=Vh
  BEFORE the U·S absorb. **INTACT**.
- **D_PAD precompute_fused_mpo OOB**: -opt impl:506-541 and -gpu
  impl:607-642 both bound inner loops by `D_act = D_mpo_actual_` while
  writing into the padded `D_use*dd` stride. **INTACT**.
- **M-opt-davidson-toggle dual-half** (bc3fcd0): `dmrg2_gpu_opt.h:73-81`
  setter intact; ctor at impl:74 still sets
  `lanczos_graph_was_user_enabled_ = true` symmetrically. **INTACT**.
- **M1-final guard consolidation** (0b9fccf): dmrg2-gpu uses shared
  `PointerModeGuard` at impl:1125, 1256 (include at impl:5). dmrg2-gpu-base
  now also uses it (3 sites — round-14 fix). dmrg2-gpu-opt: zero pointer-
  mode toggles (mode-neutral inner loop). **INTACT**.
- **M14-final dead set_quiet stub**: dmrg2-gpu.h:44 keeps the
  `void set_quiet(bool) {}  // no-op` stub — used 6× by the test driver
  (test_dmrg2_gpu.cpp:214,257,301,348,395,448). dmrg2-gpu-opt.h has no
  set_quiet. dmrg2-gpu-base.h has no set_quiet (no test caller).
  **INTACT**.
- **Non-blocking stream flag** (H1-ext-gpu): all three variants create
  streams with `hipStreamNonBlocking`. **INTACT**.
- **rocsolver_syevd round-7 H6** (-opt Davidson): impl:1525-1527 still
  uses on-device syevd for Rayleigh-Ritz; tiny D2H of `&energy` +
  `&h_dav_info` only. **INTACT**.

## Sibling propagation cross-check (technique G)

The H1-base scope-fix defect class — "PointerModeGuard scope must NOT
enclose `apply_heff*`, which uses host-stack BLAS scalars" — applied
across all three dmrg2 tiers. dmrg2-gpu (round-12 baseline, impl:1121
+ 1125): apply_heff at 1121, guard enters at 1125 — already correct
post-round-12. dmrg2-gpu-opt: Davidson hot loop is host-pointer-mode
throughout (no inner guard); apply_heff_two_site preserves host mode —
already immune. dmrg2-gpu-base: round-14 fix landed (above). **All
three siblings now correct.**

## CRITICALS

None.

## HIGHS

None.

## MEDIUMS

None within charter.

## NITS

None.

## FALSE POSITIVES VERIFIED

- The use_cpu_svd_ path in dmrg2_gpu_opt_impl.h:1259-1285 issues async
  H2D copies from `std::vector` host buffers (pageable memory) at the
  end. Looks like a "missing sync" — not a defect: subsequent post-SVD
  truncation reads `d_svd_S_/U/Vh` on the same `stream_`, so HIP's
  per-stream serialization holds. Pageable async memcpy semantics on
  HIP collapse to synchronous on the host side. Behaviorally correct.

## SIBLING-PROPAGATION OBSERVATION (out-of-charter, awareness)

The round-14 nit `f40140d` (dmrg-gpu.h:43 set_quiet stub comment
uniformity, "// no-op") is not propagation-relevant for dmrg2: the
sibling stub at dmrg2_gpu.h:44 already reads `// no-op`. Stylistic
parity already held — no fix needed in dmrg2 family. Confirmed.

## SUMMARY

Round-15 returns **0 critical, 0 high, 0 medium, 0 nits** for the dmrg2
family across all three tiers. The two regression-watch fixes from
round-14 (`5deba6d`) — H1-base-apply_heff scope correction in
dmrg2-gpu-base lanczos and M-skip-on-zero in -gpu/-opt report_timers —
are intact. The dmrg2-gpu-base.h `d_WW_` docstring correction matches
the host-side precompute. The dmrg2-gpu-only `f40140d` nit (set_quiet
stub comment) is non-applicable to dmrg2 because dmrg2 already has the
unified "// no-op" comment style. All seven prior carry-forward items
remain healthy: CR-D1 sizing, round-6 direction-L reorder + dual-stream
events, D_PAD precompute fix, M-opt-davidson-toggle, M1-final guard
consolidation (now extending to -base via the round-14 fix), M14-final
dead-stub policy, non-blocking streams, and rocsolver_syevd. This is
the fifth consecutive zero-finding sub-review for the dmrg2 family
within charter. Block GPU run? **NO**, family is ready.

Self-audit: all seven techniques completed (D N-A); regression-watch
list explicitly traced for the three round-14 dmrg2 items plus seven
carry-forwards; -base brought into scope per the round-15 lesson and
verified. Verdict: **READY**.

(Length: ~790 words.)
