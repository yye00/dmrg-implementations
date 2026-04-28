# Full conformity review — 2026-04-28

Pre-G1 GPU window gating audit. Six sub-reviews dispatched in parallel,
findings deduplicated.

## Charter proof — sub-review status

| Sub-review | Status | Findings |
|---|---|---|
| vertical-review-dmrg     | OK | 3 criticals, 5 highs, 4 mediums, 1 nit |
| vertical-review-dmrg2    | OK | 0 criticals, 2 highs, 2 mediums, 3 nits |
| vertical-review-pdmrg    | OK | 0 criticals, 3 highs, 3 mediums, 2 nits |
| horizontal-review-base   | OK | 0 criticals, 1 high, 4 mediums, 2 nits |
| horizontal-review-gpu    | OK | 0 criticals, 3 highs, 3 mediums, 2 nits |
| horizontal-review-opt    | OK | 3 criticals, 2 highs, 2 mediums, 2 nits |

All six sub-reviews ran A-E in full (D was DONE-via-grep on five of
six; clangd unavailable without ROCm headers on host — technique A
subsumes the dead-symbol case). No FAILED reviews.

---

## CRITICALS — block GPU run

### dmrg-gpu-opt (3 criticals — host LAPACK + host pointer arrays + unguarded syncs on default code path)

These are tightly correlated and form one root failure: the -opt
variant's default code path runs host computations on every site of
every sweep, contradicting the file's own docstring claim of "Zero
per-sweep host LAPACK on default code path."

- **C1. Per-site host pointer-table construction in `apply_heff` /
  env update.** `std::vector<Scalar*> h_A/h_B/h_C` built on host then
  `hipMemcpyAsync` to device — every site, every sweep. -gpu uses
  GPU kernels (`setup_batch_ptrs_wd`, `setup_batch_ptrs_env3`) and
  does no host pointer construction.
  [`gpu-rocm/dmrg-gpu-opt/src/dmrg_gpu_opt_impl.h:639-649,760-770,854-864`]
  *(found by: vertical-review-dmrg)*

- **C2. Block-Davidson Rayleigh-Ritz invokes host `lapack_syev`
  every iteration of every site.** With `use_davidson_=true`
  (default), G1 default benchmarks pay this cost on every site. The
  class header docstring at `dmrg_gpu_opt.h:38-40` claims "Zero
  per-sweep host LAPACK on default code path." `rocsolver_dsyevd`
  exists and would replace this on-device.
  [`dmrg_gpu_opt_impl.h:1585,1597`]
  *(found by: vertical-review-dmrg, vertical-review-dmrg2 (same
  defect class in dmrg2-gpu-opt at lines 1554, 1565 — see H6))*

- **C3. Six unguarded `hipStreamSynchronize(stream_)` inside
  `block_davidson_eigensolver`.** No `opts_.profile` guard — every
  Davidson iteration of every site fences the main stream globally.
  At `use_davidson_=true` this issues 6+ host fences per site, fully
  defeating the round-6 dual-stream env-update overlap.
  [`dmrg_gpu_opt_impl.h:1524,1545,1630,1657,1701,1749`]
  *(found by: vertical-review-dmrg)*

### pdmrg-gpu-opt (3 criticals — silent algorithm disable + J2 violations)

- **C4. `use_davidson_` defaults to FALSE in ctor.** Header docstring
  (`pdmrg_gpu_opt.h:19`) declares "Block-Davidson eigensolver
  replaces Lanczos" but `pdmrg_gpu_opt_impl.h:221` sets
  `use_davidson_=false`. Test driver only flips it on with explicit
  `--davidson` CLI flag (`test_pdmrg_gpu_opt.cpp:391`). Sister -opt
  variants (`dmrg_gpu_opt.h:232`, `dmrg2_gpu_opt.h:212`) default
  `true` in-class. **G1 default benchmarks for pdmrg-gpu-opt run
  Lanczos, not Davidson — the entire algorithmic premise is silently
  disabled.**
  [`pdmrg_gpu_opt_impl.h:221`]
  *(found by: horizontal-review-opt)*

- **C5. `n_recal` recalibration parameter dropped from `run()`.**
  pdmrg-gpu's signature
  `run(n_outer_sweeps, n_local_sweeps=2, n_warmup=1, n_polish=0, n_recal=0)`
  exists with the recalibration sweep wired
  (`pdmrg_gpu_impl.h:2717`). pdmrg-gpu-opt's signature truncates to
  four parameters; recalibration loop absent from impl entirely.
  Any benchmark passing `n_recal>0` for parity against -gpu cannot
  do so against -opt. **J2 violation.**
  [`pdmrg_gpu_opt.h:43` vs `pdmrg_gpu.h:39`]
  *(found by: horizontal-review-opt)*

- **C6. `d_Vh_canonical` boundary-merge swap buffer dropped.**
  pdmrg-gpu uses it to swap `d_mps_tensors_[bsite+1]` to a
  freshly-canonicalized buffer after segment-merge boundary SVD
  (`pdmrg_gpu_impl.h:2544-2559`). Absent from pdmrg-gpu-opt header
  + impl entirely; the boundary-merge swap path is structurally
  different and may produce a stale tensor pointer. **J2 violation.
  Needs a smoke test before G1.**
  [`pdmrg_gpu_opt_impl.h:3338-3360`]
  *(found by: horizontal-review-opt)*

---

## HIGHS — fix before next major event

- **H1. pdmrg-gpu-opt creates streams WITHOUT `hipStreamNonBlocking`
  flag.** `hipStreamCreate(&streams_[k])` and
  `hipStreamCreate(&worker_streams_[k][w])` use default (blocking)
  flag. pdmrg-gpu-base and pdmrg-gpu both use
  `hipStreamCreateWithFlags(..., hipStreamNonBlocking)`. The base
  header docstring explicitly says non-blocking is "required for
  correct concurrent boundary merges; not an optimization."
  **Per-segment parallelism is silently serialized in -opt — kills
  the algorithm's parallelism.** One-line fix.
  [`pdmrg_gpu_opt_impl.h:142,155`]
  *(found by: vertical-review-pdmrg)*

- **H2. pdmrg-gpu-opt `parallel_sweep` is not exception-safe.**
  Worker threads invoke `sweep_fn(this, k)` without try/catch; any
  HIP_CHECK failure escapes → `std::terminate`. pdmrg-gpu-base
  (`:1391-1407`) and pdmrg-gpu (`:2645-2662`) both use
  `std::exception_ptr` capture + `std::rethrow_exception` after
  join. **Regression vs both earlier tiers.** Copy-paste fix.
  [`pdmrg_gpu_opt_impl.h:3441-3451`]
  *(found by: vertical-review-pdmrg)*

- **H3. pdmrg-gpu missing `initialize_mps_product()` /
  `initialize_mps_neel()` initializers.** dmrg-gpu and dmrg2-gpu
  declare all three `_random` / `_product` / `_neel`; pdmrg-gpu has
  only `_random`. Asymmetric-feature pattern (the dmrg2-gpu
  dual-stream miss replayed). Either silently dropped or never
  ported; G1 benchmark scripts that touch these will break against
  pdmrg.
  [`pdmrg_gpu.h:35` vs `dmrg_gpu.h:35-36`, `dmrg2_gpu.h:35-36`]
  *(found by: horizontal-review-gpu)*

- **H4. pdmrg-gpu RSVD path has 3 unconditional
  `hipStreamSynchronize` calls.** Sit between back-to-back
  stream-ordered ops (gemm → memcpyAsync → geqrf → orgqr → gemm).
  Gratuitous — same-stream rocBLAS/rocsolver are already ordered.
  dmrg-gpu and dmrg2-gpu RSVD have zero such syncs. Serializes
  segment streams through the host on the RSVD path.
  [`pdmrg_gpu_impl.h:1746,1754,1766`]
  *(found by: horizontal-review-gpu)*

- **H5. WW (fused two-site MPO) precompute runs on the host in
  dmrg2-gpu, dmrg2-gpu-opt, pdmrg-gpu, and pdmrg-gpu-opt.** All
  four use a 7-deep nested host loop and `hipMemcpy(... HostToDevice)`
  per bond. The dmrg2-gpu-base header at `dmrg2_gpu_base.h:22-29`
  lists "the on-device WW precompute kernel" as a feature -base
  omits — i.e., explicitly claims -gpu/-opt have it. `set_mpo()`
  time so it does not affect timed sweeps, but the docstring claim
  is wrong. Either fix the docstring or port.
  [`dmrg2_gpu_impl.h:641-666`, `dmrg2_gpu_opt_impl.h:524-547`,
  `pdmrg_gpu_impl.h:818-857`, `pdmrg_gpu_opt_impl.h:667-706`]
  *(found by: vertical-review-dmrg2, vertical-review-pdmrg)*

- **H6. dmrg2-gpu-opt: Block-Davidson host `lapack_syev` per
  iteration on default code path.** Same defect class as C2 in
  dmrg-gpu-opt. `use_davidson_=true` is default; `(k,k)` projected
  eigensolve runs on host every Davidson iteration. Violates "no
  host roundtrips per sweep" rule.
  [`dmrg2_gpu_opt_impl.h:1554,1565`]
  *(found by: vertical-review-dmrg2)*

- **H7. rocBLAS pointer-mode toggle without RAII restore in
  dmrg2-gpu and pdmrg-gpu.** dmrg-gpu has `DmrgPointerModeGuard`
  RAII (round-5 fix). dmrg2-gpu and pdmrg-gpu toggle pointer mode
  with paired calls that will leak the device mode into subsequent
  rocBLAS calls if an intermediate `ROCBLAS_CHECK` throws. Round-6
  rlbfgs-gpu finding replayed across two more variants.
  `accurate_svd_gpu.h::AsvdPointerModeGuard` (lines 64-78) already
  exists — promote to `common/` or copy.
  [`dmrg2_gpu_impl.h:1141,1208,1270,1287`;
  `pdmrg_gpu_impl.h:1313,1336,1356,1447,1519,1533`]
  *(found by: horizontal-review-gpu)*

- **H8. dmrg-gpu-opt: `d_batch3_A_/B_/C_` declared, allocated, and
  freed but never read or written.** Header docstring describes
  them as "Step-3 batched GEMM pointer arrays." The Step-3 path
  uses `gemm_strided_batched` (no pointer array) on the
  large-chi/D≤2 fast path or per-(wp,sp) plain `gemm` loop
  otherwise — neither code path touches `d_batch3_*_`. **The
  round-7 dmrg2-gpu dead-stream pattern again.**
  [`dmrg_gpu_opt.h:165-167`; `dmrg_gpu_opt_impl.h:197-199,344-346`]
  *(found by: vertical-review-dmrg; horizontal-review-opt M16
  flagged this as a per-variant naming mismatch but did not run
  technique A — vertical caught the genuine dead status)*

- **H9. Dead two-site `sweep_LR_full()` / `sweep_RL_full()` in
  pdmrg-gpu-base.** Declared and defined but never called — `run()`
  only calls the `_1site` variants. Dead engineering scaffolding
  AND a foot-gun: a future maintainer who calls them for
  warmup/polish creates a CLAUDE.md PDMRG-rule violation. Same dead
  pattern that hid the dmrg2-gpu unused stream for three rounds.
  [`pdmrg_gpu_base.h:207-208`; `pdmrg_gpu_base_impl.h:1115-1132`]
  *(found by: horizontal-review-base)*

- **H10. dmrg-gpu-opt and dmrg2-gpu-opt: Lanczos α/β host-resident
  in fallback path.** -gpu uses sync-free device-pointer-mode
  Lanczos (`d_dot_result_`, `d_alpha_dev_`, `d_const_*`); -opt
  drops the device scalars and does explicit host α/β + H2D
  memcpy + sync. Documented at `dmrg_gpu_opt.h:181-183` as
  intentional but reintroduces per-iteration host roundtrips.
  Bites only the `set_use_davidson(false)` fallback path so HIGH
  not CRITICAL. **pdmrg-gpu-opt does NOT have this regression.**
  [`dmrg_gpu_opt_impl.h:1083-1109`; equivalent in dmrg2-gpu-opt]
  *(found by: horizontal-review-opt)*

- **H11. dmrg-gpu-opt sparse-MPO Step-3 issues per-nnz unbatched
  `gemm` calls.** -gpu uses `gemm_batched` for the same path. With
  90% sparse W this means -opt issues 10× more rocBLAS dispatches
  than -gpu on `opts_.sparse_mpo` runs.
  [`dmrg_gpu_opt_impl.h:680-692` vs `dmrg_gpu_impl.h:793-801`]
  *(found by: vertical-review-dmrg)*

---

## MEDIUMS

- **M1.** Three byte-equal copies of HIP_CHECK / ROCBLAS_CHECK macros
  across the -base files (and likely -gpu, -opt). Consolidate to
  `common/hip_check.h`. *(horizontal-review-base; cross-tier)*
- **M2.** Dead `d_T3_` / `ws.d_T3` scratch in dmrg2-gpu-base and
  pdmrg-gpu-base (both per-stream in pdmrg). Header comment claims
  "kept for parity" but neither sibling has it. Delete or wire.
  *(horizontal-review-base, vertical-review-pdmrg)*
- **M3.** Dead `h_batch_*_pinned` in pdmrg-gpu (`StreamWorkspace`
  declares them, sets to nullptr, never used). Header advertises
  them as "Pinned host pointer arrays (avoid heap alloc in hot
  path)." *(vertical-review-pdmrg)*
- **M4.** `set_mpo()` does not guard against double-call across all
  three -base variants (would leak previous `d_mpo_tensors_[i]`).
  *(horizontal-review-base)*
- **M5.** `set_use_davidson(false)` does not re-enable
  `lanczos_graph` in dmrg2-gpu-opt — one-way disabling.
  *(vertical-review-dmrg2)*
- **M6.** Per-bond `hipMalloc` inside `precompute_fused_mpo` (-gpu
  and -opt for dmrg2/pdmrg). One contiguous allocation would
  simplify lifetime + remove pointer-of-pointers. *(vertical-review-dmrg2)*
- **M7.** dmrg-gpu-opt: `allocate_mps_tensor` does `hipFree` +
  `hipMalloc` per call — known sync point per bond. -gpu pre-allocs
  at chi_max. *(vertical-review-dmrg)*
- **M8.** dmrg-gpu-opt missing `d_const_one_/zero_/neg_one_` device
  constants from -gpu (consistent with host-pointer Lanczos design;
  another instance of "-opt missing -gpu feature").
  *(vertical-review-dmrg)*
- **M9.** dmrg-gpu-opt missing `d_ones_D_` length-D ones vector for
  Step-3 GEMV reduction (-gpu has it for the R3-F1 full-batched-
  collapse). *(vertical-review-dmrg)*
- **M10.** dmrg-gpu-opt's `cL>=16 && cR>=16 && D<=2` Step-3 strided
  branch undocumented in header — d=3 ladder/dimer models silently
  fall to the slow per-(wp,sp) loop. *(vertical-review-dmrg)*
- **M11.** pdmrg-gpu-opt: `set_use_davidson()` setter has silent
  side effect (clobbers `opts_.lanczos_graph`) not documented in
  header. *(horizontal-review-opt)*
- **M12.** pdmrg-gpu has `set_cpu_svd` / `set_rsvd` legacy public
  setters duplicating `opts()` accessor. Two write paths to same
  flags. *(horizontal-review-gpu)*
- **M13.** Naming inconsistency: dmrg-gpu uses `nnz_rows_count_`
  (no prefix); dmrg2-gpu and pdmrg-gpu use `wl_/ww_`-prefixed.
  *(horizontal-review-gpu)*
- **M14.** `set_quiet(bool)` is a no-op in all three -opt variants.
  Pure scaffolding. *(horizontal-review-opt)*

---

## NITS — cosmetic

- `pad_mfma16` duplicated verbatim in three -opt headers. Move to
  `common/`.
- Multiple header docstrings drift relative to current code (Lanczos
  fallback wording in dmrg2-gpu-opt; key-changes list in pdmrg-gpu-opt;
  pdmrg-gpu-base boundary description ambiguity).
- pdmrg-gpu-base builds initial envs with both forward-L and
  backward-R passes; siblings only do backward-R (cosmetic divergence
  from charter expectation, but algorithmically correct).
- `prof_*` profiling counters in dmrg-gpu-opt are file-scope `static`
  globals (not thread-safe across instances; cosmetic for single-
  instance benchmarks).

---

## FALSE POSITIVES VERIFIED (to avoid re-discovery)

- **D_PAD R-env identity slot** — verified at `D_mpo_actual_ - 1`
  (correct) in all -gpu and -opt variants of all three families.
- **Round-6 dual-stream env-update overlap regression** — NOT
  regressed. event records, waits, `env_update_pending_` all live
  in dmrg-gpu-opt and dmrg2-gpu-opt; round-6 wire-up holds.
- **dmrg2-gpu's stream_env_ dead infrastructure** — verified intact
  post-round-6 (23 references in impl, all live).
- **pdmrg J1 Stoudenmire lock** — `accurate_svd_gpu` confirmed
  called inside `merge_and_optimize_boundaries` in all three pdmrg
  tiers (base:1296, gpu:2469, opt:3349).
- **pdmrg-gpu-opt worker_streams_ / use_chebyshev_ / use_batched_sweep_
  dead** — all three verified LIVE (worker streams used in apply_heff;
  Chebyshev consulted at impl:2247; batched sweep consulted at
  impl:3457, 3470).
- **pdmrg-gpu apply_heff_graph_cache cross-segment reuse** — cache
  is per-`StreamWorkspace`, key includes `(two_site, site, cL, cR)`;
  per-stream + per-shape requirement satisfied.
- **`precompute_WW` D_PAD OOB bug from 2026-04-20 ablation** — fix
  intact (all variants iterate unpadded `D_act` range, leave padded
  slots zero from value-initialized vector).

---

## SUMMARY VERDICT

**Block GPU run? YES — 6 CRITICALS.**

The blockers cluster into two failure modes:

1. **dmrg-gpu-opt host roundtrips** (C1, C2, C3) — three independent
   defects on the default code path that together violate the file's
   own "no host LAPACK / no host roundtrips" docstring claim. Per
   `dmrg_gpu_opt.h:43-44` this variant is excluded from G1, but the
   binary "must be at-least-correct" — its tier claims are not met.

2. **pdmrg-gpu-opt silent algorithm disable + J2 violations** (C4,
   C5, C6) — `use_davidson_` defaults false (algorithmic premise of
   -opt evaporates without `--davidson` CLI), `n_recal` parameter
   dropped, `d_Vh_canonical` swap buffer dropped. All three
   regressions vs pdmrg-gpu sibling.

### Top-3 actions before MI300X G1 window

1. **Fix C4 in pdmrg-gpu-opt**: change `use_davidson_=false` to `=true`
   in `pdmrg_gpu_opt_impl.h:221`. One-line fix, restores the variant's
   stated default.
2. **Fix H1 in pdmrg-gpu-opt**: change two `hipStreamCreate` to
   `hipStreamCreateWithFlags(..., hipStreamNonBlocking)` at
   `pdmrg_gpu_opt_impl.h:142, 155`. Two-line fix, restores per-segment
   parallelism.
3. **Re-add `n_recal` and `d_Vh_canonical` to pdmrg-gpu-opt** (C5,
   C6) OR explicitly document why -opt diverges from -gpu on these.
   J2 contract requires the former; if the latter, update the header.

dmrg-gpu-opt criticals (C1, C2, C3) are paper-blockers but not GPU-run
blockers per the round-6 G1 exclusion. Schedule for the post-G1 cycle
unless the GPU window has time to spare.

### What was checked vs. last conformity review

This is the first run of `/conformity-review-full`. No prior baseline
to diff against. Future runs should diff against
`reviews/conformity-20260428.md` and flag any of the above CRITICALS
that re-appear as **regressions** — those would indicate the fix did
not stick.
