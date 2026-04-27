# Pre-G1 baseline review — follow-up items

Three reviewer agents (`feature-dev:code-reviewer`) audited the baseline GPU
variants on 2026-04-27 against an N=10 campaign launch checklist. The
G1-blocking findings ("Tier A") were fixed in commit `<this-commit>`. This
doc captures the remaining "Tier B" items: real but not blocking the G1
campaign (either because the affected code path isn't exercised, or because
the variant in question isn't in the campaign).

Each item is paired with what would trigger it from "deferred" to "must
fix" so future operators know when to revisit.

---

## B1. `pdmrg-gpu-base` CLAUDE.md violations

**Files**:
- `gpu-rocm/pdmrg-gpu-base/src/pdmrg_gpu_base_impl.h:1322-1323` — polish phase calls `sweep_LR_full()` / `sweep_RL_full()` (two-site), violating CLAUDE.md rule "Polish sweeps MUST be single-site."
- `gpu-rocm/pdmrg-gpu-base/src/test_pdmrg_gpu_base.cpp:18` — comment indicates hardcoded `n_warmup=3`, violating "n_warmup ≤ 2".
- The `-base` binary has no `--warmup` / `--polish` CLI flags, so the campaign config cannot override.

**Why deferred**: `pdmrg-gpu-base` is **not in the G1 campaign config**
(`benchmarks/campaigns/g1_baseline_rebench.json` only lists
`dmrg-gpu`, `dmrg2-gpu`, `pdmrg-gpu`). The `-base` snapshots are explicitly
excluded from `gpu-rocm/build_all.sh` and have no entries in the bench
harness's `IMPL_BINARIES` map.

**Trigger to fix**: any decision to rebench `pdmrg-gpu-base` for
direct comparison with `pdmrg-gpu`. The base would need either
(a) `--warmup` / `--polish` CLI flags added and the hardcoded values made
overridable, or (b) the hardcoded values changed to comply
(`n_warmup=1`, single-site polish), with the latter making the "naive
baseline" interpretation slightly misleading.

---

## B2. RSVD per-bond heap allocation in `dmrg2-gpu` and `pdmrg-gpu`

**Files**:
- `gpu-rocm/dmrg2-gpu/src/dmrg2_gpu_impl.h:1339` — inside `svd_split()`,
  `std::vector<Scalar> h_omega(n_svd * r_use)` heap-allocated and filled
  with random values per bond per sweep when `--rsvd` is on. ~1 MB
  allocation per call at $\chi=256$.
- `gpu-rocm/pdmrg-gpu/src/pdmrg_gpu_impl.h:1727` — same pattern in
  `rsvd_split()`. ~38K scalar allocations per bond at $\chi=128$ in the
  ablation envelope.

**Why deferred**: G1 does **not** pass `--rsvd` (the campaign config has
no `--rsvd` in `extra_args_per_variant`). The default RSVD-off code path
is unaffected.

**Trigger to fix**: any future RSVD-on benchmark campaign (e.g., a
re-run of the §6.6 ablation table at $N{=}10$ statistical level). Fix:
move `h_omega` to a pre-allocated buffer in `StreamWorkspace` (alongside
the existing `h_rsvd_B` allocation pattern) so the inner loop only does
the random fill, not the malloc.

---

## B3. `--nmax` silently accepted but ignored in `-base` snapshots

**Files**:
- `gpu-rocm/dmrg-gpu-base/src/test_dmrg_gpu_base.cpp:312`
- `gpu-rocm/dmrg2-gpu-base/src/test_dmrg2_gpu_base.cpp:311`

The bench harness passes `--nmax 2` for Josephson runs; the `-base`
binaries accept the flag and silently discard it, always using
hardcoded `n_max=2`.

**Why deferred**: currently safe because
`benchmarks/bench_dmrg_gpu_ablate.py:65` and the campaign config both
use `nmax=2`, which matches the hardcoded value. Numerically identical
results today.

**Trigger to fix**: any change to the campaign's Josephson `nmax` (e.g.,
ablation across local-dimension sensitivity, or any model that wants
`nmax > 2`). Fix: parse `--nmax N` and pass to the constructor instead
of discarding. ~5 lines per file.

---

## B4. `opts_.print(stderr)` and `[D_PAD]` print on every binary
invocation

**Files**:
- `gpu-rocm/dmrg-gpu/src/dmrg_gpu_impl.h:193, 198`
- `gpu-rocm/dmrg2-gpu/src/dmrg2_gpu_impl.h:~similar` (constructor)
- `gpu-rocm/pdmrg-gpu/src/pdmrg_gpu_impl.h:262-268`

Each constructor unconditionally writes 6+ lines to stderr (one per
optimisation flag, plus a `[D_PAD]` line if D_PAD is on). Fires once per
binary invocation including the warmup sub-runs, but **not inside the
sweep loop**, so does not affect timing.

**Why deferred**: cosmetic only. Stderr noise but not perf-relevant.

**Trigger to fix**: if the bench harness ever has trouble parsing stderr
(e.g., a future logging refactor), or if the noise in benchmark logs
becomes a real problem. Fix: guard `opts_.print(stderr)` and the D_PAD
print behind `if (!opts_.quiet)` (would need adding a quiet field to
`GpuOpts`, or wiring through a constructor argument).

---

## B5. Per-rep `printf` blocks in `pdmrg-gpu::run()`

**File**: `gpu-rocm/pdmrg-gpu/src/pdmrg_gpu_impl.h:2624, 2639, 2727,
2755, 2759-2765`.

Multiple `printf` lines fire from `run()` per invocation: env build,
phase summaries (warmup / parallel / polish), final energy, total wall
time, env_build_sec/timer_scope. None inside the sweep loop. At N=10
reps per config, this is 10 blocks of stdout per cell.

**Why deferred**: not in hot loop, no GPU performance impact. The
campaign runner's stdout capture handles the volume fine.

**Trigger to fix**: only if log noise becomes a real problem. Could be
guarded by a `--quiet` mode that suppresses everything except `Final
energy` and `Total wall time` lines (which the runner parses).

---

## B6. CLI flag drift between `-gpu` and `-gpu-base`

**Files**: all three `-base/src/test_*.cpp` files lack model-parameter
flags (`--hfield`, `--ej/ec/phi`, `--j1/j2/j3`, `--jleg/jrung`,
`--j1j2`, `--j1j2j3`, `--ladder`) and `--quiet`.

**Why deferred**: G1 only runs `-gpu` baselines, not `-base` snapshots.
The `-base` snapshots are intentional pre-optimisation reference code;
their CLI surface frozen at the time of the snapshot is acceptable.

**Trigger to fix**: any decision to rebench `-base` against `-gpu` on
non-default model parameters (currently no published claim depends on
this).

---

## Round 2 deep-pass — Tier A round 2 additions (this commit)

A second-pass review on 2026-04-27 (after the round-1 Tier A fixes shipped in
`4208052`) surfaced two additional G1-blocking items missed by round 1:

- **A4** — `lanczos_process_beta_kernel` divide-by-zero guard
  (`gpu-rocm/common/scalar_traits.h:410`). The kernel computed
  `inv_nrm_out[0] = 1.0 / beta` with no guard; if `beta == 0` (invariant
  Krylov subspace exhausted), the result is `inf` → NaN propagates through
  Ritz vector construction → garbage energy. The in-loop
  `lanczos_check_beta` only fires every 3 iterations after `iter >= 4`; for
  small Krylov spaces (boundary sites with small chi) it may never run
  in-loop. Fix: clamp to 0 below `1e-300` (true subnormal underflow only).
  Single fix in shared header propagates to all 5 Lanczos-based variants
  (dmrg-gpu, dmrg2-gpu, pdmrg-gpu, pdmrg-multi-gpu, pdmrg-gpu-opt).

- **A5** — Silent flag swallow in test binaries
  (`test_dmrg_gpu.cpp:560`, `test_dmrg2_gpu.cpp:510`,
  `test_pdmrg_gpu.cpp:571`). Pattern `if (argv[i][0] == '-') continue;`
  silently ignored any unrecognized `-`-prefixed flag. A typo in the
  campaign config or a future-added flag we forget to wire would have
  produced 240 silent wrong-config runs with no error. Fix:
  `fprintf(stderr, "Unknown flag: %s\n", argv[i]); return 1;` so the
  binary exits non-zero on the first bad invocation, the campaign runner
  marks the cell as failed, and the operator sees the typo immediately.

- **B9 promoted to fix** — `gpu-rocm/common/gpu_opts.h:21` `fuse_lanczos`
  comment said "fused axpy+reorth kernel" but reorthogonalization is
  unchanged. Misleading for anyone interpreting ablation results that
  include `fuse_lanczos=on`. Fix: comment now reads
  "fused axpy+normalize kernels (reorth unchanged)". Cosmetic, no
  behavior change.

## Adversarial-round closeout (round 4) — pdmrg-family pre-G1 audit

A 4-agent adversarial review pass on the pdmrg family (pdmrg-gpu / pdmrg-gpu-base /
pdmrg-gpu-opt / pdmrg-multi-gpu) on 2026-04-27 found **3 G1-blocking bugs** that
my round-3 commits had introduced (`4033ac8`, `bb235ab`, `92bf290`). The G1 build
phase would have failed at the linker (S1) and compiler (S2); even if those were
patched, pdmrg-gpu-base would have produced wrong energies (S3). All three are
now fixed, plus one Tier A item (A1: `std::thread` exception safety).

### Tier S — fixed (would have prevented G1 from building)

- **S1** — `promote_double_to_complex` undefined in -base TUs.
  Defined in each `-gpu` impl.h but not in the `-base` versions. Hit at link
  time for `hipDoubleComplex` (Josephson) instantiation.
  Fix: copied the kernel definition into all 3 `-base` impl.h files
  (`dmrg_gpu_base_impl.h:39`, `dmrg2_gpu_base_impl.h:39`, `pdmrg_gpu_base_impl.h:41`).
  Each `-base` impl.h is a separate TU, so the `static __global__` definition
  doesn't conflict with the `-gpu` siblings.

- **S2** — `rocsolver_gesvd_auto` signature mismatch.
  All 3 `-base` SVD calls passed 3 tail args
  (`d_svd_E, rocblas_outofplace, d_svd_info`); the current scalar_traits.h
  signature requires 4 (`d_E_scratch, d_residual, d_n_sweeps, info`). I
  missed an R3-F2 signature change in `common/scalar_traits.h:167` when
  writing the round-3 commits. Fix: added `d_svdj_residual` (double*) and
  `d_svdj_n_sweeps` (rocblas_int*) to each variant's StreamWorkspace /
  member fields, allocated in constructor, freed in destructor, passed
  to all `rocsolver_gesvd_auto` call sites in svd_split + svd_split_single_site.

- **S3** — Bare `hipMemcpy` race against non-blocking streams in pdmrg-gpu-base.
  3 sites (`pdmrg_gpu_base_impl.h:814` energy readback, `:873` and `:961`
  S readback for truncation rank). Specific to pdmrg-gpu-base because round
  3 switched to `hipStreamCreateWithFlags(hipStreamNonBlocking)` for parallel
  boundary merge correctness; bare hipMemcpy on the legacy stream does NOT
  wait for non-blocking streams. dmrg-gpu-base and dmrg2-gpu-base use blocking
  streams so the same pattern there is safe. Fix: replaced with
  `hipMemcpyAsync(..., streams_[si]) + hipStreamSynchronize(streams_[si])`,
  matching the optimized pdmrg-gpu sibling.

### Tier A — fixed opportunistically

- **A1** — `std::thread` exception safety in pdmrg-gpu's `parallel_sweep`
  and `merge_and_optimize_boundaries` (and pdmrg-gpu-base's `parallel_sweep`).
  C++ contract: an exception escaping a `std::thread` function calls
  `std::terminate` immediately, abandoning peer threads' GPU work. Fix:
  wrap each lambda body in `try { ... } catch (...) { capture exception_ptr }`,
  rethrow first captured exception after `t.join()` and the per-stream sync
  barrier. On HIP error during a campaign, the operator now sees a real
  exception with stack context instead of a silent `std::terminate`.

### Tier B — deferred (not G1-blocking; documented)

These were verified real but live in variants not in the G1 campaign config
(pdmrg-gpu-opt, pdmrg-multi-gpu) or are minor hygiene issues. Trigger
conditions noted so future audits know when each becomes urgent:

- **B-Adv1** — `pdmrg-gpu-opt` and `pdmrg-multi-gpu` have the C4 "h_svd_S
  stale-read" bug: `merge_and_optimize_boundaries` reads `ws.h_svd_S[i]`
  for V = 1/S, but the GPU SVD path (default) writes to `d_svd_S` and
  never copies to host. **pdmrg-gpu is SAFE** (uses `accurate_svd` at
  boundaries which writes h_svd_S directly). **pdmrg-gpu-base is SAFE**
  (round-3 rewrite D2H-copies S for the truncation-rank decision before
  the V update reads it). Trigger to fix: any benchmark of pdmrg-gpu-opt
  or pdmrg-multi-gpu without `--cpu-svd`.

- **B-Adv2** — `pdmrg-gpu-base` `n_segments=1` returns 0.0 energy because
  the boundary-merge loop never runs and segment-sweep energies are
  discarded. Trigger: any single-segment run of this variant.

- **B-Adv3** — `pdmrg-gpu` polish skipped when outer converges, regardless
  of `n_polish > 0`. Trigger: any run with `n_polish > 0` that converges
  in the outer loop.

- **B-Adv4** — Silent flag swallow in `test_pdmrg_gpu_opt.cpp:407` and
  `test_pdmrg_multi_gpu.cpp:377` (round-2 fix never applied to these).

- **B-Adv5** — `pdmrg-gpu-opt` build script comment is false (says
  "SVD is CPU LAPACK only" — actually GPU rocsolver default).

- **B-Adv6** — `pdmrg-gpu-opt` stack-allocated `h_A[256]` arrays passed
  to `hipMemcpyAsync` in `apply_heff_single_site` (lifetime hazard).

- **B-Adv7** — `pdmrg-gpu-opt` `--batched-sweep` mode actually serializes
  on stream 0 (perf bug, not correctness).

- **B-Adv8** — `pdmrg-gpu-opt` CLI parity gap: missing 9 flags vs pdmrg-gpu.

- **B-Adv9** — `pdmrg-gpu-opt` `davidson_b` hardcoded + 134MB unconditional
  VRAM allocation regardless of `--davidson` flag.

- **B-Adv10** — `pdmrg-multi-gpu` timer scope still `include_env_build`
  (round-1 fix never applied).

- **B-Adv11** — `pdmrg-multi-gpu` hardcoded `n_polish=10` two-site polish
  (CLAUDE.md violation; same as pdmrg-gpu-base pre-round-3 was).

- **B-Adv12** — `pdmrg-multi-gpu` peer-access setup uses bare loop indices
  not `devices_[k].device_id` — broken on multi-tenant MI300X with
  non-contiguous device IDs. Critical for any future multi-MI300X campaign;
  see `docs/MULTI_GPU_INVESTIGATION.md`.

- **B-Adv13** — `pdmrg-multi-gpu` CMakeLists doesn't include `gpu-rocm/common/`;
  uses local `src/scalar_traits.h` — the β=0 guard fix from commit 9613ecd
  may not propagate to this variant.

### Tier C — cosmetic (documented, no urgency)

- pdmrg-gpu-base has stale comment "NO fused WW precompute" (it IS done now).
- pdmrg-gpu-opt timer placement correct but printf annotation missing.
- `lanczos_use_1site_` non-atomic shared bool — currently safe by use pattern,
  fragile under future refactors.

---

## Round 3 closeout — `-base` variants brought to "competent first-pass GPU"

A third deep-pass review on 2026-04-27 surfaced systemic CPU-bound patterns
in all three `-gpu-base` snapshots that went beyond what "naive" justified.
The operator instruction was unambiguous: **"zero CPU calls in any
`-gpu-*` implementation."** All three `-base` variants were rewritten to a
"competent first-pass GPU implementation" posture: device-pointer rocBLAS,
on-device tridiagonal eigensolve via `rocsolver_dsteqr`, on-device SVD
truncation via the shared `extract_cols_kernel` /
`scale_rows/cols_by_diag_kernel`, `rocsolver_gesvd_auto` instead of the
older `rocsolver_gesvd`, sweep-only timer scope matching the `-gpu`
variants, and (for the two-site / parallel variants) per-bond fused MPO
precomputation at `set_mpo()` time.

Commits:
  - `4033ac8` — `gpu-rocm/dmrg-gpu-base`
  - `bb235ab` — `gpu-rocm/dmrg2-gpu-base` (adds WW precompute)
  - `92bf290` — `gpu-rocm/pdmrg-gpu-base` (adds WW precompute, per-stream
    workspaces, non-blocking streams, full CLI surface, CLAUDE.md
    compliance for warmup/polish, single-site polish)

Round-3 punch-list (R3-1 through R3-13) status:

  - **R3-1** (timer scope drift `-base` vs `-gpu`): RESOLVED in all 3.
    The round-1 fix at commit `4208052` was applied to `-gpu` only; the
    round-3 commits apply the same fix to `-base`.
  - **R3-2** (host-pointer rocBLAS in inner loop): RESOLVED in all 3.
    Lanczos inner loop now in device-pointer mode; pointer mode toggled
    back to host before SVD/setup.
  - **R3-3** (per-iter `std::vector<Scalar> h_coeffs` heap alloc):
    RESOLVED in all 3. Reorthogonalization now uses pre-allocated
    `d_neg_overlap` device scratch.
  - **R3-4** (CPU `dstev_` instead of `rocsolver_dsteqr`): RESOLVED in
    all 3. Per-stream `d_steqr_D/E/C/info` workspaces.
  - **R3-5** (host-side SVD truncation with full D2H+H2D of U/S/Vh):
    RESOLVED in all 3. Device kernels write directly into MPS tensors
    (or into pre-allocated `d_svd_work` for the absorb GEMM).
  - **R3-6** (wrong SVD algorithm `rocsolver_gesvd` vs `_auto`):
    RESOLVED in all 3.
  - **R3-7** (dmrg2-gpu-base WW host-rebuild per Lanczos iter): RESOLVED.
    `precompute_WW()` in `set_mpo()` builds `d_WW_[bond]` once on host
    (set_mpo is outside the timed region) and uploads to device.
  - **R3-8** (pdmrg-gpu-base WW host-rebuild): RESOLVED. Same pattern
    as R3-7.
  - **R3-9** (pdmrg-gpu-base silent flag swallow): RESOLVED. Test driver
    now exits non-zero on unknown flag, identical to the round-2 fix in
    `-gpu` test files.
  - **R3-10** (pdmrg-gpu-base CLAUDE.md violations: hardcoded n_warmup=3,
    two-site polish, hardcoded n_polish=10): RESOLVED. `run()` signature
    now takes `n_polish`. Defaults `n_warmup=1, n_polish=0`. Polish phase
    uses `sweep_LR_full_1site()` / `sweep_RL_full_1site()` (single-site).
    CLI `--warmup`, `--polish`, `--segments`, `--local-sweeps` flags
    added.
  - **R3-11** (pdmrg-gpu-base blocking `hipStreamCreate` + sync hipMemcpy
    serialization): RESOLVED. `hipStreamCreateWithFlags(hipStreamNonBlocking)`.
  - **R3-12** (pdmrg-gpu-base plain `rocsolver_gesvd` at boundaries):
    NOT FIXED in this round. `accurate_svd` (Stoudenmire recursive
    refinement) is a `-gpu`-only optimization for boundary numerical
    stability; keeping `-base` on the plain rocSOLVER SVD preserves a
    real algorithmic distinction between the two variants. Documented
    here so future audits don't re-flag.
  - **R3-13** (pdmrg-gpu-base per-call `h_psi_R` heap alloc + sync
    hipMemcpy): RESOLVED. Pre-allocated `d_psi_R` in StreamWorkspace;
    `form_theta_with_V` uploads V to ws.d_svd_S as scratch and uses
    `scale_rows_by_diag_kernel` for the row scaling on device.

What this means for benchmarking:
  - All 6 `-gpu` and `-gpu-base` variants now use the same timer scope
    (sweep_only). `-gpu / -gpu-base` wall-time ratios are now apples-
    to-apples.
  - `-base` snapshots represent a "competent first-pass GPU
    implementation" baseline; `-gpu` adds dual-stream pipelining (where
    applicable), HIP graph capture, RSVD, batched GEMM + GpuOpts ablation
    framework, sparse-MPO compaction, D_PAD, and (in dmrg2-gpu /
    pdmrg-gpu) on-device WW precompute kernels and the `accurate_svd`
    boundary path. Speedup ratios now attribute to those optimizations
    rather than to artifactual CPU-roundtrip patterns.

---

## Round 2 — items remaining deferred (B7-B12)

- **B7**: RSVD heap-alloc pattern also exists in `dmrg-gpu` (round 1 only
  flagged dmrg2-gpu and pdmrg-gpu). Extends the existing B2 entry. G1
  doesn't pass `--rsvd`; revisit when running RSVD-on benchmarks.
- **B8**: `pdmrg-gpu` recalibration inflates `parallel_sec` when
  `--recal > 0` (build_initial_environments + serial 2-site sweep
  included inside the `t_start → t_parallel` interval). G1 default
  `n_recal=0`; latent if accidentally enabled.
- **B10**: `dmrg2-gpu` `if (dE < tol_ && sweep > 0) break;` enforces an
  implicit 2-sweep minimum (would prevent detecting genuine 1-sweep
  convergence). Harmless for G1 (sweeps ≥ 5 in config).
- **B11**: `pdmrg-gpu` boundary merge workspace round-robin
  (`si = idx % n_avail_streams`) could collide if `n_active >
  n_avail_streams`. Currently safe for campaign grid (`n_segments ≤ 4`,
  `n_active ≤ n_segments/2 ≤ 2`).
- **B12**: `pdmrg-gpu` HIP graph cache key omits `D_mpo_`. Currently safe
  because the campaign uses one MPO per fresh subprocess; would matter
  if the solver object were reused across MPOs.

---

## What was fixed (Tier A round 1, commit 4208052)

For reference; see commit message for full details.

- **A1**: Timing scope changed from `include_env_build` to `sweep_only`
  in all 3 baselines (`dmrg-gpu`, `dmrg2-gpu`, `pdmrg-gpu`). `t_start`
  now captured AFTER `build_initial_environments()`. "Total wall time"
  is now sweep-only; "Environment build: X s" remains as a separate
  diagnostic line. Per-phase deltas in pdmrg-gpu (warmup/parallel/polish)
  naturally become sweep-only since they reference `t_start`.
- **A2**: `invert_nrm_kernel` in `dmrg2-gpu/src/dmrg2_gpu_impl.h:1284`
  launched on `stream_` instead of default stream `0`. Eliminates a
  potential cross-stream data race that could silently corrupt Ritz
  vectors.
- **A3**: Hardcoded constructor `tol` values in
  `dmrg-gpu/src/test_dmrg_gpu.cpp` and
  `dmrg2-gpu/src/test_dmrg2_gpu.cpp` changed from `1e-12` to `1e-10` to
  match the user-stated convergence target and the constructor default.
  `pdmrg-gpu` was already at `1e-10`. Float-equality thresholds
  (`if (std::abs(J1 - 1.0) < 1e-12)`) intentionally NOT changed — those
  are unrelated parameter checks.

No CLI flag added for `--tol` — the constructor signature accepts `tol`
already, and a flag can be added later if runtime tuning is needed.
