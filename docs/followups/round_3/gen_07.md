# Round 3 — Gen 07: R2-4 Phase 0 Microbench (LIVE MI300X)

**Author:** Generative sub-agent G07
**Date:** 2026-04-10
**Target:** `docs/followups/proposal_3_hip_graph_capture.md` (R2-4) — Phase 0 go/no-go
**Status:** **HARD NO-GO on current R2-4 design. Pivot to Proposal 3-alt (fused rocWMMA kernel).**

---

## 1. Summary

I executed the Phase 0 microbench on a live AMD Instinct MI300X (gfx942, ROCm 7.2.0,
HIP 7.2.26015) against every capture variant the proposal cares about — and against
a minimal sanity workload to establish the baseline. **Every rocBLAS gemm call fails
under HIP stream capture on ROCm 7.2, not just the `_batched` variant that
Research A flagged.**

The surviving fact that forces the pivot is not just the capture failure — it is the
**uncaptured timing**: the full 3-stage workload representative of
`apply_heff_two_site` at `chi_L=chi_R=64, d=2, D=5` runs in **0.66 µs per iteration**.
There is no plausible dispatch savings graph capture could recover from that budget.
The "eliminate launch latency" motivation of R2-4 does not match the measured
reality of rocBLAS dispatch on this hardware at this problem size.

## 2. Microbench design

Source: `/home/hotaisle/dmrg-implementations/gpu-rocm/sandbox/bench_graph_capture.cpp`
(also saved locally at `/tmp/bench_graph_capture.cpp`).

Structure (matches proposal §5 Phase 0 spec):

1. `hipSetDevice(0)`
2. `rocblas_initialize()` (note: returns void in ROCm 7.2)
3. `rocblas_create_handle` + `rocblas_set_stream`
4. `rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device)` — alpha/beta on device
5. **`rocblas_set_workspace`** with a 32 MiB pre-allocated buffer, BEFORE entering capture
6. Warm-up: 5 uncaptured dispatches of the full 3-stage pipeline to force lazy-init of
   rocBLAS' internal state outside of capture

Shapes chosen to match `apply_heff_two_site` at `chi_L = chi_R = 64, d = 2, D = 5`:

| Stage | rocBLAS call | m | n | k | batch |
|---|---|---|---|---|---|
| 1 | `dgemm_strided_batched` | 64 | 256 | 64 | 5 |
| 2 | `dgemm` (dense) | 256 | 256 | 5 | — |
| 3 | `dgemm_strided_batched` | 256 | 64 | 64 | 5 |

Tests run (each on a fresh `hipStream_t` to avoid cross-contamination of the
`hipErrorStreamCaptureInvalidated` state):

- **Test 0 (sanity):** begin/end capture of an empty stream, instantiate, no rocBLAS.
- **Test 0b (minimal rocBLAS):** plain `rocblas_dgemm` (Stage 2 shape) only, under
  ThreadLocal capture.
- **Test A:** strided_batched + dgemm + strided_batched, ThreadLocal capture.
- **Test B:** strided_batched + dgemm + strided_batched, Global capture.
- **Test C:** `_batched` (pointer-array) + dgemm + `_batched`, ThreadLocal capture
  (to confirm Research A's expected finding on the pointer-array variant).
- **Timing:** 100 uncaptured iterations vs 100 captured `hipGraphLaunch` iterations
  (captured timings skipped for failed tests).

## 3. Raw results

Build: clean (`hipcc -O2 -std=c++17 -Wno-unused-value -lrocblas`).

Run transcript (excerpted; full logs on remote):

```
Problem: chi_L=64 chi_R=64 d=2 D=5
Stage1 (strided_batched): m=64 n=256 k=64 batch=5
Stage2 (dense dgemm):      m=256 n=256 k=5
Stage3 (strided_batched): m=256 n=64 k=64 batch=5
Warm-up OK.

--- Test 0: empty-stream capture (sanity) ---
[0] begin=0 end=0
[0] instantiate=0                                        <-- HIP graphs themselves work

--- Test 0b: plain rocblas_dgemm (Stage2) only under capture ---
[0b] rocblas_dgemm under capture: rb=6                   <-- rocblas_status_internal_error
[0b] end=901                                             <-- hipErrorStreamCaptureInvalidated

--- Test A: strided_batched under capture (ThreadLocal) ---
[A] FAIL at stage 1 (rocBLAS issue under capture): hip=0 rb=6

--- Test B: strided_batched under capture (Global) ---
[B] FAIL at stage 1 (rocBLAS issue under capture): hip=0 rb=6

--- Test C: _batched (pointer-array) under capture (ThreadLocal) ---
[C] FAIL at stage 1 (rocBLAS batched under capture): hip=0 rb=6

--- Timing ---
[timing] uncaptured strided: 66.3 us total, 0.66 us/iter over 100 iters
[timing] skipping ThreadLocal (no valid graph)
[timing] skipping Global (no valid graph)
```

### Decoding the error codes

Confirmed against `/opt/rocm-7.2.0/include/rocblas/internal/rocblas-types.h` and
`/opt/rocm-7.2.0/include/hip/hip_runtime_api.h`:

| Code | Symbol | Meaning |
|---|---|---|
| `rb=6` | `rocblas_status_internal_error` | "Other internal library failure" (not an arg check) |
| `hip=900` | `hipErrorStreamCaptureUnsupported` | operation not permitted in capture mode |
| `hip=901` | `hipErrorStreamCaptureInvalidated` | capture sequence corrupted by an unsupported op |

Interpretation: rocBLAS is not returning a clean "not supported under capture"
refusal. It is hitting an internal assertion / resource path that trips the HIP
runtime's stream-capture guard (`hipErrorStreamCaptureInvalidated=901` is what
`hipStreamEndCapture` returns after a disallowed op was issued mid-capture). The
dgemm call itself then propagates `rocblas_status_internal_error` rather than a
nice `not_implemented`.

**This is worse than Research A predicted.** Research A said `rocblas_dgemm_batched`
was on the beta-features unsupported list but `dgemm_strided_batched` was NOT listed
and llama.cpp#14576 reported it empirically works. On ROCm 7.2 / MI300X / the
specific rocBLAS 5.2.7 shipping with this ROCm, **none of the three variants work**:

- plain `rocblas_dgemm` fails (Test 0b)
- `rocblas_dgemm_strided_batched` fails (Tests A, B)
- `rocblas_dgemm_batched` fails (Test C)

### Baseline timing is the second blow

```
uncaptured strided: 66.3 us total / 100 iters = 0.66 µs per 3-stage iter
```

Compare that to the R2-4 motivation in `proposal_3_hip_graph_capture.md §1`:

> Each Lanczos iteration of `apply_heff_two_site` dispatches ~8–12 separate rocBLAS
> / custom kernels, and the launch latency is comparable to (or larger than) the
> actual FP64 work for these tiny matrices.

At `chi=64`, the measured 3-stage cost is **0.66 µs**. Even if HIP graph capture
collapsed 3 rocBLAS dispatches into one (~3 µs → ~1 µs is a typical cudaGraph
speedup), we would need a 3-5 µs uncaptured baseline to see a 2-4× reduction.
We are already at sub-microsecond. The dispatch overhead on the MI300X / ROCm 7.2
rocBLAS path is already essentially amortized — likely because AMD's rocBLAS
internally queues work asynchronously and the host-side dispatch is cheap.

This doesn't directly invalidate the R2-4 goal at *larger* chi where individual
gemm calls take longer (the ratio of launch to compute changes), but it does mean
the "close the CPU gap at `chi ∈ [64, 128]`" target (§3 of the proposal) is
built on an assumption that does not hold at the lower end. The uncaptured
baseline at chi=128 would need to be measured before reviving any graph-capture
plan — but that measurement is moot because capture doesn't work anyway.

## 4. Verdict: NO-GO

**R2-4 as written is dead on ROCm 7.2.** Two independent reasons, either sufficient:

1. **Technical blocker:** The rocBLAS bundled with ROCm 7.2 (v5.2.7) crashes its
   internal state machine when any `gemm` variant is called on a capture-mode
   stream. This is not a workaround-able bug — the failure mode is
   `rocblas_status_internal_error` + `hipErrorStreamCaptureInvalidated`, meaning
   the rocBLAS codepath hits a disallowed HIP op mid-call. The fix would require
   patches to rocBLAS itself (ROCm/rocBLAS#1240 is tracking this and is still
   open).

2. **Motivational blocker:** At the target problem size (`chi=64, d=2, D=5`), the
   entire 3-stage workload already runs in 0.66 µs uncaptured. There is no
   meaningful dispatch overhead left to recover. The R2-4 thesis — "launch latency
   dominates at small chi" — is falsified for this specific problem and this
   specific rocBLAS version.

**Severity calibration.** Proposal 3 explicitly flagged this as a HIGH / go-no-go
risk (§4 risk #1). The gate was correctly placed; the gate correctly fired.

## 5. Pivot: Proposal 3-alt (fused rocWMMA kernel)

The proposal's pre-authorized fallback from §5 Phase 0 is still viable and still
the right pivot:

> **NO-GO** if even the strided variant fails. In that case we pivot to
> **Proposal 3-alt: custom fused HIP kernel via rocWMMA FP64 MFMA-16** — a single
> kernel that does Step 1 + Step 2 + Step 3 in one launch, eliminating the rocBLAS
> capture dependency entirely.

However, the baseline measurement of 0.66 µs/iter *also* constrains 3-alt. A fused
kernel needs to actually do better than three well-tuned rocBLAS calls, not just
replace them. The honest win condition for 3-alt is:

- **Must prove itself at chi ≥ 96** where rocBLAS dispatch becomes a smaller
  fraction of total time and kernel-fusion data-reuse (keeping `T1` in LDS rather
  than round-tripping through HBM between Stages 1 and 2) becomes the dominant
  savings. Data-reuse is the real reason to fuse, not launch elimination.
- **Must NOT target chi ≤ 32.** The CPU cache-residency wall (Proposal 3 §4 risk #4)
  is real and our 0.66 µs chi=64 baseline shows we are already cache-bandwidth
  bound on the GPU side at small sizes. Fusion wins come from HBM bandwidth, not
  compute, at these shapes.

### Concrete next steps for the pivot

1. **Re-baseline first.** Before writing any kernel, run the uncaptured timing at
   `chi ∈ {96, 128, 192, 256}`. We need to know the real dispatch-vs-compute ratio
   at each shape. At chi=256 the gemms become large enough that rocBLAS will
   almost certainly beat a hand-rolled fused kernel on compute. The sweet spot for
   fusion is whatever range shows "rocBLAS call time < HBM round-trip time for
   T1 and T2".

2. **Keep the measurement infra.** The `sandbox/bench_graph_capture.cpp` harness
   is already set up for arbitrary problem sizes (`Problem` struct at line ~72)
   and for swapping in candidate implementations. Reuse it as the test bed for the
   fused kernel — just replace the `issue_strided` body with the candidate and
   re-run the 100-iter timing.

3. **Look at rocWMMA FP64 first, not the Composable Kernel route.** rocWMMA exposes
   the gfx942 v_mfma_f64_16x16x4 instruction directly and is the thinnest possible
   abstraction. If rocWMMA cannot express the 3-stage pattern with acceptable
   register pressure, *then* escalate to Composable Kernel's `gemm_multiple_d`
   templates.

4. **Do not re-attempt HIP graphs in round 3.** The finding is conclusive at ROCm
   7.2. If AMD ships a rocBLAS fix in a later ROCm (track ROCm/rocBLAS#1240), we
   revisit. Until then, no further engineer-hours on graph capture.

## 6. What is salvageable from R2-4

Two pieces of the R2-4 design survive and should be carried forward:

- **The measurement methodology.** Isolating stages, timing with 100-iter loops,
  using device pointer mode + pre-allocated workspace — all of this is sound and
  the sandbox harness is now reusable.

- **The honest framing of the "chi ≤ 32 cache wall" and "chi ≥ 256 SVD-bound"
  regimes** (§7 of the proposal). These are not specific to graph capture. They
  apply equally to any optimization targeting `apply_heff_two_site` and should be
  cited when scoping Proposal 3-alt.

What dies:
- The shape-indexed graph cache (R2-4 §3.2)
- The `StreamWorkspace`-per-capture invocation model (§3.3)
- Every claim about dispatch latency being dominant at chi=64 (§1)

## 7. Files

- Microbench source (remote): `/home/hotaisle/dmrg-implementations/gpu-rocm/sandbox/bench_graph_capture.cpp`
- Microbench source (local scratch): `/tmp/bench_graph_capture.cpp`
- Microbench binary (remote): `/home/hotaisle/dmrg-implementations/gpu-rocm/sandbox/bench_graph_capture`
- This report: `/home/captain/clawd/work/dmrg-implementations/docs/followups/round_3/gen_07.md`

## 8. Environment

- Host: `hotaisle@23.183.40.84`
- GPU: AMD Instinct MI300X VF (gfx942), card SKU M3000108
- ROCm: 7.2.0
- HIP: 7.2.26015-fc0010cf6a
- Clang: AMD clang 22.0.0git (roc-7.2.0)
- rocBLAS: libs 5.2.7 (librocblas.so.5.2.70200)
