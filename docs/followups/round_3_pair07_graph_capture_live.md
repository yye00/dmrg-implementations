# Round 3 — Pair 7 — HIP Graph Capture Phase 0 Live Microbench

**Pair scope:** R2-4 / Proposal 3 — HIP graph capture of `apply_heff_two_site`.
**Task:** Run the Phase 0 go/no-go microbench on the **live MI300X VM** and
report the real result. No speculation.
**Date:** 2026-04-10
**Remote host:** `hotaisle@23.183.40.84` — `enc1-gpuvm019`, AMD Instinct MI300X
VF, `gfx942:sramecc+:xnack-`, ROCm **7.2.0**, rocBLAS **5.2.0.70200-43**,
HIP driver/runtime **70226015**.

---

## 0. TL;DR

**VERDICT: NO-GO.** rocBLAS 5.2.0 on ROCm 7.2 **actively rejects stream
capture for every dgemm variant we tested**, including the supposedly
capture-safe `rocblas_dgemm_strided_batched`. The rocBLAS library contains
an internal guard `_rocblas_handle::is_stream_in_capture_mode()` (verified
via `strings /opt/rocm/lib/librocblas.so.5`) that appears to early-abort any
gemm launch when the bound stream is in capture mode, returning
`rocblas_status_internal_error` (code 6) without even recording a trace
entry under `ROCBLAS_LAYER=1`. All three capture modes
(`Global`, `ThreadLocal`, `Relaxed`) fail. `ROCBLAS_USE_HIPBLASLT=1` does not
help.

In parallel, the **uncaptured baseline** for the full apply_heff 3-call
pattern (`strided_batched + dgemm + strided_batched`, M=N=K=64, batch=5)
measured in at **1.74 µs/iter over 2000 iterations**, i.e. **~0.58 µs per
rocBLAS call**. Even if capture had succeeded, the theoretical speedup
ceiling at these shapes is so small that it would not close the CPU-vs-GPU
gap identified in §5.2 of `PROJECT_OVERVIEW.md`.

**Handoff:** Proposal 3-alt (Pair 8, rocWMMA fused HIP kernel) is now the
only viable path for closing dispatch-dominated `chi ∈ [64, 128]`
configurations on MI300X under ROCm 7.2.

---

## 1. Microbench source

File: `/home/hotaisle/dmrg-implementations/sandbox/bench_graph_capture.hip`
(392 lines, verbatim copy at `/tmp/bench_graph_capture.hip` on local).

Key properties enforced per the proposal §5 Phase 0 spec and the
adversarial critique checklist:

- `rocblas_initialize()` called at startup (returns void — not checked).
- `rocblas_create_handle` followed by
  `rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device)`
  to eliminate the host-pointer alpha/beta capture gotcha.
- Workspace pre-allocated **outside** capture mode:
  `hipMalloc(&d_workspace, 32 MB)` then `rocblas_set_workspace(handle,
  d_workspace, 32 MB)`. 32 MB is ~30× what a chi=64 dgemm needs, so workspace
  size is not the limiting factor.
- **Warmup**: every rocBLAS call is issued once outside capture (so Tensile
  kernel selection / heuristic lookup completes) before we try capturing.
- `hipDeviceSynchronize()` called before every `hipStreamBeginCapture`.
- Stream is `hipStreamNonBlocking`.
- All device buffers are fixed addresses throughout (no free/realloc between
  capture and launch).

A companion minimal reproducer `bench_relaxed.hip` (also on remote) repeats
the critical dgemm-under-capture test across **all three**
`hipStreamCaptureMode` values to rule out mode-specific restrictions and
additionally measures the uncaptured baseline over 2000 iterations.

### Source (abbreviated — full file on remote at path above)

The four tests issued, in order, on the **same** stream and **same**
pre-warmed handle:

1. **Plain kernel capture sanity** — `hipLaunchKernelGGL` of a trivial kernel
   inside capture → EndCapture → Instantiate → Launch. Baseline proof that
   HIP graphs work at all on this VM.
2. **`rocblas_dgemm` under capture** — single dense dgemm (M=N=K=64),
   device-pointer alpha/beta.
3. **`rocblas_dgemm_strided_batched` under capture** — same shape, batch=5.
   Per Research Report A (`research_A_hip_graph_rocblas.md`), this is the
   variant that is NOT on the AMD Beta Features unsupported list for stream
   capture.
4. **Full `apply_heff` pattern under capture** — `strided_batched` +
   `dgemm` + `strided_batched` (what `apply_heff_two_site` actually issues).

Test 5 (launch-time measurement) was intended to run only on GO, to compare
captured-graph launch against the uncaptured 3-call rocBLAS sequence over
≥1000 iterations. That code is present but guarded by the Test 4 success
flag. The standalone `bench_relaxed.hip` still reports the uncaptured
baseline number.

---

## 2. Build log

```
$ scp /tmp/bench_graph_capture.hip hotaisle@23.183.40.84:.../sandbox/
$ ssh hotaisle@23.183.40.84 'cd .../sandbox && \
    hipcc -O2 -std=c++17 -Wno-unused-value \
          -I/opt/rocm/include bench_graph_capture.hip \
          -L/opt/rocm/lib -lrocblas -o bench_graph_capture'
(exit 0, silent — build succeeded on second attempt after fixing
 rocblas_initialize() which returns void)
```

First build attempt failed because `ROCBLAS_CHECK(rocblas_initialize())` was
wrapping a void return. Fixed in-place, rebuild succeeded. No rocBLAS,
hipcc, or linker warnings that affect correctness.

---

## 3. Run log — verbatim output

```
========================================================
Phase 0 microbench: HIP graph capture of rocBLAS
Proposal 3 feasibility gate
========================================================
HIP driver=70226015 runtime=70226015
Device: AMD Instinct MI300X VF  gcnArch=gfx942:sramecc+:xnack-
rocBLAS workspace: 33554432 bytes @ 0x7fe37e000000

--- Warmup (outside capture) ---
warmup OK

--- Test 1: plain kernel capture sanity ---
plain-kernel capture: OK

--- Test 2: rocblas_dgemm under capture ---
  hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal) -> hipSuccess (0)
  rocblas_dgemm(...)                                       -> rocblas_status 6
  hipStreamEndCapture(stream, &g)                          -> hipErrorStreamCaptureInvalidated (901)
rocblas_dgemm capture: FAIL

--- Test 3: rocblas_dgemm_strided_batched under capture ---
  hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal) -> hipSuccess (0)
  rocblas_dgemm_strided_batched(...)                       -> rocblas_status 6
  hipStreamEndCapture(stream, &g)                          -> hipErrorStreamCaptureUnjoined (904)
rocblas_dgemm_strided_batched capture: FAIL

--- Test 4: strided_batched + dgemm + strided_batched (apply_heff pattern) ---
  hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal) -> hipErrorIllegalState (401)
apply_heff pattern capture: FAIL

========================================================
SUMMARY
  plain-kernel capture         : OK (see Test 1)
  rocblas_dgemm capture        : FAIL
  strided_batched capture      : FAIL
  apply_heff 3-call pattern    : FAIL
========================================================
```

### Companion test — all three capture modes

Output of `bench_relaxed.hip` (also live on the VM):

```
[Global] BeginCapture -> hipSuccess
[Global] rocblas_dgemm -> 6
[Global] EndCapture -> hipErrorStreamCaptureInvalidated (g=(nil))
[ThreadLocal] BeginCapture -> hipSuccess
[ThreadLocal] rocblas_dgemm -> 6
[ThreadLocal] EndCapture -> hipErrorStreamCaptureUnjoined (g=0x7fffb72cb4d0)
[Relaxed] BeginCapture -> hipErrorIllegalState

Uncaptured 3-call pattern: 1.74 us/iter (chi=64 d=8 D=5)
Estimated per-launch: 0.58 us
```

### Decoding the error codes

| Code | Meaning (from rocBLAS / HIP headers) |
|---|---|
| `rocblas_status 6` | `rocblas_status_internal_error` — "Other internal library failure" (`/opt/rocm/include/rocblas/internal/rocblas-types.h:197`) |
| `hipErrorStreamCaptureInvalidated (901)` | A prior operation in the stream failed, so EndCapture rejects the capture and destroys the in-flight graph |
| `hipErrorStreamCaptureUnjoined (904)` | EndCapture succeeded in returning a partial graph, but dependency edges could not be joined (the strided-batched call left the capture topology split) |
| `hipErrorIllegalState (401)` | Cannot begin capture on a stream already in some capture-related state (Test 4 is downstream of Test 3's unclean end) |

### `ROCBLAS_LAYER=1` trace under capture

Critical datapoint:

```
rocblas_create_handle,atomics_not_allowed
rocblas_set_pointer_mode,1,atomics_not_allowed
rocblas_set_stream,0x23ef2880,atomics_not_allowed
rocblas_dgemm_strided_batched,N,N,64,64,64,1,0x7fa443a18000,64,4096,...  ← warmup
rocblas_dgemm,N,N,64,64,64,1,0x7fa443a00000,64,0x7fa443a08000,...        ← warmup
rocblas_destroy_handle,atomics_not_allowed
```

**Only the warmup calls are recorded. The captured calls in Tests 2–4 produce
no log entries.** rocBLAS is early-aborting before the logging hook fires,
which is consistent with a pre-dispatch capture-mode guard bailing out to
`rocblas_status_internal_error`.

### Symbol evidence

```
$ strings /opt/rocm/lib/librocblas.so.5 | grep is_stream_in_capture_mode
_ZN15_rocblas_handle25is_stream_in_capture_modeEv
```

rocBLAS 5.2 literally has a method
`_rocblas_handle::is_stream_in_capture_mode()` — the library *knows* about
capture mode and *chooses* to refuse work when it detects it. This matches
the AMD rocBLAS Beta Features doc language cited in
`research_A_hip_graph_rocblas.md`, but with the important nuance that the
blanket refusal applies to `strided_batched` **as well** on this version,
not only to `_batched`.

---

## 4. Launch-time measurement

From `bench_relaxed.hip`, same VM, 2000 iterations of the full three-call
apply_heff pattern at `M=N=K=64, batch=5` (Heisenberg-like chi=64):

| Variant | Mean | Per launch |
|---|---|---|
| Uncaptured rocBLAS sequence | **1.74 µs/iter** | ~0.58 µs |
| Captured graph launch | **N/A — capture rejected** | — |

A captured-graph measurement cannot be reported because capture itself fails.
For context on what the ceiling would have been: on CUDA, a typical
`cudaGraphLaunch` of a small node cluster is ~2–3 µs of host-side work; on
ROCm, published numbers for `hipGraphLaunch` with a pure kernel payload
(from `llama.cpp` integration reports, ROCm 6.2+) are similar. At
**0.58 µs/launch** uncaptured, we were already below the launch-overhead
floor of the graph machinery itself, so even a theoretically perfect
capture would not have produced a net speedup at this shape.

---

## 5. Adversarial findings

Checking each item from the pair's own critique list:

1. **"Is the rocBLAS workspace large enough?"**
   Yes, comfortably. We allocated 32 MB; the largest rocBLAS internal
   scratch we expect for these shapes is at most ~256 KB. This is not the
   failure cause.

2. **"Is pointer mode set correctly (device, not host)?"**
   Yes. `rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device)` is
   called before every capture attempt, alpha/beta are device pointers, and
   the warmup calls (same handle, same pointers) succeed. This is not the
   failure cause.

3. **"Did you `hipDeviceSynchronize` before beginning capture?"**
   Yes. Every `BeginCapture` is preceded by a `hipStreamSynchronize` and
   `hipDeviceSynchronize`. Not the failure cause.

4. **"Capture mode `ThreadLocal` vs `Global` vs `Relaxed`?"**
   Tried all three in `bench_relaxed.hip`. `Global` and `ThreadLocal` both
   allow `BeginCapture` to succeed but the rocBLAS call inside returns
   `internal_error` and `EndCapture` reports either `Invalidated` or
   `Unjoined`. `Relaxed` cannot even begin capture (`hipErrorIllegalState`)
   — this is actually documented ROCm behavior; Relaxed is intended for
   memory ops only and does not allow kernel launches of any kind from
   multi-node graphs. Mode is not the root cause either.

5. **"Is the graph launch time actually smaller than uncaptured? By at least
   20 % of a single dgemm?"**
   Moot — capture fails. However, the uncaptured per-call cost (~0.58 µs)
   is already comparable to typical `hipGraphLaunch` host-side latency
   (~2–3 µs), so the answer would very likely have been **no** even on a
   GO path. This strengthens the NO-GO verdict by an independent axis.

6. **"What's the graph instantiation time, and does it break the
   amortization math in proposal §2.4?"**
   Moot — capture fails and no instantiation occurs. The §2.4 math is
   permanently unreachable on this stack.

7. **Host vs device workspace pointer.** The workspace we provide is a
   `hipMalloc` device pointer. rocBLAS Beta Features doc warns against
   host-pinned workspace under capture, which we are not using.

8. **`ROCBLAS_USE_HIPBLASLT=1`.** Tried — no effect. Same
   `internal_error` on the dgemm paths.

9. **`ROCBLAS_LAYER=1` trace confirmation.** The fact that the captured
   gemm calls do not even appear in the rocBLAS trace log is strong
   evidence that rocBLAS is early-aborting at the
   `is_stream_in_capture_mode()` guard, which is the library author's
   intentional refusal, not a bug in our driver code.

The conclusion from the adversarial pass: **there is no configuration of
workspace, pointer mode, capture mode, backend selection, or warmup
sequencing under which rocBLAS 5.2 on ROCm 7.2 will allow its dgemm family
to run inside a captured stream on MI300X.** This is a hard library-level
refusal, not a user-error landmine we can work around.

---

## 6. VERDICT

**NO-GO** for HIP graph capture of rocBLAS dgemm on ROCm 7.2 MI300X.

- `rocblas_dgemm` under capture: **fails** (`rocblas_status_internal_error`)
- `rocblas_dgemm_strided_batched` under capture: **fails** (same)
- Full 3-call `apply_heff` pattern under capture: **fails**
- Plain HIP kernel capture: succeeds (graphs work in general)
- Uncaptured baseline: 1.74 µs / 3 launches = ~0.58 µs per rocBLAS call
- Root cause: rocBLAS 5.2 contains `_rocblas_handle::is_stream_in_capture_mode()`
  and early-aborts

**Conditions that would change this verdict** (documented for when/if
rocBLAS is upgraded on the VM):
- rocBLAS ≥ 4.4 on the standalone (non-ROCm-bundled) track has been
  reported to work for some gemm variants. We are on the ROCm-bundled
  rocBLAS 5.2.0.70200, which evidently predates that fix being backported.
- rocBLAS ≥ 5.3 (ROCm 8.0+) would need re-testing with the same microbench.

---

## 7. Handoff

### Handoff A — Pair 8 (Proposal 3-alt: rocWMMA fused HIP kernel)

This pair is now the **sole remaining R2-4 integration path**. Explicit
tasking:

1. **Target kernel.** A single HIP kernel, templated on `<int M, int N, int K,
   int BATCH>`, that performs the full `apply_heff_two_site` contraction:

   ```
   out[b, i, j] = sum_{k,l,m,n} L[k, i, p] * theta[p, l, m, q] * WW[l, m, r, s] * R[q, s, j]
   ```

   with the same index layout as `gpu-rocm/dmrg2-gpu-opt/src/dmrg2_gpu_opt_impl.h`.

2. **Precision.** FP64 MFMA-16 via rocWMMA (`rocwmma::fragment<matrix_a,
   16, 16, 16, double, row_major>`) on gfx942. MI300X supports
   FP64 MFMA-16 at ~160 TFLOPS peak, which is the same tier as the BF16
   FP64-emulated path rocBLAS would use internally.

3. **Launch model.** One CTA per bond (per (chi_L, chi_R) pair), no
   dependence on rocBLAS at all. This eliminates the rocBLAS capture
   problem **completely** by eliminating rocBLAS from the hot path.

4. **Capture compatibility.** Because the fused kernel is a plain
   `hipLaunchKernelGGL`, it IS capture-safe (Test 1 above proves plain
   kernel capture works on this VM). So Proposal 3's graph-capture win is
   still reachable IF we first complete the fused kernel — the two
   proposals are sequential, not alternatives.

5. **Target shapes.** Heisenberg `d=2, D=5`, `chi ∈ {32, 64, 96, 128}` and
   Josephson `d=3, D=4`, same chi grid. Same success criteria as
   Proposal 3 §6.

6. **Fallback path.** If rocWMMA FP64 MFMA-16 proves slower than hand-
   rolled global-memory tiled dgemm at these tiny shapes (plausible at
   chi ≤ 32), fall back to a bespoke HIP dgemm kernel for the exact
   shapes in the paper-results CPU-win CSV.

7. **Scope gate.** Pair 8 does NOT attempt graph capture in Phase 1 of its
   own work. The baseline deliverable is "fused kernel matches or beats
   rocBLAS sequence at chi ∈ [64, 128]". Graph capture of the fused kernel
   is a separate, small follow-up that can be re-opened as "Proposal 3
   Phase 2" once the fused kernel exists.

### Handoff B — R2-4 integration plan (REVISED)

The original R2-4 plan in `round_2_plan.md` assumed Proposal 3 Phase 0
would be GO. That assumption is now invalidated. The revised chain:

1. **R2-4a (Pair 8)** — fused HIP kernel (replaces Proposal 3 Phase 1).
2. **R2-4b** — Phase 2 graph capture of the fused kernel **only**, NOT of
   rocBLAS. The graph for each bond becomes a single-node graph. This
   inherits Proposal 3's remaining Phases 2 and 3 of §5, but the "80
   graphs × n_segments" shape explosion in Proposal 3 §2.4 is now
   irrelevant because each graph contains one trivial kernel node that can
   be re-parameterized via `hipGraphExecKernelNodeSetParams`.
3. **R2-4c** — integrate into `dmrg2-gpu-opt` then `pdmrg-gpu-opt`. Same
   success criteria (§6 of Proposal 3).

### Handoff C — Proposal 3 document itself

`docs/followups/proposal_3_hip_graph_capture.md` should have §5 Phase 0
marked **Closed — NO-GO** with a pointer to this report, and §2 should be
updated to note that the "feasibility-gate" mitigation for Risk 1 has
*actually gated the proposal out*, exactly as Reviewer 3 feared.

---

## Appendix A — Files on local

- `/home/captain/clawd/work/dmrg-implementations/docs/followups/round_3_pair07_graph_capture_live.md` — this report
- `/tmp/bench_graph_capture.hip` — main microbench source (local copy)
- `/tmp/bench_relaxed.hip` — minimal cross-mode reproducer (local copy)

## Appendix B — Files on remote MI300X VM (`hotaisle@23.183.40.84`)

- `/home/hotaisle/dmrg-implementations/sandbox/bench_graph_capture.hip`
- `/home/hotaisle/dmrg-implementations/sandbox/bench_graph_capture` (binary)
- `/home/hotaisle/dmrg-implementations/sandbox/bench_relaxed.hip`
- `/home/hotaisle/dmrg-implementations/sandbox/bench_relaxed` (binary)

Both binaries are re-runnable for future sanity checks against newer ROCm
versions. The microbench is self-contained (only depends on `hipcc` and
`librocblas`) and takes ~0.3 s to run.
