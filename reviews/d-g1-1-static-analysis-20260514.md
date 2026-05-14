# D-G1-1 static analysis — 2026-05-14

Forensic analysis of the `dmrg2-gpu-opt` hang (D-G1-1, deferred per
`reviews/g1-deferred-bugs-20260509.md`). The full variant is `VARIANT_SKIP`'d
for the entire G1 campaign — this document is for the catch-up run.

## Verdict

**No static defect.** All workspaces are sized correctly for χ=128/256
across all failing models. Round-20 conformity review confirmed 0 defects.
The CR-D1 fix from round-8 is in place. The hang is **runtime behavioral**.

## Leading hypothesis (50% confidence): host-sync + Step-3 GEMM dispatch storm

### Pattern

In `gpu-rocm/dmrg2-gpu-opt/src/dmrg2_gpu_opt_impl.h:794-810`, the d>2 path
in `apply_heff_two_site` Step 3 uses a nested host loop:

```cpp
for (int s2p = 0; s2p < d; s2p++)
    for (int s1p = 0; s1p < d; s1p++)
        for (int n = 0; n < D; n++)
            rocblas_dgemm(...);  // single host-dispatched GEMM call
```

For josephson d=5: `d*d*D = 100 GEMMs per matvec`. For Davidson with
~120 matvecs per bond at χ=128: **12,000 GEMM API calls per bond**.

`block_davidson_eigensolver` (lines 1416-1693) uses **host-pointer-mode
rocBLAS** for `nrm2`, `dot`, `scal` etc. Each host-mode scalar op is
synchronous — the host blocks until the result is written to a host scalar.
Mandatory `hipStreamSynchronize` calls at lines 1473, 1515, 1527.

Result pattern:
1. CPU dispatches 100 small GEMMs (microseconds).
2. GPU runs them (microseconds).
3. CPU stalls at `hipStreamSynchronize` (10-50 µs of kernel-launch overhead).
4. GPU is **idle** waiting for next batch.

At χ=64 (local dim 16,384) the GEMMs are tiny enough that the dispatch
overhead dominates but the absolute time stays small — completes in <30 s.
At χ=128 (local dim 65,536+) the absolute time per bond explodes, and the
GPU spends ~95% of its time idle between dispatch waves. The watchdog
samples during one of these idle windows → 0% GPU → KILL at 120 s.

### Why working siblings dodge it

| Variant | Step 3 structure | Davidson scalar mode |
|---|---|---|
| `dmrg-gpu-opt` | single dense GEMM | host-mode (but only 1 GEMM per matvec) |
| `dmrg2-gpu` (default) | same 100-call loop | uses **Lanczos**, device-mode α/β |
| `dmrg2-gpu-base` | same loop, lower-tier | uses **Lanczos** |
| `pdmrg-gpu-opt` | per-segment, smaller theta | segment-parallel reduces per-bond work |
| **`dmrg2-gpu-opt`** | 100-call loop | **Davidson + host-mode** = pathological |

`dmrg2-gpu-opt` is the **only variant** that combines the Step-3 host
loop with Davidson's host-mode sync pattern.

## Secondary hypotheses

### Candidate 2 (30%): rocsolver_gesvdj Jacobi SVD slowness on ROCm 7.2

`rocsolver_gesvd_auto` in `gpu-rocm/common/scalar_traits.h:178-188` calls
the Jacobi variant `rocsolver_gesvdj`. For josephson d=5 χ=128 theta is
640×640 complex. If ROCm 7.2 has a Jacobi convergence issue for specific
dimensions, the "30 s of GPU work then 0%" pattern matches: SVD kernel
launches, runs for 30 s, gets stuck or converges extremely slowly.

**Quick test**: enable RSVD via `set_rsvd(true)` — bypasses Jacobi SVD.
If hang vanishes → confirmed.

### Candidate 3 (15%): hipFree-induced device sync in svd_split_fallback

`allocate_mps_tensor` calls `hipFree(d_mps_tensors_[site+1])` from the host
thread inside `svd_split_fallback`. Newer ROCm versions may force a device
sync on `hipFree`. If this lands while `stream_env_`'s `event_env_done_` is
pending, ordering anomalies could ensue. Unlikely to cause a true deadlock
but could corrupt event timing.

## Reproducer plan

When the GPU is free (post-campaign, before catch-up):

```bash
# Real heisenberg (simpler, d=2 — smoke killed it at 50 128 20)
HIP_LAUNCH_BLOCKING=1 AMD_LOG_LEVEL=3 \
    gpu-rocm/dmrg2-gpu-opt/build/dmrg2_gpu_opt 50 128 3 2>&1 | tee /tmp/d1-repro.log

# Complex josephson (d=5 — most pathological)
HIP_LAUNCH_BLOCKING=1 AMD_LOG_LEVEL=3 \
    gpu-rocm/dmrg2-gpu-opt/build/dmrg2_gpu_opt 16 128 3 --josephson --nmax 2 2>&1 | tee /tmp/d1-jos.log
```

The **last API call before hang** identifies the culprit:
- `rocsolver_*` last → Candidate 2 (SVD).
- `hipStreamSynchronize` after rocblas_gemm cluster → Candidate 1.
- `hipFree` → Candidate 3.

Watch GPU utilization in parallel:
```bash
while kill -0 <PID> 2>/dev/null; do
    rocm-smi --showuse | grep "GPU\[0\]"; sleep 5
done
```

## Fix proposal (Candidate 1)

Port `apply_heff_two_site` Step 3 d>2 path to `gemm_strided_batched`.
The d≤2 branch already uses this pattern at lines 777-793.

**File**: `gpu-rocm/dmrg2-gpu-opt/src/dmrg2_gpu_opt_impl.h`
**Lines to replace**: 794-810

```cpp
} else {
    // Generalized strided-batched Step 3.
    // Reduces dispatch from D*d*d = 100 calls to D*d = 20 calls for d=5.
    for (int n = 0; n < D; n++) {
        for (int s1p = 0; s1p < d; s1p++) {
            Scalar beta = (n == 0) ? Traits::zero() : Traits::one();
            ROCBLAS_CHECK(Traits::gemm_strided_batched(
                rocblas_h_,
                rocblas_operation_none, rocblas_operation_none,
                cL, cR, cR, &one,
                T2 + (size_t)(n * dd + s1p * d) * cL * cR, cL,
                    (rocblas_stride)(cL * cR),
                R_env + n * cR, cR * D, (rocblas_stride)0,
                &beta,
                d_result + s1p * cL, cL * dd, (rocblas_stride)cL,
                d));
        }
    }
}
```

This reduces dispatch from 12,000/bond to **2,400/bond** at χ=128 d=5.
Combined with the existing d≤2 strided-batched path, the GPU dispatch
pressure should be sane across all challenge sizes.

### Sibling propagation (technique G)

The same nested-loop pattern exists in:
- `gpu-rocm/dmrg2-gpu/src/dmrg2_gpu_impl.h` Step 3 (~lines 750-810).
- Possibly `gpu-rocm/dmrg2-gpu-base/src/dmrg2_gpu_base_impl.h` Step 3.

Even though these don't hang (they use Lanczos), the inefficiency is
identical. Land the fix across all 3 dmrg2 variants atomically in one PR.

## Action checklist for the catch-up run

1. Run the Reproducer Plan repros above with the campaign idle.
2. Confirm Candidate 1 vs 2 vs 3 from the log.
3. If Candidate 1: land the Step-3 strided-batched port in all 3 dmrg2
   variants. Re-smoke. Should clear χ=128 in seconds, not hang.
4. If Candidate 2: try `set_rsvd(true)` as a workaround. File a bug
   against rocm-7.2 if Jacobi SVD hangs on 640×640 complex.
5. If Candidate 3: change `allocate_mps_tensor` to lazy-allocate or
   pre-allocate at max-bond-dim once at ctor time. No mid-sweep `hipFree`.
6. Run **smoke** for the fixed variant. If clean, run **partial --full**
   (`VARIANT_ONLY=dmrg2-gpu-opt`) to fill in the missing data column for
   the paper.
7. Update `reviews/g1-deferred-bugs-20260509.md` with the resolution.

## Files

- `gpu-rocm/dmrg2-gpu-opt/src/dmrg2_gpu_opt_impl.h` (lines 794-810 = fix
  site, 1416-1693 = Davidson context).
- `gpu-rocm/common/batch_ptrs_kernels.h:181-192` = existing setup_batch_ptrs
  helper.
- `gpu-rocm/dmrg2-gpu/src/dmrg2_gpu_impl.h` = sibling for propagation.
- `gpu-rocm/dmrg2-gpu-base/src/dmrg2_gpu_base_impl.h` = sibling for propagation.
- `reviews/g1-deferred-bugs-20260509.md` = failure manifest.
- `benchmarks/run_mi300x_challenge.py:132` = `JOSEPHSON_NMAX=2` → d=5
  (critical config).
