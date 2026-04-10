# Follow-up Proposal 3 — HIP Graph Capture of `apply_heff_two_site`

**Status:** Conditionally approved (REVISE) — sole survivor of the 2026-04-10 generate-and-critique loop over `PROJECT_OVERVIEW.md`.
**Parent loop:** 10 proposals generated, 9 rejected on grounds of prior-attempt collision, arithmetic, dependency violations, or known landmines. Raw transcripts in the conversation logs.
**Date opened:** 2026-04-10

---

## 1. Motivation

At `chi ≤ 50`, MI300X loses to single-threaded CPU in **93 %** of benchmark configurations (see `PROJECT_OVERVIEW.md §5.2` and `insight_single_core_cpu_beats_gpu.md`). The documented cause is kernel-launch dispatch overhead: each Lanczos iteration of `apply_heff_two_site` dispatches ~8–12 separate rocBLAS / custom kernels, and the launch latency is comparable to (or larger than) the actual FP64 work for these tiny matrices.

The preconditions for HIP graph capture are already in place:

| Precondition | Status | Source |
|---|---|---|
| Lanczos inner loop is sync-free | Shipped (A1) | `PROJECT_OVERVIEW.md §4.1 A1` |
| Pointer arrays live on device | Shipped (A3) | `PROJECT_OVERVIEW.md §4.1 A3` |
| No host-pointer stalls during `apply_heff` | Shipped (A7) | `PROJECT_OVERVIEW.md §4.1 A7` |
| HIP graphs attempted previously | **No** | grep of overview returns zero hits |

This is genuinely unexplored territory in the codebase.

## 2. What the adversarial review forced us to change

The original Proposal 3 was rejected by Reviewer 3 in its raw form. The surviving shape incorporates five specific revisions:

1. **Narrow scope to `apply_heff_two_site` only, NOT the Lanczos outer loop.**
   The outer loop's orthogonalization step is *not* topology-stable: Lanczos iteration `j` needs `j` dot-products against `v_0..v_{j-1}`, so the BLAS-1 dependency graph grows with `j`. Attempting to capture the outer loop as a single graph is the mistake the original author made.

2. **Feasibility-gate on rocBLAS stream capture in ROCm 7.2 BEFORE integration.**
   `rocblas_dgemm_batched` historically used host-resident pointer arrays that rocBLAS touched at launch-resolve time, causing `hipErrorStreamCaptureUnsupported`. The project's `apply_heff_two_site` Step 1 / Step 3 paths use exactly this call. A 10-line standalone microbenchmark must confirm capture is supported before any integration work.

3. **Drop the "flip 93 % of CPU wins" claim.**
   The CPU advantage at `chi ≤ 50` is cache residency across Lanczos iterations, NOT merely launch overhead. A 4 GHz core with a 64 KB L1 hits ~16 GF/s on hot cache — ~60 ns/matvec for a chi=32 problem. HIP graphs cannot beat that. Reframe the target as **"close the gap at chi ∈ [64, 128] where GPU is already competitive but dispatch is a measurable fraction of sweep time."**

4. **Honestly amortize capture cost.**
   At L=32 n_segments=4, bond shapes are `(1,2), (2,4), (4,8), (8,16), (16,32), (32,16), (16,8), (8,4), (4,2), (2,1)` — up to 10 distinct shapes per segment × 4 segments × 2 sweep directions = up to 80 graphs. First-sweep capture cost at ~500 µs per graph = ~40 ms. This only amortizes over ≥3 sweeps; for single-sweep operations (polish phase), capture cost likely exceeds savings.

5. **Handle graph cache eviction.**
   Per-shape graph cache in `StreamWorkspace` with an LRU policy. Shapes in pdmrg-gpu multiply by `n_segments`; bound the cache at a fixed count and accept re-capture for unused shapes.

## 3. Scoped design

### 3.1 What gets captured

Inside `apply_heff_two_site`, the fixed topology is the three-step contraction:

```
Step 1: L_env · theta                 → T1         (batched D×d² GEMMs)
Step 2: T1 · WW                        → T2         (dense GEMM)
Step 3: T2 · R_env                     → result     (batched d²×D GEMMs)
```

Plus the existing GPU-side pointer-setup kernels from A3 (already on device, capture-safe per the A3 design).

**NOT captured**: the Lanczos BLAS-1 orthogonalization chain, the CPU tridiagonal eigensolve via `rocsolver_dsteqr` or LAPACK `dstev`, the SVD path. These remain as separate stream calls that the outer loop orchestrates.

### 3.2 Capture cache

```
struct ApplyHeffGraphCache {
    struct Key { int chi_L, chi_R, d, D_mpo; };
    std::unordered_map<Key, hipGraphExec_t, KeyHash> cache;
    size_t max_entries = 32;  // LRU eviction above this
};
```

One cache per `StreamWorkspace` (i.e. per segment in PDMRG). Graphs are created lazily on first use.

### 3.3 Invocation

```
// Inside apply_heff_two_site:
Key k{chi_L, chi_R, d, D_mpo};
hipGraphExec_t exec = cache.lookup_or_create(k, [&]() {
    hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal);
    // Issue the Step 1, Step 2, Step 3 calls as today
    hipStreamEndCapture(stream, &graph);
    hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    return exec;
});
hipGraphLaunch(exec, stream);
```

Shape invariance: the buffer pointers (`d_theta`, `d_T1`, `d_T2`, `d_L_envs_[...]`, `d_R_envs_[...]`, `d_WW`) are all **fixed addresses** inside the `StreamWorkspace`. Data changes between calls, but addresses don't. No `hipGraphExecKernelNodeSetParams` bookkeeping is needed after instantiation.

## 4. Risks (calibrated)

| # | Risk | Severity | Mitigation |
|---|------|----------|------------|
| 1 | `rocblas_dgemm_batched` not capture-safe in ROCm 7.2 | **HIGH (go/no-go)** | 10-line microbench before any integration work |
| 2 | Shape explosion for PDMRG (80 graphs × n_segments) | Medium | LRU cap, lazy capture, shape normalization via MFMA-16 padding |
| 3 | First-sweep capture cost > savings | Medium | Only capture shapes reused ≥ 3 times; pre-capture "hot shapes" at warmup |
| 4 | CPU cache residency still wins at chi ≤ 32 | Acknowledged | Do not target this regime; honest framing of the win envelope |
| 5 | Interaction with per-segment worker threads in pdmrg-gpu-opt | Low | Graph cache is per-`StreamWorkspace`, no shared state |

## 5. Measurement plan

### Phase 0 — Go/no-go microbench (1 day, MUST complete first)

**UPDATED 2026-04-10 per Research Report A:** AMD's rocBLAS Beta Features doc
explicitly lists `rocblas_dgemm_batched` under stream capture as **unsupported**
on ROCm 7.2, and GitHub issue ROCm/rocBLAS#1240 is still open. The microbench
therefore tests **`rocblas_dgemm_strided_batched`**, which is NOT listed as
unsafe and has empirical reports of working (llama.cpp #14576).

Write `bench_graph_capture.cpp` in `gpu-rocm/sandbox/` that:
1. Calls `rocblas_initialize()` at startup.
2. Sets `rocblas_pointer_mode_device`.
3. Pre-allocates a workspace via `rocblas_set_workspace` BEFORE entering capture mode.
4. Creates a stream, begins capture, issues
   `rocblas_dgemm_strided_batched` + `rocblas_dgemm` + `rocblas_dgemm_strided_batched`
   on dummy pointers, ends capture.
5. Instantiates and launches the graph.
6. Reports success or the exact `hipError_t` returned.

**GO** if all three calls capture cleanly on ROCm 7.2 using the **strided** batched
variant. This requires a one-time refactor of `apply_heff_two_site` Step 1 / Step 3
from the `_batched` to the `_strided_batched` variant (the strides are fixed at MPS
allocation, so this is mechanical).

**NO-GO** if even the strided variant fails. In that case we pivot to
**Proposal 3-alt: custom fused HIP kernel via rocWMMA FP64 MFMA-16** — a single
kernel that does Step 1 + Step 2 + Step 3 in one launch, eliminating the rocBLAS
capture dependency entirely. See `followups/round_2_plan.md` §4.

### Phase 1 — Microbench apply_heff

Standalone harness at `chi ∈ {32, 64, 96, 128}`, models Heisenberg (`d=2, D=5`) and Josephson (`d=3, D=4`). Compare captured vs uncaptured matvec throughput. Expected: 2–4 × dispatch reduction at `chi = 64`, ~1 × at `chi ≥ 256` (launches become negligible).

### Phase 2 — Full sweep benchmark

Integrate into `dmrg2-gpu-opt` only (NOT pdmrg-gpu-opt yet — keep blast radius small). Run the existing `benchmarks/run_focused_bench.py` grid on:
- Heisenberg `L ∈ {16, 32, 64}, chi ∈ {32, 64, 128}`
- Josephson `L ∈ {8, 16, 32}, chi ∈ {32, 64, 128}`

Target: no regression at any config, improvement at `chi ∈ [64, 128]`. Hard gate: energy delta < 1e-10 vs quimb-dmrg2 on all Heisenberg targets.

### Phase 3 — PDMRG integration (gated on Phase 2 success)

Port to `pdmrg-gpu-opt`, rerun the 44-config multi-GPU benchmark grid. Report per-shape graph cache hit rate and capture overhead fraction.

### Phase 4 — CUDA port

If MI300X wins materialize, replicate in `gpu-cuda/dmrg2-gpu-opt/` via `cudaGraph` (nearly identical API surface).

## 6. Success criteria

- **Minimum**: no regression anywhere, ≥ 1.1 × speedup on Heisenberg L=32 chi=64.
- **Target**: 1.3–1.5 × speedup on `dmrg2-gpu-opt` at `chi ∈ [64, 128]`.
- **Stretch**: close the GPU-vs-CPU gap on at least 5 of the 29 Heisenberg `chi ∈ [64, 128]` CPU-win configs in `benchmarks/paper_results/mi300x/wins_cpu_vs_gpu.csv`.

## 7. What we will NOT claim

- Wins at `chi ≤ 32` (CPU cache residency wall)
- Wins at `chi ≥ 256` (SVD dominates, see §5.1 of overview)
- Speedup on the polish phase of pdmrg (single-pass, capture cost dominates)
- Any effect on the CPU-SVD bottleneck itself

## 8. Dependencies on other follow-ups

None. This is the most independent follow-up in the round — it can land before Directions B and C.

## 9. Relevant code pointers

- `gpu-rocm/dmrg2-gpu/src/dmrg2_gpu_impl.h` — `apply_heff_two_site` (capture target)
- `gpu-rocm/dmrg-gpu/src/scalar_traits.h` — GPU-side pointer setup kernels (A3)
- `gpu-rocm/dmrg2-gpu-opt/src/dmrg2_gpu_opt_impl.h` — strided-batched Step-3 variant
- `benchmarks/paper_results/mi300x/wins_cpu_vs_gpu.csv` — target configs
- `PROJECT_OVERVIEW.md §4.1 A1/A3/A7` — preconditions
- `PROJECT_OVERVIEW.md §5.2` — honest framing of the CPU-win regime
