# Batched GEMM + HIP Graph Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce GPU kernel launch overhead by converting Step 3 individual GEMMs to batched calls and capturing apply_heff as a HIP Graph for replay in the Lanczos loop.

**Architecture:** Step 1 already uses batched GEMM; Step 2 is a single dense GEMM. The optimization converts Step 3 from d*D individual GEMM calls on worker streams to D batched GEMM calls (batch_count=d), then captures the entire apply_heff (3 API calls: batched + dense + D batched) as a HIP Graph. The graph is replayed 10-50 times per site in the Lanczos eigensolver, eliminating per-launch dispatch overhead. A fixed staging buffer for theta makes pointer arrays constant across replays.

**Tech Stack:** HIP (hipGraph API), rocBLAS (gemm_batched), C++17 templates

**Accuracy requirement:** 1e-10 or better on all test cases. Abort immediately if accuracy degrades.

---

## File Structure

All changes are modifications to existing files:

| File | Changes |
|------|---------|
| `dmrg-gpu/src/dmrg_gpu.h` | Add graph members, staging buffer, remove worker streams |
| `dmrg-gpu/src/dmrg_gpu_impl.h` | Batched Step 3, graph capture/replay in Lanczos |
| `dmrg2-gpu/src/dmrg2_gpu.h` | Same pattern as dmrg-gpu |
| `dmrg2-gpu/src/dmrg2_gpu_impl.h` | Same pattern, d→d² for two-site |
| `pdmrg-gpu/src/pdmrg_gpu.h` | Per-segment graph members |
| `pdmrg-gpu/src/pdmrg_gpu_impl.h` | Same pattern, per-segment graphs |

---

## Task 1: dmrg-gpu — Convert apply_heff Step 3 to batched GEMM

**Files:**
- Modify: `dmrg-gpu/src/dmrg_gpu.h` — add staging buffer, step3 pointer arrays
- Modify: `dmrg-gpu/src/dmrg_gpu_impl.h` — rewrite apply_heff Step 3

### Current Step 3 (lines 345-365 of dmrg_gpu_impl.h):
```cpp
// d worker streams, each doing D sequential GEMMs (d*D individual launches)
for (int sp = 0; sp < d; sp++) {
    int wi = sp % n_workers_;
    HIP_CHECK(hipStreamWaitEvent(worker_streams_[wi], step_done_event_, 0));
    for (int wp = 0; wp < D; wp++) {
        Scalar beta = (wp == 0) ? zero : one;
        ROCBLAS_CHECK(Traits::gemm(worker_handles_[wi], ...));
    }
    HIP_CHECK(hipEventRecord(worker_done_events_[wi], ...));
}
```

### New Step 3 (D batched GEMM calls, batch_count=d each):
```cpp
// D batched calls, each batching d independent sp values
for (int wp = 0; wp < D; wp++) {
    Scalar beta = (wp == 0) ? Traits::zero() : Traits::one();
    // Set up pointer arrays for this wp batch
    Scalar* h_A3[d_max], *h_B3[d_max], *h_C3[d_max];  // d_max = max possible d
    for (int sp = 0; sp < d; sp++) {
        int ws_out = wp * d + sp;
        h_A3[sp] = U + ws_out * cL * cR;
        h_B3[sp] = R_env + wp * cR;   // same B for all sp in this batch
        h_C3[sp] = d_result + sp * cL;
    }
    HIP_CHECK(hipMemcpyAsync(d_batch_A_, h_A3, d * sizeof(Scalar*), H2D, stream_));
    HIP_CHECK(hipMemcpyAsync(d_batch_B_, h_B3, d * sizeof(Scalar*), H2D, stream_));
    HIP_CHECK(hipMemcpyAsync(d_batch_C_, h_C3, d * sizeof(Scalar*), H2D, stream_));
    ROCBLAS_CHECK(Traits::gemm_batched(rocblas_h_,
        rocblas_operation_none, rocblas_operation_none,
        cL, cR, cR, &one,
        (const Scalar**)d_batch_A_, cL,
        (const Scalar**)d_batch_B_, cR * D,
        &beta, d_batch_C_, cL * d,
        d));
}
```

- [ ] **Step 1:** In `dmrg_gpu.h`, add `Scalar* d_theta_staging_;` member and a second set of batch pointer arrays `Scalar** d_batch_A3_, d_batch_B3_, d_batch_C3_;` (so Step 1 and Step 3 can have independent pointer arrays for graph capture). Add pinned host pointer arrays `Scalar** h_batch_A_pin_, h_batch_B_pin_, h_batch_C_pin_;` and `Scalar** h_batch_A3_pin_, h_batch_B3_pin_, h_batch_C3_pin_;` for graph-safe host memory.

- [ ] **Step 2:** In constructor, allocate the new buffers:
  ```cpp
  HIP_CHECK(hipMalloc(&d_theta_staging_, theta_size_max_ * sizeof(Scalar)));
  int batch3_max = std::max(d_, d_ * d_);  // future-proof for two-site
  HIP_CHECK(hipMalloc(&d_batch_A3_, batch3_max * sizeof(Scalar*)));
  HIP_CHECK(hipMalloc(&d_batch_B3_, batch3_max * sizeof(Scalar*)));
  HIP_CHECK(hipMalloc(&d_batch_C3_, batch3_max * sizeof(Scalar*)));
  // Pinned host arrays for graph-safe pointer upload
  int batch1_max = D_mpo_ * d_;
  HIP_CHECK(hipHostMalloc(&h_batch_A_pin_, batch1_max * sizeof(Scalar*)));
  HIP_CHECK(hipHostMalloc(&h_batch_B_pin_, batch1_max * sizeof(Scalar*)));
  HIP_CHECK(hipHostMalloc(&h_batch_C_pin_, batch1_max * sizeof(Scalar*)));
  HIP_CHECK(hipHostMalloc(&h_batch_A3_pin_, batch3_max * sizeof(Scalar*)));
  HIP_CHECK(hipHostMalloc(&h_batch_B3_pin_, batch3_max * sizeof(Scalar*)));
  HIP_CHECK(hipHostMalloc(&h_batch_C3_pin_, batch3_max * sizeof(Scalar*)));
  ```

- [ ] **Step 3:** In destructor, free the new buffers.

- [ ] **Step 4:** Rewrite apply_heff Step 1 to use pinned host arrays instead of stack VLAs:
  ```cpp
  // Step 1: use pinned host arrays (safe for graph capture)
  for (int w = 0; w < D; w++)
      for (int s = 0; s < d; s++) {
          int ws = w * d + s;
          h_batch_A_pin_[ws] = L_env + w * cL;
          h_batch_B_pin_[ws] = const_cast<Scalar*>(d_theta_in) + s * cL;
          h_batch_C_pin_[ws] = V + ws * cL * cR;
      }
  HIP_CHECK(hipMemcpyAsync(d_batch_A_, h_batch_A_pin_, D*d*sizeof(Scalar*), H2D, stream_));
  HIP_CHECK(hipMemcpyAsync(d_batch_B_, h_batch_B_pin_, D*d*sizeof(Scalar*), H2D, stream_));
  HIP_CHECK(hipMemcpyAsync(d_batch_C_, h_batch_C_pin_, D*d*sizeof(Scalar*), H2D, stream_));
  // ... same batched GEMM call
  ```

- [ ] **Step 5:** Rewrite apply_heff Step 3 to use batched GEMM with pinned arrays:
  ```cpp
  // Step 3: D batched calls, batch_count=d each
  for (int wp = 0; wp < D; wp++) {
      Scalar beta = (wp == 0) ? Traits::zero() : Traits::one();
      for (int sp = 0; sp < d; sp++) {
          int ws_out = wp * d + sp;
          h_batch_A3_pin_[sp] = U + ws_out * cL * cR;
          h_batch_B3_pin_[sp] = R_env + wp * cR;
          h_batch_C3_pin_[sp] = d_result + sp * cL;
      }
      HIP_CHECK(hipMemcpyAsync(d_batch_A3_, h_batch_A3_pin_, d*sizeof(Scalar*), H2D, stream_));
      HIP_CHECK(hipMemcpyAsync(d_batch_B3_, h_batch_B3_pin_, d*sizeof(Scalar*), H2D, stream_));
      HIP_CHECK(hipMemcpyAsync(d_batch_C3_, h_batch_C3_pin_, d*sizeof(Scalar*), H2D, stream_));
      ROCBLAS_CHECK(Traits::gemm_batched(rocblas_h_,
          rocblas_operation_none, rocblas_operation_none,
          cL, cR, cR, &one,
          (const Scalar**)d_batch_A3_, cL,
          (const Scalar**)d_batch_B3_, cR * D,
          &beta, d_batch_C3_, cL * d,
          d));
  }
  ```
  Remove worker stream event record/wait for apply_heff (keep workers for env updates).

- [ ] **Step 6:** Build and test on MI300X. Verify 1e-10 accuracy on Heisenberg L=4,8 and Josephson L=4,6.
  ```bash
  cd ~/dmrg-implementations/dmrg-gpu && mkdir -p build && cd build
  cmake .. && make -j$(nproc)
  ./test_dmrg_gpu
  ```

- [ ] **Step 7:** Commit and push.

---

## Task 2: dmrg-gpu — HIP Graph capture of apply_heff in Lanczos

**Files:**
- Modify: `dmrg-gpu/src/dmrg_gpu.h` — add graph members
- Modify: `dmrg-gpu/src/dmrg_gpu_impl.h` — graph capture/replay in lanczos_eigensolver

### Design:
- `d_theta_staging_`: fixed-address buffer. Before each apply_heff, copy theta to staging.
- `apply_heff` always reads from `d_theta_staging_`, making all pointer arrays constant.
- First Lanczos iteration: call apply_heff normally (no capture yet — need to verify dimensions work).
- Second iteration: begin capture, call apply_heff, end capture, instantiate graph.
- Iterations 2+: copy theta to staging, launch graph.
- Graph is invalidated when site changes (different cL/cR dimensions).

### Why capture on iteration 1, not 0:
Iteration 0 does the real apply_heff call. We capture on iteration 1 and replay from iteration 2+. This avoids issues with the first iteration being different (e.g., initial theta might be a corner case).

Actually, simpler: we can capture on iteration 0 itself. The key insight is that `apply_heff(site, d_theta_staging_, d_heff_result_)` uses the same pointers every time. We just need to do the staging copy before capture.

### Implementation:

- [ ] **Step 1:** In `dmrg_gpu.h`, add:
  ```cpp
  // HIP Graph for apply_heff replay in Lanczos
  hipGraph_t heff_graph_ = nullptr;
  hipGraphExec_t heff_graph_exec_ = nullptr;
  int heff_graph_site_ = -1;  // cached site (invalidate on site change)
  ```

- [ ] **Step 2:** In destructor, add cleanup:
  ```cpp
  if (heff_graph_exec_) hipGraphExecDestroy(heff_graph_exec_);
  if (heff_graph_) hipGraphDestroy(heff_graph_);
  ```

- [ ] **Step 3:** Add a helper method `setup_heff_graph(int site)` that:
  1. Invalidates old graph if site changed
  2. Sets up pinned pointer arrays for Step 1 (using `d_theta_staging_` as theta base) and Step 3
  3. Captures apply_heff into a graph:
  ```cpp
  template<typename Scalar>
  void DMRGGPU<Scalar>::setup_heff_graph(int site) {
      if (heff_graph_site_ == site) return;  // already cached
      // Destroy old graph
      if (heff_graph_exec_) { hipGraphExecDestroy(heff_graph_exec_); heff_graph_exec_ = nullptr; }
      if (heff_graph_) { hipGraphDestroy(heff_graph_); heff_graph_ = nullptr; }

      int cL = chi_L(site), cR = chi_R(site);
      int D = D_mpo_, d = d_;

      // Set up persistent pointer arrays using d_theta_staging_ as input
      // Step 1 pointers
      for (int w = 0; w < D; w++)
          for (int s = 0; s < d; s++) {
              int ws = w * d + s;
              h_batch_A_pin_[ws] = d_L_envs_[site] + w * cL;
              h_batch_B_pin_[ws] = d_theta_staging_ + s * cL;
              h_batch_C_pin_[ws] = d_T1_ + ws * cL * cR;
          }
      HIP_CHECK(hipMemcpyAsync(d_batch_A_, h_batch_A_pin_, D*d*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
      HIP_CHECK(hipMemcpyAsync(d_batch_B_, h_batch_B_pin_, D*d*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
      HIP_CHECK(hipMemcpyAsync(d_batch_C_, h_batch_C_pin_, D*d*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
      HIP_CHECK(hipStreamSynchronize(stream_));

      // Capture the graph
      HIP_CHECK(hipStreamBeginCapture(stream_, hipStreamCaptureModeGlobal));
      apply_heff(site, d_theta_staging_, d_heff_result_);
      HIP_CHECK(hipStreamEndCapture(stream_, &heff_graph_));
      HIP_CHECK(hipGraphInstantiate(&heff_graph_exec_, heff_graph_, nullptr, nullptr, 0));
      heff_graph_site_ = site;
  }
  ```

- [ ] **Step 4:** Modify `lanczos_eigensolver` to use graph replay:
  ```cpp
  // Before the loop:
  setup_heff_graph(site);

  // In the loop, replace:
  //   apply_heff(site, d_vi, d_heff_result_);
  // With:
  HIP_CHECK(hipMemcpyAsync(d_theta_staging_, d_vi, n * sizeof(Scalar),
                            hipMemcpyDeviceToDevice, stream_));
  HIP_CHECK(hipGraphLaunch(heff_graph_exec_, stream_));
  ```

- [ ] **Step 5:** Important: `apply_heff` must NOT do pointer uploads when called during graph capture. Since we upload pointers BEFORE capture (in setup_heff_graph), modify apply_heff to skip pointer uploads when pointer arrays are already set. The simplest approach: separate the pointer setup from the GEMM calls. Create `apply_heff_gemms_only(site)` that assumes pointers are already on device:
  ```cpp
  template<typename Scalar>
  void DMRGGPU<Scalar>::apply_heff_gemms_only(int site) {
      int cL = chi_L(site), cR = chi_R(site);
      int D = D_mpo_, d = d_;
      Scalar one = Traits::one(), zero_val = Traits::zero();
      Scalar* V = d_T1_;
      Scalar* U = d_T2_;
      Scalar* R_env = d_R_envs_[site + 1];

      // Step 1: batched GEMM (pointers already on device)
      ROCBLAS_CHECK(Traits::gemm_batched(rocblas_h_,
          Traits::op_t, rocblas_operation_none,
          cL, cR, cL, &one,
          (const Scalar**)d_batch_A_, cL * D,
          (const Scalar**)d_batch_B_, cL * d,
          &zero_val, d_batch_C_, cL,
          D * d));

      // Step 2: dense GEMM
      ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
          rocblas_operation_none, rocblas_operation_none,
          cL * cR, d * D, D * d, &one,
          V, cL * cR, d_W_left_[site], D * d,
          &zero_val, U, cL * cR));

      // Step 3: D batched calls
      for (int wp = 0; wp < D; wp++) {
          Scalar beta = (wp == 0) ? Traits::zero() : Traits::one();
          for (int sp = 0; sp < d; sp++) {
              int ws_out = wp * d + sp;
              h_batch_A3_pin_[sp] = U + ws_out * cL * cR;
              h_batch_B3_pin_[sp] = R_env + wp * cR;
              h_batch_C3_pin_[sp] = d_heff_result_ + sp * cL;
          }
          HIP_CHECK(hipMemcpyAsync(d_batch_A3_, h_batch_A3_pin_, d*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
          HIP_CHECK(hipMemcpyAsync(d_batch_B3_, h_batch_B3_pin_, d*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
          HIP_CHECK(hipMemcpyAsync(d_batch_C3_, h_batch_C3_pin_, d*sizeof(Scalar*), hipMemcpyHostToDevice, stream_));
          ROCBLAS_CHECK(Traits::gemm_batched(rocblas_h_,
              rocblas_operation_none, rocblas_operation_none,
              cL, cR, cR, &one,
              (const Scalar**)d_batch_A3_, cL,
              (const Scalar**)d_batch_B3_, cR * D,
              &beta, d_batch_C3_, cL * d,
              d));
      }
  }
  ```
  NOTE: Step 3 pointer uploads happen inside the capture. They use pinned host memory so they are safe for graph replay (pinned memory persists at the same address).

  Actually CRITICAL ISSUE: Step 3 pointer arrays change per wp iteration (different U offsets). But for graph capture, the pointer uploads from pinned host memory are captured with the specific values at capture time. On replay, the SAME pointer values are replayed, which is correct because U (d_T2_) is at a fixed address and the offsets are deterministic for a given site.

  HOWEVER: The `h_batch_A3_pin_` values are written by the CPU loop BEFORE each hipMemcpyAsync. During capture, the CPU loop runs, fills the pinned arrays, and the hipMemcpyAsync is captured. On replay, the hipMemcpyAsync replays from the same pinned host addresses — but the CPU loop does NOT re-run. The pinned arrays still hold the values from the LAST wp iteration of the capture!

  FIX: Pre-compute ALL Step 3 pointer arrays for all wp values BEFORE capture, store them in separate pinned buffers (one per wp), and upload all at once. OR: use D separate device pointer arrays, one per wp batch.

  SIMPLER FIX: Pre-upload ALL Step 3 pointer arrays before capture. Use D sets of device pointer arrays, allocated as one contiguous block. Each wp batch uses its own slice.

- [ ] **Step 5 (revised):** Allocate Step 3 pointer arrays as `D * d` entries (one set per wp value). Before capture, upload all D*d pointer entries. During capture, the batched GEMM calls reference fixed offsets into the pre-uploaded arrays.

  In header, replace `d_batch_A3_/B3_/C3_` with arrays sized `D_mpo * d`:
  ```cpp
  // Step 3 pointer arrays: D batches, d pointers each
  // Layout: [wp=0: d ptrs][wp=1: d ptrs]...[wp=D-1: d ptrs]
  Scalar** d_step3_A_;  // D*d entries on device
  Scalar** d_step3_B_;
  Scalar** d_step3_C_;
  ```

  In setup_heff_graph, before capture:
  ```cpp
  // Pre-compute ALL Step 3 pointers
  for (int wp = 0; wp < D; wp++)
      for (int sp = 0; sp < d; sp++) {
          int idx = wp * d + sp;
          h_batch_A3_pin_[idx] = d_T2_ + (wp * d + sp) * cL * cR;
          h_batch_B3_pin_[idx] = d_R_envs_[site + 1] + wp * cR;
          h_batch_C3_pin_[idx] = d_heff_result_ + sp * cL;
      }
  HIP_CHECK(hipMemcpyAsync(d_step3_A_, h_batch_A3_pin_, D*d*sizeof(Scalar*), H2D, stream_));
  HIP_CHECK(hipMemcpyAsync(d_step3_B_, h_batch_B3_pin_, D*d*sizeof(Scalar*), H2D, stream_));
  HIP_CHECK(hipMemcpyAsync(d_step3_C_, h_batch_C3_pin_, D*d*sizeof(Scalar*), H2D, stream_));
  HIP_CHECK(hipStreamSynchronize(stream_));
  ```

  In apply_heff_gemms_only, Step 3 becomes:
  ```cpp
  for (int wp = 0; wp < D; wp++) {
      Scalar beta = (wp == 0) ? Traits::zero() : Traits::one();
      ROCBLAS_CHECK(Traits::gemm_batched(rocblas_h_,
          rocblas_operation_none, rocblas_operation_none,
          cL, cR, cR, &one,
          (const Scalar**)(d_step3_A_ + wp * d), cL,
          (const Scalar**)(d_step3_B_ + wp * d), cR * D,
          &beta, d_step3_C_ + wp * d, cL * d,
          d));
  }
  ```
  No pointer uploads during capture — all pointers pre-uploaded and fixed.

- [ ] **Step 6:** Build, test accuracy (1e-10 on all test cases), benchmark.

- [ ] **Step 7:** Commit and push.

---

## Task 3: dmrg-gpu — Convert env update Step 3 to batched GEMM

**Files:**
- Modify: `dmrg-gpu/src/dmrg_gpu_impl.h` — update_left_env, update_right_env Step 3

### update_left_env Step 3:
- wp (D values) independent, sp (d values) accumulate
- D batched calls impossible (sp accumulates within each wp group)
- BUT: for a fixed sp, all wp values are independent → batch wp
- Restructure: d sequential batched calls, batch_count=D each

```cpp
// Step 3: d batched calls (batch_count=D), sp accumulates
for (int sp = 0; sp < d; sp++) {
    Scalar beta = (sp == 0) ? Traits::zero() : Traits::one();
    for (int wp = 0; wp < D; wp++) {
        int ws_out = wp * d + sp;
        h_batch_A3_pin_[wp] = U + ws_out * chi_in * chi_out;
        h_batch_B3_pin_[wp] = A + sp * chi_in;  // same B for all wp
        h_batch_C3_pin_[wp] = L_new + wp * chi_out;
    }
    // upload and batched GEMM call with batch_count=D
}
```

Wait — the B pointer is the SAME for all wp in the batch, and the C pointers are different but we're accumulating (beta=1). This is fine: each GEMM in the batch writes to a different C location (L_new + wp*chi_out), and they all use the same B (A + sp*chi_in). The beta=1 accumulates across sp iterations (sequential loop), not across batch elements.

### update_right_env Step 3:
- w (D values) independent, sp (d values) accumulate
- Same restructure: d sequential batched calls, batch_count=D

- [ ] **Step 1:** Rewrite update_left_env Step 3 to use batched GEMM.
- [ ] **Step 2:** Rewrite update_right_env Step 3 to use batched GEMM.
- [ ] **Step 3:** Build, test accuracy, benchmark.
- [ ] **Step 4:** Commit and push.

---

## Task 4: dmrg-gpu — Benchmark and evaluate

**Files:** None (testing only)

- [ ] **Step 1:** Run accuracy tests:
  ```bash
  ./test_dmrg_gpu  # Heisenberg L=4,8; Josephson L=4,6
  ```
  All errors must be < 1e-10.

- [ ] **Step 2:** Run performance benchmarks:
  ```bash
  # Compare before/after for representative configs
  ./test_dmrg_gpu --benchmark  # or use the benchmark binary
  ```
  Test: L=8 chi=32, L=16 chi=64, L=32 chi=64, L=32 chi=128

- [ ] **Step 3:** Commit benchmark results.

---

## Task 5: dmrg2-gpu — Apply same optimization

**Files:**
- Modify: `dmrg2-gpu/src/dmrg2_gpu.h`
- Modify: `dmrg2-gpu/src/dmrg2_gpu_impl.h`

Same pattern as dmrg-gpu, with these differences:
- apply_heff_two_site Step 3: batch_count = d² (not d), D accumulating calls
- Step 1 batch count: D*d² (not D*d)
- Pointer array sizes: D*d² for Step 1, D*d² for Step 3
- update_left_env and update_right_env: identical to dmrg-gpu (single-site tensors)
- Graph capture in lanczos_eigensolver: same pattern, different theta size

- [ ] **Step 1:** Add members to header (staging, pointer arrays, graph).
- [ ] **Step 2:** Constructor/destructor allocations/cleanup.
- [ ] **Step 3:** Convert apply_heff_two_site Step 3 to batched.
- [ ] **Step 4:** Add apply_heff_gemms_only equivalent, setup_heff_graph, graph replay in Lanczos.
- [ ] **Step 5:** Convert env update Step 3 to batched.
- [ ] **Step 6:** Build, test accuracy (1e-10), benchmark.
- [ ] **Step 7:** Commit and push.

---

## Task 6: pdmrg-gpu — Apply same optimization (per-segment)

**Files:**
- Modify: `pdmrg-gpu/src/pdmrg_gpu.h`
- Modify: `pdmrg-gpu/src/pdmrg_gpu_impl.h`

Key differences from dmrg-gpu:
- Per-segment streams: graph capture per segment stream
- Per-segment workspaces (d_T1, d_T2 are per-workspace)
- Per-segment pointer arrays and graphs: `std::vector<hipGraph_t>`, `std::vector<hipGraphExec_t>`
- apply_heff_two_site has segment index `si` parameter
- pdmrg-gpu already uses GPU kernels for pointer setup — replace with pinned host approach for consistency
- update_left_env and update_right_env also take segment index

- [ ] **Step 1:** Add per-segment members to header.
- [ ] **Step 2:** Constructor/destructor.
- [ ] **Step 3:** Convert apply_heff_two_site Step 3 to batched (per-segment stream).
- [ ] **Step 4:** Graph capture per segment in Lanczos.
- [ ] **Step 5:** Convert env update Step 3 to batched.
- [ ] **Step 6:** Build, test accuracy (1e-10), benchmark.
- [ ] **Step 7:** Commit and push.

---

## Task 7: Final benchmarking and evaluation

- [ ] **Step 1:** Run full benchmark suite on MI300X:
  ```
  L=8 chi=32, L=16 chi=64, L=32 chi=64, L=32 chi=128
  ```
  for all three implementations: dmrg-gpu, dmrg2-gpu, pdmrg-gpu.

- [ ] **Step 2:** Compare with pre-optimization baselines from prior session:
  | Config | dmrg-gpu (before) | dmrg2-gpu (before) | pdmrg-gpu (before) |
  |--------|-------------------|--------------------|--------------------|
  | L=8 chi=32 | 0.61s | 0.64s | 0.84s |
  | L=16 chi=64 | 0.95s | 1.50s | 2.6-3.1s |
  | L=32 chi=64 | 4.14s | 3.97s | 11-13s |
  | L=32 chi=128 | 4.63s | 6.35s | 26-32s |

- [ ] **Step 3:** Evaluate whether Approach B (custom fused kernels) is needed based on the speedup achieved.

- [ ] **Step 4:** Update paper_prompt.md with benchmark results.
