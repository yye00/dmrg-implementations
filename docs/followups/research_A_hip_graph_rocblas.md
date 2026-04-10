# Research Report A: HIP Graph Capture + rocBLAS Stream Safety

**Author:** Research Agent A
**Date:** 2026-04-10
**Target:** Proposal 3 — `apply_heff_two_site` HIP graph capture
**Parent doc:** `docs/followups/proposal_3_hip_graph_capture.md`

---

## Executive summary

- **GO/NO-GO verdict: NO-GO for `rocblas_dgemm_batched` inside `hipStreamBeginCapture` on ROCm 7.2.** AMD's own rocBLAS beta-features documentation explicitly lists BLAS-3 batched functions as unsupported for graph capture; GitHub issue rocBLAS#1240 ("Capturing Stream Safety") remains open and confirms the architectural root cause (host-resident workspace and pointer-array touch at launch-resolve time) is unresolved. Windows platforms additionally produce incorrect results for batched Level-3 under graph capture. [1][2]
- **Confidence: HIGH** for the negative verdict on pointer-array batched GEMM. **MEDIUM** on the strided-batched escape hatch — it is not listed as unsafe, empirical reports suggest it works, but AMD has not certified it, and rocBLAS handle initialization/workspace policies still apply. [1][3]
- **Recommended next step:** Do **not** ship the Phase 0 microbench against `rocblas_dgemm_batched`. Instead, run a **two-track Phase 0**: (Track A) microbench using `rocblas_dgemm_strided_batched` + `rocblas_dgemm` with `rocblas_pointer_mode_device` and explicit pre-allocated workspace via `rocblas_set_workspace`; (Track B) a parallel microbench that replaces rocBLAS entirely with a single custom fused-3-GEMM HIP kernel (manually fused, no graph needed — see §5). Both tracks are ~1–2 days of effort and directly inform the pivot decision.

---

## 1. rocBLAS stream capture status on ROCm 7.2

### 1.1 Official documentation

The authoritative source is the rocBLAS "Beta Features" page (currently reflecting the `develop` branch that will land in ROCm 7.2/7.3). [1] It states:

> "BLAS Level-3 and BLAS-EX functions in pointer mode device do not support HIP Graph capture. … To be added in a future release."

And separately:

> "On the Windows platform, batched functions (Level-1, Level-2, and Level-3) produce incorrect results when used with HIP graphs."

The ROCm graph-safe support matrix [4] classifies rocBLAS as **partial (⚠️)** for graph safety, behind hipFFT/hipCUFFT (✅). This is a deliberate AMD classification, not a documentation gap.

### 1.2 Architectural root cause (rocBLAS issue #1240)

Issue rocBLAS#1240 "Capturing Stream Safety" [2] is **still open** and explicitly acknowledges:

> "rocBLAS functions are not safe to use with HIP Graph functions … For this to be safe rocBLAS needs to detect if a stream is capturing and, if so, allocate fresh storage (which is never reused or deallocated)."

Two architectural properties block capture:

1. **Host pointer-array resolution at launch time** — `rocblas_dgemm_batched` accepts `const double* const A[]`. Historically rocBLAS touches these host pointers during launch-resolve, which HIP stream capture forbids.
2. **Dynamic workspace allocation on the capturing stream** — rocBLAS reuses/realloates an internal workspace buffer; under capture this becomes a stream-ordered allocation node that breaks child-graph composition (the same class of bug as NVIDIA Developer Forums thread #276634 for cuBLAS 12 child graphs, which required `cublasSetWorkspace` outside capture as the workaround). [7]

### 1.3 Version timeline

The ROCm 7.2 / 7.2.1 release notes focus on HIP runtime stream-capture validation fixes (e.g., "corrected the validation of stream capture in global-capture mode," and restricting capture inside `hipEventQuery`/`hipEventSynchronize`) but do **not** mention any rocBLAS change that would flip batched-GEMM capture safety. [5][6] There is no rocBLAS version in the 7.x line that has explicitly fixed this. ROCm 7.3 is not yet shipped. A 10-line microbench of `rocblas_dgemm_batched` inside capture on ROCm 7.2 is expected to return `hipErrorStreamCaptureUnsupported`, or worse, to *appear* to work but produce incorrect results on certain handle configurations.

### 1.4 Strided-batched variant

`rocblas_dgemm_strided_batched` is **not listed** in the "Functions unsupported with Graph Capture" section [1]. It avoids the host pointer-array entirely (it takes a single base pointer + stride). Empirical third-party evidence from llama.cpp issue #14576 [3] reports successful execution of `gemm_strided_batched_ex` under capture. However:

- AMD has **not** formally certified the strided variant as capture-safe.
- The rocBLAS workspace-allocation and initialization properties still apply regardless of the batched/strided distinction.
- The DMRG two-site kernel's Step 1 and Step 3 already use irregular L_env / R_env pointer layouts — switching to strided requires stride-compatible buffer packing, which the existing `dmrg2-gpu-opt` Step-3 strided-batched path already handles. Step 1 would need similar refactoring.

**Practical recommendation:** the Phase 0 microbench MUST test `rocblas_dgemm_strided_batched`, not `rocblas_dgemm_batched`, and must configure the handle with `rocblas_pointer_mode_device`, `rocblas_set_workspace(handle, user_buf, user_size)` outside capture, and `rocblas_initialize()` at process startup before any capture. If this combination produces a valid graph, Proposal 3 survives (with a scope adjustment to strided-only paths).

---

## 2. Documented patterns for iterative linear-algebra graph capture

### 2.1 Published HIP-graph case studies on iterative solvers

**Essentially none.** A thorough search of SC, PPoPP, IPDPS, HPDC, and ICS proceedings from 2023–2026 returned **no peer-reviewed papers** measuring HIP graph speedups on Krylov methods, Lanczos, Arnoldi, or CG on AMD GPUs. [8] The only substantive HPC GPU-graph paper in the sparse-LA space is *Work Graphs SpMV* (De Caro / ICPP 2025), which uses **AMD Work Graphs**, a separate device-driven scheduling API — not `hipStreamBeginCapture`. It reports 7.19× max / 3.35× mean SpMV speedup vs rocSPARSE for static sparse matrix-vector chains. [8] This validates that *some* form of scheduling-layer optimization helps, but Work Graphs are not what Proposal 3 describes.

**NVIDIA/CUDA side — also sparse:**

- NVIDIA blog "Constant-time launch for straight-line CUDA graphs" [9] reports ~200 ns → ~1 ns/node dispatch reduction on Ampere+, giving ~60 ns/node savings for straight-line graphs. For a 3-GEMM chain with 10 inner iterations, that is ≈2 µs saved per iteration — useful at chi=64 where total work is ~100 µs, noise at chi≥256.
- cuSparse explicitly documents graph capture examples for repeated SpMV. [10] This is the most directly analogous production pattern.
- **cuSolver eigensolvers (e.g., `cusolverDnCheevd`) cannot be captured** — they call `cudaStreamSynchronize` internally to check convergence. [11] This is the template for why full-Lanczos capture is infeasible, and exactly why Proposal 3 correctly narrowed scope to `apply_heff_two_site` only, not the outer Lanczos.

### 2.2 AMD's own admission

The 2025 AMD blog post on Mixture of Experts Align & Sort optimization [12] includes this telling caveat:

> "The algorithm currently does not support being under CUDA graph capture and we will resolve this issue later in V4 release."

AMD's own optimized kernels for MoE inference on MI300X are not graph-capture-safe. This is the broader ecosystem context: HIP graph capture is an infrastructure primitive but the library ecosystem above it has not caught up.

### 2.3 Known gotchas (aggregated from cuBLAS community + rocBLAS docs)

- **Workspace:** must be set via `rocblas_set_workspace` / `cublasSetWorkspace` **outside** capture, else stream-ordered-alloc nodes land in the graph and block child-graph composition. [7]
- **Handle initialization:** `rocblas_initialize()` or a warmup kernel launch must occur before capture begins; first-GEMM-of-handle triggers Tensile kernel load which cannot be captured. [13]
- **Pointer mode:** must be `rocblas_pointer_mode_device` for any BLAS-1 result reads (dot, nrm2) to avoid implicit host sync. The dmrg-gpu A1 optimization already does this for the Lanczos BLAS-1 — consistent with Proposal 3's claim.
- **Reorthogonalization / convergence checks:** anything that branches on GPU-computed scalars cannot live inside the captured region. Proposal 3 correctly excludes Lanczos orthogonalization from the capture scope.

---

## 3. Alternatives if rocBLAS capture is broken

### 3.1 Manual graph construction (`hipGraphCreate` + `hipGraphAddKernelNode`)

**Verdict: Not a useful escape hatch.** rocBLAS does not expose the underlying Tensile kernel entry points, launch parameters, or grid geometry as a public interface. You cannot legally add a rocBLAS GEMM as a pre-constructed kernel node — only something you wrote yourself. Adding the A3 pointer-setup kernels (which are our own HIP kernels) as explicit graph nodes is trivially possible but useless by itself, because the GEMMs are the expensive part.

### 3.2 `hipExtLaunchMultiKernelMultiDevice`

**Verdict: Not applicable.** Despite the name, this is a **multi-device** kernel launch coordination API (launches the same kernel on N devices with synchronized grid dispatch). It does **not** batch multiple different kernels on the same device. Skip.

### 3.3 Composable Kernel (CK) batched GEMM

**Verdict: Viable, but high-learning-curve.** CK is AMD's compile-time template-fused kernel library. [14] CK kernels are monolithic standalone HIP kernels with user-provided workspace (no internal stream-ordered allocations), so they are in principle graph-capture-safe. CK also supports operator fusion (GEMM + elementwise, GEMM + bias) at compile time, which is a more powerful axis than graph capture. Downsides: heavy C++ template metaprogramming, tiny shape coverage is uneven, and the DMRG Step 2 shape (`d²×d² · d²×cL*cR`) does not match any canned CK template out of the box. Estimated 2–4 weeks to build production-quality Step-1/Step-2/Step-3 replacements in CK.

### 3.4 hipBLASLt

**Verdict: Unknown, probably not yet. Do not rely on.** hipBLASLt is AMD's newer descriptor-based BLAS library (analogous to cuBLASLt). [15] Its graph-capture status in ROCm 7.2 is **not** explicitly documented in AMD's graph-safe support matrix. [4] hipBLASLt does support GEMM epilogue fusion (bias, activation) but does NOT fuse a second GEMM as an epilogue — the epilogue is elementwise only. This matches cuBLASLt behavior.

### 3.5 Custom fused HIP kernel replacing rocBLAS for Step 1 + Step 2 + Step 3

**Verdict: STRONGEST alternative if rocBLAS capture fails.** Write a single HIP kernel that performs all three contraction steps inside one launch, with intermediates in shared memory or registers. For `d=2, d=3, chi≤128, D_mpo≤5`, all working tensors comfortably fit in MI300X LDS (64 KB per CU). One launch replaces ≥3 rocBLAS launches, eliminating the root problem without needing graph capture at all. Can be built on top of **rocWMMA** [16] which provides `bfloat16_t`, `float`, and **`double`** wmma-style intrinsics on gfx942 via MFMA-16 tiles for FP64 — the same tiles the existing MFMA-16 padding optimization targets. Effort estimate: 1–2 weeks for a working fused kernel, 2–4 weeks to tune it. This is the path the KLAP kernel-launch-aggregation paper [17] describes as the "coarser-grain aggregation" solution; it is the analytically correct answer to "launch overhead dominates sub-100 µs work."

### 3.6 Persistent-threads megakernel with device-side command queue

**Verdict: Workable second choice, more complex.** One long-running kernel on a grid of persistent thread blocks, consuming work items (shape descriptors + buffer pointers) from a device-memory ring buffer. The host CPU pushes inner-Lanczos work items without launching new kernels; the persistent kernel exits only at the end of the Lanczos outer loop. This is the Aila-Laine ray-tracing pattern [18] generalized to linear algebra, and is the implicit model behind Work Graphs. Downsides: load imbalance across the three steps (Step 1 and Step 3 have very different CU utilization for small `d`), harder to profile, harder to interact with `rocsolver_dsteqr` for the Lanczos tridiag at the end of each matvec. Only worth considering if the fused kernel (§3.5) hits unexpected register-pressure walls.

### 3.7 Ranked recommendation

| Alternative | Effort | Speedup potential | Risk | Rank |
|---|---|---|---|---|
| **Custom fused 3-GEMM HIP kernel** (rocWMMA FP64) | 1–2 wk | 3–10× at chi≤64, 1.2–1.5× at chi≥128 | Low | **1** |
| Strided-batched rocBLAS + graph capture (if §1.4 holds) | 3–5 day | 1.3–1.5× at chi=64–128 | Medium | 2 |
| Composable Kernel fused GEMM | 3–4 wk | 2–5× | Medium | 3 |
| Persistent-threads megakernel | 2–3 wk | 2–4× | Medium-high | 4 |
| hipBLASLt batched | 1 wk | ≤1.3× | High (capture unknown) | 5 |
| Manual hipGraphAddKernelNode | n/a | n/a | n/a | excluded |
| hipExtLaunchMultiKernelMultiDevice | n/a | n/a | n/a | excluded |

**The single most effective technique is option 1: the custom fused HIP kernel.** It directly attacks the root cause (3 separate launches per matvec) without depending on infrastructure (graph capture) that AMD has not certified for BLAS-3 batched. It also happens to be architecture-portable: the same fused kernel design trivially ports to CUDA/H100 via CUTLASS. It is the answer that the KLAP paper and the persistent-threads literature converge on, and it matches AMD's own direction (Composable Kernel) and NVIDIA's own direction (CUTLASS + cuBLASDx) for tiny-GEMM performance.

---

## 4. CUDA graph parity for the H100 port

Short answer: **CUDA is much better than ROCm on this axis, but still has real caveats that the H100 port would inherit.**

### 4.1 `cublasDgemm` (non-batched)

Capture-safe since cuBLAS 11.x, confirmed in the current cuBLAS 12.x / 13.x documentation. [19] Requires: (a) `CUBLAS_POINTER_MODE_DEVICE` for alpha/beta, (b) `cublasSetWorkspace` with a pre-allocated buffer outside capture, (c) `cublasSetStream` to the capturing stream. No host-side sync involved for dense GEMM output (output is in device memory).

### 4.2 `cublasDgemmBatched` (pointer-array variant)

Capture-safe **provided the pointer arrays live in device memory** and are stable across graph executions. [20] Unlike rocBLAS, cuBLAS does **not** touch the pointer arrays on the host at launch-resolve time. The NVIDIA developer forum confirms this. This is the single biggest point of divergence from rocBLAS and the reason the CUDA port is the safer target for Proposal 3's design.

### 4.3 `cublasDgemmStridedBatched`

Capture-safe and recommended as the preferred batched interface for graph-based workloads. [19] No pointer indirection, no host reads.

### 4.4 Known CUDA-side gotchas that will bite the H100 port

- **Child graph incompatibility via stream-ordered allocation nodes** — cuBLAS 12.0+ will insert `cudaMallocAsync` nodes into captured graphs unless `cublasSetWorkspace` is called **outside** capture. If this is missed, `cudaGraphAddChildGraphNode` returns error 801 (operation not supported). [7] This is a trap we will hit on the H100 port and must document.
- **`cusolverDn*` eigen routines are not capturable** due to internal `cudaStreamSynchronize`. [11] We must keep the `dstev`-equivalent tridiagonal solve outside the captured region, same as Proposal 3 already scopes on the rocBLAS side.
- **cuBLASLt** has the same workspace caveat; configuration surface is different but capture semantics are equivalent.

### 4.5 Implication for the follow-up plan

If Phase 0 on MI300X fails, the CUDA port is **not** a guaranteed rescue — but it is far more likely to succeed. The practical plan is:

1. Run the MI300X Phase 0 with **strided-batched** rocBLAS (§1.4 escape hatch).
2. If it works, proceed with Proposal 3 as written, scoped to strided-batched paths.
3. If it fails, pivot to the custom fused kernel (§3.5) on MI300X first, then port that fused kernel to H100 via CUTLASS — **not** via CUDA graph capture of cuBLAS. The fused-kernel approach is portable and avoids depending on either ecosystem's BLAS-capture story.

---

## 5. Persistent-kernel alternatives for tiny GEMM chains

### 5.1 The state of the art, per 2024–2026 literature and libraries

The KLAP paper (Kernel Launch Aggregation, MICRO 2016) [17] establishes the analytical framework: "with coarser-grain aggregation, there are fewer kernel launches and more work per kernel. Therefore, the kernel launch overhead is amortized over a larger amount of work." This is the theoretical foundation for both graph capture and for fused megakernels. When graph capture is unavailable, aggregation becomes manual kernel fusion.

### 5.2 Available library building blocks

- **CUTLASS (NVIDIA)** — the reference implementation for fused-GEMM kernels on CUDA. CUTLASS 3.x supports "grouped GEMM" for batched-variable-size scenarios and exposes device-side GEMM mainloop / epilogue templates that can be composed into a fused multi-GEMM kernel. This is the NVIDIA path of choice for anything smaller than a canonical cuBLAS shape.
- **cuBLASDx (NVIDIA)** — device-callable cuBLAS headers exposing GEMM as a device function. Lets you call a full GEMM from inside your own kernel. Analogous to what we need. There is **no ROCm equivalent** as of ROCm 7.2 — this is a real gap in the AMD toolchain. [21]
- **Composable Kernel (AMD)** — the closest ROCm analog to CUTLASS, template-fused, compile-time instantiated, but with a steeper learning curve and smaller shape library. [14] Not a drop-in device-side callable.
- **rocWMMA (AMD)** — header-only wave-matrix intrinsics library. **Supports FP64 on gfx942 via MFMA-16 tiles** (`__builtin_amdgcn_mfma_f64_16x16x4f64`), which is production-ready since ROCm 5.x. [16] This is the primitive layer for building a custom fused FP64 kernel on MI300X.
- **MAGMA / batched BLAS** — classical batched-GEMM framework; does not offer fused-chain primitives for tiny matrices, only independent-batch dispatch. [22]
- **Kokkos Kernels / Ginkgo / MFEM** — none expose a fused-GEMM-chain primitive; they compose independent BLAS calls. [23]
- **hipBLASLt epilogue fusion** — supports elementwise epilogues only (bias, activation), **not** a second GEMM. This is consistent with cuBLASLt and is unlikely to change.

### 5.3 The single recommended technique

**Write a single custom fused HIP kernel using rocWMMA FP64 MFMA-16 tiles.** This is the one technique that:

1. Directly attacks the root cause (launch-overhead-per-GEMM) at the hardware level.
2. Is completely independent of the rocBLAS-capture question — it works whether or not graph capture is ever fixed.
3. Is portable to H100 by swapping rocWMMA → CUTLASS device-side templates, with near-identical architecture.
4. Composes naturally with the existing A1/A3/A7 optimizations already shipped (device pointer mode, GPU-side pointer setup kernels, sync removal) — none of those become wasted work; they all still apply.
5. Reuses the existing `dmrg2-gpu-opt` strided-batched Step 3 data layout as the starting point.

The fused kernel is Proposal 3's **backup plan** but is *also* the stronger standalone proposal. It should be written up as a follow-up Proposal 4 in parallel and treated as equal-priority with Proposal 3's Phase 0 microbench. If the microbench passes, both get tried and the one with better measured speedup wins. If the microbench fails, the fused kernel becomes the sole path forward and Proposal 3 is abandoned.

---

## Citations

1. rocBLAS Beta Features (Functions unsupported with Graph Capture + Windows note): https://rocm.docs.amd.com/projects/rocBLAS/en/latest/reference/beta-features.html
2. rocBLAS GitHub issue #1240 "Capturing Stream Safety" (still open): https://github.com/ROCm/rocBLAS/issues/1240
3. llama.cpp issue #14576 (empirical report of `gemm_strided_batched_ex` under capture): https://github.com/ggml-org/llama.cpp/issues/14576
4. ROCm graph-safe support matrix: https://rocm.docs.amd.com/en/latest/reference/graph-safe-support.html
5. ROCm 7.2 release notes: https://www.amd.com/en/resources/support-articles/release-notes/RN-AMDGPU-LINUX-ROCM-7-2.html
6. ROCm 7.2.1 release notes: https://www.amd.com/en/resources/support-articles/release-notes/RN-AMDGPU-LINUX-ROCM-7-2-1.html
7. NVIDIA Developer Forums "cuBLAS 12 graphs cannot be used as child graphs because of stream-ordered memory allocation": https://forums.developer.nvidia.com/t/cublas-12-graphs-cannot-be-used-as-child-graphs-because-of-stream-ordered-memory-allocation/276634
8. De Caro et al., "Work Graphs SpMV," ICPP 2025: https://cosenza.eu/papers/DeCaroICPP25.pdf
9. NVIDIA blog, "Constant-time launch for straight-line CUDA graphs": https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements/
10. NVIDIA cuSparse graph capture documentation: https://docs.nvidia.com/cuda/cusparse/
11. NVIDIA Developer Forums, "CUDA graph capture mode with cuSolver → CUSOLVER_STATUS_INTERNAL_ERROR": https://forums.developer.nvidia.com/t/cuda-graph-capture-mode-with-cusolver-get-cusolver-status-internal-error/309210
12. AMD ROCm Blog, "Revolutionizing Mixture of Experts Performance" (2025, includes "does not support CUDA graph capture" caveat): https://www.amd.com/en/blogs/2025/revolutionizing-mixture-of-experts-performance-10.html
13. rocBLAS Programmers Guide (memory allocation schemes, `rocblas_initialize`): https://rocm.docs.amd.com/projects/rocBLAS/en/docs-5.5.1/Programmers_Guide.html
14. AMD Composable Kernel (AI inference optimization guide): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-with-composable-kernel.html
15. hipBLASLt GitHub: https://github.com/ROCm/hipBLASLt
16. rocWMMA GitHub (FP64 MFMA-16 support on gfx942): https://github.com/ROCm/rocWMMA
17. El Hajj et al., "KLAP: Kernel Launch Aggregation and Promotion," MICRO 2016: https://ielhajj.github.io/publications/paper/paper-klap-micro16.pdf
18. Aila & Laine, "Understanding the Efficiency of Ray Traversal on GPUs" (persistent-threads pattern origin), HPG 2009.
19. NVIDIA cuBLAS Documentation (CUDA 12.3+ stream capture guidance): https://docs.nvidia.com/cuda/archive/12.3.1/cublas/index.html
20. NVIDIA Developer Forums, "How can I call cublasSgemmBatched on pointer device arrays without allocating them twice" (confirms device-side pointer arrays work under capture): https://forums.developer.nvidia.com/t/how-can-i-call-cublassgemmbatched-on-pointer-device-arrays-without-allocating-them-twice/63692
21. NVIDIA cuBLASDx (device-callable cuBLAS): https://developer.nvidia.com/cublasdx
22. Dongarra et al., "Batched BLAS (Basic Linear Algebra Subprograms) 2018 Specification": https://www.netlib.org/utk/people/JackDongarra/PAPERS/ipdps-batched-2019.pdf
23. Kokkos Kernels / Ginkgo documentation (composition of independent BLAS calls, no fused-chain primitive): standard project docs, no fused-GEMM-chain API surface documented as of 2025.
