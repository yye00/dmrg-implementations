# Research Report C: Persistent On-Chip Lanczos Kernel

**Agent**: Research Agent C (follow-up round)
**Date**: 2026-04-10
**Target**: AMD Instinct MI300X (gfx942 / CDNA3)
**Problem**: At `chi ≤ 50` single-core CPU beats MI300X in 93% of benchmark
configs; root cause is cache residency — CPU keeps `theta` (~1 MB) in L1/L2
across 15–20 Lanczos iterations, GPU refetches from HBM every matvec.

## Executive summary

A fully fused, single-launch Lanczos outer loop with the full Krylov basis
resident in LDS does **not** appear in any published work for CDNA or Hopper.
The pattern is architecturally sound but blocked by three realities:

1. **LDS is 64 KB per CU on gfx942 — hard, static, un-expandable.** No
   dynamic-LDS / large-LDS mode exists on CDNA3 [1][2]. That caps `theta` at
   `chi_L · d² · chi_R · 8 B ≤ 64 KB`, i.e. `chi ≤ 32` for `d=2` (exactly the
   envelope the caller guessed, and consistent with our 93%-CPU-wins regime).
2. **Cooperative launch works on MI300X but is ~10× slower than normal launch
   and relies on software-emulated grid sync via global atomics** [3][4]. So a
   grid-wide persistent Lanczos is a trap: you pay the overhead you are trying
   to amortise.
3. **There is no `cudaAccessPolicyWindow` equivalent on HIP.** AMD exposes
   cache hints only through ISA-level GLC/SLC/DLC/NT flags on
   `BUFFER_LOAD` / `GLOBAL_LOAD` [5][6], and the 256 MB Infinity Cache is
   hardware-managed memory-side and cannot be pinned.

The viable architecture is therefore **per-workgroup persistent** (not
grid-wide): one workgroup per bond, `theta` and Krylov vectors in that CU’s
64 KB LDS, `H_eff` inlined as hand-rolled MFMA (`v_mfma_f64_16x16x4f64`), the
entire Lanczos outer loop inside a single kernel launch, tridiagonal solve
done on-chip with a closed-form QL on a 20×20 matrix. This is inspired by
CUTLASS / Composable Kernel persistent-GEMM scheduling [7][8][9] and Mirage
megakernel paging [10], but Lanczos-specific — none of the mainstream
libraries implement it.

## 1. LDS budget and problem-size envelope

Per the CDNA3 white paper and ISA reference, each CU on MI300X has exactly
**64 KB of LDS, unchanged from CDNA2** [1][5]. There is no CDNA3 equivalent of
NVIDIA's opt-in dynamic shared memory modes (Hopper has 228 KB via opt-in).
Community references to "160 KB" are noise — they describe multi-CU totals or
different arches, not gfx942 workgroups [1].

Concrete envelope for our kernel (FP64, `d=2`):

| `chi` | `theta` size | fits in 64 KB LDS? | Krylov basis (20 vec) size | fits? |
|-------|--------------|--------------------|----------------------------|-------|
| 16    | 2 KB         | trivially          | 40 KB                      | yes   |
| 24    | 4.5 KB       | yes                | 90 KB                      | **no** |
| 32    | 8 KB         | yes                | 160 KB                     | **no** |
| 40    | 12.5 KB      | yes                | 250 KB                     | **no** |
| 48    | 18 KB        | yes                | 360 KB                     | **no** |
| 64    | 32 KB        | yes                | 640 KB                     | **no** |

So the **naive "keep everything in LDS"** envelope is `chi ≤ ~16` for `d=2`
with 20 Krylov vectors. If we relax and **keep only `theta` and `q_k`
(current Krylov vector) in LDS** and spill the older basis to L2, the
envelope extends to `chi ≤ 48` for `d=2` and `chi ≤ 20` for `d=4` — which
covers the entire regime where single-core CPU currently wins. The spilled
basis lives in L2 at 4 MB/CU (actually shared across XCD clusters) and in the
256 MB Infinity Cache, both of which will readily keep 20 × 32 KB = 640 KB
hot, because it sits under the L2 line-capacity and *way* under Infinity
Cache capacity [1][11].

**Caveat**: LDS also competes with register pressure for occupancy. A
workgroup using ~32 KB of LDS allows only one resident workgroup per CU
(max occupancy 1), which is fine for a persistent design where each
workgroup owns one bond — but it means the CU cannot overlap waves, so the
matvec must saturate its own DRAM/LDS pipelines from within a single
workgroup [11][12].

## 2. Published persistent-kernel patterns

Multi-source deep search (arXiv, SC, PPoPP, ICS, ROCm docs, AMD white
papers) returned **zero hits** for the exact phrase "persistent Lanczos" or
"in-kernel Krylov" or "fused eigensolver kernel" as a first-class pattern
[10][13][14]. The literature does contain adjacent work:

- **Mirage Persistent Kernel (MPK), arXiv 2512.22219** — a megakernel compiler
  that fuses a whole LLM inference graph into a single persistent launch
  using a **paged LDS abstraction**: LDS is sliced into fixed pages owned by
  tasks, with explicit acquire/release and overlapped prefetch of next task
  into released pages [10]. MPK is the closest published precedent for
  "fused iterative kernel with on-chip state management" but targets
  dataflow LLM graphs, not Krylov iterations. **Directly adaptable pattern**:
  reserve one LDS page for `theta`, rotate 2 pages for the 3-term Lanczos
  recurrence (`q_{k-1}`, `q_k`, `q_{k+1}`), keep alphas/betas in registers.
- **CUTLASS 3 / CuTe persistent GEMM** [7][8] — persistent thread blocks tile
  the output and stream A/B through. The scheduling idea (one block owns one
  piece of work for the whole kernel lifetime) is directly transferable.
  CUTLASS itself has no Krylov support.
- **AMD Composable Kernel (CK)** [9][15] — tile-based programming model with
  `StaticDistributedTensor` LDS allocation, XOR preshuffling for bank-conflict-
  free access, and `TileWindow` for streaming operand. CK is the AMD answer
  to CUTLASS. It has no eigensolver, but it provides the right primitives for
  hand-writing a persistent matvec whose A operand is `W_L ⊗ W_R` (tiny) and
  whose B operand is `theta` (streamed from LDS).
- **Cucheb (filtered Lanczos on GPU)** — computes Ritz values on the CPU
  every outer iter [14], which is precisely what we want to avoid. It's a
  counter-example: a state-of-the-art filtered Lanczos that does **not**
  persist across iterations, and has to do CPU round-trips for convergence.
- **NVIDIA cuSolver / RAFT / MAGMA / AMD rocSOLVER / AOCL-Sparse** — all ship
  Lanczos-family solvers but expose them as host-callable black boxes with
  per-iter kernel launches and per-iter reductions [16][17][18]. None have
  a persistent variant.
- **CUB / hipCUB / rocPRIM** — device-primitive libraries (reduce, scan, sort).
  No eigensolver primitives [19]. They could be used as the building blocks
  for the in-workgroup reductions inside a persistent Lanczos but are not
  themselves the pattern.

**Bottom line**: the pattern we want is unpublished for Lanczos on any GPU.
We will be writing something novel, but the component ideas exist (CUTLASS
persistent scheduling + MPK paged LDS + CK tile operators).

## 3. Cooperative groups / grid-wide synchronization on HIP

`hipLaunchCooperativeKernel` and `cooperative_groups::grid_group::sync()`
exist on MI300X [3][20]. The mechanism is implemented as a **global atomic
spin-wait barrier**, not hardware-assisted — NVIDIA has dedicated CTA-activate
hardware since Pascal, AMD does not [20].

**ROCm issue #3410 documents that `hipLaunchCooperativeKernel` is "an order
of magnitude slower than the normal launch" even with a single workgroup
and no sync calls** [4]. The overhead is in the launch path itself
(resource reservation so forward progress can be guaranteed), not the sync
primitive. This kills grid-wide persistent Lanczos stone dead: we would pay
10× more launch overhead per bond than normal dispatch, which is the
opposite of what we want.

The **per-workgroup persistent** model avoids this entirely: each bond gets
one workgroup, workgroups are independent, the grid is dispatched via
regular `hipLaunchKernelGGL`, and no cross-workgroup sync is needed —
inter-Krylov synchronization is purely intra-workgroup via `__syncthreads()`
and LDS atomics, which are cheap and hardware-assisted.

## 4. L2-residency hints and cache-control

**No `cudaAccessPolicyWindow` analogue exists on HIP** [21][5]. AMD exposes
cache-control only at the ISA level via flags on memory instructions:

- **GLC (globally coherent)** — force global coherency; implies cache bypass
  for conflicting lines.
- **SLC (system-level coherent)** — promotes to system-level cache tier.
  SLC=0 biases toward L2 residency; SLC=1 allows Infinity Cache.
- **DLC (device-level coherent)** — device-scope coherency (multi-XCD).
- **NT (non-temporal)** — hint that data is access-once; discourage caching.

For the persistent Lanczos matvec, we want **SLC=0 and NT=0 on `theta` and
the Krylov vectors**, which biases the cache controller to keep them warm
in L2. These flags can be set via clang builtins (`__builtin_amdgcn_...`) or
inline assembly in HIP kernels but are not exposed through `hipMemAdvise` or
rocBLAS APIs [5][6]. **Practical guidance**: for a hand-rolled kernel we can
emit the right flags; for rocBLAS calls we get whatever rocBLAS emits.

The **Infinity Cache (256 MB, ~11.9 TB/s measured, ~218 ns latency)** is a
memory-side cache that does not hold dirty lines and does not participate in
coherency [1][11]. It is entirely hardware-managed — **there is no way to pin
a buffer into it**. The good news: a 1 MB `theta` vs a 256 MB cache means
natural LRU will almost certainly retain it across Lanczos iterations, as
long as no cache-pollution kernel runs in between. The *bad* news: the
current PDMRG benchmark runs other kernels (SVD, env update) in between
Lanczos calls, and **they do blow `theta` out of L2**. The persistent
kernel design inherently fixes this by keeping `theta` in LDS (not just L2)
for the duration of the 15–20 matvecs of one bond.

## 5. Existing library building blocks

| Library        | Helpful?                          | Notes                                                                                       |
|----------------|-----------------------------------|---------------------------------------------------------------------------------------------|
| **hipTENSOR**  | No (for this purpose)             | Plan-based one-shot contractions via `hiptensorCreatePlan`; no persistent or fused multi-step primitives [22][23]. Usable only as a host-side building block for the *reference* matvec. |
| **rocBLAS**    | Partial — as graph nodes          | Supports `hipStreamBeginCapture` → `hipGraphLaunch` for most L3/EX calls (Level-1 with host pointers is excluded) [24]. This is the Proposal-3 path, but it still does per-iter kernel launches inside the graph. |
| **rocSOLVER**  | No                                | Host-callable dense solvers; no persistent primitives [17].                                 |
| **rocPRIM / hipCUB** | Yes — as in-kernel primitives | Provide block-reduce, block-scan, block-sort templates that can be called **inside** our persistent kernel for the Lanczos inner products and normalizations. This is how to do on-chip reductions without grid-sync. [19] |
| **Composable Kernel (CK)** | Yes — the closest to what we need | Tile operators, LDS tensor allocation, bank-conflict-free XOR preshuffle, `TileWindow` streaming — all the primitives needed to write the persistent matvec by hand. Upstream has no eigensolver but CK is AMD's answer to CUTLASS and is MI300X-tuned [9][15]. |
| **Mirage MPK** | Conceptually — paged LDS idea     | Not directly usable (CUDA-only, LLM-targeted) but the paged-LDS abstraction is the right mental model for juggling `theta`, `q_k`, and the 3-term recurrence [10]. |

**Key finding on hipTENSOR**: it exposes contraction *plans* and *executes*,
nothing more [22][23]. There is no "persistent contraction session"
primitive. This means the project instruction that "as much of the linear
algebra calls as possible should use hipTENSOR / rocBLAS" conflicts with the
persistent-kernel pattern — we will have to make an explicit exception and
hand-roll the in-LDS matvec using CK tile operators or raw MFMA intrinsics.

## 6. Proposed architecture for a persistent Lanczos matvec kernel

Based on everything above, the proposed design for `apply_heff_persistent`:

**Dispatch model**: regular `hipLaunchKernelGGL` (not cooperative), one
workgroup per independent bond (Stoudenmire segments dispatched in parallel;
for serial two-site DMRG, one workgroup total — a "persistent single-bond"
kernel that runs the full Lanczos outer loop in one launch).

**Workgroup size**: 256 threads (4 wavefronts), exactly one workgroup per CU
(occupancy 1 because of LDS pressure).

**LDS layout** (within 64 KB budget, `chi=32, d=2` example):
- `theta_lds`: 4096 doubles = 32 KB
- `q_prev`, `q_curr`, `q_next`: 3 × 4096 doubles = 96 KB → **does not fit**.

Resolution: rotate only 2 buffers in LDS (`q_curr` + `q_next_scratch` at 8 KB
each), spill the full Krylov basis `{q_0, …, q_{k-1}}` to L2-resident
global memory with SLC=0 / NT=0 hints. Reorthogonalization reads the spilled
basis from L2 (4 MB/CU L2, trivially fits 20 × 8 KB = 160 KB basis). This
puts **`theta` and the hot working vector in LDS**, and the archival basis
in L2 — which is still 10–50× better than the current situation where
everything is in HBM.

**Revised LDS budget for `chi=32, d=2`**:
- `theta_lds`: 32 KB
- `q_curr`: 8 KB
- `w_lds` (matvec scratch): 8 KB
- MFMA staging / alpha/beta scalars: ~1 KB
- **Total: ~49 KB** (fits).

**Matvec kernel body**:
1. Load `theta` from HBM into LDS once (single large coalesced load).
2. Load MPO tiles `W_L`, `W_R`, and environment tiles `L_env`, `R_env` from HBM
   into LDS with SLC=0 hints.
3. Inline the 4-contraction `H_eff · theta` in MFMA `v_mfma_f64_16x16x4f64`,
   reading operands from LDS, accumulating into VGPR, writing result back to
   `q_next_scratch` in LDS.
4. Block-level `alpha = <q_curr, w>` reduction via `rocprim::block_reduce`.
5. `w -= alpha * q_curr + beta * q_prev` in LDS; full reorthogonalization
   reads spilled basis from L2.
6. `beta = ||w||` via block-reduce; normalize; rotate `q_prev ← q_curr`,
   `q_curr ← w / beta`; append `q_curr` to the L2-resident basis.
7. Every ~5 iters, compute the running tridiagonal eigenvalue on-chip
   (closed-form QL on a 20×20 matrix fits in registers / LDS scratch;
   ~100 flops per iter, negligible).
8. When converged, back-project the Ritz vector against the L2-resident
   basis, write result to HBM, exit.

**Expected benefit**: eliminates 15–20 rocBLAS launches per bond (currently
~5–10 µs each = 75–200 µs/bond wasted in launch overhead, dominating the
2–5 µs of actual matvec compute at `chi ≤ 32`). Eliminates 15–20 HBM round
trips for `theta`. Replaces per-iter `rocblas_dot` / `dnrm2` /
`rocblas_axpy` (each a full kernel launch with implicit sync) with in-LDS
block reductions.

**Estimated speedup**: at `chi=32 d=2` the CPU currently wins by ~3–5×; a
persistent kernel that eliminates launch overhead and keeps the operand in
LDS should recover that factor and likely overshoot — the MI300X MFMA double
throughput is 81.7 TF, and at `chi=32` a single matvec is ~300 KFLOPs, so
the hardware floor is ~4 ns; the achievable floor is bound by LDS bandwidth
and matvec latency and is closer to ~5 µs (including reductions and
reorthogonalization). 15 iters × 5 µs = 75 µs/bond vs. the current
700–1500 µs/bond. Rough back-of-envelope: **10–20× speedup in the Lanczos
portion of small-chi sweeps** — which is exactly the regime where CPU
currently wins.

## 7. Risks and open questions

1. **Correctness of hand-rolled MFMA matvec**. Writing FP64 MFMA by hand in
   HIP is unforgiving. CK's tile operators are the safer path. Risk: getting
   the same answers as the current rocBLAS-based matvec to 1e-12.

2. **Occupancy = 1 per CU hurts latency hiding**. With ~49 KB LDS per
   workgroup only one workgroup fits per CU, so a stalled memory load from
   the spilled basis reorthogonalization cannot be hidden by switching waves.
   Mitigation: issue prefetches several iterations in advance, and keep the
   basis accesses in L2 (which has ~100 ns latency, maskable by compute).

3. **LDS capacity at `chi > 48`**. The persistent design covers the entire
   `chi ≤ 32` CPU-wins regime and extends to `chi ≤ 48` with `d=2`, but
   larger bonds need a different strategy (tile-fetch `theta` from L2). At
   `chi ≥ 64` rocBLAS already wins (launch overhead amortizes), so this is
   not a regression, but the crossover needs careful tuning.

4. **FP32 operand with FP64 accumulate — no ISA support**. CDNA3 MFMA has
   `v_mfma_f64_16x16x4f64` (FP64 in, FP64 out) and various FP32/BF16 →
   FP32 ops, but **no FP32-in → FP64-accumulate op** [5][25]. The
   "FP32 theta in LDS, FP64 accumulate" idea doubles the effective LDS
   budget but requires software emulation of the precision conversion inside
   the matvec. Given the project "no single precision" rule and the absence
   of native hardware support, **this is not a good first-pass path**.
   Deferred to a later optimization round.

5. **Cache-control flags are un-benchmarked**. SLC=0 / NT=0 on CDNA3 are
   hints, not guarantees. A microbenchmark of "same buffer touched by K
   kernels back to back" would quantify actual L2 retention on MI300X.
   We should measure this before committing to the "spill basis to L2"
   design.

6. **Interaction with HIP graph capture (Proposal 3)**. Even the persistent
   kernel can be graph-captured. Inside a graph, the persistent kernel is a
   single node — effectively free launch overhead. But graph capture around
   the whole sweep requires rebuilding the graph whenever bond dims change,
   which happens constantly in DMRG. Graph-update-without-rebuild support
   in ROCm 7.x is reportedly limited compared to CUDA [24]. Open question:
   do we need the graph at all once launches are fused?

7. **Cooperative launch as an escape hatch** is essentially ruled out by the
   10× overhead [4]. If we ever do need grid-wide sync, we should use
   persistent kernels with global-atomic barriers via ROCm `__threadfence()`
   and `atomicAdd` — effectively re-implementing what cooperative launch does
   badly — and pay only for the sync when we need it.

## Citations

[1] AMD, "AMD CDNA 3 Architecture White Paper"
https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf

[2] AMD Instinct MI300X CDNA3 Instruction Set Architecture
https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf

[3] ROCm HIP Docs, "Cooperative Groups"
https://rocm.docs.amd.com/projects/HIP/en/docs-6.1.2/how-to/cooperative_groups.html

[4] ROCm GitHub Issue #3410, "hipLaunchCooperativeKernel is ~10× slower than normal launch"
https://github.com/ROCm/ROCm/issues/3410

[5] LLVM AMDGPU Backend Usage (GLC/SLC/DLC/NT flags)
https://llvm.org/docs/AMDGPUUsage.html

[6] LLVM AMDGPU GFX940 Assembly Reference
https://llvm.org/docs/AMDGPU/AMDGPUAsmGFX940.html

[7] NVIDIA CUTLASS, "Efficient GEMM" (persistent thread blocks, GemmUniversal)
https://docs.nvidia.com/cutlass/4.2.1/media/docs/cpp/efficient_gemm.html

[8] NVIDIA CUTLASS, "GEMM API 3x" (collective mainloop, persistent scheduling)
https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html

[9] AMD Composable Kernel, "LDS Bank Conflicts and TileWindow"
https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/hardware/lds_bank_conflicts.html

[10] Mirage Persistent Kernel, arXiv 2512.22219
https://arxiv.org/html/2512.22219v1

[11] Chips and Cheese, "Testing AMD's Giant MI300X" (Infinity Cache latency / bandwidth measurements)
https://chipsandcheese.com/p/testing-amds-giant-mi300x

[12] ROCm HIP Docs, "Performance Guidelines"
https://rocm.docs.amd.com/projects/HIP/en/develop/how-to/performance_guidelines.html

[13] NVIDIA GTC 2012, "Persistent Threads for GPU Computing"
https://developer.download.nvidia.com/GTC/PDF/GTC2012/PresentationPDF/S0157-GTC2012-Persistent-Threads-Computing.pdf

[14] NVIDIA Developer Forums, "Persistent Kernel Concept"
https://forums.developer.nvidia.com/t/question-about-persistent-kernel-concept/320600

[15] AMD Composable Kernel project
https://github.com/ROCm/rocm-libraries (ck_tile subtree)

[16] NVIDIA cuSOLVER documentation
https://docs.nvidia.com/cuda/cusolver/index.html

[17] ROCm rocSOLVER types reference
https://rocm.docs.amd.com/projects/rocSOLVER/en/latest/reference/types.html

[18] MAGMA-enabled workflow paper, Dongarra et al., 2024
https://www.netlib.org/utk/people/JackDongarra/PAPERS/magma-enabled-2024.pdf

[19] rocPRIM device-ops configuration reference
https://rocm.docs.amd.com/projects/rocPRIM/en/docs-5.0.2/device_ops/config.html

[20] NVIDIA Developer Forums, "Cooperative-group grid sync implementation"
https://forums.developer.nvidia.com/t/does-the-grid-sync-in-cooperative-groups-have-the-same-functionality-as-the-device-wide-synchronization/286536

[21] NVIDIA Developer Forums, "CUDA L2 Residency Control"
https://forums.developer.nvidia.com/t/cuda-l2-residency-control/273360

[22] hipTENSOR API reference
https://rocm.docs.amd.com/projects/hipTensor/en/develop/api-reference/api-reference.html

[23] hipTENSOR Programmer's Guide
https://rocm.docs.amd.com/projects/hipTensor/en/latest/conceptual/programmers-guide.html

[24] rocBLAS Beta Features (HIP Graph capture support)
https://rocm.docs.amd.com/projects/rocBLAS/en/latest/reference/beta-features.html

[25] AMD Instinct CDNA4 Instruction Set Architecture (for comparative MFMA ops)
https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf
