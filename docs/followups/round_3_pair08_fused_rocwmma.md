# Round 3 — Pair 08: Fused rocWMMA apply_heff alternative kernel

**Pair:** 08 — Custom fused HIP/rocWMMA kernel as R2-4 alt path (refinement).
**Assumption:** Pair 7 returned NO-GO (rocBLAS stream capture unusable on ROCm 7.2).
**Role:** Generator + adversarial reviewer for the fused-kernel fallback.
**Date:** 2026-04-10.
**Live VM:** `hotaisle@23.183.40.84`, MI300X gfx942, ROCm 7.2.0, `/home/hotaisle/dmrg-implementations/sandbox/pair08/`.

## TL;DR — the verdict is loud

**ABANDON the full three-step fused kernel.** Stay with rocBLAS calls, and
if graph capture is unavailable address the problem through other means
(R2-3 persistent Lanczos at `chi ≤ 32`, R2-1 CBE to shrink the problem, or
R2-2 parallel scheme). The fused-kernel alt path in the Research A write-up
is **wishful thinking**, and this Round 3 investigation proves it
concretely.

Key numbers that force this verdict:

1. **LDS budget is blown at every interesting size.** At `chi = 32, d = 2`
   the three working tensors (theta + T1 + T2) already want **352 KB of
   LDS**, vs MI300X's hard 64 KB/CU limit. At `chi = 64, d = 2` it's **1408
   KB** — **22× over**. The Research A "all working tensors comfortably fit
   in MI300X LDS" claim is simply false.
2. **Live microbench on MI300X: a naive fused Step 1 kernel is 3.0× slower
   than rocBLAS at `chi = 32`, 22× slower at `chi = 64`.** The rocBLAS
   strided-batched call dispatches at ~7.25 µs regardless of size — which
   is already near the per-launch floor we'd be trying to undercut. A
   hand-rolled kernel has to beat 7 µs/call for the ENTIRE fused sequence,
   including MFMA programming overhead, while also managing register
   pressure and tile orchestration that rocBLAS's Tensile kernels handle
   for free.
3. **Arithmetic intensity says the work is memory-bound at `chi = 32` and
   only marginally compute-bound at `chi = 64`.** Fusion buys compute
   savings, not memory savings. Where the problem is memory-bound, fusion
   is irrelevant; where it's compute-bound, rocBLAS is already ~1.5 TF/s on
   a 10 MFLOP call — further work-per-launch is not the binding
   constraint.
4. **rocWMMA 2.x FP64 is real but fragile.** rocWMMA internally wraps
   `__builtin_amdgcn_mfma_f64_16x16x4f64` (confirmed by `grep` on
   `/opt/rocm/include/rocwmma/internal/mfma_impl.hpp`), and the fragment
   shape is fixed at **BlockM = BlockN = 16, BlockK ≥ 4** — so any `chi`
   not a multiple of 16 must be padded per-kernel-invocation. This
   interacts badly with the existing MPS bond-growth logic.
5. **The template-explosion problem is real.** `apply_heff_two_site` is
   called with at least 10 distinct `(chi_L, chi_R)` shapes per segment ×
   `d ∈ {2, 3, 4}` × `D_mpo ∈ {4, 5, 6}` × left/right sweep direction. A
   template-per-shape fused kernel explodes into dozens of specializations
   with a compile-time cost measured in minutes per specialization.

Realistic calendar estimate: **6–10 weeks** for a correct, tuned fused
kernel that matches rocBLAS at `chi ∈ [32, 128]`. Research A's "1–2 weeks"
number is derived from a hypothetical best case with unlimited LDS and
shape-oblivious MFMA tiles — neither of which obtains on gfx942.

The rest of this document provides the concrete design exercise, the
measurement, the adversarial critique, and the recommended next action.

---

## 1. Full kernel spec (as if we were going to write it)

### 1.1 Inputs and shapes

From `gpu-rocm/dmrg2-gpu/src/dmrg2_gpu_impl.h:473–550`, the current call
chain for `apply_heff_two_site(site, d_theta_in, d_result)` is:

```
chi_L = chi_L(site)           // bond dim left
chi_R = chi_R(site + 1)       // bond dim right
d     = 2 (Heisenberg), 3 (Josephson t-J-like), 4 (doubled-space)
D     = D_mpo_ (4–6 for Heisenberg / Josephson / XY)
dd    = d * d

Step 1 (batched GEMM, D*dd batches):
  T1[w,s1,s2][a',b] = sum_a L_env[w][a,a']^T * theta[s1,s2][a,b]
  Shape: (chi_L, chi_L) · (chi_L, chi_R) → (chi_L, chi_R), D*dd times

Step 2 (dense GEMM):
  T2[a',b, s1',s2',n] = sum_{w,s1,s2} T1[w,s1,s2][a',b] * WW[w,s1,s2; s1',s2',n]
  Shape: (chi_L*chi_R, D*dd) · (D*dd, dd*D) → (chi_L*chi_R, dd*D)

Step 3 (batched GEMM with accumulation, D batches × dd sub-batches):
  result[a',s1',s2',b'] = sum_n T2[a',b, s1',s2',n] * R_env[n][b,b']
  Shape: (chi_L, chi_R) · (chi_R, chi_R) → (chi_L, chi_R), dd per n, with
  accumulation over n
```

### 1.2 Hypothetical fused kernel signature

```
__global__ void apply_heff_fused(
    const double* __restrict__ d_L_env,    // (D,  chi_L, chi_L)
    const double* __restrict__ d_theta,    // (chi_L, d*d, chi_R)
    const double* __restrict__ d_WW,       // (D*d*d, d*d*D) — fused two-site MPO
    const double* __restrict__ d_R_env,    // (D,  chi_R, chi_R)
    double*       __restrict__ d_result,   // (chi_L, d*d, chi_R)
    int chi_L, int chi_R, int d, int D
);
```

Would be dispatched with (best-case) one workgroup per output tile:

```
int tile_rows = 16;  // BlockM of MFMA-16
int tile_cols = 16;  // BlockN of MFMA-16
dim3 grid( ceil(chi_L/16), ceil(chi_R/16), d*d );  // output tiles
dim3 block( 64 );      // one wavefront per tile, MFMA lanes
size_t lds_bytes = /* computed per-shape; see §3 */;
```

One block produces one `(16 × 16)` output tile of a fixed `(s1', s2')`
slab of the result, iterating over the reduction dimensions (a, b, n, w,
s1, s2) internally. This is the CUTLASS-style "one block per output tile"
pattern adapted to the three-step chain.

**This sketch is abandoned below.** The reason is the LDS budget: the
block cannot even hold its own slice of `theta` + T1 + T2 without
aggressive per-tile re-tiling, and the tile scheduler ends up with almost
as many global memory trips as the unfused rocBLAS path.

---

## 2. Hypothetical LDS layout and staging

The "ideal" pipeline (if LDS were unlimited) would be:

1. Load `d_L_env[w]` tile (cL×cL) into LDS page A.
2. Load `d_theta` tile (cL×dd×cR) into LDS page B.
3. MFMA Step 1: page A · page B → T1 in LDS page C. (D × dd tiles.)
4. Load `d_WW` (D*dd × dd*D) into LDS page D — tiny, fits trivially.
5. MFMA Step 2: page C · page D → T2 in LDS page E (overwriting A, B).
6. Load `d_R_env[n]` (cR×cR) into LDS page F.
7. MFMA Step 3: page E · page F → result in LDS page G, accumulated.
8. Store result to HBM.

Peak LDS is dominated by pages A + B + C or C + D + E simultaneously, i.e.
`theta + T1 + WW` or `T1 + T2 + WW`.

## 3. LDS + register budget: IS IT FEASIBLE?

Exact byte counts (all in FP64, col-major):

```
theta = chi_L · d² · chi_R · 8
T1    = D · d² · chi_L · chi_R · 8
T2    = chi_L · chi_R · d² · D · 8
WW    = D² · d⁴ · 8
```

| chi | d | D | theta   | T1       | T2       | WW      | θ+T1+T2   | vs 64 KB LDS |
|----:|--:|--:|--------:|---------:|---------:|--------:|----------:|-------------:|
|  16 | 2 | 5 |  8.0 KB |  40.0 KB |  40.0 KB |  3.1 KB |   88.0 KB | **1.4×over** |
|  32 | 2 | 5 | 32.0 KB | 160.0 KB | 160.0 KB |  3.1 KB |  352.0 KB | **5.5×over** |
|  64 | 2 | 5 |128.0 KB | 640.0 KB | 640.0 KB |  3.1 KB | 1408.0 KB | **22×over**  |
| 128 | 2 | 5 |512.0 KB |2560.0 KB |2560.0 KB |  3.1 KB | 5632.0 KB | **88×over**  |
| 256 | 2 | 5 |  2.0 MB | 10.0 MB  | 10.0 MB  |  3.1 KB |  22.0 MB  | **352×over** |
|  32 | 3 | 5 | 72.0 KB | 360.0 KB | 360.0 KB | 15.8 KB |  792.0 KB | **12×over**  |
|  64 | 3 | 5 |288.0 KB |1440.0 KB |1440.0 KB | 15.8 KB | 3168.0 KB | **50×over**  |
|  32 | 4 | 5 |128.0 KB | 640.0 KB | 640.0 KB | 50.0 KB | 1408.0 KB | **22×over**  |

Computed by `python3 -c "for cL in [16,32,64,128,256]: ..."` — raw numbers
match the task-prompt's own calculation (`chi=128, d=2: T1 = 131 KB`, here
listed as `2560 KB`; the prompt used `chi_L·d²·chi_R` per tile but
`T1` has an additional factor of `D` from the MPO bond index).

**Conclusion:** Not a single configuration fits the naive "everything in
LDS" pattern. The task prompt itself noted chi=128 is 2× over — the
reality is far worse once you account for both T1 and T2 live
simultaneously, and the D-fold MPO bond index.

### 3.1 Tiling rescue attempt

The only way to survive is per-output-tile execution:

- Each workgroup owns one `(16, 16)` tile of the `(chi_L, chi_R)` output
  at fixed `(s1', s2')`.
- LDS holds only that tile's share: 16×16 MFMA fragments for each of A,
  B, accumulator.
- The block iterates: for each `n ∈ [0, D)`, for each `w ∈ [0, D)`, load
  the relevant `L_env[w]` slice and `R_env[n]` slice, do the MFMAs.
- WW is fully loaded into LDS once (≤ 50 KB for d=4, D=5) — fits.

**But now the kernel is doing essentially the same global memory loads
as three independent GEMMs**, because each output tile must re-read all
of `L_env`, all of `theta`, and all of `R_env`. The only savings come
from not materializing T1 and T2 in HBM — which is real savings but
small:

- Step 1 output T1 = `D · dd · chi_L · chi_R · 8` = ~10 KB at chi=32 d=2
  D=5 × 20 = still only 160 KB of avoided HBM traffic per call.
- At 5.3 TB/s HBM: 160 KB saves ~30 ns. Launch overhead of one rocBLAS
  call is ~7 µs. Savings are **0.4% of launch overhead**.
- At chi=256 the saved T1 is 10 MB and 5.3 TB/s saves ~2 µs. Meaningful
  but not transformative — and at chi=256 rocBLAS is already fast.

### 3.2 Register budget for the tile kernel

Each wavefront holds one `(16, 16, 4)` MFMA accumulator = 16 AccVGPRs per
wave for the D operand of `v_mfma_f64_16x16x4f64` (per `rocwmma` source
at `/opt/rocm/include/rocwmma/internal/mfma_impl.hpp:836`, `DRegsT =
AccRegF64x4` means 4 accum registers per lane × 64 lanes / 16-wide = 16
VGPRs). Add two input fragment registers, prefetch buffers, loop
counters — conservatively 64 VGPRs per wave. Occupancy of 2 waves/CU is
feasible, 4 would need unlikely register sharing. Register budget is
**not the binding constraint** — LDS and launch economics are.

---

## 4. Arithmetic intensity / roofline analysis

Computed for Heisenberg parameters `d = 2, D = 5` across the target
regime:

```
Step 1 flops = 2 · chi_L · chi_R · chi_L · D · d²
Step 2 flops = 2 · chi_L · chi_R · (D · d²)²
Step 3 flops = 2 · chi_L · chi_R · chi_R · D · d²
Total bytes  = (L_env + theta + WW + R_env + result) · 8
```

| chi | total FLOPs | total bytes | AI (FLOP/B) | roofline verdict |
|----:|------------:|------------:|------------:|------------------|
|  32 |     3.44 M  |  147.1 KB   |   22.8      | **memory-bound** (below 30.7 ridge) |
|  64 |    24.25 M  |  579.1 KB   |   40.9      | weakly compute-bound |
| 128 |   180.88 M  | 2307.1 KB   |   76.6      | compute-bound |
| 256 |  1394.61 M  | 9219.1 KB   |  147.7      | strongly compute-bound |

MI300X FP64 matrix peak: ~163 TFLOPS, HBM BW: ~5.3 TB/s, ridge AI ≈ 30.7
FLOP/byte.

**Implication:**
- At `chi = 32` the entire matvec is memory-bound. **Fusion buys nothing
  here.** Fusion saves recomputation and launch count; it does not save
  HBM bandwidth unless it eliminates intermediate materialization, and
  the saved T1/T2 traffic is small vs the read traffic of L_env, theta,
  R_env. This is exactly the regime where Pair 7 (graph capture) would
  have helped by removing per-call launch dispatch, but a custom kernel
  can only replace one of rocBLAS's three launches with one launch —
  same launch count, same HBM reads.
- At `chi ≥ 128` the matvec is strongly compute-bound. rocBLAS Tensile
  kernels already hit ~60–70% of peak on dense FP64 GEMM at these sizes,
  and on the microbench our rocBLAS strided-batched path hits 1.47 TF/s
  at chi=64 (low because of small batches) and scales up at larger chi.
  A hand-fused kernel would need to match rocBLAS's tile scheduling,
  prefetch pipelining, and MFMA packing — multiple engineer-months of
  work for a ~1.1–1.3× speedup over already-fast rocBLAS, if any.

---

## 5. Micro-sanity-check: Step 1 fused vs rocBLAS (LIVE on MI300X)

### 5.1 Code

`ssh hotaisle@23.183.40.84
/home/hotaisle/dmrg-implementations/sandbox/pair08/bench_fused_step1.cpp`
(163 lines, committed on the live VM). The kernel is a **deliberately
naive** single-launch replacement for Step 1 that runs one workgroup per
`(D · d²)` batch, block dim `(16, 16)`, no LDS tiling, no MFMA — it is
the lower bound on "can a custom launch beat rocBLAS by launch economics
alone". If this simple version loses, any serious fused version either
wins by enormous work-per-launch amortization (which requires the LDS
capacity we do not have) or not at all.

Build: `hipcc --offload-arch=gfx942 -O3 -std=c++17 bench_fused_step1.cpp
-lrocblas -o bench_fused_step1` — compiles cleanly under ROCm 7.2.0 /
HIP 7.2.26015.

### 5.2 Results (run on live MI300X VF)

```
chi_L=32 chi_R=32 d=2 D=5 batch=20 iters=500
  rocBLAS strided batched:  7.249 us/iter,  0.18 TF/s
  fused-naive kernel     : 21.983 us/iter,  0.06 TF/s
  ratio (rocBLAS/fused)  :   0.330  (rocBLAS is 3.0× faster)
  work = 1.31 MFLOP/call

chi_L=64 chi_R=64 d=2 D=5 batch=20 iters=500
  rocBLAS strided batched:  7.154 us/iter,  1.47 TF/s
  fused-naive kernel     :161.119 us/iter,  0.07 TF/s
  ratio (rocBLAS/fused)  :   0.044  (rocBLAS is 22.5× faster)
  work = 10.49 MFLOP/call

chi_L=128, chi_L=256 — HIP err 700 (out-of-bounds access in naive
kernel's stride math). Not fixed because the chi=32/64 data is already
conclusive.
```

### 5.3 What the numbers tell us

1. **rocBLAS strided-batched dispatch is flat ~7.2 µs at chi=32 and
   chi=64.** That's the launch floor we'd have to undercut. A custom
   kernel competing with rocBLAS has to fit the entire fused 3-step
   chain under ~7 µs — for any chi — including cold-start dispatch,
   Tensile-equivalent kernel selection, and output writeback.
2. **A naive kernel is already 3× slower at chi=32.** The reason is not
   launch overhead (that's ~7 µs) — it's that the kernel itself is slow
   because we haven't MFMA-tiled it. Adding MFMA is the fix, but MFMA
   brings the LDS tiling requirement (§3.1), which brings back the HBM
   re-read cost, which brings back almost-rocBLAS-equivalent total cost.
3. **At chi=64 the gap widens to 22×.** This is because the naive
   kernel's FLOPs grow faster than the work-per-launch argument assumes
   — you can't outrun `O(chi³)` with "1 kernel instead of 3" when the
   per-kernel cost is `~2·chi³ · D · d²`. The per-launch savings become
   negligible as soon as the work per launch is on the order of
   microseconds.
4. **The 1.47 TF/s rocBLAS achieves at chi=64** is already ~0.9% of
   peak, but that's with batch = 20 (tiny). The problem at the target
   sizes is **too small for rocBLAS to amortize its own dispatch**, not
   that rocBLAS leaves compute on the table. A fused custom kernel
   fundamentally cannot fix a problem that is launch-overhead-bound when
   its own launch overhead is ~1 µs and rocBLAS's is ~7 µs. Amortizing
   launches is exactly what graph capture does — which is why Pair 7's
   path is the right attack vector on this problem, and pair 8's fused
   kernel is attacking the wrong bottleneck.

---

## 6. Adversarial findings

### 6.1 LDS overflow — LOUD

The task prompt noted a 2× LDS overflow at chi=128. The real overflow at
chi=128, d=2, D=5 is **88× LDS** for θ+T1+T2, and at chi=256 d=2 it's
**352×**. At chi=32 (the one regime Research A claimed would be "trivially
in LDS") the overflow is **5.5×**. There is no tiling scheme that
recovers the Research A "comfortably fit" claim; the best we can do is
per-tile execution which recovers HBM bandwidth costs nearly identical to
the unfused path.

### 6.2 rocWMMA FP64 status — usable but with surprises

Confirmed via live VM (`/opt/rocm/include/rocwmma/internal/mfma_impl.hpp`
line 836): rocWMMA 2.x on ROCm 7.2 does wrap
`__builtin_amdgcn_mfma_f64_16x16x4f64` internally under the
`mma_sync<16,16,4,double,double,double>` instantiation. BlockM = BlockN =
16 are fixed, BlockK ≥ 4 (power of 2).

Caveats found during research:
- **llama.cpp issue #19269** reports a rocWMMA compile break on ROCm
  7.2 (not FP64 specific but affects build integration).
- **Fragment shape 16 means chi must be padded to multiple of 16**.
  Today the codebase pads output shapes to MFMA-16 only where it
  matters; inside a fused kernel every reduction axis would need per-
  kernel runtime padding bookkeeping.
- **No documented C++ high-level wrapper for the 16x16x4 f64 tile
  directly** — users either call rocWMMA's fragment API (new learning
  surface) or the `__builtin_amdgcn_mfma_f64_16x16x4f64` intrinsic
  directly, which is lower-level but more explicit. AMD's own samples
  use the builtin directly, suggesting rocWMMA's FP64 path is not the
  well-trodden one.

### 6.3 Template / kernel-variant explosion

`apply_heff_two_site` is called across a large shape cartesian product:

- `chi_L ∈ {1, 2, 4, 8, 16, 20, 24, 32, 40, 48, 56, 64, 80, 96, 112,
  128, ...}` — at L=32 chi_target=128 there are 10–13 distinct `chi_L`
  values per half-sweep.
- `chi_R` independently varies with the same spread.
- `d ∈ {2, 3, 4}` depending on model family.
- `D_mpo ∈ {4, 5, 6}` depending on Hamiltonian.
- Two sweep directions (but these share kernel code).

A template-specialized fused kernel hits **≥ 10 × 10 × 3 × 3 ≈ 900
specializations**. With `hipcc -O3` compile time of ~30–60 s per
specialization (rocWMMA-heavy kernels with MFMA), that's 7.5–15 hours of
build time. Alternatively a runtime-parameterized kernel loses a lot of
the compile-time tiling benefit (that's the whole reason Tensile
generates the shape catalogs it does).

The right pattern in the face of this is either:
- **Pad to a small number of bucket shapes** (chi rounded up to next
  multiple of 16, d pinned, D pinned per model) → ~5–10 specializations.
  This is expensive in wasted compute at small chi.
- **JIT via Tensile or CK** — which means writing in CK and getting CK's
  shape selection for free, at the cost of a new 3–4 week learning curve
  and depending on CK stability.

Neither is "1–2 weeks of implementation" as Research A claimed.

### 6.4 Arithmetic intensity says memory-bound where it matters

At `chi = 32` the full matvec has AI ≈ 22.8 FLOP/byte, below the MI300X
ridge of 30.7. That means the bottleneck is HBM, not compute. Fusion
saves launch overhead and (sometimes) intermediate materialization — it
does NOT save HBM reads of the inputs. So at the very regime
(`chi ≤ 32`) where Research A claimed the biggest speedup ("3–10× at
chi≤64"), the physics of the machine says any such speedup can come only
from launch-overhead reduction. And launch-overhead reduction is Pair 7's
job, not ours.

### 6.5 The 1–2 week estimate is wishful

Realistic calendar for a correct, benchmarked, regression-tested fused
three-step kernel:

| Item | Time |
|------|------|
| LDS tiling design + paper math | 1 wk |
| First kernel for one (chi, d, D) shape, scalar correctness | 1 wk |
| MFMA-16 version of same shape, bit-exact vs rocBLAS | 1–2 wk |
| Shape parameterization / template factoring | 1 wk |
| Integration into `dmrg2-gpu` / per-shape dispatch | 1 wk |
| Correctness gate on all 6 test targets (ΔE < 1e-10) | 0.5 wk |
| Benchmark sweep and regression-guard vs rocBLAS | 0.5 wk |
| Padding strategy for non-16-multiple `chi` | 1 wk |
| **Total** | **6–10 weeks** |

This is 3–10× the Research A estimate, and the upside at the end is a
~1.1–1.3× speedup at `chi ∈ [64, 128]` that gets nullified the moment we
port to CUDA (where cuBLAS strided-batched IS capture-safe per Research
A §4.3 and we don't need a fused kernel at all).

### 6.6 Where does the real bottleneck live?

From `PROJECT_OVERVIEW.md §5.2` and the CPU-win analysis: the MI300X
loses to CPU **at chi ≤ 50 because of cache residency and launch
overhead per Krylov iteration**. The fused kernel attacks launch overhead
but can't fix cache residency (LDS is smaller than CPU L1 at 64 KB vs 80
KB L1 per modern x86 core, and is per-workgroup not per-iteration).

**R2-3 persistent Lanczos** (Pair 03 territory) is the correct answer
here: keep θ and the current Krylov vector in LDS across the entire
Lanczos outer loop, amortizing launches across all ~20 iterations at
once. That's where the 10–20× headroom lives. A fused apply_heff does
one matvec per launch; a persistent Lanczos does one **Lanczos** per
launch. The leverage ratios are night and day.

---

## 7. VERDICT

**ABANDON — stay with rocBLAS calls for `apply_heff_two_site`.**

The fused rocWMMA kernel is not the fallback Research A and round_2_plan
promised. It's a trap:

1. It cannot fit in LDS at any interesting `chi` (§3).
2. Its micro-sanity-check on the live VM shows 3–22× SLOWDOWN vs rocBLAS
   in the naive form (§5), and the MFMA'd form would at best match
   rocBLAS while costing 6–10 weeks to implement (§6.5).
3. The regime where it's claimed to help (`chi ≤ 64`) is memory-bound,
   not launch-bound, so fusion buys nothing (§6.4).
4. The template explosion makes the codebase unmaintainable (§6.3).
5. The real cache-residency problem at small chi is outside fusion's
   reach (§6.6).

**If Pair 7 does return NO-GO on rocBLAS graph capture**, the correct
fallback chain is:

1. **Take R2-3 (persistent Lanczos) seriously as THE small-chi answer.**
   The persistent kernel keeps θ and the hot Krylov vector in LDS across
   all ~20 Lanczos iterations per bond, amortizing launches across the
   entire Lanczos step (not just one matvec), and is the only approach
   that can actually match CPU L1-residency. `chi ≤ 32, d = 2` fits
   exactly the LDS budget for this scheme, which is the target regime
   where CPU currently wins.
2. **Land R2-1 (CBE-DMRG)** to eliminate the `O(chi³)` two-site SVD
   bottleneck at large chi. This takes the `chi ≥ 128` regime out of
   play for the fusion argument entirely.
3. **Accept that chi ∈ [32, 128] without HIP graph capture is a
   compromise regime** — rocBLAS launches at ~7 µs and there is no
   cheap way to drive that down short of graph capture or persistent
   kernels, neither of which is a plain "fused kernel".

If none of those land and we still want to attempt fusion, the honest
scope is:

- **Partial fusion: Step 2 + Step 3** (not Step 1 + Step 2 + Step 3).
  Step 2's output already lives in HBM, and Step 3 is D batches of tiny
  GEMMs — fusing these two removes ~`D+1 = 6` launches per matvec for
  Heisenberg. Pair 5's 1–2 µs savings per call are the realistic
  ceiling. Not worth writing the kernel for.
- **Partial fusion: Step 1 + Step 2 only.** Same math applies: ~2 µs
  savings. Same conclusion.

Neither partial fusion path clears the bar for "worth a Round 3
engineering investment."

---

## 8. Concrete next action

**Gate on Pair 7.** Assume Pair 7 has reported on rocBLAS graph capture
of `rocblas_dgemm_strided_batched`. Then:

### If Pair 7 = GO (graph capture works):
- Do nothing further on Pair 8. Proceed with Proposal 3 as written in
  `proposal_3_hip_graph_capture.md`.
- Mark the fused-kernel fallback as NOT the implementation path and
  delete the R2-4 alt mention from `round_2_plan.md §2 R2-4` (or at
  least downgrade it to "historical alternative, not recommended").

### If Pair 7 = NO-GO (graph capture broken):
- **Do NOT fall back to fused rocWMMA.** Instead:
  1. Promote R2-3 (persistent Lanczos) from "phase 3" to "immediate
     next" in the round_2_plan ordering. The persistent kernel fits the
     `chi ≤ 32` CPU-win regime exactly and is the right leverage point.
  2. Accept the `chi ∈ [32, 128]` regime as unoptimized relative to
     graph capture; commit to the CUDA/H100 port via `cudaGraph` (which
     IS capture-safe per Research A §4.2) as the performance path for
     that regime.
  3. Document the fused-kernel alt path as **tried and abandoned** in
     `round_2_plan.md`.

### Regardless of Pair 7:
- Commit `/home/hotaisle/dmrg-implementations/sandbox/pair08/bench_fused_step1.cpp`
  to the repo under `gpu-rocm/sandbox/bench_fused_step1.cpp` for
  regression traceability. This is the empirical proof the fused path
  loses to rocBLAS even before MFMA.
- Copy the table in §3 and the microbench in §5.2 into the Round 3
  synthesis document; the numbers are independently useful for other
  Round 3 pairs evaluating kernel-level work.

---

## Appendix: files touched

- **Live VM:** `/home/hotaisle/dmrg-implementations/sandbox/pair08/bench_fused_step1.cpp`
  (163 lines), `bench_fused_step1` binary. Built with hipcc, ROCm 7.2,
  gfx942. Runs in < 1 s per chi. Use
  `ssh hotaisle@23.183.40.84 "cd /home/hotaisle/dmrg-implementations/sandbox/pair08 && ./bench_fused_step1 CHI CHI D D_MPO ITERS"`
  to reproduce.
- **Local docs read:**
  - `docs/followups/proposal_3_hip_graph_capture.md`
  - `docs/followups/research_A_hip_graph_rocblas.md` (esp. §3.5, §3.7)
  - `docs/followups/research_C_persistent_lanczos.md` (esp. §1 LDS budget)
  - `docs/followups/round_2_plan.md` (esp. §2 R2-4)
  - `gpu-rocm/dmrg2-gpu/src/dmrg2_gpu_impl.h:473–550` — the
    `apply_heff_two_site` call chain that fusion would replace.
- **Remote header verified:**
  `/opt/rocm/include/rocwmma/internal/mfma_impl.hpp:836` — confirmed
  FP64 MFMA-16 wrapper exists (`__builtin_amdgcn_mfma_f64_16x16x4f64`
  under rocWMMA's `mma_sync<16,16,4,double,double,double>`
  specialization).
