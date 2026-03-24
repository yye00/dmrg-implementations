# Paper Draft Prompt: "When Does GPU DMRG Actually Pay Off?"

## Instructions

Write a research paper for submission to **Computer Physics Communications** (Elsevier). The paper should be written in a clear, direct scientific style — no hype, no overselling. Present the data honestly, including negative results. Use LaTeX formatting conventions (we will convert later). Target ~15-20 pages including figures and tables.

The paper reports on a systematic benchmarking study of 10 DMRG (Density Matrix Renormalization Group) implementations spanning Python/CPU to hand-tuned GPU (HIP/rocBLAS on AMD MI300X), tested across three physics models at varying system sizes and bond dimensions. The central finding is that GPU DMRG only pays off at high bond dimensions (chi >= 128), and that attempts to replace the SVD bottleneck with Newton-Schulz polar decomposition made performance worse at all tested sizes.

---

## Paper Structure

### Title Options (pick best)
1. "When Does GPU DMRG Actually Pay Off? A Systematic Benchmark of Ten Implementations on AMD MI300X"
2. "GPU-Accelerated DMRG: Crossover Analysis and the Newton-Schulz SVD Replacement That Didn't Work"
3. "From Python to GPU Kernels: Systematic Performance Analysis of DMRG Implementations"

### Abstract
- 10 DMRG implementations benchmarked: 4 CPU (quimb-dmrg1, quimb-dmrg2, pdmrg, a2dmrg), 6 GPU (dmrg-gpu, dmrg2-gpu, dmrg-gpu-opt, dmrg2-gpu-opt, pdmrg-gpu, pdmrg-gpu-opt)
- 3 physics models: Heisenberg spin chain (real, d=2), Josephson junction array (complex, d=5), transverse-field Ising model (real, d=2, critical point)
- System sizes L=8-128, bond dimensions chi=20-256
- Hardware: AMD Instinct MI300X (192GB HBM3, gfx942) vs multi-core CPU
- Key finding: GPU wins at chi >= 128 (74%) and chi=256 (100%), but Python+4-thread BLAS wins at chi <= 50
- Newton-Schulz + Block-Davidson optimization: 0 wins out of 50 converged comparisons (0%); diverges at chi>=128
- A2DMRG parallel approach: correct implementation but no practical speedup on single GPU

---

## The Implementations (Section 2)

### 2.1 CPU Reference Implementations

**quimb-dmrg1 / quimb-dmrg2**: Python library (quimb) using cotengra for optimal tensor contraction paths, backed by BLAS (OpenBLAS/MKL). Tested at 1, 2, 4, 8, and 12 threads. Single-site (dmrg1) and two-site (dmrg2) variants.

**pdmrg (Parallel DMRG)**: Our Python/numpy implementation of Stoudenmire & White (2013) real-space parallel DMRG. MPI-based, partitions chain into segments with boundary exchange via V = Λ⁻¹ (requires accurate SVD). Uses numpy tensordot for contractions.

**a2dmrg**: Our Python/numpy implementation of Grigori & Hassan (2025) additive two-level parallel DMRG with coarse-space correction. All sites updated independently, combined via coarse-space eigenvalue problem.

### 2.2 GPU Implementations (HIP/rocBLAS on MI300X)

All GPU implementations use:
- rocBLAS dgemm/zgemm for all tensor contractions (no CPU-side loops)
- Lanczos eigensolver on GPU (tridiagonal eigensolve on CPU — <0.01% runtime)
- rocsolver gesvd for SVD (GPU-side)
- Templates for double (real) and hipDoubleComplex (Josephson)
- Single HIP stream per implementation (except pdmrg-gpu which uses multiple streams)

**dmrg-gpu**: Single-site GPU DMRG. Baseline. All contractions via batched dgemm. The `apply_heff` matvec does 3-step GEMM: (1) D batched GEMMs with L_env, (2) dense GEMM with MPO W, (3) D batched GEMMs with R_env.

**dmrg2-gpu**: Two-site GPU DMRG. Extends single-site with fused MPO: precomputes WW[bond] = W_L ⊗ W_R, reducing apply_heff to same 3-step pattern with d→d². Converges in 2-3 sweeps vs 5-10 for single-site.

**dmrg-gpu-opt**: Single-site with Newton-Schulz polar decomposition replacing SVD, and Block-Davidson eigensolver replacing Lanczos. See Section 4.

**dmrg2-gpu-opt**: Two-site variant of the above.

**pdmrg-gpu**: GPU port of parallel DMRG. Partitions chain into segments on separate HIP streams.

**pdmrg-gpu-opt**: pdmrg-gpu with Newton-Schulz + Block-Davidson.

### 2.3 Implementation NOT included and why

**pdmrg2 (Python)**: Prototype with Newton-Schulz + Block-Davidson in Python. Not validated for benchmarking — served as algorithm development testbed before GPU port.

---

## The Physics Models (Section 3)

### 3.1 Heisenberg Spin Chain
```
H = Σᵢ (Sˣᵢ Sˣᵢ₊₁ + Sʸᵢ Sʸᵢ₊₁ + Sᶻᵢ Sᶻᵢ₊₁)
```
- d=2 (spin-1/2), real arithmetic (float64)
- Open boundary conditions
- D_MPO = 5
- Ground state energy per site → -0.4432 (Bethe ansatz, L→∞)
- Well-studied benchmark, moderate entanglement

**Why chosen:** The "hydrogen atom" of DMRG benchmarking — universally used, exact
solutions known, moderate entanglement growth makes it representative of the typical
DMRG use case. Every DMRG paper benchmarks against Heisenberg. The d=2 local dimension
means tensor contractions are small per site, stressing kernel launch overhead on GPU.
D_MPO=5 (largest of our three models) tests the MPO contraction path.

### 3.2 Josephson Junction Array
```
H = -Eⱼ/2 Σᵢ (e^{iφ_ext} e^{iφᵢ} e^{-iφᵢ₊₁} + h.c.) + Eᶜ Σᵢ nᵢ²
```
- d=5 (n_max=2, charge basis -2..+2), complex128 arithmetic
- External flux φ_ext = π/4 breaks time-reversal → complex MPO
- Parameters: Eⱼ=1.0, Eᶜ=0.5
- D_MPO = 3
- Tests complex arithmetic path, relevant to superconducting quantum computing

**Why chosen:** Forces the complex128 code path (hipDoubleComplex / zgemm), which is
fundamentally different from the real path — 4x the arithmetic per multiply, 2x the
memory bandwidth. This is the regime relevant to quantum computing applications
(superconducting circuits, Josephson junction arrays, quantum phase estimation).
The d=5 local dimension makes per-site tensors 2.5x larger than Heisenberg, shifting
the GEMM sizes and potentially changing the GPU crossover point. The external flux
breaks time-reversal symmetry, ensuring complex entries throughout (not just a real
problem in disguise).

### 3.3 Transverse-Field Ising Model (TFIM)
```
H = -J Σᵢ ZᵢZᵢ₊₁ - h Σᵢ Xᵢ
```
- d=2, real arithmetic (float64)
- h/J = 1.0 (critical point)
- D_MPO = 3
- Maximum entanglement at critical point — stress test for bond dimension

**Why chosen:** At the critical point h/J=1.0, the TFIM has logarithmic entanglement
scaling — the entanglement entropy grows as S ~ (c/6) log(L) with central charge c=1/2.
This means the bond dimension required for a given accuracy grows polynomially with L,
making it the hardest model for DMRG to converge. This stress-tests the eigensolver
(more Lanczos/Davidson iterations needed) and the SVD truncation (more singular values
near the cutoff). The D_MPO=3 (smallest of our models) means the MPO contraction is
cheap relative to the eigensolver, isolating the eigensolver performance difference
between implementations.

---

## The GPU Optimization Attempt: Newton-Schulz + Block-Davidson (Section 4)

### 4.1 Motivation
At high chi, GPU SVD (rocsolver gesvd) dominates wall time (measured at 97-99% for chi >= 256 in profiling). The SVD is inherently sequential (Householder + divide-and-conquer). The idea: replace it with iterative methods that use GPU GEMMs.

### 4.2 Newton-Schulz Iterative Polar Decomposition
Computes the polar factor A = UP where U is unitary and P is positive semi-definite:
```
U₀ = A / ||A||_F  (Frobenius norm initialization)
Uₖ₊₁ = ½ Uₖ(3I - UₖᴴUₖ)
```
Converges cubically. Typically 12-16 iterations to reach ||UᴴU - I||_F < 1e-10.

Each iteration = 2 GPU GEMMs (one for UᴴU, one for the update). Total: ~24-32 GEMMs per SVD replacement.

After convergence, compute singular values via eigendecomposition of P = AᴴU on CPU (small matrix).

**Fallback**: For matrices with k <= 4 or when NS fails to converge, fall back to CPU LAPACK SVD.

**Single-site adaptation**: Direction 'R' (left-to-right sweep): theta has shape (chi_L·d, chi_R) — tall, NS works. Direction 'L' (right-to-left): theta has shape (chi_L, d·chi_R) — wide, always falls back to SVD.

### 4.3 Block-Davidson Eigensolver
Replaces Lanczos (BLAS-2 heavy: matvec + orthogonalization via dgemv) with block subspace iteration:
- Block size b=4, max subspace 32
- Start with b random trial vectors
- Each iteration: apply H_eff to block (b matvecs), project onto subspace (GEMM), eigensolve projected H (dsyev on CPU), expand with residual corrections
- All projections use dgemm (BLAS-3)

**Fallback**: Lanczos when subspace dim <= 2b or eigendecomp fails.

### 4.4 Results: Complete Failure (0% Win Rate)

The opt implementations (Newton-Schulz + Block-Davidson) lose to the baselines (GPU SVD + Lanczos) in **every single comparison**:

**Win rates (opt vs baseline, converged runs only, 2026-03-23 re-run on MI300X):**
- 1-site: dmrg-gpu-opt wins **0/24** (0%), dmrg-gpu wins 24/24 (100%)
- 2-site: dmrg2-gpu-opt wins **0/26** (0%), dmrg2-gpu wins 26/26 (100%)

**pdmrg-gpu-opt vs best serial GPU (dmrg-gpu or dmrg2-gpu):**
- pdmrg-gpu-opt wins **0/23** (0%), always slower (0.07x to 0.73x of serial speed)
- pdmrg-gpu-opt does beat pdmrg-gpu (87% win rate), but this is misleading — pdmrg-gpu
  itself is far slower than serial GPU DMRG due to its broken coupling phase

**Numerical instability at high chi (critical finding):**
- chi<=50: NS converges correctly but 2-7x slower than rocsolver gesvd
- chi>=128: NS **diverges catastrophically** — produces garbage energies (e.g., -10²⁵)
  because Frobenius-norm-scaled iteration cannot converge when condition number is large
- chi>=256 L>=64: Crashes/segfaults from corrupted MPS tensors
- SVD fallback (U^H U - I verification) exists but triggers too late — once one bad
  SVD replacement corrupts an MPS tensor, error compounds exponentially

### 4.5 Why It Didn't Work
1. NS requires 12-16 iterations × 2 GEMMs = 24-32 GEMMs per truncation. rocsolver gesvd, while sequential, is heavily optimized for the MI300X and completes in fewer total FLOPs for chi <= 256.
2. The crossover where NS GEMMs become cheaper than gesvd requires chi >> 256 (estimated chi ~1000+), but NS diverges numerically before reaching that regime.
3. Block-Davidson's block-4 approach does more total matvecs than Lanczos (which converges in 10-50 iterations for ground state).
4. The overhead is multiplicative: slower eigensolver × slower truncation = ~2x+ penalty.
5. **The "Polar Express" adaptive NS variant** (degree-3 with tracked singular value intervals) was tried and reverted — adaptive scaling introduced worse instability than fixed-coefficient iteration.

### 4.6 Full opt vs baseline comparison table (2026-03-23 re-run, 192 configs)

Data from `benchmarks/paper_results/bench_opt_results.csv`. Only converged pairs shown (50/192).
**Every single comparison: baseline wins.**

```
Model           L  chi Type     Base(s)   Opt(s)  Speedup Winner
heisenberg     16   20 1-site       1.7      3.5    0.47x base
heisenberg     16   20 2-site       1.2      3.3    0.37x base
heisenberg     16   50 1-site       1.0      1.4    0.72x base
heisenberg     16   50 2-site       1.4      2.7    0.52x base
heisenberg     32   20 1-site       4.9     13.7    0.35x base
heisenberg     32   20 2-site       2.9     11.5    0.25x base
heisenberg     32   50 1-site       8.4     10.5    0.80x base
heisenberg     32   50 2-site       3.5      8.0    0.44x base
heisenberg     32  128 1-site       4.6      —      DIVERGED (E=-640)
heisenberg     32  256 1-site      11.8      —      DIVERGED (E=-4104)
heisenberg     64   20 1-site      14.0     41.9    0.33x base
heisenberg     64   20 2-site       8.1     37.3    0.22x base
heisenberg     64   50 1-site      23.0     10.6    ——    (opt NOT CONVERGED)
heisenberg     64  128 1-site      25.2      —      DIVERGED (E=-2×10⁶)
heisenberg     64  256 1-site      33.0      —      DIVERGED (E=-9×10²⁰)
heisenberg    128   20 1-site      13.7     80.4    0.17x base
heisenberg    128   20 2-site      27.0    143.8    0.19x base (but not converged)
heisenberg    128   50 1-site       9.7     35.2    0.27x base
heisenberg    128   50 2-site      71.7    143.1    0.50x base
josephson       8   20 1-site       1.1      2.5    0.45x base
josephson       8   20 2-site       0.8      2.1    0.37x base
josephson       8   50 1-site       0.6      1.5    0.42x base
josephson       8   50 2-site       0.7      1.9    0.36x base
josephson       8  128 1-site       0.6      1.1    0.57x base
josephson       8  128 2-site       0.7      1.7    0.40x base
josephson       8  256 1-site       0.7      1.1    0.63x base
josephson       8  256 2-site       0.7      1.7    0.39x base
josephson      16   50 1-site       3.9      9.2    0.43x base
josephson      16   50 2-site       2.9      8.4    0.35x base
josephson      16  256 1-site       4.2     24.3    0.17x base
josephson      16  256 2-site      15.8     98.4    0.16x base
josephson      32  128 1-site       8.3     42.0    0.20x base
josephson      32  128 2-site      34.3     99.3    0.34x base
josephson      32  256 1-site      18.4    104.6    0.18x base
josephson      64  128 1-site        —       —      BOTH FAIL at chi>=128
tfim           16   20 1-site       1.2      3.0    0.40x base
tfim           16   20 2-site       1.0      3.8    0.27x base
tfim           16   50 1-site       0.9      4.0    0.23x base
tfim           16   50 2-site       1.1      4.5    0.26x base
tfim           16  128 1-site       1.2      5.0    0.24x base
tfim           16  256 1-site       1.2      4.4    0.28x base
tfim           32   20 1-site       1.8      9.2    0.20x base
tfim           32   20 2-site       1.8      9.8    0.19x base
tfim           32   50 1-site       4.4      6.5    0.67x base
tfim           32   50 2-site       5.9     11.8    0.50x base
tfim           32  128 1-site       9.8     24.2    0.41x base
tfim           64   20 1-site       7.2     33.7    0.21x base
tfim           64   20 2-site       7.0     28.8    0.24x base
tfim           64   50 1-site       9.6     17.3    0.55x base
tfim           64   50 2-site      18.6     33.8    0.55x base
tfim          128   20 1-site      13.7     80.4    0.17x base
tfim          128   20 2-site      13.1     60.4    0.22x base
tfim          128   50 1-site       9.7     35.2    0.27x base
tfim          128   50 2-site      28.1     39.3    0.71x base
```

Note: chi>=128 opt entries that show DIVERGED had energies off by orders of magnitude
(Newton-Schulz numerical instability). chi>=256 L>=64 entries crashed outright (FAIL).
The previous benchmark run (pre-VM-reset) showed a few marginal opt wins at TFIM L=16
2-site and TFIM L=128 chi=128; the re-run with updated code confirms 0 wins.

---

## The A2DMRG Story (Section 5)

### 5.1 What is A2DMRG
Additive two-level parallel DMRG from Grigori & Hassan (arXiv:2505.23429v2, 2025). Instead of sequential sweeps, ALL sites are updated independently via local micro-steps, then combined via a coarse-space eigenvalue problem that finds the optimal linear combination.

### 5.2 What We Implemented
- Full Python/numpy implementation with MPI support
- O(L) incremental environment caching (fixed initial O(L²) bug)
- Pure numpy hot path (removed quimb overhead from sweep phase)
- Streaming orthogonal decompositions (peak memory O(χ²) not O(L·χ²))
- Configurable warmup sweeps (paper-faithful default: 0)
- Support for real and complex Hamiltonians

### 5.3 What Worked
- Correct for quantum chemistry systems with small d (paper's target domain)
- L=8 to L=20 Heisenberg: achieves < 1e-10 error
- Paper-faithful H6 hydrogen chain reproduction: matches quimb DMRG2 exactly

### 5.4 What Didn't Work
1. **No wall-clock speedup at P=2**: Algorithm designed for P ≈ L processors. With P=2, it's 2-4x slower than serial DMRG2.
2. **Accuracy degrades at large L**: TT-SVD compression ratio scales as (L-1):1. At L=32 chi=20, compressing 31×20 columns → 20 is a 31:1 ratio → massive truncation loss.
   - L=32, chi=50: ΔE = 2.7×10⁻⁷ (FAIL vs 1e-10 target)
   - L=48, chi=50: ΔE = 3.1×10⁻⁶ (FAIL)
   - L=64, chi=50: ΔE = 5.7×10⁻⁵ (FAIL)
3. **Complex Hamiltonians catastrophically fail**: Josephson junction L=16 chi=30: ΔE = 1.1×10³ (completely wrong)
4. **Warmup makes A2DMRG redundant**: Each warmup sweep buys 2-3 orders of magnitude in accuracy. At warmup=5, DMRG2 alone converges — the A2DMRG phase adds nothing.

### 5.5 Key Differences from Paper
| Aspect | Paper | Our Implementation |
|--------|-------|-------------------|
| Problem class | Quantum chemistry (d=12-24) | Spin chains (d=2-5, L=8-128) |
| Rank growth | r₀ → r over iterations (gentle compression) | Full χ from sweep 1 |
| Processors | P ≈ d−1 | P = 2 |
| Tolerance | 10⁻⁶ relative | 10⁻¹⁰ absolute |
| Speedup model | Per-processor FLOPs (theoretical) | Wall-clock time (actual) |

### 5.6 Lesson Learned
The algorithm is mathematically sound but designed for a specific regime (quantum chemistry with small d, many processors). Applying it to condensed matter spin chains with d=2 on a single GPU exposes the inherent additive Schwarz convergence penalty. It's faster to parallelize within tensor contractions (as done by Block2, ITensor) than across sites.

---

## Main Results: The Wins Table (Section 6)

### 6.1 Overall Win Rate

**Note:** The table below is from the original summary.csv benchmark (pre-VM-reset).
The pdmrg-gpu-opt "wins" are misleading — they beat pdmrg-gpu (broken parallel baseline)
but NOT dmrg-gpu or dmrg2-gpu. When compared against best serial GPU, pdmrg-gpu-opt
is 0/23 (always slower, 0.07x-0.73x). The corrected table should show 0 wins for all
opt and parallel variants.

| Implementation | Wins | Win % | Description |
|---|---|---|---|
| dmrg-gpu (D1-GPU) | 35 | 54.7% | 1-site GPU, rocsolver SVD |
| quimb-dmrg1 (Q-D1) | 17 | 26.6% | Python CPU, best of 1-12 threads |
| dmrg2-gpu (D2-GPU) | 9 | 14.1% | 2-site GPU, rocsolver SVD |
| quimb-dmrg2 (Q-D2) | 3 | 4.7% | Python CPU, best of 1-12 threads |
| All opt variants | 0 | 0% | dmrg-gpu-opt, dmrg2-gpu-opt, pdmrg-gpu-opt |
| All parallel variants | 0 | 0% | pdmrg-gpu, pdmrg, pdmrg2 |

64 total configs compared (converged runs only).

**The corrected 2026-03-23 re-run confirms: 0% win rate for ALL Newton-Schulz + Block-Davidson
variants across 50 converged head-to-head comparisons (24 for 1-site, 26 for 2-site).
Not a single configuration where the optimization helps.**

### 6.2 Wins by Bond Dimension

Note: PD-OPT wins from the original run are reassigned — when compared against serial
GPU (dmrg-gpu), pdmrg-gpu-opt loses every config. Corrected table:

| chi | D1-GPU | D2-GPU | Q-D1 | Q-D2 | Total |
|-----|--------|--------|------|------|-------|
| 20  | 4 (25%) | 6 (37.5%) | 6 (37.5%) | 0 | 16 |
| 50  | 7 (36.8%) | 1 (5.3%) | 9 (47.4%) | 2 (10.5%) | 19 |
| 128 | 14 (73.7%) | 2 (10.5%) | 2 (10.5%) | 1 (5.3%) | 19 |
| 256 | 10 (100%) | 0 | 0 | 0 | 10 |

### 6.3 Wins by Model

**Heisenberg** (24 configs): D1-GPU 45.8%, Q-D1 29.2%, D2-GPU 16.7%, Q-D2 8.3%
**Josephson** (17 configs): D1-GPU 58.8%, Q-D1 23.5%, D2-GPU 17.6%
**TFIM** (23 configs): D1-GPU 60.9%, Q-D1 26.1%, D2-GPU 8.7%, Q-D2 4.3%

### 6.4 Pairwise Head-to-Head Matrix (row beats column, 5% margin)

```
          D1-GPU  D2-GPU  D1-OPT  D2-OPT  PD-OPT    Q-D1    Q-D2
  D1-GPU      -- 39/60   35/37   35/36   37/43   17/38   21/36
               (65%)   (95%)   (97%)   (86%)   (45%)   (58%)
  D2-GPU 15/60      -- 23/36   37/41   42/48    8/38   19/36
         (25%)         (64%)   (90%)   (88%)   (21%)   (53%)
  D1-OPT  0/37 12/36       -- 27/34   18/25    4/15    7/15
          (0%)  (33%)         (79%)   (72%)   (27%)   (47%)
  D2-OPT  1/36  4/41    6/34       -- 12/31    1/17    6/17
          (3%)  (10%)   (18%)         (39%)    (6%)   (35%)
  PD-OPT  5/43  5/48    7/25   17/31       --  3/33    7/32
         (12%)  (10%)   (28%)   (55%)         ( 9%)   (22%)
    Q-D1 19/38 27/38   11/15   16/17   30/33       -- 28/36
         (50%)  (71%)   (73%)   (94%)   (91%)         (78%)
    Q-D2 11/36 17/36    8/15   10/17   22/32    3/36      --
         (31%)  (47%)   (53%)   (59%)   (69%)    (8%)
```

### 6.5 Geometric Mean Speedup vs quimb-dmrg2

| Implementation | Geo Mean | N configs |
|---|---|---|
| quimb-dmrg1 | 1.77x faster | 36 |
| dmrg-gpu | 1.58x faster | 36 |
| dmrg2-gpu | 1.37x faster | 36 |
| dmrg-gpu-opt | 1.28x faster | 15 |
| pdmrg-gpu-opt | 0.71x (slower) | 32 |
| dmrg2-gpu-opt | 0.73x (slower) | 17 |

### 6.6 Full Detailed Wins Table (all configs, all implementations, wall time in seconds)

Legend: D1=dmrg-gpu, D1o=dmrg-gpu-opt, D2=dmrg2-gpu, D2o=dmrg2-gpu-opt, PD=pdmrg-gpu, PDo=pdmrg-gpu-opt, Q1=quimb-dmrg1, Q2=quimb-dmrg2, Ppy=pdmrg(python), P2py=pdmrg2(python)

```
Model           L  chi Winner   WinTime   D1-GPU   D1-OPT   D2-GPU   D2-OPT   PD-GPU   PD-OPT     Q-D1     Q-D2    PD-PY   PD2-PY
heisenberg     12   20 Q-D1        0.8      1.0       --      0.9       --      6.1      1.2      0.8      1.1      1.5      1.7
heisenberg     12   50 D1-GPU      0.7      0.7       --      0.9       --      1.3      1.1      0.8      1.1      1.4      1.7
heisenberg     12  128 D1-GPU      0.7      0.7       --      0.9       --      1.2      1.1      0.8      1.0      1.4      1.8
heisenberg     16   20 D2-GPU      1.2      1.6      2.9      1.2      3.2       --       --       --       --       --       --
heisenberg     16   50 D1-GPU      1.0      1.0      1.4      1.4      2.6       --       --       --       --       --       --
heisenberg     16  128 D1-GPU      1.1      1.1      1.5      1.7      2.4       --       --       --       --       --       --
heisenberg     16  256 D1-GPU      1.4      1.4      1.4      1.6      2.5       --       --       --       --       --       --
heisenberg     20   20 Q-D1        1.1      2.3       --      1.5       --     15.0      2.2      1.1      1.1       --       --
heisenberg     20   50 Q-D1        1.1      1.5       --      1.9       --     24.7      3.6      1.1      1.4      9.8       --
heisenberg     20  128 D1-GPU      1.7      1.7       --      2.8       --    104.5      7.3      2.0      4.8     22.0       --
heisenberg     32   20 Q-D1        1.5      4.8     12.6      2.9     11.6      4.6      4.2      1.5      1.7       --       --
heisenberg     32   50 Q-D1        1.8      8.9      8.7      3.5      8.0      7.6      7.3      1.8      2.9     20.1       --
heisenberg     32  128 D1-GPU      4.6      4.6      5.8      6.2     11.4     23.1     22.7      5.2     15.2     68.3       --
heisenberg     32  256 D1-GPU     11.8     11.8     16.6     12.3     26.2     68.0     66.9       --       --       --       --
heisenberg     64   20 D2-GPU      8.4     14.2     50.5      8.4     37.2     12.3     11.4       --       --       --       --
heisenberg     64   50 Q-D2        5.1     22.8       --     10.4     30.5     21.4     20.6      5.7      5.1     61.9       --
heisenberg     64  128 Q-D1       20.6     24.9     30.3     24.1     38.6     67.4     65.4     20.6     37.7    259.0       --
heisenberg     64  256 D1-GPU     32.9     32.9     62.9     66.2    121.4    247.7    243.6       --       --       --       --
heisenberg    100   50 Q-D2        8.5     72.9       --     42.6       --       --       --     11.4      8.5    138.0       --
heisenberg    100  128 Q-D1       40.7    180.6       --     55.9       --       --       --     40.7     75.7    599.2       --
heisenberg    128   20 D2-GPU     26.2       --       --     26.2    100.7     34.7     27.1       --       --       --       --
heisenberg    128   50 PD-OPT     49.2       --       --     72.2    145.6     57.0     49.2       --       --       --       --
heisenberg    128  128 D2-GPU     94.2       --       --     94.2    192.4    174.9    167.1       --       --       --       --
heisenberg    128  256 D1-GPU    115.9    115.9    247.9    161.3       --       --       --       --       --       --       --
josephson       8   20 D2-GPU      0.8      1.3      2.6      0.8      2.0     46.8      3.6      1.5      1.3      3.2      8.3
josephson       8   50 D1-GPU      0.6      0.6      1.0      0.7      1.3     55.2      2.7      1.3      3.2      6.3     17.0
josephson       8  128 D1-GPU      0.6      0.6      1.0      0.7      1.3     26.5      6.8      3.6      6.5     19.4     57.6
josephson       8  256 D1-GPU      0.6      0.6      1.0      0.7      1.3       --       --       --       --       --       --
josephson      16   20 Q-D1        2.3      6.4       --      4.6     13.6      4.3      3.4      2.3      3.5     11.8     87.4
josephson      16   50 D2-GPU      2.9      3.7      8.2      2.9      8.8      5.7      5.2      3.7     16.3     32.2    114.9
josephson      16  128 D1-GPU      1.9      1.9      9.3      7.2     31.9     19.8     19.6     26.2    228.1    217.2       --
josephson      16  256 D1-GPU      4.3      4.3     31.8     15.3    152.5     45.6     44.9       --       --       --       --
josephson      32   20 PD-OPT      9.6       --       --     13.9     76.4     11.4      9.6       --       --       --       --
josephson      32   50 Q-D1       17.1     30.5       --     54.8       --     33.7     32.0     17.1     59.7    235.3   1144.4
josephson      32  128 D1-GPU      8.3      8.3     46.0     34.1    130.4     68.3     68.0    108.0    891.9       --       --
josephson      32  256 D1-GPU     18.4     18.4    132.7     69.9       --    181.8    180.0       --       --       --       --
josephson      48   50 Q-D1       31.0     59.8       --     89.8       --       --    296.9     31.0    360.5       --       --
josephson      48  128 D1-GPU    159.6    159.6       --    794.2       --       --       --    172.8       --       --       --
josephson      64   20 PD-OPT     29.1       --       --     44.7    298.7       --     29.1       --       --       --       --
josephson      64   50 Q-D1       41.6     82.3       --     85.4       --       --     91.4     41.6   1287.4       --       --
josephson      64  128 D2-GPU     85.1     96.1       --     85.1       --    188.9    189.8    300.8       --       --       --
josephson      64  256 D1-GPU     45.5     45.5       --    247.8       --       --       --       --       --       --       --
tfim           12   20 D1-GPU      0.7      0.7       --      0.7       --     13.2      1.7      0.8      0.8       --       --
tfim           12   50 D1-GPU      0.7      0.7       --      0.7       --      1.3      1.2      0.8      0.8       --       --
tfim           12  128 D1-GPU      0.7      0.7       --      0.8       --      1.2      1.2      0.8      0.8       --       --
tfim           16   20 D1-GPU      0.9      0.9      1.4      2.8      1.7       --       --       --       --       --       --
tfim           16   50 D1-GPU      0.9      0.9      1.4      3.0      1.6       --       --       --       --       --       --
tfim           16  128 D1-GPU      1.2      1.2      1.4      2.9      1.8       --       --       --       --       --       --
tfim           16  256 D1-GPU      1.3      1.3      3.6      1.3      1.6       --       --       --       --       --       --
tfim           20   20 Q-D1        0.9      1.0       --      1.2       --     14.7      2.1      0.9      1.0       --       --
tfim           20   50 Q-D1        1.0      1.3       --      1.5       --     27.1      3.3      1.0      1.2       --       --
tfim           20  128 Q-D2        1.9      1.9       --      3.1       --     86.2     62.9      1.9      1.9       --       --
tfim           32   20 Q-D1        1.1      1.6      4.7      1.8      5.7      3.5      3.5      1.1      1.3       --       --
tfim           32   50 Q-D1        1.6      2.1      4.0      3.6      7.0      7.5      7.4      1.6      1.8       --       --
tfim           32  128 D1-GPU      5.2      5.2      8.2     11.3     17.6     21.0     28.7      5.8      6.2       --       --
tfim           32  256 D1-GPU     11.8     11.8     21.1     33.5     69.9     87.3     89.7       --       --       --       --
tfim           64   20 D2-GPU      4.3      5.9     28.1      4.3     23.1      7.5      7.4       --       --       --       --
tfim           64   50 Q-D1        4.0      5.4     15.3      8.7     18.9     18.5     18.2      4.0      4.8       --       --
tfim           64  128 D1-GPU     18.3     18.3     28.3     43.3     60.3     84.2     81.2     22.1     26.6       --       --
tfim           64  256 D1-GPU     50.4     50.4     89.5    171.3    290.4       --       --       --       --       --       --
tfim          100   50 Q-D1        7.7      8.7       --     18.0       --       --       --      7.7      9.3       --       --
tfim          100  128 D1-GPU     36.4     36.4       --     88.4       --       --       --     40.8     55.3       --       --
tfim          128   20 D2-GPU     12.2     14.8     94.6     12.2     92.0     18.8     18.0       --       --       --       --
tfim          128   50 D1-GPU     16.1     16.1     37.8     19.6     42.7     42.3     40.8       --       --       --       --
tfim          128  128 D1-GPU     49.3     49.3     64.9    156.6    132.4    196.9    198.5       --       --       --       --
tfim          128  256 D1-GPU     85.0     85.0    192.0       --       --       --       --       --       --       --       --
```

---

## The quimb Thread Scaling Story (Section 7)

### 7.1 Finding: 4 threads is the sweet spot
quimb wins are NOT dominated by single-threaded runs:
- 4 threads: 12 wins (43%)
- 1 thread: 8 wins (29%)
- 2 threads: 3 wins
- 8 threads: 3 wins
- 12 threads: 2 wins

### 7.2 Thread contention disaster at 8+ threads
quimb-dmrg2 Heisenberg L=32 chi=50: 2.9s at 1 thread, **151.2s at 12 threads** (52x slower!)
Josephson L=32 chi=50: 104.4s at 1T, **1326.5s at 12T** (13x slower)

This is BLAS thread contention — the individual GEMMs at chi=50 are too small to benefit from 8+ threads, and the overhead of thread synchronization dominates.

### 7.3 Full quimb thread scaling data (selected configs)

```
Model           L  chi Impl               1T      2T      4T      8T     12T  Best
heisenberg     12   20 quimb-dmrg1      0.8     0.8     0.8     0.8     0.8   1T
heisenberg     20   50 quimb-dmrg1      1.1     1.1     1.1     1.1     1.1   8T
heisenberg     20  128 quimb-dmrg2      7.7     4.8     8.3    54.5   181.6   2T
heisenberg     32   50 quimb-dmrg2      2.9     3.2     5.1    43.8   151.2   1T
heisenberg     64  128 quimb-dmrg1     26.9    22.1    20.6    75.9   154.4   4T
heisenberg    100  128 quimb-dmrg1     49.3    43.8    40.7   166.6   311.8   4T
josephson       8  128 quimb-dmrg1      7.0     4.8     3.6    28.6    65.6   4T
josephson      16  128 quimb-dmrg1     72.8    43.0    26.2   213.7   485.9   4T
josephson      32   50 quimb-dmrg1     22.3    18.7    17.1   573.6  1524.7   4T
josephson      48   50 quimb-dmrg1     54.7    54.2    31.0  1025.7      --   4T
josephson      64   50 quimb-dmrg1     53.4    44.6    41.6  1315.5      --   4T
tfim           32  128 quimb-dmrg1      6.8     7.3     5.8      --      --   4T
tfim          100  128 quimb-dmrg1     46.4    42.9    40.8      --      --   4T
```

### 7.4 Why quimb wins at chi <= 50
- At chi=50, individual GEMMs are ~50×50 — fits entirely in L1/L2 cache on CPU
- GPU kernel launch overhead (~5-10μs per launch) × hundreds of launches per Lanczos iteration adds up
- HIP API overhead (hipMemcpy, hipStreamSynchronize) costs more than the actual computation
- quimb uses cotengra for optimal contraction paths — our GPU code uses a fixed (suboptimal) order
- The GPU advantage requires GEMMs large enough to saturate the MI300X compute units — this happens at chi >= 128

---

## Hardware Details (Section 8)

### GPU
- AMD Instinct MI300X (gfx942)
- 192 GB HBM3, ~5.3 TB/s memory bandwidth
- 1307 TFLOPS FP16, ~163 TFLOPS FP64
- ROCm 7.2, rocBLAS, rocSOLVER
- Single GPU (no multi-GPU runs)

### CPU (for quimb runs)
- The CPU on the MI300X VM (exact model to be confirmed — likely AMD EPYC)
- OpenBLAS (with AVX-512 or AVX2)
- Threads tested: 1, 2, 4, 8, 12

---

## Development Journey (Section 9 — optional, for narrative)

### Phase 1: CPU Implementation & Validation
- Implemented PDMRG (Stoudenmire & White 2013) in Python/numpy
- Validated against quimb reference (error < 1e-10)
- Studied contraction paths with cotengra: our manual order does 2.3-3.8x more FLOPs than optimal

### Phase 2: A2DMRG
- Implemented Grigori & Hassan (2025) additive two-level parallel DMRG
- Correct for small systems but doesn't scale to large L
- Complex Hamiltonians catastrophically fail
- Concluded: wrong tool for condensed matter spin chains

### Phase 3: GPU-Native DMRG
- Built dmrg-gpu (single-site) and dmrg2-gpu (two-site) from scratch in C++/HIP
- All contractions via rocBLAS dgemm/zgemm
- Templated for real (double) and complex (hipDoubleComplex)
- Discovered CPU LAPACK SVD was 2-6x faster than GPU rocsolver for chi < 200
- Switched to GPU rocsolver SVD after discovering OpenBLAS 0.3.20 SVD bug

### Phase 4: GPU Optimization Attempt (Failed)
- Ported Newton-Schulz polar decomposition from ML literature (claimed "1000x" SVD replacement)
- Ported Block-Davidson eigensolver to replace Lanczos
- Built 3 opt variants: dmrg-gpu-opt, dmrg2-gpu-opt, pdmrg-gpu-opt
- Also tried "Polar Express" adaptive NS (degree-3, tracked singular value intervals) — reverted, worse
- Result: **0% win rate across all 50 converged comparisons (re-run 2026-03-23)**
- At chi<=50: 2-7x slower (GEMM overhead). At chi>=128: numerically divergent (energies ~-10²⁵)
- rocsolver gesvd is too well-optimized, and NS is numerically unstable at scientific computing precision

### Phase 5: Comprehensive Benchmarking
- 192 GPU-only configs (opt vs baseline)
- 700+ total configs across all implementations
- 3 physics models × 10 implementations × multiple sizes
- Automated benchmark suite with timeout/resume on remote MI300X VM

---

## Key Lessons / Discussion Points (Section 10)

1. **The GPU crossover is at chi ~100-128**, not chi ~20 as often assumed. Below this, Python + 4-thread BLAS wins.

2. **Single-site GPU DMRG is the overall winner** — simpler algorithm, fewer sweeps needed for convergence at high chi (counter-intuitive: single-site does more sweeps but each is cheaper than two-site's d² factor).

3. **Newton-Schulz is not ready for scientific computing** — the ML community's enthusiasm (Keller & Bäuml, nanoGPT results) doesn't transfer to problems requiring 1e-10 precision. At float64 precision, NS needs ~12-16 iterations (vs ~3-5 at float16), and rocsolver SVD is highly optimized for this hardware. **Worse: NS is numerically unstable at chi>=128** — the Frobenius-norm-scaled iteration diverges when condition numbers are large, producing catastrophically wrong energies (~-10²⁵). This was confirmed in a full 192-config re-run (2026-03-23) that showed **0% win rate across 50 converged comparisons**. We also tried an adaptive "Polar Express" variant with tracked singular value intervals — it was even less stable and had to be reverted.

4. **Thread scaling is non-monotonic for BLAS-backed code** — 4 threads is optimal for chi <= 128 on this hardware. 8+ threads causes catastrophic slowdown due to thread contention on small matrices. Libraries should auto-tune this.

5. **Parallel DMRG on single GPU doesn't help** — the sequential nature of sweeps means stream-level parallelism can't overcome communication overhead. Multi-node is the path forward.

6. **The A2DMRG algorithm has a narrow regime of applicability** — it works for quantum chemistry (small d, many processors) but not for condensed matter (large L, d=2, few processors). This isn't an implementation failure; it's an inherent limitation of the additive Schwarz approach.

7. **OpenBLAS bugs matter** — we discovered OpenBLAS 0.3.20 has a broken SVD that silently produces wrong results. Fixed by building 0.3.28 from source. Numerical validation against known results is essential.

8. **Contraction path optimization matters** — our hand-coded GPU order does 2.3-3.8x more FLOPs than cotengra's optimal. At chi >= 200 this dominates. Future GPU DMRG should incorporate path optimization.

---

## Figures to Include

1. **Wall time vs chi** for each model (Heisenberg, Josephson, TFIM) at fixed L — shows crossover point
2. **Win rate pie chart** by bond dimension — shows GPU dominance at high chi
3. **NS opt vs baseline scatter** — log-log plot of baseline time vs opt time, with y=x line showing everything above it (opt slower)
4. **Thread scaling curves** for quimb — showing the contention cliff at 8+ threads
5. **Pairwise speedup heatmap** — implementation × implementation with color-coded win rates

---

## Reproducibility

- All code: https://github.com/yye00/dmrg-implementations
- Local repo: /home/captain/clawd/work/dmrg-implementations
- Hardware: AMD Instinct MI300X (gfx942), ROCm 7.2
- Convergence target: 1e-10 (dE between sweeps)
- Timeout: 300s per run
- quimb version: latest pip (tested with 1.8+)

### Benchmark result files (absolute paths)
- /home/captain/clawd/work/dmrg-implementations/benchmarks/paper_results/summary.csv — 737 rows, all 10 implementations, L=8-100, chi=20-128
- /home/captain/clawd/work/dmrg-implementations/benchmarks/paper_results/results.json — Full benchmark data (214 KB) with timing, energies, success status
- /home/captain/clawd/work/dmrg-implementations/benchmarks/paper_results/gpu_4way_results.csv — 248 rows, GPU 4-way (dmrg-gpu, dmrg2-gpu, pdmrg-gpu, pdmrg-gpu-opt), L=16-128, chi=20-256
- /home/captain/clawd/work/dmrg-implementations/benchmarks/paper_results/pdmrg1_rerun_results.csv — 88 rows, pdmrg-gpu rerun, L=16-128, chi=20-256
- /home/captain/clawd/work/dmrg-implementations/benchmarks/paper_results/gpu_bench_results.txt — Human-readable GPU results
- /home/captain/clawd/work/dmrg-implementations/benchmarks/paper_results/timing_heisenberg.png — Timing plot
- /home/captain/clawd/work/dmrg-implementations/benchmarks/paper_results/timing_josephson.png — Timing plot
- /home/captain/clawd/work/dmrg-implementations/benchmarks/paper_results/timing_tfim.png — Timing plot
- /home/captain/clawd/work/dmrg-implementations/benchmarks/paper_results/bench_opt_results.csv — 192 rows, opt vs baseline (dmrg-gpu, dmrg-gpu-opt, dmrg2-gpu, dmrg2-gpu-opt), L=16-128, chi=20-256, 3 models. Re-run 2026-03-23 on MI300X. Confirms 0% opt win rate.
- /home/captain/clawd/work/dmrg-implementations/benchmarks/paper_results/bench_opt_results.json — Same data in JSON
- /home/captain/clawd/work/dmrg-implementations/benchmarks/paper_results/a2dmrg_small_results.txt — A2DMRG (np=2) accuracy, Heisenberg L=8-20 chi=20-50
- /home/captain/clawd/work/dmrg-implementations/benchmarks/paper_results/a2dmrg_medium_results.txt — A2DMRG (np=2) accuracy, Heisenberg L=32-64 chi=50-100 + Josephson L=16-32 chi=30-50

### Source code (absolute paths)
- /home/captain/clawd/work/dmrg-implementations/dmrg-gpu/src/ — Single-site GPU DMRG baseline
- /home/captain/clawd/work/dmrg-implementations/dmrg2-gpu/src/ — Two-site GPU DMRG baseline
- /home/captain/clawd/work/dmrg-implementations/dmrg-gpu-opt/src/ — Single-site GPU DMRG with Newton-Schulz + Block-Davidson
- /home/captain/clawd/work/dmrg-implementations/dmrg2-gpu-opt/src/ — Two-site GPU DMRG with Newton-Schulz + Block-Davidson
- /home/captain/clawd/work/dmrg-implementations/pdmrg-gpu/src/ — Parallel GPU DMRG baseline
- /home/captain/clawd/work/dmrg-implementations/pdmrg-gpu-opt/src/ — Parallel GPU DMRG optimized
- /home/captain/clawd/work/dmrg-implementations/pdmrg/pdmrg/ — Python PDMRG (Stoudenmire & White)
- /home/captain/clawd/work/dmrg-implementations/a2dmrg/a2dmrg/ — Python A2DMRG (Grigori & Hassan)
- /home/captain/clawd/work/dmrg-implementations/benchmarks/lib/runners/quimb_runner.py — quimb DMRG runner

### A2DMRG analysis documents
- /home/captain/clawd/work/dmrg-implementations/a2dmrg/docs/a2dmrg-critical-analysis.md — 53 KB deep analysis
- /home/captain/clawd/work/dmrg-implementations/a2dmrg/bench_a2dmrg.py — Quick accuracy benchmark
- /home/captain/clawd/work/dmrg-implementations/a2dmrg/bench_medium.py — Medium-scale benchmark
- /home/captain/clawd/work/dmrg-implementations/docs/superpowers/plans/2026-03-19-a2dmrg-performance-rewrite.md — Performance rewrite plan

---

## Generating Plots from the Correct Data

All plots should be generated from the CSV files committed to GitHub (not from any
stale /tmp or remote files). Use only these authoritative data sources:

### Primary data files (absolute paths)

1. **Main 10-implementation comparison:**
   `/home/captain/clawd/work/dmrg-implementations/benchmarks/paper_results/summary.csv`
   Columns: impl, model, L, chi, threads_or_np_or_seg, energy, solve_time, wall_time, success

2. **Opt vs baseline (2026-03-23 re-run, authoritative for Newton-Schulz results):**
   `/home/captain/clawd/work/dmrg-implementations/benchmarks/paper_results/bench_opt_results.csv`
   Columns: model, impl, L, chi, energy, wall_time, sweeps, converged

3. **GPU 4-way scalability:**
   `/home/captain/clawd/work/dmrg-implementations/benchmarks/paper_results/gpu_4way_results.csv`

4. **A2DMRG accuracy (text format, parse manually):**
   `/home/captain/clawd/work/dmrg-implementations/benchmarks/paper_results/a2dmrg_small_results.txt`
   `/home/captain/clawd/work/dmrg-implementations/benchmarks/paper_results/a2dmrg_medium_results.txt`

### Plot 1: Wall time vs chi (crossover plot)
```python
# From summary.csv, for each model:
# x-axis: chi (20, 50, 128, 256)
# y-axis: wall time (log scale)
# Lines: dmrg-gpu, dmrg2-gpu, quimb-dmrg1 (best threads), quimb-dmrg2 (best threads)
# Fixed L (e.g., L=32 or L=64 — pick L with most data)
# Shows crossover where GPU lines cross below CPU lines
# For quimb, use best wall_time across all thread counts for each (model, L, chi)
```

### Plot 2: Opt vs baseline scatter (the failure plot)
```python
# From bench_opt_results.csv (2026-03-23 re-run):
# x-axis: baseline wall time (dmrg-gpu or dmrg2-gpu)
# y-axis: opt wall time (dmrg-gpu-opt or dmrg2-gpu-opt)
# Both log scale. y=x diagonal line.
# All points should be ABOVE the line (opt always slower)
# Color by model, shape by 1-site vs 2-site
# Only include converged=True pairs
# Annotate the diverged cases (chi>=128) as X marks or arrows off the top
```

### Plot 3: Win rate by chi (bar chart)
```python
# From summary.csv wins analysis:
# x-axis: chi values (20, 50, 128, 256)
# y-axis: win percentage
# Stacked or grouped bars for: dmrg-gpu, dmrg2-gpu, quimb-dmrg1, quimb-dmrg2
# Shows clean transition from CPU-wins to GPU-wins at chi~100-128
```

### Plot 4: Thread scaling curves
```python
# From summary.csv, filter quimb-dmrg1 and quimb-dmrg2:
# x-axis: threads (1, 2, 4, 8, 12)
# y-axis: wall time
# One subplot per (model, L, chi) showing the contention cliff at 8+ threads
# Pick 4-6 representative configs
```

### Plot 5: A2DMRG accuracy degradation
```python
# From a2dmrg_small_results.txt + a2dmrg_medium_results.txt:
# x-axis: L (8, 10, 12, 16, 20, 32, 48, 64)
# y-axis: |E_a2dmrg - E_ref| (log scale)
# Lines for chi=20 and chi=50
# Horizontal lines at 1e-10 (PASS), 1e-8 (WARN)
# Shows accuracy degradation with system size
```

### Plot 6: NS divergence energy trajectory (if raw sweep data available)
```python
# From bench_opt_results.csv, select diverged cases:
# Show final energy vs chi for opt implementations
# Correct energy (from baseline) as reference line
# Dramatically illustrates the divergence: correct at chi=50, garbage at chi=128
```

---

## Additional Findings to Include in Paper

### Finding: Single-site GPU DMRG is the surprise winner
Counter-intuitive: single-site DMRG (dmrg-gpu) outperforms two-site (dmrg2-gpu) at
high chi despite needing more sweeps. The reason: two-site optimizes over d² sites at
once, making each matvec d² times more expensive. At d=2, this is 4x. The extra sweeps
for single-site (10-16 vs 2-5) don't compensate for the 4x per-step cost when chi is
large enough that GEMM dominates. This finding contradicts conventional wisdom that
two-site DMRG is "always better" due to automatic bond dimension growth.

### Finding: CPU SVD was initially faster than GPU SVD
Early in development, CPU LAPACK SVD (dgesvd) was 2-6x faster than GPU rocsolver
for chi<200. We switched to GPU SVD only after discovering OpenBLAS 0.3.20 has a
broken SVD that silently produces wrong singular values (fixed in 0.3.28). This
OpenBLAS bug discovery is itself noteworthy — it means any DMRG code using the
system-default OpenBLAS on Ubuntu 22.04 may produce silently wrong results.

### Finding: Contraction path optimization matters more than hardware
Our hand-coded GPU contraction order does 2.3-3.8x more FLOPs than cotengra's optimal
path (measured via cotengra opt_einsum). At chi>=200, this FLOP penalty dominates wall
time. quimb uses cotengra internally, which partially explains its competitiveness.
Future GPU DMRG implementations should incorporate automatic path optimization, not
just brute-force the contractions.

### Finding: BLAS thread contention is a catastrophic failure mode
quimb at 12 threads is up to 52x slower than at 1 thread for small chi. This is not
gradual degradation — it's a cliff at 8 threads. The failure mode: OpenBLAS spawns
thread pools per GEMM call, and for small matrices (50x50), the thread synchronization
overhead exceeds the computation. Libraries like quimb should auto-detect this and
cap thread count based on matrix size.

### Finding: A2DMRG's narrow applicability window
The Grigori-Hassan A2DMRG algorithm works for quantum chemistry (d=12-24, P~d-1
processors) but fails for condensed matter (d=2, large L). The additive Schwarz
correction's convergence rate depends on the ratio P/L — at P=2, L=64, the coarse
space is too coarse. This is not an implementation deficiency; the algorithm's
mathematical convergence theory explicitly requires P~d-1. With d=2, you'd need
P=1, which is serial DMRG.

---

## References to Cite

- White, S. R. (1992). Density matrix formulation for quantum renormalization groups. PRL 69, 2863.
- White, S. R. (1993). Density-matrix algorithms for quantum renormalization groups. PRB 48, 10345.
- Stoudenmire, E. M., & White, S. R. (2013). Real-space parallel density matrix renormalization group. PRB 87, 155137.
- Grigori, L., & Hassan, M. (2025). An additive two-level parallel variant of the DMRG algorithm. arXiv:2505.23429v2.
- Keller, T., & Bäuml, M. (2024). Newton-Schulz iteration for matrix square roots. (ML context for NS SVD replacement)
- quimb: Gray, J. (2018). quimb: a python package for quantum information and many-body calculations. JOSS 3(29), 819.
- cotengra: Gray, J., & Kourtis, S. (2021). Hyper-optimized tensor network contraction. Quantum 5, 410.
- ITensor / Block2 / TeNPy references for context on other implementations
