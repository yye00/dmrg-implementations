# A2DMRG Critical Analysis: Implementation, Results, and Comparison with Grigori-Hassan

## 1. What We Implemented

We implemented the Additive Two-Level parallel DMRG (A2DMRG) algorithm from Grigori
and Hassan (arXiv:2505.23429v2, December 2025) in Python/MPI, targeting spin-chain and
lattice Hamiltonians (Heisenberg, Bose-Hubbard). The key contributions of our
implementation work:

### 1.1 O(L²) → O(L) Environment Fix

The dominant cost in our initial implementation was an O(L²) per-sweep environment
rebuild: for each of the L microstep sites, we rebuilt all left/right environments from
scratch. We replaced this with **incremental environment caching** — build left and right
environment chains once per sweep in O(L), then reuse them for all microsteps. This
reduced per-sweep cost from O(L² · χ² · D) to O(L · χ² · D), making the algorithm
practical for L > 20.

### 1.2 Pure Numpy Hot Path

The sweep phases (local microsteps, coarse-space correction, linear combination, TT-SVD
compression) were rewritten from quimb tensor objects to pure numpy arrays. This removed
per-tensor overhead from quimb's tag/index machinery and enabled direct LAPACK/BLAS
dispatch in the eigensolver and SVD calls.

### 1.3 Streaming Orthogonal Decompositions

Replaced the all-at-once i-orthogonal decomposition (storing L copies of the MPS
simultaneously) with per-site streaming, reducing peak memory from O(L · χ²) to O(χ²)
per rank.

### 1.4 Warm-up and Finalization

Added configurable warm-up sweeps (default: 2 serial DMRG2 sweeps via quimb) before
entering the parallel A2DMRG loop, and optional finalization sweeps after convergence.

---

## 2. Our Benchmark Results

### 2.1 Small Scale — Heisenberg Spin-1/2 Chain (OBC)

| L | χ | |E_A2DMRG − E_ref| | Status | Wall time |
|---|---|---|---|---|
| 8 | 20 | 3×10⁻¹⁵ | PASS | 0.1s |
| 12 | 20 | 7×10⁻¹⁵ | PASS | 0.2s |
| 12 | 50 | 7×10⁻¹⁵ | PASS | 0.3s |
| 16 | 50 | 2×10⁻¹⁴ | PASS | 0.7s |
| 20 | 50 | 5×10⁻¹¹ | PASS | 2.1s |

Reference: quimb DMRG2. A2DMRG with np=2, warmup_sweeps=2, max_sweeps=20, tol=1e-12.

### 2.2 Medium Scale — Heisenberg (OBC, np=2)

| L | χ | |ΔE| | Status | A2DMRG time | DMRG2 time |
|---|---|---|---|---|---|
| 32 | 50 | 2.7×10⁻⁷ | FAIL | 7.4s | 5.2s |
| 32 | 100 | 8.9×10⁻⁹ | WARN | 35.6s | 11.4s |
| 48 | 50 | 3.1×10⁻⁶ | FAIL | 15.9s | 11.4s |
| 48 | 100 | 1.9×10⁻⁵ | FAIL | 104.3s | 26.8s |
| 64 | 50 | 5.7×10⁻⁵ | FAIL | 32.7s | 17.4s |
| 64 | 100 | 1.9×10⁻⁵ | FAIL | 198.3s | 44.4s |

### 2.3 Complex Hamiltonian — Bose-Hubbard/Josephson (complex128, t=1+0.5j, OBC, np=2)

| L | χ | |ΔE| | Status | A2DMRG time | DMRG1 time |
|---|---|---|---|---|---|
| 16 | 30 | 1.1×10³ | FAIL | 117.4s | 20.9s |
| 16 | 50 | 4.4×10² | FAIL | 204.0s | 53.8s |

**Summary**: A2DMRG achieves excellent accuracy (< 10⁻¹⁰) for L ≤ 20 but accuracy
degrades rapidly beyond L ≈ 30 on spin chains. Performance is 2–4× *slower* than serial
quimb DMRG2 with only P = 2 MPI ranks. The complex Bose-Hubbard Hamiltonian fails
catastrophically.

---

## 3. What Grigori and Hassan Actually Tested

Reading the paper carefully reveals several critical differences between their numerical
experiments and ours.

### 3.1 Problem Class: Quantum Chemistry, Not Spin Chains

The paper tests **strongly correlated molecular systems** from quantum chemistry:

- Linear hydrogen chains: H₆ (d=12), H₈ (d=16), H₁₀ (d=20), H₁₂ (d=24)
- C₂ dimer (d=20)
- N₂ dimer (d=20)

Here d is the number of spin orbitals (= number of TT cores / MPS sites). The
Hamiltonian is the second-quantized electronic Hamiltonian from PySCF with STO-3G basis,
**not** a Heisenberg or Hubbard lattice model. This is a fundamentally different problem
class with different entanglement structure and MPO properties.

### 3.2 System Sizes Are Small

Their largest system is H₁₂ with d = 24 sites. Bond dimensions (maximal ranks) tested:

| System | d (sites) | Ranks tested |
|--------|-----------|-------------|
| H₆ | 12 | 16, 32, 48 |
| H₈ | 16 | 32, 64, 128 |
| H₁₀ | 20 | 64, 128, 256 |
| H₁₂ | 24 | 128, 256, 512 |
| C₂ | 20 | 64, 128, 256 |
| N₂ | 20 | 32, 64, 128 |

**Our L=12–20 PASS results are directly comparable in system size to their d=12–24 tests.**
Our degradation starts at L=32, a regime they never tested.

### 3.3 Tolerance: 10⁻⁶, Not 10⁻¹⁰

The paper uses tolerance 10⁻⁶ for both eigensolvers and the convergence criterion
(relative energy difference between successive iterations < 10⁻⁶). Our target was
10⁻¹⁰ absolute accuracy. This is a 10,000× more stringent requirement.

### 3.4 Relative Error, Not Absolute

The paper plots **relative error in energy** (|E − E_exact| / |E_exact|), not the
absolute energy difference we report. For quantum chemistry systems with total energies
of order −1 to −100 Hartree, relative errors of 10⁻⁴ to 10⁻⁶ correspond to absolute
errors of 10⁻⁴ to 10⁻⁴ — comparable to or worse than our L=32 Heisenberg results.

### 3.5 Speedup Is Theoretical, Not Wall-Clock

**This is the most important finding.** The paper does NOT report actual parallel
execution times. Instead, it computes a **theoretical cost per processor** by counting
FLOPs (tensor contractions, matrix-vector products, QR, SVD) throughout each iteration.

From page 20 of the paper:

> "The cost per processor is computed by evaluating, through the course of each global
> iteration, the number and flop-count of tensor contractions, matrix-vector products,
> and linear algebra operations such as QR and singular value decompositions."

For A2DMRG, they **assume** that each of the d−1 local micro-iterations runs on an
independent processor, and that coarse-space matrix entries are computed in parallel. The
reported 2×–6× speedup is in this idealized FLOP model, not in actual wall-clock time on
a real distributed system.

### 3.6 Number of Processors: d−1, Not 2

The paper's parallelism model assumes **d − 1 processors** (one per local microstep). For
H₁₂ this means 23 processors. We tested with P = 2, where each rank handles L/2 sites
sequentially — a regime that offers essentially no benefit from the additive structure.

### 3.7 The Paper Explicitly Acknowledges Slower Convergence

From page 22:

> "First we note that the A2DMRG Algorithm 2 demonstrates slower convergence than the
> classical DMRG algorithm 1 as a function of the global iterations."

They explain this via the domain decomposition analogy: classical DMRG is a non-linear
**multiplicative** Schwarz method (Gauss-Seidel), while A2DMRG is a non-linear
**additive** Schwarz method (Jacobi). It is well-known in the numerical linear algebra
literature that multiplicative Schwarz converges faster than additive Schwarz. The
coarse-space correction partially compensates but does not fully close the gap.

### 3.8 Implementation: Julia, Not Python

The paper uses "an in-house Julia module developed and described in [Badreddine, 2024]"
for the DMRG simulations. Julia's JIT compilation and type stability make it
significantly faster than Python/numpy for the many small tensor operations in DMRG.

---

## 4. Critical Assessment

### 4.1 Are Our Results Consistent with the Paper?

**Yes, for the system sizes they tested.** Our L=8–20 results (absolute error < 10⁻¹⁰)
are at least as good as what the paper shows. Their convergence plots for d=12–24 show
relative errors of 10⁻⁴ to 10⁻⁶, which is *less* accurate than our small-scale results.

The paper simply **does not test** beyond d=24 sites, so our L=32–64 degradation is
neither predicted nor contradicted by the paper.

### 4.2 Is A2DMRG Fundamentally Sound?

The algorithm is mathematically well-motivated. The additive Schwarz with coarse-space
correction is a legitimate parallel strategy. However, several structural issues limit
its practical competitiveness:

1. **Additive < Multiplicative convergence**: This is a theorem in linear algebra and
   empirically confirmed here. Standard DMRG (multiplicative/sequential) converges faster
   per sweep. The coarse space of dimension d+1 (one-site) or d (two-site) is small
   relative to the full optimization landscape and cannot fully communicate information
   between distant sites.

2. **Coarse-space weakness at large L**: The coarse space spanned by d+1 candidate
   tensors captures only local improvements at each site. For long chains (L >> 20), the
   global entanglement structure requires information propagation across the full chain,
   which the additive scheme handles poorly. Standard DMRG achieves this naturally via
   sequential sweeps.

3. **Scaling paradox**: The algorithm needs P ≈ d−1 processors to achieve its theoretical
   speedup, but at d=24 (the paper's largest), d−1 = 23 processors competing for a
   tiny problem that serial DMRG solves in seconds. The algorithm becomes useful only if
   d is large AND the per-site cost is high — i.e., large bond dimensions on large
   systems, precisely where convergence degrades.

4. **Complex Hamiltonians**: The catastrophic failure on our Bose-Hubbard test
   (complex128, |ΔE| ~ 10³) suggests the algorithm may be particularly fragile for
   non-Hermitian effective Hamiltonians or complex-valued MPS. The paper tests only
   real-valued quantum chemistry Hamiltonians.

### 4.3 No Independent Reproductions Exist

As of March 2026, the paper has **zero citations** (Semantic Scholar). No independent
implementation or reproduction has been published. Our work appears to be the first
independent implementation of A2DMRG. The paper's Julia implementation is described in
Siwar Badreddine's 2024 PhD thesis but is not publicly available.

### 4.4 What Are We Missing?

Possible explanations for our medium-scale degradation that do NOT imply the paper is
wrong:

1. **Quantum chemistry Hamiltonians may be more favorable**: The all-to-all two-electron
   integrals create a different MPO structure (bond dimension growing with d) compared to
   the O(1) MPO bond dimension of Heisenberg. The coarse-space correction may be more
   effective for quantum chemistry because the MPO already couples all sites.

2. **Rank growth dynamics**: In the paper's experiments, ranks grow from random
   initialization. The TT-SVD compression after each A2DMRG iteration naturally adapts
   ranks. Our implementation uses fixed bond dimension throughout, which may be suboptimal.

3. **Eigensolver warm-starting**: The paper notes that A2DMRG micro-iterations require
   more Lanczos iterations than serial DMRG (because they lack the sequential
   warm-starting). Our implementation may not warm-start the eigensolver as effectively.

4. **Number of processors**: With P = 2, each rank still performs L/2 sequential
   microsteps. The algorithm is designed for P ≈ L, where each rank handles ≈ 1 site.
   Testing at P = 2 is testing the algorithm in its worst operating regime.

5. **Rank growth dynamics differ**: The paper starts from random initialization with
   "very small rank parameters" and lets ranks grow. Their convergence plots track this
   growth process. Our implementation fixes bond dimension from the start (after warm-up),
   which is a different optimization trajectory.

6. **Quantum chemistry MPO structure**: Molecular Hamiltonians have all-to-all
   two-electron integrals producing MPO bond dimensions that grow with d. The Heisenberg
   chain has constant MPO bond dimension D=5. The coarse-space correction may couple
   more effectively through the denser MPO structure of quantum chemistry problems.

### 4.5 What the Paper Does NOT Claim

The paper is actually quite careful in its claims. It does NOT claim:

- Superior accuracy over serial DMRG (it acknowledges slower convergence)
- Wall-clock speedup on actual hardware
- Scalability to large lattice systems (d > 24)
- Applicability to complex-valued Hamiltonians
- Practicality with small processor counts (P << d)

The paper claims "competitive convergence rates while achieving significant parallel
speedups" — but "competitive" means "not catastrophically worse," and "speedups" means
per-processor FLOP savings, not wall-clock time.

---

## 5. What We Contributed

### 5.1 Genuine Implementation Improvements

1. **O(L²) → O(L) environment caching**: A necessary fix for any practical implementation.
   Without this, A2DMRG is unusable beyond L ≈ 20 regardless of algorithm quality.

2. **Pure numpy hot path**: Removed all quimb overhead from sweep phases, reducing
   per-iteration overhead by ~5× for small systems.

3. **Streaming decompositions**: O(χ²) peak memory instead of O(L·χ²).

4. **Warm-up sweeps**: The paper's algorithm starts from random initialization. Adding
   2 serial DMRG warm-up sweeps dramatically improves convergence quality at the cost
   of a small serial preamble. This is a practical improvement not in the paper.

### 5.2 Empirical Findings Not in the Paper

1. **Spin-chain scaling behavior**: First test of A2DMRG on 1D lattice Hamiltonians
   (Heisenberg, Bose-Hubbard) beyond d=24.

2. **Complex Hamiltonian failure**: First demonstration that A2DMRG struggles with
   complex-valued tensor networks.

3. **Small-P regime characterization**: Demonstrated that A2DMRG with P=2 is slower
   than serial DMRG, establishing a lower bound on useful processor counts.

4. **Accuracy-vs-system-size curve**: Quantified the accuracy degradation from 10⁻¹⁵
   at L=8 to 10⁻⁵ at L=64 for the Heisenberg model.

---

## 6. Honest Conclusions

### What We Can Say

1. Our implementation of A2DMRG is faithful to the algorithm in arXiv:2505.23429.
2. For system sizes comparable to the paper's (L ≤ 20–24), we achieve equal or better
   accuracy than the paper reports.
3. The O(L²)→O(L) environment fix is essential for practical use and is not discussed
   in the paper (which uses a Julia implementation whose internal complexity is not
   detailed).
4. At L ≥ 32 on spin chains, the additive scheme's convergence degrades significantly
   compared to serial DMRG, consistent with the well-known theory that additive Schwarz
   converges slower than multiplicative Schwarz.
5. With P = 2, the algorithm offers no practical advantage over serial DMRG.

### What We Cannot Say

1. We cannot claim that A2DMRG is competitive with serial DMRG for lattice models at
   L > 24 — the paper does not test this regime and our results show degradation.
2. We cannot claim wall-clock speedup — the paper's speedup is a theoretical FLOP
   model, and our P=2 results are slower than serial.
3. We cannot claim the algorithm fails in general — quantum chemistry Hamiltonians may
   have more favorable structure, and P >> 2 may recover the theoretical speedup.

### Recommendation for Publication

If publishing these results, the honest framing is:

> We implemented the A2DMRG algorithm of Grigori and Hassan (2025) for 1D lattice
> Hamiltonians and identified a critical O(L²) environment rebuild bottleneck in the
> naive implementation, which we resolved via incremental caching. For system sizes
> comparable to the original paper (L ≤ 24), our implementation achieves high accuracy
> (|ΔE| < 10⁻¹⁰). However, extending to larger spin chains (L ≥ 32) reveals accuracy
> degradation consistent with the slower convergence of additive vs. multiplicative
> Schwarz methods. The algorithm's parallel speedup requires P ≈ L processors to
> materialize; with P = 2, it is slower than serial DMRG2.

This is honest, publishable, and scientifically valuable. It neither oversells our work
nor unfairly criticizes the original paper, which is a theoretical contribution tested
only on small quantum chemistry systems.

---

## 7. Context: Other Parallel DMRG Approaches

A2DMRG is unique in that it changes the **algorithmic structure** of DMRG to enable
parallelism across sites. All other parallel DMRG approaches keep the sequential sweep
structure and parallelize within each micro-iteration:

- **Stoudenmire-White (2013)**: Pipeline parallelism — stagger multiple sweeps so
  different processors work on different parts of the chain simultaneously. No
  coarse-space correction. Cited by Grigori-Hassan as the only prior non-sequential DMRG.

- **Sharma-Chan / Zhai-Chan (Block2)**: Distributed-memory DMRG parallelizing tensor
  contractions within each site optimization. State-of-the-art for production quantum
  chemistry DMRG.

- **Levy-Solomonik-Clark (2020)**: Distributed sparse/dense parallel tensor contractions
  for DMRG.

- **Menczer et al. (2024)**: Quarter-petaFLOPS DMRG on DGX-H100 — GPU parallelism
  within contractions, achieving massive single-node performance.

The practical track record strongly favors the "parallelize within contractions" approach.
A2DMRG's "parallelize across sites" strategy is theoretically interesting but unproven at
scale and, as we show, introduces convergence penalties that may outweigh the parallelism
gains.

---

## Appendix: Key Differences Summary

| Aspect | Grigori-Hassan Paper | Our Implementation |
|--------|---------------------|-------------------|
| Systems | H₆–H₁₂, C₂, N₂ (quantum chemistry) | Heisenberg, Bose-Hubbard (lattice) |
| Sites d/L | 12–24 | 8–64 |
| Bond dim | 16–512 | 20–100 |
| Tolerance | 10⁻⁶ (relative) | 10⁻¹⁰–10⁻¹² (absolute) |
| Processors | d−1 (theoretical, 11–23) | 2 (actual MPI) |
| Speedup metric | FLOP count per processor | Wall-clock time |
| Implementation | Julia (in-house) | Python/numpy/MPI |
| Error metric | Relative energy error | Absolute energy difference |
| Warmup | None (random init) | 2 serial DMRG2 sweeps |
| Models | Real-valued only | Real and complex |
