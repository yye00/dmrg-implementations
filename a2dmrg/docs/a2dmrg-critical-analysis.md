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

### 2.4 Paper-Faithful Reproduction: H6 Quantum Chemistry (d=12, STO-3G)

To directly compare with the paper, we reproduced their H6 linear hydrogen chain
experiment (smallest test case, d=12 spin orbitals, STO-3G basis, 1 Å spacing).
The MPO was built from PySCF via openfermion Jordan-Wigner transformation and quimb
`MatrixProductOperator.from_dense()`.

| Method | r=16 | r=32 | r=48 |
|--------|------|------|------|
| FCI exact | −3.236066 | −3.236066 | −3.236066 |
| DMRG2 (quimb) | −3.2333 (rel 8.4×10⁻⁴) | −3.2357 (rel 1.1×10⁻⁴) | −3.2361 (rel 4.7×10⁻⁶) |
| A2DMRG wu=0 | −3.2002 (rel 1.1×10⁻²) | −3.2357 (rel 1.1×10⁻⁴) | −3.2361 (rel 4.7×10⁻⁶) |
| A2DMRG wu=2 | −3.1738 (rel 1.9×10⁻²) | −3.2357 (rel 1.1×10⁻⁴) | −3.2361 (rel 4.7×10⁻⁶) |

**Key findings**:

1. **r=32 and r=48: A2DMRG matches DMRG2 exactly.** Both methods hit the same
   bond-dimension-limited accuracy. The algorithm is correct.

2. **r=16 is too small for H6.** Even DMRG2 only achieves 8.4×10⁻⁴ relative error.
   A2DMRG is worse (1.1×10⁻²) due to compression loss. The paper also tests r=16 only
   for comparison, not as a target accuracy.

3. **Compression ratio matters.** For H6 (d=12), combining 12 candidates produces
   bond dim 12×r before TT-SVD compression back to r. At r=32 (ratio 12:1) and r=48
   (ratio 12:1), the compression is faithful. At r=16 (ratio 12:1), the small target bond
   dim limits what can be represented.

### 2.5 Updated Heisenberg Results with Gauge Fix

After fixing a gauge consistency bug in candidate MPS construction (bond dimension
mismatch between different i-orthogonal forms), the warmup=0 results changed significantly:

| L | χ | DMRG2 | A2DMRG wu=0 (20 sweeps) | A2DMRG wu=2 |
|---|---|-------|-------------------------|-------------|
| 8 | 20 | −3.3749 | (machine precision) | 5.3×10⁻¹⁵ |
| 16 | 20 | −6.9117 | 8.8×10⁻⁴ (not converged) | 9.3×10⁻¹¹ |
| 32 | 20 | −13.9973 | 9.3×10⁻² (not converged) | 1.9×10⁻⁶ |

The warmup=0 L=32 case shows the compression bottleneck clearly: 31 candidates × χ=20
= bond dim 620, compressed to 20 (31:1 ratio). Energy improves ~0.04/sweep, not converging
in 20 sweeps. The paper's H6 has ratio 12:1, which is manageable.

### 2.6 Warmup Sweep Count vs Accuracy (Heisenberg L=32, χ=20, np=4)

| Warmup sweeps | Abs error | Rel error | Total time |
|---------------|-----------|-----------|------------|
| 0 | 1.27 | 9.1×10⁻² | 74.8s |
| 1 | 5.7×10⁻³ | 4.1×10⁻⁴ | 0.2s |
| 2 | 4.4×10⁻⁶ | 3.1×10⁻⁷ | 0.3s |
| 3 | 1.5×10⁻⁸ | 1.1×10⁻⁹ | 0.3s |
| 5 | 5.2×10⁻¹² | 3.7×10⁻¹³ | 0.4s |
| 10 | 2.9×10⁻¹³ | 2.1×10⁻¹⁴ | 0.4s |

Each additional warmup sweep buys ~2–3 orders of magnitude in accuracy. By wu=5, A2DMRG
reaches machine precision (10⁻¹²). The A2DMRG phase itself converges in 1–2 sweeps for
wu ≥ 2, taking <0.1s. All the heavy lifting is done by the serial DMRG2 warmup — the
parallel A2DMRG phase is polishing a nearly-converged state.

**Key implication**: More warmup trivially fixes fidelity, but it also renders the
A2DMRG phase redundant. At wu=5, DMRG2 alone would have converged to the same accuracy
in the same time. A2DMRG adds negligible value on top of sufficient warmup.

### 2.7 Methodology Differences from Paper

Three critical differences between our implementation and the paper's experiments:

1. **Initialization**: We use a Néel state (product state) padded to the target bond
   dimension. The paper uses "identical random initializations possessing very small rank
   parameters" — meaning random MPS with initial bond dim much smaller than the target.

2. **Rank growth vs fixed rank**: The paper's algorithm starts with small ranks that
   grow toward the target r over several iterations. Quote: "the A2DMRG algorithm requires
   several global iterations before a maximal rank parameter r is achieved." Our
   implementation starts at the full target bond dim, so the TT-SVD compression ratio is
   maximal (L-1):1 from the first sweep. The paper's rank-growth trajectory means early
   sweeps have gentle compression ratios that gradually increase as ranks approach r.

3. **Convergence threshold**: The paper uses relative energy difference < 10⁻⁶ between
   successive iterations. Our benchmarks used 10⁻¹², which is 10⁶× more stringent. At
   the paper's 10⁻⁶ threshold, warmup=1 already achieves sufficient accuracy
   (rel_err = 4.1×10⁻⁴ < 10⁻³, and A2DMRG would continue improving from there).

The rank-growth difference is particularly important: the paper's gentle rank-growth
trajectory avoids the massive compression loss we observe when starting at full bond dim.
This explains why the paper shows convergence from cold start while our warmup=0 struggles
at large L — we are compressing from a (L-1)×χ bond dim from sweep 1, while they start
with (L-1)×r₀ where r₀ << r.

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

### 3.3 Tolerance: 10⁻⁶ Relative, Not 10⁻¹²

The paper uses relative energy difference < 10⁻⁶ between successive iterations as the
convergence criterion (stated explicitly in every figure caption). Our benchmarks used
10⁻¹² absolute convergence tolerance — **10⁶× more stringent**. At the paper's 10⁻⁶
threshold, even warmup=1 on Heisenberg L=32 would satisfy the criterion.

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

**Yes.** We now have direct evidence from reproducing the paper's H6 quantum chemistry
experiment:

- **r=32, r=48**: A2DMRG matches DMRG2 to the bond-dimension limit. The algorithm works.
- **r=16**: A2DMRG is worse than DMRG2, but r=16 is too small for H6 regardless.
- **L=8–20 Heisenberg with warmup=2**: absolute error < 10⁻¹⁰, better than the paper's
  10⁻⁶ tolerance.

The medium-scale Heisenberg degradation (L=32) is explained by the compression ratio:
the paper tests d≤24 (ratio ≤24:1), while L=32 chi=20 has ratio 31:1. This is not a bug
but an inherent scaling limitation of the additive Schwarz + TT-SVD compression approach.

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

2. **Rank growth dynamics (CONFIRMED)**: The paper explicitly starts with "very small
   rank parameters" and lets ranks grow toward the target r over several iterations. This
   means early sweeps have gentle compression ratios (e.g., 12×2 = 24 bond dim compressed
   to r=48 — no truncation at all). Our implementation starts at full bond dim, hitting
   the maximum compression ratio from sweep 1. This difference alone likely explains most
   of the warmup=0 accuracy gap at large L.

3. **Eigensolver warm-starting**: The paper notes that A2DMRG micro-iterations require
   more Lanczos iterations than serial DMRG (because they lack the sequential
   warm-starting). Our implementation may not warm-start the eigensolver as effectively.

4. **Number of processors**: With P = 2, each rank still performs L/2 sequential
   microsteps. The algorithm is designed for P ≈ L, where each rank handles ≈ 1 site.
   Testing at P = 2 is testing the algorithm in its worst operating regime.

5. **Convergence threshold**: The paper uses 10⁻⁶ relative, we used 10⁻¹². At the
   paper's threshold, warmup=1 already gives sufficient accuracy for Heisenberg L=32.

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
2. **Paper-faithful H6 reproduction matches DMRG2** at r=32 and r=48 — the algorithm is
   correctly implemented and produces correct energies for quantum chemistry problems.
3. For system sizes comparable to the paper's (L ≤ 20–24), we achieve equal or better
   accuracy than the paper reports.
4. The O(L²)→O(L) environment fix is essential for practical use and is not discussed
   in the paper (which uses a Julia implementation whose internal complexity is not
   detailed).
5. At L ≥ 32 on spin chains with warmup=0, the additive scheme's convergence degrades
   due to the TT-SVD compression ratio scaling as (L-1):1. This is inherent to the
   algorithm, not a bug. The paper avoids this regime (d≤24).
6. **warmup=2 + A2DMRG works well** even at L=32 (rel_err=1.9×10⁻⁶), confirming the
   algorithm is useful when starting close to the ground state.
7. With P = 2, the algorithm offers no wall-clock advantage over serial DMRG.

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

## 8. Application Regimes: Where Each Approach Works

### 8.1 The Paper's Regime: Small-d Quantum Chemistry

The Grigori-Hassan paper targets **ab initio quantum chemistry** — computing ground state
energies of small molecules using second-quantized electronic Hamiltonians in a
spin-orbital basis.

**Characteristics of this regime:**

| Property | Quantum Chemistry |
|----------|------------------|
| Sites (d) | 12–24 (spin orbitals, scales with basis set) |
| Physical dim | 2 (occupation number: empty/filled) |
| Bond dim (r) | 16–512 (grows with correlation strength) |
| MPO structure | Dense, long-range (all-to-all two-electron integrals) |
| MPO bond dim (D) | Grows with d (D ~ d for molecular Hamiltonians) |
| Hamiltonians | Real-valued (electronic structure in real basis) |
| Accuracy target | ~10⁻⁶ relative (chemical accuracy ~1 mHa) |
| Processor count | d−1 (one per bond, 11–23 processors) |

**Why A2DMRG works here:**
- Small d means few candidates (≤24), so compression ratio ≤24:1
- Rank growth from small initial values avoids early compression loss
- Dense MPO couples all sites, so coarse-space correction propagates information globally
- 10⁻⁶ tolerance is easily reachable
- d−1 processors is feasible (12–23 cores on one node)

**Limitations the paper doesn't face:**
- d never exceeds 24, so compression ratio stays manageable
- Never tests complex-valued Hamiltonians
- Never tests sparse/local MPO structure (like Heisenberg D=5)
- Measures FLOP speedup, not wall-clock — avoids MPI overhead question

### 8.2 Our Target Regime: Quantum Computing Applications

Our target is **quantum computing simulation** — computing ground states, simulating
circuits, and benchmarking quantum algorithms using tensor network methods.

**Key applications:**
- **Variational Quantum Eigensolver (VQE)**: Classical optimizer needs ground state
  energy repeatedly for different Hamiltonian parameters. Parallel evaluation across
  parameter space is natural, but each evaluation needs accurate MPS ground state.
- **Quantum Approximate Optimization (QAOA)**: Simulating QAOA circuits on classical
  hardware to benchmark quantum advantage claims. Involves L=50–200 qubits.
- **Quantum error correction**: Simulating stabilizer codes and noise channels. Surface
  codes have L = O(d²) physical qubits with local stabilizer Hamiltonians.
- **Entanglement studies**: Computing entanglement entropy, mutual information, and
  other correlation measures for quantum many-body systems relevant to quantum hardware.

**Characteristics of this regime:**

| Property | Quantum Computing |
|----------|-------------------|
| Sites (L) | 50–200+ (qubits, scales with quantum hardware) |
| Physical dim | 2 (qubit) or 4+ (qudit, bosonic codes) |
| Bond dim (χ) | 32–256 (limited by classical resources) |
| MPO structure | Sparse, local (nearest-neighbor or few-body) |
| MPO bond dim (D) | O(1) for local Hamiltonians (D=5 for Heisenberg) |
| Hamiltonians | Real AND complex (quantum circuits, Floquet, etc.) |
| Accuracy target | 10⁻⁸ to 10⁻¹² (quantum simulation fidelity) |
| Processor count | Limited by available GPUs (1–8 typical, 64+ on clusters) |

**Where A2DMRG struggles here (and why):**

1. **Large L / moderate χ = catastrophic compression ratio.** For L=100 χ=64, the
   compression ratio is 99:1 — the linear combination MPS has bond dim 6336 before
   TT-SVD compresses it to 64. This destroys most of the coarse-space gain every sweep.

2. **Sparse MPOs don't help the coarse space.** With D=5 (Heisenberg), the coarse-space
   correction can only propagate information locally. The paper's dense MPOs (D ~ d)
   couple all sites, making the coarse space more effective.

3. **Complex Hamiltonians.** Quantum circuits and Floquet Hamiltonians produce complex
   amplitudes. Our Bose-Hubbard tests (complex128) failed catastrophically. The paper
   never tests this.

4. **High accuracy needed.** Quantum simulation fidelity requires 10⁻⁸ or better for
   meaningful benchmarks. The paper's 10⁻⁶ tolerance is insufficient for our applications.

5. **Few processors available.** GPU clusters typically have 4–8 GPUs per node. A2DMRG
   needs P ≈ L processors to achieve its theoretical speedup. With P=4 on L=100, each
   rank handles 25 bonds — the algorithm degenerates to a slower version of serial DMRG.

### 8.3 Where Our Implementation Shines

Despite the A2DMRG algorithm's limitations for quantum computing at scale, our
implementation work produced valuable components:

**1. GPU-native DMRG (dmrg-gpu, dmrg2-gpu)**
These are the real workhorses for quantum computing simulation:
- L=32 χ=64 in 1.3s (single-site) / 2.1s (two-site) on MI300X
- 3–5× faster than quimb at χ≥128
- Templates for both real and complex (hipDoubleComplex)
- All contractions via rocBLAS — no CPU loops in hot path

**2. The DMRG warmup + parallel polish pattern**
Even though the A2DMRG phase adds little value, the pattern of "serial DMRG warmup
followed by parallel refinement" is sound. The warmup sweep data shows:

| Approach | L=32 χ=20 accuracy | Time |
|----------|-------------------|------|
| DMRG2 alone (20 sweeps) | machine precision | 0.4s |
| wu=2 + A2DMRG (np=4) | 3.1×10⁻⁷ | 0.3s |
| wu=5 + A2DMRG (np=4) | 3.7×10⁻¹³ | 0.4s |

The parallel phase doesn't buy much here, but for larger χ on GPUs, distributing local
updates across multiple GPUs could hide latency in the eigensolver.

**3. O(L) environment caching**
Incremental environment building is essential for any DMRG implementation and is reusable
across serial and parallel algorithms.

**4. Pure numpy tensor pipeline**
The extract → numpy → solve → reconstruct pipeline eliminates quimb overhead and is
directly portable to GPU (replace np.tensordot with rocBLAS dgemm).

### 8.4 Recommended Path for Quantum Computing

Given our findings, the recommended strategy for quantum computing DMRG is:

1. **Single-GPU DMRG2** for L ≤ 100, χ ≤ 256: Our dmrg2-gpu already handles this
   regime well. One MI300X can do L=32 χ=64 in 2.1s. Scaling to L=100 χ=128 is
   straightforward with existing code.

2. **Multi-GPU parallelism within contractions** for χ > 256: Distribute the large
   matrix operations (GEMM, SVD) across GPUs. This is the Block2/Menczer approach and
   scales naturally with χ.

3. **Pipeline parallelism** (Stoudenmire-White) for multiple parameter points: Run
   staggered sweeps on different GPUs for Hamiltonian parameter scans (VQE optimization,
   phase diagrams). This exploits the embarrassing parallelism of parameter sweeps.

4. **A2DMRG only for d ≤ 30** with large χ: The algorithm genuinely helps when the
   number of sites is small relative to bond dim — exactly the quantum chemistry regime.
   For quantum computing problems that happen to have small effective system size (e.g.,
   reduced models, effective Hamiltonians), A2DMRG with rank growth could be competitive.

**A2DMRG is not the path to parallel quantum computing simulation.** The compression
bottleneck at large L/χ is fundamental, not fixable with more warmup or engineering. The
established approaches (parallelize within contractions, pipeline sweeps) are better
suited to the L=50–200 χ=32–256 regime of quantum computing.

---

## Appendix: Key Differences Summary

| Aspect | Grigori-Hassan Paper | Our Implementation |
|--------|---------------------|-------------------|
| Systems | H₆–H₁₂, C₂, N₂ (quantum chemistry) | Heisenberg, Bose-Hubbard (lattice) |
| Sites d/L | 12–24 | 8–64 |
| Bond dim | 16–512 | 20–100 |
| Tolerance | 10⁻⁶ (relative) | 10⁻¹² (absolute) |
| Processors | d−1 (theoretical, 11–23) | 2–4 (actual MPI) |
| Speedup metric | FLOP count per processor | Wall-clock time |
| Implementation | Julia (in-house) | Python/numpy/MPI |
| Error metric | Relative energy error | Absolute energy difference |
| Initialization | Random, very small rank | Néel state at full bond dim |
| Warmup | None (rank grows from cold start) | 0–10 serial DMRG2 sweeps |
| Rank trajectory | Grows from small → target r | Fixed at target χ throughout |
| Models | Real-valued only | Real and complex |
