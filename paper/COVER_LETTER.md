# Cover Letter — Computer Physics Communications

**Manuscript:** GPU-Accelerated DMRG on AMD MI300X: Systematic Benchmarking and Lessons from Failed Optimizations

**Author:** Yaakoub El Khamra (corresponding)

**Submitted to:** Computer Physics Communications (Program Library)

---

Dear Editors,

We are submitting the enclosed manuscript for consideration at *Computer Physics
Communications*. It is a systematic benchmarking study of eleven density matrix
renormalization group (DMRG) implementations — spanning single-site and two-site
DMRG, real-space parallel DMRG, and an additive two-level variant (A2DMRG) — on
the AMD Instinct MI300X GPU, tested across three physics models (Heisenberg
chain, Josephson-junction array, transverse-field Ising) and a range of system
sizes and bond dimensions.

The manuscript is submitted to the CPC Program Library track. The associated
code repository is publicly available at
https://github.com/yye00/dmrg-implementations under an MIT license, and every
result JSON in the repository embeds the exact git commit hash, binary SHA-256,
runtime CLI, and environment-variable snapshot that produced it.

### Why CPC

The manuscript is a benchmark-plus-implementation study with the following
characteristics that match the CPC audience:

1. **Publicly released code**, eleven DMRG variants, with build instructions
   for ROCm 7.2 on MI300X (gfx942).
2. **Systematic negative results** on several algorithmic hypotheses
   (Newton-Schulz polar decomposition for SVD replacement; Block-Davidson
   eigensolver; Chebyshev-filtered subspace iteration; cross-segment batched
   GEMM dispatch; A2DMRG additive parallelism). Each hypothesis is documented
   and disproved with its own benchmark evidence.
3. **Two narrowly-applicable positive results** on GPU micro-optimizations
   (randomized SVD in two-site DMRG: 5-6x; HIP graph capture in segment-
   parallel DMRG: 1.2-1.8x). Both wins target the locally-binding bottleneck
   of their variant, establishing a transferable pattern for practitioners.
4. **Practitioner bottom-line** for the moderate bond-dimension regime
   (chi ≤ 256): when to use GPU DMRG, when to use single-threaded CPU BLAS,
   and which algorithmic "GPU optimizations" do not pay off.

### Key findings

- At chi = 128, GPU wins 100% of short-chain (L ≤ 20) configurations but only
  50% at L ≥ 100. At chi ≤ 50, the CPU wins *all* medium-to-large systems.
- In 93% of cases where the CPU wins, **a single CPU core outperforms the
  304-compute-unit MI300X**. The GPU loses not to parallelized CPU BLAS, but
  to kernel launch latency on matrices small enough to fit in L2 cache.
- CPU LAPACK SVD consumes 97-98% of per-sweep wall time at chi = 256, imposing
  a ceiling no eigensolver or environment-update optimization can breach.
- Newton-Schulz polar decomposition achieves a 0% win rate across 50
  configurations and diverges numerically at chi ≥ 128. Chebyshev filtering
  is 1.9-11x slower. Cross-segment batched GEMM is slower in 18/19 configs.
- Of six GPU micro-optimizations tested uniformly across all variants, only
  two deliver robust wins — and only in the specific variants where they
  target the measured bottleneck.

### Suggested reviewers

- (Please add 2-3 DMRG/GPU-computing researchers with no recent conflicts.)

### Conflicts and prior publication

The manuscript has not been submitted elsewhere and is not under consideration
by another journal. All authors have read and approved the manuscript. We
declare no competing interests. See Declaration of competing interest in the
manuscript for the formal statement.

### Reproducibility statement

Every benchmark result JSON includes the git commit SHA, binary SHA-256,
runtime arguments, and the relevant environment variable snapshot. The
repository build (including a fixed OpenBLAS 0.3.28 workaround for a known
singular-vector bug in 0.3.20) is documented. An ablation harness with
correctness gate (|Delta E| < 5e-10 vs. flags-off baseline) is included and
runs on any rebuilt binary.

Thank you for your consideration.

Sincerely,

Yaakoub El Khamra
