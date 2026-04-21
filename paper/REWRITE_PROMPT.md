# Paper Rewrite Prompt: From Listicle to Cohesive Narrative

## Task

Rewrite `paper/main.tex` into a cohesive narrative. The paper has correct data and good content but reads as a catalog — it lists methods, then lists results, then lists failures. Your job is to **rewrite it as a story with a single argumentative thread** while preserving all data, tables, equations, and citations exactly as they are.

## The narrative arc

The paper tells a three-act story:

**Act 1 — The promise (Introduction + Methods):** GPUs offer massive BLAS-3 throughput. DMRG's inner loop is dominated by tensor contractions (GEMMs). The MI300X has 81.7 TFLOPS FP64 matrix throughput. This should be a perfect match. *Tension: but is it?*

**Act 2 — The straightforward port works... sometimes (Results §5.1–5.3):** A direct HIP/rocBLAS port of single-site and two-site DMRG delivers real speedups — but only when χ ≥ 128 AND L ≤ 20. As systems get larger or bond dimensions drop, the CPU wins. The crossover isn't a simple threshold — it's a 2D surface in (L, χ) space. Two-site DMRG converges 3.3× faster at L=100 due to adaptive bond growth. Parallel DMRG on a single GPU never wins. *Tension: can we push the crossover lower?*

**Act 3 — Everything we tried to fix it failed (Results §5.4 + Discussion):** We systematically attacked the bottleneck from every angle:
- Replace SVD with GEMM-heavy Newton–Schulz → 0% win rate, diverges at χ ≥ 128
- Replace Lanczos with BLAS-3 Davidson → slower, subspace overhead dominates
- Chebyshev filtering → 2–11× slower, wrong algorithm for single-eigenvalue problems
- Cross-segment batched GEMM → slower in 18/19 configs
- A2DMRG (additive Schwarz) → accurate only at small L, serial warmup makes parallel phase redundant

*Resolution:* Profiling reveals the root cause — CPU SVD is 97–98% of wall time at χ=256. None of our optimizations touch the bottleneck. The GPU's GEMM throughput is irrelevant when the dominant operation is a CPU factorization. This boundary (χ ≈ 256, moderate L) is where CPU linear algebra is simply too good to beat with iterative GPU methods.

## Critical insight to incorporate: Single-core CPU beats the MI300X

**This is a key finding that the current draft glosses over.** The paper currently says "we report the best wall time across all thread counts (1, 2, 4, 8, 12)" for CPU benchmarks. But analysis of the raw data reveals:

- Of 29 configurations where CPU beats GPU, **27 (93%) are won by single-threaded CPU alone**. Threading is not the reason the CPU wins.
- Only 2 configurations require multi-threading to beat the GPU:
  1. Heisenberg L=64 χ=128: 1-thread loses by 15%, 4-thread wins by 13%
  2. Josephson L=48 χ=128: 1-thread is 3.3× slower, 4-thread barely wins (d=5 makes matrices larger)
- At χ ≤ 50, a **single CPU core** outperforms the 304-CU MI300X for all medium-to-large systems
- 8 and 12 threads are actively harmful in 77% of runs (thread contention)
- 4 threads is the sweet spot but only provides 20-40% speedup at χ≥128

**This must be woven into the narrative.** The story becomes even starker: the GPU doesn't lose to parallelized CPU BLAS — it loses to kernel launch overhead at small matrix sizes. Each 50×50 GEMM completes in microseconds but costs 5–10 μs to launch. The MI300X's 81.7 TFLOPS is stranded behind a latency wall.

### Where to incorporate this:
1. **Results §5.1 (crossover)**: Add a paragraph or sub-table showing that the "best CPU" is almost always 1-thread at χ ≤ 50. Make clear the GPU is losing to a single core.
2. **Discussion §6.1 (why BLAS-3 replacements fail)**: Strengthen the "DMRG matrices are small by GPU standards" paragraph with the single-core finding — if the matrices are so small that even threading doesn't help, they're certainly too small for GPU parallelism.
3. **Discussion (new angle)**: The thread-scaling data tells its own story — 8+ threads hurt because of contention on small matrices. This is the same phenomenon that makes the GPU lose: too many execution units fighting over too little work.
4. **Conclusions**: The practitioner takeaway should note that at χ ≤ 50, even single-core CPU wins.

## Specific rewriting instructions

### 1. Introduction
Restructure as promise → gap → our contribution. Currently it lists all 11 implementations and all 4 failed optimizations upfront, spoiling the story. Instead:
- **Para 1**: DMRG context, MPS, computational cost (keep existing)
- **Para 2**: GPUs are natural candidates — GEMM throughput. MI300X specs. (keep existing, tighten)
- **Para 3**: The open question: "At what problem sizes does GPU acceleration become worthwhile? Can algorithmic modifications designed to increase GEMM utilization extend the favorable regime?" (reframe existing)
- **Para 4**: Our contribution — preview the answer honestly. "We find a narrow window of GPU advantage at large χ and small L, and show through systematic negative results why it cannot be widened at moderate bond dimensions. Remarkably, a single CPU core outperforms the MI300X at χ ≤ 50."
- **Para 5**: Brief roadmap (keep short)

Do NOT list all 4 failed optimizations in the introduction. Mention "several algorithmic modifications designed to exploit BLAS-3 throughput" and say they all failed — details belong in the methods and results.

### 2. Methods section
Keep all formalism but add *motivation sentences* that connect each method to the narrative:
- MPS/canonical forms → "The SVD-based truncation in canonical form maintenance will prove to be the dominant computational bottleneck (Section 6.2)."
- GPU implementation → The 3-step batched GEMM is the operation we hoped would be the GPU's strength. Explain *why* we structured it this way (maximize BLAS-3 fraction).
- Parallel DMRG → Motivated by hope that segment parallelism could amortize the per-bond overhead that limits serial GPU DMRG.

Each subsection should end with a forward pointer connecting it to the results.

### 3. Algorithmic variants (§3)
Reframe each variant as a **hypothesis under test**:
- Newton–Schulz: "Hypothesis: replacing the BLAS-2-dominated SVD with GEMM iterations should shift computation to the GPU's strength."
- Block-Davidson: "Hypothesis: BLAS-3 subspace projections should outperform BLAS-1/2 Lanczos iterations."
- Chebyshev: "Hypothesis: polynomial filtering can reduce eigenvalue iteration count."
- Cross-segment batched GEMM: "Hypothesis: batching across segments reduces launch overhead."
- A2DMRG: "Hypothesis: additive parallelism avoids the serial coupling bottleneck of real-space decomposition."

Each should close with a one-line foreshadowing: "Section 5.4 reports the outcome."

### 4. Results section
Reorganize by **narrative progression** (the data/tables stay the same, the prose changes):

**§5.1 — The baseline: where does GPU win?**
Present the crossover tables. Add the single-core insight: "Strikingly, in 27 of 29 CPU-win configurations (93%), the single-threaded CPU already outperforms the GPU — multi-threaded BLAS is not the cause of the GPU's loss." Frame the crossover as a 2D surface in (L, χ) space, not a simple threshold.

**§5.2 — The best GPU strategy: single-site vs two-site**
Practical recommendations. Connect to §5.1: "Given the narrow GPU-favorable regime, which GPU variant should a practitioner choose?"

**§5.3 — Can parallelism widen the regime?**
Parallel DMRG + A2DMRG results. Transition from §5.2: "Since single-GPU performance is limited by the crossover boundary, we investigated whether parallelism could help." Cover both Stoudenmire real-space decomposition and Grigori additive Schwarz. Neither helps.

**§5.4 — Can smarter algorithms widen the regime?**
All four GPU optimization variants. Transition: "Since parallelism did not help, we turned to algorithmic modifications designed to increase the fraction of computation performed as BLAS-3 operations." Present each failure with the hypothesis → result → why structure.

**§5.5 — The root cause: profiling the SVD bottleneck**
The profiling table (97-98% SVD). This is the *reveal* — the climax of the narrative. "These failures share a common explanation: the bottleneck is not where we assumed." Connect back to the single-core finding: the matrices are so small that neither GPU parallelism nor CPU threading helps — what matters is the serial factorization cost.

### 5. Discussion
Currently repeats results. Instead, use for **synthesis and generalization**:

**§6.1 — The small-matrix wall**: Unify the three failure modes (GPU launch overhead, threading overhead, iterative algorithm overhead) as the same phenomenon: DMRG at moderate χ produces matrices too small for parallelism of any kind. The single-core CPU finding is the starkest evidence. This is not specific to DMRG — any tensor network method at moderate bond dimension will hit this wall.

**§6.2 — The SVD ceiling**: Even if all non-SVD operations were infinitely fast, wall time drops by at most 2-3%. No amount of GEMM optimization can overcome this. The SVD is inherently sequential (QR iteration) and CPU LAPACK's implementation is extremely mature.

**§6.3 — The A2DMRG paradox**: The warmup-makes-it-redundant finding deserves its own discussion paragraph. It's a cautionary tale: additive parallel methods that require a good initial guess are in a chicken-and-egg trap — getting a good initial guess is equivalent to solving the problem.

**§6.4 — When would GPU DMRG win?**: χ ≥ 1024 (GEMM sizes reach GPU sweet spot), multi-GPU (each segment gets dedicated hardware), randomized SVD (when truncation rank ≪ χ), mixed precision.

### 6. Conclusions
Keep the numbered findings but:
- Add the single-core insight as a finding
- Add a practitioner "bottom line" paragraph: "For χ ≤ 50, use CPU quimb single-site DMRG — even a single core suffices. For χ ≥ 128 with L ≤ 64, use GPU single-site DMRG. Do not invest in BLAS-3 SVD replacements below χ ≈ 1024."
- End with the big-picture takeaway: "These results delineate a fundamental boundary — not specific to our implementation but to the problem structure — between regimes where massive GPU parallelism adds value and regimes where a single CPU core's low-latency sequential execution cannot be beaten."

## Hard constraints

- **Do NOT change any numbers, table data, equations, or citations.** The data is verified and correct.
- **Do NOT add new tables or remove existing ones.** You may rearrange their order to match the new narrative flow.
- **Do NOT add new sections beyond what's described above.** Reorganize within the existing structure.
- **Keep the total length approximately the same** (CPC target: ~15-20 pages). You're replacing catalog prose with narrative prose, not adding volume.
- **Preserve the elsarticle LaTeX format** and all existing packages.
- **Maintain the honest, no-hype tone.** This is a paper about negative results. The story is "we tried hard and learned why it didn't work," not "GPU DMRG is amazing."
- **Add transition sentences between subsections.** Every subsection should begin by connecting to the previous one and end by motivating the next.
- **The abstract is good as-is** — it already tells the story. Add one sentence about the single-core finding. Otherwise leave it.

## Test for cohesion

A reader should be able to read only the first and last sentence of each section and follow the entire argument. If you delete all tables and equations, the remaining text should tell a complete story. If it doesn't, add more connective tissue.
