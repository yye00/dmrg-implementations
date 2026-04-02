# Adversarial Reviewer Prompt

You are a hostile but fair CPC referee tasked with verifying every factual claim in the paper `paper/main.tex` against the actual codebase, benchmark data, and published literature. Your job is to find **lies, exaggerations, mischaracterizations, and unsupported claims**. You are not reviewing writing quality — only factual accuracy.

## Your mandate

For every verifiable claim in the paper, do ONE of:
1. **VERIFIED** — you checked the code/data and the claim is correct
2. **WRONG** — the claim contradicts the code/data (give evidence)
3. **UNSUPPORTED** — the claim could be true but you found no evidence for it
4. **MISLEADING** — technically true but presented in a way that gives the wrong impression

## What to check

### A. Algorithm descriptions vs. code

For each implementation described in the paper, read the actual source code and verify:

- **Sweep types**: Does each phase (warmup, segment, coupling, polish) use single-site or two-site as claimed?
  - Check: `pdmrg-gpu/src/pdmrg_gpu_impl.h` — the `run()` method
  - Check: `pdmrg-gpu-opt/src/pdmrg_gpu_opt_impl.h` — the `run()` method
  - Check: `pdmrg/pdmrg/dmrg.py` — `pdmrg_main()` and `serial_warmup()`

- **apply_heff structure**: Is the 3-step GEMM pattern (Step 1: L_env batched, Step 2: dense WW, Step 3: R_env batched) accurately described?
  - Check: `dmrg-gpu/src/dmrg_gpu_impl.h` or `dmrg2-gpu/src/dmrg2_gpu_impl.h`

- **Newton-Schulz iteration**: Is the recurrence X_{k+1} = 0.5 * X_k * (3I - X_k^H X_k) correctly stated? How many iterations does the code actually use? Does it match "5-10 iterations" or "12-16 iterations" (the paper and prompt disagree)?
  - Check: `dmrg-gpu-opt/src/dmrg_gpu_opt_impl.h` — search for `newton_schulz` or `ns_`

- **Block-Davidson**: Is block size really 4? Max subspace really 32?
  - Check: `dmrg-gpu-opt/src/` or `pdmrg-gpu-opt/src/`

- **Chebyshev**: Is degree really 15? Outer iterations really 20? Does the 3-term recurrence match what's written?
  - Check: `pdmrg-gpu-opt/src/pdmrg_gpu_opt_impl.h` — search for `chebyshev`

- **Lanczos convergence**: Paper claims "10-50 iterations" and "~15 iterations". Which is it? What does the code default to?
  - Check: Lanczos max iterations in any GPU impl

- **SVD**: Paper claims CPU LAPACK `dgesvd` is default. Verify this. Paper claims GPU SVD is available via `--gpu-svd`. Verify.
  - Check: SVD code path in `dmrg2-gpu/src/`

- **Segment sweeps**: Paper says "staggered directions" — verify even segments go LR while odd go RL (or whatever the actual pattern is).

- **Fused MPO**: Paper says WW[b] = W_L ⊗ W_R is precomputed. Verify this tensor product is computed once, not per-sweep.

### B. Numerical claims vs. benchmark data

Load the actual data files and verify every number cited in the paper:

```python
import json, csv

# Main dataset
results = json.load(open("benchmarks/paper_results/results.json"))

# GPU opt study
gpu_opt = json.load(open("benchmarks/paper_results/gpu_opt_bench.json"))
```

Check these specific claims:

- **"74% win rate at chi=128"** — compute actual win rate from data
- **"100% win rate at chi=256"** — verify
- **"dmrg2-gpu 3.3x faster than dmrg-gpu at L=100 chi=128 (53s vs 177s)"** — find these exact entries
- **"0% win rate for Newton-Schulz across 50 configurations"** — verify from `bench_opt_results.csv`
- **"Chebyshev 1.9-11x slower"** — verify the two data points (L=8 chi=32: 0.73s vs 1.40s; L=20 chi=50: 5.27s vs 59.15s)
- **"Batched sweep: 1/19 configs faster, only chi=256 seg=2 at 1.18x"** — verify from gpu_opt_bench.json
- **"CPU SVD = 97-98% of per-sweep runtime at chi>=128"** — this comes from profiling in `pdmrg-gpu/OPTIMIZATION_REPORT.md`, not from benchmark JSON. Verify the report says this.
- **Table 1 (crossover win rates)**: Recompute from data
- **Table 2 (serial GPU times)**: Verify each number
- **Table 3 (parallel GPU times)**: Verify each number
- **Table 4 (opt vs baseline)**: Verify each number
- **Table 5 (Chebyshev)**: Verify each number
- **Table 6 (batched sweep)**: Verify each number
- **Table 7 (profiling breakdown)**: Verify against optimization report

### C. Literature claims

Use web search to verify:

- **"Bethe ansatz ground state energy density e_0 = 1/4 - ln(2) ≈ -0.4432"** — is this per bond or per site? The standard result is E/L → 1/4 - ln(2) for the Heisenberg antiferromagnet. Verify the sign and whether it's per site or per bond.
- **"TFIM central charge c = 1/2"** — verify for h/J = 1.0 critical point
- **"MI300X: 304 CUs, 192 GB HBM3, 5.3 TB/s bandwidth, 81.7 TFLOPS FP64 matrix"** — verify against AMD specs. Are these the right numbers for the MI300X (not MI300A or MI250X)?
- **"Newton-Schulz converges cubically"** — verify this is the correct convergence rate for the standard iteration
- **"Lanczos converges in O(sqrt(kappa)) iterations"** — this is a common claim but verify it's stated correctly (it's O(sqrt(kappa)) for the eigenvalue gap, not the condition number per se)
- **"CheFSI designed for DFT" citing Zhou et al 2006** — verify this is the correct reference
- **Stoudenmire & White 2013 for real-space parallel DMRG** — verify this is the correct reference

### D. Consistency checks

- Does the paper cite "10 implementations" in the abstract? Count them. Are there really 10?
- Does the abstract say "712 configurations"? The conclusion says "712". The data files have 604 + 108 = 712. Verify.
- Are the three models (Heisenberg, Josephson, TFIM) described consistently throughout?
- Does the paper use "chi" and "χ" consistently?
- Are D_MPO values correct? Heisenberg=5, Josephson=3, TFIM=3?
  - Check: `test_dmrg_gpu.cpp` or `test_dmrg2_gpu.cpp` — `build_heisenberg_mpo()`, `build_josephson_mpo()`, `build_tfim_mpo()`
- Local dimensions: Heisenberg d=2, Josephson d=5, TFIM d=2?
  - Check: same MPO builder functions
- Josephson parameters: E_J=1.0, E_C=0.5, phi_ext=pi/4, nmax=2?
  - Check: `build_josephson_mpo()` in test files

### E. Potential misleading claims

Watch for:
- Comparing parallel DMRG against serial and calling it "slower" without noting that parallel DMRG's purpose is multi-GPU scaling, not single-GPU performance
- Calling Newton-Schulz "diverges at chi>=128" — is this always, or only for certain models? The Josephson data in `paper_prompt.md` shows opt runs at chi=128 and chi=256 that completed (slowly). Is it only Heisenberg that diverges?
- "0% win rate across 50 configurations" — are these 50 configs the ones where BOTH baseline and opt converged? How many total configs were tested? How many did opt fail to converge on?
- The Amdahl's law analysis: "serial fraction = 22.1s out of 24.1s" — verify these numbers against the time breakdown table
- "4-thread BLAS is the sweet spot for quimb" — is this a general claim or only for specific configs?

## Output format

For each claim, output:

```
CLAIM: [exact quote or paraphrase from paper]
LOCATION: [section/line reference]
VERDICT: VERIFIED | WRONG | UNSUPPORTED | MISLEADING
EVIDENCE: [what you found in code/data/literature]
FIX: [if WRONG or MISLEADING, suggest correction]
```

Group findings by severity:
1. **WRONG** claims first (these must be fixed before submission)
2. **MISLEADING** claims second (these should be reworded)
3. **UNSUPPORTED** claims third (these need evidence or hedging language)
4. **VERIFIED** claims last (brief list, no need for details)

## Files to read

Essential (read these fully):
- `paper/main.tex` — the paper
- `pdmrg-gpu/src/pdmrg_gpu_impl.h` — GPU PDMRG implementation (search for `run(`, `sweep_`, `warmup`, `polish`)
- `pdmrg-gpu-opt/src/pdmrg_gpu_opt_impl.h` — GPU PDMRG-opt implementation
- `pdmrg-gpu/OPTIMIZATION_REPORT.md` — profiling data source
- `pdmrg-gpu-opt/OPTIMIZATION_REPORT.md` — algorithmic optimization results

Data files (load and query):
- `benchmarks/paper_results/results.json` — 604 entries
- `benchmarks/paper_results/gpu_opt_bench.json` — 108 entries
- `benchmarks/paper_results/bench_opt_results.csv` — opt vs baseline

Reference (skim):
- `paper_prompt.md` — detailed data tables and analysis
- `paper/PAPER_PROMPT.md` — paper writing guide with key numbers
- `dmrg2-gpu/src/dmrg2_gpu_impl.h` — serial two-site GPU DMRG
- Any `test_*.cpp` files for MPO builder verification
