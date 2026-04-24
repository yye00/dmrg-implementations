# Paper Writing Prompt

## What This Paper Is

A systematic benchmarking study of DMRG implementations spanning Python/CPU to hand-tuned GPU (HIP/rocBLAS on AMD MI300X). The paper honestly reports what worked, what didn't, and where the crossover points are. It includes significant negative results (Newton-Schulz, Chebyshev, batched sweep all failed to improve performance) that are valuable for the community.

**Target journal**: Computer Physics Communications (CPC), Elsevier
**Style**: Clear, direct scientific writing. No hype. Honest about negative results. ~15-20 pages including figures and tables.
**Format**: LaTeX (`elsarticle` class). Write in `paper/` subdirectory.

---

## CPC Submission Guidelines

### Submission Category
**Regular Research Article** (computational physics with detailed benchmarking). Not a "Computer Programs in Physics" submission — we are not releasing a standalone software package, but a systematic performance study with open-source code.

### Format Requirements
- **Document class**: `\documentclass[preprint,12pt]{elsarticle}` for review submission (single-column, 12pt, double-spaced). Final production uses `\documentclass[5p,twocolumn]{elsarticle}`.
- **Abstract**: 250-500 words (CPC allows longer abstracts than most journals). Should summarize the models, hardware, key positive and negative results, and the main takeaway.
- **Keywords**: 4-6 keywords after abstract. E.g.: *DMRG, GPU computing, tensor networks, AMD MI300X, parallel algorithms, performance benchmarking*
- **References**: Numbered [1], [2], ... in order of appearance (Vancouver style). Use `\bibliographystyle{elsarticle-num}`.
- **Figures**: Vector PDF preferred (matplotlib `savefig('fig.pdf')`). Minimum 300 DPI for raster. Must be readable in grayscale — use markers/line styles in addition to color.
- **Line numbering**: Enable with `\usepackage{lineno}` and `\linenumbers` for review.

### Required Sections (CPC standard structure)
1. **Introduction** — Context, motivation, what gap this fills
2. **Methods** — Models, algorithms, implementation details
3. **Computational Details** — Hardware specs, software versions, benchmark methodology
4. **Results** — Performance data, comparisons, analysis
5. **Discussion** — Negative results analysis, bottleneck identification, lessons learned
6. **Conclusions** — Summary of findings, when to use GPU DMRG, future directions
7. **Acknowledgments** — Hardware access, funding
8. **Data Availability Statement** — REQUIRED. Point to GitHub repository with code and benchmark data.
9. **References**

### CPC-Specific Requirements
- **Code availability**: Mandatory. Include GitHub URL, commit hash, and brief build instructions. CPC strongly values reproducibility.
- **Data availability statement**: Required even for non-program papers. Template: *"The source code, benchmark scripts, and raw benchmark data are available at [GitHub URL]. The repository includes build instructions for ROCm 7.2 on AMD MI300X."*
- **Reproducibility**: CPC reviewers will check that results can in principle be reproduced. Include: exact compiler flags, ROCm version, GPU model, number of runs, whether times are best-of or average.

### LaTeX Preamble
```latex
\documentclass[preprint,12pt,authoryear]{elsarticle}
\usepackage{lineno}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{siunitx}
\usepackage{algorithm}
\usepackage{algpseudocode}

\linenumbers

\journal{Computer Physics Communications}

\begin{document}
\begin{frontmatter}
\title{GPU-Accelerated DMRG on AMD MI300X: Systematic Benchmarking and Lessons from Failed Optimizations}

\author[inst1]{...}
\affiliation[inst1]{organization={...}, city={...}, country={...}}

\begin{abstract}
% 250-500 words
\end{abstract}

\begin{keyword}
DMRG \sep GPU computing \sep tensor networks \sep AMD MI300X \sep performance benchmarking
\end{keyword}

\end{frontmatter}
```

---

## Repository Layout

Everything lives in `/home/captain/clawd/work/dmrg-implementations/`. Here is the complete map of what matters:

### Source Code (10 implementations)

**CPU Implementations:**
```
pdmrg/                     # Python/numpy parallel DMRG (Stoudenmire & White 2013)
pdmrg-opt/                 # Same + Newton-Schulz/Davidson (CPU prototype, not benchmarked)
pdmrg-cotengra/            # PDMRG with cotengra optimal contraction paths
a2dmrg/                    # Additive two-level DMRG (Grigori & Hassan 2025)
```
quimb is used as external reference (installed in venvs, not our code).

**GPU Implementations (HIP/C++ on MI300X):**
```
dmrg-gpu/src/              # Single-site GPU DMRG (Lanczos + SVD)
dmrg2-gpu/src/             # Two-site GPU DMRG (Lanczos + SVD, fused MPO)
pdmrg-gpu/src/             # Parallel segment GPU DMRG (per-segment HIP streams)
dmrg-gpu-opt/src/          # Single-site + Newton-Schulz/Davidson/MFMA padding
dmrg2-gpu-opt/src/         # Two-site + Newton-Schulz/Davidson/MFMA padding
pdmrg-gpu-opt/src/         # Parallel segment + all Tier 1+2 optimizations
```

Each GPU impl has: `*_gpu.h` (class decl), `*_gpu_impl.h` (template impl), `scalar_traits.h` (BLAS dispatch + GPU kernels), `test_*.cpp` (test harness with MPO builders).

All GPU code is templated on `Scalar` (double or hipDoubleComplex) via `ScalarTraits<Scalar>`.

### Benchmark Data (READ THESE FIRST)

```
benchmarks/paper_results/results.json          # 604 entries — THE main dataset
                                                # All implementations, all models, all sizes
                                                # Fields: impl, model, L, chi, sweeps, segments,
                                                #         energy, wall_time, success, threads, etc.

benchmarks/paper_results/gpu_opt_bench.json    # 108 entries — GPU-opt scaling study
                                                # All 6 GPU impls + batched sweep A/B comparison
                                                # Fields: impl, model, L, chi, segments, label,
                                                #         wall_time, energy, error, status

benchmarks/paper_results/bench_opt_results.csv # Opt vs baseline head-to-head (192 configs)
benchmarks/paper_results/bench_opt_results.json # Same data, JSON format

benchmarks/paper_results/summary.csv           # Running analysis notes (large, grep for sections)
benchmarks/paper_results/wins_cpu_vs_gpu.csv   # CPU vs GPU winner per config
benchmarks/paper_results/wins_gpu_serial_vs_parallel.csv  # Serial vs parallel GPU
benchmarks/paper_results/wins_quimb_vs_mpi.csv # quimb vs MPI implementations

benchmarks/paper_results/gpu_4way_results.csv  # Earlier 4-way GPU comparison
benchmarks/paper_results/gpu_bench_results.txt # Earlier GPU benchmark text output
benchmarks/paper_results/timing_*.png          # Timing plots (Heisenberg, Josephson, TFIM)
benchmarks/paper_results/a2dmrg_*_results.txt  # A2DMRG validation results
```

### Documentation (READ THESE FOR NARRATIVE)

```
paper_prompt.md                    # *** PREVIOUS PAPER PROMPT — very detailed ***
                                    # Contains: full paper outline, all win tables,
                                    # opt vs baseline comparison (50 configs),
                                    # A2DMRG story, pairwise head-to-head matrix,
                                    # geometric mean speedups, analysis by chi/model
                                    # THIS IS YOUR PRIMARY REFERENCE FOR PAPER STRUCTURE

pdmrg-gpu/OPTIMIZATION_REPORT.md   # GPU infrastructure optimizations report
                                    # Device-pointer Lanczos, GPU pointer kernels,
                                    # batched Step-3, pinned memory race condition,
                                    # boundary coupling attempts, Amdahl's law analysis,
                                    # CPU SVD dominance finding (97-98% at chi>=128)
                                    # DETAILED PERFORMANCE BREAKDOWNS WITH TIMINGS

pdmrg-gpu-opt/OPTIMIZATION_REPORT.md # Algorithmic optimization attempts report
                                      # Newton-Schulz (correct but slower)
                                      # Block-Davidson (comparable to Lanczos)
                                      # MFMA-16 padding (5-10% improvement)
                                      # Chebyshev eigensolver (1.9-11x SLOWER)
                                      # Cross-segment batched GEMM (slower except chi=256 seg=2)
                                      # KEY LESSON: BLAS-2→BLAS-3 doesn't help at chi<=256

optimizations.md                    # Tier 1/2/3 optimization design document
                                    # What was planned vs what was implemented

PDMRG_GPU_OPT_PROMPT.md           # Implementation spec for pdmrg-gpu-opt
                                    # Newton-Schulz algorithm details, Block-Davidson details

docs/CPU_ARCHITECTURE.md            # CPU implementation architecture reference
                                    # PDMRG vs PDMRG-OPT differences table

a2dmrg/docs/a2dmrg-critical-analysis.md  # A2DMRG failure analysis
                                          # Why A2DMRG doesn't work for spin chains

benchmarks/BENCHMARK_STATUS.md      # Benchmark infrastructure status
benchmarks/README.md                # Benchmark runner documentation
```

### Benchmark Scripts

```
benchmarks/paper_benchmark.py       # Main production benchmark (724 configs)
benchmarks/bench_opt.py             # Opt vs baseline comparison
benchmarks/bench_gpu_all.py         # All GPU implementations benchmark
benchmarks/bench_gpu_opt_scaling.py # Batched sweep scaling study
benchmarks/rerun_*.py               # Various rerun scripts for specific fixes
benchmarks/lib/                     # Benchmark library (registry, runners, models)
```

---

## The Story Arc

### Act 1: The Landscape (Sections 1-3)
DMRG is the gold standard for 1D quantum many-body problems. GPU acceleration is attractive because tensor contractions map to GEMM. We systematically benchmark 10 implementations across 3 models on MI300X.

### Act 2: What Works (Sections 4-5)
- **GPU DMRG wins at chi >= 128** (74% win rate), dominates at chi=256 (100%)
- **CPU wins at chi <= 50** — quimb with 4-thread BLAS beats GPU due to kernel launch overhead
- **Two-site converges 3x faster** than single-site at large L (53s vs 177s at L=100 chi=128)
- The crossover is at chi ≈ 64-128 depending on model and L

### Act 3: What Failed — and Why (Section 6, the meat of the paper)
Three categories of failures, each instructive:

**A. Newton-Schulz + Block-Davidson (0% win rate across 50 configs)**
- Premise: Replace BLAS-2 algorithms with BLAS-3 (GEMM-heavy) alternatives
- Reality: NS requires 24-32 GEMMs per SVD replacement; LAPACK SVD is one call
- Diverges numerically at chi >= 128 (condition number)
- Block-Davidson has no advantage over Lanczos for single eigenvalue
- The "BLAS-3 everything" thesis fails because DMRG matrices (128×128 to 512×512) are too small for GPU GEMM throughput to compensate for iteration overhead

**B. Chebyshev-Filtered Subspace Iteration (1.9-11x slower)**
- Premise: Polynomial filter avoids orthogonalization, sync-free
- Reality: Does degree×outer = up to 300 matvecs; Lanczos converges in 15
- Designed for many eigenvalues (DFT), not single ground state
- Wrong algorithm for the problem

**C. Cross-Segment Batched GEMM (slower at 18/19 configs)**
- Premise: Batch GEMM calls across PDMRG segments to reduce launch overhead
- Reality: Serializes BLAS-1 operations that were running concurrently
- Lock-step sweep loses async execution advantage of thread-per-segment
- One bright spot: chi=256 seg=2 shows 1.18x speedup (large enough GEMMs)

**D. A2DMRG (correct but impractical)**
- Premise: All-sites-parallel DMRG via additive Schwarz + coarse-space correction
- Reality: TT-SVD compression ratio (L-1):1 kills accuracy at L > 20
- Complex Hamiltonians catastrophically fail (ΔE = 10³)
- Designed for quantum chemistry (d=12-24), not spin chains (d=2)

### Act 4: The Real Bottleneck (Section 7)
- CPU SVD consumes 97-98% of per-sweep runtime at chi >= 128
- All other optimizations combined affect 2-3% of wall time
- The path forward is: (a) chi >> 1000 where GPU GEMM iteration amortizes, (b) multi-GPU with per-GPU segments, (c) fundamentally different SVD algorithms (not iterative polar)

### Act 5: Conclusions
- GPU DMRG is worth it starting at chi ≈ 128 on MI300X
- Don't replace working LAPACK SVD with iterative GPU methods at chi < 1000
- Negative results are valuable: saves the community from repeating these experiments
- Open-source code enables reproduction

---

## Key Data Tables to Include

### Table 1: Implementation overview
Read from: `paper_prompt.md` Section 2, `docs/CPU_ARCHITECTURE.md`

### Table 2: Win rates by bond dimension
Read from: `paper_prompt.md` Section 6.2
```
chi=20:  GPU 62.5%, CPU 37.5%
chi=50:  GPU 42.1%, CPU 57.9%
chi=128: GPU 84.2%, CPU 15.8%
chi=256: GPU 100%, CPU 0%
```

### Table 3: Opt vs baseline (50 configs, 0% win rate)
Read from: `paper_prompt.md` Section 4.6 (full table lines 187-243)
Also in: `benchmarks/paper_results/bench_opt_results.csv`

### Table 4: Serial GPU comparison (latest, 6 impls)
Read from: `benchmarks/paper_results/gpu_opt_bench.json` (Phase 1 entries, segments=null)
```python
import json
d = json.load(open("benchmarks/paper_results/gpu_opt_bench.json"))
serial = [r for r in d if not r.get("segments") and not r.get("label")]
```

### Table 5: Batched sweep scaling (19 pairs)
Read from: `benchmarks/paper_results/gpu_opt_bench.json` (Phase 3 entries, label="baseline"/"batched")

### Table 6: Chebyshev vs Lanczos
From commit messages / summary.csv:
```
L=8  chi=32  seg=2: Lanczos 0.732s, Chebyshev 1.395s (1.9x slower)
L=20 chi=50  seg=2: Lanczos 5.273s, Chebyshev 59.15s (11x slower)
```

### Table 7: Pairwise head-to-head matrix
Read from: `paper_prompt.md` Section 6.4

### Table 8: A2DMRG accuracy degradation
Read from: `paper_prompt.md` Section 5.4, `a2dmrg/docs/a2dmrg-critical-analysis.md`

---

## Figures to Generate

1. **Wall time vs chi** (3 panels: Heisenberg, Josephson, TFIM) — crossover curves
   - Existing PNGs: `benchmarks/paper_results/timing_*.png` (may need regeneration with latest data)
   - Data: `results.json`

2. **GPU speedup heatmap** (L × chi, color = speedup over best CPU)
   - Data: `results.json`, compute `min(GPU times) / min(CPU times)` per (model, L, chi)

3. **Opt vs baseline scatter** — every config, x=baseline time, y=opt time, diagonal = parity
   - Data: `bench_opt_results.csv`
   - Every point below diagonal = opt wins (expect: zero points below)

4. **SVD time fraction** — stacked bar showing SVD vs Lanczos vs env vs other
   - Data: `pdmrg-gpu/OPTIMIZATION_REPORT.md` lines 86-94 (profiling breakdown)

5. **Batched sweep scaling** — baseline vs batched wall time, grouped by segment count
   - Data: `gpu_opt_bench.json` Phase 3

6. **Chebyshev matvec budget** — bar chart: Lanczos (~15 matvecs) vs Chebyshev (up to 300)

---

## Physics Models Details

### Heisenberg (d=2, real, D_MPO=5)
H = Σᵢ (SˣᵢSˣᵢ₊₁ + SʸᵢSʸᵢ₊₁ + SᶻᵢSᶻᵢ₊₁), OBC
MPO construction: `test_*.cpp` → `build_heisenberg_mpo()`

### Josephson Junction Array (d=5, complex128, D_MPO=3)
H = -Eⱼ/2 Σᵢ (e^{iφ_ext} e^{iφᵢ} e^{-iφᵢ₊₁} + h.c.) + Eᶜ Σᵢ nᵢ²
Parameters: Eⱼ=1.0, Eᶜ=0.5, φ_ext=π/4, nmax=2 (charge basis -2..+2)
MPO construction: `test_*.cpp` → `build_josephson_mpo()`

### TFIM at Critical Point (d=2, real, D_MPO=3)
H = -J Σᵢ ZᵢZᵢ₊₁ - h Σᵢ Xᵢ, h/J=1.0 (critical)
MPO construction: `test_*.cpp` → `build_tfim_mpo()`

---

## Hardware Specs

**GPU**: AMD Instinct MI300X (gfx942)
- 304 Compute Units
- 192 GB HBM3 (5.3 TB/s bandwidth)
- FP64 peak: 81.7 TFLOPS (matrix), 40.9 TFLOPS (vector)
- ROCm 7.2

**CPU**: AMD EPYC (host CPU on cloud VM)
- Multi-core, used for quimb benchmarks at 1/2/4/8/12 threads
- OpenBLAS 0.3.28 (compiled from source — system 0.3.20 has broken SVD)

---

## Loading the Data

```python
import json

# Main benchmark dataset (604 entries)
results = json.load(open("benchmarks/paper_results/results.json"))
# Each entry: {impl, model, L, chi, sweeps, segments, energy, wall_time, success, threads, ...}

# GPU-opt scaling study (108 entries)
gpu_opt = json.load(open("benchmarks/paper_results/gpu_opt_bench.json"))
# Each entry: {impl, model, L, chi, segments, label, wall_time, energy, error, status, ...}

# Opt vs baseline comparison (CSV, 192 rows)
import csv
with open("benchmarks/paper_results/bench_opt_results.csv") as f:
    reader = csv.DictReader(f)
    opt_results = list(reader)
```

---

## Key Numerical Results to Cite

- **CPU SVD = 97-98% of runtime** at chi >= 128 (from profiling, `pdmrg-gpu/OPTIMIZATION_REPORT.md` line 218)
- **dmrg2-gpu 3.3x faster than dmrg-gpu** at L=100 chi=128 (53s vs 177s, two-site convergence)
- **Newton-Schulz: 0/50 wins** (0% win rate, all configs baseline wins)
- **Newton-Schulz diverges at chi >= 128** (energies off by 10⁶ to 10²⁰)
- **Chebyshev: 1.9-11x slower** than Lanczos (wrong algorithm for single eigenvalue)
- **Batched sweep: 1/19 configs faster** (only at chi=256 seg=2, 1.18x)
- **A2DMRG accuracy**: < 1e-10 at L=8-20, degrades to 10⁻⁵ at L=64, catastrophic at complex
- **GPU crossover**: chi ≈ 64-128 (GPU wins 74% at chi=128, 100% at chi=256)
- **Pinned memory race condition**: critical finding for GPU programming (see OPTIMIZATION_REPORT.md)

---

## What NOT to Do

- Don't oversell the GPU results. CPU wins at small chi. Say so clearly.
- Don't hide the negative results. The Newton-Schulz failure is the most interesting part.
- Don't claim the -opt variants are "optimized" — they're slower. Call them "algorithmic variants" or "alternative implementations."
- Don't compare pdmrg-gpu-opt against pdmrg-gpu and claim victory — compare against best serial GPU (dmrg-gpu or dmrg2-gpu).
- Don't speculate about multi-GPU without data. We only tested single GPU.

---

## LaTeX Structure

```
paper/
├── main.tex              # Main document (elsarticle preprint format)
├── figures/              # Generated figures (vector PDF from matplotlib)
├── scripts/              # Python scripts to generate figures/tables from JSON data
├── refs.bib              # Bibliography (elsarticle-num style)
└── Makefile              # Build: pdflatex + bibtex
```

Use `elsarticle` document class with `preprint,12pt` options for review submission. Enable line numbering with `\linenumbers`. Use `\bibliographystyle{elsarticle-num}` for numbered references.

### Manuscript Sections (CPC order)
```latex
\section{Introduction}
\section{Methods}
  \subsection{DMRG algorithm}
  \subsection{Two-site optimization and bond adaptation}
  \subsection{Parallel DMRG (domain decomposition)}
  \subsection{GPU implementation strategy}
\section{Algorithmic variants}
  \subsection{Newton-Schulz polar decomposition}
  \subsection{Block-Davidson eigensolver}
  \subsection{Chebyshev-filtered subspace iteration}
  \subsection{Cross-segment batched GEMM}
\section{Computational details}
  \subsection{Hardware and software environment}
  \subsection{Benchmark methodology}
  \subsection{Physics models}
\section{Results}
  \subsection{CPU vs GPU crossover}
  \subsection{Single-site vs two-site convergence}
  \subsection{Parallel segment scaling}
  \subsection{Algorithmic variant performance}
\section{Discussion}
  \subsection{Why BLAS-3 replacements fail at moderate chi}
  \subsection{CPU SVD bottleneck analysis}
  \subsection{Implications for future GPU DMRG}
\section{Conclusions}

% CPC required sections:
\section*{Declaration of competing interest}
\section*{Data availability}
\section*{Acknowledgments}
```

---

## References to Cite

- White 1992, 1993 (original DMRG)
- Stoudenmire & White 2013 (real-space parallel DMRG)
- Grigori & Hassan 2025 (additive two-level DMRG, arXiv:2505.23429v2)
- quimb: Gray 2018 (quimb tensor network library)
- cotengra: Gray & Kourtis 2021 (optimal contraction paths)
- Newton-Schulz: Schulz 1933; Nakamura & Higham 2020 (polar decomposition)
- Block-Davidson: Davidson 1975; Liu 1978
- Chebyshev filtering: Zhou et al 2006 (CheFSI for DFT)
- ROCm/rocBLAS: AMD documentation
- ITensor: Fishman et al 2022 (comparison context)
- Block2: Zhai & Chan 2021 (comparison context)
- QDWH: Nakamura & Higham 2013 (related work on GPU polar decomposition)
