# DMRG Implementations — Project Overview

One-stop reference for every implementation in this repository, what we tried,
what worked, what didn't, and why. Read this before editing anything; it will
save you from re-running the experiments we already ran.

Scope: targeting **quantum computing workloads** (qubit chains, `d=2`, `L=50–200+`,
moderate `chi`, real + complex Hamiltonians — VQE/QAOA/circuit-sim regime). Not
quantum chemistry (which would favor `d=12–24`, huge `chi`, real-only).

---

## 1. Repository Layout

```
dmrg-implementations/
├── cpu/                          Python / MPI implementations
│   ├── pdmrg/                      Real-space parallel DMRG (Stoudenmire-White)
│   ├── pdmrg-cotengra/             PDMRG using cotengra contraction paths
│   ├── pdmrg-opt/                  PDMRG + BLAS-3 prep (Newton-Schulz, Block-Davidson)
│   └── a2dmrg/                     Additive two-level DMRG (Grigori-Hassan 2025)
│
├── gpu-rocm/                     AMD MI300X (gfx942) implementations, C++/HIP
│   ├── dmrg-gpu/                   Single-site, tuned (Lanczos + SVD)
│   ├── dmrg-gpu-opt/               Single-site + MFMA padding, strided batched GEMM
│   ├── dmrg-gpu-base/              Single-site naive baseline (no optimizations)
│   ├── dmrg2-gpu/                  Two-site, tuned (fused WW, batched GEMM)
│   ├── dmrg2-gpu-opt/              Two-site + Newton-Schulz, Block-Davidson, MFMA padding
│   ├── dmrg2-gpu-base/             Two-site naive baseline
│   ├── pdmrg-gpu/                  Stream-parallel PDMRG, tuned
│   ├── pdmrg-gpu-opt/              Stream-parallel PDMRG + every -opt experiment
│   ├── pdmrg-gpu-base/             Stream-parallel PDMRG naive baseline
│   └── pdmrg-multi-gpu/            4-device PDMRG across 4 MI300Xs
│
├── gpu-cuda/                     NVIDIA H100 CUDA ports (mirrors gpu-rocm/)
│   └── {dmrg-gpu, dmrg-gpu-opt, dmrg2-gpu, dmrg2-gpu-opt, pdmrg-gpu, pdmrg-gpu-opt}
│
├── benchmarks/
│   ├── run.py                      Unified CLI entry point
│   ├── run_mi300x_challenge.py     Large-scale MI300X sweep
│   ├── run_h100_challenge.py       H100 equivalent
│   ├── paper_benchmark.py          Publication-grade suite
│   ├── lib/                        Registry, dispatch, runners
│   ├── data/                       Seeded binary MPS/MPO test data
│   ├── results/{mi300x,h100}/      Raw benchmark results
│   └── paper_results/{mi300x,h100}/  Publication-grade results + CSVs
│
├── reports/{mi300x,h100}/        Timing-breakdown JSONs
├── paper/                        CPC manuscript (main.tex, figures)
├── docs/                         Design prompts, optimization notes, this file
└── benchmark_data/               Challenge-problem MPS/MPO fixtures
```

### Where benchmark results live

| Path                                              | What                                        |
|---------------------------------------------------|----------------------------------------------|
| `benchmarks/paper_results/mi300x/summary.csv`     | 737-row master table of all (impl, model, L, chi) runs |
| `benchmarks/paper_results/mi300x/challenge/`      | Challenge-size (large) JSONs per implementation |
| `benchmarks/paper_results/mi300x/wins_cpu_vs_gpu.csv` | Side-by-side CPU vs GPU winners (GPU win rate: 18%) |
| `benchmarks/paper_results/mi300x/results.json`    | Full benchmark run (417 results, phases 0-3) |
| `benchmarks/paper_results/mi300x/gpu_opt_bench.json` | 108-config gpu-opt scaling study |
| `reports/mi300x/`                                 | Per-sweep timing breakdowns (JSON) |

---

## 2. Implementation Taxonomy

### 2.1 CPU implementations (`cpu/`)

| Package            | Algorithm                                    | Key files |
|--------------------|----------------------------------------------|-----------|
| `pdmrg/`           | Real-space parallel DMRG, MPI domain decomposition | `pdmrg/pdmrg/dmrg.py` (main), `parallel/merge.py` (V=Λ⁻¹ boundary merge) |
| `pdmrg-cotengra/`  | Same + cotengra contraction paths             | Same as pdmrg, with cotengra H_eff |
| `pdmrg-opt/`       | Same + BLAS-3 prep (Newton-Schulz polar, Block-Davidson) | `pdmrg/numerics/linalg_utils.py` |
| `a2dmrg/`          | Additive two-level DMRG: local micro-steps + coarse GEV | `a2dmrg/a2dmrg/dmrg.py`, `numerics/coarse_eigenvalue.py` |

**All three parallel CPU variants require `np >= 2`.** With `np=1`, `pdmrg` and `a2dmrg` return the serial quimb DMRG2 warmup energy without executing the parallel algorithm — so "PDMRG np=1" timings actually measure quimb.

**Contraction strategy (all CPU variants):** hand-coded `np.tensordot` chains, no opt_einsum/cotengra in the hot path (except `pdmrg-cotengra`). Full contraction-index conventions and optimal ordering are in `docs/CPU_ARCHITECTURE.md`.

**Precision policy:** never single precision. All arrays are `float64` or `complex128`. A2DMRG validates dtype at entry.

**Caveats documented in `README.md`:**
- `pdmrg` boundary-merge optimization path is permanently disabled ("H_eff spurious eigenvalue problem"). Performance impact undocumented.
- A2DMRG intentionally skips canonicalization to preserve bond-dim structure for coarse-space linear combination.

### 2.2 GPU implementations (`gpu-rocm/`)

All GPU code is templated on `Scalar ∈ {double, hipDoubleComplex}` via `scalar_traits.h` (rocBLAS dispatch shim). Hard-coded BLAS-3 everywhere per project rules: **no CPU for-loops or CPU BLAS calls in the hot path** except explicit exceptions (CPU LAPACK for tridiagonal eigensolves, early-experiment CPU SVD).

Three tiers exist for each algorithm family:

| Tier       | Purpose                                                        |
|------------|---------------------------------------------------------------|
| `*-base/`  | Naive baseline. No optimizations. Used as speedup denominator. |
| `*-gpu/`   | Production-tuned. All optimizations that actually helped.      |
| `*-gpu-opt/` | Kitchen-sink experiment. Every BLAS-3 idea we tried.         |

#### Single-site: `dmrg-gpu{-base,-opt}` / `dmrg-gpu-opt`

- All environment updates, `H_eff` matvec, Lanczos Krylov iters run on GPU via rocBLAS.
- Only tridiagonal eigensolve (~100 elements) runs on CPU LAPACK.
- SVD: CPU LAPACK default (2–6× faster for `chi < 200`), `--gpu-svd` flag switches to rocsolver.

#### Two-site: `dmrg2-gpu{-base,-opt}` / `dmrg2-gpu-opt`

- Same 3-step GEMM pattern as single-site but with `d → d²` after two-site theta formation.
- **Fused MPO cache** `WW[bond] = W_L ⊗ W_R` precomputed once per `set_mpo`.
- `apply_heff`: Step 1 = D×d² batched GEMMs; Step 2 = one dense GEMM against `WW`; Step 3 = d²×D batched GEMMs.
- SVD split at the physical bond: `theta (cL·d, d·cR)` → `U=MPS[site]`, `S·Vh=MPS[site+1]`.

#### Stream-parallel: `pdmrg-gpu{-base,-opt}` / `pdmrg-gpu-opt`

- Partitions chain into N segments, assigns each to its own **HIP stream + rocBLAS handle**.
- `std::thread` drives each segment (one CPU thread per segment) — this is **non-optional**, see §4.4.
- Full three-phase algorithm: **warmup** (single-site, serial, stream 0) → **outer** (parallel segments + boundary coupling) → **polish** (single-site, serial, stream 0).
- Stoudenmire boundary merge uses `V = Λ⁻¹` with safety floor at 1e-12.

#### Multi-GPU: `pdmrg-multi-gpu/`

- Distributes segments across 4 physical MI300X devices (not just 4 streams on one device).
- Uses peer-to-peer memory and per-device rocBLAS handles.
- Partial benchmark results (27/44 configs) in `benchmarks/paper_results/mi300x/challenge/pdmrg-multi-gpu_mi300x_challenge_20260407_partial.json` — VM died before completion.

### 2.3 CUDA ports (`gpu-cuda/`)

- All 6 ROCm implementations ported to CUDA: dmrg-gpu, dmrg-gpu-opt, dmrg2-gpu, dmrg2-gpu-opt, pdmrg-gpu, pdmrg-gpu-opt.
- 18,407 lines, each subdirectory fully independent (no shared code with `gpu-rocm/`).
- All 6 compile and pass `L=4` correctness on H100.
- **Known workaround:** cuSOLVER 13.0 `cusolverDnDgesvd` rejects `m < n`. Every CUDA impl transposes wide matrices before SVD and swaps `U`/`Vh` afterwards. SVD workspace is dynamically requeried because non-square workspaces can exceed the max-size query. See `gpu-cuda/README.md`.

### 2.4 Reference baselines (external)

- `quimb-dmrg1` — quimb single-site DMRG.
- `quimb-dmrg2` — quimb two-site DMRG. This is the gold standard for correctness (`1e-10` tolerance), and is what we chase in wall time.

---

## 3. What Each Naive Baseline Strips Away

The `*-base/` variants (added in commit `8926b71`) exist so speedup numbers have
an unambiguous denominator. They are not meant to be fast — they are meant to be
honest.

### Stripped from `dmrg-gpu-base/` and `dmrg2-gpu-base/`

- No fused two-site MPO cache — `WW` is rebuilt on the **host** from raw MPO
  tensors on every `apply_heff` call and uploaded to a GPU scratch buffer.
- No `gemm_batched` / batched-pointer setup: all environment updates and matvec
  steps are nested for-loops of single rocBLAS gemm calls.
- Host-pointer-mode rocBLAS throughout. Lanczos reductions (`nrm2`, `dot`) incur
  an implicit `hipStreamSynchronize` every call.
- Host-pointer-mode Lanczos + CPU LAPACK `dstev_` for the tridiagonal eigenproblem.
- rocSOLVER `gesvd` + full D2H copy + host-side truncation / S-scaling for every
  SVD split. No randomized SVD, no on-device truncation.
- No custom GPU kernels except the complex-conjugate helper (bra correctness
  requires this on complex environments).

### Stripped from `pdmrg-gpu-base/`

All of the above, and additionally:

- No pre-computed batch pointer arrays; no device scalars for Lanczos.
- No custom `scale_rows_by_real` kernel — the Stoudenmire boundary merge pulls
  ψ_R to host, scales rows by `V` on host, uploads to scratch, then GEMMs with ψ_L.
- No per-segment worker stream pool.

### Preserved in the baseline (non-optimizations)

- Single-site warmup + polish sweeps: these are part of the PDMRG algorithm, not
  an optimization. Removing them is wrong.
- `std::thread` per segment driving its own HIP stream: without this, the
  segments run sequentially despite using separate streams (see §4.4).
- Stoudenmire `V = Λ⁻¹` boundary coupling with floor at 1e-12.

### Hard-coded defaults (no CLI tuning knobs)

| Parameter      | Value | Scope |
|----------------|-------|-------|
| `n_warmup`     | 3     | pdmrg-gpu-base (single-site warmup sweeps) |
| `n_local`      | 2     | pdmrg-gpu-base (local sweeps per outer iter) |
| `n_segments`   | 2     | pdmrg-gpu-base (tests) |
| `polish`       | 10    | pdmrg-gpu-base (two-site full-chain polish sweeps) |
| SVD backend    | rocSOLVER gesvd + host trunc | all bases |
| eigensolver    | Lanczos, CPU `dstev_` | all bases |

CLI on all base binaries is just positional `L chi_max n_sweeps` plus problem
selection `--josephson | --tfim`. `--nmax` is accepted-and-ignored for
benchmark-runner compatibility.

---

## 4. What We Tried — Exhaustive Record

This section is the whole point of the document. Every optimization below was
implemented; most were benchmarked; some were reverted. The format is:

> **Name** — what, why, result.

### 4.1 Optimizations that HELPED (shipped in `*-gpu/`)

#### A1: Device-pointer-mode Lanczos

**What**: Switched rocBLAS from `rocblas_pointer_mode_host` to `...device` for
`dot`/`nrm2`/`axpy`/`scal` inside the Lanczos eigensolver. Allocated per-stream
device scalars in `StreamWorkspace`: `d_dot_result`, `d_nrm2_result`,
`d_neg_alpha`, `d_neg_overlap`, `d_inv_nrm`, `d_alpha_dev`, `d_beta_dev`,
`d_const_one/zero/neg_one`.

**Why**: Host-pointer mode forces an implicit `hipStreamSynchronize` after every
reduction. With device-pointer mode, the reduction writes to device memory and
the next BLAS call consumes a device pointer — no sync.

**Result**: Eliminated ~100 sync stalls per bond optimization. Lanczos itself is
only 1–2% of runtime at `chi ≥ 128`, so the wall-clock impact is small (~0.1–0.3s
per sweep). Nevertheless essential for opening the door to async execution.

Commit: `37980ec`.

#### A3: GPU-side pointer-setup kernels

**What**: Six small kernels in `scalar_traits.h` that compute batched-GEMM
pointer arrays **directly on the GPU**:
- `setup_heff_A_ptrs` (L_env slices, cached per site)
- `setup_heff_B_ptrs` (theta slices, recomputed per Lanczos iter)
- `setup_heff_C_ptrs` (T1 slices, cached per site)
- `setup_lenv_ptrs` / `setup_renv_ptrs` (env update pointer triples)

Each launches 1 block of `batch_count` threads (typically 20–36).

**Why**: The original code allocated `std::vector<Scalar*>` on the heap, filled
them in a CPU loop, then `hipMemcpyAsync`ed to device — called 10–50× per
Lanczos iter per bond.

**Saga (read this before touching pointer upload code!)**: The first attempted
fix (commit `146cd85`) used **pinned host memory** with `hipMemcpyAsync`. This
introduced a nasty race — see §5.3 for the full story. Final fix in commit
`50eb73c` bypasses host-to-device transfer entirely via GPU kernels.

**Result**: Eliminated thousands of heap allocations + H2D transfers per sweep.
Also killed the pinned-memory race. Wall-clock savings: ~0.5–1s per sweep at
`chi=128`. Commits: `146cd85` → `50eb73c`.

#### A4: Batched GEMM for Step 3 of `apply_heff_two_site`

**What**: Step 3 used to issue `d²·D` separate GEMM calls in a triple-nested
loop (20 for Heisenberg `d=2,D=5`; 36 for Josephson `d=3,D=4`). Replaced with
`rocblas_gemm_batched` using the same GPU-side pointer setup.

**Result**: Reduces per-GEMM dispatch overhead. Measurable at small `chi`
(launch overhead significant); marginal at `chi ≥ 128`. Initially committed
(`27a4967`), reverted during debug (`5eb5a7f`), restored with pointer kernels
(`50eb73c`).

#### A7: Remove redundant stream sync from sweep loops

**What**: Removed `hipStreamSynchronize(streams_[0])` after every bond in
`sweep_LR_full()` / `sweep_RL_full()`.

**Why**: The CPU-SVD path in `svd_split` already forces sync via pageable-memory
staging + an explicit `hipStreamSynchronize` before LAPACK.

**Result**: Negligible at `chi ≥ 128` (SVD dominates). Would matter with GPU SVD.
Commit: `6df2548`.

#### B2: Adaptive warmup

**What**: Warmup sweeps exit early when `dE < tol` after the first sweep.

**Result**: Saves 1–2 warmup sweeps when convergence is fast. No savings at
`L=64 chi=128 --warmup 1` (already minimum). Commit: `1ff1b07`.

#### B5: Skip polish when outer loop converged

**What**: If the outer PDMRG loop converges, skip the polish phase entirely.
A convergence-tracking bug was fixed in `d18ba03` (the original code compared
`energy_` vs `energy_prev` after the loop where both were already equal).

**Result**: Saves 1–2 full-chain sweeps when the outer loop converges. Rarely
triggers at typical parameters. Commits: `1ff1b07`, `d18ba03`.

#### MFMA-16 dimension padding

**What**: Pad `chi_max` to the next multiple of 16 for MI300X matrix-core FP64
tile alignment. All allocations use the padded dimension; actual bond dims stay
unpadded. Capped at `chi_max_user_` (not padded value) to avoid quality
regressions at the user-requested truncation.

**Why**: MI300X MFMA units operate on 16×16 FP64 tiles. Non-aligned dims waste
tile utilization.

**Result**: 5–10% improvement when `chi` is not already a multiple of 16.
Shipped in all `*-opt` variants. Commits: `e7153b7`, `bd4d09c`, `0815243`.

#### Strided batched Step-3 GEMMs (`*-opt`)

**What**: Replace the loop of `D·d²` individual GEMM calls in `apply_heff`
Step 3 with `rocblas_gemm_strided_batched`, exploiting the regular stride of
`R_env` slices. Applied to `dmrg-gpu-opt`, `dmrg2-gpu-opt`, `pdmrg-gpu-opt`.

**Result**: Works, measurable at small `chi`. At `chi=16` on `dmrg-gpu-opt`, a
threshold was added (commits `2a6e271`, `15a0407`) because the strided batched
path regressed for very small `chi`. `batch_count <= 2` is also restricted.

#### Per-segment worker stream pool (`pdmrg-gpu-opt`)

**What**: Each PDMRG segment gets a pool of worker HIP streams for dispatching
independent GEMM groups concurrently within environment updates.

**Result**: Modest improvement by overlapping independent small GEMMs within a
single segment's work.

#### Full-chain coupling sweep (over boundary-only)

**What**: After segment sweeps, run full LR + RL sweeps across the whole chain
to couple segments, instead of only resweeping ±W sites around boundaries.

**Why**: The boundary-only path failed — see §4.3, B1.

**Result**: Correct. This is the path shipped in `pdmrg-gpu/`. It's also the
thing that makes PDMRG slower than `dmrg2-gpu` on a single device, because the
full-chain sweep serializes the parallelism.

#### GPU-side tridiagonal eigensolver (`rocsolver_dsteqr`)

**What**: Replaced CPU LAPACK `dstev_` with `rocsolver_dsteqr` for the Lanczos
tridiagonal subproblem. Part of the "no CPU in the hot path" cleanup.

**Result**: Eliminates a D2H/H2D round-trip per Lanczos outer iter. Small win
(~µs per iter) but closes the last CPU dependency in the hot loop. Commits:
`e9ccfbb`, `e500d52`.

### 4.2 Optimizations that WORKED but didn't HELP

#### Newton-Schulz polar decomposition for canonicalization (`*-opt`)

**What**: `X_{k+1} = ½·X_k·(3I - X_k^H X_k)` replaces QR-based canonicalization.
Converges to 1e-10 in 5–10 iterations, pure BLAS-3.

**Result**: Correct on both left (tall/square `A = U·P`) and right (wide
`A = L·Q`) canonicalization. But: NS requires 5–10 GEMMs vs 1 QR, and
canonicalization is not the bottleneck — **SVD split is**. Modest improvement
that doesn't move the needle.

#### Block-Davidson eigensolver (`*-opt`)

**What**: Replaces Lanczos with Block-Davidson: maintain subspace, project `H`,
solve small eigenproblem, expand with preconditioned residual.

**Why**: Davidson can use BLAS-3 for subspace projection when block size > 1;
Lanczos is BLAS-1/2-dominated.

**Result**: Correct — converges to same energies. Within ±10% of Lanczos wall
time at all tested sizes. **No significant speedup** because Lanczos typically
converges in 15–20 matvecs for a single ground state, and Davidson uses
comparable matvecs but with more bookkeeping (QR restarts, projected
eigenproblem management).

#### Randomized SVD (Halko–Martinsson–Tropp)

**What**: `Y = theta @ Omega` (random projection) → QR → `B = Q^H @ theta` →
SVD of small `(r × n)` matrix. Available via `--rsvd`.

**Result**: Works, correct. Faster than full SVD when `chi` is large relative
to truncation rank. But the oversampling requirement (`r = chi_max + p`, `p ≈
10–20`) means savings are **modest** for typical DMRG bond dimensions.

### 4.3 Optimizations that FAILED (implemented, reverted or flagged OFF)

#### B1: Boundary-only coupling sweep (`pdmrg-gpu`)

**What**: After segment sweeps, resweep only `±W` sites around each segment
boundary instead of the whole chain.

**Result**: **Energy error 8.41 at L=32 chi=64** — completely failed to converge.
For 8 segments of 8 sites with `W=4`, boundary coverage is ~89% of the chain (so
barely any saving), but the segment sweeps disrupt global entanglement structure
enough that boundary-only coupling cannot recover.

**Lesson**: PDMRG segment sweeps disrupt the converged MPS in ways that require
**full-chain resweeping** to fix. Narrow boundary coupling is insufficient.
Commit `d18ba03` reverted it from the outer loop.

#### B1 (alternative): Single-direction coupling sweep

**What**: Coupling sweep does LR only (not LR+RL), halving the cost. Direction
alternates by outer iteration.

**Result**: Energy quality dropped: `-28.167` vs correct `-28.175`. Two-site
DMRG requires both sweep directions — LR grows `chi` from the left but doesn't
optimize the right canonical form. **Full LR+RL coupling is essential.**

#### Newton-Schulz split (NS_split) replacing SVD

**What**: Replace `svd_split` with Newton-Schulz polar → eigendecompose `P^H P`.
Singular values = `sqrt(eigvals(P^H P))`, right singular vectors = eigvecs.

**Why**: LAPACK `dgesvd` is inherently sequential (bulge chase). NS + eigendecomp
should saturate GPU compute.

**Result**: **Correct but slower than CPU SVD for `chi ≤ 256`**. The NS iteration
(5–10 GEMMs) + host-side eigendecomp (O(chi³) — same as SVD) takes more total
wall time than one LAPACK SVD call.

L=32 chi=128 seg=2: CPU SVD split 16.5s vs NS split ~20–25s.

**Lesson**: NS replaces one O(n³) factorization with another at higher constant
factors. The win would require `chi ≫ 256` where GPU GEMM dominates over LAPACK
SVD.

#### Cross-segment batched GEMM sweep (`pdmrg-gpu-opt` Item 4)

**What**: Replace thread-per-segment parallel sweep with a lock-step sweep that
batches GEMM calls across segments into single rocBLAS batched calls. Segments
with matching `(cL, cR)` are grouped for one batched dispatch.

**Result**: **Slower at all scales tested**, benchmarked across 19 configurations.

| Config                  | Baseline | Batched  | Ratio |
|-------------------------|----------|----------|-------|
| L=20 chi=50 seg=2       | 2.35s    | 3.04s    | 0.77× |
| L=32 chi=64 seg=2       | 9.70s    | 94.30s   | 0.10× |
| L=32 chi=128 seg=2      | 16.55s   | 231.81s  | 0.07× |
| L=32 chi=256 seg=2      | 59.91s   | 50.66s   | **1.18×** |
| L=64 chi=128 seg=2      | 72.23s   | TIMEOUT  | —     |

**Why it failed**:
1. Lock-step sweep serializes Lanczos BLAS-1 (`dot`, `nrm2`, `axpy`) onto a
   single stream, losing the concurrent execution thread-per-segment achieves.
2. Segments often have mismatched bond dims, preventing effective batching.
3. At high segment counts (`seg=8+`), baseline thread-per-segment mode fails to
   converge within timeout while batched finishes — this is a *convergence
   behavior* difference from lock-step vs async, not a performance advantage.

Status: implemented, defaulted **OFF**. Available via `--batched-sweep`.
Commit: `0359be3`.

#### Chebyshev-filtered subspace iteration (`pdmrg-gpu-opt` Item 5)

**What**: Replace Lanczos with Chebyshev polynomial filtering: 10-step Lanczos
for spectral bounds `[λ_min, λ_max]`, then degree-15 Chebyshev filter via
three-term recurrence. Energy via Rayleigh quotient. Sync-free, just repeated
`apply_heff` + BLAS-1.

**Result**: **Correct (1e-13 accuracy) but 1.9–11× slower than Lanczos.**

| Config             | Lanczos | Chebyshev | Ratio |
|--------------------|---------|-----------|-------|
| L=8 chi=32 seg=2   | 0.73s   | 1.40s     | 0.52× |
| L=20 chi=50 seg=2  | 5.27s   | 59.15s    | 0.09× |

**Why it failed**: fundamental algorithm-problem mismatch.
- Chebyshev does `degree × outer_iters` matvecs (up to 15 × 20 = 300).
- Lanczos converges in ~15 matvecs for a single ground state.
- Chebyshev filtering is designed for computing **many eigenvalues
  simultaneously** (as in DFT/Kohn-Sham), not a single ground state where Krylov
  methods are already optimal.

Status: implemented, defaulted **OFF**. Available via `--chebyshev`.
Commit: `7197d2e`.

#### Pinned-memory batched pointer upload (reverted → replaced)

**What**: Use `hipHostMalloc` pinned host memory + `hipMemcpyAsync` for batched
GEMM pointer arrays.

**Result**: Illegal memory access crashes at `L=64 chi=128`, LAPACK `dstev`
failures, NaN propagation. **It's a race** — see §5.3 for details. Replaced
with GPU-side pointer setup kernels.

#### Zero-sync Lanczos with GPU convergence detection

**What**: Detect Lanczos convergence on the GPU without D2H transfers, to
eliminate all CPU-side reads during inner loop.

**Result**: Reverted in commit `1517f29` after NaN propagation and convergence
regressions. The CPU convergence check is cheap; cutting it doesn't save
meaningful time.

### 4.4 The std::thread discovery (critical for PDMRG correctness + perf)

Without explicit `std::thread` driving each segment, `hipStreamSynchronize`
inside `optimize_bond` **blocks the CPU**, making all segments run sequentially
even when they use separate streams. This is not an optimization — it's a
correctness-of-parallelism requirement.

```cpp
auto parallel_sweep = [this](auto sweep_fn) {
    std::vector<std::thread> threads(n_segments_);
    for (int k = 0; k < n_segments_; k++)
        threads[k] = std::thread([this, k, &sweep_fn]{ sweep_fn(this, k); });
    for (auto& t : threads) t.join();
    for (int s = 0; s < n_segments_; s++)
        HIP_CHECK(hipStreamSynchronize(streams_[s]));
};
```

Also in this commit (2026-03-12):
- Removed `hipStreamSynchronize` before Lanczos (same-stream ops are already ordered).
- Replaced `printf` with `std::cout` (thread-safe).

### 4.5 Contraction-path observation (not yet applied on GPU)

Analysis in `.claude/.../memory/pdmrg-contraction-analysis.md`:

```
H_eff: result[a,p,q,f] = L[a,w,c]·theta[c,s,r,e]·W_L[w,m,s,p]·W_R[m,n,r,q]·R[f,n,e]

Our manual order (L → W_L → W_R → R):  2.3–3.8× more FLOPs than optimal.
Cotengra optimal (theta×R → W_R → W_L → L): starts from the right, avoids
  large intermediates.
```

At `chi=20–50`, cotengra Python overhead eats the FLOP savings (~5% slower than
raw `tensordot`). At `chi=200`, cotengra is 1.62× faster. **The GPU ports
currently use the manual order.** A path rewrite remains an open optimization —
potentially 2–3× at large `chi` if it compiles cleanly through rocBLAS.

---

## 5. Lessons and Pitfalls (read before debugging)

### 5.1 CPU SVD is the dominant bottleneck

At `chi=256`, CPU LAPACK SVD consumes **97–98% of per-sweep runtime** in
`dmrg2-gpu`. Confirmed by profiling:

```
Sweep 2: lanczos=467ms (990 iters) svd=38132ms env=26ms other=4ms
```

SVD of `(chi·d, d·chi) = (512, 512)` matrices takes ~302ms per LAPACK call; 126
SVDs per sweep = ~38s. **All Lanczos/GEMM/env optimizations combined touch only
2–3% of runtime** at these sizes.

GPU SVD (`rocsolver_dgesvd`) gives a 13% improvement on `dmrg2-gpu` at L=64
chi=256 (141.7s → 123.4s). But it made **PDMRG slower** (140s vs 113s) because
parallel segment sweeps produce smaller effective `chi`, where rocSOLVER's
launch overhead dominates.

### 5.2 Single-core CPU beats MI300X at `chi ≤ 50`

Of 29 configurations where CPU beats GPU, **27 (93%) are won by single-threaded
CPU alone.** Only 2 exceptions where threading flips the outcome:
1. Heisenberg L=64 chi=128: 1-thread loses by 15%, 4-thread wins by 13%.
2. Josephson L=48 chi=128: 1-thread 3.3× slower, 4-thread barely wins.

Why: GPU kernel launch overhead (~5–10 µs × hundreds of launches per Lanczos
iter) dominates when individual GEMMs are `50×50` matrices that complete in µs.
The MI300X's 81.7 TFLOPS FP64 matrix throughput is irrelevant at these sizes.

Threading patterns: **4 threads is the sweet spot** (20–40% speedup at
`chi ≥ 128`). **8 and 12 threads are harmful** in 77% of runs (contention).
Josephson `(d=5)` benefits most (up to 3.6× at chi=128).

**Paper implication**: GPU loses to launch overhead, not to threaded CPU BLAS.
Frame the negative result this way.

### 5.3 Pinned-memory async race (read before touching pointer uploads)

`hipMemcpyAsync` from **pinned** host memory is truly asynchronous. The DMA
engine reads the source buffer at **execution time**, not enqueue time. In a
tight Lanczos loop, the CPU overwrites the pinned buffer before the DMA reads
it, causing:
- Illegal memory access crashes.
- LAPACK `dstev` fed with corrupted alpha values (~1e-10 instead of O(1)).
- Intermittent NaN propagation.

The original `std::vector` (pageable) code worked because HIP stages
pageable-to-device transfers through an **internal pinned buffer**, making the
copy effectively synchronous w.r.t. the source.

**Solution**: bypass host-to-device entirely with GPU-side pointer setup
kernels (A3). **Moral**: switching pageable → pinned memory is not a free
performance win. It can introduce races if the surrounding code assumed
synchronous-like behavior. Audit carefully.

### 5.4 OpenBLAS 0.3.20 has a broken `dgesvd`

OpenBLAS 0.3.20 (Ubuntu 22.04 default) produces garbage SVD output for
non-square (especially tall-skinny) matrices. Both `dgesvd` and `dgesdd` return
`info=0` but give non-orthonormal `U` columns and ~130% reconstruction error.
`OMP_NUM_THREADS=1` partially mitigates (fixes `184×92`) but doesn't fix
`200×100+`. Square matrices are fine.

**Fix**: built OpenBLAS 0.3.28 from source, installed to `~/openblas-0.3.28` on
the remote MI300X. All GPU `CMakeLists.txt` prefer this over system LAPACK:

```bash
git clone --branch v0.3.28 https://github.com/OpenMathLib/OpenBLAS.git
cd OpenBLAS && make -j$(nproc) USE_OPENMP=0 NO_LAPACK=0
make PREFIX=~/openblas-0.3.28 install
```

`dmrg-gpu`'s workspace query was also fixed to check all `(m,n)` combinations,
not just square.

### 5.5 `chi` cap must respect user intent

`dmrg2-gpu-opt` / `pdmrg-gpu-opt` originally capped bond dims at the
MFMA-padded `chi_max_` instead of the user-supplied value. Fixed in
`0815243`, `bb5b6e4` — always cap at `chi_max_user_`. Padding is for alignment
only; it must never leak into the truncation criterion.

### 5.6 Accurate SVD for Stoudenmire boundary `V = Λ⁻¹`

Standard LAPACK SVD has good absolute accuracy but **poor relative accuracy**
for small singular values. When computing `V = 1/σ`, relative errors in small
`σ` are amplified by `1/σ²`, producing huge absolute errors in `V`. This makes
the boundary merge `Ψ' = ψ'_L · V · ψ'_R` a poor initial guess for Lanczos,
degrading convergence.

**Stoudenmire & White (arXiv:1301.3494, Appendix)** gives a recursive refinement:
1. Standard SVD: `M = A·Λ·B`.
2. Find smallest `p` where `λ_p/λ_1 < ε` (recommended `ε = 1e-4`).
3. If none, converged.
4. Compute `X = A† M B†` restricted to last `(n-p)` rows/cols.
5. Recursively SVD `X → Ã, Λ̃, B̃`.
6. Update `A[:,p:] = A[:,p:]·Ã`, `B[p:,:] = B̃·B[p:,:]`, `λ[p:] = λ̃`.

Use this **only** for `compute_boundary_V` where `V = 1/σ` is needed. Regular
`svd_split` can use standard LAPACK (no inversion).

### 5.7 Two-site polish sweeps for PDMRG segment convergence

Originally polish used single-site sweeps for speed. This was incorrect — it
didn't recover all the truncation error introduced during segment sweeps. Fixed
in `a681377` and documented in `2760e73` (the warmup is single-site for the
right reasons, the polish must be two-site).

---

## 6. Benchmark Summary

See `benchmarks/paper_results/mi300x/summary.csv` (737 rows) for the master
table. A condensed cpu-vs-gpu winner breakdown
(`wins_cpu_vs_gpu.csv`): **CPU wins 31, GPU wins 7. GPU win rate: 18%.**

The 7 GPU wins are all Josephson (d=5) at chi=128 or L large:

```
josephson,  8, 128  →  5.68× GPU   (dmrg-gpu)
josephson, 16, 128  →  2.67× GPU   (dmrg-gpu)
josephson, 32, 128  →  3.65× GPU   (dmrg-gpu)
josephson, 48, 128  →  1.08× GPU   (dmrg-gpu)
josephson, 64, 128  →  1.06× GPU   (dmrg-gpu)
```

**GPU wins cluster at d=5** because the Lanczos / env updates (which GPU is
good at) grow faster with `d` than the SVD step (which GPU is bad at at small
matrix sizes).

### Serial GPU comparison (Heisenberg OBC)

| L   | chi | dmrg-gpu | dmrg2-gpu | dmrg-gpu-opt | dmrg2-gpu-opt |
|-----|-----|----------|-----------|--------------|---------------|
| 8   | 32  | 0.31s    | 0.34s     | 0.68s        | 0.78s         |
| 32  | 128 | 4.21s    | 5.63s     | 5.09s        | 8.87s         |
| 64  | 128 | 24.04s   | 22.83s    | 26.63s       | 30.64s        |
| 100 | 128 | 176.80s  | 53.13s    | 157.13s      | 69.32s        |

### PDMRG vs dmrg2-gpu (Heisenberg, 8 segments)

| System        | dmrg2-gpu | PDMRG (optimized) | Speedup |
|---------------|-----------|---------------------|---------|
| L=32 chi=64   | 2.1s      | 5.5s                | 0.38×   |
| L=64 chi=128  | 27.0s     | 24.1s               | 1.12×   |
| L=64 chi=256  | 141.7s    | 113–119s            | 1.19–1.25× |
| L=128 chi=128 | 74.4s     | 76.0s               | 0.98×   |

**PDMRG wins modestly at `chi ≥ 128` and `L ≥ 64`**. Below that, parallel
segment sweeps disrupt convergence and the coupling sweep dominates.

### PDMRG Amdahl analysis (L=64 chi=128, warmup=1, outer=2)

| Phase                             | Time  | % of Total |
|-----------------------------------|-------|------------|
| Env build                         | 0.5s  | 2%         |
| Warmup (1 LR+RL sweep)            | 8.8s  | 37%        |
| Outer 0 (segments + coupling)     | 4.4s  | 18%        |
| Outer 1 (segments + coupling)     | 2.2s  | 9%         |
| Polish (2 LR+RL sweeps)           | 7.7s  | 32%        |
| **Total**                         | 24.1s |            |

Serial work dominates: warmup (8.8s) + coupling×2 (3.4+2.2s) + polish (7.7s) =
22.1s. Even with zero-cost segments, the serial ceiling is 27.0/22.1 = **1.22×**
over `dmrg2-gpu` (27.0s). This is Amdahl talking.

---

## 7. Build and Run

### Local development
- `cpu/pdmrg/venv/bin/python run_pdmrg_np1.py`
- `cpu/a2dmrg/venv/bin/python run_a2dmrg_np1.py`
- Unified benchmarks: `benchmarks/run.py {list|validate|benchmark|scale|report}`

### Remote MI300X (ROCm 7.2.0, gfx942)
- `ssh hotaisle@<current-ip>` (IP rotates; most recent memory says
  `23.183.40.74`, but verify — previous IPs .75/.79/.82/.87 are all dead).
- Persistent session: `tmux attach -t test_remote` or `paper_bench`.
- Working dir: `~/dmrg-implementations`.
- Build pattern (same for all GPU variants):

```bash
cd ~/dmrg-implementations/gpu-rocm/<impl>
mkdir -p build && cd build
cmake .. -DGPU_TARGETS=gfx942
make -j$(nproc)
```

- Typical correctness smoke test:

```bash
./dmrg_gpu          8 32 10                  # Heisenberg L=8 chi=32
./dmrg2_gpu         8 32 10 --josephson      # Josephson complex
./pdmrg_gpu         8 32 10                  # Heisenberg, 2 segments
./pdmrg_gpu_opt     8 32 10 --segments 2     # with opts
./<impl>_base       8 32 10                  # naive baseline
```

### Remote H100 (CUDA 12.x, sm_90)
- `ssh -o StrictHostKeyChecking=no ubuntu@217.18.55.100`
- Build:
```bash
cd gpu-cuda/<impl>
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=90
make -j
```
- `--allow-unsupported-compiler` is set in CMakeLists for GCC 15.

---

## 8. Documentation Index

### In-tree docs

| Path                                             | Purpose |
|--------------------------------------------------|---------|
| `README.md`                                      | Project front door |
| `docs/PROJECT_OVERVIEW.md`                       | **This file** |
| `docs/CPU_ARCHITECTURE.md`                       | CPU contraction paths, index conventions, MPO layout |
| `docs/optimizations.md`                          | Tier 1/2/3 GPU optimization ideas (research notes) |
| `docs/GPU_FIX_QUICK_REFERENCE.md`                | GPU-CPU parity fix cheatsheet |
| `docs/GPU_PARITY_FIX_PROMPT.md`                  | Full GPU parity-fix specification (long) |
| `docs/PDMRG_GPU_OPT_PROMPT.md`                   | PDMRG-OPT optimization plan |
| `docs/PDMRG_GPU_PERF_PROMPT.md`                  | PDMRG-GPU performance plan |
| `docs/PDMRG_GPU_PROMPT.md`                       | Initial PDMRG-GPU design doc |
| `docs/PDMRG_STOUDENMIRE_FIX_PROMPT.md`           | Stoudenmire boundary merge fix |
| `gpu-rocm/pdmrg-gpu/OPTIMIZATION_REPORT.md`      | Detailed pdmrg-gpu per-commit perf notes |
| `gpu-rocm/pdmrg-gpu-opt/OPTIMIZATION_REPORT.md`  | Detailed pdmrg-gpu-opt experiments (Tier 1/2) |
| `gpu-rocm/pdmrg-gpu/BOUNDARY_MERGE_FIX_PROMPT.md`| Boundary merge debugging history |
| `gpu-rocm/dmrg-gpu/GPU_NATIVE_DMRG_PROMPT.md`    | dmrg-gpu design |
| `gpu-rocm/dmrg-gpu/GPU_TWO_SITE_DMRG_PROMPT.md`  | dmrg2-gpu design |
| `benchmarks/README.md`                           | Benchmark CLI and registry |
| `benchmarks/BENCHMARK_STATUS.md`                 | Recent run status |
| `gpu-cuda/README.md`                             | CUDA port status + cuSOLVER gesvd m<n workaround |
| `paper/PAPER_PROMPT.md`                          | CPC manuscript brief |
| `paper/REVIEWER_PROMPT.md`                       | Peer review prep |
| `paper/main.tex`                                 | Manuscript source |

### Auto-memory (`~/.claude/projects/.../memory/`)

| File                                          | Contents |
|-----------------------------------------------|----------|
| `MEMORY.md`                                   | Index of everything below |
| `project_pdmrg_performance.md`                | std::thread discovery, PDMRG speedup table |
| `project_accurate_svd.md`                     | Stoudenmire recursive SVD algorithm |
| `project_target_application.md`               | Quantum-computing regime rationale |
| `project_benchmark_plan.md`                   | 724-config paper benchmark plan |
| `insight_single_core_cpu_beats_gpu.md`        | 93% of CPU-wins are single-thread |
| `lessons_openblas_svd_bug.md`                 | OpenBLAS 0.3.20 `dgesvd` is broken |
| `pdmrg-contraction-analysis.md`               | Manual vs cotengra FLOP counts |
| `reference_h100_access.md`                    | H100 SSH |

---

## 9. Open Problems / Future Work

### Active follow-up round (opened 2026-04-10)

After a 10-pair generate-and-critique loop over this document, one proposal
survived in revised form and two structural gaps were identified. See:

- `docs/followups/proposal_3_hip_graph_capture.md` — surviving HIP-graph
  capture proposal, narrowed to `apply_heff_two_site` only.
- `docs/followups/round_2_plan.md` — dedicated research round covering
  (a) SVD call-count reduction (not SVD speed), (b) persistent on-chip
  Lanczos kernel with theta cached in LDS across iterations, (c) novel
  parallel DMRG schemes beyond Stoudenmire PDMRG and Grigori–Hasan A2DMRG.

### Standing items

1. **Contraction path rewrite** — switch to cotengra-optimal order on GPU.
   Expected 2–3× at `chi ≥ 200`. Currently the manual L → W_L → W_R → R order
   does 2.3–3.8× more FLOPs.
2. **Multi-GPU scaling** — `pdmrg-multi-gpu` has partial results (27/44). VM
   died mid-run. Full benchmark pending.
3. **Larger benchmarks** — the regime where PDMRG parallelism pays off is
   `L ≥ 128 chi ≥ 256`. All current data is below this.
4. **Alternative coupling strategies** — full-chain LR+RL coupling is the
   bottleneck. Subspace expansion or density-matrix perturbation at boundaries
   might replace brute-force resweeping.
5. **Mixed-precision SVD** — single-precision for warmup/segments, double for
   polish. Could halve SVD time for ~90% of computation. Conflicts with the
   "never single precision" policy — would need explicit exception.
6. **GPU SVD tuning at `chi ≥ 512`** — rocSOLVER should dominate here but
   hasn't been profiled.
7. **CUDA benchmarks at scale** — H100 port compiles and passes `L=4`, but no
   large runs have been done.

---

## 10. Known Correctness Caveats (from README.md)

- **`pdmrg` np=1 early return**: runs serial quimb DMRG2 warmup, returns that
  energy without ever entering the parallel algorithm. "PDMRG np=1" and "A2DMRG
  np=1" benchmark numbers are therefore quimb DMRG2 numbers in disguise.
- **`pdmrg` boundary merge optimization disabled**: permanently, due to an
  "H_eff spurious eigenvalue problem" that was never resolved.
- **`a2dmrg` canonicalization policy**: skipped intentionally to preserve
  bond-dim structure for coarse-space linear combination. This is a design
  decision.

When interpreting CPU benchmark numbers, check the metadata schema in
`BENCHMARK_METADATA_SCHEMA.md`.
