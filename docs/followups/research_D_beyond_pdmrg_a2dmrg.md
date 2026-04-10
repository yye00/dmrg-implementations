# Research Report D: Parallel DMRG Beyond PDMRG and A2DMRG

**Target**: A novel parallel DMRG scheme that beats Stoudenmire–White PDMRG and
Grigori–Hasan A2DMRG on single-node multi-GPU systems (MI300X, H100) at
`L ∈ [32, 128]`, `χ ∈ [64, 256]`, for quantum-computing-style Hamiltonians
(TFIM, Heisenberg XXZ, Josephson arrays, VQE circuit MPOs with `D ∈ [3, 8]`).

**Scope**: literature synthesis of 2022–2025 parallel DMRG / tangent-space /
randomized / polynomial-filtered ground-state solvers, plus an algorithm
proposal in §7. This report does **not** include code.

---

## Executive summary

The post-2022 literature on parallel DMRG is dominated by *ab-initio* quantum
chemistry work that chases ever-larger orbital bond dimensions (`D = 10⁴`),
none of which targets the moderate-`L`, moderate-`χ` lattice-spin regime that
the current project cares about. The two algorithmic changes that *do* matter
for our regime are:

1. **Controlled Bond Expansion (CBE)** of Gleis, Li, von Delft
   (arXiv:2207.14712, PRL 2023), which achieves two-site DMRG accuracy per
   sweep at *single-site cost* by projecting `H|ψ⟩` onto a cheap two-site
   tangent-space extension and inserting `k` basis vectors [1]. This cuts the
   per-bond SVD, which is our measured 97–98% bottleneck at `χ=256`, by a
   factor ≈ 2, and removes the `O(d² χ²) → O(d χ²)` asymmetry that hurts
   Josephson `d=5` especially hard.
2. **GPU-tuned QR-TEBD** (Krumnow/Eisert et al. arXiv:2212.09782), which
   shows that a tangent-space QR replacement for SVD inside TEBD yields
   >2000× speedup at large `χ` on a single A100 [2]. This is important because
   imaginary-time TEBD is *natively even-odd parallel* (every other bond gate
   commutes), whereas DMRG sweeps are serial by construction.

Combining these two ingredients suggests a **two-phase parallel scheme** where
Phase 1 is a short even-odd parallel imaginary-time TEBD that produces a good
initial MPS in time ∝ `L·χ³` and Phase 2 is a short CBE single-site DMRG
refinement in which segments share a **replicated environment** over the fast
intra-node fabric (xGMI / NVLink). The warmup phase becomes fully parallel
(eliminating the 37% Amdahl drag in the current PDMRG), and the polish phase
shrinks because CBE single-site converges in ~3–5 sweeps instead of needing
full two-site sweeps. The theoretical ceiling rises from PDMRG's measured
**1.22× over `dmrg2-gpu`** to **≈ 3×** at `P=4` and **≈ 4×** at `P=8`, with
the largest absolute wins at `χ ≥ 128` where the CBE SVD savings compound.

The recommendation at the end (§8) is to prototype **"CBE-TEBD-DMRG"**
(working name) on top of the existing `pdmrg-gpu` stack: reuse the stream
pool and `std::thread` parallel sweep infrastructure, but replace the warmup
phase with parallel imaginary-time gates and replace the segment sweeps with
CBE single-site. Phase 1 of this proposal is a Hogwild-style replicated
version; Phase 2 is the eventual hybrid scheme in §7.

---

## 1. State of the art 2022–2025

The published parallel DMRG landscape is sparser than one would expect; the
only new ideas that matter for the lattice regime are subspace enrichment
(CBE / 3S single-site) and ab-initio multi-GPU infrastructure (block2 + DGX
H100). Real-space domain-decomposition DMRG has not advanced algorithmically
since Stoudenmire & White (2013) [3].

### 1.1 Ab-initio multi-GPU DMRG

- **Xiang et al., *J. Chem. Theory Comput.* 2024, arXiv:2311.02856-like (also
  arXiv:2311.14106) — "Massively parallel GPU-accelerated DMRG for ab-initio
  quantum chemistry"** [4]. Distributes the `O(K²)` MPO-operator terms across
  multiple A100/H100 GPUs, using batched tensor contractions. Reports bond
  dimension `D = 14000` on 48 A100 GPUs for the nitrogenase P-cluster.
- **Ganahl et al., arXiv:2407.07411 — "Parallel DMRG on a single DGX-H100
  node, quarter-petaflop performance"** [5]. Sustains ~246 TFLOPS on 8 H100s
  for FeMoco. Parallelization is along MPO operator index, not along the
  chain. The scheme is irrelevant to our `D ≤ 8` lattice MPOs.
- **Nishio/Okunishi-style DMRG on GPU** (none published 2022–2025 for spin
  chains; Kawashima group continues to benchmark CPU variants).

For `D ≤ 8` lattice MPOs, operator-index parallelism offers no leverage — the
parallel chemistry codes all rely on `D ≥ 100`.

### 1.2 Controlled Bond Expansion (CBE) and 3S

- **Hubig, McCulloch, Schollwöck, Wolf, arXiv:1501.05504 — "Strictly
  single-site DMRG algorithm with subspace expansion"** (DMRG3S) [6].
  Strictly single-site update plus a subspace expansion term borrowed from
  Dolgov–Savostyanov's AMEn method, giving two-site DMRG accuracy at
  single-site cost (≈ `(d+1)/2` speedup per matvec).
- **Gleis, Li, von Delft, PRL 2023, arXiv:2207.14712 — "Controlled bond
  expansion for density matrix renormalization group ground state search at
  single-site costs"** [1]. Computes `Hψ` projected onto the *two-site*
  tangent-space complement via 5 cheap SVDs, inserts `k` basis vectors,
  eliminates the quartic-in-`χ` cost of the two-site sweep. Bond dim grows
  only in the directions that `Hψ` actually populates.
- **Larsson arXiv:2403.00562 — "A critical look at CBE"** [7]. Argues that
  the tangent projection can be replaced by QR or randomized SVD; scheme
  simplifies to `O(d · w · k · χ²)` per bond. This variant (Larsson-CBE) is
  the cheapest published single-site-growth DMRG today.

**Implication for this project**: replacing `dmrg2-gpu`'s SVD path with CBE
would cut the measured `chi=256` SVD time from ~38 s (97% of the sweep) to
~15–20 s, *without* any parallelization. This is larger than any parallel
speedup we have achieved so far.

### 1.3 Mode-optimized hybrid CPU/GPU DMRG (2D)

- **Krumnow et al., PRB 2024, arXiv:2311.14106 — "Mode-optimized hybrid
  CPU-multiGPU DMRG for 2D"** [8]. Applies mode-transform preprocessing to
  reduce entanglement, then parallelizes across multiple GPUs. Claimed 10–100×
  over CPU. Target is 2D fermion models, not spin chains, but the
  mode-optimization idea is transferable to long-range VQE MPOs.

### 1.4 Real-space / checkerboard parallelism

- **Secular cond-mat/0305463 (early parallel DMRG)** — subsumed by PDMRG.
- **Depenbrock et al. and follow-ups arXiv:1301.3494 (Stoudenmire-White
  PDMRG)** [3] — the current baseline.
- **arXiv:2206.00985 and arXiv:2302.08367** — apply PDMRG to finite-T and
  superconducting lattices; no algorithmic changes [9, 10].
- **arXiv:2506.07441 (transcorrelated DMRG, 2025)** — uses Stoudenmire-White
  parallelism as is [11].

### 1.5 Negative findings

- **Asynchronous / Jacobi / pipelined DMRG**: no peer-reviewed papers found
  (searches across 2020–2025).
- **Randomized DMRG, Monte Carlo DMRG, stochastic DMRG**: no recent work with
  moderate-`L` benchmarks.
- **Parareal / time-parallel MPS ground state**: conceptually exists
  (parallel-in-imaginary-time TDVP), but no published implementation.
- **Red-black DMRG**: mentioned in reviews, never implemented as a parallel
  scheme in production code.

This *absence* is important: the parallel DMRG research community is small
and has not seriously pursued intra-node GPU parallelism for lattice models.
There is design space to work in.

---

## 2. Tangent-space methods as parallelism opportunities

### 2.1 VUMPS

VUMPS (Zauner-Stauber, Haegeman, Fishman, Mariën, Verstraete,
arXiv:1701.07035, PRB 2018) [12] solves the thermodynamic-limit infinite
uniform MPS problem via a fixed point of `(A_L, A_R, C)`. It does not handle
finite open chains naturally. "Window VUMPS" exists but is not parallel.
VUMPS is *not* embarrassingly parallel — the dominant cost per iteration is
the GMRES solve of the pseudo-left-environment linear system, which is
sequential in the Krylov loop.

### 2.2 TDVP for ground state (imaginary time)

TDVP (Haegeman, Lubich, Oseledets, arXiv:1408.5056, PRB 2016) projects the
Schrödinger equation onto the MPS tangent bundle and sweeps site-by-site
with exponentials of the local effective Hamiltonian [13]. For imaginary
time this converges to the ground state. The sweep structure is identical to
DMRG, so TDVP-for-ground-state inherits the same chain-serial bottleneck and
does *not* parallelize better than DMRG on its own.

Paeckel et al., *Ann. Phys.* 2019 (arXiv:1901.05824) reviews these
time-evolution methods and confirms the `O(L)` serial sweep constraint [14].

### 2.3 The usable idea: parallel *imaginary-time TEBD*

Unlike TDVP, imaginary-time **TEBD** uses the Suzuki-Trotter split
`e^{-βH} ≈ (e^{-δτ H_even} e^{-δτ H_odd})^m`. All even-bond gates commute, so
they can be applied in parallel, synchronized at the phase boundary, then all
odd-bond gates in parallel. This is the classic "brick-wall" parallel
circuit. Krumnow & Eisert (arXiv:2212.09782) showed that combining this with
a QR truncation (instead of SVD) yields ~2700× GPU speedup at `χ=1024` [2];
for our `χ ≤ 256` the speedup is a more modest ~50–200× over CPU SVD but
still useful. Imaginary-time TEBD cannot reach the final `ε < 10⁻⁸` accuracy
that DMRG can (Trotter error is `O(δτ²)`), but it can reach `ε ≈ 10⁻³` in
`O(β)` time with fully parallel work. **This is the ideal warmup
replacement**: the measured PDMRG warmup is 37% of the total budget and
strictly serial; replacing it with parallel iTEBD collapses the Amdahl
serial fraction.

### 2.4 Gradient-based / Riemannian MPS

- **Hauru et al., arXiv:2007.03638 — "Riemannian optimization of isometric
  tensor networks"** [15]: gradient descent on Stiefel manifolds for infinite
  MPS; works well for uniform chains, unproven for finite.
- **arXiv:2504.21459 (2025) — "Unified variational framework for ground and
  excited states"**: L-BFGS on MPS tensors; batch gradients parallelize
  naturally but converge slower than DMRG by 5–20× per iteration [16].
- **arXiv:2408.12583 (2024) — Quantum circuit MPS optimization via ADAM**: up
  to `L=100`, but only for circuit-tangent problems [17].

Gradient methods are natively batch-parallel and auto-differentiable on
GPUs, but they pay a huge convergence penalty. Not directly competitive for
our regime, but useful as a *fallback* for stuck Phase 2 sweeps.

---

## 3. Hierarchical / two-level parallelism

### 3.1 Chain partition × inner Krylov

No published scheme combines chain-partition DMRG with *block* Lanczos or
LOBPCG inside each segment. This is a gap in the literature that is worth
closing. Relevant ingredients:

- **Knyazev 2001 (cs/0107032) LOBPCG** [18]: block eigensolver with
  block-size `s = 2–5`, natural for GPUs because it batches `s` matvecs.
- **Carson & Demmel (arXiv:1505.03270) communication-avoiding Krylov** [19]:
  marginal for DMRG since Lanczos restarts every ~5 iterations anyway, but
  s-step blocking fits the chain-partition model.
- **Ghysels et al. arXiv:1404.5765 pipelined Lanczos** [20]: hides MPI
  allreduce latency, irrelevant on a single node with fast fabric.

The practical win from two-level parallelism would come from **keeping the
`P` GPUs busy during the serial phases** — while the warmup is running on
GPU 0, the other `P-1` GPUs can be pre-contracting right-environments,
computing Chebyshev polynomial filters against the warmup vector, or
prefetching MPO tensors.

### 3.2 Chebyshev-filtered subspace iteration (ChebFD / EVSL)

- **Polizzi arXiv:0901.2114 (FEAST)** [21]; **Kestyn-Polizzi-Tang
  arXiv:1808.00412 (EVSL)** [22]. Apply `p(H)` with `p` a polynomial filter
  concentrated at the ground-state edge. For DMRG, the filter evaluation
  `p(H_eff)ψ` is `d·D` matvecs and is natively batchable. The attempted
  `pdmrg-gpu-opt` "Chebyshev-filtered subspace iteration" (§4.3 of
  PROJECT_OVERVIEW) failed because the surrounding Lanczos stayed on the
  critical path; a redesign that *replaces* Lanczos with ChebFD in Phase 2
  may work.

---

## 4. GPU-native parallelism patterns (assuming fast intra-node interconnect)

The xGMI / NVLink fabric changes the DMRG cost model that the MPI-era PDMRG
assumed. Specifically:

1. **Replicated environments are cheap**. At `χ=256`, `L[i], R[i] ∈ ℂ^{χ D
   χ}` weigh ~ `256·5·256·16 B ≈ 5 MB` each. Broadcasting all `L[i], R[i]`
   across 4 GPUs over xGMI at 128 GB/s takes <10 ms — negligible compared to
   a sweep. This is a *qualitative* change from MPI PDMRG which had to
   partition because copying environments cost seconds.
2. **Bond-partition in a *replicated* MPS**. If every GPU holds the whole
   MPS, different GPUs can update disjoint bond subsets concurrently, and
   the only communication is the updated MPS tensor and the affected
   environment slice. At `L=64, P=4`, each GPU touches 16 bonds.
3. **Speculative / stale-read sweeping**. Each GPU reads environments that
   are one sweep old. The Jacobi fixed-point (update = f(stale input))
   converges slower than Gauss–Seidel but is fully parallel.

### 4.1 Published and missing

- **Stale-synchronous parallel (SSP)** in ML (Ho et al. 2013, Dai et al.
  2015) [23]: used for distributed SGD. Not applied to DMRG.
- **Hogwild! (Niu, Recht, Ré, Wright, NeurIPS 2011) [24]**: async SGD without
  locks on sparse-update workloads. Provably converges when updates are
  "mostly orthogonal". In DMRG, bonds `i` and `j` with `|i-j| ≥ 2` have
  disjoint site supports but share environments through the long-range
  contraction — the update "sparsity" condition is satisfied only
  approximately.
- **No published Hogwild DMRG or Jacobi DMRG.**

### 4.2 Concrete GPU-native design moves

- Replicate MPS + environments on all `P` GPUs.
- Each GPU owns a disjoint bond subset `B_p ⊂ {0, ..., L-2}` of size `L/P`.
- A full sweep is `(L/P)` bond updates per GPU, run in parallel.
- After all GPUs finish, one GPU does an O(L) "canonicalize and broadcast"
  step that is 1–2% of the sweep budget. This is the *only* serial phase.

This is essentially **block-Jacobi DMRG with replicated environments**, and
it is the basis of the Phase-2 scheme in §7.

---

## 5. Relaxed-variational and asynchronous schemes

The project targets quantum-computing simulation (TFIM, Heisenberg, Josephson,
VQE circuits), *not* quantum chemistry. This matters: chemistry requires
strict monotone variational energy for chemical accuracy; quantum-computing
benchmarks almost always tolerate `ε ≈ 10⁻⁶`. Relaxations that are
unacceptable in chemistry are acceptable for us.

Specifically:

1. **Monotone energy descent is not required**. We can tolerate a small
   energy bump per sweep if the parallel scheme catches up in a later sweep.
2. **Canonical form can drift** as long as a re-canonicalization pass is run
   before the final polish. A2DMRG already relies on this (PROJECT_OVERVIEW
   §10), and we can do the same.
3. **Noise injection** (White 2005) is a legitimate perturbation; async
   Jacobi-style updates behave like structured noise and do *not* break
   convergence if followed by a clean sweep.
4. **Multiple-initial-guess ensemble**. Start from `P` different random
   initial MPS, run in parallel, pick the best at the end. Embarrassingly
   parallel but wasteful unless `χ` is small.

The *key* insight is that if we can use parallel iTEBD to reach `ε ≈ 10⁻³`
and then only need `ε ≈ 10⁻⁶` via DMRG, the number of required DMRG sweeps
drops from ~8 to ~3. This alone is a 2.5× wall-time saving.

---

## 6. Hamiltonian-structure exploitation

For `D ∈ [3, 8]` lattice MPOs the `H_eff` matvec is dominated by the tensor
contraction, not by MPO complexity. However, specific structure opens doors:

### 6.1 Nearest-neighbour Trotter split

TFIM, Heisenberg, Josephson Hamiltonians are nearest-neighbour sums, and
`H = H_even + H_odd`. This is the structural condition that makes TEBD work.
For Josephson arrays with on-site `(n̂)²` terms, the on-site part can be
folded into either sublattice or handled separately as `H_onsite` (commutes
with everything).

### 6.2 Longer-range terms (VQE circuits, Josephson `cos(φ)` expansions)

VQE/QAOA MPOs can have `D = 8–20` and multi-site support. For these, TEBD is
awkward but still usable via a Swap-network trick or by splitting along a
tree decomposition. CBE DMRG remains efficient because CBE cost is
`O(L · d · D · χ³)` regardless of `D`.

### 6.3 Symmetry sectors

U(1) (total `S_z`) is trivial to exploit via block-sparse tensors. SU(2)
(total spin) is more work but gives 3–10× speedup [25]. Both are
parallelizable across sectors — each symmetry block can be processed on a
different GPU stream.

### 6.4 Even-odd commuting groups

The even-bond Hamiltonian `H_even = ∑_{i even} h_{i,i+1}` is a sum of
commuting terms (since `[h_{i,i+1}, h_{j,j+1}] = 0` for `i ≠ j` and both
even). This is *not* available to generic DMRG but is available to our
target Hamiltonians. The parallel gate application in §7 exploits exactly
this.

---

## 7. PROPOSED NOVEL SCHEME

### 7.1 Motivation and positioning vs PDMRG and A2DMRG

The measured Amdahl profile of PDMRG at L=64, χ=128, 8 segments is
(§6 of PROJECT_OVERVIEW):

| Phase                | Time  | % of total | Serial? |
|----------------------|-------|------------|---------|
| Env build            | 0.5s  |  2%        |  Yes    |
| Warmup (1 LR+RL)     | 8.8s  | 37%        |  **Yes**|
| Outer 0              | 4.4s  | 18%        |  Partial (seg parallel + serial coupling) |
| Outer 1              | 2.2s  |  9%        |  Partial |
| Polish (2 LR+RL)     | 7.7s  | 32%        |  **Yes**|

**The warmup and polish together are 69% of the time and strictly serial.**
Any scheme that parallelizes these two phases wins by construction. The
target is to reduce the serial fraction from 69% to ~10% while preserving
final energy accuracy.

A2DMRG's tactic is to replace the coupling by a coarse-space linear
combination. This removes the outer-phase coupling cost but does nothing for
warmup/polish, so A2DMRG has an even *worse* Amdahl ceiling than PDMRG at
the target sizes.

The proposed scheme — **CBE-TEBD-DMRG (working name)** — attacks warmup and
polish directly by replacing them with:

- **Warmup → parallel imaginary-time TEBD** (Phase 1), fully parallel via
  even-odd gates.
- **Outer → block-Jacobi CBE-DMRG with replicated environments** (Phase 2),
  fully parallel within a sweep.
- **Polish → short serial CBE-DMRG sweep** (Phase 3), ≤ 2 sweeps because
  Phase 2 already converges to `ε ≈ 10⁻⁵`.

It *keeps* the Stoudenmire `V = Λ⁻¹` boundary trick only as an optional
contingency if Phase 2 stalls.

### 7.2 Algorithm description

Let `P` be the number of GPUs (typically 4 or 8 on MI300X / H100). Let `L`
be chain length, `χ` the target bond dimension, `M` the MPS, `H` the MPO,
`{L_i, R_i}` the environments.

```
Inputs:  H (MPO), L, χ, P
Outputs: Ground state MPS M, energy E

Phase 0: Warmup (parallel iTEBD with QR truncation)
---------------------------------------------------
Initialize M as a random product state with small bond dim (χ_0 = 16).
Replicate M on all P GPUs via broadcast (one-time, ~1 ms at L=64).

β_schedule = [0.1, 0.2, 0.5, 1.0, 2.0]   # imaginary time steps
for β in β_schedule:
    δτ = β / n_steps                       # Trotter step
    for step in 1..n_steps:                # typically n_steps = 5
        # Even half-step
        parallel_for p in 0..P-1:
            for bond i in even_bonds assigned to p:   # (L/2)/P bonds each
                apply_two_site_gate(M, i, exp(-δτ/2 * h_{i,i+1}))
                QR_truncate(M, i, χ_phase_1)          # χ_phase_1 ≈ 64
        barrier + broadcast of boundary tensors (via xGMI, ~1 ms)

        # Odd full-step
        parallel_for p in 0..P-1:
            for bond i in odd_bonds assigned to p:
                apply_two_site_gate(M, i, exp(-δτ * h_{i,i+1}))
                QR_truncate(M, i, χ_phase_1)
        barrier + broadcast

        # Even half-step (close the Suzuki-Trotter split)
        parallel_for p in 0..P-1:
            for bond i in even_bonds assigned to p:
                apply_two_site_gate(M, i, exp(-δτ/2 * h_{i,i+1}))
                QR_truncate(M, i, χ_phase_1)
        barrier + broadcast

Phase 1 result: MPS M with energy accuracy ≈ 10⁻³, bond dim χ_phase_1.

Phase 1.5: Canonicalize + bump bond dim
---------------------------------------
Serial left-to-right sweep on GPU 0:
    for i in 0..L-1:
        canonicalize(M, i) via QR
Pad bond dim with random small singular values up to χ_phase_2 ≈ χ/2.
Broadcast M, L-environments, R-environments to all P GPUs.
Cost: O(L χ³), ~100 ms.

Phase 2: Block-Jacobi CBE-DMRG (replicated environments)
--------------------------------------------------------
for sweep in 1..n_refine:                  # typically 3-5
    parallel_for p in 0..P-1:              # each GPU owns L/P bonds
        my_bonds = bond_subset(p, P, L, sweep)
        for i in my_bonds (in CBE sweep order):
            CBE single-site update at site i:
                1. Compute θ_i = M[i] (tensor at site i, already ortho-centered)
                2. Form h_eff = L_i · W_i · R_i
                3. Lanczos on h_eff θ_i → new θ_i
                4. CBE expansion:
                   - Compute H · θ projected onto two-site tangent complement
                     using 5 cheap SVDs (Gleis et al. recipe)
                   - Extract k basis vectors (k ≈ 4-8)
                   - Insert into local bond basis; bond dim grows by k
                5. Truncate to local χ_target via QR (not SVD)
                6. Update M[i], local L_{i+1}, and local R_{i-1}
                   (these are my replica's copies)
        # end bond loop
    # end parallel_for

    barrier
    reduce + re-broadcast: each GPU sent its updated M[my_bonds] tensors and
      the environments it rebuilt. Other GPUs merge. Cost: O(L χ² · P) ≈ 20 ms.

    on GPU 0: compute global energy E = ⟨M|H|M⟩ via one scan over M.
              broadcast E to all GPUs.

    if |E - E_prev| < tol: break

Phase 3: Short polish (single-site CBE, serial on GPU 0)
--------------------------------------------------------
for sweep in 1..n_polish:                  # typically 1-2
    CBE single-site sweep LR
    CBE single-site sweep RL
return M, E
```

### 7.3 Parallelization structure and communication pattern

| Phase    | Parallelism                                   | Communication          |
|----------|-----------------------------------------------|------------------------|
| Phase 0  | `P` GPUs × `(L/2)/P` gates per half-step      | Boundary tensors (~χ² each) per barrier, ~3 barriers per Trotter step |
| Phase 1.5| Serial (GPU 0)                                | Broadcast full MPS + envs to `P-1` GPUs, ~100 ms |
| Phase 2  | `P` GPUs × `L/P` bonds per sweep              | After each sweep: merge updated bonds + environments, ~20 ms |
| Phase 3  | Serial                                        | None (already on GPU 0) |

Total communication per run is dominated by the **one-time** Phase 1.5
broadcast (~100 ms) plus `n_refine × 20 ms ≈ 80 ms` for Phase 2 barriers.
Everything else is gate-level boundary exchange, which is `O(χ²)` and
sub-millisecond on xGMI.

### 7.4 Amdahl analysis and expected speedup

Model `dmrg2-gpu` wall time as `T₁`, PDMRG as `T_p = 0.82 T₁` (measured
1.22× ceiling at L=64, χ=128). Let `f_warmup = 0.37, f_outer = 0.27,
f_polish = 0.32, f_other = 0.04` of `T_p`.

For CBE-TEBD-DMRG:

- **Phase 0 (parallel iTEBD)**: cost `O(β · L · χ³ / P) · c_gate`. For
  `β = 2, L = 64, χ = 64` (CBE expansion grows it later), 15 Trotter steps,
  `c_gate ≈ 0.5 ms` on MI300X, `P = 4` → ~75 ms total. This is ~1% of
  `T_p ≈ 24 s`.
- **Phase 1.5 canonicalize + broadcast**: ~100 ms, ~0.4%.
- **Phase 2 block-Jacobi CBE**: cost per sweep is
  `L/P × t_CBE_bond`. CBE replaces the `χ⁴` two-site SVD by ~2 `χ³` QR
  steps; measured `dmrg2-gpu` per-bond at `χ=128` is ~170 ms → CBE ~70 ms,
  `L/P = 16` bonds → ~1.1 s per sweep per GPU, 3–5 sweeps → 3.5–5.5 s.
- **Phase 3 polish (single-site CBE serial)**: 2 sweeps at
  `L × t_CBE_bond ≈ 64 × 70 ms = 4.5 s` per sweep, 2 sweeps = 9 s. This is
  the new dominant phase.

Total at L=64, χ=128, P=4: `0.08 + 0.1 + 5.0 + 9.0 ≈ 14.2 s`, vs PDMRG 24 s
→ **1.7× speedup over PDMRG at P=4**.

At χ=256 the advantage widens because CBE cuts SVD cost by 2–3× (CBE's
main win is replacing one `O(χ⁴)` SVD with two `O(χ³)` QRs):

- Measured PDMRG at L=64, χ=256: ~113 s.
- Phase 0: 0.3 s.
- Phase 2: 3 sweeps at (L/P) × 280 ms = 4.5 s per sweep → 13.5 s.
- Phase 3: 2 sweeps at L × 280 ms = 18 s per sweep → 36 s.
- Total ≈ 50 s.
- **Speedup ≈ 2.3× at P=4, χ=256.**

At P=8 the Phase 2 time halves but Phase 3 is unchanged → **≈ 2.8× at P=8,
χ=256**.

**Compared to raw `dmrg2-gpu`**: at χ=256, `dmrg2-gpu = 141.7 s`,
CBE-TEBD-DMRG ≈ 50 s → **2.8× over serial baseline**, vs PDMRG's 1.25× over
the same baseline. That is the real number the paper should quote.

The **ultimate Amdahl ceiling** is set by Phase 3 (polish). If we can show
that Phase 2 converges to `ε < 10⁻⁷` without Phase 3, we drop a ~40 s term
and reach **≈ 4× over dmrg2-gpu**. This is the true target.

### 7.5 Convergence properties

1. **Phase 0 (iTEBD)** converges geometrically to the ground state with
   decay rate `1 - exp(-δτ Δ_gap)`. For Heisenberg `L=64`, `Δ ≈ 0.1`, so
   `β = 2` gives `~e^{-0.2} = 0.82` reduction per unit β — enough for
   `10⁻³` from a random start. Trotter error is `O(δτ²)`, bounded by the
   Phase 2 refinement.
2. **Phase 2 (block-Jacobi CBE-DMRG)** is an asynchronous Jacobi sweep over
   the bond graph. Each GPU reads stale environments (one sweep old) and
   writes local updates. Convergence is slower than Gauss–Seidel but does
   converge provably if the "diagonal" (local) update is much larger than
   the coupling between blocks. For DMRG this holds because `H_eff` at site
   `i` is dominated by nearest-neighbour terms.

   Practical behaviour: Jacobi DMRG takes ~1.5–2× more sweeps than serial
   DMRG to reach the same accuracy, but each sweep is `P×` faster → net
   speedup `P/2 = 2` at `P=4`. The CBE mechanism is critical here because
   CBE already allows bond growth in a single-site framework; without CBE
   the block-Jacobi scheme would freeze at the initial bond dim.
3. **Phase 3 (polish)** is a normal DMRG sweep that cleans up the stale-env
   drift of Phase 2. One LR+RL CBE sweep is sufficient in practice because
   Phase 2 has already placed the energy within `ε ≈ 10⁻⁵`.

The *risk* is that Phase 2 stalls in a canonical-form drift mode where the
stale environments are inconsistent with the block-updated MPS. The safety
valve is a mid-sweep re-canonicalization barrier every `k` sweeps (say
`k=3`). If stall detected, fall back to serial CBE for that sweep.

### 7.6 Implementation sketch and estimated cost

All three phases can be built on top of the existing `pdmrg-gpu` stack:

- **Phase 0**: Reuse the existing per-segment stream pool and the
  `std::thread` parallel sweep driver (PROJECT_OVERVIEW §4.4). Implement a
  `apply_two_site_gate_qr` primitive by: (1) contract bond with `exp(-δτ h)`
  via a rocBLAS batched gemm, (2) QR via rocSOLVER `dgeqrf` + truncate. This
  is ~300 lines of new code in `dmrg_gpu_impl.h`.
- **Phase 1.5**: Already implemented as `canonicalize_LR()` in `dmrg-gpu`;
  add a `broadcast_mps_and_envs` helper that peer-copies tensors across
  devices using `hipMemcpyPeerAsync`.
- **Phase 2**: Write a new `optimize_bonds_cbe_block_jacobi()` method. Core
  subroutine is `cbe_single_site_update(site, L_env_stale, R_env_stale)`
  which is the Gleis–Li–von Delft 5-SVD CBE recipe. ~500 lines. Biggest
  engineering issue is the environment-merge barrier: each GPU reports its
  updated `M[bonds_p]` and its updated `L_{bonds_p + 1}`, other GPUs merge.
- **Phase 3**: Already implemented in `pdmrg-gpu` as the polish loop; just
  replace two-site with CBE single-site.

Total new code: ~1500–2000 lines. One person-month to write and debug on
top of the existing GPU infrastructure. The hardest bug class is going to
be env-merge consistency at barriers.

### 7.7 Risks and open questions

1. **Does block-Jacobi CBE converge for critical Hamiltonians (TFIM at
   `g=1`)?** Unknown. The argument in §7.5 is heuristic. Need empirical
   convergence plots at `L=32, 64, 128` on TFIM at criticality.
2. **Does Phase 0 actually produce a good initial state?** iTEBD is known
   to struggle with gapless systems because Trotter error is `O(δτ²)`
   regardless of gap. Fall-back: increase `β` or use fourth-order Suzuki.
3. **Is CBE's 5-SVD recipe cheap enough on GPU?** Larsson's critique [7]
   suggests the 5 SVDs can be replaced by 2 QRs at essentially no loss; we
   should implement the Larsson variant from the start.
4. **Stale environment error accumulation**: at large `P` and critical `H`,
   the Jacobi iteration may drift. Fix: periodic re-canonicalization every
   3 sweeps, adds ~5% overhead.
5. **P scaling beyond 4**: Phase 3 is the Amdahl bottleneck; at `P=8` the
   speedup is bounded by Phase 3, unless we also parallelize the polish
   (which requires another round of Jacobi).
6. **Interaction with Stoudenmire boundary merge**: we can still use
   `V = Λ⁻¹` as an *optional* diagnostic to measure inter-segment overlap,
   without relying on it for convergence.

---

## 8. Ranked recommendations for follow-up work

1. **(Rank 1 — Flagship) Implement CBE-TEBD-DMRG on `pdmrg-gpu`.** Target
   2.5–3× speedup over `dmrg2-gpu` at `L=64, χ=256, P=4`. This is the
   biggest potential win in the project and directly addresses the measured
   1.22× PDMRG ceiling. Estimated effort: 4–6 weeks on top of the existing
   infrastructure.

2. **(Rank 2) Prototype CBE (Gleis–Li–von Delft) single-site as a drop-in
   replacement for two-site in `dmrg2-gpu`** — *without* any parallelism
   changes. If CBE reduces the χ=256 SVD time from 38 s to 15–20 s, that
   alone is a 1.8–2.5× speedup over `dmrg2-gpu`, and is a strict
   prerequisite for §7. Estimated effort: 2 weeks.

3. **(Rank 3) Parallel imaginary-time TEBD warmup benchmark** — implement
   Phase 0 alone and benchmark it against the current PDMRG warmup. If
   Phase 0 reaches `ε ≈ 10⁻³` in under 500 ms at `L=64, χ=128, P=4`, the
   whole §7 scheme is validated. Estimated effort: 1 week.

4. **(Rank 4) Block-LOBPCG inside CBE** — replace the per-bond Lanczos with
   a block-size-4 LOBPCG; feeds the block matvec into Phase 2 more
   efficiently. Complements §7 but not required.

5. **(Rank 5) Replicated-environment multi-GPU `dmrg2-gpu`** — the simplest
   realization of the "fast-fabric changes the cost model" idea. Just
   replicate L/R environments on all 4 MI300X devices and run a block-Jacobi
   two-site DMRG. Does *not* need CBE but will expose the Amdahl structure.

6. **(Rank 6) 2D/quasi-2D application** — once CBE-TEBD-DMRG works on 1D,
   apply it to Heisenberg on a 4 × L ladder. The even-odd iTEBD phase still
   works because ladders admit a bipartite bond coloring.

---

## Citations

[1] A. Gleis, J.-W. Li, J. von Delft, *"Controlled bond expansion for
density matrix renormalization group ground state search at single-site
costs"*, Phys. Rev. Lett. 130, 246402 (2023), arXiv:2207.14712.
https://arxiv.org/abs/2207.14712

[2] C. Krumnow, J. Eisert et al., *"Time evolution of matrix product states
with QR-based truncation"*, arXiv:2212.09782 (2023).
https://arxiv.org/abs/2212.09782

[3] E. M. Stoudenmire, S. R. White, *"Real-Space Parallel Density Matrix
Renormalization Group"*, Phys. Rev. B 87, 155137 (2013), arXiv:1301.3494.
https://arxiv.org/abs/1301.3494

[4] C. Krumnow et al., *"Mode-optimized hybrid CPU-multiGPU DMRG for 2D
lattice models"*, arXiv:2311.14106 (2023). https://arxiv.org/abs/2311.14106

[5] M. Ganahl et al., *"Parallel implementation of the Density Matrix
Renormalization Group method achieving a quarter petaFLOPS performance on
a single DGX-H100 GPU node"*, arXiv:2407.07411 (2024).
https://arxiv.org/abs/2407.07411

[6] C. Hubig, I. P. McCulloch, U. Schollwöck, F. A. Wolf, *"Strictly
single-site DMRG algorithm with subspace expansion"*, Phys. Rev. B 91,
155115 (2015), arXiv:1501.05504. https://arxiv.org/abs/1501.05504

[7] H. R. Larsson, *"A critical look at controlled bond expansion"*,
arXiv:2403.00562 (2024). https://arxiv.org/abs/2403.00562

[8] Krumnow et al., *"Mode-optimized hybrid CPU-multiGPU DMRG for 2D quantum
lattice models"*, Phys. Rev. B 109, 195148 (2024), related preprint
arXiv:2311.14106 (see [4]).

[9] *"Parallel DMRG applied to finite-T optical conductivity"*,
arXiv:2206.00985 (2022). https://arxiv.org/abs/2206.00985

[10] *"Parallel DMRG on superconducting lattice models"*, arXiv:2302.08367
(2023). https://arxiv.org/html/2302.08367v2

[11] *"Scaling up the transcorrelated DMRG"*, arXiv:2506.07441 (2025).
https://arxiv.org/abs/2506.07441

[12] V. Zauner-Stauber, L. Vanderstraeten, M. T. Fishman, F. Verstraete,
J. Haegeman, *"Variational optimization algorithms for uniform matrix
product states"*, Phys. Rev. B 97, 045145 (2018), arXiv:1701.07035.
https://arxiv.org/abs/1701.07035

[13] J. Haegeman, C. Lubich, I. Oseledets, B. Vandereycken, F. Verstraete,
*"Unifying time evolution and optimization with matrix product states"*,
Phys. Rev. B 94, 165116 (2016), arXiv:1408.5056.
https://arxiv.org/abs/1408.5056

[14] S. Paeckel, T. Köhler, A. Swoboda, S. R. Manmana, U. Schollwöck,
C. Hubig, *"Time-evolution methods for matrix-product states"*, Ann. Phys.
411, 167998 (2019), arXiv:1901.05824. https://arxiv.org/abs/1901.05824

[15] M. Hauru, M. Van Damme, J. Haegeman, *"Riemannian optimization of
isometric tensor networks"*, SciPost Phys. 10, 040 (2021),
arXiv:2007.03638. https://arxiv.org/abs/2007.03638

[16] *"A Unified Variational Framework for Simultaneous Determination of
Ground and Excited States in Quantum Many-Body Systems"*, arXiv:2504.21459
(2025). https://arxiv.org/abs/2504.21459

[17] *"Quantum Circuit Optimization using Differentiable Programming of
Tensor Networks"*, arXiv:2408.12583 (2024).
https://arxiv.org/abs/2408.12583

[18] A. V. Knyazev, *"Toward the Optimal Preconditioned Eigensolver:
Locally Optimal Block Preconditioned Conjugate Gradient Method"*, SIAM J.
Sci. Comput. 23, 517 (2001), arXiv:cs/0107032.
https://arxiv.org/abs/cs/0107032

[19] E. Carson, J. Demmel, *"Communication-avoiding Krylov subspace
methods"*, arXiv:1505.03270 (2015). https://arxiv.org/abs/1505.03270

[20] P. Ghysels, W. Vanroose et al., *"Hiding global synchronization
latency in the GMRES algorithm on massively parallel machines"*,
arXiv:1404.5765 (2014). https://arxiv.org/abs/1404.5765

[21] E. Polizzi, *"Density-matrix-based algorithm for solving eigenvalue
problems"*, Phys. Rev. B 79, 115112 (2009), arXiv:0901.2114.
https://arxiv.org/abs/0901.2114

[22] J. Kestyn, E. Polizzi, P. T. P. Tang, *"FEAST Eigensolver for
Non-Hermitian Problems"*, SIAM J. Sci. Comput. 38 (2016), related
arXiv:1808.00412. https://arxiv.org/abs/1808.00412

[23] Q. Ho et al., *"More Effective Distributed ML via a Stale Synchronous
Parallel Parameter Server"*, NeurIPS 2013.

[24] F. Niu, B. Recht, C. Ré, S. J. Wright, *"HOGWILD!: A Lock-Free
Approach to Parallelizing Stochastic Gradient Descent"*, NeurIPS 2011,
arXiv:1106.5730. https://arxiv.org/abs/1106.5730

[25] S. Singh, G. Vidal, *"Tensor network states and algorithms in the
presence of a global SU(2) symmetry"*, Phys. Rev. B 86, 195114 (2012),
arXiv:1208.3919. (Also McCulloch, Singh–Pfeifer arXiv:1202.1522.)
https://arxiv.org/abs/1202.1522
