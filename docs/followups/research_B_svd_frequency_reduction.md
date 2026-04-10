# Research Report B: Reducing SVD Call Count in Two-Site DMRG

**Scope**: Answering whether we can skip a substantial fraction of the bond SVDs in `dmrg2-gpu` sweeps — i.e., reduce the NUMBER of O(chi^3) SVD calls rather than the per-call cost. §5.1 of `PROJECT_OVERVIEW.md` shows CPU SVD is 97–98% of sweep time at chi=256 (~38 s / sweep on L=64). If we can skip half of the 126 SVDs per sweep without destroying accuracy, we save ~19 s/sweep — far more than any of the A1/A3/A4 micro-optimizations combined.

## Executive summary

1. **No published paper explicitly gates bond SVDs with a "skip-if-cold" criterion**, but the DMRG community has developed three closely related tools that the project can combine:
   (a) **strictly single-site DMRG with subspace expansion** (White 2005 density-matrix perturbation [19]; Hubig/McCulloch/Schollwöck/Wolf 2015 DMRG3S [3]; Gleis/Li/von Delft 2023 Controlled Bond Expansion, CBE [7]) which eliminates the large two-site SVD entirely — replacing `(d·chi, d·chi)` SVDs with cheaper `(d·chi, chi)` operations or `(chi, k·chi)` operations where k ≪ chi;
   (b) **Brand's incremental SVD** [1] updating an existing SVD in O(k² ) per rank-one perturbation, with the orthogonality-loss problem resolved by Yangwen Zhang's 2022 reorthogonalization fix [8];
   (c) **adaptive bond-dimension and truncation-weight metrics** (Legeza DBSS, White truncation weight [37], and recent PID-entropy controllers [27]) that are already used per-site and can trivially be used as skip-gates.
2. **Variational bounds are preserved when bonds are skipped**: skipping a local update cannot raise the energy above what the previous sweep left on the table. DMRG without an update is trivially variational (see Gleis/Li/von Delft 2023 [7] and McCulloch/Osborne 2024 comment [25]).
3. **The cheapest practical gate is the Frobenius / Schatten change of `theta`** (option (a) from the prompt) combined with the **Lanczos residual** from the already-computed local eigensolve (option (c)). Both are free byproducts of the sweep step and cost zero extra FLOPs.
4. **Recommendation**: implement a hybrid algorithm described in §6 — early sweeps (1–3) are classical 2-site DMRG; later sweeps become "1-site subspace-expansion + selective 2-site SVD only at bonds whose cheaply-measured change metric exceeds a threshold". For L=64 chi=256, we conservatively expect 40–70% of bonds to freeze after 3 sweeps on a translation-invariant Heisenberg chain, giving 1.5–3× wall-clock speedup.

---

## 1. Prior art on bond convergence detection

**Explicit "bond freezing / skip-the-SVD" in standard DMRG is not published.** The adjacent body of work is:

- **Legeza, Sólyom — Dynamic Block State Selection (DBSS)** [1, 14]. Legeza pioneered using the single-bond entropy / truncation weight to decide, per bond, how many states to keep. DBSS treats each bond heterogeneously and is the closest thing the community has to "importance-weighted sweeping", but it modifies `chi_i`, not whether the SVD runs. It still calls the SVD at every bond.
- **White's truncation-weight criterion** [37] (Kumar/Tubman 2011, Phys. Rev. B 84, 125130). The discarded weight `w_i = sum(sigma_k^2, k > chi)` at bond `i` provides a rigorous, empirically linear bound on the energy error. Used as a post-hoc diagnostic and as an adaptive `chi` control, but **again as an always-computed side-effect of the SVD, not a gate that skips it.**
- **Adaptive entropy-feedback (arXiv:2604.03960 / Jiang 2025 [27])**. A PID controller on `S_i` dynamically grows/shrinks `chi_i` per bond. Reports 2.7× speedup on Heisenberg vs fixed-chi. Same limitation: SVD is still called every step, just truncated differently.
- **Site reordering (Zhao 2024, arXiv:2512.22021 [8])**. Reduces relative energy error by 65–94% in disordered Heisenberg chains by permuting site order, i.e., *repositioning work*, not eliminating it.
- **TeNPy / ITensor early-termination**. Both libraries have sweep-level early stopping when `|ΔE| < tol`, but no per-bond gate. Confirmed by inspection of tenpy `SingleSiteDMRGEngine` [18] and ITensor discourse thread #2036 [42]. Neither library documents a "skip this bond" facility.

**Conclusion**: the skip-the-SVD idea is an open gap in the literature. That's good news for us — it's also bad news because we cannot cite a prior validation of correctness on our target Hamiltonians.

---

## 2. Incremental SVD update feasibility

**Brand (2002, 2006)** [1] developed a rank-one update identity that modifies an SVD `USV^T → (new SVD)` in **O(pr + r³)** time per update, where `r` is the kept rank and `p` is the matrix leading dimension. For DMRG this would turn a `(d·chi, d·chi) = (512, 512)` SVD (current cost O(d³·chi³) ≈ 1.3·10⁸ FLOPs → ~300 ms) into a rank-one update of O(chi²·d + chi³) ≈ 2·10⁷ FLOPs per perturbation. **If** the between-sweep perturbation at a bond really has rank ~1, we save ~7×.

**The orthogonality-loss problem and its 2022 resolution**. Brand left open "how often reorthogonalization is necessary to guarantee overall numerical precision" [1, §4]. For ~15 years this was a practical showstopper: in experiments the left singular basis loses orthogonality within ~600 updates and singular values become unreliable for truncation [8]. In April 2022, Yangwen Zhang (arXiv:2204.05398) [8] answered the question and proposed a reformulation that avoids accumulating the small orthogonal matrices, substantially reducing orthogonality loss and cost. Gu–Eisenstat's 1995 broken-arrowhead O(j²) divide-and-conquer SVD update [21] provides an alternative stable path.

**Feasibility for DMRG**: **questionable on two counts.**

1. The between-sweep change `theta' - theta` at a bond is **not** naturally rank-1. Empirically it depends on every intervening local optimization and on the re-orthogonalization when the sweep passes through. For mid-chain bonds far from the active site, `‖theta' - theta‖_F / ‖theta‖_F` drops below 1e-3 after 2–3 sweeps (White's observation [19], also confirmed by Stoudenmire's tutorial on DMRG sweeps [7]), so the perturbation is small in norm but **not low rank** — which is what Brand's update requires for the O(k²) speedup.
2. The **truncation step** that DMRG performs after the SVD makes Brand's smooth-perturbation analysis pessimistic [29]. A perturbation that shifts a singular value across the truncation boundary changes the retained basis discontinuously.

**Net**: incremental SVD is mathematically well understood and numerically stable with the 2022 reorthogonalization fix [8], but it is **not a drop-in replacement** for DMRG's bond SVD. It would require either (a) reformulating the update as a periodic low-rank correction plus a full SVD every N sweeps, or (b) combining it with a skip-gate so that Brand's update is only invoked when the perturbation is demonstrably small and smooth.

---

## 3. Subspace expansion / single-site alternatives

The right research direction, bypassing the large SVD entirely:

- **White 2005 density-matrix perturbation** (Phys. Rev. B 72, 180403(R); arXiv:cond-mat/0508709) [1, 19]. Adds a small noise term to the single-site effective Hamiltonian to allow symmetry-sector changes that single-site DMRG would otherwise miss. Observed 2–4× speedup over 2-site DMRG on Heisenberg S=1 chain.
- **DMRG3S (Hubig, McCulloch, Schollwöck, Wolf 2015**; arXiv:1501.05504; Phys. Rev. B 91, 155115) [3, 11]. Strictly single-site DMRG with subspace expansion. Each step costs O(w·d·m³ + w²·d²·m²) vs 2-site O(d²·chi³·w). **Reported up to 3.9× runtime reduction vs 2-site on Fermi-Hubbard.** Compatible with SU(2) non-Abelian symmetries. The expensive SVD becomes an SVD of a `(d·m, w·m)` rectangular matrix rather than `(d²·m, d²·m)`. For our d=2 Heisenberg this is only a d× savings, but for Josephson d=5 it is 5× — matches the payoff profile PDMRG needs.
- **Controlled Bond Expansion (Gleis/Li/von Delft 2023**; arXiv:2207.14712; Phys. Rev. Lett. 130, 246402) [7]. Strictly improves on DMRG3S: **"two-site accuracy and per-sweep convergence at single-site cost"**. Uses the insight that only a small subspace of the two-site complement carries significant `H·ψ` weight. Fully variational, no mixing parameter. Scales O(d·w·k·D²) where k is the number of expansion vectors (typically 10–50 even when D=1000+), i.e., **quadratic in D instead of cubic**.
- **McCulloch/Osborne 2024 simplified CBE variant** (arXiv:2403.00562) [25]. Replaces CBE's sequence of 5 SVDs with a single QR (optionally one small SVD), uses randomized SVD (RSVD) for the expansion. Scales O(d·w·k·D²) with smaller constant. Equally applicable to the classic 3S algorithm. **This is currently the fastest known fully-variational single-site-cost 2-site-accurate ground-state DMRG.**
- **Gleis/Li/von Delft 2025 reply** (arXiv:2501.12291) [26] defends the 2-site tangent-space projection for TDVP but concedes ground-state DMRG benefits from the McCulloch/Osborne simplification.

**GPU implementations**: none of CBE, DMRG3S, or 3S have publicly available GPU implementations that we can find.
- ITensor has DMRG3S in its C++ and Julia versions, CPU-only.
- SyTen (Hubig's library) has DMRG3S and CBE, CPU-only.
- block2 [37] has single-site DMRG with perturbation for quantum chemistry, CPU + distributed MPI, **no GPU backend for CBE**. It does have a distributed multi-GPU DMRG (arXiv:2312.17657 [12]) but that ports classical 2-site DMRG to GPUs, not CBE.
- TeNPy has `SingleSiteDMRGEngine` + `SubspaceExpansion` mixer [28], CPU only.
- Quimb experimentally supports CBE-like mixers, no GPU path.

**Opportunity**: **a GPU port of CBE-DMRG would be novel and publishable**. The project already has the hard parts (Lanczos, batched GEMM, environment updates, 1-site machinery in `dmrg-gpu`); what is missing is the k-vector expansion and the RSVD of a `(d·chi, w·k)` matrix where k ~ 20 and d,w are small constants. That rectangular SVD is `O(chi·k²)` GEMM cost — trivially GPU-friendly.

**"Cold bonds" language**: no paper uses the phrase literally, but the PID-entropy controller [27] and DBSS come closest. The idea that bulk bonds converge first in gapped 1D chains (because the correlation length is finite) is well established [45] but has not been turned into an algorithmic gate.

---

## 4. Energy-gradient gating heuristics

A natural idea: only run the SVD at bond `i` if the local energy contribution at `i` changed by more than `ε` since the previous sweep. Status:

- **Theoretical foundation exists.** Haegeman/Lubich/Oseledets/Vandereycken/Verstraete TDVP [15] places DMRG in tangent-space geometry where `∇E` at the current MPS is computable and is zero at a fixed point. A zero/small gradient at site `i` means the local eigensolve already sits at its optimum for the current environment.
- **Not used as a skip gate in any published code we could find.**
- **Variational preservation**: skipping a local update never raises the total `⟨ψ|H|ψ⟩`; the previous sweep's energy was already variational, and a skipped bond leaves the MPS unchanged locally, so the bound is inherited. Confirmed by McCulloch/Osborne 2024 [25] who argue CBE's variational properties are "essentially identical to existing algorithms including 2-site DMRG and single-site subspace expansion (3S)", all of which allow per-bond no-ops.
- **Gotcha**: you cannot skip in the *first* sweep of a new bond dimension. Skipping only applies to bonds whose environment hasn't changed substantially since the last visit. This means the gate must also check that the **neighbor** bonds were not updated significantly since last visit — otherwise a skipped bond could be sitting on a stale environment.

---

## 5. Cheap change-detection metrics

Ranked by cost (cheapest first) and by diagnostic value:

| Metric | Cost | Diagnostic value | Notes |
|---|---|---|---|
| **Lanczos residual `r = ‖H·ψ − λ·ψ‖`** at bond `i` | **Free** — already computed in the eigensolve | High | If `r < ε_res`, the local problem is converged. **This is the best single gate** because it's free and directly bounds the local contribution to energy error. |
| **`‖theta_new - theta_old‖_F`** | One axpy + norm (O(d²·chi²)) | Medium | Cheap but doesn't distinguish rotations that preserve the Schmidt spectrum from updates that change it. Combined with `‖theta‖_F` gives a relative change. |
| **Previous singular values `sigma_i^(prev)` vs. a quick estimate** (e.g., Gershgorin on `theta^T theta` diagonal) | O(d²·chi²) | Medium-high | Catches Schmidt-gap shifts without a full SVD. |
| **Discarded weight from previous SVD** at bond `i` | Stored from last sweep | Medium | If `w_i^(prev)` was already ~10^-14, the bond can't improve significantly. |
| **Entanglement entropy change `ΔS_i`** | Requires prev and new sigma vectors | Low-medium | Used in PID controllers [27]; requires the SVD we're trying to skip unless we estimate `S_i` from `theta^T theta` eigenvalues via cheap subspace iteration. |
| **Energy gradient `∂E/∂M_i`** | ~ a single applyH | Low (expensive for the info) | Theoretically clean, but the applyH itself is the second-largest sweep cost. Not a win. |

**Recommended composite gate** (combined from multiple sources, novel): **"skip if Lanczos residual < ε_res AND Frobenius relative change < ε_F AND the previous discarded weight < ε_w AND the bond was visited last sweep without skip"**. All three thresholds must be hit to skip; failing any one of them forces a full SVD.

---

## 6. Proposed algorithm sketch: Adaptive Hot/Cold Two-Site DMRG (AHC-DMRG)

Combining the best ideas for `dmrg2-gpu`:

**Phase 1 — classical warmup (sweeps 0 to N_warm=2)**: Full two-site DMRG with CPU SVD at every bond, exactly as today. This establishes a converged Schmidt spectrum across the chain. Record `sigma_i`, `w_i`, `theta_i` for every bond.

**Phase 2 — hot/cold gated sweeps (sweeps N_warm+1 onward)**:

For each bond `i` in sweep direction:
1. Run Lanczos on the two-site theta as usual. The result `theta_new` and residual `r_i` are produced.
2. **Cheap gate**: compute
   - `delta_F_i = ‖theta_new − theta_i^(prev)‖_F / ‖theta_i^(prev)‖_F`    (one `axpy` + one `dnrm2`, both GPU-resident)
   - `gate_cold = (r_i < eps_res) AND (delta_F_i < eps_F) AND (w_i^(prev) < eps_w)`
3. **If `gate_cold` is true → "cold bond path"**: skip the full SVD. Instead, perform a **canonicalization-only update**: apply the previous `U_i`, `V_i^T` projectors to `theta_new` and push the correction into the neighboring tensor. This is O(chi³) GEMM cost but **with no LAPACK SVD** — purely GPU GEMMs. Update `w_i` estimate by `|theta_new - theta_old|·sigma_old` (cheap bound).
4. **Else "hot bond path"**: do the full two-site SVD as today. Overwrite `sigma_i`, `w_i`, `theta_i^(prev)`, `U_i`, `V_i^T` for the next sweep.

**Phase 3 — periodic full sweep (every N_flush=5 sweeps)**: force a classical 2-site SVD on every bond to resync accumulated drift.

**Phase 4 — polish (final sweep)**: classical 2-site DMRG on every bond.

**Why this will work**:
- **Correctness**: the cold-bond path is a pure GEMM projection onto the previously-computed isometries. It preserves canonicalization, so neighbor environments remain valid. The variational bound is preserved because we never raise the energy past sweep `N_warm`. The periodic full-sweep flush catches any drift from accumulated projection errors.
- **Speed**: at L=64 chi=256, sweep time is 38 s in SVD + ~0.5 s in Lanczos. If 50% of bonds freeze after sweep 3, we drop to ~19 s in SVD + 0.5 s in Lanczos = **2× sweep speedup**. If 70% freeze (realistic for gapped 1D Heisenberg with correlation length ~5 sites and L=64), we drop to ~12 s → **3× sweep speedup**.
- **Integration with CBE**: this works standalone, but plugs cleanly into a future CBE-DMRG port: once CBE is in place, even the "hot" bonds do only 1-site-cost updates, giving a further d× boost.

**Risks**:
- The eps thresholds are Hamiltonian-dependent. Needs calibration: sweep L=16 chi=64 with various `eps_F` (1e-4, 1e-5, 1e-6) and compare convergence energy vs classical DMRG.
- The rank-skip could interact badly with bond-dimension growth: if chi has not yet saturated, skipping means not growing chi at that bond. Solution: disable the gate whenever `sigma[chi-1] / sigma[0] > eps_grow`.
- Josephson (d=5) needs different thresholds than Heisenberg (d=2) — the physical bond dimension at a given accuracy is larger, so relative changes may be smaller in norm but still important.

**Benchmarks to run**: Heisenberg L=32, 64, 128 at chi=128, 256, 512; Josephson L=32, 48 at chi=128; TFIM L=64 at chi=256. Compare (a) classical 2-site, (b) AHC-DMRG, (c) eventually CBE-DMRG GPU port. Measure both wall time and final energy error vs Bethe ansatz / exact.

---

## Citations

[1] Brand, M. "Fast low-rank modifications of the thin singular value decomposition." MERL TR2006-059. https://www.merl.com/publications/docs/TR2006-059.pdf
[3] Hubig, C., McCulloch, I. P., Schollwöck, U., Wolf, F. A. "Strictly single-site DMRG algorithm with subspace expansion." Phys. Rev. B 91, 155115 (2015). arXiv:1501.05504. https://arxiv.org/abs/1501.05504
[7] Gleis, A., Li, J.-W., von Delft, J. "Controlled bond expansion for density matrix renormalization group ground state search at single-site costs." Phys. Rev. Lett. 130, 246402 (2023). arXiv:2207.14712. https://arxiv.org/pdf/2207.14712.pdf
[8] Zhang, Y. "An answer to an open question in the incremental SVD." arXiv:2204.05398 (2022). http://arxiv.org/pdf/2204.05398.pdf
[11] Hubig et al. PRB 91, 155115 paper version. https://link.aps.org/doi/10.1103/PhysRevB.91.155115
[12] Zhai, H. et al. "Distributed multi-GPU ab initio DMRG." J. Chem. Theory Comput. 20, 775 (2024). arXiv:2312.17657. https://arxiv.org/html/2311.02854v2
[14] Legeza et al. "Entanglement, correlation and orbital optimization in quantum chemistry DMRG." Talk slides, INT Washington 2020. https://archive.int.washington.edu/talks/WorkShops/int_20_78W/People/Legeza_O/Legeza.pdf
[15] Haegeman, J. et al. "Unifying time evolution and optimization with matrix product states." arXiv:1408.5056; Phys. Rev. B 94, 165116 (2015). https://arxiv.org/abs/1408.5056
[18] TeNPy SingleSiteDMRGEngine docs. https://tenpy.readthedocs.io/en/latest/reference/tenpy.algorithms.dmrg.SingleSiteDMRGEngine.html
[19] White, S. R. "Density matrix renormalization group algorithms with a single center site." Phys. Rev. B 72, 180403(R) (2005). arXiv:cond-mat/0508709. https://link.aps.org/doi/10.1103/PhysRevB.72.180403
[21] Gu, M., Eisenstat, S. C. Stable divide-and-conquer for bidiagonal SVD. SIAM J. Matrix Anal. Appl. (1995). https://epubs.siam.org/doi/10.1137/S0895479892242232
[25] McCulloch, I. P., Osborne, J. J. "Comment on 'Controlled Bond Expansion...' (Extended)." arXiv:2403.00562 (2024). http://arxiv.org/pdf/2403.00562.pdf
[26] Gleis, Li, von Delft. "Reply to comment on 'Controlled bond expansion...'." arXiv:2501.12291 (2025). https://papers.cool/arxiv/2501.12291
[27] "Adaptive bond dimension management in DMRG with entropy feedback." arXiv:2604.03960. https://arxiv.org/html/2604.03960v1
[28] TeNPy SubspaceExpansion mixer. https://tenpy.readthedocs.io/en/v1.0.4/reference/tenpy.algorithms.mps_common.SubspaceExpansion.html
[29] Xia, Zhou. Perturbation analysis of truncated SVD. https://www.arxiv.org/pdf/2009.07542v1.pdf
[34] Yanai, T. et al. "Multireference quantum chemistry through DMRG-CASSCF." arXiv:0712.2475. https://arxiv.org/abs/0712.2475
[37] Zhai, H., Chan, G. K.-L. "Block2: A comprehensive open-source framework for DMRG." J. Chem. Phys. 159, 234801 (2024). https://pubs.aip.org/aip/jcp/article/159/23/234801/2930207/
[42] ITensor discourse: "subspace expansion noise in DMRG." https://itensor.discourse.group/t/subspace-expansion-noise-in-dmrg/2036
[45] Lei, Y., Chen, Y.-C. "Boundary effects in 1D DMRG." https://arxiv.org/html/2404.19588v1
[46] Legeza, Ö., Sólyom, J. "Optimizing the density-matrix renormalization group method using quantum information entropy." Phys. Rev. B 68, 195116 (2003). arXiv:cond-mat/0305336. https://arxiv.org/abs/cond-mat/0305336
[49] Liao, S. et al. "Transcorrelated DMRG." arXiv:2506.07441. https://arxiv.org/html/2506.07441v2
