# Round 3 — Pair 04: Block-Jacobi Single-Site CBE Convergence Risk

**Scope:** refinement of R2-2 Phase 2 (block-Jacobi CBE single-site sweeps with
replicated environments), ref. `docs/followups/research_D_beyond_pdmrg_a2dmrg.md`
§7.2 and `docs/followups/round_2_plan.md` §Follow-up R2-2.

**Role:** generate-and-critique pair. One voice defends the block-Jacobi choice;
the other tears it apart.

**TL;DR:** Block-Jacobi DMRG has **zero published convergence results**. The
closest relative (A2DMRG, arXiv:2505.23429, May 2025) explicitly added a
**global coarse-space correction step on top of its Jacobi-like local phase**
*because pure Jacobi was deemed insufficient*. R2-2 Phase 2 as written drops
that safety net. Without a line-search / damping / coarse-correction fallback,
the scheme is a convergence-rate gamble, and at P=4 the Amdahl math requires
that block-Jacobi cost no more than ~1.8× the serial sweep count or the whole
speedup collapses.

---

## 1. Block-Jacobi CBE single-site sweep — concrete pseudocode

Let `P` workers (GPUs) each own a contiguous site range `S_p = [p·L/P, (p+1)·L/P)`
for `p = 0..P-1`. Before the sweep, every worker holds a **full replicated copy**
of the MPS `M` and of the complete left/right environment stacks `{L_i, R_i}`,
all `i ∈ [0, L]`. (Cost analyzed in §3.)

```
Procedure BlockJacobi_CBE_Sweep(M_in, L_env_in, R_env_in, H_MPO, P, k_cbe, chi_cap):
    # All workers start from the same global state
    broadcast M_in, L_env_in, R_env_in to all P workers
    tol_local = 1e-8   # per-site inner eigensolver

    parallel_for p in 0..P-1:                       # ≈ L/P sites per worker
        M_p     = copy(M_in)                        # local scratch MPS
        L_env_p = copy(L_env_in[: p·L/P + 1])       # stale for everything left of S_p
        R_env_p = copy(R_env_in[(p+1)·L/P :])       # stale for everything right of S_p

        # Sweep direction inside the segment is GAUSS-SEIDEL within S_p.
        # Jacobi only at segment boundaries.
        for i in S_p (left-to-right):
            # 1. Single-site CBE update at site i
            H_eff   = apply_heff_single_site(L_env_p[i], H_MPO[i], R_env_p[i-(p+1)·L/P+1])
            theta_i = Lanczos(H_eff, M_p[i], tol_local, max_iter=20)

            # 2. CBE expansion (Gleis-Li-von Delft, or Larsson QR variant)
            #    Produces k_cbe extra basis directions from the tangent space
            U_k     = CBE_tangent_expand(theta_i, L_env_p[i], R_env_p[i-(p+1)·L/P+1],
                                         H_MPO[i], H_MPO[i+1], k=k_cbe)
            M_p[i]  = QR_truncate(concat(theta_i, U_k), chi_cap)

            # 3. Update LOCAL L_env_p forward (Gauss-Seidel within segment)
            L_env_p[i+1] = build_L_env_step(L_env_p[i], M_p[i], H_MPO[i])

        # At this point, M_p[S_p] is updated. R_env_p within S_p is stale
        # relative to other workers' updates because those workers ran in parallel.

    # ==== SYNCHRONIZATION BARRIER ====
    # Collect updated tensors from each worker:
    for p in 0..P-1:
        M_out[S_p] = M_p[S_p]       # disjoint writes, no conflict on sites
    # NOTE: at segment boundaries site (p+1)·L/P the bond connects two segments
    # that have never seen each other's updates. This is the dangerous region.

    # Re-canonicalize (serial, on GPU 0, ~100 ms at L=64)
    M_out = canonicalize_LR(M_out)

    # Rebuild environments from scratch. This is the ONLY way to ensure
    # self-consistency; incremental merge of worker-local environments
    # would propagate the stale-env error.
    L_env_out, R_env_out = rebuild_all_environments(M_out, H_MPO)

    # Compute global energy
    E = energy_from_envs(L_env_out, R_env_out, M_out, H_MPO)
    return M_out, L_env_out, R_env_out, E
```

### Key properties of this sweep

1. **Inside a segment, the update is Gauss-Seidel** — site `i+1` sees the newly
   updated site `i`. This is the Stoudenmire-White trick at the segment level.
2. **Between segments, the update is pure Jacobi** — worker `p` never sees
   worker `p+1`'s updates within this sweep. Segment `p+1` is still reading the
   old `L_env[p·L/P]` built from the un-updated sites in `S_p`.
3. **The barrier is non-trivial.** Rebuilding the full environment stack after
   every sweep is `O(L · χ² · D)` work and cannot be overlapped with the next
   sweep. This is the new serial tail. At `L=64, χ=256, D=5` that is
   roughly `100 ms`, consistent with R2-2's §7.3 table.
4. **No V=Λ⁻¹ bridge.** Unlike Stoudenmire PDMRG, there is no explicit boundary
   operator that enforces continuity at segment boundaries. The only
   reconciliation is the global canonicalize + env rebuild. This is a **major
   departure** from the only published parallel DMRG that is known to converge.

---

## 2. Convergence rate argument

### 2.1 Theoretical: spectral radius of the block-Jacobi iteration

Classical numerical analysis (Saad, *Iterative Methods for Sparse Linear
Systems* §4) says: for a fixed-point iteration `x^{(k+1)} = G x^{(k)}` with
spectral radius `ρ(G) < 1`, the error decays as `‖e^{(k)}‖ ~ ρ(G)^k`. For
Jacobi vs Gauss-Seidel applied to the same matrix `A`,
`ρ_Jacobi² = ρ_GS` for consistently ordered matrices (Young's theorem) — so
**Jacobi takes 2× as many iterations as Gauss-Seidel** in the well-behaved
case.

DMRG is **not** a linear system, and the "iteration matrix" is not explicit,
but the fixed-point framing applies: DMRG is an ALS iteration on a variational
manifold, and the local site update is a contraction whose Lipschitz constant
depends on (a) the spectral gap of `H_eff`, (b) the correlation length of the
MPS, and (c) the staleness of the environments.

For Heisenberg at `L=64`, spectral gap `Δ ≈ 0.1`, correlation length
`ξ ≈ 6–8`. If the segment length `L/P ≥ 2ξ`, the "information horizon" of one
sweep covers each segment, and the stale-env coupling between segments is
exponentially suppressed in `L/P - ξ`. At `P=4, L=64`, segment length is 16,
which is ~2–3× the correlation length — marginal but plausible.

For TFIM at criticality (`g=1`) `ξ` diverges as `L → ∞`, the information horizon
argument fails, and block-Jacobi is likely to stall. **Gapless critical
Hamiltonians are the known failure mode.**

### 2.2 Empirical: what published results say

Summary from perplexity literature survey (citations below):

- **No peer-reviewed paper implements block-Jacobi DMRG with stale
  environments.** This is the single most important finding. The search covered
  2010–2026. It is not a search-term gap; block-Jacobi MPS ALS is simply not
  in the literature.
- **Stoudenmire-White PDMRG (arXiv:1301.3494):** does **not** do block-Jacobi.
  It uses Gauss-Seidel inside each segment **plus an explicit V=Λ⁻¹ merge at
  segment boundaries after every segment sweep.** Reported ~10× speedup with
  10× compute on chemistry benchmarks — but no per-segment iteration-count
  table.
- **A2DMRG (Naumov/Grigori/Hasan, arXiv:2505.23429, May 2025):** additive
  Schwarz-inspired. Local phase is **Jacobi-like** (independent domain solves
  with fixed interfaces), but the authors **explicitly add a global
  coarse-space correction step** because pure Jacobi is insufficient.
  Reported speedup 2–8× on strongly correlated molecular systems.
  **The authors acknowledge in-paper that their scheme "exhibits slower
  convergence in terms of global iterations compared to classical DMRG"**
  and that the Jacobi-vs-GS analogy is exactly why.
- **CBE (Gleis-Li-von Delft, PRL 130 246402, arXiv:2207.14712):** reports 2.5×
  speedup over single-site DMRG3S and 3.9× over two-site DMRG on Fermi-Hubbard.
  The paper **claims compatibility with real-space parallelization** but
  **does not implement or test it**. There is no CBE-block-Jacobi result
  anywhere.

### 2.3 The optimistic reading

Under the following conditions, block-Jacobi CBE could converge within a
constant factor of Gauss-Seidel:

1. **Segment length ≥ 3× correlation length** (i.e., `L/P ≳ 3ξ`). For gapped
   Hamiltonians with `ξ ≤ 8`, this means `L/P ≥ 24`, i.e., at `L=64, P=2` only,
   or at `L=128, P=4`.
2. **CBE's adaptive subspace enrichment partially compensates** for stale-env
   fragility — the same tangent-space expansion that makes CBE immune to
   single-site freezing also gives the local update extra "noise" in the
   direction that the stale environments fail to explore. This is a real but
   unproven mechanism.
3. **The warmup (Phase 0) and polish (Phase 3) clean up Jacobi drift**, so
   Phase 2 only needs to reduce the energy by a few orders of magnitude, not
   to machine precision.

### 2.4 CBE single-site vs 2-site: does CBE change the convergence picture?

Yes, plausibly. The standard argument for why 1-site DMRG is fragile to stale
environments is that the site optimization cannot grow the bond basis: if the
current basis at bond `i` is missing a direction that becomes important due to
distant updates, 1-site DMRG cannot discover it. 2-site DMRG discovers new
basis directions automatically through the SVD after each bond update.

CBE inserts a small tangent-space enrichment (`k ≈ 4–8` extra basis vectors)
that covers the 2-site tangent complement. This is enough to break the
stale-env trap: even with stale environments, the per-bond update can grow
the bond basis in directions that the local `H_eff` suggests. In that sense,
**CBE is the best candidate single-site variant for block-Jacobi**, better
than DMRG3S (which only enriches with `Hψ` projected on the *current* basis)
and vastly better than plain 1-site DMRG.

That argument is qualitative and has no numerical validation.

---

## 3. Comparison table

| Aspect | Serial CBE (baseline) | Stoudenmire PDMRG (V=Λ⁻¹) | A2DMRG (additive Schwarz) | **Block-Jacobi CBE (R2-2 Phase 2)** |
|---|---|---|---|---|
| Parallelism source | None | Segments + boundary merge | Segments + coarse correction | Segments |
| Ordering | Gauss-Seidel, global | Gauss-Seidel within segment, serial boundary merge | Jacobi local + global correction | Gauss-Seidel within segment, **pure Jacobi across segments** |
| Boundary bridge | n/a | `V = Λ⁻¹` exact SVD operator | Coarse-space eigenproblem | **None** (rebuild envs + canonicalize) |
| Published convergence proof | Yes (variational monotone) | Yes (empirical, ~10×) | Yes (empirical, 2–8×) | **No** |
| Monotone energy guarantee | Yes | Yes (per-merge check) | Yes (via coarse correction) | **No** — can oscillate |
| Extra sweeps vs serial | 1× | ~1.2–1.5× (measured in project: 2–3 outer sweeps + polish) | Authors: "more global iterations than classical DMRG"; empirical 1.5–2× | **Unknown — theory 1.8–2.2×, critical Hamiltonians possibly ∞** |
| Critical-point stability | Yes | Yes (with enough sweeps) | Yes (coarse correction absorbs gap) | **Unknown — high risk** |
| Memory per worker | 1× | ~`L/P` segment + 2 boundary envs | `L/P` segment + coarse space | **Full MPS + full env stack × P** (see §5) |
| Known failure mode | None | Boundary phase slip → caught by merge | Coarse space rank-deficient | **Gapless critical Ham → stalls**; **non-monotone oscillation** |
| Implementation risk | Low | Medium (merge already debugged in pdmrg-gpu) | High (coarse solve) | **High (no published reference)** |

---

## 4. Extra-sweep estimate at L=64, χ=128 (Heisenberg)

**Serial CBE baseline (projected from CBE literature):**

- DMRG2 reference on Heisenberg L=64 converges in ~5 sweeps to ΔE ~ 1e-10.
- CBE single-site (Gleis et al.) reports 2.5× runtime speedup over DMRG3S and
  converges in a similar number of sweeps (CBE's speedup is per-bond, not
  fewer sweeps). Estimate: **5 serial CBE sweeps** to reach ΔE ~ 1e-10.

**Block-Jacobi CBE estimate at P=4, L/P=16 (~2ξ for Heisenberg):**

Two regimes:

1. **Optimistic (CBE enrichment compensates, Jacobi ~= 1.8× Gauss-Seidel):**
   Sweeps needed = `ceil(5 × 1.8) = 9 sweeps`. This is the lower bound.
2. **Pessimistic (no CBE compensation, Jacobi ~= 2.5× Gauss-Seidel, which is
   the upper end of Young's bound for banded matrices):**
   Sweeps needed = `ceil(5 × 2.5) = 13 sweeps`.

Wall-time speedup formula:
```
speedup(P) = serial_sweeps × L / (jacobi_sweeps × L/P) = serial_sweeps × P / jacobi_sweeps
```

| P | Optimistic (Jacobi=9) | Pessimistic (Jacobi=13) | Break-even (speedup=1) requires |
|---|---|---|---|
| 2 | 5·2/9 = **1.11×** | 5·2/13 = **0.77×** ❌ | Jacobi ≤ 10 |
| 4 | 5·4/9 = **2.22×** | 5·4/13 = **1.54×** | Jacobi ≤ 20 |
| 8 | 5·8/9 = **4.44×** | 5·8/13 = **3.08×** | Jacobi ≤ 40 |

**At P=2 under the pessimistic assumption, block-Jacobi CBE is a net LOSS.**
This is damning. The R2-2 projection of 1.7× at `L=64 χ=128 P=4` over PDMRG
assumes the optimistic case, and does not include the full env-rebuild
barrier (~100 ms per sweep × 9 sweeps = 900 ms extra).

Adding the env-rebuild tail and the Phase 3 serial polish at P=4:

```
T_R2-2(P=4, L=64, χ=128) ≈
    0.08  (Phase 0 iTEBD)
    0.1   (Phase 1.5 canonicalize+broadcast)
    9 × (16 bonds × 70 ms + 100 ms env rebuild)  =  9 × 1220 ms = 11.0 s   (Phase 2)
    9.0   (Phase 3 serial CBE polish, R2-2 estimate)
    -----
    total ~ 20.2 s
```

vs R2-2's original 14.2 s claim. The slipped estimate narrows the
PDMRG-relative speedup from **1.7× → 1.19×**, which is within the noise of
the measured PDMRG ceiling (1.22×).

**If the pessimistic Jacobi count (13) holds:**
```
13 × 1220 ms = 15.9 s (Phase 2) + 9 s (Phase 3) = ~25 s total
```
vs PDMRG's measured 24 s. **Block-Jacobi CBE is no faster than PDMRG.**

---

## 5. Adversarial findings

### 5.1 Is there any published block-Jacobi DMRG result?

**No.** Confirmed via dedicated deep-research search (perplexity Sonar Deep
Research, 2010–2026, keywords: "block-Jacobi DMRG", "asynchronous DMRG",
"Hogwild DMRG", "stale environment DMRG"). The literature is silent.
A2DMRG (arXiv:2505.23429) is the closest, and its authors explicitly describe
their scheme as "additive Schwarz-inspired" and **explicitly warn that the
additive (Jacobi-like) structure converges slower than classical DMRG in
terms of global iteration count.** A2DMRG adds a global coarse-space
correction to paper over the Jacobi penalty; **R2-2 Phase 2 has no such
correction.**

### 5.2 What if the block-Jacobi update increases the energy between sweeps?

This is a real risk. Three standard fixes, none yet designed into R2-2:

1. **Damped Jacobi** `M_new = (1-ω) M_old + ω M_Jacobi` with ω ∈ (0.5, 0.8).
   Requires solving the projection back onto the MPS manifold, which is not
   just a linear combination of tensors — it's a variational projection.
   Cost: one extra canonicalization pass. Feasible but adds ~5% per sweep.
2. **Line search** along the Jacobi descent direction. Same issue as damping:
   the MPS manifold is not linear.
3. **Rollback on energy increase.** If `E_new > E_old`, discard the Jacobi
   update for the offending segment and fall back to a serial Gauss-Seidel
   sweep on that segment. This is the **only** rigorously monotone fix for
   non-convex ALS iterations. Adds a serial-fallback code path.

**Recommendation:** R2-2 must implement rollback-on-energy-increase as a
minimum safety net. The current §7.2 pseudocode assumes monotonic descent and
has no check.

### 5.3 At what P does convergence degradation eat the speedup?

From §4: break-even under the pessimistic assumption (Jacobi sweeps = 13)
is **P ≥ 3**. At **P = 2 the pessimistic case is a net loss** (0.77× vs
serial), and the optimistic case is only 1.11×. This means **block-Jacobi
CBE at P=2 is not worth attempting** — the convergence risk is not
compensated by the parallelism gain.

At P=4 the pessimistic case is 1.54× over *serial CBE*, which is barely above
the PDMRG Amdahl ceiling (1.22× over serial two-site) and does not clear the
95 % confidence interval of PDMRG's measured speedup.

At P=8 the pessimistic case is 3.08× over serial CBE, which is meaningful but
requires the segment length `L/P = 8` to still exceed 2 correlation lengths.
For Heisenberg `ξ ≈ 6–8`, **segment length 8 is right at the information
horizon**, and critical TFIM at `L=64 P=8` almost certainly fails.

**Conclusion:** the usable P window is `4 ≤ P ≤ 8`, and only for gapped
Hamiltonians with `ξ < L/P`.

### 5.4 Memory scaling

For replicated envs at L=64, D=5, P=4, `complex128`:

| χ | Single env tensor | L+1 envs (L-stack) | 2 stacks (L+R) | MPS | **Per-GPU total** |
|---|---|---|---|---|---|
| 128 | 1.31 MB | 85.2 MB | 170.4 MB | 33.6 MB | **~204 MB** |
| 256 | 5.24 MB | 340.8 MB | 681.6 MB | 134.2 MB | **~816 MB** |
| 512 | 20.97 MB | 1363.2 MB | 2726.4 MB | 536.9 MB | **~3263 MB** |

All of these fit in MI300X's 192 GB HBM with room for CBE scratch
(`U_k`, Lanczos Krylov basis, etc.). **Memory is not the bottleneck.**
Broadcasting 204 MB (χ=128) at xGMI 128 GB/s is ~1.6 ms. Broadcasting
816 MB (χ=256) is ~6.4 ms. Both are negligible.

At χ=1024 the per-GPU state hits ~13 GB, still within the 192 GB envelope,
but the broadcast is now ~100 ms which starts to eat into Phase 1.5's
"serial ~100 ms" budget. Not a showstopper but worth tracking.

### 5.5 What if two adjacent segments disagree on the bond state?

This is **the** hard question. After the parallel sweep, the bond tensor
`M[p·L/P]` at the boundary of segments `S_{p-1}` and `S_p` has been touched
by worker `p-1`'s Gauss-Seidel pass (as the last site in that segment) and
then read — but not written — by worker `p`'s pass. **There is no
explicit merge.** The scheme relies on the post-sweep global canonicalize +
env rebuild to "heal" the inconsistency.

This is optimistic in three ways:

1. **Canonicalization does not minimize energy.** It only re-orthogonalizes.
   If the two workers' updates are in conflict at the boundary, canonicalize
   will just pick one canonical form; the energy may still be higher than
   either worker's local-minimum estimate.
2. **The env rebuild uses the new M everywhere**, which is the correct
   thing to do, but it means the stale-env error from the previous sweep is
   not corrected — it is **propagated into the next sweep's starting envs.**
   This is the standard Jacobi error-accumulation failure mode.
3. **Stoudenmire's V=Λ⁻¹ merge is the known-good solution** for this exact
   problem (Stoudenmire-White 2013 §III-C). Dropping it and replacing it with
   "canonicalize and hope" is a design regression.

**Recommendation:** re-introduce a Stoudenmire-style V=Λ⁻¹ merge at each
segment boundary after the parallel sweep. This moves R2-2 Phase 2 from
"block-Jacobi with no safety net" to "Jacobi-within-segment + explicit
boundary merge", which is architecturally closer to PDMRG and has an
empirical convergence track record. The cost is `P−1` SVDs of χ×χ boundary
matrices, ~10 ms total at χ=256, so it is not a performance concern.

---

## 6. VERDICT

**NEEDS SAFER SYNC.**

The block-Jacobi CBE Phase 2 as written in `research_D_beyond_pdmrg_a2dmrg.md`
§7.2 is theoretically plausible but has **no published reference
implementation, no empirical convergence results, no monotonic-energy
guarantee, and no merge step at segment boundaries.** The speedup
projection of 1.7× at `L=64 χ=128 P=4` relies on an optimistic (1.8×) Jacobi
sweep-count penalty. Under the pessimistic (2.5×) penalty that classical
numerical analysis permits, the scheme is **no faster than measured PDMRG**
and at `P=2` is a net loss.

The scheme is not hopeless — CBE's subspace enrichment is the best
theoretical argument for stale-env robustness in the 1-site DMRG family, and
the gapped-Hamiltonian information-horizon argument is sound for `L/P ≫ ξ`.
But without the three safety nets listed below, it is a convergence gamble
that should not be committed to as R2-2's flagship.

---

## 7. Concrete next actions

These are ordered by cost and by how much they de-risk R2-2:

1. **[Prerequisite, 1 week, CPU-only] Python block-Jacobi CBE reference
   implementation** in `cpu/cbe-reference/bj_cbe.py`. Use numpy + the existing
   `cpu/pdmrg/` env infrastructure. Run on Heisenberg L=32,64 and TFIM
   L=32,64 at criticality (`g=1`), with P ∈ {2, 4, 8}, for 20 sweeps.
   Plot energy vs sweep and count sweeps-to-1e-8. **This is the go/no-go
   gate for R2-2 Phase 2.** Cost: 1 week, no GPU needed.

2. **[Algorithmic fix, 2–3 days] Add V=Λ⁻¹ boundary merge to the sweep
   pseudocode.** After the parallel sweep and before the global canonicalize,
   execute a Stoudenmire-style merge at each of the `P−1` boundaries. The
   code for this already exists in `cpu/pdmrg/pdmrg/dmrg.py:recompute_boundary_v`
   and `boundary_merge`. Port to the new scheme. This is the single most
   important risk mitigation.

3. **[Safety net, 2–3 days] Add rollback-on-energy-increase.** After each
   parallel sweep, if the new global energy exceeds the old global energy,
   discard the sweep and fall back to a single serial CBE sweep. This is an
   O(L) serial fallback invoked only when Phase 2 misbehaves. Cost is
   trivial in code (a conditional branch) but it is the **only** monotonic
   guarantee available.

4. **[Stretch, 1 week] Damped Jacobi `ω ∈ (0.5, 0.8)`** as an optional mode
   in the CPU reference. Compare convergence rate with/without damping on
   the TFIM critical benchmark. If damping closes the gap to serial
   Gauss-Seidel, lift it to the GPU implementation.

5. **[Update R2-2 plan] Rewrite §7.2 of `research_D_beyond_pdmrg_a2dmrg.md`**
   and §R2-2 of `round_2_plan.md` to include:
   - the V=Λ⁻¹ boundary merge in the Phase 2 pseudocode,
   - the rollback-on-energy-increase safety net,
   - the pessimistic speedup projection (1.19× instead of 1.7× at
     P=4 χ=128) as the **default number**, with 1.7× reserved as a
     best-case stretch goal,
   - an explicit gate: **R2-2 does not proceed to GPU implementation until
     the CPU reference from action #1 shows a sweep-count ≤ 1.8× serial on
     the three reference Hamiltonians.**

---

## References (selected)

- A. Gleis, J.-W. Li, J. von Delft, *"Controlled bond expansion for DMRG
  ground state search at single-site costs"*, PRL 130, 246402 (2023),
  arXiv:2207.14712.
- E. M. Stoudenmire, S. R. White, *"Real-Space Parallel DMRG"*, PRB 87,
  155137 (2013), arXiv:1301.3494.
- M. Naumov, L. Grigori et al., *"A2DMRG: Additive Two-Level DMRG"* (title
  tentative), arXiv:2505.23429 (2025). **This is the closest published
  relative to the R2-2 Phase 2 scheme, and it includes a coarse-space
  correction that R2-2 does not.**
- F. Niu, B. Recht, C. Ré, S. J. Wright, *"HOGWILD!: A Lock-Free Approach to
  Parallelizing SGD"*, NeurIPS 2011, arXiv:1106.5730. **Closest ML-side
  analog for async parallel updates. Provably converges only when updates
  are 'mostly orthogonal'; for DMRG this is true only across segments
  separated by more than ξ.**
- Y. Saad, *Iterative Methods for Sparse Linear Systems* (SIAM, 2003), §4
  (Jacobi vs Gauss-Seidel, Young's theorem).
- C. Hubig, I. P. McCulloch, U. Schollwöck, F. A. Wolf, *"Strictly
  single-site DMRG algorithm with subspace expansion"* (DMRG3S), PRB 91,
  155115 (2015), arXiv:1501.05504.
- H. R. Larsson, *"A critical look at controlled bond expansion"*,
  arXiv:2403.00562 (2024).
- Perplexity deep-research survey (Sonar, April 2026), full text in
  `.claude/projects/.../tool-results/toolu_019QSKid95TfmjipdHBH8noU.txt`.

**Key absence:** no paper in the 2010–2026 literature implements or tests
block-Jacobi DMRG with stale environments. This is a negative result from
an exhaustive search, not a search-term gap.
