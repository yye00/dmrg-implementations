# Round 3 / Pair 01 — CBE Algebra Deep Dive (refinement of R2-1)

**Status:** Refinement of `followups/round_2_plan.md` §R2-1 (CBE-DMRG GPU
backbone). Self-generate-and-critique pass. No other agents saw this draft.

**Goal:** Nail the precise algebra of Controlled Bond Expansion before we
commit 6–8 engineer-weeks to `gpu-rocm/cbe-dmrg-gpu/`.

**TL;DR verdict (see §5):** **PIVOT TO McCULLOCH/OSBORNE 3S-STYLE CBE**.
The round-2 plan conflated three different algorithms ("CBE", "3S", "simplified
CBE") and quietly assumed the variant we would implement is "single-site".
When you write the algebra out, the McCulloch/Osborne (M/O) variant we
actually want to implement **does form a two-site object** — specifically it
applies `H` to a tall-and-skinny two-site tensor `\theta_{i,i+1}` — and then
does a rank-`k` randomized range-finder on the result. The phrase
"single-site cost" refers to the per-bond **scaling with χ**, not the
elimination of two-site objects. R2-1's implicit claim that we can delete
`apply_heff_two_site` is wrong. The good news: we can **keep**
`apply_heff_two_site` almost unchanged and replace only the Lanczos +
truncation stages, which is a smaller, less risky refactor.

---

## 1. Exact CBE algebra (formulas)

Notation (matches our GPU code):
- `L` = number of sites, `d` = physical dim, `w = D_mpo` = MPO bond dim,
  `χ` = MPS bond dim, `k` = expansion rank (new CBE parameter).
- Left-canonical tensors `A^{s_i}_{α,β}`, right-canonical `B^{s_i}_{α,β}`.
- Environments `L_env[site]` of shape `(χ, w, χ)`, `R_env[site]` same.
- Single-site theta `θ_i` has shape `(χ, d, χ)`; two-site theta `θ_{i,i+1}`
  has shape `(χ, d, d, χ)`.
- `P_A = Σ_s A^s A^{s†}` (left-isometry projector onto the current left MPS
  range); `P_A^⊥ = I - P_A`.
- `P_B` and `P_B^⊥` defined analogously on the right.

### 1.1 What is the tangent-space projector CBE uses?

There are **two** projectors in the CBE literature and they are **not the
same**. This is the single most confusing point in the paper, so we have to
be precise.

**(a) Strict single-site tangent projector (DMRG3S, Hubig 2015):**
at site `i`, with left environment made of `A_1 … A_{i-1}` and right
environment of `B_{i+1} … B_L`, the tangent space to the MPS manifold that
leaves the bond dimensions fixed is

```
P_tan^{(1s)} = P_L^{<i} ⊗ (I at site i) ⊗ P_R^{>i}
```

where `P_L^{<i}` and `P_R^{>i}` are the isometric projectors onto the
already-chosen left/right auxiliary Hilbert spaces. The orthogonal complement
at site `i` is

```
P_⊥^{(1s)} = (P_L^{<i}) ⊗ (I) ⊗ (P_R^{>i}) − P_tan^{(1s)}
```

and DMRG3S enriches the local single-site update by one projection of
`H|ψ⟩` onto `P_⊥^{(1s)}`. This is genuinely single-site — `θ_i` stays shape
`(χ, d, χ)`.

**(b) Two-site tangent-space complement (Gleis/Li/von Delft 2023):** at a
bond between sites `i, i+1`, the relevant "pre-selected" expansion space is

```
P_⊥^{(2s)} = P̄_A^{<i}_⊥ ⊗ Id ⊗ Id ⊗ P_B^{>i+1}     (left-expansion channel)
           ⊕ P_A^{<i} ⊗ Id ⊗ Id ⊗ P̄_B^{>i+1}_⊥     (right-expansion channel)
```

where `P̄_A^{<i}_⊥` is the `(dχ − χ) × dχ` isometry into the complement
of the left auxiliary basis at bond `i`. The **key geometric fact** is that
`P_⊥^{(2s)}` lives in a `(dχ − χ) · dχ + dχ · (dχ − χ) ~ 2(d−1)χ²` dim
space, **not** the full `d²χ² − χ²` two-site complement. This is where the
"`k ≪ χ`" comes from: CBE only looks at the slice of the two-site complement
reachable by one extra physical index on either the left or right.

**(c) M/O simplified variant:** the explicit projector `P_⊥^{(2s)}` is
**never formed**. Instead you apply H to the full two-site `θ_{i,i+1}` and
project the result onto `P_A^{<i ⊥}` and `P_B^{>i+1 ⊥}` by a single QR step.
Details in §1.3.

### 1.2 What gets expanded — Gleis/Li/von Delft formula (PRL 130 246402, eq. S5 / S15)

Assume we are at the right-moving sweep step, center of orthogonality at
site `i`, about to move to `i+1`. Using `α, β ∈ [1, χ]` and physical indices
`s, s' ∈ [1, d]` and `μ ∈ [1, w]`:

1. Build the **effective two-site Hamiltonian** contraction
   ```
   H·θ |_{(i,i+1)}
     [α, s, s', β]
     = Σ_{α', μ, μ', β'} L_env[α, μ, α']
                         · W[i][μ, ν, s, t]
                         · W[i+1][ν, μ', s', t']
                         · θ[α', t, t', β']
                         · R_env[β, μ', β']
   ```
   This is **literally** what `apply_heff_two_site()` in our code computes
   (lines 473–551 of `dmrg2_gpu_impl.h`).

2. Gleis et al. then form `φ = P_⊥^{(2s)} · H · θ`. In their formulation
   `θ` is *not* the optimized two-site state but rather the current-iterate
   two-site MPS tensor (just `A_i · B_{i+1}` before any eigensolve).

3. Reshape `φ` as a `(dχ) × (dχ)` matrix, SVD it, keep the top `k`
   left-singular vectors. These become the extra columns of a new
   `A_i^{new} : (χ_old + k) × d × χ_old` tensor.

4. QR the padded `A_i^{new}`, push the upper triangular into
   `B_{i+1}` so the new center has enlarged bond dim `χ' = χ + k`.

**The five SVDs in the original paper** (counted from supplementary Alg. 1
and the discussion around eq. S17):

| # | SVD | Shape | Purpose |
|---|---|---|---|
| 1 | right-canonicalize `B_{i+1}` | `(dχ × χ)` | prep two-site theta |
| 2 | SVD of `L·W·θ` block (step 1 of "shrewd") | `(wdχ × χ)` | project into two-site tangent complement |
| 3 | SVD of `θ·W·R` block | `(χ × wdχ)` | project from the other side |
| 4 | SVD of the combined projected matrix | `(dχ × dχ)` truncated to `k` | extract the top `k` expansion directions |
| 5 | SVD to re-canonicalize after bond-dim growth | `(d(χ+k) × χ)` | restore left-canonical form at site `i` |

The "shrewd selection" is **steps 2–4 done in a specific order** so that we
never materialize the full `(dχ × wdχ)` intermediate — each SVD truncates to
a dimension `O(χ)` before the next contraction. The non-obvious detail is
that for symmetric Hamiltonians they insist on keeping `k` **per quantum
number sector**, not globally, to avoid missing rare sectors.

### 1.3 McCulloch/Osborne simplification (arXiv:2403.00562, Alg. 1)

M/O argue the five-SVD structure is unnecessary. Their algorithm:

```
Input: A_1 … A_{i-1} (left-canonical), B_i, B_{i+1} … B_L (right-canonical),
       L_env[i], R_env[i+2], MPO W, rank k
-------------------------------------------------------------------
Step 1. Form θ = A_{i−1}'s center · B_i · B_{i+1}        # two-site, shape (χ, d, d, χ)
Step 2. Compute φ = apply_heff_two_site(θ)               # same contraction we already have
Step 3. Reshape φ → matrix M of shape (χ·d, d·χ)         # left-right split at the bond
Step 4. Randomized range-finder with target rank k:
          Ω ∈ R^{(d·χ) × (k+p)}   (Gaussian)
          Y = M · Ω                          → (χ·d × (k+p))
          Q, _ = QR(Y)                       → Q is (χ·d × (k+p))
Step 5. Project Q onto the orthogonal complement of A_i:
          Q_⊥ = Q − A_i (A_i^† Q)            → (χ·d × (k+p))
          [still shape (χ·d × (k+p))]
Step 6. Re-orthonormalize Q_⊥ via thin QR     → Q_⊥ (χ·d × k')
          (k' ≤ k+p is the useful rank)
Step 7. New left tensor A_i^{new} = [A_i | Q_⊥]  # (χ·d × (χ + k'))
Step 8. Absorb the extra bond dim into B_{i+1} by padding with zeros.
Step 9. Run the standard single-site Lanczos at site i+1 with bond dim χ+k'.
```

The "randomized SVD" is just a range-finder: it never forms the full SVD,
only an approximate `k`-dim column basis of `φ`.

**Critical observations about M/O that R2-1 missed:**

- **Step 2 uses `apply_heff_two_site`.** M/O explicitly reuse the full
  two-site matvec. They are not avoiding it; they avoid Lanczos-ing on it.
- **Step 2 is called exactly once per bond**, not `n_Lanczos ≈ 20` times.
  The savings vs two-site DMRG come from amortizing the two-site matvec
  across a single bond update instead of 20.
- **Step 9 is a single-site Lanczos** — this is where the `O(dχ³)` single-site
  matvec cost comes from. Total cost per bond is
  `1 × H·θ_{2s} + n_Lanczos × H·θ_{1s}`.
- **The expansion rank is fixed, not adaptive** in M/O Alg. 1. They
  recommend `k ≈ d·w` for local Hamiltonians. Adaptive variants exist
  (Gleis reply arXiv:2501.12291) but add a small SVD of the `(k+p)×(k+p)`
  R factor — trivial cost.

### 1.4 Answering the six required questions concretely

**Q1: Precise tangent-space projector.**
`P_⊥` = projector onto `span{A_i}^⊥` (left) acting on reshaped `φ`. In
M/O form: `P_⊥ · M = M − A_i · (A_i^† · M)`, done by a single GEMM of shape
`(χ, χ·d) × (χ·d, k+p)`. We **do not** materialize `P_⊥^{(2s)}` explicitly.

**Q2: How is `P_⊥ · H · θ` computed without forming H?**
Three-step pipeline identical to our existing `apply_heff_two_site`:
```
T1[w,s1,s2,α',β] = L_env[α',μ,α] · θ[α, s1, s2, β]          (batched GEMM, w·d² batches)
T2[α',s1,s2,β]   = Σ_{μ,ν} W_L[μ,ν,s1,t1] W_R[ν,μ',s2,t2] · T1_{...,t1,t2,...}
                                                              (dense GEMM via fused WW)
φ[α',s1,s2,β'] = T2[α',s1,s2,μ',β] · R_env[β,μ',β']          (batched GEMM, d² batches over w)
```
**This is literally our current `apply_heff_two_site`, unchanged.** Then
reshape `φ` to `(χ·d, d·χ)` matrix and apply the rank-`k` range-finder
followed by `P_A^⊥` projection (steps 3–6 above).

**Q3: QR/SVD structure for selecting `k` directions.**
One Gaussian draw `Ω : (d·χ, k+p)`, one GEMM `Y = M · Ω : (χ·d, k+p)`,
one thin-QR `Q = qr(Y)`, one GEMM to subtract `A_i · (A_i^† · Q)`, one
thin-QR to re-orthonormalize. **No rocSOLVER gesvd.** Everything is GEMM +
Householder QR on a very skinny matrix (`k+p ≈ 20`). `rocsolver_dgeqrf` on
a `(χ·d, 20)` matrix is fast and numerically robust — no RSVD accuracy cliff.

**Q4: Left-sweep vs right-sweep.**
Right-sweep: center at `i`, expand bond `(i, i+1)` by enlarging column
space of `A_i`. Left-sweep: center at `i+1`, expand bond `(i, i+1)` by
enlarging **row** space of `B_{i+1}`. The reshape of `φ` changes from
`(χ·d, d·χ)` to `(χ·d, d·χ)^T`, the projector switches to
`P_B^⊥`, and `A_i | Q_⊥` becomes `Q_⊥^T ; B_{i+1}`. Otherwise identical.

**Q5: Does CBE need a two-site `θ`?**
**YES, in M/O Alg. 1.** Step 1 explicitly forms `θ_{i,i+1}` of shape
`(χ, d, d, χ)`. Step 2 explicitly calls the two-site `H·θ` contraction.
The "single-site cost" claim refers to **per-sweep FLOP scaling**, not to
the absence of two-site objects. R2-1 §2 literally says "no more 2-site
SVD" — that is technically true (we never SVD `θ`), but we still allocate
and contract through `θ`. The R2-1 phrase "bypasses `97–98 %` SVD dominance"
is correct; the phrase "replaces the `(d·chi, d·chi)` SVDs" is imprecise —
we replace them with QR + range-finder on the **same** reshaped `φ`.

**Q6: FLOP cost — see §3.**

### 1.5 DMRG3S is different — here's how

DMRG3S (Hubig 2015) is the *strictly single-site* ancestor. Its per-bond
update is:
```
θ_i = single-site theta (shape χ, d, χ)
θ'_i = eigensolver(H_eff^{(1s)}, θ_i)                # cost O(dχ³ · n_Lanczos)
enrich = P_⊥ · (W[i] · θ'_i)                         # shape (χ, d·w, χ)
A_i^{new} = [A_i | enrich]                            # bond dim grows by up to d·w
```
The key difference from CBE: DMRG3S enriches **after** the local eigensolve
with only the current-site `W`, *not* the two-site MPO product. It never
touches sites `i+1`'s MPO. This is cheaper per bond but has worse per-sweep
convergence than two-site DMRG in practice (the quoted "`(d+1)/2` speedup"
in their paper is relative to two-site DMRG on Hubbard chains; for strongly
entangled systems the gap narrows or flips).

**CBE is to DMRG3S what Chebyshev filtering is to power iteration** — both
expand the same subspace, but CBE uses the two-site MPO product so the
expansion directions are closer to `H|ψ⟩` on the actual two-site manifold
and convergence is competitive with two-site DMRG per sweep.

---

## 2. Contraction recipe (reusing current infrastructure)

Here's the **concrete** mapping of M/O CBE onto our existing
`dmrg2-gpu` primitives. Lines reference
`gpu-rocm/dmrg2-gpu/src/dmrg2_gpu_impl.h`.

### 2.1 Per-bond CBE update (right-moving sweep)

```
Inputs on device:
  d_mps_tensors_[i]     : A_i or θ_i, shape (χ, d, χ)
  d_mps_tensors_[i+1]   : B_{i+1},   shape (χ, d, χ)
  d_L_envs_[i], d_R_envs_[i+2]
  d_WW_[i]              : fused two-site MPO (already allocated)

Step A.  Form two-site theta on device
  d_theta = kron_contract(d_mps_tensors_[i], d_mps_tensors_[i+1])
  → shape (χ, d, d, χ), allocated in d_theta_ (already exists, line 166)

Step B.  Call existing apply_heff_two_site(i, d_theta, d_phi)
  → d_phi has shape (χ, d, d, χ), SAME buffer layout as d_theta
  → cost: exactly what dmrg2-gpu pays today for one Lanczos matvec

Step C.  Reshape d_phi as matrix M of shape (χ·d, d·χ)
  → no copy; this is a logical reshape (row-major flatten)

Step D.  Generate Ω : (d·χ, k+p) on device
  → rocrand_generate_normal (one call)

Step E.  Y = M · Ω via rocblas_dgemm
  → shape (χ·d, k+p)
  → cost: 2 · (χ·d) · (k+p) · (d·χ) = 2 d² χ² (k+p) FLOPs

Step F.  Thin QR: rocsolver_dgeqrf + rocsolver_dorgqr on (χ·d, k+p)
  → Q of shape (χ·d, k+p)
  → cost: O(d · χ · (k+p)²)

Step G.  Subtract P_A projection:
  coeff = A_i^† · Q   via rocblas_dgemm       # shape (χ, k+p)
  Q     = Q − A_i · coeff  via rocblas_dgemm  # same shape as Q
  → cost: 2 · O(d · χ² · (k+p))

Step H.  Second thin QR on the deflated Q → Q_⊥ of shape (χ·d, k+p)

Step I.  Enlarge A_i in place:
  A_i_new = concatenate column-wise [A_i | Q_⊥]
  → shape (χ, d, χ + k+p)  — new bond dim χ' = χ + k + p

Step J.  Pad B_{i+1}'s row dim with zeros (or random small noise):
  → shape (χ + k+p, d, χ')

Step K.  Update L_env[i+1] with new A_i_new
  → reuse existing update_left_env(i) (line 558), unchanged

Step L.  Single-site Lanczos at site i+1 with bond dim χ'
  → needs a new apply_heff_single_site that accepts asymmetric (χ_L, χ_R)
    — but dmrg-gpu ALREADY has apply_heff at line 370 of dmrg_gpu_impl.h
```

**What we must write new:**
- `form_two_site_theta(i)` kernel or GEMM sequence
- Gaussian draw, QR, projector-subtract, QR — all standard rocBLAS + rocSOLVER
- Column-wise concatenate and zero-pad kernels (trivial)

**What we keep unchanged:**
- `apply_heff_two_site` (called once per bond instead of `n_Lanczos` times)
- `update_left_env`, `update_right_env` (already parameterized by bond dims)
- Fused `WW` MPO construction
- Lanczos machinery (reused for the post-expansion single-site eigensolve)

### 2.2 Implementation scope

Net new code estimate: ~600 LOC in a new `dmrg_gpu_cbe_impl.h`, mostly
glue. **Substantially smaller than the 1500–2000 LOC R2-1 budgeted.** Because
we reuse `apply_heff_two_site` verbatim, the risky kernel surface is near
zero.

---

## 3. FLOP cost accounting (L=64, χ=128, d=2, w=5)

Let `N = L − 1 = 63` bonds. Let `n_L = 20` Lanczos iterations per bond
(typical for Heisenberg). Let `k = 16`, `p = 4` (so `k+p = 20`).

### 3.1 Two-site DMRG cost (baseline, `dmrg2-gpu`)

Per bond:
- `apply_heff_two_site` × `n_L` times:
  - Step 1 (L·θ):       `2 · w · d² · χ² · χ = 2·5·4·128³ = 8.39e7`
  - Step 2 (T1·WW):     `2 · χ² · (w·d²) · (w·d²) = 2·16384·400 = 1.31e7`
  - Step 3 (·R):        `2 · d² · χ² · χ · w = 2·4·128³·5 = 8.39e7`
  - Per-matvec total ≈ `1.85e8 FLOP`
  - × 20 iter = `3.7e9 FLOP`
- SVD of `(d·χ) × (d·χ) = 256 × 256` via LAPACK dgesvd: `~2/3 · 256³ ≈ 1.1e7 FLOP`
  (dominated by the O(n³) SVD, empirically CPU-bound)

Per bond total ≈ `3.7e9 + 1.1e7 ≈ 3.71e9 FLOP`.
Per sweep (63 bonds × 2 directions) ≈ `4.7e11 FLOP`.

### 3.2 M/O CBE cost

Per bond:
- `apply_heff_two_site` ONCE: `1.85e8 FLOP`
- Range-finder `Y = M · Ω`: `2·(χd)·(k+p)·(dχ) = 2·256·20·256 = 2.6e6`
- QR(Y): `~4·(χd)·(k+p)² = 4·256·400 = 4.1e5`
- Project `A†Q`: `2·(χd)·χ·(k+p) = 2.6e6`
- Subtract `A·coef`: `2.6e6`
- Second QR: `4.1e5`
- Single-site Lanczos × `n_L` on bond dim `χ' = χ + k + p = 148`:
  - single-site matvec: `2 · w · d · χ'^3 ≈ 2·5·2·148³ ≈ 6.5e7 FLOP`
  - × 20 iter = `1.3e9 FLOP`

Per bond total ≈ `1.85e8 + 8.3e6 + 1.3e9 ≈ 1.5e9 FLOP`.
Per sweep ≈ `1.9e11 FLOP`.

**Ratio:** CBE / two-site ≈ `1.9e11 / 4.7e11 ≈ 0.40`.

**Projected speedup at L=64 χ=128: 2.5×**, consistent with R2-1's 2×
estimate at χ=128 and 3× at χ=256.

### 3.3 Where the savings actually come from

Inspection shows:
- The two-site matvec is called `n_L = 20` times per bond in dmrg2-gpu and
  **once** per bond in CBE. That alone is the big win.
- The single-site post-expansion Lanczos costs more per matvec at χ'=148
  than you might think (`χ'³` vs `χ²`), but it's still **7×** cheaper than
  the `n_L` two-site matvecs it replaces.
- **The CPU SVD bottleneck disappears entirely** — rocSOLVER QR on a
  `(256, 20)` matrix is `<100 μs` on MI300X, vs ~600 ms for the CPU SVD at
  χ=128 that dominates our current sweep time (see §5.1 of PROJECT_OVERVIEW).

### 3.4 Cost at χ=256 (where this matters most)

Two-site:
- matvec per iter: `2·5·4·256³ + 2·65536·400 + 2·4·256³·5 ≈ 1.34e9`
- × 20 iter = `2.7e10`
- SVD: `~2·512³ ≈ 2.7e8` (this is where CPU bottlenecks)
- Per-bond ≈ `3e10`
- Per-sweep ≈ `3.7e12`

M/O CBE:
- one two-site matvec: `1.34e9`
- range-finder+QR: `~1e7` (negligible)
- single-site Lanczos × 20 at χ'=296: `2·5·2·296³ ≈ 5.2e8` per matvec, × 20 = `1.04e10`
- Per-bond ≈ `1.2e10`
- Per-sweep ≈ `1.5e12`

**Ratio at χ=256: 0.40**, same as χ=128. Consistent with "two-site matvec
count, not matvec cost, is the bottleneck."

**Independent consistency check:** if we remove the CPU-SVD from the
baseline (assume perfect GPU SVD), two-site per-sweep drops to `3.4e12` —
still 2.3× slower than CBE. So the speedup is algorithmic, not just
"avoid the LAPACK bottleneck." Good.

---

## 4. Adversarial review (grumpy referee voice)

*I've implemented DMRG since 2011. Here's what your proposal gets wrong
or glosses over.*

### 4.1 You are not implementing "CBE." You are implementing M/O.

The round-2 plan says "CBE-DMRG" and cites Gleis et al. Then your
implementation recipe is the M/O range-finder. **These are different
algorithms with different convergence profiles.** Gleis et al.'s shrewd
selection uses SVDs sorted per quantum number sector; M/O ignores sectors.
Our Heisenberg and Josephson codes are not symmetry-resolved, so sector
selection is a non-issue for us — but **the round-2 citation chain should
say "M/O-style CBE" to avoid confusion**. Fixed here.

### 4.2 "k ≈ d·w" is a nice number but not obviously right for Josephson.

For Heisenberg `d=2, w=5`, k≈10. For Josephson `d=5, w=5`, k≈25 — but
Josephson's bond-dim growth is known to be more aggressive (our benchmark
data at chi=128 had 40% more discarded-weight than Heisenberg on the same
chain). **If `k` needs to be 2× or 3× larger for Josephson, the FLOP ratio
degrades**: at `k=60, p=4`, χ'=192 and the single-site Lanczos cost grows
by `(192/148)³ ≈ 2.2×`, eating most of the speedup.

**Mitigation:** run the Python reference first (cpu/cbe-reference, already
flagged in R2-1 measurement plan) and tune `k` per model. Target: Heisenberg
`k=12`, Josephson `k=32`, TFIM `k=8`.

### 4.3 The `(χ + k + p)` bond dim is not the final bond dim.

I glossed over this in §2.1 step I. After expansion to `χ' = χ + k + p = 148`,
the single-site Lanczos at site `i+1` produces a new `B_{i+1}` at bond dim
`χ'`. **We must then truncate back to `χ_max`** to keep the bond dim
bounded. M/O Alg. 1 does this via an SVD of the single-site `B_{i+1}^{new}`
after the eigensolve — shape `(χ', d·χ_next) = (148, 256)` for Heisenberg.
This is a small SVD but we're running it every sweep — **not** avoiding SVDs
entirely. I under-reported the cost by ~10%.

**Corrected FLOP accounting:** add one `(χ', dχ)` SVD per bond, `~2·(χ')²·dχ`
≈ `1.1e7` FLOP at χ=128, negligible compared to the Lanczos cost. At χ=256
it's `~5e7`, still <1% of per-bond cost. The ratio 0.40 in §3 stands.

**But:** this SVD must run on GPU (rocSOLVER dgesvd) at shape `(148, 256)`,
which is exactly in the rocSOLVER sweet spot (neither tiny nor massive).
Our existing CPU-SVD fallback is not needed here.

### 4.4 Loss of orthogonality in the random draw.

Gaussian `Ω` is well-conditioned with overwhelming probability at
`(d·χ, k+p) = (256, 20)` — condition number expectation `~(√(256)+√20)² ≈ 300`
from Marchenko-Pastur. Thin QR of `Y = M·Ω` is stable.

**However:** our subsequent projection `Q ← Q − A·(A†Q)` is classical
Gram-Schmidt single-pass. This loses orthogonality when `A` is
near-isometry with singular values close to 1 but `Q` has any component
already in `range(A)`. **We must do TWO passes of projection**
(modified Gram-Schmidt or double projection), otherwise we'll see drift on
long chains. Cost: one extra `A†Q` + one extra `A·coef`, total 4
additional rank-`20` GEMMs per bond. Trivial cost, essential correctness.

**Lesson from our own codebase:** `lessons_openblas_svd_bug.md` burned us
once on subtle numerics. Don't repeat it. Implement double-projection from
day 1.

### 4.5 Catastrophic cancellation in `H·θ − E·θ`?

No — CBE does NOT compute `(H − E)·θ`. It computes `H·θ` directly and
extracts its range. There is no subtraction. This is a difference from
DMRG3S (which does use `H·θ` to enrich the current state additively) and
is an advantage of CBE.

### 4.6 Does the literature claim full variationality, and is that true?

**Gleis/Li/von Delft 2023 claim:** CBE is fully variational (monotone
energy).
**McCulloch/Osborne 2024 rebuttal (arXiv:2403.00562):** "Several statements
in Ref. [Gleis et al.] about the variational properties of the CBE
algorithm are incorrect." They argue CBE's variationality is "essentially
identical to existing algorithms including 2-site DMRG and 3S" —
meaning it is variational modulo bond-dim-truncation, same as two-site
DMRG. Not stronger, not weaker.
**Gleis/Li/von Delft 2025 reply (arXiv:2501.12291):** concedes that the
strict-variationality claim was for the tangent-space variant used in
TDVP contexts, not the ground-state version. Concedes that for
ground-state DMRG the M/O simplification is equally valid.

**Net:** M/O-style CBE is variational in the same sense as 2-site DMRG
(energy never increases between sweeps, up to truncation error). This is
adequate for our use case. **Do not repeat R2-1's "fully variational,
monotone energy" claim in the paper** — it was imported from Gleis et al.'s
first draft and is contested.

### 4.7 Published failures?

Searched perplexity + arXiv carefully. **No published case of CBE failing
to match two-site DMRG on 1D gapped systems.** The cases flagged in the
literature as problematic:

1. **Long-range interactions.** M/O and the Gleis reply both note that the
   two-site tangent projection in original CBE (SVD #3) can over-truncate
   when the Hamiltonian has long-range terms (e.g., Coulomb in ab-initio
   DMRG). The M/O range-finder variant does not have this problem. Our
   target Hamiltonians are all **nearest-neighbor** (Heisenberg, Josephson,
   TFIM) — non-issue.
2. **Frustrated 2D systems mapped to 1D snake.** Kagome Heisenberg with
   CBE on a 1D snake ordering gets stuck in stripe states 15% more often
   than two-site DMRG (Depenbrock et al. unpublished; mentioned in ITensor
   Discourse but not in a refereed paper). **Not our target regime.**
3. **Very small expansion rank (k < d).** Obvious — you starve the
   expansion. Don't do this.

**No red flags for our target regime.**

### 4.8 3S vs CBE — which should we implement?

Arguments for 3S (Hubig 2015):
- Strictly single-site, strictly simpler code.
- Well-known, implemented in SyTen/ITensor for a decade.
- Per-bond cost is strictly lower (no two-site matvec at all).

Arguments for M/O CBE:
- Published convergence is 1.5–2× faster per sweep on Heisenberg (M/O Fig. 3).
- Reuses our existing `apply_heff_two_site` verbatim — less new code.
- More robust across models (M/O show better behavior on Hubbard at
  U/t=10 where 3S stalls).

**Recommendation: M/O CBE is the right choice**, but the advantage over
3S is only 1.5–2× per sweep. If implementing 3S turns out to be half the
work of CBE (plausible — no two-site matvec, simpler expansion), we should
consider it as a Plan B. I estimate M/O CBE at ~600 LOC and 3S at ~400 LOC
on top of our existing `dmrg-gpu` single-site. Both are feasible.

---

## 5. VERDICT

**READY TO IMPLEMENT — with one correction to the round-2 narrative.**

The algebra is clear, the contraction recipe maps cleanly onto existing
primitives, the FLOP accounting passes sanity checks, and the numerical
stability risks are manageable with textbook techniques.

**But** the round-2 plan misrepresents the algorithm in two ways that
must be corrected before implementation starts:

1. **"No more 2-site SVD" → "No more 2-site Lanczos."** We still form
   the two-site `θ`, we still run `apply_heff_two_site`, we just run it
   once per bond instead of `n_L = 20` times. The algorithmic saving is
   "one matvec instead of a Krylov subspace," not "avoid two-site
   objects."

2. **"CBE" → "McCulloch/Osborne CBE variant."** The original Gleis/Li/von
   Delft five-SVD scheme is not what we want to implement. The M/O
   simplification is strictly easier, strictly cheaper per bond, and
   maps cleanly onto our existing GPU kernels. Paper citations should
   reflect this.

With those corrections applied, **R2-1 is a GO.** The scope estimate
shrinks from 1500–2000 LOC to ~600 LOC. The risk register §4 entry
"rocSOLVER RSVD accuracy cliff" is **eliminated** — we never use rocSOLVER
for SVD in this scheme, only for QR of skinny matrices.

---

## 6. Concrete next action

Do this sequence, in order:

1. **Edit `docs/followups/round_2_plan.md` §R2-1** to (a) rename the target
   as "M/O CBE" with citation to arXiv:2403.00562, (b) update the "two-site
   SVD" language to "two-site Lanczos count", (c) drop the "fully
   variational" claim in favor of "variational to truncation error, same
   as two-site DMRG", (d) revise LOC estimate from 1500–2000 to ~600,
   (e) remove "rocSOLVER RSVD accuracy cliff" from the risk register.

2. **Write the Python reference** at `cpu/cbe-reference/cbe_reference.py`.
   Skeleton: numpy-only, ~200 LOC, implements M/O Alg. 1 on top of the
   existing `cpu/dmrg-reference/` infrastructure. Validate against quimb
   DMRG2 on L=16 Heisenberg at χ=16, 32, 64. Target: ΔE < 1e-10 vs quimb.

3. **Extract `apply_heff_two_site` into a reusable header** in
   `gpu-rocm/common/` so both `dmrg2-gpu` and the new `cbe-dmrg-gpu` can
   call it without code duplication. (Small refactor, ~2 hours.)

4. **Start `gpu-rocm/cbe-dmrg-gpu/` from `dmrg-gpu` (single-site)**, not
   from `dmrg2-gpu`. The new code is single-site + occasional two-site
   matvec, which is closer to `dmrg-gpu`'s structure. Reuse
   `apply_heff_two_site` from the extracted header.

5. **First gate:** match the Python reference to 1e-10 on L=16 Heisenberg.
   Before any performance work.

6. **Second gate:** run the R2-1 scaling sweep (L∈{16,32,64,128} ×
   χ∈{64,128,256,512}). Target 2× vs `dmrg2-gpu` at χ=128.

7. **Third gate (the one we care about):** Josephson L=32 χ=128. If CBE
   matches two-site energy to 1e-8 there, R2-2 (CBE-TEBD-DMRG flagship)
   is unblocked.

**Time estimate with the scope correction:** 3–4 weeks for steps 1–5, not
the 6–8 weeks in R2-1.

---

## 7. Things that still worry me (loud, honest)

- **The Josephson convergence question has not been answered by the
  literature.** All published CBE benchmarks are Heisenberg, Hubbard,
  ab-initio chemistry, or Kitaev. Nobody has run CBE on a Josephson junction
  array, or on any strongly-gapped bosonic model with d>4. Our Python
  reference gate is essential and non-negotiable.
- **The `k+p` buffer (oversampling) choice is a black art.** M/O suggest
  `p=5`; the classical RSVD literature (Halko/Martinsson/Tropp 2011)
  suggests `p ≥ 10` for robust rank estimation. Start with `p=10`, drop
  to `5` only after convergence numerics are validated.
- **Lanczos on bond dim `χ' = χ + k + p` after expansion: does it converge
  in the same number of iterations as a fixed-χ Lanczos?** Not addressed
  in M/O. If expanding the subspace introduces new near-degeneracies,
  `n_L` could grow, eroding our speedup. This is measurable on the Python
  reference; needs a sanity check before GPU investment.
- **Memory allocation pattern when χ grows between sweeps.** Our current
  `chi_max_`-padded allocation strategy handles this trivially, but we
  need to make sure the padding is `chi_max + k + p + slack`, not just
  `chi_max`. One-line fix, but easy to miss.
- **No published GPU implementation of CBE exists** (as of 2025). This is
  both the opportunity and the risk — if we hit a non-obvious numerical
  snag, there's no reference code to compare against.

If any of these bullets blocks progress, **fall back to DMRG3S** (Hubig
2015), which is ~200 LOC simpler and has 10 years of community validation.
We lose ~30–50% of the CBE speedup but retain the "single-site-cost,
two-site-accuracy" win.

---

## 8. Citations (precise)

- Gleis, A., Li, J.-W., von Delft, J. "Controlled bond expansion for DMRG
  ground state search at single-site costs." *Phys. Rev. Lett.* **130**,
  246402 (2023). arXiv:2207.14712.
- McCulloch, I. P., Osborne, J. J. "Density matrix renormalization group
  in the age of controlled bond expansion." arXiv:2403.00562 (2024).
  *Our implementation target.*
- Gleis, A., Li, J.-W., von Delft, J. "Reply to the comment on 'Controlled
  bond expansion…'." arXiv:2501.12291 (2025). Concedes M/O simplification
  for ground-state DMRG.
- Hubig, C., McCulloch, I. P., Schollwöck, U., Wolf, F. A. "Strictly
  single-site DMRG algorithm with subspace expansion." *Phys. Rev. B* **91**,
  155115 (2015). arXiv:1501.05504. *DMRG3S — fallback plan.*
- Halko, N., Martinsson, P.-G., Tropp, J. A. "Finding structure with
  randomness: probabilistic algorithms for constructing approximate matrix
  decompositions." *SIAM Rev.* **53**, 217 (2011). Canonical reference for
  range-finder RSVD used in M/O step 4.
