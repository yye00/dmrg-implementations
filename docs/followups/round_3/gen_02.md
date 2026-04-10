# Round 3 — Gen 02: CBE α-Schedule and χ-Growth Policy for R2-1

**Sub-agent:** G02 (pair 02) · **Target:** R2-1 CBE-DMRG GPU backbone
**Scope:** Concrete shift parameter α schedule and bond-dimension growth rule,
calibrated for Heisenberg L∈{32,64,128}, χ∈{128,256} and Josephson d=3 D=4
L∈{16,32}, χ∈{64,128} on MI300X.

---

## 1. Background and notation

Following McCulloch/Osborne (arXiv:2403.00562) and Gleis/Li/von Delft
(arXiv:2207.14712), CBE adds a `k`-vector expansion block to the left/right
isometry of the currently active site. The shift parameter α plays two roles
depending on the variant used:

1. In DMRG3S / Hubig et al. it is the **mixing coefficient** multiplying the
   perturbation term `α · (H_L ⊗ I) · ψ` added to the reduced density matrix
   before SVD truncation. Large α opens the bond basis to new symmetry
   sectors; too small α stalls.
2. In CBE proper it rescales the **tangent-space enrichment block** before the
   small RSVD that selects the expansion directions.

Our implementation follows (2). The enrichment block `T_i ∈ R^{(d·χ) × (w·k)}`
is formed, then we solve an RSVD of `α · T_i`. The singular values of the
enrichment block are compared against `σ_keep` (the smallest kept singular
value of the current site), and the top `k_eff ≤ k` are admitted if they
exceed `α · σ_keep`. α therefore acts as a **relative acceptance threshold**,
not a mixing coefficient, and lives on the natural scale of `σ_keep / σ_max`.

Published CBE works use α in the range `[10⁻³ , 10⁻¹]` with most examples at
α ≈ 10⁻². Hubig-DMRG3S uses α ≈ 10⁻⁴ because the "shift" there is an additive
noise rather than a relative cutoff.

## 2. Starting value α₀

**α₀ = 5·10⁻² = 0.05** for both Heisenberg and Josephson.

Justification: at the start of the sweep after TEBD warmup (R2-2 handoff),
the MPS has entanglement spectrum roughly consistent with a `χ=χ_warm` state
where `χ_warm` is 1/2 to 1/4 of the target `χ_max`. The enrichment block must
therefore be aggressive enough to double the bond basis in 2–3 sweeps. With
α₀ = 0.05 and typical Heisenberg L=64 runs, the smallest kept singular value
sits around σ_keep/σ_max ≈ 10⁻³, so admission threshold `α·σ_keep ≈ 5·10⁻⁵`
sweeps up directions that would have been truncated at σ ≈ 10⁻⁴–10⁻⁵ — exactly
the "dark" Schmidt vectors CBE targets. This matches §IV of Gleis/Li/von Delft
and the default in SyTen's CBE mode.

## 3. Per-site update rule

After each bond update we compute two quantities already free in the sweep:

- `w_i` = discarded weight from the site-i truncation (sum of squared dropped σ)
- `r_i` = Lanczos residual from the local eigensolve

Then:

```
α_new = clip( α_old · g(w_i, r_i, s) ,  α_min,  α_max )

g = (w_i / w_target)^p · max(1, r_i / r_target)^q · β^s
```

with defaults:

| Symbol | Value | Rationale |
|---|---|---|
| `w_target` | `1·10⁻⁸` | Standard DMRG trunc-weight target for ΔE≈1e-10 |
| `r_target` | `1·10⁻⁶` | 100× tighter than inner Lanczos tol |
| `p` | `0.5` | Square-root decay — Legeza DBSS coefficient |
| `q` | `0.5` | Same, for residual branch |
| `β` | `0.85` | Per-sweep damping |
| `α_min` | `1·10⁻⁴` | Floor — below this behaves like frozen 1-site |
| `α_max` | `2·10⁻¹` | Ceiling — above this the enrichment is noise-dominated |
| `s` | sweep index (0-based from end of warmup) |

The `β^s` term enforces geometric cooling; combined with the `w_i/w_target`
term it self-adapts: a bond that has already dropped below `w_target` drives
α down faster, a bond sitting at `w_i ≫ w_target` keeps α high.

## 4. χ growth rule

**Hard cap:** `χ_i ≤ χ_max`.

**Growth gate:** at the end of each sweep, allow `χ_i := min(χ_max, χ_i + Δχ)`
on bonds where the enrichment RSVD produced `≥ 0.75·k` accepted vectors AND
`w_i > w_target`. This is the standard Hubig "full-block accepted → grow"
rule.

| Target | Δχ per sweep | Sweeps to saturation |
|---|---|---|
| Heisenberg χ=128 | +24 | ≤ 6 (from χ_warm=48) |
| Heisenberg χ=256 | +32 | ≤ 8 (from χ_warm=64) |
| Josephson χ=64  | +12 | ≤ 5 (from χ_warm=24) |
| Josephson χ=128 | +20 | ≤ 6 (from χ_warm=40) |

We disable growth once the median `w_i` over the chain drops below
`w_target / 10 = 10⁻⁹`.

## 5. TEBD warmup handoff (from R2-2 Phase 0)

TEBD should produce an MPS with:

- **Heisenberg L=64 χ_max=256:** `χ_warm = 64`, entanglement spectrum
  converged to `σ_min/σ_max ≥ 10⁻³`. TEBD Trotter step `dt=0.05`, imaginary
  time `τ=4.0` (gapped — overkill). Pass α₀ = 0.05 directly to CBE.
- **Heisenberg L=128 χ_max=256:** `χ_warm = 96`, same α₀.
- **Josephson L=32 χ_max=128:** `χ_warm = 40`, α₀ = 0.08 (slightly larger
  because Josephson has d=3 and richer local Hilbert space).
- **Josephson L=16 χ_max=64:** `χ_warm = 24`, α₀ = 0.08.

TEBD does NOT produce an α — it just produces MPS + canonicalization. CBE
initializes α = α₀ on the first pass and begins the update rule of §3 from
sweep index s=0.

## 6. Termination rule

Converged when ALL three hold on the last full sweep:

1. `|E_{n} − E_{n-1}| < 5·10⁻¹¹` (half the publication threshold for safety)
2. `max_i w_i < 5·10⁻⁹`
3. `max_i r_i < 1·10⁻⁷`

If (1) holds but (2)/(3) don't, do up to 2 extra polish sweeps with α fixed at
`α_min`. If still not converged, emit a warning and return last state — the
caller (R2-2 or a benchmark harness) decides whether to grow χ_max.

## 7. Stall detection and fallback

Stall is declared when either:

- Sign (a): `(E_n − E_{n-1}) > 0` — energy went UP (rare, indicates numerical
  drift or enrichment noise)
- Sign (b): `|E_n − E_{n-1}| < 1·10⁻⁸` AND `max_i w_i > 10⁻⁷` — energy frozen
  but truncation weight still dirty (α is too small to explore)

Fallback ladder (applied in order until one sweep makes progress):

1. Double α: `α := min(α · 2, α_max)`, increase `k` (enrichment rank) by 50%.
2. Force one FULL 2-site sweep using the dmrg2-gpu-opt kernel (stored in a
   sibling static library — see §8). This re-seeds the bond basis.
3. If two consecutive fallbacks fail, abort and return stall status.

## 8. Comparison to the dmrg2-gpu-opt baseline

| Aspect | dmrg2-gpu-opt (current) | CBE-DMRG (proposed) |
|---|---|---|
| Bond expansion | Implicit via 2-site SVD of (d·χ, d·χ) theta | Explicit via α·T enrichment + RSVD |
| Truncation policy | Keep top `χ_max` σ, always | Adaptive: α-gated, Δχ-per-sweep growth |
| χ growth | χ_i set at construction by min-cut cap | Grows from `χ_warm` to `χ_max` over ≤ 8 sweeps |
| Noise / stall recovery | None (fixed-form sweep) | α doubling + 2-site fallback |
| Cost per site | O(d²·χ³·w) SVD | O(d·w·k·χ²) enrichment + O(d·χ³) gauge |
| Sweeps to converge (L=64 χ=256 Heisenberg) | 6–8 measured | **Predicted ≤ 7** (see §9) |

The key behavioral change is that dmrg2-gpu-opt has no handle on exploration
vs exploitation — every bond gets the same full 2-site treatment. CBE with
this α-schedule explicitly cools over sweeps and can be put into an
"exploration-heavy" or "polish" mode by scaling α₀.

## 9. Testable prediction

**CBE with α₀=0.05, β=0.85, p=q=0.5, Δχ=+32/sweep, starting from a
χ_warm=64 TEBD warmup, converges Heisenberg L=64 χ=256 to |ΔE|<1e-10 vs quimb
DMRG2 in ≤ 7 CBE sweeps** (TEBD warmup not counted). Secondary predictions:

- Heisenberg L=32 χ=128: ≤ 5 CBE sweeps
- Heisenberg L=128 χ=256: ≤ 9 CBE sweeps
- Josephson L=16 χ=64: ≤ 6 CBE sweeps
- Josephson L=32 χ=128: ≤ 8 CBE sweeps
- Stall-fallback engagement rate < 5% of sweeps on Heisenberg, < 15% on
  Josephson (which has richer local Hilbert space and is harder to cool).

Falsifier: if Heisenberg L=64 χ=256 requires > 10 CBE sweeps, the α-schedule
is wrong and should be re-tuned (likely α₀ too low or β too aggressive).

---

## Three-sentence summary

Start CBE with α₀=0.05 (Heisenberg) or 0.08 (Josephson) and update per site
via `α ← α·(w_i/10⁻⁸)^0.5·max(1,r_i/10⁻⁶)^0.5·0.85^s`, clipped to
[10⁻⁴, 2·10⁻¹], with bond-dim growth of Δχ=+24/+32 per sweep gated on
`≥0.75k` accepted enrichment vectors. Handoff from R2-2 TEBD warmup is MPS
only (χ_warm ≈ χ_max/4 to χ_max/2), and convergence is declared when
|ΔE|<5·10⁻¹¹, `max w_i<5·10⁻⁹`, `max r_i<10⁻⁷`; stall (|ΔE|<10⁻⁸ with dirty
truncation) triggers an α-doubling then a forced 2-site sweep via the
dmrg2-gpu-opt kernel. Testable prediction: Heisenberg L=64 χ=256 converges in
≤ 7 CBE sweeps to 1e-10 vs quimb DMRG2.
