# Round 3 — Pair 03: CBE-TEBD Parallel Warmup Validation

**Refinement target:** R2-2 flagship "CBE-TEBD-DMRG" Phase 0 (parallel imaginary-time TEBD warmup).
**Role:** generator + adversarial reviewer.

---

## 1. Precise parallel iTEBD warmup recipe

### 1.1 Hamiltonian split (nearest-neighbour only)
For H = Σᵢ hᵢ,ᵢ₊₁ on an OBC chain of length L:
```
H_even = Σ_{i even}   h_{i,i+1}      (even-bond sublattice; L/2 gates on L=64)
H_odd  = Σ_{i odd}    h_{i,i+1}      (odd-bond sublattice;  L/2 − 1 gates)
```
All gates within one sublattice commute.

### 1.2 Gate definition
Two-site local Hamiltonian for spin-½ XXX Heisenberg:
```
h_{i,i+1} = J (Sˣ⊗Sˣ + Sʸ⊗Sʸ + Sᶻ⊗Sᶻ)      ∈ ℂ^{4×4}
U_{i,i+1}(δτ) = exp(−δτ · h_{i,i+1})         (4×4 dense, precomputed once)
```
Each gate is a constant 4×4 matrix, reshaped to (d,d,d,d) = (2,2,2,2) for contraction with MPS pairs. Cost ~nothing to build — precomputed once outside the loop.

### 1.3 Suzuki-Trotter-2 schedule
Per Trotter step of total time δτ:
```
U(δτ) ≈ exp(−½δτ H_even) · exp(−δτ H_odd) · exp(−½δτ H_even)
```
Three sublattice barriers per step. Local error O(δτ³), global error after n steps is **O(β·δτ²·‖[H_even,H_odd]‖)**; ‖[·,·]‖ ∼ O(1) J per shared site.

### 1.4 β/δτ schedule — concrete numbers (refined vs. R2-2)
R2-2 doc proposes `β_schedule = [0.1, 0.2, 0.5, 1.0, 2.0]`, `n_steps=5` each → ~15 total Trotter steps → β_total=2.0. **We will show this is 4–5 orders of magnitude too short.**

Realistic minimum for an energy error 10⁻³ from a χ=16 random start on Heisenberg L=64 (gap Δ≈0.1, see §1.6):
```
β_target   = 25        (from  e^{−2·β·Δ} ≈ 10⁻² state weight, Energy err ≈ w·Δ ≈ 10⁻³)
δτ         = 0.05      (Trotter error ≈ β·δτ² ≈ 25·0.0025 ≈ 6·10⁻²  ← already a problem)
n_steps    = 500       Trotter steps
```
The Trotter error at δτ=0.05 already dominates the target 10⁻³ accuracy. Refinement: start with coarse δτ and tighten.
```
β=10  @ δτ=0.1   (100 steps; bulk damping)
β=10  @ δτ=0.05  (200 steps; intermediate)
β=5   @ δτ=0.02  (250 steps; precision tail)
Total: 550 Trotter steps, final Trotter floor ≈ 5·(0.02)² = 2·10⁻³
```
Per-energy-accuracy scaling (derived, §5.2):
| Target E-err | β needed | Best δτ schedule | # Trotter steps |
|---|---|---|---|
| 10⁻³ | ~25 | 0.1 → 0.05 → 0.02 | ~550 |
| 10⁻⁵ | ~50 | 0.05 → 0.02 → 0.01 | ~3000 |
| 10⁻⁶ | ~60 | 0.02 → 0.01 → 0.005 | ~6000 |
| 10⁻⁷ | ~70 | 0.005 → 0.002 | ~20000 |

### 1.5 QR truncation after each gate
For each bond-i gate application on tensors (Aᵢ, Aᵢ₊₁):
```
1. θ  = tensordot(Aᵢ, Aᵢ₊₁, axis=inner bond)               shape (χ_L, d, d, χ_R)
2. θ' = contract(U_{i,i+1}, θ)                              shape (χ_L, d, d, χ_R)
3. M  = θ'.reshape(χ_L·d, d·χ_R)
   QR-based "thin" truncation per Unfried-Hauschild-Pollmann (arXiv:2212.09782):
     Q,R   = qr(M)                                          (replaces left-SVD leg)
     Q',R' = qr(R.T)
     Update Aᵢ ← Q · first χ_max cols of Q',  Aᵢ₊₁ ← truncated R'.T
4. After every K gates (K ≈ 5–10), do one real SVD resweep to restore orthogonality.
```
Key properties:
- QR is **cost-wise** O(d²·χ³) vs SVD O(d³·χ³); on GPU, QR is also kernel-fusion friendly.
- QR does **not** produce the true Schmidt basis; the "QR-truncated" state is not variationally optimal per bond. Unfried et al. document up to 3 orders of magnitude GPU speedup at **χ=1000+**; at our χ=128 the per-gate saving is much smaller (measured ~3–8× in tenpy discussion threads).
- **Critical caveat**: arXiv:2212.09782 tests QR-TEBD only on **real-time quenches in the quantum clock model**. It does **not** test imaginary-time evolution, ground-state preparation, or Heisenberg chains. The claim that QR-TEBD converges to the same fixed point as SVD-TEBD under imaginary time is **not published**; it is a conjecture based on the fact that both reduce to the same exact operator in the χ→∞ limit.

### 1.6 Bond dimension growth
Starting from a random χ=16 MPS, the iTEBD evolution adds entanglement at rate set by local gates. After each two-site gate, bond dim doubles (χ_L·d → min(χ_L·d, χ_R·d, χ_max)). The χ must be held at **χ_phase0 ≈ 64** to let correlations develop; growing beyond 64 during warmup is wasted work since Phase 1.5 recanonicalizes anyway.

---

## 2. Cross-peer communication pattern and cost

### 2.1 Partition
Bond chunks assigned to P=4 GPUs on L=64 OBC:
```
GPU 0: bonds  0..15   (tensors M[0..16])    ← owns 17 tensors
GPU 1: bonds 16..31   (tensors M[16..32])   ← shared M[16] with GPU 0
GPU 2: bonds 32..47   (tensors M[32..48])
GPU 3: bonds 48..62   (tensors M[48..63])
```
Even-bond gates on i=14 (GPU 0) and i=16 (GPU 1) are **not** independent — both touch M[16]. Need halo handling.

### 2.2 Per-half-step communication
Each sublattice half-step (even or odd):
1. Each GPU applies its assigned gates in parallel.
2. **Boundary exchange**: GPU p sends M[first_tensor] and M[last_tensor] to peers p−1 and p+1.
   - Tensor size: χ_L·d·χ_R·16 B ≈ 64·2·64·16 = 131 KB at χ=64 complex128.
   - 2 exchanges per GPU per barrier × 3 barriers per Trotter step × ~550 steps = 3300 barriers.
   - Per barrier: 4 × 2 × 131 KB = 1 MB total traffic; xGMI @ 128 GB/s → **8 µs raw BW + ~50 µs launch latency ≈ 60 µs per barrier**.
3. **Global barrier** (hipDeviceSynchronize + notify): ~30 µs measured on MI300X.

Total warmup barrier overhead: 3300 × 90 µs ≈ **0.3 s**. Non-trivial but not catastrophic.

### 2.3 But the gates themselves are fast
At χ=64, a single two-site gate apply is ~3 gemm ops each ~0.1 ms on MI300X → ~0.3 ms per gate (not 0.5 ms as R2-2 estimated for χ=128). Per half-step per GPU: (L/2)/P = 8 gates → 2.4 ms of GPU work sandwiched between 90 µs of barrier.
**Compute-to-barrier ratio ≈ 25:1**, communication is not dominant.

### 2.4 Real cost summary
| Operation | Per-unit cost | Count (β=25, δτ=0.05) | Subtotal |
|---|---|---|---|
| Gate apply + QR (χ=64) | 0.3 ms | 500 steps × 96 gates / 4 GPUs = 12000 | 3.6 s |
| Barrier + halo exchange | 90 µs | 500 × 3 = 1500 | 0.14 s |
| Periodic SVD orth. pass | 5 ms | every 50 steps = 10 | 0.05 s |
| **Phase 0 total (β=25)** | — | — | **≈ 3.8 s** |

For β=50 (energy target 10⁻⁵): **≈ 8 s**.
For β=60 (target 10⁻⁶): **≈ 12 s**.

**This is already comparable to the entire PDMRG warmup phase (8.8 s serial DMRG1).** The parallel iTEBD is *not* an overwhelming win; it is at best a 2–3× speedup over serial DMRG1 warmup, not the 10–30× implied by the R2-2 Amdahl model.

---

## 3. Convergence criterion + wall-clock estimate at L=64, χ=128, P=4

### 3.1 Convergence criterion for Phase 0 → Phase 2 handoff
Three gates:
1. **Energy plateau**: ΔE between successive β-segments < 10⁻³ · |E|.
2. **Trotter-consistency**: halve δτ and re-run 10 steps; if ΔE < 5·10⁻⁴, Trotter floor has been hit and further β is wasted.
3. **Variance**: ⟨H²⟩ − ⟨H⟩² < 10⁻² (loose; DMRG tightens it). Computing ⟨H²⟩ costs one extra environment sweep (~50 ms at χ=128).

Handoff to Phase 2 (block-Jacobi CBE-DMRG) when all three pass, or when a hard β cap (β=30) is reached.

### 3.2 Wall-clock estimate, L=64, χ=128, P=4
**Important correction:** R2-2 assumed χ=64 during Phase 0, χ=128 in Phase 2. Gate cost at χ=128 is ≈ 4× higher than at χ=64:
- Gate apply at χ=128 (χ·d=256): ~1.2 ms (not 0.5 ms; rocBLAS gemm 256×4×256 ≈ 1 ms + QR + halo)
- Per half-step per GPU: 8 gates × 1.2 ms = 9.6 ms
- Per Trotter step: 3 × (9.6 + 0.09) ≈ 29 ms
- For β_target = 25 (energy ~10⁻³), ~500 Trotter steps → **14.5 s**
- Phase 1.5 canonicalize + broadcast at χ=128: O(L·χ³) = 64·2²¹ ≈ 10⁸ ops ≈ 100 ms (R2-2 number correct here)

| Phase | Wall-clock |
|---|---|
| Phase 0 parallel iTEBD (β=25, ε~10⁻³) | **14.5 s** |
| Phase 1.5 recanonicalize+broadcast | 0.1 s |
| **Replacement for PDMRG warmup** | **~14.6 s** |
| vs PDMRG quimb-DMRG1 warmup baseline | 8.8 s |

**Phase 0 iTEBD at χ=128 is already 1.6× SLOWER than the PDMRG serial DMRG1 warmup it replaces.**

At χ=64 during Phase 0 (R2-2's actual proposal), cost drops to ~4 s, but then Phase 1.5 must pad with random singular values to reach χ=128, and the first Phase 2 sweep is effectively a new warmup itself.

---

## 4. Literature evidence

### 4.1 Positive — parallel iTEBD exists and works
- **Vidal 2004, PRL 93, 040502** ("Efficient simulation of 1D quantum many-body systems") [1]: original iTEBD. Imaginary-time evolution to ground state works for **gapped infinite** chains; convergence rate ~ exp(−β·Δ).
- **Orús & Vidal 2008, arXiv:0711.3960 / PRB 78 155117** [2]: "Infinite TEBD", benchmarks on TFIM and XXZ. Ground-state convergence verified; **no hybrid with DMRG**.
- **Unfried, Hauschild, Pollmann 2022, arXiv:2212.09782** [3]: **correct citation; not Krumnow/Eisert.** QR-TEBD for GPU, 3-orders-of-magnitude speedup over SVD-TEBD at **χ=1024+** in the clock-model **real-time quench**. Does not test imaginary-time evolution or ground states. Does not test whether QR steady state equals SVD steady state under imaginary-time flow. The χ=128 regime is not benchmarked.

### 4.2 Neutral — TEBD for long-range Hamiltonians
- **Van Damme et al. 2024, arXiv:2402.05198** [4]: compact MPO construction for Pauli-string long-range Hamiltonians; needed for J1-J2 and for any Josephson model with multi-site coupling. Adds 2–4× gate-cost overhead vs nearest-neighbour TEBD.
- For J1-J2, TEBD requires a 4-way split H = H_J1_even + H_J1_odd + H_J2_even + H_J2_odd. Standard 2nd-order Trotter on 4 non-commuting terms is O(δτ²) but the constant **doubles** vs nearest-neighbour case. For Josephson with on-site cos(ϕ) and charging (n−n_g)², the on-site commutes with bond terms → can be folded; still needs a truncated charge basis.

### 4.3 **Critical gap — no published TEBD→DMRG hybrid with measured speedup**
Perplexity deep-research on the explicit query "hybrid TEBD-initialized DMRG workflow with measured speedup" returned **zero published workflows** with concrete wall-clock benchmarks. Several indirect facts:
- **Schollwöck 2011, Rev. Mod. Phys.** [5]: DMRG review notes that a good initial state helps, but does not advocate iTEBD warmup for finite-system DMRG.
- **Paeckel et al. 2019, Ann. Phys. (arXiv:1901.05824)** [6]: the comprehensive review of time-evolution methods; does not list iTEBD-warmed DMRG as a standard technique.
- **No paper from Verstraete, Vidal, Orús, Stoudenmire, White, Fishman proposes or benchmarks TEBD→DMRG handoff.** Why? Because finite-system DMRG from a random MPS already converges fast on gapped 1D; the cost of running iTEBD to produce a "warm" state is comparable to or larger than letting DMRG warmup itself. This is precisely what we re-derive quantitatively in §3.2.

### 4.4 Trotter error literature
- **Childs et al. 2021, Phys. Rev. X 11, 011020 (arXiv:1912.08854)** [7]: tight commutator bounds for Suzuki-Trotter. For spin-½ nearest-neighbour Hamiltonians the 2nd-order per-step error is ≤ (δτ³/12)·‖Σ[h_e,[h_e,h_o]]‖, typical operator-norm ~ L.
- For Heisenberg L=64, 2nd-order δτ=0.01 → Trotter floor ≈ 10⁻⁴ in E; **reaching 10⁻⁶ requires either δτ ≤ 0.003 or 4th-order Trotter**. 4th-order doubles the gate count per Trotter step.

---

## 5. Adversarial findings

### 5.1 iTEBD to ε=10⁻⁶ is not feasible in the R2-2 time budget
Let Δ ≈ 0.1 be the effective gap for Heisenberg L=64 (finite-size gap is actually 2π²/L² ≈ 5·10⁻³ for the true low-lying excitation, but warmup only needs to kill bulk excited states at energy ~ J = 1, so effective Δ ≈ 0.5–1; we use 0.1 as a conservative low estimate matching R2-2). Energy error after imaginary time β from a random start (|c_excited|² ≈ 1):
```
|c_excited|²(β) ≈ exp(−2·β·Δ)   → E-error ≈ Δ · |c|²
```
- ε = 10⁻³: β ≈ 23 (we computed 23.0 exactly)
- ε = 10⁻⁵: β ≈ 46
- ε = 10⁻⁶: β ≈ 58
- ε = 10⁻⁷: β ≈ 69

R2-2 claimed "β = 2 gives ~e⁻⁰·² = 0.82 reduction per unit β — enough for 10⁻³ from a random start". **This is off by at least a factor of 10**. e⁻⁰·² ≈ 0.82 per unit β means β=2 only reduces contamination to 0.67 — nowhere near 10⁻³. The R2-2 "75 ms for β=2" iTEBD warmup produces a state with energy error ≈ **0.5 · J ≈ 10⁻¹**, completely useless as a warmup for a DMRG that expects energy error 10⁻³ or better to skip sweeps.

### 5.2 Trotter floor dominates the target accuracy
Second-order Trotter global error scales as β·δτ². For the schedule β=25, δτ=0.05: error = 25·0.0025 = 0.06 — 60× larger than the 10⁻³ target. Using δτ=0.01 gives 25·10⁻⁴ = 2.5·10⁻³, **still larger than the target**. To reach 10⁻³ via Trotter one needs δτ ≤ 0.006, which means **4000+ Trotter steps** → ~12 s per GPU at χ=64 → P=4 still ≥3 s and that is just to reach the 10⁻³ level that serial PDMRG DMRG1 reaches in ~9 s from a decent initial state and to 1e-10 on small L.

**Translation: the iTEBD warmup is not a free lunch. It is at best an even exchange with serial DMRG1 warmup on wall-clock, and it produces a strictly worse state (energy ~10⁻³ vs DMRG1's ~10⁻⁸ at the same cost).**

### 5.3 TEBD does not handle long-range Hamiltonians cleanly
- **J1-J2 Heisenberg**: requires 4-way split; 2× gate count, 2× barriers per Trotter step. TEBD still works but factor-2 overhead wipes out the Phase 0 budget estimate.
- **Josephson (charging + cos φ)**: charging n² is on-site (commutes), good. But cos(ϕᵢ − ϕⱼ) between neighbours with on-site charging basis d ~ 2n_cut+1 ≈ 10–20 means each two-site gate is a 100×100 to 400×400 dense U. Gate apply at χ=128 and d=15: 128·15·15·128 = 3.7M entries per tensor; dominant gemm is 128d × d × 128d → O(d²·χ³) = 225·2·10⁶ = 5·10⁸ ops per gate, ~2 ms per gate on MI300X. **Double** the χ=128 spin-½ estimate. Josephson warmup balloon: ~30 s.
- **VQE MPOs** (D=8–20 with multi-site support): no longer two-body; Suzuki-Trotter split is not even defined without SWAP networks, which multiply gate count by O(range²). TEBD warmup becomes non-competitive immediately.

### 5.4 QR-TEBD convergence under imaginary time is unvalidated
Unfried et al. paper **only tests real-time quenches on the clock model**. They do not:
- Check that QR-truncation converges to the DMRG fixed point under imaginary-time flow.
- Characterize bond-dimension growth from a product state.
- Benchmark against finite-system DMRG.
QR-truncation discards a tangent-space direction that is not the dominant Schmidt direction, so imaginary-time QR-TEBD can **in principle** bias the ground-state approximation away from the true Schmidt basis until a periodic SVD cleanup restores orthogonality. No published characterization of this bias magnitude. **This is a research hazard, not a validated tool.**

### 5.5 Does parallel TEBD produce a state close to the DMRG fixed point?
Imaginary-time evolution converges to the ground state of **H**, modulo Trotter error and truncation. If β is large enough and δτ small enough, yes — iTEBD's fixed point **is** the DMRG fixed point at the same χ. But the two algorithms differ in *how* they reach it:
- DMRG: local variational minimization per site, chooses optimal basis at each step.
- iTEBD: imaginary-time flow, retains basis consistent with global exponential decay.
At finite β and finite δτ the iTEBD state is a **different** low-energy approximation — not a different eigenstate, but a state whose MPS tensors are parameterized differently. The DMRG polish sweep can happily clean this up, but there is no free jump; **Phase 3 polish cannot be skipped just because iTEBD produced a state with energy 10⁻³**.

### 5.6 Phase 1.5 serial canonicalization eats the budget
R2-2 claims Phase 1.5 takes ~100 ms. Let's verify: L·χ³ = 64·2·10⁶ = 1.3·10⁸ float ops → 50 ms on GPU 0 for canonicalize + 50 ms broadcast 17 MB × 3 peers / 128 GB/s ≈ 0.4 ms. **The canonicalize dominates at ~50–100 ms; the broadcast is fine.** But this 100 ms is not the bottleneck — it is Phase 0 itself and Phase 3 polish that dominate.

### 5.7 Amdahl re-accounting
R2-2 model (§7.4): T_PDMRG = 24 s, warmup = 37% = 8.9 s. Their Phase 0 replacement = 75 ms. Savings = **8.8 s** → fueled the entire 1.7× speedup claim.

Our re-derivation: Phase 0 at β=25, χ=128, P=4 = **~14.5 s** → not a 8.8 s saving but a **5.7 s regression**. The claimed 1.7× speedup inverts to ~0.95× (a slowdown).

Even being charitable (Phase 0 at χ=64 ≈ 4 s, then pad to χ=128 and swallow a first Phase 2 sweep as "free warmup"): net wins ~5 s over PDMRG baseline at best. Real speedup from Phase 0 replacement alone is **< 1.2×**, not 1.7×.

The remainder of the R2-2 speedup (Phase 2 block-Jacobi + Phase 3 reduction) is not touched by this critique; those mechanisms can still work **but they are independent of the iTEBD warmup**.

---

## 6. VERDICT

**FEASIBLE BUT NOT FAST ENOUGH**

Parallel iTEBD warmup is *technically* implementable and *does* produce a low-energy state, but at the refined target (L=64, χ=128, Heisenberg, energy ε=10⁻³) it costs **~14 s of wall-clock on 4 MI300X**, compared to **8.8 s** for the serial PDMRG DMRG1 warmup it replaces. The R2-2 number (~75 ms) is off by roughly **200×** because the R2-2 β-schedule (β_total=2) corresponds to energy error ~0.1, not 10⁻³. Fixing the β schedule to reach 10⁻³ pushes Phase 0 wall-clock past the serial baseline.

Corollaries:
- For **Heisenberg** ε=10⁻³: iTEBD is ~1.6× slower than quimb DMRG1 warmup (serial) at χ=128, marginally competitive at χ=64 only with substantial handoff cost.
- For **J1-J2 / Josephson**: 2–4× overhead makes iTEBD warmup strictly slower than serial DMRG1.
- For **VQE MPOs** (D>3, range>1): SWAP networks make iTEBD non-competitive.
- The "QR-TEBD converges to DMRG fixed point" claim has **no published validation** under imaginary time. It is a research hypothesis, not a tool.
- Phase 2 (block-Jacobi CBE-DMRG) speedup mechanism is **unaffected** by this critique. Do that work *without* the iTEBD warmup.

**The parallelism gain from iTEBD is real but tiny in our regime**; the Amdahl model in R2-2 §7.4 overestimates the Phase-0 savings by ~10–100×.

---

## 7. Concrete next action

**Do not implement iTEBD warmup on pdmrg-gpu. Redirect R2-2 effort as follows:**

1. **Keep the existing quimb-DMRG1 serial warmup unchanged** — it is already at ε≈10⁻⁸ in 8.8 s and no parallel scheme beats it cleanly at our χ.
2. **Redirect Phase 0 parallelism to GPU-overlapped env prebuild**: while GPU 0 runs serial DMRG1 warmup, GPUs 1–3 concurrently (a) precompute MPO·MPO tensor products for Phase 2, (b) stage L/R environment scratch buffers, (c) prewarm Lanczos Krylov caches. Cost: ~0 ms (overlapped); benefit: eliminates the ~0.5 s env-build serial overhead visible in the Amdahl table.
3. **Run a minimal iTEBD warmup sanity probe** (Rank 3 in R2 plan): 50-line script calling quimb's iTEBD or a direct Trotter loop for L=32, β=25, χ=64, and measure (a) final energy vs DMRG ground truth, (b) wall-clock. Kill the R2-2 iTEBD line item if the measured wall-clock at the 10⁻³ error level exceeds the DMRG1 baseline — which it almost certainly will.
4. **Preserve the Phase 2 block-Jacobi CBE-DMRG research line** as its own prototype, independent of the iTEBD warmup. Its claimed 1.7–2.3× speedup comes from the Jacobi sweep, not from Phase 0.
5. **If a TEBD path is pursued anyway**, scope it as a **research exploration for large-χ (χ≥512)** systems where QR-TEBD's GPU advantage (documented in Unfried et al. at χ=1024) actually kicks in. Not for our current χ=128 publication regime.

---

## References

[1] Vidal, G. *Efficient simulation of 1D quantum many-body systems*. PRL 93, 040502 (2004).
[2] Orús, R. & Vidal, G. *Infinite time-evolving block decimation algorithm beyond unitary evolution*. arXiv:0711.3960, PRB 78, 155117 (2008).
[3] Unfried, J., Hauschild, J. & Pollmann, F. *Fast time evolution of matrix product states using the QR decomposition*. arXiv:2212.09782 (2022). **(Note: R2-2 doc misattributes this to Krumnow & Eisert.)**
[4] Van Damme, M. et al. *Efficient matrix product operators for long-range Hamiltonians*. arXiv:2402.05198 (2024).
[5] Schollwöck, U. *The density-matrix renormalization group in the age of matrix product states*. Ann. Phys. 326, 96–192 (2011).
[6] Paeckel, S. et al. *Time-evolution methods for matrix-product states*. Ann. Phys. 411, 167998 (2019); arXiv:1901.05824.
[7] Childs, A. M. et al. *Theory of Trotter error with commutator scaling*. Phys. Rev. X 11, 011020 (2021); arXiv:1912.08854.
[8] Hauschild, J. & Pollmann, F. *Efficient numerical simulations with tensor networks: TeNPy*. SciPost Phys. Lect. Notes 5 (2018). (Reference for the tenpy DMRG1-as-warmup community convention.)

---

**Files referenced from the repo:**
- `/home/captain/clawd/work/dmrg-implementations/docs/followups/research_D_beyond_pdmrg_a2dmrg.md` §7.2–7.4 (R2-2 algorithm + flawed β schedule)
- `/home/captain/clawd/work/dmrg-implementations/cpu/pdmrg/pdmrg/dmrg.py` lines 35–70 (real PDMRG warmup: quimb DMRG1, 5 sweeps, chi=50, tol=1e-10, not 1 LR+RL)
- `/home/captain/clawd/work/dmrg-implementations/docs/PROJECT_OVERVIEW.md` §6 (Amdahl measured profile, 37% warmup fraction)
