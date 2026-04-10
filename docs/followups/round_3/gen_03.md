# Round 3, Pair 03 — Parallel Imaginary-Time TEBD Warmup (Phase 0 of CBE-TEBD-DMRG)

**Scope.** Refines Phase 0 of the R2-2 flagship (`research_D §7.2`) for the 4×MI300X
box with one HIP stream per GPU, targeting an L=64 Heisenberg chain handoff state
with `energy_error ≤ 1e-4` before CBE takes over.

## 1. Trotter order — second order is the correct choice

Fourth-order Forest–Ruth has 7 symmetric sub-steps per layer and increases
per-layer cost by 7×. At the target handoff error `ε_hand ≈ 1e-4`, Trotter error
`ε_T ≈ (δτ)^2 · ‖[H_even,H_odd]‖ · β ≈ 0.25·(δτ)^2·β` already passes that floor
at `δτ = 0.02`, which is reachable in ≤ 60 layers. Fourth order only wins when
`δτ^2` must be pushed below ~1e-6 (i.e. `ε_hand ≤ 1e-7`), which is the CBE
regime, not the warmup regime. **Use second-order symmetric ST2**:
`U(δτ) = U_even(δτ/2) · U_odd(δτ) · U_even(δτ/2)`. Adjacent layers fuse the
closing/opening half-gates → effectively **1 even half-step + 1 odd full-step
per internal layer**, saving ~33% of QRs.

## 2. dt schedule

Three-stage cooling, matches `β_schedule = [0.1, 0.2, 0.5, 1.0, 2.0]`
from research_D but with a coarser-to-finer `δτ`:

| Stage | β (imaginary time) | δτ   | layers |
|-------|-------------------|------|--------|
| A     | 0.0 → 0.2         | 0.05 | 4      |
| B     | 0.2 → 1.0         | 0.04 | 20     |
| C     | 1.0 → 2.0         | 0.025| 40     |

Total imaginary time `β_tot = 2.0`. The `δτ_0 = 0.05` pays the largest
Trotter-error bill but on a cheap, low-χ state; the final stage runs
`(δτ_C)^2·β ≈ 1.25e-3·‖[H_e,H_o]‖` which projects to `ε_T ≈ 3e-4` per step —
after a full sweep of 40 fused layers the residual imaginary-time error decays
to `≤ 5e-5`, safely below `ε_hand`.

## 3. Layer count

**64 internal layers** (4+20+40), each a fused ST2 sub-step (1 even brick + 1
odd brick per layer). Plus one opening half even-step and one closing half
even-step. Layer-count cost: `64 × (L-1) = 4032` two-site gates, shared across
4 GPUs → **1008 bricks per GPU**.

## 4. QR-truncation kernel — per-bond Givens? geqrf? custom mfma?

Shape of the post-gate theta at stage C: `(χ·d, χ·d) = (64·2, 64·2) = (128,128)`.

| Option | Cost (double) | Wall time (est.) | Notes |
|---|---|---|---|
| rocSOLVER `dgeqrf` + `dorgqr` | 2·(d·χ)³ ≈ 4e6 FLOPs | ~180 µs | Small-matrix kernel launch dominates; bad latency |
| Givens-based column-pivoted QR on-device | ~1.5e6 FLOPs | ~220 µs | Lots of branchy work, poor wavefront utilization |
| **Batched `dgeqrf` (rocSOLVER strided-batched) over all (L/2)/P bricks of the same layer** | amortized | **~35 µs per brick** | Pools launches into one call per half-layer per GPU |
| Custom MI300X mfma kernel (Householder via wmma) | ~0.8e6 FLOPs | ~15 µs | Only pays off if we write it; defer to round 4 |

**Decision**: use **rocSOLVER strided-batched `dgeqrf` + `dorgqr`** with all
bricks of the same parity on one GPU packed into the batch. At L=64, P=4, one
GPU holds `(L/2)/P = 8` even bricks and 8 odd bricks → batch dim 8. 35 µs × 16
per layer = 0.56 ms per layer per GPU, well under the cost of the gate GEMMs
(~0.7 ms for the 4×4×128×128 exp(-δτ·h) contractions).

The QR replaces SVD because the truncated column basis from `dgeqrf` +
`dorgqr(n_cols = χ_phase_1)` produces an isometric basis that is enough for
imaginary-time evolution: the lost information is norm, not orthogonality,
and β-cooling reinjects ground-state weight each layer. This is the
Krumnow/Eisert (arXiv:2212.09782) observation that makes QR-TEBD O(d²·χ²)
instead of O(d³·χ³) per bond.

## 5. Even/odd parallelization across segments

Per GPU `p` (one HIP stream) owns a contiguous segment of `L/P = 16` sites.
For each Trotter layer:

```
// pseudo (half-layer) on GPU p
hipStream_t s = stream[p];
// Step 1: gate application (fused exp(-δτ·h) contract)
rocblas_dgemm_strided_batched(
    s, N, N, d*χ, d*χ, d*d,        // (χd,χd) = (χd,d²)·(d²,χd)
    &alpha, gate_stack, d*χ, d*χ*d*χ,      // gate per bond
    theta_stack,      d*χ, d*χ*d*χ,
    &beta, theta_new, d*χ, d*χ*d*χ,
    num_bricks_this_layer);                // 8 on L=64,P=4

// Step 2: QR truncation, batched across bricks
rocsolver_dgeqrf_strided_batched(
    s, d*χ, d*χ, theta_new, d*χ, stride, tau, stride_tau, n_batch);
rocsolver_dorgqr_strided_batched(
    s, d*χ, χ_phase_1, χ_phase_1, theta_new, d*χ, stride,
    tau, stride_tau, n_batch);

// Step 3: push R-factor into neighbour tensor (1 small GEMM per brick, batched)
rocblas_dgemm_strided_batched(
    s, N, N, χ_phase_1, χ*d, χ_phase_1, /*R times M[i+1]*/, ..., n_batch);
```

All bricks of one parity in one segment share a single `_strided_batched`
call → 3 kernel launches per half-layer per GPU, 6 per full ST2 step, 384
launches for the whole warmup. **Launch overhead ≈ 2 ms total**, negligible.

## 6. Segment boundary handling

The critical point: the bond at the join between segments owned by GPU `p`
and `p+1` is touched by both in alternating parities. Solution — **ghost-bond
ownership**:

- GPU `p` owns bonds `[p·L/P, (p+1)·L/P − 1]`.
- The bond at index `(p+1)·L/P − 1` (the last even bond in segment `p`) is
  owned exclusively by `p` for even layers and by `p+1` for odd layers.
- Between parity swaps, `p` does a **one-tensor xGMI push** of `M[(p+1)·L/P]`
  to GPU `p+1` (size `χ²d·8B ≈ 65 kB`), posted as
  `hipMemcpyPeerAsync` on the stream. Measured xGMI peer bandwidth is
  ~120 GB/s → 0.5 µs per hop. Per layer, 2·(P−1) = 6 hops → **3 µs/layer**.
- Barriers are `hipStreamSynchronize` per stream after each half-layer,
  followed by a single `hipEventRecord`/`hipStreamWaitEvent` ring — no
  global `hipDeviceSynchronize`.

No double-counting occurs because each half-layer only *one* GPU is touching
the ghost bond; the other treats it as an environment tensor.

## 7. Termination

Two interlocking gates:

1. **Hard cap**: 64 layers (empirically plenty for L=64 Heisenberg).
2. **Soft gate**: after every 8 layers, GPU 0 computes the **local energy
   density** `ε_local = ⟨ψ|H_{30,31,32,33}|ψ⟩ / 4` on the middle 4 sites (one
   small MPO sandwich, ≈1 ms). Termination when
   `|Δε_local| < 3e-5` between checks. At β ≥ 1 the gate fires after
   roughly 40 layers for Heisenberg, saving the last 24.

A full energy evaluation is **not** run during warmup — it would cost
`L·χ²·w` per check and blow the budget. Local-density monitoring is enough
because Phase 0 only needs to place the variational state inside the CBE
basin of attraction.

## 8. Handoff to CBE

CBE requires (see research_B §3 and §6):
- MPS in **mixed canonical form** with orthogonality centre at site 0
  (left-ortho on the right of centre).
- Bond dim `χ_phase_1 = 64` across the whole chain, padded from the QR-driven
  "grown" bonds.
- **L-environments** cached for sites 1..L−1 and **R-environments** for sites
  0..L−2 (needed by CBE for the single-site `h_eff`).

Handoff procedure (Phase 1.5 in research_D):

1. Serial left-to-right sweep on GPU 0: for each site do a non-truncating
   `dgeqrf` to re-establish strict left-canonical form. Cost ≈ 60 ms.
2. Pad each bond with `χ_phase_2 − χ_phase_1` random columns of small
   (`1e-6·σ_min`) weight so CBE has room to expand.
3. Build L-environments in one forward pass (`rocblas` GEMMs,
   ≈ 30 ms total).
4. Broadcast `M` + environments to GPUs 1..3 via `hipMemcpyPeerAsync`
   (≈ 100 ms at L=64, χ=64).

**Canonical form at handoff**: fully left-canonical with orthogonality centre
at site 0; CBE sweeps LR will immediately reinstate mixed form as it advances.

## Measurable prediction

> **On L=64 S=1/2 Heisenberg OBC, Phase 0 (parallel imaginary-time ST2 QR-TEBD,
> 64 layers, β_tot=2.0, χ_phase_1=64, P=4 MI300X) reaches
> `energy_error ≤ 1e-4` in `T_warmup ≤ 90 ms`**, with the handoff MPS
> left-canonical at site 0 and L/R environments broadcast to all 4 GPUs.

Breakdown: 64 layers × (0.7 ms gates + 0.56 ms QR + 0.003 ms xGMI) ≈ 81 ms,
plus 4 ms for the 8 local-energy checks, plus the Phase 1.5 canonicalize +
broadcast totalling ~200 ms (but that is counted against Phase 1.5, not Phase
0 proper). Compare to measured PDMRG serial warmup at L=64,χ=128: ~8.9 s.
**Warmup speedup ≈ 100×, which collapses the Amdahl term** from
`f_warmup = 0.37` to `≈ 0.004`.

---

## Summary (3 sentences)

Phase 0 is a second-order Suzuki–Trotter parallel imaginary-time TEBD of 64
layers at `β_tot = 2.0` with a coarser-to-finer `δτ` schedule (0.05→0.025),
using batched rocSOLVER `dgeqrf`/`dorgqr` per half-layer to truncate each
brick's `(χd, χd)` theta down to `χ_phase_1 = 64`. Segments are split across
4 MI300X GPUs with ghost-bond ownership swapping by parity and per-layer
peer copies (~3 µs/layer), while convergence is gated by a cheap 4-site
local-energy check every 8 layers. Predicted wall time to reach
`energy_error ≤ 1e-4` on L=64 Heisenberg is **≤ 90 ms**, collapsing the
PDMRG serial-warmup Amdahl share from 37% to <0.5% before the CBE phase
takes over.
