# Round 3 — Gen 06: Orthogonalization Strategy for the Persistent Lanczos Kernel

**Sub-agent:** G06 (pair 06) · **Target:** R2-3 per-workgroup persistent Lanczos on MI300X
**Scope:** Choice, algorithm, LDS layout, and bit-stability budget for the
reorthogonalization step inside the single-workgroup Lanczos kernel that
replaces the host-driven `lanczos_eigensolver` in `dmrg2-gpu-opt/src/
dmrg2_gpu_opt_impl.h` (lines 689–853).

---

## 1. Strategy: full reorthogonalization is the right default

`m_max = 20` and `n = chi^2 · d^2 ≤ 64·64·4 = 16384` for the Heisenberg L=32
χ=64 target. Full reorthogonalization (FR) cost is
`O(m · n) = 20 · 16384 ≈ 3.3·10⁵` FMAs per Lanczos step, dwarfed by the
`apply_heff` matvec at `O(chi^3 · d^2 · D) ≈ 10⁹` FMAs. FR overhead is ≤ 0.1%
of matvec. Selective reorthogonalization (Parlett-Scott, 1979) and partial
reorthogonalization (Simon, 1984) are engineered to save FR cost when
`m ∼ 10³`; at `m ≤ 20` they only add branching and host coordination that we
explicitly want to avoid in a persistent kernel. **Decision: FR every step,
unconditional.**

The current baseline (line 748–757) already does a once-through classical
Gram-Schmidt via a single gemv pair. That is mathematically CGS-1. CGS-1 has
a loss-of-orthogonality bound `‖I - V^T V‖ = O(κ(V)·ε)` where
`κ(V) → ∞` as vectors become near-linearly dependent — exactly the situation
at convergence of a Krylov method. This is the root cause of ghost
eigenvalues we have occasionally observed on L=64 runs when Davidson falls
back to Lanczos.

## 2. Algorithm: CGS-2 (classical Gram-Schmidt, applied twice)

We run classical Gram-Schmidt, check the norm reduction, and if the new
vector lost more than half its length we run CGS a second time. In practice
for an m≤20 Krylov basis we always run both passes unconditionally — the
branch-free version is friendlier to the wavefront.

**Justification.** Giraud, Langou and Rozložník ("The loss of orthogonality
in the Gram-Schmidt orthogonalization process", *Computers & Mathematics with
Applications* 50 (2005), 1069–1075) prove that CGS-2 satisfies
`‖I - Q^T Q‖ ≤ c·m·ε` **independent of κ(V)**, provided
`c·m·κ(V)·ε < 1/2` at the CGS-1 stage — a trivial condition for
`m=20, κ < 10¹⁵`. In contrast, MGS gives `‖I - Q^T Q‖ ≤ c·m·κ(V)·ε` which
degrades with κ. The famous Paige (1980) analysis of plain Lanczos produces
precisely the κ-dependent loss; CGS-2 is the standard cure.

**Why not MGS.** MGS orthogonalizes against each previous basis vector
sequentially: each dot product depends on the residual updated by the
previous dot. Inside one workgroup this serializes the wavefront — the
`(m-1)` dots cannot be merged into a single wave-level reduction. With 64
wavefronts in the CU and `m ≤ 20` dots, MGS would leave 44 wavefronts idle
per step. CGS-2's "dot against all at once" pattern lets us pack all `(iter+1)`
dots into a single batched reduction.

**Why not block-CGS-2.** Block-CGS-2 amortizes Householder-like costs over a
block of `k > 1` new vectors; Lanczos is strictly one-new-vector-per-step, so
the block size collapses to 1 and block-CGS-2 reduces to CGS-2.

## 3. Kernel mapping inside one workgroup

Wavefront = 64 lanes. Workgroup = 256 threads = 4 wavefronts. Assume the
current Lanczos vector `w ∈ ℝⁿ` is tiled across the 4 wavefronts in LDS +
registers, with `n ≤ 16384` so each thread owns 64 doubles.

Per CGS pass, for iteration `i` with `i+1` previous vectors `v₀…v_i`:

```
// Pass 1 (CGS)
for j in 0..=i:
    partial[lane] = sum over owned slice of conj(v_j[k]) * w[k]
    c_j = wavefront_reduce(partial)         // __shfl_xor butterfly
    c_j = workgroup_reduce(c_j, LDS)        // 4-wave tree, 1 barrier
for j in 0..=i:
    w[k] -= c_j * v_j[k]                    // vectorized, no barrier

// Pass 2 (CGS repeat — same body)
for j in 0..=i:
    partial[lane] = sum over owned slice of conj(v_j[k]) * w[k]
    d_j = workgroup_reduce(partial, LDS)
for j in 0..=i:
    w[k] -= d_j * v_j[k]
    c_j += d_j                               // accumulate for α_i bookkeeping
```

All dot reductions happen inside the workgroup with one `__syncthreads`
(amdgcn `s_barrier`) per reduction. No host-pointer rocBLAS dot, no
host-to-device synchronization. `α_i` is `c_i` from the first pass against
`v_i` itself; `β_i = ‖w‖` is computed identically with one extra workgroup
reduction.

## 4. LDS layout (handoff from G05)

G05 owns the basis archive. We request 1 LDS tile of `m_max × 1 double`
(160 B) for the `c_j` partials and `m_max × 1 double` for `d_j` (160 B), plus
a shared `α[]`, `β[]`, and a scalar `breakdown_flag`. Total scratch this
kernel demands beyond G05's basis archive: **≤ 512 B of LDS**, well inside
the 64 KB budget. No spill risk.

## 5. Breakdown and exit signalling

After the second pass, compute `β_i`. If `β_i < max(tol_break, ε·‖H‖·√n)` we
write `breakdown_flag = 1` into a workgroup-visible LDS word and proceed to
the final Ritz recombination using only the `iter+1` vectors built so far.
`‖H‖` is approximated by `max_j |α_j| + 2·max_j |β_j|` accumulated on the fly
(Gershgorin). The flag is flushed to a single device-side `int` that the
host polls only at kernel return — no mid-kernel device→host stall.

The same mechanism carries the "Ritz energy converged to 1e-10" early-exit:
the tridiagonal solve is done on the device by a single thread with a
20×20 QR (`O(m³) = 8000` FMAs, ~1 µs). Eigenvalue delta is compared to
`tol_eig_conv = 1e-12`.

## 6. Bit-stability prediction

**Claim.** With CGS-2, single-workgroup reductions in fixed lane order, and
the same LAPACK `dstev` tridiagonal solve as the baseline (done on-device via
a fixed-order mini-QR with identical reflector sign conventions), the
persistent kernel will reproduce the host-driven baseline ground-state
energy on Heisenberg L=32 χ=64 to within **Δ E ≤ 4·ε·|E| ≈ 1.4·10⁻¹⁵**.
This is *not* 1 ULP but it is within the accumulated rounding noise of a
20-step Lanczos (`O(m · ε · ‖H‖)` per Kahan's rounding bound).

A **strict 1 ULP** match would require CGS-2 *plus* a final full-precision
Gram-Schmidt polish of the reconstructed Ritz vector `|ψ⟩ = Σᵢ c_i v_i`
against the basis, which adds one extra reduction and costs < 0.1%. We
propose shipping with the polish enabled behind `DMRG_LANCZOS_POLISH=1`
(default on for bit-compare regression tests, off for production) because
dstev itself is not bitwise deterministic across rocBLAS versions.

## 7. Summary of decisions

| Knob | Choice | Cost vs matvec |
|---|---|---|
| Strategy | Full reorthogonalization every step | < 0.1% |
| Algorithm | CGS-2 unconditional | 2 × (m-1) dots/axpys |
| Reduction | Wavefront butterfly + 4-wave tree in LDS | 1 barrier/dot |
| Breakdown tol | `max(1e-14, ε·‖H‖·√n)` | free |
| Exit signal | Single LDS flag → device int, polled on kernel return | 0 stalls |
| LDS scratch | ≤ 512 B beyond G05's basis archive | negligible |
| Bit-stability target | `ΔE ≤ 4·ε·|E|` default; strict 1 ULP with `POLISH=1` | +0.1% |

---

## Three-sentence summary

Use classical Gram-Schmidt applied twice (CGS-2) unconditionally for full
reorthogonalization at every Lanczos step, relying on Giraud-Langou-Rozložník
(2005) which proves CGS-2 is κ-independent backward stable while MGS is not,
and map each pass to a single workgroup-scoped butterfly reduction so no
dependency chain serializes the wavefronts. LDS scratch stays under 512 B on
top of G05's basis archive, breakdown and convergence are signalled through
a single device-int polled only at kernel return, and `‖H‖` is tracked
on-the-fly via Gershgorin from the already-computed `α_i`,`β_i`. The
resulting persistent kernel is predicted to match the host-driven baseline
Heisenberg L=32 χ=64 ground-state energy to `ΔE ≤ 4·ε·|E| ≈ 1.4·10⁻¹⁵`, with
strict 1-ULP parity available behind an optional final Ritz-vector polish
pass.
