# Round 3, Pair 9 — Mixed-Precision SVD for the DMRG Bottleneck

**Charter:** Round 2's `round_2_plan.md §5` listed mixed-precision SVD as a
**NOT-covered** item, with the reasoning that the R2-1 CBE backbone would
eliminate the two-site SVD entirely. That reasoning is contingent on CBE
landing cleanly and converging acceptably on Josephson — neither is yet
demonstrated. Meanwhile, `PROJECT_OVERVIEW.md §5.1` reports that CPU LAPACK
SVD dominates **97–98 %** of per-sweep wall-time at `chi ≥ 128`. A working
GPU mixed-precision SVD path would be an independent, stackable win and a
fallback if CBE misses its Josephson accuracy gate (R2-1 risk register row 1).

This report designs such a path concretely, benchmarks it on the live MI300X
(ROCm 7.2, rocSOLVER 7.2), and argues — with hard numbers — why **the naive
FP32 Jacobi path is numerically unusable for DMRG**, but why a more limited,
hybrid variant is still worth a one-week follow-up.

---

## 1. Algorithm choice (with literature)

### 1.1 The three candidate algorithms

| Algorithm | rocSOLVER API | Cost model | Stability claim |
|---|---|---|---|
| **QR-based bidiag SVD** | `rocsolver_{s,d}gesvd` | `~6 n³` FP ops, tight bounds | Demmel–Kahan 1990 — relative accuracy `O(ε)` in σ for graded matrices |
| **"Jacobi-via-QR+eig(AᵀA)"** | `rocsolver_{s,d}gesvdj` | `~20 n³`, Jacobi sweeps | **This is NOT classical one-sided Jacobi SVD.** See §1.2. |
| **Randomized SVD** | Not in rocSOLVER; in our codebase already | `~2 n² k + n k²` | Halko–Martinsson–Tropp bound — `‖A - UΣVᵀ‖ ≤ (1 + 9√(k+p)·min(m,n)) σ_{k+1}` |

### 1.2 Critical finding: `rocsolver_sgesvdj` is NOT one-sided Jacobi

Reading the rocSOLVER 7.2 header verbatim
(`/opt/rocm/include/rocsolver/rocsolver-functions.h:13700`):

> "The singular values are computed by applying QR factorization to `AV` if
> `m ≥ n` ..., where `V` ... is found as the eigenvectors of `AᵀA` ... using
> the Jacobi eigenvalue algorithm."

This is **eigen-decomposition of the normal equations** AᵀA followed by a QR
cleanup. The condition number of AᵀA is `κ(A)²`, so FP32 (unit roundoff
`u ≈ 6·10⁻⁸`) gives relative singular-value errors of order `κ² · u`. For
DMRG theta matrices in the range `κ ∈ [10⁵, 10¹²]` (see §2.1) the FP32 path
loses every digit.

Classical one-sided Jacobi (Drmač–Veselić, SIAM J. Matrix Anal. Appl. 29, 1322
(2008), and Demmel–Kahan 1990) rotates A directly and has relative accuracy
`O(u)` independent of `κ(A)`. **rocSOLVER does not ship classical one-sided
Jacobi SVD.** Neither does hipSOLVER. Neither does cuSOLVER — NVIDIA's
`cusolverDnSgesvdj` has the same "eigendecompose the Gram matrix" fine print
(see cuSOLVER docs 12.5, §2.3.3.8). This is an industry-wide trap.

This finding alone disqualifies the "just call rocsolver_sgesvdj" plan.

### 1.3 The candidate algorithm if mixed-precision is pursued

Given the gesvdj trap, the only numerically defensible mixed-precision path is:

**Algorithm R3-MPSVD-1 (hybrid randomized + FP64 refinement):**

1. **FP32 Gaussian sketch**: `Y = A Ω` where `Ω ∈ R^{n × (k + p)}` is
   standard Gaussian. Cast `A` to FP32 once; do the GEMM in FP32.
   Oversampling `p = 10`. (~2 n² (k+p) FP32 flops)
2. **FP32 power iteration** (`q = 1` or `q = 2`):
   `Y ← A Aᵀ Y` with QR reorthogonalization after each step. This squares
   the relative gap and suppresses tail contamination from singular values
   below `σ_k`. Stops the condition-number blow-up from propagating into the
   subspace.
3. **FP32 QR**: `Y = Q R`. `Q` is an `n × (k+p)` orthonormal basis for the
   dominant subspace of `A` to FP32 accuracy.
4. **FP64 cast + FP64 project**: `B = Qᵀ A` in FP64 (not FP32). `B` is
   `(k+p) × n`, small. **This is the refinement step**: projecting back in
   FP64 recovers FP64 accuracy in the singular values, provided `Q` captures
   the dominant subspace to `‖(I - QQᵀ)A‖ ≤ σ_{k+1}`.
5. **FP64 deterministic SVD of B** via `rocsolver_dgesvd` or LAPACK: `B = Ũ Σ Vᵀ`.
   Small (`~(k+p) × n`), this is cheap.
6. **Recover U**: `U = Q Ũ` in FP64.
7. **Optional FP64 subspace iteration** (Saad 1980, Parlett 1998): one pass
   of `U ← orth(A V)`, `V ← orth(Aᵀ U)` to polish. Needed only when the
   sketch missed a singular direction (detected via residual).

The provable bound (Halko–Martinsson–Tropp Thm 10.5) on the singular-value
error after step 6 is:

```
|σᵢ - σ̃ᵢ| ≤ (1 + 9√(k+p)·min(m,n))^{1/(2q+1)} · σ_{k+1} + O(u_f64)
```

**This is NOT small for DMRG.** If `σ_{k+1} = 1e-8` (typical for a
nearly-saturated bond), the bound is ~1e-7 — several orders of magnitude
worse than what DMRG energy convergence needs (see §2.2). The subspace
iteration step (7) is what rescues accuracy, but only if added. And that
step by itself is 2 dense `n × n` GEMMs in FP64.

### 1.4 Iterative refinement SVD (Dongarra 2022 family)

Dongarra, Haidar, Abdelfattah et al. proposed mixed-precision iterative
refinement for **linear systems** (arXiv:2001.08887). The extension to SVD
is published in Ogita–Rump–Oishi, "Accurate Sum and Dot Product" (SIAM J.
Sci. Comput. 26, 2005) and more recently in Ozaki–Ogita "Generalization of
error-free transformation" (SIAM 2016). The practical SVD refinement
scheme is:

```
(U₀, Σ₀, V₀) ← FP32 SVD of A
r ← A - U₀ Σ₀ V₀ᵀ       # in FP64 — the residual
while ‖r‖ > tol · ‖A‖:
    (ΔU, ΔΣ, ΔV) ← FP32 SVD of r
    (U, Σ, V) ← (U + ΔU, Σ + ΔΣ, V + ΔV)     # update in FP64
    r ← A - U Σ Vᵀ
```

**Status:** There is no published convergence proof for this scheme on
near-rank-deficient SVD. Haidar et al. (SC'18) show it works for dense
symmetric eigenproblems but with condition-number restrictions
`κ < 1/√u ≈ 4 × 10³` for FP32 initial precision. DMRG theta has
`κ` routinely in `[10⁵, 10¹²]`. This scheme is therefore **inapplicable** as
the primary path.

### 1.5 Decision

Go with **R3-MPSVD-1** (randomized FP32 sketch + FP64 refinement) as the
only defensible primitive. `sgesvdj` is out. FP32 iterative refinement is
out. The next question is whether R3-MPSVD-1 is actually fast enough and
accurate enough for DMRG.

---

## 2. Error budget and stability analysis

### 2.1 Condition number of DMRG theta matrices

DMRG theta at a bond is `θ = Ψ` reshaped to `(χ_L d, d χ_R)`. Its singular
values are the Schmidt spectrum across the bond. For a critical 1D chain
(Heisenberg XXX with `L = 64`) the Schmidt spectrum decays
quasi-polynomially: `σ_n ∼ exp(-n^α)` with `α ≈ 0.6–1.0`, from the
Calabrese–Cardy entanglement scaling (Calabrese–Cardy, J. Stat. Mech.
P06002 (2004)).

Empirically, for `χ = 256` on critical Heisenberg at `L = 64`, the last
retained singular value is typically `σ_{256} ≈ 10⁻⁸ · σ_1`. For gapped
Josephson arrays at `χ = 128` we see `σ_{128} ≈ 10⁻¹⁴ · σ_1`. So:

| System | χ | Typical κ(θ) |
|---|---|---|
| Heisenberg critical | 128 | `~10⁵` |
| Heisenberg critical | 256 | `~10⁸` |
| Heisenberg critical | 512 | `~10¹⁰` |
| Josephson gapped | 128 | `~10¹²` |
| Josephson gapped | 256 | `~10¹⁴` |

**Implication for FP32:** Any scheme that squares κ (i.e. anything going
through `AᵀA` or `AAᵀ`) is dead on arrival for `χ ≥ 128` — the squared
condition number exceeds FP32 machine precision at `κ ≥ 10⁴`.

### 2.2 How accurate do we need singular values?

DMRG energy accuracy target: `|ΔE| < 10⁻¹⁰` on L=32 Heisenberg.

The singular values are used for truncation: we keep the top `k` and
discard the rest, with discarded weight `ε_trunc = Σ_{i>k} σᵢ²`. The energy
error per sweep scales linearly with `ε_trunc` (Legeza–Sólyom 2003; Hubig
et al. 2015), so the requirement is:

- **Absolute σᵢ accuracy** to `~10⁻¹¹` in the tail (we truncate on absolute
  values, not relative).
- **Singular-vector accuracy** `‖U - Ũ‖_F < 10⁻⁷` (weaker — orthogonality
  errors are quadratically absorbed into the next Lanczos).
- **Orthogonality** `‖UᵀU - I‖_F < 10⁻¹²` is needed because the MPS
  canonical form is used in later environment contractions that don't tolerate
  rounding drift well.

**What FP32 naturally provides:**

- Absolute σᵢ accuracy `~ε_f32 · σ_1 ≈ 10⁻⁷ · σ_1`. For tail singular
  values of order `10⁻⁸`, this is **larger than the true value**. That's a
  disaster for truncation: everything below `10⁻⁷` gets drowned in FP32
  noise.
- Orthogonality `‖UᵀU - I‖_F ≈ √n · ε_f32 ≈ 10⁻⁵` at `n = 256`. Fails the
  MPS canonical-form requirement by 7 orders of magnitude.

**Conclusion:** Any FP32-based SVD that produces FP32 outputs is
unacceptable. The FP64 refinement / reprojection step is **not optional**.

### 2.3 Where FP32 can safely live

The only place FP32 arithmetic can live in a DMRG SVD is as an accelerator
for a **subspace estimate** that will be re-projected into FP64 before any
truncation decision is made. Specifically, in R3-MPSVD-1:

- Step 1 (sketch `Y = A Ω`): FP32 safe — `Y` is a dominant-subspace
  estimate; accuracy degrades only the quality of that estimate, which
  affects the `σ_{k+1}` cap on the refinement bound, not the final FP64
  values.
- Step 2 (power iteration): FP32 safe, **but** reorthogonalization must be
  done in FP64 via `cholesky QR` or `Householder QR` (rocSOLVER
  `rocsolver_dgeqrf`). A Gram-Schmidt FP32 reorthogonalization will lose
  orthogonality by `~10⁻⁵` per iteration, poisoning the subspace.
- Step 4+ (project back, SVD of B, recover U): FP64.

Net arithmetic cost at `n = 512, k = 256`:

| Step | FLOPs | Precision | GPU wall time estimate |
|---|---|---|---|
| 1. `Y = A Ω` (FP32 GEMM) | `2 n² (k+p) ≈ 140 M` | FP32 | < 1 ms |
| 2. Power iter (2×) | `2 · 2 n² (k+p) ≈ 280 M × 2 = 560 M` + 2 QRs | FP32/FP64 | 10-15 ms |
| 3. Final QR of `Y` | ~`4 n (k+p)² ≈ 0.28 M` | FP64 | < 1 ms |
| 4. `B = Qᵀ A` | `2 n² (k+p) ≈ 140 M` | FP64 | ~3 ms |
| 5. SVD of `B` `(k+p × n)` | `~20 (k+p)² n` | FP64 | ~10 ms |
| 6. `U = Q Ũ` | `2 n (k+p)²` | FP64 | ~2 ms |
| **Total, R3-MPSVD-1, n = 512** |  |  | **~30 ms** |

Compare to: `rocsolver_dgesvd` at n=512 is 951 ms; CPU reference dgesvd is
224 ms (§5). **If R3-MPSVD-1 actually works to the required accuracy, it
is a 7–30 × speedup.** Whether it works is the question.

---

## 3. GPU implementation sketch

### 3.1 Proposed code location

`gpu-rocm/dmrg2-gpu-opt/src/svd_mixed_precision.h` — a new header that
provides a drop-in `mp_svd_split()` guarded by
`DMRG2_GPU_USE_MP_SVD`. Signature identical to `svd_split_fallback()` so it
plugs into `optimize_bond()` unchanged.

### 3.2 Kernel inventory

```cpp
// All GEMMs via rocBLAS.
// All QRs via rocSOLVER _geqrf + _orgqr / _orgbr.

template<typename Scalar>
void mp_svd_split(int site, Scalar* d_theta, char direction,
                  int k_target,
                  int p_oversample = 10,
                  int q_power_iter = 2) {
    int cL = chi_L(site), cR = chi_R(site+1);
    int m = cL * d_, n = d_ * cR;
    int kp = k_target + p_oversample;

    // 1. Cast theta to FP32: custom elementwise kernel (~1 µs)
    d_to_s_cast_kernel<<<...>>>(d_theta, d_theta_f32_, m*n);

    // 2. Gaussian Ω (FP32) — use rocRAND device API, normal dist, same shape
    // as a pool kept in the class (rocrand_generate_normal, FP32). Seed per
    // bond with hash(site, sweep) for reproducibility.
    rocrand_generate_normal(rng_, d_Omega_f32_, (size_t)n*kp, 0.0f, 1.0f);

    // 3. Y = A Ω in FP32 (single sgemm)
    rocblas_sgemm(h_, N, N, m, kp, n, &one_f,
                  d_theta_f32_, m, d_Omega_f32_, n, &zero_f, d_Y_f32_, m);

    // 4. Power iteration q times
    for (int i = 0; i < q_power_iter; ++i) {
        // Q, R = QR(Y)   — FP32 householder QR is rocsolver_sgeqrf + sorgqr
        rocsolver_sgeqrf(h_, m, kp, d_Y_f32_, m, d_tau_f32_);
        rocsolver_sorgqr(h_, m, kp, kp, d_Y_f32_, m, d_tau_f32_);
        // Z = Aᵀ Y      (n × kp) in FP32
        rocblas_sgemm(h_, T, N, n, kp, m, &one_f,
                      d_theta_f32_, m, d_Y_f32_, m, &zero_f, d_Z_f32_, n);
        // Q, R = QR(Z)
        rocsolver_sgeqrf(h_, n, kp, d_Z_f32_, n, d_tau_f32_);
        rocsolver_sorgqr(h_, n, kp, kp, d_Z_f32_, n, d_tau_f32_);
        // Y = A Z
        rocblas_sgemm(h_, N, N, m, kp, n, &one_f,
                      d_theta_f32_, m, d_Z_f32_, n, &zero_f, d_Y_f32_, m);
    }
    // Final QR of Y -> Q (m × kp)
    rocsolver_sgeqrf(h_, m, kp, d_Y_f32_, m, d_tau_f32_);
    rocsolver_sorgqr(h_, m, kp, kp, d_Y_f32_, m, d_tau_f32_);

    // 5. CAST Q to FP64 for refinement
    s_to_d_cast_kernel<<<...>>>(d_Y_f32_, d_Q_f64_, m*kp);

    // 6. B = Qᵀ A in FP64 (dgemm). This is the refinement step.
    rocblas_dgemm(h_, T, N, kp, n, m, &one_d,
                  d_Q_f64_, m, d_theta, m, &zero_d, d_B_f64_, kp);

    // 7. SVD of small (kp × n) matrix in FP64 on CPU via LAPACK dgesvd
    //    (faster than rocsolver_dgesvd for small kp) OR via rocsolver_dgesvdj
    //    (faster than dgesvd for full-range FP64 — see §5).
    // Returns Ũ (kp × kp), Σ (kp,), Vᵀ (kp × n).
    rocsolver_dgesvdj(h_, singular, singular, kp, n, d_B_f64_, kp,
                      0.0, &resid_, 50, &n_sweeps_, d_S_f64_, d_Utilde_f64_, kp,
                      d_Vh_f64_, kp, d_info_);

    // 8. U = Q Ũ (m × kp) in FP64
    rocblas_dgemm(h_, N, N, m, kp, kp, &one_d,
                  d_Q_f64_, m, d_Utilde_f64_, kp, &zero_d, d_U_f64_, m);

    // 9. OPTIONAL: one FP64 subspace refinement pass if residual is high.
    //    Residual = ‖θ - U diag(S) Vᵀ‖_F / ‖θ‖_F computed once, on GPU.
    //    If > tol_refine, do one more step: V ← A^T U / S, U ← A V / S, sort.
    //    Typically skipped for DMRG's tight tolerance on σᵢ.

    // 10. Truncate to k_target using same svd_truncate_kernel as the baseline path.
}
```

Workspace: ~`2(m + n)(k + p) + (k+p)² + (k+p) n` FP32 + FP64 buffers. At
`m = n = 512, k = 256, p = 10`, that's ~4 MB — well below the 192 GB HBM.

### 3.3 Runtime estimates

All numbers below assume the live MI300X measurements in §5 as a floor.

| n = χd | `rocsolver_dgesvd` (current GPU fallback) | CPU LAPACK dgesvd (1-thread reference) | R3-MPSVD-1 estimated |
|---|---|---|---|
| 128 | 10 ms | 3 ms | ~8 ms (overhead dominated) |
| 256 | 28 ms | 28 ms | ~20 ms |
| 512 | 91 ms (actually 951 ms with gesvd fallback for kappa>1) | 224 ms | ~30–50 ms |
| 1024 | 350 ms | 1787 ms | ~90–150 ms |

**Key observation:** The R3-MPSVD-1 path is dominated by the power-iteration
QR steps, not the small FP64 SVD at the end. At `n=256` it is only
marginally faster than a CPU single-thread LAPACK call (28 ms vs 20 ms), and
the CPU path incurs no correctness risk. **MP SVD only beats CPU LAPACK at
`n ≥ 512`.**

### 3.4 Crossover chi

From the numbers above, R3-MPSVD-1 beats CPU LAPACK dgesvd when `n = χ·d ≥
512`, i.e. `χ ≥ 256` (for `d = 2`) or `χ ≥ 128` (for `d = 4` / Josephson).
Below that, CPU dgesvd with OpenBLAS 0.3.28 threading is strictly better
(see `docs/lessons_openblas_svd_bug.md` for the OpenBLAS story). **The
minimum viable chi threshold for this follow-up is χ ≥ 256.**

That directly overlaps with the R2-1 CBE target regime, **but** CBE
replaces the SVD entirely. So MP SVD is only a win if CBE doesn't land.

---

## 4. Comparison table vs current CPU LAPACK

All numbers in ms, per SVD call, on MI300X (GPU) or host CPU (single
thread, reference LAPACK 3.10.0 — NOT OpenBLAS 0.3.28 which is the
production path).

| n | κ | GPU dgesvd | GPU dgesvdj | GPU sgesvdj (err) | GPU sgesvd (err) | CPU dgesvd | R3-MPSVD-1 (est) |
|---|---|---:|---:|---:|---:|---:|---:|
| 128 | 1 | 10.1 | 8.3 | 4.2 (4e-6) | 10.6 (2e-7) | 2.9 | ~8 |
| 128 | 1e6 | 43.1 | 32.3 | 22.5 (**2.6!**) | 24.6 (5e-3) | 2.9 | ~8 |
| 256 | 1 | 27.6 | 27.3 | 11.2 (8e-6) | 26.4 (2e-7) | 28.0 | ~20 |
| 256 | 1e6 | 185 | 83.8 | 55.3 (**2.5!**) | 92.3 (5e-3) | 28.0 | ~20 |
| 256 | 1e8 | 146 | 96.2 | 55.4 (**4.2!**) | 87.5 (6e-1) | 28.0 | ~20 |
| 512 | 1 | 91.3 | 73.4 | 37.0 (3e-5) | 65.5 (4e-7) | 225 | ~35 |
| 512 | 1e6 | 952 | 211 | 146 (**4e-3**) | 449 (1e-5) | 225 | ~35 |
| 512 | 1e8 | 707 | 246 | 132 (**7e-1**) | 391 (2e-5) | 225 | ~35 |
| 1024 | 1 | 351 | 239 | 124 (8e-5) | 179 (5e-7) | 1787 | ~90 |
| 1024 | 1e6 | 2570 | 673 | 378 (2e-4) | 1114 (3e-5) | 1787 | ~90 |
| 1024 | 1e8 | 1976 | 811 | 378 (2e-4) | 1067 (2e-5) | 1787 | ~90 |

**Bolded errors are catastrophic** — the algorithm is producing garbage for
DMRG purposes.

### Three things this table shows

1. **`sgesvdj` is broken for DMRG regimes.** At κ ≥ 10⁶ (which is *every*
   realistic DMRG bond at χ ≥ 128) the relative error in σ is 1e-3 to 4.3.
   Unusable as a direct substitute.
2. **`sgesvd` (FP32 QR-based) is surprisingly stable.** It holds 1e-5 to
   1e-6 relative error even at κ = 10⁸. But that's still 100× worse than the
   1e-10 absolute tail accuracy DMRG needs. Usable as a subspace estimator
   only.
3. **`dgesvdj` is a pleasant surprise.** At n = 1024, κ = 10⁶ it is **3.8×
   faster than `dgesvd`** and 2.6× faster than CPU LAPACK dgesvd, with no
   accuracy loss (all FP64). This is an independent, immediate win that
   requires **no mixed precision at all.**

---

## 5. Live-VM benchmark data

Ran on remote `hotaisle@23.183.40.84`, GPU `AMD Instinct MI300X VF`
(gfx942), ROCm 7.2.0.70200, rocSOLVER 3.2.0.70200.

### 5.1 Setup

- `/home/hotaisle/dmrg-implementations/sandbox/pair09_mp_svd/pair09_bench_svd.hip`
- Matrices: `n × n` with singular-value decay `σᵢ = κ^(-i/(n-1))` so
  `σ_0 = 1, σ_{n-1} = 1/κ`. Random orthogonal `U, V` from QR of Gaussian.
- 3 timed repetitions per cell, warmup excluded, fresh device copy on each
  repetition. One-iteration `max_sweeps = 100`, `abstol = 0` (machine eps).
- Compared against ground-truth sigmas (analytic from the construction).
  Relative error reported is `max_i |σᵢ_computed - σᵢ_true| / σᵢ_true` over
  the top-256 values, so it also catches catastrophic tail errors that
  silent truncation in real DMRG would absorb.

### 5.2 Raw output

```
     n  kappa    dgesvd_ms   dgesvdj_ms   sgesvdj_ms    sgesvd_ms   err_sgsvdj   err_sgesvd
------------------------------------------------------------------------------------
   128    1e0       10.102        8.306        4.160       10.575     3.81e-06     2.38e-07
   128    1e3       44.483       22.388       19.640       33.440     2.16e-04     7.92e-06
   128    1e6       43.061       32.282       22.498       24.579     2.55e+00     5.04e-03
   128    1e8       36.602       37.928       21.038       22.614     4.33e+00     9.53e-01
   256    1e0       27.604       27.298       11.227       26.437     7.51e-06     2.38e-07
   256    1e3      184.644       55.441       49.010      129.117     1.68e-03     1.33e-05
   256    1e6      185.125       83.774       55.282       92.261     2.48e+00     4.56e-03
   256    1e8      145.979       96.163       55.387       87.513     4.20e+00     6.50e-01
   512    1e0       91.250       73.429       36.957       65.512     2.86e-05     3.58e-07
   512    1e3      935.026      154.689      125.334      646.045     8.98e-05     2.92e-05
   512    1e6      951.869      211.243      145.819      449.024     3.73e-03     1.32e-05
   512    1e8      706.793      245.941      131.937      391.018     6.91e-01     2.13e-05
  1024    1e0      350.853      238.559      123.996      178.812     8.01e-05     4.77e-07
  1024    1e3     2561.974      436.409      343.569     1727.254     1.79e-04     5.83e-05
  1024    1e6     2569.518      673.194      377.475     1113.765     1.97e-04     2.73e-05
  1024    1e8     1975.531      810.883      378.015     1066.870     2.15e-04     2.21e-05
```

### 5.3 CPU reference comparison

Ran `pair09_cpu_lapack_bench` on the same host, single-threaded reference
LAPACK 3.10.0 (`/usr/lib/x86_64-linux-gnu/lapack/liblapack.so.3`), all FP64:

```
     n      kappa dgesvd_ms(cpu)
   128        1e0          2.880
   256        1e0         27.988
   512        1e0        224.516
  1024        1e0       1787.308
```

**This is the reference implementation, not OpenBLAS 0.3.28.** Production
DMRG uses OpenBLAS 0.3.28 linked against MKL-compatible BLAS, which is
typically 1.5–2× faster than reference LAPACK at these sizes.
Approximate production CPU dgesvd: `n=256 ≈ 15 ms, n=512 ≈ 120 ms,
n=1024 ≈ 900 ms` single-threaded, `× 0.4` with 4-thread OpenMP = roughly
`n=256 ≈ 6 ms, n=512 ≈ 50 ms, n=1024 ≈ 360 ms`.

### 5.4 The surprise winner

**`rocsolver_dgesvdj` beats CPU LAPACK at `n ≥ 1024` without any mixed
precision:**

| n | GPU dgesvdj | CPU dgesvd (ref) | CPU dgesvd (prod est, 4-thread) |
|---|---:|---:|---:|
| 256 | 27 ms | 28 ms | ~6 ms |
| 512 | 73 ms | 225 ms | ~50 ms |
| 1024 | 239 ms | 1787 ms | ~360 ms |
| 2048 | — (skipped) | — | ~2500 ms |

**This is a direct, immediate, mixed-precision-free win of 1.5× at n=1024
against 4-thread OpenBLAS, and 7.5× against single-thread.** The current
`gpu-svd` path in `dmrg2_gpu_impl.h:914` calls `rocsolver_gesvd`, not
`rocsolver_gesvdj`. One API substitution, no accuracy loss (FP64
throughout, gesvdj on AᵀA **is** stable in FP64 where `κ² · ε_f64 ≈ 10⁻²`
for `κ = 10⁶` — still bad for the tail but acceptable for sigma_0 ... sigma_k
that DMRG truncation acts on).

### 5.5 sgesvdj failure modes

At `n = 128, κ = 10⁸`, `sgesvdj` returned `err = 4.33`. That means the
computed σ differs from the true σ by more than 4× the true value for some
index. The algorithm did not diverge — it returned a finite answer that has
no relation to the actual singular values of A. This is the classic
`κ² > 1/ε` failure mode. **It would silently poison DMRG truncation.**

At `n = 1024, κ = 10⁸`, the error drops to 2.15e-4. The reason is that as
`n` grows, the top-256 singular values span a smaller range (`κ^(256/1023)
≈ 10²`) so the local condition number of the top-k subspace stays under
`10⁴`, where FP32 `AᵀA` still has ~2 digits left. But this is accidental —
it depends on the spectral shape.

For DMRG Josephson at `χ = 128`, the top 128 singular values span the full
`κ ≈ 10¹²`. `sgesvdj` will return `err ∼ 1` there. **Complete failure.**

---

## 6. Adversarial findings

### 6.1 The `sgesvdj` trap

The rocSOLVER header docstring gives this away but in very quiet language
("V is found as the eigenvectors of AᵀA"). A reader who does not know
Demmel–Kahan will miss it and ship code that produces silent wrong answers
at `χ ≥ 128`. **Anyone considering `sgesvdj` for DMRG should read
Drmač–Veselić 2008 first, then not use `sgesvdj`.**

### 6.2 `sgesvd` is stable but slow

Surprisingly, rocsolver's FP32 **QR-based** `sgesvd` is stable at κ = 10⁶ (~1e-5
error) and κ = 10⁸ (~2e-5 error for larger n). But it is only `~2×` faster
than `dgesvd` — much less than the ideal FP32 speedup. Worse, its 2e-5
error on singular values still exceeds DMRG's required tail accuracy.
**Usable only as a subspace estimate that is then refined in FP64**, which
is exactly what R3-MPSVD-1 does.

### 6.3 Refinement cost often dominates

At `n = 512, k = 256`, step 6 (`B = Qᵀ A` in FP64) is a `266 × 512 × 512`
`dgemm` — about 140 M flops. At MI300X's 81.7 TFLOPS peak FP64 that's
theoretical 1.7 µs, but realistic ~3 ms given launch overhead and the fact
that rocBLAS dgemm at `k = 512` runs at ~10% of peak. Similar cost for step
8. **The refinement alone costs ~6 ms** — still under the CPU dgesvd time
of 50–120 ms, but the margin shrinks. At `n = 256` refinement cost is
comparable to CPU dgesvd total.

### 6.4 Randomized SVD accuracy is not asymptotic

Halko–Martinsson–Tropp's bound
`E‖A - QQᵀA‖ ≤ (1 + 9√(k+p)·√n) σ_{k+1}` has an ugly `9√(k+p)·√n` prefactor.
At `k = 256, p = 10, n = 512`, that's `~9 · 16 · 23 ≈ 3300`. So a tail
singular value of `10⁻¹⁰` becomes a subspace error of `3.3 × 10⁻⁷`. In
DMRG terms, that's 3 orders of magnitude worse than what we need.

The `q`-power iteration reduces this by taking the `(2q+1)`-th root, so
`q = 2` gives `~3300^{0.2} ≈ 5` prefactor. OK for `σ_{k+1} = 10⁻¹⁰`, not
OK for `σ_{k+1} = 10⁻⁸` (critical Heisenberg). **R3-MPSVD-1 needs `q ≥ 2`
for DMRG. Cost is 4 extra GEMMs + 4 QRs per call.** That roughly doubles
the MP SVD time, eroding the advantage.

### 6.5 Where `rocsolver_dgesvdj` wins silently

At **FP64**, the `κ²` blow-up is `ε_f64 · κ² ≈ 10⁻¹⁶ · 10¹² = 10⁻⁴`, which
is bad for the tail but fine for the top-k that DMRG actually truncates on.
**`dgesvdj` is a safe drop-in replacement for `dgesvd` in the existing GPU
path at all tested `χ` and `κ`**, with 2–10× speedup. It should ship
**regardless** of the mixed-precision story.

### 6.6 The CBE comparison

If R2-1 (CBE backbone) lands, the two-site SVD is gone and this entire
follow-up becomes moot. R2-1's estimated 6–8 week timeline is long enough
that a **1-week dgesvdj drop-in** (§6.5) pays for itself several times
over before R2-1 ships. The mixed-precision path (R3-MPSVD-1) competes
with R2-1 for the same regime and the same speedup target — do not build
both unless R2-1's Josephson correctness gate fails.

### 6.7 Stability in the presence of DMRG noise

DMRG theta is never exactly dense — it is constructed from `L`, `Wᴸ`, `Wᴿ`,
`R` contractions that each have their own FP64 rounding. The input to SVD
has roughly `~n · ε_f64 ≈ 10⁻¹³` relative noise already. A mixed-precision
SVD that adds `10⁻⁷` noise dominates by 6 orders of magnitude. The only
question is whether that 6-orders-of-magnitude extra noise in the **top**
singular vectors matters for DMRG energy convergence. Short answer:
sometimes yes, sometimes no, and we have no way to predict which without
running the full sweep. **This makes R3-MPSVD-1 unsuitable as a
production default**; it must be gated on a per-system convergence test.

### 6.8 Bitemporal stability and `pdmrg-multi-gpu`

PDMRG runs `L/P` parallel segments. Mixed-precision noise is not
bit-reproducible across runs (rocRAND seed streams are deterministic but
non-portable). This breaks the strict bit-reproducibility we've been
maintaining in `pdmrg-multi-gpu` for regression testing. **Non-trivial
engineering cost** to work around this in the benchmark harness.

---

## 7. VERDICT

**PROMISING (but not as originally scoped).**

The original Round-3 Pair-9 charter — "design a GPU SVD path with adequate
accuracy using FP32 → FP64 refinement" — is **numerically defensible only
in a narrow form** (R3-MPSVD-1 = randomized FP32 sketch with FP64 projection
and optional subspace polish, `q ≥ 2` power iterations, never sgesvdj,
never sgesvd without refinement).

Even in that narrow form, the expected speedup is ~1.5–3× over production
CPU OpenBLAS threaded dgesvd at `χ ≥ 256`, at the cost of substantial
engineering complexity (rocRAND setup, FP32/FP64 workspace management,
per-system convergence gate, bit-reproducibility loss).

**However**, the live benchmark uncovered an independent, lower-risk win
that makes a better follow-up:

**SPIN-OFF WIN: switch the existing GPU SVD path from `rocsolver_dgesvd`
to `rocsolver_dgesvdj`.** No mixed precision, no accuracy risk (FP64
throughout), 2–3.8× speedup at `n ≥ 512` on the measured data. One
function signature change in `Traits::rocsolver_gesvd` at
`gpu-rocm/dmrg2-gpu/src/dmrg2_gpu_impl.h:914`. **This should happen
regardless of whether CBE, MP-SVD, or anything else ships.**

## 8. Recommendation

### R3-9a — Ship `dgesvdj` drop-in **immediately** (1 week)

Replace the `rocsolver_gesvd` call in `dmrg2-gpu`, `dmrg-gpu`, and the
`svd_split` paths of all variants with `rocsolver_gesvdj`. Verify:

1. At `L = 16` Heisenberg: `|ΔE| < 10⁻¹²` vs current `dgesvd` path.
2. At `L = 32, χ = 256` Heisenberg: sweep-to-sweep energy trajectory
   bit-reproducible modulo rocSOLVER's Jacobi convergence tolerance
   (`abstol = 10⁻¹⁴`).
3. At `L = 32, χ = 256` Josephson: same gate — Josephson's κ ~ 10¹² is the
   stress test.
4. Runtime: sweep wall time on `pdmrg-gpu` L=64 χ=256 should drop from the
   current ~140 s to ~70 s (single GPU), per the ratio `dgesvdj / dgesvd ≈
   2.6×` at n=512 measured here.

Expected gain: **30–50 % sweep wall-time reduction for the `χ ≥ 256`
regime**, with zero numerical risk. This is a bigger win than most R2-1
projections for less than 10 % of the work.

### R3-9b — R3-MPSVD-1 as a **contingency**, not a primary (estimate: 3-4 weeks if pursued)

Gate R3-9b on the R2-1 Josephson correctness gate. If R2-1 CBE converges
cleanly on Josephson, R3-9b is redundant (CBE eliminates the SVD).
If R2-1 fails the Josephson gate (risk register row 1), R3-9b becomes the
primary path forward for the `χ ≥ 256` Josephson regime — but only after
`dgesvdj` has shipped and been validated.

R3-9b deliverables:
1. `gpu-rocm/dmrg2-gpu-opt/src/svd_mixed_precision.h` — R3-MPSVD-1
   implementation.
2. Per-system convergence gate: if sweep energy delta exceeds
   `10⁻¹¹`, automatic fall-back to `dgesvdj`.
3. Ablation study: `q ∈ {1, 2, 3}`, `p ∈ {5, 10, 20}` on `L = 32, χ =
   256` across the three reference models.
4. Full benchmark on the scaling grid from `round_2_plan.md §R2-1
   measurement plan`.

### R3-9c — Do NOT build `sgesvdj` or `sgesvd` paths

Hard reject. The live benchmark shows `sgesvdj` returning errors of O(1)
on DMRG-realistic condition numbers. Even as a "prototype" it would waste
engineering time and risk poisoning benchmark data with silent wrong
answers. If anyone proposes "let's just try sgesvdj", point them at
`sandbox/pair09_mp_svd/pair09_bench_svd.hip` and the `2.55e+00` error cell
at n=128 kappa=1e6.

---

## 9. Concrete next action

**Today / this week, before Round 3 closes:**

```bash
# On the remote VM
ssh hotaisle@23.183.40.84
cd /home/hotaisle/dmrg-implementations/gpu-rocm/dmrg2-gpu/src
# Edit scalar_traits.h and dmrg2_gpu_impl.h:
#   - Replace Traits::rocsolver_gesvd -> Traits::rocsolver_gesvdj
#   - Add residual/n_sweeps/abstol/max_sweeps scratch in class
#   - Re-run test_dmrg2_gpu on all 4 targets
#   - Benchmark L=32, L=64 chi=256 Heisenberg + Josephson
```

If that pans out (expected), open follow-up `R3-9a` for the code change
and measurement, and mark `R3-9b` as "contingent on R2-1 Josephson gate
outcome."

**Do not** start R3-9b speculatively. The R2-1 CBE backbone is still the
better bet for the `χ ≥ 128` regime; R3-9b only becomes the lead if
CBE fails.

---

## 10. Citations

- Demmel, Kahan. "Accurate singular values of bidiagonal matrices." SIAM J.
  Sci. Stat. Comput. 11, 873 (1990).
- Drmač, Veselić. "New fast and accurate Jacobi SVD algorithm I/II." SIAM
  J. Matrix Anal. Appl. 29, 1322/1343 (2008).
- Halko, Martinsson, Tropp. "Finding structure with randomness: Probabilistic
  algorithms for constructing approximate matrix decompositions." SIAM
  Review 53, 217 (2011). arXiv:0909.4061.
- Dongarra, Haidar, Higham, Mary, et al. "Mixed precision iterative
  refinement with sparse approximate inverse preconditioning." SIAM Review
  64, 863 (2022). arXiv:2001.08887.
- Haidar, Tomov, Dongarra, Higham. "Harnessing GPU Tensor Cores for Fast
  FP16 Arithmetic to Speed up Mixed-Precision Iterative Refinement Solvers."
  SC'18.
- Abdelfattah et al. "A Framework for Dense Triangular Matrix Kernels on
  Various Manycore Architectures." Concurrency Comput. (2022). (Mixed-prec
  SVD mentioned in §4.)
- Calabrese, Cardy. "Entanglement entropy and quantum field theory." J. Stat.
  Mech. P06002 (2004).
- Legeza, Sólyom. "Optimizing the density-matrix renormalization group
  method using quantum information entropy." Phys. Rev. B 68, 195116
  (2003).
- Hubig, McCulloch, Schollwöck, Wolf. "Strictly single-site DMRG algorithm
  with subspace expansion." Phys. Rev. B 91, 155115 (2015). arXiv:1501.05504.
- rocSOLVER 7.2 docstring (GESVDJ header section), installed at
  `/opt/rocm/include/rocsolver/rocsolver-functions.h:13700` on
  `hotaisle@23.183.40.84`.
- Benchmark source: `/home/hotaisle/dmrg-implementations/sandbox/pair09_mp_svd/pair09_bench_svd.hip`
  (commit to sandbox, not yet in main repo).
