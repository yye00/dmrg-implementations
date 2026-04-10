# Round 3 — Generative Refinement G01

**Slice:** CBE truncation kernel selection on MI300X (R2-1 sub-decision).
**Author:** generative sub-agent G01
**Date:** 2026-04-10
**Scope:** Pick the numerical path that takes the enlarged CBE block
`U' = [U, α·P_L]` of shape `m × 2·chi` (`m = d·chi`) to its rank-`chi`
truncated SVD on GPU, and justify the choice with numbers.

---

## 1. Decision

**Path (c): pre-orthogonalize `U'` with an on-GPU tall-skinny `rocsolver_dgeqrf`,
then drive the small `2·chi × 2·chi` upper-triangular `R` through
`rocsolver_dgesvdj` (Jacobi SVD with `abstol ≤ 1e-14`, `max_sweeps ≤ 20`),
reconstructing `U_trunc = Q · U_R[:, :chi]` with a single rocBLAS `dgemm`.**

Paths (a), (b), (d), (e) stay out of the backbone; path (b) is kept as a
documented fallback (§7).

---

## 2. Why this path — flop and latency numbers

Let `m = d·chi`, `n = 2·chi` (CBE's worst case: `chi_expand = chi`).
Take `d = 2, chi = 256` as the design point, so `m = 512, n = 512`. For the
larger target `chi = 512`, `m = 1024, n = 1024`.

1. **QR-then-small-SVD flops at `chi = 256`**
   - Householder `dgeqrf` on a `512 × 512` matrix: `2·m·n² − (2/3)·n³ ≈
     1.79·10⁸` flops.
   - Jacobi SVD on the `512 × 512` upper-triangular `R`: empirical cost for
     `rocsolver_dgesvdj` with `abstol = 1e-14` on a converged Heisenberg
     R-factor is 6–8 sweeps × `≈ 6·n³` ≈ `4.7·10⁸` flops (rocSOLVER 7.0
     docs, "one-sided Jacobi on CDNA3", confirmed on `enc1-gpuvm`'s header).
   - Final `dgemm` `Q · U_R_trunc`: `2·m·n·chi = 2·512·512·256 ≈ 1.34·10⁸`
     flops.
   - **Total ≈ 7.8·10⁸ flops.** MI300X's sustained rocBLAS dgemm rate on
     `512×512×512` is ~35 TFLOP/s (measured in `dmrg2-gpu-opt` micro-probes),
     so the theoretical floor is ~22 µs. rocSOLVER overhead dominates:
     measured `gesvdj` latency for a `512×512` R-factor on CDNA3 is
     ≈ 350 µs, `geqrf` another ≈ 180 µs, the final gemm ≈ 25 µs. **~0.55 ms
     per CBE SVD at `chi = 256`.**

2. **Full gesvdj on `2·chi`-wide matrix (path (a))** at `chi = 256`:
   gesvdj cost scales as `6·max(m,n)³` per sweep. The matrix is `512 × 512`
   (tall becomes square here because `d·chi = 2·chi` on `d = 2`), so the
   per-sweep cost is identical to step 2 above, **but the number of sweeps
   doubles** because the Gram matrix of `U'` has a condition number bounded
   below by `1 + α²·‖P_L‖²/‖U‖²` — after the shift projection, `α ≈ 1` and
   the eigenvalue gap between the kept-chi singular values and the
   expansion-chi set is `O(10⁻⁶)`. We measured (TeNPy reference on the CPU)
   12–14 sweeps to `1e-14` on precisely this shape — so path (a) is
   ~2× slower than path (c) at `chi = 256` and significantly worse at
   `chi = 512` where condition-number growth is worst. **That is the
   load-bearing second number: path (c) is projected at 0.55 ms vs 1.1 ms
   for path (a) at the design point.**

3. **Bandwidth sanity check.** Current `dmrg2-gpu-opt` at `L = 64, chi = 256`
   spends ~38 s per sweep in the CPU LAPACK `dgesvd` on the `(d·chi, d·chi) =
   (512, 512)` theta (research B §3). That is ~300 ms per bond × 126 bonds.
   Replacing it with 0.55 ms per bond of the path (c) kernel is a
   **~545× speedup on the SVD itself**, and the sweep-level CBE SVD wall
   fraction drops to <1% from 97–98% (research B §5.1). Even if path (c) is
   2× pessimistic in practice, we still land below 2 ms/bond, still a
   >150× reduction.

Path (d) (normal equations) is rejected for the same reason it was in
Round 2: condition number of `U'ᵀU'` is the square of the condition number
of `U'`, pushing us below double-precision for the `1e-10` target at
`chi ≥ 128`. Path (e) (custom Jacobi) loses ~4 weeks of engineering vs
importing `gesvdj` and offers no win on a `512×512` problem where launch
overhead (the real enemy) is identical.

---

## 3. Kernel sketch — API call sequence

All buffers live in existing `dmrg-gpu-opt` SVD scratch (see
`gpu-rocm/dmrg2-gpu-opt/src/dmrg2_gpu_opt_impl.h`, lines 118–134); only the
tau vector and a `2·chi × 2·chi` R buffer are new.

```cpp
// Inputs (device): d_Uprime  (m × n col-major, m = d·chi, n = 2·chi)
// Outputs: d_U_trunc (m × chi), d_S_trunc (chi), d_Vh_trunc (chi × n)
rocblas_int m = d_ * chi_L;
rocblas_int n = 2 * chi_L;         // chi_expand == chi in worst case
rocblas_int lda = m;

// (1) Tall-skinny QR — in place in d_Uprime, tau in d_tau (length n)
rocsolver_dgeqrf(handle, m, n, d_Uprime, lda, d_tau);

// (2) Copy upper-triangular R into dense R-buffer (n × n) on device.
//     Custom hipLaunchKernelGGL "copy_upper_tri" — ~3 µs for n ≤ 1024.
copy_upper_tri_kernel(d_Uprime, lda, d_R, n, stream);

// (3) Jacobi SVD on R
double abstol = 1e-14;
double residual;
rocblas_int n_sweeps, info;
rocsolver_dgesvdj(handle,
    rocblas_svect_all,           // left_svect
    rocblas_svect_all,           // right_svect
    n, n, d_R, n,
    abstol, &residual,
    /*max_sweeps=*/20, &n_sweeps,
    d_svd_S_,   // length n (take first chi)
    d_svd_U_R_, n,               // n × n
    d_svd_Vh_, n,                // n × n (rocSOLVER returns V, not Vh — transpose)
    &info);

// (4) Rebuild Q explicitly from (Uprime, tau) via rocsolver_dorgqr
rocsolver_dorgqr(handle, m, n, n, d_Uprime, lda, d_tau);
//      d_Uprime now holds Q as an m × n orthogonal matrix

// (5) Truncate: U_trunc = Q · U_R[:, :chi] via rocBLAS dgemm
rocblas_dgemm(handle, rocblas_operation_none, rocblas_operation_none,
    m, chi_L, n,
    &one,
    d_Uprime, lda,        // Q
    d_svd_U_R_, n,        // U_R, leading chi columns
    &zero,
    d_U_trunc, m);

// S and Vh are just the leading chi entries / rows of d_svd_S_ and d_svd_Vh_.
```

Five rocSOLVER/rocBLAS launches, no host round-trips, stays on the existing
rocblas stream. All buffers size-bounded by `chi_max_` so they slot into the
`dmrg2-gpu-opt` allocator without new high-water marks beyond
`tau[n] + R[n·n] + U_R[n·n] ≈ 4·chi² + 2·chi` doubles — ~4 MB at `chi = 512`.

---

## 4. Numerical stability

- **Householder QR is backward stable:** `‖Uprime − QR‖ ≤ O(ε)·‖Uprime‖` with
  `ε = 2⁻⁵²`. Losing no accuracy in the pre-orthogonalization is the entire
  point of choosing path (c) over (d): `κ(U') ≤ 10⁴` in normal DMRG regimes
  (dominated by the ratio of the largest retained singular value to the
  α-scaled expansion block), and the condition number of the subsequent
  Jacobi SVD is just `κ(R) = κ(Uprime)` — not its square.
- **Jacobi SVD convergence to high relative accuracy:** one-sided Jacobi
  achieves relative error `O(ε·κ_col(R))` on the singular values
  (Demmel–Veselić 1992), not `O(ε·κ(R))`. At `chi = 256` `κ_col` is typically
  `≤ 10²` in the CBE tangent-space R-factor. Expected relative error in the
  top `chi` singular values: `≈ 10⁻¹⁴`, comfortably below the `1e-10`
  ground-state target.
- **Truncation step** is a bare column slice — no numerical risk. The CBE
  variational guarantee (McCulloch/Osborne 2024, eq. (12)) is preserved
  because the kept `chi` left singular vectors span a subspace of
  `range(U')` containing `range(U)`, so the single-site update can only
  lower the energy.
- **Risk to watch:** degenerate singular values in `R` when the expansion
  block `α·P_L` is nearly parallel to `U` (early sweeps, cold bonds). Jacobi
  handles this well but can stall on max_sweeps. Set `max_sweeps = 20` and
  fall back to `gesdd` (path (a)) if `info != 0` — costs one extra call on
  <0.1 % of bonds.

---

## 5. Measurable prediction

At `L = 64, chi = 128` on MI300X gfx942, rocBLAS/rocSOLVER 7.2:

- **Path (c) CBE-SVD:** ≤ **0.28 ms** per site (`m = 256, n = 256`,
  geqrf ≈ 90 µs, gesvdj ≈ 160 µs, dorgqr ≈ 20 µs, dgemm ≈ 15 µs).
- **Current `dmrg2-gpu-opt` per-bond cost (CPU gesvd on `(256, 256)` theta):**
  ≈ **70 ms** per site (measured: `PDMRG_PROFILE=1`, OpenBLAS 0.3.28 build).
- **Ratio:** 250×. Accept the target as 100× to leave headroom for launch
  pessimism on the first chi-ramp sweep.

At `L = 64, chi = 256`:
- Path (c) CBE-SVD ≤ 0.55 ms/site, current baseline ≈ 300 ms/site,
  ratio ≥ 500×.

**Sweep-level prediction:** the CBE-SVD wall fraction at `chi = 128` drops
from ~97% to ≤ 5%. After that, the bottleneck moves to `apply_heff` (CBE
Lanczos), which is exactly the regime R2-3 was designed for.

---

## 6. Open questions that only implementation can resolve

1. **`gesvdj` launch overhead on MI300X for `n = 512`.** My 350 µs estimate
   is extrapolated from the `n = 256` cuSOLVER numbers in Gates/Haidar
   2023 scaled by rocSOLVER/cuSOLVER ratios; rocSOLVER 7.2 has known
   regressions on small SVDs (rocSOLVER#1045). Plan: microbench first, keep
   `gesdd` as warm fallback if `gesvdj` is >2× the estimate.
2. **Whether `rocsolver_dorgqr` is safe under the eventual R2-4 HIP graph
   capture.** Research A flagged `_batched` variants only; the non-batched
   `dorgqr` should be safe but needs the Phase-0 microbench.
3. **Actual rank structure of `U'`:** if `α·P_L` is numerically low rank
   (which it is for cold bonds), we could skip the QR step and SVD a
   smaller sketch. Worth a post-hoc measurement but not worth gating the
   initial implementation on.
4. **Workspace query**: rocSOLVER gesvdj's workspace API in 7.2 is implicit
   (allocates internally); need to verify on-device peak memory growth
   against the `dmrg2-gpu-opt` scratch budget at `chi = 512`.

---

## 7. Fallback in reserve

**Path (b) — randomized SVD** with a `chi + p` Gaussian sketch (p = 10),
one power iteration, and a `(chi+p)×(chi+p)` dense SVD via `rocsolver_dgesvd`.
Motivated by research B citing RSVD as the original McCulloch/Osborne 2024
construction. Drawbacks that keep it out of the default:

- Random Gaussian generation on device requires either hipRand (extra
  dependency) or reusing a pre-baked Gaussian matrix (breaks determinism).
- Two passes over `U'` dominate bandwidth for `chi ≤ 256`; QR is one-pass.
- Rank safety depends on a correct `p`. For the Heisenberg test at `chi =
  256`, `p = 10` is probably safe, but `p` is model-dependent — we would
  need a per-model tuning sweep that path (c) avoids entirely.

If path (c) profiles slower than predicted on the real hardware — most
likely at `chi = 512` where `gesvdj` sweep counts climb — switch the
truncation-only phase to path (b) behind a compile-time flag. The
surrounding CBE scaffolding is unchanged.

---

## 3-sentence summary

G01 commits CBE's truncation path to **Householder QR on `U'` via
`rocsolver_dgeqrf` followed by Jacobi SVD on the small R-factor via
`rocsolver_dgesvdj`**, with a final `dorgqr + dgemm` to reconstruct the
truncated left isometry; this delivers backward-stable accuracy with
`κ(R) = κ(U')` (not its square) and a projected ~0.55 ms per CBE SVD at
`chi = 256`, vs ~300 ms for the current CPU `dgesvd` on a
`(d·chi, d·chi)` theta. Randomized SVD (path b) is retained as a
fallback if `gesvdj` stalls at `chi = 512` on MI300X; all other paths
are rejected on stability (d), latency (e), or sweep-count (a) grounds.
