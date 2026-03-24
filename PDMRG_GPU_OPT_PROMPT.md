# PDMRG-OPT-GPU Implementation Prompt

## Objective

Build `pdmrg-gpu-opt/`, a new GPU implementation of stream-parallel DMRG that ports the algorithmic improvements from the Python `pdmrg-opt/` codebase onto the proven GPU infrastructure of `pdmrg-gpu/` and `dmrg2-gpu/`. The two key algorithmic changes are:

1. **Newton-Schulz polar decomposition** replaces QR/SVD for MPS canonicalization
2. **Block-Davidson eigensolver** replaces Lanczos for the ground-state optimization

Both algorithms are BLAS-3 dominant (GEMM-heavy), which is the fundamental advantage over the BLAS-2-heavy Lanczos and the monolithic LAPACK SVD used in the current GPU codes.

## Why This Matters

Our `pdmrg-gpu/` optimization report (`OPTIMIZATION_REPORT.md`) established that **CPU SVD consumes 97-98% of per-sweep runtime** at chi ≥ 128. All other optimizations (device-pointer Lanczos, GPU pointer kernels, batched GEMMs) collectively affect only 2-3% of wall time. The SVD is called once per bond optimization to split the optimized two-site tensor back into two single-site tensors.

The current SVD path:
```
theta (chi*d × d*chi) → LAPACK dgesvd → U, S, Vh → truncate → MPS tensors
```

The pdmrg-opt approach replaces this with Newton-Schulz polar decomposition during canonicalization (not at the SVD split point, but during the sweep gauge shifts). Meanwhile, Block-Davidson replaces Lanczos for the eigensolver, converting the BLAS-2 dominated matvec loop into a BLAS-3 subspace projection.

## Hardware Target

- AMD Instinct MI300X (gfx942, 304 CUs, 192 GB HBM3)
- ROCm 7.2, rocBLAS, rocsolver
- HIP/C++ with templates for `double` and `hipDoubleComplex`

## Repository Layout

```
pdmrg-gpu-opt/
├── CMakeLists.txt
└── src/
    ├── pdmrg_gpu_opt.h           # Class declaration
    ├── pdmrg_gpu_opt_impl.h      # Full template implementation
    ├── scalar_traits.h        # Reuse from pdmrg-gpu + new kernels
    └── test_pdmrg_gpu_opt.cpp    # Test harness (same as pdmrg-gpu)
```

## Existing Code to Reuse

From `pdmrg-gpu/src/` (proven, tested):
- `scalar_traits.h` — rocBLAS dispatch, GPU pointer setup kernels, Lanczos kernels
- `test_pdmrg_gpu.cpp` — Heisenberg + Josephson MPO builders, test harness
- Stream/handle management, MPS tensor allocation, MPO precomputation
- Environment update methods (`update_left_env`, `update_right_env`)
- `apply_heff_two_site` — the 3-step GEMM contraction (Step 1: batched L_env×theta, Step 2: dense T1×WW, Step 3: batched T2×R_env)

From `dmrg2-gpu/src/`:
- `svd_split` — keep as fallback and for the boundary merge SVD
- Full-chain sweep methods

---

## Part 1: Newton-Schulz Polar Decomposition on GPU

### Algorithm

Replace QR/SVD-based canonicalization with iterative polar decomposition. The Python implementation is in `pdmrg-opt/pdmrg/numerics/linalg_utils.py`:

```python
def newton_schulz_polar(A, tol=1e-10):
    """A = U @ P where U is isometric, P is positive semi-definite."""
    m, n = A.shape
    if m >= n:  # tall/square (common case)
        U = A / norm(A, 'fro')           # scale to singular values ≤ 1
        I_n = eye(n)
        for _ in range(100):
            UtU = U.conj().T @ U          # (n, n) GEMM
            U_new = 0.5 * U @ (3*I_n - UtU)  # (m, n) GEMM
            if norm(U_new - U) < tol:
                break
            U = U_new
        P = U.conj().T @ A               # (n, n) GEMM
        return U, P
    else:  # wide (rare, chain boundaries)
        return qr(A)
```

### GPU Implementation

Each Newton-Schulz iteration is 2 GEMMs + 1 scale, all on the same stream:

```
Iteration k:
  1. UtU = U^H × U            [rocblas_dgemm: (n,m) × (m,n) → (n,n)]
  2. temp = 3I - UtU           [GPU kernel: subtract from scaled identity]
  3. U_new = 0.5 * U × temp    [rocblas_dgemm: (m,n) × (n,n) → (m,n)]
  4. diff = ||U_new - U||_F    [rocblas_dnrm2 on (U_new - U)]
  5. U ← U_new
```

**Matrix sizes in DMRG context:**
- Left-canonize site: A is (chi_L × d, chi_R), e.g., (128, 64) to (2048, 1024)
- Right-canonize site: A^H is (d × chi_R, chi_L), similar sizes
- These are the GEMM sizes that rocBLAS handles efficiently on MI300X

**Convergence properties:**
- Newton-Schulz has **cubic convergence** once started (each iteration triples the number of correct digits)
- After Frobenius normalization, singular values are in (0, 1], so convergence is guaranteed
- For well-conditioned MPS tensors: typically **5-8 iterations** to reach 1e-10
- For ill-conditioned tensors (near singular): up to **15-20 iterations**
- Each iteration is 2 GEMMs — so total cost is 10-40 GEMMs per canonicalization
- Compare to SVD (LAPACK dgesvd): O(mn²) flops for m×n matrix, implemented as a single monolithic call that's hard to overlap with other work

**Key advantage over SVD:**
- Newton-Schulz is **purely GEMM-based** — rocBLAS GEMMs saturate GPU compute at chi ≥ 64
- SVD (LAPACK or rocsolver) has complex internal structure with Householder reflections, bidiagonalization, QR iteration — much harder to keep the GPU busy
- Newton-Schulz iterations can potentially overlap with other streams (segment parallelism)
- No CPU fallback needed — entire computation stays on GPU

**Where to use Newton-Schulz (replacing SVD/QR in current code):**
1. **Segment sweep canonicalization**: After optimizing bond (site, site+1), we currently do SVD to split theta into MPS[site] and MPS[site+1]. Instead, reshape theta into the left-canonical shape and apply Newton-Schulz to get the isometric factor (U) and remainder (P). The SVD truncation step is replaced by truncating P's columns based on approximate singular values (the diagonal of P, which approximates S).
2. **Pre-sweep gauge fixing**: `canonize_block()` in pdmrg-opt right-canonizes all segment sites before a LR sweep and vice versa. Each site canonicalization is one Newton-Schulz call.
3. **Environment rebuild canonicalization**: Before rebuilding environments, ensure canonical form via Newton-Schulz.

**Important: Newton-Schulz does NOT provide singular values.** For the bond optimization SVD split where we need truncation based on singular values, we have two options:
- **Option A (recommended)**: Use Newton-Schulz for the isometric factor, then compute singular values of the remainder P via a small (chi × chi) SVD or eigendecomposition of P^H P. This is much cheaper than the full (chi*d × d*chi) SVD.
- **Option B**: Keep SVD for the bond split (it gives U, S, Vh directly), but use Newton-Schulz for all other canonicalization calls (gauge shifts during sweeps, pre-sweep canonicalization).

### GPU Workspace

Add to `StreamWorkspace`:
```cpp
// Newton-Schulz workspace
Scalar* d_ns_U;       // (m, n) current iterate
Scalar* d_ns_UtU;     // (n, n) product
Scalar* d_ns_temp;    // (n, n) 3I - UtU
Scalar* d_ns_diff;    // (m, n) U_new - U for convergence
RealType* d_ns_nrm;   // 1 element: ||diff||_F
```

### GPU Kernel Needed

```cpp
// Compute temp = alpha * I - A  (for 3I - UtU step)
template<typename Scalar>
__global__ void scaled_identity_minus(Scalar* A, int n, Scalar alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n * n) {
        int row = i % n, col = i / n;
        Scalar diag = (row == col) ? alpha : Scalar(0);
        A[i] = diag - A[i];
    }
}
```

---

## Part 2: Block-Davidson Eigensolver on GPU

### Algorithm

Replace Lanczos with Block-Davidson (LOBPCG-style subspace expansion). The Python implementation is in `pdmrg-opt/pdmrg/numerics/linalg_utils.py`:

```python
def block_davidson(matvec, dim, x_init, b=4, max_iter=30, tol=1e-10):
    V = empty((dim, b))           # subspace basis (dim × k)
    V[:, 0] = x_init / norm       # first basis vector
    V[:, 1:b] = random + orthog   # fill remaining columns
    AV = [matvec(V[:, j]) for j in range(b)]  # H × V columns

    for iteration in range(max_iter):
        k = V.shape[1]
        H_proj = V.conj().T @ AV              # (k, k) GEMM — Rayleigh-Ritz
        eigvals, eigvecs = eigh(H_proj)        # small dense eigenproblem
        X = V @ eigvecs                        # (dim, k) GEMM — Ritz vectors
        AX = AV @ eigvecs                      # (dim, k) GEMM — H × Ritz vectors

        # Convergence check
        r = AX[:, 0] - eigvals[0] * X[:, 0]
        if norm(r) < tol and |dE| < tol: return

        # Expand subspace with residual corrections
        new_vecs = [r_i / norm(r_i) for i in range(b)]
        W = new_vecs - V @ (V.T @ new_vecs)   # orthogonalize (GEMM)
        Q, R = qr(W)                          # small QR for numerical stability

        # Restart if subspace too large
        if k + new > max_subspace:
            V = X[:, :b]; AV = recompute; continue

        # Expand
        AQ = [matvec(Q[:, j]) ...]
        V = [V, Q]; AV = [AV, AQ]

    return best_energy, best_vec
```

### GPU Implementation

The Block-Davidson has a fundamentally different GPU profile than Lanczos:

**Lanczos (current):**
- Each iteration: 1 matvec (3-step batched GEMM) + 2 reductions (dot, nrm2) + 2 axpy
- BLAS-2 dominated: reductions require device→host sync or device-pointer tricks
- Sequential dependency: v_{k+1} depends on v_k
- Typically 10-50 iterations

**Block-Davidson (proposed):**
- Each iteration: b matvecs (b=4 calls to apply_heff) + 2 large GEMMs (V^H @ AV, V @ eigvecs)
- BLAS-3 dominated: projections are large matrix-matrix multiplies
- b matvecs can potentially be batched (apply H to multiple vectors simultaneously)
- Typically 5-15 iterations (but each iteration does b matvecs)

**Total matvec count comparison:**
- Lanczos: 10-50 matvecs per bond
- Block-Davidson: 5-15 iterations × 4-8 matvecs = 20-120 matvecs per bond
- Block-Davidson does MORE matvecs but the BLAS-3 projections and potential batching make each "unit of work" more GPU-efficient

**Key GPU operations per Block-Davidson iteration:**

```
1. Apply H to b new vectors:
   for j in 0..b-1:
     AQ[:, j] = apply_heff_two_site(Q[:, j])    # existing 3-step GEMM

2. Rayleigh-Ritz projection:
   H_proj = V^H × AV                             # rocblas_dgemm: (k, dim) × (dim, k) → (k, k)

3. Dense eigenproblem:
   eigh(H_proj)                                   # CPU: k×k matrix, k ≤ 32, negligible cost

4. Ritz vector reconstruction:
   X = V × eigvecs                                # rocblas_dgemm: (dim, k) × (k, k) → (dim, k)
   AX = AV × eigvecs                              # rocblas_dgemm: (dim, k) × (k, k) → (dim, k)

5. Residual + orthogonalization:
   W = new_vecs - V × (V^H × new_vecs)            # 2 GEMMs
   QR(W)                                           # small QR on CPU or GPU
```

**Batched matvec opportunity:**
The b matvecs in step 1 are independent and can be done as a single batched contraction. Instead of calling `apply_heff_two_site` b times sequentially, we can reshape V into a block and do the 3-step GEMM on the entire block at once:
- Step 1: batch_count = D × d² × b (instead of D × d²)
- Step 2: one larger GEMM with b RHS vectors
- Step 3: same scaling

This is a major advantage — rocBLAS batched GEMM with more batches is more efficient.

### GPU Workspace for Block-Davidson

```cpp
// Block-Davidson workspace (replaces Lanczos workspace)
int davidson_block_size;      // b = 4
int davidson_max_subspace;    // b * 8 = 32
Scalar* d_V_basis;            // (dim, max_subspace) — subspace basis
Scalar* d_AV_basis;           // (dim, max_subspace) — H × basis
Scalar* d_H_proj;             // (max_subspace, max_subspace) — projected H
Scalar* d_ritz_vecs;          // (dim, max_subspace) — X = V @ eigvecs
Scalar* d_ritz_Hvecs;         // (dim, max_subspace) — AX = AV @ eigvecs
Scalar* d_residuals;          // (dim, b) — residual vectors
Scalar* d_new_vecs;           // (dim, b) — orthogonalized new directions

// Host buffers for small eigenproblem
std::vector<Scalar> h_H_proj;        // (max_sub, max_sub)
std::vector<RealType> h_eigvals;     // (max_sub,)
std::vector<Scalar> h_eigvecs;       // (max_sub, max_sub)
```

### Convergence and Fallback

Block-Davidson may fail to converge for:
- Very small systems (dim < max_subspace) — use direct dense eigensolver
- Highly degenerate spectra — rare in DMRG ground state problems
- Numerical breakdown during orthogonalization

**Fallback**: Keep the existing Lanczos eigensolver as fallback. If Block-Davidson fails after max_iter, switch to Lanczos for that bond.

---

## Part 3: Parallel Segment Architecture

### Segment Sweep with Newton-Schulz Canonicalization

The pdmrg-opt Python code does the following before each segment sweep:

```python
# Before LR sweep: right-canonize all sites (polar decomposition)
for j in range(n_local - 1, 0, -1):
    M = A[j].reshape(chi_L, d * chi_R)
    U, P = newton_schulz_polar(M.conj().T)
    A[j] = U.conj().T.reshape(-1, d, chi_R)    # right-isometric
    A[j-1] = A[j-1] @ P.conj().T               # absorb remainder

# Rebuild R_env from right-canonical sites
for j in range(n_local-1, 0, -1):
    R_env[j] = update_right_env(R_env[j+1], A[j], W[j])
```

On GPU, each Newton-Schulz call is 5-8 iterations × 2 GEMMs = 10-16 GEMMs. For a segment of 8 sites, that's 7 × 12 ≈ 84 GEMMs for the pre-sweep canonicalization. This is roughly equivalent to 1-2 Lanczos iterations worth of work — negligible.

### Bond Optimization Flow (GPU)

```
For each bond (site, site+1) in segment:
  1. form_theta: MPS[site] × MPS[site+1]          [1 GEMM]
  2. Block-Davidson eigensolver:
     - b=4 initial matvecs (apply_heff)            [4 × 3-step GEMM]
     - 5-15 Rayleigh-Ritz iterations:
       - b residual matvecs                         [4 × 3-step GEMM]
       - V^H @ AV projection                       [1 GEMM]
       - V @ eigvecs reconstruction                 [1 GEMM]
       - Small eigenproblem on CPU                  [negligible]
  3. Split theta → MPS[site], MPS[site+1]:
     Option A: Newton-Schulz + small SVD of P      [~12 GEMMs + 1 small SVD]
     Option B: Standard SVD (LAPACK/rocsolver)      [1 SVD call]
  4. Update environment                             [3-step GEMM]
```

### Coupling Sweep

Same as current pdmrg-gpu: full-chain env rebuild + LR + RL sweep on stream 0. But now using Block-Davidson instead of Lanczos, and optionally Newton-Schulz for gauge shifts.

---

## Part 4: Implementation Plan

### Phase 1: Newton-Schulz Polar Decomposition
1. Add `newton_schulz_polar()` method to the GPU class
2. Add `scaled_identity_minus` kernel to `scalar_traits.h`
3. Add workspace buffers to `StreamWorkspace`
4. Test standalone: compare U, P outputs against CPU QR for random matrices
5. Integrate into segment sweep canonicalization (replace the gauge-shift in svd_split)
6. Verify correctness on all 3 test cases

### Phase 2: Block-Davidson Eigensolver
1. Add `block_davidson_eigensolver()` method replacing `lanczos_eigensolver()`
2. Add workspace buffers for subspace management
3. Implement batched matvec (apply_heff to b vectors at once) — or just loop
4. Small dense eigenproblem solved on CPU (LAPACK dsyev for k ≤ 32)
5. Keep Lanczos as fallback
6. Verify correctness and compare convergence behavior

### Phase 3: Integration
1. Wire Newton-Schulz into segment sweep pre-canonicalization
2. Wire Block-Davidson into optimize_bond
3. Implement the split strategy (Option A or B)
4. Full correctness validation
5. Performance benchmarking vs pdmrg-gpu and dmrg2-gpu

### Phase 4: Performance Tuning
1. Tune Block-Davidson block size b (4 vs 8)
2. Tune Newton-Schulz tolerance (1e-10 vs 1e-8 for non-final sweeps)
3. Profile to find new bottleneck (should be the split SVD or the matvecs)
4. Consider batched matvec for Block-Davidson if per-vector matvec is too small for GPU saturation

---

## Part 5: Expected Performance Impact

### What Changes

| Component | Current (pdmrg-gpu) | New (pdmrg-gpu-opt) | Expected Impact |
|-----------|---------------------|-------------------|-----------------|
| Eigensolver | Lanczos (BLAS-2) | Block-Davidson (BLAS-3) | Neutral-to-positive: more total FLOPs but better GPU utilization |
| Bond split | Full SVD (chi*d × d*chi) | Newton-Schulz + small SVD | **Major**: replaces the 97-98% bottleneck with GEMM iterations |
| Canonicalization | Implicit (in SVD) | Newton-Schulz pre-sweep | Small overhead (~84 GEMMs per segment, negligible) |
| Sweep structure | Same | Same | No change |

### Newton-Schulz vs SVD Cost Analysis

At chi=256, d=2 (Heisenberg), the bond split operates on a (512 × 512) matrix:

**LAPACK dgesvd (current):**
- ~0.6ms per call on CPU (measured)
- ~126 calls per sweep → ~76ms per sweep for SVD alone? No — the profile shows 38s per sweep. The 0.6ms is for the LAPACK call itself; the transfers and overhead add up.
- Actually: 38s / 126 bonds = 300ms per bond SVD (including D2H, LAPACK, scaling, H2D)

**Newton-Schulz on GPU (proposed):**
- Per iteration: 2 GEMMs on (512 × 512) matrices
- rocBLAS dgemm at (512, 512, 512): ~0.01ms on MI300X (compute-bound, ~2.7 TFLOPS peak f64)
- 8 iterations × 2 GEMMs × 0.01ms = 0.16ms per canonicalization
- Plus small SVD of P (256 × 256): ~0.05ms on CPU
- Total: ~0.2ms per bond vs ~300ms for full LAPACK SVD
- **Potential speedup: 1000× on the SVD step**

Even accounting for the small P-matrix SVD and data movement, this could be a **10-100× reduction** in the SVD bottleneck, which currently consumes 97-98% of runtime.

### Projected Speedups

If Newton-Schulz eliminates 90% of the SVD cost (conservative estimate):

| System | dmrg2-gpu | pdmrg-gpu (current) | pdmrg-gpu-opt (projected) |
|--------|-----------|----------------------|------------------------|
| L=64 chi=128 | 27.0s | 24.1s | ~5-8s |
| L=64 chi=256 | 141.7s | 119s | ~15-25s |

These are rough projections. The actual speedup depends on:
- Newton-Schulz iteration count (condition number of MPS tensors)
- Block-Davidson total matvec count vs Lanczos
- Whether the small P-matrix SVD becomes the new bottleneck
- Memory bandwidth limits for the larger workspace

### Risk Factors

1. **Newton-Schulz convergence**: Ill-conditioned MPS tensors may require 15-20 iterations, reducing the advantage. Mitigation: monitor iteration count, fall back to SVD if > 20 iterations.

2. **Truncation accuracy**: Newton-Schulz gives U and P, but P is not diagonal like S from SVD. We need to extract singular values from P for truncation. Options:
   - Eigendecompose P^H P (chi × chi, much smaller than the original matrix)
   - Use the diagonal of P as approximate singular values (fast but less accurate)
   - SVD of P (chi × chi, much cheaper than chi*d × d*chi)

3. **Block-Davidson overhead**: More total matvecs than Lanczos. If the matvec is already fast (as it is at chi ≥ 128), the extra projections may not help.

4. **Memory**: Block-Davidson needs b × dim workspace for the subspace. At chi=256, dim = 256×4×256 = 262144, and b=4 gives 4 × 262144 × 8 bytes = 8 MB per basis vector. Max subspace of 32 = 256 MB. Fits comfortably in MI300X's 192 GB.

---

## Correctness Requirements

All optimizations must preserve accuracy to < 1e-10 on these tests:

| Test | Exact Energy |
|------|-------------|
| Heisenberg L=8 chi=32 segments=2 | -3.374932598688 |
| Heisenberg L=32 chi=64 segments=4 | -13.997315618007 |
| Josephson L=6 chi=32 segments=2 | -1.748843818181493 |

## Build and Test

```bash
ssh hotaisle@23.183.40.82
cd ~/dmrg-implementations && git pull
cd pdmrg-gpu-opt/build && cmake .. -DGPU_TARGETS=gfx942 && make -j$(nproc)

# Correctness (all must show PASS with error < 1e-10):
./pdmrg_gpu_opt 8 32 5 --segments 2 --warmup 3
./pdmrg_gpu_opt 32 64 3 --segments 4 --warmup 3
./pdmrg_gpu_opt 6 32 5 --segments 2 --warmup 3 --josephson

# Performance benchmarks (compare to dmrg2-gpu and pdmrg-gpu):
./pdmrg_gpu_opt 64 128 2 --segments 8 --warmup 1 --local-sweeps 1
./pdmrg_gpu_opt 64 256 2 --segments 8 --warmup 1 --local-sweeps 1

# dmrg2-gpu baseline:
cd ~/dmrg-implementations/dmrg2-gpu/build
./dmrg2_gpu 64 128 5
./dmrg2_gpu 64 256 5
```

## Success Criteria

1. All correctness tests pass with error < 1e-10
2. PDMRG-OPT-GPU speedup over dmrg2-gpu at L=64 chi=128: target ≥ 2×
3. PDMRG-OPT-GPU speedup over dmrg2-gpu at L=64 chi=256: target ≥ 3×
4. Newton-Schulz converges in ≤ 15 iterations for all test cases
5. Block-Davidson converges for all test cases (with Lanczos fallback rate < 5%)

## Key Design Decisions to Make During Implementation

1. **Split strategy**: Option A (Newton-Schulz + small SVD of P) vs Option B (keep full SVD for split, Newton-Schulz only for gauge shifts). Start with Option B for safety, then try Option A for maximum speedup.

2. **Block-Davidson block size**: b=4 (Python default) vs b=2 (fewer matvecs, less memory) vs b=8 (faster convergence, more memory). Profile to find sweet spot.

3. **Newton-Schulz for coupling sweep**: The full-chain coupling sweep currently uses SVD at every bond. Replace with Newton-Schulz here too for maximum benefit.

4. **Tolerance relaxation**: Use tighter Newton-Schulz tolerance (1e-12) for polish sweeps, looser (1e-8) for warmup/segments. Same for Block-Davidson.

## Reference Files

- `pdmrg-opt/pdmrg/numerics/linalg_utils.py` — Python Newton-Schulz + Block-Davidson
- `pdmrg-opt/pdmrg/numerics/eigensolver.py` — Python eigensolver wrapper
- `pdmrg-opt/pdmrg/mps/canonical.py` — Python Newton-Schulz canonicalization
- `pdmrg-opt/pdmrg/parallel/merge.py` — Python boundary merge with accurate SVD
- `pdmrg-gpu/src/pdmrg_gpu_impl.h` — Current GPU PDMRG implementation
- `pdmrg-gpu/OPTIMIZATION_REPORT.md` — Performance analysis and bottleneck identification
- `dmrg2-gpu/src/dmrg2_gpu_impl.h` — Reference GPU two-site DMRG
