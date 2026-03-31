# GPU Optimization Techniques for -gpu-opt Variants

Research conducted 2026-03-31 targeting AMD MI300X (gfx942) with ROCm 7.2.
Constraint: strict FP64 only — no mixed precision.

## Tier 1 — High Impact, Moderate Effort

### 1. HIP Graphs (1.3-2x speedup for chi≤64)
- Capture repeated Lanczos iteration kernel sequence (GEMM→custom kernels→GEMM) as a graph
- Eliminates per-kernel launch overhead which dominates at small chi
- MI300X kernel launch ~5-10μs each; Lanczos does 50-100 launches per iteration
- Pure infrastructure optimization — no algorithmic change

### 2. QDWH-SVD (2-5x on SVD step for chi≥64)
- Replace rocsolver SVD with QDWH polar decomposition → eigendecomposition
- Pure Level-3 BLAS (GEMM-only), saturates MI300X compute
- rocsolver SVD is inherently sequential (bulge chase); QDWH is parallel
- Strict FP64 compatible — uses Householder QR for stability
- Algorithm: X_{k+1} = X_k(aI + bX_k^H X_k)(I + cX_k^H X_k)^{-1} converges to U_p (polar factor)
- Then SVD from polar: A = U_p * H, eigendecompose H = VΣV^H, so A = (U_p V)ΣV^H

### 3. Dimension Padding to MFMA Multiples (1.1-1.5x)
- MI300X MFMA units operate on 16×16 FP64 tiles
- Pad chi to next multiple of 16 (or 32) for better utilization
- Simple to implement — just allocate padded arrays, zero-fill
- Affects all GEMM calls in apply_heff, environment updates, SVD

## Tier 2 — Moderate Impact, Higher Effort

### 4. Batched GEMM for PDMRG Segments (1.5-2x for pdmrg-gpu)
- Use rocBLAS batched GEMM to process all PDMRG segments in one call
- Currently segments run on separate streams — batched is more efficient for small matrices

### 5. Chebyshev-Filtered Subspace Iteration (1.3-2x for chi≥64)
- Replace Lanczos with polynomial-filtered iteration
- No orthogonalization needed (sync-free), just repeated apply_heff calls
- Needs spectral bounds estimate (cheap 5-step Lanczos)

## Tier 3 — Worth Investigating

### 6. Persistent Kernels / Fused apply_heff
- Fuse the 3-step GEMM sequence into one kernel to avoid launch overhead

### 7. Memory Pool / Custom Allocator
- Reduce hipMalloc overhead with pre-allocated pools

### 8. Strassen-like GEMM
- Theoretical benefit at very large chi, but rocBLAS already well-optimized
