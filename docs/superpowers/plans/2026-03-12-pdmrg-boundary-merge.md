# PDMRG Boundary Merge Fix — Stoudenmire Algorithm

> **For agentic workers:** REQUIRED: Use superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the O(L) full-chain coupling sweep with O(P) boundary-only coupling per Stoudenmire & White (arXiv:1301.3494), achieving parallel speedup.

**Architecture:** Implement the paper's 4-step cycle (distribute→sweep→merge→optimize). V=Λ⁻¹ computed via recursive accurate SVD (paper Appendix). Boundary coupling rebuilds environments, then optimizes only at P-1 boundary bonds with V-weighted initial guess.

**Tech Stack:** HIP/ROCm, rocBLAS, LAPACK (CPU SVD+BLAS), existing `column_scale_real` GPU kernel for V-weighting.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/accurate_svd.h` | Create | Recursive accurate SVD (Stoudenmire Appendix) |
| `src/scalar_traits.h` | No change | Already has `column_scale_real` kernel |
| `src/pdmrg_gpu.h` | Modify | Add 2 method declarations |
| `src/pdmrg_gpu_impl.h` | Modify | Add 3 methods, modify `compute_boundary_V` and `run()` |
| `src/test_pdmrg_gpu.cpp` | No change | Existing tests cover correctness |

---

### Task 1: Create `accurate_svd.h`

**Files:** Create `src/accurate_svd.h`

- [ ] Implement recursive accurate SVD per Stoudenmire Appendix
- [ ] BLAS dgemm_/zgemm_ externs for CPU-side A†MB† computation
- [ ] Template function works for both double and hipDoubleComplex
- [ ] Epsilon = 1e-4 threshold for recursion split

### Task 2: Add method declarations to `pdmrg_gpu.h`

**Files:** Modify `src/pdmrg_gpu.h:108-137`

- [ ] Add `double optimize_boundary_bond(int boundary_idx, char direction, int si)`
- [ ] Add `double boundary_coupling_sweep()`

### Task 3: Implement boundary methods in `pdmrg_gpu_impl.h`

**Files:** Modify `src/pdmrg_gpu_impl.h`

- [ ] `#include "accurate_svd.h"`
- [ ] Rewrite `compute_boundary_V()` to use `accurate_svd` instead of standard LAPACK
- [ ] Implement `optimize_boundary_bond()`: V-weighted theta (column_scale_real + GEMM) → Lanczos → accurate SVD split → V update
- [ ] Implement `boundary_coupling_sweep()`: env rebuild → L→R boundary pass → R env rebuild → R→L boundary pass

### Task 4: Modify `run()` algorithm loop

**Files:** Modify `src/pdmrg_gpu_impl.h:1123-1262`

- [ ] After warmup: compute V at each boundary via `compute_boundary_V()`
- [ ] Replace Phase 2 (full-chain sweep) with `boundary_coupling_sweep()`
- [ ] Cap polish phase at 2 full-chain sweeps

### Task 5: Build and test on remote MI300X

- [ ] Push to git, pull on remote
- [ ] Build: `cmake .. -DGPU_TARGETS=gfx942 && make -j$(nproc)`
- [ ] Test Heisenberg L=8 chi=32 segments=2: error < 1e-10
- [ ] Test Heisenberg L=32 chi=64 segments=4: error < 1e-10
- [ ] Test Josephson L=6 chi=32 segments=2: error < 1e-10
