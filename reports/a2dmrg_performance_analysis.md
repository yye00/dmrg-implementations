# A2DMRG Performance Analysis Report

**Date:** 2026-02-20  
**System:** L=20 sites, D=50 bond dimension, complex128  
**Benchmark:** Josephson junction (Bose-Hubbard with phase twist)

---

## Executive Summary

A2DMRG with np=1 took **3.1 hours** (11,271s) compared to PDMRG's **17 minutes** (1,041s) for the same problem—a **10.8× slowdown**. The root cause is the **O(L²) coarse-space matrix construction** that dominates runtime and is not parallelized.

---

## Benchmark Results

### Reference (quimb)
| Method | Energy | Time |
|--------|--------|------|
| DMRG1 | -15.678132929191 | 318s |
| DMRG2 | -15.678132897760 | 722s |

### PDMRG
| np | Energy | ΔE vs DMRG2 | Time | Speedup |
|----|--------|-------------|------|---------|
| 1 | -15.678132894987 | 2.77e-09 | 1041s | 1.0× |
| 2 | -15.678132895489 | 2.27e-09 | 305s | 3.4× |
| 4 | -15.678132896853 | 9.07e-10 | 426s | 2.4× |
| 8 | -15.678132897545 | 2.15e-10 | 686s | 1.5× |

### A2DMRG
| np | Energy | ΔE vs DMRG2 | Time | Speedup |
|----|--------|-------------|------|---------|
| 1 | -15.678132885851 | 1.19e-08 | 11271s | 1.0× |
| 2 | -15.678132883805 | 1.40e-08 | 7742s | 1.5× |

---

## Root Cause Analysis

### Algorithm Structure (per sweep)

```
A2DMRG Sweep:
├── Phase 1: Skip canonicalization (disabled)
├── Phase 2: Parallel local micro-steps
│   ├── prepare_orthogonal_decompositions() → L copies of MPS
│   └── local_microstep_1site() × L sites
├── Phase 3: Coarse-space minimization  ← BOTTLENECK
│   ├── build_coarse_matrices() → (L+1)² matrix elements
│   └── solve_coarse_eigenvalue_problem()
├── Phase 4: Form linear combination
└── Phase 5: Compression
```

### The Bottleneck: `build_coarse_matrices()`

For L=20 sites, this function computes two (L+1)×(L+1) = 21×21 matrices:

```
H_coarse[i,j] = ⟨Y⁽ⁱ⁾|H|Y⁽ʲ⁾⟩   (Hamiltonian matrix)
S_coarse[i,j] = ⟨Y⁽ⁱ⁾|Y⁽ʲ⁾⟩     (Overlap matrix)
```

**Cost per sweep:**
- 21² = **441 matrix elements** to compute
- Each off-diagonal H element requires a **full MPS-MPO-MPS contraction**
- Equivalent to ~880 energy computations per sweep
- This runs **serially on rank 0** (not parallelized)

### Why Scaling is Poor

| Transition | Expected | Actual | Efficiency |
|------------|----------|--------|------------|
| np=1 → np=2 | 2.0× | 1.46× | 73% |

The "embarrassingly parallel" local micro-steps (Phase 2) parallelize well, but Phase 3 remains serial. As L grows, the O(L²) coarse-space computation increasingly dominates.

### Comparison with PDMRG

| Aspect | PDMRG | A2DMRG |
|--------|-------|--------|
| Per-sweep structure | Standard DMRG sweep | 5-phase algorithm |
| Orthogonal forms | 1 (current center) | L (all sites) |
| Coarse-space | None | O(L²) matrix build |
| Parallelism | Domain decomposition | Local micro-steps |

---

## Timing Breakdown (Estimated)

Based on code analysis for L=20, D=50:

| Phase | Operations | Est. Time/Sweep |
|-------|------------|-----------------|
| prepare_orthogonal_decompositions | 20 MPS canonicalizations | ~5% |
| parallel_local_microsteps | 20 eigensolves + environments | ~15% |
| **build_coarse_matrices** | **441 tensor contractions** | **~70%** |
| form_linear_combination | 21 MPS weighted sum | ~5% |
| compress_mps | SVD compression | ~5% |

---

## Recommendations

1. **Parallelize coarse-space computation**: Distribute the (L+1)² matrix element calculations across ranks

2. **Reduce candidate count**: The paper's filtering (`overlap_threshold=0.99`) helps, but more aggressive filtering could reduce the matrix size

3. **Cache environment tensors**: Many contractions share intermediate results

4. **Consider hybrid approach**: Use A2DMRG only when parallel speedup overcomes serial overhead (large L with many processors)

---

## Conclusion

A2DMRG's theoretical parallelism in local micro-steps is overshadowed by the serial O(L²) coarse-space bottleneck. For moderate system sizes (L~20), standard PDMRG significantly outperforms A2DMRG even with multiple processors. The algorithm may only show benefits for very large systems where the parallel local updates dominate runtime.
