# GPU-CPU Parity Fix: Quick Reference

**Full details**: See `GPU_PARITY_FIX_PROMPT.md` (11,000 words)
**This doc**: TL;DR for quick consultation

---

## Critical Issues (Must Fix)

### 🔴 Issue #1: Accuracy Failure
- **Current**: 12.5% energy error
- **Target**: < 1e-10 error
- **Status**: BLOCKING for production

### 🔴 Issue #2: Missing Canonicalization Fix
- **Problem**: Environments built from non-canonical MPS → N ≠ I
- **CPU fix**: Commit 6dbeabf (2026-03-09)
- **GPU status**: NOT IMPLEMENTED
- **Impact**: Numerical instability, wrong energies

### 🟡 Issue #3: Code Quality
- **Problems**: 77 TODOs, 6 backup files, unclear production code
- **Impact**: Maintenance burden

---

## Top 3 Fixes to Implement First

### Fix #1: Environment Canonicalization (HIGHEST PRIORITY)

**What**: Left-canonicalize MPS before building L_envs, right-canonicalize before building R_envs.

**CPU reference**: `pdmrg/pdmrg/dmrg.py` lines 310-368

**GPU implementation**:
```cpp
// Before building L_envs: QR sweep (left-canonicalization)
for (int i = 0; i < n_sites - 1; i++) {
    int chi_L = mps[i].dim0;
    int d = mps[i].dim1;
    int chi_R = mps[i].dim2;

    // Reshape to matrix: (chi_L*d) x chi_R
    double* M = mps[i].reshape(chi_L * d, chi_R);

    // QR: M = Q @ R
    double *Q, *R;
    rocsolver_dgeqrf(handle, chi_L*d, chi_R, M, &tau);
    rocsolver_dorgqr(handle, chi_L*d, chi_R, chi_R, M, tau, &Q);

    // Update MPS
    mps[i] = Q.reshape(chi_L, d, chi_R);  // Left-canonical

    // Absorb R into next site
    mps[i+1] = contract(R, mps[i+1]);  // R @ mps[i+1]
}

// Now build L_envs from left-canonical tensors
L_env = init_left_env(...);
for (int i = 0; i < start_site; i++) {
    L_env = update_left_env(L_env, mps[i], mpo[i]);
}
```

**Why this matters**: L_env from left-canonical tensors gives L_norm = I (identity), which the eigensolver assumes. Using non-canonical tensors breaks this assumption.

---

### Fix #2: Return Canonical Tensor from Boundary Merge

**What**: After SVD at boundary merge, return BOTH mixed-form (S @ Vh) and canonical (Vh) tensors.

**CPU reference**: `pdmrg/pdmrg/parallel/merge.py` lines 88-97

**GPU implementation**:
```cpp
struct BoundaryMergeResult {
    Tensor A_left;           // U (left-canonical)
    Tensor A_right;          // S @ Vh (for MPS storage)
    Vector V_new;            // 1/S
    double energy;
    double trunc_error;
    Tensor A_right_canonical;  // Vh (for R_env update) ← NEW
};

BoundaryMergeResult merge_boundary_tensors(...) {
    // ... optimize theta, get theta_opt ...

    // SVD: theta_opt → U, S, Vh
    rocsolver_dgesvd(..., &U, &S, &Vt);

    // Compute V = 1/S
    std::vector<double> V(k);
    for (int i = 0; i < k; i++) {
        V[i] = (S[i] > 1e-14) ? (1.0 / S[i]) : 0.0;
    }

    // Reshape tensors
    Tensor A_left = U.reshape(chi_L, d_L, k);
    Tensor A_right_mixed = matmul(diag(S), Vt).reshape(k, d_R, chi_R);
    Tensor A_right_canonical = Vt.reshape(k, d_R, chi_R);  // ← NEW

    return {A_left, A_right_mixed, V, energy, trunc_err, A_right_canonical};
}
```

**Usage in stream coordinator**:
```cpp
auto result = merge_boundary_tensors(...);

// Update MPS with mixed form
stream_left.mps[last_site] = result.A_left;
stream_right.mps[first_site] = result.A_right;  // S @ Vh

// Update R_env with CANONICAL form
stream_right.R_env[boundary_site] = update_right_env(
    old_R_env,
    result.A_right_canonical,  // ← Use Vh, not S@Vh
    mpo[boundary_site]
);
```

**Why this matters**: Using S @ Vh for R_env gives R_norm = S² ≠ I, breaking eigensolver assumption.

---

### Fix #3: V = 1/S (Verify It's Correct)

**What**: Ensure V = 1/S is computed and applied at EVERY boundary merge.

**CPU reference**: `pdmrg/pdmrg/numerics/accurate_svd.py`

**GPU implementation**:
```cpp
// After SVD: U, S, Vh
std::vector<double> V(k);
for (int i = 0; i < k; i++) {
    if (S[i] > 1e-14) {
        V[i] = 1.0 / S[i];
    } else {
        V[i] = 0.0;  // Safeguard against division by zero
    }
}

// CRITICAL: Store V and use it in next merge
// Merge equation: Psi' = psi_left . diag(V) . psi_right
Tensor theta = contract_with_V(psi_left, V, psi_right);
```

**Validation**:
```cpp
// Print V to check it's reasonable
std::cout << "V = [";
for (int i = 0; i < std::min(5, (int)V.size()); i++) {
    std::cout << V[i] << ", ";
}
std::cout << "...]\n";

// V should NOT be all 1's (that's identity approximation, WRONG)
// V should NOT be all 0's (that's degenerate)
// V should be O(1) to O(100) typically
```

---

## Quick Validation Tests

### Test 1: Does V look right?
```bash
./test_boundary_merge | grep "V ="
# Should show: V = [1.2, 1.5, 1.8, 2.1, ...]
# NOT: V = [1.0, 1.0, 1.0, 1.0, ...]  ← WRONG (identity)
```

### Test 2: Are environments canonical?
```bash
./test_stream_segment | grep "norm"
# Should show: ||L_env|| ~ 1.0 (order of magnitude)
# NOT: ||L_env|| ~ 1e10 or 1e-10  ← WRONG (norm explosion/collapse)
```

### Test 3: Single-stream accuracy
```bash
./test_heisenberg_multistream 8 32 1 30
# Should show: |Error| < 1e-10  ✓ PASS
# NOT: |Error| = 0.125  ✗ FAIL  ← CURRENT STATE
```

### Test 4: Multi-stream consistency
```bash
for n in 1 2 4; do
    ./test_heisenberg_multistream 8 32 $n 30 | grep "Final Energy"
done
# All energies should match to ~1e-11
```

---

## Code Locations to Check

### GPU Files (Priority Order)

1. **`src/stream_segment.cpp`** (1030 lines)
   - Function: `build_environments()` ← ADD CANONICALIZATION HERE
   - Function: `qr_sweep()`, `lq_sweep()` ← VERIFY THESE WORK

2. **`src/boundary_merge_gpu.cpp`** (900 lines)
   - Function: `merge_boundary_tensors()` ← RETURN 6th VALUE (A_right_canonical)
   - Function: `compute_v_from_boundary_tensor()` ← VERIFY V = 1/S

3. **`src/stream_coordinator.cpp`** (400 lines)
   - Function: `even_boundary_merge()`, `odd_boundary_merge()` ← USE CANONICAL TENSOR FOR R_ENV

4. **`src/heff_optimized_gpu.cpp`** (500 lines)
   - Function: `apply_heff()` ← VERIFY CONTRACTION ORDER MATCHES CPU

### CPU Reference Files

1. **`pdmrg/pdmrg/dmrg.py`**
   - Function: `build_local_environments()` lines 310-368 ← CANONICALIZATION LOGIC
   - Function: `boundary_merge()` lines 243-280 ← CANONICAL TENSOR USAGE

2. **`pdmrg/pdmrg/parallel/merge.py`**
   - Function: `merge_boundary_tensors()` lines 13-99 ← COMPLETE MERGE LOGIC

3. **`pdmrg/pdmrg/numerics/accurate_svd.py`**
   - Function: `compute_v_from_svd()` lines 44-64 ← V = 1/S COMPUTATION

---

## Common Mistakes to Avoid

### ❌ Mistake #1: V = ones(chi)
**Wrong**:
```cpp
std::vector<double> V(chi_bond, 1.0);  // Identity approximation
```

**Right**:
```cpp
std::vector<double> V(chi_bond);
for (int i = 0; i < chi_bond; i++) {
    V[i] = (S[i] > 1e-14) ? (1.0 / S[i]) : 0.0;
}
```

### ❌ Mistake #2: Using S @ Vh for R_env
**Wrong**:
```cpp
R_env_new = update_right_env(R_env, A_right_mixed, mpo);  // A_right = S @ Vh
```

**Right**:
```cpp
R_env_new = update_right_env(R_env, A_right_canonical, mpo);  // Vh only
```

### ❌ Mistake #3: No Canonicalization Before Environments
**Wrong**:
```cpp
// Build L_env directly from warmup MPS (non-canonical)
for (int i = 0; i < start; i++) {
    L_env = update_left_env(L_env, mps[i], mpo[i]);
}
```

**Right**:
```cpp
// QR sweep first to left-canonicalize
for (int i = 0; i < n_sites - 1; i++) {
    mps[i] = left_canonicalize(mps[i]);
}
// Then build L_env
for (int i = 0; i < start; i++) {
    L_env = update_left_env(L_env, mps[i], mpo[i]);
}
```

---

## Success Metrics

### Must Achieve
- ✅ Heisenberg L=8, 1 stream: **|Error| < 1e-10**
- ✅ Multi-stream: **|E(n=1) - E(n=4)| < 1e-11**
- ✅ CPU-GPU parity: **|E_CPU - E_GPU| < 1e-10**

### Stretch Goals
- 🎯 Heisenberg L=64: Converges without crashes
- 🎯 GPU ≥ 2x faster than CPU (after correctness proven)
- 🎯 8 streams ≥ 60% parallel efficiency

---

## Debugging Workflow

### If accuracy is still wrong after fixes:

1. **Test eigensolver in isolation**
   ```bash
   ./test_boundary_merge --verbose
   # Check: Are eigenvalues real? Negative? Reasonable magnitude?
   ```

2. **Test H_eff application**
   ```cpp
   // In test code:
   Tensor theta_test = random_tensor(chi_L, d_L, d_R, chi_R);
   Tensor result = apply_heff(theta_test, L_env, R_env, W_L, W_R);
   double norm_ratio = frobenius_norm(result) / frobenius_norm(theta_test);
   std::cout << "||H_eff(theta)|| / ||theta|| = " << norm_ratio << "\n";
   // Should be O(1), like 0.5 to 2.0
   // NOT O(1e10) or O(1e-10) ← indicates contraction error
   ```

3. **Compare GPU vs CPU tensors**
   ```python
   # Save CPU intermediate tensors
   np.savez('cpu_intermediate.npz', L_env=L_env, theta=theta, ...)
   ```
   ```cpp
   // Load in GPU, compute difference
   Tensor L_env_cpu = load_numpy("cpu_intermediate.npz", "L_env");
   Tensor L_env_gpu = compute_l_env_gpu(...);
   double diff = frobenius_norm(L_env_cpu - L_env_gpu);
   std::cout << "||L_env_CPU - L_env_GPU|| = " << diff << "\n";
   // Should be < 1e-12
   ```

4. **Check tensor shapes at every step**
   ```cpp
   #define DEBUG_SHAPES
   #ifdef DEBUG_SHAPES
   #define PRINT_SHAPE(T) std::cout << #T << " shape: " << T.shape_str() << "\n"
   #else
   #define PRINT_SHAPE(T)
   #endif

   PRINT_SHAPE(L_env);
   PRINT_SHAPE(theta);
   // Run test, verify shapes match expectations
   ```

---

## When to Ask for Help

- ✅ **After implementing Fix #1-3**: If accuracy still > 1e-8, something deeper is wrong
- ✅ **After fixing all TODOs**: If crashes persist, may be rocSOLVER/hipTensor bug
- ✅ **After CPU-GPU tensor comparison**: If differences > 1e-10, algorithm mismatch

---

## Timeline Estimate

- **Day 1**: Implement Fix #1 (canonicalization), test unit tests
- **Day 2**: Implement Fix #2 (canonical tensor return), test boundary merge
- **Day 3**: Implement Fix #3 (verify V), test single-stream accuracy
- **Day 4**: Test multi-stream consistency, debug any failures
- **Day 5**: Cross-validate with CPU, clean up code, write validation report

**Total**: 3-5 days (full-time work)

---

## Key Files Reference

### GPU Implementation
- `src/stream_segment.cpp` (1030 lines) - **MODIFY**
- `src/boundary_merge_gpu.cpp` (900 lines) - **MODIFY**
- `src/stream_coordinator.cpp` (400 lines) - **MODIFY**
- `src/accurate_svd_gpu.cpp` (300 lines) - **VERIFY**
- `src/heff_optimized_gpu.cpp` (500 lines) - **VERIFY**

### CPU Reference (commit 6dbeabf)
- `pdmrg/pdmrg/dmrg.py` (lines 310-368) - **CANONICALIZATION**
- `pdmrg/pdmrg/parallel/merge.py` (lines 88-97) - **CANONICAL RETURN**
- `pdmrg/pdmrg/numerics/accurate_svd.py` (lines 44-64) - **V COMPUTATION**
- `pdmrg/pdmrg/numerics/eigensolver.py` (entire file) - **EIGENSOLVER**
- `pdmrg/pdmrg/numerics/environments.py` (entire file) - **ENV UPDATES**

---

## One-Liner Summary

**Fix environments via canonicalization (QR/LQ before building L_env/R_env), return canonical tensor from merge (Vh separate from S@Vh), verify V=1/S is correct → achieve < 1e-10 accuracy.**
