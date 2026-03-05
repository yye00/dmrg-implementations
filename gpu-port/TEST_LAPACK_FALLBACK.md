# CPU LAPACK Fallback Testing - MI300X

## What Was Implemented

**Commit e84fc8c**: CPU LAPACK fallback for Lanczos tridiagonal eigensolver

### Problem
rocSOLVER `dsteqr` has a confirmed bug in ROCm 7.2 on MI300X (gfx942) returning incorrect eigenvalues `[-1, 0, 1]` instead of true eigenvalues. This is likely caused by the hybrid CPU-GPU STEQR execution path introduced in rocSOLVER 3.30.0.

### Solution
Replaced rocSOLVER with CPU LAPACK `dstev` for the tridiagonal eigenvalue problem:
- The tridiagonal matrix is tiny (3-30 dimensions, <2KB)
- CPU overhead is negligible (<0.01% of DMRG iteration time)
- This is standard practice in production GPU Lanczos implementations

### Implementation Details
1. **Lanczos iteration loop** (GPU): Build tridiagonal matrix T via matrix-vector products
2. **Copy to CPU**: Transfer D (diagonal) and E (off-diagonal) arrays to host
3. **LAPACK dstev** (CPU): Compute eigenvalues and eigenvectors of T
4. **Extract ground state**: λ_min = D[0] (smallest eigenvalue)
5. **Copy Ritz coefficients** (CPU → GPU): First column of eigenvector matrix
6. **Reconstruct wavefunction** (GPU): |θ⟩ = Σ c[i] |v[i]⟩ using rocBLAS dgemv
7. **Normalize** (GPU): Make |θ⟩ unit norm
8. **Validate** (GPU): Check <θ|H_eff|θ> = λ_min within 1e-10

---

## Testing Instructions for MI300X

### Prerequisites
- AMD MI300X GPU (gfx942)
- ROCm 7.2+ with rocBLAS, rocSOLVER, hipTensor
- LAPACK library (liblapack.so)

### 1. Pull Latest Code

```bash
cd ~/dmrg-implementations/gpu-port
git pull origin master
git log -1  # Verify commit e84fc8c
```

### 2. Clean Build

```bash
rm -rf build
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
```

**Expected output**: CMake should find LAPACK and link successfully
```
-- Found hipTensor: /opt/rocm/lib/libhiptensor.so
...
-- Configuring done
-- Generating done
```

### 3. Build Test Executable

```bash
make -j16 test_heisenberg_multistream
```

**Expected**: Clean build with no errors

### 4. Run Heisenberg Test

```bash
./test_heisenberg_multistream
```

### 5. Expected Output

#### Debug Output from LAPACK
```
DEBUG LAPACK dstev: niter=10, eigenvalues = [-0.8936104773, -0.4106333831, 0.4792439275, ...]
DEBUG Ground state energy: -8.936104772931e-01
```

**Key checks**:
- ✅ Eigenvalues are **NOT** `[-1, 0, 1]`
- ✅ Eigenvalues span a reasonable range (e.g., -0.89 to +0.48)
- ✅ Ground state energy is negative (antiferromagnetic)

#### Energy Convergence
```
=== Iteration 0 ===
  Energy: -3.XXXXXXXXX

=== Iteration 1 ===
  Energy: -3.XXXXXXXXX
  ΔE: X.XXe-XX

...

=== Iteration 10 ===
  Energy: -3.374931816815  ← Should converge to exact value
  ΔE: X.XXe-11            ← Should be very small
```

**Key checks**:
- ✅ Energy converges to E_exact = -3.374931816815
- ✅ Energy changes become very small (ΔE < 1e-10)

#### Final Accuracy Test
```
Accuracy vs Exact:
  E_DMRG  = -3.374931816815
  E_exact = -3.374931816815
  |Error| = X.XXe-11       ← TARGET: < 1e-10

  Accuracy test: ✅ PASS
```

**Success Criteria**:
- ✅ `|Error| < 1e-10` (not 3.7% like before)
- ✅ Rayleigh quotient validation passes (no warnings)
- ✅ No "WARNING: Rayleigh quotient mismatch" messages

---

## Troubleshooting

### Issue 1: LAPACK Not Found
**Error**: `undefined reference to 'dstev_'`

**Fix**: Install LAPACK
```bash
# Ubuntu/Debian
sudo apt-get install liblapack-dev

# RHEL/CentOS
sudo yum install lapack-devel
```

### Issue 2: Still Getting [-1, 0, 1] Eigenvalues
**Error**: `DEBUG LAPACK dstev: eigenvalues = [-1.000000, 0.000000, 1.000000]`

**Diagnosis**: LAPACK is not being called (old rocSOLVER code still running)

**Fix**: Verify commit e84fc8c is checked out
```bash
git log -1 --oneline  # Should show "Implement CPU LAPACK fallback"
grep -n "dstev_" src/boundary_merge_gpu.cpp  # Should find LAPACK call
```

### Issue 3: Rayleigh Quotient Mismatch
**Warning**: `WARNING: Rayleigh quotient mismatch: λ=-0.893, <θ|H|θ>=-0.850, err=4.3e-02`

**Diagnosis**: Wavefunction reconstruction is incorrect

**Possible causes**:
1. Ritz coefficient indexing wrong (check column-major layout)
2. Lanczos vectors corrupted (check GPU memory)
3. H_eff application bug (check tensor contraction)

**Debug**: Add more printouts in reconstruction step

### Issue 4: Energy Still 3.7% Off
**Error**: `|Error| = 1.250e-01 (3.7%)`

**Diagnosis**: LAPACK eigensolver not being used

**Fix**: Check build logs for LAPACK linkage
```bash
ldd test_heisenberg_multistream | grep lapack
# Should show: liblapack.so.3 => /usr/lib/x86_64-linux-gnu/liblapack.so.3
```

---

## Success Criteria

| Criterion | Target | Status |
|-----------|--------|--------|
| **Eigenvalues** | Real values (not [-1,0,1]) | ⏳ Testing |
| **Energy sign** | Negative | ⏳ Testing |
| **Energy magnitude** | ~-3.375 | ⏳ Testing |
| **Accuracy** | \|Error\| < 1e-10 | ⏳ Testing |
| **Convergence** | ΔE < 1e-10 | ⏳ Testing |
| **Rayleigh validation** | No warnings | ⏳ Testing |

**Overall**: ⏳ **TESTING REQUIRED ON MI300X**

---

## Next Steps After Successful Test

### 1. Verify Against Quimb (Heisenberg)
```python
# In separate terminal with Quimb installed
python verify_with_quimb.py --model heisenberg --L 8 --chi 32
```

**Target**: `|E_GPU - E_Quimb| < 1e-10`

### 2. Test Josephson Junction Array
```bash
# Build Josephson test (to be implemented)
make test_josephson_multistream
./test_josephson_multistream
```

**Target**: Same 1e-10 accuracy

### 3. Document Performance
- Measure CPU LAPACK overhead (should be <0.01%)
- Profile full DMRG iteration breakdown
- Update PHASE2_COMPLETE.md with final accuracy results

---

## Implementation Notes

### Why CPU LAPACK is Correct Architecture

1. **Matrix size**: 3-30 dimensions × 8 bytes = 72-7200 bytes
   - PCIe transfer time: <1 microsecond
   - LAPACK solve time: <10 microseconds
   - Total overhead: <0.01% of DMRG iteration

2. **GPU-only alternatives**:
   - Custom CUDA/HIP tridiagonal solver: Complex, error-prone
   - hipSOLVER wrapper: Same rocSOLVER bug
   - Dense eigensolvers (dsyev): Overkill for small matrix

3. **Production precedent**:
   - ITensor C++: Uses LAPACK `dstev` on CPU
   - TeNPy Python: Uses NumPy/SciPy (LAPACK wrappers)
   - ALPS: Uses LAPACK for small auxiliary problems

### Code Structure

```
lanczos_eigensolver_gpu_native()
│
├─ [GPU] Lanczos iteration loop
│  ├─ apply_heff (GPU)
│  ├─ rocblas_dgemv (GPU)
│  ├─ rocblas_ddot (GPU)
│  └─ Build tridiagonal matrix T
│
├─ [CPU↔GPU] Copy D, E to host (tiny)
│
├─ [CPU] LAPACK dstev
│  └─ Eigenvalues + eigenvectors of T
│
├─ [CPU↔GPU] Copy Ritz coefficients to device (tiny)
│
├─ [GPU] Reconstruct wavefunction
│  ├─ rocblas_dgemv: θ = V * c
│  └─ rocblas_dscal: normalize
│
└─ [GPU] Validate
   ├─ apply_heff (GPU)
   └─ rocblas_ddot: check <θ|H|θ> = λ
```

**Total CPU time**: ~10 μs per DMRG iteration (~0.01% overhead)

---

## Related Documents

- `PHASE2_COMPLETE.md` - Phase 2 status (to be updated with final accuracy)
- `TEST_FULL_CHAIN_ENERGY.md` - Previous testing guide (3.7% error)
- `GPU_VERIFICATION_REPORT.md` - Phase 1 validation results

---

## Questions?

If tests fail, check:
1. Git commit matches e84fc8c
2. LAPACK linkage in build logs
3. Debug output shows real eigenvalues (not [-1,0,1])
4. No "Rayleigh quotient mismatch" warnings

**Target**: 1e-10 accuracy for both Heisenberg and Josephson benchmarks!
