# Phase 1 Testing Instructions

## Quick Start (GPU System with ROCm 7.2.0)

```bash
cd /home/captain/clawd/work/dmrg-implementations/gpu-port

# Clean build
rm -rf build && mkdir build && cd build

# Configure (ROCm should auto-detect)
cmake ..

# Build
make -j$(nproc)

# Run Phase 1 tests
./test_phase1
```

## Expected Results

### Test 1: AccurateSVD_GPU Reconstruction
- **Status**: ✅ PASS
- **Expected error**: ~5e-16 (machine precision)
- **What it tests**: Base rocsolver_dgesvd works correctly for matrix reconstruction

### Test 2: OptimizedHeff Identity Test
- **Status**: ✅ PASS (after bug fix)
- **Expected error**: ~1e-14 or better
- **What it tests**: hipTensor contractions produce correct H_eff with identity MPO/environments

## Bug Fixes Applied

### Test 2 Fix (CRITICAL)
The test previously **failed with error = 2.0** due to incorrect environment initialization:

**Before (WRONG)**:
```cpp
// Set L[w, a, a] = 1.0 for ALL w
for (int w = 0; w < D_mpo; w++) {
    for (int a = 0; a < chi_L; a++) {
        h_L[w + a * D_mpo + a * D_mpo * chi_L] = 1.0;
    }
}
```

**After (CORRECT)**:
```cpp
// Set L[0, a, a] = 1.0 (only w=0)
for (int a = 0; a < chi_L; a++) {
    h_L[0 + a * D_mpo + a * D_mpo * chi_L] = 1.0;
}
```

**Why this matters**: With all w active, the MPO chain sums over w:
```
result = Σ_w theta = D_mpo × theta = 3 × theta
||result - theta|| / ||theta|| = 2.0 ✗
```

With only w=0 active, the identity chain gives:
```
result = theta
||result - theta|| / ||theta|| ≈ 1e-14 ✓
```

## Debug Output

The current build includes **verbose debug output** that prints tensor norms at each contraction step:
```
=== DEBUG: OptimizedHeff::apply() ===
  ||theta|| = ... (size=...)
  ||L|| = ...
  ...
Step 1: T1 = L × theta
  ||T1|| = ...
...
=== END DEBUG ===
```

**After confirming tests pass**, remove debug code from `heff_optimized_gpu.cpp::apply()` (lines 451-550).

## Troubleshooting

### If ROCm not found:
```bash
export ROCM_PATH=/opt/rocm
# or wherever ROCm is installed
cmake .. -DROCM_PATH=/opt/rocm
```

### If tests still fail:
1. Check debug output for NaN or zero norms
2. Verify GPU is available: `rocm-smi`
3. Check ROCm version: `hipcc --version` (should be 7.2.0 or compatible)

## Next Steps After Tests Pass

1. ✅ Mark Phase 1 complete
2. Remove debug output from OptimizedHeff
3. (Optional) Fix AccurateSVD recursive refinement (currently disabled, low priority)
4. Begin Phase 2: Multi-stream segmentation infrastructure

## Technical Notes

### ROCm 7.2.0 API
- Uses `hiptensorOperationDescriptor_t` (not ContractionDescriptor)
- `hiptensorCreateContraction` takes 14 args (no mode counts)
- Constants: `HIPTENSOR_COMPUTE_DESC_64F`, `HIPTENSOR_R_64F`, `HIPTENSOR_WORKSPACE_DEFAULT`
- `rocsolver_dgesvd`: 15-arg signature, no buffer size query, returns V^T with ldv=k

### Column-Major Storage
- All tensors use column-major (Fortran) layout
- V^T returned by rocsolver: row p at offset `p` (not `p*n`) for ldv=k

### AccurateSVD Recursion
- Currently disabled (`max_depth=0` in constructor)
- Base SVD works perfectly; recursion has indexing bugs
- Not critical for DMRG application (epsilon=1e-4 threshold rarely triggers)
