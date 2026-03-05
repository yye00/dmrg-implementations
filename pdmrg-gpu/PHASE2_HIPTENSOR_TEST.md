# Phase 2: hipTensor Environment Contractions - Testing Guide

## Summary

**Completed**: Full hipTensor environment contraction implementation for Phase 2 multi-stream DMRG.

**Code Quality**:
- Refactored from 1322 to 1030 lines (22% reduction)
- Replaced 400+ lines of repetitive boilerplate with clean helper function calls
- Uses correct hipTensor ROCm 7.2.0 API

**Status**: Ready for hardware testing on MI300X.

---

## What Was Implemented

### 1. **hipTensor Helper Function** (`stream_segment.cpp` lines 1265-1322)

Encapsulates the complete hipTensor contraction workflow:
```cpp
void StreamSegment::hiptensor_contract(
    const double* A, int nmodeA, const int64_t* extentA, const int32_t* modesA,
    const double* B, int nmodeB, const int64_t* extentB, const int32_t* modesB,
    double* C, int nmodeC, const int64_t* extentC, const int32_t* modesC,
    double alpha, double beta);
```

**Correct API Used**:
- `hiptensorCreateContraction` (not CreateContractionDescriptor)
- `hiptensorCreatePlanPreference` (not CreateContractionFind)
- `hiptensorEstimateWorkspaceSize` (not ContractionGetWorkspaceSize)
- `hiptensorCreatePlan` (not CreateContractionPlan)
- `hiptensorContract` (not hiptensorContraction)
- `HIPTENSOR_COMPUTE_DESC_64F` (not COMPUTE_64F)
- `HIPTENSOR_WORKSPACE_DEFAULT` (not WORKSPACE_RECOMMENDED)

### 2. **Environment Rebuilding Functions**

#### `rebuild_right_boundary_env()` (lines 585-695)
Builds L_env at right boundary via 3-step contraction:
1. `temp1 = L * A` - Contract left environment with MPS tensor
2. `temp2 = temp1 * W` - Apply MPO operator
3. `L_new = temp2 * A` - Contract with conjugate MPS tensor

#### `rebuild_left_boundary_env()` (lines 697-774)
Builds R_env at left boundary (mirror of above):
1. `temp1 = R * A` - Contract right environment with MPS tensor
2. `temp2 = temp1 * W` - Apply MPO operator
3. `R_new = temp2 * A` - Contract with conjugate MPS tensor

### 3. **Einstein Summation Notation**

Environment contraction formula:
```
L_new[b, wp, b'] = sum_{a, a', w, s, s'}
    L[a, w, a'] * A[a, s, b] * W[w, s, s', wp] * conj(A[a', s', b'])
```

For real tensors (no complex conjugation needed), decomposed into 3 steps for efficiency.

---

## Build Instructions (MI300X)

On `enc1-gpuvm015` (HotAisle):

```bash
cd ~/dmrg-implementations/gpu-port
./build_mi300x.sh
```

Expected outputs:
- `build/dmrg_benchmark` - Phase 1 benchmark
- `build/test_heisenberg_multistream` - **Phase 2 multi-iteration test**

---

## Testing Protocol

### Test 1: Basic Compilation
```bash
./build_mi300x.sh
```
**Expected**: Clean build with no errors.

**Verifies**:
- hipTensor API compatibility
- Correct constant usage
- Header includes

### Test 2: Multi-Iteration DMRG Pipeline
```bash
cd build
./test_heisenberg_multistream
```

**Expected Output**:
```
========================================
Testing Phase 2 Multi-Stream Iterative DMRG
========================================

Parameters:
  L = 8 sites
  d = 2
  chi_max = 16
  n_streams = 2
  D_mpo = 3 (identity)

✓ Initialization complete

=== Iteration 0 ===
  Segment sweeps...
  Boundary merges...
  Energy: 1.0000000000

=== Iteration 1 ===
  ...
  Energy: 1.0000000000

Result: PASS
```

**Verifies**:
- hipTensor contractions execute without errors
- Multi-stream coordination works
- Iterative DMRG pipeline is stable
- Energy remains constant with identity MPO (expected)

### Test 3: Check for Memory Leaks
```bash
rocprof --stats ./test_heisenberg_multistream
```

**Verifies**:
- No HIP memory leaks from hipTensor workspace allocations
- Proper cleanup of tensor descriptors

### Test 4: Verify GPU Utilization
```bash
rocm-smi --showuse &
./test_heisenberg_multistream
```

**Expected**:
- GPU utilization > 80% during contractions
- Multiple kernels executing (environment rebuilds)

---

## Expected Performance

### hipTensor Contractions

For chi=16, d=2, D_mpo=3:
- **Step 1** (L * A): Contract 3-tensor (16×3×16) with 3-tensor (16×2×16) → 4-tensor (3×16×2×16)
- **Step 2** (temp1 * W): Contract 4-tensor with 4-tensor (3×2×2×3) → 4-tensor (16×2×16×3)
- **Step 3** (temp2 * A): Contract 4-tensor with 3-tensor (16×2×16) → 3-tensor (16×3×16)

**Performance Target**:
- Each 3-step contraction < 1ms on MI300X
- Total environment rebuild < 5ms per iteration

---

## Debugging Failed Tests

### Build Fails with "undefined reference to hiptensor..."
**Cause**: hipTensor library not linked.
**Fix**: Check `CMakeLists.txt` has `-lhiptensor` and `find_package(hiptensor)`.

### Runtime Error: "hiptensorCreateContraction failed"
**Cause**: Invalid tensor mode specification or extent mismatch.
**Debug**:
```cpp
// Add before each hiptensor_contract() call:
std::cout << "Contracting: nmodeA=" << nmodeA << " nmodeB=" << nmodeB << std::endl;
```

### Segmentation Fault in hipTensor
**Cause**: Device pointer not allocated or incorrect size.
**Debug**:
```bash
# Run with compute-sanitizer
compute-sanitizer ./test_heisenberg_multistream
```

### Energy Not Converging (When Real Hamiltonian Added)
**Cause**: Environment contractions may have incorrect mode ordering.
**Fix**: Verify mode arrays match Einstein summation in comments.

---

## Next Steps After Successful Test

1. **Add Real Heisenberg Hamiltonian MPO**
   - Replace identity MPO with actual Heisenberg operators
   - Expected ground state energy: E_0 ≈ -0.886 * L for L=8

2. **Validate Against Quimb**
   - Run equivalent DMRG in Quimb with same parameters
   - Verify |E_gpu - E_cpu| < 1e-10

3. **Scale to Larger Systems**
   - Test with L=16, chi=64
   - Benchmark multi-stream vs single-stream speedup

4. **Optimize hipTensor Workspace**
   - Profile workspace allocation overhead
   - Consider reusing workspace across contractions

---

## Code Locations

| Component | File | Lines |
|-----------|------|-------|
| hipTensor helper | `src/stream_segment.cpp` | 1265-1322 |
| Right boundary env | `src/stream_segment.cpp` | 585-695 |
| Left boundary env | `src/stream_segment.cpp` | 697-774 |
| Test program | `src/test_heisenberg_multistream.cpp` | 1-121 |
| StreamSegment header | `src/stream_segment.h` | 1-273 |

---

## Implementation Notes

### Why 3-Step Decomposition?

Original formula requires contracting 5 tensors simultaneously:
```
L[a,w,a'] * A[a,s,b] * W[w,s,s',wp] * A[a',s',b'] → L_new[b,wp,b']
```

**Problem**: hipTensor only supports binary contractions (A * B → C).

**Solution**: Decompose into sequential pairwise contractions:
1. Intermediate `temp1` contracts L with first A
2. Intermediate `temp2` contracts temp1 with W
3. Final result contracts temp2 with second A

**Efficiency**: Each step optimized by hipTensor's JIT compiler for MI300X.

### Mode Numbering Convention

Modes are abstract labels for tensor dimensions in Einstein summation:
- **Contracted indices** appear in both input tensors
- **Free indices** appear in output tensor
- **Column-major storage**: First dimension is fastest-varying

Example:
```
L[a,w,a'] * A[a,s,b] = temp1[w,a',s,b]
modes L:    {0, 1, 2}
modes A:    {0, 3, 4}
modes temp1: {1, 2, 3, 4}  (contract over mode 0)
```

---

## Success Criteria

✅ **Build Success**: No compilation errors
✅ **Runtime Success**: test_heisenberg_multistream completes without crashes
✅ **Energy Stability**: Energy constant across iterations with identity MPO
✅ **Memory Safety**: No memory leaks detected
✅ **GPU Utilization**: > 80% during contractions

**When all criteria met**: Phase 2 hipTensor implementation is validated.

---

## Contact / Questions

See `.claude-memory/MEMORY.md` for implementation history and design decisions.

Commit: fbce21c ("Refactor: Replace verbose hipTensor code with helper function calls")
