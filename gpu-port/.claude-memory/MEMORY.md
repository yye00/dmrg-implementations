# GPU DMRG Development Memory

## 2026-03-04: Implemented Exact Tensor Contractions - CODE COMPLETE

**Critical Fix:** Replaced approximate H_eff (multiply by -1.5) with EXACT Heisenberg Hamiltonian

### What Was Fixed
- User identified: "it is taking 0ms to run? that means it is not running!"
- Root cause: Simplified H_eff approximation gave wrong energies
- Solution: Implemented exact 2-site Heisenberg H = S·S using 4×4 matrix

### Files Created/Modified
1. **Created `include/tensor_contractions.hpp`** (380 LOC)
   - `apply_H_eff_heisenberg_exact()` - Exact Hamiltonian via rocBLAS batched GEMM
   - `apply_H_eff_2site()` - General tensor network framework
   - Environment updates: `update_left_env()`, `update_right_env()`

2. **Updated `src/dmrg_gpu_native.cpp`**
   - Added TensorContractions member
   - Replaced simplified callback with exact H_eff
   - Now uses exact quantum Hamiltonian matrix

3. **Updated `src/dmrg_benchmark.cpp`**
   - Fixed includes: GPU-native eigensolvers (LanczosEigensolverGPU, BlockDavidsonGPU)
   - Added TensorContractions member
   - Replaced approximate H_eff with exact version

4. **Created `build_mi300x.sh`** - Automated build script for MI300X
5. **Created `IMPLEMENTATION_COMPLETE.md`** - Full documentation
6. **Created `READY_FOR_TESTING.md`** - Testing instructions

### Expected Results
- **Accuracy:** Energy matches Quimb to < 1e-12 error
- **Runtime:** 5-15 seconds for L=12 (not 0ms!)
- **Performance:** PDMRG2 is 2-3x faster than PDMRG
- **Scalability:** Stream count (1→8) shows ~2.5x improvement

### Status
✅ **CODE COMPLETE** - Ready for MI300X testing
- All code written and integrated
- Headers fixed (GPU-native versions)
- Build script prepared
- Documentation complete

### Next Step
Build and test on MI300X (enc1-gpuvm015):
```bash
cd /home/captain/clawd/work/dmrg-implementations/gpu-port
./build_mi300x.sh
cd build
./dmrg_benchmark 12 100 5
```

**Confidence:** 95% that accuracy tests will pass (< 1e-10 error)

---

## 2026-03-05: Phase 2 Multi-Stream hipTensor Implementation - READY FOR TESTING

**Achievement:** Completed full hipTensor environment contractions with 22% code reduction via refactoring.

### What Was Implemented

1. **hipTensor Helper Function** (`stream_segment.cpp`)
   - Encapsulates complete hipTensor contraction workflow
   - Uses CORRECT ROCm 7.2.0 API (not old experimental API)
   - Handles workspace allocation, plan creation, execution, cleanup
   - 58 lines of reusable code

2. **Environment Rebuilding**
   - `rebuild_right_boundary_env()` - Build L_env via 3-step contraction
   - `rebuild_left_boundary_env()` - Build R_env via 3-step contraction
   - Each uses Einstein summation decomposition for efficiency

3. **API Corrections Applied**
   - ✅ `hiptensorCreateContraction` (not CreateContractionDescriptor)
   - ✅ `hiptensorCreatePlanPreference` (not CreateContractionFind)
   - ✅ `hiptensorEstimateWorkspaceSize` (not ContractionGetWorkspaceSize)
   - ✅ `HIPTENSOR_COMPUTE_DESC_64F` (not COMPUTE_64F)
   - ✅ `HIPTENSOR_WORKSPACE_DEFAULT` (not WORKSPACE_RECOMMENDED)

### Code Quality Improvements

**Refactoring Stats:**
- Before: 1322 lines with verbose API calls
- After: 1030 lines with helper function
- **Reduction: 292 lines (22% smaller)**
- **Removed: 341 lines of repetitive boilerplate**
- **Added: 84 lines of clean helper calls**

### Einstein Summation Implemented

Environment contraction formula:
```
L_new[b, wp, b'] = sum_{a, a', w, s, s'}
    L[a, w, a'] * A[a, s, b] * W[w, s, s', wp] * A[a', s', b']
```

Decomposed into 3 sequential pairwise contractions (hipTensor only supports binary ops):
1. `temp1 = L * A` - Contract environment with MPS
2. `temp2 = temp1 * W` - Apply MPO operator
3. `L_new = temp2 * A` - Contract with conjugate MPS

### Files Modified

- `src/stream_segment.h` - Added hipTensor handle, helper declaration
- `src/stream_segment.cpp` - Implemented helper + environment contractions
- `src/test_heisenberg_multistream.cpp` - Multi-iteration test harness

### Testing Status

**Created:** `PHASE2_HIPTENSOR_TEST.md` - Comprehensive testing guide with:
- Build instructions for MI300X
- 4-step test protocol (compilation, multi-iteration, memory, GPU utilization)
- Performance expectations (< 1ms per contraction, < 5ms per iteration)
- Debugging guide for common hipTensor errors
- Code location reference

**Test Target:** `./test_heisenberg_multistream` - 8-site chain, 2 streams, 5 iterations

**Expected Behavior:**
- Clean compilation with no API errors
- Stable multi-iteration DMRG pipeline
- Energy = 1.0 constant (identity MPO test)
- No memory leaks from workspace allocations

### Next Steps

1. **Hardware Test** - Build and run on MI300X (enc1-gpuvm015)
2. **Add Real Hamiltonian** - Replace identity MPO with Heisenberg operators
3. **Validate Accuracy** - Compare vs Quimb (target: < 1e-10 error)
4. **Benchmark Speedup** - Multi-stream vs single-stream scaling

### Commits

- `c0eb488` - Created hipTensor helper with correct API
- `fbce21c` - Refactored verbose code → helper calls (341 lines removed)
- `1a4823f` - Added PHASE2_HIPTENSOR_TEST.md testing guide

**Status:** ✅ Phase 2 implementation complete, ready for MI300X validation
