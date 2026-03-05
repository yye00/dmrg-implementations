# hipTensor API Fix Summary - ROCm 7.2.0 Compatibility

## Files Fixed
1. `/home/captain/clawd/work/dmrg-implementations/gpu-port/src/heff_optimized_gpu.cpp`
2. `/home/captain/clawd/work/dmrg-implementations/gpu-port/src/test_phase1.cpp`

## Changes Made

### 1. hiptensorCreateContraction - Signature Update
**Old (17 arguments with integer counts):**
```cpp
HIPTENSOR_CHECK(hiptensorCreateContraction(
    handle,
    &contraction_1,
    desc_L, 3, modesL_1, HIPTENSOR_OP_IDENTITY,
    desc_theta, 4, modesTheta_1, HIPTENSOR_OP_IDENTITY,
    desc_T1, 5, modesT1_1,
    desc_T1, 5, modesT1_1,
    HIPTENSOR_COMPUTE_64F
));
```

**New (14 arguments, no counts, handle dereferenced):**
```cpp
HIPTENSOR_CHECK(hiptensorCreateContraction(
    *handle,
    &contraction_1,
    desc_L, modesL_1, HIPTENSOR_OP_IDENTITY,
    desc_theta, modesTheta_1, HIPTENSOR_OP_IDENTITY,
    desc_T1, modesT1_1, HIPTENSOR_OP_IDENTITY,
    desc_T1, modesT1_1,
    HIPTENSOR_COMPUTE_DESC_64F
));
```

**Key changes:**
- Removed integer mode counts (3, 4, 5, etc.)
- Added `HIPTENSOR_OP_IDENTITY` for output descriptor
- Changed `HIPTENSOR_COMPUTE_64F` → `HIPTENSOR_COMPUTE_DESC_64F`
- Dereferenced handle: `*handle` instead of `handle`

### 2. hiptensorDestroyContractionDescriptor → hiptensorDestroyOperationDescriptor
**Old:**
```cpp
hiptensorDestroyContractionDescriptor(contraction_1);
hiptensorDestroyContractionDescriptor(contraction_2);
hiptensorDestroyContractionDescriptor(contraction_3);
hiptensorDestroyContractionDescriptor(contraction_4);
```

**New:**
```cpp
hiptensorDestroyOperationDescriptor(contraction_1);
hiptensorDestroyOperationDescriptor(contraction_2);
hiptensorDestroyOperationDescriptor(contraction_3);
hiptensorDestroyOperationDescriptor(contraction_4);
```

### 3. hiptensorCreateTensorDescriptor - Handle Dereference
**Changed all occurrences:**
```cpp
// Old
hiptensorCreateTensorDescriptor(handle, &desc_L, ...)

// New
hiptensorCreateTensorDescriptor(*handle, &desc_L, ...)
```

**Applied to descriptors:**
- desc_L, desc_R, desc_W1, desc_W2
- desc_theta, desc_T1, desc_T2, desc_T3
- desc_result

### 4. Other API Calls - Handle Dereference
**Changed in heff_optimized_gpu.cpp:**
```cpp
// hiptensorCreatePlanPreference
hiptensorCreatePlanPreference(*handle, &pref, ...)

// hiptensorEstimateWorkspaceSize
hiptensorEstimateWorkspaceSize(*handle, contraction_1, ...)

// hiptensorCreatePlan
hiptensorCreatePlan(*handle, &plan_1, ...)

// hiptensorContract
hiptensorContract(*handle, plan_1, ...)
```

### 5. hiptensorDestroy - Remove Address-of Operator
**Changed in test_phase1.cpp:**
```cpp
// Old
hiptensorDestroy(&handle);

// New
hiptensorDestroy(handle);
```

## Reference Implementation
All changes based on working code from:
- `/home/captain/clawd/work/dmrg-implementations/gpu-port/src/pdmrg_gpu.cpp` (lines 116-121)
- `/home/captain/clawd/work/dmrg-implementations/gpu-port/src/pdmrg2_gpu.cpp`

**Reference signature from pdmrg_gpu.cpp:**
```cpp
HT_CHECK(hiptensorCreateContraction(handle, &opDesc,
    descA, modesA, opA,
    descB, modesB, opB,
    descD, modesD, HIPTENSOR_OP_IDENTITY,
    descD, modesD,
    HIPTENSOR_COMPUTE_DESC_C64F));
```

Note: In pdmrg_gpu.cpp, `handle` is already a value (not a pointer), so it's used directly. In heff_optimized_gpu.cpp and test_phase1.cpp, `handle` is a pointer (`hiptensorHandle_t*`), so we dereference it with `*handle`.

## Header File Status
The header file `/home/captain/clawd/work/dmrg-implementations/gpu-port/src/heff_optimized_gpu.h` was already updated correctly:
- Uses `hiptensorOperationDescriptor_t` (lines 59-62) ✓

## Compilation Status
Files are ready for compilation with ROCm 7.2.0. To build:
```bash
cd /home/captain/clawd/work/dmrg-implementations/gpu-port
./build_mi300x.sh
```

Or manually:
```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DGPU_TARGETS=gfx942
make test_phase1
```

## Summary of All API Changes
| API Call | Change Required |
|----------|----------------|
| `hiptensorCreateContraction` | Remove mode counts, add output op, dereference handle, use `COMPUTE_DESC_` |
| `hiptensorDestroyContractionDescriptor` | Rename to `hiptensorDestroyOperationDescriptor` |
| `hiptensorCreateTensorDescriptor` | Dereference handle: `*handle` |
| `hiptensorCreatePlanPreference` | Dereference handle: `*handle` |
| `hiptensorEstimateWorkspaceSize` | Dereference handle: `*handle` |
| `hiptensorCreatePlan` | Dereference handle: `*handle` |
| `hiptensorContract` | Dereference handle: `*handle` |
| `hiptensorDestroy` | Remove address-of: `handle` not `&handle` |

## Descriptor Type Change
Header already correct:
```cpp
// Old (ROCm < 7.2.0)
hiptensorContractionDescriptor_t contraction_1;

// New (ROCm 7.2.0)
hiptensorOperationDescriptor_t contraction_1;  // ✓ Already done
```

## All Fixed Contractions
1. **Contraction 1:** `T1 = L × theta` (line 242-250)
2. **Contraction 2:** `T2 = W1 × T1` (line 260-268)
3. **Contraction 3:** `T3 = W2 × T2` (line 278-286)
4. **Contraction 4:** `result = T3 × R` (line 296-304)

All now use the correct 14-argument signature with dereferenced handle.
