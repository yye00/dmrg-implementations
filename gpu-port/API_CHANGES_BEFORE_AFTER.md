# hipTensor API Changes: Before and After

## ROCm 7.2.0 Compatibility Fix

This document shows the exact changes made to adapt to the ROCm 7.2.0 hipTensor API.

---

## Change 1: hiptensorCreateContraction Signature

### BEFORE (ROCm < 7.2.0)
```cpp
// 17 arguments with integer mode counts
HIPTENSOR_CHECK(hiptensorCreateContraction(
    handle,                                  // pointer not dereferenced
    &contraction_1,
    desc_L, 3, modesL_1, HIPTENSOR_OP_IDENTITY,          // mode count: 3
    desc_theta, 4, modesTheta_1, HIPTENSOR_OP_IDENTITY,  // mode count: 4
    desc_T1, 5, modesT1_1,                               // mode count: 5, no op
    desc_T1, 5, modesT1_1,                               // mode count: 5, no op
    HIPTENSOR_COMPUTE_64F                                // old constant name
));
```

### AFTER (ROCm 7.2.0)
```cpp
// 14 arguments without integer mode counts
HIPTENSOR_CHECK(hiptensorCreateContraction(
    *handle,                                 // dereferenced pointer
    &contraction_1,
    desc_L, modesL_1, HIPTENSOR_OP_IDENTITY,             // no mode count
    desc_theta, modesTheta_1, HIPTENSOR_OP_IDENTITY,     // no mode count
    desc_T1, modesT1_1, HIPTENSOR_OP_IDENTITY,           // no mode count, has op
    desc_T1, modesT1_1,                                  // no mode count, no op
    HIPTENSOR_COMPUTE_DESC_64F                           // new constant name
));
```

### Key Differences
1. **Handle dereferencing**: `handle` → `*handle`
2. **Remove mode counts**: Delete the integer arguments (3, 4, 5)
3. **Add output operation**: Third descriptor now has `HIPTENSOR_OP_IDENTITY`
4. **Update compute type**: `HIPTENSOR_COMPUTE_64F` → `HIPTENSOR_COMPUTE_DESC_64F`

---

## Change 2: Descriptor Type and Destroy Function

### BEFORE
```cpp
// Header file (heff_optimized_gpu.h)
hiptensorContractionDescriptor_t contraction_1;
hiptensorContractionDescriptor_t contraction_2;
hiptensorContractionDescriptor_t contraction_3;
hiptensorContractionDescriptor_t contraction_4;

// Implementation file (heff_optimized_gpu.cpp)
hiptensorDestroyContractionDescriptor(contraction_1);
hiptensorDestroyContractionDescriptor(contraction_2);
hiptensorDestroyContractionDescriptor(contraction_3);
hiptensorDestroyContractionDescriptor(contraction_4);
```

### AFTER
```cpp
// Header file (heff_optimized_gpu.h)
hiptensorOperationDescriptor_t contraction_1;
hiptensorOperationDescriptor_t contraction_2;
hiptensorOperationDescriptor_t contraction_3;
hiptensorOperationDescriptor_t contraction_4;

// Implementation file (heff_optimized_gpu.cpp)
hiptensorDestroyOperationDescriptor(contraction_1);
hiptensorDestroyOperationDescriptor(contraction_2);
hiptensorDestroyOperationDescriptor(contraction_3);
hiptensorDestroyOperationDescriptor(contraction_4);
```

### Key Differences
1. **Type rename**: `ContractionDescriptor` → `OperationDescriptor`
2. **Function rename**: `DestroyContraction` → `DestroyOperation`

---

## Change 3: Handle Dereferencing in All API Calls

### BEFORE
```cpp
// Tensor descriptor creation
hiptensorCreateTensorDescriptor(handle, &desc_L, ...);

// Plan preference creation
hiptensorCreatePlanPreference(handle, &pref, ...);

// Workspace size estimation
hiptensorEstimateWorkspaceSize(handle, contraction_1, ...);

// Plan creation
hiptensorCreatePlan(handle, &plan_1, ...);

// Contract execution
hiptensorContract(handle, plan_1, ...);
```

### AFTER
```cpp
// Tensor descriptor creation
hiptensorCreateTensorDescriptor(*handle, &desc_L, ...);

// Plan preference creation
hiptensorCreatePlanPreference(*handle, &pref, ...);

// Workspace size estimation
hiptensorEstimateWorkspaceSize(*handle, contraction_1, ...);

// Plan creation
hiptensorCreatePlan(*handle, &plan_1, ...);

// Contract execution
hiptensorContract(*handle, plan_1, ...);
```

### Key Differences
1. **All API calls**: Dereference handle pointer: `handle` → `*handle`

### Why?
In `heff_optimized_gpu.cpp`, the handle is stored as `hiptensorHandle_t*` (pointer), but the API expects `hiptensorHandle_t` (value), so we must dereference it.

---

## Change 4: hiptensorDestroy Call

### BEFORE (test_phase1.cpp)
```cpp
hiptensorHandle_t handle;
hiptensorCreate(&handle);
// ... use handle ...
hiptensorDestroy(&handle);  // ❌ Wrong: passing address
```

### AFTER (test_phase1.cpp)
```cpp
hiptensorHandle_t handle;
hiptensorCreate(&handle);
// ... use handle ...
hiptensorDestroy(handle);   // ✓ Correct: passing value
```

### Key Differences
1. **hiptensorDestroy**: Remove address-of operator: `&handle` → `handle`

---

## Complete Example: Full Contraction Setup

### BEFORE
```cpp
// Create contraction (old API)
hiptensorContractionDescriptor_t contraction;
hiptensorCreateContraction(
    handle,
    &contraction,
    descA, 3, modesA, HIPTENSOR_OP_IDENTITY,
    descB, 4, modesB, HIPTENSOR_OP_IDENTITY,
    descC, 5, modesC,
    descC, 5, modesC,
    HIPTENSOR_COMPUTE_64F
);

// ... use contraction ...

// Destroy (old API)
hiptensorDestroyContractionDescriptor(contraction);
```

### AFTER
```cpp
// Create contraction (new API)
hiptensorOperationDescriptor_t contraction;
hiptensorCreateContraction(
    *handle,
    &contraction,
    descA, modesA, HIPTENSOR_OP_IDENTITY,
    descB, modesB, HIPTENSOR_OP_IDENTITY,
    descC, modesC, HIPTENSOR_OP_IDENTITY,
    descC, modesC,
    HIPTENSOR_COMPUTE_DESC_64F
);

// ... use contraction ...

// Destroy (new API)
hiptensorDestroyOperationDescriptor(contraction);
```

---

## Reference Implementation

The correct API usage is based on working code in:
- `gpu-port/src/pdmrg_gpu.cpp` (lines 116-121, 145)
- `gpu-port/src/pdmrg2_gpu.cpp`

**Reference from pdmrg_gpu.cpp:**
```cpp
hiptensorOperationDescriptor_t opDesc;
HT_CHECK(hiptensorCreateContraction(handle, &opDesc,
    descA, modesA, opA,
    descB, modesB, opB,
    descD, modesD, HIPTENSOR_OP_IDENTITY,
    descD, modesD,
    HIPTENSOR_COMPUTE_DESC_C64F));
```

Note: In pdmrg_gpu.cpp, `handle` is a value, not a pointer, so it's used directly.

---

## Files Modified

1. **gpu-port/src/heff_optimized_gpu.cpp**
   - 4 × `hiptensorCreateContraction` calls updated
   - 9 × `hiptensorCreateTensorDescriptor` calls updated
   - 1 × `hiptensorCreatePlanPreference` call updated
   - 4 × `hiptensorEstimateWorkspaceSize` calls updated
   - 4 × `hiptensorCreatePlan` calls updated
   - 4 × `hiptensorContract` calls updated
   - 4 × `hiptensorDestroyContractionDescriptor` → `hiptensorDestroyOperationDescriptor`

2. **gpu-port/src/test_phase1.cpp**
   - 3 × `hiptensorDestroy(&handle)` → `hiptensorDestroy(handle)`

3. **gpu-port/src/heff_optimized_gpu.h** (already correct)
   - 4 × `hiptensorContractionDescriptor_t` → `hiptensorOperationDescriptor_t`

---

## Verification

All changes verified with `/home/captain/clawd/work/dmrg-implementations/gpu-port/verify_api_fixes.sh`

**Result:** ✅ All 10 tests passed

Files are ready for compilation with ROCm 7.2.0.
