#!/bin/bash
# Verification script for hipTensor API fixes
# Run this to verify all API calls are updated correctly

set -e

echo "=========================================="
echo "hipTensor API Fix Verification"
echo "=========================================="
echo ""

PASS=0
FAIL=0

# Test 1: Check old COMPUTE_64F is replaced
echo "Test 1: Checking HIPTENSOR_COMPUTE_64F is replaced..."
if grep -q "HIPTENSOR_COMPUTE_64F" src/heff_optimized_gpu.cpp src/test_phase1.cpp 2>/dev/null; then
    echo "  ❌ FAIL: Old HIPTENSOR_COMPUTE_64F found"
    FAIL=$((FAIL+1))
else
    echo "  ✓ PASS: No old HIPTENSOR_COMPUTE_64F found"
    PASS=$((PASS+1))
fi

# Test 2: Check new COMPUTE_DESC_64F is present
echo "Test 2: Checking HIPTENSOR_COMPUTE_DESC_64F is used..."
COUNT=$(grep -c "HIPTENSOR_COMPUTE_DESC_64F" src/heff_optimized_gpu.cpp)
if [ "$COUNT" -eq 4 ]; then
    echo "  ✓ PASS: Found 4 occurrences of HIPTENSOR_COMPUTE_DESC_64F"
    PASS=$((PASS+1))
else
    echo "  ❌ FAIL: Expected 4 occurrences, found $COUNT"
    FAIL=$((FAIL+1))
fi

# Test 3: Check old destroy function is replaced
echo "Test 3: Checking hiptensorDestroyContractionDescriptor is replaced..."
if grep -q "hiptensorDestroyContractionDescriptor" src/heff_optimized_gpu.cpp 2>/dev/null; then
    echo "  ❌ FAIL: Old hiptensorDestroyContractionDescriptor found"
    FAIL=$((FAIL+1))
else
    echo "  ✓ PASS: No old hiptensorDestroyContractionDescriptor found"
    PASS=$((PASS+1))
fi

# Test 4: Check new destroy function is present
echo "Test 4: Checking hiptensorDestroyOperationDescriptor is used..."
COUNT=$(grep -c "hiptensorDestroyOperationDescriptor" src/heff_optimized_gpu.cpp)
if [ "$COUNT" -eq 4 ]; then
    echo "  ✓ PASS: Found 4 occurrences of hiptensorDestroyOperationDescriptor"
    PASS=$((PASS+1))
else
    echo "  ❌ FAIL: Expected 4 occurrences, found $COUNT"
    FAIL=$((FAIL+1))
fi

# Test 5: Check handle dereferencing in hiptensorCreateContraction
echo "Test 5: Checking handle dereferencing in hiptensorCreateContraction..."
COUNT=$(grep -A1 "hiptensorCreateContraction" src/heff_optimized_gpu.cpp | grep -c "\*handle")
if [ "$COUNT" -eq 4 ]; then
    echo "  ✓ PASS: All 4 hiptensorCreateContraction calls dereference handle"
    PASS=$((PASS+1))
else
    echo "  ❌ FAIL: Expected 4 dereferenced handles, found $COUNT"
    FAIL=$((FAIL+1))
fi

# Test 6: Check handle dereferencing in hiptensorCreateTensorDescriptor
echo "Test 6: Checking handle dereferencing in hiptensorCreateTensorDescriptor..."
COUNT=$(grep -A1 "hiptensorCreateTensorDescriptor" src/heff_optimized_gpu.cpp | grep -c "\*handle")
if [ "$COUNT" -eq 9 ]; then
    echo "  ✓ PASS: All 9 hiptensorCreateTensorDescriptor calls dereference handle"
    PASS=$((PASS+1))
else
    echo "  ❌ FAIL: Expected 9 dereferenced handles, found $COUNT"
    FAIL=$((FAIL+1))
fi

# Test 7: Check handle dereferencing in hiptensorContract
echo "Test 7: Checking handle dereferencing in hiptensorContract..."
COUNT=$(grep -A1 "hiptensorContract" src/heff_optimized_gpu.cpp | grep -c "\*handle")
if [ "$COUNT" -eq 4 ]; then
    echo "  ✓ PASS: All 4 hiptensorContract calls dereference handle"
    PASS=$((PASS+1))
else
    echo "  ❌ FAIL: Expected 4 dereferenced handles, found $COUNT"
    FAIL=$((FAIL+1))
fi

# Test 8: Check argument count in hiptensorCreateContraction (no integer counts)
echo "Test 8: Checking hiptensorCreateContraction has no integer mode counts..."
if grep -E "desc_[A-Z0-9]+, [0-9]+, modes" src/heff_optimized_gpu.cpp 2>/dev/null; then
    echo "  ❌ FAIL: Found integer mode counts in hiptensorCreateContraction"
    FAIL=$((FAIL+1))
else
    echo "  ✓ PASS: No integer mode counts found"
    PASS=$((PASS+1))
fi

# Test 9: Check hiptensorDestroy doesn't use &handle
echo "Test 9: Checking hiptensorDestroy uses handle (not &handle)..."
if grep -q "hiptensorDestroy(&handle)" src/test_phase1.cpp 2>/dev/null; then
    echo "  ❌ FAIL: Found hiptensorDestroy(&handle) - should be hiptensorDestroy(handle)"
    FAIL=$((FAIL+1))
else
    echo "  ✓ PASS: hiptensorDestroy uses correct signature"
    PASS=$((PASS+1))
fi

# Test 10: Check header uses OperationDescriptor
echo "Test 10: Checking header uses hiptensorOperationDescriptor_t..."
if grep -q "hiptensorOperationDescriptor_t contraction" src/heff_optimized_gpu.h 2>/dev/null; then
    echo "  ✓ PASS: Header uses hiptensorOperationDescriptor_t"
    PASS=$((PASS+1))
else
    echo "  ❌ FAIL: Header doesn't use hiptensorOperationDescriptor_t"
    FAIL=$((FAIL+1))
fi

echo ""
echo "=========================================="
echo "Verification Results"
echo "=========================================="
echo "PASSED: $PASS/10"
echo "FAILED: $FAIL/10"
echo ""

if [ "$FAIL" -eq 0 ]; then
    echo "✅ ALL TESTS PASSED!"
    echo ""
    echo "Files are ready for compilation with ROCm 7.2.0"
    echo "To build: ./build_mi300x.sh"
    exit 0
else
    echo "❌ SOME TESTS FAILED!"
    echo ""
    echo "Please review the failures above and fix the issues."
    exit 1
fi
