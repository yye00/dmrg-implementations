#!/bin/bash
# HotAisle MI300X Environment Diagnostic Script
# Run this first to verify ROCm, hipTensor, and MPI setup

echo "=========================================="
echo "HotAisle MI300X Environment Diagnostic"
echo "=========================================="
echo ""

# Check for AMD GPUs
echo "[1/10] Checking for AMD GPUs..."
if command -v rocm-smi &> /dev/null; then
    rocm-smi --showproductname
    echo "✓ rocm-smi found"
else
    echo "✗ rocm-smi not found - ROCm may not be installed"
fi
echo ""

# Check ROCm version
echo "[2/10] Checking ROCm version..."
if command -v rocminfo &> /dev/null; then
    ROCM_VERSION=$(rocminfo | grep "Runtime Version" | head -1)
    echo "$ROCM_VERSION"

    # Check for MI300X (gfx942)
    if rocminfo | grep -q "gfx942"; then
        echo "✓ MI300X (gfx942) detected"
    else
        echo "⚠ MI300X (gfx942) not detected - found:"
        rocminfo | grep "Name:" | grep "gfx"
    fi
else
    echo "✗ rocminfo not found"
fi
echo ""

# Check HIP compiler
echo "[3/10] Checking HIP compiler..."
if command -v hipcc &> /dev/null; then
    hipcc --version | head -3
    echo "✓ hipcc found"
else
    echo "✗ hipcc not found"
fi
echo ""

# Check for hipTensor
echo "[4/10] Checking for hipTensor library..."
HIPTENSOR_PATHS=(
    "/opt/rocm/lib/libhiptensor.so"
    "/opt/rocm/lib64/libhiptensor.so"
    "/usr/lib/libhiptensor.so"
)

FOUND_HIPTENSOR=false
for path in "${HIPTENSOR_PATHS[@]}"; do
    if [ -f "$path" ]; then
        echo "✓ Found hipTensor at: $path"
        ls -lh "$path"
        FOUND_HIPTENSOR=true
        break
    fi
done

if [ "$FOUND_HIPTENSOR" = false ]; then
    echo "✗ hipTensor not found in standard locations"
    echo "  Search manually: find /opt/rocm -name 'libhiptensor*'"
fi
echo ""

# Check rocBLAS
echo "[5/10] Checking for rocBLAS..."
if [ -f "/opt/rocm/lib/librocblas.so" ] || [ -f "/opt/rocm/lib64/librocblas.so" ]; then
    echo "✓ rocBLAS found"
else
    echo "✗ rocBLAS not found"
fi
echo ""

# Check rocSOLVER
echo "[6/10] Checking for rocSOLVER..."
if [ -f "/opt/rocm/lib/librocsolver.so" ] || [ -f "/opt/rocm/lib64/librocsolver.so" ]; then
    echo "✓ rocSOLVER found"
else
    echo "✗ rocSOLVER not found"
fi
echo ""

# Check MPI
echo "[7/10] Checking MPI installation..."
if command -v mpirun &> /dev/null; then
    mpirun --version | head -3
    echo "✓ MPI found"

    # Check if ROCm-aware
    echo "  Checking for ROCm-aware MPI..."
    if mpirun --version 2>&1 | grep -i "ucx\|rocm\|gpu"; then
        echo "  ✓ Possibly ROCm-aware (mentions UCX/ROCm/GPU)"
    else
        echo "  ⚠ May not be ROCm-aware - GPU-direct may not work"
    fi
else
    echo "✗ mpirun not found"
fi
echo ""

# Check CMake
echo "[8/10] Checking CMake..."
if command -v cmake &> /dev/null; then
    cmake --version | head -1
    CMAKE_VERSION=$(cmake --version | grep -oP '\d+\.\d+' | head -1)
    if (( $(echo "$CMAKE_VERSION >= 3.20" | bc -l) )); then
        echo "✓ CMake version sufficient (>= 3.20)"
    else
        echo "⚠ CMake version may be too old (need >= 3.20)"
    fi
else
    echo "✗ cmake not found"
fi
echo ""

# Check C++ compiler
echo "[9/10] Checking C++ compiler..."
if command -v g++ &> /dev/null; then
    g++ --version | head -1
    echo "✓ g++ found"
elif command -v clang++ &> /dev/null; then
    clang++ --version | head -1
    echo "✓ clang++ found"
else
    echo "✗ No C++ compiler found"
fi
echo ""

# Test simple HIP compilation
echo "[10/10] Testing HIP compilation..."
cat > /tmp/test_hip.cpp << 'EOF'
#include <hip/hip_runtime.h>
#include <iostream>

__global__ void hello_kernel() {
    printf("Hello from GPU thread %d\n", threadIdx.x);
}

int main() {
    int deviceCount;
    hipGetDeviceCount(&deviceCount);
    std::cout << "Found " << deviceCount << " GPU(s)" << std::endl;

    if (deviceCount > 0) {
        hipDeviceProp_t prop;
        hipGetDeviceProperties(&prop, 0);
        std::cout << "GPU 0: " << prop.name << std::endl;
        std::cout << "Architecture: " << prop.gcnArchName << std::endl;
        std::cout << "Memory: " << prop.totalGlobalMem / (1024*1024*1024) << " GB" << std::endl;

        hello_kernel<<<1, 4>>>();
        hipDeviceSynchronize();
    }

    return 0;
}
EOF

if hipcc -o /tmp/test_hip /tmp/test_hip.cpp 2>/dev/null; then
    echo "✓ HIP compilation successful"
    echo "  Running test kernel..."
    /tmp/test_hip
    rm /tmp/test_hip /tmp/test_hip.cpp
else
    echo "✗ HIP compilation failed"
    echo "  Try manually: hipcc -o /tmp/test_hip /tmp/test_hip.cpp"
fi
echo ""

# Summary
echo "=========================================="
echo "Summary"
echo "=========================================="
echo ""
echo "Environment Variables:"
echo "  PATH: $PATH"
echo "  LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "  ROCM_PATH: ${ROCM_PATH:-not set}"
echo "  HIP_PLATFORM: ${HIP_PLATFORM:-not set}"
echo ""

echo "Next Steps:"
echo "1. If hipTensor missing: Install with 'apt-get install hiptensor' or check ROCm installation"
echo "2. If MPI not ROCm-aware: May need to rebuild OpenMPI with UCX + ROCm support"
echo "3. Save this output to share with Claude when starting GPU port"
echo ""
echo "Save diagnostic results:"
echo "  ./diagnostic.sh > hotaisle_diagnostic_$(date +%Y%m%d).txt"
