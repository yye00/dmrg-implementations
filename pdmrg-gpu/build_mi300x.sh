#!/bin/bash
# Build script for MI300X
# Run this on enc1-gpuvm015 (HotAisle)

set -e  # Exit on error

echo "=========================================="
echo "Building GPU DMRG for MI300X"
echo "=========================================="
echo ""

# Set ROCm paths
export ROCM_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH

echo "ROCm path: $ROCM_PATH"
echo "GPU target: gfx942 (MI300X)"
echo ""

# Check for ROCm
if ! command -v hipcc &> /dev/null; then
    echo "❌ ERROR: hipcc not found. Is ROCm installed?"
    echo "   Try: export PATH=/opt/rocm/bin:\$PATH"
    exit 1
fi

echo "✓ hipcc found: $(which hipcc)"
hipcc --version | head -3
echo ""

# Check for GPU
if command -v rocm-smi &> /dev/null; then
    echo "✓ GPU detected:"
    rocm-smi --showproductname
    echo ""
else
    echo "⚠️  Warning: rocm-smi not found, cannot verify GPU"
fi

# Clean build
echo "Cleaning previous build..."
rm -rf build
mkdir build
cd build

echo ""
echo "Configuring with CMake..."
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=$ROCM_PATH/llvm/bin/clang++ \
  -DCMAKE_C_COMPILER=$ROCM_PATH/llvm/bin/clang \
  -DGPU_TARGETS=gfx942 \
  || {
    echo ""
    echo "❌ CMake configuration failed!"
    echo "   Check that ROCm libraries are installed:"
    echo "   - rocBLAS"
    echo "   - rocSOLVER"
    echo "   - hipTensor"
    exit 1
  }

echo ""
echo "Building targets..."
echo "  - dmrg_benchmark (PDMRG vs PDMRG2 comparison)"
echo "  - dmrg_gpu_native (minimal transfers version)"
echo ""

make -j8 dmrg_benchmark dmrg_gpu_native || {
    echo ""
    echo "❌ Build failed!"
    echo "   Check compiler errors above."
    exit 1
}

echo ""
echo "=========================================="
echo "✅ BUILD SUCCESSFUL!"
echo "=========================================="
echo ""
echo "Executables created:"
ls -lh dmrg_benchmark dmrg_gpu_native 2>/dev/null || echo "Warning: executables not found"
echo ""
echo "Quick test:"
echo "  ./dmrg_benchmark 6 50 5"
echo ""
echo "Full benchmark:"
echo "  ./dmrg_benchmark 12 100 5"
echo ""
echo "GPU-native version:"
echo "  ./dmrg_gpu_native 12 100 5 pdmrg 4"
echo "  ./dmrg_gpu_native 12 100 5 pdmrg2 8"
echo ""
