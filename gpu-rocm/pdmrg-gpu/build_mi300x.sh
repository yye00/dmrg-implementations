#!/bin/bash
# Build script for Parallel DMRG-GPU on MI300X
set -e

echo "=========================================="
echo "Building PDMRG-GPU for MI300X"
echo "=========================================="

export ROCM_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$PATH

if ! command -v hipcc &> /dev/null; then
    echo "ERROR: hipcc not found. Is ROCm installed?"
    exit 1
fi

echo "hipcc: $(which hipcc)"
echo ""

rm -rf build
mkdir build
cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=$ROCM_PATH/llvm/bin/clang++ \
  -DCMAKE_C_COMPILER=$ROCM_PATH/llvm/bin/clang \
  -DGPU_TARGETS=gfx942

make -j8 pdmrg_gpu

echo ""
echo "=========================================="
echo "BUILD SUCCESSFUL"
echo "=========================================="
echo ""
echo "Run: ./build/pdmrg_gpu --warmup 1 --polish 0 [options]"
echo "  Example: ./build/pdmrg_gpu --L 32 --chi 64 --n_sweeps 20 --warmup 1 --polish 0"
echo ""
