#!/bin/bash
# Build script for Naive Baseline DMRG2-GPU-BASE on MI300X
# Competent first-pass GPU implementation (precomputed WW at set_mpo time,
# device-pointer rocBLAS, on-device SVD truncation). Round-3 rewrite.
set -e

echo "=========================================="
echo "Building Naive Baseline DMRG2-GPU-BASE for MI300X"
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

make -j8 dmrg2_gpu_base

echo ""
echo "=========================================="
echo "BUILD SUCCESSFUL"
echo "=========================================="
echo ""
echo "Run: ./build/dmrg2_gpu_base [L] [chi_max] [n_sweeps]"
echo "  Example: ./build/dmrg2_gpu_base 8 32 30"
echo ""
