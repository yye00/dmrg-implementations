#!/bin/bash
# Build script for Optimized Two-Site DMRG-GPU (Block-Davidson + CPU-LAPACK SVD) on MI300X
set -e

echo "=========================================="
echo "Building DMRG2-GPU-OPT for MI300X"
echo "=========================================="
# Note: Block-Davidson (b=4) is HARDCODED in source; there is no runtime flag.
# opts_.device_k and opts_.rsvd are declared but never read in this variant.

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

make -j8 dmrg2_gpu_opt

echo ""
echo "=========================================="
echo "BUILD SUCCESSFUL"
echo "=========================================="
echo ""
echo "Run: ./build/dmrg2_gpu_opt [L] [chi_max] [n_sweeps]"
echo "  Example: ./build/dmrg2_gpu_opt 8 32 30"
echo ""
