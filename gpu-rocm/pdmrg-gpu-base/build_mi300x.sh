#!/bin/bash
# Build script for Naive Baseline PDMRG-GPU-BASE on MI300X
# Competent first-pass stream-parallel GPU DMRG (per-stream device-pointer
# Lanczos, precomputed WW, non-blocking streams, single-site warmup/polish).
# Round-3 rewrite, CLAUDE.md compliant.
set -e

echo "=========================================="
echo "Building Naive Baseline PDMRG-GPU-BASE for MI300X"
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

make -j8 pdmrg_gpu_base

echo ""
echo "=========================================="
echo "BUILD SUCCESSFUL"
echo "=========================================="
echo ""
echo "Run: ./build/pdmrg_gpu_base [L] [chi_max] [n_outer] [...flags]"
echo "  Example: ./build/pdmrg_gpu_base 32 64 20 --segments 2 --warmup 1 --polish 0"
echo ""
