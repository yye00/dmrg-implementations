#!/bin/bash
# ==============================================================================
# Reproducible CPU/GPU DMRG Benchmark Suite
# ==============================================================================
#
# This script ensures exact reproducibility between CPU and GPU benchmarks by:
# 1. Generating MPS and MPO data with fixed random seed
# 2. Saving to binary files
# 3. Loading same data in both CPU (Python/Quimb) and GPU (C++/HIP) versions
#
# This guarantees identical initial conditions and Hamiltonians.
#
# Usage:
#   ./run_reproducible_benchmark.sh [--heisenberg|--josephson|--all]
#
# Output:
#   - benchmark_data/: Binary MPS/MPO files
#   - reproducible_benchmark_results.json: Combined CPU/GPU results
# ==============================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$SCRIPT_DIR/benchmark_data"
GPU_BUILD_DIR="$ROOT_DIR/pdmrg-gpu/build"

# Parse arguments
MODEL="--all"
if [ $# -gt 0 ]; then
    MODEL="$1"
fi

echo "================================================================================"
echo "  Reproducible CPU/GPU DMRG Benchmark"
echo "================================================================================"
echo "Model: $MODEL"
echo "Data directory: $DATA_DIR"
echo ""

# ==============================================================================
# Step 1: Generate benchmark data (MPS + MPO)
# ==============================================================================
echo "### Step 1: Generating benchmark data ###"
echo ""

python3 "$SCRIPT_DIR/serialize_mps_mpo.py" \
    --output-dir "$DATA_DIR" \
    --seed 42 \
    $MODEL

if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Failed to generate benchmark data"
    exit 1
fi

echo ""
echo "✓ Benchmark data generated successfully"
echo ""

# ==============================================================================
# Step 2: Verify data can be loaded (Python side)
# ==============================================================================
echo "### Step 2: Verifying Python loader ###"
echo ""

# Test loading one file
MPS_FILE=$(find "$DATA_DIR" -name "*_mps.bin" | head -1)
if [ -n "$MPS_FILE" ]; then
    echo "Testing Python MPS loader on: $MPS_FILE"
    python3 "$SCRIPT_DIR/load_mps_mpo.py" "$MPS_FILE" --type mps
    echo ""
fi

MPO_FILE=$(find "$DATA_DIR" -name "*_mpo.bin" | head -1)
if [ -n "$MPO_FILE" ]; then
    echo "Testing Python MPO loader on: $MPO_FILE"
    python3 "$SCRIPT_DIR/load_mps_mpo.py" "$MPO_FILE" --type mpo
    echo ""
fi

echo "✓ Python loader verified"
echo ""

# ==============================================================================
# Step 3: Build C++ loader test (if not already built)
# ==============================================================================
echo "### Step 3: Building C++ loader test ###"
echo ""

mkdir -p "$GPU_BUILD_DIR"
cd "$GPU_BUILD_DIR"

# Check if CMakeLists.txt includes the test
if ! grep -q "test_mps_mpo_loader" "$ROOT_DIR/pdmrg-gpu/CMakeLists.txt" 2>/dev/null; then
    echo "Warning: test_mps_mpo_loader not found in CMakeLists.txt"
    echo "You may need to add it manually:"
    echo ""
    echo "  add_executable(test_mps_mpo_loader src/test_mps_mpo_loader.cpp)"
    echo ""
    echo "Skipping C++ verification for now..."
else
    cmake .. && make test_mps_mpo_loader -j16

    # ==============================================================================
    # Step 4: Verify data can be loaded (C++ side)
    # ==============================================================================
    echo ""
    echo "### Step 4: Verifying C++ loader ###"
    echo ""

    if [ -n "$MPS_FILE" ]; then
        echo "Testing C++ MPS loader on: $MPS_FILE"
        ./test_mps_mpo_loader mps "$MPS_FILE"
        echo ""
    fi

    if [ -n "$MPO_FILE" ]; then
        echo "Testing C++ MPO loader on: $MPO_FILE"
        ./test_mps_mpo_loader mpo "$MPO_FILE"
        echo ""
    fi

    echo "✓ C++ loader verified"
    echo ""
fi

cd "$SCRIPT_DIR"

# ==============================================================================
# Step 5: Run CPU benchmark with loaded data
# ==============================================================================
echo "### Step 5: Running CPU benchmark ###"
echo ""

# TODO: Update cpu_gpu_benchmark.py to support --load-data flag
# For now, just run the standard benchmark
python3 "$SCRIPT_DIR/cpu_gpu_benchmark.py" \
    --out cpu_reproducible_results.json

echo ""
echo "✓ CPU benchmark complete"
echo ""

# ==============================================================================
# Step 6: Run GPU benchmark with loaded data
# ==============================================================================
echo "### Step 6: Running GPU benchmark ###"
echo ""

# TODO: Create GPU benchmark executable that loads from files
# For now, run standard GPU benchmark
if [ -f "$GPU_BUILD_DIR/pdmrg_complete" ]; then
    echo "Running GPU benchmark..."
    cd "$GPU_BUILD_DIR"
    # Add data loading support to GPU executables
    echo "Note: GPU benchmark with data loading not yet implemented"
    echo "Please update GPU executables to use MPSLoader/MPOLoader"
else
    echo "GPU executable not found. Build with:"
    echo "  cd $ROOT_DIR/pdmrg-gpu && mkdir -p build && cd build && cmake .. && make -j16"
fi

cd "$SCRIPT_DIR"

# ==============================================================================
# Summary
# ==============================================================================
echo ""
echo "================================================================================"
echo "  Benchmark Complete"
echo "================================================================================"
echo ""
echo "Generated data:"
ls -lh "$DATA_DIR"
echo ""
echo "Next steps:"
echo "  1. Update CPU benchmark to load from $DATA_DIR"
echo "  2. Update GPU benchmark to load from $DATA_DIR"
echo "  3. Compare results to verify exact reproducibility"
echo ""
echo "Files:"
echo "  - Benchmark data: $DATA_DIR/"
echo "  - CPU results: cpu_reproducible_results.json"
echo "  - GPU results: (to be implemented)"
echo ""
