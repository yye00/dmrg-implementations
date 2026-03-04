#!/bin/bash
# ============================================================================
# Production DMRG-GPU Benchmark Suite
# Compares PDMRG (BLAS-2) vs PDMRG2 (BLAS-3) on AMD MI300X
# Tests: Heisenberg + Josephson models, stream scaling
# ============================================================================

set -e

BUILD_DIR="build"
RESULTS_DIR="benchmark_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${RESULTS_DIR}/benchmark_${TIMESTAMP}.log"

# Default parameters (can override via env vars)
L=${L:-12}
MAX_D=${MAX_D:-100}
SWEEPS=${SWEEPS:-10}
STREAMS=${STREAMS:-"1,2,4,8"}

echo "=============================================="
echo "PDMRG vs PDMRG2 GPU Benchmark Suite"
echo "AMD MI300X - Production Benchmark"
echo "=============================================="
echo ""
echo "Parameters:"
echo "  Chain length (L): $L"
echo "  Max bond dim (D): $MAX_D"
echo "  Sweeps:           $SWEEPS"
echo "  Streams:          $STREAMS"
echo ""

# Check if executables exist
if [ ! -f "${BUILD_DIR}/pdmrg_gpu" ] || [ ! -f "${BUILD_DIR}/pdmrg2_gpu" ]; then
    echo "Executables not found. Building..."
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    cmake .. 2>&1 | tail -5
    make -j$(nproc) pdmrg_gpu pdmrg2_gpu 2>&1 | tail -10
    cd ..
    echo "Build complete."
    echo ""
fi

# Create results directory
mkdir -p "$RESULTS_DIR"

echo "Logging to: $LOG_FILE"
echo ""

# Tee all output to log file
exec > >(tee -a "$LOG_FILE") 2>&1

# Print GPU info
echo "=============================================="
echo "GPU Information"
echo "=============================================="
rocm-smi --showid --showproductname 2>/dev/null || echo "(rocm-smi not available)"
echo ""

# ============================================================================
# Test 1: Heisenberg Model - PDMRG
# ============================================================================
echo "=============================================="
echo "TEST 1: Heisenberg - PDMRG (BLAS-2)"
echo "  L=$L, D=$MAX_D, sweeps=$SWEEPS, streams=$STREAMS"
echo "=============================================="
echo ""

PDMRG_HEIS_START=$(date +%s%N)
"${BUILD_DIR}/pdmrg_gpu" \
    --model heisenberg \
    --L "$L" \
    --max-D "$MAX_D" \
    --sweeps "$SWEEPS" \
    --streams "$STREAMS"
PDMRG_HEIS_END=$(date +%s%N)
PDMRG_HEIS_TIME=$(echo "scale=3; ($PDMRG_HEIS_END - $PDMRG_HEIS_START) / 1000000000" | bc)
echo ""
echo "[PDMRG Heisenberg total wall time: ${PDMRG_HEIS_TIME}s]"
echo ""

# ============================================================================
# Test 2: Heisenberg Model - PDMRG2
# ============================================================================
echo "=============================================="
echo "TEST 2: Heisenberg - PDMRG2 (BLAS-3)"
echo "  L=$L, D=$MAX_D, sweeps=$SWEEPS, streams=$STREAMS"
echo "=============================================="
echo ""

PDMRG2_HEIS_START=$(date +%s%N)
"${BUILD_DIR}/pdmrg2_gpu" \
    --model heisenberg \
    --L "$L" \
    --max-D "$MAX_D" \
    --sweeps "$SWEEPS" \
    --streams "$STREAMS"
PDMRG2_HEIS_END=$(date +%s%N)
PDMRG2_HEIS_TIME=$(echo "scale=3; ($PDMRG2_HEIS_END - $PDMRG2_HEIS_START) / 1000000000" | bc)
echo ""
echo "[PDMRG2 Heisenberg total wall time: ${PDMRG2_HEIS_TIME}s]"
echo ""

# ============================================================================
# Test 3: Josephson Junction - PDMRG
# ============================================================================
echo "=============================================="
echo "TEST 3: Josephson Junction - PDMRG (BLAS-2)"
echo "  L=$L, D=$MAX_D, sweeps=$SWEEPS, streams=$STREAMS"
echo "  n_max=2, E_J=1.0, E_C=0.5, phi_ext=pi/4"
echo "=============================================="
echo ""

PDMRG_JOS_START=$(date +%s%N)
"${BUILD_DIR}/pdmrg_gpu" \
    --model josephson \
    --L "$L" \
    --max-D "$MAX_D" \
    --sweeps "$SWEEPS" \
    --streams "$STREAMS" \
    --n-max 2 \
    --E-J 1.0 \
    --E-C 0.5
PDMRG_JOS_END=$(date +%s%N)
PDMRG_JOS_TIME=$(echo "scale=3; ($PDMRG_JOS_END - $PDMRG_JOS_START) / 1000000000" | bc)
echo ""
echo "[PDMRG Josephson total wall time: ${PDMRG_JOS_TIME}s]"
echo ""

# ============================================================================
# Test 4: Josephson Junction - PDMRG2
# ============================================================================
echo "=============================================="
echo "TEST 4: Josephson Junction - PDMRG2 (BLAS-3)"
echo "  L=$L, D=$MAX_D, sweeps=$SWEEPS, streams=$STREAMS"
echo "  n_max=2, E_J=1.0, E_C=0.5, phi_ext=pi/4"
echo "=============================================="
echo ""

PDMRG2_JOS_START=$(date +%s%N)
"${BUILD_DIR}/pdmrg2_gpu" \
    --model josephson \
    --L "$L" \
    --max-D "$MAX_D" \
    --sweeps "$SWEEPS" \
    --streams "$STREAMS" \
    --n-max 2 \
    --E-J 1.0 \
    --E-C 0.5
PDMRG2_JOS_END=$(date +%s%N)
PDMRG2_JOS_TIME=$(echo "scale=3; ($PDMRG2_JOS_END - $PDMRG2_JOS_START) / 1000000000" | bc)
echo ""
echo "[PDMRG2 Josephson total wall time: ${PDMRG2_JOS_TIME}s]"
echo ""

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "=============================================="
echo "BENCHMARK SUMMARY"
echo "=============================================="
echo ""
echo "Configuration: L=$L, D=$MAX_D, sweeps=$SWEEPS"
echo ""
echo "Model          | Algorithm | Wall Time"
echo "---------------|-----------|----------"
echo "Heisenberg     | PDMRG     | ${PDMRG_HEIS_TIME}s"
echo "Heisenberg     | PDMRG2    | ${PDMRG2_HEIS_TIME}s"
echo "Josephson      | PDMRG     | ${PDMRG_JOS_TIME}s"
echo "Josephson      | PDMRG2    | ${PDMRG2_JOS_TIME}s"
echo ""
echo "Expected results:"
echo "  Heisenberg L=12: E ~ -5.142091"
echo "  PDMRG2 should be faster than PDMRG for D >= ~50"
echo ""
echo "Full log: $LOG_FILE"
echo "=============================================="
