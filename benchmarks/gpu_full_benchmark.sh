#!/bin/bash
# ============================================================================
# Comprehensive GPU DMRG Benchmark Suite for AMD MI300X
# ============================================================================
# Tests all 3 GPU implementations across Heisenberg and Josephson models
# with stream scaling study (1, 2, 4, 8 streams).
#
# Implementations:
#   1. dmrg_with_environments - Basic GPU DMRG with hipTensor
#   2. pdmrg_gpu             - Stream-parallelized BLAS-2
#   3. pdmrg2_gpu            - GPU-optimized BLAS-3
#
# Usage:
#   ./gpu_full_benchmark.sh              # Full benchmark
#   ./gpu_full_benchmark.sh --quick      # Quick test (small cases only)
#   ./gpu_full_benchmark.sh --heisenberg # Heisenberg only
#   ./gpu_full_benchmark.sh --josephson  # Josephson only

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
GPU_DIR="$ROOT_DIR/gpu-port"
BUILD_DIR="$GPU_DIR/build"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$SCRIPT_DIR"
JSON_OUT="$RESULTS_DIR/gpu_benchmark_results_${TIMESTAMP}.json"
LOG_FILE="$RESULTS_DIR/gpu_benchmark_${TIMESTAMP}.log"

# Parse arguments
QUICK_MODE=false
HEIS_ONLY=false
JOS_ONLY=false
for arg in "$@"; do
    case $arg in
        --quick) QUICK_MODE=true ;;
        --heisenberg) HEIS_ONLY=true ;;
        --josephson) JOS_ONLY=true ;;
    esac
done

# Build GPU executables if needed
build_gpu() {
    echo "Checking/building GPU executables..."
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    if [[ ! -f "pdmrg_gpu" || ! -f "pdmrg2_gpu" || ! -f "dmrg_with_environments" ]]; then
        cmake "$GPU_DIR" -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -5
        make -j$(nproc) pdmrg_gpu pdmrg2_gpu dmrg_with_environments 2>&1 | tail -10
        echo "Build complete."
    else
        echo "Executables already built."
    fi
    cd "$SCRIPT_DIR"
}

# Initialize JSON results
init_json() {
    cat > "$JSON_OUT" << EOF
{
  "timestamp": "$(date -Iseconds)",
  "platform": "GPU",
  "gpu_info": "$(rocm-smi --showproductname 2>/dev/null | grep 'GPU' | head -1 || echo 'AMD MI300X')",
  "results": {
    "heisenberg": {},
    "josephson": {}
  },
  "stream_scaling": {}
}
EOF
}

# Run a single GPU benchmark and capture output
# Args: exe model L D sweeps streams [extra_args...]
run_gpu_test() {
    local exe=$1
    local model=$2
    local L=$3
    local D=$4
    local sweeps=$5
    local streams=$6
    shift 6
    local extra="$@"

    local label="${exe}|${model}|L=${L}|D=${D}|streams=${streams}"
    echo ""
    echo "--- $label ---"

    local cmd="$BUILD_DIR/$exe --model $model --L $L --max-D $D --sweeps $sweeps --streams $streams $extra"
    echo "  CMD: $cmd"

    local start_ns=$(date +%s%N)
    local output
    output=$($cmd 2>&1) || true
    local end_ns=$(date +%s%N)
    local wall_ms=$(( (end_ns - start_ns) / 1000000 ))
    local wall_s=$(echo "scale=4; $wall_ms / 1000" | bc)

    # Extract energy from output (look for typical patterns)
    local energy=$(echo "$output" | grep -oP '(?<=E=)[0-9eE.+-]+' | tail -1)
    if [[ -z "$energy" ]]; then
        energy=$(echo "$output" | grep -oP '(?<=energy[= :]+)-?[0-9]+\.[0-9]+' | tail -1)
    fi
    if [[ -z "$energy" ]]; then
        energy=$(echo "$output" | grep -oP '(?<=Final E: )-?[0-9]+\.[0-9]+' | tail -1)
    fi

    echo "  Energy: ${energy:-N/A}"
    echo "  Wall time: ${wall_s}s"
    echo "$output" | tail -5

    # Append to log
    echo "" >> "$LOG_FILE"
    echo "=== $label ===" >> "$LOG_FILE"
    echo "$output" >> "$LOG_FILE"
    echo "Wall time: ${wall_s}s" >> "$LOG_FILE"
}

# dmrg_with_environments has no CLI args (hardcoded L=12, d=2, D=100)
run_dmrg_with_env() {
    echo ""
    echo "--- dmrg_with_environments (L=12, D=100, Heisenberg) ---"

    local start_ns=$(date +%s%N)
    local output
    output=$("$BUILD_DIR/dmrg_with_environments" 2>&1) || true
    local end_ns=$(date +%s%N)
    local wall_ms=$(( (end_ns - start_ns) / 1000000 ))
    local wall_s=$(echo "scale=4; $wall_ms / 1000" | bc)

    local energy=$(echo "$output" | grep -oP '(?<=Final E: )-?[0-9]+\.[0-9]+' | tail -1)
    if [[ -z "$energy" ]]; then
        energy=$(echo "$output" | grep -oP '(?<=energy[= :]+)-?[0-9]+\.[0-9]+' | tail -1)
    fi

    echo "  Energy: ${energy:-N/A}"
    echo "  Wall time: ${wall_s}s"
    echo "$output" | tail -5

    echo "" >> "$LOG_FILE"
    echo "=== dmrg_with_environments ===" >> "$LOG_FILE"
    echo "$output" >> "$LOG_FILE"
}

# ============================================================================
# MAIN
# ============================================================================

echo "============================================================"
echo "COMPREHENSIVE GPU DMRG BENCHMARK SUITE"
echo "AMD MI300X - $(date)"
echo "============================================================"
echo ""

# Build
build_gpu

# Init
init_json
echo "Logging to: $LOG_FILE"
echo "JSON output: $JSON_OUT"
echo ""

exec > >(tee -a "$LOG_FILE") 2>&1

# GPU info
echo "============================================================"
echo "GPU Information"
echo "============================================================"
rocm-smi --showid --showproductname 2>/dev/null || echo "(rocm-smi not available)"
echo ""

# ============================================================================
# Heisenberg Benchmarks
# ============================================================================
if ! $JOS_ONLY; then
    echo ""
    echo "########################################################"
    echo "# HEISENBERG MODEL (d=2, real)"
    echo "########################################################"

    # dmrg_with_environments (fixed parameters)
    echo ""
    echo "=== dmrg_with_environments ==="
    run_dmrg_with_env

    if $QUICK_MODE; then
        HEIS_CONFIGS="12:100:10"
    else
        HEIS_CONFIGS="12:100:20 20:100:30 40:200:40"
    fi

    for config in $HEIS_CONFIGS; do
        IFS=':' read -r L D SWEEPS <<< "$config"

        echo ""
        echo "=== Heisenberg L=$L D=$D ==="

        # pdmrg_gpu with stream scaling
        for S in 1 2 4 8; do
            run_gpu_test pdmrg_gpu heisenberg $L $D $SWEEPS $S
        done

        # pdmrg2_gpu with stream scaling
        for S in 1 2 4 8; do
            run_gpu_test pdmrg2_gpu heisenberg $L $D $SWEEPS $S
        done
    done
fi

# ============================================================================
# Josephson Junction Benchmarks
# ============================================================================
if ! $HEIS_ONLY; then
    echo ""
    echo "########################################################"
    echo "# JOSEPHSON JUNCTION (d=5, complex128)"
    echo "########################################################"

    if $QUICK_MODE; then
        JOS_CONFIGS="8:50:10:2"
    else
        JOS_CONFIGS="8:50:20:2 12:50:30:2 16:100:40:2"
    fi

    for config in $JOS_CONFIGS; do
        IFS=':' read -r L D SWEEPS NMAX <<< "$config"

        echo ""
        echo "=== Josephson L=$L D=$D n_max=$NMAX (d=$((NMAX+1))) ==="

        # pdmrg_gpu with stream scaling
        for S in 1 2 4 8; do
            run_gpu_test pdmrg_gpu josephson $L $D $SWEEPS $S --n-max $NMAX --E-J 1.0 --E-C 0.5
        done

        # pdmrg2_gpu with stream scaling
        for S in 1 2 4 8; do
            run_gpu_test pdmrg2_gpu josephson $L $D $SWEEPS $S --n-max $NMAX --E-J 1.0 --E-C 0.5
        done
    done
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "============================================================"
echo "GPU BENCHMARK COMPLETE"
echo "============================================================"
echo "Log: $LOG_FILE"
echo "JSON: $JSON_OUT"
echo "Timestamp: $(date)"
echo "============================================================"
