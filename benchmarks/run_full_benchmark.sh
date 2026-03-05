#!/bin/bash
# ============================================================================
# Master Benchmark Orchestration Script
# ============================================================================
# Runs both CPU and GPU DMRG benchmarks and generates comparison report.
#
# Usage:
#   ./run_full_benchmark.sh              # Full benchmark (CPU + GPU)
#   ./run_full_benchmark.sh --cpu-only   # CPU benchmarks only
#   ./run_full_benchmark.sh --gpu-only   # GPU benchmarks only
#   ./run_full_benchmark.sh --quick      # Quick mode (small cases)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

CPU_ONLY=false
GPU_ONLY=false
QUICK=false
SKIP_LARGE=false

for arg in "$@"; do
    case $arg in
        --cpu-only) CPU_ONLY=true ;;
        --gpu-only) GPU_ONLY=true ;;
        --quick) QUICK=true; SKIP_LARGE=true ;;
        --skip-large) SKIP_LARGE=true ;;
    esac
done

echo "============================================================"
echo "DMRG BENCHMARK SUITE - CPU vs GPU"
echo "============================================================"
echo "Date:      $(date)"
echo "Host:      $(hostname)"
echo "CPU:       $(lscpu 2>/dev/null | grep 'Model name' | head -1 | sed 's/Model name: *//')"
echo "CPU-only:  $CPU_ONLY"
echo "GPU-only:  $GPU_ONLY"
echo "Quick:     $QUICK"
echo "============================================================"
echo ""

# ============================================================================
# CPU Benchmarks
# ============================================================================
if ! $GPU_ONLY; then
    echo ""
    echo "########################################################"
    echo "# RUNNING CPU BENCHMARKS (Quimb DMRG1 + DMRG2)"
    echo "########################################################"
    echo ""

    PYTHON="$ROOT_DIR/pdmrg/venv/bin/python"
    if [[ ! -f "$PYTHON" ]]; then
        PYTHON=$(which python3)
    fi

    CPU_ARGS=""
    if $SKIP_LARGE; then
        CPU_ARGS="$CPU_ARGS --skip-large"
    fi

    CPU_JSON="$SCRIPT_DIR/cpu_benchmark_results_${TIMESTAMP}.json"
    $PYTHON "$SCRIPT_DIR/cpu_gpu_benchmark.py" $CPU_ARGS --out "$CPU_JSON"

    echo ""
    echo "CPU results saved to: $CPU_JSON"
fi

# ============================================================================
# GPU Benchmarks
# ============================================================================
if ! $CPU_ONLY; then
    echo ""
    echo "########################################################"
    echo "# RUNNING GPU BENCHMARKS"
    echo "########################################################"
    echo ""

    # Check for ROCm/HIP
    if command -v rocm-smi &> /dev/null; then
        GPU_ARGS=""
        if $QUICK; then
            GPU_ARGS="--quick"
        fi
        bash "$SCRIPT_DIR/gpu_full_benchmark.sh" $GPU_ARGS
    else
        echo "WARNING: ROCm/HIP not available on this machine."
        echo "GPU benchmarks require AMD MI300X with ROCm installed."
        echo ""
        echo "To run GPU benchmarks on MI300X:"
        echo "  1. Copy this directory to the MI300X machine"
        echo "  2. cd gpu-port/build && cmake .. && make -j"
        echo "  3. cd ../../benchmarks && ./gpu_full_benchmark.sh"
    fi
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "============================================================"
echo "BENCHMARK SUITE COMPLETE"
echo "============================================================"
echo "Timestamp: $TIMESTAMP"
echo ""
echo "To generate the comparison report, run:"
echo "  python3 $SCRIPT_DIR/generate_report.py"
echo "============================================================"
