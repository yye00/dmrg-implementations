#!/bin/bash
# ============================================================================
# GPU vs CPU Benchmark Script for PDMRG and PDMRG2
# ============================================================================
# Compares GPU implementations (AMD MI300X) against CPU Python implementations
# Tests both Heisenberg (d=2) and Josephson junction (d=5) models
#
# Usage:
#   ./gpu_benchmark.sh [--quick]   # Quick test with small sizes
#   ./gpu_benchmark.sh [--full]    # Full benchmark with all sizes

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
GPU_DIR="$ROOT_DIR/gpu-port"
BUILD_DIR="$GPU_DIR/build"
PDMRG_DIR="$ROOT_DIR/pdmrg"
RESULTS_FILE="$SCRIPT_DIR/gpu_benchmark_results.txt"

QUICK_MODE=false
if [[ "$1" == "--quick" ]]; then
    QUICK_MODE=true
fi

echo "============================================================"
echo "PDMRG/PDMRG2 GPU vs CPU Benchmark"
echo "============================================================"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo ""

# Check GPU
if command -v rocm-smi &> /dev/null; then
    echo "GPU Info:"
    rocm-smi --showproductname 2>/dev/null || true
    echo ""
fi

# Build GPU executables if needed
if [[ ! -f "$BUILD_DIR/pdmrg_gpu" || ! -f "$BUILD_DIR/pdmrg2_gpu" ]]; then
    echo "Building GPU executables..."
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    cmake "$GPU_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
        -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
        -DGPU_TARGETS=gfx942
    make -j8 pdmrg_gpu pdmrg2_gpu
    cd "$SCRIPT_DIR"
    echo "Build complete."
    echo ""
fi

# Results header
cat > "$RESULTS_FILE" << 'HEADER'
============================================================
PDMRG/PDMRG2 GPU vs CPU Performance Comparison
============================================================

HEADER

echo "Date: $(date)" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# ============================================================================
# GPU Benchmarks
# ============================================================================

echo "============================================================"
echo "GPU Benchmarks"
echo "============================================================"

run_gpu_benchmark() {
    local exe=$1
    local model=$2
    local L=$3
    local max_D=$4
    local sweeps=$5
    local extra_args=$6
    local label=$7

    echo ""
    echo "--- $label ---"
    echo "Running: $BUILD_DIR/$exe --model $model --L $L --max-D $max_D --sweeps $sweeps $extra_args"

    local result
    result=$("$BUILD_DIR/$exe" --model "$model" --L "$L" --max-D "$max_D" --sweeps "$sweeps" $extra_args 2>&1)
    echo "$result" | tail -20

    # Extract energy and time from output
    local energy=$(echo "$result" | grep "Final E:" | tail -1 | awk '{print $NF}')
    local time_val=$(echo "$result" | grep "Time:" | tail -1 | awk '{print $2}')

    echo "" >> "$RESULTS_FILE"
    echo "$label" >> "$RESULTS_FILE"
    echo "  Energy: $energy" >> "$RESULTS_FILE"
    echo "  Time: ${time_val}s" >> "$RESULTS_FILE"
}

# Heisenberg benchmarks
echo ""
echo "=== Heisenberg Model (d=2) ==="
echo "" >> "$RESULTS_FILE"
echo "=== Heisenberg Model (d=2) ===" >> "$RESULTS_FILE"

if $QUICK_MODE; then
    HEISENBERG_SIZES="8"
    HEISENBERG_D="50"
    SWEEPS=5
else
    HEISENBERG_SIZES="8 10 12"
    HEISENBERG_D="100"
    SWEEPS=10
fi

for L in $HEISENBERG_SIZES; do
    run_gpu_benchmark "pdmrg_gpu" "heisenberg" "$L" "$HEISENBERG_D" "$SWEEPS" "--streams 1" \
        "PDMRG_GPU  | Heisenberg L=$L D=$HEISENBERG_D streams=1"

    run_gpu_benchmark "pdmrg2_gpu" "heisenberg" "$L" "$HEISENBERG_D" "$SWEEPS" "--streams 1" \
        "PDMRG2_GPU | Heisenberg L=$L D=$HEISENBERG_D streams=1"
done

# Josephson benchmarks
echo ""
echo "=== Josephson Junction Model (d=5, complex128) ==="
echo "" >> "$RESULTS_FILE"
echo "=== Josephson Junction Model (d=5, complex128) ===" >> "$RESULTS_FILE"

if $QUICK_MODE; then
    JOSEPHSON_SIZES="4"
    JOSEPHSON_D="20"
    SWEEPS_J=5
else
    JOSEPHSON_SIZES="4 6 8"
    JOSEPHSON_D="50"
    SWEEPS_J=10
fi

for L in $JOSEPHSON_SIZES; do
    run_gpu_benchmark "pdmrg_gpu" "josephson" "$L" "$JOSEPHSON_D" "$SWEEPS_J" \
        "--streams 1 --n-max 2" \
        "PDMRG_GPU  | Josephson L=$L D=$JOSEPHSON_D n_max=2 streams=1"

    run_gpu_benchmark "pdmrg2_gpu" "josephson" "$L" "$JOSEPHSON_D" "$SWEEPS_J" \
        "--streams 1 --n-max 2" \
        "PDMRG2_GPU | Josephson L=$L D=$JOSEPHSON_D n_max=2 streams=1"
done

# ============================================================================
# CPU Benchmarks (Python PDMRG)
# ============================================================================

echo ""
echo "============================================================"
echo "CPU Benchmarks (Python PDMRG)"
echo "============================================================"
echo "" >> "$RESULTS_FILE"
echo "=== CPU Reference (Python PDMRG) ===" >> "$RESULTS_FILE"

# Check if Python environment is available
PYTHON_AVAILABLE=false
if [[ -d "$ROOT_DIR/.venv-bench" ]]; then
    source "$ROOT_DIR/.venv-bench/bin/activate" 2>/dev/null && PYTHON_AVAILABLE=true
elif [[ -d "$PDMRG_DIR/venv" ]]; then
    source "$PDMRG_DIR/venv/bin/activate" 2>/dev/null && PYTHON_AVAILABLE=true
fi

if $PYTHON_AVAILABLE; then
    run_cpu_benchmark() {
        local model=$1
        local L=$2
        local bond_dim=$3
        local sweeps=$4
        local label=$5

        echo ""
        echo "--- $label ---"

        local start_time=$(date +%s%N)
        local result
        result=$(python -c "
import sys
sys.path.insert(0, '$PDMRG_DIR')
from pdmrg.hamiltonians.heisenberg import build_heisenberg_mpo
from pdmrg.dmrg import pdmrg_main

class FakeMPI:
    def Get_rank(self): return 0
    def Get_size(self): return 1
    def gather(self, data, root=0): return [data]
    def send(self, *a, **kw): pass
    def recv(self, *a, **kw): return None

class FakeComm:
    COMM_WORLD = FakeMPI()

if '$model' == 'heisenberg':
    mpo = build_heisenberg_mpo($L)
    energy, _ = pdmrg_main(
        L=$L, mpo=mpo, max_sweeps=$sweeps, bond_dim=$bond_dim,
        bond_dim_warmup=min(50, $bond_dim), n_warmup_sweeps=5,
        tol=1e-10, dtype='float64', comm=FakeMPI(), verbose=False
    )
    print(f'Energy: {energy:.12f}')
" 2>&1)
        local end_time=$(date +%s%N)
        local elapsed=$(( (end_time - start_time) / 1000000 ))
        local elapsed_sec=$(echo "scale=4; $elapsed / 1000" | bc)

        echo "$result"
        echo "  Wall time: ${elapsed_sec}s"

        echo "" >> "$RESULTS_FILE"
        echo "$label" >> "$RESULTS_FILE"
        local energy_val=$(echo "$result" | grep "Energy:" | awk '{print $2}')
        echo "  Energy: $energy_val" >> "$RESULTS_FILE"
        echo "  Time: ${elapsed_sec}s" >> "$RESULTS_FILE"
    }

    for L in $HEISENBERG_SIZES; do
        run_cpu_benchmark "heisenberg" "$L" "$HEISENBERG_D" "$SWEEPS" \
            "CPU_PDMRG  | Heisenberg L=$L D=$HEISENBERG_D"
    done
else
    echo "Python environment not available. Skipping CPU benchmarks."
    echo "  (CPU benchmarks skipped - no Python environment)" >> "$RESULTS_FILE"
fi

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "============================================================"
echo "Benchmark Complete"
echo "============================================================"
echo "Results written to: $RESULTS_FILE"
echo ""

cat "$RESULTS_FILE"
