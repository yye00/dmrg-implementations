#!/bin/bash
#
# GPU Multi-Stream Benchmark Suite
#
# Runs comprehensive GPU DMRG benchmarks with multiple stream counts
# and compares against CPU PDMRG/PDMRG2 results
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PDMRG_GPU_DIR="${SCRIPT_DIR}/../pdmrg-gpu"
BUILD_DIR="${PDMRG_GPU_DIR}/build"
RESULTS_DIR="${SCRIPT_DIR}/gpu_results"

# Create results directory
mkdir -p "${RESULTS_DIR}"

echo "============================================================"
echo "GPU Multi-Stream Benchmark Suite"
echo "============================================================"
echo "Script dir:   ${SCRIPT_DIR}"
echo "PDMRG-GPU:    ${PDMRG_GPU_DIR}"
echo "Build dir:    ${BUILD_DIR}"
echo "Results:    ${RESULTS_DIR}"
echo ""

# Check if GPU executable exists
GPU_EXE="${BUILD_DIR}/test_heisenberg_multistream"
if [ ! -f "${GPU_EXE}" ]; then
    echo "❌ GPU executable not found: ${GPU_EXE}"
    echo ""
    echo "Build with:"
    echo "  cd ${PDMRG_GPU_DIR}"
    echo "  mkdir -p build && cd build"
    echo "  cmake -DCMAKE_BUILD_TYPE=Release .."
    echo "  make -j16 test_heisenberg_multistream"
    echo ""
    exit 1
fi

echo "✓ GPU executable found: ${GPU_EXE}"
echo ""

# Check Python dependencies
echo "Checking Python dependencies..."
python3 -c "import quimb" 2>/dev/null && echo "✓ quimb installed" || echo "⚠️  quimb not installed (will use hardcoded reference)"
python3 -c "import numpy" && echo "✓ numpy installed" || { echo "❌ numpy required"; exit 1; }
echo ""

# ==============================================================================
# Benchmark 1: Heisenberg L=8, chi=32 (Small, Fast)
# ==============================================================================
echo "============================================================"
echo "Benchmark 1: Heisenberg L=8, chi=32 (Small)"
echo "============================================================"

python3 "${SCRIPT_DIR}/gpu_heisenberg_benchmark.py" \
    --L 8 \
    --chi 32 \
    --max-iter 20 \
    --streams 1,2,4,8 \
    --out "${RESULTS_DIR}/heisenberg_L8_chi32.json" \
    --check-speedup \
    --min-efficiency 0.60

echo ""
echo "✓ Benchmark 1 complete"
echo ""

# ==============================================================================
# Benchmark 2: Heisenberg L=12, chi=64 (Medium)
# ==============================================================================
echo "============================================================"
echo "Benchmark 2: Heisenberg L=12, chi=64 (Medium)"
echo "============================================================"

python3 "${SCRIPT_DIR}/gpu_heisenberg_benchmark.py" \
    --L 12 \
    --chi 64 \
    --max-iter 30 \
    --streams 1,2,4,8 \
    --out "${RESULTS_DIR}/heisenberg_L12_chi64.json" \
    --check-speedup \
    --min-efficiency 0.60

echo ""
echo "✓ Benchmark 2 complete"
echo ""

# ==============================================================================
# Benchmark 3: Heisenberg L=16, chi=128 (Large) - Optional
# ==============================================================================
if [ "${RUN_LARGE:-0}" = "1" ]; then
    echo "============================================================"
    echo "Benchmark 3: Heisenberg L=16, chi=128 (Large)"
    echo "============================================================"

    python3 "${SCRIPT_DIR}/gpu_heisenberg_benchmark.py" \
        --L 16 \
        --chi 128 \
        --max-iter 50 \
        --streams 1,2,4,8 \
        --out "${RESULTS_DIR}/heisenberg_L16_chi128.json" \
        --check-speedup \
        --min-efficiency 0.50

    echo ""
    echo "✓ Benchmark 3 complete"
    echo ""
fi

# ==============================================================================
# Summary
# ==============================================================================
echo "============================================================"
echo "Benchmark Suite Complete"
echo "============================================================"
echo ""
echo "Results saved to: ${RESULTS_DIR}/"
echo ""
ls -lh "${RESULTS_DIR}"/*.json
echo ""

# Display summary from JSON files
echo "============================================================"
echo "Summary: Accuracy"
echo "============================================================"

for json_file in "${RESULTS_DIR}"/*.json; do
    if [ -f "$json_file" ]; then
        echo ""
        echo "File: $(basename $json_file)"
        python3 -c "
import json
import sys

with open('$json_file') as f:
    data = json.load(f)

ref_E = data['reference_energy']
print(f'  Reference: {ref_E:.12f}')

for r in data['results']:
    if r['implementation'] == 'GPU_DMRG':
        streams = r['num_streams']
        passed = r.get('passed', False)
        delta_E = r.get('delta_E', None)
        status = '✓' if passed else '❌'
        if delta_E is not None:
            print(f'  {status} {streams} streams: ΔE = {delta_E:.2e}')
        else:
            print(f'  {status} {streams} streams: ERROR')
"
    fi
done

echo ""
echo "============================================================"
echo "Summary: Scalability"
echo "============================================================"

for json_file in "${RESULTS_DIR}"/*.json; do
    if [ -f "$json_file" ]; then
        echo ""
        echo "File: $(basename $json_file)"
        python3 -c "
import json

with open('$json_file') as f:
    data = json.load(f)

if 'scalability' not in data or 'baseline_time' not in data['scalability']:
    print('  No scalability data')
else:
    baseline = data['scalability']['baseline_time']
    speedups = data['scalability'].get('speedups', {})
    efficiencies = data['scalability'].get('efficiencies', {})

    print(f'  Baseline (1 stream): {baseline:.3f}s')
    print(f'  {"Streams":<10} {"Speedup":<10} {"Efficiency":<12}')
    print(f'  {"-"*32}')

    for streams_str, speedup in sorted(speedups.items(), key=lambda x: int(x[0])):
        streams = int(streams_str)
        eff = efficiencies.get(streams_str, 0.0)
        print(f'  {streams:<10} {speedup:<10.2f} {eff:<12.1%}')
"
    fi
done

echo ""
echo "============================================================"
echo "To run large benchmark (L=16, chi=128):"
echo "  RUN_LARGE=1 ./run_gpu_suite.sh"
echo "============================================================"
echo ""

exit 0
