#!/bin/bash
# GPU Scalability Benchmark: pdmrg2-gpu vs baselines
# Compares accuracy and timing across segment counts
set -e

DMRG_GPU="../dmrg-gpu/build/dmrg_gpu"
DMRG2_GPU="../dmrg2-gpu/build/dmrg2_gpu"
PDMRG_GPU="../pdmrg-gpu/build/pdmrg_gpu"
PDMRG2_GPU="../pdmrg2-gpu/build/pdmrg2_gpu"

# Test configurations: L chi_max
CONFIGS=(
    "32 64"
    "64 128"
)

N_SWEEPS=20
N_OUTER=20
N_WARMUP=3
N_LOCAL=2

echo "=============================================="
echo "GPU Scalability Benchmark"
echo "=============================================="
echo ""

for CONFIG in "${CONFIGS[@]}"; do
    read -r L CHI <<< "$CONFIG"

    echo "=============================================="
    echo "Configuration: L=$L, chi=$CHI"
    echo "=============================================="

    # 1. dmrg-gpu (single-site, single stream baseline)
    echo ""
    echo "--- dmrg-gpu (single-site, 1 stream) ---"
    $DMRG_GPU $L $CHI $N_SWEEPS 2>&1 | grep -E "Final energy|Exact energy|Absolute error|PASS|FAIL|Total wall"
    echo ""

    # 2. dmrg2-gpu (two-site, single stream baseline)
    echo "--- dmrg2-gpu (two-site, 1 stream) ---"
    $DMRG2_GPU $L $CHI $N_SWEEPS 2>&1 | grep -E "Final energy|Exact energy|Absolute error|PASS|FAIL|Total wall"
    echo ""

    # 3. pdmrg-gpu with segments=2,4,8
    for SEGS in 2 4 8; do
        if [ $SEGS -gt $((L / 4)) ]; then
            echo "--- pdmrg-gpu (segments=$SEGS) --- SKIPPED (too many segments for L=$L)"
            continue
        fi
        echo "--- pdmrg-gpu (segments=$SEGS) ---"
        $PDMRG_GPU $L $CHI $N_OUTER --segments $SEGS --warmup $N_WARMUP --local-sweeps $N_LOCAL 2>&1 | grep -E "Final energy|Exact energy|Absolute error|PASS|FAIL|Total wall|Total parallel|coupling"
        echo ""
    done

    # 4. pdmrg2-gpu with segments=2,4,8
    for SEGS in 2 4 8; do
        if [ $SEGS -gt $((L / 4)) ]; then
            echo "--- pdmrg2-gpu (segments=$SEGS) --- SKIPPED (too many segments for L=$L)"
            continue
        fi
        echo "--- pdmrg2-gpu (segments=$SEGS) ---"
        $PDMRG2_GPU $L $CHI $N_OUTER --segments $SEGS --warmup $N_WARMUP --local-sweeps $N_LOCAL 2>&1 | grep -E "Final energy|Exact energy|Absolute error|PASS|FAIL|Total wall|Total parallel|coupling"
        echo ""
    done

    echo ""
done

echo "=============================================="
echo "Benchmark complete"
echo "=============================================="
