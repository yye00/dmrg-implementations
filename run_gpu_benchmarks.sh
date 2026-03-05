#!/bin/bash
# GPU Benchmark Suite - Tests PDMRG_GPU and PDMRG2_GPU with multiple stream counts

set -e
cd ~/dmrg-implementations

OUTPUT_FILE="gpu_benchmark_results_$(date +%Y%m%d_%H%M%S).json"

echo "{"
echo "  \"timestamp\": \"$(date -Iseconds)\","
echo "  \"platform\": \"AMD MI300X\","
echo "  \"results\": {"

# Helper function to run and parse
run_gpu_test() {
    local impl=$1
    local model=$2
    local L=$3
    local D=$4
    local streams=$5
    local extra_args=$6

    local exe="./gpu-port/build/${impl}"
    local cmd="$exe --model $model --L $L --max-D $D --sweeps 20 --streams $streams $extra_args"

    echo "    Running: $impl $model L=$L D=$D streams=$streams" >&2

    local output=$($cmd 2>&1)
    local energy=$(echo "$output" | grep "Final E:" | awk '{print $NF}')
    local time=$(echo "$output" | grep "time=" | tail -1 | sed 's/.*time=\([0-9.]*\)s.*/\1/')

    echo "      \"${impl}_${model}_L${L}_D${D}_streams${streams}\": {"
    echo "        \"implementation\": \"$impl\","
    echo "        \"model\": \"$model\","
    echo "        \"L\": $L,"
    echo "        \"D\": $D,"
    echo "        \"streams\": $streams,"
    echo "        \"energy\": $energy,"
    echo "        \"time_s\": ${time:-0.0}"
    echo "      },"
}

echo "    \"heisenberg\": {"

# Heisenberg benchmarks
for impl in pdmrg_gpu pdmrg2_gpu; do
    for streams in 1 2 4 8; do
        run_gpu_test "$impl" "heisenberg" 12 100 $streams ""
    done
done

echo "      \"_done\": true"
echo "    },"
echo "    \"josephson\": {"

# Josephson benchmarks
for impl in pdmrg_gpu pdmrg2_gpu; do
    for streams in 1 2 4 8; do
        run_gpu_test "$impl" "josephson" 8 50 $streams "--n-max 2"
    done
done

echo "      \"_done\": true"
echo "    }"
echo "  },"
echo "  \"reference_energy\": {"
echo "    \"heisenberg_L12\": -5.142090632841,"
echo "    \"josephson_L8\": -2.843801043139"
echo "  }"
echo "}"

echo "" >&2
echo "Results saved to: $OUTPUT_FILE" >&2
