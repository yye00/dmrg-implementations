#!/usr/bin/env bash
# Probe: match R3 compute load (3 warmup, 10 polish) with current binary
# which has single-site polish. If times match pre-R3 baseline → F1+F2 clean.
set -u
cd ~/dmrg-implementations
TAG="probe_$(date -u +%Y%m%dT%H%M%SZ)"
OUT="benchmarks/paper_results/mi300x/challenge/_probe_logs/${TAG}"
mkdir -p "$OUT"

BIN=gpu-rocm/pdmrg-gpu/build/pdmrg_gpu
# model L chi (n_outer=20, matches pre-R3 and R3 default)
configs=(
  "heisenberg 50 128"
  "heisenberg 100 128"
  "tfim 50 128"
  "tfim 100 128"
)

echo "=== Probe started $(date -u) ===" | tee "${OUT}/SUMMARY.txt"
for cfg in "${configs[@]}"; do
  set -- $cfg
  model=$1; L=$2; chi=$3
  flag=""
  [ "$model" = "tfim" ] && flag="--tfim"
  total_times=()
  for rep in 1 2 3; do
    log="${OUT}/${model}_L${L}_chi${chi}_rep${rep}.log"
    t0=$(date +%s.%N)
    "$BIN" $L $chi 20 $flag --segments 2 --local-sweeps 2 --warmup 3 --polish 10 > "$log" 2>&1
    t1=$(date +%s.%N)
    t=$(awk "BEGIN{printf \"%.3f\", $t1 - $t0}")
    total_times+=($t)
    echo "  $model L=$L chi=$chi rep=$rep elapsed=${t}s" | tee -a "${OUT}/SUMMARY.txt"
  done
  # median
  IFS=$"\n" sorted=($(printf "%s\n" "${total_times[@]}" | sort -g))
  med=${sorted[1]}
  echo "[${model} L=${L} chi=${chi}] times=${total_times[*]} median=${med}" | tee -a "${OUT}/SUMMARY.txt"
done
echo "=== Probe done $(date -u) ===" | tee -a "${OUT}/SUMMARY.txt"
echo "Logs in $OUT"
