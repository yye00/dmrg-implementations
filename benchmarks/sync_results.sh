#!/bin/bash
# Periodic sync of benchmark results from remote MI300X
REMOTE="hotaisle@23.183.40.83"
REMOTE_DIR="~/dmrg-implementations/benchmarks/paper_results"
LOCAL_DIR="/home/captain/clawd/work/dmrg-implementations/benchmarks/paper_results"

while true; do
    scp -q "$REMOTE:$REMOTE_DIR/results.json" "$LOCAL_DIR/results.json" 2>/dev/null && \
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Synced results.json ($(wc -c < "$LOCAL_DIR/results.json") bytes)"
    scp -q "$REMOTE:$REMOTE_DIR/benchmark.log" "$LOCAL_DIR/benchmark.log" 2>/dev/null && \
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Synced benchmark.log ($(wc -c < "$LOCAL_DIR/benchmark.log") bytes)"
    sleep 120
done
