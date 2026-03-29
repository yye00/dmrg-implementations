#!/bin/bash
# Periodic sync of benchmark results from remote MI300X to local + push to GitHub.
# Usage: ./benchmarks/sync_results.sh [interval_seconds] [remote_host]
INTERVAL=${1:-300}
REMOTE=${2:-hotaisle@23.183.40.74}
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_REPO="~/dmrg-implementations"
RESULTS_DIR="benchmarks/paper_results"

cd "$REPO_DIR" || exit 1

echo "[$(date)] Starting result sync loop (every ${INTERVAL}s from ${REMOTE})"

while true; do
    # Check if remote is reachable
    if ! ssh -o ConnectTimeout=5 "$REMOTE" 'echo ok' &>/dev/null; then
        echo "[$(date)] Remote unreachable - VM may be gone. Final push of whatever we have..."
        if ! git diff --quiet "$RESULTS_DIR/" 2>/dev/null; then
            N=$(python3 -c "import json; print(len(json.load(open('$RESULTS_DIR/results.json'))))" 2>/dev/null)
            git add "$RESULTS_DIR/" 2>/dev/null
            git commit -m "data: emergency sync benchmark results ($N results) - VM lost" 2>/dev/null
            git push 2>/dev/null
        fi
        echo "[$(date)] Exiting."
        break
    fi

    # Copy results + log
    scp -q "$REMOTE:$REMOTE_REPO/$RESULTS_DIR/results.json" "$REPO_DIR/$RESULTS_DIR/results.json" 2>/dev/null
    scp -q "$REMOTE:$REMOTE_REPO/$RESULTS_DIR/rerun_gpu_opt.log" "$REPO_DIR/$RESULTS_DIR/rerun_gpu_opt.log" 2>/dev/null

    # Check if anything changed
    if git diff --quiet "$RESULTS_DIR/"; then
        echo "[$(date)] No new results"
    else
        N=$(python3 -c "import json; print(len(json.load(open('$RESULTS_DIR/results.json'))))" 2>/dev/null)
        echo "[$(date)] New results ($N total), pushing to GitHub..."
        git add "$RESULTS_DIR/results.json" "$RESULTS_DIR/rerun_gpu_opt.log" 2>/dev/null
        git commit -m "data: auto-sync benchmark results ($N results)" 2>/dev/null
        git push 2>/dev/null
    fi

    # Check if benchmark is still running
    RUNNING=$(ssh -o ConnectTimeout=5 "$REMOTE" 'ps aux | grep rerun_gpu_opt | grep -v grep | wc -l' 2>/dev/null)
    if [ "$RUNNING" = "0" ]; then
        echo "[$(date)] Benchmark finished! Final sync..."
        scp -q "$REMOTE:$REMOTE_REPO/$RESULTS_DIR/results.json" "$REPO_DIR/$RESULTS_DIR/results.json" 2>/dev/null
        scp -q "$REMOTE:$REMOTE_REPO/$RESULTS_DIR/rerun_gpu_opt.log" "$REPO_DIR/$RESULTS_DIR/rerun_gpu_opt.log" 2>/dev/null
        N=$(python3 -c "import json; print(len(json.load(open('$RESULTS_DIR/results.json'))))" 2>/dev/null)
        git add "$RESULTS_DIR/results.json" "$RESULTS_DIR/rerun_gpu_opt.log" 2>/dev/null
        git commit -m "data: final benchmark results ($N results, gpu-opt DMRG1 warmup+polish)" 2>/dev/null
        git push 2>/dev/null
        echo "[$(date)] Final push done. Exiting."
        break
    fi

    sleep "$INTERVAL"
done
