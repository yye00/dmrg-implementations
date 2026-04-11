#!/usr/bin/env bash
# R3 benchmark ladder — ultra-trim (18 configs) × 3 repeats × 5 impls × 3 models.
#
# Execution order: all optimized impls first (so we have new numbers ASAP),
# then the baseline -base variants. pdmrg-gpu-base is intentionally skipped
# because pdmrg optimized alone is ~11.7 h/pass at full grid; a -base pass
# would blow the wall-time budget without changing the paper narrative.
#
# After each (impl, model) block we commit + push so partial progress is
# always on the remote.
#
# Designed to run inside a tmux session on the MI300X host.

set -u

REPO="${REPO:-$HOME/dmrg-implementations}"
cd "$REPO" || { echo "cannot cd $REPO" >&2; exit 1; }

PY=python3
RUN="benchmarks/run_mi300x_challenge.py"
OUT_DIR="benchmarks/paper_results/mi300x/challenge"
LOG_DIR="benchmarks/paper_results/mi300x/challenge/_logs"
DOC="docs/followups/r3_benchmark_run.md"

mkdir -p "$LOG_DIR"

REPEATS="${REPEATS:-3}"

# Order chosen so fastest wins land first:
#   dmrg-gpu (optimized)     ~28 min / 18 cfg
#   dmrg2-gpu (optimized)    ~77 min
#   pdmrg-gpu (optimized)    ~70 min
#   dmrg-gpu-base            ~85 min
#   dmrg2-gpu-base           ~230 min
IMPLS=(
    dmrg-gpu
    dmrg2-gpu
    pdmrg-gpu
    dmrg-gpu-base
    dmrg2-gpu-base
)
MODELS=(heisenberg josephson tfim)

append_doc() {
    # $1 = markdown line(s) to append
    printf '%s\n' "$1" >> "$DOC"
}

commit_push() {
    # $1 = commit message
    local msg="$1"
    git add "$OUT_DIR" "$DOC" 2>/dev/null || true
    if git diff --cached --quiet; then
        echo "[ladder] nothing to commit"
        return 0
    fi
    git commit -m "$msg" >/dev/null 2>&1 || { echo "[ladder] commit failed"; return 1; }
    # Retry push twice in case of transient network
    for attempt in 1 2 3; do
        if git push origin HEAD >/dev/null 2>&1; then
            echo "[ladder] pushed: $msg"
            return 0
        fi
        echo "[ladder] push attempt $attempt failed, retrying..."
        sleep 5
    done
    echo "[ladder] PUSH FAILED after 3 tries: $msg"
    return 1
}

START_TS=$(date -u +%Y%m%dT%H%M%SZ)
echo "[ladder] start $START_TS"
append_doc ""
append_doc "## Run started $START_TS"
append_doc ""
append_doc "- Grid: ULTRA_TRIM (18 configs/model)"
append_doc "- Repeats: $REPEATS"
append_doc "- Impls: ${IMPLS[*]}"
append_doc "- Models: ${MODELS[*]}"
append_doc ""
append_doc "| impl | model | start | end | walltime | status |"
append_doc "|------|-------|-------|-----|----------|--------|"
commit_push "bench(r3): ladder start $START_TS" || true

TOTAL_BLOCKS=$(( ${#IMPLS[@]} * ${#MODELS[@]} ))
BLOCK_N=0
LADDER_START=$(date +%s)

for impl in "${IMPLS[@]}"; do
    for model in "${MODELS[@]}"; do
        BLOCK_N=$(( BLOCK_N + 1 ))
        BLOCK_TS=$(date -u +%Y%m%dT%H%M%SZ)
        TAG="r3_${BLOCK_TS}"
        LOG_FILE="$LOG_DIR/${impl}_${model}_${BLOCK_TS}.log"

        echo "================================================================"
        echo "[ladder] block $BLOCK_N/$TOTAL_BLOCKS  impl=$impl  model=$model"
        echo "[ladder] log: $LOG_FILE"
        echo "================================================================"

        BLOCK_START=$(date +%s)
        $PY "$RUN" --trim --repeats "$REPEATS" \
                    --impl "$impl" --model "$model" \
                    --tag "$TAG" 2>&1 | tee "$LOG_FILE"
        rc=${PIPESTATUS[0]}
        BLOCK_END=$(date +%s)
        BLOCK_DUR=$(( BLOCK_END - BLOCK_START ))
        BLOCK_DUR_HUMAN=$(printf '%dm%02ds' $((BLOCK_DUR/60)) $((BLOCK_DUR%60)))

        if [ "$rc" -eq 0 ]; then
            status="OK"
        else
            status="FAIL(rc=$rc)"
        fi

        append_doc "| $impl | $model | $BLOCK_TS | $(date -u +%H:%M:%SZ) | ${BLOCK_DUR_HUMAN} | $status |"
        commit_push "bench(r3): ${impl}/${model} ${status} (${BLOCK_DUR_HUMAN})" || true

        echo "[ladder] block $BLOCK_N done in $BLOCK_DUR_HUMAN  (status=$status)"
    done
done

LADDER_END=$(date +%s)
TOTAL_DUR=$(( LADDER_END - LADDER_START ))
TOTAL_HUMAN=$(printf '%dh%02dm' $((TOTAL_DUR/3600)) $(((TOTAL_DUR%3600)/60)))

echo "[ladder] ALL DONE in $TOTAL_HUMAN"
append_doc ""
append_doc "**Ladder total: $TOTAL_HUMAN**"
commit_push "bench(r3): ladder complete (${TOTAL_HUMAN})" || true
