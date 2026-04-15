#!/usr/bin/env bash
# PDMRG-GPU performance study — warmup/polish sweep on challenge configs.
#
# Runs pdmrg-gpu with 4 (warmup, polish) variants on the ultra-trim grid,
# 3 repeats each. Every --warmup and --polish flag is explicit (CLAUDE.md rule).
#
# After each variant×model block, commits+pushes (or skips if not a git repo).

set -u

REPO="${REPO:-$HOME/dmrg-implementations}"
cd "$REPO" || { echo "cannot cd $REPO" >&2; exit 1; }

PY=python3
RUN="benchmarks/run_mi300x_challenge.py"
OUT_DIR="benchmarks/paper_results/mi300x/challenge"
LOG_DIR="benchmarks/paper_results/mi300x/challenge/_logs"
DOC="docs/followups/pdmrg_study_run.md"

mkdir -p "$LOG_DIR"

REPEATS="${REPEATS:-3}"
IMPL="pdmrg-gpu"
MODELS=(heisenberg josephson tfim)

# 4 study variants: (warmup, polish)
declare -a VARIANT_TAGS=( "w0p0" "w1p0" "w1p1" "w1p2" )
declare -a VARIANT_WARMUP=( 0 1 1 1 )
declare -a VARIANT_POLISH=( 0 0 1 2 )

append_doc() {
    printf '%s\n' "$1" >> "$DOC"
}

commit_push() {
    local msg="$1"
    if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        echo "[study] not a git repo — skipping commit/push: $msg"
        return 0
    fi
    git add "$OUT_DIR" "$DOC" 2>/dev/null || true
    if git diff --cached --quiet; then
        echo "[study] nothing to commit"
        return 0
    fi
    git commit -m "$msg" >/dev/null 2>&1 || { echo "[study] commit failed"; return 1; }
    for attempt in 1 2 3; do
        if git push origin HEAD >/dev/null 2>&1; then
            echo "[study] pushed: $msg"
            return 0
        fi
        echo "[study] push attempt $attempt failed, retrying..."
        sleep 5
    done
    echo "[study] PUSH FAILED: $msg"
    return 1
}

START_TS=$(date -u +%Y%m%dT%H%M%SZ)
echo "[study] start $START_TS"

cat > "$DOC" << HEADER
# PDMRG-GPU Performance Study — Warmup/Polish Sweep

**Date:** $START_TS
**Grid:** ULTRA_TRIM (18 configs/model × 3 models = 54 configs)
**Repeats:** $REPEATS
**Impl:** $IMPL

## Variants

| Tag | --warmup | --polish | Purpose |
|-----|----------|----------|---------|
| w0p0 | 0 | 0 | Pure PDMRG kernel — no serial overhead |
| w1p0 | 1 | 0 | Minimal warmup, no polish |
| w1p1 | 1 | 1 (1-site) | Light warmup + 1 single-site polish |
| w1p2 | 1 | 2 (1-site) | Light warmup + 2 single-site polish |

## Log

| variant | model | start | end | walltime | status |
|---------|-------|-------|-----|----------|--------|
HEADER

commit_push "bench(pdmrg-study): start $START_TS" || true

TOTAL_BLOCKS=$(( ${#VARIANT_TAGS[@]} * ${#MODELS[@]} ))
BLOCK_N=0
LADDER_START=$(date +%s)

for vi in "${!VARIANT_TAGS[@]}"; do
    TAG="${VARIANT_TAGS[$vi]}"
    WARMUP="${VARIANT_WARMUP[$vi]}"
    POLISH="${VARIANT_POLISH[$vi]}"

    for model in "${MODELS[@]}"; do
        BLOCK_N=$(( BLOCK_N + 1 ))
        BLOCK_TS=$(date -u +%Y%m%dT%H%M%SZ)
        LOG_FILE="$LOG_DIR/pdmrg-study_${TAG}_${model}_${BLOCK_TS}.log"

        echo "================================================================"
        echo "[study] block $BLOCK_N/$TOTAL_BLOCKS  variant=$TAG  model=$model"
        echo "[study]   --warmup $WARMUP --polish $POLISH"
        echo "[study]   log: $LOG_FILE"
        echo "================================================================"

        BLOCK_START=$(date +%s)
        $PY "$RUN" --trim --repeats "$REPEATS" \
                    --impl "$IMPL" --model "$model" \
                    --pdmrg-warmup "$WARMUP" --pdmrg-polish "$POLISH" \
                    --tag "pdmrg_${TAG}" 2>&1 | tee "$LOG_FILE"
        rc=${PIPESTATUS[0]}
        BLOCK_END=$(date +%s)
        BLOCK_DUR=$(( BLOCK_END - BLOCK_START ))
        BLOCK_DUR_HUMAN=$(printf '%dm%02ds' $((BLOCK_DUR/60)) $((BLOCK_DUR%60)))

        if [ "$rc" -eq 0 ]; then
            status="OK"
        else
            status="FAIL(rc=$rc)"
        fi

        append_doc "| $TAG | $model | $BLOCK_TS | $(date -u +%H:%M:%SZ) | ${BLOCK_DUR_HUMAN} | $status |"
        commit_push "bench(pdmrg-study): ${TAG}/${model} ${status} (${BLOCK_DUR_HUMAN})" || true

        echo "[study] block $BLOCK_N done in $BLOCK_DUR_HUMAN  (status=$status)"
    done
done

LADDER_END=$(date +%s)
TOTAL_DUR=$(( LADDER_END - LADDER_START ))
TOTAL_HUMAN=$(printf '%dh%02dm' $((TOTAL_DUR/3600)) $(((TOTAL_DUR%3600)/60)))

echo "[study] ALL DONE in $TOTAL_HUMAN"
append_doc ""
append_doc "**Study total: $TOTAL_HUMAN**"
commit_push "bench(pdmrg-study): complete (${TOTAL_HUMAN})" || true
