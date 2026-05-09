#!/bin/bash
# Continuous result backup — runs forever in the GPU window.
# Stages and pushes any new/modified file under benchmarks/paper_results/,
# benchmarks/results/, reports/, reviews/, plus repo-root *.log files.
#
# Design lessons from the 2026-05-08 lost-VM incident:
# - 30s interval (was 300s; VM died in the 5-min gap with 0 commits).
# - Immediate first commit BEFORE the first sleep (so even a 60-second
#   VM lifetime produces at least one commit).
# - Heartbeat commit every 10 min even if no result files changed
#   (proves the loop is alive; helps detect VM zombies).
# - Pull-with-rebase before push to handle concurrent commits from
#   the orchestrator side.

set -u

REPO="${REPO:-$HOME/dmrg-implementations}"
LOG="${LOG:-/tmp/result_backup.log}"
INTERVAL="${INTERVAL:-30}"
HEARTBEAT_EVERY="${HEARTBEAT_EVERY:-600}"   # heartbeat commit every 10 min

cd "$REPO" || { echo "ERR: $REPO not found" >> "$LOG"; exit 1; }

# Make sure git can commit autonomously
git config user.email "hotaisle-mi300x@local" 2>/dev/null
git config user.name  "Hot Aisle MI300X"      2>/dev/null

echo "$(date -Iseconds) backup start interval=${INTERVAL}s heartbeat=${HEARTBEAT_EVERY}s repo=$REPO" >> "$LOG"

last_heartbeat=$(date +%s)

stage_and_push() {
    local msg="$1"
    git pull --rebase --autostash --quiet origin main 2>>"$LOG" || true

    # Mirror /tmp/g1_bootstrap.log into the repo so it gets backed up like
    # any other tracked artifact. Doing `git add /tmp/...` directly fails
    # with "is outside repository" and — worse — fails the WHOLE git add
    # invocation, leaving every other pathspec unstaged. That bug ran for
    # 1h25m on 2026-05-09 with 4 result JSONs un-staged the entire smoke.
    if [[ -f /tmp/g1_bootstrap.log ]]; then
        mkdir -p reports/.heartbeat
        cp /tmp/g1_bootstrap.log reports/.heartbeat/g1_bootstrap.log 2>/dev/null || true
    fi

    # Only stage paths that actually exist as repo-relative paths. Trailing
    # globs that match no files (e.g. g1_full_*.log before --full launches)
    # would otherwise fail the whole add too — guard each glob.
    local -a add_paths=(benchmarks/paper_results/ benchmarks/results/ reports/ reviews/)
    shopt -s nullglob
    local smoke_logs=(g1_smoke_*.log)
    local full_logs=(g1_full_*.log)
    shopt -u nullglob
    (( ${#smoke_logs[@]} )) && add_paths+=("${smoke_logs[@]}")
    (( ${#full_logs[@]} )) && add_paths+=("${full_logs[@]}")

    git add -A "${add_paths[@]}" 2>>"$LOG" || true

    if git diff --cached --quiet; then
        return 1
    fi
    if git commit -m "$msg" --quiet 2>>"$LOG"; then
        if git push origin main --quiet 2>>"$LOG"; then
            echo "$(date -Iseconds) PUSH: $msg" >> "$LOG"
            return 0
        fi
        # Push failed — local commit still preserves the data on this VM's
        # disk. Log so /g1-poll can scp the .git bundle as an emergency
        # rescue if push auth never gets configured.
        echo "$(date -Iseconds) PUSH FAILED (commit kept locally): $msg" >> "$LOG"
        return 2
    fi
    return 2
}

# Immediate first commit — do not wait for the first sleep window.
stage_and_push "data: auto-backup boot $(date -u +%Y-%m-%dT%H:%M:%SZ)" || \
    echo "$(date -Iseconds) initial cycle: nothing to stage yet" >> "$LOG"

while true; do
    sleep "$INTERVAL"
    cd "$REPO" || continue
    now=$(date +%s)
    msg="data: auto-backup $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if stage_and_push "$msg"; then
        last_heartbeat="$now"
    fi
    # Heartbeat: if nothing was committed in HEARTBEAT_EVERY seconds, force
    # an empty/timestamp commit so we see liveness in origin/main.
    if (( now - last_heartbeat > HEARTBEAT_EVERY )); then
        # Touch a heartbeat marker
        mkdir -p reports/.heartbeat
        echo "$(date -Iseconds) hostname=$(hostname) gpu=$(rocm-smi --showuse 2>/dev/null | grep 'GPU use' | awk -F: '{print $NF}' | tr -d ' ')%" \
            > reports/.heartbeat/g1.txt
        if stage_and_push "data: heartbeat $(date -u +%Y-%m-%dT%H:%M:%SZ)"; then
            last_heartbeat="$now"
        fi
    fi
done
