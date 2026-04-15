#!/usr/bin/env bash
# Local-side sync loop: every 5 min, rsync R3 benchmark results from the
# remote MI300X VM back to this local repo, then commit+push.
#
# The remote has no GitHub auth (fresh VM), so all git operations happen
# here on the local machine.

set -u

REPO="${REPO:-$HOME/clawd/work/dmrg-implementations}"
REMOTE="${REMOTE:-hotaisle@23.183.40.79}"
REMOTE_REPO="${REMOTE_REPO:-~/dmrg-implementations}"
INTERVAL="${INTERVAL:-300}"  # seconds

cd "$REPO" || { echo "cannot cd $REPO" >&2; exit 1; }

log() {
    printf '[sync %s] %s\n' "$(date -u +%H:%M:%SZ)" "$*"
}

pull_once() {
    # Pull results dir + progress doc from remote
    rsync -az --exclude '*.tmp' \
        "${REMOTE}:${REMOTE_REPO}/benchmarks/paper_results/mi300x/challenge/" \
        benchmarks/paper_results/mi300x/challenge/ 2>&1 | grep -v '^$' || true
    rsync -az \
        "${REMOTE}:${REMOTE_REPO}/docs/followups/r3_benchmark_run.md" \
        docs/followups/r3_benchmark_run.md 2>&1 | grep -v '^$' || true
    rsync -az \
        "${REMOTE}:${REMOTE_REPO}/docs/followups/pdmrg_study_run.md" \
        docs/followups/pdmrg_study_run.md 2>&1 | grep -v '^$' || true
}

commit_push_once() {
    git add benchmarks/paper_results/mi300x/challenge/ \
            docs/followups/r3_benchmark_run.md \
            docs/followups/pdmrg_study_run.md 2>/dev/null || true
    if git diff --cached --quiet; then
        log "no changes"
        return 0
    fi
    local msg="bench(r3): auto-sync $(date -u +%Y%m%dT%H%MZ)"
    if git commit -m "$msg" >/dev/null 2>&1; then
        if git push origin HEAD >/dev/null 2>&1; then
            log "pushed: $msg"
        else
            log "PUSH FAILED (will retry next cycle)"
        fi
    else
        log "commit failed"
    fi
}

log "starting sync loop (interval=${INTERVAL}s, remote=${REMOTE})"

while true; do
    pull_once
    commit_push_once
    sleep "$INTERVAL"
done
