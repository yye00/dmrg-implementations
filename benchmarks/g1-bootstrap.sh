#!/bin/bash
# G1 baseline bootstrap — autonomous end-to-end orchestrator.
#
# Single command on a fresh Hot Aisle MI300X VM:
#   bash benchmarks/g1-bootstrap.sh [--single-gpu|--multi-gpu]
#
# What it does, in order, autonomously (will NOT pause for confirmation):
#   1. flock to prevent double-bootstrap
#   2. Pre-flight (cmake, gfortran, libopenblas-dev, git push auth, free disk)
#   3. Build OpenBLAS 0.3.28 from source if not cached
#   4. Kill any orphan benchmark processes from a prior run
#   5. Start result_backup.sh (30s interval, immediate first commit)
#   6. Start hang_watcher.sh (120s threshold, GPU<10% × 2)
#   7. Smoke run (max 4h)
#   8. EXIT 0. The /g1-poll Claude slash-command (run via /loop 30m)
#      decides what comes next: fix bugs and re-smoke, or launch --full.
#      The bootstrap intentionally does NOT auto-chain — paper numbers
#      come from a smoke-validated binary, not a partially-broken one.
#
#   With `--auto-full`: bootstrap chains smoke → --full if pass-rate
#   ≥ 80%, otherwise exits. Use only when no human/Claude oversight
#   is expected and you accept partial-variant numbers.
#
# Lessons from the 2026-05-08 / 09 lost-VM incident:
#   • backup loop must commit IMMEDIATELY, not after first sleep
#   • smoke must auto-chain to --full — VM idle = wasted GPU $$
#   • all decision logic lives here, NOT in human-in-the-loop confirms
#
# Stop conditions:
#   • create file `~/dmrg-implementations/STOP_G1` to halt next phase
#   • SIGINT/SIGTERM kills cleanly via trap

set -u
trap 'echo "[bootstrap] interrupted"; exit 130' INT TERM

REPO="${REPO:-$HOME/dmrg-implementations}"
LOG="${LOG:-/tmp/g1_bootstrap.log}"
LOCK="${LOCK:-/tmp/g1-bootstrap.lock}"
STOP_FILE="${STOP_FILE:-$REPO/STOP_G1}"

TARGET="single-gpu"
AUTO_FULL=0
for arg in "$@"; do
    case "$arg" in
        --single-gpu) TARGET="single-gpu" ;;
        --multi-gpu)  TARGET="multi-gpu" ;;
        --auto-full)  AUTO_FULL=1 ;;
    esac
done

log() { echo "$(date -Iseconds) [bootstrap] $*" | tee -a "$LOG" ; }

# ─── Single-instance guard ────────────────────────────────────────────────
exec 200>"$LOCK"
flock -n 200 || { log "ABORT: another bootstrap holds $LOCK"; exit 2; }

cd "$REPO" 2>/dev/null || {
    log "Repo $REPO missing — cloning"
    cd "$HOME"
    git clone https://github.com/yye00/dmrg-implementations.git || { log "ABORT: clone failed"; exit 3; }
    cd "$REPO"
}

log "==== G1 BOOTSTRAP START target=$TARGET HEAD=$(git rev-parse --short HEAD) ===="
git pull --quiet origin main || log "WARN: git pull failed; continuing on local HEAD"

# ─── 1. Pre-flight ────────────────────────────────────────────────────────
log "[1/9] Pre-flight checks"

# 1a. ROCm + GPU
command -v hipcc >/dev/null || { log "FAIL: hipcc not found"; exit 4; }
rocm-smi --showproductname 2>&1 | grep -q MI300X || log "WARN: not an MI300X"

# 1b. cmake / gfortran / libopenblas-dev
need_apt=()
command -v cmake     >/dev/null || need_apt+=(cmake)
command -v gfortran  >/dev/null || need_apt+=(gfortran)
[[ ! -f /usr/include/openblas-pthread/cblas.h && ! -f /usr/include/cblas-openblas.h && ! -f /usr/include/cblas.h ]] && need_apt+=(libopenblas-dev liblapack-dev)
if (( ${#need_apt[@]} > 0 )); then
    log "Installing apt packages: ${need_apt[*]}"
    sudo apt-get install -y "${need_apt[@]}" >>"$LOG" 2>&1 || { log "FAIL: apt-get install"; exit 5; }
fi

# 1c. OpenBLAS 0.3.28 (cached if present)
OPENBLAS_PATH="$HOME/openblas-0.3.28"
if [[ ! -f "$OPENBLAS_PATH/lib/libopenblas.so" ]]; then
    log "Building OpenBLAS 0.3.28 from source (one-time, ~10 min)"
    cd "$HOME"
    [[ ! -f OpenBLAS-0.3.28.tar.gz ]] && \
        wget -q https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.28/OpenBLAS-0.3.28.tar.gz
    rm -rf OpenBLAS-0.3.28 && tar -xzf OpenBLAS-0.3.28.tar.gz
    cd OpenBLAS-0.3.28
    if make -j16 USE_OPENMP=1 NUM_THREADS=104 DYNAMIC_ARCH=1 >>"$LOG" 2>&1 \
       && make PREFIX="$OPENBLAS_PATH" install >>"$LOG" 2>&1; then
        log "OpenBLAS 0.3.28 installed"
    else
        log "WARN: OpenBLAS source build failed; will fall back to system libopenblas-0.3.20 (SVD bug present)"
    fi
    cd "$REPO"
fi

# 1d. Git push auth (do a probe commit/push) — SOFT failure: results
#     accumulate locally even if push doesn't work; the backup loop will
#     keep retrying every 30s and pick up auth the moment it's configured.
git config user.email "hotaisle-mi300x@local" 2>/dev/null
git config user.name  "Hot Aisle MI300X"      2>/dev/null
mkdir -p reports/.heartbeat
echo "$(date -Iseconds) bootstrap-probe hostname=$(hostname)" > reports/.heartbeat/probe.txt
git add reports/.heartbeat/probe.txt
PUSH_OK=0
if git commit -m "data: bootstrap probe $(date -u +%Y-%m-%dT%H:%M:%SZ)" --quiet 2>>"$LOG"; then
    if git push --quiet origin main 2>>"$LOG"; then
        log "git push auth verified"
        PUSH_OK=1
    fi
fi
if [[ "$PUSH_OK" -ne 1 ]]; then
    log "WARN: git push auth not configured. Results will accumulate locally."
    log "WARN: To enable backup, set credentials and the backup loop will catch up."
    log "WARN: e.g.: git config credential.helper store && (echo URL with token) > ~/.git-credentials"
fi

# 1e. Disk space
free_gb=$(df -BG "$REPO" | awk 'NR==2 {gsub("G","",$4); print $4}')
[[ "$free_gb" -lt 10 ]] && { log "FAIL: free disk = ${free_gb} GB < 10 GB"; exit 7; }
log "Free disk: ${free_gb} GB"

# 1f. Defect registry must be clean at HEAD
log "Defect registry pre-flight"
if ! bash .claude/scripts/defect-registry.sh > /tmp/registry.log 2>&1; then
    log "WARN: registry returned non-zero (review $LOG and /tmp/registry.log)"
fi
hits=$(grep -E "^TOTAL HITS:" /tmp/registry.log | awk '{print $NF}')
log "Registry: TOTAL HITS = $hits"
[[ "$hits" != "0" ]] && { log "FAIL: registry has hits — aborting"; exit 8; }

# ─── 2. Kill orphans, free GPU ────────────────────────────────────────────
log "[2/9] Killing orphan processes from prior runs"
pkill -9 -f result_backup.sh    2>/dev/null
pkill -9 -f hang_watcher.sh     2>/dev/null
pkill -9 -f run_g1_baseline     2>/dev/null
pkill -9 -f run_mi300x_challenge 2>/dev/null
pkill -9 -f /build/dmrg          2>/dev/null
pkill -9 -f /build/pdmrg         2>/dev/null
sleep 2
log "GPU after cleanup: $(rocm-smi --showuse 2>&1 | grep 'GPU use' | awk -F: '{print $NF}' | tr -d ' ')%"

# ─── 3. Start result-backup loop FIRST ────────────────────────────────────
log "[3/9] Starting result-backup loop (30s interval, immediate first commit)"
chmod +x benchmarks/result_backup.sh benchmarks/hang_watcher.sh
INTERVAL=30 nohup setsid benchmarks/result_backup.sh </dev/null >/dev/null 2>&1 &
disown
sleep 3
pgrep -f benchmarks/result_backup.sh >/dev/null || { log "FAIL: backup loop didn't start"; exit 9; }
log "Backup loop alive (PID $(pgrep -f benchmarks/result_backup.sh | head -1))"

# ─── 4. Start hang watchdog ───────────────────────────────────────────────
log "[4/9] Starting hang watchdog (120s threshold, GPU<10%)"
THRESHOLD_SEC=120 GPU_THRESHOLD=10 INTERVAL=20 \
    nohup setsid benchmarks/hang_watcher.sh </dev/null >/dev/null 2>&1 &
disown
sleep 2
pgrep -f benchmarks/hang_watcher.sh >/dev/null || { log "FAIL: watchdog didn't start"; exit 10; }
log "Watchdog alive (PID $(pgrep -f benchmarks/hang_watcher.sh | head -1))"

# ─── 5. Smoke ─────────────────────────────────────────────────────────────
log "[5/9] Running smoke (max 4h, $TARGET)"
[[ -f "$STOP_FILE" ]] && { log "STOP_G1 file present — halting before smoke"; exit 0; }

SMOKE_LOG="g1_smoke_$(date -u +%Y%m%d-%H%M).log"
( bash benchmarks/run_g1_baseline.sh --$TARGET --smoke > "$SMOKE_LOG" 2>&1 ) &
SMOKE_PID=$!
log "Smoke PID=$SMOKE_PID log=$SMOKE_LOG"
SMOKE_DEADLINE=$(( $(date +%s) + 4*3600 ))
while kill -0 "$SMOKE_PID" 2>/dev/null; do
    sleep 60
    [[ -f "$STOP_FILE" ]] && { log "STOP_G1 — killing smoke"; kill -9 "$SMOKE_PID"; exit 0; }
    if [[ $(date +%s) -gt $SMOKE_DEADLINE ]]; then
        log "Smoke exceeded 4h — killing and proceeding to --full anyway"
        kill -9 "$SMOKE_PID"
        break
    fi
done
SMOKE_RC=$?

# ─── 6. Smoke pass-rate gate ──────────────────────────────────────────────
log "[6/9] Smoke gate"
total=$(grep -cE "^        rep 1/1: " "$SMOKE_LOG" 2>/dev/null || echo 0)
fails=$(grep -c "rep 1/1: FAIL" "$SMOKE_LOG" 2>/dev/null || echo 0)
if [[ "$total" -gt 0 ]]; then
    pass=$(( total - fails ))
    rate=$(( 100 * pass / total ))
else
    rate=0
fi
log "Smoke: $pass/$total = ${rate}% pass, $fails FAIL"

# Identify variants whose pass-rate is < 50% (skip in --full)
SKIP_LIST=""
for v in dmrg-gpu-base dmrg-gpu dmrg-gpu-opt dmrg2-gpu-base dmrg2-gpu dmrg2-gpu-opt pdmrg-gpu-base pdmrg-gpu pdmrg-gpu-opt; do
    v_total=$(grep -A 100 "  $v —" "$SMOKE_LOG" | grep -cE "^        rep 1/1: " 2>/dev/null | head -1)
    v_fails=$(grep -A 100 "  $v —" "$SMOKE_LOG" | grep -c "rep 1/1: FAIL" 2>/dev/null | head -1)
    if [[ "$v_total" -gt 0 ]] && [[ $((v_fails * 2)) -gt $v_total ]]; then
        SKIP_LIST="${SKIP_LIST}${SKIP_LIST:+,}$v"
        log "Variant $v failed >50% in smoke — adding to --full skip-list"
    fi
done

if [[ "$rate" -lt 80 ]]; then
    log "Smoke pass-rate $rate% < 80% — exiting; /g1-poll will diagnose and fix"
    exit 0
fi

# Default: stop here. /g1-poll (via /loop) reads SMOKE_LOG and triggers --full.
if [[ "$AUTO_FULL" -ne 1 ]]; then
    log "Smoke clean (${rate}%). Exiting — invoke /g1-poll or /g1-launch-full to chain"
    exit 0
fi

# ─── 7. --full (only with --auto-full) ────────────────────────────────────
log "[7/9] Auto-chaining to --full (skip=${SKIP_LIST:-none})"
[[ -f "$STOP_FILE" ]] && { log "STOP_G1 — halting before --full"; exit 0; }

FULL_LOG="g1_full_$(date -u +%Y%m%d-%H%M).log"
SKIP_ENV=""
[[ -n "$SKIP_LIST" ]] && SKIP_ENV="VARIANT_SKIP=$SKIP_LIST"
( env $SKIP_ENV bash benchmarks/run_g1_baseline.sh --$TARGET --skip-smoke > "$FULL_LOG" 2>&1 ) &
FULL_PID=$!
log "--full PID=$FULL_PID log=$FULL_LOG"
while kill -0 "$FULL_PID" 2>/dev/null; do
    sleep 120
    [[ -f "$STOP_FILE" ]] && { log "STOP_G1 — killing --full"; kill -9 "$FULL_PID"; exit 0; }
done
log "--full done"

# ─── 8. Stretch (until window closes or STOP) ─────────────────────────────
log "[8/9] Stretch sweep — REPEATS=20 on configs that ran < 30s in --full"
# Best-effort: re-run --full at REPEATS=20 (the harness handles dedup
# via --tag).  If the user wanted a different stretch axis, they can
# touch STOP_G1 to halt and inspect.
[[ -f "$STOP_FILE" ]] && { log "STOP_G1 — skipping stretch"; exit 0; }
STRETCH_LOG="g1_stretch_$(date -u +%Y%m%d-%H%M).log"
( REPEATS=20 env $SKIP_ENV bash benchmarks/run_g1_baseline.sh --$TARGET --skip-smoke > "$STRETCH_LOG" 2>&1 ) &
STRETCH_PID=$!
log "Stretch PID=$STRETCH_PID log=$STRETCH_LOG"
while kill -0 "$STRETCH_PID" 2>/dev/null; do
    sleep 300
    [[ -f "$STOP_FILE" ]] && { log "STOP_G1 — killing stretch"; kill -9 "$STRETCH_PID"; break; }
done

log "[9/9] G1 BOOTSTRAP COMPLETE"
