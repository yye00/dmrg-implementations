#!/bin/bash
# Auto-kill hung benchmark binaries.
#
# Heuristic: a non-warmup benchmark binary running > 120s with GPU CU
# occupancy < 10% is hung — kill it so the launcher records FAIL and
# moves to the next config instead of burning the GPU window.
#
# Lessons from the 2026-05-08 dmrg2-gpu-opt χ≥128 hangs:
# - Use the simple regex /build/dmrg or /build/pdmrg as two separate
#   pgrep calls; the alternation regex `dmrg.*_gpu\|pdmrg.*_gpu` did
#   not match on the actual cmdline format and the watchdog never
#   fired. (See lessons memory `g1_watchdog_regex_bug`.)
# - Skip warmup invocations whose argv contains "4 4 2" (positional
#   L=4 chi=4 sweeps=2).
# - Fail open: if rocm-smi can't be parsed, assume hung and kill.

set -u

LOG="${LOG:-/tmp/hang_watcher.log}"
THRESHOLD_SEC="${THRESHOLD_SEC:-120}"
GPU_THRESHOLD="${GPU_THRESHOLD:-10}"
INTERVAL="${INTERVAL:-20}"

echo "$(date -Iseconds) watchdog start threshold=${THRESHOLD_SEC}s gpu<${GPU_THRESHOLD}%" >> "$LOG"

while true; do
    sleep "$INTERVAL"
    PIDS="$(pgrep -f /build/dmrg 2>/dev/null; pgrep -f /build/pdmrg 2>/dev/null)"
    [ -z "$PIDS" ] && continue
    for pid in $PIDS; do
        cmdline=$(cat /proc/$pid/cmdline 2>/dev/null | tr '\0' ' ')
        # Skip warmup
        if echo "$cmdline" | grep -qE " 4 4 2( |$)"; then continue; fi
        elapsed=$(ps -p $pid -o etimes= 2>/dev/null | tr -d ' ')
        [ -z "$elapsed" ] && continue
        if [ "$elapsed" -gt "$THRESHOLD_SEC" ]; then
            cu=$(rocm-smi --showuse 2>/dev/null | grep "GPU use" | awk -F: '{print $NF}' | tr -d ' ')
            # Fail open: empty/unparseable => assume idle
            cu="${cu:-0}"
            if [ "$cu" -lt "$GPU_THRESHOLD" ] 2>/dev/null; then
                echo "$(date -Iseconds) KILL pid=$pid elapsed=${elapsed}s gpu=${cu}%: $cmdline" >> "$LOG"
                kill -9 "$pid" 2>/dev/null
            fi
        fi
    done
done
