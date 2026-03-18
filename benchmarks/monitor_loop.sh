#!/bin/bash
# Background monitor loop - runs every 10 minutes until done
# Survives session termination via nohup

SCRIPT_DIR="/home/captain/clawd/work/dmrg-implementations/benchmarks"
PIDFILE="/tmp/benchmark_monitor_loop.pid"

# Write PID
echo $$ > "$PIDFILE"

while true; do
    /bin/bash "$SCRIPT_DIR/monitor_and_cleanup.sh"
    
    # Check if cleanup marked as done
    if grep -q "done=true" /tmp/benchmark_monitor_state 2>/dev/null; then
        echo "[$(date)] Monitor loop: cleanup completed, exiting." >> "$SCRIPT_DIR/paper_results/monitor.log"
        rm -f "$PIDFILE"
        exit 0
    fi
    
    sleep 600  # 10 minutes
done
