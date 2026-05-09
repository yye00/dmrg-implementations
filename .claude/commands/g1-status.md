---
description: Read-only G1 GPU host status — phase, progress, FAILs, watchdog/backup health. Use this when you just want to know what's happening without acting.
allowed-tools: Bash, Read
---

Read-only status query against the G1 GPU host (read `~/.gpu-host` for the
SSH target). Do NOT take any action; just print:

1. **Bootstrap state**: `tail -10 /tmp/g1_bootstrap.log` on remote.
2. **Smoke**: configs done / 162, FAILs, last variant.
3. **Full**: configs done / total, FAILs, last variant.
4. **Watchdog**: alive? last 3 KILL entries from `/tmp/hang_watcher.log`.
5. **Backup**: alive? last 3 PUSH entries from `/tmp/result_backup.log`.
6. **GPU**: rocm-smi --showuse one-liner.
7. **Last 3 commits on origin/main** (so the user can see auto-backup is firing).

End with the inferred phase tag (bootstrap-running / smoke-in-progress /
smoke-clean / smoke-broken / full-in-progress / full-done / idle / dead).
