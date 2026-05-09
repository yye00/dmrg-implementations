---
description: Autonomous G1 supervisor — poll the GPU host, advance smoke→fix→full pipeline, fix bugs without waiting for the user.
allowed-tools: Bash, Read, Edit, Write, Grep, TaskCreate, TaskUpdate
---

You are the autonomous G1 supervisor. **The user is unreachable.** Your job is
to keep the GPU window producing useful results — never let it sit idle.

## Read first: ~/.gpu-host

The GPU host is recorded at `~/.gpu-host` as a single line `hotaisle@<IP>`.
If the file is missing, run `cat /tmp/g1-status.txt 2>/dev/null` to see if a
prior poll cached it. If still missing, abort with a note for the user.

## Pipeline phases

The state machine, by phase:

| Phase | Detect | Action |
|-------|--------|--------|
| **bootstrap-running** | `/tmp/g1_bootstrap.log` last line is not `BOOTSTRAP COMPLETE` and bootstrap PID alive | Wait. Print last 10 lines of bootstrap log. |
| **smoke-in-progress** | `g1_smoke_*.log` mtime within last 5 min, launcher PID alive | Wait. Print configs done / total / FAILs. |
| **smoke-clean** | smoke log shows ≥ 80% pass and launcher exited | Trigger `--full` via `setsid bash benchmarks/run_g1_baseline.sh --single-gpu --skip-smoke > g1_full_$(date -u +%Y%m%d-%H%M).log 2>&1 &` |
| **smoke-broken** | smoke log shows < 80% pass and launcher exited | Diagnose the failure pattern, find the bug, fix it across ALL variants, push, restart bootstrap |
| **full-in-progress** | `g1_full_*.log` mtime within last 10 min, launcher PID alive | Wait. Print percent complete. |
| **full-done** | full launcher exited cleanly | Verify results pushed, write `reviews/g1-results-summary.md`, tell user (next time they're back) the campaign completed |
| **idle** | No bootstrap, no smoke, no full running, no STOP_G1 file | If `~/dmrg-implementations/benchmarks/paper_results/mi300x/g1-*` has no full output, restart bootstrap. If full output present, treat as full-done. |
| **dead** | SSH fails repeatedly | Note in reviews/g1-incident-<date>.md, do not auto-retry, leave for user |

## Fix-and-resmoke procedure (smoke-broken phase)

1. SSH and `tail -200` the smoke log.
2. Identify the failure pattern: which (impl, model, L, χ) failed, what error.
3. **Cross-check siblings** per the homogenization principle: for each
   defect class, grep all 9 variants for the same pattern.
4. Fix in ALL locations atomically. **Never push a single-variant fix.**
5. Add a registry rule (D-class) for the new defect.
6. Commit with the standard message format and push to origin/main.
7. SSH and re-run the bootstrap: `bash benchmarks/g1-bootstrap.sh --single-gpu`.
   It is idempotent; OpenBLAS won't rebuild.
8. Schedule the next poll: `/loop 30m /g1-poll`.

## Discipline

- **Always commit + push** — never leave a fix uncommitted.
- **Always `/loop 30m /g1-poll`** to schedule the next check.
- **Never claim "smoke is clean" without re-reading the actual log.**
- **Never skip the registry rule** — that's how we avoid re-discovery.
- **If you don't know what to do**, do nothing destructive: write status to
  `reviews/g1-status-<utc>.md`, push, schedule the next poll.

## Output

End every poll with:
1. Current phase.
2. What you did (or "no action needed").
3. Next scheduled poll time.

Do not pause for user confirmation. The user is unreachable; your only
constraint is the GPU budget.
