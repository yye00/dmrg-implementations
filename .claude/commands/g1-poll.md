---
description: Autonomous G1 supervisor — poll the GPU host, advance smoke→fix→full pipeline, fix bugs without waiting for the user.
allowed-tools: Bash, Read, Edit, Write, Grep, TaskCreate, TaskUpdate
---

You are the autonomous G1 supervisor. **The user is unreachable.** Your job is
to keep the GPU window producing useful results — never let it sit idle.

## Read first: ~/.gpu-host

The GPU host is recorded at `~/.gpu-host` as a single line `hotaisle@<IP>`.
If the file is missing, run `cat /tmp/g1-status.txt 2>/dev/null` to see if a
prior poll cached it. If still missing, try the last-known IP from
`CLAUDE.md` (the "Remote MI300X Access" section). Only abort if that also
fails to SSH.

## MANDATORY rescue step (every poll, before anything else)

The VM's on-VM `result_backup.sh` commits locally but cannot push when
GitHub auth is unconfigured (the typical fresh-VM state). Therefore the
**ONLY durable copy of in-progress results is on the local machine**, and
local is the only host that can push to origin.

**Every /g1-poll fire MUST run the rescue BEFORE inspecting state:**

```bash
LOCAL_REPO=/home/captain/clawd/work/dmrg-implementations
REMOTE=$(cat ~/.gpu-host)

# 1. rsync result artifacts off the VM. -u means "skip files that are
#    newer on the destination" so we never overwrite a committed local
#    copy with a stale remote one. --partial keeps partial transfers
#    if the SSH drops mid-run.
rsync -avzu --partial --timeout=60 \
    "${REMOTE}":dmrg-implementations/benchmarks/paper_results/mi300x/g1-*/ \
    "${LOCAL_REPO}/benchmarks/paper_results/mi300x/" 2>&1 | tail -5

# 2. Pull launcher logs (gitignored locally — keep outside the repo
#    or they'll be ignored on add).
mkdir -p "${LOCAL_REPO}/reviews/g1-vm-logs"
rsync -avzu --partial --timeout=60 \
    "${REMOTE}":dmrg-implementations/g1_smoke_*.log \
    "${REMOTE}":dmrg-implementations/g1_full_*.log \
    "${LOCAL_REPO}/reviews/g1-vm-logs/" 2>&1 | tail -3

# 3. Pull diagnostic logs from /tmp (bootstrap, watchdog, backup).
rsync -avzu --partial --timeout=60 \
    "${REMOTE}":/tmp/g1_bootstrap.log \
    "${REMOTE}":/tmp/hang_watcher.log \
    "${REMOTE}":/tmp/result_backup.log \
    "${LOCAL_REPO}/reviews/g1-vm-logs/" 2>&1 | tail -3

# 4. Commit + push from local. The result JSONs are not gitignored;
#    .log files generally are, so we hand-include the rescued logs.
cd "${LOCAL_REPO}"
git add -A benchmarks/paper_results/mi300x/
git add -f reviews/g1-vm-logs/  # gitignored, force-add
if ! git diff --cached --quiet; then
    git commit -m "g1-rescue: $(date -u +%Y-%m-%dT%H:%M:%SZ) /g1-poll periodic backup" \
        --quiet
    git push origin main --quiet 2>&1 | tail -3
fi
```

This step runs every 30 min regardless of pipeline phase. If the VM dies
between fires, at most 30 minutes of paper results are lost — and we keep
the launcher/watchdog/backup logs for forensic recovery.

**Do NOT skip this step** even if smoke is "obviously running fine" — that
exact assumption cost us the 2026-05-08 lost-VM incident.

After the rescue completes, proceed with the pipeline-phase state machine
below.

## Pipeline phases

The state machine, by phase:

| Phase | Detect | Action |
|-------|--------|--------|
| **bootstrap-running** | `/tmp/g1_bootstrap.log` last line is not `BOOTSTRAP COMPLETE` and bootstrap PID alive | Wait. Print last 10 lines of bootstrap log. |
| **smoke-in-progress** | `g1_smoke_*.log` mtime within last 5 min, launcher PID alive | Wait. Print configs done / total / FAILs. |
| **smoke-clean** | smoke log shows ≥ 80% pass AND smoke launcher exited AND **no `g1_full_*.log` exists yet** | Trigger `--full` with χ=512 skipped (paper budget) AND dmrg2-gpu-opt skipped (D-G1-1 hang, see `reviews/g1-deferred-bugs-20260509.md`): `CHI_SKIP=512 VARIANT_SKIP=dmrg2-gpu-opt setsid bash benchmarks/run_g1_baseline.sh --single-gpu --skip-smoke > g1_full_$(date -u +%Y%m%d-%H%M).log 2>&1 &`. The `g1_full_*.log` existence check guards against re-firing if a prior poll already launched it but the launcher hasn't started writing yet. |
| **smoke-broken** | smoke log shows < 80% pass and launcher exited | Diagnose the failure pattern, find the bug, fix it across ALL variants, push, restart bootstrap |
| **full-in-progress** | `g1_full_*.log` mtime within last 10 min, launcher PID alive | Wait. Print percent complete. |
| **full-done** (χ≤256 main pass) | full launcher exited cleanly AND no `g1_chi512_*.log` exists yet | Trigger χ=512 mop-up at REPEATS=1, also skipping the deferred dmrg2-gpu-opt: `CHI_ONLY=512 VARIANT_SKIP=dmrg2-gpu-opt REPEATS=1 setsid bash benchmarks/run_g1_baseline.sh --single-gpu --skip-smoke > g1_chi512_$(date -u +%Y%m%d-%H%M).log 2>&1 &`. User explicitly opted into χ=512 single-rep coverage as a follow-on (2026-05-09). Heisenberg/TFIM only — josephson has no χ=512 row. |
| **chi512-in-progress** | `g1_chi512_*.log` mtime within last 10 min, launcher PID alive | Wait. Print percent complete. |
| **chi512-done** | chi512 launcher exited cleanly | Write `reviews/g1-results-summary.md`, declare campaign complete. |
| **idle** | No bootstrap, no smoke, no full, no chi512 running, no STOP_G1 file | If no full output present, restart bootstrap. If full+chi512 both present, treat as chi512-done. If only full present and chi512 not yet run, kick off the mop-up. |
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
8. The cron-scheduled poll fires every 30 min — no need to re-schedule.

## Escalation: fallback to unaffected variants

After **two unsuccessful fix attempts** for the same failing variant
(i.e., smoke still shows that variant FAIL after two re-pushes), do
NOT keep blocking the GPU window:

1. Mark the variant as **deferred** in `reviews/g1-deferred-bugs-<utc>.md`
   with the failure pattern, attempted fixes, and reproduction config.
2. Run `--full` with that variant excluded:
   `VARIANT_SKIP=<variant1>,<variant2> bash benchmarks/run_g1_baseline.sh --single-gpu --skip-smoke`
3. The remaining variants still produce paper-grade numbers — partial
   data is far better than no data when the GPU clock is ticking.
4. Commit the deferred-bugs file. The user picks up the failures next
   time they're online.

**The GPU must always be doing useful work.** Idle GPU = wasted budget.
If a fix is genuinely beyond your reach, run unaffected configurations.

## Discipline

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
