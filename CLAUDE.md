# Claude Agent Information for DMRG Implementations Project

## Remote MI300X GPU Host Access

**CRITICAL**: This project uses a remote AMD MI300X GPU system for GPU development work.

### Access Methods

#### Method 1: tmux Session (Recommended)
```bash
# Attach to existing persistent session
tmux attach-session -t test_remote
```

**Session Info**:
- Session name: `test_remote`
- Created: Wed Mar 4 09:41:56 2026
- Always running (persistent connection to remote)

#### Method 2: Direct SSH (Passwordless)
```bash
# Direct SSH connection (no password required)
ssh hotaisle@23.183.40.87
```

**Remote Host Details**:
- Hostname: `enc1-gpuvm014`
- IP: `23.183.40.82`
- User: `hotaisle`
- GPU: AMD Instinct MI300X (gfx942)
- ROCm: Version 7.2+

### Remote Repository Location

```bash
/home/hotaisle/dmrg-implementations
```

**Current state** (as of 2026-03-10):
- ✅ Repository cloned
- ✅ Latest commit: `b5ad273` (GPU parity fix documentation)
- ✅ GitHub token configured
- ✅ Git user configured (Claude Opus 4.6)
- ✅ GPU_PARITY_FIX_PROMPT.md present (836 lines)
- ✅ GPU_FIX_QUICK_REFERENCE.md present (386 lines)

### Quick Commands

```bash
# Attach to remote session
tmux attach -t test_remote

# Send commands to remote session (from local)
tmux send-keys -t test_remote 'cd ~/dmrg-implementations/gpu-rocm/pdmrg-gpu' Enter

# Capture output from remote session
tmux capture-pane -t test_remote -p | tail -20

# Direct SSH
ssh hotaisle@23.183.40.87
```

### GPU Verification

```bash
# Check GPU availability (on remote)
rocm-smi --showproductname

# Expected output:
# GPU[0]: Card Series: AMD Instinct MI300X VF
# GPU[0]: GFX Version: gfx942
```

### Important Notes

1. **Always use test_remote tmux session** for persistent work
2. **SSH connection is passwordless** - no password prompt
3. **GitHub token is configured** on remote for commits/pushes
4. **Working directory**: `/home/hotaisle/dmrg-implementations`
5. **GPU code location**: `/home/hotaisle/dmrg-implementations/gpu-rocm/pdmrg-gpu`

---

## Local Development

**Local machine**: captain@localhost
**Local repo**: `/home/captain/clawd/work/dmrg-implementations`
**Branch**: `main`

### Workflow

1. **Local work**: Edit, commit, push from local machine
2. **Remote work**: Pull changes on remote, compile GPU code, test
3. **Remote commits**: Can commit/push from remote (GitHub token configured)

---

## PDMRG Rules (ALL variants: pdmrg, pdmrg-gpu, pdmrg-gpu-opt, pdmrg-multigpu)

**MANDATORY — these rules apply to ALL pdmrg variants in ALL contexts (benchmarks, tests, development):**

1. **Warmup sweeps MUST be single-site.** NEVER two-site warmup. The functions
   `sweep_LR_full_1site()` / `sweep_RL_full_1site()` are correct. NEVER call
   `sweep_LR_full()` / `sweep_RL_full()` for warmup.
2. **Polish sweeps MUST be single-site.** NEVER two-site polish. Same rule —
   use `_1site` variants only.
3. **Warmup and polish sweep counts MUST NOT exceed 2.** `n_warmup <= 2`,
   `n_polish <= 2`. The defaults in code and benchmark scripts must reflect this.
4. **Zero warmup (`n_warmup=0`) and zero polish (`n_polish=0`) MUST be
   supported** as valid configurations.
5. **Benchmarks use challenge-sized problems ONLY.** The ULTRA_TRIM or
   CHALLENGE_SIZES grids in `run_mi300x_challenge.py`. Smaller problems
   (L=4, L=8, etc.) are for smoke-testing / correctness verification during
   development, NEVER for published performance numbers.
6. **When running benchmarks, ALWAYS explicitly pass `--warmup N` and
   `--polish N`** to the binary. NEVER rely on compiled-in defaults — defaults
   drift and cause wasted benchmark runs.

---

**Last Updated**: 2026-04-15
**Purpose**: DMRG GPU implementations + PDMRG performance study

## Local Directory Layout (updated 2026-04-02)

After repository reorganization:
- **`cpu/`** -- CPU Python implementations: `pdmrg/`, `pdmrg-cotengra/`, `pdmrg-opt/`, `a2dmrg/`
- **`gpu-rocm/`** -- AMD MI300X GPU implementations: `dmrg-gpu/`, `dmrg-gpu-opt/`, `dmrg2-gpu/`, `dmrg2-gpu-opt/`, `pdmrg-gpu/`, `pdmrg-gpu-opt/`
- **`gpu-cuda/`** -- Planned NVIDIA H100 CUDA ports (empty scaffolding)
- **`benchmarks/results/{mi300x,h100}/`** -- Raw benchmark results tagged by architecture
- **`benchmarks/paper_results/{mi300x,h100}/`** -- Publication-grade results tagged by architecture
- **`reports/{mi300x,h100}/`** -- Timing report JSONs tagged by architecture
- **`docs/`** -- GPU development prompts and reference docs

---

## Path B workflow (paper revision, Apr-May 2026)

When working on Path B defects (paper rewrite + MI300X rebench campaign):

1. **Facts**: cite `docs/PATH_B_GROUND_TRUTH.md` -- the locked inventory
   produced by the I-1/I-2/I-3 audit, pinned to commit `6f45533`. Do NOT
   re-audit code. If you find anything contradicting it, STOP and post a
   comment on the "Path B execution tracker" issue.

2. **Roles**: all work happens via the subagents in `.claude/agents/`. The
   top-level session does not implement, review, or fix directly -- it
   dispatches.
   - `orchestrator` -- one supervisor tick: inspects PR state, dispatches the
     next role for one cluster, updates the tracking issue.
   - `implementer` -- takes a cluster brief, opens PR on
     `claude/path-b-<cluster-id>`, labels `needs-review`. Respects bail-out.
   - `reviewer` -- audits PR vs brief + ground truth, labels `review-clean` or
     `review-blocked`. Checklist-driven, not taste-driven.
   - `fixer` -- addresses reviewer comments in one follow-up commit, loops at
     most 3 times then escalates with `human-review-required`.

3. **Slash commands**:
   - `/path-b-tick` -- one orchestrator tick.
   - `/path-b-status` -- read-only status dump from the tracking issue.
   - `/review-pr <n>` -- fire reviewer on PR `<n>`.
   - `/fix-pr <n>` -- fire fixer on PR `<n>`.

4. **Unattended operation**: compose with `/loop 30m /path-b-tick` for a
   heartbeat-driven supervisor. PR review comments (agent or human) both
   route through `review-blocked` -> fixer; the two feedback channels are
   identical from the fixer's perspective.

5. **Tracking**: the "Path B execution tracker" issue on
   `yye00/dmrg-implementations` holds cluster state. One row per cluster,
   status in {not_started, implementer_running, needs-review, review-clean,
   review-blocked, waiting_gpu, human-review-required, merged}.

6. **GPU authorization**: all MI300X-bound commands (`ssh hotaisle@...`,
   `tmux send-keys -t test_remote ...`, `tmux attach -t test_remote`) are
   denied by the `.claude/hooks/gpu_auth_guard.sh` PreToolUse hook unless
   `PATH_B_GPU_AUTH=1` is exported in the shell that launched Claude Code.
   Export only when a Hot Aisle window has been booked.

7. **Branch convention**: Path B PRs live on `claude/path-b-<cluster-id>` and
   merge into `main`. No long-lived integration branch. Do not push to any
   existing feature branch (e.g., `claude/dmrg-gpu-concurrency-*`) -- those
   are obsolete pre-fork snapshots.

8. **Bail-out clause (all agents)**: if a brief contradicts ground truth, a
   named file/line no longer exists, or a reviewer comment asks for
   scope-creep or number-changes without statistical evidence -- STOP and
   escalate. Do NOT silently re-interpret. This is the single most
   important rule.
