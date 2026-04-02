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

**Last Updated**: 2026-03-10
**Purpose**: GPU-CPU parity fixes for gpu-rocm/pdmrg-gpu implementation

## Local Directory Layout (updated 2026-04-02)

After repository reorganization:
- **`cpu/`** -- CPU Python implementations: `pdmrg/`, `pdmrg-cotengra/`, `pdmrg-opt/`, `a2dmrg/`
- **`gpu-rocm/`** -- AMD MI300X GPU implementations: `dmrg-gpu/`, `dmrg-gpu-opt/`, `dmrg2-gpu/`, `dmrg2-gpu-opt/`, `pdmrg-gpu/`, `pdmrg-gpu-opt/`
- **`gpu-cuda/`** -- Planned NVIDIA H100 CUDA ports (empty scaffolding)
- **`benchmarks/results/{mi300x,h100}/`** -- Raw benchmark results tagged by architecture
- **`benchmarks/paper_results/{mi300x,h100}/`** -- Publication-grade results tagged by architecture
- **`reports/{mi300x,h100}/`** -- Timing report JSONs tagged by architecture
- **`docs/`** -- GPU development prompts and reference docs
