"""
Benchmark provenance capture — git commit hash, branch, dirty flag,
host/GPU identification, binary metadata, environment variables.

Intended use: every benchmark JSON should embed a `provenance` dict at
the top level so results are reproducible to the exact code version and
runtime configuration. Used by run_mi300x_challenge.py,
run_h100_challenge.py, paper_benchmark.py, bench_dmrg_gpu_ablate.py.
"""

import hashlib
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _run(cmd, cwd=None, timeout=5):
    """Run a shell command, return stdout stripped. Empty string on any failure."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, cwd=cwd)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def git_info(repo_root=None):
    """Collect git commit / branch / dirty / last-commit info for a repo."""
    cwd = str(repo_root) if repo_root else None
    commit = _run(["git", "rev-parse", "HEAD"], cwd=cwd)
    short  = _run(["git", "rev-parse", "--short=12", "HEAD"], cwd=cwd)
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd)
    status = _run(["git", "status", "--porcelain"], cwd=cwd)
    dirty_files = [line[3:] for line in status.splitlines()] if status else []
    last_msg = _run(["git", "log", "-1", "--format=%s"], cwd=cwd)
    last_iso = _run(["git", "log", "-1", "--format=%cI"], cwd=cwd)
    try:
        ahead  = _run(["git", "rev-list", "--count", "@{u}..HEAD"], cwd=cwd) or "0"
        behind = _run(["git", "rev-list", "--count", "HEAD..@{u}"], cwd=cwd) or "0"
    except Exception:
        ahead, behind = "0", "0"

    return {
        "commit":          commit,
        "commit_short":    short,
        "branch":          branch,
        "dirty":           bool(dirty_files),
        "dirty_files":     dirty_files,
        "last_commit_msg": last_msg,
        "last_commit_iso": last_iso,
        "ahead_of_upstream":  int(ahead)  if ahead.isdigit()  else 0,
        "behind_upstream":    int(behind) if behind.isdigit() else 0,
    }


def host_info():
    """Hostname, OS, Python version, CPU."""
    info = {
        "hostname":     socket.gethostname(),
        "fqdn":         socket.getfqdn(),
        "platform":     platform.platform(),
        "python":       sys.version.split()[0],
        "cpu":          platform.processor() or platform.machine(),
    }
    return info


def gpu_info():
    """Detect GPU (AMD or NVIDIA). Returns dict with vendor/model/driver."""
    # Try ROCm first
    rocm_out = _run(["rocm-smi", "--showproductname", "--csv"])
    if rocm_out and "GPU" in rocm_out:
        lines = [l for l in rocm_out.splitlines() if l.strip() and not l.startswith("device")]
        return {
            "vendor":       "AMD",
            "rocm_smi":     rocm_out,
            "count":        max(1, len(lines) - 1),  # minus header
        }

    # Try NVIDIA
    nv_out = _run(["nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader"])
    if nv_out:
        lines = [l for l in nv_out.splitlines() if l.strip()]
        return {
            "vendor":       "NVIDIA",
            "nvidia_smi":   nv_out,
            "count":        len(lines),
        }

    return {"vendor": "unknown"}


def binary_info(path):
    """Capture binary path, mtime, size, and SHA-256 (for stale-detection)."""
    p = Path(path)
    if not p.exists():
        return {"path": str(path), "exists": False}
    st = p.stat()
    try:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 16), b""):
                h.update(chunk)
        sha = h.hexdigest()
    except Exception:
        sha = ""
    return {
        "path":    str(p.resolve()),
        "exists":  True,
        "size":    st.st_size,
        "mtime":   datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
        "sha256":  sha,
    }


def env_snapshot(prefixes=("DMRG_GPU_",), extra_keys=("OMP_NUM_THREADS",
                                                        "OPENBLAS_NUM_THREADS",
                                                        "MKL_NUM_THREADS",
                                                        "HIP_VISIBLE_DEVICES",
                                                        "CUDA_VISIBLE_DEVICES")):
    """Snapshot of env vars relevant to the benchmark. Captures every var whose
    name starts with any prefix (e.g. DMRG_GPU_OPT_*, DMRG_GPU_PROFILE) plus
    the explicit extras."""
    snap = {}
    for k, v in os.environ.items():
        if any(k.startswith(p) for p in prefixes):
            snap[k] = v
    for k in extra_keys:
        if k in os.environ:
            snap[k] = os.environ[k]
    return snap


def provenance_block(repo_root=None, script_argv=None):
    """Full provenance block suitable for embedding at the top level of
    benchmark JSON output. Call once per run.

    Args:
        repo_root: Path to git repo. If None, uses the containing repo of
                   the calling script.
        script_argv: List of command-line args the benchmark script was
                     invoked with (sys.argv[1:] typically).
    """
    if repo_root is None:
        # Walk up from this file until we find a .git directory
        here = Path(__file__).resolve()
        for ancestor in [here, *here.parents]:
            if (ancestor / ".git").exists():
                repo_root = ancestor
                break

    return {
        "generated_utc":   datetime.now(timezone.utc).isoformat(),
        "git":             git_info(repo_root),
        "host":            host_info(),
        "gpu":             gpu_info(),
        "env":             env_snapshot(),
        "script": {
            "path":      str(Path(sys.argv[0]).resolve()) if sys.argv else "",
            "argv":      script_argv if script_argv is not None else sys.argv[1:],
            "cwd":       os.getcwd(),
        },
    }
