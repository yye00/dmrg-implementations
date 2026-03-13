"""
Runner for GPU DMRG implementations: dmrg-gpu, dmrg2-gpu, pdmrg-gpu, pdmrg2-gpu.

Invokes compiled C++/HIP executables via subprocess.
"""

import os
import re
import subprocess

from benchmarks.lib.hardware import get_repo_root


def _parse_gpu_output(stdout):
    """Parse energy and timing from GPU executable stdout."""
    energy = None
    wall_time = None

    for line in stdout.split("\n"):
        # Match "Final energy: -1.234567" or "Ground state energy: -1.234567"
        m = re.search(r"(?:Final|Ground state)\s+energy:\s+([-\d.eE+]+)", line)
        if m:
            energy = float(m.group(1))

        # Match "Total wall time: 1.234 s" or "Wall time: 1.234s"
        m = re.search(r"(?:Total\s+)?[Ww]all\s+time:\s+([-\d.eE+]+)", line)
        if m:
            wall_time = float(m.group(1))

    return energy, wall_time


def run(model, L, chi, max_sweeps=30, executable=None, np_count=None,
        n_max=2, **kwargs):
    """Run a GPU DMRG executable.

    Args:
        model: 'heisenberg' or 'josephson'
        L, chi, max_sweeps: standard DMRG parameters
        executable: path to compiled binary (relative to repo root)
        np_count: number of HIP streams (for parallel GPU impls)
        n_max: charge truncation for Josephson

    Returns:
        dict with keys: energy, time, success
    """
    if executable is None:
        raise ValueError("GPU runner requires 'executable' parameter")

    root = get_repo_root()
    exe_path = os.path.join(root, executable)

    if not os.path.exists(exe_path):
        return {
            "energy": None, "time": None, "success": False,
            "error": f"Executable not found: {exe_path}",
        }

    # Build command
    cmd = [exe_path, str(L), str(chi), str(max_sweeps)]

    # Model-specific flags
    if model == "josephson":
        cmd.append("--josephson")

    # Parallel GPU implementations use --segments for stream count
    if np_count is not None and np_count > 1:
        # Check if this is a pdmrg-style executable (uses --segments)
        if "pdmrg" in executable:
            cmd.extend(["--segments", str(np_count)])
            cmd.extend(["--warmup", "3"])
            cmd.extend(["--local-sweeps", "2"])

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
        )

        if result.returncode != 0:
            return {
                "energy": None, "time": None, "success": False,
                "error": result.stderr[-500:] if result.stderr else f"Exit code {result.returncode}",
            }

        energy, wall_time = _parse_gpu_output(result.stdout)

        if energy is None:
            return {
                "energy": None, "time": None, "success": False,
                "error": f"Could not parse energy from output: {result.stdout[-200:]}",
            }

        return {
            "energy": energy,
            "time": wall_time,
            "success": True,
        }

    except subprocess.TimeoutExpired:
        return {"energy": None, "time": None, "success": False, "error": "Timeout (600s)"}
    except FileNotFoundError:
        return {"energy": None, "time": None, "success": False, "error": f"Cannot execute: {exe_path}"}
