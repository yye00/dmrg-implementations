"""
Runner for MPI-based parallel DMRG implementations: pdmrg, pdmrg2, pdmrg-cotengra, a2dmrg.

Launches via mpirun subprocess with a temporary worker script.
"""

import json
import os
import subprocess
import sys
import tempfile
import time

from benchmarks.lib.hardware import get_repo_root, get_venv_python, get_mpi_env


def _make_pdmrg_script(package, entry_module, function_name, model, L, chi,
                        max_sweeps, tol, n_max, dtype):
    """Generate a temporary Python script for mpirun to execute."""
    # Build the model setup code
    repo_root = get_repo_root()
    package_dir = os.path.join(repo_root, package)

    if model == "heisenberg":
        model_setup = f"""
import quimb.tensor as qtn
mpo = qtn.MPO_ham_heis(L={L}, j=1.0, bz=0.0, cyclic=False)
"""
    elif model == "josephson":
        model_setup = f"""
from benchmarks.lib.models import build_josephson_mpo
mpo = build_josephson_mpo({L}, n_max={n_max})
"""
    else:
        raise ValueError(f"Unknown model: {model}")

    script = f"""#!/usr/bin/env python3
import sys, os, json, time
import numpy as np
from mpi4py import MPI

# Add package dir first (so 'pdmrg' resolves to the installed package, not the repo dir)
sys.path.insert(0, '{package_dir}')
# Add repo root for benchmarks.lib imports
sys.path.insert(0, '{repo_root}')

{model_setup}

from {entry_module} import {function_name}

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

t0 = time.perf_counter()
"""

    if function_name == "a2dmrg_main":
        script += f"""
energy, mps = {function_name}(
    L={L},
    mpo=mpo,
    max_sweeps={max_sweeps},
    bond_dim={chi},
    tol={tol},
    dtype=np.dtype('{dtype}'),
    comm=comm,
    warmup_sweeps=5,
    verbose=False,
)
elapsed = time.perf_counter() - t0
if rank == 0:
    print(json.dumps({{"energy": float(np.real(energy)), "time": elapsed, "success": True}}))
"""
    else:
        script += f"""
result = {function_name}(
    L={L},
    mpo=mpo,
    max_sweeps={max_sweeps},
    bond_dim={chi},
    bond_dim_warmup={chi},
    n_warmup_sweeps=5,
    tol={tol},
    dtype='{dtype}',
    comm=comm,
    verbose=False,
    return_metadata=True,
)
elapsed = time.perf_counter() - t0

if isinstance(result, tuple):
    energy = result[0]
    metadata = result[2] if len(result) > 2 else {{}}
else:
    energy = result
    metadata = {{}}

if rank == 0:
    sweeps = metadata.get('sweeps', {max_sweeps}) if isinstance(metadata, dict) else {max_sweeps}
    converged = metadata.get('converged', True) if isinstance(metadata, dict) else True
    print(json.dumps({{
        "energy": float(np.real(energy)),
        "time": elapsed,
        "sweeps": sweeps,
        "converged": converged,
        "success": True,
    }}))
"""
    return script


def run(model, L, chi, max_sweeps=30, tol=1e-11, np_count=2, threads=1,
        package="pdmrg", entry="pdmrg.dmrg", function="pdmrg_main",
        n_max=2, **kwargs):
    """Run an MPI-based DMRG implementation.

    Args:
        model: 'heisenberg' or 'josephson'
        L, chi, max_sweeps, tol: standard DMRG parameters
        np_count: number of MPI ranks (mpirun -np)
        threads: OPENBLAS_NUM_THREADS per rank
        package: implementation package directory name
        entry: Python module path for import
        function: function name to call
        n_max: charge truncation for Josephson

    Returns:
        dict with keys: energy, time, sweeps, converged, success
    """
    dtype = "complex128" if model == "josephson" else "float64"

    script_content = _make_pdmrg_script(
        package=package, entry_module=entry, function_name=function,
        model=model, L=L, chi=chi, max_sweeps=max_sweeps, tol=tol,
        n_max=n_max, dtype=dtype,
    )

    # Write temporary script
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script_content)
        script_path = f.name

    try:
        # Find the right Python interpreter
        try:
            venv_python = get_venv_python(package)
        except FileNotFoundError:
            venv_python = sys.executable

        env = get_mpi_env(threads=threads)

        cmd = [
            "mpirun", "-np", str(np_count),
            "--oversubscribe",
            "--mca", "btl", "tcp,self",
            "--mca", "btl_tcp_if_include", "lo",
            venv_python, script_path,
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, env=env, timeout=600,
        )

        if result.returncode != 0:
            return {
                "energy": None, "time": None, "success": False,
                "error": result.stderr[-500:] if result.stderr else "Unknown error",
            }

        # Parse JSON from stdout (last line from rank 0)
        for line in reversed(result.stdout.strip().split("\n")):
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)

        return {
            "energy": None, "time": None, "success": False,
            "error": f"No JSON output. stdout: {result.stdout[-200:]}",
        }

    except subprocess.TimeoutExpired:
        return {"energy": None, "time": None, "success": False, "error": "Timeout (600s)"}
    finally:
        os.unlink(script_path)
