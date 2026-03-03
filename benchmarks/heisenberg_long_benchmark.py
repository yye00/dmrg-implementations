#!/usr/bin/env python3
"""Long Heisenberg Model Benchmark (L=32)

This is the same benchmark as `heisenberg_benchmark.py` but with a longer chain.

Tests all implementations against quimb reference:
- quimb DMRG1 (reference)
- quimb DMRG2 (reference)
- PDMRG: np=1, 2, 4, 8
- PDMRG2: np=1, 2, 4, 8
- A2DMRG: np=1, 2, 4, 8

Reports:
- Energy accuracy (vs quimb DMRG2 as ground truth)
- Timing
- Number of sweeps/iterations
- Any issues (convergence, scaling, etc.)

NOTE: This benchmark is intended to be run *alone* on the machine.
"""

import json
import time
import subprocess
import os
import numpy as np

# Configuration (Long Heisenberg)
L = 48  # System size (long)
BOND_DIM = 20
MAX_SWEEPS = 50
# Single tolerance criterion used everywhere (convergence + pass/fail)
TOL = 1e-10
CUTOFF = 1e-14

# Pass/fail threshold is the same as solver tol
PASS_TOL = TOL

# Per-run timeout for MPI jobs (seconds)
MPI_TIMEOUT = 3600


def run_quimb_reference():
    """Run quimb DMRG1 and DMRG2 as reference."""
    import quimb.tensor as qtn

    mpo = qtn.MPO_ham_heis(L=L, j=1.0, bz=0.0, cyclic=False)
    results = {}

    # DMRG1
    t0 = time.time()
    dmrg1 = qtn.DMRG1(mpo, bond_dims=BOND_DIM, cutoffs=CUTOFF)
    dmrg1.solve(max_sweeps=MAX_SWEEPS, tol=TOL, verbosity=0)
    t1 = time.time()
    results["quimb_DMRG1"] = {
        "energy": float(np.real(dmrg1.energy)),
        "time": t1 - t0,
        "sweeps": len(dmrg1.energies) if hasattr(dmrg1, "energies") else "N/A",
    }

    # DMRG2
    t0 = time.time()
    dmrg2 = qtn.DMRG2(mpo, bond_dims=BOND_DIM, cutoffs=CUTOFF)
    dmrg2.solve(max_sweeps=MAX_SWEEPS, tol=TOL, verbosity=0)
    t1 = time.time()
    results["quimb_DMRG2"] = {
        "energy": float(np.real(dmrg2.energy)),
        "time": t1 - t0,
        "sweeps": len(dmrg2.energies) if hasattr(dmrg2, "energies") else "N/A",
    }

    return results


def run_pdmrg(np_count):
    """Run PDMRG with given number of processes."""
    script = f"""
import sys
import time
import json
import numpy as np
import quimb.tensor as qtn

sys.path.insert(0, '/home/captain/clawd/work/dmrg-implementations/pdmrg')

from mpi4py import MPI
from pdmrg.dmrg import pdmrg_main

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

L = {L}
mpo = qtn.MPO_ham_heis(L=L, j=1.0, bz=0.0, cyclic=False)

t0 = time.time()
energy, pmps = pdmrg_main(
    L=L,
    mpo=mpo,
    max_sweeps={MAX_SWEEPS},
    bond_dim={BOND_DIM},
    bond_dim_warmup={BOND_DIM},
    n_warmup_sweeps=5,
    tol={TOL},
    comm=comm,
    verbose=False,
)
t1 = time.time()

if rank == 0:
    print(json.dumps({{
        'energy': float(energy),
        'time': t1 - t0,
        'sweeps': 'N/A'
    }}))
"""

    script_path = "/tmp/pdmrg_bench_long_heis.py"
    with open(script_path, "w") as f:
        f.write(script)

    cmd = [
        "mpirun",
        "-np",
        str(np_count),
        "--oversubscribe",
        "--mca",
        "btl",
        "tcp,self",
        "--mca",
        "btl_tcp_if_include",
        "lo",
        "/home/captain/clawd/work/dmrg-implementations/pdmrg/venv/bin/python",
        script_path,
    ]

    env = os.environ.copy()
    env["PATH"] = "/usr/lib64/openmpi/bin:" + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = "/usr/lib64/openmpi/lib:" + env.get("LD_LIBRARY_PATH", "")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=MPI_TIMEOUT, env=env)
        if result.returncode != 0:
            return {"error": result.stderr, "energy": None, "time": None}

        lines = [l for l in result.stdout.strip().split("\n") if l.startswith("{")]
        if lines:
            return json.loads(lines[-1])
        return {"error": "No output", "stdout": result.stdout, "stderr": result.stderr}
    except subprocess.TimeoutExpired:
        return {"error": f"Timeout ({MPI_TIMEOUT}s)", "energy": None, "time": None}
    except Exception as e:
        return {"error": str(e), "energy": None, "time": None}


def run_pdmrg2(np_count):
    """Run PDMRG2 with given number of processes."""
    script = f"""
import sys
import time
import json
import numpy as np
import quimb.tensor as qtn

sys.path.insert(0, '/home/captain/clawd/work/dmrg-implementations/pdmrg2')

from mpi4py import MPI
from pdmrg.dmrg import pdmrg_main

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

L = {L}
mpo = qtn.MPO_ham_heis(L=L, j=1.0, bz=0.0, cyclic=False)

t0 = time.time()
energy, pmps = pdmrg_main(
    L=L,
    mpo=mpo,
    max_sweeps={MAX_SWEEPS},
    bond_dim={BOND_DIM},
    bond_dim_warmup={BOND_DIM},
    n_warmup_sweeps=5,
    tol={TOL},
    comm=comm,
    verbose=False,
)
t1 = time.time()

if rank == 0:
    print(json.dumps({{
        'energy': float(energy),
        'time': t1 - t0,
        'sweeps': 'N/A'
    }}))
"""

    script_path = "/tmp/pdmrg2_bench_long_heis.py"
    with open(script_path, "w") as f:
        f.write(script)

    cmd = [
        "mpirun",
        "-np",
        str(np_count),
        "--oversubscribe",
        "--mca",
        "btl",
        "tcp,self",
        "--mca",
        "btl_tcp_if_include",
        "lo",
        "/home/captain/clawd/work/dmrg-implementations/pdmrg2/venv/bin/python",
        script_path,
    ]

    env = os.environ.copy()
    env["PATH"] = "/usr/lib64/openmpi/bin:" + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = "/usr/lib64/openmpi/lib:" + env.get("LD_LIBRARY_PATH", "")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=MPI_TIMEOUT, env=env)
        if result.returncode != 0:
            return {"error": result.stderr, "energy": None, "time": None}

        lines = [l for l in result.stdout.strip().split("\n") if l.startswith("{")]
        if lines:
            return json.loads(lines[-1])
        return {"error": "No output", "stdout": result.stdout, "stderr": result.stderr}
    except subprocess.TimeoutExpired:
        return {"error": f"Timeout ({MPI_TIMEOUT}s)", "energy": None, "time": None}
    except Exception as e:
        return {"error": str(e), "energy": None, "time": None}


def run_a2dmrg(np_count):
    """Run A2DMRG with given number of processes."""
    script = f"""
import sys
import time
import json
import numpy as np
import quimb.tensor as qtn

sys.path.insert(0, '/home/captain/clawd/work/dmrg-implementations/a2dmrg')

from mpi4py import MPI
from a2dmrg.dmrg import a2dmrg_main

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

L = {L}
mpo = qtn.MPO_ham_heis(L=L, j=1.0, bz=0.0, cyclic=False)

t0 = time.time()
energy, mps = a2dmrg_main(
    L=L,
    mpo=mpo,
    max_sweeps={MAX_SWEEPS},
    bond_dim={BOND_DIM},
    tol={TOL},
    comm=comm,
    warmup_sweeps=5,
    verbose=False,
)
t1 = time.time()

if rank == 0:
    print(json.dumps({{
        'energy': float(np.real(energy)),
        'time': t1 - t0,
        'sweeps': {MAX_SWEEPS}
    }}))
"""

    script_path = "/tmp/a2dmrg_bench_long_heis.py"
    with open(script_path, "w") as f:
        f.write(script)

    cmd = [
        "mpirun",
        "-np",
        str(np_count),
        "--oversubscribe",
        "--mca",
        "btl",
        "tcp,self",
        "--mca",
        "btl_tcp_if_include",
        "lo",
        "/home/captain/clawd/work/dmrg-implementations/a2dmrg/venv/bin/python",
        script_path,
    ]

    env = os.environ.copy()
    env["PATH"] = "/usr/lib64/openmpi/bin:" + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = "/usr/lib64/openmpi/lib:" + env.get("LD_LIBRARY_PATH", "")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=MPI_TIMEOUT, env=env)
        if result.returncode != 0:
            return {"error": result.stderr, "energy": None, "time": None}

        lines = [l for l in result.stdout.strip().split("\n") if l.startswith("{")]
        if lines:
            return json.loads(lines[-1])
        return {"error": "No output", "stdout": result.stdout, "stderr": result.stderr}
    except subprocess.TimeoutExpired:
        return {"error": f"Timeout ({MPI_TIMEOUT}s)", "energy": None, "time": None}
    except Exception as e:
        return {"error": str(e), "energy": None, "time": None}


def main():
    print("=" * 70)
    print("HEISENBERG MODEL BENCHMARK - LONG (L=32)")
    print("=" * 70)
    print(f"Configuration: L={L}, bond_dim={BOND_DIM}, max_sweeps={MAX_SWEEPS}")
    print(f"Tolerance: {TOL}, Cutoff: {CUTOFF}")
    print(f"Pass threshold: |ΔE| < {PASS_TOL}")
    print()

    all_results = {}
    issues = []

    # quimb references
    print("Running quimb references...")
    ref_results = run_quimb_reference()
    all_results.update(ref_results)

    E_ref = ref_results["quimb_DMRG2"]["energy"]
    # Annotate reference entries with delta_E / passed
    ref_results["quimb_DMRG2"]["delta_E"] = 0.0
    ref_results["quimb_DMRG2"]["passed"] = True
    dE_dmrg1 = ref_results["quimb_DMRG1"]["energy"] - E_ref
    ref_results["quimb_DMRG1"]["delta_E"] = dE_dmrg1
    ref_results["quimb_DMRG1"]["passed"] = bool(abs(dE_dmrg1) < PASS_TOL)
    print(
        f"  quimb DMRG1: E = {ref_results['quimb_DMRG1']['energy']:.15f} ({ref_results['quimb_DMRG1']['time']:.2f}s)"
    )
    print(f"  quimb DMRG2: E = {E_ref:.15f} ({ref_results['quimb_DMRG2']['time']:.2f}s) [REFERENCE]")
    print()

    # PDMRG
    print("Running PDMRG tests...")
    for np_count in [1, 2, 4, 8]:
        print(f"  PDMRG np={np_count}...", end=" ", flush=True)
        result = run_pdmrg(np_count)
        all_results[f"PDMRG_np{np_count}"] = result

        if "error" in result:
            result["delta_E"] = None
            result["passed"] = False
            print(f"ERROR: {result['error'][:80]}...")
            issues.append(f"PDMRG np={np_count}: {result['error']}")
        else:
            dE = result["energy"] - E_ref
            result["delta_E"] = dE
            result["passed"] = bool(abs(dE) < PASS_TOL)
            status = "✓ PASS" if result["passed"] else "✗ FAIL"
            print(f"E = {result['energy']:.15f}, ΔE = {dE:.2e}, t = {result['time']:.2f}s {status}")
            if not result["passed"]:
                issues.append(f"PDMRG np={np_count}: ΔE = {dE:.2e} exceeds threshold")
    print()

    # PDMRG2
    print("Running PDMRG2 tests...")
    for np_count in [1, 2, 4, 8]:
        print(f"  PDMRG2 np={np_count}...", end=" ", flush=True)
        result = run_pdmrg2(np_count)
        all_results[f"PDMRG2_np{np_count}"] = result

        if "error" in result:
            result["delta_E"] = None
            result["passed"] = False
            print(f"ERROR: {result['error'][:80]}...")
            issues.append(f"PDMRG2 np={np_count}: {result['error']}")
        else:
            dE = result["energy"] - E_ref
            result["delta_E"] = dE
            result["passed"] = bool(abs(dE) < PASS_TOL)
            status = "✓ PASS" if result["passed"] else "✗ FAIL"
            print(f"E = {result['energy']:.15f}, ΔE = {dE:.2e}, t = {result['time']:.2f}s {status}")
            if not result["passed"]:
                issues.append(f"PDMRG2 np={np_count}: ΔE = {dE:.2e} exceeds threshold")
    print()

    # A2DMRG
    print("Running A2DMRG tests...")
    for np_count in [1, 2, 4, 8]:
        print(f"  A2DMRG np={np_count}...", end=" ", flush=True)
        result = run_a2dmrg(np_count)
        all_results[f"A2DMRG_np{np_count}"] = result

        if "error" in result:
            result["delta_E"] = None
            result["passed"] = False
            print(f"ERROR: {result['error'][:80]}...")
            issues.append(f"A2DMRG np={np_count}: {result['error']}")
        else:
            dE = result["energy"] - E_ref
            result["delta_E"] = dE
            result["passed"] = bool(abs(dE) < PASS_TOL)
            status = "✓ PASS" if result["passed"] else "✗ FAIL"
            print(f"E = {result['energy']:.15f}, ΔE = {dE:.2e}, t = {result['time']:.2f}s {status}")
            if not result["passed"]:
                issues.append(f"A2DMRG np={np_count}: ΔE = {dE:.2e} exceeds threshold")

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Method':<20} {'Energy':<20} {'ΔE':<12} {'Time (s)':<10} {'Status'}")
    print("-" * 70)

    for name, result in all_results.items():
        if "error" in result:
            print(f"{name:<20} {'ERROR':<20} {'-':<12} {'-':<10} ✗")
        else:
            dE = result["energy"] - E_ref if name != "quimb_DMRG2" else 0.0
            status = "✓" if abs(dE) < PASS_TOL else "✗"
            print(f"{name:<20} {result['energy']:<20.15f} {dE:<12.2e} {result['time']:<10.2f} {status}")

    print()
    print("=" * 70)
    print("ISSUES DETECTED")
    print("=" * 70)
    if issues:
        for issue in issues:
            print(f"  ⚠ {issue}")
    else:
        print("  ✓ No issues detected - all tests passed!")

    output_path = "/home/captain/clawd/work/dmrg-implementations/benchmarks/heisenberg_long_benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
