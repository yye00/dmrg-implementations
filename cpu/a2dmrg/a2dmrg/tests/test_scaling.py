"""
Scalability report framework for A2DMRG.

Generates JSON reports with serial timings and placeholder structure
for parallel timings (to fill when MPI is available).
"""

import json
import time
import numpy as np
import pytest
from quimb.tensor import SpinHam1D

from a2dmrg.dmrg import a2dmrg_main
from a2dmrg.mpi_compat import MPI

pytestmark = pytest.mark.mpi


def _build_heisenberg_mpo(L):
    """Build Heisenberg chain MPO."""
    builder = SpinHam1D(S=1 / 2)
    builder += 1.0, 'X', 'X'
    builder += 1.0, 'Y', 'Y'
    builder += 1.0, 'Z', 'Z'
    return builder.build_mpo(L)


def time_a2dmrg_serial(L, bond_dim, max_sweeps=30, tol=1e-12, warmup_sweeps=2):
    """Time a serial A2DMRG run and return (energy, elapsed_seconds).

    Parameters
    ----------
    L : int
        Number of sites.
    bond_dim : int
        Bond dimension.
    max_sweeps : int
        Maximum number of sweeps.
    tol : float
        Convergence tolerance.
    warmup_sweeps : int
        Number of warm-up sweeps.

    Returns
    -------
    energy : float
        Ground state energy.
    elapsed : float
        Wall time in seconds.
    """
    mpo = _build_heisenberg_mpo(L)
    start = time.perf_counter()
    energy, _ = a2dmrg_main(
        L=L,
        mpo=mpo,
        max_sweeps=max_sweeps,
        bond_dim=bond_dim,
        tol=tol,
        comm=MPI.COMM_WORLD,
        dtype=np.float64,
        one_site=True,
        verbose=False,
        warmup_sweeps=warmup_sweeps,
    )
    elapsed = time.perf_counter() - start
    return energy, elapsed


def test_scaling_report(tmp_path):
    """Generate a JSON scaling report with serial timings.

    Runs A2DMRG on several (L, bond_dim) combinations and writes
    timing data to a JSON file. Parallel entries are placeholders
    for when MPI becomes available.
    """
    configs = [
        {"L": 10, "bond_dim": 20},
        {"L": 20, "bond_dim": 30},
    ]

    serial_results = []
    for cfg in configs:
        energy, elapsed = time_a2dmrg_serial(
            L=cfg["L"],
            bond_dim=cfg["bond_dim"],
            max_sweeps=20,
            tol=1e-10,
        )
        serial_results.append({
            "L": cfg["L"],
            "bond_dim": cfg["bond_dim"],
            "np": 1,
            "energy": float(energy),
            "energy_per_site": float(energy / cfg["L"]),
            "time_seconds": round(elapsed, 4),
        })

    # Placeholder structure for parallel timings
    parallel_placeholder = []
    for cfg in configs:
        for np_count in [2, 4, 8]:
            parallel_placeholder.append({
                "L": cfg["L"],
                "bond_dim": cfg["bond_dim"],
                "np": np_count,
                "energy": None,
                "energy_per_site": None,
                "time_seconds": None,
                "speedup": None,
                "efficiency": None,
                "note": "Requires MPI (mpi4py) to run",
            })

    report = {
        "description": "A2DMRG scalability report",
        "serial": serial_results,
        "parallel": parallel_placeholder,
    }

    # Write JSON report
    report_path = tmp_path / "scaling_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    # Verify report structure
    loaded = json.loads(report_path.read_text())
    assert "serial" in loaded
    assert "parallel" in loaded
    assert len(loaded["serial"]) == len(configs)
    assert len(loaded["parallel"]) == len(configs) * 3

    # Verify serial results are populated
    for entry in loaded["serial"]:
        assert entry["energy"] is not None
        assert entry["energy"] < 0
        assert entry["time_seconds"] > 0

    # Print report summary
    print(f"\n=== Scaling Report ===")
    print(f"Report written to: {report_path}")
    for entry in loaded["serial"]:
        print(
            f"  L={entry['L']:3d}, bond_dim={entry['bond_dim']:3d}, "
            f"np={entry['np']}: E={entry['energy']:.10f}, "
            f"E/L={entry['energy_per_site']:.6f}, "
            f"t={entry['time_seconds']:.2f}s"
        )
    print(f"  Parallel entries: {len(loaded['parallel'])} (placeholder)")
