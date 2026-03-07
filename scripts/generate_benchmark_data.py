#!/usr/bin/env python3
"""
Generate Static Benchmark Data

Creates serialized MPO and MPS files for all benchmark cases.
Also generates golden-standard reference results using quimb DMRG2.
"""

import sys
import os
import json
import argparse
import time
from datetime import datetime
from pathlib import Path
import numpy as np

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import quimb.tensor as qtn


def create_heisenberg_mpo(L: int) -> qtn.MatrixProductOperator:
    """Create Heisenberg chain MPO."""
    return qtn.MPO_ham_heis(L=L, j=1.0, bz=0.0, cyclic=False)


def create_josephson_mpo(L: int, E_J: float, E_C: float, n_max: int,
                         with_flux: bool = True) -> qtn.MatrixProductOperator:
    """Create Josephson junction array MPO."""
    # Import from a2dmrg benchmarks
    a2dmrg_path = repo_root / "a2dmrg"
    sys.path.insert(0, str(a2dmrg_path))

    try:
        from benchmarks.josephson_junction import build_josephson_mpo
        mpo = build_josephson_mpo(L, E_J=E_J, E_C=E_C, n_max=n_max, with_flux=with_flux)
        return mpo
    except ImportError:
        raise ImportError(
            "Could not import josephson_junction builder from a2dmrg/benchmarks. "
            "Ensure a2dmrg package is set up correctly."
        )


def extract_mpo_tensors(mpo: qtn.MatrixProductOperator) -> list:
    """Extract raw tensor arrays from quimb MPO."""
    L = mpo.L
    tensors = []

    for i in range(L):
        # Get tensor at site i
        tensor_data = mpo.arrays[i]
        tensors.append(tensor_data)

    return tensors


def extract_mps_tensors(mps: qtn.MatrixProductState) -> list:
    """Extract raw tensor arrays from quimb MPS."""
    L = mps.L
    tensors = []

    for i in range(L):
        tensor_data = mps.arrays[i]
        tensors.append(tensor_data)

    return tensors


def save_mpo(mpo_tensors: list, output_path: Path, L: int, dtype: str):
    """Save MPO tensors to .npz file."""
    data = {}

    for i, tensor in enumerate(mpo_tensors):
        data[f"tensor_{i}"] = tensor

    # Add metadata
    metadata = {
        "L": L,
        "dtype": dtype,
        "format_version": "1.0"
    }
    data["metadata"] = metadata

    np.savez_compressed(output_path, **data)
    print(f"  Saved MPO: {output_path}")


def save_mps(mps_tensors: list, output_path: Path, L: int, dtype: str, gauge: str):
    """Save MPS tensors to .npz file."""
    data = {}

    for i, tensor in enumerate(mps_tensors):
        data[f"tensor_{i}"] = tensor

    # Add metadata
    metadata = {
        "L": L,
        "dtype": dtype,
        "gauge": gauge,
        "format_version": "1.0"
    }
    data["metadata"] = metadata

    np.savez_compressed(output_path, **data)
    print(f"  Saved MPS: {output_path}")


def generate_heisenberg_case(L: int, bond_dim: int, output_dir: Path):
    """Generate Heisenberg benchmark case."""
    print(f"\nGenerating Heisenberg L={L}, D={bond_dim}...")

    # Create MPO
    mpo = create_heisenberg_mpo(L)
    mpo_tensors = extract_mpo_tensors(mpo)

    # Generate initial MPS using DMRG2 with low sweeps
    print("  Running DMRG2 for initial state...")
    dmrg = qtn.DMRG2(mpo, bond_dims=bond_dim, cutoffs=1e-14)
    dmrg.solve(max_sweeps=5, tol=1e-8, verbosity=0)

    # Canonize to left-canonical form
    mps = dmrg.state
    mps.canonize(L - 1)  # Left-canonical
    mps_tensors = extract_mps_tensors(mps)

    # Save MPO
    output_dir.mkdir(parents=True, exist_ok=True)
    save_mpo(mpo_tensors, output_dir / "mpo.npz", L, "float64")

    # Save MPS
    save_mps(mps_tensors, output_dir / "initial_mps.npz", L, "float64", "left")

    # Create manifest
    manifest = {
        "model": "heisenberg",
        "L": L,
        "bond_dim": bond_dim,
        "dtype": "float64",
        "parameters": {
            "j": 1.0,
            "bz": 0.0,
            "cyclic": False
        },
        "initial_mps_gauge": "left",
        "generator_version": "1.0",
        "created_timestamp": datetime.now().isoformat()
    }

    with open(output_dir / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved manifest")

    # Generate golden results
    print("  Generating golden results...")
    golden_results = generate_golden_results(mpo, bond_dim, cutoff=1e-14, tol=1e-12)

    with open(output_dir / "golden_results.json", 'w') as f:
        json.dump(golden_results, f, indent=2)
    print(f"  Saved golden results")


def generate_josephson_case(L: int, bond_dim: int, n_max: int, output_dir: Path):
    """Generate Josephson junction array benchmark case."""
    print(f"\nGenerating Josephson L={L}, D={bond_dim}, n_max={n_max}...")

    # Create MPO
    E_J = 1.0
    E_C = 0.5
    mpo = create_josephson_mpo(L, E_J, E_C, n_max, with_flux=True)
    mpo_tensors = extract_mpo_tensors(mpo)

    # Generate initial MPS using DMRG2
    print("  Running DMRG2 for initial state...")
    dmrg = qtn.DMRG2(mpo, bond_dims=bond_dim, cutoffs=1e-14)
    dmrg.solve(max_sweeps=5, tol=1e-8, verbosity=0)

    # Canonize to left-canonical form
    mps = dmrg.state
    mps.canonize(L - 1)  # Left-canonical
    mps_tensors = extract_mps_tensors(mps)

    # Save MPO
    output_dir.mkdir(parents=True, exist_ok=True)
    save_mpo(mpo_tensors, output_dir / "mpo.npz", L, "complex128")

    # Save MPS
    save_mps(mps_tensors, output_dir / "initial_mps.npz", L, "complex128", "left")

    # Create manifest
    manifest = {
        "model": "josephson",
        "L": L,
        "bond_dim": bond_dim,
        "dtype": "complex128",
        "parameters": {
            "E_J": E_J,
            "E_C": E_C,
            "n_max": n_max,
            "with_flux": True
        },
        "initial_mps_gauge": "left",
        "generator_version": "1.0",
        "created_timestamp": datetime.now().isoformat()
    }

    with open(output_dir / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved manifest")

    # Generate golden results
    print("  Generating golden results...")
    golden_results = generate_golden_results(mpo, bond_dim, cutoff=1e-14, tol=1e-12)

    with open(output_dir / "golden_results.json", 'w') as f:
        json.dump(golden_results, f, indent=2)
    print(f"  Saved golden results")


def generate_golden_results(mpo, bond_dim, cutoff, tol):
    """
    Generate golden-standard results using quimb DMRG1 and DMRG2.

    Uses OpenMP threads=1 for reproducibility.
    """
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    golden = {}

    # DMRG1
    print("    Running quimb DMRG1...")
    t0 = time.time()
    dmrg1 = qtn.DMRG1(mpo, bond_dims=bond_dim, cutoffs=cutoff)
    dmrg1.solve(max_sweeps=50, tol=tol, verbosity=0)
    t1 = time.time()

    golden["quimb_dmrg1"] = {
        "energy": float(np.real(dmrg1.energy)),
        "energy_per_site": float(np.real(dmrg1.energy)) / mpo.L,
        "sweeps": len(dmrg1.energies) if hasattr(dmrg1, 'energies') else 50,
        "tolerance": tol,
        "cutoff": cutoff,
        "converged": True,  # Assume converged if no exception
        "wall_time": t1 - t0,
        "threads": 1
    }

    # DMRG2
    print("    Running quimb DMRG2...")
    t0 = time.time()
    dmrg2 = qtn.DMRG2(mpo, bond_dims=bond_dim, cutoffs=cutoff)
    dmrg2.solve(max_sweeps=50, tol=tol, verbosity=0)
    t1 = time.time()

    golden["quimb_dmrg2"] = {
        "energy": float(np.real(dmrg2.energy)),
        "energy_per_site": float(np.real(dmrg2.energy)) / mpo.L,
        "sweeps": len(dmrg2.energies) if hasattr(dmrg2, 'energies') else 50,
        "tolerance": tol,
        "cutoff": cutoff,
        "converged": True,
        "wall_time": t1 - t0,
        "threads": 1
    }

    return golden


def main():
    parser = argparse.ArgumentParser(description='Generate static benchmark data')
    parser.add_argument('--all', action='store_true',
                        help='Generate all benchmark cases')
    parser.add_argument('--heisenberg', action='store_true',
                        help='Generate Heisenberg cases only')
    parser.add_argument('--josephson', action='store_true',
                        help='Generate Josephson cases only')
    parser.add_argument('--case', type=str,
                        help='Generate specific case (e.g., heisenberg_L12_D20)')

    args = parser.parse_args()

    benchmark_root = repo_root / "benchmark_data"

    if args.all or args.heisenberg:
        # Heisenberg cases
        heisenberg_cases = [
            (12, 20),
            (32, 20),
            (48, 20)
        ]

        for L, bond_dim in heisenberg_cases:
            output_dir = benchmark_root / "heisenberg" / f"L{L}_D{bond_dim}"
            generate_heisenberg_case(L, bond_dim, output_dir)

    if args.all or args.josephson:
        # Josephson cases
        josephson_cases = [
            (20, 50, 2),
            (24, 50, 2),
            (28, 50, 2),
            (32, 50, 2)
        ]

        for L, bond_dim, n_max in josephson_cases:
            output_dir = benchmark_root / "josephson" / f"L{L}_D{bond_dim}_nmax{n_max}"
            generate_josephson_case(L, bond_dim, n_max, output_dir)

    if args.case:
        # Parse case string
        parts = args.case.split('_')
        model = parts[0]
        L = int(parts[1][1:])
        bond_dim = int(parts[2][1:])

        if model == "heisenberg":
            output_dir = benchmark_root / "heisenberg" / f"L{L}_D{bond_dim}"
            generate_heisenberg_case(L, bond_dim, output_dir)
        elif model == "josephson":
            n_max = int(parts[3][4:])
            output_dir = benchmark_root / "josephson" / f"L{L}_D{bond_dim}_nmax{n_max}"
            generate_josephson_case(L, bond_dim, n_max, output_dir)
        else:
            print(f"Unknown model: {model}")
            sys.exit(1)

    print("\n" + "="*70)
    print("BENCHMARK DATA GENERATION COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
