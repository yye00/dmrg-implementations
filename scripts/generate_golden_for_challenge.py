#!/usr/bin/env python3
"""
Generate golden reference results for existing challenge benchmark cases.

For cases where MPO/MPS already exist but golden results are missing.
Primarily intended for Josephson L=24/28/32 on HPC systems.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import quimb.tensor as qtn

from benchmark_data_loader import _resolve_case_path, load_mpo_from_disk, load_manifest


def run_quimb_dmrg(mpo, method='dmrg2', tol=1e-12, cutoff=1e-14, max_sweeps=100, threads=1):
    """
    Run quimb DMRG to generate golden reference energy.

    Parameters
    ----------
    mpo : quimb MPO
        The Hamiltonian
    method : str
        'dmrg1' (single-site) or 'dmrg2' (two-site)
    tol : float
        Energy convergence tolerance
    cutoff : float
        Singular value cutoff
    max_sweeps : int
        Maximum number of sweeps
    threads : int
        Number of OpenMP threads

    Returns
    -------
    dict
        Results including energy, sweeps, wall_time, converged
    """
    # Set thread count
    os.environ['OMP_NUM_THREADS'] = str(threads)
    os.environ['MKL_NUM_THREADS'] = str(threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(threads)

    print(f"  Running quimb {method.upper()} (tol={tol}, cutoff={cutoff}, max_sweeps={max_sweeps}, threads={threads})...")

    start_time = time.time()

    # Create DMRG object
    if method == 'dmrg1':
        dmrg = qtn.DMRG1(mpo, bond_dims=[20])  # Will expand as needed
    elif method == 'dmrg2':
        dmrg = qtn.DMRG2(mpo, bond_dims=[20])
    else:
        raise ValueError(f"Unknown method: {method}")

    # Run DMRG
    dmrg.solve(
        tol=tol,
        sweep_sequence='RL',
        cutoff=cutoff,
        max_sweeps=max_sweeps,
        verbosity=1
    )

    wall_time = time.time() - start_time

    # Extract results
    energy = float(dmrg.energy)
    L = len(mpo.tensors)
    energy_per_site = energy / L
    sweeps = dmrg.sweeps
    converged = dmrg.converged

    result = {
        "energy": energy,
        "energy_per_site": energy_per_site,
        "sweeps": sweeps,
        "tolerance": tol,
        "cutoff": cutoff,
        "converged": converged,
        "wall_time": wall_time,
        "threads": threads
    }

    print(f"    ✓ E = {energy:.15f}, sweeps = {sweeps}, time = {wall_time:.2f}s, converged = {converged}")

    return result


def convert_tensors_to_quimb_mpo(mpo_tensors):
    """Convert list of MPO tensors to quimb MPO."""
    L = len(mpo_tensors)

    # Build MPO from tensors
    arrays = []
    for i, W in enumerate(mpo_tensors):
        # W has shape (d, d, D_left, D_right) - need to transpose to quimb format
        # quimb expects (D_left, d, d, D_right)
        W_quimb = np.transpose(W, (2, 0, 1, 3))
        arrays.append(W_quimb)

    # Create quimb MPO
    mpo = qtn.MatrixProductOperator(arrays)

    return mpo


def generate_golden_results(model, case, tol=1e-12, cutoff=1e-14, max_sweeps=100, threads=1):
    """
    Generate golden reference results for a benchmark case.

    Parameters
    ----------
    model : str
        Model name (e.g., "josephson", "heisenberg")
    case : str
        Case identifier (e.g., "L24_D50_nmax2")
    tol : float
        DMRG convergence tolerance
    cutoff : float
        Singular value cutoff
    max_sweeps : int
        Maximum DMRG sweeps
    threads : int
        OpenMP threads

    Returns
    -------
    dict
        Golden results with quimb_dmrg1 and quimb_dmrg2
    """
    print(f"\n{'='*80}")
    print(f"GENERATING GOLDEN RESULTS: {model}/{case}")
    print(f"{'='*80}")

    # Load existing MPO
    print(f"\nLoading MPO and manifest...")
    mpo_tensors, mpo_metadata = load_mpo_from_disk(model, case)
    manifest = load_manifest(model, case)

    print(f"  Model: {manifest.get('model', 'unknown')}")
    print(f"  L: {manifest['L']}")
    print(f"  Bond dim: {manifest.get('bond_dim', 'N/A')}")
    print(f"  Physical dim: {manifest.get('physical_dim', 'N/A')}")
    print(f"  Dtype: {manifest.get('dtype', 'unknown')}")

    # Convert to quimb MPO
    print(f"\nConverting to quimb MPO...")
    mpo = convert_tensors_to_quimb_mpo(mpo_tensors)

    # Run DMRG1
    dmrg1_result = run_quimb_dmrg(
        mpo,
        method='dmrg1',
        tol=tol,
        cutoff=cutoff,
        max_sweeps=max_sweeps,
        threads=threads
    )

    # Run DMRG2
    dmrg2_result = run_quimb_dmrg(
        mpo,
        method='dmrg2',
        tol=tol,
        cutoff=cutoff,
        max_sweeps=max_sweeps,
        threads=threads
    )

    # Create results dictionary
    golden_results = {
        "quimb_dmrg1": dmrg1_result,
        "quimb_dmrg2": dmrg2_result,
        "generation_info": {
            "date": datetime.now().isoformat(),
            "tolerance": tol,
            "cutoff": cutoff,
            "max_sweeps": max_sweeps,
            "threads": threads
        }
    }

    return golden_results


def save_golden_results(model, case, golden_results):
    """Save golden results to JSON file."""
    case_path = _resolve_case_path(model, case)
    output_path = case_path / "golden_results.json"

    print(f"\nSaving golden results to: {output_path}")

    with open(output_path, 'w') as f:
        json.dump(golden_results, f, indent=2)

    print(f"  ✓ Saved")


def main():
    parser = argparse.ArgumentParser(
        description="Generate golden reference results for existing challenge benchmarks"
    )
    parser.add_argument(
        '--model',
        required=True,
        choices=['heisenberg', 'josephson'],
        help='Model name'
    )
    parser.add_argument(
        '--case',
        required=True,
        help='Case identifier (e.g., L24_D50_nmax2)'
    )
    parser.add_argument(
        '--tolerance',
        type=float,
        default=1e-12,
        help='DMRG energy convergence tolerance (default: 1e-12)'
    )
    parser.add_argument(
        '--cutoff',
        type=float,
        default=1e-14,
        help='Singular value cutoff (default: 1e-14)'
    )
    parser.add_argument(
        '--max-sweeps',
        type=int,
        default=100,
        help='Maximum DMRG sweeps (default: 100)'
    )
    parser.add_argument(
        '--threads',
        type=int,
        default=1,
        help='OpenMP threads for quimb (default: 1)'
    )

    args = parser.parse_args()

    try:
        # Generate golden results
        golden_results = generate_golden_results(
            model=args.model,
            case=args.case,
            tol=args.tolerance,
            cutoff=args.cutoff,
            max_sweeps=args.max_sweeps,
            threads=args.threads
        )

        # Save to file
        save_golden_results(args.model, args.case, golden_results)

        print(f"\n{'='*80}")
        print("GOLDEN RESULTS GENERATION COMPLETE")
        print(f"{'='*80}")
        print(f"\nSummary:")
        print(f"  DMRG1 energy: {golden_results['quimb_dmrg1']['energy']:.15f}")
        print(f"  DMRG2 energy: {golden_results['quimb_dmrg2']['energy']:.15f}")
        print(f"  ΔE (DMRG2 - DMRG1): {golden_results['quimb_dmrg2']['energy'] - golden_results['quimb_dmrg1']['energy']:.2e}")
        print(f"  DMRG1 converged: {golden_results['quimb_dmrg1']['converged']}")
        print(f"  DMRG2 converged: {golden_results['quimb_dmrg2']['converged']}")
        print(f"  Total wall time: {golden_results['quimb_dmrg1']['wall_time'] + golden_results['quimb_dmrg2']['wall_time']:.2f}s")
        print()

        return 0

    except Exception as e:
        print(f"\n❌ ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
