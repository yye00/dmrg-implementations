#!/usr/bin/env python3
"""
Benchmark Data Loader

Provides consistent loading of static benchmark data across all DMRG implementations.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import quimb.tensor as qtn


def get_benchmark_root() -> Path:
    """Get the benchmark_data directory path."""
    repo_root = Path(__file__).parent
    return repo_root / "benchmark_data"


def list_available_benchmarks() -> Dict[str, List[str]]:
    """
    List all available benchmark cases.

    Returns
    -------
    dict
        {"heisenberg": ["L12_D20", ...], "josephson": ["L20_D50_nmax2", ...]}
    """
    bench_root = get_benchmark_root()
    result = {}

    for model_dir in bench_root.iterdir():
        if model_dir.is_dir() and model_dir.name not in [".git", "__pycache__"]:
            cases = []
            for case_dir in model_dir.iterdir():
                if case_dir.is_dir() and (case_dir / "manifest.json").exists():
                    cases.append(case_dir.name)
            if cases:
                result[model_dir.name] = sorted(cases)

    return result


def load_mpo_from_disk(model: str, case: str) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Load serialized MPO from disk.

    Parameters
    ----------
    model : str
        Model name (e.g., "heisenberg", "josephson")
    case : str
        Case identifier (e.g., "L12_D20", "L20_D50_nmax2")

    Returns
    -------
    mpo_tensors : list of ndarray
        List of MPO tensors
    metadata : dict
        MPO metadata
    """
    bench_root = get_benchmark_root()
    mpo_path = bench_root / model / case / "mpo.npz"

    if not mpo_path.exists():
        raise FileNotFoundError(f"MPO not found: {mpo_path}")

    data = np.load(mpo_path, allow_pickle=True)

    # Extract tensors
    L = int(data["metadata"].item()["L"])
    mpo_tensors = [data[f"tensor_{i}"] for i in range(L)]

    # Extract metadata
    metadata = data["metadata"].item()

    return mpo_tensors, metadata


def load_mps_from_disk(model: str, case: str) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Load serialized initial MPS from disk.

    Parameters
    ----------
    model : str
        Model name
    case : str
        Case identifier

    Returns
    -------
    mps_tensors : list of ndarray
        List of MPS tensors
    metadata : dict
        MPS metadata including gauge information
    """
    bench_root = get_benchmark_root()
    mps_path = bench_root / model / case / "initial_mps.npz"

    if not mps_path.exists():
        raise FileNotFoundError(f"Initial MPS not found: {mps_path}")

    data = np.load(mps_path, allow_pickle=True)

    # Extract tensors
    L = int(data["metadata"].item()["L"])
    mps_tensors = [data[f"tensor_{i}"] for i in range(L)]

    # Extract metadata
    metadata = data["metadata"].item()

    return mps_tensors, metadata


def load_manifest(model: str, case: str) -> Dict[str, Any]:
    """
    Load benchmark manifest.

    Parameters
    ----------
    model : str
        Model name
    case : str
        Case identifier

    Returns
    -------
    dict
        Manifest with system parameters
    """
    bench_root = get_benchmark_root()
    manifest_path = bench_root / model / case / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, 'r') as f:
        return json.load(f)


def load_golden_results(model: str, case: str) -> Dict[str, Any]:
    """
    Load golden-standard reference results.

    Parameters
    ----------
    model : str
        Model name
    case : str
        Case identifier

    Returns
    -------
    dict
        Golden results from quimb DMRG1/DMRG2
    """
    bench_root = get_benchmark_root()
    golden_path = bench_root / model / case / "golden_results.json"

    if not golden_path.exists():
        raise FileNotFoundError(f"Golden results not found: {golden_path}")

    with open(golden_path, 'r') as f:
        return json.load(f)


def load_benchmark_case(model: str, case: str) -> Dict[str, Any]:
    """
    Load complete benchmark case (MPO, MPS, manifest, golden results).

    Parameters
    ----------
    model : str
        Model name
    case : str
        Case identifier

    Returns
    -------
    dict
        {
            "mpo_tensors": list of ndarray,
            "mpo_metadata": dict,
            "mps_tensors": list of ndarray,
            "mps_metadata": dict,
            "manifest": dict,
            "golden_results": dict
        }
    """
    mpo_tensors, mpo_metadata = load_mpo_from_disk(model, case)
    mps_tensors, mps_metadata = load_mps_from_disk(model, case)
    manifest = load_manifest(model, case)

    try:
        golden_results = load_golden_results(model, case)
    except FileNotFoundError:
        golden_results = None

    return {
        "mpo_tensors": mpo_tensors,
        "mpo_metadata": mpo_metadata,
        "mps_tensors": mps_tensors,
        "mps_metadata": mps_metadata,
        "manifest": manifest,
        "golden_results": golden_results
    }


def convert_tensors_to_quimb_mpo(mpo_tensors: List[np.ndarray]) -> qtn.MatrixProductOperator:
    """
    Convert list of MPO tensors to quimb MPO object.

    Parameters
    ----------
    mpo_tensors : list of ndarray
        Raw MPO tensors

    Returns
    -------
    qtn.MatrixProductOperator
        quimb MPO object
    """
    L = len(mpo_tensors)
    arrays = []

    for i, tensor in enumerate(mpo_tensors):
        # quimb expects MPO tensors as (bond_left, bond_right, phys_up, phys_down)
        # Our format is already this
        arrays.append(tensor)

    # Create quimb MPO
    mpo = qtn.MatrixProductOperator(arrays, shape='lrud')
    return mpo


def convert_tensors_to_quimb_mps(mps_tensors: List[np.ndarray]) -> qtn.MatrixProductState:
    """
    Convert list of MPS tensors to quimb MPS object.

    Parameters
    ----------
    mps_tensors : list of ndarray
        Raw MPS tensors (left_bond, phys, right_bond)

    Returns
    -------
    qtn.MatrixProductState
        quimb MPS object
    """
    L = len(mps_tensors)
    arrays = []

    for i, tensor in enumerate(mps_tensors):
        # quimb expects MPS tensors as (bond_left, phys, bond_right)
        arrays.append(tensor)

    # Create quimb MPS
    mps = qtn.MatrixProductState(arrays)
    return mps


if __name__ == '__main__':
    # Test: list available benchmarks
    print("Available benchmarks:")
    benchmarks = list_available_benchmarks()
    for model, cases in benchmarks.items():
        print(f"\n{model}:")
        for case in cases:
            print(f"  - {case}")

    # Test: load a case if any exist
    if benchmarks:
        model = list(benchmarks.keys())[0]
        case = benchmarks[model][0]
        print(f"\nTesting load: {model}/{case}")

        try:
            data = load_benchmark_case(model, case)
            print(f"✓ Loaded successfully")
            print(f"  MPO shape: {[t.shape for t in data['mpo_tensors'][:3]]} ...")
            print(f"  MPS shape: {[t.shape for t in data['mps_tensors'][:3]]} ...")
            print(f"  Manifest: L={data['manifest']['L']}, bond_dim={data['manifest']['bond_dim']}")
        except Exception as e:
            print(f"✗ Failed: {e}")
