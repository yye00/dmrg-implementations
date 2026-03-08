"""Load benchmark MPS/MPO data from the benchmark_data directory.

Provides functions to load stored MPS and MPO tensors (npz format),
convert them to quimb objects, and list available benchmark cases.
"""

import json
from pathlib import Path

import numpy as np
import quimb.tensor as qtn

BENCHMARK_ROOT = Path(__file__).parent.parent / "benchmark_data"


def list_available_benchmarks(tier=None):
    """List available benchmark cases, optionally filtered by tier.

    Returns dict: {model: [case_names]} for cases that have at least
    mpo.npz and manifest.json.
    """
    result = {}
    tiers = [tier] if tier else ["regular", "challenge"]

    for t in tiers:
        tier_dir = BENCHMARK_ROOT / t
        if not tier_dir.exists():
            continue
        for model_dir in sorted(tier_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            for case_dir in sorted(model_dir.iterdir()):
                if not case_dir.is_dir():
                    continue
                if (case_dir / "mpo.npz").exists() and (case_dir / "manifest.json").exists():
                    result.setdefault(model, []).append(case_dir.name)

    return result


def _find_case_dir(model, case):
    """Find the directory for a benchmark case across tiers."""
    for tier in ["regular", "challenge"]:
        d = BENCHMARK_ROOT / tier / model / case
        if d.exists():
            return d
    raise FileNotFoundError(f"Benchmark case not found: {model}/{case}")


def load_benchmark_case(model, case):
    """Load a complete benchmark case: MPO tensors, MPS tensors, manifest, golden results.

    Returns dict with keys: mpo_tensors, mps_tensors, manifest, golden_results.
    mps_tensors and golden_results may be None if files don't exist.
    """
    case_dir = _find_case_dir(model, case)

    # Load manifest
    with open(case_dir / "manifest.json") as f:
        manifest = json.load(f)

    # Load MPO tensors
    mpo_data = np.load(case_dir / "mpo.npz")
    L = manifest["L"]
    mpo_tensors = [mpo_data[f"tensor_{i}"] for i in range(L)]

    # Load MPS tensors (optional)
    mps_path = case_dir / "initial_mps.npz"
    if mps_path.exists():
        mps_data = np.load(mps_path)
        mps_tensors = [mps_data[f"tensor_{i}"] for i in range(L)]
    else:
        mps_tensors = None

    # Load golden results (optional)
    golden_path = case_dir / "golden_results.json"
    if golden_path.exists():
        with open(golden_path) as f:
            golden_results = json.load(f)
    else:
        golden_results = None

    return {
        "mpo_tensors": mpo_tensors,
        "mps_tensors": mps_tensors,
        "manifest": manifest,
        "golden_results": golden_results,
    }


def convert_tensors_to_quimb_mpo(mpo_tensors):
    """Convert a list of MPO tensor arrays to a quimb MPO.

    Stored convention (open boundary):
      - boundary (i=0):   (D, d, d)   — 3D, left bond dim absent
      - bulk:             (D, D, d, d) — 4D
      - boundary (i=L-1): (D, d, d)   — 3D, right bond dim absent

    quimb detects open boundary from first tensor being 3D.
    """
    return qtn.MatrixProductOperator(list(mpo_tensors))


def convert_tensors_to_quimb_mps(mps_tensors, dtype=None):
    """Convert a list of MPS tensor arrays to a quimb MPS.

    Stored convention: (chi_L, d, chi_R) for bulk, (d, chi) or (chi, d) for boundaries.
    Quimb convention: (chi_L, chi_R, d) for bulk, (chi, d) for boundaries.
    """
    L = len(mps_tensors)
    arrays = []
    for i, t in enumerate(mps_tensors):
        if dtype is not None:
            t = t.astype(dtype)
        if t.ndim == 3:
            # Bulk: (chi_L, d, chi_R) -> (chi_L, chi_R, d)
            t = t.transpose(0, 2, 1)
        elif t.ndim == 2:
            if i == 0:
                # Left boundary: (d, chi_R) -> (chi_R, d)
                t = t.transpose(1, 0)
            # Right boundary: (chi_L, d) -> already correct
        arrays.append(t)
    return qtn.MatrixProductState(arrays)
