"""
MPS/MPO data loading for benchmarks.

Supports two data formats:
  1. Binary (.bin) files with complex128 data (for CPU/GPU interop)
  2. NPZ (.npz) files with manifest.json (legacy benchmark_data format)

All loaders return raw numpy arrays. Conversion to quimb objects is separate.
"""

import json
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data"
BENCHMARK_DATA_DIR = Path(__file__).parent.parent.parent / "benchmark_data"


# ---------------------------------------------------------------------------
# Binary format loaders (primary format for CPU/GPU benchmarks)
# ---------------------------------------------------------------------------

def load_mps_from_binary(filepath, quiet=False):
    """Load MPS tensors from a .bin file.

    Returns (tensors, metadata) where each tensor has shape (D_left, d, D_right).
    """
    filepath = Path(filepath)
    with open(filepath, "rb") as f:
        num_sites = np.fromfile(f, dtype=np.int64, count=1)[0]
        bond_dims = np.fromfile(f, dtype=np.int64, count=num_sites + 1)
        phys_dims = np.fromfile(f, dtype=np.int64, count=num_sites)

        tensors = []
        for site in range(num_sites):
            shape = tuple(np.fromfile(f, dtype=np.int64, count=3))
            assert shape[0] == bond_dims[site], f"Bond dim mismatch at site {site}"
            assert shape[2] == bond_dims[site + 1], f"Bond dim mismatch at site {site}"
            assert shape[1] == phys_dims[site], f"Phys dim mismatch at site {site}"
            data = np.fromfile(f, dtype=np.complex128, count=int(np.prod(shape)))
            tensors.append(data.reshape(shape))

    metadata = {}
    meta_path = filepath.with_suffix(".json")
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)

    if not quiet:
        print(f"Loaded MPS: {filepath.name} ({num_sites} sites, "
              f"bond_dims={list(bond_dims)}, phys_dims={list(phys_dims)})")

    return tensors, metadata


def load_mpo_from_binary(filepath, quiet=False):
    """Load MPO tensors from a .bin file.

    Returns (tensors, metadata) where each tensor has shape (D_left, d, d, D_right).
    """
    filepath = Path(filepath)
    with open(filepath, "rb") as f:
        num_sites = np.fromfile(f, dtype=np.int64, count=1)[0]
        mpo_bond_dims = np.fromfile(f, dtype=np.int64, count=num_sites + 1)
        phys_dims = np.fromfile(f, dtype=np.int64, count=num_sites)

        tensors = []
        for site in range(num_sites):
            shape = tuple(np.fromfile(f, dtype=np.int64, count=4))
            assert shape[0] == mpo_bond_dims[site]
            assert shape[3] == mpo_bond_dims[site + 1]
            assert shape[1] == phys_dims[site]
            assert shape[2] == phys_dims[site]
            data = np.fromfile(f, dtype=np.complex128, count=int(np.prod(shape)))
            tensors.append(data.reshape(shape))

    metadata = {}
    meta_path = filepath.with_suffix(".json")
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)

    if not quiet:
        print(f"Loaded MPO: {filepath.name} ({num_sites} sites, "
              f"mpo_bond_dims={list(mpo_bond_dims)}, phys_dims={list(phys_dims)})")

    return tensors, metadata


# ---------------------------------------------------------------------------
# NPZ format loaders (legacy benchmark_data/ format)
# ---------------------------------------------------------------------------

def list_available_benchmarks(tier=None):
    """List available NPZ benchmark cases. Returns {model: [case_names]}."""
    result = {}
    tiers = [tier] if tier else ["regular", "challenge"]
    for t in tiers:
        tier_dir = BENCHMARK_DATA_DIR / t
        if not tier_dir.exists():
            continue
        for model_dir in sorted(tier_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            for case_dir in sorted(model_dir.iterdir()):
                if not case_dir.is_dir():
                    continue
                if (case_dir / "mpo.npz").exists() and (case_dir / "manifest.json").exists():
                    result.setdefault(model_dir.name, []).append(case_dir.name)
    return result


def load_benchmark_case(model, case):
    """Load a benchmark case from NPZ format.

    Returns dict with keys: mpo_tensors, mps_tensors, manifest, golden_results.
    """
    case_dir = None
    for tier in ["regular", "challenge"]:
        d = BENCHMARK_DATA_DIR / tier / model / case
        if d.exists():
            case_dir = d
            break
    if case_dir is None:
        raise FileNotFoundError(f"Benchmark case not found: {model}/{case}")

    with open(case_dir / "manifest.json") as f:
        manifest = json.load(f)

    L = manifest["L"]
    mpo_data = np.load(case_dir / "mpo.npz")
    mpo_tensors = [mpo_data[f"tensor_{i}"] for i in range(L)]

    mps_path = case_dir / "initial_mps.npz"
    mps_tensors = None
    if mps_path.exists():
        mps_data = np.load(mps_path)
        mps_tensors = [mps_data[f"tensor_{i}"] for i in range(L)]

    golden_path = case_dir / "golden_results.json"
    golden_results = None
    if golden_path.exists():
        with open(golden_path) as f:
            golden_results = json.load(f)

    return {
        "mpo_tensors": mpo_tensors,
        "mps_tensors": mps_tensors,
        "manifest": manifest,
        "golden_results": golden_results,
    }


# ---------------------------------------------------------------------------
# Quimb conversions
# ---------------------------------------------------------------------------

def tensors_to_quimb_mpo(mpo_tensors):
    """Convert raw MPO tensor arrays to a quimb MatrixProductOperator."""
    import quimb.tensor as qtn
    return qtn.MatrixProductOperator(list(mpo_tensors))


def tensors_to_quimb_mps(mps_tensors, dtype=None, phys_dim=None):
    """Convert raw MPS tensor arrays to a quimb MatrixProductState.

    Auto-detects internal format (chi_L, d, chi_R) vs quimb format (chi_L, chi_R, d)
    and transposes if needed.
    """
    import quimb.tensor as qtn

    L = len(mps_tensors)

    if phys_dim is None:
        for t in mps_tensors:
            if t.ndim == 3:
                phys_dim = min(t.shape)
                break
        if phys_dim is None:
            phys_dim = 2

    # Detect format from a bulk tensor
    needs_transpose = True
    for t in mps_tensors[L // 3: 2 * L // 3]:
        if t.ndim == 3 and t.shape[1] != t.shape[2]:
            if t.shape[2] == phys_dim and t.shape[1] != phys_dim:
                needs_transpose = False
            break

    arrays = []
    for i, t in enumerate(mps_tensors):
        if dtype is not None:
            t = t.astype(dtype)
        if t.ndim == 3 and needs_transpose:
            t = t.transpose(0, 2, 1)
        elif t.ndim == 2 and i == 0 and needs_transpose:
            t = t.transpose(1, 0)
        arrays.append(t)

    return qtn.MatrixProductState(arrays)


# ---------------------------------------------------------------------------
# Data file discovery
# ---------------------------------------------------------------------------

def find_data_files(model, L, chi=None, n_max=None, data_dir=None):
    """Find MPS and MPO binary files for a given model and system size.

    Returns (mps_path, mpo_path) or raises FileNotFoundError.
    """
    if data_dir is None:
        data_dir = DATA_DIR

    data_dir = Path(data_dir)

    if model == "heisenberg":
        mpo_path = data_dir / f"heisenberg_L{L}_mpo.bin"
        if chi is not None:
            mps_path = data_dir / f"heisenberg_L{L}_chi{chi}_mps.bin"
        else:
            # Find any MPS file for this L
            candidates = sorted(data_dir.glob(f"heisenberg_L{L}_chi*_mps.bin"))
            if not candidates:
                raise FileNotFoundError(f"No Heisenberg MPS data for L={L} in {data_dir}")
            mps_path = candidates[0]
    elif model == "josephson":
        nm = n_max or 2
        mpo_path = data_dir / f"josephson_L{L}_n{nm}_mpo.bin"
        if chi is not None:
            mps_path = data_dir / f"josephson_L{L}_n{nm}_chi{chi}_mps.bin"
        else:
            candidates = sorted(data_dir.glob(f"josephson_L{L}_n{nm}_chi*_mps.bin"))
            if not candidates:
                raise FileNotFoundError(f"No Josephson MPS data for L={L} in {data_dir}")
            mps_path = candidates[0]
    else:
        raise ValueError(f"Unknown model: {model}")

    if not mps_path.exists():
        raise FileNotFoundError(f"MPS file not found: {mps_path}")
    if not mpo_path.exists():
        raise FileNotFoundError(f"MPO file not found: {mpo_path}")

    return mps_path, mpo_path
