"""
Generate binary MPS/MPO data files for reproducible benchmarks.

Creates Heisenberg and Josephson test data with a fixed random seed (42)
so that CPU and GPU implementations start from identical initial states.

Output format: complex128 binary files with int64 headers.
See lib/data_loader.py for the loader.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import quimb.tensor as qtn

from benchmarks.lib.models import build_josephson_mpo


DATA_DIR = Path(__file__).parent


def save_mps_to_binary(mps, filepath, model_info):
    """Save MPS tensors to binary file with shape (D_left, d, D_right)."""
    tensors = [t.data for t in mps.tensors] if hasattr(mps, "tensors") else list(mps)
    tensors = [np.asarray(t, dtype=np.complex128) for t in tensors]
    L = len(tensors)

    # Normalize quimb tensor shapes to (D_left, d, D_right)
    normalized = []
    for i, t in enumerate(tensors):
        if i == 0 and t.ndim == 2:
            t = t[np.newaxis, :, :]
            t = np.transpose(t, (0, 2, 1))
        elif i == L - 1 and t.ndim == 2:
            t = t[:, :, np.newaxis]
        elif t.ndim == 3:
            t = np.transpose(t, (0, 2, 1))
        normalized.append(t)

    filepath = Path(filepath)
    with open(filepath, "wb") as f:
        np.array([L], dtype=np.int64).tofile(f)
        bond_dims = [1] + [t.shape[2] for t in normalized]
        np.array(bond_dims, dtype=np.int64).tofile(f)
        phys_dims = [t.shape[1] for t in normalized]
        np.array(phys_dims, dtype=np.int64).tofile(f)
        for t in normalized:
            np.array(t.shape, dtype=np.int64).tofile(f)
            np.ascontiguousarray(t).tofile(f)

    # Save metadata JSON
    meta = {"model": model_info, "num_sites": L,
            "shapes": [list(t.shape) for t in normalized],
            "dtype": "complex128", "total_elements": sum(t.size for t in normalized)}
    with open(filepath.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)

    mb = filepath.stat().st_size / 1024**2
    print(f"  MPS: {filepath.name} ({mb:.2f} MB, {L} sites)")


def save_mpo_to_binary(mpo, filepath, model_info):
    """Save MPO tensors to binary file with shape (D_left, d, d, D_right)."""
    tensors = [t.data for t in mpo.tensors] if hasattr(mpo, "tensors") else list(mpo)
    tensors = [np.asarray(t, dtype=np.complex128) for t in tensors]
    L = len(tensors)

    # Normalize quimb tensor shapes to (D_mpo_left, d, d, D_mpo_right)
    normalized = []
    for i, t in enumerate(tensors):
        if t.ndim == 3:
            if i == 0:
                t = t[np.newaxis, :, :, :]
                t = np.transpose(t, (0, 2, 3, 1))
            else:
                t = t[:, :, :, np.newaxis]
        elif t.ndim == 4:
            t = np.transpose(t, (0, 2, 3, 1))
        normalized.append(t)

    filepath = Path(filepath)
    with open(filepath, "wb") as f:
        np.array([L], dtype=np.int64).tofile(f)
        mpo_bond_dims = [1] + [t.shape[3] for t in normalized]
        np.array(mpo_bond_dims, dtype=np.int64).tofile(f)
        phys_dims = [t.shape[1] for t in normalized]
        np.array(phys_dims, dtype=np.int64).tofile(f)
        for t in normalized:
            np.array(t.shape, dtype=np.int64).tofile(f)
            np.ascontiguousarray(t).tofile(f)

    meta = {"model": model_info, "num_sites": L,
            "shapes": [list(t.shape) for t in normalized],
            "dtype": "complex128", "total_elements": sum(t.size for t in normalized)}
    with open(filepath.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)

    mb = filepath.stat().st_size / 1024**2
    print(f"  MPO: {filepath.name} ({mb:.2f} MB, {L} sites)")


def generate_heisenberg(L, chi_init, output_dir, seed=42):
    """Generate Heisenberg MPS and MPO."""
    print(f"\nHeisenberg L={L}, chi_init={chi_init}")
    np.random.seed(seed)
    mpo = qtn.MPO_ham_heis(L=L, j=1.0, bz=0.0, cyclic=False)
    mps = qtn.MPS_rand_state(L=L, bond_dim=chi_init, phys_dim=2, dtype=complex)
    mps.normalize()
    info = {"model": "heisenberg", "L": L, "d": 2, "chi_init": chi_init,
            "j": 1.0, "bz": 0.0, "cyclic": False, "seed": seed}
    save_mps_to_binary(mps, output_dir / f"heisenberg_L{L}_chi{chi_init}_mps.bin", info)
    save_mpo_to_binary(mpo, output_dir / f"heisenberg_L{L}_mpo.bin", info)


def generate_josephson(L, n_max, chi_init, output_dir, seed=42):
    """Generate Josephson MPS and MPO."""
    d = 2 * n_max + 1
    print(f"\nJosephson L={L}, n_max={n_max} (d={d}), chi_init={chi_init}")
    np.random.seed(seed)
    mpo = build_josephson_mpo(L, n_max=n_max)
    mps = qtn.MPS_rand_state(L=L, bond_dim=chi_init, phys_dim=d, dtype=complex)
    mps.normalize()
    info = {"model": "josephson", "L": L, "d": d, "n_max": n_max,
            "chi_init": chi_init, "E_J": 1.0, "E_C": 0.5, "seed": seed}
    save_mps_to_binary(mps, output_dir / f"josephson_L{L}_n{n_max}_chi{chi_init}_mps.bin", info)
    save_mpo_to_binary(mpo, output_dir / f"josephson_L{L}_n{n_max}_mpo.bin", info)


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark MPS/MPO data")
    parser.add_argument("--output-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--heisenberg", action="store_true")
    parser.add_argument("--josephson", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    if args.heisenberg or args.all:
        for L, chi in [(12, 10), (20, 10), (40, 20)]:
            generate_heisenberg(L, chi, output_dir, seed=args.seed)

    if args.josephson or args.all:
        for L, n_max, chi in [(8, 2, 10), (12, 2, 10), (16, 2, 20)]:
            generate_josephson(L, n_max, chi, output_dir, seed=args.seed)

    print(f"\nDone. Files in: {output_dir}")


if __name__ == "__main__":
    main()
