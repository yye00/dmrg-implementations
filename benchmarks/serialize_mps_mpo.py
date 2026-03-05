#!/usr/bin/env python3
"""
MPS/MPO Serialization Utility
==============================

Generates and saves MPS and MPO tensors to binary files for use in both
CPU (Python/Quimb) and GPU (C++/HIP) benchmarks.

This ensures exact reproducibility: both implementations use identical
initial MPS states and Hamiltonian MPO operators.

Output format:
- Binary files with metadata header
- Complex128 data (16 bytes per element: 8-byte real + 8-byte imag)
- Compatible with both Python (numpy) and C++ (std::complex<double>)

Files generated:
- heisenberg_L{L}_mps.bin - Initial MPS state
- heisenberg_L{L}_mpo.bin - Hamiltonian MPO
- josephson_L{L}_n{n_max}_mps.bin - Josephson MPS
- josephson_L{L}_n{n_max}_mpo.bin - Josephson MPO
- metadata.json - Human-readable metadata
"""

import numpy as np
import json
import argparse
from pathlib import Path
import sys
import os

# Add paths for quimb
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

import quimb.tensor as qtn


def save_tensor_metadata(filepath, tensors, model_info):
    """Save tensor metadata as JSON."""
    metadata = {
        "model": model_info,
        "num_sites": len(tensors),
        "shapes": [list(t.shape) for t in tensors],
        "dtype": str(tensors[0].dtype),
        "total_elements": sum(t.size for t in tensors),
    }

    meta_path = filepath.with_suffix('.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  Metadata: {meta_path}")
    return metadata


def save_mps_to_binary(mps, filepath, model_info):
    """
    Save MPS tensors to binary file.

    Format:
    - Header: num_sites (int64), bond_dims[] (int64 array)
    - For each site i:
      - shape: (D_left, d, D_right) as 3 int64 values
      - data: D_left * d * D_right complex128 values

    Parameters
    ----------
    mps : quimb MPS or list of tensors
        Matrix Product State (list of rank-3 tensors)
    filepath : Path
        Output file path
    model_info : dict
        Model metadata
    """
    # Extract tensor arrays from quimb MPS
    if hasattr(mps, 'tensors'):
        tensors = [t.data for t in mps.tensors]
    else:
        tensors = mps

    L = len(tensors)

    # Convert to complex128 if needed
    tensors = [np.asarray(t, dtype=np.complex128) for t in tensors]

    # Normalize tensor shapes: quimb boundary tensors lack one bond dimension
    # First tensor: (D_right, d) → (1, d, D_right)
    # Middle tensors: (D_left, D_right, d) → (D_left, d, D_right)
    # Last tensor: (D_left, d) → (D_left, d, 1)
    normalized_tensors = []

    for i, t in enumerate(tensors):
        if i == 0:
            # First site: (D_right, d) → (1, d, D_right)
            if len(t.shape) == 2:
                t_norm = t[np.newaxis, :, :]  # Add left bond of size 1
                t_norm = np.transpose(t_norm, (0, 2, 1))  # (1, D_right, d) → (1, d, D_right)
            else:
                # Already 3D: (D_left, D_right, d) → (D_left, d, D_right)
                t_norm = np.transpose(t, (0, 2, 1))
        elif i == L - 1:
            # Last site: (D_left, d) → (D_left, d, 1)
            if len(t.shape) == 2:
                t_norm = t[:, :, np.newaxis]  # Add right bond of size 1
            else:
                # Already 3D: (D_left, D_right, d) → (D_left, d, D_right)
                t_norm = np.transpose(t, (0, 2, 1))
        else:
            # Middle site: (D_left, D_right, d) → (D_left, d, D_right)
            t_norm = np.transpose(t, (0, 2, 1))

        normalized_tensors.append(t_norm)

    tensors = normalized_tensors

    # Save metadata
    metadata = save_tensor_metadata(filepath, tensors, model_info)

    with open(filepath, 'wb') as f:
        # Header: number of sites
        np.array([L], dtype=np.int64).tofile(f)

        # Bond dimensions (size L+1: left and right bonds for all sites)
        bond_dims = [1]  # Left boundary
        for t in tensors:
            bond_dims.append(t.shape[2])  # Right bond
        np.array(bond_dims, dtype=np.int64).tofile(f)

        # Physical dimensions (size L)
        phys_dims = [t.shape[1] for t in tensors]
        np.array(phys_dims, dtype=np.int64).tofile(f)

        # Write each tensor
        for i, t in enumerate(tensors):
            # Verify shape is (D_left, d, D_right)
            assert len(t.shape) == 3, f"MPS tensor {i} must be rank-3, got shape {t.shape}"

            # Write shape
            np.array(t.shape, dtype=np.int64).tofile(f)

            # Write data in C-contiguous order
            t_contig = np.ascontiguousarray(t)
            t_contig.tofile(f)

    total_mb = filepath.stat().st_size / 1024**2
    print(f"  MPS saved: {filepath} ({total_mb:.2f} MB)")
    print(f"  Sites: {L}, Bond dims: {bond_dims}, Phys dims: {phys_dims}")

    return metadata


def save_mpo_to_binary(mpo, filepath, model_info):
    """
    Save MPO tensors to binary file.

    Format:
    - Header: num_sites (int64), mpo_bond_dims[] (int64 array)
    - For each site i:
      - shape: (D_mpo_left, d, d, D_mpo_right) as 4 int64 values
      - data: D_mpo_left * d * d * D_mpo_right complex128 values

    Parameters
    ----------
    mpo : quimb MPO or list of tensors
        Matrix Product Operator (list of rank-4 tensors)
    filepath : Path
        Output file path
    model_info : dict
        Model metadata
    """
    # Extract tensor arrays from quimb MPO
    if hasattr(mpo, 'tensors'):
        tensors = [t.data for t in mpo.tensors]
    else:
        tensors = mpo

    L = len(tensors)

    # Convert to complex128
    tensors = [np.asarray(t, dtype=np.complex128) for t in tensors]

    # Normalize tensor shapes: quimb boundary tensors lack one bond dimension
    # Quimb format: (bond_left, bond_right, ket, bra) OR (bond, ket, bra) at boundaries
    # Target format: (bond_left, ket, bra, bond_right)
    normalized_tensors = []

    for i, t in enumerate(tensors):
        if len(t.shape) == 3:
            # Boundary tensor: (bond, ket, bra)
            if i == 0:
                # First site: (D_mpo_right, ket, bra) → (1, ket, bra, D_mpo_right)
                t_norm = t[np.newaxis, :, :, :]  # Add left bond of size 1
                t_norm = np.transpose(t_norm, (0, 2, 3, 1))  # (1, D_mpo_right, ket, bra) → (1, ket, bra, D_mpo_right)
            else:
                # Last site: (D_mpo_left, ket, bra) → (D_mpo_left, ket, bra, 1)
                t_norm = t[:, :, :, np.newaxis]  # Add right bond of size 1
        elif len(t.shape) == 4:
            # Middle tensor: (D_mpo_left, D_mpo_right, ket, bra) → (D_mpo_left, ket, bra, D_mpo_right)
            t_norm = np.transpose(t, (0, 2, 3, 1))
        else:
            raise ValueError(f"MPO tensor {i} has unexpected shape {t.shape}")

        normalized_tensors.append(t_norm)

    tensors = normalized_tensors

    # Save metadata
    metadata = save_tensor_metadata(filepath, tensors, model_info)

    with open(filepath, 'wb') as f:
        # Header: number of sites
        np.array([L], dtype=np.int64).tofile(f)

        # MPO bond dimensions (size L+1)
        mpo_bond_dims = [1]  # Left boundary
        for t in tensors:
            mpo_bond_dims.append(t.shape[3])  # Right MPO bond
        np.array(mpo_bond_dims, dtype=np.int64).tofile(f)

        # Physical dimensions (size L)
        phys_dims = [t.shape[1] for t in tensors]
        np.array(phys_dims, dtype=np.int64).tofile(f)

        # Write each tensor
        for i, t in enumerate(tensors):
            # Verify shape is (D_mpo_left, d, d, D_mpo_right)
            assert len(t.shape) == 4, f"MPO tensor {i} must be rank-4, got shape {t.shape}"
            assert t.shape[1] == t.shape[2], f"MPO tensor {i} must have matching physical dims"

            # Write shape
            np.array(t.shape, dtype=np.int64).tofile(f)

            # Write data in C-contiguous order
            t_contig = np.ascontiguousarray(t)
            t_contig.tofile(f)

    total_mb = filepath.stat().st_size / 1024**2
    print(f"  MPO saved: {filepath} ({total_mb:.2f} MB)")
    print(f"  Sites: {L}, MPO bond dims: {mpo_bond_dims}, Phys dims: {phys_dims}")

    return metadata


def build_josephson_mpo(L, E_J=1.0, E_C=0.5, mu=0.0, n_max=2):
    """Build Josephson junction array MPO (same as cpu_gpu_benchmark.py)."""
    d = 2 * n_max + 1

    charges = np.arange(-n_max, n_max + 1, dtype=np.float64)
    n_op = np.diag(charges.astype(np.complex128))

    exp_iphi = np.zeros((d, d), dtype=np.complex128)
    for i in range(d - 1):
        exp_iphi[i + 1, i] = 1.0 + 0j

    exp_miphi = exp_iphi.conj().T

    S = (d - 1) / 2
    builder = qtn.SpinHam1D(S=S)

    phi_ext = np.pi / 4
    flux_phase = np.exp(1j * phi_ext)

    builder.add_term(-E_J / 2 * flux_phase, exp_iphi, exp_miphi)
    builder.add_term(-E_J / 2 * np.conj(flux_phase), exp_miphi, exp_iphi)

    n2 = n_op @ n_op
    builder.add_term(E_C, n2)

    if mu != 0:
        builder.add_term(-mu, n_op)

    return builder.build_mpo(L)


def generate_heisenberg_data(L, chi_init, output_dir, seed=None):
    """Generate and save Heisenberg model MPS and MPO."""
    print(f"\n{'='*70}")
    print(f"Generating Heisenberg Model (L={L}, chi_init={chi_init})")
    print(f"{'='*70}")

    if seed is not None:
        np.random.seed(seed)

    # Build MPO
    mpo = qtn.MPO_ham_heis(L=L, j=1.0, bz=0.0, cyclic=False)

    # Generate random initial MPS with bond dimension chi_init
    mps = qtn.MPS_rand_state(L=L, bond_dim=chi_init, phys_dim=2, dtype=complex)
    mps.normalize()

    # Save
    model_info = {
        "model": "heisenberg",
        "L": L,
        "d": 2,
        "chi_init": chi_init,
        "j": 1.0,
        "bz": 0.0,
        "cyclic": False,
        "seed": seed,
    }

    mps_path = output_dir / f"heisenberg_L{L}_chi{chi_init}_mps.bin"
    mpo_path = output_dir / f"heisenberg_L{L}_mpo.bin"

    save_mps_to_binary(mps, mps_path, model_info)
    save_mpo_to_binary(mpo, mpo_path, model_info)

    return mps_path, mpo_path


def generate_josephson_data(L, n_max, chi_init, output_dir, seed=None):
    """Generate and save Josephson junction MPS and MPO."""
    print(f"\n{'='*70}")
    print(f"Generating Josephson Model (L={L}, n_max={n_max}, chi_init={chi_init})")
    print(f"{'='*70}")

    if seed is not None:
        np.random.seed(seed)

    d = 2 * n_max + 1

    # Build MPO
    mpo = build_josephson_mpo(L, n_max=n_max)

    # Generate random initial MPS
    mps = qtn.MPS_rand_state(L=L, bond_dim=chi_init, phys_dim=d, dtype=complex)
    mps.normalize()

    # Save
    model_info = {
        "model": "josephson",
        "L": L,
        "d": d,
        "n_max": n_max,
        "chi_init": chi_init,
        "E_J": 1.0,
        "E_C": 0.5,
        "mu": 0.0,
        "phi_ext": np.pi / 4,
        "seed": seed,
    }

    mps_path = output_dir / f"josephson_L{L}_n{n_max}_chi{chi_init}_mps.bin"
    mpo_path = output_dir / f"josephson_L{L}_n{n_max}_mpo.bin"

    save_mps_to_binary(mps, mps_path, model_info)
    save_mpo_to_binary(mpo, mpo_path, model_info)

    return mps_path, mpo_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate and serialize MPS/MPO for CPU/GPU benchmarks")
    parser.add_argument("--output-dir", type=str, default="benchmark_data",
                        help="Output directory for binary files")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--heisenberg", action="store_true",
                        help="Generate Heisenberg test cases")
    parser.add_argument("--josephson", action="store_true",
                        help="Generate Josephson test cases")
    parser.add_argument("--all", action="store_true",
                        help="Generate all test cases")
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Output directory: {output_dir.absolute()}")
    print(f"Random seed: {args.seed}")

    generated_files = []

    # Heisenberg test cases (matching cpu_gpu_benchmark.py)
    if args.heisenberg or args.all:
        heisenberg_cases = [
            {"L": 12, "chi_init": 10},
            {"L": 20, "chi_init": 10},
            {"L": 40, "chi_init": 20},
        ]

        for case in heisenberg_cases:
            mps_path, mpo_path = generate_heisenberg_data(
                L=case["L"],
                chi_init=case["chi_init"],
                output_dir=output_dir,
                seed=args.seed
            )
            generated_files.extend([mps_path, mpo_path])

    # Josephson test cases (matching cpu_gpu_benchmark.py)
    if args.josephson or args.all:
        josephson_cases = [
            {"L": 8, "n_max": 2, "chi_init": 10},
            {"L": 12, "n_max": 2, "chi_init": 10},
            {"L": 16, "n_max": 2, "chi_init": 20},
        ]

        for case in josephson_cases:
            mps_path, mpo_path = generate_josephson_data(
                L=case["L"],
                n_max=case["n_max"],
                chi_init=case["chi_init"],
                output_dir=output_dir,
                seed=args.seed
            )
            generated_files.extend([mps_path, mpo_path])

    # Save index of all generated files
    index = {
        "seed": args.seed,
        "files": [str(f) for f in generated_files],
    }

    index_path = output_dir / "index.json"
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Generation complete!")
    print(f"{'='*70}")
    print(f"Generated {len(generated_files)} files")
    print(f"Index: {index_path}")
    print("\nTo use in benchmarks:")
    print(f"  CPU: python cpu_gpu_benchmark.py --load-data {output_dir}")
    print(f"  GPU: ./benchmark --load-data {output_dir}")


if __name__ == "__main__":
    main()
