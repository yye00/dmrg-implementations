#!/usr/bin/env python3
"""
MPS/MPO Binary File Loader
===========================

Loads MPS and MPO tensors from binary files created by serialize_mps_mpo.py.

Can be used to:
1. Verify serialization/deserialization correctness
2. Load data into CPU benchmarks for reproducibility
3. Inspect saved tensor data
"""

import numpy as np
import json
from pathlib import Path


def load_mps_from_binary(filepath):
    """
    Load MPS tensors from binary file.

    Parameters
    ----------
    filepath : str or Path
        Path to .bin file

    Returns
    -------
    tensors : list of ndarray
        List of MPS tensors, each with shape (D_left, d, D_right)
    metadata : dict
        Metadata loaded from corresponding .json file
    """
    filepath = Path(filepath)

    with open(filepath, 'rb') as f:
        # Read header
        num_sites = np.fromfile(f, dtype=np.int64, count=1)[0]

        # Read bond dimensions
        bond_dims = np.fromfile(f, dtype=np.int64, count=num_sites + 1)

        # Read physical dimensions
        phys_dims = np.fromfile(f, dtype=np.int64, count=num_sites)

        # Read each tensor
        tensors = []
        for site in range(num_sites):
            # Read shape
            shape = tuple(np.fromfile(f, dtype=np.int64, count=3))

            # Validate
            assert shape[0] == bond_dims[site], f"Bond dim mismatch at site {site}"
            assert shape[2] == bond_dims[site + 1], f"Bond dim mismatch at site {site}"
            assert shape[1] == phys_dims[site], f"Phys dim mismatch at site {site}"

            # Read data
            num_elements = np.prod(shape)
            data = np.fromfile(f, dtype=np.complex128, count=num_elements)

            # Reshape to (D_left, d, D_right)
            tensor = data.reshape(shape)
            tensors.append(tensor)

    # Load metadata if available
    metadata = {}
    meta_path = filepath.with_suffix('.json')
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            metadata = json.load(f)

    print(f"Loaded MPS from {filepath}")
    print(f"  Sites: {num_sites}")
    print(f"  Bond dims: {list(bond_dims)}")
    print(f"  Phys dims: {list(phys_dims)}")

    return tensors, metadata


def load_mpo_from_binary(filepath):
    """
    Load MPO tensors from binary file.

    Parameters
    ----------
    filepath : str or Path
        Path to .bin file

    Returns
    -------
    tensors : list of ndarray
        List of MPO tensors, each with shape (D_mpo_left, d, d, D_mpo_right)
    metadata : dict
        Metadata loaded from corresponding .json file
    """
    filepath = Path(filepath)

    with open(filepath, 'rb') as f:
        # Read header
        num_sites = np.fromfile(f, dtype=np.int64, count=1)[0]

        # Read MPO bond dimensions
        mpo_bond_dims = np.fromfile(f, dtype=np.int64, count=num_sites + 1)

        # Read physical dimensions
        phys_dims = np.fromfile(f, dtype=np.int64, count=num_sites)

        # Read each tensor
        tensors = []
        for site in range(num_sites):
            # Read shape
            shape = tuple(np.fromfile(f, dtype=np.int64, count=4))

            # Validate
            assert shape[0] == mpo_bond_dims[site], f"MPO bond dim mismatch at site {site}"
            assert shape[3] == mpo_bond_dims[site + 1], f"MPO bond dim mismatch at site {site}"
            assert shape[1] == phys_dims[site], f"Phys dim mismatch at site {site}"
            assert shape[2] == phys_dims[site], f"Phys dim mismatch at site {site}"

            # Read data
            num_elements = np.prod(shape)
            data = np.fromfile(f, dtype=np.complex128, count=num_elements)

            # Reshape to (D_mpo_left, d, d, D_mpo_right)
            tensor = data.reshape(shape)
            tensors.append(tensor)

    # Load metadata if available
    metadata = {}
    meta_path = filepath.with_suffix('.json')
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            metadata = json.load(f)

    print(f"Loaded MPO from {filepath}")
    print(f"  Sites: {num_sites}")
    print(f"  MPO bond dims: {list(mpo_bond_dims)}")
    print(f"  Phys dims: {list(phys_dims)}")

    return tensors, metadata


def convert_to_quimb_mps(tensors):
    """
    Convert list of tensor arrays to quimb MPS object.

    Parameters
    ----------
    tensors : list of ndarray
        List of MPS tensors with shape (D_left, d, D_right)

    Returns
    -------
    mps : quimb MPS
        Quimb MPS object
    """
    import quimb.tensor as qtn

    # Create MPS from tensor arrays
    # Quimb expects tensors in a specific index ordering
    L = len(tensors)

    # Create a product state MPS first, then replace tensors
    mps = qtn.MPS_computational_state('0' * L)

    # Replace tensors with loaded data
    # Need to transpose from our convention to quimb's convention
    for i, t in enumerate(tensors):
        mps.tensors[i].modify(data=t)

    return mps


def convert_to_quimb_mpo(tensors):
    """
    Convert list of tensor arrays to quimb MPO object.

    Parameters
    ----------
    tensors : list of ndarray
        List of MPO tensors with shape (D_mpo_left, d_bra, d_ket, D_mpo_right)

    Returns
    -------
    mpo : quimb MPO
        Quimb MPO object
    """
    import quimb.tensor as qtn

    # Quimb MPO expects (bra, ket, bond_left, bond_right)
    # We have (bond_left, bra, ket, bond_right)
    # Need to transpose to quimb convention

    L = len(tensors)
    d = tensors[0].shape[1]  # Physical dimension

    # Create identity MPO first as template
    mpo = qtn.MPO_identity(L=L, phys_dim=d)

    # Replace tensors with loaded data (after transposing)
    for i, t in enumerate(tensors):
        # Transpose from (bond_left, bra, ket, bond_right) to (bra, ket, bond_left, bond_right)
        t_quimb = np.transpose(t, (1, 2, 0, 3))
        mpo.tensors[i].modify(data=t_quimb)

    return mpo


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Load and inspect MPS/MPO binary files")
    parser.add_argument("file", help="Path to .bin file")
    parser.add_argument("--type", choices=["mps", "mpo"], required=True,
                        help="Type of tensor")
    parser.add_argument("--to-quimb", action="store_true",
                        help="Convert to quimb object")
    args = parser.parse_args()

    if args.type == "mps":
        tensors, metadata = load_mps_from_binary(args.file)
        print(f"\nMetadata: {metadata}")
        print(f"\nTensor shapes:")
        for i, t in enumerate(tensors):
            print(f"  Site {i}: {t.shape}")

        if args.to_quimb:
            mps = convert_to_quimb_mps(tensors)
            print(f"\nConverted to quimb MPS")
            print(f"  Bond dimensions: {mps.bond_sizes()}")

    elif args.type == "mpo":
        tensors, metadata = load_mpo_from_binary(args.file)
        print(f"\nMetadata: {metadata}")
        print(f"\nTensor shapes:")
        for i, t in enumerate(tensors):
            print(f"  Site {i}: {t.shape}")

        if args.to_quimb:
            mpo = convert_to_quimb_mpo(tensors)
            print(f"\nConverted to quimb MPO")
