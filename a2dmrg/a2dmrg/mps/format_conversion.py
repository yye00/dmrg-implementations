"""
MPS format conversion utilities.

This module provides functions to convert between different MPS tensor formats,
particularly for converting quimb DMRG2 output to A2DMRG-compatible format.
"""

import numpy as np
import quimb.tensor as qtn
from .canonical import _pad_to_bond_dimensions


def convert_quimb_dmrg_to_a2dmrg_format(mps_dmrg, target_bond_dim):
    """
    Convert quimb DMRG2 MPS to A2DMRG-compatible format with uniform bonds.

    Quimb DMRG2 produces MPS with:
    - Variable bond dimensions (adaptively reduced)
    - Physical index ordering: (bond_left, phys, bond_right) for middle sites

    A2DMRG requires:
    - Uniform bond dimensions for parallel decomposition
    - Physical index ordering: (bond_left, bond_right, phys) for middle sites

    Parameters
    ----------
    mps_dmrg : quimb.tensor.MatrixProductState
        MPS from quimb DMRG (may have variable bonds and different index order)
    target_bond_dim : int
        Target uniform bond dimension for A2DMRG

    Returns
    -------
    mps_a2dmrg : quimb.tensor.MatrixProductState
        MPS in A2DMRG-compatible format with uniform bond dimensions

    Notes
    -----
    The conversion performs two steps:
    1. Transpose tensors to match A2DMRG index ordering
    2. Pad bond dimensions to uniform size with zeros

    The padding is safe because:
    - Padded dimensions represent "empty" subspace
    - After A2DMRG compression, zero-filled dimensions are removed

    Examples
    --------
    >>> from quimb.tensor import DMRG2, MPO_ham_heis
    >>> mpo = MPO_ham_heis(6, j=1.0)
    >>> dmrg = DMRG2(mpo, bond_dims=20)
    >>> dmrg.solve(tol=1e-8)
    >>> mps_warmstart = convert_quimb_dmrg_to_a2dmrg_format(dmrg.state, 20)
    """
    L = mps_dmrg.L

    # Extract tensors and convert to A2DMRG format
    tensors_a2dmrg = []

    for i in range(L):
        tensor = mps_dmrg[i].data.copy()  # Get numpy array and make a copy

        if i == 0:
            # DMRG2 format: (phys, bond_right)
            # A2DMRG format: (bond_right, phys)
            if tensor.ndim == 2:
                phys, bond_right = tensor.shape
                tensor_a2dmrg = tensor.T  # Transpose to (bond_right, phys)
            else:
                raise ValueError(f"Unexpected tensor shape at site 0: {tensor.shape}")

        elif i == L - 1:
            # DMRG2 format: (bond_left, phys)
            # A2DMRG format: (bond_left, phys) - SAME!
            if tensor.ndim == 2:
                tensor_a2dmrg = tensor  # No transpose needed
            else:
                raise ValueError(f"Unexpected tensor shape at site {i}: {tensor.shape}")

        else:
            # DMRG2 format: (bond_left, phys, bond_right)
            # A2DMRG format: (bond_left, bond_right, phys)
            if tensor.ndim == 3:
                bond_left, phys, bond_right = tensor.shape
                # Transpose: (0, 1, 2) -> (0, 2, 1)
                tensor_a2dmrg = tensor.transpose(0, 2, 1)  # -> (bond_left, bond_right, phys)
            else:
                raise ValueError(f"Unexpected tensor shape at site {i}: {tensor.shape}")

        tensors_a2dmrg.append(tensor_a2dmrg)

    # Create new MPS from converted tensors
    # Use shape='lrp' to specify (left_bond, right_bond, phys) ordering
    mps_a2dmrg = qtn.MatrixProductState(tensors_a2dmrg, shape='lrp')

    # Now pad to uniform bond dimensions
    # Calculate target bond dimensions for each site
    target_bonds = []
    for i in range(L):
        if i == 0:
            # First site: only right bond
            target_bonds.append(target_bond_dim)
        elif i == L - 1:
            # Last site: only left bond
            target_bonds.append(target_bond_dim)
        else:
            # Middle sites: (left_bond, right_bond)
            target_bonds.append((target_bond_dim, target_bond_dim))

    # Apply padding
    _pad_to_bond_dimensions(mps_a2dmrg, target_bonds)

    return mps_a2dmrg
