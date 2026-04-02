"""
MPS initialization and utility functions.

This module provides functions for creating and manipulating Matrix Product States
using quimb as the underlying tensor network library.

Quimb MPS tensor index conventions:
- First site (i=0): shape = (bond_right, phys)
- Middle sites: shape = (bond_left, bond_right, phys)
- Last site (i=L-1): shape = (bond_left, phys)
"""

import numpy as np
import quimb.tensor as qtn


def create_random_mps(L, bond_dim, phys_dim=2, dtype='float64', canonical='left'):
    """
    Create a random Matrix Product State in canonical form.

    Parameters
    ----------
    L : int
        Number of sites (length of the chain)
    bond_dim : int
        Maximum bond dimension (chi)
    phys_dim : int, optional
        Physical dimension at each site (default: 2 for spin-1/2)
    dtype : str or dtype, optional
        Data type for tensors ('float64' or 'complex128')
    canonical : str, optional
        Canonical form: 'left', 'right', or None (default: 'left')

    Returns
    -------
    mps : MatrixProductState
        Random MPS in the specified canonical form

    Notes
    -----
    - Left-canonical: All tensors except rightmost satisfy Σ_s A^s† A^s = I
    - Right-canonical: All tensors except leftmost satisfy Σ_s B^s B^{s†} = I
    - The MPS is normalized (⟨ψ|ψ⟩ = 1)

    Examples
    --------
    >>> mps = create_random_mps(L=10, bond_dim=4, dtype='float64')
    >>> mps.L  # Number of sites
    10
    >>> mps = create_random_mps(L=10, bond_dim=4, dtype='complex128')
    >>> mps[0].dtype
    dtype('complex128')
    """
    # Create random MPS using quimb
    mps = qtn.MPS_rand_state(
        L=L,
        bond_dim=bond_dim,
        phys_dim=phys_dim,
        dtype=dtype,
        cyclic=False,
        normalize=True
    )

    # Apply canonicalization if requested
    if canonical == 'left':
        mps.left_canonize()
    elif canonical == 'right':
        mps.right_canonize()
    elif canonical is None:
        pass  # Keep as-is (random, non-canonical)
    else:
        raise ValueError(f"Unknown canonical form: {canonical}")

    return mps


def verify_left_canonical(mps, tol=1e-10):
    """
    Verify that an MPS is in left-canonical form.

    Parameters
    ----------
    mps : MatrixProductState
        MPS to verify
    tol : float, optional
        Tolerance for numerical errors (default: 1e-10)

    Returns
    -------
    is_canonical : bool
        True if MPS is left-canonical within tolerance
    max_error : float
        Maximum deviation from orthogonality

    Notes
    -----
    For left-canonical form, all tensors except the rightmost should satisfy:
        Σ_s A[i][:, :, s].H @ A[i][:, :, s] == I (sum over physical index)

    Quimb tensor shapes:
    - First site: (right_bond, phys)
    - Middle sites: (left_bond, right_bond, phys)
    - Last site: (left_bond, phys)
    """
    max_error = 0.0
    L = mps.L

    # Check all sites except the last one
    for i in range(L - 1):
        tensor = mps[i].data  # Get numpy array from quimb Tensor

        if tensor.ndim == 2:
            # First site: shape (right_bond, phys)
            # For left-canonical: contract over phys, should give identity on right_bond
            right_bond, phys = tensor.shape
            # Reshape to (phys, right_bond) then check orthogonality
            mat = tensor.T  # Shape: (phys, right_bond)
            product = mat.conj().T @ mat  # Should be identity (right_bond, right_bond)
            identity = np.eye(right_bond, dtype=tensor.dtype)

        elif tensor.ndim == 3:
            # Middle sites: shape (left_bond, right_bond, phys)
            # Reshape to (left_bond * phys, right_bond), check A†A = I
            left_bond, right_bond, phys = tensor.shape
            mat = tensor.transpose(0, 2, 1).reshape(left_bond * phys, right_bond)
            product = mat.conj().T @ mat
            identity = np.eye(right_bond, dtype=tensor.dtype)

        else:
            raise ValueError(f"Unexpected tensor rank {tensor.ndim} at site {i}")

        error = np.linalg.norm(product - identity)
        max_error = max(max_error, error)

    is_canonical = max_error < tol
    return is_canonical, max_error


def verify_right_canonical(mps, tol=1e-10):
    """
    Verify that an MPS is in right-canonical form.

    Parameters
    ----------
    mps : MatrixProductState
        MPS to verify
    tol : float, optional
        Tolerance for numerical errors (default: 1e-10)

    Returns
    -------
    is_canonical : bool
        True if MPS is right-canonical within tolerance
    max_error : float
        Maximum deviation from orthogonality

    Notes
    -----
    For right-canonical form, all tensors except the leftmost should satisfy:
        Σ_s B[i][:, :, s] @ B[i][:, :, s].H == I (sum over physical index)

    Quimb tensor shapes:
    - First site: (right_bond, phys)
    - Middle sites: (left_bond, right_bond, phys)
    - Last site: (left_bond, phys)
    """
    max_error = 0.0
    L = mps.L

    # Check all sites except the first one
    for i in range(1, L):
        tensor = mps[i].data  # Get numpy array from quimb Tensor

        if tensor.ndim == 2:
            # Last site: shape (left_bond, phys)
            # For right-canonical: B B† = I on left_bond
            left_bond, phys = tensor.shape
            mat = tensor  # Shape: (left_bond, phys)
            product = mat @ mat.conj().T  # Should be identity (left_bond, left_bond)
            identity = np.eye(left_bond, dtype=tensor.dtype)

        elif tensor.ndim == 3:
            # Middle sites: shape (left_bond, right_bond, phys)
            # Reshape to (left_bond, right_bond * phys), check B B† = I
            left_bond, right_bond, phys = tensor.shape
            mat = tensor.reshape(left_bond, right_bond * phys)
            product = mat @ mat.conj().T
            identity = np.eye(left_bond, dtype=tensor.dtype)

        else:
            raise ValueError(f"Unexpected tensor rank {tensor.ndim} at site {i}")

        error = np.linalg.norm(product - identity)
        max_error = max(max_error, error)

    is_canonical = max_error < tol
    return is_canonical, max_error


def create_neel_state(L, bond_dim=None, dtype='float64'):
    """
    Create a Neel state MPS: |↑↓↑↓↑↓...⟩ for spin-1/2 chain.

    The Neel state is a classical product state with alternating spin up and spin down.
    For the Heisenberg model, this gives energy around E ≈ -0.25*L (negative!), which
    is much closer to the ground state than a random state.

    Parameters
    ----------
    L : int
        Number of sites (length of the chain)
    bond_dim : int, optional
        Maximum bond dimension (for consistency with other MPS, but Neel state
        is a product state so actual bond dim = 1). If provided, tensors will be
        padded with zeros to match this bond dimension.
    dtype : str or dtype, optional
        Data type for tensors ('float64' or 'complex128')

    Returns
    -------
    mps : MatrixProductState
        Neel state MPS in left-canonical form

    Notes
    -----
    The Neel state is defined as:
    - Site i (even): spin up |0⟩ = [1, 0]
    - Site i (odd): spin down |1⟩ = [0, 1]

    For a product state, the MPS representation has bond dimension 1:
    - First site (i=0): shape = (1, 2) for spin up
    - Middle sites: shape = (1, 1, 2)
    - Last site (i=L-1): shape = (1, 2)

    Examples
    --------
    >>> mps = create_neel_state(L=10, dtype='float64')
    >>> mps.L
    10
    >>> # Neel state has negative energy for Heisenberg
    >>> # E ≈ -0.25*L, much better than random (E ≈ +0.01)
    """
    # Convert dtype string to numpy dtype
    if isinstance(dtype, str):
        dtype = np.dtype(dtype)

    # Create tensors for Neel state
    tensors = []

    for i in range(L):
        # Determine spin at this site (alternating)
        # Even sites: spin up |0⟩ = [1, 0]
        # Odd sites: spin down |1⟩ = [0, 1]
        if i % 2 == 0:
            # Spin up
            phys_state = np.array([1.0, 0.0], dtype=dtype)
        else:
            # Spin down
            phys_state = np.array([0.0, 1.0], dtype=dtype)

        # Create MPS tensor for product state (bond dim = 1)
        if i == 0:
            # First site: shape (right_bond=1, phys=2)
            tensor = phys_state.reshape(1, 2)
        elif i == L - 1:
            # Last site: shape (left_bond=1, phys=2)
            tensor = phys_state.reshape(1, 2)
        else:
            # Middle sites: shape (left_bond=1, right_bond=1, phys=2)
            tensor = phys_state.reshape(1, 1, 2)

        tensors.append(tensor)

    # Create quimb MPS from tensors
    mps = qtn.MatrixProductState(tensors, shape='lrp')

    # If bond_dim is specified and > 1, pad tensors to match target bond dimension
    if bond_dim is not None and bond_dim > 1:
        from .canonical import _pad_to_bond_dimensions

        # Prepare target bond dimensions
        target_bonds = []
        for i in range(L):
            if i == 0:
                target_bonds.append(bond_dim)  # right bond
            elif i == L - 1:
                target_bonds.append(bond_dim)  # left bond
            else:
                target_bonds.append((bond_dim, bond_dim))  # (left, right)

        _pad_to_bond_dimensions(mps, target_bonds)

        # CRITICAL FIX: Add small random noise to avoid zero effective Hamiltonians
        # The padded Neel state has exact zeros in the bond dimensions, which causes
        # ARPACK error -9 (starting vector is zero) when solving local eigenvalue problems.
        # Adding noise ~1e-10 breaks the exact product state structure while preserving
        # the Neel state as the dominant component.
        noise_level = 1e-10
        for i in range(L):
            tensor = mps.tensors[i]
            noise_shape = tensor.data.shape
            if dtype == np.complex128:
                noise = noise_level * (np.random.randn(*noise_shape) + 1j * np.random.randn(*noise_shape))
            else:
                noise = noise_level * np.random.randn(*noise_shape)
            tensor.modify(data=tensor.data + noise)

        # Renormalize MPS after adding noise
        norm = mps.norm()
        mps /= norm

    return mps


def create_product_state_mps(L, bond_dim, state_index=0, phys_dim=2, dtype='float64'):
    """
    Create a product state MPS with all sites in the same local state.

    Parameters
    ----------
    L : int
        Number of sites (length of the chain)
    bond_dim : int
        Maximum bond dimension (product state has bond_dim=1, will be padded)
    state_index : int, optional
        Index of the local state (0 to phys_dim-1). Default: 0
        For spin-1/2: 0 = spin up |↑⟩, 1 = spin down |↓⟩
        For bosons: state_index = occupation number (0, 1, 2, ...)
    phys_dim : int, optional
        Physical dimension at each site (default: 2 for spin-1/2)
        For bosons with nmax=3, phys_dim=4 (states: |0⟩, |1⟩, |2⟩, |3⟩)
    dtype : str or dtype, optional
        Data type for tensors ('float64' or 'complex128')

    Returns
    -------
    mps : MatrixProductState
        Product state MPS with all sites in |state_index⟩

    Notes
    -----
    The product state is |n,n,n,...,n⟩ where n = state_index.
    Bond dimensions are padded to bond_dim with small random noise to avoid
    numerical issues in DMRG eigensolvers.

    Examples
    --------
    >>> # Create |0,0,0,0⟩ for spin-1/2 (all spin up)
    >>> mps = create_product_state_mps(L=4, bond_dim=8, state_index=0, phys_dim=2)

    >>> # Create |1,1,1,1,1⟩ for bosons (one boson per site)
    >>> mps = create_product_state_mps(L=5, bond_dim=8, state_index=1, phys_dim=4)
    """
    # Convert dtype string to numpy dtype
    if isinstance(dtype, str):
        dtype = np.dtype(dtype)

    # Validate state_index
    if state_index < 0 or state_index >= phys_dim:
        raise ValueError(f"state_index={state_index} must be in range [0, {phys_dim-1}]")

    # Create local state vector |state_index⟩
    phys_state = np.zeros(phys_dim, dtype=dtype)
    phys_state[state_index] = 1.0

    # Create tensors for product state
    tensors = []

    for i in range(L):
        # Create MPS tensor for product state (bond dim = 1)
        if i == 0:
            # First site: shape (right_bond=1, phys)
            tensor = phys_state.reshape(1, phys_dim)
        elif i == L - 1:
            # Last site: shape (left_bond=1, phys)
            tensor = phys_state.reshape(1, phys_dim)
        else:
            # Middle sites: shape (left_bond=1, right_bond=1, phys)
            tensor = phys_state.reshape(1, 1, phys_dim)

        tensors.append(tensor)

    # Create quimb MPS from tensors
    mps = qtn.MatrixProductState(tensors, shape='lrp')

    # If bond_dim > 1, pad tensors to match target bond dimension
    if bond_dim > 1:
        from .canonical import _pad_to_bond_dimensions

        # Prepare target bond dimensions
        target_bonds = []
        for i in range(L):
            if i == 0:
                target_bonds.append(bond_dim)  # right bond
            elif i == L - 1:
                target_bonds.append(bond_dim)  # left bond
            else:
                target_bonds.append((bond_dim, bond_dim))  # (left, right)

        _pad_to_bond_dimensions(mps, target_bonds)

        # Add small random noise to avoid zero effective Hamiltonians
        # This prevents ARPACK error -9 (starting vector is zero)
        noise_level = 1e-10
        for i in range(L):
            tensor = mps.tensors[i]
            noise_shape = tensor.data.shape
            if dtype == np.complex128:
                noise = noise_level * (np.random.randn(*noise_shape) + 1j * np.random.randn(*noise_shape))
            else:
                noise = noise_level * np.random.randn(*noise_shape)
            tensor.modify(data=tensor.data + noise)

        # Renormalize MPS after adding noise
        norm = mps.norm()
        mps /= norm

    return mps


def get_mps_norm(mps):
    """
    Compute the norm of an MPS: ||ψ|| = sqrt(⟨ψ|ψ⟩).

    Parameters
    ----------
    mps : MatrixProductState
        MPS to compute norm of

    Returns
    -------
    norm : float
        Norm of the MPS
    """
    return mps.norm()
