"""
Canonical form transformations for Matrix Product States.

This module implements conversions between different canonical forms:
- Left-canonical (left-orthogonal)
- Right-canonical (right-orthogonal)
- i-orthogonal (orthogonality center at site i)
- TT-SVD compression to project back to rank manifold
"""

import numpy as np
from quimb.tensor import MatrixProductState
from typing import Optional
from ..numerics.truncated_svd import truncated_svd


def left_canonicalize(mps, normalize=True):
    """
    Convert MPS to left-canonical form via QR decomposition.

    Parameters
    ----------
    mps : MatrixProductState
        MPS to canonicalize (modified in-place)
    normalize : bool, optional
        Whether to normalize the MPS (default: True)

    Returns
    -------
    mps : MatrixProductState
        The same MPS object, now in left-canonical form

    Notes
    -----
    In left-canonical form, all tensors except the rightmost satisfy:
        Σ_s A[i][:, s, :].H @ A[i][:, s, :] == I

    This is achieved by sweeping left to right with QR decomposition.
    The norm accumulates in the rightmost tensor.
    """
    mps.left_canonize(normalize=normalize)
    return mps


def right_canonicalize(mps, normalize=True):
    """
    Convert MPS to right-canonical form via LQ decomposition.

    Parameters
    ----------
    mps : MatrixProductState
        MPS to canonicalize (modified in-place)
    normalize : bool, optional
        Whether to normalize the MPS (default: True)

    Returns
    -------
    mps : MatrixProductState
        The same MPS object, now in right-canonical form

    Notes
    -----
    In right-canonical form, all tensors except the leftmost satisfy:
        Σ_s B[i][:, s, :] @ B[i][:, s, :].H == I

    This is achieved by sweeping right to left with LQ decomposition.
    The norm accumulates in the leftmost tensor.
    """
    mps.right_canonize(normalize=normalize)
    return mps


def move_orthogonality_center(mps, site, normalize=True):
    """
    Create i-orthogonal decomposition with orthogonality center at specified site.

    Parameters
    ----------
    mps : MatrixProductState
        MPS to transform (modified in-place)
    site : int
        Site index where orthogonality center should be placed (0 <= site < L)
    normalize : bool, optional
        Whether to normalize the MPS (default: True)

    Returns
    -------
    mps : MatrixProductState
        The same MPS object, now with orthogonality center at specified site

    Notes
    -----
    In i-orthogonal form with center at site i:
    - Sites 0..i-1 are left-orthogonal: Σ_s A^s† A^s = I
    - Sites i+1..L-1 are right-orthogonal: Σ_s B^s B^{s†} = I
    - Site i contains the orthogonality center (and the norm)

    This form is crucial for A2DMRG local micro-steps, as it allows
    efficient updates of individual sites.
    """
    if not (0 <= site < mps.L):
        raise ValueError(f"Site {site} out of range for MPS with L={mps.L}")

    # quimb's canonize method with 'where' parameter does exactly this
    mps.canonize(where=site)

    # Normalize if requested
    if normalize:
        norm = mps.norm()
        if norm > 0:
            mps /= norm

    return mps


def prepare_orthogonal_decompositions(mps, sites=None):
    """
    Prepare multiple i-orthogonal decompositions for parallel A2DMRG.

    Parameters
    ----------
    mps : MatrixProductState
        Base MPS to create decompositions from
    sites : list of int, optional
        Sites to create orthogonality centers at.
        If None, creates decompositions for all sites.

    Returns
    -------
    decompositions : list of MatrixProductState
        List of MPS copies, each with orthogonality center at different site

    Notes
    -----
    This is Phase 1 of the A2DMRG algorithm. Each MPI rank will receive
    one or more of these decompositions for parallel local micro-steps.

    The decompositions are deep copies, so they can be modified independently.

    CRITICAL for A2DMRG: We preserve the original bond dimensions after
    canonicalization. This ensures all candidate MPS have compatible shapes
    for the coarse-space minimization step.
    """
    if sites is None:
        sites = list(range(mps.L))

    # Record original bond dimensions before any canonicalization
    L = mps.L
    original_bond_dims = []
    for i in range(L):
        tensor = mps[i].data
        if i == 0:
            # First site: (right_bond, phys)
            bond_dim = tensor.shape[0]
        elif i == L - 1:
            # Last site: (left_bond, phys)
            bond_dim = tensor.shape[0]
        else:
            # Middle site: (left_bond, right_bond, phys)
            # Record both left and right bonds
            bond_dim = (tensor.shape[0], tensor.shape[1])
        original_bond_dims.append(bond_dim)

    decompositions = []
    for site in sites:
        # Create a deep copy of the MPS
        mps_copy = mps.copy()
        # Move orthogonality center to this site
        move_orthogonality_center(mps_copy, site, normalize=True)

        # CRITICAL: Pad bonds back to original dimensions
        # Canonicalization may reduce bond dimensions, but A2DMRG requires
        # all candidate MPS to have the same bond structure
        _pad_to_bond_dimensions(mps_copy, original_bond_dims)

        decompositions.append(mps_copy)

    return decompositions


def pad_to_uniform_bond_dim(mps, target_bond_dim):
    """
    Pad MPS tensors to have uniform bond dimension throughout.
    
    This is necessary after quimb's compress() which may produce
    variable bond dimensions depending on singular value truncation.
    
    Parameters
    ----------
    mps : MatrixProductState
        MPS to pad (modified in-place)
    target_bond_dim : int
        Target bond dimension for all interior bonds
        
    Notes
    -----
    Edge bonds (connecting to site 0 and site L-1) may be smaller
    depending on the physical dimension.
    
    Zero-padding preserves the state mathematically (the padded dimensions
    represent empty subspace). This is needed for A2DMRG to ensure
    all candidate MPS have the same bond structure.
    """
    L = mps.L
    
    for i in range(L):
        tensor = mps[i]
        data = tensor.data
        inds = tensor.inds
        
        # Find physical index
        phys_name = f'k{i}'
        phys_pos = None
        for idx, ind in enumerate(inds):
            if ind.startswith('k'):
                phys_pos = idx
                break
        
        if data.ndim == 2:
            # Edge site
            if i == 0:
                # First site: shape could be (phys, right) or (right, phys)
                if phys_pos == 0:
                    phys, right = data.shape
                    if right < target_bond_dim:
                        padded = np.zeros((phys, target_bond_dim), dtype=data.dtype)
                        padded[:, :right] = data
                        tensor.modify(data=padded)
                else:
                    right, phys = data.shape
                    if right < target_bond_dim:
                        padded = np.zeros((target_bond_dim, phys), dtype=data.dtype)
                        padded[:right, :] = data
                        tensor.modify(data=padded)
            else:
                # Last site: shape could be (left, phys) or (phys, left)
                if phys_pos == 1:
                    left, phys = data.shape
                    if left < target_bond_dim:
                        padded = np.zeros((target_bond_dim, phys), dtype=data.dtype)
                        padded[:left, :] = data
                        tensor.modify(data=padded)
                else:
                    phys, left = data.shape
                    if left < target_bond_dim:
                        padded = np.zeros((phys, target_bond_dim), dtype=data.dtype)
                        padded[:, :left] = data
                        tensor.modify(data=padded)
        
        elif data.ndim == 3:
            # Middle site: find bond positions
            bond_pos = [j for j in range(3) if j != phys_pos]
            
            # Get current shape
            shape = list(data.shape)
            needs_padding = any(shape[j] < target_bond_dim for j in bond_pos)
            
            if needs_padding:
                new_shape = list(shape)
                for j in bond_pos:
                    new_shape[j] = max(shape[j], target_bond_dim)
                
                padded = np.zeros(new_shape, dtype=data.dtype)
                
                # Build slice for original data
                slices = [slice(None)] * 3
                for j in bond_pos:
                    slices[j] = slice(0, shape[j])
                
                padded[tuple(slices)] = data
                tensor.modify(data=padded)


def _pad_to_bond_dimensions(mps, target_bond_dims):
    """
    Pad MPS tensors to match target bond dimensions.

    Parameters
    ----------
    mps : MatrixProductState
        MPS to pad (modified in-place)
    target_bond_dims : list
        Target bond dimensions for each site

    Notes
    -----
    CRITICAL: We pad with zeros. This is acceptable for A2DMRG because:
    1. The padded dimensions represent "empty" subspace
    2. The coarse-space linear combination will not use these dimensions
    3. After compression, these zero-filled dimensions will be removed

    The padding is necessary to ensure all candidate MPS have the same
    bond structure for the parallel local micro-steps. Without it, the
    effective Hamiltonian dimensions don't match the environment tensors.
    """
    L = mps.L
    for i in range(L):
        tensor = mps[i].data

        if i == 0:
            # First site: (right_bond, phys)
            curr_right, phys = tensor.shape
            target_right = target_bond_dims[i]

            if curr_right < target_right:
                # Pad right bond
                padded = np.zeros((target_right, phys), dtype=tensor.dtype)
                padded[:curr_right, :] = tensor
                mps[i].modify(data=padded)

        elif i == L - 1:
            # Last site: (left_bond, phys)
            curr_left, phys = tensor.shape
            target_left = target_bond_dims[i]

            if curr_left < target_left:
                # Pad left bond
                padded = np.zeros((target_left, phys), dtype=tensor.dtype)
                padded[:curr_left, :] = tensor
                mps[i].modify(data=padded)

        else:
            # Middle site: (left_bond, right_bond, phys)
            curr_left, curr_right, phys = tensor.shape
            target_left, target_right = target_bond_dims[i]

            if curr_left < target_left or curr_right < target_right:
                # Pad both bonds
                new_left = max(curr_left, target_left)
                new_right = max(curr_right, target_right)
                padded = np.zeros((new_left, new_right, phys), dtype=tensor.dtype)
                padded[:curr_left, :curr_right, :] = tensor
                mps[i].modify(data=padded)


# Public alias for cross-module use
pad_to_bond_dimensions = _pad_to_bond_dimensions


def verify_i_orthogonal(mps, center_site, tol=1e-10):
    """
    Verify that MPS has orthogonality center at specified site.

    Parameters
    ----------
    mps : MatrixProductState
        MPS to verify
    center_site : int
        Expected location of orthogonality center
    tol : float, optional
        Tolerance for numerical errors (default: 1e-10)

    Returns
    -------
    is_i_orthogonal : bool
        True if MPS has orthogonality center at specified site
    errors : dict
        Dictionary with 'left', 'right' keys containing maximum errors

    Notes
    -----
    Checks:
    1. Sites 0..center_site-1 are left-orthogonal
    2. Sites center_site+1..L-1 are right-orthogonal
    3. Site center_site can be arbitrary (contains the norm)

    Quimb tensor shapes:
    - First site: (right_bond, phys)
    - Middle sites: (left_bond, right_bond, phys)
    - Last site: (left_bond, phys)
    """
    L = mps.L
    errors = {'left': 0.0, 'right': 0.0}

    # Check left-orthogonality for sites before center
    for i in range(center_site):
        tensor = mps[i].data

        if tensor.ndim == 2:
            # First site: shape (right_bond, phys)
            right_bond, phys = tensor.shape
            mat = tensor.T  # Shape: (phys, right_bond)
            product = mat.conj().T @ mat
            identity = np.eye(right_bond, dtype=tensor.dtype)
        elif tensor.ndim == 3:
            # Middle sites: shape (left_bond, right_bond, phys)
            left_bond, right_bond, phys = tensor.shape
            mat = tensor.transpose(0, 2, 1).reshape(left_bond * phys, right_bond)
            product = mat.conj().T @ mat
            identity = np.eye(right_bond, dtype=tensor.dtype)
        else:
            raise ValueError(f"Unexpected tensor rank {tensor.ndim} at site {i}")

        error = np.linalg.norm(product - identity)
        errors['left'] = max(errors['left'], error)

    # Check right-orthogonality for sites after center
    for i in range(center_site + 1, L):
        tensor = mps[i].data

        if tensor.ndim == 2:
            # Last site: shape (left_bond, phys)
            left_bond, phys = tensor.shape
            mat = tensor  # Shape: (left_bond, phys)
            product = mat @ mat.conj().T
            identity = np.eye(left_bond, dtype=tensor.dtype)
        elif tensor.ndim == 3:
            # Middle sites: shape (left_bond, right_bond, phys)
            left_bond, right_bond, phys = tensor.shape
            mat = tensor.reshape(left_bond, right_bond * phys)
            product = mat @ mat.conj().T
            identity = np.eye(left_bond, dtype=tensor.dtype)
        else:
            raise ValueError(f"Unexpected tensor rank {tensor.ndim} at site {i}")

        error = np.linalg.norm(product - identity)
        errors['right'] = max(errors['right'], error)

    is_i_orthogonal = (errors['left'] < tol) and (errors['right'] < tol)
    return is_i_orthogonal, errors


def compress_mps(mps, max_rank=None, tol=None, normalize=True, pad_bonds=True):
    """
    Compress MPS using TT-SVD algorithm to project back to rank manifold.

    This implements Phase 4 of A2DMRG: after forming the linear combination
    of candidate states, the TT-ranks may have increased. This function
    applies the standard TT-SVD algorithm to compress the MPS back to the
    target rank manifold.

    Parameters
    ----------
    mps : MatrixProductState
        MPS to compress (modified in-place)
    max_rank : int, optional
        Maximum bond dimension to keep at each bond
    tol : float, optional
        Relative tolerance for SVD truncation (keep singular values >= sigma_max * tol)
    normalize : bool, optional
        Whether to normalize the MPS after compression (default: True)
    pad_bonds : bool, optional
        Whether to pad all bonds to exactly max_rank (default: True).
        This is CRITICAL for A2DMRG to ensure uniform bond dimensions across
        all candidate MPS, which is required for subsequent sweeps to work correctly.

    Returns
    -------
    mps : MatrixProductState
        The same MPS object, now compressed and in left-canonical form
    truncation_errors : list of float
        Truncation error at each bond (Frobenius norm of discarded singular values)

    Notes
    -----
    Algorithm (standard TT-SVD, NOT recursive accurate SVD):
    1. Start from leftmost core
    2. For each bond i = 0 to L-2:
       a. Reshape tensor U_i into matrix: M = U_i.reshape(r_{i-1} * n_i, -1)
       b. Compute truncated SVD: M = P @ diag(Σ) @ Q^H
       c. Keep top r_i singular values (determined by max_rank and/or tol)
       d. Update U_i = P[:, :r_i].reshape(r_{i-1}, n_i, r_i)
       e. Propagate remainder to next site: U_{i+1} = (diag(Σ[:r_i]) @ Q[:r_i, :]) @ U_{i+1}
    3. Result is compressed left-orthogonal MPS with bounded ranks
    4. If pad_bonds=True, pad all bonds to exactly max_rank for uniform structure

    The resulting MPS is in left-canonical form with bond dimensions controlled
    by max_rank and/or tolerance. With pad_bonds=True, all internal bonds will
    have exactly max_rank dimensions (padded with zeros if necessary), which
    ensures compatibility for A2DMRG's parallel local micro-steps.

    References
    ----------
    Oseledets, SIAM J. Sci. Comput. 33(5), 2295-2317 (2011)
    Grigori & Hassan, arXiv:2505.23429v2 (2025), Algorithm 1, Phase 4
    """
    if max_rank is None and tol is None:
        raise ValueError("Must specify at least one of max_rank or tol for compression")

    L = mps.L
    truncation_errors = []

    # Sweep from left to right, performing SVD at each bond
    for i in range(L - 1):
        # Get tensor at site i
        tensor = mps[i].data

        # Determine tensor shape based on position
        if i == 0:
            # First site: shape is (right_bond, phys) in quimb
            # Reshape to (phys, right_bond) for SVD
            if tensor.ndim == 2:
                r_right, d = tensor.shape
                M = tensor.T  # Shape: (d, r_right)
                r_left = 1
            else:
                raise ValueError(f"First tensor should be 2D, got shape {tensor.shape}")
        else:
            # Middle sites: shape is (left_bond, right_bond, phys) in quimb
            if tensor.ndim == 3:
                r_left, r_right, d = tensor.shape
                # Reshape to matrix: (r_left * d, r_right)
                M = tensor.transpose(0, 2, 1).reshape(r_left * d, r_right)
            else:
                raise ValueError(f"Middle tensor at site {i} should be 3D, got shape {tensor.shape}")

        # Perform truncated SVD
        U, S, Vh, trunc_error = truncated_svd(
            M, max_rank=max_rank, tol=tol, return_truncation_error=True
        )
        truncation_errors.append(trunc_error)

        # Determine actual rank kept
        r_new = len(S)

        # Update tensor at site i (becomes left-orthogonal)
        if i == 0:
            # Reshape back: U has shape (d, r_new)
            # Quimb expects (r_new, d) for first site
            new_tensor = U.T
        else:
            # Reshape back: U has shape (r_left * d, r_new)
            # Quimb expects (r_left, r_new, d)
            new_tensor = U.reshape(r_left, d, r_new).transpose(0, 2, 1)

        mps[i].modify(data=new_tensor)

        # Propagate S @ Vh to next site
        # S @ Vh has shape (r_new, r_right)
        next_tensor = mps[i + 1].data

        if i + 1 == L - 1:
            # Next site is last site: shape is (left_bond, phys) in quimb
            if next_tensor.ndim == 2:
                r_left_next, d_next = next_tensor.shape
                # Contract: (r_new, r_right) @ (r_right, d_next) = (r_new, d_next)
                # First reshape next_tensor from (r_left_next, d_next) to (r_left_next, d_next)
                # Then contract S*Vh @ next_tensor
                propagate = np.diag(S) @ Vh  # Shape: (r_new, r_right)
                new_next = propagate @ next_tensor  # Shape: (r_new, d_next)
                mps[i + 1].modify(data=new_next)
            else:
                raise ValueError(f"Last tensor should be 2D, got shape {next_tensor.shape}")
        else:
            # Next site is middle site: shape is (left_bond, right_bond, phys) in quimb
            if next_tensor.ndim == 3:
                r_left_next, r_right_next, d_next = next_tensor.shape
                # Reshape next_tensor to matrix: (r_left_next, r_right_next * d_next)
                next_matrix = next_tensor.reshape(r_left_next, r_right_next * d_next)
                # Contract: (r_new, r_right) @ (r_right, r_right_next * d_next)
                propagate = np.diag(S) @ Vh  # Shape: (r_new, r_right)
                new_next_matrix = propagate @ next_matrix  # Shape: (r_new, r_right_next * d_next)
                # Reshape back: (r_new, r_right_next, d_next)
                new_next = new_next_matrix.reshape(r_new, r_right_next, d_next)
                mps[i + 1].modify(data=new_next)
            else:
                raise ValueError(f"Middle tensor at site {i+1} should be 3D, got shape {next_tensor.shape}")

    # Pad bonds to uniform size if requested
    if pad_bonds and max_rank is not None:
        # Ensure all internal bonds have exactly max_rank dimensions
        # This is CRITICAL for A2DMRG to work correctly in subsequent sweeps
        for i in range(L):
            tensor = mps[i].data

            # Determine current bond dimensions
            if i == 0:
                # First site: (chi_R, d)
                chi_R_curr, d = tensor.shape
                # Pad right bond if needed
                if chi_R_curr < max_rank:
                    # Pad right bond to max_rank
                    chi_R_new = max_rank
                    padded = np.zeros((chi_R_new, d), dtype=tensor.dtype)
                    padded[:chi_R_curr, :] = tensor
                    mps[i].modify(data=padded)
            elif i == L - 1:
                # Last site: (chi_L, d)
                chi_L_curr, d = tensor.shape
                # Pad left bond if needed
                if chi_L_curr < max_rank:
                    # Pad left bond to max_rank
                    chi_L_new = max_rank
                    padded = np.zeros((chi_L_new, d), dtype=tensor.dtype)
                    padded[:chi_L_curr, :] = tensor
                    mps[i].modify(data=padded)
            else:
                # Middle site: (chi_L, chi_R, d)
                chi_L_curr, chi_R_curr, d = tensor.shape
                # Pad both bonds if needed
                chi_L_new = max_rank if chi_L_curr < max_rank else chi_L_curr
                chi_R_new = max_rank if chi_R_curr < max_rank else chi_R_curr
                if chi_L_curr < chi_L_new or chi_R_curr < chi_R_new:
                    padded = np.zeros((chi_L_new, chi_R_new, d), dtype=tensor.dtype)
                    padded[:chi_L_curr, :chi_R_curr, :] = tensor
                    mps[i].modify(data=padded)

    # Normalize if requested
    if normalize:
        norm = mps.norm()
        if norm > 0:
            mps /= norm

    return mps, truncation_errors
