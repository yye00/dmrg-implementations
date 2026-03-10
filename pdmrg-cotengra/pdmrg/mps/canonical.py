"""Canonical form utilities for MPS tensors.

Our internal convention for raw numpy arrays:
  MPS tensor: (left_bond, phys, right_bond)

quimb convention (MPS_rand_state):
  Edge: (bond, phys)  with physical LAST
  Bulk: (bond_L, bond_R, phys)  with physical LAST

We provide helpers to convert between these conventions.
"""

import numpy as np
import quimb.tensor as qtn


def get_tensor_data(mps, site):
    """Extract tensor data from quimb MPS in our convention: (left, phys, right).

    Parameters
    ----------
    mps : quimb MatrixProductState
    site : int

    Returns
    -------
    data : ndarray, shape (left_bond, phys, right_bond)
    """
    tensor = mps[site]
    data = tensor.data
    inds = tensor.inds

    # Find the physical index position
    phys_name = mps.site_ind_id.format(site)
    phys_pos = list(inds).index(phys_name)

    ndim = data.ndim
    if ndim == 2:
        # Edge tensor: 2D means one bond dimension is missing (size 1)
        if phys_pos == 0:
            # shape is (phys, bond) -> (1, phys, bond)
            return data[np.newaxis, :, :]
        else:
            # shape is (bond, phys)
            if site == 0:
                # Left edge: bond is right_bond -> (1, phys, right_bond)
                return data.T[np.newaxis, :, :]
            else:
                # Right edge: bond is left_bond -> (left_bond, phys, 1)
                return data[:, :, np.newaxis]
    elif ndim == 3:
        # Bulk tensor: quimb has (bond_L, bond_R, phys) for rand_state
        # Physical is at phys_pos
        if phys_pos == 2:
            # (bond_L, bond_R, phys) -> need (bond_L, phys, bond_R)
            return np.transpose(data, (0, 2, 1))
        elif phys_pos == 1:
            # Already (bond_L, phys, bond_R) — our convention
            return data.copy()
        else:  # phys_pos == 0
            # (phys, bond_L, bond_R) -> (bond_L, phys, bond_R)
            return np.transpose(data, (1, 0, 2))
    else:
        raise ValueError(f"Unexpected tensor ndim={ndim} at site {site}")


def set_tensor_data(mps, site, data_lpr):
    """Set tensor data in quimb MPS from our convention: (left, phys, right).

    Parameters
    ----------
    mps : quimb MatrixProductState
    site : int
    data_lpr : ndarray, shape (left_bond, phys, right_bond)
    """
    tensor = mps[site]
    inds = tensor.inds
    phys_name = mps.site_ind_id.format(site)
    phys_pos = list(inds).index(phys_name)

    ndim = len(inds)
    if ndim == 2:
        if phys_pos == 0:
            # quimb wants (phys, bond)
            # Our data: (1, phys, bond) or (bond, phys, 1)
            if data_lpr.shape[0] == 1:
                new_data = data_lpr[0, :, :]  # (phys, right)
            else:
                new_data = data_lpr[:, :, 0]  # this shouldn't happen here
        else:
            # quimb wants (bond, phys)
            if data_lpr.shape[0] == 1:
                # Left edge: (1, phys, right) -> (right, phys)
                new_data = data_lpr[0, :, :].T  # NO
                # Actually quimb has (bond, phys) at position (0, 1)
                # and bond IS the right bond for site 0
                new_data = np.transpose(data_lpr[0, :, :])  # (right, phys)
            else:
                # Right edge: (left, phys, 1) -> (left, phys)
                new_data = data_lpr[:, :, 0]
    elif ndim == 3:
        if phys_pos == 2:
            # quimb wants (..., ..., phys) -> (left, right, phys)
            new_data = np.transpose(data_lpr, (0, 2, 1))
        elif phys_pos == 1:
            new_data = data_lpr.copy()
        else:
            new_data = np.transpose(data_lpr, (1, 0, 2))
    else:
        raise ValueError(f"Unexpected ndim={ndim}")

    tensor.modify(data=new_data)


def get_mpo_tensor_data(mpo, site):
    """Extract MPO tensor data in our convention: (mpo_left, mpo_right, phys_up, phys_down).

    quimb MPO convention:
      Edge: (mpo_bond, phys_up, phys_down) with 3 dims
      Bulk: (mpo_left, mpo_right, phys_up, phys_down) with 4 dims

    Parameters
    ----------
    mpo : quimb MPO
    site : int

    Returns
    -------
    data : ndarray, shape (mpo_left, mpo_right, d, d)
        For edge sites, the missing bond dimension is size 1.
    """
    tensor = mpo[site]
    data = tensor.data
    inds = tensor.inds

    upper_name = mpo.upper_ind_id.format(site)
    lower_name = mpo.lower_ind_id.format(site)

    ndim = data.ndim
    if ndim == 3:
        # Edge site: (mpo_bond, phys_up, phys_down)
        upper_pos = list(inds).index(upper_name)
        lower_pos = list(inds).index(lower_name)
        bond_pos = [i for i in range(3) if i != upper_pos and i != lower_pos][0]

        # Transpose to (bond, upper, lower)
        perm = [bond_pos, upper_pos, lower_pos]
        data_ordered = np.transpose(data, perm)

        # Add missing bond dimension
        if site == 0:
            # Left edge: missing left bond -> (1, mpo_right, d, d)
            return data_ordered[np.newaxis, :, :, :]
        else:
            # Right edge: missing right bond -> (mpo_left, 1, d, d)
            return data_ordered[:, np.newaxis, :, :]
    elif ndim == 4:
        # Bulk: (mpo_left, mpo_right, phys_up, phys_down) in quimb
        upper_pos = list(inds).index(upper_name)
        lower_pos = list(inds).index(lower_name)
        bond_positions = [i for i in range(4) if i != upper_pos and i != lower_pos]
        # bond_positions[0] should be left bond, bond_positions[1] right bond
        perm = [bond_positions[0], bond_positions[1], upper_pos, lower_pos]
        return np.transpose(data, perm)
    else:
        raise ValueError(f"Unexpected MPO tensor ndim={ndim} at site {site}")


def left_canonize_site(A):
    """Left-canonize an MPS tensor via QR decomposition.

    A has shape (left, phys, right).
    Returns Q (left, phys, new_right), R (new_right, right).
    Q is left-isometric: sum_s Q^H Q = I.

    Parameters
    ----------
    A : ndarray, shape (chi_L, d, chi_R)

    Returns
    -------
    Q : ndarray, shape (chi_L, d, min(chi_L*d, chi_R))
    R : ndarray, shape (min(chi_L*d, chi_R), chi_R)
    """
    chi_L, d, chi_R = A.shape
    M = A.reshape(chi_L * d, chi_R)
    Q, R = np.linalg.qr(M)
    new_dim = Q.shape[1]
    Q = Q.reshape(chi_L, d, new_dim)
    return Q, R


def right_canonize_site(B):
    """Right-canonize an MPS tensor via RQ decomposition.

    B has shape (left, phys, right).
    Returns L (left, new_left), Q (new_left, phys, right).
    Q is right-isometric: sum_s Q Q^H = I.

    Parameters
    ----------
    B : ndarray, shape (chi_L, d, chi_R)

    Returns
    -------
    L : ndarray, shape (chi_L, new_left)
    Q : ndarray, shape (new_left, d, chi_R)
    """
    chi_L, d, chi_R = B.shape
    M = B.reshape(chi_L, d * chi_R)
    # RQ = QR of M^T, then transpose back
    Q, R = np.linalg.qr(M.conj().T)
    # M^H = Q R -> M = R^H Q^H
    L = R.conj().T  # shape (chi_L, new_dim)
    Qr = Q.conj().T  # shape (new_dim, d*chi_R)
    new_dim = Qr.shape[0]
    Qr = Qr.reshape(new_dim, d, chi_R)
    return L, Qr
