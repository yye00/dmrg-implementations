"""Numpy array extraction from quimb MPS/MPO objects.

Standard conventions:
- MPS: (chi_left, d, chi_right) -- always 3D, even at boundaries (chi=1)
- MPO: (D_left, D_right, d_up, d_down) -- always 4D, even at boundaries (D=1)
  d_up connects to ket MPS, d_down connects to bra MPS.
  This matches observables.py's _get_mpo_array convention and compute_energy() contractions.
"""

import numpy as np
import quimb.tensor as qtn


def extract_mps_arrays(mps):
    """Extract MPS tensors as list of numpy arrays in (chi_L, d, chi_R) format.

    All tensors are returned as 3D arrays. Boundary sites are padded to 3D:
    - Site 0: (1, d, chi_R)
    - Site L-1: (chi_L, d, 1)

    Parameters
    ----------
    mps : quimb.tensor.MatrixProductState
        The MPS to extract arrays from.

    Returns
    -------
    list of numpy.ndarray
        List of L arrays, each with shape (chi_L, d, chi_R).
    """
    from a2dmrg.numerics.observables import _get_mps_array
    L = mps.L
    return [_get_mps_array(mps, i) for i in range(L)]


def extract_mpo_arrays(mpo):
    """Extract MPO tensors as list of numpy arrays in (D_L, D_R, d_up, d_down) format.

    Uses the same convention as observables.py's _get_mpo_array:
    - axis 0: mpo_left bond
    - axis 1: mpo_right bond
    - axis 2: upper physical (connects to ket MPS)
    - axis 3: lower physical (connects to bra MPS)

    This convention matches compute_energy() contractions exactly.

    Parameters
    ----------
    mpo : quimb.tensor.MatrixProductOperator
        The MPO to extract arrays from.

    Returns
    -------
    list of numpy.ndarray
        List of L arrays, each with shape (D_L, D_R, d_up, d_down).
    """
    from a2dmrg.numerics.observables import _get_mpo_array
    L = mpo.L
    return [_get_mpo_array(mpo, i) for i in range(L)]


def arrays_to_quimb_mps(arrays, dtype=None):
    """Convert list of numpy arrays back to quimb MPS.

    Uses quimb's built-in MatrixProductState constructor which generates
    proper UUID-based index names compatible with quimb's DMRG2 solver.

    Parameters
    ----------
    arrays : list of ndarray
        MPS tensors in (chi_L, d, chi_R) format.
    dtype : numpy dtype, optional
        Override dtype.

    Returns
    -------
    quimb.tensor.MatrixProductState
    """
    L = len(arrays)
    # Convert from our (chi_L, d, chi_R) convention to quimb's expected format:
    # - Site 0: 2D (d, chi_R)
    # - Middle sites: 3D (chi_L, d, chi_R) — same as ours
    # - Site L-1: 2D (chi_L, d)
    quimb_arrays = []
    for i in range(L):
        a = arrays[i]
        if dtype is not None:
            a = a.astype(dtype)
        if i == 0:
            quimb_arrays.append(a[0, :, :])  # (d, chi_R)
        elif i == L - 1:
            quimb_arrays.append(a[:, :, 0])  # (chi_L, d)
        else:
            quimb_arrays.append(a)  # (chi_L, d, chi_R)
    return qtn.MatrixProductState(quimb_arrays, shape='lpr')
