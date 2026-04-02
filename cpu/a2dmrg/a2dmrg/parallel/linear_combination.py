"""
Linear Combination Formation for A2DMRG

This module implements Phase 3 of the A2DMRG algorithm: forming the optimal
linear combination of candidate MPS states.

Key Algorithm:
--------------
Given:
- Candidate MPS list: [Y^(0), Y^(1), ..., Y^(d)]
- Optimal coefficients: c* from coarse-space solver

Compute:
- Ỹ = Σⱼ c*ⱼ Y^(j)

This forms a new MPS that is the optimal linear combination. The bond dimension
of Ỹ is larger than the original states since it combines multiple MPS.

MPI Implementation:
------------------
For parallel formation:
1. Rank 0 computes coefficients c* from coarse eigenvalue problem
2. Broadcast c* to all processors
3. All processors form the same linear combination
4. No Gather needed - linear combination is a local operation

Reference:
----------
Grigori & Hassan, arXiv:2505.23429v2, Section 3.2
"""

import numpy as np
from typing import List, Optional
import copy

try:
    from a2dmrg.mpi_compat import MPI
except ImportError:
    MPI = None


def form_linear_combination(candidate_arrays_list, coefficients, comm=None, assigned_sites=None):
    """Form weighted sum of candidate MPS (numpy arrays).

    IMPORTANT: The coefficient for each candidate is applied to site 0 ONLY.
    An MPS state |psi> = A[0] A[1] ... A[L-1], so multiplying c into every
    site would give c^L amplitude instead of c.

    Parameters
    ----------
    candidate_arrays_list : List[List[ndarray]]
        List of candidate MPS states as numpy array lists.
        Each candidate is a list of ndarray in (chi_L, d, chi_R) format.
    coefficients : np.ndarray
        Coefficients c*, shape (d+1,)
        Obtained from coarse eigenvalue problem solver.
    comm : MPI.Comm, optional
        MPI communicator (None for serial mode).
    assigned_sites : List[int], optional
        Currently unused - all processors form the full combination.

    Returns
    -------
    combined : List[ndarray]
        Linear combination as list of numpy arrays in (chi_L, d, chi_R) format.
        Bond dimensions are expanded (block-diagonal for middle sites).
    """
    if len(candidate_arrays_list) == 0:
        raise ValueError("candidate_arrays_list cannot be empty")
    if len(coefficients) != len(candidate_arrays_list):
        raise ValueError(
            f"coefficients length ({len(coefficients)}) must match "
            f"candidate_arrays_list length ({len(candidate_arrays_list)})"
        )

    # Broadcast coefficients in parallel mode
    if comm is not None:
        coefficients = comm.bcast(coefficients, root=0)

    n_cands = len(candidate_arrays_list)
    L = len(candidate_arrays_list[0])

    # Filter near-zero coefficients
    active = [(j, coefficients[j]) for j in range(n_cands) if abs(coefficients[j]) > 1e-14]
    if not active:
        raise ValueError("All coefficients are zero")

    combined = []
    for i in range(L):
        if i == 0:
            # First site: (1, d, chi_R) -- apply coeff and concatenate along chi_R
            tensors = [c * candidate_arrays_list[j][i] for j, c in active]
            combined.append(np.concatenate(tensors, axis=2))
        elif i == L - 1:
            # Last site: (chi_L, d, 1) -- concatenate along chi_L
            tensors = [candidate_arrays_list[j][i] for j, c in active]
            combined.append(np.concatenate(tensors, axis=0))
        else:
            # Middle: block-diagonal (chi_L*n, d, chi_R*n)
            tensors = [candidate_arrays_list[j][i] for j, c in active]
            total_L = sum(t.shape[0] for t in tensors)
            d = tensors[0].shape[1]
            total_R = sum(t.shape[2] for t in tensors)
            block = np.zeros((total_L, d, total_R), dtype=tensors[0].dtype)
            row, col = 0, 0
            for t in tensors:
                block[row:row+t.shape[0], :, col:col+t.shape[2]] = t
                row += t.shape[0]
                col += t.shape[2]
            combined.append(block)

    return combined


def verify_linear_combination_energy(
    combined_mps,
    mpo,
    coefficients: np.ndarray,
    H_coarse: np.ndarray,
    S_coarse: np.ndarray,
    tolerance: float = 1e-8
) -> bool:
    """
    Verify that the energy of the combined MPS matches the coarse-space prediction.

    For the linear combination Ỹ = Σⱼ c*ⱼ Y^(j), the energy should be:
        E(Ỹ) = c*† H_coarse c* / (c*† S_coarse c*)

    This is a sanity check that the linear combination was formed correctly.

    Parameters
    ----------
    combined_mps : MPS
        The combined MPS Ỹ
    mpo : MPO
        Hamiltonian as Matrix Product Operator
    coefficients : np.ndarray
        Coefficients c*, shape (d+1,)
    H_coarse : np.ndarray
        Coarse-space Hamiltonian matrix
    S_coarse : np.ndarray
        Coarse-space overlap matrix
    tolerance : float, optional
        Relative error tolerance

    Returns
    -------
    is_valid : bool
        True if energies match within tolerance

    Notes
    -----
    The predicted energy is:
        E_predicted = (c*† H_coarse c*) / (c*† S_coarse c*)

    The actual energy is:
        E_actual = ⟨Ỹ|H|Ỹ⟩ / ⟨Ỹ|Ỹ⟩
    """
    from a2dmrg.numerics.observables import compute_energy

    # Compute actual energy of combined MPS
    E_actual = compute_energy(combined_mps, mpo, normalize=True)

    # Compute predicted energy from coarse-space matrices
    # E_predicted = c*† H_coarse c* / (c*† S_coarse c*)
    numerator = np.conj(coefficients) @ H_coarse @ coefficients
    denominator = np.conj(coefficients) @ S_coarse @ coefficients

    E_predicted = np.real(numerator / denominator)

    # Extract scalar from E_actual if it's a Tensor object
    if hasattr(E_actual, 'data'):
        E_actual = np.asarray(E_actual.data).ravel()[0]
    E_actual = np.real(E_actual)

    # Compute relative error
    if abs(E_predicted) > 1e-12:
        relative_error = abs(E_actual - E_predicted) / abs(E_predicted)
    else:
        relative_error = abs(E_actual - E_predicted)

    is_valid = relative_error < tolerance

    return is_valid, E_actual, E_predicted, relative_error
