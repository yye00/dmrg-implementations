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


def form_linear_combination(
    candidate_mps_list: List,
    coefficients: np.ndarray,
    comm=None,
    assigned_sites: Optional[List[int]] = None
):
    """
    Form optimal MPS as linear combination Ỹ = Σⱼ c*ⱼ Y^(j).

    This function combines the candidate MPS states with the given coefficients
    to form the optimal state in the coarse space.

    Parameters
    ----------
    candidate_mps_list : List[MPS]
        List of candidate MPS states: [Y^(0), Y^(1), ..., Y^(d)]
        All must have same length L and same physical dimension d_phys
    coefficients : np.ndarray
        Coefficients c*, shape (d+1,)
        Obtained from coarse eigenvalue problem solver
    comm : MPI.Comm, optional
        MPI communicator (None for serial mode)
    assigned_sites : List[int], optional
        List of site indices assigned to this processor (for parallel mode)
        Currently unused - all processors form the full combination

    Returns
    -------
    combined_mps : MPS
        Linear combination Ỹ = Σⱼ c*ⱼ Y^(j)
        Has same length L but larger bond dimension

    Notes
    -----
    Uses quimb's built-in MPS addition which automatically handles bond dimension
    stacking. When adding two MPS with bond dimension χ, the result has bond
    dimension at most 2χ. For d+1 states, bond dimension grows to at most (d+1)χ.

    Later compression (Phase 4) reduces this back down via truncated SVD.

    In parallel mode:
    - Coefficients are broadcast from rank 0
    - All processors form the same linear combination
    - The assigned_sites parameter is for future optimization

    Example
    -------
    Serial mode:
    >>> combined_mps = form_linear_combination(candidate_list, coefficients)

    Parallel mode:
    >>> from a2dmrg.mpi_compat import MPI
    >>> comm = MPI.COMM_WORLD
    >>> my_sites = [2, 3, 4]
    >>> combined_mps = form_linear_combination(
    ...     candidate_list, coefficients, comm=comm, assigned_sites=my_sites
    ... )
    """
    # Validate inputs
    if len(candidate_mps_list) == 0:
        raise ValueError("candidate_mps_list cannot be empty")

    if len(coefficients) != len(candidate_mps_list):
        raise ValueError(
            f"coefficients length ({len(coefficients)}) must match "
            f"candidate_mps_list length ({len(candidate_mps_list)})"
        )

    # Broadcast coefficients in parallel mode
    if comm is not None:
        # Ensure all processors have the same coefficients
        coefficients = comm.bcast(coefficients, root=0)

    # Form linear combination using quimb's MPS arithmetic
    # Handle case where some coefficients might be zero or very small
    # Find first non-zero coefficient to initialize
    init_idx = None
    for i in range(len(candidate_mps_list)):
        if abs(coefficients[i]) > 1e-14:  # Skip near-zero coefficients
            init_idx = i
            break

    if init_idx is None:
        raise ValueError("All coefficients are zero - cannot form linear combination")

    # Start with the first non-zero term
    combined_mps = coefficients[init_idx] * candidate_mps_list[init_idx]

    # Add remaining non-zero terms
    for i in range(len(candidate_mps_list)):
        if i != init_idx and abs(coefficients[i]) > 1e-14:
            combined_mps = combined_mps + coefficients[i] * candidate_mps_list[i]

    return combined_mps


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
