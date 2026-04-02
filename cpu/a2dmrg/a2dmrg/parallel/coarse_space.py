"""
Coarse-Space Matrix Construction for A2DMRG

This module implements Phase 3 of the A2DMRG algorithm: computing the coarse-space
matrices H_coarse and S_coarse for the generalized eigenvalue problem.

Key Algorithm:
--------------
Given candidate MPS list [Y^(0), Y^(1), ..., Y^(d)] where:
- Y^(0) is the original state
- Y^(i) for i>0 are locally-updated states

Compute:
- H_coarse[i,j] = ⟨Y^(i), H Y^(j)⟩  (Hamiltonian matrix)
- S_coarse[i,j] = ⟨Y^(i), Y^(j)⟩    (overlap/mass matrix)

Properties:
- Both matrices are (d+1) × (d+1)
- Both are Hermitian for complex MPS
- S_coarse is positive semi-definite (may be ill-conditioned)
- Diagonal H_coarse[i,i] = ⟨Y^(i), H Y^(i)⟩ is real (energy)

MPI Implementation:
------------------
Each processor computes contributions for its assigned sites:
1. Build local H_coarse_local and S_coarse_local (mostly zeros)
2. Allreduce with MPI.SUM to combine contributions
3. All processors receive identical global matrices

Reference:
----------
Grigori & Hassan, arXiv:2505.23429v2, Section 3.2
"""

import numpy as np
from typing import List, Tuple, Optional

try:
    from a2dmrg.mpi_compat import MPI
except ImportError:
    MPI = None


def filter_redundant_candidates(
    candidate_mps_list: List,
    overlap_threshold: float = 0.99
) -> Tuple[List, List[int]]:
    """
    Filter out redundant candidates that are nearly linearly dependent.

    When local micro-steps produce nearly identical MPS (overlap ~ 1.0),
    the coarse-space mass matrix S becomes singular. This function removes
    redundant candidates to prevent numerical instability.

    Algorithm:
    1. Start with first candidate (always keep the original MPS)
    2. For each subsequent candidate:
       - Compute overlap with all retained candidates
       - If max overlap < threshold, keep it
       - Otherwise, discard it (redundant)

    Parameters
    ----------
    candidate_mps_list : List[MPS]
        List of candidate MPS states
    overlap_threshold : float, optional
        Candidates with |overlap| > threshold are considered redundant (default: 0.99)

    Returns
    -------
    filtered_list : List[MPS]
        List of non-redundant candidates
    retained_indices : List[int]
        Indices of retained candidates in original list
    """
    from a2dmrg.numerics.observables import compute_overlap_numpy
    import numpy as np

    if len(candidate_mps_list) == 0:
        return [], []

    # Always keep first candidate (original MPS)
    filtered = [candidate_mps_list[0]]
    retained_indices = [0]

    # Check each subsequent candidate for redundancy
    for i in range(1, len(candidate_mps_list)):
        candidate = candidate_mps_list[i]

        # Compute overlap with all retained candidates
        is_redundant = False
        for retained in filtered:
            overlap = compute_overlap_numpy(candidate, retained)

            # Normalize overlap by norms
            norm_candidate = np.sqrt(abs(compute_overlap_numpy(candidate, candidate)))
            norm_retained = np.sqrt(abs(compute_overlap_numpy(retained, retained)))

            if norm_candidate < 1e-14 or norm_retained < 1e-14:
                # Degenerate case: zero norm MPS
                is_redundant = True
                break

            normalized_overlap = np.abs(overlap) / (norm_candidate * norm_retained)

            if normalized_overlap > overlap_threshold:
                # This candidate is too similar to a retained one
                is_redundant = True
                break

        if not is_redundant:
            filtered.append(candidate)
            retained_indices.append(i)

    return filtered, retained_indices


def build_coarse_matrices(
    candidate_mps_list: List,
    mpo_arrays,
    comm=None,
    assigned_sites: Optional[List[int]] = None,
    filter_redundant: bool = False,
    overlap_threshold: float = 0.99,
    return_filtered_candidates: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build coarse-space matrices H_coarse and S_coarse for the generalized eigenvalue problem.

    This function computes the (d+1) × (d+1) matrices:
    - H_coarse[i,j] = ⟨Y^(i), H Y^(j)⟩
    - S_coarse[i,j] = ⟨Y^(i), Y^(j)⟩

    where Y^(0), Y^(1), ..., Y^(d) are candidate MPS states.

    In parallel mode, each processor computes contributions for its assigned sites.
    The matrices are then combined via MPI.Allreduce.

    Parameters
    ----------
    candidate_mps_list : List[List[ndarray]]
        List of candidate MPS states as numpy array lists: [Y^(0), Y^(1), ..., Y^(d)]
        Y^(0) is the original state, Y^(i>0) are locally-updated states
        Each candidate is a list of ndarray in (chi_L, d, chi_R) format.
    mpo_arrays : List[ndarray]
        Pre-extracted MPO arrays in (D_L, D_R, d_up, d_down) format.
    comm : MPI.Comm, optional
        MPI communicator (None for serial mode)
    assigned_sites : List[int], optional
        List of site indices assigned to this processor (for parallel mode)
        If None, all rows are computed (serial mode)
    filter_redundant : bool, optional
        Whether to filter out redundant candidates (default: True)
        Candidates with overlap > overlap_threshold are removed to prevent
        singular mass matrices
    overlap_threshold : float, optional
        Threshold for filtering redundant candidates (default: 0.99)
        Candidates with normalized overlap > threshold are considered redundant

    Returns
    -------
    H_coarse : np.ndarray
        Hamiltonian matrix, shape (d', d') where d' <= d+1 (after filtering)
        H_coarse[i,j] = ⟨Y^(i), H Y^(j)⟩
    S_coarse : np.ndarray
        Overlap/mass matrix, shape (d', d')
        S_coarse[i,j] = ⟨Y^(i), Y^(j)⟩
    filtered_candidates : List[MPS] or None
        If filter_redundant=True, returns the filtered candidate list
        If filter_redundant=False, returns None

    Properties
    ----------
    - Both matrices are Hermitian (for complex MPS: H^† = H, S^† = S)
    - Diagonal elements H_coarse[i,i] are real (energies)
    - S_coarse is positive semi-definite
    - For complex MPS: dtype is complex128, for real: float64

    Notes
    -----
    - Uses compute_energy() and compute_overlap() from a2dmrg.numerics.observables
    - In parallel mode, only computes rows for assigned_sites
    - MPI.Allreduce combines contributions from all processors
    - All processors receive identical global matrices after Allreduce

    Example
    -------
    Serial mode:
    >>> H_coarse, S_coarse = build_coarse_matrices(candidate_list, mpo)

    Parallel mode:
    >>> from a2dmrg.mpi_compat import MPI
    >>> comm = MPI.COMM_WORLD
    >>> my_sites = [2, 3, 4]  # This processor handles sites 2, 3, 4
    >>> H_coarse, S_coarse = build_coarse_matrices(
    ...     candidate_list, mpo, comm=comm, assigned_sites=my_sites
    ... )
    """
    # Import numpy-based observable computation functions
    from a2dmrg.numerics.observables import compute_cross_energy_numpy, compute_overlap_numpy

    # Filter redundant candidates if requested.
    # NOTE: Default is False to keep this function a pure matrix builder.
    if filter_redundant:
        candidate_mps_list, _retained_indices = filter_redundant_candidates(
            candidate_mps_list,
            overlap_threshold=overlap_threshold,
        )

    # Determine matrix dimension
    d_plus_1 = len(candidate_mps_list)

    # Infer dtype from first MPS (numpy array list)
    dtype = np.complex128 if np.iscomplexobj(candidate_mps_list[0][0]) else np.float64

    # Determine which rows to compute
    if assigned_sites is None:
        # Serial mode: compute all rows
        rows_to_compute = list(range(d_plus_1))
    else:
        # Parallel mode: only compute rows for assigned sites
        # Note: assigned_sites are site indices (0 to d-1)
        # But matrix indices include Y^(0), so we add 1
        # Y^(0) is computed by rank 0 (site -1 or special case)
        rows_to_compute = [i + 1 for i in assigned_sites]

        # If this is rank 0 (or no sites assigned), compute row 0 (original state)
        if comm is not None and comm.rank == 0:
            rows_to_compute = [0] + rows_to_compute

    # Initialize local matrices (will be mostly zeros except for assigned rows)
    H_local = np.zeros((d_plus_1, d_plus_1), dtype=dtype)
    S_local = np.zeros((d_plus_1, d_plus_1), dtype=dtype)

    # Compute matrix elements for assigned rows
    for i in rows_to_compute:
        # Only compute upper triangle (including diagonal) and fill by Hermitian symmetry.
        for j in range(i, d_plus_1):
            # Compute H_coarse[i,j] = ⟨Y^(i), H Y^(j)⟩
            # This is: ⟨Y^(i)| H |Y^(j)⟩
            # Use the formula: (bra.H @ mpo @ ket) where bra and ket are MPS

            bra = candidate_mps_list[i]
            ket = candidate_mps_list[j]

            # For H matrix element: ⟨bra| H |ket⟩
            # We use: bra.H @ (mpo @ ket) and extract the scalar
            # But compute_energy normalizes, so we need to be careful

            # Approach: Use compute_overlap for numerator and denominator separately
            # H[i,j] = ⟨bra| H |ket⟩
            # First compute |H|ket⟩
            # Then compute ⟨bra| (H|ket⟩)

            # Actually, for H matrix element, we can use:
            # If i == j: H[i,i] = energy of state i
            # If i != j: H[i,j] = ⟨i| H |j⟩

            if i == j:
                # Diagonal: ⟨bra|H|bra⟩ via numpy sweep (single pass)
                H_ij = compute_cross_energy_numpy(bra, mpo_arrays, bra)
                if not np.iscomplexobj(H_local):
                    H_ij = np.real(H_ij)
                H_local[i, j] = H_ij
            else:
                # Off-diagonal: ⟨bra|H|ket⟩ via numpy sweep (no cotengra)
                H_ij = compute_cross_energy_numpy(bra, mpo_arrays, ket)
                if not np.iscomplexobj(H_local):
                    H_ij = np.real(H_ij)
                H_local[i, j] = H_ij

            # Compute S_coarse[i,j] = ⟨Y^(i), Y^(j)⟩
            S_ij = compute_overlap_numpy(bra, ket)
            if not np.iscomplexobj(S_local):
                S_ij = np.real(S_ij)
            S_local[i, j] = S_ij

            # Fill lower-triangular entry by Hermitian symmetry so we only do half the work.
            if j != i:
                H_local[j, i] = np.conj(H_local[i, j])
                S_local[j, i] = np.conj(S_local[i, j])

    # Combine contributions from all ranks if work was distributed
    # CRITICAL: Only Allreduce when ranks computed partial results
    # (assigned_sites is not None). When assigned_sites is None, every rank
    # computed the full matrices independently, so summing would multiply
    # all elements by the number of ranks.
    if comm is not None and assigned_sites is not None:
        H_coarse = np.zeros((d_plus_1, d_plus_1), dtype=dtype)
        S_coarse = np.zeros((d_plus_1, d_plus_1), dtype=dtype)

        # NOTE: We intentionally avoid MPI.Allreduce here for determinism.
        # Floating-point summation is not associative and different MPI reduction
        # trees can yield tiny np-dependent roundoff differences.
        # For publication + reproducibility tests we want bitwise-stable results
        # across different `np`, so we gather to rank 0, sum in rank order, then
        # broadcast the result back out.
        H_parts = comm.gather(H_local, root=0)
        S_parts = comm.gather(S_local, root=0)

        if comm.Get_rank() == 0:
            H_coarse = np.zeros((d_plus_1, d_plus_1), dtype=dtype)
            S_coarse = np.zeros((d_plus_1, d_plus_1), dtype=dtype)
            for h in H_parts:
                H_coarse += h
            for s in S_parts:
                S_coarse += s

        H_coarse = comm.bcast(H_coarse, root=0)
        S_coarse = comm.bcast(S_coarse, root=0)
    else:
        # Serial mode or all ranks computed full matrices: use local directly
        H_coarse = H_local
        S_coarse = S_local

    # Optionally return the filtered candidate list (backwards compatible default: return 2 values).
    if return_filtered_candidates:
        if filter_redundant:
            return H_coarse, S_coarse, candidate_mps_list
        else:
            return H_coarse, S_coarse, None

    return H_coarse, S_coarse


def verify_hermitian(matrix: np.ndarray, name: str = "Matrix", tol: float = 1e-10) -> bool:
    """
    Verify that a matrix is Hermitian (A^† = A).

    For real matrices, this means symmetric (A^T = A).
    For complex matrices, this means A.conj().T = A.

    Parameters
    ----------
    matrix : np.ndarray
        Matrix to verify, shape (N, N)
    name : str, optional
        Name for error messages
    tol : float, optional
        Tolerance for comparison (default: 1e-10)

    Returns
    -------
    bool
        True if matrix is Hermitian within tolerance

    Raises
    ------
    AssertionError
        If matrix is not square or not Hermitian
    """
    assert matrix.shape[0] == matrix.shape[1], f"{name} must be square"

    # Compute Hermitian conjugate
    matrix_H = matrix.conj().T

    # Check if equal within tolerance
    max_diff = np.max(np.abs(matrix - matrix_H))

    if max_diff > tol:
        raise AssertionError(
            f"{name} is not Hermitian: max|A - A^†| = {max_diff:.2e} > {tol:.2e}"
        )

    return True


def verify_positive_semidefinite(matrix: np.ndarray, name: str = "Matrix", tol: float = -1e-10) -> bool:
    """
    Verify that a matrix is positive semi-definite (all eigenvalues >= 0).

    Parameters
    ----------
    matrix : np.ndarray
        Matrix to verify, shape (N, N)
    name : str, optional
        Name for error messages
    tol : float, optional
        Tolerance for smallest eigenvalue (default: -1e-10, allowing small numerical errors)

    Returns
    -------
    bool
        True if matrix is positive semi-definite

    Raises
    ------
    AssertionError
        If matrix has negative eigenvalues beyond tolerance
    """
    eigenvalues = np.linalg.eigvalsh(matrix)
    min_eigenvalue = np.min(eigenvalues)

    if min_eigenvalue < tol:
        raise AssertionError(
            f"{name} is not positive semi-definite: "
            f"min eigenvalue = {min_eigenvalue:.2e} < {tol:.2e}"
        )

    return True
