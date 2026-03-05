"""
Coarse-space eigenvalue problem solver for A2DMRG.

This module implements the solution of the generalized eigenvalue problem
that arises in the coarse-space minimization phase (Phase 3) of A2DMRG:

    H_coarse c = λ S_coarse c

where:
- H_coarse[i,j] = ⟨Y^(i), H Y^(j)⟩  (Hamiltonian matrix)
- S_coarse[i,j] = ⟨Y^(i), Y^(j)⟩    (overlap/mass matrix)

The mass matrix S_coarse may be ill-conditioned due to near-linear dependence
of the candidate states Y^(i). This module uses SVD-based regularization to
handle this problem robustly.
"""

import numpy as np
from scipy.linalg import eigh
from typing import Tuple, Optional


def solve_coarse_eigenvalue_problem(
    H_coarse: np.ndarray,
    S_coarse: np.ndarray,
    regularization: float = 1e-10,
    return_all: bool = False,
    return_diagnostics: bool = False
):
    """
    Solve the generalized eigenvalue problem H c = λ S c.

    Uses SVD-based regularization to handle ill-conditioned mass matrix S.

    Algorithm:
    1. Compute S = U Σ V^H via SVD
    2. Regularize: S^(-1/2) = U Σ^(-1/2) V^H, truncate σ < regularization
    3. Transform: H_transformed = S^(-1/2) H S^(-1/2)
    4. Solve standard eigenvalue problem: H_transformed c̃ = λ c̃
    5. Transform back: c* = S^(-1/2) c̃
    6. Normalize: c* = c* / ||c*||

    Parameters
    ----------
    H_coarse : np.ndarray
        Hamiltonian matrix, shape (n, n), Hermitian
    S_coarse : np.ndarray
        Mass/overlap matrix, shape (n, n), Hermitian positive semi-definite
    regularization : float, optional
        Threshold for truncating small singular values, default 1e-10
        Singular values σ < regularization are set to infinity in S^(-1/2)
    return_all : bool, optional
        If True, return all eigenvalues and eigenvectors, not just the minimum

    Returns
    -------
    energy : float
        Lowest eigenvalue
    coefficients : np.ndarray
        Corresponding eigenvector c*, shape (n,), normalized to ||c*|| = 1

    Or if return_all=True:
    energies : np.ndarray
        All eigenvalues, sorted from lowest to highest
    coefficients_matrix : np.ndarray
        Eigenvector matrix, shape (n, n), column i is eigenvector for eigenvalue i

    Notes
    -----
    For real-valued H_coarse and S_coarse, the eigenvalues are real and the
    eigenvectors are real. For complex-valued matrices, eigenvalues are still
    real (since H and S are Hermitian), but eigenvectors may be complex.

    The regularization parameter controls how small singular values are treated.
    Too small: numerical instability. Too large: loss of information.
    The default 1e-10 works well for typical A2DMRG problems.
    """
    # Validate inputs
    if H_coarse.shape != S_coarse.shape:
        raise ValueError(f"H_coarse and S_coarse must have same shape, got {H_coarse.shape} and {S_coarse.shape}")

    if H_coarse.shape[0] != H_coarse.shape[1]:
        raise ValueError(f"H_coarse and S_coarse must be square, got shape {H_coarse.shape}")

    n = H_coarse.shape[0]

    if n == 0:
        raise ValueError("H_coarse and S_coarse cannot be empty")

    if regularization < 0:
        raise ValueError(f"regularization must be non-negative, got {regularization}")

    # Determine dtype - preserve complex if either matrix is complex
    dtype = H_coarse.dtype
    if S_coarse.dtype == np.complex128 or H_coarse.dtype == np.complex128:
        dtype = np.complex128
    else:
        dtype = np.float64

    # Ensure matrices are the right dtype
    H_coarse = np.asarray(H_coarse, dtype=dtype)
    S_coarse = np.asarray(S_coarse, dtype=dtype)

    # Step 1-2: Handle rank-deficient / ill-conditioned overlap matrix S.
    # We follow the paper's approach: work in the subspace of significant eigenvalues
    # of S rather than always perturbing S (which can shift eigenvalues even when S=I).
    s_eig, U = np.linalg.eigh(S_coarse)

    # Keep only the significant subspace. Interpret `regularization` as the eigenvalue
    # cutoff for S (absolute).
    cutoff = float(regularization)
    keep = s_eig > cutoff

    if not np.any(keep):
        raise np.linalg.LinAlgError(
            "Overlap matrix S has no eigenvalues above regularization cutoff; "
            "coarse space is numerically rank-deficient."
        )

    s_keep = s_eig[keep]
    U_keep = U[:, keep]

    # Build S^{-1/2} on the kept subspace.
    s_inv_sqrt = 1.0 / np.sqrt(s_keep)

    # Transform to standard eigenproblem on the kept subspace:
    # H_t = S^{-1/2} U^H H U S^{-1/2}
    if dtype == np.complex128:
        H_rot = U_keep.conj().T @ H_coarse @ U_keep
    else:
        H_rot = U_keep.T @ H_coarse @ U_keep

    H_t = (s_inv_sqrt[:, None] * H_rot) * s_inv_sqrt[None, :]

    # Solve standard eigenproblem
    eigenvalues_red, vecs_red = np.linalg.eigh(H_t)

    # Map eigenvectors back to the original coarse space:
    # c = U S^{-1/2} v
    coefficients_matrix = U_keep @ (s_inv_sqrt[:, None] * vecs_red)
    eigenvalues = eigenvalues_red

    # Step 6: Normalize eigenvectors with respect to S (original, unregularized)
    # For generalized eigenvalue problem, we want c^H S c = 1
    for i in range(coefficients_matrix.shape[1]):
        c = coefficients_matrix[:, i]
        if dtype == np.complex128:
            norm_squared = c.conj() @ S_coarse @ c
        else:
            norm_squared = c @ S_coarse @ c

        norm_squared = np.real(norm_squared)  # Should be real for Hermitian S

        if norm_squared > 1e-14:
            coefficients_matrix[:, i] /= np.sqrt(norm_squared)

    # Return results
    diagnostics = {"n_effective": int(np.sum(keep))}
    if return_all:
        if return_diagnostics:
            return eigenvalues, coefficients_matrix, diagnostics
        return eigenvalues, coefficients_matrix
    else:
        if return_diagnostics:
            return eigenvalues[0], coefficients_matrix[:, 0], diagnostics
        return eigenvalues[0], coefficients_matrix[:, 0]


def verify_solution(
    H_coarse: np.ndarray,
    S_coarse: np.ndarray,
    energy: float,
    coefficients: np.ndarray,
    tolerance: float = 1e-8
) -> Tuple[bool, float]:
    """
    Verify that the solution satisfies H c = λ S c.

    Parameters
    ----------
    H_coarse : np.ndarray
        Hamiltonian matrix
    S_coarse : np.ndarray
        Mass matrix
    energy : float
        Eigenvalue
    coefficients : np.ndarray
        Eigenvector
    tolerance : float, optional
        Relative error tolerance for verification

    Returns
    -------
    is_valid : bool
        True if the solution is valid within tolerance
    residual : float
        Relative residual ||H c - λ S c|| / ||H c||
    """
    # Compute H @ c
    Hc = H_coarse @ coefficients

    # Compute λ S @ c
    lambda_Sc = energy * (S_coarse @ coefficients)

    # Compute residual
    residual_vec = Hc - lambda_Sc
    residual_norm = np.linalg.norm(residual_vec)
    Hc_norm = np.linalg.norm(Hc)

    if Hc_norm < 1e-14:
        # Degenerate case
        relative_residual = residual_norm
    else:
        relative_residual = residual_norm / Hc_norm

    is_valid = relative_residual < tolerance

    return is_valid, relative_residual


def estimate_condition_number(S_coarse: np.ndarray) -> float:
    """
    Estimate the condition number of the mass matrix S_coarse.

    The condition number is the ratio of largest to smallest singular value.
    Large condition numbers indicate ill-conditioning.

    Parameters
    ----------
    S_coarse : np.ndarray
        Mass matrix

    Returns
    -------
    condition_number : float
        Condition number κ(S) = σ_max / σ_min
        Returns infinity if smallest singular value is zero
    """
    s = np.linalg.svd(S_coarse, compute_uv=False, hermitian=True)

    if s[-1] < 1e-14:
        return np.inf
    else:
        return s[0] / s[-1]
