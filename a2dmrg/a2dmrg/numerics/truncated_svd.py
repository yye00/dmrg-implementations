"""
Truncated SVD for tensor compression in A2DMRG.

This module provides standard truncated SVD (NOT recursive accurate SVD).
Truncation is controlled by:
- tolerance: keep singular values >= sigma_max * tol
- max_rank: keep at most max_rank singular values
- Combined: use minimum of both constraints
"""

import numpy as np
from typing import Optional, Tuple


def truncated_svd(
    M: np.ndarray,
    max_rank: Optional[int] = None,
    tol: Optional[float] = None,
    return_truncation_error: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform truncated SVD on matrix M.

    Computes M ≈ U @ diag(S) @ Vh where the number of singular values
    is determined by tolerance and/or rank constraints.

    Parameters
    ----------
    M : np.ndarray
        Matrix to decompose, shape (m, n)
    max_rank : int, optional
        Maximum number of singular values to keep
    tol : float, optional
        Relative tolerance - keep singular values >= sigma_max * tol
    return_truncation_error : bool, default False
        If True, return truncation error as 4th element

    Returns
    -------
    U : np.ndarray
        Left singular vectors, shape (m, k)
    S : np.ndarray
        Singular values (1D array), shape (k,)
    Vh : np.ndarray
        Right singular vectors (conjugate transposed), shape (k, n)
    truncation_error : float (optional)
        Frobenius norm of truncated singular values

    Notes
    -----
    - Uses standard numpy.linalg.svd (NOT recursive accurate SVD)
    - If both max_rank and tol are None, keeps all singular values
    - If both are specified, uses the minimum (most restrictive)
    - Singular values are always returned in descending order
    - Supports both real and complex matrices
    """
    # Handle empty or zero matrix
    if M.size == 0:
        raise ValueError("Cannot perform SVD on empty matrix")

    # Compute full SVD
    U_full, S_full, Vh_full = np.linalg.svd(M, full_matrices=False)

    # Determine number of singular values to keep
    n_sv = len(S_full)

    # Start with all singular values
    keep_rank = n_sv

    # Apply tolerance-based truncation
    if tol is not None:
        if tol < 0:
            raise ValueError(f"Tolerance must be non-negative, got {tol}")

        sigma_max = S_full[0] if n_sv > 0 else 0.0
        threshold = sigma_max * tol

        # Find number of singular values above threshold
        # Keep all values >= threshold
        n_above_threshold = np.sum(S_full >= threshold)
        keep_rank = min(keep_rank, n_above_threshold)

    # Apply rank-based truncation
    if max_rank is not None:
        if max_rank < 0:
            raise ValueError(f"max_rank must be non-negative, got {max_rank}")
        keep_rank = min(keep_rank, max_rank)

    # Ensure at least one singular value is kept (unless all are zero)
    if keep_rank == 0 and n_sv > 0:
        # Check if all singular values are truly zero
        if S_full[0] > 0:
            keep_rank = 1  # Keep at least one non-zero singular value

    # Truncate
    U = U_full[:, :keep_rank]
    S = S_full[:keep_rank]
    Vh = Vh_full[:keep_rank, :]

    if return_truncation_error:
        # Compute truncation error as Frobenius norm of discarded singular values
        truncation_error = np.linalg.norm(S_full[keep_rank:])
        return U, S, Vh, truncation_error
    else:
        return U, S, Vh


def reconstruct_from_svd(U: np.ndarray, S: np.ndarray, Vh: np.ndarray) -> np.ndarray:
    """
    Reconstruct matrix from SVD decomposition.

    Parameters
    ----------
    U : np.ndarray
        Left singular vectors, shape (m, k)
    S : np.ndarray
        Singular values (1D array), shape (k,)
    Vh : np.ndarray
        Right singular vectors (conjugate transposed), shape (k, n)

    Returns
    -------
    M : np.ndarray
        Reconstructed matrix M ≈ U @ diag(S) @ Vh, shape (m, n)
    """
    return U @ np.diag(S) @ Vh


def truncation_error_bound(S_full: np.ndarray, keep_rank: int) -> float:
    """
    Compute theoretical truncation error bound.

    The truncation error (in Frobenius norm) from keeping only the
    first keep_rank singular values is exactly:

    ||M - M_truncated||_F = sqrt(sum_{i=keep_rank}^{n} sigma_i^2)

    Parameters
    ----------
    S_full : np.ndarray
        Full array of singular values
    keep_rank : int
        Number of singular values to keep

    Returns
    -------
    error : float
        Frobenius norm of truncation error
    """
    if keep_rank >= len(S_full):
        return 0.0
    return np.linalg.norm(S_full[keep_rank:])
