"""Recursive accurate SVD for PDMRG.

Standard SVD has poor relative accuracy for small singular values.
When computing V = 1/S, small singular value errors get amplified.
This recursive refinement (from the paper's appendix) fixes that.
"""

import numpy as np


def accurate_svd(M, epsilon=1e-4):
    """Compute SVD with improved relative accuracy for small singular values.

    Uses recursive refinement: extracts the subspace of poorly-determined
    singular values and re-SVDs the projected matrix for better accuracy.

    Parameters
    ----------
    M : ndarray, shape (m, n)
        Matrix to decompose.
    epsilon : float
        Ratio threshold below which singular values are considered
        inaccurate and need refinement.

    Returns
    -------
    U : ndarray, shape (m, k)
    S : ndarray, shape (k,)
    Vh : ndarray, shape (k, n)
        Where k = min(m, n). M ≈ U @ diag(S) @ Vh.
    """
    U, S, Vh = np.linalg.svd(M, full_matrices=False)

    if len(S) == 0:
        return U, S, Vh

    # Find where relative accuracy degrades
    p = None
    s_max = S[0]
    if s_max == 0.0:
        return U, S, Vh

    for i in range(len(S)):
        if S[i] / s_max < epsilon:
            p = i
            break

    if p is None or p >= len(S) - 1:
        return U, S, Vh  # All singular values are accurate enough

    # Project onto the inaccurate subspace and re-SVD
    # X = U[:, p:]^H @ M @ Vh[p:, :]^H
    X = U[:, p:].conj().T @ M @ Vh[p:, :].conj().T
    U_sub, S_sub, Vh_sub = accurate_svd(X, epsilon)

    # Update the inaccurate portion
    U[:, p:] = U[:, p:] @ U_sub
    Vh[p:, :] = Vh_sub @ Vh[p:, :]
    S[p:] = S_sub

    return U, S, Vh


def compute_v_from_svd(S, regularization=1e-12):
    """Compute V = 1/S with regularization to prevent blowup.

    Parameters
    ----------
    S : ndarray, shape (k,)
        Singular values.
    regularization : float
        Minimum value for singular values before inversion.

    Returns
    -------
    V : ndarray, shape (k,)
        Inverse singular values, regularized.
    """
    return 1.0 / np.clip(S, regularization, None)


def truncated_svd(M, max_bond, cutoff=0.0):
    """SVD with truncation to max bond dimension.

    Parameters
    ----------
    M : ndarray, shape (m, n)
        Matrix to decompose.
    max_bond : int
        Maximum number of singular values to keep.
    cutoff : float
        Discard singular values smaller than this (default: 0, no cutoff).

    Returns
    -------
    U : ndarray, shape (m, k)
    S : ndarray, shape (k,)
    Vh : ndarray, shape (k, n)
    trunc_err : float
        Sum of discarded squared singular values (truncation error).
    """
    U, S, Vh = np.linalg.svd(M, full_matrices=False)

    # Apply bond dimension limit only
    k = min(len(S), max_bond)
    k = max(k, 1)

    trunc_err = float(np.sum(S[k:] ** 2))

    return U[:, :k], S[:k], Vh[:k, :], trunc_err
