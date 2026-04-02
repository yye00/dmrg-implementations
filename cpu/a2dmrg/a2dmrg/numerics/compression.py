"""Numpy TT-SVD compression for MPS arrays."""
import numpy as np


def tt_svd_compress(arrays, max_bond, normalize=True):
    """Compress MPS arrays via left-to-right TT-SVD.

    Parameters
    ----------
    arrays : list of ndarray
        MPS tensors in (chi_L, d, chi_R) format.
    max_bond : int
        Maximum bond dimension after compression.
    normalize : bool
        Normalize the result.

    Returns
    -------
    compressed : list of ndarray
        Compressed MPS tensors.
    """
    L = len(arrays)
    result = [a.copy() for a in arrays]

    # Left-to-right SVD sweep
    for i in range(L - 1):
        chi_L, d, chi_R = result[i].shape
        mat = result[i].reshape(chi_L * d, chi_R)
        U, S, Vh = np.linalg.svd(mat, full_matrices=False)

        # Truncate to max_bond
        chi_new = min(max_bond, len(S))
        U = U[:, :chi_new]
        S = S[:chi_new]
        Vh = Vh[:chi_new, :]

        result[i] = U.reshape(chi_L, d, chi_new)
        # Absorb S @ Vh into next tensor
        SVh = np.diag(S) @ Vh
        result[i + 1] = np.tensordot(SVh, result[i + 1], axes=(1, 0))

    if normalize:
        # Normalize last tensor
        norm = np.linalg.norm(result[-1])
        if norm > 1e-15:
            result[-1] = result[-1] / norm

    return result
