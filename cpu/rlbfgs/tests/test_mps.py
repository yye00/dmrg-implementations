"""Unit tests for MPS utilities (canonical forms, inner product)."""

import numpy as np
import pytest

from rlbfgs.mps import (
    random_mps,
    right_canonicalize,
    left_canonicalize,
    mps_inner,
    mps_norm_squared,
)


@pytest.mark.parametrize("L,chi", [(4, 3), (6, 5), (8, 4)])
def test_right_canonical_isometry(L, chi):
    X = random_mps(L, chi, d=2, seed=42)
    X = right_canonicalize(X)
    for i in range(1, L):
        A = X[i]
        chi_L, d, chi_R = A.shape
        A_mat = A.reshape(chi_L, d * chi_R)
        prod = A_mat @ A_mat.conj().T
        np.testing.assert_allclose(
            prod, np.eye(chi_L), atol=1e-10,
            err_msg=f"site {i} is not right-canonical",
        )


@pytest.mark.parametrize("L,chi", [(4, 3), (6, 5)])
def test_left_canonical_isometry(L, chi):
    X = random_mps(L, chi, d=2, seed=7)
    X = left_canonicalize(X)
    for i in range(0, L - 1):
        A = X[i]
        chi_L, d, chi_R = A.shape
        A_mat = A.reshape(chi_L * d, chi_R)
        prod = A_mat.conj().T @ A_mat
        np.testing.assert_allclose(
            prod, np.eye(chi_R), atol=1e-10,
            err_msg=f"site {i} is not left-canonical",
        )


def test_inner_product_preserves_norm_under_canonicalization():
    X = random_mps(8, 5, d=2, seed=0)
    n0 = mps_norm_squared(X)
    X1 = right_canonicalize(X)
    X2 = left_canonicalize(X)
    assert n0 > 0
    np.testing.assert_allclose(n0, mps_norm_squared(X1), rtol=1e-10)
    np.testing.assert_allclose(n0, mps_norm_squared(X2), rtol=1e-10)
