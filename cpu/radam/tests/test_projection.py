"""Tangent-space projection and retraction sanity checks."""

import numpy as np
import pytest

from radam.mps import random_mps, right_canonicalize, mps_norm_squared, zeros_like_mps
from radam.projection import project_right_canonical
from radam.retraction import retract_and_recanonicalize


def _random_cores(X, seed):
    rng = np.random.default_rng(seed)
    return [
        (rng.standard_normal(A.shape) + 1j * rng.standard_normal(A.shape)).astype(A.dtype)
        for A in X
    ]


@pytest.mark.parametrize("L,chi", [(4, 3), (6, 5)])
def test_projection_is_idempotent(L, chi):
    X = random_mps(L, chi, seed=1)
    V = _random_cores(X, seed=2)
    V1 = project_right_canonical(X, V)
    V2 = project_right_canonical(X, V1)
    for i in range(L):
        np.testing.assert_allclose(V1[i], V2[i], atol=1e-10)


@pytest.mark.parametrize("L,chi", [(4, 3), (6, 5)])
def test_projection_is_gauge_orthogonal(L, chi):
    """For i >= 1 the projected core satisfies V_i @ A_i^H = 0."""
    X = random_mps(L, chi, seed=3)
    V = _random_cores(X, seed=4)
    Vp = project_right_canonical(X, V)
    for i in range(1, L):
        A = X[i]
        chi_L, d, chi_R = A.shape
        A_mat = A.reshape(chi_L, d * chi_R)
        V_mat = Vp[i].reshape(chi_L, d * chi_R)
        prod = V_mat @ A_mat.conj().T
        np.testing.assert_allclose(
            prod, np.zeros_like(prod), atol=1e-10,
            err_msg=f"site {i} projection failed gauge-orthogonality",
        )


def test_retraction_zero_delta_is_identity():
    X = random_mps(6, 4, seed=5)
    zero = zeros_like_mps(X)
    Y = retract_and_recanonicalize(X, zero)
    # Right-canonicalizing a right-canonical MPS is the identity up to signs.
    # Compare norms and inner products instead (phase/sign-invariant check).
    from radam.mps import mps_inner
    overlap = mps_inner(X, Y)
    nx = mps_norm_squared(X)
    ny = mps_norm_squared(Y)
    np.testing.assert_allclose(nx, ny, rtol=1e-10)
    np.testing.assert_allclose(abs(overlap) ** 2 / (nx * ny), 1.0, rtol=1e-9)


def test_retraction_small_delta_small_change():
    """For a tiny tangent step, ||X_new - X|| is O(|delta|) in overlap."""
    X = random_mps(5, 3, seed=7)
    V = project_right_canonical(X, _random_cores(X, seed=8))
    # Scale V small.
    V_small = [1e-3 * v for v in V]
    Y = retract_and_recanonicalize(X, V_small)

    from radam.mps import mps_inner, mps_norm_squared
    overlap = mps_inner(X, Y)
    nx = mps_norm_squared(X)
    ny = mps_norm_squared(Y)
    fidelity = abs(overlap) ** 2 / (nx * ny)
    # Overlap should be very close to 1.
    assert fidelity > 0.99
