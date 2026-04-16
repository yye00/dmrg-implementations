"""Tangent-space projection + retraction sanity tests."""

import numpy as np
import pytest

from rlbfgs.mps import random_mps, mps_norm_squared, zeros_like_mps, mps_inner
from rlbfgs.projection import project_right_canonical
from rlbfgs.retraction import retract_and_recanonicalize


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
        )


def test_retraction_zero_delta_is_identity_up_to_phase():
    X = random_mps(6, 4, seed=5)
    zero = zeros_like_mps(X)
    Y = retract_and_recanonicalize(X, zero)
    overlap = mps_inner(X, Y)
    nx = mps_norm_squared(X)
    ny = mps_norm_squared(Y)
    # Retraction normalizes to <Y|Y>=1, so norm may differ from <X|X>.
    assert ny > 0
    np.testing.assert_allclose(abs(overlap) ** 2 / (nx * ny), 1.0, rtol=1e-9)


def test_retraction_normalizes_norm():
    X = random_mps(6, 4, seed=5)
    zero = zeros_like_mps(X)
    Y = retract_and_recanonicalize(X, zero)
    np.testing.assert_allclose(mps_norm_squared(Y), 1.0, atol=1e-10)
