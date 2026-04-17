"""Gradient correctness tests: finite-difference check against
projected Euclidean gradient."""

import numpy as np
import pytest

from rlbfgs.gradient import euclidean_gradient, energy_only
from rlbfgs.mpo import build_heisenberg_mpo
from rlbfgs.mps import random_mps


@pytest.mark.parametrize("L,chi,site", [(4, 3, 0), (4, 3, 1), (4, 3, 2), (4, 3, 3), (6, 4, 2)])
def test_gradient_matches_finite_difference(L, chi, site):
    X = random_mps(L, chi, d=2, seed=11)
    H = build_heisenberg_mpo(L, j=1.0, bz=0.1)
    grads, E0 = euclidean_gradient(X, H, return_energy=True)

    rng = np.random.default_rng(2024)
    V = (
        rng.standard_normal(X[site].shape)
        + 1j * rng.standard_normal(X[site].shape)
    )
    eps = 1e-6
    X_plus = [A.copy() for A in X]
    X_plus[site] = X_plus[site] + eps * V
    X_minus = [A.copy() for A in X]
    X_minus[site] = X_minus[site] - eps * V
    E_plus = energy_only(X_plus, H)
    E_minus = energy_only(X_minus, H)

    fd = (E_plus - E_minus) / (2 * eps)
    analytic = 2.0 * np.real(np.vdot(grads[site], V))
    np.testing.assert_allclose(fd, analytic, rtol=1e-4, atol=1e-6)
