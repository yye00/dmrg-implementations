"""Correctness tests for the Euclidean gradient of the Rayleigh quotient.

Numerically verifies ``dE/dA_i*`` against finite differences.  Because
the energy is real-valued in both ``A`` and ``A*``, we test the
gradient by perturbing in a *real* direction: for any ndarray ``V``
with the same shape as ``A_i``, the directional derivative of
``E(A + t V, A* + t V*) - E(A, A*)`` at ``t = 0`` equals
``2 * Re(<grad, V>)`` where
``<grad, V> = sum_{entries} conj(grad[k]) V[k]``.
"""

import numpy as np
import pytest

from radam.mps import random_mps
from radam.mpo import build_heisenberg_mpo
from radam.gradient import euclidean_gradient


def _energy(X, H):
    from radam.environments import build_left_envs_H, build_left_envs_norm
    L = len(X)
    xhx = complex(build_left_envs_H(X, H)[L][0, 0, 0])
    xx = float(np.real(build_left_envs_norm(X)[L][0, 0]))
    return np.real(xhx / xx)


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
    E_plus = _energy(X_plus, H)
    E_minus = _energy(X_minus, H)

    fd = (E_plus - E_minus) / (2 * eps)
    analytic = 2.0 * np.real(np.vdot(grads[site], V))

    np.testing.assert_allclose(fd, analytic, rtol=1e-4, atol=1e-6)
