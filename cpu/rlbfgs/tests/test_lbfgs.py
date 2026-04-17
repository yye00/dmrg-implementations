"""Tests for the R-LBFGS optimizer and the warm-start driver.

Includes a regression test on Heisenberg L=8 chi=8 that the warm-start
driver converges to within 1e-5 of the ED ground state (the chi=8
truncation error is ~1e-6 on this problem, so 1e-5 is a tight but safe
bound).  The benchmark suite exercises the larger-chi convergence to
< 1e-9 vs single-site DMRG.
"""

import numpy as np
import pytest

from rlbfgs.driver import run_rlbfgs, run_rlbfgs_warmstart
from rlbfgs.mpo import build_heisenberg_mpo
from rlbfgs.mps import random_mps
from rlbfgs.optimizer import (
    LBFGSState, lbfgs_two_loop, line_search_armijo,
)
from rlbfgs.tangent import inner_real


def test_lbfgs_two_loop_empty_history_is_minus_g():
    X = random_mps(L=4, chi=4, d=2, seed=0)
    g = [np.ones_like(A) for A in X]
    d = lbfgs_two_loop(g, history=[], X_current=X)
    for di, gi in zip(d, g):
        np.testing.assert_allclose(di, -gi, atol=1e-12)


def test_armijo_line_search_decreases_energy():
    H = build_heisenberg_mpo(L=6, j=1.0)
    X = random_mps(L=6, chi=4, d=2, seed=0)

    from rlbfgs.gradient import euclidean_gradient
    from rlbfgs.projection import project_right_canonical
    from rlbfgs.tangent import scale_cores

    grads_eucl, E0 = euclidean_gradient(X, H, return_energy=True)
    g = project_right_canonical(X, grads_eucl)
    direction = scale_cores(g, -1.0)
    alpha, X_new, E_new = line_search_armijo(X, direction, g, E0, H)
    assert alpha > 0
    assert E_new < E0


def test_warmstart_rlbfgs_heisenberg_L8_chi8():
    """End-to-end: Heisenberg L=8 chi=8 warm-start reaches < 1e-5 of ED."""
    H = build_heisenberg_mpo(L=8, j=1.0)
    # ED reference.
    M = np.zeros((1, 1, 1), dtype=H[0].dtype)
    M[0, 0, 0] = 1.0
    for W in H:
        M = np.einsum("lab,lrpq->rapbq", M, W)
        mR = M.shape[0]
        M = M.reshape(mR, M.shape[1] * M.shape[2], M.shape[3] * M.shape[4])
    M = M[0]
    ED = float(np.linalg.eigvalsh(M)[0])

    res = run_rlbfgs_warmstart(
        H, L=8, chi=8, d=2,
        warmup_epochs=150, warmup_history=20,
        polish_epochs=400, polish_history=20, polish_tol=1e-12,
        polish_ridge=1e-8,
        seed=0, log_every=0,
    )
    gap = res["energy"] - ED
    assert gap < 1e-5, f"R-LBFGS energy gap {gap:.3e} > 1e-5 vs ED {ED}"
    assert res["polish_epochs"] > 0


def test_rlbfgs_plain_runs_from_random():
    """Plain R-LBFGS without warmstart decreases energy from random init."""
    H = build_heisenberg_mpo(L=6, j=1.0)
    res = run_rlbfgs(
        H, L=6, chi=4, d=2,
        history_size=5, max_epochs=30, tol=1e-12,
        precondition=False, seed=0, log_every=0,
    )
    assert res["history"][0]["energy"] > res["energy"]
    assert res["energy"] < 0.0


def test_rlbfgs_returns_canonical_mps():
    H = build_heisenberg_mpo(L=5, j=1.0)
    res = run_rlbfgs(
        H, L=5, chi=4, d=2,
        history_size=5, max_epochs=10, tol=1e-12,
        precondition=False, seed=1, log_every=0,
    )
    X = res["mps"]
    for i in range(1, len(X)):
        A = X[i]
        chi_L, d, chi_R = A.shape
        Amat = A.reshape(chi_L, d * chi_R)
        prod = Amat @ Amat.conj().T
        np.testing.assert_allclose(
            prod, np.eye(chi_L), atol=1e-9,
            err_msg=f"site {i} not right-canonical after R-LBFGS",
        )
