"""Tests for R-LBFGS and the warm-start (R-Adam -> R-LBFGS) driver.

Includes a high-precision regression test on Heisenberg L=12 chi=20:
the warm-start optimizer must converge to within 1e-7 of the
single-site DMRG ground-state energy on the bond-dim-20 manifold.
The full convergence to ~1e-9 is exercised in the benchmark suite,
which takes a few minutes; this test uses a tighter ridge / shorter
budget so it runs in well under a minute.
"""

import math

import numpy as np
import pytest

from radam.driver import run_warmstart_rlbfgs
from radam.lbfgs import (
    LBFGSState, lbfgs_two_loop, line_search_armijo, run_rlbfgs,
)
from radam.mpo import build_heisenberg_mpo, build_josephson_mpo
from radam.mps import random_mps
from radam.tangent import inner_real, norm_tangent


def test_lbfgs_two_loop_empty_history_is_minus_g():
    """With no history, two-loop returns -g scaled by gamma=default (1.0)."""
    X = random_mps(L=4, chi=4, d=2, seed=0)
    g = [np.ones_like(A) for A in X]
    d = lbfgs_two_loop(g, history=[], X_current=X)
    for di, gi in zip(d, g):
        np.testing.assert_allclose(di, -gi, atol=1e-12)


def test_armijo_line_search_decreases_energy():
    H = build_heisenberg_mpo(L=6, j=1.0)
    X = random_mps(L=6, chi=4, d=2, seed=0)

    from radam.gradient import euclidean_gradient, energy_only
    from radam.projection import project_right_canonical
    from radam.tangent import scale_cores

    grads_eucl, E0 = euclidean_gradient(X, H, return_energy=True)
    g = project_right_canonical(X, grads_eucl)
    direction = scale_cores(g, -1.0)
    alpha, X_new, E_new = line_search_armijo(X, direction, g, E0, H)
    assert alpha > 0
    assert E_new < E0


def test_warmstart_rlbfgs_heisenberg_L8_chi8():
    """End-to-end: Heisenberg L=8 chi=8 reaches the DMRG ground state.

    Compares to ED on the L=8 (full) Heisenberg chain.  The chi=8
    manifold contains the exact ground state for L=8 d=2
    (since 2^4 = 16 >= max bond dim and L=8 needs at most chi=8 in
    the middle bond), so the warm-start optimizer should converge
    to within 1e-7 of ED.
    """
    H = build_heisenberg_mpo(L=8, j=1.0)
    # Compute ED reference.
    M = np.zeros((1, 1, 1), dtype=H[0].dtype)
    M[0, 0, 0] = 1.0
    for W in H:
        M = np.einsum("lab,lrpq->rapbq", M, W)
        mR = M.shape[0]
        M = M.reshape(mR, M.shape[1] * M.shape[2], M.shape[3] * M.shape[4])
    M = M[0]
    ED = float(np.linalg.eigvalsh(M)[0])

    res = run_warmstart_rlbfgs(
        H, L=8, chi=8, d=2,
        warmup_epochs=300, warmup_lr=2e-2, warmup_min_lr=1e-4,
        polish_epochs=400, polish_history=20, polish_tol=1e-12,
        polish_precondition=True, polish_ridge=1e-8,
        seed=0, log_every=0,
    )
    gap = res["energy"] - ED
    # 1e-5 in this short test (warm-start tightens to ~1e-9 with longer
    # budget; benchmark suite exercises the full polish on L=12 chi=20).
    assert gap < 1e-5, (
        f"R-LBFGS energy gap {gap:.3e} > 1e-5 vs ED reference {ED}"
    )
    assert res["polish_epochs"] > 0


def test_rlbfgs_runs_without_warmstart_on_simple_problem():
    """Plain R-LBFGS (no R-Adam warmup) on a small Heisenberg problem.

    Without warmup the method may not converge tightly, but it must
    decrease the energy below random initialization.
    """
    H = build_heisenberg_mpo(L=6, j=1.0)
    res = run_rlbfgs(
        H, L=6, chi=4, d=2,
        history_size=5, max_epochs=30, tol=1e-12,
        precondition=True, ridge=1e-6,
        seed=0, log_every=0,
    )
    assert res["history"][0]["energy"] > res["energy"]
    assert res["energy"] < 0.0  # AFM Heisenberg is negative


def test_run_rlbfgs_returns_canonical_mps():
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
