"""End-to-end R-Adam smoke test on small Heisenberg / TFIM chains.

We just verify that the optimizer reduces the energy and drives the
Riemannian gradient norm down from the random initialization.  Exact
ground-state energy comparison is not part of this test -- it is a
correctness *regression* test, not a physics-accuracy test.
"""

import numpy as np
import pytest

from radam.driver import run_heisenberg, run_tfim


def test_radam_heisenberg_L4_chi4_matches_ed():
    """L=4 Heisenberg ground state from ED is -1.6160; R-Adam should get within 1e-2."""
    exact = -1.6160254037844388
    res = run_heisenberg(
        L=4, chi=4, j=1.0, bz=0.0,
        lr=2e-2, max_epochs=500,
        lr_schedule="cosine", min_lr=1e-7,
        seed=0, log_every=0, tol=1e-8,
    )
    assert res["energy"] - exact < 1e-2, (
        f"energy {res['energy']} did not reach exact ground state {exact}"
    )
    # Gradient norm should shrink by at least an order of magnitude.
    assert res["grad_norm"] < res["history"][0]["grad_norm"] / 10.0


def test_radam_tfim_L6_chi16_decreases_energy():
    """L=6 critical TFIM ED ground state is -3.1577.

    Critical TFIM is notoriously hard to converge (power-law correlations);
    we only require a ~10% energy gap close and an order-of-magnitude drop
    in the Riemannian gradient norm as a regression check.
    """
    exact = -3.1577410691232437
    res = run_tfim(
        L=6, chi=16, j=1.0, hx=1.0,
        lr=2e-2, max_epochs=500,
        lr_schedule="cosine", min_lr=1e-7,
        seed=0, log_every=0, tol=1e-8,
    )
    assert res["energy"] - exact < 1e-1
    assert res["grad_norm"] < res["history"][0]["grad_norm"] / 10.0


def test_radam_preserves_canonical_form():
    """After every retraction, the MPS should remain right-canonical."""
    import numpy as np
    res = run_heisenberg(
        L=5, chi=4, j=1.0,
        lr=1e-2, max_epochs=20,
        lr_schedule="cosine", seed=1, log_every=0,
    )
    X = res["mps"]
    for i in range(1, len(X)):
        A = X[i]
        chi_L, d, chi_R = A.shape
        A_mat = A.reshape(chi_L, d * chi_R)
        prod = A_mat @ A_mat.conj().T
        np.testing.assert_allclose(
            prod, np.eye(chi_L), atol=1e-9,
            err_msg=f"site {i} is not right-canonical after optimization",
        )
