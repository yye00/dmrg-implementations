"""Smoke test for the benchmark runner wiring.

Verifies that ``benchmarks.lib.runners.radam_runner.run`` executes
end-to-end for each supported model on tiny problems and returns the
expected result-dict keys.  Does NOT compare against a gold-standard
energy -- that is the job of the larger validate / benchmark suite.
"""

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


@pytest.mark.parametrize("model,L,chi", [
    ("heisenberg", 6, 8),
    ("tfim",       6, 8),
    ("josephson",  4, 8),
])
def test_benchmark_runner_returns_valid_dict(model, L, chi):
    from benchmarks.lib.runners.radam_runner import run
    result = run(
        model=model, L=L, chi=chi,
        max_sweeps=2, tol=1e-8, threads=1, n_max=2,
        epochs_per_sweep=30,
    )
    # Required keys per the runner contract.
    for key in ("energy", "time", "sweeps", "converged", "success"):
        assert key in result, f"missing key {key!r} in runner result"
    assert result["success"] is True
    assert isinstance(result["energy"], float)
    assert result["time"] > 0.0
    assert result["sweeps"] >= 1
    # R-Adam specific extra keys.
    assert "epochs" in result
    assert "grad_norm" in result


def test_benchmark_runner_rejects_unknown_model():
    from benchmarks.lib.runners.radam_runner import run
    with pytest.raises(ValueError, match="radam runner does not support"):
        run(model="bogus", L=4, chi=4, max_sweeps=2, tol=1e-6, threads=1)
