"""Runner for the Riemannian Adam (R-Adam) CPU implementation.

R-Adam is not a DMRG variant -- there are no sweeps or local eigensolves.
The ``max_sweeps`` parameter from the benchmark harness is reinterpreted
as a budget: we run ``epochs_per_sweep * max_sweeps`` R-Adam epochs with
a cosine-annealed learning rate, or stop early on gradient-norm
tolerance.

Runs in-process via direct Python import of the ``radam`` package at
``cpu/radam``.  Small-scale problems only (see
``benchmarks/lib/registry.py::SIZES``); no challenge-grid support.

For a quasi-Newton variant that reaches ~1e-9 against DMRG on the same
small grid, see ``benchmarks/lib/runners/rlbfgs_runner.py`` (``rlbfgs``
registry entry).
"""

import os
import sys
import time

import numpy as np

from benchmarks.lib.hardware import get_repo_root


def _ensure_radam_on_path():
    radam_dir = os.path.join(get_repo_root(), "cpu", "radam")
    if radam_dir not in sys.path:
        sys.path.insert(0, radam_dir)


# Sensible per-model defaults tuned for small problems (L<=20, chi<=50).
# These keep R-Adam honest on a cold-start random MPS; users can override
# via kwargs.  The learning rate is cosine-annealed inside run_radam.
_DEFAULTS = {
    "heisenberg": {"lr": 2e-2, "epochs_per_sweep": 60, "min_lr": 1e-6},
    "tfim":       {"lr": 2e-2, "epochs_per_sweep": 60, "min_lr": 1e-6},
    "josephson":  {"lr": 1e-2, "epochs_per_sweep": 80, "min_lr": 1e-6},
}


def run(
    model,
    L,
    chi,
    max_sweeps=30,
    tol=1e-11,
    threads=1,
    n_max=2,
    lr=None,
    epochs_per_sweep=None,
    min_lr=None,
    seed=0,
    **kwargs,
):
    """Run R-Adam on a small-scale ground-state problem.

    Args:
        model: 'heisenberg', 'tfim', or 'josephson'.
        L: chain length.
        chi: MPS bond dimension.
        max_sweeps: DMRG-sweep budget from the benchmark harness; used
            here as a scaling factor for R-Adam epochs.
        tol: gradient-norm early-stop tolerance.
        threads: OPENBLAS_NUM_THREADS etc.
        n_max: charge truncation for Josephson (d = 2*n_max + 1).
        lr, epochs_per_sweep, min_lr: R-Adam hyperparameter overrides.
            Per-model defaults are in ``_DEFAULTS``.
        seed: RNG seed for the initial MPS.

    Returns:
        dict with keys energy, time, sweeps, converged, success plus
        R-Adam specific keys (epochs, grad_norm).
    """
    thread_str = str(threads)
    for var in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[var] = thread_str

    _ensure_radam_on_path()
    from radam.driver import run_radam
    from radam.mpo import (
        build_heisenberg_mpo,
        build_tfim_mpo,
        build_josephson_mpo,
    )

    defaults = _DEFAULTS.get(model)
    if defaults is None:
        raise ValueError(
            f"radam runner does not support model '{model}'; "
            "supported: heisenberg, tfim, josephson"
        )
    _lr = lr if lr is not None else defaults["lr"]
    _eps = (
        epochs_per_sweep if epochs_per_sweep is not None
        else defaults["epochs_per_sweep"]
    )
    _min_lr = min_lr if min_lr is not None else defaults["min_lr"]

    if model == "heisenberg":
        H = build_heisenberg_mpo(L, j=1.0, bz=0.0)
        d = 2
    elif model == "tfim":
        H = build_tfim_mpo(L, j=1.0, hx=1.0)
        d = 2
    elif model == "josephson":
        H = build_josephson_mpo(L, E_J=1.0, E_C=0.5, mu=0.0, n_max=n_max)
        d = 2 * n_max + 1
    else:  # unreachable
        raise ValueError(f"Unknown model: {model}")

    max_epochs = max(int(_eps * max_sweeps), 50)

    t0 = time.perf_counter()
    result = run_radam(
        H, L=L, chi=chi, d=d,
        lr=_lr, beta1=0.9, beta2=0.999, eps=1e-8,
        max_epochs=max_epochs, tol=tol,
        lr_schedule="cosine", min_lr=_min_lr,
        seed=seed, log_every=0,
    )
    elapsed = time.perf_counter() - t0

    return {
        "energy": float(result["energy"]),
        "time": elapsed,
        "sweeps": result["epochs"],
        "epochs": result["epochs"],
        "grad_norm": result["grad_norm"],
        "converged": bool(result["converged"]),
        "success": True,
    }
