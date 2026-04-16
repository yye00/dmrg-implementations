"""Runner for the Riemannian Adam (R-Adam) CPU implementation.

R-Adam is not a DMRG variant -- there are no sweeps or local eigensolves.
The "max_sweeps" parameter from the benchmark harness is reinterpreted
as a budget: we run ``epochs_per_sweep * max_sweeps`` R-Adam epochs (with
a cosine-annealed learning rate) or stop early on gradient-norm tolerance.

Runs in-process via direct Python import; only the ``cpu/radam`` package
and numpy are required.  Small-scale problems only (see
``benchmarks/lib/registry.py::SIZES``); no challenge-grid support.
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
_DEFAULT_RADAM_HYPERPARAMS = {
    "heisenberg": {"lr": 2e-2, "epochs_per_sweep": 60, "min_lr": 1e-6},
    "tfim":       {"lr": 2e-2, "epochs_per_sweep": 60, "min_lr": 1e-6},
    "josephson":  {"lr": 1e-2, "epochs_per_sweep": 80, "min_lr": 1e-6},
}

# Defaults for the warm-start (R-Adam warmup -> R-LBFGS polish) algorithm.
# Tuned to reach |E - E_DMRG1_chi| < 1e-9 on the small grid.
_DEFAULT_WARMSTART_HYPERPARAMS = {
    "heisenberg": dict(
        warmup_lr=2e-2, warmup_min_lr=1e-4, warmup_epochs=300,
        polish_epochs=400, polish_history=20, polish_tol=1e-12,
        polish_precondition=True, polish_ridge=1e-8,
    ),
    "tfim": dict(
        warmup_lr=2e-2, warmup_min_lr=1e-4, warmup_epochs=300,
        polish_epochs=400, polish_history=20, polish_tol=1e-12,
        polish_precondition=True, polish_ridge=1e-8,
    ),
    "josephson": dict(
        warmup_lr=2e-2, warmup_min_lr=1e-4, warmup_epochs=800,
        polish_epochs=2000, polish_history=30, polish_tol=1e-13,
        polish_precondition=True, polish_ridge=1e-4,
    ),
}


def run(
    model,
    L,
    chi,
    max_sweeps=30,
    tol=1e-11,
    threads=1,
    n_max=2,
    algorithm="radam",          # 'radam' | 'warmstart' (warmup + R-LBFGS polish)
    lr=None,
    epochs_per_sweep=None,
    min_lr=None,
    seed=0,
    **kwargs,
):
    """Run R-Adam (or R-Adam warm-up + R-LBFGS polish) on a small problem.

    Args:
        model: 'heisenberg', 'tfim', or 'josephson'.
        L: chain length.
        chi: MPS bond dimension.
        max_sweeps: DMRG-sweep budget from the benchmark harness; used
            here as a scaling factor for R-Adam epochs (algorithm='radam')
            or ignored (algorithm='warmstart' uses fixed budgets).
        tol: gradient-norm early-stop tolerance.
        threads: OPENBLAS_NUM_THREADS etc.
        n_max: charge truncation for Josephson (d = 2*n_max + 1).
        algorithm: 'radam' for plain Riemannian Adam, or 'warmstart' for
            R-Adam warmup followed by metric-preconditioned R-LBFGS
            polish.  'warmstart' is the configuration that reaches
            <1e-9 energy gap to single-site DMRG on the small grid.
        lr, epochs_per_sweep, min_lr: hyperparameter overrides for the
            'radam' algorithm.  Per-model defaults are in
            ``_DEFAULT_RADAM_HYPERPARAMS`` / ``_DEFAULT_WARMSTART_HYPERPARAMS``.
        seed: RNG seed for the initial MPS.

    Returns:
        dict with keys energy, time, sweeps, converged, success plus
        algorithm-specific keys (epochs, grad_norm; for warmstart also
        warmup_epochs, polish_epochs).
    """
    thread_str = str(threads)
    for var in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[var] = thread_str

    _ensure_radam_on_path()
    from radam.driver import run_radam, run_warmstart_rlbfgs
    from radam.mpo import (
        build_heisenberg_mpo,
        build_tfim_mpo,
        build_josephson_mpo,
    )

    if model == "heisenberg":
        H = build_heisenberg_mpo(L, j=1.0, bz=0.0)
        d = 2
    elif model == "tfim":
        H = build_tfim_mpo(L, j=1.0, hx=1.0)
        d = 2
    elif model == "josephson":
        H = build_josephson_mpo(L, E_J=1.0, E_C=0.5, mu=0.0, n_max=n_max)
        d = 2 * n_max + 1
    else:
        raise ValueError(
            f"radam runner does not support model '{model}'; "
            "supported: heisenberg, tfim, josephson"
        )

    if algorithm == "radam":
        defaults = _DEFAULT_RADAM_HYPERPARAMS[model]
        _lr = lr if lr is not None else defaults["lr"]
        _eps_per_sw = (
            epochs_per_sweep if epochs_per_sweep is not None
            else defaults["epochs_per_sweep"]
        )
        _min_lr = min_lr if min_lr is not None else defaults["min_lr"]
        max_epochs = max(int(_eps_per_sw * max_sweeps), 50)
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
            "algorithm": "radam",
        }

    elif algorithm == "warmstart":
        params = dict(_DEFAULT_WARMSTART_HYPERPARAMS[model])
        # Apply optional per-call overrides.
        for k, v in kwargs.items():
            if k in params:
                params[k] = v
        t0 = time.perf_counter()
        result = run_warmstart_rlbfgs(
            H, L=L, chi=chi, d=d,
            seed=seed, log_every=0,
            **params,
        )
        elapsed = time.perf_counter() - t0
        return {
            "energy": float(result["energy"]),
            "time": elapsed,
            "sweeps": result["epochs"],
            "epochs": result["epochs"],
            "warmup_epochs": result["warmup_epochs"],
            "polish_epochs": result["polish_epochs"],
            "grad_norm": result["grad_norm"],
            "converged": bool(result["converged"]),
            "success": True,
            "algorithm": "warmstart",
        }

    else:
        raise ValueError(
            f"unknown algorithm '{algorithm}'; expected 'radam' or 'warmstart'"
        )
