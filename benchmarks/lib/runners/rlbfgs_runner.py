"""Runner for the Riemannian L-BFGS (R-LBFGS) CPU implementation.

Runs in-process via direct Python import of the ``rlbfgs`` package at
``cpu/rlbfgs``.  On the small benchmark grid this reaches
|E - E_DMRG1_chi| < 1e-9 -- ~5-8 orders of magnitude tighter than
plain R-Adam.

The ``max_sweeps`` argument from the harness is ignored -- R-LBFGS
uses fixed epoch budgets baked into ``_DEFAULTS``.  That is because
R-LBFGS's convergence behaviour is very different from a DMRG sweep
and the DMRG sweep budget is not a useful proxy for an L-BFGS
iteration count.  The caller can override via kwargs
(``warmup_epochs``, ``polish_epochs``, ``polish_ridge``, etc.).

Warm-up strategy
----------------

By default the runner uses **R-Adam** (imported from the sibling
``radam`` package at ``cpu/radam``) as the cold-start warmup, then
feeds the resulting MPS into the ``rlbfgs`` package's metric-
preconditioned polish phase.  Empirically this reaches ~1e-9 vs
single-site DMRG on Josephson L=8 chi=20.

If ``cpu/radam`` is not importable, the runner falls back to R-LBFGS's
own two-phase driver (``run_rlbfgs_warmstart``), which uses
un-preconditioned L-BFGS as the warmup.  That reaches ~1e-7 on
Josephson (still much tighter than plain R-Adam at ~1e-2).

The ``rlbfgs`` package itself has no dependency on ``radam``; the
cross-package orchestration lives in this runner.

Small-scale problems only (see ``benchmarks/lib/registry.py::SIZES``);
no challenge-grid support, no GPU path.
"""

import os
import sys
import time

import numpy as np

from benchmarks.lib.hardware import get_repo_root


def _ensure_rlbfgs_on_path():
    rlbfgs_dir = os.path.join(get_repo_root(), "cpu", "rlbfgs")
    if rlbfgs_dir not in sys.path:
        sys.path.insert(0, rlbfgs_dir)


def _ensure_radam_on_path():
    radam_dir = os.path.join(get_repo_root(), "cpu", "radam")
    if radam_dir not in sys.path:
        sys.path.insert(0, radam_dir)


# Per-model defaults.  Tuned on the small grid to reach < 1e-9 vs
# single-site DMRG at the same bond dim.  The warmup uses R-Adam
# (first-order) which is a better initialiser than un-preconditioned
# L-BFGS for d=5 problems with complex128 arithmetic (Josephson).
_DEFAULTS = {
    "heisenberg": dict(
        use_radam_warmup=True,
        warmup_epochs=300, warmup_lr=2e-2, warmup_min_lr=1e-4,
        polish_epochs=500, polish_history=20, polish_tol=1e-13,
        polish_ridge=1e-8,
    ),
    "tfim": dict(
        use_radam_warmup=True,
        warmup_epochs=300, warmup_lr=2e-2, warmup_min_lr=1e-4,
        polish_epochs=500, polish_history=20, polish_tol=1e-13,
        polish_ridge=1e-8,
    ),
    "josephson": dict(
        use_radam_warmup=True,
        warmup_epochs=800, warmup_lr=2e-2, warmup_min_lr=1e-4,
        polish_epochs=2000, polish_history=30, polish_tol=1e-13,
        polish_ridge=1e-4,
    ),
}


def run(
    model,
    L,
    chi,
    max_sweeps=30,          # ignored -- see module docstring
    tol=1e-11,              # ignored -- we use polish_tol inside _DEFAULTS
    threads=1,
    n_max=2,
    seed=0,
    **kwargs,
):
    """Run R-LBFGS (warm-start) on a small-scale ground-state problem.

    Args:
        model: 'heisenberg', 'tfim', or 'josephson'.
        L: chain length.
        chi: MPS bond dimension.
        max_sweeps: ignored -- R-LBFGS uses fixed epoch budgets.
        tol: ignored -- see ``polish_tol`` in ``_DEFAULTS``.
        threads: OPENBLAS_NUM_THREADS etc.
        n_max: charge truncation for Josephson (d = 2*n_max + 1).
        seed: RNG seed for the initial MPS.

    kwargs (all optional overrides of ``_DEFAULTS[model]``):
        warmup_epochs, warmup_history,
        polish_epochs, polish_history, polish_tol, polish_ridge.

    Returns:
        dict with keys energy, time, sweeps, converged, success plus
        R-LBFGS specific keys (epochs, warmup_epochs, polish_epochs,
        grad_norm).
    """
    thread_str = str(threads)
    for var in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[var] = thread_str

    _ensure_rlbfgs_on_path()
    from rlbfgs.driver import run_rlbfgs, run_rlbfgs_warmstart
    from rlbfgs.mpo import (
        build_heisenberg_mpo,
        build_tfim_mpo,
        build_josephson_mpo,
    )

    params = dict(_DEFAULTS.get(model) or {})
    if not params:
        raise ValueError(
            f"rlbfgs runner does not support model '{model}'; "
            "supported: heisenberg, tfim, josephson"
        )
    # Apply overrides.
    for k, v in kwargs.items():
        if k in params:
            params[k] = v

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

    use_radam = bool(params.pop("use_radam_warmup", True))
    warmup_epochs = int(params.pop("warmup_epochs"))
    warmup_lr = float(params.pop("warmup_lr", 2e-2))
    warmup_min_lr = float(params.pop("warmup_min_lr", 1e-4))
    polish_epochs = int(params.pop("polish_epochs"))
    polish_history = int(params.pop("polish_history"))
    polish_tol = float(params.pop("polish_tol"))
    polish_ridge = float(params.pop("polish_ridge"))

    t0 = time.perf_counter()

    # Phase 1: warmup -- R-Adam if available, else un-preconditioned L-BFGS.
    initial_mps = None
    warmup_wall = 0.0
    used_radam = False
    if use_radam:
        try:
            _ensure_radam_on_path()
            from radam.driver import run_radam as radam_run
            t_w = time.perf_counter()
            warm = radam_run(
                H, L=L, chi=chi, d=d,
                lr=warmup_lr, max_epochs=warmup_epochs, tol=0.0,
                lr_schedule="cosine", min_lr=warmup_min_lr,
                seed=seed, log_every=0,
            )
            initial_mps = warm["mps"]
            warmup_wall = time.perf_counter() - t_w
            used_radam = True
        except ImportError:
            initial_mps = None  # falls through to L-BFGS warmstart

    if initial_mps is not None:
        # Phase 2: preconditioned R-LBFGS polish.
        polish = run_rlbfgs(
            H, L=L, chi=chi, d=d,
            initial_mps=initial_mps,
            history_size=polish_history,
            max_epochs=polish_epochs,
            tol=polish_tol,
            precondition=True,
            ridge=polish_ridge,
            seed=seed,
            log_every=0,
        )
        epochs_total = warmup_epochs + polish["epochs"]
        final = {
            "energy": polish["energy"],
            "grad_norm": polish["grad_norm"],
            "epochs": epochs_total,
            "warmup_epochs": warmup_epochs,
            "polish_epochs": polish["epochs"],
            "converged": polish["converged"],
        }
    else:
        # Fallback: rlbfgs's own two-phase driver (L-BFGS warmup +
        # preconditioned polish).
        result = run_rlbfgs_warmstart(
            H, L=L, chi=chi, d=d,
            warmup_epochs=warmup_epochs, warmup_history=20,
            polish_epochs=polish_epochs, polish_history=polish_history,
            polish_tol=polish_tol, polish_ridge=polish_ridge,
            seed=seed, log_every=0,
        )
        final = {
            "energy": result["energy"],
            "grad_norm": result["grad_norm"],
            "epochs": result["epochs"],
            "warmup_epochs": result["warmup_epochs"],
            "polish_epochs": result["polish_epochs"],
            "converged": result["converged"],
        }

    elapsed = time.perf_counter() - t0

    return {
        "energy": float(final["energy"]),
        "time": elapsed,
        "sweeps": final["epochs"],
        "epochs": final["epochs"],
        "warmup_epochs": final["warmup_epochs"],
        "polish_epochs": final["polish_epochs"],
        "grad_norm": final["grad_norm"],
        "converged": bool(final["converged"]),
        "success": True,
        "warmup_impl": "radam" if used_radam else "lbfgs",
    }
