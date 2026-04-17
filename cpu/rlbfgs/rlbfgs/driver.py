"""Top-level R-LBFGS drivers.

Two entry points:

* :func:`run_rlbfgs` -- a single-phase R-LBFGS loop (no warmup).
* :func:`run_rlbfgs_warmstart` -- **two-phase** R-LBFGS.  Phase 1 is
  plain (un-preconditioned) R-LBFGS, which is well-conditioned at
  random initialisation.  Phase 2 turns on the per-site metric
  preconditioner ``L_norm[i]^{-1}`` and delivers quasi-second-order
  convergence to high precision.  This is the recommended entry point.

Both drivers are self-contained: no dependency on the ``radam``
package.  If you do have a good warm-start MPS from elsewhere (e.g.,
a previous R-Adam run) pass it as ``initial_mps`` to
:func:`run_rlbfgs_warmstart` and it will be right-canonicalized and
used as the starting point.
"""

from __future__ import annotations

import math
import time
from typing import Callable, List, Optional

import numpy as np

from .mpo import build_heisenberg_mpo, build_tfim_mpo, build_josephson_mpo
from .mps import random_mps, right_canonicalize
from .optimizer import LBFGSState, rlbfgs_step


def _log(ep, info, phase=""):
    print(
        f"rlbfgs {phase}ep {ep:5d}  E={info['energy']: .14f}  "
        f"||G||={info['grad_norm']:.3e}  alpha={info['step_size']:.2e}  "
        f"|hist|={info['history_len']}"
        + ("  [fallback]" if info["fallback_to_grad"] else "")
        + ("  [ls-fail]" if info["ls_failed"] else "")
    )


def run_rlbfgs(
    H,
    L: int,
    chi: int,
    d: int = 2,
    *,
    initial_mps=None,
    history_size: int = 20,
    max_epochs: int = 500,
    tol: float = 1e-10,
    seed: Optional[int] = 0,
    log_every: int = 0,
    line_search: str = "wolfe",
    line_search_kwargs: Optional[dict] = None,
    callback: Optional[Callable[[int, dict], None]] = None,
    precondition: bool = False,
    ridge: float = 1e-10,
    dtype=np.complex128,
    _phase_label: str = "",
):
    """Run R-LBFGS for up to ``max_epochs`` iterations.

    Parameters
    ----------
    H : list of ndarray
        MPO cores (see :mod:`rlbfgs.mpo`).
    L, chi, d : int
        Problem dimensions; only used when ``initial_mps`` is ``None``.
    initial_mps : list of ndarray or None
        Starting MPS.  If ``None``, a random right-canonical MPS is
        drawn with seed ``seed``; otherwise the supplied MPS is
        right-canonicalized.
    history_size : int
        L-BFGS memory ``m``.
    max_epochs : int
    tol : float
        Early-stop threshold on the physical Riemannian gradient norm.
    seed : int or None
    log_every : int
        Print every ``log_every`` iterations (0 disables).
    line_search : {'wolfe', 'armijo'}
    line_search_kwargs : dict or None
        Forwarded to the line search.
    callback : callable(ep, info) or None
    precondition : bool
        Enable metric preconditioning by ``L_norm[i]^{-1}``.
    ridge : float
        Ridge regularisation for the preconditioner inverse.
    """
    if initial_mps is None:
        X = random_mps(L, chi, d=d, dtype=dtype, seed=seed)
    else:
        X = right_canonicalize([A.copy() for A in initial_mps])

    state = LBFGSState(X=X, history_size=history_size)
    log = []

    t0 = time.perf_counter()
    converged = False
    for ep in range(1, max_epochs + 1):
        info = rlbfgs_step(
            state, H,
            line_search=line_search,
            line_search_kwargs=line_search_kwargs,
            precondition=precondition,
            ridge=ridge,
        )
        info["epoch"] = ep
        log.append(info)
        if callback is not None:
            callback(ep, info)
        if log_every > 0 and (ep == 1 or ep % log_every == 0):
            _log(ep, info, phase=_phase_label)
        if info["grad_norm"] < tol:
            converged = True
            break

    wall = time.perf_counter() - t0
    final = log[-1]
    if log_every > 0:
        print(
            f"rlbfgs {_phase_label}done  converged={converged}  "
            f"epochs={len(log)}  E={final['energy']:.14f}  "
            f"||G||={final['grad_norm']:.3e}  wall={wall:.2f}s"
        )

    return {
        "mps": state.X,
        "energy": final["energy"],
        "grad_norm": final["grad_norm"],
        "epochs": len(log),
        "history": log,
        "wall_time": wall,
        "converged": converged,
    }


def run_rlbfgs_warmstart(
    H,
    L: int,
    chi: int,
    d: int = 2,
    *,
    initial_mps=None,
    # Phase 1: no-preconditioning warmup.
    warmup_epochs: int = 500,
    warmup_history: int = 20,
    warmup_tol: float = 0.0,                # never early-stop in warmup
    # Phase 2: preconditioned polish.
    polish_epochs: int = 1500,
    polish_history: int = 30,
    polish_tol: float = 1e-12,
    polish_ridge: float = 1e-6,
    # Common.
    line_search: str = "wolfe",
    line_search_kwargs: Optional[dict] = None,
    seed: Optional[int] = 0,
    log_every: int = 0,
    dtype=np.complex128,
):
    """Two-phase R-LBFGS: unpreconditioned warmup + preconditioned polish.

    Phase 1 uses plain L-BFGS with Strong-Wolfe line search (no
    metric preconditioning).  This is the best single option at random
    initialisation because ``L_norm[i]`` may be rank-deficient on a
    fresh random MPS and the preconditioner ``L_norm^{-1}`` blows up.

    Phase 2 turns on metric preconditioning.  Near the minimum the
    preconditioned direction approximates the Newton direction on the
    manifold and convergence accelerates from geometric to
    quasi-quadratic.

    Returns the same shape as :func:`run_rlbfgs` with two extra
    keys: ``warmup_epochs``, ``polish_epochs``.
    """
    if log_every > 0:
        print(
            f"--- R-LBFGS warm-up (no precond): {warmup_epochs} epochs, "
            f"history={warmup_history} ---"
        )
    warm = run_rlbfgs(
        H, L=L, chi=chi, d=d,
        initial_mps=initial_mps,
        history_size=warmup_history,
        max_epochs=warmup_epochs,
        tol=warmup_tol,
        seed=seed,
        log_every=log_every,
        line_search=line_search,
        line_search_kwargs=line_search_kwargs,
        precondition=False,
        dtype=dtype,
        _phase_label="warmup ",
    )

    if log_every > 0:
        print(
            f"--- R-LBFGS polish (precond, ridge={polish_ridge}): "
            f"{polish_epochs} epochs, history={polish_history}, "
            f"tol={polish_tol} ---"
        )
    polish = run_rlbfgs(
        H, L=L, chi=chi, d=d,
        initial_mps=warm["mps"],
        history_size=polish_history,
        max_epochs=polish_epochs,
        tol=polish_tol,
        seed=seed,
        log_every=log_every,
        line_search=line_search,
        line_search_kwargs=line_search_kwargs,
        precondition=True,
        ridge=polish_ridge,
        dtype=dtype,
        _phase_label="polish ",
    )

    return {
        "mps": polish["mps"],
        "energy": polish["energy"],
        "grad_norm": polish["grad_norm"],
        "epochs": warm["epochs"] + polish["epochs"],
        "warmup_epochs": warm["epochs"],
        "polish_epochs": polish["epochs"],
        "warmup_history": warm["history"],
        "polish_history": polish["history"],
        "wall_time": warm["wall_time"] + polish["wall_time"],
        "warmup_wall_time": warm["wall_time"],
        "polish_wall_time": polish["wall_time"],
        "converged": polish["converged"],
    }


def run_heisenberg(L: int, chi: int, j: float = 1.0, bz: float = 0.0, **kw):
    H = build_heisenberg_mpo(L, j=j, bz=bz)
    return run_rlbfgs_warmstart(H, L=L, chi=chi, d=2, **kw)


def run_tfim(L: int, chi: int, j: float = 1.0, hx: float = 1.0, **kw):
    H = build_tfim_mpo(L, j=j, hx=hx)
    return run_rlbfgs_warmstart(H, L=L, chi=chi, d=2, **kw)


def run_josephson(
    L: int,
    chi: int,
    *,
    E_J: float = 1.0,
    E_C: float = 0.5,
    mu: float = 0.0,
    n_max: int = 2,
    phi_ext: float = math.pi / 4,
    **kw,
):
    d = 2 * n_max + 1
    H = build_josephson_mpo(
        L, E_J=E_J, E_C=E_C, mu=mu, n_max=n_max, phi_ext=phi_ext,
    )
    return run_rlbfgs_warmstart(H, L=L, chi=chi, d=d, **kw)
