"""Top-level R-Adam optimization driver.

Run Riemannian Adam on an MPS ground-state problem with optional
cosine-annealing learning-rate schedule and early-stopping based on the
Riemannian gradient norm.
"""

from __future__ import annotations

import math
import time
from typing import Callable, List, Optional

import numpy as np

from .lbfgs import run_rlbfgs
from .mps import random_mps
from .mpo import build_heisenberg_mpo, build_tfim_mpo, build_josephson_mpo
from .optimizer import RAdamState, radam_step


def _cosine_lr(base_lr: float, step: int, total: int, min_lr: float = 0.0) -> float:
    """Cosine-annealing learning rate."""
    if total <= 1:
        return base_lr
    t = min(step, total) / total
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t))


def run_radam(
    H,
    L: int,
    chi: int,
    d: int = 2,
    *,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    max_epochs: int = 500,
    tol: float = 1e-6,
    lr_schedule: Optional[str] = "cosine",
    min_lr: float = 1e-6,
    seed: Optional[int] = 0,
    log_every: int = 10,
    callback: Optional[Callable[[int, dict], None]] = None,
    initial_mps: Optional[list] = None,
    dtype=np.complex128,
):
    """Optimize an MPS against the Hamiltonian ``H`` using R-Adam.

    Parameters
    ----------
    H : list of ndarray
        MPO cores.
    L : int
        Number of sites (must match ``len(H)``).
    chi : int
        Target bond dimension.
    d : int
        Local Hilbert-space dimension (default 2).
    lr, beta1, beta2, eps : float
        Adam hyperparameters.
    max_epochs : int
        Maximum number of R-Adam iterations.
    tol : float
        Convergence tolerance on the Riemannian gradient norm.
    lr_schedule : {'cosine', None}
        Optional learning-rate schedule.
    min_lr : float
        Lower bound for the cosine schedule.
    seed : int or None
        RNG seed for MPS initialization.
    log_every : int
        Print progress every ``log_every`` epochs.
    callback : callable or None
        Called as ``callback(epoch, info_dict)`` every epoch.
    initial_mps : list of ndarray or None
        If provided, used as the starting MPS instead of a random one.

    Returns
    -------
    result : dict
        ``{"mps": final MPS cores, "energy": float, "grad_norm": float,
        "epochs": int, "history": list of dicts, "wall_time": float,
        "converged": bool}``.
    """
    if len(H) != L:
        raise ValueError(f"len(H)={len(H)} does not match L={L}")

    if initial_mps is None:
        X = random_mps(L, chi, d, dtype=dtype, seed=seed)
    else:
        X = [A.copy() for A in initial_mps]

    state = RAdamState(X=X, lr=lr, beta1=beta1, beta2=beta2, eps=eps)

    history = []
    t0 = time.perf_counter()
    converged = False

    for epoch in range(1, max_epochs + 1):
        if lr_schedule == "cosine":
            state.lr = _cosine_lr(lr, epoch - 1, max_epochs, min_lr=min_lr)
        elif lr_schedule is None:
            state.lr = lr

        info = radam_step(state, H)
        info["epoch"] = epoch
        info["lr"] = state.lr
        history.append(info)

        if callback is not None:
            callback(epoch, info)

        if log_every > 0 and (epoch == 1 or epoch % log_every == 0):
            print(
                f"epoch {epoch:5d}  E={info['energy']: .10f}  "
                f"||G||={info['grad_norm']:.3e}  "
                f"lr={state.lr:.2e}  |step|={info['step_size']:.2e}"
            )

        if info["grad_norm"] < tol:
            converged = True
            break

    wall = time.perf_counter() - t0
    final = history[-1]
    if log_every > 0:
        print(
            f"done  converged={converged}  epochs={len(history)}  "
            f"final E={final['energy']:.10f}  final ||G||={final['grad_norm']:.3e}  "
            f"wall={wall:.2f}s"
        )

    return {
        "mps": state.X,
        "energy": final["energy"],
        "grad_norm": final["grad_norm"],
        "epochs": len(history),
        "history": history,
        "wall_time": wall,
        "converged": converged,
    }


def run_heisenberg(L: int, chi: int, j: float = 1.0, bz: float = 0.0, **kw):
    """Convenience wrapper: build a Heisenberg MPO and run R-Adam."""
    H = build_heisenberg_mpo(L, j=j, bz=bz)
    return run_radam(H, L=L, chi=chi, d=2, **kw)


def run_tfim(L: int, chi: int, j: float = 1.0, hx: float = 1.0, **kw):
    """Convenience wrapper: build a TFIM MPO and run R-Adam."""
    H = build_tfim_mpo(L, j=j, hx=hx)
    return run_radam(H, L=L, chi=chi, d=2, **kw)


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
    """Convenience wrapper: build a Josephson-array MPO and run R-Adam."""
    d = 2 * n_max + 1
    H = build_josephson_mpo(
        L, E_J=E_J, E_C=E_C, mu=mu, n_max=n_max, phi_ext=phi_ext,
    )
    return run_radam(H, L=L, chi=chi, d=d, **kw)


def run_warmstart_rlbfgs(
    H,
    L: int,
    chi: int,
    d: int = 2,
    *,
    # R-Adam warm-up parameters.
    warmup_epochs: int = 200,
    warmup_lr: float = 1e-2,
    warmup_min_lr: float = 1e-4,
    warmup_lr_schedule: Optional[str] = "cosine",
    # R-LBFGS polish parameters.
    polish_epochs: int = 1000,
    polish_history: int = 20,
    polish_tol: float = 1e-10,
    polish_precondition: bool = False,
    polish_ridge: float = 1e-10,
    line_search_kwargs: Optional[dict] = None,
    # Common.
    seed: Optional[int] = 0,
    log_every: int = 0,
    dtype=np.complex128,
):
    """Two-stage optimizer: R-Adam warm-up followed by R-LBFGS polish.

    The R-Adam stage is responsible for finding the right basin of
    attraction (it is robust to a poor random initialization but only
    converges geometrically on the Rayleigh-quotient).  Once close to
    the minimum the R-LBFGS stage takes over and delivers
    quasi-second-order convergence to high precision.

    Returns a dict with the same keys as :func:`run_radam` plus
    ``warmup_history`` and ``polish_history`` for diagnostics.
    """
    if len(H) != L:
        raise ValueError(f"len(H)={len(H)} does not match L={L}")

    if log_every > 0:
        print(f"--- R-Adam warm-up: {warmup_epochs} epochs, lr={warmup_lr} ---")
    warm = run_radam(
        H,
        L=L,
        chi=chi,
        d=d,
        lr=warmup_lr,
        max_epochs=warmup_epochs,
        tol=0.0,                              # never stop early in warmup
        lr_schedule=warmup_lr_schedule,
        min_lr=warmup_min_lr,
        seed=seed,
        log_every=log_every,
        dtype=dtype,
    )

    if log_every > 0:
        print(
            f"--- R-LBFGS polish: max {polish_epochs} epochs, "
            f"history={polish_history}, tol={polish_tol} ---"
        )
    polish = run_rlbfgs(
        H,
        L=L,
        chi=chi,
        d=d,
        initial_mps=warm["mps"],
        history_size=polish_history,
        max_epochs=polish_epochs,
        tol=polish_tol,
        seed=seed,
        log_every=log_every,
        line_search_kwargs=line_search_kwargs,
        precondition=polish_precondition,
        ridge=polish_ridge,
        dtype=dtype,
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
