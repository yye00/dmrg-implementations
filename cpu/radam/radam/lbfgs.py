"""Riemannian L-BFGS optimizer for the fixed-rank TT / MPS manifold.

This is a quasi-Newton optimizer with the standard L-BFGS two-loop
recursion lifted to the TT manifold via vector transport.  The intended
use mirrors the rationale at the top of :mod:`radam.optimizer`: get
quasi-second-order convergence (matching DMRG's local-eigensolve
convergence rate in the asymptotic regime) while still updating all
cores simultaneously per iteration.

Algorithm
---------

At iteration ``k``, given the current MPS ``X_k`` (right-canonical with
centre at site 0) and a buffer of the last ``m`` ``(s, y)`` pairs:

1. **Riemannian gradient**:
   ``g_k = P_{X_k}(grad_E_k)`` where ``grad_E_k = dE/dA_i*`` is the
   Euclidean gradient (one core per site) and ``P_X`` is the
   right-canonical tangent-space projection.

2. **Search direction** via the two-loop recursion, with vector
   transport applied **at the time of use** to every stored pair:

       q = g_k
       for (s_i, y_i, rho_i) in reverse(history):
           s_i_t = P_{X_k}(s_i);   y_i_t = P_{X_k}(y_i)
           rho_i_t = 1 / <y_i_t, s_i_t>
           alpha_i = rho_i_t * <s_i_t, q>
           q -= alpha_i * y_i_t
           save alpha_i

       gamma_k = <s_last_t, y_last_t> / <y_last_t, y_last_t>     (or 1.0)
       r = gamma_k * q

       for (s_i, y_i) in history:
           ...transport again...
           beta = rho_i_t * <y_i_t, r>
           r += (alpha_i - beta) * s_i_t

       d_k = -r

   The two-loop is implemented in a single pass that transports each
   pair once and caches the transported vectors, so the cost is
   ``O(m)`` per step.

3. **Descent guard**: if ``<g_k, d_k> >= 0`` the L-BFGS direction is
   not a descent direction (Hessian estimate is non-positive-definite).
   We fall back to ``d_k = -g_k`` and clear the history.

4. **Backtracking Armijo line search**:
   try ``alpha = alpha_init`` (typically ``1.0``), shrink by ``beta``
   until ``E(R(X_k, alpha d_k)) <= E_k + c1 * alpha * <g_k, d_k>``.

5. **Pair update**:
   ``s_k = alpha * d_k`` (the actual tangent step taken)
   ``y_k = g_{k+1} - P_{X_{k+1}}(g_k)`` (gradient-difference,
   transported)
   Apply the **cautious update** rule: only push ``(s_k, y_k)`` if
   ``<s_k, y_k> > epsilon * ||s_k|| * ||y_k||``.

6. **Convergence**: ``||g_k|| < tol``.

References
----------
* Riemannian Optimization on Manifolds, Boumal (2023)
* Steinlechner, "Riemannian optimization for high-dimensional tensor
  completion" (2016)
* Nocedal & Wright, Numerical Optimization, Ch. 7
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, List, Optional, Tuple

import numpy as np

from .gradient import energy_only, euclidean_gradient, precondition_with_metric
from .projection import project_right_canonical
from .retraction import retract_and_recanonicalize
from .tangent import (
    add_cores,
    copy_cores,
    inner_real,
    norm_tangent,
    scale_cores,
    sub_cores,
)


@dataclass
class LBFGSState:
    """In-memory state of the R-LBFGS optimizer.

    Attributes
    ----------
    X : list of ndarray
        Current MPS (right-canonical, centre at site 0).
    history : deque of (s, y, rho)
        Buffer of the last ``m`` pairs.  ``s, y`` are tangent vectors
        stored in the tangent space at the iterate at which they were
        recorded; transport is applied when they are used.  ``rho``
        is ``1 / <y, s>`` evaluated at the *original* tangent space
        and is recomputed after transport during the two-loop.
    history_size : int
        Maximum number of pairs to keep (the L-BFGS memory ``m``).
    k : int
        Iteration counter.
    """

    X: List[np.ndarray]
    history: Deque[Tuple[List[np.ndarray], List[np.ndarray], float]] = field(default=None)
    history_size: int = 10
    k: int = 0

    def __post_init__(self):
        if self.history is None:
            self.history = deque(maxlen=self.history_size)
        else:
            self.history = deque(self.history, maxlen=self.history_size)


def _transport_pair_and_compute_rho(X_current, s, y):
    """Transport ``(s, y)`` to the tangent space at ``X_current``."""
    s_t = project_right_canonical(X_current, s)
    y_t = project_right_canonical(X_current, y)
    sy = inner_real(s_t, y_t)
    return s_t, y_t, sy


def lbfgs_two_loop(g, history, X_current, *, default_gamma: float = 1.0):
    """L-BFGS two-loop recursion with on-the-fly vector transport.

    Returns the search direction ``d`` (negative descent direction
    pre-applied so the caller can directly do ``X + alpha * d``).
    """
    # Transport every pair once and cache.
    transported: List[Tuple[list, list, float]] = []
    for s_old, y_old, _rho_old in history:
        s_t, y_t, sy_t = _transport_pair_and_compute_rho(X_current, s_old, y_old)
        if sy_t > 0.0:
            transported.append((s_t, y_t, 1.0 / sy_t))

    q = copy_cores(g)
    alphas: List[float] = []
    for s_t, y_t, rho_t in reversed(transported):
        a = rho_t * inner_real(s_t, q)
        q = sub_cores(q, scale_cores(y_t, a))
        alphas.append(a)
    alphas.reverse()

    if transported:
        s_last, y_last, _ = transported[-1]
        sy = inner_real(s_last, y_last)
        yy = inner_real(y_last, y_last)
        gamma = sy / yy if yy > 0.0 else default_gamma
    else:
        gamma = default_gamma

    r = scale_cores(q, gamma)
    for i, (s_t, y_t, rho_t) in enumerate(transported):
        b = rho_t * inner_real(y_t, r)
        r = add_cores(r, scale_cores(s_t, alphas[i] - b))

    return scale_cores(r, -1.0)


def line_search_armijo(
    X,
    direction,
    g_current,
    E_current: float,
    H,
    *,
    c1: float = 1e-4,
    alpha_init: float = 1.0,
    beta: float = 0.5,
    max_iter: int = 25,
    min_alpha: float = 1e-14,
):
    """Backtracking line search satisfying the Armijo (sufficient-decrease)
    condition.

    Returns ``(alpha, X_trial, E_trial)``.  If the line search exits
    without satisfying the condition, the *last* trial point is
    returned (callers should detect non-decrease and act accordingly).
    """
    slope = inner_real(g_current, direction)
    # Numerical safety: if slope is non-negative the direction is not
    # descent.  Fall back to the negative gradient.
    if slope >= 0.0:
        direction = scale_cores(g_current, -1.0)
        slope = inner_real(g_current, direction)
        if slope >= 0.0:
            # gradient itself is zero -- nothing to do
            return 0.0, X, E_current

    alpha = alpha_init
    X_trial = X
    E_trial = E_current
    for _ in range(max_iter):
        X_trial = retract_and_recanonicalize(X, scale_cores(direction, alpha))
        try:
            E_trial = energy_only(X_trial, H)
        except ValueError:
            # MPS became degenerate (norm ~ 0); shrink and retry.
            alpha *= beta
            if alpha < min_alpha:
                return alpha, X_trial, E_current
            continue
        if E_trial <= E_current + c1 * alpha * slope:
            return alpha, X_trial, E_trial
        alpha *= beta
        if alpha < min_alpha:
            return alpha, X_trial, E_trial
    return alpha, X_trial, E_trial


def line_search_strong_wolfe(
    X,
    direction,
    g_phys,
    E_current: float,
    H,
    *,
    c1: float = 1e-4,
    c2: float = 0.9,
    alpha_init: float = 1.0,
    alpha_max: float = 10.0,
    max_zoom_iter: int = 25,
    max_bracket_iter: int = 25,
    min_alpha: float = 1e-14,
):
    """Line search satisfying the Strong Wolfe conditions on the manifold.

    Uses the bracket-then-zoom algorithm of Nocedal & Wright, Ch. 3,
    Algorithms 3.5/3.6.  Each candidate ``alpha`` is materialized via
    the manifold retraction; the slope at the candidate is computed
    using the **transported** descent direction and the new Riemannian
    gradient.

    The slope must be ``<g_E, direction>`` where ``g_E`` is the
    *un-preconditioned* (physical) Riemannian gradient -- this is
    what predicts ``dE/dalpha``.  The caller passes ``g_phys``
    accordingly.

    Returns ``(alpha, X_new, E_new, g_phys_new)``.  ``g_phys_new``
    is the un-preconditioned Riemannian gradient at the new point.
    """
    slope0 = inner_real(g_phys, direction)
    if slope0 >= 0.0:
        # Not a descent direction -- fall back to ``-g_phys``.
        direction = scale_cores(g_phys, -1.0)
        slope0 = inner_real(g_phys, direction)
        if slope0 >= 0.0:
            return 0.0, X, E_current, g_phys

    # Helper to evaluate a trial alpha.
    def _phi(a):
        X_a = retract_and_recanonicalize(X, scale_cores(direction, a))
        try:
            E_a = energy_only(X_a, H)
        except ValueError:
            return None, None, None, None
        if not np.isfinite(E_a):
            return None, None, None, None
        # Riemannian gradient at the new point (un-preconditioned).
        grads_eucl, _ = euclidean_gradient(X_a, H, return_energy=True)
        g_a = project_right_canonical(X_a, grads_eucl)
        # Transport ``direction`` to the new tangent space and take
        # the slope along it (a faithful realisation of the Wolfe
        # curvature condition for manifold optimization).
        d_a = project_right_canonical(X_a, direction)
        slope_a = inner_real(g_a, d_a)
        return X_a, E_a, g_a, slope_a

    def _zoom(alpha_lo, alpha_hi, X_lo, E_lo, slope_lo, X_hi, E_hi):
        """Zoom phase: bracket has [alpha_lo, alpha_hi]."""
        prev_alpha_j = None
        prev_E_j = None
        for _ in range(max_zoom_iter):
            # Use bisection (robust; quadratic interpolation also fine).
            alpha_j = 0.5 * (alpha_lo + alpha_hi)
            X_j, E_j, g_j, slope_j = _phi(alpha_j)
            if E_j is None:
                # Degenerate -- shrink towards ``alpha_lo``.
                alpha_hi = alpha_j
                continue
            armijo = E_j <= E_current + c1 * alpha_j * slope0
            if (not armijo) or (prev_E_j is not None and E_j >= prev_E_j):
                alpha_hi = alpha_j
                X_hi = X_j
                E_hi = E_j
            else:
                if abs(slope_j) <= -c2 * slope0:
                    return alpha_j, X_j, E_j, g_j
                if slope_j * (alpha_hi - alpha_lo) >= 0.0:
                    alpha_hi = alpha_lo
                    X_hi = X_lo
                    E_hi = E_lo
                alpha_lo = alpha_j
                X_lo = X_j
                E_lo = E_j
                slope_lo = slope_j
            prev_alpha_j = alpha_j
            prev_E_j = E_j
            if abs(alpha_hi - alpha_lo) < min_alpha:
                break
        # Accept the lowest-energy point we found.
        return alpha_lo, X_lo, E_lo, None  # caller will recompute g

    # --- bracket phase ---
    alpha_prev = 0.0
    E_prev = E_current
    X_prev = X
    g_prev = g_phys
    slope_prev = slope0
    alpha_i = alpha_init

    last_X = X
    last_E = E_current
    last_g = g_phys
    for i in range(max_bracket_iter):
        X_i, E_i, g_i, slope_i = _phi(alpha_i)
        if E_i is None:
            # Degenerate alpha; shrink and retry.
            alpha_i = 0.5 * (alpha_prev + alpha_i)
            if alpha_i < min_alpha:
                return alpha_prev, X_prev, E_prev, g_prev
            continue

        last_X, last_E, last_g = X_i, E_i, g_i
        armijo = E_i <= E_current + c1 * alpha_i * slope0
        if (not armijo) or (i > 0 and E_i >= E_prev):
            a_star, X_star, E_star, g_star = _zoom(
                alpha_prev, alpha_i, X_prev, E_prev, slope_prev, X_i, E_i,
            )
            if g_star is None:
                # Re-compute Riemannian gradient if zoom didn't supply it.
                grads_eucl, _ = euclidean_gradient(X_star, H, return_energy=True)
                g_star = project_right_canonical(X_star, grads_eucl)
            return a_star, X_star, E_star, g_star

        if abs(slope_i) <= -c2 * slope0:
            return alpha_i, X_i, E_i, g_i

        if slope_i >= 0.0:
            a_star, X_star, E_star, g_star = _zoom(
                alpha_i, alpha_prev, X_i, E_i, slope_i, X_prev, E_prev,
            )
            if g_star is None:
                grads_eucl, _ = euclidean_gradient(X_star, H, return_energy=True)
                g_star = project_right_canonical(X_star, grads_eucl)
            return a_star, X_star, E_star, g_star

        # Expand bracket.
        alpha_prev = alpha_i
        X_prev = X_i
        E_prev = E_i
        g_prev = g_i
        slope_prev = slope_i
        alpha_i = min(2.0 * alpha_i, alpha_max)
        if alpha_i >= alpha_max:
            break

    # Bracket loop exhausted -- return the best we found.
    return alpha_i, last_X, last_E, last_g


def rlbfgs_step(
    state: LBFGSState,
    H,
    *,
    line_search: str = "wolfe",
    line_search_kwargs: Optional[dict] = None,
    cautious_eps: float = 1e-10,
    precondition: bool = False,
    ridge: float = 1e-10,
):
    """Perform one R-LBFGS iteration in-place on ``state``.

    Parameters
    ----------
    state : LBFGSState
    H : list of ndarray
        MPO cores.
    line_search : {"wolfe", "armijo"}
        Line-search variant.  The Strong Wolfe variant returns the new
        gradient as a side effect, saving one gradient evaluation per
        iteration.
    line_search_kwargs : dict or None
    cautious_eps : float
        Threshold for the cautious-update curvature check.

    Returns
    -------
    info : dict
        ``{"energy", "grad_norm", "step_size", "history_len",
           "fallback_to_grad", "ls_failed"}``.
    """
    state.k += 1
    line_search_kwargs = dict(line_search_kwargs or {})

    # 1. Riemannian gradient at X_k.
    if precondition:
        grads_eucl, energy, (L_N, R_N) = euclidean_gradient(
            state.X, H, return_energy=True, return_envs=True,
        )
        # Project first, then precondition.  L_N[i]^{-1} preserves the
        # gauge-orthogonality condition ``V_i A_i^H = 0`` at sites
        # ``i >= 1``, so no second projection is needed.  At site 0
        # the preconditioner is identity (L_N[0] = R_N[1] are 1x1).
        g_phys = project_right_canonical(state.X, grads_eucl)
        g = precondition_with_metric(g_phys, state.X, L_N, R_N, ridge=ridge)
    else:
        grads_eucl, energy = euclidean_gradient(state.X, H, return_energy=True)
        g = project_right_canonical(state.X, grads_eucl)
        g_phys = g
    g_norm = norm_tangent(g_phys)

    # 2. Search direction (uses preconditioned gradient).
    direction = lbfgs_two_loop(g, state.history, state.X)

    # 3. Descent guard: check direction is descending in the *physical*
    #    gradient (since the line search uses the physical slope).
    slope = inner_real(g_phys, direction)
    fallback = False
    if slope >= 0.0:
        # Hessian estimate failed; reset history and use steepest descent
        # along the (preconditioned) Riemannian direction.
        state.history.clear()
        direction = scale_cores(g, -1.0)
        slope = inner_real(g_phys, direction)
        if slope >= 0.0:
            # Even the steepest preconditioned direction is not descent
            # (can only happen at numerical noise level).  Use g_phys.
            direction = scale_cores(g_phys, -1.0)
            slope = inner_real(g_phys, direction)
        fallback = True

    # 4. Line search.
    if line_search == "wolfe":
        alpha, X_new, E_new, g_phys_new = line_search_strong_wolfe(
            state.X, direction, g_phys, energy, H, **line_search_kwargs
        )
    elif line_search == "armijo":
        alpha, X_new, E_new = line_search_armijo(
            state.X, direction, g_phys, energy, H, **line_search_kwargs
        )
        ge_new, _ = euclidean_gradient(X_new, H, return_energy=True)
        g_phys_new = project_right_canonical(X_new, ge_new)
    else:
        raise ValueError(f"unknown line_search='{line_search}'")

    # Re-compute the *preconditioned* gradient at X_new for storing
    # in the (s, y) pair (so the Hessian estimate is consistent).
    if precondition:
        ge_new_for_pre, _, (Ln_new, Rn_new) = euclidean_gradient(
            X_new, H, return_energy=True, return_envs=True,
        )
        gp_new = precondition_with_metric(ge_new_for_pre, X_new, Ln_new, Rn_new, ridge=ridge)
        g_new = project_right_canonical(X_new, gp_new)
    else:
        g_new = g_phys_new

    ls_failed = E_new > energy + 1e-15  # did not improve

    # 5. Compute new gradient and pair (s, y) at X_new.
    if alpha > 0.0 and not ls_failed:
        # Step taken in the tangent space at X_k.
        s_k = scale_cores(direction, alpha)
        # Transport the old gradient to the new tangent space, then form y.
        g_old_t = project_right_canonical(X_new, g)
        y_k = sub_cores(g_new, g_old_t)
        # Store both vectors *in the X_new tangent space* so future
        # transports start from a faithful representative.
        s_k_new = project_right_canonical(X_new, s_k)
        sy = inner_real(s_k_new, y_k)
        s_norm = norm_tangent(s_k_new)
        y_norm = norm_tangent(y_k)
        # Cautious update: skip if curvature condition is not safely positive.
        if sy > cautious_eps * max(s_norm * y_norm, 1e-30):
            state.history.append((s_k_new, y_k, 1.0 / sy))

    state.X = X_new

    return {
        "energy": float(E_new),
        "grad_norm": g_norm,
        "step_size": float(alpha),
        "history_len": len(state.history),
        "fallback_to_grad": bool(fallback),
        "ls_failed": bool(ls_failed),
    }


def run_rlbfgs(
    H,
    L: int,
    chi: int,
    d: int = 2,
    *,
    initial_mps=None,
    history_size: int = 10,
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
):
    """Run R-LBFGS to convergence (or until ``max_epochs``).

    Parameters
    ----------
    H : list of ndarray
        MPO cores.
    L, chi, d : int
        Problem dimensions; used only when ``initial_mps`` is ``None``.
    initial_mps : list of ndarray or None
        Starting MPS.  If ``None``, a random right-canonical MPS is
        used; otherwise the supplied MPS is right-canonicalized in
        place.
    history_size : int
        L-BFGS memory ``m``.
    max_epochs : int
        Maximum R-LBFGS iterations.
    tol : float
        Convergence tolerance on Riemannian gradient norm.
    seed : int or None
        RNG seed for ``random_mps`` (only used if ``initial_mps``
        is None).
    log_every : int
        Print every ``log_every`` iterations (0 disables logging).
    line_search_kwargs : dict or None
        Forwarded to :func:`line_search_armijo`.
    callback : callable
        Called as ``callback(epoch, info)`` after each iteration.

    Returns
    -------
    dict
        ``{"mps", "energy", "grad_norm", "epochs", "history",
           "wall_time", "converged"}``.
    """
    import time

    from .mps import random_mps, right_canonicalize

    if initial_mps is None:
        X = random_mps(L, chi, d=d, dtype=dtype, seed=seed)
    else:
        X = right_canonicalize([A.copy() for A in initial_mps])

    state = LBFGSState(X=X, history_size=history_size)
    history_log = []

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
        history_log.append(info)
        if callback is not None:
            callback(ep, info)
        if log_every > 0 and (ep == 1 or ep % log_every == 0):
            print(
                f"rlbfgs ep {ep:5d}  E={info['energy']: .12f}  "
                f"||G||={info['grad_norm']:.3e}  alpha={info['step_size']:.2e}  "
                f"|hist|={info['history_len']}"
                + ("  [fallback]" if info["fallback_to_grad"] else "")
                + ("  [ls-fail]" if info["ls_failed"] else "")
            )
        if info["grad_norm"] < tol:
            converged = True
            break

    wall = time.perf_counter() - t0
    final = history_log[-1]

    if log_every > 0:
        print(
            f"rlbfgs done converged={converged} epochs={len(history_log)} "
            f"E={final['energy']:.12f} ||G||={final['grad_norm']:.3e} "
            f"wall={wall:.2f}s"
        )

    return {
        "mps": state.X,
        "energy": final["energy"],
        "grad_norm": final["grad_norm"],
        "epochs": len(history_log),
        "history": history_log,
        "wall_time": wall,
        "converged": converged,
    }
