"""Riemannian Adam update for the fixed-rank TT / MPS manifold.

The R-Adam step as implemented here (see the module :mod:`radam` for a
longer derivation):

1. Compute the Euclidean gradient ``grad_i = dE/dA_i*`` for every core.
2. Project to the tangent space: ``G_i = P_X(grad_i)``.
3. Vector-transport the momentum: ``M <- P_X(M_prev)``.
4. First-moment update: ``M_i = b1 * M_i + (1-b1) * G_i``.
5. Second-moment update (scalar!): ``v = b2 * v + (1-b2) * ||G||^2``
   where ``||G||^2 = sum_i ||G_i||_F^2``.
6. Bias correction: ``M_hat = M / (1 - b1^k)``, ``v_hat = v / (1 - b2^k)``.
7. Step direction on the tangent space:
      ``Delta_i = -alpha * M_hat_i / (sqrt(v_hat) + eps)``
8. Retract: ``X <- right_canonicalize([A + Delta for A, Delta in zip(X, Delta)])``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .gradient import euclidean_gradient
from .mps import mps_frob_norm_squared_cores, zeros_like_mps
from .projection import project_right_canonical, transport_momentum
from .retraction import retract_and_recanonicalize


@dataclass
class RAdamState:
    """In-memory state of the R-Adam optimizer.

    Attributes
    ----------
    X : list of ndarray
        Current MPS (right-canonical, centre at site 0).
    M : list of ndarray
        First-moment (momentum) estimate, one tensor per site.
    v : float
        Scalar second-moment estimate (variance of gradient norm).
    k : int
        Iteration counter (starts at 0; incremented before first use).
    lr : float
        Learning rate (``alpha``).
    beta1 : float
    beta2 : float
    eps : float
    """

    X: List[np.ndarray]
    M: List[np.ndarray] = field(default=None)
    v: float = 0.0
    k: int = 0
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    def __post_init__(self):
        if self.M is None:
            self.M = zeros_like_mps(self.X)


def radam_step(state: RAdamState, H):
    """Perform one R-Adam step in-place on ``state``.

    Parameters
    ----------
    state : RAdamState
        Optimizer state.  ``state.X`` must be right-canonical.
    H : list of ndarray
        MPO cores for the target Hamiltonian.

    Returns
    -------
    info : dict
        ``{"energy": float, "grad_norm": float, "step_size": float}``.
    """
    state.k += 1
    k = state.k
    b1, b2, eps, lr = state.beta1, state.beta2, state.eps, state.lr

    # 1. Euclidean gradient + current energy.
    grads_eucl, energy = euclidean_gradient(state.X, H, return_energy=True)

    # 2. Riemannian gradient: project onto tangent space.
    G = project_right_canonical(state.X, grads_eucl)

    # 3. Vector transport of previous momentum.
    M_trans = transport_momentum(state.X, state.M)

    # 4. First-moment update.
    M_new = [b1 * m + (1.0 - b1) * g for m, g in zip(M_trans, G)]

    # 5. Second-moment (scalar) update.
    g_norm_sq = mps_frob_norm_squared_cores(G)
    v_new = b2 * state.v + (1.0 - b2) * g_norm_sq

    # 6. Bias correction.
    bc1 = 1.0 - b1 ** k
    bc2 = 1.0 - b2 ** k
    M_hat = [m / bc1 for m in M_new]
    v_hat = v_new / bc2

    # 7. Tangent step.
    step_scale = -lr / (float(np.sqrt(v_hat)) + eps)
    Delta = [step_scale * m for m in M_hat]

    # 8. Retract.
    X_new = retract_and_recanonicalize(state.X, Delta)

    # Update state.
    state.X = X_new
    state.M = M_new
    state.v = v_new

    return {
        "energy": float(energy),
        "grad_norm": float(np.sqrt(g_norm_sq)),
        "step_size": float(abs(step_scale)),
    }
