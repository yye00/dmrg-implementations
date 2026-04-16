"""Riemannian L-BFGS (R-LBFGS) for MPS / Tensor Train ground state optimization.

A self-contained CPU + numpy implementation of the quasi-Newton
Riemannian L-BFGS optimizer on the fixed-rank Tensor Train (MPS)
manifold, with a recommended two-phase warm-start entry point
(:func:`run_rlbfgs_warmstart`) that reaches < 1e-9 of single-site DMRG
ground-state energy on the small benchmark grid.

Conventions (shared with the ``radam`` package):

* MPS core: ``(chi_L, d, chi_R)``
* MPO core: ``(mpo_L, mpo_R, d_up, d_down)``
"""

from .mps import (
    random_mps,
    zeros_like_mps,
    copy_mps,
    scale_mps,
    right_canonicalize,
    left_canonicalize,
    mps_inner,
    mps_norm_squared,
    mps_frob_norm_squared_cores,
)
from .mpo import build_heisenberg_mpo, build_tfim_mpo, build_josephson_mpo
from .environments import build_right_envs_H, build_left_envs_H
from .gradient import euclidean_gradient, energy_only, precondition_with_metric
from .projection import project_right_canonical, transport_momentum
from .retraction import retract_and_recanonicalize
from .optimizer import (
    LBFGSState,
    rlbfgs_step,
    lbfgs_two_loop,
    line_search_armijo,
    line_search_strong_wolfe,
)
from .driver import (
    run_rlbfgs,
    run_rlbfgs_warmstart,
    run_heisenberg,
    run_tfim,
    run_josephson,
)
from .tangent import inner_real, norm_tangent, scale_cores, add_cores, sub_cores

__all__ = [
    # MPS
    "random_mps", "zeros_like_mps", "copy_mps", "scale_mps",
    "right_canonicalize", "left_canonicalize",
    "mps_inner", "mps_norm_squared", "mps_frob_norm_squared_cores",
    # MPO
    "build_heisenberg_mpo", "build_tfim_mpo", "build_josephson_mpo",
    # Environments
    "build_right_envs_H", "build_left_envs_H",
    # Gradient
    "euclidean_gradient", "energy_only", "precondition_with_metric",
    # Geometry
    "project_right_canonical", "transport_momentum",
    "retract_and_recanonicalize",
    # Optimizer
    "LBFGSState", "rlbfgs_step",
    "lbfgs_two_loop", "line_search_armijo", "line_search_strong_wolfe",
    # Drivers
    "run_rlbfgs", "run_rlbfgs_warmstart",
    "run_heisenberg", "run_tfim", "run_josephson",
    # Tangent
    "inner_real", "norm_tangent", "scale_cores", "add_cores", "sub_cores",
]
