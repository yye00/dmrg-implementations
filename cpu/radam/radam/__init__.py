"""Riemannian Adam (R-Adam) for Tensor Train / MPS optimization.

A CPU numpy implementation of Riemannian Adam applied to the fixed-rank
Tensor Train (Matrix Product State) manifold, with support for
Heisenberg, Transverse Field Ising (TFIM), and Josephson-junction-array
Hamiltonians.

Convention for all tensor shapes in this package:

  MPS core:  (chi_L, d, chi_R)
  MPO core:  (mpo_L, mpo_R, d_up, d_down)

The full MPS / MPO are Python lists of these cores.  Edge sites use a
trivial bond of size 1.

For a quasi-Newton optimizer with tighter convergence on the same
problems see the sibling ``rlbfgs`` package (``cpu/rlbfgs/``).
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
from .gradient import euclidean_gradient
from .projection import project_right_canonical, transport_momentum
from .retraction import retract_and_recanonicalize
from .optimizer import RAdamState, radam_step
from .driver import run_radam, run_heisenberg, run_tfim, run_josephson

__all__ = [
    "random_mps",
    "zeros_like_mps",
    "copy_mps",
    "scale_mps",
    "right_canonicalize",
    "left_canonicalize",
    "mps_inner",
    "mps_norm_squared",
    "mps_frob_norm_squared_cores",
    "build_heisenberg_mpo",
    "build_tfim_mpo",
    "build_josephson_mpo",
    "build_right_envs_H",
    "build_left_envs_H",
    "euclidean_gradient",
    "project_right_canonical",
    "transport_momentum",
    "retract_and_recanonicalize",
    "RAdamState",
    "radam_step",
    "run_radam",
    "run_heisenberg",
    "run_tfim",
    "run_josephson",
]
