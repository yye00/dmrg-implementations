"""Environment update functions for DMRG sweeps.

Environments encode the contraction of the MPS, MPO, and conjugate MPS
for all sites to the left (L_env) or right (R_env) of the current bond.

Uses cotengra for optimized tensor contraction paths. Each environment
update is a 4-tensor contraction expressed as a single einsum.

Index conventions (all arrays in our internal format):
  MPS tensor A: (left_bond, phys, right_bond)
  MPO tensor W: (mpo_left, mpo_right, phys_up, phys_down)
  L_env: (bra_bond, mpo_bond, ket_bond)
  R_env: (bra_bond, mpo_bond, ket_bond)
"""

import numpy as np
import cotengra as ctg

# Cache for pre-compiled contraction expressions
_lenv_expr_cache = {}
_renv_expr_cache = {}


def _get_lenv_expr(L_shape, A_shape, W_shape):
    """Get or create a pre-compiled expression for left env update."""
    key = (L_shape, A_shape, W_shape)
    if key not in _lenv_expr_cache:
        A_conj_shape = A_shape  # same shape as A
        # L_new[b, v, g] = L[a,w,c] * A[c,s,g] * W[w,v,s,d] * A*[a,d,b]
        #
        # W shape: (mpo_L, mpo_R, phys_up, phys_down)
        #   phys_up (axis 2) = ket side, contracts with A's phys (s)
        #   phys_down (axis 3) = bra side, contracts with A*'s phys (d)
        #
        # Index legend:
        #   a = bra (contracted), w = mpo_L (contracted), c = ket (contracted)
        #   s = phys_ket (contracted A-W phys_up)
        #   d = phys_bra (contracted W phys_down - A*)
        #   g = ket' (output), v = mpo_R (output), b = bra' (output)
        _lenv_expr_cache[key] = ctg.einsum_expression(
            'awc,csg,wvsd,adb->bvg',
            L_shape, A_shape, W_shape, A_conj_shape,
            optimize='auto',
        )
    return _lenv_expr_cache[key]


def _get_renv_expr(R_shape, B_shape, W_shape):
    """Get or create a pre-compiled expression for right env update."""
    key = (R_shape, B_shape, W_shape)
    if key not in _renv_expr_cache:
        B_conj_shape = B_shape  # same shape as B
        # R_new[b, v, g] = R[a,w,c] * B[g,s,c] * W[v,w,s,d] * B*[b,d,a]
        #
        # W shape: (mpo_L, mpo_R, phys_up, phys_down)
        #   phys_up (axis 2) = ket side, contracts with B's phys (s)
        #   phys_down (axis 3) = bra side, contracts with B*'s phys (d)
        #
        # Index legend:
        #   a = bra (contracted), w = mpo_R (contracted), c = ket (contracted)
        #   s = phys_ket (contracted B-W phys_up)
        #   d = phys_bra (contracted W phys_down - B*)
        #   g = ket' (output), v = mpo_L (output), b = bra' (output)
        _renv_expr_cache[key] = ctg.einsum_expression(
            'awc,gsc,vwsd,bda->bvg',
            R_shape, B_shape, W_shape, B_conj_shape,
            optimize='auto',
        )
    return _renv_expr_cache[key]


def update_left_env(L_env, A, W):
    """Grow left environment by one site (sweeping right).

    L_new[a', w', c'] = L[a,w,c] * A[c,s,c'] * W[w,w',t,s] * A*[a,t,a']

    Uses cotengra to find and cache the optimal contraction order.

    Parameters
    ----------
    L_env : ndarray, shape (χ_bra, D, χ_ket)
    A : ndarray, shape (χ_ket, d, χ_ket')  — ket tensor
    W : ndarray, shape (D, D', d, d) — MPO tensor (mpo_L, mpo_R, up, down)

    Returns
    -------
    L_new : ndarray, shape (χ_bra', D', χ_ket')
    """
    A_conj = A.conj()
    expr = _get_lenv_expr(L_env.shape, A.shape, W.shape)
    return expr(L_env, A, W, A_conj)


def update_right_env(R_env, B, W):
    """Grow right environment by one site (sweeping left).

    R_new[a', w', c'] = R[a,w,c] * B[c',s,c] * W[w',w,t,s] * B*[a',t,a]

    Uses cotengra to find and cache the optimal contraction order.

    Parameters
    ----------
    R_env : ndarray, shape (χ_bra, D, χ_ket)
    B : ndarray, shape (χ_ket', d, χ_ket)  — ket tensor
    W : ndarray, shape (D', D, d, d) — MPO tensor (mpo_L, mpo_R, up, down)

    Returns
    -------
    R_new : ndarray, shape (χ_bra', D', χ_ket')
    """
    B_conj = B.conj()
    expr = _get_renv_expr(R_env.shape, B.shape, W.shape)
    return expr(R_env, B, W, B_conj)


def init_left_env(chi_L, D, dtype=np.float64):
    """Create trivial left environment for the left edge of the system.

    L[bra=0, mpo=0, ket=0] = 1, all others = 0.
    """
    L = np.zeros((chi_L, D, chi_L), dtype=dtype)
    L[0, 0, 0] = 1.0
    return L


def init_right_env(chi_R, D, dtype=np.float64):
    """Create trivial right environment for the right edge of the system."""
    R = np.zeros((chi_R, D, chi_R), dtype=dtype)
    R[-1, -1, -1] = 1.0
    return R
