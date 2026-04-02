"""Effective Hamiltonian for two-site DMRG optimization.

H_eff acts on a two-site wavefunction θ of shape (χ_L, d, d, χ_R).
Built from L_env, W_i, W_{i+1}, R_env.

Uses cotengra for optimized tensor contraction paths. The full 5-tensor
contraction is expressed as a single einsum and cotengra finds the
optimal pairwise contraction order, which is then cached for reuse.

Index conventions (our internal convention):
- MPS site tensor: (left_bond, phys, right_bond)
- Two-site wavefunction θ: (χ_L, d_left, d_right, χ_R)
- Environment L: (bra_bond, mpo_bond, ket_bond)
- Environment R: (bra_bond, mpo_bond, ket_bond)
- MPO tensor W[bulk]: (mpo_left, mpo_right, phys_up, phys_down)
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator
import cotengra as ctg


# Cache for pre-compiled contraction expressions keyed by tensor shapes
_heff_expr_cache = {}


def _get_heff_expr(L_shape, theta_shape, WL_shape, WR_shape, R_shape):
    """Get or create a pre-compiled contraction expression for H_eff."""
    key = (L_shape, theta_shape, WL_shape, WR_shape, R_shape)
    if key not in _heff_expr_cache:
        # Full contraction:
        #   result[a, p, q, f] = L[a,w,c] * theta[c,s,r,e] *
        #                        W_L[w,m,s,p] * W_R[m,n,r,q] * R[f,n,e]
        #
        # W shape: (mpo_L, mpo_R, phys_up, phys_down)
        #   phys_up (axis 2) = ket side, contracts with theta
        #   phys_down (axis 3) = bra side, free output index
        #
        # Index legend:
        #   a = bra_L, w = mpo_L, c = ket_L (contracted)
        #   s = phys_L_ket (contracted via W_L axis 2), r = phys_R_ket (contracted via W_R axis 2)
        #   e = ket_R (contracted), m = mpo_mid (contracted), n = mpo_R (contracted)
        #   p = phys_L_bra (W_L axis 3, output), q = phys_R_bra (W_R axis 3, output)
        #   f = bra_R (output)
        _heff_expr_cache[key] = ctg.einsum_expression(
            'awc,csre,wmsp,mnrq,fne->apqf',
            L_shape, theta_shape, WL_shape, WR_shape, R_shape,
            optimize='auto',
        )
    return _heff_expr_cache[key]


def apply_heff(L_env, R_env, W_left, W_right, theta):
    """Apply the effective Hamiltonian to a two-site wavefunction.

    Full contraction (Einstein notation):
      result[a, p, q, f] = L[a,w,c] * theta[c,s,r,e] *
                           W_L[w,m,p,s] * W_R[m,n,q,r] * R[f,n,e]

    Uses cotengra to find and cache the optimal contraction path.

    Parameters
    ----------
    L_env : ndarray, shape (χ_L, D_L, χ_L)
    R_env : ndarray, shape (χ_R, D_R, χ_R)
    W_left : ndarray, shape (D_L, D_m, d, d)
    W_right : ndarray, shape (D_m, D_R, d, d)
    theta : ndarray, shape (χ_L, d, d, χ_R)

    Returns
    -------
    result : ndarray, shape (χ_L, d, d, χ_R)
    """
    expr = _get_heff_expr(
        L_env.shape, theta.shape, W_left.shape, W_right.shape, R_env.shape
    )
    return expr(L_env, theta, W_left, W_right, R_env)


def build_heff_operator(L_env, R_env, W_left, W_right, dtype=None):
    """Build a LinearOperator for the effective Hamiltonian.

    Parameters
    ----------
    L_env : ndarray, shape (χ_L, D_L, χ_L)
    R_env : ndarray, shape (χ_R, D_R, χ_R)
    W_left : ndarray, shape (D_L, D_m, d_L, d_L)
    W_right : ndarray, shape (D_m, D_R, d_R, d_R)
    dtype : dtype, optional

    Returns
    -------
    H_eff : LinearOperator
    shape_4d : tuple (χ_L, d_L, d_R, χ_R)
    """
    chi_L = L_env.shape[0]
    chi_R = R_env.shape[0]
    d_L = W_left.shape[2]
    d_R = W_right.shape[2]

    shape_4d = (chi_L, d_L, d_R, chi_R)
    dim = chi_L * d_L * d_R * chi_R

    if dtype is None:
        dtype = np.result_type(L_env.dtype, R_env.dtype,
                               W_left.dtype, W_right.dtype)

    # Pre-compile the expression once for this operator
    expr = _get_heff_expr(
        L_env.shape, shape_4d, W_left.shape, W_right.shape, R_env.shape
    )

    def matvec(v):
        theta = v.reshape(shape_4d)
        result = expr(L_env, theta, W_left, W_right, R_env)
        return result.ravel()

    return LinearOperator((dim, dim), matvec=matvec, dtype=dtype), shape_4d
