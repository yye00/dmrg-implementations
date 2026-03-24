"""Effective Hamiltonian for two-site DMRG optimization.

H_eff acts on a two-site wavefunction θ of shape (χ_L, d, d, χ_R).
Built from L_env, W_i, W_{i+1}, R_env.

NEVER form H_eff as an explicit matrix — use LinearOperator for
matrix-free application via scipy.sparse.linalg.eigsh.

Index conventions (our internal convention):
- MPS site tensor: (left_bond, phys, right_bond)
- Two-site wavefunction θ: (χ_L, d_left, d_right, χ_R)
- Environment L: (bra_bond, mpo_bond, ket_bond)
- Environment R: (bra_bond, mpo_bond, ket_bond)
- MPO tensor W[bulk]: (mpo_left, mpo_right, phys_up, phys_down)
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator


def apply_heff(L_env, R_env, W_left, W_right, theta):
    """Apply the effective Hamiltonian to a two-site wavefunction.

    Full contraction (Einstein notation, all unique letters):
      result[a, p, q, f] = L[a,w,c] * theta[c,s,r,e] *
                           W_L[w,m,p,s] * W_R[m,n,q,r] * R[f,n,e]

    Where:
      a = bra_L (from L), c = ket_L (from L), w = mpo_L (from L)
      s = phys_L_ket, r = phys_R_ket, e = ket_R (from theta)
      m = mpo_mid, n = mpo_R
      p = phys_L_bra, q = phys_R_bra, f = bra_R (from R)

    Environment convention: L/R[bra, mpo, ket].
      L: theta connects to ket (axis 2), result comes from bra (axis 0).
      R: theta connects to ket (axis 2), result comes from bra (axis 0).

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
    # Step 1: L[a,w,c] * theta[c,s,r,e] -> X[a,w,s,r,e]
    # Contract over c (ket_L): L axis 2, theta axis 0
    X = np.tensordot(L_env, theta, axes=(2, 0))
    # X shape: (a, w, s, r, e)

    # Step 2: X[a,w,s,r,e] * W_L[w,m,s',s] -> Y[a,r,e,m,s']
    # W has shape (mpo_L, mpo_R, phys_up, phys_down) where phys_up=ket, phys_down=bra
    # Contract over w (mpo_L): X axis 1, W_L axis 0
    # Contract over s (phys_L_ket): X axis 2, W_L axis 2 (phys_up = ket side)
    Y = np.tensordot(X, W_left, axes=([1, 2], [0, 2]))
    # Remaining X: a(0), r(3), e(4) -> Y positions 0,1,2
    # Remaining W_L: m(1), p(2) -> Y positions 3,4
    # Y shape: (a, r, e, m, p)

    # Step 3: Y[a,r,e,m,s'] * W_R[m,n,r',r] -> Z[a,e,s',n,r']
    # W has shape (mpo_L, mpo_R, phys_up, phys_down) where phys_up=ket, phys_down=bra
    # Contract over r (phys_R_ket): Y axis 1, W_R axis 2 (phys_up = ket side)
    # Contract over m (mpo_mid): Y axis 3, W_R axis 0
    Z = np.tensordot(Y, W_right, axes=([1, 3], [2, 0]))
    # Remaining Y: a(0), e(2), p(4) -> Z positions 0,1,2
    # Remaining W_R: n(1), q(2) -> Z positions 3,4
    # Z shape: (a, e, p, n, q)

    # Step 4: Z[a,e,p,n,q] * R[f,n,e] -> result[a,p,q,f]
    # Contract over e (ket_R): Z axis 1, R axis 2
    # Contract over n (mpo_R): Z axis 3, R axis 1
    result = np.tensordot(Z, R_env, axes=([1, 3], [2, 1]))
    # Remaining Z: a(0), p(2), q(4) -> positions 0,1,2
    # Remaining R: f(0) -> position 3
    # result shape: (a, p, q, f) = (χ_L, d, d, χ_R) ✓

    return result


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

    def matvec(v):
        theta = v.reshape(shape_4d)
        result = apply_heff(L_env, R_env, W_left, W_right, theta)
        return result.ravel()

    return LinearOperator((dim, dim), matvec=matvec, dtype=dtype), shape_4d
