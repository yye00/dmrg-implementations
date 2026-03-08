"""Eigensolver wrappers for DMRG optimization.

Uses block-Davidson (LOBPCG-style) eigensolver as primary method,
with scipy.sparse.linalg.eigsh (Lanczos) as fallback.

The block-Davidson solver is BLAS-3 friendly (batched matvecs, dense
Rayleigh-Ritz projection) and structurally ready for GPU porting.
"""

import numpy as np
from scipy.sparse.linalg import eigsh

from pdmrg.numerics.effective_ham import build_heff_operator
from pdmrg.numerics.linalg_utils import block_davidson


def optimize_two_site(L_env, R_env, W_left, W_right, theta_init,
                      max_iter=30, tol=1e-10):
    """Find the ground state of H_eff starting from theta_init.

    Parameters
    ----------
    L_env : ndarray, shape (chi_L, D_L, chi_L)
    R_env : ndarray, shape (chi_R, D_R, chi_R)
    W_left : ndarray
        MPO tensor for left site.
    W_right : ndarray
        MPO tensor for right site.
    theta_init : ndarray, shape (chi_L, d, d, chi_R)
        Initial guess for two-site wavefunction.
    max_iter : int
        Maximum solver iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    energy : float
        Ground state energy of H_eff.
    theta_opt : ndarray, shape (chi_L, d, d, chi_R)
        Optimized two-site wavefunction.
    """
    dtype = np.result_type(theta_init.dtype, L_env.dtype)
    H_eff, shape_4d = build_heff_operator(L_env, R_env, W_left, W_right,
                                          dtype=dtype)

    v0 = theta_init.ravel().astype(dtype)
    dim = H_eff.shape[0]

    try:
        energy, v = block_davidson(H_eff.matvec, dim, v0,
                                   max_iter=max_iter, tol=tol)
        theta_opt = v.reshape(shape_4d)
    except Exception:
        # Fallback: Lanczos with relaxed tolerance
        E, V = eigsh(H_eff, k=1, v0=v0, which='SA',
                     maxiter=max_iter * 5, tol=tol * 100)
        energy = float(E[0].real)
        theta_opt = V[:, 0].reshape(shape_4d)

    return energy, theta_opt
