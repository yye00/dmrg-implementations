"""Eigensolver wrappers for DMRG optimization.

Uses scipy.sparse.linalg.eigsh (Lanczos) to find the ground state
of the effective Hamiltonian acting on the two-site wavefunction.
"""

import numpy as np
from scipy.sparse.linalg import eigsh

from pdmrg.numerics.effective_ham import build_heff_operator


def optimize_two_site(L_env, R_env, W_left, W_right, theta_init,
                      max_iter=30, tol=1e-10):
    """Find the ground state of H_eff starting from theta_init.

    Parameters
    ----------
    L_env : ndarray, shape (χ_L, D_L, χ_L)
    R_env : ndarray, shape (χ_R, D_R, χ_R)
    W_left : ndarray
        MPO tensor for left site.
    W_right : ndarray
        MPO tensor for right site.
    theta_init : ndarray, shape (χ_L, d, d, χ_R)
        Initial guess for two-site wavefunction.
    max_iter : int
        Maximum Lanczos iterations.
    tol : float
        Convergence tolerance for eigsh.

    Returns
    -------
    energy : float
        Ground state energy of H_eff.
    theta_opt : ndarray, shape (χ_L, d, d, χ_R)
        Optimized two-site wavefunction.
    """
    dtype = np.result_type(theta_init.dtype, L_env.dtype)
    H_eff, shape_4d = build_heff_operator(L_env, R_env, W_left, W_right,
                                          dtype=dtype)

    v0 = theta_init.ravel().astype(dtype)

    # For real symmetric: 'SA' (smallest algebraic) is correct and fastest.
    # For complex Hermitian: eigenvalues are real but eigsh with 'SA' can
    # have convergence issues; 'SA' still works since H is Hermitian.
    try:
        E, V = eigsh(H_eff, k=1, v0=v0, which='SA',
                     maxiter=max_iter, tol=tol)
    except Exception:
        # Fallback: increase iterations substantially
        E, V = eigsh(H_eff, k=1, v0=v0, which='SA',
                     maxiter=max_iter * 10, tol=tol * 10)

    energy = float(E[0].real)
    theta_opt = V[:, 0].reshape(shape_4d)

    return energy, theta_opt
