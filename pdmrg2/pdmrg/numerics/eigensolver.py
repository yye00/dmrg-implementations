"""Eigensolver wrappers for DMRG optimization.

Uses Block-Davidson (LOBPCG-style) to find the ground state of the
effective Hamiltonian acting on the two-site wavefunction.

The block of *b* trial vectors replaces single-vector Lanczos, enabling
batched BLAS-3 Hamiltonian applications and a dense Rayleigh-Ritz
projection at each iteration.
"""

import numpy as np

from pdmrg.numerics.effective_ham import build_heff_operator
from pdmrg.numerics.linalg_utils import block_davidson


def optimize_two_site(L_env, R_env, W_left, W_right, theta_init,
                      max_iter=30, tol=1e-10, block_size=4):
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
        Maximum Block-Davidson iterations.
    tol : float
        Convergence tolerance (residual norm and energy change).
    block_size : int
        Number of trial vectors in the Davidson block (default 4).

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
    dim = v0.size

    try:
        energy, v = block_davidson(
            H_eff.matvec, dim, v0,
            b=block_size, max_iter=max_iter, tol=tol
        )
    except Exception:
        # Fallback to scipy eigsh if block_davidson fails (e.g. dim < b)
        from scipy.sparse.linalg import eigsh
        E, V = eigsh(H_eff, k=1, v0=v0, which='SA',
                     maxiter=max_iter * 5, tol=tol * 100)
        energy = float(E[0].real)
        v = V[:, 0]

    theta_opt = v.reshape(shape_4d)
    return energy, theta_opt
