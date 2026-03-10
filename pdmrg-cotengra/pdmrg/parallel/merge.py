"""Boundary merge operations for PDMRG.

When two processors meet at a shared bond, merge their wavefunctions
using the V = Lambda^-1 prescription (Eq. 5 of Stoudenmire & White),
optimize the two-site wavefunction, and split via SVD.
"""

import numpy as np
from pdmrg.numerics.eigensolver import optimize_two_site
from pdmrg.numerics.accurate_svd import accurate_svd, compute_v_from_svd


def merge_boundary_tensors(psi_left, psi_right, V,
                           L_env, R_env, W_left, W_right,
                           max_bond, max_iter=30, tol=1e-10):
    """Merge two boundary tensors using V, optimize, and split.

    Implements Eq. 5: Psi' = psi_left . diag(V) . psi_right
    The V matrix bridges the independently-evolved wavefunctions.

    Parameters
    ----------
    psi_left : ndarray, shape (chi_L, d, chi_bond)
        Left boundary tensor (right edge of left rank).
    psi_right : ndarray, shape (chi_bond, d, chi_R)
        Right boundary tensor (left edge of right rank).
    V : ndarray, shape (chi_bond,)
        Current V = Lambda^-1 at the shared bond.
    L_env : ndarray, shape (chi_L, D, chi_L)
        Left environment from left rank.
    R_env : ndarray, shape (chi_R, D, chi_R)
        Right environment from right rank.
    W_left : ndarray, shape (D_L, D_R, d, d)
        MPO tensor at left boundary site.
    W_right : ndarray, shape (D_L, D_R, d, d)
        MPO tensor at right boundary site.
    max_bond : int
        Maximum bond dimension after SVD truncation.
    max_iter : int
        Maximum eigensolver iterations.
    tol : float
        Eigensolver tolerance.

    Returns
    -------
    A_left_new : ndarray, shape (chi_L, d, k)
        New left-canonical tensor for left boundary site.
    A_right_new : ndarray, shape (k, d, chi_R)
        New right-canonical tensor for right boundary site.
    V_new : ndarray, shape (k,)
        Updated V = 1/S for next iteration.
    energy : float
        Energy from the optimization.
    trunc_err : float
        Truncation error from SVD.
    """
    # Step 1: Form theta = psi_left . diag(V) . psi_right  (Eq. 5)
    # psi_left: (chi_L, d, chi_bond), V: (chi_bond,), psi_right: (chi_bond, d, chi_R)
    V_psi_right = V[:, None, None] * psi_right
    theta = np.einsum('ija,akl->ijkl', psi_left, V_psi_right)

    # Step 2: Optimize with eigensolver
    energy, theta_opt = optimize_two_site(
        L_env, R_env, W_left, W_right, theta,
        max_iter=max_iter, tol=tol
    )

    # Step 3: SVD to split using accurate_svd (per Stoudenmire & White paper,
    # accurate SVD is needed at merge boundaries for high-precision V = 1/S)
    chi_L, d_L, d_R, chi_R = theta_opt.shape
    M = theta_opt.reshape(chi_L * d_L, d_R * chi_R)

    U, S, Vh = accurate_svd(M)

    # Truncate to max_bond
    k = min(len(S), max_bond)
    k = max(k, 1)

    trunc_err = float(np.sum(S[k:] ** 2))
    U = U[:, :k]
    S = S[:k]
    Vh = Vh[:k, :]

    # Compute new V for next iteration
    V_new = compute_v_from_svd(S)

    # Split: A_left is left-canonical (U), A_right absorbs S
    A_left_new = U.reshape(chi_L, d_L, k)
    A_right_new = (np.diag(S) @ Vh).reshape(k, d_R, chi_R)

    # Also return the right-canonical part (Vh) for correct environment building.
    # After merge, boundary environments must be computed from CANONICAL tensors:
    #   L_env from A_left_new (U, left-canonical) → norm = I
    #   R_env from A_right_canonical (Vh, right-canonical) → norm = I
    # Using A_right_new (S*Vh) for R_env gives norm = S² ≠ I, which breaks
    # the standard eigenvalue assumption (N_eff = I) in subsequent sweeps.
    A_right_canonical = Vh.reshape(k, d_R, chi_R)

    return A_left_new, A_right_new, V_new, energy, trunc_err, A_right_canonical
