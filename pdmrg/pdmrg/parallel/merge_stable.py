"""
Stable Boundary Merge Implementation for PDMRG

This module implements numerically stable boundary merge using exact SVD
without inverse singular-value amplification.

Key Difference from Original:
- REMOVES: V = Lambda^-1 amplification (lines 65-66 of merge.py)
- ADDS: Direct two-site wavefunction formation using Lambda (not Lambda^-1)
- ADDS: Diagnostics for singular value spectrum and conditioning

Root Cause (Original Implementation):
When singular values S approach truncation threshold, V = 1/S becomes very large,
amplifying numerical errors in theta = psi_left · diag(V) · psi_right.

Solution:
Form theta using Lambda = S directly, avoiding inversion and amplification.
"""

import numpy as np
from pdmrg.numerics.eigensolver import optimize_two_site
from pdmrg.numerics.accurate_svd import accurate_svd
from pdmrg.numerics.effective_ham import apply_heff


def merge_boundary_tensors_stable(psi_left, psi_right, Lambda,
                                   L_env, R_env, W_left, W_right,
                                   max_bond, max_iter=30, tol=1e-10,
                                   skip_optimization=False,
                                   return_diagnostics=False):
    """
    Merge two boundary tensors using exact SVD approach (stable).

    CRITICAL CHANGE: Uses Lambda (singular values) directly instead of
    V = 1/Lambda to avoid numerical amplification.

    Parameters
    ----------
    psi_left : ndarray, shape (chi_L, d, chi_bond)
        Left boundary tensor (right edge of left rank).
    psi_right : ndarray, shape (chi_bond, d, chi_R)
        Right boundary tensor (left edge of right rank).
    Lambda : ndarray, shape (chi_bond,)
        Singular values at the shared bond (NOT inverted).
        This is S from SVD, not V = 1/S.
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
    skip_optimization : bool
        If True, skip eigensolver and just compute energy.
    return_diagnostics : bool
        If True, return diagnostic information about conditioning.

    Returns
    -------
    A_left_new : ndarray, shape (chi_L, d, k)
        New left-canonical tensor for left boundary site.
    A_right_new : ndarray, shape (k, d, chi_R)
        New right-canonical tensor for right boundary site.
    Lambda_new : ndarray, shape (k,)
        Updated singular values (NOT inverted).
    energy : float
        Energy from the optimization.
    trunc_err : float
        Truncation error from SVD.
    diagnostics : dict (optional)
        Diagnostic information if return_diagnostics=True.
    """
    # Step 1: Form theta = psi_left . diag(Lambda) . psi_right
    # Key difference: multiply by Lambda (small), not V = 1/Lambda (large)
    # This avoids amplifying numerical errors.
    #
    # psi_left: (chi_L, d, chi_bond)
    # Lambda: (chi_bond,)
    # psi_right: (chi_bond, d, chi_R)
    #
    # Lambda_psi_right = diag(Lambda) @ psi_right
    Lambda_psi_right = Lambda[:, None, None] * psi_right
    theta = np.einsum('ija,akl->ijkl', psi_left, Lambda_psi_right)

    # Compute diagnostics BEFORE optimization
    if return_diagnostics:
        diagnostics = {
            "lambda_min": float(Lambda.min()),
            "lambda_max": float(Lambda.max()),
            "lambda_condition": float(Lambda.max() / max(Lambda.min(), 1e-16)),
            "lambda_mean": float(Lambda.mean()),
            "lambda_spectrum": Lambda.tolist(),
            "theta_norm": float(np.linalg.norm(theta)),
        }
    else:
        diagnostics = None

    # Step 2: Optimize with Lanczos (or skip if already converged)
    if skip_optimization:
        # Just compute the energy without optimization
        H_theta = apply_heff(L_env, R_env, W_left, W_right, theta)
        theta_flat = theta.ravel()
        H_theta_flat = H_theta.ravel()
        energy = float(np.real(np.vdot(theta_flat, H_theta_flat) / np.vdot(theta_flat, theta_flat)))
        theta_opt = theta
    else:
        energy, theta_opt = optimize_two_site(
            L_env, R_env, W_left, W_right, theta,
            max_iter=max_iter, tol=tol
        )

    # Step 3: Exact SVD to split
    chi_L, d_L, d_R, chi_R = theta_opt.shape
    M = theta_opt.reshape(chi_L * d_L, d_R * chi_R)

    U, S, Vh = accurate_svd(M)

    # Add SVD diagnostics
    if return_diagnostics:
        diagnostics["svd_spectrum"] = S.tolist()
        diagnostics["svd_min"] = float(S.min())
        diagnostics["svd_max"] = float(S.max())
        diagnostics["svd_condition"] = float(S.max() / max(S.min(), 1e-16))

    # Truncate to max_bond
    k = min(len(S), max_bond)
    k = max(k, 1)

    trunc_err = float(np.sum(S[k:] ** 2))
    U = U[:, :k]
    S_trunc = S[:k]
    Vh = Vh[:k, :]

    # Add truncation diagnostics
    if return_diagnostics:
        diagnostics["truncation_rank"] = k
        diagnostics["truncation_error"] = trunc_err
        diagnostics["kept_weight"] = float(np.sum(S_trunc**2) / np.sum(S**2))

    # New Lambda is S_trunc (singular values, NOT inverted)
    Lambda_new = S_trunc

    # Split: A_left is left-canonical (U), A_right absorbs S
    A_left_new = U.reshape(chi_L, d_L, k)
    A_right_new = (np.diag(S_trunc) @ Vh).reshape(k, d_R, chi_R)

    if return_diagnostics:
        return A_left_new, A_right_new, Lambda_new, energy, trunc_err, diagnostics
    else:
        return A_left_new, A_right_new, Lambda_new, energy, trunc_err


def merge_boundary_tensors_stable_from_V(psi_left, psi_right, V,
                                          L_env, R_env, W_left, W_right,
                                          max_bond, max_iter=30, tol=1e-10,
                                          skip_optimization=False,
                                          return_diagnostics=False):
    """
    Wrapper that converts V = 1/Lambda back to Lambda for stable merge.

    This function provides backward compatibility with code that passes V.
    It converts V -> Lambda, calls the stable merge, then returns results.

    IMPORTANT: This conversion itself may have issues if V was heavily
    regularized. The ideal solution is to stop storing V altogether and
    store Lambda throughout PDMRG.

    Parameters
    ----------
    V : ndarray
        V = 1/Lambda (with regularization). Will be converted to Lambda.
    (other parameters same as merge_boundary_tensors_stable)

    Returns
    -------
    Same as merge_boundary_tensors_stable
    """
    # Convert V back to Lambda
    # V was computed as: V = 1.0 / np.clip(S, 1e-12, None)
    # So Lambda ≈ 1/V, but must handle regularization carefully
    #
    # Problem: if V was clipped at 1/1e-12 = 1e12, then Lambda = 1/V = 1e-12
    # This is correct.
    #
    # However, if V has very large values due to small S, converting back
    # may still be problematic.
    #
    # Safest approach: clip Lambda to reasonable range
    Lambda = 1.0 / V
    Lambda = np.clip(Lambda, 1e-14, 1.0)  # Reasonable singular value range

    # Call stable merge
    result = merge_boundary_tensors_stable(
        psi_left, psi_right, Lambda,
        L_env, R_env, W_left, W_right,
        max_bond, max_iter, tol,
        skip_optimization, return_diagnostics
    )

    if return_diagnostics:
        A_left_new, A_right_new, Lambda_new, energy, trunc_err, diagnostics = result
        # Convert Lambda_new back to V for compatibility
        V_new = 1.0 / np.clip(Lambda_new, 1e-12, None)
        return A_left_new, A_right_new, V_new, energy, trunc_err, diagnostics
    else:
        A_left_new, A_right_new, Lambda_new, energy, trunc_err = result
        # Convert Lambda_new back to V for compatibility
        V_new = 1.0 / np.clip(Lambda_new, 1e-12, None)
        return A_left_new, A_right_new, V_new, energy, trunc_err
