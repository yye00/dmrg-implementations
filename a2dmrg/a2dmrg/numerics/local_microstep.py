"""
Fast numpy-based DMRG micro-steps for A2DMRG.

This module implements one-site and two-site local micro-step operations
for Phase 2 of the Additive Two-Level DMRG (A2DMRG) algorithm using pure
numpy tensordot contractions and pre-built cached environments.

The fast micro-steps (fast_microstep_1site, fast_microstep_2site) operate
on pre-extracted numpy arrays and pre-computed environment tensors, avoiding
any quimb overhead.  Canonicalization and environment construction are handled
externally by build_environments_incremental().

Reference: Grigori & Hassan, "An Additive Two-Level Parallel Variant
of the DMRG Algorithm with Coarse-Space Correction",
arXiv:2505.23429v2 (2025).
"""

import numpy as np

def _build_heff_numpy_1site(L_env, W, R_env):
    """Build 1-site effective Hamiltonian from pre-extracted numpy arrays.

    Takes already-extracted numpy arrays (no quimb objects).

    L_env: (bra, mpo, ket)
    W: (D_L, D_R, d_up, d_down) — from extract_mpo_arrays
    R_env: (bra, mpo, ket)
    """
    from scipy.sparse.linalg import LinearOperator

    bra_L, mpo_L, ket_L = L_env.shape
    bra_R, mpo_R, ket_R = R_env.shape
    d = W.shape[2]
    size = ket_L * d * ket_R

    def matvec(v):
        psi = v.reshape(ket_L, d, ket_R)
        # L[bra,mpo,ket] @ psi[ket,d,ket'] -> X[bra,mpo,d_ket,ket']
        X = np.tensordot(L_env, psi, axes=(2, 0))
        # X[bra,mpo,d_ket,ket'] @ W[D_L,D_R,d_up,d_down]
        # Contract mpo with D_L, d_ket with d_up
        Y = np.tensordot(X, W, axes=([1, 2], [0, 2]))
        # Y[bra,ket',D_R,d_down] @ R[bra_R,mpo_R,ket_R]
        # Contract ket' with ket_R, D_R with mpo_R
        result = np.tensordot(Y, R_env, axes=([1, 2], [2, 1]))
        # result shape: (bra, d_down, bra_R) = (bra_L, d_bra, bra_R)
        return result.ravel()

    return LinearOperator(shape=(size, size), matvec=matvec, dtype=L_env.dtype)


def _build_heff_numpy_2site(L_env, W1, W2, R_env):
    """Build 2-site effective Hamiltonian from pre-extracted numpy arrays.

    L_env: (bra, mpo, ket)
    W1, W2: (D_L, D_R, d_up, d_down) — from extract_mpo_arrays
    R_env: (bra, mpo, ket)
    """
    from scipy.sparse.linalg import LinearOperator

    bra_L, mpo_L, ket_L = L_env.shape
    bra_R, mpo_R, ket_R = R_env.shape
    d1 = W1.shape[2]
    d2 = W2.shape[2]
    size = ket_L * d1 * d2 * ket_R

    def matvec(v):
        theta = v.reshape(ket_L, d1, d2, ket_R)
        # L[bra,mpo,ket] @ theta[ket,d1,d2,ket_R'] -> X[bra,mpo,d1,d2,ket_R']
        X = np.tensordot(L_env, theta, axes=(2, 0))
        # X[bra,mpo,d1,d2,ket_R'] @ W1[D_L,D_R,d_up,d_down]
        # Contract mpo with D_L, d1 with d_up
        Y = np.tensordot(X, W1, axes=([1, 2], [0, 2]))
        # Y[bra,d2,ket_R',D_R1,d_down1]
        # Y @ W2[D_L2,D_R2,d_up2,d_down2]
        # Contract d2 (Y axis 1) with d_up2 (W2 axis 2), D_R1 (Y axis 3) with D_L2 (W2 axis 0)
        Z = np.tensordot(Y, W2, axes=([1, 3], [2, 0]))
        # Z[bra,ket_R',d_down1,D_R2,d_down2]
        # Z @ R[bra_R,mpo_R,ket_R]
        # Contract ket_R' (Z axis 1) with ket_R (R axis 2), D_R2 (Z axis 3) with mpo_R (R axis 1)
        result = np.tensordot(Z, R_env, axes=([1, 3], [2, 1]))
        # result[bra, d_down1, d_down2, bra_R]
        return result.ravel()

    return LinearOperator(shape=(size, size), matvec=matvec, dtype=L_env.dtype)


def fast_microstep_1site(center_tensor, W, L_env, R_env, site, L, tol=1e-10):
    """One-site micro-step using pre-built environments.

    No MPS copy, no canonization, no environment rebuild.

    Parameters
    ----------
    center_tensor : ndarray, shape (chi_L, d, chi_R)
    W : ndarray, shape (D_L, D_R, d_up, d_down)
    L_env : ndarray, shape (bra, mpo, ket)
    R_env : ndarray, shape (bra, mpo, ket)
    site : int
    L : int
    tol : float

    Returns
    -------
    optimized_tensor : ndarray, shape (chi_L, d, chi_R)
    eigenvalue : float
    """
    from .eigensolver import solve_effective_hamiltonian

    H_eff = _build_heff_numpy_1site(L_env, W, R_env)
    v0 = center_tensor.ravel()
    energy, eigvec = solve_effective_hamiltonian(H_eff, v0=v0, tol=tol)
    optimized = eigvec.reshape(center_tensor.shape)
    return optimized, float(np.real(energy))


def fast_microstep_2site(tensor_i, tensor_ip1, W_i, W_ip1,
                         L_env, R_env, site, L,
                         max_bond=None, cutoff=0.0, tol=1e-10):
    """Two-site micro-step using pre-built environments.

    Parameters
    ----------
    tensor_i : ndarray, shape (chi_L, d1, chi_M)
    tensor_ip1 : ndarray, shape (chi_M, d2, chi_R)
    W_i, W_ip1 : MPO tensors, shape (D_L, D_R, d_up, d_down)
    L_env : left environment at bond i
    R_env : right environment at bond i+2
    site : int
    L : int
    max_bond : int or None
    cutoff : float
    tol : float

    Returns
    -------
    U_tensor : ndarray, shape (chi_L, d1, chi_new)
    SVh_tensor : ndarray, shape (chi_new, d2, chi_R)
    eigenvalue : float
    """
    from .eigensolver import solve_effective_hamiltonian

    chi_L, d1, chi_M = tensor_i.shape
    _, d2, chi_R = tensor_ip1.shape

    # Form two-site theta
    theta = np.tensordot(tensor_i, tensor_ip1, axes=(2, 0))

    H_eff = _build_heff_numpy_2site(L_env, W_i, W_ip1, R_env)
    v0 = theta.ravel()
    energy, eigvec = solve_effective_hamiltonian(H_eff, v0=v0, tol=tol)

    # SVD split
    theta_opt = eigvec.reshape(chi_L * d1, d2 * chi_R)
    U, S, Vh = np.linalg.svd(theta_opt, full_matrices=False)

    # Truncate
    if max_bond is not None:
        chi_new = min(max_bond, len(S))
    else:
        chi_new = len(S)
    if cutoff > 0:
        mask = S > cutoff
        chi_new = min(chi_new, max(1, mask.sum()))

    U = U[:, :chi_new]
    S = S[:chi_new]
    Vh = Vh[:chi_new, :]

    U_tensor = U.reshape(chi_L, d1, chi_new)
    SVh_tensor = (np.diag(S) @ Vh).reshape(chi_new, d2, chi_R)

    return U_tensor, SVh_tensor, float(np.real(energy))
