"""
Environment tensor construction for DMRG.

Environment tensors encode the contraction of MPS and MPO tensors
on one side of the site(s) being optimized.

Conventions:
- L[i] = Left environment at bond i (sites 0..i-1 contracted)
- R[i] = Right environment at bond i (sites i..L-1 contracted)
- Environment shape: (bra_bond, mpo_bond, ket_bond)
"""

import numpy as np


# ===================================================================
# Incremental environment builder (O(L) total, not O(L) per site)
# ===================================================================

def build_environments_incremental(mps_arrays, mpo_arrays):
    """Build all left and right environments in O(L) via two sweeps.

    Parameters
    ----------
    mps_arrays : list of ndarray
        MPS tensors in (chi_L, d, chi_R) format.
    mpo_arrays : list of ndarray
        MPO tensors in (D_L, D_R, d_up, d_down) format (from extract_mpo_arrays).

    Returns
    -------
    L_envs : list of ndarray, length L+1
        L_envs[i] has shape (chi_bra, D, chi_ket) for bond i.
    R_envs : list of ndarray, length L+1
        R_envs[i] has shape (chi_bra, D, chi_ket) for bond i.
    canon_arrays : list of ndarray
        canon_arrays[i] is the center tensor at site i in i-orthogonal form.
    left_canonical : list of ndarray
        left_canonical[i] is the left-canonical tensor at site i (from QR in left sweep).
        Use left_canonical[0..i-1] + canon_arrays[i] + right_canonical[i+1..L-1]
        to build a consistent candidate MPS at site i.
    right_canonical : list of ndarray
        right_canonical[i] is the right-canonical tensor at site i (from step 1).
        The full list forms a consistent right-canonical MPS.
    """
    L = len(mps_arrays)
    dtype = mps_arrays[0].dtype

    arrays = [a.copy() for a in mps_arrays]

    # --- Step 1: Right-canonicalize, build R_envs ---
    R_envs = [None] * (L + 1)
    R_envs[L] = np.ones((1, 1, 1), dtype=dtype)

    for i in range(L - 1, 0, -1):
        chi_L, d, chi_R = arrays[i].shape
        mat = arrays[i].reshape(chi_L, d * chi_R)
        Q, R = np.linalg.qr(mat.T)
        arrays[i] = Q.T.reshape(-1, d, chi_R)
        arrays[i - 1] = np.tensordot(arrays[i - 1], R.T, axes=(2, 0))
        R_envs[i] = _update_right_env_numpy(R_envs[i + 1], arrays[i], mpo_arrays[i])

    # Build R_envs[0] from site 0
    R_envs[0] = _update_right_env_numpy(R_envs[1], arrays[0], mpo_arrays[0])

    # Save the right-canonical MPS (consistent bond dims, from step 1)
    right_canonical = [a.copy() for a in arrays]

    # --- Step 2: Left-sweep, build L_envs ---
    L_envs = [None] * (L + 1)
    L_envs[0] = np.ones((1, 1, 1), dtype=dtype)

    canon_arrays = [None] * L
    left_canonical = [None] * L

    for i in range(L):
        canon_arrays[i] = arrays[i].copy()

        if i < L - 1:
            chi_L, d, chi_R = arrays[i].shape
            mat = arrays[i].reshape(chi_L * d, chi_R)
            Q, R = np.linalg.qr(mat)
            new_chi = Q.shape[1]
            arrays[i] = Q.reshape(chi_L, d, new_chi)
            left_canonical[i] = arrays[i].copy()
            arrays[i + 1] = np.tensordot(R, arrays[i + 1], axes=(1, 0))
            L_envs[i + 1] = _update_left_env_numpy(L_envs[i], arrays[i], mpo_arrays[i])
        else:
            left_canonical[i] = arrays[i].copy()

    # Build L_envs[L] from the last site (for completeness / symmetry with R_envs)
    L_envs[L] = _update_left_env_numpy(L_envs[L - 1], arrays[L - 1], mpo_arrays[L - 1])

    return L_envs, R_envs, canon_arrays, left_canonical, right_canonical


def _update_left_env_numpy(L_env, A, W):
    """Update left environment: L_new = contract(L, A, W, A*).

    L_env: (bra, mpo, ket)
    A: (chi_L, d, chi_R) -- ket tensor (left-canonical)
    W: (D_L, D_R, d_up, d_down) -- MPO tensor

    Convention: d_up connects to ket, d_down connects to bra.
    Same contraction pattern as compute_energy() in observables.py.
    """
    A_conj = A.conj()
    # Step 1: L[bra,mpo,ket] @ A[ket,d,ket'] -> X[bra,mpo,d_ket,ket']
    X = np.tensordot(L_env, A, axes=(2, 0))
    # Step 2: X[bra,mpo,d_ket,ket'] @ W[D_L,D_R,d_up,d_down]
    # Contract mpo (X axis 1) with D_L (W axis 0), d_ket (X axis 2) with d_up (W axis 2)
    Y = np.tensordot(X, W, axes=([1, 2], [0, 2]))
    # Y shape: (bra, ket', D_R, d_down)
    # Step 3: Y[bra,ket',D_R,d_down] @ A*[chi_L,d,chi_R]
    # Contract bra (Y axis 0) with chi_L (A* axis 0), d_down (Y axis 3) with d (A* axis 1)
    L_new = np.tensordot(Y, A_conj, axes=([0, 3], [0, 1]))
    # L_new shape: (ket', D_R, chi_R_conj) = (ket', mpo', bra')
    return L_new.transpose(2, 1, 0)


def _update_right_env_numpy(R_env, B, W):
    """Update right environment: R_new = contract(R, B, W, B*).

    R_env: (bra, mpo, ket)
    B: (chi_L, d, chi_R) -- ket tensor (right-canonical)
    W: (D_L, D_R, d_up, d_down) -- MPO tensor

    Convention: d_up connects to ket, d_down connects to bra.
    """
    B_conj = B.conj()
    # Step 1: B[chi_L,d,chi_R] @ R[bra,mpo,ket] -> X[chi_L,d_ket,bra,mpo]
    X = np.tensordot(B, R_env, axes=(2, 2))
    # Step 2: X[chi_L,d_ket,bra,mpo] @ W[D_L,D_R,d_up,d_down]
    # Contract d_ket (X axis 1) with d_up (W axis 2), mpo (X axis 3) with D_R (W axis 1)
    Y = np.tensordot(X, W, axes=([1, 3], [2, 1]))
    # Y shape: (chi_L, bra, D_L, d_down)
    # Step 3: Y[chi_L,bra,D_L,d_down] @ B*[chi_L,d,chi_R]
    # Contract bra (Y axis 1) with chi_R (B* axis 2), d_down (Y axis 3) with d (B* axis 1)
    R_new = np.tensordot(Y, B_conj, axes=([1, 3], [2, 1]))
    # R_new shape: (chi_L, D_L, chi_L_conj) = (ket_L, mpo_L, bra_L)
    return R_new.transpose(2, 1, 0)
