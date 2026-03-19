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
from typing import List, Optional
import quimb.tensor as qtn


def build_left_environments(mps: qtn.MatrixProductState,
                            mpo: qtn.MatrixProductOperator) -> List[np.ndarray]:
    """
    Build all left environment tensors L[0], L[1], ..., L[L].

    L[i] encodes the contraction of sites 0..i-1 with the MPO.
    L[0] is initialized as a 1x1x1 identity.

    Update formula (sweeping left to right):
        L[i+1] = contract(L[i], A[i], W[i], A[i].conj())

    where the contraction is:
        L_new[bra', mpo', ket'] = Σ_{bra,mpo,ket,s,s'}
            L[bra, mpo, ket] *
            A_conj[bra, s, bra'] *
            W[mpo, s, mpo', s'] *
            A[ket, s', ket']

    Uses sequential tensordot contractions instead of a 4-tensor einsum.
    MPO tensors are reshaped using _reshape_mpo_tensor for consistency with
    effective_ham.py (both use the same (mpo_L, phys_out, mpo_R, phys_in) convention).

    Parameters
    ----------
    mps : MatrixProductState
        The MPS (should be in left-canonical or mixed canonical form)
    mpo : MatrixProductOperator
        The MPO (Hamiltonian)

    Returns
    -------
    left_envs : List[np.ndarray]
        List of left environments L[0], L[1], ..., L[L]
        Each has shape (bra_bond, mpo_bond, ket_bond)
    """
    from a2dmrg.numerics.observables import _get_mps_array

    L = mps.L  # Number of sites

    # Initialize L[0] as 1x1x1 identity
    L_env = np.ones((1, 1, 1), dtype=mps.dtype)
    left_envs = [L_env]

    # Sweep left to right, building L[1], L[2], ..., L[L]
    for i in range(L):
        A = _get_mps_array(mps, i)            # (left, phys, right) - ket tensor
        W = _reshape_mpo_tensor(mpo, i)       # (mpo_L, phys_out, mpo_R, phys_in)
        A_conj = A.conj()

        # Contraction: L[a,m,k] @ A_conj[a,s,A] @ W[m,s,M,p] @ A[k,p,K] -> L_new[A,M,K]
        # Step 1: L_env[a,m,k] contracted with A_conj[a,s,A] over bra (axis 0 of both)
        # -> X[m, k, s, A]
        X = np.tensordot(L_env, A_conj, axes=(0, 0))

        # Step 2: X[m,k,s,A] contracted with W[m,s,M,p] over mpo_L (m) and phys_out (s)
        # Contract X axis 0 (m=mpo) with W axis 0 (mpo_L)
        # Contract X axis 2 (s=phys_out) with W axis 1 (phys_out)
        # -> Y[k, A, M, p]
        Y = np.tensordot(X, W, axes=([0, 2], [0, 1]))

        # Step 3: Y[k,A,M,p] contracted with A[k,p,K] over ket_left (k) and phys_in (p)
        # Contract Y axis 0 (k=ket) with A axis 0 (left)
        # Contract Y axis 3 (p=phys_in) with A axis 1 (phys)
        # -> L_new[A, M, K] = (bra', mpo', ket')
        L_env = np.tensordot(Y, A, axes=([0, 3], [0, 1]))

        left_envs.append(L_env)

    return left_envs


def build_right_environments(mps: qtn.MatrixProductState,
                             mpo: qtn.MatrixProductOperator) -> List[np.ndarray]:
    """
    Build all right environment tensors R[L], R[L-1], ..., R[0].

    R[i] encodes the contraction of sites i..L-1 with the MPO.
    R[L] is initialized as a 1x1x1 identity.

    Update formula (sweeping right to left):
        R[i-1] = contract(R[i], B[i], W[i], B[i].conj())

    Uses sequential tensordot contractions for efficiency.
    MPO tensors use _reshape_mpo_tensor for consistency with effective_ham.py.

    Parameters
    ----------
    mps : MatrixProductState
        The MPS (should be in right-canonical or mixed canonical form)
    mpo : MatrixProductOperator
        The MPO (Hamiltonian)

    Returns
    -------
    right_envs : List[np.ndarray]
        List of right environments R[0], R[1], ..., R[L]
        Each has shape (bra_bond, mpo_bond, ket_bond)
        Indexed naturally: right_envs[i] = R[i]
    """
    from a2dmrg.numerics.observables import _get_mps_array

    L = mps.L  # Number of sites

    # Initialize R[L] as 1x1x1 identity
    R_env = np.ones((1, 1, 1), dtype=mps.dtype)
    right_envs = [R_env]

    # Sweep right to left, building R[L-1], R[L-2], ..., R[0]
    for i in range(L - 1, -1, -1):
        B = _get_mps_array(mps, i)            # (left, phys, right) - ket tensor
        W = _reshape_mpo_tensor(mpo, i)       # (mpo_L, phys_out, mpo_R, phys_in)
        B_conj = B.conj()

        # Contraction: B_conj[a,s,A] @ W[m,s,M,p] @ B[k,p,K] @ R[A,M,K] -> R_new[a,m,k]
        # Step 1: R_env[A,M,K] contracted with B[k,p,K] over ket_right (axis 2 of both)
        # -> X[A, M, k, p]
        X = np.tensordot(R_env, B, axes=(2, 2))

        # Step 2: X[A,M,k,p] contracted with W[m,s,M,p] over mpo_R (M) and phys_in (p)
        # Contract X axis 1 (M=mpo_R) with W axis 2 (mpo_R)
        # Contract X axis 3 (p=phys_in) with W axis 3 (phys_in)
        # -> Y[A, k, m, s]
        Y = np.tensordot(X, W, axes=([1, 3], [2, 3]))

        # Step 3: Y[A,k,m,s] contracted with B_conj[a,s,A] over bra_right (A) and phys_out (s)
        # Contract Y axis 0 (A=bra_right) with B_conj axis 2 (right=bra_right)
        # Contract Y axis 3 (s=phys_out) with B_conj axis 1 (phys)
        # -> Z[k, m, a]
        Z = np.tensordot(Y, B_conj, axes=([0, 3], [2, 1]))

        # Transpose Z[k,m,a] -> R_new[a,m,k] = (bra, mpo, ket)
        R_env = np.transpose(Z, (2, 1, 0))

        right_envs.append(R_env)

    # Reverse the list so right_envs[i] = R[i]
    right_envs.reverse()

    return right_envs


def _reshape_mps_tensor_from_quimb(tensor, site: int, L: int) -> np.ndarray:
    """
    Reshape quimb MPS tensor to standard form (left_bond, phys, right_bond).

    Uses quimb's index names to correctly identify dimensions instead of heuristics.

    Parameters
    ----------
    tensor : quimb.tensor.Tensor or np.ndarray
        MPS tensor (with .inds attribute if quimb Tensor)
    site : int
        Site index
    L : int
        Total number of sites

    Returns
    -------
    A_reshaped : np.ndarray
        Tensor in standard form (left_bond, phys, right_bond)
    """
    # If it's a quimb Tensor, use index names
    if hasattr(tensor, 'inds') and hasattr(tensor, 'data'):
        data = np.asarray(tensor.data)
        inds = tensor.inds

        # Find physical index (starts with 'k')
        phys_name = f'k{site}'
        if phys_name not in inds:
            # Fallback: look for any 'k' index
            phys_name = None
            for ind in inds:
                if ind.startswith('k'):
                    phys_name = ind
                    break

        if phys_name is not None and phys_name in inds:
            phys_pos = list(inds).index(phys_name)

            if data.ndim == 2:
                # Edge site
                if site == 0:
                    # First site: (phys, right) or (right, phys)
                    if phys_pos == 0:
                        # (phys, right) -> (1, phys, right)
                        return data[None, :, :]
                    else:
                        # (right, phys) -> (1, phys, right)
                        return np.transpose(data, (1, 0))[None, :, :]
                else:
                    # Last site: (left, phys) or (phys, left)
                    if phys_pos == 1:
                        # (left, phys) -> (left, phys, 1)
                        return data[:, :, None]
                    else:
                        # (phys, left) -> (left, phys, 1)
                        return np.transpose(data, (1, 0))[:, :, None]
            else:
                # Middle site: 3D tensor
                # Find which positions are left/right bonds
                bond_positions = [i for i in range(3) if i != phys_pos]
                # quimb typically uses (left, phys, right) or (left, right, phys)
                # We want (left, phys, right)
                if phys_pos == 1:
                    # Already (left, phys, right)
                    return data
                elif phys_pos == 2:
                    # (left, right, phys) -> (left, phys, right)
                    return np.transpose(data, (0, 2, 1))
                else:
                    # (phys, left, right) -> (left, phys, right)
                    return np.transpose(data, (1, 0, 2))

    # Fallback for raw numpy arrays
    A = np.asarray(tensor.data if hasattr(tensor, 'data') else tensor)
    return _reshape_mps_tensor(A, site, L)


def _reshape_mps_tensor(A: np.ndarray, site: int, L: int) -> np.ndarray:
    """
    Reshape raw MPS numpy array to standard form (left_bond, phys, right_bond).

    DEPRECATED: Use _reshape_mps_tensor_from_quimb for quimb Tensors.
    This fallback uses heuristics that fail when phys_dim ≈ bond_dim.
    """
    if A.ndim == 2:
        # Either first or last site
        if site == 0:
            # First site: (right_bond, phys) -> (1, phys, right_bond)
            right_bond, phys = A.shape
            return A.T.reshape(1, phys, right_bond)
        else:
            # Last site: (left_bond, phys) -> (left_bond, phys, 1)
            left_bond, phys = A.shape
            return A.reshape(left_bond, phys, 1)
    elif A.ndim == 3:
        # Quimb native format: (left_bond, right_bond, phys)
        # Our format: (left_bond, phys, right_bond)
        # Always assume quimb format for 3D arrays
        return A.transpose(0, 2, 1)
    else:
        return A


def _reshape_mpo_tensor(mpo_or_array, site: int, L: int = None) -> np.ndarray:
    """
    Extract MPO tensor and reshape to (mpo_left, phys_out, mpo_right, phys_in).

    When *mpo_or_array* is a quimb ``MatrixProductOperator``, uses the
    ``.inds`` attribute to resolve index positions (preferred path).
    When it is a raw numpy array (legacy callers in effective_ham.py),
    falls back to positional heuristics.

    Parameters
    ----------
    mpo_or_array : MatrixProductOperator or np.ndarray
        Either the full quimb MPO (preferred) or a raw numpy array (legacy).
    site : int
        Site index.
    L : int, optional
        Total number of sites.  Required only for the legacy (raw array)
        path; ignored when *mpo_or_array* is an MPO.

    Returns
    -------
    W : np.ndarray
        Tensor in standard form (mpo_left, phys_out, mpo_right, phys_in).
    """

    # -----------------------------------------------------------
    # Preferred path: quimb MPO object — use index-name resolution
    # -----------------------------------------------------------
    if hasattr(mpo_or_array, 'upper_ind_id'):
        mpo = mpo_or_array
        t = mpo[site]
        data = np.asarray(t.data)
        inds = t.inds

        # Identify the physical indices by name
        upper_name = mpo.upper_ind_id.format(site)  # phys_out / ket, e.g. 'k0'
        lower_name = mpo.lower_ind_id.format(site)  # phys_in  / bra, e.g. 'b0'

        upper_pos = list(inds).index(upper_name)
        lower_pos = list(inds).index(lower_name)

        if data.ndim == 2:
            # Single-site MPO stored as (phys_out, phys_in) with no bond dims.
            if upper_pos == 0 and lower_pos == 1:
                op = data
            else:
                op = np.transpose(data, (1, 0))
            return op[None, :, None, :]

        if data.ndim == 3:
            # Edge site: one bond index + two physical indices.
            bond_pos = [i for i in range(3) if i not in (upper_pos, lower_pos)][0]

            if site == 0:
                # bond is mpo_right; mpo_left is implicit size-1
                perm = [upper_pos, bond_pos, lower_pos]
                return np.transpose(data, perm)[None, :, :, :]
            else:
                # bond is mpo_left; mpo_right is implicit size-1
                perm = [bond_pos, upper_pos, lower_pos]
                ordered = np.transpose(data, perm)
                return ordered[:, :, None, :]

        # Middle site: 4D — two bond indices + two physical indices.
        bond_positions = [i for i in range(4) if i not in (upper_pos, lower_pos)]
        # quimb keeps bonds in (left, right) order within the index tuple
        mpo_left_pos = bond_positions[0]
        mpo_right_pos = bond_positions[1]

        perm = [mpo_left_pos, upper_pos, mpo_right_pos, lower_pos]
        return np.transpose(data, perm)

    # -----------------------------------------------------------
    # Legacy fallback: raw numpy array with positional heuristics
    # (used by effective_ham.py which receives pre-extracted arrays)
    # -----------------------------------------------------------
    W = np.asarray(mpo_or_array)

    if W.ndim == 3:
        if site == 0:
            W_transposed = W.transpose(1, 0, 2)
            return np.expand_dims(W_transposed, axis=0)
        else:
            return np.expand_dims(W, axis=2)
    elif W.ndim == 4:
        return W.transpose(0, 2, 1, 3)
    else:
        return W


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
        MPS tensors after canonicalization. canon_arrays[i] is the center tensor
        when the MPS is in i-orthogonal form.
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

    # --- Step 2: Left-sweep, build L_envs ---
    L_envs = [None] * (L + 1)
    L_envs[0] = np.ones((1, 1, 1), dtype=dtype)

    canon_arrays = [None] * L

    for i in range(L):
        canon_arrays[i] = arrays[i].copy()

        if i < L - 1:
            chi_L, d, chi_R = arrays[i].shape
            mat = arrays[i].reshape(chi_L * d, chi_R)
            Q, R = np.linalg.qr(mat)
            new_chi = Q.shape[1]
            arrays[i] = Q.reshape(chi_L, d, new_chi)
            arrays[i + 1] = np.tensordot(R, arrays[i + 1], axes=(1, 0))
            L_envs[i + 1] = _update_left_env_numpy(L_envs[i], arrays[i], mpo_arrays[i])

    # Build L_envs[L] from the last site (for completeness / symmetry with R_envs)
    L_envs[L] = _update_left_env_numpy(L_envs[L - 1], arrays[L - 1], mpo_arrays[L - 1])

    return L_envs, R_envs, canon_arrays


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
