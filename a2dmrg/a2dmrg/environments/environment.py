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
        W_raw = np.asarray(mpo[i].data)
        W = _reshape_mpo_tensor(W_raw, i, L)  # (mpo_L, phys_out, mpo_R, phys_in)
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
        W_raw = np.asarray(mpo[i].data)
        W = _reshape_mpo_tensor(W_raw, i, L)  # (mpo_L, phys_out, mpo_R, phys_in)
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


def _reshape_mpo_tensor(W: np.ndarray, site: int, L: int) -> np.ndarray:
    """
    Reshape MPO tensor to form (left_mpo, phys_out, right_mpo, phys_in).

    Quimb MPO_ham_heis returns tensors with shapes:
    - First site (i=0): (right_mpo, phys_out, phys_in) with left_mpo=1 implicit
    - Middle sites: (left_mpo, right_mpo, phys_out, phys_in)
    - Last site (i=L-1): (left_mpo, phys_out, phys_in) with right_mpo=1 implicit

    We want: (mpo_left, phys_out, mpo_right, phys_in)

    Parameters
    ----------
    W : np.ndarray
        Raw MPO tensor from quimb
    site : int
        Site index
    L : int
        Total number of sites

    Returns
    -------
    W_reshaped : np.ndarray
        Tensor in standard form (left_mpo, phys_out, mpo_right, phys_in)
    """
    if W.ndim == 3:
        # Either first or last site
        if site == 0:
            # First site: (right_mpo, phys_out, phys_in) -> (1, phys_out, right_mpo, phys_in)
            # Need to transpose (right_mpo, phys_out, phys_in) -> (phys_out, right_mpo, phys_in)
            # Then expand_dims to add left_mpo=1 at axis 0
            W_transposed = W.transpose(1, 0, 2)  # (phys_out, right_mpo, phys_in)
            return np.expand_dims(W_transposed, axis=0)  # (1, phys_out, right_mpo, phys_in)
        else:
            # Last site: (left_mpo, phys_out, phys_in) -> (left_mpo, phys_out, 1, phys_in)
            # Already in correct order, just need to add right_mpo=1 at axis 2
            return np.expand_dims(W, axis=2)  # (left_mpo, phys_out, 1, phys_in)
    elif W.ndim == 4:
        # Middle site: (left_mpo, right_mpo, phys_out, phys_in)
        # We want: (left_mpo, phys_out, right_mpo, phys_in)
        # Transpose: (0, 2, 1, 3)
        return W.transpose(0, 2, 1, 3)
    else:
        return W
