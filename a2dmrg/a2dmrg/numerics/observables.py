"""
Observable computations for MPS.

This module provides functions to compute:
- Energy: ⟨ψ|H|ψ⟩ for an MPS ψ and MPO H
- Overlap: ⟨φ|ψ⟩ between two MPS states

These are essential for:
1. Coarse-space minimization (Phase 3 of A2DMRG)
2. Convergence checking
3. Validation against reference implementations
"""

import numpy as np
import quimb.tensor as qtn
from typing import Union


def _get_mps_array(mps, site):
    """Extract MPS tensor as numpy array in standard format (left, phys, right)."""
    t = mps[site]
    data = np.asarray(t.data)
    inds = t.inds
    L = mps.L
    
    # Find physical index position (starts with 'k')
    phys_name = mps.site_ind_id.format(site)
    phys_pos = list(inds).index(phys_name)
    
    if data.ndim == 2:
        # Edge site: 2D tensor
        if site == 0:
            # First site: (phys, right) -> (1, phys, right)
            if phys_pos == 0:
                return data[None, :, :]
            else:  # (right, phys)
                return np.transpose(data, (1, 0))[None, :, :]
        else:
            # Last site: (left, phys) -> (left, phys, 1)
            if phys_pos == 1:
                return data[:, :, None]
            else:  # (phys, left)
                return np.transpose(data, (1, 0))[:, :, None]
    else:
        # Middle site: typically 3D tensor (left, phys, right), but quimb can also
        # represent a single-site MPS tensor as a 1D vector (phys,).
        if data.ndim == 1:
            # Single-site MPS stored as (phys,), promote to (1, phys, 1)
            return data[None, :, None]

        # 3D tensor, need to reorder to (left, phys, right)
        if phys_pos == 1:
            return data  # Already (left, phys, right)
        elif phys_pos == 0:
            return np.transpose(data, (1, 0, 2))  # (phys, left, right) -> (left, phys, right)
        else:
            return np.transpose(data, (0, 2, 1))  # (left, right, phys) -> (left, phys, right)


def _get_mpo_array(mpo, site):
    """Extract MPO tensor as numpy array in format (mpo_left, mpo_right, phys_up, phys_down)."""
    t = mpo[site]
    data = np.asarray(t.data)
    inds = t.inds
    L = mpo.L
    
    # Find physical indices (ket='k', bra='b')
    upper_name = mpo.upper_ind_id.format(site)  # 'k{site}'
    lower_name = mpo.lower_ind_id.format(site)  # 'b{site}'
    
    upper_pos = list(inds).index(upper_name)
    lower_pos = list(inds).index(lower_name)
    
    if data.ndim == 2:
        # Single-site MPO can be stored as a plain operator matrix (phys_up, phys_down)
        # with inds ('k0','b0'). Promote to (1, 1, phys_up, phys_down).
        # Ensure order is (phys_up, phys_down).
        if upper_pos == 0 and lower_pos == 1:
            op = data
        elif upper_pos == 1 and lower_pos == 0:
            op = np.transpose(data, (1, 0))
        else:
            raise ValueError("Unexpected physical index positions for 2D MPO")
        return op[None, None, :, :]

    if data.ndim == 3:
        # Edge site: 3D tensor (mpo_bond, phys_up, phys_down)
        bond_pos = [i for i in range(3) if i not in [upper_pos, lower_pos]][0]

        # Transpose to (mpo_bond, phys_up, phys_down)
        perm = [bond_pos, upper_pos, lower_pos]
        data_ordered = np.transpose(data, perm)

        if site == 0:
            return data_ordered[None, :, :, :]  # (1, mpo_right, phys_up, phys_down)
        else:
            return data_ordered[:, None, :, :]  # (mpo_left, 1, phys_up, phys_down)

    # Middle site: 4D tensor
    bond_positions = [i for i in range(4) if i not in [upper_pos, lower_pos]]
    # In quimb, bonds are typically ordered (left, right) in the index list
    perm = [bond_positions[0], bond_positions[1], upper_pos, lower_pos]
    return np.transpose(data, perm)


def compute_energy(
    mps: qtn.MatrixProductState,
    mpo,
    normalize: bool = True
) -> float:
    """
    Compute the energy ⟨ψ|H|ψ⟩ for an MPS and MPO using direct numpy contractions.

    Uses efficient left-to-right sweep instead of full tensor network contraction.
    This is much faster than quimb's tensor network for repeated computations.

    Parameters
    ----------
    mps : quimb.tensor.MatrixProductState
        The quantum state as an MPS
    mpo : quimb MPO
        The Hamiltonian as a Matrix Product Operator
    normalize : bool, optional
        If True, computes ⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩ (default: True)

    Returns
    -------
    energy : float
        The energy expectation value (real number)
    """
    L = mps.L
    
    # Extract first site tensors
    A0 = _get_mps_array(mps, 0)
    W0 = _get_mpo_array(mpo, 0)
    
    # Initialize left environment: (bra, mpo, ket)
    chi_L = A0.shape[0]  # Should be 1 for left edge
    D_L = W0.shape[0]    # Should be 1 for left edge
    L_env = np.zeros((chi_L, D_L, chi_L), dtype=A0.dtype)
    L_env[0, 0, 0] = 1.0
    
    # Sweep left to right, updating L_env
    for i in range(L):
        A = _get_mps_array(mps, i)
        W = _get_mpo_array(mpo, i)
        A_conj = A.conj()
        
        # L[a,w,c] @ A[c,s,c'] -> X[a,w,s,c']
        X = np.tensordot(L_env, A, axes=(2, 0))
        
        # X[a,w,s,c'] @ W[w,w',s',s] -> Y[a,c',w',s']
        # W has shape (mpo_L, mpo_R, phys_up, phys_down)
        Y = np.tensordot(X, W, axes=([1, 2], [0, 2]))
        
        # Y[a,c',w',s'] @ A*[a,s',a'] -> L_new[c',w',a']
        L_env = np.tensordot(Y, A_conj, axes=([0, 3], [0, 1]))
        L_env = np.transpose(L_env, (2, 1, 0))
    
    # Final trace to get energy
    energy = np.real(np.trace(L_env[:, 0, :]))
    
    # Normalize if requested
    if normalize:
        norm_sq = np.abs(mps.norm()) ** 2
        energy = energy / norm_sq
    
    return float(energy)


def compute_cross_energy(
    bra: qtn.MatrixProductState,
    mpo,
    ket: qtn.MatrixProductState,
) -> complex:
    """
    Compute ⟨bra|H|ket⟩ for two MPS states using direct numpy contractions.

    Uses efficient left-to-right sweep. Works for bra ≠ ket (cross terms)
    and bra = ket (diagonal). Returns unnormalized ⟨bra|H|ket⟩.

    Parameters
    ----------
    bra : quimb.tensor.MatrixProductState
    mpo : quimb MPO
    ket : quimb.tensor.MatrixProductState

    Returns
    -------
    complex
        The matrix element ⟨bra|H|ket⟩
    """
    assert bra.L == ket.L, "MPS lengths must match"
    L = bra.L

    A0_bra = _get_mps_array(bra, 0)
    A0_ket = _get_mps_array(ket, 0)
    W0 = _get_mpo_array(mpo, 0)

    chi_bra_L = A0_bra.shape[0]
    D_L = W0.shape[0]
    chi_ket_L = A0_ket.shape[0]
    dtype = np.result_type(A0_bra.dtype, A0_ket.dtype, W0.dtype)

    L_env = np.zeros((chi_bra_L, D_L, chi_ket_L), dtype=dtype)
    L_env[0, 0, 0] = 1.0

    for i in range(L):
        A_bra = _get_mps_array(bra, i)
        A_ket = _get_mps_array(ket, i)
        W = _get_mpo_array(mpo, i)
        A_bra_conj = A_bra.conj()

        # L[a,w,c] @ A_ket[c,s,c'] -> X[a,w,s,c']
        X = np.tensordot(L_env, A_ket, axes=(2, 0))

        # X[a,w,s,c'] @ W[w,w',s',s] -> Y[a,c',w',s']
        Y = np.tensordot(X, W, axes=([1, 2], [0, 2]))

        # Y[a,c',w',s'] @ A_bra*[a,s',a'] -> L_new[c',w',a']
        L_env = np.tensordot(Y, A_bra_conj, axes=([0, 3], [0, 1]))
        L_env = np.transpose(L_env, (2, 1, 0))

    return complex(np.trace(L_env[:, 0, :]))


def compute_overlap(
    bra: qtn.MatrixProductState,
    ket: qtn.MatrixProductState
) -> complex:
    """
    Compute the overlap ⟨φ|ψ⟩ between two MPS states using tensordot.

    Uses efficient left-to-right sweep with pairwise contractions.

    Parameters
    ----------
    bra : quimb.tensor.MatrixProductState
        The bra state ⟨φ|
    ket : quimb.tensor.MatrixProductState
        The ket state |ψ⟩

    Returns
    -------
    overlap : complex
        The overlap ⟨φ|ψ⟩ (complex number in general)
    """
    L = bra.L
    assert L == ket.L, "MPS lengths must match"
    
    # Initialize left boundary: (bra_bond, ket_bond)
    A0_bra = _get_mps_array(bra, 0)
    A0_ket = _get_mps_array(ket, 0)
    
    chi_bra = A0_bra.shape[0]  # Should be 1 for left edge
    chi_ket = A0_ket.shape[0]  # Should be 1 for left edge
    
    # Transfer matrix: T[bra_bond, ket_bond]
    T = np.zeros((chi_bra, chi_ket), dtype=np.result_type(A0_bra.dtype, A0_ket.dtype))
    T[0, 0] = 1.0
    
    # Sweep left to right using tensordot
    for i in range(L):
        A_bra = _get_mps_array(bra, i)  # (left, phys, right)
        A_ket = _get_mps_array(ket, i)  # (left, phys, right)
        A_bra_conj = A_bra.conj()
        
        # T[a,c] @ A_bra*[a,s,b] -> X[c,s,b]
        X = np.tensordot(T, A_bra_conj, axes=(0, 0))
        # X shape: (c, s, b)
        
        # X[c,s,b] @ A_ket[c,s,d] -> T_new[b,d]
        # Contract over c (axis 0 of X, axis 0 of A_ket) and s (axis 1 of both)
        T = np.tensordot(X, A_ket, axes=([0, 1], [0, 1]))
        # T shape: (b, d)
    
    # Final trace to get overlap
    overlap = np.trace(T)
    
    return complex(overlap)
