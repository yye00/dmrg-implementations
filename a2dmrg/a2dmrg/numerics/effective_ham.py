"""
Effective Hamiltonian construction for DMRG.

This module builds the effective Hamiltonians H_eff as LinearOperator objects
for both one-site and two-site DMRG updates.

Index conventions (must match environment.py):
- MPS tensors: (left_bond, phys, right_bond)
- MPO tensors: (mpo_L, phys_out, mpo_R, phys_in)
- L_env: (bra, mpo, ket) 
- R_env: (bra, mpo, ket)
- theta (two-site): (left_bond, phys_1, phys_2, right_bond)

Author: A2DMRG Implementation
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator

# Import reshape functions for consistency with environment building
from ..environments.environment import _reshape_mpo_tensor, _reshape_mps_tensor_from_quimb


def build_effective_hamiltonian_2site(L_env, W1, W2, R_env, site, L=None):
    """
    Build effective Hamiltonian for two-site DMRG update.
    
    Parameters
    ----------
    L_env : ndarray
        Left environment, shape (bra_L, mpo_L, ket_L)
    W1 : ndarray  
        MPO tensor at site (raw from quimb)
    W2 : ndarray
        MPO tensor at site+1 (raw from quimb)
    R_env : ndarray
        Right environment, shape (bra_R, mpo_R, ket_R)
    site : int
        First site of the two-site block
    L : int
        Total number of sites
        
    Returns
    -------
    LinearOperator
        Effective Hamiltonian for the two-site block
    """
    # Convert to numpy arrays
    L_env = np.asarray(L_env.data if hasattr(L_env, 'data') else L_env)
    W1 = np.asarray(W1.data if hasattr(W1, 'data') else W1)
    W2 = np.asarray(W2.data if hasattr(W2, 'data') else W2)
    R_env = np.asarray(R_env.data if hasattr(R_env, 'data') else R_env)
    
    # Backwards-compatible API: older tests pass `site` as a tuple of shapes.
    if isinstance(site, tuple):
        site_idx = 0
    else:
        site_idx = int(site)

    if L is None:
        # Infer minimal chain length; for 2-site effective Hamiltonian we need at least 2.
        L = 2

    # Reshape MPO tensors to standard form: (mpo_L, phys_out, mpo_R, phys_in)
    # Use the SAME function as environment building for consistency!
    W1_std = _reshape_mpo_tensor(W1, site_idx, L)
    W2_std = _reshape_mpo_tensor(W2, site_idx + 1, L)
    
    # Get dimensions
    bra_L, mpo_L, ket_L = L_env.shape
    bra_R, mpo_R, ket_R = R_env.shape
    _, phys1, _, _ = W1_std.shape
    _, phys2, _, _ = W2_std.shape
    
    # theta shape: (ket_L, phys1, phys2, ket_R)
    size = ket_L * phys1 * phys2 * ket_R
    
    def matvec(v):
        """
        Apply H_eff to vector v.
        
        Contraction order (optimized for typical DMRG dimensions):
        1. L_env @ theta -> contract ket_L bond
        2. @ W1 -> contract mpo_L and phys1 (ket)
        3. @ W2 -> contract mpo_mid and phys2 (ket)
        4. @ R_env -> contract mpo_R and ket_R
        Result: (bra_L, phys1_bra, phys2_bra, bra_R)
        
        Index convention:
        - L_env: (a, w, b) = (bra_L, mpo_L, ket_L)
        - theta: (b, p, q, d) = (ket_L, phys1, phys2, ket_R)
        - W1: (w, s, W, p) = (mpo_L, phys_out, mpo_mid, phys_in)
        - W2: (W, S, V, q) = (mpo_mid, phys_out, mpo_R, phys_in)
        - R_env: (c, V, d) = (bra_R, mpo_R, ket_R)
        - result: (a, s, S, c) = (bra_L, phys1_bra, phys2_bra, bra_R)
        """
        theta = v.reshape(ket_L, phys1, phys2, ket_R)
        
        # Step 1: L @ theta over ket_L (index b)
        # L[a,w,b] @ theta[b,p,q,d] -> X[a,w,p,q,d]
        X = np.tensordot(L_env, theta, axes=(2, 0))
        
        # Step 2: X @ W1 over mpo_L (w) and phys_ket_1 (p)
        # X[a,w,p,q,d] @ W1[w,s,W,p] -> Y[a,q,d,s,W]
        Y = np.tensordot(X, W1_std, axes=([1, 2], [0, 3]))
        
        # Step 3: Y @ W2 over mpo_mid (W) and phys_ket_2 (q)
        # Y[a,q,d,s,W] @ W2[W,S,V,q] -> Z[a,d,s,S,V]
        Z = np.tensordot(Y, W2_std, axes=([1, 4], [3, 0]))
        
        # Step 4: Z @ R_env over mpo_R (V) and ket_R (d)
        # Z[a,d,s,S,V] @ R[c,V,d] -> result[a,s,S,c]
        result = np.tensordot(Z, R_env, axes=([1, 4], [2, 1]))
        
        return result.ravel()
    
    return LinearOperator(shape=(size, size), matvec=matvec, dtype=L_env.dtype)


def build_effective_hamiltonian_1site(L_env, W, R_env, site, L=None):
    """
    Build effective Hamiltonian for one-site DMRG update.
    
    Parameters
    ----------
    L_env : ndarray
        Left environment, shape (bra_L, mpo_L, ket_L)
    W : ndarray
        MPO tensor at site (raw from quimb)
    R_env : ndarray
        Right environment, shape (bra_R, mpo_R, ket_R)
    site : int
        Site index
    L : int
        Total number of sites
        
    Returns
    -------
    LinearOperator
        Effective Hamiltonian for the one-site block
    """
    # Convert to numpy arrays
    L_env = np.asarray(L_env.data if hasattr(L_env, 'data') else L_env)
    W = np.asarray(W.data if hasattr(W, 'data') else W)
    R_env = np.asarray(R_env.data if hasattr(R_env, 'data') else R_env)
    
    # Backwards-compatible API: older tests pass `site` as a (chi_L, chi_R, d) tuple.
    # In that case infer `site_idx` and total length from MPO tensor rank.
    if isinstance(site, tuple):
        site_idx = 0
    else:
        site_idx = int(site)

    if L is None:
        # Infer from MPO tensor rank: edge MPO tensors are 3D, middle are 4D.
        L = 1 if W.ndim == 3 else 2

    # Reshape MPO to standard form
    W_std = _reshape_mpo_tensor(W, site_idx, L)
    
    # Get dimensions
    bra_L, mpo_L, ket_L = L_env.shape
    bra_R, mpo_R, ket_R = R_env.shape
    _, phys, _, _ = W_std.shape
    
    # MPS tensor shape: (ket_L, phys, ket_R)
    size = ket_L * phys * ket_R
    
    def matvec(v):
        """
        Apply H_eff to vector v.
        
        Index convention:
        - L_env: (a, w, b) = (bra_L, mpo_L, ket_L)
        - psi: (b, p, d) = (ket_L, phys, ket_R)
        - W: (w, s, V, p) = (mpo_L, phys_out, mpo_R, phys_in)
        - R_env: (c, V, d) = (bra_R, mpo_R, ket_R)
        - result: (a, s, c) = (bra_L, phys_bra, bra_R)
        """
        psi = v.reshape(ket_L, phys, ket_R)
        
        # Contract: L @ psi @ W @ R
        # L[a,w,b] @ psi[b,p,d] -> X[a,w,p,d]
        X = np.tensordot(L_env, psi, axes=(2, 0))
        
        # X[a,w,p,d] @ W[w,s,V,p] -> Y[a,d,s,V]
        Y = np.tensordot(X, W_std, axes=([1, 2], [0, 3]))
        
        # Y[a,d,s,V] @ R[c,V,d] -> result[a,s,c]
        result = np.tensordot(Y, R_env, axes=([1, 3], [2, 1]))
        
        return result.ravel()
    
    return LinearOperator(shape=(size, size), matvec=matvec, dtype=L_env.dtype)
