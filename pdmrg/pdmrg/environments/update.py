"""Environment update functions for DMRG sweeps.

Environments encode the contraction of the MPS, MPO, and conjugate MPS
for all sites to the left (L_env) or right (R_env) of the current bond.

Index conventions (all arrays in our internal format):
  MPS tensor A: (left_bond, phys, right_bond)
  MPO tensor W: (mpo_left, mpo_right, phys_up, phys_down)
  L_env: (bra_bond, mpo_bond, ket_bond)
  R_env: (bra_bond, mpo_bond, ket_bond)
"""

import numpy as np


def update_left_env(L_env, A, W):
    """Grow left environment by one site (sweeping right).

    L_new[a', w', c'] = sum_{a,w,c,s,s'} L[a,w,c] * A[a,s,a'] * W[w,w',s',s] * A*[c,s',c']

    Wait — careful with bra vs ket. In DMRG:
      - ket = |ψ⟩ uses A
      - bra = ⟨ψ| uses A* (conjugate)
      The environment accumulates ⟨ψ|H|ψ⟩.

    Actually the convention in the spec is:
      L[bra, mpo, ket] where bra contracts with A* and ket with A.

    So:
      L_new[a', w', c'] = L[a,w,c] * A*[a,s',a'] * W[w,w',s',s] * A[c,s,c']

    Wait, let me be very precise:
      L[bra=a, mpo=w, ket=c]
      A[left=c, phys=s, right=c']     (ket side)
      A*[left=a, phys=s', right=a']   (bra side, conjugate)
      W[mpo_left=w, mpo_right=w', phys_up=s', phys_down=s]

    Contract:
      L_new[a', w', c'] = L[a,w,c] * conj(A)[a,s',a'] * W[w,w',s',s] * A[c,s,c']

    Parameters
    ----------
    L_env : ndarray, shape (χ_bra, D, χ_ket)
    A : ndarray, shape (χ_ket, d, χ_ket')  — ket tensor
    W : ndarray, shape (D, D', d, d) — MPO tensor (mpo_L, mpo_R, up, down)

    Returns
    -------
    L_new : ndarray, shape (χ_bra', D', χ_ket')
    """
    A_conj = A.conj()

    # Step 1: L[a,w,c] * A[c,s,c'] -> X[a,w,s,c']  (sum over c)
    X = np.tensordot(L_env, A, axes=(2, 0))
    # X shape: (a, w, s, c')

    # Step 2: X[a,w,s,c'] * W[w,w',s',s] -> Y[a,c',w',s']  (sum over w,s)
    # W has shape (mpo_L, mpo_R, phys_up, phys_down) where:
    #   - phys_up (axis 2) connects to ket (A)
    #   - phys_down (axis 3) connects to bra (A*)
    # Contract X's mpo_bond (axis 1) with W's mpo_L (axis 0)
    # Contract X's ket_phys (axis 2) with W's phys_up (axis 2)
    Y = np.tensordot(X, W, axes=([1, 2], [0, 2]))
    # X axes 1(w),2(s) contract with W axes 0(mpo_L),2(phys_up)
    # Remaining X: (a=0, c'=3) -> pos 0,1
    # Remaining W: (mpo_R=1, phys_down=3) -> pos 2,3
    # Y shape: (a, c', w', s_down) where s_down connects to bra

    # Step 3: Y[a,c',w',s'] * A*[a,s',a'] -> L_new[c',w',a']  (sum over a,s')
    L_new = np.tensordot(Y, A_conj, axes=([0, 3], [0, 1]))
    # Y axes 0(a),3(s') contract with A* axes 0(a),1(s')
    # Remaining Y: (c'=1, w'=2) -> pos 0,1
    # Remaining A*: (a'=2) -> pos 2
    # L_new shape: (c', w', a')

    # We want output shape (bra'=a', mpo'=w', ket'=c')
    # Currently (c', w', a') -> transpose to (a', w', c')
    L_new = np.transpose(L_new, (2, 1, 0))

    return L_new


def update_right_env(R_env, B, W):
    """Grow right environment by one site (sweeping left).

    R_new[a', w', c'] = R[a,w,c] * conj(B)[a',s',a] * W[w',w,s',s] * B[c',s,c]

    Parameters
    ----------
    R_env : ndarray, shape (χ_bra, D, χ_ket)
    B : ndarray, shape (χ_ket', d, χ_ket)  — ket tensor
    W : ndarray, shape (D', D, d, d) — MPO tensor (mpo_L, mpo_R, up, down)

    Returns
    -------
    R_new : ndarray, shape (χ_bra', D', χ_ket')
    """
    B_conj = B.conj()

    # Step 1: R[a,w,c] * B[c',s,c] -> X[a,w,c',s]  (sum over c)
    X = np.tensordot(R_env, B, axes=(2, 2))
    # R axis 2(c=ket) contracts with B axis 2(right=c)
    # X shape: (a, w, c', s)

    # Step 2: X[a,w,c',s] * W[w',w,s',s] -> Y[a,c',w',s']  (sum over w,s)
    # W has shape (mpo_L, mpo_R, phys_up, phys_down) where:
    #   - phys_up (axis 2) connects to ket (B)
    #   - phys_down (axis 3) connects to bra (B*)
    # Contract X's mpo_bond (axis 1) with W's mpo_R (axis 1)
    # Contract X's ket_phys (axis 3) with W's phys_up (axis 2)
    Y = np.tensordot(X, W, axes=([1, 3], [1, 2]))
    # X axes 1(w),3(s) contract with W axes 1(mpo_R),2(phys_up)
    # Remaining X: (a=0, c'=2) -> pos 0,1
    # Remaining W: (mpo_L=0, phys_down=3) -> pos 2,3
    # Y shape: (a, c', w', s_down) where s_down connects to bra

    # Step 3: Y[a,c',w',s'] * B*[a',s',a] -> R_new[c',w',a']  (sum over a,s')
    R_new = np.tensordot(Y, B_conj, axes=([0, 3], [2, 1]))
    # Y axes 0(a),3(s') contract with B* axes 2(right=a),1(phys=s')
    # Remaining Y: (c'=1, w'=2) -> pos 0,1
    # Remaining B*: (a'=0) -> pos 2
    # R_new shape: (c', w', a')

    # Transpose to (bra'=a', mpo'=w', ket'=c')
    R_new = np.transpose(R_new, (2, 1, 0))

    return R_new


def init_left_env(chi_L, D, dtype=np.float64):
    """Create trivial left environment for the left edge of the system.

    L[bra=0, mpo=0, ket=0] = 1, all others = 0.
    Shape: (1, 1, 1) for the boundary.

    But with actual bond dimensions:
    For an MPS starting at site 0, the left bond dim is 1.
    """
    L = np.zeros((chi_L, D, chi_L), dtype=dtype)
    L[0, 0, 0] = 1.0
    return L


def init_right_env(chi_R, D, dtype=np.float64):
    """Create trivial right environment for the right edge of the system.

    Shape: (1, 1, 1) for the boundary, or (chi_R, D, chi_R) with
    R[-1, -1, -1] = 1 for the right boundary of an MPO.
    """
    R = np.zeros((chi_R, D, chi_R), dtype=dtype)
    # For the right boundary, the MPO bond index is the last one
    R[-1, -1, -1] = 1.0
    return R
