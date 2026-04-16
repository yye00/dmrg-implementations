"""Left/right environment contractions for an MPO sandwiched between an MPS.

Conventions
-----------

MPS core ``A`` has shape ``(chi_L, d, chi_R)`` where the middle axis is
the physical index of the ket.  The bra ``<X|`` is obtained by
complex-conjugating each core.

MPO core ``W`` has shape ``(mpo_L, mpo_R, d_up, d_dn)`` where
``d_up`` contracts with the **bra** physical index and ``d_dn``
contracts with the **ket** physical index.  This matches the usual
operator convention ``W[p_up, p_dn] = <p_up | W | p_dn>``.

Environment shapes: ``(chi_bra, mpo, chi_ket)``.

``L_env[i]`` holds the contraction of sites ``0..i-1``.  ``L_env[0]``
is the trivial 1x1x1 tensor of ones, ``L_env[L]`` equals ``<X|H|X>``.

``R_env[i]`` holds the contraction of sites ``i..L-1``.  ``R_env[L]``
is trivial, ``R_env[0]`` equals ``<X|H|X>``.
"""

from __future__ import annotations

import numpy as np


def build_left_envs_H(X, H):
    L = len(X)
    dtype = np.result_type(X[0].dtype, H[0].dtype)
    Ls = [None] * (L + 1)
    Ls[0] = np.ones((1, 1, 1), dtype=dtype)
    for i in range(L):
        Ls[i + 1] = _grow_left(X[i], H[i], Ls[i])
    return Ls


def build_right_envs_H(X, H):
    L = len(X)
    dtype = np.result_type(X[0].dtype, H[0].dtype)
    Rs = [None] * (L + 1)
    Rs[L] = np.ones((1, 1, 1), dtype=dtype)
    for i in range(L - 1, -1, -1):
        Rs[i] = _grow_right(X[i], H[i], Rs[i + 1])
    return Rs


def _grow_left(A, W, Lprev):
    """Absorb site ``i`` into the left environment.

    Index order used in einsums:
      a = a_bra, l = m_L (old), b = b_ket, q = p_dn, n = m_R, p = p_up,
      c = b_bra.
    """
    # T1[a, l, q, b] = Lprev[a, l, a_ket] * A[a_ket, q, b]
    T1 = np.einsum("alk,kqb->alqb", Lprev, A)
    # T2[a, n, p, b] = T1[a, l, q, b] * W[l, n, p, q]
    T2 = np.einsum("alqb,lnpq->anpb", T1, W)
    # out[c, n, b] = T2[a, n, p, b] * conj(A)[a, p, c]
    out = np.einsum("anpb,apc->cnb", T2, A.conj())
    return out


def _grow_right(A, W, Rnext):
    """Absorb site ``i`` into the right environment."""
    # T1[a_ket, q, c_bra, n_R] = A[a_ket, q, b_ket] * Rnext[c_bra, n_R, b_ket]
    T1 = np.einsum("aqb,cnb->aqcn", A, Rnext)
    # T2[a_ket, p, c_bra, l_L] = T1[a_ket, q, c_bra, n_R] * W[l_L, n_R, p, q]
    T2 = np.einsum("aqcn,lnpq->apcl", T1, W)
    # out[d_bra, l_L, a_ket] = T2[a_ket, p, c_bra, l_L] * conj(A)[d_bra, p, c_bra]
    out = np.einsum("apcl,dpc->dla", T2, A.conj())
    return out


def build_left_envs_norm(X):
    """Left norm environments, shape ``(chi_bra, chi_ket)``.

    ``L[L][0,0]`` equals ``<X|X>``.
    """
    L = len(X)
    Ls = [None] * (L + 1)
    Ls[0] = np.ones((1, 1), dtype=X[0].dtype)
    for i in range(L):
        A = X[i]
        # L[i+1][b_bra, b_ket] = L[i][a_bra, a_ket] * conj(A)[a_bra, p, b_bra] * A[a_ket, p, b_ket]
        Ls[i + 1] = np.einsum("ac,apb,cpd->bd", Ls[i], A.conj(), A)
    return Ls


def build_right_envs_norm(X):
    """Right norm environments, shape ``(chi_bra, chi_ket)``.

    ``R[0][0,0]`` equals ``<X|X>``.
    """
    L = len(X)
    Rs = [None] * (L + 1)
    Rs[L] = np.ones((1, 1), dtype=X[0].dtype)
    for i in range(L - 1, -1, -1):
        A = X[i]
        # R[i][a_bra, a_ket] = conj(A)[a_bra, p, b_bra] * A[a_ket, p, b_ket] * R[i+1][b_bra, b_ket]
        Rs[i] = np.einsum("apb,cpd,bd->ac", A.conj(), A, Rs[i + 1])
    return Rs
