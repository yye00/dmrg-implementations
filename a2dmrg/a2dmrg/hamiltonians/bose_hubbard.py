"""Bose-Hubbard / Josephson MPO constructors.

We keep this small and explicit (manual MPO) to support complex hopping.

Hamiltonian:
    H = -t Σ_i (a†_i a_{i+1} + h.c.) + (U/2) Σ_i n_i(n_i-1) - μ Σ_i n_i

Local basis: |0>, |1>, ..., |nmax>
"""

from __future__ import annotations

import numpy as np
import quimb.tensor as qtn


def bose_hubbard_mpo(
    L: int,
    t: complex = 1.0,
    U: float = 1.0,
    mu: float = 0.0,
    nmax: int = 3,
):
    """Create a Bose-Hubbard MPO.

    Parameters
    ----------
    L:
        Number of sites.
    t:
        Hopping amplitude (can be complex).
    U:
        On-site interaction.
    mu:
        Chemical potential.
    nmax:
        Maximum occupation; local dimension is nmax+1.
    """
    if L < 1:
        raise ValueError("L must be >= 1")

    d = nmax + 1

    # Bosonic creation operator a^
    a_dag = np.zeros((d, d), dtype=np.complex128)
    for n in range(d - 1):
        a_dag[n + 1, n] = np.sqrt(n + 1)

    # Annihilation operator a
    a = a_dag.T.conj()

    # Number operator n
    n_op = np.diag(np.arange(d, dtype=float)).astype(np.complex128)

    # n(n-1)
    nn1 = np.diag(np.arange(d, dtype=float) * (np.arange(d, dtype=float) - 1)).astype(np.complex128)

    I = np.eye(d, dtype=np.complex128)

    # Local onsite term
    onsite = (-mu) * n_op + (U / 2.0) * nn1

    mpo_tensors = []

    if L == 1:
        mpo_tensors.append(onsite)
    else:
        # First site: (right_bond=4, ket, bra)
        W0 = np.zeros((4, d, d), dtype=np.complex128)
        W0[0, :, :] = onsite
        W0[1, :, :] = (-t) * a_dag
        W0[2, :, :] = (-np.conj(t)) * a
        W0[3, :, :] = I
        mpo_tensors.append(W0)

        # Middle sites: (left_bond=4, right_bond=4, ket, bra)
        for _ in range(1, L - 1):
            W = np.zeros((4, 4, d, d), dtype=np.complex128)
            W[0, 0, :, :] = I
            W[0, 1, :, :] = (-t) * a_dag
            W[0, 2, :, :] = (-np.conj(t)) * a
            W[0, 3, :, :] = onsite
            W[1, 0, :, :] = a
            W[2, 0, :, :] = a_dag
            W[3, 3, :, :] = I
            mpo_tensors.append(W)

        # Last site: (left_bond=4, ket, bra)
        WL = np.zeros((4, d, d), dtype=np.complex128)
        WL[0, :, :] = onsite
        WL[1, :, :] = a
        WL[2, :, :] = a_dag
        WL[3, :, :] = I
        mpo_tensors.append(WL)

    return qtn.MatrixProductOperator(mpo_tensors)
