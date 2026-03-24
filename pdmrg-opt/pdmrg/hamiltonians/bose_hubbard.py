"""Bose-Hubbard / Josephson junction model Hamiltonian.

H = -t sum_i (a†_i a_{i+1} + h.c.) + (U/2) sum_i n_i(n_i-1) - mu sum_i n_i

For Josephson junction physics, complex128 dtype is needed.
"""

import numpy as np
import quimb.tensor as qtn


def build_bose_hubbard_mpo(L, t=1.0, U=4.0, mu=2.0, n_max=3,
                            dtype='complex128'):
    """Build Bose-Hubbard MPO using quimb's SpinHam1D.

    Parameters
    ----------
    L : int
        Number of sites.
    t : float
        Hopping amplitude.
    U : float
        On-site interaction.
    mu : float
        Chemical potential.
    n_max : int
        Maximum boson occupancy per site (truncated Hilbert space).
    dtype : str
        Data type ('float64' or 'complex128').

    Returns
    -------
    mpo : quimb MPO
    """
    d = n_max + 1  # Local Hilbert space dimension

    # Build operator matrices for truncated bosons
    # a† (creation): a†|n> = sqrt(n+1)|n+1>
    a_dag = np.zeros((d, d), dtype=dtype)
    for n in range(d - 1):
        a_dag[n + 1, n] = np.sqrt(n + 1)

    # a (annihilation): a|n> = sqrt(n)|n-1>
    a = a_dag.conj().T

    # n = a†a (number operator)
    n_op = a_dag @ a

    # Build with SpinHam1D using custom local dimension (S=(d-1)/2 gives dim d)
    builder = qtn.SpinHam1D(S=(d - 1) / 2)

    # Hopping: -t (a†_i a_{i+1} + h.c.)
    builder.add_term(-t, a_dag, a)
    builder.add_term(-t, a, a_dag)

    # On-site: (U/2) n(n-1) - mu*n
    n2 = n_op @ n_op
    onsite = (U / 2.0) * (n2 - n_op) - mu * n_op
    builder.add_term(1.0, onsite)

    return builder.build_mpo(L)
