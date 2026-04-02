"""Heisenberg model Hamiltonian via quimb."""

import quimb.tensor as qtn


def build_heisenberg_mpo(L, j=1.0, bz=0.0, cyclic=False):
    """Build Heisenberg XXX chain MPO.

    H = J sum_i (S^x_i S^x_{i+1} + S^y_i S^y_{i+1} + S^z_i S^z_{i+1})
      + bz sum_i S^z_i

    Parameters
    ----------
    L : int
        Number of sites.
    j : float
        Coupling constant.
    bz : float
        External magnetic field along z.
    cyclic : bool
        Periodic boundary conditions.

    Returns
    -------
    mpo : quimb MPO
    """
    return qtn.MPO_ham_heis(L=L, j=j, bz=bz, cyclic=cyclic)
