"""Heisenberg MPO constructors."""

from __future__ import annotations

import quimb.tensor as qtn


def heisenberg_mpo(L: int, J: float = 1.0, bz: float = 0.0, cyclic: bool = False):
    """Build the spin-1/2 Heisenberg XXX chain MPO via quimb.

    Parameters
    ----------
    L:
        Number of sites.
    J:
        Coupling constant.
    bz:
        Z-field term (quimb parameter).
    cyclic:
        Periodic boundary conditions.
    """
    return qtn.MPO_ham_heis(L=L, j=J, bz=bz, cyclic=cyclic)
