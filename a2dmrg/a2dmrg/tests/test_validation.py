"""
Validation tests: Verify A2DMRG matches quimb serial DMRG.

Tests compare A2DMRG (np=1, serial mode) against quimb's DMRG2 solver
for the antiferromagnetic Heisenberg chain with open boundary conditions.
"""

import numpy as np
import pytest
import quimb.tensor as qtn
from a2dmrg.mpi_compat import MPI

from a2dmrg.dmrg import a2dmrg_main


def _create_heisenberg_mpo(L, J=1.0):
    """Create Heisenberg chain MPO using quimb's SpinHam1D."""
    from quimb.tensor import SpinHam1D

    builder = SpinHam1D(S=1 / 2)
    builder += J, 'X', 'X'
    builder += J, 'Y', 'Y'
    builder += J, 'Z', 'Z'
    return builder.build_mpo(L)


def _run_quimb_dmrg(L, bond_dim):
    """Run quimb DMRG2 and return the ground state energy."""
    H = qtn.MPO_ham_heis(L, cyclic=False)
    dmrg = qtn.DMRG2(H, bond_dims=bond_dim)
    dmrg.solve(tol=1e-12, max_sweeps=50, verbosity=0)
    return dmrg.energy


def _run_a2dmrg(L, bond_dim, warmup_sweeps=2, one_site=True):
    """Run A2DMRG in serial mode and return the ground state energy and MPS.
    
    Uses one-site updates by default (as per paper recommendation).
    """
    mpo = _create_heisenberg_mpo(L)
    energy, mps = a2dmrg_main(
        L=L,
        mpo=mpo,
        max_sweeps=30,
        bond_dim=bond_dim,
        tol=1e-12,
        comm=MPI.COMM_SELF,
        dtype=np.float64,
        one_site=one_site,
        verbose=False,
        warmup_sweeps=warmup_sweeps,
    )
    return energy, mps


def test_heisenberg_serial_l10():
    """Quick test: A2DMRG matches quimb DMRG for L=10 Heisenberg chain.
    
    Paper tolerance: 1e-6 (Grigori & Hassan, arXiv:2505.23429)
    """
    L = 10
    bond_dim = 32

    E_quimb = _run_quimb_dmrg(L, bond_dim)
    E_a2dmrg, _ = _run_a2dmrg(L, bond_dim, warmup_sweeps=2)

    diff = abs(E_a2dmrg - E_quimb)
    # Paper uses 1e-6 tolerance; we aim for better
    assert diff < 1e-6, (
        f"L={L}: |E_a2dmrg - E_quimb| = {diff:.3e} exceeds paper tolerance 1e-6 "
        f"(E_a2dmrg={E_a2dmrg:.12f}, E_quimb={E_quimb:.12f})"
    )


def test_heisenberg_serial_l20():
    """Validation test: A2DMRG matches quimb DMRG for L=20 Heisenberg chain.
    
    Paper tolerance: 1e-6 (Grigori & Hassan, arXiv:2505.23429)
    """
    L = 20
    bond_dim = 50

    E_quimb = _run_quimb_dmrg(L, bond_dim)
    E_a2dmrg, _ = _run_a2dmrg(L, bond_dim, warmup_sweeps=2)

    diff = abs(E_a2dmrg - E_quimb)
    # Paper uses 1e-6 tolerance; we aim for better
    assert diff < 1e-6, (
        f"L={L}: |E_a2dmrg - E_quimb| = {diff:.3e} exceeds paper tolerance 1e-6 "
        f"(E_a2dmrg={E_a2dmrg:.12f}, E_quimb={E_quimb:.12f})"
    )


def test_heisenberg_serial_l40():
    """Validation test: A2DMRG matches quimb DMRG for L=40 Heisenberg chain.

    Also verifies energy per site E/L ≈ -0.443 (approaching infinite chain limit).
    
    Paper tolerance: 1e-6 (Grigori & Hassan, arXiv:2505.23429)
    """
    L = 40
    bond_dim = 50

    E_quimb = _run_quimb_dmrg(L, bond_dim)
    E_a2dmrg, _ = _run_a2dmrg(L, bond_dim, warmup_sweeps=2)

    # Verify E/L ≈ -0.443 (infinite chain limit: -ln(2) + 1/4)
    E_per_site = E_a2dmrg / L
    assert abs(E_per_site - (-0.443)) < 0.01, (
        f"L={L}: E/L = {E_per_site:.6f}, expected ≈ -0.443"
    )

    # Verify A2DMRG matches quimb within paper tolerance (1e-6)
    diff = abs(E_a2dmrg - E_quimb)
    assert diff < 1e-6, (
        f"L={L}: |E_a2dmrg - E_quimb| = {diff:.3e} exceeds paper tolerance 1e-6 "
        f"(E_a2dmrg={E_a2dmrg:.12f}, E_quimb={E_quimb:.12f})"
    )
