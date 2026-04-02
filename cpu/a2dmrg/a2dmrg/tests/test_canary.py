"""Fast canary test: a2dmrg accuracy at L=8, L=12 chi=20 vs quimb DMRG2.

Tests the numpy performance rewrite achieves 1e-10 accuracy vs quimb DMRG2.
finalize_sweeps=0 because quimb DMRG2 finalization degrades A2DMRG accuracy
(A2DMRG alone achieves ~1e-11, finalization raises it to ~1e-8 due to
MPS conversion and re-optimization to a different local minimum).
"""
import time

import numpy as np
import pytest
import quimb.tensor as qtn

from a2dmrg.hamiltonians.heisenberg import heisenberg_mpo
from a2dmrg.mpi_compat import MPI


def _quimb_reference(L, chi):
    """Run quimb DMRG2 and return ground state energy."""
    mpo = heisenberg_mpo(L)
    dmrg = qtn.DMRG2(mpo, bond_dims=chi, cutoffs=1e-14)
    dmrg.solve(max_sweeps=30, tol=1e-12, verbosity=0)
    return float(np.real(dmrg.energy))


@pytest.mark.mpi
def test_canary_heisenberg_l8():
    """A2DMRG must match quimb DMRG2 within 1e-10 for L=8 chi=20."""
    from a2dmrg.dmrg import a2dmrg_main

    comm = MPI.COMM_WORLD
    L, chi = 8, 20

    mpo = heisenberg_mpo(L)
    t0 = time.perf_counter()
    energy, mps = a2dmrg_main(
        L=L, mpo=mpo, bond_dim=chi, max_sweeps=20, tol=1e-12,
        warmup_sweeps=2, finalize_sweeps=0,
        comm=comm, verbose=False, one_site=False,
    )
    wall = time.perf_counter() - t0

    if comm.Get_rank() == 0:
        E_ref = _quimb_reference(L, chi)
        diff = abs(energy - E_ref)
        print(f"\n[canary L=8] A2DMRG: {energy:.12f}  quimb: {E_ref:.12f}"
              f"  diff: {diff:.2e}  wall: {wall:.1f}s")
        assert diff < 1e-10, f"Energy diff {diff:.2e} exceeds 1e-10"


@pytest.mark.mpi
def test_canary_heisenberg_l12():
    """Larger canary: L=12 chi=20."""
    from a2dmrg.dmrg import a2dmrg_main

    comm = MPI.COMM_WORLD
    L, chi = 12, 20

    mpo = heisenberg_mpo(L)
    t0 = time.perf_counter()
    energy, mps = a2dmrg_main(
        L=L, mpo=mpo, bond_dim=chi, max_sweeps=20, tol=1e-12,
        warmup_sweeps=2, finalize_sweeps=0,
        comm=comm, verbose=False, one_site=False,
    )
    wall = time.perf_counter() - t0

    if comm.Get_rank() == 0:
        E_ref = _quimb_reference(L, chi)
        diff = abs(energy - E_ref)
        print(f"\n[canary L=12] A2DMRG: {energy:.12f}  quimb: {E_ref:.12f}"
              f"  diff: {diff:.2e}  wall: {wall:.1f}s")
        assert diff < 1e-10, f"Energy diff {diff:.2e} exceeds 1e-10"
