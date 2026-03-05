#!/usr/bin/env python3
import sys
if "pytest" in sys.modules:
    import pytest
    pytest.skip("dev script (not a unit test)", allow_module_level=True)

"""
Test #61: Reproducibility across different np with fixed seed

This tests that A2DMRG gives the same result whether run with 1, 2, or 4 processors.
"""

import fix_quimb_python313

import numpy as np
from a2dmrg.mpi_compat import MPI, HAS_MPI
from a2dmrg.dmrg import a2dmrg_main
from a2dmrg.tests.test_a2dmrg_main import create_heisenberg_mpo


def test_reproducibility_across_np():
    """Test that results are identical across different np values."""

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Test parameters
    L = 10  # Small system for speed
    bond_dim = 8
    seed = 42

    if rank == 0:
        print("="*70)
        print("Test #61: Reproducibility across different np")
        print("="*70)
        print(f"L = {L}, bond_dim = {bond_dim}, seed = {seed}")
        print(f"Current np = {size}")
        print()

    # Create Hamiltonian
    mpo = create_heisenberg_mpo(L, J=1.0, cyclic=False)

    # Set seed for reproducibility
    np.random.seed(seed)

    # Run A2DMRG
    energy, mps = a2dmrg_main(
        L=L,
        mpo=mpo,
        max_sweeps=10,
        bond_dim=bond_dim,
        tol=1e-10,
        comm=comm,
        dtype=np.float64,
        one_site=True,
        verbose=(rank == 0)
    )

    if rank == 0:
        print()
        print("="*70)
        print("RESULT")
        print("="*70)
        print(f"np = {size}")
        print(f"Energy: {energy:.15f}")
        print(f"Energy per site: {energy/L:.15f}")
        print()
        print("To test reproducibility:")
        print(f"  Run with np=1: python3.13 test_reproducibility.py")
        print(f"  Run with np=2: mpirun -np 2 python3.13 test_reproducibility.py")
        print(f"  Run with np=4: mpirun -np 4 python3.13 test_reproducibility.py")
        print()
        print("All runs should give IDENTICAL energies to machine precision")


if __name__ == "__main__":
    test_reproducibility_across_np()
