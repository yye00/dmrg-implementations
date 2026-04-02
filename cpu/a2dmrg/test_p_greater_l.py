#!/usr/bin/env python3
import sys
if "pytest" in sys.modules:
    import pytest
    pytest.skip("dev script (not a unit test)", allow_module_level=True)

"""
Test #59: MPI edge case - More processors than sites (P > L)

This tests the scenario where we have more MPI processors than lattice sites.
Some processors should get zero sites assigned and handle this gracefully.
"""

import fix_quimb_python313  # Apply numba fix for Python 3.13+

import numpy as np
from a2dmrg.mpi_compat import MPI, HAS_MPI
import quimb.tensor as qtn
from a2dmrg.dmrg import a2dmrg_main
from a2dmrg.tests.test_a2dmrg_main import create_heisenberg_mpo


def test_p_greater_than_l():
    """Test P > L edge case with L=4, np=8."""

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Test parameters
    L = 4  # Small system
    bond_dim = 4
    np_expected = 8  # We expect to run with 8 processors

    # Only run this test if we actually have 8 processors
    if size != np_expected:
        if rank == 0:
            print(f"SKIP: Test requires np={np_expected}, but running with np={size}")
            print(f"Run with: mpirun -np {np_expected} python3.13 test_p_greater_l.py")
        return

    # Create Hamiltonian (all ranks do this)
    mpo = create_heisenberg_mpo(L, J=1.0, cyclic=False)

    if rank == 0:
        print("="*70)
        print(f"Test #59: P > L Edge Case")
        print("="*70)
        print(f"L = {L} sites")
        print(f"np = {size} processors")
        print(f"Processors {L} through {size-1} should get zero sites")
        print()

    # Run A2DMRG
    try:
        energy, mps = a2dmrg_main(
            L=L,
            mpo=mpo,
            max_sweeps=5,
            bond_dim=bond_dim,
            tol=1e-10,
            comm=comm,
            dtype=np.float64,
            one_site=True,
            verbose=(rank == 0)  # Only rank 0 prints
        )

        if rank == 0:
            print()
            print("="*70)
            print("RESULT")
            print("="*70)
            print(f"Final energy: {energy:.12f}")
            print(f"Energy per site: {energy/L:.12f}")
            print()

            # Sanity checks
            assert np.isfinite(energy), "Energy should be finite"
            assert energy < 0, "Ground state energy should be negative"
            assert energy > -L * 10, f"Energy seems unreasonable: {energy}"

            print("✓ Test PASSED: P > L edge case handled correctly")
            print("✓ Idle processors didn't cause errors")
            print("✓ Result is physically reasonable")

    except Exception as e:
        if rank == 0:
            print(f"\n✗ Test FAILED with error:")
            print(f"  {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    test_p_greater_than_l()
