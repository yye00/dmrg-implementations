"""Test #36: L=40 validation against quimb DMRG."""

import pytest
import numpy as np
from a2dmrg.mpi_compat import MPI
from quimb.tensor import SpinHam1D, DMRG2
from a2dmrg.dmrg import a2dmrg_main

pytestmark = pytest.mark.mpi


def test_l40_matches_quimb():
    """Test #36: A2DMRG matches quimb for L=40.

    This is a larger system test to validate that A2DMRG can handle
    longer chains and still match serial DMRG to machine precision.
    """
    L = 40
    bond_dim = 100

    # Step 1-2: Run quimb DMRG2
    builder = SpinHam1D(S=1/2)
    builder += 1.0, "X", "X"
    builder += 1.0, "Y", "Y"
    builder += 1.0, "Z", "Z"
    mpo = builder.build_mpo(L)

    print(f"\n{'='*60}")
    print(f"Test #36: Heisenberg L={L} Validation")
    print(f"{'='*60}")
    print(f"Running quimb DMRG2 (reference solution)...")

    dmrg = DMRG2(mpo, bond_dims=bond_dim)
    dmrg.solve(tol=1e-10, verbosity=1)
    E_serial = dmrg.energy
    print(f"\nQuimb DMRG2 energy: {E_serial:.15f}")

    # Step 3-4: Run A2DMRG
    print(f"\nRunning A2DMRG (np=1, one-site)...")
    energy, mps = a2dmrg_main(
        L=L,
        mpo=mpo,
        max_sweeps=30,
        bond_dim=bond_dim,
        tol=1e-10,
        comm=MPI.COMM_WORLD,
        dtype=np.float64,
        one_site=True,
        verbose=True
    )
    print(f"A2DMRG energy:      {energy:.15f}")

    # Step 5: Verify match
    diff = abs(energy - E_serial)
    print(f"\nDifference:         {diff:.3e}")
    print(f"Target:             < 1e-10")

    print(f"{'='*60}")
    if diff < 1e-10:
        print("✓ TEST PASSED: A2DMRG L=40 matches serial DMRG")
    else:
        print("✗ TEST FAILED: Energy mismatch")
    print(f"{'='*60}\n")

    assert diff < 1e-10, f"Energies differ by {diff}"


if __name__ == "__main__":
    test_l40_matches_quimb()
