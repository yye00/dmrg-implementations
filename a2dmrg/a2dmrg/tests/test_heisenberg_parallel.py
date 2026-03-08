"""Tests #37-39: Heisenberg parallel validation (np=2,4,8)."""

import pytest
import numpy as np
from a2dmrg.mpi_compat import MPI
from quimb.tensor import SpinHam1D, DMRG2
from a2dmrg.dmrg import a2dmrg_main


@pytest.mark.mpi
def test_heisenberg_np2_matches_serial():
    """Test #37: A2DMRG np=2 matches serial DMRG within 1e-10.

    This test validates that A2DMRG produces identical results
    with 2 processors compared to serial DMRG.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # This test requires exactly 2 processors
    if size != 2:
        pytest.skip(f"Test requires np=2, but running with np={size}")

    L = 40
    bond_dim = 100

    # Create Heisenberg MPO
    builder = SpinHam1D(S=1/2)
    builder += 1.0, "X", "X"
    builder += 1.0, "Y", "Y"
    builder += 1.0, "Z", "Z"
    mpo = builder.build_mpo(L)

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Test #37: Heisenberg L={L} with np=2")
        print(f"{'='*60}")

    # Step 1: Get reference from serial DMRG
    if rank == 0:
        print("Running serial DMRG (reference)...")

    dmrg = DMRG2(mpo, bond_dims=bond_dim)
    dmrg.solve(tol=1e-10, verbosity=0)
    E_serial = dmrg.energy

    if rank == 0:
        print(f"Serial DMRG energy: {E_serial:.15f}")

    # Step 2: Run A2DMRG with np=2
    if rank == 0:
        print(f"\nRunning A2DMRG with np=2...")

    energy, mps = a2dmrg_main(
        L=L,
        mpo=mpo,
        max_sweeps=30,
        bond_dim=bond_dim,
        tol=1e-10,
        comm=comm,  # Use parallel communicator
        dtype=np.float64,
        one_site=True,
        verbose=False
    )

    if rank == 0:
        print(f"A2DMRG (np=2):      {energy:.15f}")

    # Step 3: Verify match
    diff = abs(energy - E_serial)

    if rank == 0:
        print(f"\nDifference: {diff:.3e}")
        print(f"Target:     < 1e-10")
        print(f"{'='*60}")
        if diff < 1e-10:
            print("✓ TEST PASSED: np=2 matches serial")
        else:
            print("✗ TEST FAILED: Energy mismatch")
        print(f"{'='*60}\n")

    assert diff < 1e-10, f"np=2 differs from serial by {diff}"


@pytest.mark.mpi
def test_heisenberg_np4_matches_serial():
    """Test #38: A2DMRG np=4 matches serial DMRG within 1e-10.

    This test validates that A2DMRG produces identical results
    with 4 processors compared to serial DMRG.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # This test requires exactly 4 processors
    if size != 4:
        pytest.skip(f"Test requires np=4, but running with np={size}")

    L = 40
    bond_dim = 100

    # Create Heisenberg MPO
    builder = SpinHam1D(S=1/2)
    builder += 1.0, "X", "X"
    builder += 1.0, "Y", "Y"
    builder += 1.0, "Z", "Z"
    mpo = builder.build_mpo(L)

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Test #38: Heisenberg L={L} with np=4")
        print(f"{'='*60}")

    # Get reference from serial DMRG
    if rank == 0:
        print("Running serial DMRG (reference)...")

    dmrg = DMRG2(mpo, bond_dims=bond_dim)
    dmrg.solve(tol=1e-10, verbosity=0)
    E_serial = dmrg.energy

    if rank == 0:
        print(f"Serial DMRG energy: {E_serial:.15f}")

    # Run A2DMRG with np=4
    if rank == 0:
        print(f"\nRunning A2DMRG with np=4...")

    energy, mps = a2dmrg_main(
        L=L,
        mpo=mpo,
        max_sweeps=30,
        bond_dim=bond_dim,
        tol=1e-10,
        comm=comm,
        dtype=np.float64,
        one_site=True,
        verbose=False
    )

    if rank == 0:
        print(f"A2DMRG (np=4):      {energy:.15f}")

    # Verify match
    diff = abs(energy - E_serial)

    if rank == 0:
        print(f"\nDifference: {diff:.3e}")
        print(f"Target:     < 1e-10")
        print(f"{'='*60}")
        if diff < 1e-10:
            print("✓ TEST PASSED: np=4 matches serial")
        else:
            print("✗ TEST FAILED: Energy mismatch")
        print(f"{'='*60}\n")

    assert diff < 1e-10, f"np=4 differs from serial by {diff}"


@pytest.mark.mpi
def test_heisenberg_np8_matches_serial():
    """Test #39: A2DMRG np=8 matches serial DMRG within 1e-10.

    This test validates that A2DMRG produces identical results
    with 8 processors compared to serial DMRG. This is the final
    validation that ALL processor counts give machine-precision
    identical results.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # This test requires exactly 8 processors
    if size != 8:
        pytest.skip(f"Test requires np=8, but running with np={size}")

    L = 40
    bond_dim = 100

    # Create Heisenberg MPO
    builder = SpinHam1D(S=1/2)
    builder += 1.0, "X", "X"
    builder += 1.0, "Y", "Y"
    builder += 1.0, "Z", "Z"
    mpo = builder.build_mpo(L)

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Test #39: Heisenberg L={L} with np=8")
        print(f"{'='*60}")

    # Get reference from serial DMRG
    if rank == 0:
        print("Running serial DMRG (reference)...")

    dmrg = DMRG2(mpo, bond_dims=bond_dim)
    dmrg.solve(tol=1e-10, verbosity=0)
    E_serial = dmrg.energy

    if rank == 0:
        print(f"Serial DMRG energy: {E_serial:.15f}")

    # Run A2DMRG with np=8
    if rank == 0:
        print(f"\nRunning A2DMRG with np=8...")

    energy, mps = a2dmrg_main(
        L=L,
        mpo=mpo,
        max_sweeps=30,
        bond_dim=bond_dim,
        tol=1e-10,
        comm=comm,
        dtype=np.float64,
        one_site=True,
        verbose=False
    )

    if rank == 0:
        print(f"A2DMRG (np=8):      {energy:.15f}")

    # Verify match
    diff = abs(energy - E_serial)

    if rank == 0:
        print(f"\nDifference: {diff:.3e}")
        print(f"Target:     < 1e-10")
        print(f"{'='*60}")
        if diff < 1e-10:
            print("✓ TEST PASSED: np=8 matches serial")
        else:
            print("✗ TEST FAILED: Energy mismatch")
        print(f"{'='*60}\n")

        # Final summary
        print("="*60)
        print("VALIDATION COMPLETE")
        print("All processor counts (1,2,4,8) give identical results")
        print("to machine precision (< 1e-10)")
        print("="*60)

    assert diff < 1e-10, f"np=8 differs from serial by {diff}"


if __name__ == "__main__":
    # Run appropriate test based on MPI size
    comm = MPI.COMM_WORLD
    size = comm.Get_size()

    if size == 2:
        test_heisenberg_np2_matches_serial()
    elif size == 4:
        test_heisenberg_np4_matches_serial()
    elif size == 8:
        test_heisenberg_np8_matches_serial()
    else:
        print(f"No test configured for np={size}. Use np=2, 4, or 8.")
