"""Test #41: Josephson junction - Complex128 parallel (np=2,4) matches serial."""

import pytest
import numpy as np
from a2dmrg.mpi_compat import MPI
from quimb.tensor import DMRG2
from a2dmrg.dmrg import a2dmrg_main
from a2dmrg.tests.test_bose_hubbard import create_bose_hubbard_mpo
from a2dmrg.mps.format_conversion import convert_quimb_dmrg_to_a2dmrg_format


@pytest.mark.mpi
def test_complex128_parallel_np2():
    """Test #41 (part 1): A2DMRG complex128 with np=2 matches serial.

    This test validates that A2DMRG works correctly with complex128 dtype
    in parallel with 2 processors for Josephson junction problems.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # This test requires exactly 2 processors
    if size != 2:
        pytest.skip(f"Test requires np=2, but running with np={size}")

    L = 6
    bond_dim = 20
    nmax = 3  # 4 states per site: |0⟩, |1⟩, |2⟩, |3⟩

    # Step 1: Create Bose-Hubbard MPO with complex hopping
    t_mag = 1.0
    phase = np.pi / 4  # 45 degrees
    t = t_mag * np.exp(1j * phase)  # Complex hopping
    U = 2.0  # Interaction
    mu = 0.5  # Chemical potential

    mpo = create_bose_hubbard_mpo(L, t=t, U=U, mu=mu, nmax=nmax)

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Test #41: Josephson Parallel (np=2) - Complex128")
        print(f"{'='*60}")
        print(f"System: L={L}, nmax={nmax} (local dim = {nmax+1})")
        print(f"Parameters:")
        print(f"  Hopping t = {t_mag:.2f} * exp(i*{phase:.4f}) = {t:.6f}")
        print(f"  Interaction U = {U:.2f}")
        print(f"  Chemical potential μ = {mu:.2f}")
        print(f"  Bond dimension = {bond_dim}")
        print(f"  MPI processors = {size}")

    # Step 1: Get reference from serial DMRG (warm-start)
    dmrg_warmstart = DMRG2(mpo, bond_dims=bond_dim)
    dmrg_warmstart.solve(tol=1e-10, max_sweeps=10, verbosity=0)

    # Convert to A2DMRG format
    initial_mps = convert_quimb_dmrg_to_a2dmrg_format(dmrg_warmstart.state, bond_dim)

    # Get reference from fully converged serial DMRG
    dmrg = DMRG2(mpo, bond_dims=bond_dim)
    dmrg.solve(tol=1e-10, max_sweeps=15, verbosity=0)
    E_serial = dmrg.energy

    if rank == 0:
        print(f"\nSerial DMRG2 reference: {E_serial:.15f}")
        print(f"  Real part:  {E_serial.real:.15f}")
        print(f"  Imag part:  {E_serial.imag:.3e}")

    # Step 2: Run A2DMRG with np=2
    if rank == 0:
        print(f"\n--- Running A2DMRG (np=2, complex128) ---")

    energy, mps = a2dmrg_main(
        L=L,
        mpo=mpo,
        max_sweeps=15,
        bond_dim=bond_dim,
        tol=1e-10,
        comm=comm,  # Use parallel communicator
        dtype=np.complex128,
        one_site=True,
        initial_mps=initial_mps,
        verbose=False
    )

    if rank == 0:
        print(f"A2DMRG (np=2):      {energy:.15f}")
        print(f"  Real part:  {energy.real:.15f}")
        print(f"  Imag part:  {energy.imag:.3e}")

    # Step 4: Verify match
    diff = abs(energy - E_serial)

    if rank == 0:
        print(f"\nDifference: {diff:.3e}")
        print(f"Target:     1e-10")
        print(f"\n{'='*60}")
        if diff < 1e-10:
            print("✓ TEST PASSED: A2DMRG np=2 matches serial")
        else:
            print("✗ TEST FAILED: Energy mismatch")
        print(f"{'='*60}\n")

    assert diff < 1e-10, f"Energies differ by {diff} (np=2)"


@pytest.mark.mpi
def test_complex128_parallel_np4():
    """Test #41 (part 2): A2DMRG complex128 with np=4 matches serial.

    This test validates that A2DMRG works correctly with complex128 dtype
    in parallel with 4 processors for Josephson junction problems.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # This test requires exactly 4 processors
    if size != 4:
        pytest.skip(f"Test requires np=4, but running with np={size}")

    L = 8  # Larger system for np=4
    bond_dim = 20
    nmax = 3

    # Create Bose-Hubbard MPO
    t_mag = 1.0
    phase = np.pi / 4
    t = t_mag * np.exp(1j * phase)
    U = 2.0
    mu = 0.5

    mpo = create_bose_hubbard_mpo(L, t=t, U=U, mu=mu, nmax=nmax)

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Test #41: Josephson Parallel (np=4) - Complex128")
        print(f"{'='*60}")
        print(f"System: L={L}, nmax={nmax} (local dim = {nmax+1})")
        print(f"  MPI processors = {size}")

    # Get reference from serial DMRG
    dmrg_warmstart = DMRG2(mpo, bond_dims=bond_dim)
    dmrg_warmstart.solve(tol=1e-10, max_sweeps=10, verbosity=0)

    initial_mps = convert_quimb_dmrg_to_a2dmrg_format(dmrg_warmstart.state, bond_dim)

    dmrg = DMRG2(mpo, bond_dims=bond_dim)
    dmrg.solve(tol=1e-10, max_sweeps=15, verbosity=0)
    E_serial = dmrg.energy

    if rank == 0:
        print(f"\nSerial DMRG2 reference: {E_serial:.15f}")

    # Run A2DMRG with np=4
    if rank == 0:
        print(f"\n--- Running A2DMRG (np=4, complex128) ---")

    energy, mps = a2dmrg_main(
        L=L,
        mpo=mpo,
        max_sweeps=15,
        bond_dim=bond_dim,
        tol=1e-10,
        comm=comm,
        dtype=np.complex128,
        one_site=True,
        initial_mps=initial_mps,
        verbose=False
    )

    if rank == 0:
        print(f"A2DMRG (np=4):      {energy:.15f}")

    # Verify match
    diff = abs(energy - E_serial)

    if rank == 0:
        print(f"\nDifference: {diff:.3e}")
        print(f"Target:     1e-10")
        print(f"\n{'='*60}")
        if diff < 1e-10:
            print("✓ TEST PASSED: A2DMRG np=4 matches serial")
        else:
            print("✗ TEST FAILED: Energy mismatch")
        print(f"{'='*60}\n")

    assert diff < 1e-10, f"Energies differ by {diff} (np=4)"


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    size = comm.Get_size()

    if size == 2:
        test_complex128_parallel_np2()
    elif size == 4:
        test_complex128_parallel_np4()
    else:
        print(f"No test configured for np={size}. Use np=2 or 4.")
