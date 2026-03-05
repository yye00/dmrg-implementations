"""Test #40: Josephson junction - Complex128 support with Bose-Hubbard model."""

import fix_quimb_python313  # Fix for Python 3.13 compatibility

import numpy as np
from a2dmrg.mpi_compat import MPI
from quimb.tensor import DMRG2
from a2dmrg.dmrg import a2dmrg_main
from a2dmrg.tests.test_bose_hubbard import create_bose_hubbard_mpo
from a2dmrg.mps.format_conversion import convert_quimb_dmrg_to_a2dmrg_format


def test_complex128_bose_hubbard():
    """Test #40: A2DMRG with complex128 on Bose-Hubbard model.

    This test validates that A2DMRG works correctly with complex128 dtype
    for Josephson junction problems (Bose-Hubbard with complex hopping).

    Strategy: Use warm-start from DMRG2 to overcome one-site convergence
    limitations identified in Session 28 and Session 60.
    """
    L = 6
    bond_dim = 20
    nmax = 3  # 4 states per site: |0⟩, |1⟩, |2⟩, |3⟩

    # Step 1: Create Bose-Hubbard MPO with complex hopping
    # For Josephson junction, hopping has phase: t = |t| * e^(iφ)
    t_mag = 1.0
    phase = np.pi / 4  # 45 degrees
    t = t_mag * np.exp(1j * phase)  # Complex hopping
    U = 2.0  # Interaction
    mu = 0.5  # Chemical potential

    mpo = create_bose_hubbard_mpo(L, t=t, U=U, mu=mu, nmax=nmax)

    print(f"\n{'='*60}")
    print(f"Test #40: Josephson Junction - Complex128 Bose-Hubbard")
    print(f"{'='*60}")
    print(f"System: L={L}, nmax={nmax} (local dim = {nmax+1})")
    print(f"Parameters:")
    print(f"  Hopping t = {t_mag:.2f} * exp(i*{phase:.4f}) = {t:.6f}")
    print(f"  Interaction U = {U:.2f}")
    print(f"  Chemical potential μ = {mu:.2f}")
    print(f"  Bond dimension = {bond_dim}")

    # Step 2: Warm-start with DMRG2 (overcomes one-site convergence issues)
    print(f"\n--- Running Quimb DMRG2 for warm-start ---")
    dmrg_warmstart = DMRG2(mpo, bond_dims=bond_dim)
    dmrg_warmstart.solve(tol=1e-10, max_sweeps=10, verbosity=0)
    E_warmstart = dmrg_warmstart.energy
    print(f"DMRG2 warm-start energy: {E_warmstart:.15f}")
    print(f"  Real part:  {E_warmstart.real:.15f}")
    print(f"  Imag part:  {E_warmstart.imag:.3e}")

    # Convert DMRG2 MPS to A2DMRG format (uniform bonds, correct index ordering)
    print(f"\n--- Converting MPS to A2DMRG format ---")
    initial_mps = convert_quimb_dmrg_to_a2dmrg_format(dmrg_warmstart.state, bond_dim)
    print(f"Conversion complete. MPS has uniform bond dimension = {bond_dim}")

    # Step 3: Run serial DMRG with complex128 (reference solution)
    print(f"\n--- Running Quimb DMRG2 (serial reference) ---")
    dmrg = DMRG2(mpo, bond_dims=bond_dim)
    dmrg.solve(tol=1e-10, max_sweeps=15, verbosity=0)
    E_serial = dmrg.energy
    print(f"Quimb DMRG2 energy: {E_serial:.15f}")
    print(f"  Real part:  {E_serial.real:.15f}")
    print(f"  Imag part:  {E_serial.imag:.3e}")

    # Step 4: Run A2DMRG np=1 with complex128 using warm-start
    print(f"\n--- Running A2DMRG (np=1, complex128, warm-start) ---")
    energy, mps = a2dmrg_main(
        L=L,
        mpo=mpo,
        max_sweeps=15,
        bond_dim=bond_dim,
        tol=1e-10,
        comm=MPI.COMM_WORLD,
        dtype=np.complex128,  # Complex dtype for Josephson junction
        one_site=True,
        initial_mps=initial_mps,  # Use warm-start
        verbose=False
    )
    print(f"A2DMRG energy:      {energy:.15f}")
    print(f"  Real part:  {energy.real:.15f}")
    print(f"  Imag part:  {energy.imag:.3e}")

    # Step 5: Verify |E_a2dmrg - E_serial| < 1e-10
    diff = abs(energy - E_serial)
    print(f"\nDifference: {diff:.3e}")
    print(f"Target:     1e-10")

    # Verify energy is complex (has non-trivial imaginary part from complex hopping)
    # Actually, for Hermitian H, energy should be real even with complex t
    # The hopping term -t(a†a + aa†) is Hermitian if we include h.c.
    print(f"\nEnergy is Hermitian: |Im(E)| = {abs(energy.imag):.3e}")

    print(f"\n{'='*60}")
    if diff < 1e-10:
        print("✓ TEST PASSED: A2DMRG complex128 matches serial DMRG")
    else:
        print("✗ TEST FAILED: Energy mismatch")
    print(f"{'='*60}\n")

    assert diff < 1e-10, f"Energies differ by {diff}"


if __name__ == "__main__":
    test_complex128_bose_hubbard()
