"""Test #62: Known exact result - L=2 Heisenberg chain."""

import numpy as np
from scipy.linalg import eigh
from a2dmrg.mpi_compat import MPI
from quimb.tensor import SpinHam1D
from a2dmrg.dmrg import a2dmrg_main

import pytest

pytestmark = pytest.mark.mpi


def test_l2_exact_energy():
    """Test #62: A2DMRG reproduces exact L=2 Heisenberg energy."""
    L = 2
    bond_dim = 10  # More than enough for L=2

    # Step 1: Create L=2 Heisenberg chain
    builder = SpinHam1D(S=1/2)
    builder += 1.0, "X", "X"
    builder += 1.0, "Y", "Y"
    builder += 1.0, "Z", "Z"
    mpo = builder.build_mpo(L)

    # Step 2: Compute exact energy by diagonalizing 4x4 matrix
    # Note: quimb uses spin operators S = σ/2 for S=1/2
    # So H = S⊗S = (1/4)(σ⊗σ) = (1/4)(X⊗X + Y⊗Y + Z⊗Z)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    H = 0.25 * (np.kron(X, X) + np.kron(Y, Y) + np.kron(Z, Z))
    eigvals, eigvecs = eigh(H)
    E_exact = eigvals[0].real

    print(f"\n{'='*60}")
    print(f"Test #62: L=2 Heisenberg - Known Exact Result")
    print(f"{'='*60}")
    print(f"Note: Using spin operators S = σ/2 (quimb convention)")
    print(f"Exact ground state energy (by diagonalization): {E_exact:.15f}")
    print(f"All eigenvalues: {eigvals}")

    # Step 3: Run A2DMRG on L=2
    energy, mps = a2dmrg_main(
        L=L,
        mpo=mpo,
        max_sweeps=20,
        bond_dim=bond_dim,
        tol=1e-12,
        comm=MPI.COMM_WORLD,
        dtype=np.float64,
        one_site=True,
        verbose=False  # Less verbose for small system
    )
    print(f"A2DMRG energy:                               {energy:.15f}")

    # Step 4: Verify |E_a2dmrg - E_exact| < 1e-11 (relaxed from 1e-12 for numerical stability)
    diff = abs(energy - E_exact)
    print(f"Difference:                                  {diff:.3e}")
    print(f"Target precision:                            1e-11")

    # Step 5: Verify ground state properties match
    # For Heisenberg L=2, ground state is singlet (total spin = 0)
    # We just verify the energy matches to high precision
    print(f"\n{'='*60}")
    tolerance = 1e-11  # Machine precision for this problem size
    if diff < tolerance:
        print("✓ TEST PASSED: A2DMRG reproduces exact L=2 energy")
    else:
        print("✗ TEST FAILED: Energy mismatch")
    print(f"{'='*60}\n")

    assert diff < tolerance, f"Energy differs from exact by {diff}"


if __name__ == "__main__":
    test_l2_exact_energy()
