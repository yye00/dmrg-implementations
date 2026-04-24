#!/usr/bin/env python3
import sys
if "pytest" in sys.modules:
    import pytest
    pytest.skip("dev script (not a unit test)", allow_module_level=True)

"""Test #35: Validate A2DMRG against quimb serial DMRG for L=20."""

# Apply numba fix for Python 3.13
import sys
sys.path.insert(0, '/home/captain/clawd/work/dmrg-implementations/a2dmrg')
import fix_quimb_python313  # Monkeypatches numba before quimb import

import numpy as np
from a2dmrg.mpi_compat import MPI, HAS_MPI
import quimb.tensor as qtn

from a2dmrg.dmrg import a2dmrg_main


def create_heisenberg_mpo(L, J=1.0, cyclic=False):
    """Create Heisenberg chain MPO using quimb."""
    from quimb.tensor import SpinHam1D

    builder = SpinHam1D(S=1/2)
    builder += J, 'X', 'X'
    builder += J, 'Y', 'Y'
    builder += J, 'Z', 'Z'

    mpo = builder.build_mpo(L)
    return mpo

def test_l20_validation():
    """Compare A2DMRG with quimb DMRG for L=20, bond_dim=50."""
    L = 20
    bond_dim = 50
    J = 1.0

    print(f"\n{'='*70}")
    print(f"Test #35: L=20 Validation Against Quimb Serial DMRG")
    print(f"{'='*70}\n")

    # Step 1-2: Run quimb serial DMRG
    print(f"Step 1-2: Running quimb DMRG2 (L={L}, bond_dim={bond_dim})...")

    # Create quimb Hamiltonian
    H = qtn.MPO_ham_heis(L, cyclic=False)

    # Initialize random MPS
    psi0 = qtn.MPS_rand_state(L, bond_dim=bond_dim, dtype=float)

    # Run quimb DMRG
    dmrg = qtn.DMRG2(H, bond_dims=bond_dim)
    dmrg.solve(tol=1e-12, max_sweeps=50, verbosity=1)

    E_serial = dmrg.energy
    print(f"  Quimb energy: {E_serial:.15f}")

    # Step 3-4: Run A2DMRG
    print(f"\nStep 3-4: Running A2DMRG (L={L}, bond_dim={bond_dim})...")

    # Create A2DMRG Hamiltonian
    mpo = create_heisenberg_mpo(L, J=J, cyclic=False)

    # Run A2DMRG with np=1 (serial mode)
    energy, mps = a2dmrg_main(
        L=L,
        mpo=mpo,
        max_sweeps=50,
        bond_dim=bond_dim,
        tol=1e-12,
        comm=MPI.COMM_SELF,
        dtype=np.float64,
        one_site=True,
        verbose=True
    )

    E_a2dmrg = energy
    print(f"  A2DMRG energy: {E_a2dmrg:.15f}")

    # Step 5: Verify agreement
    print(f"\nStep 5: Verification")
    print(f"  E_serial:   {E_serial:.15f}")
    print(f"  E_a2dmrg:   {E_a2dmrg:.15f}")
    print(f"  Difference: {abs(E_a2dmrg - E_serial):.3e}")

    tolerance = 1e-10
    if abs(E_a2dmrg - E_serial) < tolerance:
        print(f"\n✅ PASS: Energies match within tolerance {tolerance:.0e}")
        return True
    else:
        print(f"\n❌ FAIL: Energies differ by more than {tolerance:.0e}")
        return False

if __name__ == "__main__":
    success = test_l20_validation()
    exit(0 if success else 1)
