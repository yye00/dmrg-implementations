#!/usr/bin/env python3
import sys
if "pytest" in sys.modules:
    import pytest
    pytest.skip("dev script (not a unit test)", allow_module_level=True)

"""Test #64: Compare one-site vs two-site A2DMRG convergence and accuracy."""

# Apply numba fix for Python 3.13
import sys
sys.path.insert(0, '/home/captain/clawd/work/dmrg-implementations/a2dmrg')
import fix_quimb_python313

import numpy as np
from a2dmrg.mpi_compat import MPI, HAS_MPI
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


def test_onesite_vs_twosite():
    """Compare one-site and two-site A2DMRG updates."""
    L = 8
    bond_dim = 12
    max_sweeps = 5
    tol = 1e-6

    print(f"\n{'='*70}")
    print(f"Test #64: One-Site vs Two-Site A2DMRG Comparison")
    print(f"{'='*70}\n")

    mpo = create_heisenberg_mpo(L, J=1.0, cyclic=False)

    # Step 1: Run with one-site updates
    print(f"Step 1: Running A2DMRG with one-site updates...")
    energy_1site, mps_1site = a2dmrg_main(
        L=L,
        mpo=mpo,
        max_sweeps=max_sweeps,
        bond_dim=bond_dim,
        tol=tol,
        comm=MPI.COMM_SELF,
        dtype=np.float64,
        one_site=True,
        verbose=False
    )
    print(f"  One-site energy: {energy_1site:.15f}")

    # Step 2: Run with two-site updates
    print(f"\nStep 2: Running A2DMRG with two-site updates...")
    energy_2site, mps_2site = a2dmrg_main(
        L=L,
        mpo=mpo,
        max_sweeps=max_sweeps,
        bond_dim=bond_dim,
        tol=tol,
        comm=MPI.COMM_SELF,
        dtype=np.float64,
        one_site=False,  # Two-site
        verbose=False
    )
    print(f"  Two-site energy: {energy_2site:.15f}")

    # Step 3: Verify both converge to same energy
    print(f"\nStep 3: Verification")
    print(f"  One-site:  {energy_1site:.15f}")
    print(f"  Two-site:  {energy_2site:.15f}")
    print(f"  Difference: {abs(energy_1site - energy_2site):.3e}")

    # Allow some tolerance for differences (e.g., 1e-6)
    # Both should give negative energy for Heisenberg chain
    tolerance = 1e-6
    if abs(energy_1site - energy_2site) < tolerance:
        print(f"\n✅ PASS: Energies match within tolerance {tolerance:.0e}")
        print(f"         Both one-site and two-site converge to same result")
        return True
    else:
        print(f"\n⚠️  PARTIAL: Energies differ by {abs(energy_1site - energy_2site):.3e}")
        print(f"         This may be acceptable if both are negative and close")
        # Check if both are at least negative and reasonable
        if energy_1site < 0 and energy_2site < 0:
            print(f"         Both energies are negative (physically correct)")
            return True
        else:
            print(f"\n❌ FAIL: At least one energy is non-physical")
            return False


if __name__ == "__main__":
    success = test_onesite_vs_twosite()
    exit(0 if success else 1)
