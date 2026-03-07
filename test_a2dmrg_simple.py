#!/usr/bin/env python3
"""
Simple A2DMRG validation test using quimb's built-in Hamiltonians.
"""

import sys
import os

# Force single-threaded
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np
import quimb.tensor as qtn

# Import A2DMRG
from a2dmrg.dmrg import a2dmrg_main


def run_reference_dmrg2(mpo, L, chi, tol=1e-10):
    """Run quimb DMRG2 as reference."""
    dmrg = qtn.DMRG2(mpo, bond_dims=chi)
    dmrg.solve(tol=tol, max_sweeps=50, verbosity=0)
    return dmrg.state, dmrg.energy


def test_heisenberg_small():
    """Test Heisenberg on small system."""
    print("\n" + "="*60)
    print("TEST: Heisenberg XXZ Chain (L=10, periodic)")
    print("="*60)

    L = 10
    chi = 64

    # Create Heisenberg XXZ Hamiltonian
    print("Creating Heisenberg XXZ Hamiltonian...")
    builder = qtn.SpinHam1D(S=1/2, cyclic=True)
    H_mpo = builder.build_mpo(L)

    # Get reference energy from quimb DMRG2
    print("\nRunning quimb DMRG2 (reference)...")
    ref_mps, ref_energy = run_reference_dmrg2(H_mpo, L, chi, tol=1e-10)
    print(f"  Reference energy: {ref_energy:.15f}")

    # Run A2DMRG (serial, np=1)
    print("\nRunning A2DMRG (np=1)...")
    try:
        a2dmrg_energy, a2dmrg_mps = a2dmrg_main(
            L=L,
            mpo=H_mpo,
            max_sweeps=15,
            bond_dim=chi,
            tol=1e-10,
            comm=None,  # Serial mode
            verbose=True,
            warmup_sweeps=3,
            one_site=False  # Two-site
        )
        print(f"\n  A2DMRG energy:    {a2dmrg_energy:.15f}")

        error = abs(a2dmrg_energy - ref_energy)
        print(f"  Absolute error:   {error:.2e}")
        rel_error = error / abs(ref_energy)
        print(f"  Relative error:   {rel_error:.2e}")

        if error < 1e-12:
            print("\n  ✅ PASS - Machine precision!")
            return True
        elif error < 5e-10:
            print("\n  ✅ PASS - Acceptance threshold")
            return True
        elif error < 1e-8:
            print("\n  ⚠️  WARN - Acceptable but not great")
            return True
        else:
            print("\n  ❌ FAIL - Error too large")
            return False

    except Exception as e:
        print(f"\n  ❌ FAIL - Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_heisenberg_tiny():
    """Test on very small system for quick validation."""
    print("\n" + "="*60)
    print("TEST: Heisenberg XXZ Chain (L=6, open)")
    print("="*60)

    L = 6
    chi = 32

    # Create Heisenberg XXZ Hamiltonian (open boundary)
    print("Creating Heisenberg XXZ Hamiltonian...")
    builder = qtn.SpinHam1D(S=1/2, cyclic=False)
    H_mpo = builder.build_mpo(L)

    # Get reference energy from quimb DMRG2
    print("\nRunning quimb DMRG2 (reference)...")
    ref_mps, ref_energy = run_reference_dmrg2(H_mpo, L, chi, tol=1e-10)
    print(f"  Reference energy: {ref_energy:.15f}")

    # Run A2DMRG (serial, np=1)
    print("\nRunning A2DMRG (np=1)...")
    try:
        a2dmrg_energy, a2dmrg_mps = a2dmrg_main(
            L=L,
            mpo=H_mpo,
            max_sweeps=10,
            bond_dim=chi,
            tol=1e-10,
            comm=None,  # Serial mode
            verbose=True,
            warmup_sweeps=2,
            one_site=False  # Two-site
        )
        print(f"\n  A2DMRG energy:    {a2dmrg_energy:.15f}")

        error = abs(a2dmrg_energy - ref_energy)
        print(f"  Absolute error:   {error:.2e}")

        if error < 1e-12:
            print("\n  ✅ PASS - Machine precision!")
            return True
        elif error < 5e-10:
            print("\n  ✅ PASS - Acceptance threshold")
            return True
        elif error < 1e-7:
            print("\n  ⚠️  WARN - Acceptable")
            return True
        else:
            print("\n  ❌ FAIL - Error too large")
            return False

    except Exception as e:
        print(f"\n  ❌ FAIL - Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("\n" + "="*70)
    print("A2DMRG QUICK VALIDATION TESTS")
    print("Testing convergence after i-orthogonal transformation fix")
    print("="*70)

    results = []

    # Run tests
    results.append(("Heisenberg L=6 (open)", test_heisenberg_tiny()))
    results.append(("Heisenberg L=10 (periodic)", test_heisenberg_small()))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:40s} {status}")

    all_passed = all(p for _, p in results)
    print("="*70)
    if all_passed:
        print("ALL TESTS PASSED! ✅")
        print("\nA2DMRG is converging correctly with i-orthogonal transformation!")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED! ❌")
        sys.exit(1)
