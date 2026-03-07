#!/usr/bin/env python3
"""
Basic A2DMRG convergence test - just verify it runs and converges.
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


def test_heisenberg():
    """Test A2DMRG on Heisenberg chain."""
    print("\n" + "="*70)
    print("TEST: Heisenberg Spin Chain (L=8, periodic)")
    print("="*70)

    L = 8
    chi = 64

    # Create Heisenberg Hamiltonian
    print("\nCreating Heisenberg Hamiltonian...")
    builder = qtn.SpinHam1D(S=1/2, cyclic=True)
    H_mpo = builder.build_mpo(L)
    print(f"  MPO created for L={L} sites")

    # Compute exact energy for comparison (small system)
    print("\nComputing exact ground state energy (exact diagonalization)...")
    H_op = builder.build_local_ham(L)
    exact_energy = qtn.groundenergy(H_op)
    print(f"  Exact energy: {exact_energy:.15f}")

    # Run A2DMRG
    print("\nRunning A2DMRG (np=1, two-site)...")
    print("-" * 70)
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
        print("-" * 70)
        print(f"\nA2DMRG energy:    {a2dmrg_energy:.15f}")
        print(f"Exact energy:     {exact_energy:.15f}")

        error = abs(a2dmrg_energy - exact_energy)
        print(f"Absolute error:   {error:.2e}")

        if error < 1e-12:
            print("\n✅ EXCELLENT - Machine precision!")
            return True
        elif error < 1e-10:
            print("\n✅ PASS - Very good accuracy")
            return True
        elif error < 1e-8:
            print("\n✅ PASS - Good accuracy")
            return True
        elif error < 1e-6:
            print("\n⚠️  WARN - Acceptable but not great")
            return True
        else:
            print("\n❌ FAIL - Error too large")
            return False

    except Exception as e:
        print(f"\n❌ FAIL - Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_heisenberg_open():
    """Test A2DMRG on open boundary Heisenberg."""
    print("\n" + "="*70)
    print("TEST: Heisenberg Spin Chain (L=10, open)")
    print("="*70)

    L = 10
    chi = 64

    # Create Heisenberg Hamiltonian (open)
    print("\nCreating Heisenberg Hamiltonian (open boundary)...")
    builder = qtn.SpinHam1D(S=1/2, cyclic=False)
    H_mpo = builder.build_mpo(L)
    print(f"  MPO created for L={L} sites")

    # Compute exact energy
    print("\nComputing exact ground state energy (exact diagonalization)...")
    H_op = builder.build_local_ham(L)
    exact_energy = qtn.groundenergy(H_op)
    print(f"  Exact energy: {exact_energy:.15f}")

    # Run A2DMRG
    print("\nRunning A2DMRG (np=1, two-site)...")
    print("-" * 70)
    try:
        a2dmrg_energy, a2dmrg_mps = a2dmrg_main(
            L=L,
            mpo=H_mpo,
            max_sweeps=10,
            bond_dim=chi,
            tol=1e-10,
            comm=None,
            verbose=True,
            warmup_sweeps=2,
            one_site=False
        )
        print("-" * 70)
        print(f"\nA2DMRG energy:    {a2dmrg_energy:.15f}")
        print(f"Exact energy:     {exact_energy:.15f}")

        error = abs(a2dmrg_energy - exact_energy)
        print(f"Absolute error:   {error:.2e}")

        if error < 1e-12:
            print("\n✅ EXCELLENT - Machine precision!")
            return True
        elif error < 1e-10:
            print("\n✅ PASS - Very good accuracy")
            return True
        elif error < 1e-8:
            print("\n✅ PASS - Good accuracy")
            return True
        elif error < 1e-6:
            print("\n⚠️  WARN - Acceptable")
            return True
        else:
            print("\n❌ FAIL - Error too large")
            return False

    except Exception as e:
        print(f"\n❌ FAIL - Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("\n" + "="*70)
    print("A2DMRG BASIC CONVERGENCE TESTS")
    print("Testing i-orthogonal transformation implementation")
    print("="*70)

    results = []

    # Run tests
    results.append(("Heisenberg L=8 (periodic)", test_heisenberg()))
    results.append(("Heisenberg L=10 (open)", test_heisenberg_open()))

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
        print("\n🎉 ALL TESTS PASSED!")
        print("\nA2DMRG with i-orthogonal transformation is working correctly!")
        print("The algorithm converges to the correct ground state energy.")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED")
        sys.exit(1)
