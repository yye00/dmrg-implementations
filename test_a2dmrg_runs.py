#!/usr/bin/env python3
"""
Simple A2DMRG smoke test - just verify it runs without crashing.
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


def test_heisenberg_small():
    """Test A2DMRG on small Heisenberg chain."""
    print("\n" + "="*70)
    print("TEST 1: Heisenberg Spin Chain (L=8, periodic)")
    print("="*70)

    L = 8
    chi = 32

    # Create Heisenberg Hamiltonian
    print("\nCreating Heisenberg Hamiltonian...")
    builder = qtn.SpinHam1D(S=1/2, cyclic=True)
    H_mpo = builder.build_mpo(L)
    print(f"  ✓ MPO created for L={L} sites")

    # Run A2DMRG
    print("\nRunning A2DMRG...")
    print("-" * 70)
    try:
        a2dmrg_energy, a2dmrg_mps = a2dmrg_main(
            L=L,
            mpo=H_mpo,
            max_sweeps=8,
            bond_dim=chi,
            tol=1e-9,
            comm=None,  # Serial mode
            verbose=True,
            warmup_sweeps=2,
            one_site=False  # Two-site
        )
        print("-" * 70)
        print(f"\n  Final energy: {a2dmrg_energy:.12f}")

        # Expected energy per site for Heisenberg XXZ is around -0.44 to -0.47
        energy_per_site = a2dmrg_energy / L
        print(f"  Energy/site:  {energy_per_site:.6f}")

        # Sanity check: energy should be negative and reasonable
        if -0.6 < energy_per_site < -0.3:
            print("\n  ✅ PASS - Energy is in expected range")
            return True
        else:
            print(f"\n  ❌ FAIL - Energy per site {energy_per_site:.4f} outside expected range [-0.6, -0.3]")
            return False

    except Exception as e:
        print(f"\n  ❌ FAIL - Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_heisenberg_open():
    """Test A2DMRG on open boundary Heisenberg."""
    print("\n" + "="*70)
    print("TEST 2: Heisenberg Spin Chain (L=10, open)")
    print("="*70)

    L = 10
    chi = 48

    # Create Heisenberg Hamiltonian (open)
    print("\nCreating Heisenberg Hamiltonian (open boundary)...")
    builder = qtn.SpinHam1D(S=1/2, cyclic=False)
    H_mpo = builder.build_mpo(L)
    print(f"  ✓ MPO created for L={L} sites")

    # Run A2DMRG
    print("\nRunning A2DMRG...")
    print("-" * 70)
    try:
        a2dmrg_energy, a2dmrg_mps = a2dmrg_main(
            L=L,
            mpo=H_mpo,
            max_sweeps=8,
            bond_dim=chi,
            tol=1e-9,
            comm=None,
            verbose=True,
            warmup_sweeps=2,
            one_site=False
        )
        print("-" * 70)
        print(f"\n  Final energy: {a2dmrg_energy:.12f}")

        # Check energy is reasonable
        energy_per_site = a2dmrg_energy / L
        print(f"  Energy/site:  {energy_per_site:.6f}")

        if -0.6 < energy_per_site < -0.3:
            print("\n  ✅ PASS - Energy is in expected range")
            return True
        else:
            print(f"\n  ❌ FAIL - Energy per site {energy_per_site:.4f} outside expected range")
            return False

    except Exception as e:
        print(f"\n  ❌ FAIL - Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_heisenberg_tiny():
    """Test A2DMRG on very small system."""
    print("\n" + "="*70)
    print("TEST 3: Heisenberg Spin Chain (L=6, periodic, one-site)")
    print("="*70)

    L = 6
    chi = 24

    # Create Heisenberg Hamiltonian
    print("\nCreating Heisenberg Hamiltonian...")
    builder = qtn.SpinHam1D(S=1/2, cyclic=True)
    H_mpo = builder.build_mpo(L)
    print(f"  ✓ MPO created for L={L} sites")

    # Run A2DMRG with ONE-SITE updates
    print("\nRunning A2DMRG (one-site updates)...")
    print("-" * 70)
    try:
        a2dmrg_energy, a2dmrg_mps = a2dmrg_main(
            L=L,
            mpo=H_mpo,
            max_sweeps=8,
            bond_dim=chi,
            tol=1e-9,
            comm=None,
            verbose=True,
            warmup_sweeps=2,
            one_site=True  # One-site updates
        )
        print("-" * 70)
        print(f"\n  Final energy: {a2dmrg_energy:.12f}")

        energy_per_site = a2dmrg_energy / L
        print(f"  Energy/site:  {energy_per_site:.6f}")

        if -0.6 < energy_per_site < -0.3:
            print("\n  ✅ PASS - Energy is in expected range")
            return True
        else:
            print(f"\n  ❌ FAIL - Energy per site {energy_per_site:.4f} outside expected range")
            return False

    except Exception as e:
        print(f"\n  ❌ FAIL - Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("\n" + "="*70)
    print("A2DMRG SMOKE TESTS")
    print("Verifying A2DMRG runs and converges after i-orthogonal fix")
    print("="*70)

    results = []

    # Run tests
    results.append(("Heisenberg L=8 (periodic, 2-site)", test_heisenberg_small()))
    results.append(("Heisenberg L=10 (open, 2-site)", test_heisenberg_open()))
    results.append(("Heisenberg L=6 (periodic, 1-site)", test_heisenberg_tiny()))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:45s} {status}")

    all_passed = all(p for _, p in results)
    print("="*70)
    if all_passed:
        print("\n🎉 ALL SMOKE TESTS PASSED!")
        print("\nA2DMRG with i-orthogonal transformation is working!")
        print("- Algorithm runs without crashing")
        print("- Converges to reasonable ground state energies")
        print("- Both one-site and two-site updates work")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED")
        sys.exit(1)
