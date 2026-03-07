#!/usr/bin/env python3
"""
Quick test of A2DMRG on small systems to verify convergence after i-orthogonal fix.
"""

import sys
import os
from pathlib import Path

# Force single-threaded
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# Add paths
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / 'benchmarks'))

import numpy as np
import quimb.tensor as qtn
from load_mps_mpo import load_mpo_from_binary, convert_to_quimb_mpo

def load_mpo(filepath):
    """Load MPO from binary file."""
    tensors = load_mpo_from_binary(filepath)
    return convert_to_quimb_mpo(tensors)

# Import A2DMRG
from a2dmrg.dmrg import a2dmrg_main

def run_reference_dmrg2(mpo, L, chi, tol=1e-11):
    """Run quimb DMRG2 as reference."""
    dmrg = qtn.DMRG2(mpo, bond_dims=chi)
    dmrg.solve(tol=tol, max_sweeps=40, verbosity=0)
    return dmrg.state, dmrg.energy


def test_heisenberg_L8():
    """Test Heisenberg L=8."""
    print("\n" + "="*60)
    print("TEST: Heisenberg L=8")
    print("="*60)

    L = 8
    chi = 100

    # Load MPO
    mpo = load_mpo('benchmarks/benchmark_data/heisenberg_L8_mpo.bin')

    # Get reference energy from quimb DMRG2
    print("Running quimb DMRG2 (reference)...")
    ref_mps, ref_energy = run_reference_dmrg2(mpo, L, chi, tol=1e-11)
    print(f"  Reference energy: {ref_energy:.15f}")

    # Run A2DMRG (serial, np=1)
    print("\nRunning A2DMRG (np=1)...")
    try:
        a2dmrg_energy, a2dmrg_mps = a2dmrg_main(
            L=L,
            mpo=mpo,
            max_sweeps=20,
            bond_dim=chi,
            tol=1e-11,
            comm=None,  # Serial mode
            verbose=False,
            warmup_sweeps=2,
            one_site=False  # Two-site
        )
        print(f"  A2DMRG energy:    {a2dmrg_energy:.15f}")

        error = abs(a2dmrg_energy - ref_energy)
        print(f"  Error: {error:.2e}")

        if error < 1e-12:
            print("  ✅ PASS - Machine precision!")
        elif error < 5e-10:
            print("  ✅ PASS - Acceptance threshold")
        else:
            print("  ❌ FAIL - Error too large")
            return False

    except Exception as e:
        print(f"  ❌ FAIL - Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_heisenberg_L12():
    """Test Heisenberg L=12."""
    print("\n" + "="*60)
    print("TEST: Heisenberg L=12")
    print("="*60)

    L = 12
    chi = 100

    # Load MPO
    mpo = load_mpo('benchmarks/benchmark_data/heisenberg_L12_mpo.bin')

    # Get reference energy from quimb DMRG2
    print("Running quimb DMRG2 (reference)...")
    ref_mps, ref_energy = run_reference_dmrg2(mpo, L, chi, tol=1e-11)
    print(f"  Reference energy: {ref_energy:.15f}")

    # Run A2DMRG (serial, np=1)
    print("\nRunning A2DMRG (np=1)...")
    try:
        a2dmrg_energy, a2dmrg_mps = a2dmrg_main(
            L=L,
            mpo=mpo,
            max_sweeps=20,
            bond_dim=chi,
            tol=1e-11,
            comm=None,
            verbose=False,
            warmup_sweeps=2,
            one_site=False
        )
        print(f"  A2DMRG energy:    {a2dmrg_energy:.15f}")

        error = abs(a2dmrg_energy - ref_energy)
        print(f"  Error: {error:.2e}")

        if error < 1e-12:
            print("  ✅ PASS - Machine precision!")
        elif error < 5e-10:
            print("  ✅ PASS - Acceptance threshold")
        else:
            print("  ❌ FAIL - Error too large")
            return False

    except Exception as e:
        print(f"  ❌ FAIL - Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_josephson_L8():
    """Test Josephson L=8."""
    print("\n" + "="*60)
    print("TEST: Josephson L=8, n=2")
    print("="*60)

    L = 8
    chi = 100

    # Load MPO
    mpo = load_mpo('benchmarks/benchmark_data/josephson_L8_n2_mpo.bin')

    # Get reference energy from quimb DMRG2
    print("Running quimb DMRG2 (reference)...")
    ref_mps, ref_energy = run_reference_dmrg2(mpo, L, chi, tol=1e-11)
    print(f"  Reference energy: {ref_energy:.15f}")

    # Run A2DMRG (serial, np=1)
    print("\nRunning A2DMRG (np=1)...")
    try:
        a2dmrg_energy, a2dmrg_mps = a2dmrg_main(
            L=L,
            mpo=mpo,
            max_sweeps=20,
            bond_dim=chi,
            tol=1e-11,
            comm=None,
            verbose=False,
            warmup_sweeps=2,
            one_site=False
        )
        print(f"  A2DMRG energy:    {a2dmrg_energy:.15f}")

        error = abs(a2dmrg_energy - ref_energy)
        print(f"  Error: {error:.2e}")

        if error < 1e-12:
            print("  ✅ PASS - Machine precision!")
        elif error < 5e-10:
            print("  ✅ PASS - Acceptance threshold")
        else:
            print("  ❌ FAIL - Error too large")
            return False

    except Exception as e:
        print(f"  ❌ FAIL - Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == '__main__':
    print("\n" + "="*60)
    print("A2DMRG QUICK VALIDATION TESTS")
    print("Testing convergence after i-orthogonal transformation fix")
    print("="*60)

    results = []

    # Run tests
    results.append(("Heisenberg L=8", test_heisenberg_L8()))
    results.append(("Heisenberg L=12", test_heisenberg_L12()))
    results.append(("Josephson L=8", test_josephson_L8()))

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:30s} {status}")

    all_passed = all(p for _, p in results)
    print("="*60)
    if all_passed:
        print("ALL TESTS PASSED! ✅")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED! ❌")
        sys.exit(1)
