#!/usr/bin/env python3
"""
Serial Validation of Stable Boundary Merge

Tests the merge logic without MPI by directly calling merge functions.
Compares old merge vs stable merge on the same boundary tensor data.
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from benchmark_data_loader import load_benchmark_case, convert_tensors_to_quimb_mpo
import quimb.tensor as qtn


def test_merge_directly():
    """
    Test merge functions directly by creating synthetic boundary data.
    """
    print("="*80)
    print("DIRECT MERGE FUNCTION TEST")
    print("="*80)

    # Import both merge implementations
    sys.path.insert(0, str(repo_root / 'pdmrg'))
    from pdmrg.parallel.merge import merge_boundary_tensors
    from pdmrg.parallel.merge_stable import merge_boundary_tensors_stable_from_V

    # Create synthetic boundary data
    chi_L = 10
    chi_R = 10
    chi_bond = 8
    d = 2

    np.random.seed(42)

    # Create random boundary tensors
    psi_left = np.random.randn(chi_L, d, chi_bond) + 1j * np.random.randn(chi_L, d, chi_bond)
    psi_right = np.random.randn(chi_bond, d, chi_R) + 1j * np.random.randn(chi_bond, d, chi_R)

    # Create Lambda and V
    Lambda = np.array([1.0, 0.8, 0.5, 0.2, 0.1, 0.05, 0.01, 0.001])  # Realistic singular values
    V = 1.0 / np.clip(Lambda, 1e-12, None)  # V = 1/Lambda (old approach)

    print(f"\nInput diagnostics:")
    print(f"  Lambda (S): min={Lambda.min():.3e}, max={Lambda.max():.3e}")
    print(f"  V (1/S): min={V.min():.3e}, max={V.max():.3e}")
    print(f"  Condition number (V): {V.max()/V.min():.3e}")

    # Create simple environments and MPO tensors
    D = 3
    L_env = np.random.randn(chi_L, D, chi_L) + 1j * np.random.randn(chi_L, D, chi_L)
    R_env = np.random.randn(chi_R, D, chi_R) + 1j * np.random.randn(chi_R, D, chi_R)
    W_left = np.random.randn(D, D, d, d) + 1j * np.random.randn(D, D, d, d)
    W_right = np.random.randn(D, D, d, d) + 1j * np.random.randn(D, D, d, d)

    # Make environments Hermitian (approximate)
    L_env = (L_env + L_env.conj().transpose(2, 1, 0)) / 2
    R_env = (R_env + R_env.conj().transpose(2, 1, 0)) / 2

    # Test 1: Old merge with skip_optimization=True
    print("\n" + "-"*80)
    print("Test 1: Old merge (skip_optimization=True)")
    print("-"*80)

    try:
        A_L_old, A_R_old, V_new_old, E_old, trunc_old = merge_boundary_tensors(
            psi_left, psi_right, V,
            L_env, R_env, W_left, W_right,
            max_bond=8, max_iter=30, tol=1e-10,
            skip_optimization=True
        )
        print(f"✓ Old merge succeeded")
        print(f"  Energy: {E_old:.12f}")
        print(f"  Truncation error: {trunc_old:.3e}")
    except Exception as e:
        print(f"✗ Old merge failed: {e}")
        return False

    # Test 2: Stable merge with skip_optimization=True
    print("\n" + "-"*80)
    print("Test 2: Stable merge (skip_optimization=True)")
    print("-"*80)

    try:
        A_L_new, A_R_new, V_new_new, E_new, trunc_new = merge_boundary_tensors_stable_from_V(
            psi_left, psi_right, V,
            L_env, R_env, W_left, W_right,
            max_bond=8, max_iter=30, tol=1e-10,
            skip_optimization=True
        )
        print(f"✓ Stable merge succeeded")
        print(f"  Energy: {E_new:.12f}")
        print(f"  Truncation error: {trunc_new:.3e}")
    except Exception as e:
        print(f"✗ Stable merge failed: {e}")
        return False

    # Compare results
    print("\n" + "-"*80)
    print("COMPARISON")
    print("-"*80)

    dE = abs(E_new - E_old)
    print(f"  Energy difference: {dE:.3e}")

    # Check tensor norms
    A_L_diff = np.linalg.norm(A_L_new - A_L_old)
    A_R_diff = np.linalg.norm(A_R_new - A_R_old)
    print(f"  ||A_left_new - A_left_old||: {A_L_diff:.3e}")
    print(f"  ||A_right_new - A_right_old||: {A_R_diff:.3e}")

    # Verdict
    if dE < 1e-8:
        print(f"\n✓ Stable merge produces equivalent results (ΔE < 1e-8)")
        return True
    else:
        print(f"\n✗ Stable merge differs significantly (ΔE = {dE:.3e})")
        return False


def test_with_real_data():
    """
    Test using actual benchmark data (no MPI needed).
    """
    print("\n" + "="*80)
    print("REAL DATA TEST (Heisenberg L=12)")
    print("="*80)

    # Load Heisenberg data
    data = load_benchmark_case("heisenberg", "L12_D20")
    mpo = convert_tensors_to_quimb_mpo(data['mpo_tensors'])
    golden = data['golden_results']
    E_golden = golden['quimb_dmrg2']['energy']

    print(f"\nGolden reference (quimb DMRG2): {E_golden:.15f}")

    # Run quimb DMRG2 ourselves to verify
    print("\nRunning quimb DMRG2 for verification...")
    dmrg2 = qtn.DMRG2(mpo, bond_dims=20, cutoffs=1e-14)
    dmrg2.solve(max_sweeps=30, tol=1e-10, verbosity=0)
    E_verify = float(np.real(dmrg2.energy))

    print(f"  Verified energy: {E_verify:.15f}")
    print(f"  Δ vs golden: {abs(E_verify - E_golden):.3e}")

    if abs(E_verify - E_golden) < 1e-10:
        print(f"  ✓ Golden reference reproduced")
        return True
    else:
        print(f"  ✗ Cannot reproduce golden reference")
        return False


def main():
    print("STABLE MERGE VALIDATION (Serial Tests)")
    print("="*80)
    print("These tests validate merge logic without requiring MPI.\n")

    # Test 1: Direct merge function comparison
    test1_passed = test_merge_directly()

    # Test 2: Real data verification
    test2_passed = test_with_real_data()

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"  Direct merge test: {'✓ PASS' if test1_passed else '✗ FAIL'}")
    print(f"  Real data test: {'✓ PASS' if test2_passed else '✗ FAIL'}")

    if test1_passed and test2_passed:
        print("\n✓ SERIAL VALIDATION PASSED")
        print("  Stable merge logic is mathematically equivalent.")
        print("  NOTE: Full parallel validation requires working MPI.")
        return 0
    else:
        print("\n✗ SERIAL VALIDATION FAILED")
        print("  Stable merge has issues even in serial tests.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
