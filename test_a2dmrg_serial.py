#!/usr/bin/env python3
"""
Test A2DMRG in serial mode (no MPI) on small cases.

Tests:
- Heisenberg L12_D20
- Josephson L20_D50_nmax2

Validates that the fixed A2DMRG implementation converges correctly.
"""

import sys
import os
from pathlib import Path
import time
import numpy as np

# Force single-threaded BLAS
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / 'a2dmrg'))

from benchmark_data_loader import load_benchmark_case, convert_tensors_to_quimb_mpo
import quimb.tensor as qtn
from a2dmrg.dmrg import a2dmrg_main

# Validation thresholds
GOLDEN_TOLERANCE = 1e-11
MACHINE_PRECISION_THRESHOLD = 1e-12
ACCEPTANCE_THRESHOLD = 5e-10


def run_quimb_dmrg2(model, case):
    """Run quimb DMRG2 reference."""
    data = load_benchmark_case(model, case)
    mpo = convert_tensors_to_quimb_mpo(data['mpo_tensors'])
    manifest = data['manifest']

    t0 = time.time()
    dmrg = qtn.DMRG2(mpo, bond_dims=manifest['bond_dim'], cutoffs=1e-14)
    dmrg.solve(max_sweeps=50, tol=GOLDEN_TOLERANCE, verbosity=0)
    t1 = time.time()

    return float(np.real(dmrg.energy)), t1 - t0


def run_a2dmrg_serial(model, case):
    """Run A2DMRG in serial mode (no MPI)."""
    data = load_benchmark_case(model, case)
    mpo = convert_tensors_to_quimb_mpo(data['mpo_tensors'])
    manifest = data['manifest']

    dtype = np.complex128 if manifest['dtype'] == 'complex128' else np.float64

    t0 = time.time()
    try:
        # Run without comm (serial mode)
        energy, mps = a2dmrg_main(
            L=manifest['L'],
            mpo=mpo,
            max_sweeps=40,
            bond_dim=manifest['bond_dim'],
            tol=GOLDEN_TOLERANCE,
            dtype=dtype,
            comm=None,  # Serial mode
            warmup_sweeps=5,
            verbose=False
        )
        t1 = time.time()
        return float(np.real(energy)), t1 - t0, None
    except Exception as e:
        return None, None, str(e)


def test_case(model, case):
    """Test a single benchmark case."""
    print(f"\n{'='*80}")
    print(f"TEST CASE: {model}/{case}")
    print(f"{'='*80}\n")

    # Load golden reference
    data = load_benchmark_case(model, case)
    manifest = data['manifest']
    print(f"Configuration: L={manifest['L']}, D={manifest['bond_dim']}, dtype={manifest['dtype']}")

    # Get golden energy
    golden_data = data['golden_results']
    E_golden = golden_data['quimb_dmrg2']['energy']
    print(f"\nGolden Reference: {E_golden:.15f}")

    # Run quimb DMRG2
    print("\n  Running quimb DMRG2...")
    E_quimb, t_quimb = run_quimb_dmrg2(model, case)
    dE_quimb = E_quimb - E_golden
    print(f"    E = {E_quimb:.15f}")
    print(f"    ΔE = {dE_quimb:.3e}")
    print(f"    t = {t_quimb:.3f}s")

    # Run A2DMRG serial
    print("\n  Running A2DMRG (serial, no MPI)...")
    E_a2dmrg, t_a2dmrg, error = run_a2dmrg_serial(model, case)

    if error:
        print(f"    ✗ FAILED: {error}")
        return {
            'golden': E_golden,
            'quimb': {'energy': E_quimb, 'delta_E': dE_quimb, 'time': t_quimb},
            'a2dmrg': {'success': False, 'error': error}
        }

    dE = E_a2dmrg - E_golden
    machine_prec = abs(dE) < MACHINE_PRECISION_THRESHOLD
    accepted = abs(dE) < ACCEPTANCE_THRESHOLD

    if machine_prec:
        status = "✓✓ MACHINE PRECISION"
    elif accepted:
        status = "✓  ACCEPTED"
    else:
        status = "✗  FAILED"

    print(f"    {status}")
    print(f"    E = {E_a2dmrg:.15f}")
    print(f"    ΔE = {dE:.3e}")
    print(f"    t = {t_a2dmrg:.3f}s")
    print(f"    Speedup vs quimb: {t_quimb/t_a2dmrg:.2f}×")

    return {
        'golden': E_golden,
        'quimb': {'energy': E_quimb, 'delta_E': dE_quimb, 'time': t_quimb},
        'a2dmrg': {
            'success': True,
            'energy': E_a2dmrg,
            'delta_E': dE,
            'time': t_a2dmrg,
            'machine_precision': machine_prec,
            'accepted': accepted
        }
    }


def main():
    print("="*80)
    print("A2DMRG SERIAL VALIDATION TEST")
    print("="*80)
    print("Testing fixed A2DMRG implementation (serial mode)")
    print()
    print(f"Environment:")
    print(f"  OMP_NUM_THREADS = {os.environ.get('OMP_NUM_THREADS')}")
    print(f"  OPENBLAS_NUM_THREADS = {os.environ.get('OPENBLAS_NUM_THREADS')}")
    print()

    results = {}

    # Test Heisenberg L12
    print("\n" + "="*80)
    print("HEISENBERG CHAIN TEST")
    print("="*80)
    heisenberg_result = test_case('heisenberg', 'L12_D20')
    results['heisenberg/L12_D20'] = heisenberg_result

    # Test Josephson L20
    print("\n" + "="*80)
    print("JOSEPHSON JUNCTION TEST")
    print("="*80)
    josephson_result = test_case('josephson', 'L20_D50_nmax2')
    results['josephson/L20_D50_nmax2'] = josephson_result

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")

    all_passed = True
    for case_name, case_data in results.items():
        a2dmrg_data = case_data['a2dmrg']
        if a2dmrg_data.get('success'):
            if a2dmrg_data['machine_precision']:
                print(f"✓✓ {case_name}: ΔE = {a2dmrg_data['delta_E']:.3e} (MACHINE PRECISION)")
            elif a2dmrg_data['accepted']:
                print(f"✓  {case_name}: ΔE = {a2dmrg_data['delta_E']:.3e} (ACCEPTED)")
            else:
                print(f"✗  {case_name}: ΔE = {a2dmrg_data['delta_E']:.3e} (FAILED - TOO LARGE)")
                all_passed = False
        else:
            print(f"✗  {case_name}: FAILED - {a2dmrg_data.get('error', 'Unknown error')}")
            all_passed = False

    print(f"\n{'='*80}")
    if all_passed:
        print("✓✓ ALL TESTS PASSED")
        print("="*80)
        print("\nA2DMRG implementation is correctly converging to reference energies!")
        print("The bond dimension preservation fix is working as expected.")
    else:
        print("✗ SOME TESTS FAILED")
        print("="*80)
        print("\nPlease review the errors above.")

    print()
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
