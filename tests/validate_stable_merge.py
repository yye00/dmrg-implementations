#!/usr/bin/env python3
"""
Validate Stable Boundary Merge Fix

Compares old merge (V = 1/Lambda) vs new stable merge (Lambda directly)
against golden quimb DMRG2 results.

Tests:
1. Heisenberg L=12, D=20 (real)
2. Josephson L=20, D=50, n_max=2 (complex)

For each test, runs:
- PDMRG with use_stable_merge=False (old path)
- PDMRG with use_stable_merge=True (new path)
- PDMRG2 with both paths

Reports:
- Final energy
- Delta vs golden
- Convergence
- Number of sweeps
"""

import sys
import os
import json
import time
import numpy as np
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

# Force single-threaded for reproducibility
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from mpi4py import MPI
from benchmark_data_loader import load_benchmark_case, convert_tensors_to_quimb_mpo


def run_pdmrg_test(model, case, use_stable_merge, implementation='pdmrg'):
    """
    Run PDMRG or PDMRG2 with specified merge path.

    Parameters
    ----------
    model : str
        "heisenberg" or "josephson"
    case : str
        Case identifier
    use_stable_merge : bool
        True = new stable merge, False = old merge
    implementation : str
        "pdmrg" or "pdmrg2"

    Returns
    -------
    dict
        {energy, time, metadata}
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Load benchmark data
    data = load_benchmark_case(model, case)
    mpo = convert_tensors_to_quimb_mpo(data['mpo_tensors'])
    manifest = data['manifest']

    # Import appropriate implementation
    if implementation == 'pdmrg':
        sys.path.insert(0, str(repo_root / 'pdmrg'))
        from pdmrg.dmrg import pdmrg_main
    elif implementation == 'pdmrg2':
        sys.path.insert(0, str(repo_root / 'pdmrg2'))
        from pdmrg.dmrg import pdmrg_main
    else:
        raise ValueError(f"Unknown implementation: {implementation}")

    # Run PDMRG
    t0 = time.time()
    try:
        result = pdmrg_main(
            L=manifest['L'],
            mpo=mpo,
            max_sweeps=30,
            bond_dim=manifest['bond_dim'],
            bond_dim_warmup=manifest['bond_dim'],
            n_warmup_sweeps=5,
            tol=1e-10,
            dtype=manifest['dtype'],
            comm=comm,
            verbose=(rank == 0),
            use_stable_merge=use_stable_merge,
            return_metadata=True
        )

        if len(result) == 3:
            energy, pmps, metadata = result
        else:
            energy, pmps = result
            metadata = None

        t1 = time.time()

        return {
            'success': True,
            'energy': float(np.real(energy)),
            'time': t1 - t0,
            'metadata': metadata
        }
    except Exception as e:
        t1 = time.time()
        return {
            'success': False,
            'error': str(e),
            'time': t1 - t0
        }


def run_validation_suite():
    """Run complete validation suite."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_procs = comm.Get_size()

    if n_procs != 2:
        if rank == 0:
            print("ERROR: This validation requires exactly np=2")
            print(f"Usage: mpirun -np 2 python {__file__}")
        sys.exit(1)

    if rank == 0:
        print("="*80)
        print("STABLE BOUNDARY MERGE VALIDATION")
        print("="*80)
        print(f"np={n_procs}, threads=1 (OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')})")
        print()

    test_cases = [
        ("heisenberg", "L12_D20"),
        ("josephson", "L20_D50_nmax2")
    ]

    all_results = {}

    for model, case in test_cases:
        if rank == 0:
            print(f"\n{'='*80}")
            print(f"TEST CASE: {model}/{case}")
            print(f"{'='*80}")

        # Load golden results
        data = load_benchmark_case(model, case)
        golden = data['golden_results']
        E_golden = golden['quimb_dmrg2']['energy']

        if rank == 0:
            print(f"\nGolden Reference (quimb DMRG2):")
            print(f"  Energy: {E_golden:.15f}")
            print(f"  Sweeps: {golden['quimb_dmrg2']['sweeps']}")
            print(f"  Time: {golden['quimb_dmrg2']['wall_time']:.2f}s")

        # Test matrix:
        # 1. PDMRG old merge (use_stable_merge=False)
        # 2. PDMRG new merge (use_stable_merge=True)
        # 3. PDMRG2 old merge (use_stable_merge=False)
        # 4. PDMRG2 new merge (use_stable_merge=True)

        tests = [
            ('pdmrg', False, 'PDMRG (old merge)'),
            ('pdmrg', True, 'PDMRG (stable merge)'),
            ('pdmrg2', False, 'PDMRG2 (old merge)'),
            ('pdmrg2', True, 'PDMRG2 (stable merge)'),
        ]

        case_results = {}

        for impl, use_stable, label in tests:
            if rank == 0:
                print(f"\n{label}:")
                print(f"  Running with use_stable_merge={use_stable}...")

            result = run_pdmrg_test(model, case, use_stable, impl)

            if result['success']:
                E = result['energy']
                dE = E - E_golden

                case_results[label] = {
                    'energy': E,
                    'delta_E': dE,
                    'time': result['time'],
                    'metadata': result['metadata']
                }

                if rank == 0:
                    print(f"  Energy: {E:.15f}")
                    print(f"  ΔE vs golden: {dE:.3e}")
                    if result['metadata']:
                        print(f"  Converged: {result['metadata'].get('converged', 'N/A')}")
                        print(f"  Final sweep: {result['metadata'].get('final_sweep', 'N/A')}")
                    print(f"  Time: {result['time']:.2f}s")

                    # Check validation threshold
                    if abs(dE) < 1e-10:
                        print(f"  ✓ PASS (|ΔE| < 1e-10)")
                    elif abs(dE) < 1e-8:
                        print(f"  ⚠ MARGINAL (|ΔE| < 1e-8)")
                    else:
                        print(f"  ✗ FAIL (|ΔE| > 1e-8)")
            else:
                case_results[label] = {
                    'success': False,
                    'error': result['error'],
                    'time': result['time']
                }
                if rank == 0:
                    print(f"  ✗ FAILED: {result['error']}")

        all_results[f"{model}/{case}"] = case_results

    # Final summary
    if rank == 0:
        print(f"\n{'='*80}")
        print("VALIDATION SUMMARY")
        print(f"{'='*80}")

        for test_case, case_results in all_results.items():
            print(f"\n{test_case}:")

            for label, result in case_results.items():
                if result.get('success', True):
                    dE = result['delta_E']
                    status = "✓" if abs(dE) < 1e-10 else ("⚠" if abs(dE) < 1e-8 else "✗")
                    print(f"  {status} {label}: ΔE = {dE:.3e}")
                else:
                    print(f"  ✗ {label}: FAILED")

        # Compare old vs new merge
        print(f"\n{'='*80}")
        print("STABLE MERGE VALIDATION VERDICT")
        print(f"{'='*80}")

        verdict_passed = True
        for test_case, case_results in all_results.items():
            pdmrg_old = case_results.get('PDMRG (old merge)', {})
            pdmrg_new = case_results.get('PDMRG (stable merge)', {})

            if pdmrg_old.get('success', True) and pdmrg_new.get('success', True):
                dE_old = abs(pdmrg_old['delta_E'])
                dE_new = abs(pdmrg_new['delta_E'])

                print(f"\n{test_case}:")
                print(f"  Old merge: |ΔE| = {dE_old:.3e}")
                print(f"  New merge: |ΔE| = {dE_new:.3e}")

                if dE_new <= dE_old:
                    improvement = (dE_old - dE_new) / dE_old * 100 if dE_old > 0 else 0
                    print(f"  ✓ Stable merge is BETTER or EQUAL (improved by {improvement:.1f}%)")
                else:
                    degradation = (dE_new - dE_old) / dE_old * 100
                    print(f"  ✗ Stable merge is WORSE (degraded by {degradation:.1f}%)")
                    verdict_passed = False

        print(f"\n{'='*80}")
        if verdict_passed:
            print("✓ VERDICT: STABLE MERGE IS VALID")
            print("  New boundary merge is at least as accurate as old path.")
            print("  Safe to use as default (use_stable_merge=True).")
        else:
            print("✗ VERDICT: STABLE MERGE FAILS VALIDATION")
            print("  New boundary merge degrades accuracy vs old path.")
            print("  DO NOT use as default until fixed.")
        print(f"{'='*80}")

        # Save results
        output_file = repo_root / "tests" / "stable_merge_validation_results.json"
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nDetailed results saved to: {output_file}")

        return 0 if verdict_passed else 1


if __name__ == '__main__':
    sys.exit(run_validation_suite())
