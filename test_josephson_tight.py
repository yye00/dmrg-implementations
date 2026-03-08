#!/usr/bin/env python3
"""
Test A2DMRG on Josephson with tighter tolerance.

The Josephson L20 case is showing ~4e-09 error with tol=1e-11.
Let's try tighter tolerance to see if we can achieve better convergence.
"""

import sys
import os
from pathlib import Path
import time
import numpy as np

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / 'a2dmrg'))

from benchmark_data_loader import load_benchmark_case, convert_tensors_to_quimb_mpo
import quimb.tensor as qtn
from a2dmrg.dmrg import a2dmrg_main

# Load Josephson case
data = load_benchmark_case('josephson', 'L20_D50_nmax2')
mpo = convert_tensors_to_quimb_mpo(data['mpo_tensors'])
manifest = data['manifest']
E_golden = data['golden_results']['quimb_dmrg2']['energy']

print("="*80)
print("JOSEPHSON L20 CONVERGENCE TEST")
print("="*80)
print(f"Configuration: L={manifest['L']}, D={manifest['bond_dim']}, dtype={manifest['dtype']}")
print(f"Golden Reference: {E_golden:.15f}")
print()

# Test with different tolerances
tolerances = [1e-11, 1e-12, 1e-13]
max_sweeps_list = [40, 60, 80]

results = []

for tol in tolerances:
    for max_sweeps in max_sweeps_list:
        print(f"\nTesting: tol={tol:.1e}, max_sweeps={max_sweeps}")

        t0 = time.time()
        try:
            energy, mps = a2dmrg_main(
                L=manifest['L'],
                mpo=mpo,
                max_sweeps=max_sweeps,
                bond_dim=manifest['bond_dim'],
                tol=tol,
                dtype=np.complex128,
                comm=None,
                warmup_sweeps=5,
                verbose=False
            )
            t1 = time.time()

            E = float(np.real(energy))
            dE = E - E_golden

            print(f"  E = {E:.15f}")
            print(f"  ΔE = {dE:.3e}")
            print(f"  t = {t1-t0:.2f}s")

            if abs(dE) < 1e-12:
                print(f"  ✓✓ MACHINE PRECISION")
            elif abs(dE) < 5e-10:
                print(f"  ✓  ACCEPTED")
            else:
                print(f"  ✗  TOO LARGE")

            results.append({
                'tol': tol,
                'max_sweeps': max_sweeps,
                'energy': E,
                'delta_E': dE,
                'time': t1-t0
            })

        except Exception as e:
            print(f"  ✗  FAILED: {e}")
            results.append({
                'tol': tol,
                'max_sweeps': max_sweeps,
                'error': str(e)
            })

print(f"\n{'='*80}")
print("BEST RESULT:")
print(f"{'='*80}")

valid_results = [r for r in results if 'energy' in r]
if valid_results:
    best = min(valid_results, key=lambda r: abs(r['delta_E']))
    print(f"Tolerance: {best['tol']:.1e}")
    print(f"Max sweeps: {best['max_sweeps']}")
    print(f"Energy: {best['energy']:.15f}")
    print(f"ΔE: {best['delta_E']:.3e}")
    print(f"Time: {best['time']:.2f}s")

    if abs(best['delta_E']) < 1e-12:
        print("\n✓✓ Achieved MACHINE PRECISION")
    elif abs(best['delta_E']) < 5e-10:
        print("\n✓ Achieved ACCEPTANCE threshold")
    else:
        print("\n✗ Still above acceptance threshold")
else:
    print("No successful runs")
