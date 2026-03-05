#!/usr/bin/env python3
"""
Josephson Junction Array: Full Correctness Benchmark

Validates all DMRG implementations converge to the same energy within 1e-10.

Runs:
  1-2:  quimb DMRG1, DMRG2 (serial reference)
  3-6:  PDMRG np=1,2,4,8
  7-10: A2DMRG np=1,2,4,8

Total: 10 runs, all must match to 1e-10
"""

import subprocess
import time
import json
import os
import sys
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.josephson_junction import build_josephson_mpo, verify_complex_dtype


# Configuration
L = 20
BOND_DIM = 50
N_MAX = 2  # d = 5
MAX_SWEEPS = 30
TOL = 1e-14  # Target precision for solver
MATCH_TOL = 1e-10  # Required agreement between implementations

NP_LIST = [1, 2, 4, 8]


def run_quimb_dmrg1(mpo, bond_dim, max_sweeps=30, tol=1e-14):
    """Run quimb DMRG1 (single-site)."""
    import quimb.tensor as qtn
    
    t0 = time.time()
    dmrg = qtn.DMRG1(mpo, bond_dims=bond_dim)
    dmrg.solve(max_sweeps=max_sweeps, tol=tol, verbosity=0)
    t1 = time.time()
    
    return {
        'method': 'quimb_DMRG1',
        'energy': float(np.real(dmrg.energy)),
        'time': t1 - t0,
        'success': True
    }


def run_quimb_dmrg2(mpo, bond_dim, max_sweeps=30, tol=1e-14):
    """Run quimb DMRG2 (two-site)."""
    import quimb.tensor as qtn
    
    t0 = time.time()
    dmrg = qtn.DMRG2(mpo, bond_dims=bond_dim)
    dmrg.solve(max_sweeps=max_sweeps, tol=tol, verbosity=0)
    t1 = time.time()
    
    return {
        'method': 'quimb_DMRG2',
        'energy': float(np.real(dmrg.energy)),
        'time': t1 - t0,
        'success': True
    }


def run_pdmrg(np_count, L=20, bond_dim=50, n_max=2, max_sweeps=30, tol=1e-14):
    """Run PDMRG with specified number of processes."""
    
    script = f'''
import sys
import time
import numpy as np
sys.path.insert(0, '/home/captain/clawd/work/dmrg-implementations/a2dmrg')
from benchmarks.josephson_junction import build_josephson_mpo

from a2dmrg.mpi_compat import MPI, HAS_MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

mpo = build_josephson_mpo({L}, E_J=1.0, E_C=0.5, n_max={n_max}, with_flux=True)

sys.path.insert(0, '/home/captain/clawd/work/dmrg-implementations/pdmrg')
from pdmrg.dmrg import pdmrg_main

t0 = time.time()
energy, mps = pdmrg_main(
    L={L},
    mpo=mpo,
    bond_dim={bond_dim},
    max_sweeps={max_sweeps},
    tol={tol},
    dtype='complex128',
    comm=comm,
    verbose=False
)
t1 = time.time()

if rank == 0:
    print(f"RESULT:{{np.real(energy):.15f}}:{{t1-t0:.4f}}")
'''
    
    script_path = '/tmp/pdmrg_correctness.py'
    with open(script_path, 'w') as f:
        f.write(script)
    
    pdmrg_python = '/home/captain/clawd/work/dmrg-implementations/pdmrg/venv/bin/python'
    
    cmd = f'''
source /etc/profile.d/modules.sh && module load mpi/openmpi-x86_64 && \
mpirun -np {np_count} --oversubscribe --mca btl tcp,self --mca btl_tcp_if_include lo \
{pdmrg_python} -u {script_path} 2>&1
'''
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1800)
        output = result.stdout + result.stderr
        
        for line in output.split('\n'):
            if line.startswith('RESULT:'):
                parts = line.split(':')
                return {
                    'method': f'PDMRG_np{np_count}',
                    'energy': float(parts[1]),
                    'time': float(parts[2]),
                    'success': True,
                    'np': np_count
                }
        
        return {
            'method': f'PDMRG_np{np_count}',
            'success': False,
            'error': output[-2000:] if len(output) > 2000 else output
        }
    except Exception as e:
        return {'method': f'PDMRG_np{np_count}', 'success': False, 'error': str(e)}


def run_a2dmrg(np_count, L=20, bond_dim=50, n_max=2, max_sweeps=30, tol=1e-14):
    """Run A2DMRG with specified number of processes."""
    
    script = f'''
import sys
import time
import numpy as np
sys.path.insert(0, '/home/captain/clawd/work/dmrg-implementations/a2dmrg')
from benchmarks.josephson_junction import build_josephson_mpo
from a2dmrg.dmrg import a2dmrg_main

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

mpo = build_josephson_mpo({L}, E_J=1.0, E_C=0.5, n_max={n_max}, with_flux=True)

t0 = time.time()
energy, mps = a2dmrg_main(
    L={L},
    mpo=mpo,
    max_sweeps={max_sweeps},
    bond_dim={bond_dim},
    tol={tol},
    dtype=np.complex128,
    comm=comm,
    warmup_sweeps=3,
    one_site=True,
    verbose=False
)
t1 = time.time()

if rank == 0:
    print(f"RESULT:{{np.real(energy):.15f}}:{{t1-t0:.4f}}")
'''
    
    script_path = '/tmp/a2dmrg_correctness.py'
    with open(script_path, 'w') as f:
        f.write(script)
    
    a2dmrg_python = '/home/captain/clawd/work/dmrg-implementations/a2dmrg/venv/bin/python'
    
    cmd = f'''
source /etc/profile.d/modules.sh && module load mpi/openmpi-x86_64 && \
mpirun -np {np_count} --oversubscribe --mca btl tcp,self --mca btl_tcp_if_include lo \
{a2dmrg_python} -u {script_path} 2>&1
'''
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1800)
        output = result.stdout + result.stderr
        
        for line in output.split('\n'):
            if line.startswith('RESULT:'):
                parts = line.split(':')
                return {
                    'method': f'A2DMRG_np{np_count}',
                    'energy': float(parts[1]),
                    'time': float(parts[2]),
                    'success': True,
                    'np': np_count
                }
        
        return {
            'method': f'A2DMRG_np{np_count}',
            'success': False,
            'error': output[-2000:] if len(output) > 2000 else output
        }
    except Exception as e:
        return {'method': f'A2DMRG_np{np_count}', 'success': False, 'error': str(e)}


def main():
    print("=" * 80)
    print("JOSEPHSON JUNCTION ARRAY - CORRECTNESS VALIDATION")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Parameters: L={L}, D={BOND_DIM}, n_max={N_MAX} (d={2*N_MAX+1}), dtype=complex128")
    print(f"Target precision: all energies must agree within {MATCH_TOL:.0e}")
    print("=" * 80)
    
    # Build MPO once for quimb
    print("\n[0/10] Building Josephson Junction MPO...")
    mpo = build_josephson_mpo(L, E_J=1.0, E_C=0.5, n_max=N_MAX, with_flux=True)
    is_complex, msg = verify_complex_dtype(mpo)
    print(f"  {msg}")
    
    results = []
    
    # Run 1: quimb DMRG1
    print("\n[1/10] quimb DMRG1 (serial, single-site)...")
    r = run_quimb_dmrg1(mpo, BOND_DIM, MAX_SWEEPS, TOL)
    results.append(r)
    print(f"  Energy: {r['energy']:.12f}")
    print(f"  Time:   {r['time']:.2f}s")
    
    # Run 2: quimb DMRG2
    print("\n[2/10] quimb DMRG2 (serial, two-site)...")
    r = run_quimb_dmrg2(mpo, BOND_DIM, MAX_SWEEPS, TOL)
    results.append(r)
    print(f"  Energy: {r['energy']:.12f}")
    print(f"  Time:   {r['time']:.2f}s")
    
    # Runs 3-6: PDMRG
    for i, np_count in enumerate(NP_LIST):
        run_num = 3 + i
        print(f"\n[{run_num}/10] PDMRG np={np_count}...")
        r = run_pdmrg(np_count, L, BOND_DIM, N_MAX, MAX_SWEEPS, TOL)
        results.append(r)
        if r['success']:
            print(f"  Energy: {r['energy']:.12f}")
            print(f"  Time:   {r['time']:.2f}s")
        else:
            print(f"  FAILED: {r.get('error', 'unknown')[:200]}")
    
    # Runs 7-10: A2DMRG
    for i, np_count in enumerate(NP_LIST):
        run_num = 7 + i
        print(f"\n[{run_num}/10] A2DMRG np={np_count}...")
        r = run_a2dmrg(np_count, L, BOND_DIM, N_MAX, MAX_SWEEPS, TOL)
        results.append(r)
        if r['success']:
            print(f"  Energy: {r['energy']:.12f}")
            print(f"  Time:   {r['time']:.2f}s")
        else:
            print(f"  FAILED: {r.get('error', 'unknown')[:200]}")
    
    # Analysis
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]
    
    if successful:
        E_ref = successful[0]['energy']
        
        print(f"\n{'Method':<20} | {'Energy':>18} | {'ΔE':>12} | {'Time':>10} | {'Status':<10}")
        print("-" * 80)
        
        all_match = True
        for r in results:
            if r.get('success', False):
                diff = abs(r['energy'] - E_ref)
                status = "✓ PASS" if diff < MATCH_TOL else "✗ FAIL"
                if diff >= MATCH_TOL:
                    all_match = False
                print(f"{r['method']:<20} | {r['energy']:>18.12f} | {diff:>12.2e} | {r['time']:>9.2f}s | {status}")
            else:
                print(f"{r['method']:<20} | {'N/A':>18} | {'N/A':>12} | {'N/A':>10} | ✗ FAILED")
                all_match = False
        
        # Energy spread
        energies = [r['energy'] for r in successful]
        E_min, E_max = min(energies), max(energies)
        spread = E_max - E_min
        
        print("-" * 80)
        print(f"Reference Energy: {E_ref:.15f}")
        print(f"Energy spread:    {spread:.2e} (max-min across all runs)")
        print(f"Target tolerance: {MATCH_TOL:.0e}")
        
        print("\n" + "=" * 80)
        if all_match and len(failed) == 0:
            print("✓ ALL 10 RUNS PASSED - Implementations are correct!")
        else:
            n_pass = sum(1 for r in successful if abs(r['energy'] - E_ref) < MATCH_TOL)
            print(f"✗ {n_pass}/10 runs within tolerance, {len(failed)} failed")
        print("=" * 80)
    else:
        print("ERROR: No successful runs!")
    
    # Save detailed results
    output = {
        'benchmark': 'josephson_correctness',
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'L': L,
            'bond_dim': BOND_DIM,
            'n_max': N_MAX,
            'd': 2 * N_MAX + 1,
            'dtype': 'complex128',
            'tolerance': MATCH_TOL
        },
        'results': results,
        'summary': {
            'total_runs': len(results),
            'successful_runs': len(successful),
            'failed_runs': len(failed),
            'energy_spread': float(spread) if successful else None,
            'all_match': all_match if successful else False
        }
    }
    
    output_file = '/home/captain/clawd/work/dmrg-implementations/a2dmrg/benchmarks/josephson_correctness_results.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {output_file}")
    
    return 0 if (successful and all_match and len(failed) == 0) else 1


if __name__ == '__main__':
    sys.exit(main())
