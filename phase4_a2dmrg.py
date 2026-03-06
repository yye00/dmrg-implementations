#!/usr/bin/env python3
"""Phase 4: A2DMRG CPU Benchmarks"""
import json
import time
import sys
import os
import subprocess
import numpy as np
from datetime import datetime

REPO_ROOT = os.path.expanduser("~/dmrg-implementations")
sys.path.insert(0, os.path.join(REPO_ROOT, 'a2dmrg'))

VALIDATION_TOL = 1e-10
N_REPETITIONS = 5

HEISENBERG_REF = -5.142090632840
JOSEPHSON_REF = None

def load_references():
    """Load reference energies."""
    global JOSEPHSON_REF
    try:
        with open(os.path.join(REPO_ROOT, 'benchmark_results_phase2_josephson.json')) as f:
            data = json.load(f)
            JOSEPHSON_REF = data['josephson_short']['quimb_dmrg2']['energy_mean']
            print(f"✓ Loaded references: Heis={HEISENBERG_REF:.12f}, Jos={JOSEPHSON_REF:.12f}")
    except Exception as e:
        print(f"❌ Could not load references: {e}")
        sys.exit(1)

def run_a2dmrg_mpi(mpo_builder, np_count, **params):
    """Run A2DMRG with MPI."""
    dtype_str = params.get('dtype', 'float64')
    script = f'''
import sys, time, json
import numpy as np
sys.path.insert(0, "{REPO_ROOT}/a2dmrg")
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

{mpo_builder}

from a2dmrg.dmrg import a2dmrg_main

t0 = time.time()
energy, mps = a2dmrg_main(
    L={params['L']},
    mpo=mpo,
    bond_dim={params['D']},
    max_sweeps={params['sweeps']},
    tol={params['tol']},
    dtype=np.{dtype_str},
    comm=comm,
    verbose=False
)
elapsed = time.time() - t0

if rank == 0:
    print(json.dumps({{"energy": float(np.real(energy)), "time": elapsed}}))
'''
    
    with open('/tmp/run_a2dmrg_temp.py', 'w') as f:
        f.write(script)
    
    cmd = ['mpirun', '-np', str(np_count), '--oversubscribe', 
           'python3', '/tmp/run_a2dmrg_temp.py']
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if result.returncode != 0:
            return None, None, result.stderr
        for line in result.stdout.strip().split('\n'):
            if line.startswith('{'):
                data = json.loads(line)
                return data['energy'], data['time'], None
        return None, None, "No JSON output found"
    except Exception as e:
        return None, None, str(e)

def validate_energy(test_energy, reference_energy, test_name):
    """Validate energy."""
    diff = abs(test_energy - reference_energy)
    if diff > VALIDATION_TOL:
        print(f"\n{'='*80}")
        print(f"❌ VALIDATION FAILURE!")
        print(f"Test: {test_name}")
        print(f"Reference: {reference_energy:.15f}")
        print(f"Test: {test_energy:.15f}")
        print(f"Difference: {diff:.2e}")
        print(f"{'='*80}")
        return False
    return True

def run_a2dmrg_benchmarks(model_name, mpo_builder, reference_energy, **params):
    """Run A2DMRG benchmarks."""
    print(f"\n{'='*80}")
    print(f"  A2DMRG: {model_name}")
    print(f"  Reference: {reference_energy:.12f}")
    print(f"{'='*80}")
    
    results = {}
    
    for np_val in [1, 2, 4, 8]:
        print(f"\n[A2DMRG np={np_val}]...")
        energies, times = [], []
        
        for rep in range(N_REPETITIONS):
            energy, elapsed, error = run_a2dmrg_mpi(mpo_builder, np_val, **params)
            if error:
                print(f"  ❌ Error: {error}")
                return None
            
            energies.append(energy)
            times.append(elapsed)
            print(f"  Rep {rep+1}/{N_REPETITIONS}: E={energy:.12f}, time={elapsed:.3f}s")
            
            if not validate_energy(energy, reference_energy, f"A2DMRG np={np_val}"):
                return None
        
        results[f'a2dmrg_np{np_val}'] = {
            'energies': energies,
            'times': times,
            'energy_mean': np.mean(energies),
            'time_mean': np.mean(times),
            'time_std': np.std(times)
        }
        print(f"  ✓ Validated! Avg time: {np.mean(times):.3f}s")
    
    return results

def main():
    print(f"\n{'#'*80}")
    print(f"# PHASE 4: A2DMRG CPU BENCHMARKS")
    print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# WARNING: A2DMRG is slow - this will take time!")
    print(f"{'#'*80}")
    
    load_references()
    
    all_results = {
        'metadata': {
            'start_time': datetime.now().isoformat(),
            'validation_tol': VALIDATION_TOL,
            'n_repetitions': N_REPETITIONS
        }
    }
    
    # Heisenberg A2DMRG
    print(f"\n{'#'*80}")
    print(f"# HEISENBERG A2DMRG")
    print(f"{'#'*80}")
    heis_mpo_builder = f'''import quimb.tensor as qtn
mpo = qtn.MPO_ham_heis(L=12, j=1.0, bz=0.0, cyclic=False)'''
    
    heis_results = run_a2dmrg_benchmarks(
        "Heisenberg-Short", heis_mpo_builder, HEISENBERG_REF,
        L=12, D=100, sweeps=20, tol=1e-10, dtype='float64'
    )
    if heis_results is None:
        return 1
    all_results['heisenberg_short_a2dmrg'] = heis_results
    
    # Josephson A2DMRG
    print(f"\n{'#'*80}")
    print(f"# JOSEPHSON A2DMRG")
    print(f"{'#'*80}")
    jos_mpo_builder = f'''sys.path.insert(0, "{REPO_ROOT}/a2dmrg")
from benchmarks.josephson_junction import build_josephson_mpo
mpo = build_josephson_mpo(8, E_J=1.0, E_C=0.5, n_max=2, with_flux=True)'''
    
    jos_results = run_a2dmrg_benchmarks(
        "Josephson-Short", jos_mpo_builder, JOSEPHSON_REF,
        L=8, D=50, sweeps=20, tol=1e-10, dtype='complex128', n_max=2
    )
    if jos_results is None:
        return 1
    all_results['josephson_short_a2dmrg'] = jos_results
    
    # Save results
    with open('benchmark_results_phase4_a2dmrg.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'#'*80}")
    print(f"# PHASE 4 A2DMRG COMPLETE!")
    print(f"# ALL BENCHMARKS COMPLETE!")
    print(f"{'#'*80}\n")
    return 0

if __name__ == '__main__':
    sys.exit(main())
