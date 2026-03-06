#!/usr/bin/env python3
"""Phase 2: Josephson CPU Benchmarks"""
import json
import time
import sys
import os
import subprocess
import numpy as np
from datetime import datetime

REPO_ROOT = os.path.expanduser("~/dmrg-implementations")
sys.path.insert(0, os.path.join(REPO_ROOT, 'pdmrg'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'pdmrg2'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'a2dmrg'))

import quimb.tensor as qtn

VALIDATION_TOL = 1e-10
N_REPETITIONS = 5
JOSEPHSON_SHORT = {"name": "Josephson-Short", "L": 8, "D": 50, "n_max": 2, "sweeps": 20, "tol": 1e-10}

def build_josephson_mpo(L, n_max=2):
    """Build Josephson junction MPO."""
    from benchmarks.josephson_junction import build_josephson_mpo as build_mpo
    return build_mpo(L, E_J=1.0, E_C=0.5, n_max=n_max, with_flux=True)

def run_quimb_dmrg(mpo, bond_dim, max_sweeps, tol, method='DMRG1'):
    """Run Quimb DMRG."""
    t0 = time.time()
    if method == 'DMRG1':
        dmrg = qtn.DMRG1(mpo, bond_dims=bond_dim, cutoffs=1e-14)
    else:
        dmrg = qtn.DMRG2(mpo, bond_dims=bond_dim, cutoffs=1e-14)
    dmrg.solve(max_sweeps=max_sweeps, tol=tol, verbosity=0)
    elapsed = time.time() - t0
    energy = float(np.real(dmrg.energy))
    return energy, elapsed

def run_pdmrg_mpi(mpo_builder, np_count, impl='pdmrg', **params):
    """Run PDMRG/PDMRG2 with MPI."""
    script = f'''
import sys, time, json
import numpy as np
sys.path.insert(0, "{REPO_ROOT}/{impl}")
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

{mpo_builder}

from pdmrg.dmrg import pdmrg_main

t0 = time.time()
energy, mps = pdmrg_main(
    L={params['L']},
    mpo=mpo,
    bond_dim={params['D']},
    max_sweeps={params['sweeps']},
    tol={params['tol']},
    dtype="{params.get('dtype', 'complex128')}",
    comm=comm,
    verbose=False
)
elapsed = time.time() - t0

if rank == 0:
    print(json.dumps({{"energy": float(np.real(energy)), "time": elapsed}}))
'''
    
    with open('/tmp/run_pdmrg_temp.py', 'w') as f:
        f.write(script)
    
    cmd = ['mpirun', '-np', str(np_count), '--oversubscribe', 
           'python3', '/tmp/run_pdmrg_temp.py']
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
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

def main():
    print(f"\n{'#'*80}")
    print(f"# PHASE 2: JOSEPHSON CPU BENCHMARKS")
    print(f"{'#'*80}\n")
    
    problem = JOSEPHSON_SHORT
    print(f"Problem: {problem['name']} (L={problem['L']}, D={problem['D']}, n_max={problem['n_max']})")
    
    results = {}
    mpo = build_josephson_mpo(problem['L'], problem['n_max'])
    
    # Quimb DMRG1
    print(f"\n[1/2] Quimb DMRG1 (reference)...")
    dmrg1_energies, dmrg1_times = [], []
    for rep in range(N_REPETITIONS):
        energy, elapsed = run_quimb_dmrg(mpo, problem['D'], problem['sweeps'], problem['tol'], 'DMRG1')
        dmrg1_energies.append(energy)
        dmrg1_times.append(elapsed)
        print(f"  Rep {rep+1}/{N_REPETITIONS}: E={energy:.12f}, time={elapsed:.3f}s")
    
    dmrg1_ref = np.mean(dmrg1_energies)
    results['quimb_dmrg1'] = {
        'energies': dmrg1_energies, 'times': dmrg1_times,
        'energy_mean': dmrg1_ref, 'time_mean': np.mean(dmrg1_times)
    }
    
    # Quimb DMRG2
    print(f"\n[2/2] Quimb DMRG2 (reference)...")
    dmrg2_energies, dmrg2_times = [], []
    for rep in range(N_REPETITIONS):
        energy, elapsed = run_quimb_dmrg(mpo, problem['D'], problem['sweeps'], problem['tol'], 'DMRG2')
        dmrg2_energies.append(energy)
        dmrg2_times.append(elapsed)
        print(f"  Rep {rep+1}/{N_REPETITIONS}: E={energy:.12f}, time={elapsed:.3f}s")
        if not validate_energy(energy, dmrg1_ref, "Quimb DMRG2"):
            return 1
    
    dmrg2_ref = np.mean(dmrg2_energies)
    results['quimb_dmrg2'] = {
        'energies': dmrg2_energies, 'times': dmrg2_times,
        'energy_mean': dmrg2_ref, 'time_mean': np.mean(dmrg2_times)
    }
    
    print(f"\n✓ Quimb references: DMRG1={dmrg1_ref:.12f}, DMRG2={dmrg2_ref:.12f}")
    
    # PDMRG
    mpo_builder = f'''sys.path.insert(0, "{REPO_ROOT}/a2dmrg")
from benchmarks.josephson_junction import build_josephson_mpo
mpo = build_josephson_mpo({problem['L']}, E_J=1.0, E_C=0.5, n_max={problem['n_max']}, with_flux=True)'''
    
    for np_val in [1, 2, 4, 8]:
        print(f"\n[PDMRG np={np_val}]...")
        pdmrg_energies, pdmrg_times = [], []
        for rep in range(N_REPETITIONS):
            energy, elapsed, error = run_pdmrg_mpi(mpo_builder, np_val, 'pdmrg', **problem, dtype='complex128')
            if error:
                print(f"  ❌ Error: {error}")
                return 1
            pdmrg_energies.append(energy)
            pdmrg_times.append(elapsed)
            print(f"  Rep {rep+1}/{N_REPETITIONS}: E={energy:.12f}, time={elapsed:.3f}s")
            if not validate_energy(energy, dmrg2_ref, f"PDMRG np={np_val}"):
                return 1
        
        results[f'pdmrg_np{np_val}'] = {
            'energies': pdmrg_energies, 'times': pdmrg_times,
            'energy_mean': np.mean(pdmrg_energies), 'time_mean': np.mean(pdmrg_times)
        }
        print(f"  ✓ Validated!")
    
    # PDMRG2
    for np_val in [1, 2, 4, 8]:
        print(f"\n[PDMRG2 np={np_val}]...")
        pdmrg2_energies, pdmrg2_times = [], []
        for rep in range(N_REPETITIONS):
            energy, elapsed, error = run_pdmrg_mpi(mpo_builder, np_val, 'pdmrg2', **problem, dtype='complex128')
            if error:
                print(f"  ❌ Error: {error}")
                return 1
            pdmrg2_energies.append(energy)
            pdmrg2_times.append(elapsed)
            print(f"  Rep {rep+1}/{N_REPETITIONS}: E={energy:.12f}, time={elapsed:.3f}s")
            if not validate_energy(energy, dmrg2_ref, f"PDMRG2 np={np_val}"):
                return 1
        
        results[f'pdmrg2_np{np_val}'] = {
            'energies': pdmrg2_energies, 'times': pdmrg2_times,
            'energy_mean': np.mean(pdmrg2_energies), 'time_mean': np.mean(pdmrg2_times)
        }
        print(f"  ✓ Validated!")
    
    # Save results
    output = {
        'metadata': {'start_time': datetime.now().isoformat(), 'validation_tol': VALIDATION_TOL, 'n_repetitions': N_REPETITIONS},
        'josephson_short': results
    }
    with open('benchmark_results_phase2_josephson.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Phase 2 Josephson complete! Results saved.\n")
    return 0

if __name__ == '__main__':
    sys.exit(main())
