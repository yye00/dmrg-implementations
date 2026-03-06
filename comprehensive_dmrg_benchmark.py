#!/usr/bin/env python3
"""
Comprehensive DMRG Benchmark Suite
Validates all implementations against Quimb DMRG1/2 with strict tolerance checking
"""
import json
import time
import sys
import os
import subprocess
import numpy as np
from datetime import datetime

# Add paths
REPO_ROOT = os.path.expanduser("~/dmrg-implementations")
sys.path.insert(0, os.path.join(REPO_ROOT, 'pdmrg'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'pdmrg2'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'a2dmrg'))

import quimb.tensor as qtn

# Validation tolerance
VALIDATION_TOL = 1e-10
N_REPETITIONS = 5

# Problem definitions
HEISENBERG_SHORT = {"name": "Heisenberg-Short", "L": 12, "D": 100, "sweeps": 20, "tol": 1e-10}
JOSEPHSON_SHORT = {"name": "Josephson-Short", "L": 8, "D": 50, "n_max": 2, "sweeps": 20, "tol": 1e-10}

def build_josephson_mpo(L, n_max=2):
    """Build Josephson junction MPO (Bose-Hubbard)."""
    sys.path.insert(0, os.path.join(REPO_ROOT, 'a2dmrg'))
    from benchmarks.josephson_junction import build_josephson_mpo as build_mpo
    return build_mpo(L, E_J=1.0, E_C=0.5, n_max=n_max, with_flux=True)

def run_quimb_dmrg(mpo, bond_dim, max_sweeps, tol, method='DMRG1'):
    """Run Quimb DMRG1 or DMRG2."""
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
    """Run PDMRG or PDMRG2 with MPI."""
    script = f'''
import sys, time, json
import numpy as np
sys.path.insert(0, "{REPO_ROOT}/{impl}")
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Build MPO
{mpo_builder}

# Import and run
from pdmrg.dmrg import pdmrg_main

t0 = time.time()
energy, mps = pdmrg_main(
    L={params['L']},
    mpo=mpo,
    bond_dim={params['D']},
    max_sweeps={params['sweeps']},
    tol={params['tol']},
    dtype="{params.get('dtype', 'float64')}",
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
        # Find JSON line
        for line in result.stdout.strip().split('\n'):
            if line.startswith('{'):
                data = json.loads(line)
                return data['energy'], data['time'], None
        return None, None, "No JSON output found"
    except Exception as e:
        return None, None, str(e)

def validate_energy(test_energy, reference_energy, test_name, reference_name="Quimb"):
    """Validate energy against reference."""
    diff = abs(test_energy - reference_energy)
    if diff > VALIDATION_TOL:
        print(f"\n{'='*80}")
        print(f"❌ VALIDATION FAILURE!")
        print(f"{'='*80}")
        print(f"Test: {test_name}")
        print(f"Reference ({reference_name}): {reference_energy:.15f}")
        print(f"Test energy: {test_energy:.15f}")
        print(f"Difference: {diff:.2e} (tolerance: {VALIDATION_TOL:.2e})")
        print(f"{'='*80}")
        return False
    return True

def run_heisenberg_benchmarks(problem, phase_name):
    """Run Heisenberg benchmarks for one problem size."""
    print(f"\n{'='*80}")
    print(f"  {phase_name}: {problem['name']} (L={problem['L']}, D={problem['D']})")
    print(f"{'='*80}")
    
    results = {}
    mpo = qtn.MPO_ham_heis(L=problem['L'], j=1.0, bz=0.0, cyclic=False)
    
    # Step 1: Run Quimb DMRG1/2 (reference)
    print(f"\n[1/2] Running Quimb DMRG1 (reference)...")
    dmrg1_energies = []
    dmrg1_times = []
    for rep in range(N_REPETITIONS):
        energy, elapsed = run_quimb_dmrg(mpo, problem['D'], problem['sweeps'], problem['tol'], 'DMRG1')
        dmrg1_energies.append(energy)
        dmrg1_times.append(elapsed)
        print(f"  Rep {rep+1}/{N_REPETITIONS}: E={energy:.12f}, time={elapsed:.3f}s")
    
    dmrg1_energy_ref = np.mean(dmrg1_energies)
    results['quimb_dmrg1'] = {
        'energies': dmrg1_energies,
        'times': dmrg1_times,
        'energy_mean': dmrg1_energy_ref,
        'time_mean': np.mean(dmrg1_times),
        'time_std': np.std(dmrg1_times)
    }
    
    print(f"\n[2/2] Running Quimb DMRG2 (reference)...")
    dmrg2_energies = []
    dmrg2_times = []
    for rep in range(N_REPETITIONS):
        energy, elapsed = run_quimb_dmrg(mpo, problem['D'], problem['sweeps'], problem['tol'], 'DMRG2')
        dmrg2_energies.append(energy)
        dmrg2_times.append(elapsed)
        print(f"  Rep {rep+1}/{N_REPETITIONS}: E={energy:.12f}, time={elapsed:.3f}s")
        
        # Validate DMRG2 vs DMRG1
        if not validate_energy(energy, dmrg1_energy_ref, "Quimb DMRG2", "Quimb DMRG1"):
            return None
    
    dmrg2_energy_ref = np.mean(dmrg2_energies)
    results['quimb_dmrg2'] = {
        'energies': dmrg2_energies,
        'times': dmrg2_times,
        'energy_mean': dmrg2_energy_ref,
        'time_mean': np.mean(dmrg2_times),
        'time_std': np.std(dmrg2_times)
    }
    
    print(f"\n✓ Quimb references: DMRG1={dmrg1_energy_ref:.12f}, DMRG2={dmrg2_energy_ref:.12f}")
    
    # Step 2: Run PDMRG (np=1,2,4,8) if in Phase 1
    if 'Phase 1' in phase_name:
        for np_val in [1, 2, 4, 8]:
            print(f"\n[PDMRG np={np_val}] Running...")
            mpo_builder = f"import quimb.tensor as qtn\nmpo = qtn.MPO_ham_heis(L={problem['L']}, j=1.0, bz=0.0, cyclic=False)"
            
            pdmrg_energies = []
            pdmrg_times = []
            for rep in range(N_REPETITIONS):
                energy, elapsed, error = run_pdmrg_mpi(mpo_builder, np_val, 'pdmrg', **problem, dtype='float64')
                if error:
                    print(f"  ❌ Error: {error}")
                    return None
                pdmrg_energies.append(energy)
                pdmrg_times.append(elapsed)
                print(f"  Rep {rep+1}/{N_REPETITIONS}: E={energy:.12f}, time={elapsed:.3f}s")
                
                # Validate against Quimb
                if not validate_energy(energy, dmrg2_energy_ref, f"PDMRG np={np_val}"):
                    return None
            
            results[f'pdmrg_np{np_val}'] = {
                'energies': pdmrg_energies,
                'times': pdmrg_times,
                'energy_mean': np.mean(pdmrg_energies),
                'time_mean': np.mean(pdmrg_times),
                'time_std': np.std(pdmrg_times)
            }
            print(f"  ✓ PDMRG np={np_val} validated!")
        
        # Step 3: Run PDMRG2 (np=1,2,4,8)
        for np_val in [1, 2, 4, 8]:
            print(f"\n[PDMRG2 np={np_val}] Running...")
            mpo_builder = f"import quimb.tensor as qtn\nmpo = qtn.MPO_ham_heis(L={problem['L']}, j=1.0, bz=0.0, cyclic=False)"
            
            pdmrg2_energies = []
            pdmrg2_times = []
            for rep in range(N_REPETITIONS):
                energy, elapsed, error = run_pdmrg_mpi(mpo_builder, np_val, 'pdmrg2', **problem, dtype='float64')
                if error:
                    print(f"  ❌ Error: {error}")
                    return None
                pdmrg2_energies.append(energy)
                pdmrg2_times.append(elapsed)
                print(f"  Rep {rep+1}/{N_REPETITIONS}: E={energy:.12f}, time={elapsed:.3f}s")
                
                # Validate against Quimb
                if not validate_energy(energy, dmrg2_energy_ref, f"PDMRG2 np={np_val}"):
                    return None
            
            results[f'pdmrg2_np{np_val}'] = {
                'energies': pdmrg2_energies,
                'times': pdmrg2_times,
                'energy_mean': np.mean(pdmrg2_energies),
                'time_mean': np.mean(pdmrg2_times),
                'time_std': np.std(pdmrg2_times)
            }
            print(f"  ✓ PDMRG2 np={np_val} validated!")
    
    return results

def main():
    print(f"{'#'*80}")
    print(f"# COMPREHENSIVE DMRG BENCHMARK SUITE")
    print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Validation tolerance: {VALIDATION_TOL}")
    print(f"# Repetitions per configuration: {N_REPETITIONS}")
    print(f"{'#'*80}")
    
    all_results = {
        'metadata': {
            'start_time': datetime.now().isoformat(),
            'validation_tol': VALIDATION_TOL,
            'n_repetitions': N_REPETITIONS
        }
    }
    
    # Phase 1: CPU benchmarks (Quimb + PDMRG + PDMRG2)
    print(f"\n{'#'*80}")
    print(f"# PHASE 1: CPU BENCHMARKS (Quimb DMRG1/2, PDMRG, PDMRG2)")
    print(f"{'#'*80}")
    
    # Heisenberg short
    heis_results = run_heisenberg_benchmarks(HEISENBERG_SHORT, "Phase 1: CPU")
    if heis_results is None:
        print("\n❌ BENCHMARK ABORTED DUE TO VALIDATION FAILURE")
        return 1
    all_results['heisenberg_short'] = heis_results
    
    # Save intermediate results
    with open('benchmark_results_phase1_heisenberg.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Phase 1 Heisenberg complete! Results saved.")
    
    # TODO: Add Josephson and GPU benchmarks here
    
    print(f"\n{'#'*80}")
    print(f"# BENCHMARK SUITE COMPLETE")
    print(f"{'#'*80}")
    return 0

if __name__ == '__main__':
    sys.exit(main())
