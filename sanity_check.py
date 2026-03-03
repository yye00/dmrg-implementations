#!/usr/bin/env python3
"""
Quick Sanity Check: PDMRG and A2DMRG (np=1) vs quimb DMRG1/DMRG2

Single-process implementations should not be significantly slower than quimb.
"""

import subprocess
import time
import sys
import os

# Configuration
L = 20
BOND_DIM = 50
N_MAX = 2  # d = 5 local dimension
MAX_SWEEPS = 20
TOL = 1e-10

def run_quimb_benchmarks():
    """Run quimb DMRG1 and DMRG2 as references."""
    sys.path.insert(0, '/home/captain/clawd/work/dmrg-implementations/a2dmrg')
    from benchmarks.josephson_junction import build_josephson_mpo
    import quimb.tensor as qtn
    import numpy as np
    
    print(f"\nBuilding Josephson Junction MPO: L={L}, n_max={N_MAX}, d={2*N_MAX+1}")
    mpo = build_josephson_mpo(L, E_J=1.0, E_C=0.5, n_max=N_MAX, with_flux=True)
    
    results = {}
    
    # DMRG1
    print("\n[quimb DMRG1] Running...")
    t0 = time.time()
    dmrg1 = qtn.DMRG1(mpo, bond_dims=BOND_DIM)
    dmrg1.solve(max_sweeps=MAX_SWEEPS, tol=TOL, verbosity=1)
    t1 = time.time()
    results['quimb_dmrg1'] = {
        'energy': float(np.real(dmrg1.energy)),
        'time': t1 - t0
    }
    print(f"  Energy: {results['quimb_dmrg1']['energy']:.12f}")
    print(f"  Time:   {results['quimb_dmrg1']['time']:.2f}s")
    
    # DMRG2
    print("\n[quimb DMRG2] Running...")
    t0 = time.time()
    dmrg2 = qtn.DMRG2(mpo, bond_dims=BOND_DIM)
    dmrg2.solve(max_sweeps=MAX_SWEEPS, tol=TOL, verbosity=1)
    t1 = time.time()
    results['quimb_dmrg2'] = {
        'energy': float(np.real(dmrg2.energy)),
        'time': t1 - t0
    }
    print(f"  Energy: {results['quimb_dmrg2']['energy']:.12f}")
    print(f"  Time:   {results['quimb_dmrg2']['time']:.2f}s")
    
    return results

def run_pdmrg_np1():
    """Run PDMRG with np=1."""
    script = f'''
import sys
import time
import numpy as np
sys.path.insert(0, '/home/captain/clawd/work/dmrg-implementations/a2dmrg')
from benchmarks.josephson_junction import build_josephson_mpo

from mpi4py import MPI
comm = MPI.COMM_WORLD

mpo = build_josephson_mpo({L}, E_J=1.0, E_C=0.5, n_max={N_MAX}, with_flux=True)

sys.path.insert(0, '/home/captain/clawd/work/dmrg-implementations/pdmrg')
from pdmrg.dmrg import pdmrg_main

t0 = time.time()
energy, mps = pdmrg_main(
    L={L},
    mpo=mpo,
    bond_dim={BOND_DIM},
    max_sweeps={MAX_SWEEPS},
    tol={TOL},
    dtype='complex128',
    comm=comm,
    verbose=True
)
t1 = time.time()

print(f"RESULT:{{np.real(energy):.15f}}:{{t1-t0:.4f}}")
'''
    
    script_path = '/tmp/pdmrg_sanity.py'
    with open(script_path, 'w') as f:
        f.write(script)
    
    pdmrg_python = '/home/captain/clawd/work/dmrg-implementations/pdmrg/venv/bin/python'
    
    cmd = f'''
source /etc/profile.d/modules.sh && module load mpi/openmpi-x86_64 && \
mpirun -np 1 --oversubscribe --mca btl tcp,self --mca btl_tcp_if_include lo \
{pdmrg_python} -u {script_path} 2>&1
'''
    
    print("\n[PDMRG np=1] Running...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
    output = result.stdout + result.stderr
    print(output)
    
    for line in output.split('\n'):
        if line.startswith('RESULT:'):
            parts = line.split(':')
            return {
                'energy': float(parts[1]),
                'time': float(parts[2])
            }
    return None

def run_a2dmrg_np1():
    """Run A2DMRG with np=1."""
    script = f'''
import sys
import time
import numpy as np
sys.path.insert(0, '/home/captain/clawd/work/dmrg-implementations/a2dmrg')
from benchmarks.josephson_junction import build_josephson_mpo
from a2dmrg.dmrg import a2dmrg_main

from mpi4py import MPI
comm = MPI.COMM_WORLD

mpo = build_josephson_mpo({L}, E_J=1.0, E_C=0.5, n_max={N_MAX}, with_flux=True)

t0 = time.time()
energy, mps = a2dmrg_main(
    L={L},
    mpo=mpo,
    max_sweeps={MAX_SWEEPS},
    bond_dim={BOND_DIM},
    tol={TOL},
    dtype=np.complex128,
    comm=comm,
    warmup_sweeps=2,
    one_site=True,
    verbose=True
)
t1 = time.time()

print(f"RESULT:{{np.real(energy):.15f}}:{{t1-t0:.4f}}")
'''
    
    script_path = '/tmp/a2dmrg_sanity.py'
    with open(script_path, 'w') as f:
        f.write(script)
    
    a2dmrg_python = '/home/captain/clawd/work/dmrg-implementations/a2dmrg/venv/bin/python'
    
    cmd = f'''
source /etc/profile.d/modules.sh && module load mpi/openmpi-x86_64 && \
mpirun -np 1 --oversubscribe --mca btl tcp,self --mca btl_tcp_if_include lo \
{a2dmrg_python} -u {script_path} 2>&1
'''
    
    print("\n[A2DMRG np=1] Running...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
    output = result.stdout + result.stderr
    print(output)
    
    for line in output.split('\n'):
        if line.startswith('RESULT:'):
            parts = line.split(':')
            return {
                'energy': float(parts[1]),
                'time': float(parts[2])
            }
    return None

def main():
    print("=" * 70)
    print("DMRG SANITY CHECK: Single-process performance comparison")
    print("=" * 70)
    print(f"Parameters: L={L}, bond_dim={BOND_DIM}, n_max={N_MAX}, tol={TOL}")
    
    # Run quimb first
    quimb_results = run_quimb_benchmarks()
    E_ref = quimb_results['quimb_dmrg2']['energy']
    
    # Run PDMRG np=1
    pdmrg_result = run_pdmrg_np1()
    
    # Run A2DMRG np=1
    a2dmrg_result = run_a2dmrg_np1()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Method':<20} | {'Energy':>18} | {'ΔE':>12} | {'Time':>10}")
    print("-" * 70)
    
    print(f"{'quimb DMRG1':<20} | {quimb_results['quimb_dmrg1']['energy']:>18.12f} | "
          f"{abs(quimb_results['quimb_dmrg1']['energy'] - E_ref):>12.2e} | "
          f"{quimb_results['quimb_dmrg1']['time']:>9.1f}s")
    
    print(f"{'quimb DMRG2':<20} | {quimb_results['quimb_dmrg2']['energy']:>18.12f} | "
          f"{'(ref)':>12} | "
          f"{quimb_results['quimb_dmrg2']['time']:>9.1f}s")
    
    if pdmrg_result:
        print(f"{'PDMRG np=1':<20} | {pdmrg_result['energy']:>18.12f} | "
              f"{abs(pdmrg_result['energy'] - E_ref):>12.2e} | "
              f"{pdmrg_result['time']:>9.1f}s")
    else:
        print(f"{'PDMRG np=1':<20} | {'FAILED':>18}")
    
    if a2dmrg_result:
        print(f"{'A2DMRG np=1':<20} | {a2dmrg_result['energy']:>18.12f} | "
              f"{abs(a2dmrg_result['energy'] - E_ref):>12.2e} | "
              f"{a2dmrg_result['time']:>9.1f}s")
    else:
        print(f"{'A2DMRG np=1':<20} | {'FAILED':>18}")
    
    print("-" * 70)
    
    # Performance analysis
    if pdmrg_result and a2dmrg_result:
        quimb_time = quimb_results['quimb_dmrg2']['time']
        pdmrg_ratio = pdmrg_result['time'] / quimb_time
        a2dmrg_ratio = a2dmrg_result['time'] / quimb_time
        
        print(f"\nPerformance vs quimb DMRG2:")
        print(f"  PDMRG np=1:  {pdmrg_ratio:.1f}x slower")
        print(f"  A2DMRG np=1: {a2dmrg_ratio:.1f}x slower")
        
        if pdmrg_ratio > 3:
            print(f"\n⚠️  PDMRG np=1 is significantly slower than expected!")
        if a2dmrg_ratio > 3:
            print(f"\n⚠️  A2DMRG np=1 is significantly slower than expected!")
    
    print("\n" + "=" * 70)

if __name__ == '__main__':
    main()
