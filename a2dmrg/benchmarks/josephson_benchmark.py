#!/usr/bin/env python3
"""
Josephson Junction Array: PDMRG vs A2DMRG Benchmark

This benchmark compares parallel DMRG implementations on a physically
realistic superconducting quantum computing model with complex128 coefficients.

Requirements:
- All parallel results must match serial quimb DMRG2 to ~1e-10
- Total time to solution (including warmup/finalization)
- np = 1, 2, 4, 8 cores
"""

import subprocess
import time
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.josephson_junction import build_josephson_mpo, verify_complex_dtype


def run_serial_reference(L, bond_dim, n_max=2, max_sweeps=20, tol=1e-14):
    """Run serial quimb DMRG2 as reference."""
    import quimb.tensor as qtn
    
    print("  Building Josephson Junction MPO...")
    mpo = build_josephson_mpo(L, E_J=1.0, E_C=0.5, n_max=n_max, with_flux=True)
    
    is_complex, msg = verify_complex_dtype(mpo)
    print(f"  {msg}")
    
    print("  Running quimb DMRG2...")
    t0 = time.time()
    dmrg = qtn.DMRG2(mpo, bond_dims=bond_dim)
    dmrg.solve(max_sweeps=max_sweeps, tol=tol, verbosity=0)
    t1 = time.time()
    
    energy = float(np.real(dmrg.energy))
    elapsed = t1 - t0
    
    return {'energy': energy, 'time': elapsed, 'mpo': mpo}


def run_pdmrg(L, bond_dim, np_count, n_max=2, max_sweeps=20, tol=1e-14):
    """Run PDMRG with specified number of processes."""
    # PDMRG needs to be called with the josephson model
    # We'll create a temporary script that builds the model
    
    script = f'''
import sys
import time
import numpy as np
sys.path.insert(0, '/home/captain/clawd/work/dmrg-implementations/a2dmrg')
from benchmarks.josephson_junction import build_josephson_mpo

from a2dmrg.mpi_compat import MPI, HAS_MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Build MPO
mpo = build_josephson_mpo({L}, E_J=1.0, E_C=0.5, n_max={n_max}, with_flux=True)

# Import and run PDMRG
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
    verbose=(rank == 0)
)
t1 = time.time()

if rank == 0:
    print(f"PDMRG_RESULT:{{np.real(energy)}}:{{t1-t0}}")
'''
    
    # Write temp script
    script_path = '/tmp/pdmrg_josephson_test.py'
    with open(script_path, 'w') as f:
        f.write(script)
    
    # Use PDMRG's virtual environment Python
    pdmrg_python = '/home/captain/clawd/work/dmrg-implementations/pdmrg/venv/bin/python'
    
    cmd = f'''
source /etc/profile.d/modules.sh && module load mpi/openmpi-x86_64 && \
mpirun -np {np_count} --oversubscribe --mca btl tcp,self --mca btl_tcp_if_include lo \
{pdmrg_python} {script_path} 2>&1
'''
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1800)
        output = result.stdout + result.stderr
        
        # Parse result
        for line in output.split('\n'):
            if 'PDMRG_RESULT:' in line:
                parts = line.split(':')
                energy = float(parts[1])
                elapsed = float(parts[2])
                return {'energy': energy, 'time': elapsed, 'success': True}
        
        return {'energy': None, 'time': None, 'success': False, 'output': output}
    except Exception as e:
        return {'energy': None, 'time': None, 'success': False, 'error': str(e)}


def run_a2dmrg(L, bond_dim, np_count, n_max=2, max_sweeps=20, warmup_sweeps=2, tol=1e-14):
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

# Build MPO
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
    warmup_sweeps={warmup_sweeps},
    one_site=True,
    verbose=(rank == 0)
)
t1 = time.time()

if rank == 0:
    print(f"A2DMRG_RESULT:{{np.real(energy)}}:{{t1-t0}}")
'''
    
    script_path = '/tmp/a2dmrg_josephson_test.py'
    with open(script_path, 'w') as f:
        f.write(script)
    
    # Use A2DMRG's virtual environment Python
    a2dmrg_python = '/home/captain/clawd/work/dmrg-implementations/a2dmrg/venv/bin/python'
    
    cmd = f'''
source /etc/profile.d/modules.sh && module load mpi/openmpi-x86_64 && \
mpirun -np {np_count} --oversubscribe --mca btl tcp,self --mca btl_tcp_if_include lo \
{a2dmrg_python} {script_path} 2>&1
'''
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1800)
        output = result.stdout + result.stderr
        
        for line in output.split('\n'):
            if 'A2DMRG_RESULT:' in line:
                parts = line.split(':')
                energy = float(parts[1])
                elapsed = float(parts[2])
                return {'energy': energy, 'time': elapsed, 'success': True}
        
        return {'energy': None, 'time': None, 'success': False, 'output': output}
    except Exception as e:
        return {'energy': None, 'time': None, 'success': False, 'error': str(e)}


def main():
    # Benchmark parameters
    L = 20
    bond_dim = 50
    n_max = 2  # d = 5
    np_list = [1, 2, 4, 8]
    
    print("=" * 70)
    print("Josephson Junction Array Benchmark")
    print(f"L={L}, D={bond_dim}, d={2*n_max+1}, complex128")
    print("=" * 70)
    
    # Serial reference
    print("\n[1/3] Serial Reference (quimb DMRG2)")
    print("-" * 50)
    ref = run_serial_reference(L, bond_dim, n_max)
    E_ref = ref['energy']
    print(f"  Energy: {E_ref:.12f}")
    print(f"  Time: {ref['time']:.2f}s")
    
    # PDMRG
    print("\n[2/3] PDMRG Scaling")
    print("-" * 50)
    pdmrg_results = []
    for np_count in np_list:
        print(f"  np={np_count}...", end=" ", flush=True)
        res = run_pdmrg(L, bond_dim, np_count, n_max)
        if res['success']:
            diff = abs(res['energy'] - E_ref)
            print(f"E={res['energy']:.10f}, t={res['time']:.2f}s, diff={diff:.2e}")
            res['energy_diff'] = diff
        else:
            print("FAILED")
        res['np'] = np_count
        pdmrg_results.append(res)
    
    # A2DMRG
    print("\n[3/3] A2DMRG Scaling")
    print("-" * 50)
    a2dmrg_results = []
    for np_count in np_list:
        print(f"  np={np_count}...", end=" ", flush=True)
        res = run_a2dmrg(L, bond_dim, np_count, n_max)
        if res['success']:
            diff = abs(res['energy'] - E_ref)
            print(f"E={res['energy']:.10f}, t={res['time']:.2f}s, diff={diff:.2e}")
            res['energy_diff'] = diff
        else:
            print("FAILED")
        res['np'] = np_count
        a2dmrg_results.append(res)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Josephson Junction (L={}, D={}, complex128)".format(L, bond_dim))
    print("=" * 70)
    print(f"Reference Energy: {E_ref:.12f}")
    print()
    print(f"{'np':>4} | {'PDMRG Time':>12} | {'PDMRG ΔE':>12} | {'A2DMRG Time':>12} | {'A2DMRG ΔE':>12}")
    print("-" * 70)
    
    for i, np_count in enumerate(np_list):
        pd = pdmrg_results[i]
        a2 = a2dmrg_results[i]
        
        pd_t = f"{pd['time']:.2f}s" if pd['success'] else "FAIL"
        pd_e = f"{pd.get('energy_diff', 0):.2e}" if pd['success'] else "N/A"
        a2_t = f"{a2['time']:.2f}s" if a2['success'] else "FAIL"
        a2_e = f"{a2.get('energy_diff', 0):.2e}" if a2['success'] else "N/A"
        
        print(f"{np_count:>4} | {pd_t:>12} | {pd_e:>12} | {a2_t:>12} | {a2_e:>12}")
    
    # Save results
    results = {
        'benchmark': 'josephson_junction',
        'L': L,
        'bond_dim': bond_dim,
        'n_max': n_max,
        'd': 2 * n_max + 1,
        'dtype': 'complex128',
        'reference_energy': E_ref,
        'reference_time': ref['time'],
        'pdmrg': pdmrg_results,
        'a2dmrg': a2dmrg_results,
    }
    
    output_file = 'benchmarks/josephson_benchmark_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")


if __name__ == '__main__':
    main()
