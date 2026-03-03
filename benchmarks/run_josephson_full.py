#!/usr/bin/env python3
"""
Full Josephson Junction Array Benchmark - No Timeouts

Runs all methods with consistent convergence criteria:
- quimb DMRG1 (reference)
- quimb DMRG2 (reference)
- PDMRG np=1,2,4,8
- A2DMRG np=1,2,4,8

All methods use the same convergence criteria (tol=1e-10).
Results saved to josephson_full_results.json
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime

import numpy as np

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'a2dmrg'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pdmrg'))

import quimb.tensor as qtn

# ============================================================================
# Physical Parameters
# ============================================================================

L = 20              # Number of junctions
E_C = 1.0           # Charging energy
E_J = 2.0           # Josephson energy
N_G = 0.0           # Gate charge offset
N_MAX = 2           # Charge truncation (d=5)
PHI_EXT = np.pi / 4 # External flux (requires complex128)

# DMRG parameters - consistent across all methods
BOND_DIM = 50
MAX_SWEEPS = 200    # Effectively unlimited - convergence decides
TOL = 1e-10         # Convergence tolerance
CUTOFF = 1e-14      # SVD cutoff

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ============================================================================
# MPO Construction
# ============================================================================

def build_josephson_mpo():
    """Build Josephson junction array MPO."""
    d = 2 * N_MAX + 1
    S = (d - 1) / 2
    
    # Charge operators
    charges = np.arange(-N_MAX, N_MAX + 1, dtype='complex128')
    n_op = np.diag(charges)
    
    # Phase operators
    exp_iphi = np.zeros((d, d), dtype='complex128')
    for i in range(d - 1):
        exp_iphi[i + 1, i] = 1.0
    exp_miphi = exp_iphi.conj().T
    
    # Build MPO
    builder = qtn.SpinHam1D(S=S)
    flux_phase = np.exp(1j * PHI_EXT)
    builder.add_term(-E_J / 2 * flux_phase, exp_iphi, exp_miphi)
    builder.add_term(-E_J / 2 * np.conj(flux_phase), exp_miphi, exp_iphi)
    n_squared = n_op @ n_op
    builder.add_term(E_C, n_squared)
    
    return builder.build_mpo(L)


# ============================================================================
# Reference Calculations
# ============================================================================

def run_quimb_dmrg1(mpo):
    """Run quimb DMRG1 (1-site)."""
    print("Running quimb DMRG1...", flush=True)
    t0 = time.time()
    dmrg = qtn.DMRG1(mpo, bond_dims=BOND_DIM, cutoffs=CUTOFF)
    dmrg.solve(max_sweeps=MAX_SWEEPS, tol=TOL, verbosity=0)
    t1 = time.time()
    return {'energy': float(dmrg.energy.real), 'time': t1 - t0, 'sweeps': dmrg.sweep}


def run_quimb_dmrg2(mpo):
    """Run quimb DMRG2 (2-site)."""
    print("Running quimb DMRG2...", flush=True)
    t0 = time.time()
    dmrg = qtn.DMRG2(mpo, bond_dims=BOND_DIM, cutoffs=CUTOFF)
    dmrg.solve(max_sweeps=MAX_SWEEPS, tol=TOL, verbosity=0)
    t1 = time.time()
    return {'energy': float(dmrg.energy.real), 'time': t1 - t0, 'sweeps': dmrg.sweep}


# ============================================================================
# PDMRG
# ============================================================================

def run_pdmrg(np_count):
    """Run PDMRG with given processor count."""
    print(f"Running PDMRG np={np_count}...", flush=True)
    
    script = f'''
import sys
sys.path.insert(0, '{BASE_DIR}/pdmrg')
import numpy as np
import quimb.tensor as qtn
from mpi4py import MPI
import time

# Build MPO
n_max = {N_MAX}
d = 2 * n_max + 1
S = (d - 1) / 2

charges = np.arange(-n_max, n_max + 1, dtype='complex128')
n_op = np.diag(charges)
exp_iphi = np.zeros((d, d), dtype='complex128')
for i in range(d - 1):
    exp_iphi[i + 1, i] = 1.0
exp_miphi = exp_iphi.conj().T

builder = qtn.SpinHam1D(S=S)
flux_phase = np.exp(1j * {PHI_EXT})
builder.add_term(-{E_J} / 2 * flux_phase, exp_iphi, exp_miphi)
builder.add_term(-{E_J} / 2 * np.conj(flux_phase), exp_miphi, exp_iphi)
n_squared = n_op @ n_op
builder.add_term({E_C}, n_squared)
mpo = builder.build_mpo({L})

# Run PDMRG
from pdmrg.dmrg import pdmrg_main
t0 = time.time()
energy, mps = pdmrg_main(
    L={L},
    mpo=mpo,
    bond_dim={BOND_DIM},
    bond_dim_warmup={BOND_DIM},
    max_sweeps={MAX_SWEEPS},
    tol={TOL},
    n_warmup_sweeps=5,
    dtype='complex128',
    comm=MPI.COMM_WORLD,
    verbose=False
)
t1 = time.time()

if MPI.COMM_WORLD.Get_rank() == 0:
    print(f"RESULT:energy={{energy.real:.15f}},time={{t1-t0:.2f}}")
'''
    
    script_path = f'/tmp/pdmrg_jj_np{np_count}.py'
    with open(script_path, 'w') as f:
        f.write(script)
    
    env = os.environ.copy()
    env['PATH'] = '/usr/lib64/openmpi/bin:' + env.get('PATH', '')
    env['LD_LIBRARY_PATH'] = '/usr/lib64/openmpi/lib:' + env.get('LD_LIBRARY_PATH', '')
    
    result = subprocess.run(
        ['mpirun', '-np', str(np_count), '--oversubscribe',
         '--mca', 'btl', 'tcp,self', '--mca', 'btl_tcp_if_include', 'lo',
         sys.executable, script_path],
        capture_output=True, text=True, env=env
    )
    
    # Parse result
    for line in result.stdout.split('\n'):
        if line.startswith('RESULT:'):
            parts = line[7:].split(',')
            energy = float(parts[0].split('=')[1])
            time_s = float(parts[1].split('=')[1])
            return {'energy': energy, 'time': time_s}
    
    print(f"PDMRG np={np_count} FAILED:")
    print(result.stderr[:500] if result.stderr else "No stderr")
    return None


# ============================================================================
# A2DMRG
# ============================================================================

def run_a2dmrg(np_count):
    """Run A2DMRG with given processor count."""
    print(f"Running A2DMRG np={np_count}...", flush=True)
    
    script = f'''
import sys
sys.path.insert(0, '{BASE_DIR}/a2dmrg')
import numpy as np
import quimb.tensor as qtn
from mpi4py import MPI
import time

# Build MPO
n_max = {N_MAX}
d = 2 * n_max + 1
S = (d - 1) / 2

charges = np.arange(-n_max, n_max + 1, dtype='complex128')
n_op = np.diag(charges)
exp_iphi = np.zeros((d, d), dtype='complex128')
for i in range(d - 1):
    exp_iphi[i + 1, i] = 1.0
exp_miphi = exp_iphi.conj().T

builder = qtn.SpinHam1D(S=S)
flux_phase = np.exp(1j * {PHI_EXT})
builder.add_term(-{E_J} / 2 * flux_phase, exp_iphi, exp_miphi)
builder.add_term(-{E_J} / 2 * np.conj(flux_phase), exp_miphi, exp_iphi)
n_squared = n_op @ n_op
builder.add_term({E_C}, n_squared)
mpo = builder.build_mpo({L})

# Run A2DMRG
from a2dmrg.dmrg import a2dmrg_main
t0 = time.time()
energy, mps = a2dmrg_main(
    L={L},
    mpo=mpo,
    max_sweeps={MAX_SWEEPS},
    bond_dim={BOND_DIM},
    tol={TOL},
    warmup_sweeps=5,
    dtype=np.complex128,
    comm=MPI.COMM_WORLD,
    verbose=False
)
t1 = time.time()

if MPI.COMM_WORLD.Get_rank() == 0:
    print(f"RESULT:energy={{energy.real:.15f}},time={{t1-t0:.2f}}")
'''
    
    script_path = f'/tmp/a2dmrg_jj_np{np_count}.py'
    with open(script_path, 'w') as f:
        f.write(script)
    
    env = os.environ.copy()
    env['PATH'] = '/usr/lib64/openmpi/bin:' + env.get('PATH', '')
    env['LD_LIBRARY_PATH'] = '/usr/lib64/openmpi/lib:' + env.get('LD_LIBRARY_PATH', '')
    
    result = subprocess.run(
        ['mpirun', '-np', str(np_count), '--oversubscribe',
         '--mca', 'btl', 'tcp,self', '--mca', 'btl_tcp_if_include', 'lo',
         sys.executable, script_path],
        capture_output=True, text=True, env=env
    )
    
    # Parse result
    for line in result.stdout.split('\n'):
        if line.startswith('RESULT:'):
            parts = line[7:].split(',')
            energy = float(parts[0].split('=')[1])
            time_s = float(parts[1].split('=')[1])
            return {'energy': energy, 'time': time_s}
    
    print(f"A2DMRG np={np_count} FAILED:")
    print(result.stderr[:500] if result.stderr else "No stderr")
    return None


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("JOSEPHSON JUNCTION ARRAY - FULL BENCHMARK")
    print("=" * 70)
    print(f"Parameters: L={L}, E_J/E_C={E_J/E_C}, n_max={N_MAX}, bond_dim={BOND_DIM}")
    print(f"Convergence: tol={TOL}, max_sweeps={MAX_SWEEPS}")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)
    print()
    
    results = {
        'parameters': {
            'L': L, 'E_C': E_C, 'E_J': E_J, 'n_max': N_MAX,
            'phi_ext': PHI_EXT, 'bond_dim': BOND_DIM,
            'tol': TOL, 'max_sweeps': MAX_SWEEPS
        },
        'timestamp': datetime.now().isoformat(),
        'quimb1': None,
        'quimb2': None,
        'pdmrg': {},
        'a2dmrg': {}
    }
    
    # Build MPO
    print("Building MPO...", flush=True)
    mpo = build_josephson_mpo()
    print(f"  MPO dtype: {mpo[0].data.dtype}")
    print()
    
    # Reference calculations
    print("-" * 70)
    print("REFERENCE CALCULATIONS")
    print("-" * 70)
    
    results['quimb1'] = run_quimb_dmrg1(mpo)
    print(f"  DMRG1: E = {results['quimb1']['energy']:.12f}, time = {results['quimb1']['time']:.2f}s")
    
    results['quimb2'] = run_quimb_dmrg2(mpo)
    print(f"  DMRG2: E = {results['quimb2']['energy']:.12f}, time = {results['quimb2']['time']:.2f}s")
    print()
    
    ref_energy = results['quimb2']['energy']  # Use DMRG2 as reference
    
    # PDMRG
    print("-" * 70)
    print("PDMRG RESULTS")
    print("-" * 70)
    for np_count in [1, 2, 4, 8]:
        res = run_pdmrg(np_count)
        if res:
            delta = res['energy'] - ref_energy
            results['pdmrg'][f'np{np_count}'] = res
            results['pdmrg'][f'np{np_count}']['delta_e'] = delta
            print(f"  np={np_count}: E = {res['energy']:.12f}, ΔE = {delta:.2e}, time = {res['time']:.2f}s")
        else:
            print(f"  np={np_count}: FAILED")
    print()
    
    # A2DMRG
    print("-" * 70)
    print("A2DMRG RESULTS")
    print("-" * 70)
    for np_count in [1, 2, 4, 8]:
        res = run_a2dmrg(np_count)
        if res:
            delta = res['energy'] - ref_energy
            results['a2dmrg'][f'np{np_count}'] = res
            results['a2dmrg'][f'np{np_count}']['delta_e'] = delta
            print(f"  np={np_count}: E = {res['energy']:.12f}, ΔE = {delta:.2e}, time = {res['time']:.2f}s")
        else:
            print(f"  np={np_count}: FAILED")
    print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Reference (DMRG2): E = {ref_energy:.12f}")
    print()
    print("Method        np   Energy              ΔE           Time")
    print("-" * 70)
    
    for method in ['pdmrg', 'a2dmrg']:
        for key in ['np1', 'np2', 'np4', 'np8']:
            if key in results[method]:
                r = results[method][key]
                np_val = key[2:]
                print(f"{method.upper():12}  {np_val:2}   {r['energy']:.12f}   {r['delta_e']:+.2e}   {r['time']:.2f}s")
    
    # Save results
    output_path = os.path.join(os.path.dirname(__file__), 'josephson_full_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print()
    print(f"Results saved to: {output_path}")
    print(f"Finished: {datetime.now().isoformat()}")


if __name__ == '__main__':
    main()
