#!/usr/bin/env python3
"""
Quick Correctness Check - verify all implementations match
Small system (L=10) for fast validation
"""

import subprocess
import time
import sys
import numpy as np

# Small configuration for quick testing
L = 10
BOND_DIM = 30  
N_MAX = 2
MAX_SWEEPS = 15
TOL = 1e-10

sys.path.insert(0, '/home/captain/clawd/work/dmrg-implementations/a2dmrg')
from benchmarks.josephson_junction import build_josephson_mpo
import quimb.tensor as qtn

print("=" * 70)
print("QUICK CORRECTNESS CHECK")
print("=" * 70)
print(f"L={L}, D={BOND_DIM}, n_max={N_MAX}, tol={TOL}")
print("=" * 70)

# Build MPO
print("\nBuilding Josephson Junction MPO...")
mpo = build_josephson_mpo(L, E_J=1.0, E_C=0.5, n_max=N_MAX, with_flux=True)

results = {}

# quimb DMRG2 (reference)
print("\n[1/4] quimb DMRG2 (reference)...")
t0 = time.time()
dmrg2 = qtn.DMRG2(mpo, bond_dims=BOND_DIM)
dmrg2.solve(max_sweeps=MAX_SWEEPS, tol=TOL, verbosity=0)
t1 = time.time()
E_ref = float(np.real(dmrg2.energy))
results['quimb_DMRG2'] = {'energy': E_ref, 'time': t1-t0}
print(f"  Energy: {E_ref:.12f}")
print(f"  Time:   {t1-t0:.1f}s")

# PDMRG np=1
print("\n[2/4] PDMRG np=1...")
script = f'''
import sys, time, numpy as np
sys.path.insert(0, '/home/captain/clawd/work/dmrg-implementations/a2dmrg')
from benchmarks.josephson_junction import build_josephson_mpo
from mpi4py import MPI
comm = MPI.COMM_WORLD
mpo = build_josephson_mpo({L}, E_J=1.0, E_C=0.5, n_max={N_MAX}, with_flux=True)
sys.path.insert(0, '/home/captain/clawd/work/dmrg-implementations/pdmrg')
from pdmrg.dmrg import pdmrg_main
t0 = time.time()
energy, _ = pdmrg_main(L={L}, mpo=mpo, bond_dim={BOND_DIM}, max_sweeps={MAX_SWEEPS}, tol={TOL}, dtype='complex128', comm=comm, verbose=False)
print(f"RESULT:{{np.real(energy):.15f}}:{{time.time()-t0:.2f}}")
'''
with open('/tmp/pdmrg_check.py', 'w') as f:
    f.write(script)

cmd = 'source /etc/profile.d/modules.sh && module load mpi/openmpi-x86_64 && mpirun -np 1 --oversubscribe --mca btl tcp,self --mca btl_tcp_if_include lo /home/captain/clawd/work/dmrg-implementations/pdmrg/venv/bin/python -u /tmp/pdmrg_check.py 2>&1'
result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)

for line in result.stdout.split('\n'):
    if line.startswith('RESULT:'):
        parts = line.split(':')
        E = float(parts[1])
        T = float(parts[2])
        results['PDMRG_np1'] = {'energy': E, 'time': T}
        print(f"  Energy: {E:.12f} (ΔE = {abs(E-E_ref):.2e})")
        print(f"  Time:   {T:.1f}s")
        break
else:
    print(f"  FAILED: {result.stdout[-200:]}")
    results['PDMRG_np1'] = {'error': 'failed'}

# A2DMRG np=1
print("\n[3/4] A2DMRG np=1...")
script = f'''
import sys, time, numpy as np
sys.path.insert(0, '/home/captain/clawd/work/dmrg-implementations/a2dmrg')
from benchmarks.josephson_junction import build_josephson_mpo
from a2dmrg.dmrg import a2dmrg_main
from mpi4py import MPI
comm = MPI.COMM_WORLD
mpo = build_josephson_mpo({L}, E_J=1.0, E_C=0.5, n_max={N_MAX}, with_flux=True)
t0 = time.time()
energy, _ = a2dmrg_main(L={L}, mpo=mpo, max_sweeps={MAX_SWEEPS}, bond_dim={BOND_DIM}, tol={TOL}, dtype=np.complex128, comm=comm, warmup_sweeps=2, verbose=False)
print(f"RESULT:{{np.real(energy):.15f}}:{{time.time()-t0:.2f}}")
'''
with open('/tmp/a2dmrg_check.py', 'w') as f:
    f.write(script)

cmd = 'source /etc/profile.d/modules.sh && module load mpi/openmpi-x86_64 && mpirun -np 1 --oversubscribe --mca btl tcp,self --mca btl_tcp_if_include lo /home/captain/clawd/work/dmrg-implementations/a2dmrg/venv/bin/python -u /tmp/a2dmrg_check.py 2>&1'
result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)

for line in result.stdout.split('\n'):
    if line.startswith('RESULT:'):
        parts = line.split(':')
        E = float(parts[1])
        T = float(parts[2])
        results['A2DMRG_np1'] = {'energy': E, 'time': T}
        print(f"  Energy: {E:.12f} (ΔE = {abs(E-E_ref):.2e})")
        print(f"  Time:   {T:.1f}s")
        break
else:
    print(f"  FAILED: {result.stdout[-200:]}")
    results['A2DMRG_np1'] = {'error': 'failed'}

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\n{'Method':<20} | {'Energy':>18} | {'ΔE from ref':>12} | {'Time':>8}")
print("-" * 70)

for name, r in results.items():
    if 'error' in r:
        print(f"{name:<20} | {'FAILED':>18} | {'N/A':>12} | {'N/A':>8}")
    else:
        dE = abs(r['energy'] - E_ref) if name != 'quimb_DMRG2' else 0
        status = "✓" if dE < 1e-8 else "✗"
        print(f"{name:<20} | {r['energy']:>18.12f} | {dE:>12.2e} | {r['time']:>7.1f}s {status}")

print("-" * 70)
print(f"Reference energy: {E_ref:.15f}")

# Check all match
all_match = all(
    'energy' in r and abs(r['energy'] - E_ref) < 1e-8 
    for name, r in results.items() if name != 'quimb_DMRG2'
)
print(f"\n{'✓ ALL MATCH' if all_match else '✗ MISMATCH DETECTED'} (tolerance: 1e-8)")
print("=" * 70)
