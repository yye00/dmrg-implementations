#!/usr/bin/env python3
"""
Quick Sanity Check: smaller system for faster comparison
"""

import subprocess
import time
import sys
import numpy as np

# Smaller configuration for quick testing
L = 10
BOND_DIM = 30
N_MAX = 2
MAX_SWEEPS = 10
TOL = 1e-8

sys.path.insert(0, '/home/captain/clawd/work/dmrg-implementations/a2dmrg')
from benchmarks.josephson_junction import build_josephson_mpo
import quimb.tensor as qtn

print("=" * 70)
print("QUICK SANITY CHECK")
print("=" * 70)
print(f"Parameters: L={L}, bond_dim={BOND_DIM}, n_max={N_MAX}, tol={TOL}")

print(f"\nBuilding MPO...")
mpo = build_josephson_mpo(L, E_J=1.0, E_C=0.5, n_max=N_MAX, with_flux=True)

# quimb DMRG2
print("\n[quimb DMRG2]")
t0 = time.time()
dmrg2 = qtn.DMRG2(mpo, bond_dims=BOND_DIM)
dmrg2.solve(max_sweeps=MAX_SWEEPS, tol=TOL, verbosity=1)
t1 = time.time()
E_ref = float(np.real(dmrg2.energy))
print(f"  Energy: {E_ref:.12f}")
print(f"  Time:   {t1-t0:.2f}s")

# Now run PDMRG np=1
print("\n[PDMRG np=1]")
script = f'''
import sys, time
import numpy as np
sys.path.insert(0, '/home/captain/clawd/work/dmrg-implementations/a2dmrg')
from benchmarks.josephson_junction import build_josephson_mpo
from mpi4py import MPI
comm = MPI.COMM_WORLD
mpo = build_josephson_mpo({L}, E_J=1.0, E_C=0.5, n_max={N_MAX}, with_flux=True)
sys.path.insert(0, '/home/captain/clawd/work/dmrg-implementations/pdmrg')
from pdmrg.dmrg import pdmrg_main
t0 = time.time()
energy, mps = pdmrg_main(L={L}, mpo=mpo, bond_dim={BOND_DIM}, max_sweeps={MAX_SWEEPS}, tol={TOL}, dtype='complex128', comm=comm, verbose=True)
t1 = time.time()
print(f"RESULT:{{np.real(energy):.15f}}:{{t1-t0:.4f}}")
'''
with open('/tmp/pdmrg_quick.py', 'w') as f:
    f.write(script)

pdmrg_python = '/home/captain/clawd/work/dmrg-implementations/pdmrg/venv/bin/python'
cmd = f'source /etc/profile.d/modules.sh && module load mpi/openmpi-x86_64 && mpirun -np 1 --oversubscribe --mca btl tcp,self --mca btl_tcp_if_include lo {pdmrg_python} -u /tmp/pdmrg_quick.py 2>&1'

result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
print(result.stdout)
if result.stderr:
    print(result.stderr[-500:])

pdmrg_energy = None
pdmrg_time = None
for line in result.stdout.split('\n'):
    if line.startswith('RESULT:'):
        parts = line.split(':')
        pdmrg_energy = float(parts[1])
        pdmrg_time = float(parts[2])

# A2DMRG np=1
print("\n[A2DMRG np=1]")
script = f'''
import sys, time
import numpy as np
sys.path.insert(0, '/home/captain/clawd/work/dmrg-implementations/a2dmrg')
from benchmarks.josephson_junction import build_josephson_mpo
from a2dmrg.dmrg import a2dmrg_main
from mpi4py import MPI
comm = MPI.COMM_WORLD
mpo = build_josephson_mpo({L}, E_J=1.0, E_C=0.5, n_max={N_MAX}, with_flux=True)
t0 = time.time()
energy, mps = a2dmrg_main(L={L}, mpo=mpo, max_sweeps={MAX_SWEEPS}, bond_dim={BOND_DIM}, tol={TOL}, dtype=np.complex128, comm=comm, warmup_sweeps=2, one_site=True, verbose=True)
t1 = time.time()
print(f"RESULT:{{np.real(energy):.15f}}:{{t1-t0:.4f}}")
'''
with open('/tmp/a2dmrg_quick.py', 'w') as f:
    f.write(script)

a2dmrg_python = '/home/captain/clawd/work/dmrg-implementations/a2dmrg/venv/bin/python'
cmd = f'source /etc/profile.d/modules.sh && module load mpi/openmpi-x86_64 && mpirun -np 1 --oversubscribe --mca btl tcp,self --mca btl_tcp_if_include lo {a2dmrg_python} -u /tmp/a2dmrg_quick.py 2>&1'

result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
print(result.stdout)
if result.stderr:
    print(result.stderr[-500:])

a2dmrg_energy = None
a2dmrg_time = None
for line in result.stdout.split('\n'):
    if line.startswith('RESULT:'):
        parts = line.split(':')
        a2dmrg_energy = float(parts[1])
        a2dmrg_time = float(parts[2])

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\n{'Method':<20} | {'Energy':>18} | {'ΔE':>12} | {'Time':>10}")
print("-" * 70)

print(f"{'quimb DMRG2':<20} | {E_ref:>18.12f} | {'(ref)':>12} | {t1-t0:>9.1f}s")

if pdmrg_energy is not None:
    print(f"{'PDMRG np=1':<20} | {pdmrg_energy:>18.12f} | {abs(pdmrg_energy - E_ref):>12.2e} | {pdmrg_time:>9.1f}s")
    pdmrg_ratio = pdmrg_time / (t1-t0)
else:
    print(f"{'PDMRG np=1':<20} | {'FAILED':>18}")
    pdmrg_ratio = None

if a2dmrg_energy is not None:
    print(f"{'A2DMRG np=1':<20} | {a2dmrg_energy:>18.12f} | {abs(a2dmrg_energy - E_ref):>12.2e} | {a2dmrg_time:>9.1f}s")
    a2dmrg_ratio = a2dmrg_time / (t1-t0)
else:
    print(f"{'A2DMRG np=1':<20} | {'FAILED':>18}")
    a2dmrg_ratio = None

print("-" * 70)

if pdmrg_ratio:
    print(f"\nPDMRG np=1 is {pdmrg_ratio:.1f}x vs quimb DMRG2")
if a2dmrg_ratio:
    print(f"A2DMRG np=1 is {a2dmrg_ratio:.1f}x vs quimb DMRG2")

print("\n" + "=" * 70)
