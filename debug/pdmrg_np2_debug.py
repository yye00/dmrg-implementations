#!/usr/bin/env python3
"""
PDMRG np=2 Debug - investigate boundary merge issue
"""

import sys
import time
import json
import numpy as np
import quimb.tensor as qtn

sys.path.insert(0, '/home/captain/clawd/work/dmrg-implementations/pdmrg')

from mpi4py import MPI

# Load reference
with open('/home/captain/clawd/work/dmrg-implementations/debug/reference_heisenberg.json', 'r') as f:
    reference = json.load(f)

config = reference['config']
E_ref = reference['quimb_DMRG2']['energy']

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_procs = comm.Get_size()

if rank == 0:
    print("="*60)
    print(f"PDMRG np=2 BOUNDARY MERGE DEBUG")
    print("="*60)
    print(f"Reference energy: {E_ref:.15f}")

# Build MPO
mpo = qtn.MPO_ham_heis(L=config['L'], j=1.0, bz=0.0, cyclic=False)

# Import PDMRG components
from pdmrg.dmrg import serial_warmup, distribute_mps, build_local_environments
from pdmrg.dmrg import canonize_block, boundary_merge, check_convergence
from pdmrg.mps.canonical import get_mpo_tensor_data
from pdmrg.parallel.distribute import compute_site_distribution

# Phase 0: Serial warmup
if rank == 0:
    print("\n=== Phase 0: Serial Warmup ===")
    mps_arrays, mpo_arrays, warmup_energy = serial_warmup(
        mpo, config['L'], bond_dim_warmup=config['bond_dim'],
        n_warmup_sweeps=5, dtype='float64'
    )
    print(f"Warmup energy: {warmup_energy:.15f}")
    print(f"ΔE from ref: {abs(warmup_energy - E_ref):.2e}")
else:
    mps_arrays = None
    
mpo_arrays = [get_mpo_tensor_data(mpo, i) for i in range(config['L'])]

# Phase 1: Distribute
if rank == 0:
    print("\n=== Phase 1: Distribute MPS ===")

pmps = distribute_mps(mps_arrays, mpo_arrays, comm, dtype=np.float64)

global_arrays = getattr(pmps, '_global_arrays', None)
env_mgr = build_local_environments(pmps, mpo_arrays, dtype=np.float64,
                                    global_mps_arrays=global_arrays)
if hasattr(pmps, '_global_arrays'):
    del pmps._global_arrays

site_ranges = compute_site_distribution(config['L'], n_procs)
print(f"Rank {rank}: sites {list(site_ranges[rank])}, local_arrays shapes: {[a.shape for a in pmps.arrays]}")

# Skip local energy computation - just proceed to sweeps
print(f"Rank {rank}: ready for sweeps")

comm.Barrier()

# Now run one sweep + merge
bond_dim = config['bond_dim']
tol = config['tol']
eigsolver_tol = tol / 10

if rank == 0:
    print("\n=== Phase 2: QR sweep right ===")
    
canonize_block(pmps, env_mgr, mpo_arrays, 'left')

comm.Barrier()

if rank == 0:
    print("\n=== Phase 3: Even boundary merge ===")
    
E_merge1 = boundary_merge(
    pmps, env_mgr, mpo_arrays, comm, 'even',
    max_bond=bond_dim, max_iter=30, tol=eigsolver_tol
)

print(f"Rank {rank}: merge energy = {E_merge1:.15f}")

comm.Barrier()

if rank == 0:
    print("\n=== Phase 4: QR sweep left ===")
    
canonize_block(pmps, env_mgr, mpo_arrays, 'right')

comm.Barrier()

if rank == 0:
    print("\n=== Phase 5: Odd boundary merge ===")
    
E_merge2 = boundary_merge(
    pmps, env_mgr, mpo_arrays, comm, 'odd',
    max_bond=bond_dim, max_iter=30, tol=eigsolver_tol
)

print(f"Rank {rank}: merge energy = {E_merge2:.15f} (should be 0 for np=2)")

# Check final convergence
merge_energies = [e for e in [E_merge1, E_merge2] if e != 0.0]
E_best = min(merge_energies) if merge_energies else 0.0

converged, E_global = check_convergence(E_best, 0.0, tol, comm)

if rank == 0:
    dE = abs(E_global - E_ref)
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Warmup energy:     {warmup_energy:.15f}")
    print(f"After merge:       {E_global:.15f}")
    print(f"Reference:         {E_ref:.15f}")
    print(f"ΔE (warmup→merge): {abs(E_global - warmup_energy):.2e}")
    print(f"ΔE (merge→ref):    {dE:.2e}")
    print(f"Status: {'✓ PASS' if dE < 1e-10 else '✗ FAIL'}")
