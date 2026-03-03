#!/usr/bin/env python3
"""
A2DMRG Debugging Script - Heisenberg Model

Must produce IDENTICAL results to quimb DMRG2 (within 1e-10).
Uses the same:
1. MPO construction
2. Tensor contraction patterns (via cotengra + optuna)
3. Convergence criteria
"""

import sys
import time
import json
import numpy as np
import quimb.tensor as qtn

# Add paths
sys.path.insert(0, '/home/captain/clawd/work/dmrg-implementations/a2dmrg')

from mpi4py import MPI

# Load reference configuration
with open('/home/captain/clawd/work/dmrg-implementations/debug/reference_heisenberg.json', 'r') as f:
    reference = json.load(f)

config = reference['config']
E_ref = reference['quimb_DMRG2']['energy']

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_procs = comm.Get_size()

if rank == 0:
    print("="*60)
    print(f"A2DMRG DEBUGGING - Heisenberg Model (np={n_procs})")
    print("="*60)
    print(f"Configuration: {config}")
    print(f"Reference energy (quimb DMRG2): {E_ref:.15f}")

# Build the SAME MPO as quimb reference
mpo = qtn.MPO_ham_heis(L=config['L'], j=1.0, bz=0.0, cyclic=False)

if rank == 0:
    print(f"\nMPO: L={mpo.L}, bond_dim={mpo.max_bond()}")

# Import A2DMRG
from a2dmrg.dmrg import a2dmrg_main

# Run A2DMRG with IDENTICAL parameters
t0 = time.time()
energy, mps = a2dmrg_main(
    L=config['L'],
    mpo=mpo,
    max_sweeps=1,  # Single sweep for debugging
    bond_dim=config['bond_dim'],
    tol=config['tol'],
    dtype=np.float64,
    comm=comm,
    warmup_sweeps=5,      # More warmup for A2DMRG
    finalize_sweeps=0,    # DISABLED - has a bug with MPS handoff
    one_site=False,       # Two-site for better convergence
    verbose=(rank == 0),
)
t1 = time.time()

if rank == 0:
    dE = abs(energy - E_ref)
    status = "✓ PASS" if dE < 1e-10 else "✗ FAIL"
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"A2DMRG energy:   {energy:.15f}")
    print(f"Reference (DMRG2): {E_ref:.15f}")
    print(f"Difference:      {dE:.2e} {status}")
    print(f"Time:            {t1-t0:.2f}s")
    print(f"Processors:      {n_procs}")
    
    # Save result
    result = {
        'method': 'A2DMRG',
        'np': n_procs,
        'energy': float(np.real(energy)),
        'time': t1 - t0,
        'dE': dE,
        'pass': dE < 1e-10,
    }
    
    with open(f'/home/captain/clawd/work/dmrg-implementations/debug/a2dmrg_np{n_procs}_result.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nResult saved to debug/a2dmrg_np{n_procs}_result.json")
