#!/usr/bin/env python
"""
DEPRECATED: This script is no longer functional.

PDMRG now requires np >= 2 (it is a parallel real-space algorithm).
For serial execution, use quimb.DMRG2 instead.

See: cpu-audit branch changes (2026-03-07)
"""
import sys
print("=" * 80)
print("ERROR: This script is deprecated")
print("PDMRG requires np >= 2 (parallel real-space algorithm)")
print("For serial execution, use quimb.DMRG2 instead")
print("=" * 80)
sys.exit(1)

import sys
import os
import time
import numpy as np

# Add relative paths to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'a2dmrg'))
from benchmarks.josephson_junction import build_josephson_mpo

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Parameters
L = 20
bond_dim = 50
n_max = 2
max_sweeps = 20
tol = 1e-14  # Machine precision

print(f"Building Josephson Junction MPO (L={L}, n_max={n_max})...")
mpo = build_josephson_mpo(L, E_J=1.0, E_C=0.5, n_max=n_max, with_flux=True)

# Import and run PDMRG
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'pdmrg'))
from pdmrg.dmrg import pdmrg_main

print(f"Starting PDMRG (np={comm.Get_size()}, D={bond_dim})...")
t0 = time.time()
energy, mps = pdmrg_main(
    L=L,
    mpo=mpo,
    bond_dim=bond_dim,
    max_sweeps=max_sweeps,
    tol=tol,
    dtype='complex128',
    comm=comm,
    verbose=(rank == 0)
)
t1 = time.time()

if rank == 0:
    print(f"\n{'='*60}")
    print(f"PDMRG np=1 COMPLETE")
    print(f"Energy: {np.real(energy):.12f}")
    print(f"Time: {t1-t0:.2f}s")
    print(f"{'='*60}")
