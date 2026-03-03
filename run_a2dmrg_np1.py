#!/usr/bin/env python
"""Run A2DMRG np=1 benchmark standalone."""
import sys
import time
import numpy as np

sys.path.insert(0, '/home/captain/clawd/work/dmrg-implementations/a2dmrg')
from benchmarks.josephson_junction import build_josephson_mpo
from a2dmrg.dmrg import a2dmrg_main

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Parameters  
L = 20
bond_dim = 50
n_max = 2
max_sweeps = 20
warmup_sweeps = 5
tol = 1e-14  # Machine precision

print(f"Building Josephson Junction MPO (L={L}, n_max={n_max})...")
mpo = build_josephson_mpo(L, E_J=1.0, E_C=0.5, n_max=n_max, with_flux=True)

print(f"Starting A2DMRG (np={comm.Get_size()}, D={bond_dim})...")
t0 = time.time()
energy, mps = a2dmrg_main(
    L=L,
    mpo=mpo,
    max_sweeps=max_sweeps,
    bond_dim=bond_dim,
    tol=tol,
    dtype=np.complex128,
    comm=comm,
    warmup_sweeps=warmup_sweeps,
    one_site=True,
    verbose=(rank == 0)
)
t1 = time.time()

if rank == 0:
    print(f"\n{'='*60}")
    print(f"A2DMRG np=1 COMPLETE")
    print(f"Energy: {np.real(energy):.12f}")
    print(f"Time: {t1-t0:.2f}s")
    print(f"{'='*60}")
