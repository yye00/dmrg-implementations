#!/usr/bin/env python3
"""
Very quick benchmark to verify A2DMRG timing improvements.
Uses minimal system (L=6, D=16) to get fast results.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'a2dmrg'))

import quimb.tensor as qtn
from a2dmrg.dmrg import a2dmrg_main
from a2dmrg.mpi_compat import MPI

# Tiny system for fast testing
L = 6
BOND_DIM = 16
MAX_SWEEPS = 5
TOL = 1e-6

def build_heisenberg_mpo():
    """Build simple Heisenberg MPO."""
    builder = qtn.SpinHam1D(S=1/2)
    builder += 1.0, 'X', 'X'
    builder += 1.0, 'Y', 'Y'
    builder += 1.0, 'Z', 'Z'
    return builder.build_mpo(L)

def main():
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    
    if rank == 0:
        print("="*70)
        print("A2DMRG QUICK TIMING TEST")
        print("="*70)
        print(f"System: L={L}, D={BOND_DIM}, dtype=float64")
        print(f"MPI ranks: {size}")
        print(f"Max sweeps: {MAX_SWEEPS}, tol={TOL}")
        print("="*70)
    
    mpo = build_heisenberg_mpo()
    
    t0 = time.time()
    energy, mps = a2dmrg_main(
        L=L,
        mpo=mpo,
        max_sweeps=MAX_SWEEPS,
        bond_dim=BOND_DIM,
        tol=TOL,
        warmup_sweeps=2,
        dtype=np.float64,
        comm=comm,
        verbose=True
    )
    t1 = time.time()
    
    if rank == 0:
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        print(f"Energy: {energy:.12f}")
        print(f"Total time: {t1-t0:.2f}s")
        print("="*70)

if __name__ == "__main__":
    main()
