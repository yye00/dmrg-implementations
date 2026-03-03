#!/usr/bin/env python3
"""
Benchmark to test FULL A2DMRG optimizations (coarse-space + local microsteps).
Compares timing breakdown before/after optimizations.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'a2dmrg'))

import quimb.tensor as qtn
from a2dmrg.dmrg import a2dmrg_main
from a2dmrg.mpi_compat import MPI

# Test system
L = 8
BOND_DIM = 20
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
        print("A2DMRG FULL OPTIMIZATION BENCHMARK")
        print("="*70)
        print(f"System: L={L}, D={BOND_DIM}, dtype=float64")
        print(f"MPI ranks: {size}")
        print(f"Max sweeps: {MAX_SWEEPS}, tol={TOL}")
        print("="*70)
        print()
        print("This tests BOTH optimizations:")
        print("  1. Coarse-space: Hermitian symmetry + energy prefilter + parallel")
        print("  2. Local microsteps: Environment caching + lazy decompositions")
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
        print("FINAL RESULTS")
        print("="*70)
        print(f"Energy: {energy:.12f}")
        print(f"Total time: {t1-t0:.2f}s")
        print("="*70)

if __name__ == "__main__":
    main()
