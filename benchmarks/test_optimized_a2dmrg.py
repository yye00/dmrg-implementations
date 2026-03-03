#!/usr/bin/env python3
"""
Quick benchmark to test A2DMRG optimizations.
Runs a small system (L=10, D=30) to verify speedup from coarse-space fixes.
"""

import sys
import os
import time
import numpy as np

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'a2dmrg'))

import quimb.tensor as qtn
from a2dmrg.dmrg import a2dmrg_main
from a2dmrg.mpi_compat import MPI

# Parameters - small system for quick test
L = 10
BOND_DIM = 30
MAX_SWEEPS = 10
TOL = 1e-8

# Josephson junction parameters
E_C = 1.0
E_J = 2.0
N_MAX = 2
PHI_EXT = np.pi / 4

def build_josephson_mpo():
    """Build Josephson junction array MPO."""
    d = 2 * N_MAX + 1
    S = (d - 1) / 2
    
    charges = np.arange(-N_MAX, N_MAX + 1, dtype='complex128')
    n_op = np.diag(charges)
    
    exp_iphi = np.zeros((d, d), dtype='complex128')
    for i in range(d - 1):
        exp_iphi[i + 1, i] = 1.0
    exp_miphi = exp_iphi.conj().T
    
    builder = qtn.SpinHam1D(S=S)
    flux_phase = np.exp(1j * PHI_EXT)
    builder.add_term(-E_J / 2 * flux_phase, exp_iphi, exp_miphi)
    builder.add_term(-E_J / 2 * np.conj(flux_phase), exp_miphi, exp_iphi)
    n_squared = n_op @ n_op
    builder.add_term(E_C, n_squared)
    
    return builder.build_mpo(L)

def main():
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    
    if rank == 0:
        print("="*70)
        print("A2DMRG OPTIMIZATION BENCHMARK")
        print("="*70)
        print(f"System: L={L}, D={BOND_DIM}, dtype=complex128")
        print(f"MPI ranks: {size}")
        print(f"Max sweeps: {MAX_SWEEPS}, tol={TOL}")
        print("="*70)
    
    # Build MPO
    mpo = build_josephson_mpo()
    
    # Run A2DMRG with timing
    t0 = time.time()
    energy, mps = a2dmrg_main(
        L=L,
        mpo=mpo,
        max_sweeps=MAX_SWEEPS,
        bond_dim=BOND_DIM,
        tol=TOL,
        warmup_sweeps=3,
        dtype=np.complex128,
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
