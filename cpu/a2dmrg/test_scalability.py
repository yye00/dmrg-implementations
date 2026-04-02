#!/usr/bin/env python3
import sys
if "pytest" in sys.modules:
    import pytest
    pytest.skip("dev script (not a unit test)", allow_module_level=True)

"""
A2DMRG Scalability Study
========================
Run A2DMRG with np=1,2,4,... and measure timing and verify correctness.
"""

import sys
sys.path.insert(0, '.')

# Disable numba caching
try:
    import numba
    from numba.core.dispatcher import Dispatcher
    from numba.np.ufunc import ufuncbuilder
    Dispatcher.enable_caching = lambda self: None
    ufuncbuilder.UFuncDispatcher.enable_caching = lambda self: None
except Exception:
    pass

import numpy as np
import time
from a2dmrg.mpi_compat import MPI, HAS_MPI
from quimb.tensor import SpinHam1D
from a2dmrg.dmrg import a2dmrg_main

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def run_scalability_test(L=20, bond_dim=50, max_sweeps=10):
    """Run scalability test for current np."""
    
    # Build Heisenberg Hamiltonian
    builder = SpinHam1D(S=1/2)
    builder += 1.0, "X", "X"
    builder += 1.0, "Y", "Y"
    builder += 1.0, "Z", "Z"
    mpo = builder.build_mpo(L)
    
    # Warmup run
    if rank == 0:
        print(f"[np={size}] Warmup run...", flush=True)
    a2dmrg_main(L=L, mpo=mpo, max_sweeps=2, bond_dim=bond_dim, 
                tol=1e-6, comm=comm, dtype=np.float64, one_site=True, verbose=False)
    
    # Timed run
    comm.Barrier()
    start = time.time()
    
    energy, mps = a2dmrg_main(
        L=L,
        mpo=mpo,
        max_sweeps=max_sweeps,
        bond_dim=bond_dim,
        tol=1e-10,
        comm=comm,
        dtype=np.float64,
        one_site=True,
        verbose=False
    )
    
    comm.Barrier()
    elapsed = time.time() - start
    
    if rank == 0:
        print(f"[np={size}] L={L}, χ={bond_dim}, sweeps={max_sweeps}")
        print(f"[np={size}] Energy: {energy:.15f}")
        print(f"[np={size}] Time:   {elapsed:.3f}s")
        print(f"[np={size}] Time/sweep: {elapsed/max_sweeps:.3f}s")
        print()
    
    return energy, elapsed


def main():
    L = 20
    bond_dim = 50
    max_sweeps = 10
    
    if rank == 0:
        print("=" * 60)
        print(f"A2DMRG Scalability Study")
        print(f"L={L}, bond_dim={bond_dim}, max_sweeps={max_sweeps}")
        print(f"Running with np={size}")
        print("=" * 60)
        print()
    
    energy, elapsed = run_scalability_test(L, bond_dim, max_sweeps)
    
    if rank == 0:
        # Output in machine-parseable format
        print(f"RESULT: np={size} L={L} chi={bond_dim} E={energy:.15f} t={elapsed:.3f}")


if __name__ == "__main__":
    main()
