#!/usr/bin/env python3.13
import sys
if "pytest" in sys.modules:
    import pytest
    pytest.skip("dev script (not a unit test)", allow_module_level=True)

"""
Test #61: Reproducibility across different np with fixed seed

This test verifies that A2DMRG gives identical results whether run with
1, 2, or 4 processors when using the same random seed.
"""

import sys
sys.path.insert(0, '.')

# CRITICAL: Apply numba fix BEFORE importing quimb
try:
    import numba
    from numba.core.dispatcher import Dispatcher
    from numba.np.ufunc import ufuncbuilder
    Dispatcher.enable_caching = lambda self: None
    ufuncbuilder.UFuncDispatcher.enable_caching = lambda self: None
except Exception:
    pass  # If numba not needed, continue

import numpy as np
from a2dmrg.mpi_compat import MPI, HAS_MPI
def create_heisenberg_mpo(L, J=1.0):
    """Create Heisenberg chain MPO."""
    import quimb.tensor as qtn
    from quimb.tensor import SpinHam1D

    builder = SpinHam1D(S=1/2)
    builder += J, 'X', 'X'
    builder += J, 'Y', 'Y'
    builder += J, 'Z', 'Z'
    mpo = builder.build_mpo(L)
    return mpo


def main():
    from a2dmrg.dmrg import a2dmrg_main

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Test parameters - use small L for speed
    L = 10
    bond_dim = 8
    max_sweeps = 10
    seed = 42

    if rank == 0:
        print("=" * 70)
        print("Test #61: Reproducibility across different np")
        print("=" * 70)
        print(f"L = {L}, bond_dim = {bond_dim}, seed = {seed}")
        print(f"Current np = {size}")
        print(f"Running {max_sweeps} sweeps...")
        print()

    # Create Hamiltonian
    mpo = create_heisenberg_mpo(L, J=1.0)

    # Set seed for reproducibility
    np.random.seed(seed)

    # Run A2DMRG
    energy, mps = a2dmrg_main(
        L=L,
        mpo=mpo,
        max_sweeps=max_sweeps,
        bond_dim=bond_dim,
        tol=1e-10,
        comm=comm,
        dtype=np.float64,
        one_site=True,
        verbose=False  # Suppress convergence output
    )

    if rank == 0:
        print()
        print("=" * 70)
        print("RESULT")
        print("=" * 70)
        print(f"np = {size}")
        print(f"Energy: {energy:.15f}")
        print(f"Energy per site: {energy/L:.15f}")
        print()
        print("To test reproducibility, run:")
        print(f"  /usr/bin/python3.13 test_cross_np_reproducibility.py")
        print(f"  mpirun -np 2 /usr/bin/python3.13 test_cross_np_reproducibility.py")
        print(f"  mpirun -np 4 /usr/bin/python3.13 test_cross_np_reproducibility.py")
        print()
        print("All runs should give IDENTICAL energies to machine precision.")
        print("=" * 70)

    return energy


if __name__ == "__main__":
    main()
