#!/usr/bin/env python3
import sys
if "pytest" in sys.modules:
    import pytest
    pytest.skip("dev script (not a unit test)", allow_module_level=True)

"""Minimal A2DMRG validation - just verify the algorithm works."""

import sys
sys.path.insert(0, '.')

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
from quimb.tensor import SpinHam1D, DMRG2
from a2dmrg.dmrg import a2dmrg_main

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Very small test
L = 6
bond_dim = 8
max_sweeps = 5
tol = 1e-6

if rank == 0:
    print(f"Minimal A2DMRG Test (L={L}, χ={bond_dim}, sweeps={max_sweeps}, np={size})")
    print()

# Build Hamiltonian
builder = SpinHam1D(S=1/2)
builder += 1.0, "X", "X"
builder += 1.0, "Y", "Y"
builder += 1.0, "Z", "Z"
mpo = builder.build_mpo(L)

# Run quimb DMRG
if rank == 0:
    dmrg = DMRG2(mpo, bond_dims=bond_dim)
    dmrg.solve(tol=tol, verbosity=0)
    E_quimb = dmrg.energy.real
    print(f"Quimb:  E = {E_quimb:.10f}")
else:
    E_quimb = None

E_quimb = comm.bcast(E_quimb, root=0)

# Run A2DMRG
start = time.time()
energy, mps = a2dmrg_main(
    L=L, mpo=mpo, max_sweeps=max_sweeps, bond_dim=bond_dim,
    tol=tol, comm=comm, dtype=np.float64, one_site=True, verbose=False
)
elapsed = time.time() - start

if rank == 0:
    print(f"A2DMRG: E = {energy:.10f}  (t={elapsed:.1f}s)")
    diff = abs(energy - E_quimb)
    print(f"Diff:   {diff:.3e}")
    if diff < 1e-4:
        print("✓ Energies are close (within 1e-4)")
    else:
        print(f"✗ Energy difference too large: {diff}")
