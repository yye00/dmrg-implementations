#!/usr/bin/env python3
import sys
if "pytest" in sys.modules:
    import pytest
    pytest.skip("dev script (not a unit test)", allow_module_level=True)

"""Quick A2DMRG validation - single system size."""

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

# Test parameters
L = 10
bond_dim = 20
max_sweeps = 20
tol = 1e-10

if rank == 0:
    print(f"=" * 60)
    print(f"Quick A2DMRG Validation Test")
    print(f"L={L}, χ={bond_dim}, sweeps={max_sweeps}, np={size}")
    print(f"=" * 60)
    print()

# Build Hamiltonian
builder = SpinHam1D(S=1/2)
builder += 1.0, "X", "X"
builder += 1.0, "Y", "Y"
builder += 1.0, "Z", "Z"
mpo = builder.build_mpo(L)

# Run quimb DMRG (only on rank 0)
if rank == 0:
    print("Running quimb DMRG...", flush=True)
    dmrg = DMRG2(mpo, bond_dims=bond_dim)
    start = time.time()
    dmrg.solve(tol=tol, verbosity=1)
    t_quimb = time.time() - start
    E_quimb = dmrg.energy.real
    print(f"Quimb:  E = {E_quimb:.15f}  (t={t_quimb:.2f}s)")
    print()
else:
    E_quimb = None

E_quimb = comm.bcast(E_quimb, root=0)

# Run A2DMRG
if rank == 0:
    print("Running A2DMRG...", flush=True)

comm.Barrier()
start = time.time()

energy, mps = a2dmrg_main(
    L=L,
    mpo=mpo,
    max_sweeps=max_sweeps,
    bond_dim=bond_dim,
    tol=tol,
    comm=comm,
    dtype=np.float64,
    one_site=True,
    verbose=(rank == 0)
)

comm.Barrier()
t_a2dmrg = time.time() - start

if rank == 0:
    print(f"A2DMRG: E = {energy:.15f}  (t={t_a2dmrg:.2f}s)")
    print()
    
    diff = abs(energy - E_quimb)
    print(f"Difference: {diff:.3e}")
    print()
    
    if diff < 1e-8:
        print("✓ PASS: Energies match to machine precision!")
    else:
        print(f"✗ FAIL: Energy difference {diff:.3e} exceeds 1e-8")
        sys.exit(1)
