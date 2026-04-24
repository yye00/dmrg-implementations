#!/usr/bin/env python3
import sys
if "pytest" in sys.modules:
    import pytest
    pytest.skip("dev script (not a unit test)", allow_module_level=True)

"""Quick test with warm-up sweeps."""

import sys
sys.path.insert(0, '.')

try:
    import numba
    from numba.core.dispatcher import Dispatcher
    Dispatcher.enable_caching = lambda self: None
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

L = 6
bond_dim = 16
warmup = 2
a2dmrg_sweeps = 3

# Build Hamiltonian
builder = SpinHam1D(S=1/2)
builder += 1.0, "X", "X"
builder += 1.0, "Y", "Y"
builder += 1.0, "Z", "Z"
mpo = builder.build_mpo(L)

if rank == 0:
    print(f"Test: L={L}, bond_dim={bond_dim}, warmup={warmup}, np={size}", flush=True)
    print(flush=True)
    
    # Reference
    dmrg_ref = DMRG2(mpo, bond_dims=bond_dim)
    dmrg_ref.solve(tol=1e-10, verbosity=0)
    E_ref = float(np.real(dmrg_ref.energy))
    print(f"Quimb DMRG: E = {E_ref:.12f}", flush=True)
else:
    E_ref = None

E_ref = comm.bcast(E_ref, root=0)

# Run A2DMRG with warm-up
start = time.time()
energy, mps = a2dmrg_main(
    L=L,
    mpo=mpo,
    max_sweeps=a2dmrg_sweeps,
    bond_dim=bond_dim,
    tol=1e-6,
    comm=comm,
    dtype=np.float64,
    one_site=True,
    verbose=True,
    warmup_sweeps=warmup
)
elapsed = time.time() - start

if rank == 0:
    diff = abs(energy - E_ref)
    print(flush=True)
    print(f"A2DMRG:    E = {energy:.12f}", flush=True)
    print(f"Reference: E = {E_ref:.12f}", flush=True)
    print(f"Diff: {diff:.3e}", flush=True)
    print(f"Time: {elapsed:.1f}s", flush=True)
    
    if diff < 1e-6:
        print("✓ PASS: Energies match!", flush=True)
    else:
        print(f"✗ FAIL: Energy difference too large", flush=True)
