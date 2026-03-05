#!/usr/bin/env python3
import sys
if "pytest" in sys.modules:
    import pytest
    pytest.skip("dev script (not a unit test)", allow_module_level=True)

"""Debug A2DMRG to find the energy issue."""

import sys
sys.path.insert(0, '.')

try:
    import numba
    from numba.core.dispatcher import Dispatcher
    Dispatcher.enable_caching = lambda self: None
except Exception:
    pass

import numpy as np
from a2dmrg.mpi_compat import MPI, HAS_MPI
from quimb.tensor import SpinHam1D, DMRG2
from a2dmrg.dmrg import a2dmrg_main
from a2dmrg.mps.mps_utils import create_neel_state
from a2dmrg.numerics.observables import compute_energy
from a2dmrg.environments.environment import build_left_environments, build_right_environments

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

L = 6
bond_dim = 8

# Build Hamiltonian
builder = SpinHam1D(S=1/2)
builder += 1.0, "X", "X"
builder += 1.0, "Y", "Y"
builder += 1.0, "Z", "Z"
mpo = builder.build_mpo(L)

# Run quimb DMRG
dmrg = DMRG2(mpo, bond_dims=bond_dim)
dmrg.solve(tol=1e-6, verbosity=0)
E_quimb = dmrg.energy.real
print(f"Quimb:  E = {E_quimb:.10f}")
print(f"Quimb MPS bond dims: {[dmrg._k[i].shape for i in range(L)]}")
print()

# Create initial Neel state
mps = create_neel_state(L, bond_dim=bond_dim)
E_init = compute_energy(mps, mpo, normalize=True)
print(f"Initial Neel state energy: {E_init:.10f}")
print(f"Neel MPS shapes: {[mps[i].shape for i in range(L)]}")
print()

# Check environments
print("Building environments...")
L_envs = build_left_environments(mps, mpo)
R_envs = build_right_environments(mps, mpo)

print(f"L_envs: {[e.shape for e in L_envs]}")
print(f"R_envs: {[e.shape for e in R_envs]}")
print()

# Check each site
for site in range(L):
    mps_tensor = mps[site].data
    chi_L = mps_tensor.shape[0] if len(mps_tensor.shape) == 3 else (1 if site == 0 else mps_tensor.shape[0])
    chi_R = mps_tensor.shape[1] if len(mps_tensor.shape) == 3 else (mps_tensor.shape[0] if site == 0 else 1)
    
    L_env = L_envs[site]
    R_env = R_envs[L - site - 1]
    
    env_chi_L = L_env.shape[2] if len(L_env.shape) == 3 else L_env.shape[0]
    env_chi_R = R_env.shape[2] if len(R_env.shape) == 3 else R_env.shape[0]
    
    match = "✓" if env_chi_L == chi_L and env_chi_R == chi_R else "✗ MISMATCH"
    print(f"Site {site}: mps={mps_tensor.shape}, L_env={L_env.shape}, R_env={R_env.shape}, chi_L={chi_L}, chi_R={chi_R}, env_chi_L={env_chi_L}, env_chi_R={env_chi_R} {match}")
