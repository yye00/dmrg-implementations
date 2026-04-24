#!/usr/bin/env python3
import sys
if "pytest" in sys.modules:
    import pytest
    pytest.skip("dev script (not a unit test)", allow_module_level=True)

"""Debug the eigenvector in local microstep."""

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
from quimb.tensor import SpinHam1D
from a2dmrg.mps.mps_utils import create_neel_state
from a2dmrg.numerics.observables import compute_energy
from a2dmrg.numerics.effective_ham import build_effective_hamiltonian_1site
from a2dmrg.numerics.eigensolver import solve_effective_hamiltonian
from a2dmrg.environments.environment import build_left_environments, build_right_environments

L = 6
bond_dim = 8

# Build Hamiltonian
builder = SpinHam1D(S=1/2)
builder += 1.0, "X", "X"
builder += 1.0, "Y", "Y"
builder += 1.0, "Z", "Z"
mpo = builder.build_mpo(L)

# Initial state
mps = create_neel_state(L, bond_dim=bond_dim)

# Build environments once
L_envs = build_left_environments(mps, mpo)
R_envs = build_right_environments(mps, mpo)

for site in [0, 1, 2]:  # Debug first 3 sites
    print(f"=== Site {site} ===")
    
    mps_tensor = mps[site].data
    original_shape = mps_tensor.shape
    print(f"MPS tensor shape: {original_shape}")
    
    # Determine dimensions
    if site == 0:
        chi_R = original_shape[0]
        d = original_shape[1]
        chi_L = 1
    elif site == L - 1:
        chi_L = original_shape[0]
        d = original_shape[1]
        chi_R = 1
    else:
        chi_L = original_shape[0]
        chi_R = original_shape[1]
        d = original_shape[2]
    
    mps_shape = (chi_L, chi_R, d)
    print(f"mps_shape for H_eff: {mps_shape}")
    
    L_env = L_envs[site]
    R_env = R_envs[L - site - 1]
    W_i = mpo[site].data
    
    print(f"L_env shape: {L_env.shape}")
    print(f"R_env shape: {R_env.shape}")
    print(f"W_i shape: {W_i.shape}")
    
    # Build H_eff
    H_eff = build_effective_hamiltonian_1site(L_env, W_i, R_env, mps_shape)
    print(f"H_eff shape: {H_eff.shape}")
    
    # Solve
    v0 = mps_tensor.ravel()
    print(f"v0 shape: {v0.shape}, v0 norm: {np.linalg.norm(v0):.6f}")
    
    energy, eigvec = solve_effective_hamiltonian(H_eff, v0=v0, tol=1e-6)
    print(f"eigvec shape: {eigvec.shape}, eigvec norm: {np.linalg.norm(eigvec):.6f}")
    print(f"energy: {energy:.6f}")
    
    # Reshape eigenvector
    if site == 0:
        new_tensor = eigvec.reshape(mps_shape).squeeze(axis=0)
    elif site == L - 1:
        new_tensor = eigvec.reshape(mps_shape).squeeze(axis=1)
    else:
        new_tensor = eigvec.reshape(mps_shape)
    
    print(f"new_tensor shape: {new_tensor.shape}")
    print(f"new_tensor norm: {np.linalg.norm(new_tensor):.6f}")
    
    # Check if shape matches original
    if new_tensor.shape != original_shape:
        print(f"SHAPE MISMATCH! original={original_shape}, new={new_tensor.shape}")
    
    print()
