#!/usr/bin/env python3
import sys
if "pytest" in sys.modules:
    import pytest
    pytest.skip("dev script (not a unit test)", allow_module_level=True)

"""Check why eigenvector is orthogonal to neighbors."""

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
site = 2

# Build environments
L_envs = build_left_environments(mps, mpo)
R_envs = build_right_environments(mps, mpo)

mps_tensor = mps[site].data
chi_L, chi_R, d = mps_tensor.shape
mps_shape = (chi_L, chi_R, d)

L_env = L_envs[site]
R_env = R_envs[L - site - 1]
W_i = mpo[site].data

# Build H_eff
H_eff = build_effective_hamiltonian_1site(L_env, W_i, R_env, mps_shape)

# Original tensor as vector
v0 = mps_tensor.ravel()
print(f"Original tensor v0 norm: {np.linalg.norm(v0):.6f}")

# Apply H_eff to v0
Hv0 = H_eff.matvec(v0)
print(f"H_eff @ v0 norm: {np.linalg.norm(Hv0):.6f}")
print(f"<v0|H|v0>: {np.real(np.vdot(v0, Hv0)):.6f}")

# Get eigenvector
energy, eigvec = solve_effective_hamiltonian(H_eff, v0=v0, tol=1e-6)
eigvec_real = np.real(eigvec)
print(f"\nEigvec norm: {np.linalg.norm(eigvec_real):.6f}")
print(f"Eigenvalue: {energy:.6f}")

# Overlap between original and eigenvector
overlap = np.abs(np.vdot(v0, eigvec_real))
print(f"\nOverlap |<v0|eigvec>|: {overlap:.6f}")

# Check what the Neel state tensor looks like
print(f"\nOriginal tensor (v0) reshaped to {mps_shape}:")
v0_tensor = v0.reshape(mps_shape)
print(f"  Non-zero elements: {np.count_nonzero(np.abs(v0_tensor) > 1e-10)}")
print(f"  Total elements: {v0_tensor.size}")

# Show where the non-zeros are
for i in range(chi_L):
    for j in range(chi_R):
        for k in range(d):
            if abs(v0_tensor[i,j,k]) > 1e-10:
                print(f"  v0[{i},{j},{k}] = {v0_tensor[i,j,k]:.6f}")

print(f"\nEigenvector reshaped to {mps_shape}:")
eig_tensor = eigvec_real.reshape(mps_shape)
print(f"  Non-zero elements: {np.count_nonzero(np.abs(eig_tensor) > 1e-10)}")

# Show largest elements
flat = eig_tensor.ravel()
largest_idx = np.argsort(np.abs(flat))[-5:]
print(f"  Largest 5 elements:")
for idx in largest_idx[::-1]:
    i, j, k = np.unravel_index(idx, mps_shape)
    print(f"    [{i},{j},{k}] = {flat[idx]:.6f}")
