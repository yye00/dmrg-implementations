#!/usr/bin/env python3
import sys
if "pytest" in sys.modules:
    import pytest
    pytest.skip("dev script (not a unit test)", allow_module_level=True)

"""Full step-by-step trace of local_microstep_1site."""

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
from a2dmrg.numerics.observables import compute_energy

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
print(f"Original MPS norm: {mps.norm():.10f}")

site = 2  # Debug site 2

print(f"\n=== Site {site} ===")

# Step 1: Create a copy
mps_updated = mps.copy()
print(f"After copy, norm: {mps_updated.norm():.10f}")

# Check all tensors
print("Tensor norms after copy:")
for i in range(L):
    print(f"  Tensor {i}: {np.linalg.norm(mps_updated[i].data):.6f}")

# Step 2: Build environments
L_envs = build_left_environments(mps_updated, mpo)
R_envs = build_right_environments(mps_updated, mpo)

# Get the tensor at site
mps_tensor = mps_updated[site].data
original_shape = mps_tensor.shape
print(f"\nMPS tensor shape: {original_shape}")
print(f"MPS tensor norm: {np.linalg.norm(mps_tensor):.6f}")

# Determine dimensions
chi_L = original_shape[0]
chi_R = original_shape[1]
d = original_shape[2]
mps_shape = (chi_L, chi_R, d)
print(f"mps_shape: {mps_shape}")

# Get environment tensors
L_env = L_envs[site]
R_env = R_envs[L - site - 1]
W_i = mpo[site].data

print(f"L_env shape: {L_env.shape}")
print(f"R_env shape: {R_env.shape}")
print(f"W_i shape: {W_i.shape}")

# Step 3: Build H_eff
H_eff = build_effective_hamiltonian_1site(L_env, W_i, R_env, mps_shape)

# Use current MPS tensor as initial guess
v0 = mps_tensor.ravel()
print(f"\nv0 norm: {np.linalg.norm(v0):.6f}")

# Solve for ground state
energy, eigvec = solve_effective_hamiltonian(H_eff, v0=v0, tol=1e-6)
print(f"eigvec norm: {np.linalg.norm(eigvec):.6f}")
print(f"eigvec dtype: {eigvec.dtype}")

# Step 4: Update MPS[site] with new eigenvector
# For middle site, just reshape
new_tensor = eigvec.reshape(mps_shape)
print(f"new_tensor shape: {new_tensor.shape}")
print(f"new_tensor norm: {np.linalg.norm(new_tensor):.6f}")

# CRITICAL: Check dtype mismatch!
print(f"\nnew_tensor dtype: {new_tensor.dtype}")
print(f"mps_tensor dtype: {mps_tensor.dtype}")

# Try converting to real
if np.iscomplexobj(new_tensor):
    print("Converting to real...")
    new_tensor_real = np.real(new_tensor)
    print(f"new_tensor_real norm: {np.linalg.norm(new_tensor_real):.6f}")
else:
    new_tensor_real = new_tensor

# Check if we need to cast
print(f"\nBefore modify, mps_updated[{site}].data dtype: {mps_updated[site].data.dtype}")

# Update the MPS tensor data
mps_updated[site].modify(data=new_tensor_real)

print(f"After modify, mps_updated[{site}].data dtype: {mps_updated[site].data.dtype}")
print(f"After modify, tensor norm: {np.linalg.norm(mps_updated[site].data):.6f}")
print(f"After modify, MPS norm: {mps_updated.norm():.10e}")

# Check all tensors after modify
print("\nTensor norms after modify:")
for i in range(L):
    print(f"  Tensor {i}: {np.linalg.norm(mps_updated[i].data):.6f}")
