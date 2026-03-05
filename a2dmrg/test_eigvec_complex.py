#!/usr/bin/env python3
import sys
if "pytest" in sys.modules:
    import pytest
    pytest.skip("dev script (not a unit test)", allow_module_level=True)

"""Check if eigenvector is complex."""

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
from a2dmrg.environments.environment import build_left_environments, build_right_environments
from scipy.sparse.linalg import eigsh

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

# Build environments
L_envs = build_left_environments(mps, mpo)
R_envs = build_right_environments(mps, mpo)

for site in [2, 3]:  # Debug sites that fail
    print(f"=== Site {site} ===")
    
    mps_tensor = mps[site].data
    original_shape = mps_tensor.shape
    
    chi_L = original_shape[0]
    chi_R = original_shape[1]
    d = original_shape[2]
    mps_shape = (chi_L, chi_R, d)
    
    L_env = L_envs[site]
    R_env = R_envs[L - site - 1]
    W_i = mpo[site].data
    
    print(f"L_env dtype: {L_env.dtype}")
    print(f"R_env dtype: {R_env.dtype}")
    print(f"W_i dtype: {W_i.dtype}")
    print(f"mps_tensor dtype: {mps_tensor.dtype}")
    
    # Build H_eff
    H_eff = build_effective_hamiltonian_1site(L_env, W_i, R_env, mps_shape)
    print(f"H_eff dtype: {H_eff.dtype}")
    
    # Test H_eff by applying it to a vector
    v0 = mps_tensor.ravel()
    print(f"v0 dtype: {v0.dtype}")
    
    Hv = H_eff.matvec(v0)
    print(f"Hv dtype: {Hv.dtype}")
    print(f"Hv has complex components: {np.any(np.iscomplex(Hv))}")
    print(f"Hv imaginary part norm: {np.linalg.norm(np.imag(Hv)):.6e}")
    
    # Solve with explicit return of values
    eigenvalues, eigenvectors = eigsh(H_eff, k=1, which='SA', tol=1e-6, v0=v0)
    eigvec = eigenvectors[:, 0]
    
    print(f"eigvec dtype: {eigvec.dtype}")
    print(f"eigvec has complex components: {np.any(np.iscomplex(eigvec))}")
    if np.iscomplexobj(eigvec):
        print(f"eigvec imaginary part norm: {np.linalg.norm(np.imag(eigvec)):.6e}")
        print(f"eigvec real part norm: {np.linalg.norm(np.real(eigvec)):.6e}")
    print(f"eigvec norm: {np.linalg.norm(eigvec):.6f}")
    print()
