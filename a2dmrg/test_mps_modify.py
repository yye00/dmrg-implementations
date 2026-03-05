#!/usr/bin/env python3
import sys
if "pytest" in sys.modules:
    import pytest
    pytest.skip("dev script (not a unit test)", allow_module_level=True)

"""Debug MPS modification."""

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

# Test copying and modifying
for site in [0, 1, 2, 3]:
    print(f"\n=== Testing site {site} ===")
    
    # Copy MPS
    mps_copy = mps.copy()
    print(f"After copy, norm: {mps_copy.norm():.10f}")
    
    # Get tensor data
    old_tensor = mps_copy[site].data
    print(f"Old tensor shape: {old_tensor.shape}, norm: {np.linalg.norm(old_tensor):.6f}")
    
    # Create new random tensor with same shape
    new_tensor = np.random.randn(*old_tensor.shape)
    new_tensor /= np.linalg.norm(new_tensor)
    print(f"New tensor shape: {new_tensor.shape}, norm: {np.linalg.norm(new_tensor):.6f}")
    
    # Modify
    mps_copy[site].modify(data=new_tensor)
    print(f"After modify, MPS norm: {mps_copy.norm():.10f}")
    
    # Check tensor
    check_tensor = mps_copy[site].data
    print(f"Check tensor shape: {check_tensor.shape}, norm: {np.linalg.norm(check_tensor):.6f}")
    print(f"Tensor data match: {np.allclose(new_tensor, check_tensor)}")
