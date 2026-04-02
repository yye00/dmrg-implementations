#!/usr/bin/env python3
import sys
if "pytest" in sys.modules:
    import pytest
    pytest.skip("dev script (not a unit test)", allow_module_level=True)

"""Debug the local microstep output."""

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
from a2dmrg.numerics.local_microstep import local_microstep_1site

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
print(f"Original MPS shapes: {[mps[i].shape for i in range(L)]}")
print()

# Check each site
for site in range(L):
    updated_mps, energy = local_microstep_1site(mps, mpo, site, tol=1e-6)
    norm = updated_mps.norm()
    E = compute_energy(updated_mps, mpo, normalize=True)
    
    print(f"Site {site}:")
    print(f"  Updated MPS norm: {norm:.10e}")
    print(f"  Energy: {E:.10f}")
    print(f"  Shapes: {[updated_mps[i].shape for i in range(L)]}")
    
    # Check tensor norms
    for i in range(L):
        t_norm = np.linalg.norm(updated_mps[i].data)
        if t_norm < 1e-8:
            print(f"  WARNING: Tensor {i} has near-zero norm: {t_norm:.3e}")
    print()
