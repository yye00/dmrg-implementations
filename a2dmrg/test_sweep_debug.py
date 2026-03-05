#!/usr/bin/env python3
import sys
if "pytest" in sys.modules:
    import pytest
    pytest.skip("dev script (not a unit test)", allow_module_level=True)

"""Debug one A2DMRG sweep."""

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
from a2dmrg.mps.mps_utils import create_neel_state
from a2dmrg.numerics.observables import compute_energy
from a2dmrg.numerics.local_microstep import local_microstep_1site
from a2dmrg.parallel.local_steps import gather_local_results, prepare_candidate_mps_list
from a2dmrg.parallel.coarse_space import build_coarse_matrices
from a2dmrg.numerics.coarse_eigenvalue import solve_coarse_eigenvalue_problem
from a2dmrg.parallel.linear_combination import form_linear_combination

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

# Reference
dmrg = DMRG2(mpo, bond_dims=bond_dim)
dmrg.solve(tol=1e-6, verbosity=0)
E_quimb = dmrg.energy.real
print(f"Quimb:  E = {E_quimb:.10f}")
print()

# Initial state
mps = create_neel_state(L, bond_dim=bond_dim)
E_init = compute_energy(mps, mpo, normalize=True)
print(f"Initial: E = {E_init:.10f}")
print()

# Perform local microsteps at each site
print("=== Local Microsteps ===")
local_results = {}
for site in range(L):
    updated_mps, energy = local_microstep_1site(mps, mpo, site, tol=1e-6)
    local_results[site] = (updated_mps, energy)
    E_check = compute_energy(updated_mps, mpo, normalize=True)
    print(f"Site {site}: local_energy={energy:.6f}, actual_energy={E_check:.6f}")

print()
print("=== Candidate MPS List ===")
candidates = prepare_candidate_mps_list(mps, local_results)
print(f"Number of candidates: {len(candidates)}")

for i, c in enumerate(candidates):
    E_c = compute_energy(c, mpo, normalize=True)
    print(f"  Candidate {i}: E = {E_c:.10f}")

print()
print("=== Coarse-Space ===")
H_coarse, S_coarse, filtered = build_coarse_matrices(candidates, mpo, comm=None)
print(f"H_coarse shape: {H_coarse.shape}")
print(f"H_coarse:\n{H_coarse}")
print(f"S_coarse:\n{S_coarse}")

print()
print("=== Eigenvalue Problem ===")
E_coarse, coeffs = solve_coarse_eigenvalue_problem(H_coarse, S_coarse, regularization=1e-8)
print(f"Coarse energy: {E_coarse:.10f}")
print(f"Coefficients: {coeffs}")

print()
print("=== Linear Combination ===")
combined_mps = form_linear_combination(candidates, coeffs)
E_combined = compute_energy(combined_mps, mpo, normalize=True)
print(f"Combined MPS energy: {E_combined:.10f}")
print(f"Expected from coarse: {E_coarse:.10f}")
print(f"Difference: {abs(E_combined - E_coarse):.3e}")
