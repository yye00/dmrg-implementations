import sys
if "pytest" in sys.modules:
    import pytest
    pytest.skip("dev script (not a unit test)", allow_module_level=True)

"""Test warm-start with smaller problem."""
import fix_quimb_python313
import numpy as np
from a2dmrg.mpi_compat import MPI, HAS_MPI
from quimb.tensor import DMRG2
from a2dmrg.dmrg import a2dmrg_main
from a2dmrg.tests.test_bose_hubbard import create_bose_hubbard_mpo
from a2dmrg.mps.format_conversion import convert_quimb_dmrg_to_a2dmrg_format

# Very small test
L = 4
bond_dim = 8
nmax = 2

# Create Bose-Hubbard MPO
t = 1.0 * np.exp(1j * np.pi / 4)
U = 2.0
mu = 0.5

mpo = create_bose_hubbard_mpo(L, t=t, U=U, mu=mu, nmax=nmax)
print(f"System: L={L}, bond_dim={bond_dim}, nmax={nmax}")

# Warm-start with DMRG2
print("\n--- Running DMRG2 for warm-start ---")
dmrg_warmstart = DMRG2(mpo, bond_dims=bond_dim)
dmrg_warmstart.solve(tol=1e-10, max_sweeps=5, verbosity=0)
E_warmstart = dmrg_warmstart.energy
print(f"DMRG2 energy: {E_warmstart:.10f}")

# Convert to A2DMRG format
print("\n--- Converting MPS ---")
initial_mps = convert_quimb_dmrg_to_a2dmrg_format(dmrg_warmstart.state, bond_dim)
print(f"Conversion complete")

# Run A2DMRG with warm-start
print("\n--- Running A2DMRG with warm-start ---")
energy, mps = a2dmrg_main(
    L=L,
    mpo=mpo,
    max_sweeps=2,  # Just 2 sweeps for testing
    bond_dim=bond_dim,
    tol=1e-10,
    comm=MPI.COMM_WORLD,
    dtype=np.complex128,
    one_site=True,
    initial_mps=initial_mps,
    verbose=True
)
print(f"\nA2DMRG energy: {energy:.10f}")
print(f"Difference from DMRG2: {abs(energy - E_warmstart):.3e}")
