import sys
if "pytest" in sys.modules:
    import pytest
    pytest.skip("dev script (not a unit test)", allow_module_level=True)

"""Debug compression issue in A2DMRG."""
import fix_quimb_python313
import numpy as np
from a2dmrg.mpi_compat import MPI, HAS_MPI
from a2dmrg.tests.test_bose_hubbard import create_bose_hubbard_mpo
from a2dmrg.numerics.observables import compute_energy
from a2dmrg.mps.mps_utils import create_product_state_mps
from a2dmrg.parallel.linear_combination import form_linear_combination
from a2dmrg.mps.canonical import compress_mps

# Very small test case
L = 4
nmax = 2
bond_dim = 4
t = 1.0 + 0.5j
U = 2.0
mu = 0.5

mpo = create_bose_hubbard_mpo(L, t=t, U=U, mu=mu, nmax=nmax)

# Create some candidate MPS
phys_dim = nmax + 1
candidates = []
for state_idx in range(min(3, phys_dim)):
    mps = create_product_state_mps(L, bond_dim=bond_dim, state_index=state_idx,
                                    phys_dim=phys_dim, dtype=np.complex128)
    energy = compute_energy(mps, mpo)
    print(f'Candidate {state_idx} (state |{state_idx}{state_idx}{state_idx}{state_idx}⟩): E = {energy:.6f}')
    candidates.append(mps)

# Form linear combination with coefficients
coeffs = np.array([0.2, 0.6, 0.2])
coeffs /= np.linalg.norm(coeffs)  # Normalize
print(f'\nCoefficients: {coeffs}')

combined = form_linear_combination(candidates, coeffs)
print(f'\nCombined MPS norm: {combined.norm():.6f}')
energy_combined = compute_energy(combined, mpo)
print(f'Combined MPS energy: {energy_combined:.6f}')

# Now compress
combined_compressed = combined.copy()
compress_mps(combined_compressed, max_rank=bond_dim, tol=1e-12, pad_bonds=True)
combined_compressed /= combined_compressed.norm()

energy_compressed = compute_energy(combined_compressed, mpo)
print(f'Compressed MPS energy: {energy_compressed:.6f}')
print(f'Energy change from compression: {energy_compressed - energy_combined:.6e}')
