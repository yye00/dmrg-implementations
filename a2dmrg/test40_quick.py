"""Test #40: Quick version with reduced sweeps."""
import fix_quimb_python313
import numpy as np
from a2dmrg.mpi_compat import MPI, HAS_MPI
from quimb.tensor import DMRG2
from a2dmrg.dmrg import a2dmrg_main
from a2dmrg.tests.test_bose_hubbard import create_bose_hubbard_mpo
from a2dmrg.mps.format_conversion import convert_quimb_dmrg_to_a2dmrg_format

# Test parameters
L = 6
bond_dim = 20
nmax = 3

# Create Bose-Hubbard MPO with complex hopping
t_mag = 1.0
phase = np.pi / 4
t = t_mag * np.exp(1j * phase)
U = 2.0
mu = 0.5

mpo = create_bose_hubbard_mpo(L, t=t, U=U, mu=mu, nmax=nmax)

print(f"\n{'='*60}")
print(f"Test #40: Josephson Junction - Complex128 Bose-Hubbard")
print(f"{'='*60}")
print(f"System: L={L}, nmax={nmax}, bond_dim={bond_dim}")
print(f"Parameters: t={t:.6f}, U={U:.2f}, μ={mu:.2f}")

# Warm-start with DMRG2
print(f"\n--- DMRG2 warm-start (5 sweeps) ---")
dmrg_warmstart = DMRG2(mpo, bond_dims=bond_dim)
dmrg_warmstart.solve(tol=1e-10, max_sweeps=5, verbosity=0)
E_warmstart = dmrg_warmstart.energy
print(f"DMRG2 warm-start energy: {E_warmstart:.12f}")

# Convert to A2DMRG format
print(f"\n--- Converting MPS ---")
initial_mps = convert_quimb_dmrg_to_a2dmrg_format(dmrg_warmstart.state, bond_dim)
print("Conversion complete")

# Serial DMRG reference (more sweeps for convergence)
print(f"\n--- DMRG2 reference (10 sweeps) ---")
dmrg = DMRG2(mpo, bond_dims=bond_dim)
dmrg.solve(tol=1e-10, max_sweeps=10, verbosity=0)
E_serial = dmrg.energy
print(f"DMRG2 reference energy: {E_serial:.12f}")

# A2DMRG with warm-start (fewer sweeps since starting from good state)
print(f"\n--- A2DMRG (np=1, complex128, warm-start, 5 sweeps) ---")
energy, mps = a2dmrg_main(
    L=L,
    mpo=mpo,
    max_sweeps=5,  # Reduced from 15
    bond_dim=bond_dim,
    tol=1e-10,
    comm=MPI.COMM_WORLD,
    dtype=np.complex128,
    one_site=True,
    initial_mps=initial_mps,
    verbose=False  # Less verbose for speed
)
print(f"A2DMRG energy: {energy:.12f}")

# Verify
diff = abs(energy - E_serial)
print(f"\n{'='*60}")
print(f"Difference from DMRG2: {diff:.3e}")
print(f"Target: 1e-10")

if diff < 1e-10:
    print("✓ TEST PASSED: A2DMRG complex128 matches serial DMRG")
    print(f"{'='*60}\n")
    exit(0)
else:
    print("✗ TEST FAILED: Energy mismatch")
    print(f"{'='*60}\n")
    exit(1)
