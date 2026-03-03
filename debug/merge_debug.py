#!/usr/bin/env python3
"""Debug the boundary merge step"""

import sys
import numpy as np
import quimb.tensor as qtn

sys.path.insert(0, '/home/captain/clawd/work/dmrg-implementations/pdmrg')

from pdmrg.dmrg import serial_warmup
from pdmrg.mps.canonical import get_mpo_tensor_data
from pdmrg.environments.update import init_left_env, init_right_env, update_left_env, update_right_env
from pdmrg.parallel.merge import merge_boundary_tensors

L = 12
mpo = qtn.MPO_ham_heis(L=L, j=1.0, bz=0.0, cyclic=False)
mpo_arrays = [get_mpo_tensor_data(mpo, i) for i in range(L)]

# Get warmup state
mps_arrays, _, warmup_energy = serial_warmup(mpo, L, bond_dim_warmup=20, n_warmup_sweeps=5)
print(f"Warmup energy: {warmup_energy:.15f}")

# Build R_env first (from site 11 to 7)
chi_R_last = mps_arrays[-1].shape[2]
D_last = mpo_arrays[-1].shape[1]
R_env = init_right_env(chi_R_last, D_last, np.float64)
for i in range(L-1, 6, -1):
    R_env = update_right_env(R_env, mps_arrays[i], mpo_arrays[i])

# Simulate canonize_block AND build L_env consistently
# For 'left' direction: QR sweep right on sites 0-4, OC at site 5
# Build L_env with the TRANSFORMED tensors
chi_L0 = mps_arrays[0].shape[0]
D_0 = mpo_arrays[0].shape[0]
L_env = init_left_env(chi_L0, D_0, np.float64)

for j in range(5):
    A = mps_arrays[j]
    chi_L, d, chi_R = A.shape
    M = A.reshape(chi_L * d, chi_R)
    Q, R = np.linalg.qr(M)
    mps_arrays[j] = Q.reshape(chi_L, d, -1)
    mps_arrays[j + 1] = np.tensordot(R, mps_arrays[j + 1], axes=(1, 0))
    # Update L_env with the TRANSFORMED tensor
    L_env = update_left_env(L_env, mps_arrays[j], mpo_arrays[j])

print(f"After QR sweep: L_env shape = {L_env.shape}")

# Now sites 0-4 are left-canonical, site 5 is OC
# L_env is correct up to site 5

# Skip full energy computation - just focus on local energy
print(f"(Skipping full energy computation - focusing on local energy)")

# Now do the merge
psi_left = mps_arrays[5]
psi_right = mps_arrays[6]
V = np.ones(psi_left.shape[2], dtype=np.float64)  # Identity V

# Compute energy of the two-site wavefunction before optimization
theta_before = np.einsum('ijk,klm->ijlm', psi_left, psi_right)

# Use optimize_two_site to compute local energy
from pdmrg.numerics.eigensolver import optimize_two_site
E_local_before, _ = optimize_two_site(L_env, R_env, mpo_arrays[5], mpo_arrays[6], theta_before, max_iter=1, tol=1e-12)
print(f"Local energy before optimization (1 iter): {E_local_before:.15f}")

# Run the merge
A_left_new, A_right_new, V_new, E_merge, trunc_err = merge_boundary_tensors(
    psi_left, psi_right, V,
    L_env, R_env, mpo_arrays[5], mpo_arrays[6],
    max_bond=20, max_iter=30, tol=1e-12
)

print(f"Merge energy: {E_merge:.15f}")
print(f"Truncation error: {trunc_err:.2e}")

# Update MPS
mps_arrays[5] = A_left_new
mps_arrays[6] = A_right_new

# Compute local energy after merge
theta_after = np.einsum('ijk,klm->ijlm', A_left_new, A_right_new)
E_local_after, _ = optimize_two_site(L_env, R_env, mpo_arrays[5], mpo_arrays[6], theta_after, max_iter=1, tol=1e-12)
print(f"Local energy after merge (1 iter): {E_local_after:.15f}")

print(f"\nReference: {warmup_energy:.15f}")
print(f"ΔE (local before→local after): {E_local_after - E_local_before:.2e}")
print(f"ΔE (merge→ref): {E_merge - warmup_energy:.2e}")
