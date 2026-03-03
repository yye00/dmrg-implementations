#!/usr/bin/env python3
"""Compare PDMRG H_eff with quimb's two-site optimization"""

import sys
import numpy as np
import quimb.tensor as qtn

sys.path.insert(0, '/home/captain/clawd/work/dmrg-implementations/pdmrg')

from pdmrg.dmrg import serial_warmup
from pdmrg.mps.canonical import get_mpo_tensor_data
from pdmrg.environments.update import init_left_env, init_right_env, update_left_env, update_right_env
from pdmrg.numerics.eigensolver import optimize_two_site
from pdmrg.numerics.effective_ham import build_heff_operator

L = 12
bond_dim = 20
mpo = qtn.MPO_ham_heis(L=L, j=1.0, bz=0.0, cyclic=False)
mpo_arrays = [get_mpo_tensor_data(mpo, i) for i in range(L)]

# Get warmup state using quimb DMRG2 with tighter settings
dmrg = qtn.DMRG2(mpo, bond_dims=bond_dim, cutoffs=1e-14)
dmrg.solve(max_sweeps=30, tol=1e-12, verbosity=0)
E_ref = float(np.real(dmrg.energy))
print(f"quimb DMRG2 energy: {E_ref:.15f}")

# Get the MPS from quimb and convert to arrays
quimb_mps = dmrg._k.copy()
mps_arrays = []
for i in range(L):
    # quimb tensors have indices (left, phys, right) for middle sites
    # Edge sites might be (phys, right) or (left, phys)
    t = quimb_mps[i]
    if i == 0:
        # First site: might be (bond, phys) shape, need (1, phys, bond)
        arr = t.data
        if arr.ndim == 2:
            arr = arr[None, :, :]  # (1, phys, bond)
        elif arr.ndim == 3:
            pass  # Already (left, phys, right)
    elif i == L-1:
        # Last site: might be (bond, phys) shape, need (bond, phys, 1)
        arr = t.data
        if arr.ndim == 2:
            arr = arr[:, :, None]  # (bond, phys, 1)
        elif arr.ndim == 3:
            pass
    else:
        arr = t.data
    
    # Ensure shape is (left, phys, right)
    if arr.ndim == 2:
        # This shouldn't happen after the above handling
        print(f"Warning: Site {i} has 2D array: {arr.shape}")
    mps_arrays.append(arr)

print(f"\nMPS shapes: {[a.shape for a in mps_arrays]}")

# QR sweep on sites 0-4 to get mixed-canonical form
for j in range(5):
    A = mps_arrays[j]
    chi_L, d, chi_R = A.shape
    M = A.reshape(chi_L * d, chi_R)
    Q, R = np.linalg.qr(M)
    mps_arrays[j] = Q.reshape(chi_L, d, -1)
    mps_arrays[j + 1] = np.tensordot(R, mps_arrays[j + 1], axes=(1, 0))

# Build environments for bond 5-6
chi_L0 = mps_arrays[0].shape[0]
D_0 = mpo_arrays[0].shape[0]
L_env = init_left_env(chi_L0, D_0, np.float64)
for i in range(5):
    L_env = update_left_env(L_env, mps_arrays[i], mpo_arrays[i])

chi_R_last = mps_arrays[-1].shape[2]
D_last = mpo_arrays[-1].shape[1]
R_env = init_right_env(chi_R_last, D_last, np.float64)
for i in range(L-1, 6, -1):
    R_env = update_right_env(R_env, mps_arrays[i], mpo_arrays[i])

# Form theta at bond 5-6
theta = np.einsum('ijk,klm->ijlm', mps_arrays[5], mps_arrays[6])
print(f"theta shape: {theta.shape}")

# Compute energy using PDMRG's H_eff
H_eff, shape_4d = build_heff_operator(L_env, R_env, mpo_arrays[5], mpo_arrays[6], dtype=np.float64)
theta_vec = theta.ravel()
H_theta = H_eff @ theta_vec
E_pdmrg = np.real(np.vdot(theta_vec, H_theta) / np.vdot(theta_vec, theta_vec))
print(f"\nPDMRG H_eff expectation value: {E_pdmrg:.15f}")

# Optimize using PDMRG's eigensolver
E_opt_pdmrg, theta_opt = optimize_two_site(
    L_env, R_env, mpo_arrays[5], mpo_arrays[6], theta,
    max_iter=30, tol=1e-12
)
print(f"PDMRG eigsh energy: {E_opt_pdmrg:.15f}")

# Now use quimb to do the same optimization
# Sweep the quimb MPS to bond 5-6 and get its energy
print(f"\n=== Comparison ===")
print(f"quimb DMRG2 final:    {E_ref:.15f}")
print(f"PDMRG H_eff (theta):  {E_pdmrg:.15f}")
print(f"PDMRG eigsh (opt):    {E_opt_pdmrg:.15f}")
print(f"\nΔE (H_eff vs ref):    {E_pdmrg - E_ref:.2e}")
print(f"ΔE (eigsh vs ref):    {E_opt_pdmrg - E_ref:.2e}")
