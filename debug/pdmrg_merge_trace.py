#!/usr/bin/env python3
"""Trace the PDMRG merge step to understand energy deviation"""

import sys
import numpy as np
import quimb.tensor as qtn

sys.path.insert(0, '/home/captain/clawd/work/dmrg-implementations/pdmrg')

from pdmrg.dmrg import serial_warmup
from pdmrg.mps.canonical import get_mpo_tensor_data
from pdmrg.environments.update import init_left_env, init_right_env, update_left_env, update_right_env
from pdmrg.parallel.merge import merge_boundary_tensors
from pdmrg.numerics.eigensolver import optimize_two_site

L = 12
bond_dim = 20
mpo = qtn.MPO_ham_heis(L=L, j=1.0, bz=0.0, cyclic=False)
mpo_arrays = [get_mpo_tensor_data(mpo, i) for i in range(L)]

# Get warmup state
mps_arrays_orig, _, warmup_energy = serial_warmup(mpo, L, bond_dim_warmup=bond_dim, n_warmup_sweeps=5)
mps_arrays = [a.copy() for a in mps_arrays_orig]
print(f"Warmup energy: {warmup_energy:.15f}")

# Reference: quimb DMRG2
dmrg = qtn.DMRG2(mpo, bond_dims=bond_dim, cutoffs=1e-14)
dmrg.solve(max_sweeps=30, tol=1e-12, verbosity=0)
E_ref = float(np.real(dmrg.energy))
print(f"Reference energy: {E_ref:.15f}")

# Test 1: Standard serial DMRG two-site optimization at bond 5-6
# Build environments from the ORIGINAL MPS (no QR sweep)
print("\n=== Test 1: Serial DMRG optimization at bond 5-6 ===")

chi_L0 = mps_arrays[0].shape[0]
D_0 = mpo_arrays[0].shape[0]
L_env_serial = init_left_env(chi_L0, D_0, np.float64)
for i in range(5):
    L_env_serial = update_left_env(L_env_serial, mps_arrays[i], mpo_arrays[i])

chi_R_last = mps_arrays[-1].shape[2]
D_last = mpo_arrays[-1].shape[1]
R_env_serial = init_right_env(chi_R_last, D_last, np.float64)
for i in range(L-1, 6, -1):
    R_env_serial = update_right_env(R_env_serial, mps_arrays[i], mpo_arrays[i])

theta_serial = np.einsum('ijk,klm->ijlm', mps_arrays[5], mps_arrays[6])
E_serial, theta_opt_serial = optimize_two_site(
    L_env_serial, R_env_serial, mpo_arrays[5], mpo_arrays[6], theta_serial,
    max_iter=30, tol=1e-12
)
print(f"Serial DMRG energy at bond 5-6: {E_serial:.15f}")

# Test 2: After QR sweep on rank 0, rebuild L_env
print("\n=== Test 2: After QR sweep on sites 0-4 ===")

for j in range(5):
    A = mps_arrays[j]
    chi_L, d, chi_R = A.shape
    M = A.reshape(chi_L * d, chi_R)
    Q, R = np.linalg.qr(M)
    mps_arrays[j] = Q.reshape(chi_L, d, -1)
    mps_arrays[j + 1] = np.tensordot(R, mps_arrays[j + 1], axes=(1, 0))

# Rebuild L_env from QR-transformed tensors
L_env = init_left_env(chi_L0, D_0, np.float64)
for i in range(5):
    L_env = update_left_env(L_env, mps_arrays[i], mpo_arrays[i])

# R_env stays the same (sites 7-11 unchanged)
R_env = R_env_serial.copy()

theta_qr = np.einsum('ijk,klm->ijlm', mps_arrays[5], mps_arrays[6])
E_qr, theta_opt_qr = optimize_two_site(
    L_env, R_env, mpo_arrays[5], mpo_arrays[6], theta_qr,
    max_iter=30, tol=1e-12
)
print(f"Energy after QR sweep: {E_qr:.15f}")

print(f"\n=== Summary ===")
print(f"Warmup:     {warmup_energy:.15f}")
print(f"Reference:  {E_ref:.15f}")
print(f"Serial opt: {E_serial:.15f}")
print(f"After QR:   {E_qr:.15f}")
print(f"\nΔE (warmup → ref):   {warmup_energy - E_ref:.2e}")
print(f"ΔE (serial → ref):   {E_serial - E_ref:.2e}")
print(f"ΔE (QR → ref):       {E_qr - E_ref:.2e}")
