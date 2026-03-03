#!/usr/bin/env python3
"""Debug gauge consistency issue"""

import sys
import numpy as np
import quimb.tensor as qtn

sys.path.insert(0, '/home/captain/clawd/work/dmrg-implementations/pdmrg')

from pdmrg.dmrg import serial_warmup
from pdmrg.mps.canonical import get_mpo_tensor_data
from pdmrg.environments.update import init_left_env, init_right_env, update_left_env, update_right_env
from pdmrg.numerics.eigensolver import optimize_two_site

L = 12
mpo = qtn.MPO_ham_heis(L=L, j=1.0, bz=0.0, cyclic=False)
mpo_arrays = [get_mpo_tensor_data(mpo, i) for i in range(L)]

# Get warmup state
mps_arrays, _, warmup_energy = serial_warmup(mpo, L, bond_dim_warmup=20, n_warmup_sweeps=5)
mps_orig = [a.copy() for a in mps_arrays]  # Keep original copy

print(f"Warmup energy: {warmup_energy:.15f}")

# Test 1: Local energy with ORIGINAL tensors (no QR sweep)
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

theta = np.einsum('ijk,klm->ijlm', mps_arrays[5], mps_arrays[6])
E_orig, _ = optimize_two_site(L_env, R_env, mpo_arrays[5], mpo_arrays[6], theta, max_iter=1, tol=1e-12)
print(f"Local energy with ORIGINAL tensors: {E_orig:.15f}")

# Test 2: QR sweep rank 0, original tensors rank 1
mps_arrays = [a.copy() for a in mps_orig]  # Reset

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
    L_env = update_left_env(L_env, mps_arrays[j], mpo_arrays[j])

# R_env still from original (rank 1 didn't QR sweep)
R_env = init_right_env(chi_R_last, D_last, np.float64)
for i in range(L-1, 6, -1):
    R_env = update_right_env(R_env, mps_orig[i], mpo_arrays[i])  # ORIGINAL!

theta = np.einsum('ijk,klm->ijlm', mps_arrays[5], mps_arrays[6])
E_mixed, _ = optimize_two_site(L_env, R_env, mpo_arrays[5], mpo_arrays[6], theta, max_iter=1, tol=1e-12)
print(f"Local energy with MIXED gauge (QR left, orig right): {E_mixed:.15f}")

# Test 3: QR sweep BOTH halves
mps_arrays = [a.copy() for a in mps_orig]  # Reset

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
    L_env = update_left_env(L_env, mps_arrays[j], mpo_arrays[j])

# Also QR sweep sites 6-10
for j in range(6, 11):
    A = mps_arrays[j]
    chi_L, d, chi_R = A.shape
    M = A.reshape(chi_L * d, chi_R)
    Q, R = np.linalg.qr(M)
    mps_arrays[j] = Q.reshape(chi_L, d, -1)
    mps_arrays[j + 1] = np.tensordot(R, mps_arrays[j + 1], axes=(1, 0))

# Now rebuild R_env from QR-transformed tensors
R_env = init_right_env(chi_R_last, D_last, np.float64)
for i in range(L-1, 6, -1):
    R_env = update_right_env(R_env, mps_arrays[i], mpo_arrays[i])  # QR-transformed!

theta = np.einsum('ijk,klm->ijlm', mps_arrays[5], mps_arrays[6])
E_both_qr, _ = optimize_two_site(L_env, R_env, mpo_arrays[5], mpo_arrays[6], theta, max_iter=1, tol=1e-12)
print(f"Local energy with BOTH QR swept: {E_both_qr:.15f}")

print(f"\nReference: {warmup_energy:.15f}")
print(f"ΔE (orig): {E_orig - warmup_energy:.2e}")
print(f"ΔE (mixed): {E_mixed - warmup_energy:.2e}")
print(f"ΔE (both QR): {E_both_qr - warmup_energy:.2e}")
