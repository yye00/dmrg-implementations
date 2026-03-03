#!/usr/bin/env python3
"""Detailed trace of PDMRG merge to find the 1.24e-9 error"""

import sys
import numpy as np
import quimb.tensor as qtn

sys.path.insert(0, '/home/captain/clawd/work/dmrg-implementations/pdmrg')

from pdmrg.dmrg import serial_warmup
from pdmrg.mps.canonical import get_mpo_tensor_data
from pdmrg.environments.update import init_left_env, init_right_env, update_left_env, update_right_env
from pdmrg.numerics.eigensolver import optimize_two_site
from pdmrg.numerics.effective_ham import apply_heff

L = 12
bond_dim = 20
mpo = qtn.MPO_ham_heis(L=L, j=1.0, bz=0.0, cyclic=False)
mpo_arrays = [get_mpo_tensor_data(mpo, i) for i in range(L)]

# Get warmup state
mps_arrays_orig, _, warmup_energy = serial_warmup(mpo, L, bond_dim_warmup=bond_dim, n_warmup_sweeps=5)
print(f"Warmup energy: {warmup_energy:.15f}")

# Reference
dmrg = qtn.DMRG2(mpo, bond_dims=bond_dim, cutoffs=1e-14)
dmrg.solve(max_sweeps=30, tol=1e-12, verbosity=0)
E_ref = float(np.real(dmrg.energy))
print(f"Reference energy: {E_ref:.15f}")

# Simulate the PDMRG with 2 ranks
# Rank 0: sites 0-5
# Rank 1: sites 6-11

mps_arrays = [a.copy() for a in mps_arrays_orig]

print("\n=== Step 1: canonize_block('left') on rank 0 (sites 0-5) ===")
# QR sweep from site 0 to site 4, leaving site 5 as OC
for j in range(5):
    A = mps_arrays[j]
    chi_L, d, chi_R = A.shape
    M = A.reshape(chi_L * d, chi_R)
    Q, R = np.linalg.qr(M)
    mps_arrays[j] = Q.reshape(chi_L, d, -1)
    mps_arrays[j + 1] = np.tensordot(R, mps_arrays[j + 1], axes=(1, 0))

# Build L_env[5] from left-canonical sites 0-4
L_env = init_left_env(1, mpo_arrays[0].shape[0], np.float64)
for i in range(5):
    L_env = update_left_env(L_env, mps_arrays[i], mpo_arrays[i])
print(f"L_env[5] shape: {L_env.shape}")

print("\n=== Step 2: canonize_block('left') on rank 1 (sites 6-11) ===")
# QR sweep from site 6 to site 10, leaving site 11 as OC
for j in range(6, 11):
    A = mps_arrays[j]
    chi_L, d, chi_R = A.shape
    M = A.reshape(chi_L * d, chi_R)
    Q, R = np.linalg.qr(M)
    mps_arrays[j] = Q.reshape(chi_L, d, -1)
    mps_arrays[j + 1] = np.tensordot(R, mps_arrays[j + 1], axes=(1, 0))

print("\n=== Step 3: rebuild_boundary_r_env on rank 1 ===")
# Right-canonize sites 7-11, leaving site 6 as OC
for j in range(11, 6, -1):
    A = mps_arrays[j]
    chi_L, d, chi_R = A.shape
    M = A.reshape(chi_L, d * chi_R)
    Q, R = np.linalg.qr(M.conj().T)
    L = R.conj().T
    Q_right = Q.conj().T.reshape(-1, d, chi_R)
    mps_arrays[j] = Q_right
    mps_arrays[j - 1] = np.tensordot(mps_arrays[j - 1], L, axes=(2, 0))

# Build R_env[6] from right-canonical sites 7-11
R_env = init_right_env(1, mpo_arrays[11].shape[1], np.float64)
for i in range(11, 6, -1):
    R_env = update_right_env(R_env, mps_arrays[i], mpo_arrays[i])
print(f"R_env[6] shape: {R_env.shape}")

print("\n=== Step 4: boundary_merge at bond 5-6 ===")
psi_left = mps_arrays[5]
psi_right = mps_arrays[6]
print(f"psi_left shape: {psi_left.shape}")
print(f"psi_right shape: {psi_right.shape}")

# Check gauges
def check_gauge(A, name):
    chi_L, d, chi_R = A.shape
    A_mat = A.reshape(chi_L * d, chi_R)
    AdA = A_mat.conj().T @ A_mat
    left_err = np.linalg.norm(AdA - np.eye(chi_R))
    A_mat2 = A.reshape(chi_L, d * chi_R)
    AAd = A_mat2 @ A_mat2.conj().T
    right_err = np.linalg.norm(AAd - np.eye(chi_L))
    if left_err < 1e-10:
        print(f"  {name}: LEFT-canonical (err={left_err:.2e})")
    elif right_err < 1e-10:
        print(f"  {name}: RIGHT-canonical (err={right_err:.2e})")
    else:
        print(f"  {name}: OC (left_err={left_err:.2e}, right_err={right_err:.2e})")

print("\nGauge check:")
for i in [4, 5, 6, 7]:
    check_gauge(mps_arrays[i], f"site {i}")

# Form theta and compute energy
V = np.ones(psi_left.shape[2], dtype=np.float64)  # Identity V
theta = np.einsum('ijk,klm->ijlm', psi_left, psi_right)
print(f"\ntheta shape: {theta.shape}")
print(f"||theta|| = {np.linalg.norm(theta.ravel()):.10f}")
print(f"<theta|theta> = {np.vdot(theta.ravel(), theta.ravel()):.10f}")

# Apply H_eff and compute energy
H_theta = apply_heff(L_env, R_env, mpo_arrays[5], mpo_arrays[6], theta)
E_before_opt = np.real(np.vdot(theta.ravel(), H_theta.ravel()) / np.vdot(theta.ravel(), theta.ravel()))
print(f"\nEnergy before optimization: {E_before_opt:.15f}")

# Optimize
E_opt, theta_opt = optimize_two_site(L_env, R_env, mpo_arrays[5], mpo_arrays[6], theta, max_iter=30, tol=1e-12)
print(f"Energy after optimization: {E_opt:.15f}")

print(f"\n=== Summary ===")
print(f"Warmup:     {warmup_energy:.15f}")
print(f"Reference:  {E_ref:.15f}")
print(f"Before opt: {E_before_opt:.15f}")
print(f"After opt:  {E_opt:.15f}")
print(f"\nΔE (before→ref): {E_before_opt - E_ref:.2e}")
print(f"ΔE (after→ref):  {E_opt - E_ref:.2e}")

# Check if theta is an eigenstate of H_eff
print("\n=== Eigenstate check ===")
# For an eigenstate: H|psi> = E|psi>, so ||H|psi> - E|psi>|| should be ~0
residual = H_theta - E_before_opt * theta
residual_norm = np.linalg.norm(residual.ravel())
print(f"||H|theta> - E|theta>|| = {residual_norm:.2e}")

# For comparison, check theta_opt
H_theta_opt = apply_heff(L_env, R_env, mpo_arrays[5], mpo_arrays[6], theta_opt)
E_opt_check = np.real(np.vdot(theta_opt.ravel(), H_theta_opt.ravel()) / np.vdot(theta_opt.ravel(), theta_opt.ravel()))
residual_opt = H_theta_opt - E_opt_check * theta_opt
residual_opt_norm = np.linalg.norm(residual_opt.ravel())
print(f"||H|theta_opt> - E_opt|theta_opt>|| = {residual_opt_norm:.2e}")

# Check overlap between theta and theta_opt
overlap = np.abs(np.vdot(theta.ravel(), theta_opt.ravel()))
print(f"|<theta|theta_opt>| = {overlap:.10f}")

# Check if theta is normalized
print(f"||theta|| = {np.linalg.norm(theta.ravel()):.10f}")
print(f"||theta_opt|| = {np.linalg.norm(theta_opt.ravel()):.10f}")

# Compute FULL energy of the optimized state
print("\n=== Full energy check ===")
# SVD theta_opt to get new tensors for sites 5 and 6
chi_L, d_L, d_R, chi_R = theta_opt.shape
M_opt = theta_opt.reshape(chi_L * d_L, d_R * chi_R)
U_opt, S_opt, Vh_opt = np.linalg.svd(M_opt, full_matrices=False)
k = min(len(S_opt), bond_dim)
U_opt = U_opt[:, :k]
S_opt = S_opt[:k]
Vh_opt = Vh_opt[:k, :]
A5_new = U_opt.reshape(chi_L, d_L, k)
A6_new = (np.diag(S_opt) @ Vh_opt).reshape(k, d_R, chi_R)

# Update the MPS
mps_opt = [a.copy() for a in mps_arrays]
mps_opt[5] = A5_new
mps_opt[6] = A6_new

# Compute full energy by computing <psi|H|psi>/<psi|psi> directly
print("Computing full energy using direct tensor contraction...")

L_sites = 12  # System size

# Build full left-to-right contraction for <psi|H|psi>
# Start with identity at left boundary
chi_L_init = mps_opt[0].shape[0]
D_L_init = mpo_arrays[0].shape[0]
L_full = np.zeros((chi_L_init, D_L_init, chi_L_init), dtype=np.float64)
L_full[0, 0, 0] = 1.0

for i in range(L_sites):
    L_full = update_left_env(L_full, mps_opt[i], mpo_arrays[i])

# Final trace (L_full should be 1x1x1)
E_full_opt = np.trace(L_full[:, -1, :])
print(f"Full energy of optimized state: {E_full_opt:.15f}")
print(f"ΔE (full_opt - ref): {E_full_opt - E_ref:.2e}")

# Also compute full norm to verify normalization
chi_L_init = mps_opt[0].shape[0]
norm_mat = np.eye(chi_L_init)
for i in range(L_sites):
    A = mps_opt[i]
    norm_mat = np.einsum('ac,asd,cse->de', norm_mat, A.conj(), A)
full_norm = np.trace(norm_mat)
print(f"Full norm of optimized state: {full_norm:.15f}")
