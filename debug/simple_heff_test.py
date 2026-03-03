#!/usr/bin/env python3
"""Simple test: compare quimb DMRG energy computation with PDMRG's H_eff"""

import numpy as np
import quimb.tensor as qtn

import sys
sys.path.insert(0, '/home/captain/clawd/work/dmrg-implementations/pdmrg')

from pdmrg.mps.canonical import get_tensor_data, get_mpo_tensor_data
from pdmrg.environments.update import init_left_env, init_right_env, update_left_env, update_right_env
from pdmrg.numerics.effective_ham import apply_heff

L = 4
mpo = qtn.MPO_ham_heis(L=L, j=1.0, bz=0.0, cyclic=False)

# Run quimb DMRG to convergence
dmrg = qtn.DMRG2(mpo, bond_dims=10, cutoffs=1e-14)
dmrg.solve(max_sweeps=20, tol=1e-12, verbosity=0)
E_quimb = float(np.real(dmrg.energy))
print(f"quimb DMRG2 energy: {E_quimb:.15f}")

# Extract MPS in our convention
mps = dmrg._k
mps_arrays = [get_tensor_data(mps, i) for i in range(L)]
mpo_arrays = [get_mpo_tensor_data(mpo, i) for i in range(L)]

print(f"\nMPS shapes: {[a.shape for a in mps_arrays]}")
print(f"MPO shapes: {[w.shape for w in mpo_arrays]}")

# Put in mixed-canonical form for bond 1-2
# Sites 0 left-canonical, sites 1-2 are the two-site block, site 3 right-canonical
# quimb ends with all sites left-canonical, so we need to right-canonize sites 2,3

# Right-canonize site 3 (LQ decomposition: A = L @ Q where Q is right-isometric)
A3 = mps_arrays[3]
chi_L3, d3, chi_R3 = A3.shape
M3 = A3.reshape(chi_L3, d3 * chi_R3)
Q3, R3 = np.linalg.qr(M3.conj().T)  # QR of M^H -> M = R^H @ Q^H
L3 = R3.conj().T  # L = R^H
Q3_right = Q3.conj().T.reshape(-1, d3, chi_R3)  # Q_right = Q^H reshaped
mps_arrays[3] = Q3_right
mps_arrays[2] = np.tensordot(mps_arrays[2], L3, axes=(2, 0))

# Right-canonize site 2
A2 = mps_arrays[2]
chi_L2, d2, chi_R2 = A2.shape
M2 = A2.reshape(chi_L2, d2 * chi_R2)
Q2, R2 = np.linalg.qr(M2.conj().T)
L2 = R2.conj().T
Q2_right = Q2.conj().T.reshape(-1, d2, chi_R2)
mps_arrays[2] = Q2_right
mps_arrays[1] = np.tensordot(mps_arrays[1], L2, axes=(2, 0))

print(f"\nAfter QR:")
print(f"MPS shapes: {[a.shape for a in mps_arrays]}")

# Build environments
# L_env at site 1: includes site 0
chi_L0 = mps_arrays[0].shape[0]
D0 = mpo_arrays[0].shape[0]
L_env = init_left_env(chi_L0, D0, np.float64)
L_env = update_left_env(L_env, mps_arrays[0], mpo_arrays[0])
print(f"\nL_env shape: {L_env.shape}")

# R_env at site 2: includes site 3
chi_R3 = mps_arrays[3].shape[2]
D3 = mpo_arrays[3].shape[1]
R_env = init_right_env(chi_R3, D3, np.float64)
R_env = update_right_env(R_env, mps_arrays[3], mpo_arrays[3])
print(f"R_env shape: {R_env.shape}")

# Form two-site wavefunction
theta = np.einsum('ijk,klm->ijlm', mps_arrays[1], mps_arrays[2])
print(f"theta shape: {theta.shape}")

# Apply H_eff
H_theta = apply_heff(L_env, R_env, mpo_arrays[1], mpo_arrays[2], theta)
print(f"H_theta shape: {H_theta.shape}")

# Compute expectation value
E_local = np.real(np.vdot(theta.ravel(), H_theta.ravel()) / np.vdot(theta.ravel(), theta.ravel()))
print(f"\nPDMRG H_eff energy at bond 1-2: {E_local:.15f}")

print(f"\n=== Comparison ===")
print(f"quimb DMRG2:     {E_quimb:.15f}")
print(f"PDMRG H_eff:     {E_local:.15f}")
print(f"ΔE:              {E_local - E_quimb:.2e}")

# Debug: trace through environment construction step by step
print(f"\n=== Debug: L_env construction ===")
L_init = init_left_env(1, 1, np.float64)  # chi_L=1 for leftmost, D=1 for left MPO edge
print(f"L_init shape: {L_init.shape}, L_init[0,0,0]={L_init[0,0,0]}")

# Manually do the update
A0 = mps_arrays[0]
W0 = mpo_arrays[0]
print(f"A0 shape: {A0.shape}, W0 shape: {W0.shape}")

# The correct init should use D from the CURRENT MPO tensor (left bond)
D_left = mpo_arrays[0].shape[0]  # This is 1 for the left edge
chi_L = mps_arrays[0].shape[0]  # This is 1 for the left edge
print(f"chi_L={chi_L}, D_left={D_left}")

L_correct = init_left_env(chi_L, D_left, np.float64)
print(f"L_correct shape: {L_correct.shape}")

L_after = update_left_env(L_correct, A0, W0)
print(f"L_after shape: {L_after.shape}")
print(f"L_after sum: {np.abs(L_after).sum():.6f}")

print(f"\n=== Debug: R_env construction ===")
R_init = init_right_env(1, 1, np.float64)  # chi_R=1 for rightmost, D=1 for right MPO edge
print(f"R_init shape: {R_init.shape}, R_init[0,0,0]={R_init[0,0,0]}")

A3 = mps_arrays[3]
W3 = mpo_arrays[3]
print(f"A3 shape: {A3.shape}, W3 shape: {W3.shape}")

D_right = mpo_arrays[3].shape[1]  # This is 1 for the right edge
chi_R = mps_arrays[3].shape[2]  # This is 1 for the right edge
print(f"chi_R={chi_R}, D_right={D_right}")

R_correct = init_right_env(chi_R, D_right, np.float64)
print(f"R_correct shape: {R_correct.shape}")

R_after = update_right_env(R_correct, A3, W3)
print(f"R_after shape: {R_after.shape}")
print(f"R_after sum: {np.abs(R_after).sum():.6f}")

# Check theta normalization
theta_norm = np.linalg.norm(theta.ravel())
print(f"\n=== Debug: theta normalization ===")
print(f"||theta|| = {theta_norm:.6f}")

# The local norm should be computed as <theta|I_env|theta> where I_env has MPO = Identity
# For mixed-canonical MPS with OC at sites 1-2, ||theta|| should be 1.0

# Let's check if the MPS was already normalized
mps_norm = 0.0
for i, A in enumerate(mps_arrays):
    mps_norm += np.linalg.norm(A.ravel())**2
print(f"Sum of ||A_i||^2 = {mps_norm:.6f}")

# Check individual tensor norms
for i, A in enumerate(mps_arrays):
    print(f"  ||A[{i}]||^2 = {np.linalg.norm(A.ravel())**2:.6f}")

# Check if we're computing <theta|H|theta> correctly
# The issue might be that we need to account for overlaps properly
# Let me compute <theta|theta> using the environments with identity MPO

print(f"\n=== Debug: <theta|theta> via environments ===")
# For this we need identity MPO environments
# Actually, for mixed-canonical form at bond 1-2:
# <psi|psi> = Tr(L_norm @ theta^dag @ theta @ R_norm)
# where L_norm and R_norm are the norm environments (MPO = identity)

# The bra-ket contraction of theta with itself
theta_sq = np.vdot(theta.ravel(), theta.ravel())
print(f"<theta|theta> (flat) = {theta_sq:.6f}")

# If the MPS is in proper mixed-canonical form, <theta|theta> should equal 1
# But if sites 0 is left-canonical and sites 2,3 are right-canonical...

# Actually let's compute <psi|H|psi> manually for the full MPS
print(f"\n=== Manual full energy computation ===")
# <psi|H|psi> = contract everything
# This is expensive but let's do it for debugging

# Contract: L_init @ A0* @ W0 @ A0 @ L_after_1 @ A1* @ W1 @ A1 @ ... @ R_init
# Actually this gets complicated. Let me just verify the issue is in environments.
