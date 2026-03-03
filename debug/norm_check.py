#!/usr/bin/env python3
"""Check normalization through the MPS extraction and QR sweep"""

import numpy as np
import quimb.tensor as qtn

import sys
sys.path.insert(0, '/home/captain/clawd/work/dmrg-implementations/pdmrg')

from pdmrg.mps.canonical import get_tensor_data

L = 4
mpo = qtn.MPO_ham_heis(L=L, j=1.0, bz=0.0, cyclic=False)

# Run quimb DMRG
dmrg = qtn.DMRG2(mpo, bond_dims=10, cutoffs=1e-14)
dmrg.solve(max_sweeps=20, tol=1e-12, verbosity=0)
print(f"quimb DMRG2 energy: {dmrg.energy:.15f}")

# Check quimb MPS normalization
quimb_mps = dmrg._k
print(f"\nquimb MPS norm: {quimb_mps.norm():.10f}")

# What is the OC after DMRG?
# quimb DMRG2 ends with OC at some position
print(f"quimb DMRG finished at position/direction: might vary")

# Extract tensors
mps_arrays = [get_tensor_data(quimb_mps, i) for i in range(L)]
print(f"\nExtracted MPS shapes: {[a.shape for a in mps_arrays]}")

# Check gauge of extracted tensors
print("\nGauge check for extracted tensors:")
for i, A in enumerate(mps_arrays):
    chi_L, d, chi_R = A.shape
    A_mat = A.reshape(chi_L * d, chi_R)
    AdA = A_mat.conj().T @ A_mat
    left_err = np.linalg.norm(AdA - np.eye(chi_R))
    
    A_mat2 = A.reshape(chi_L, d * chi_R)
    AAd = A_mat2 @ A_mat2.conj().T
    right_err = np.linalg.norm(AAd - np.eye(chi_L))
    
    if left_err < 1e-10:
        gauge = "LEFT"
    elif right_err < 1e-10:
        gauge = "RIGHT"
    else:
        gauge = "OC"
    print(f"  Site {i}: {gauge} (left_err={left_err:.2e}, right_err={right_err:.2e})")

# Compute full MPS norm by contracting all tensors
# <psi|psi> = Tr(A0^dag @ A0 @ A1^dag @ A1 @ ... @ A_{L-1}^dag @ A_{L-1})
# For left-canonical sites: A^dag @ A = I
# For right-canonical sites: A @ A^dag = I
# So the norm is concentrated at the OC

# Contract to compute norm
print("\nComputing MPS norm via contraction:")
# Start with identity at left boundary
norm_mat = np.eye(1)
for i, A in enumerate(mps_arrays):
    chi_L, d, chi_R = A.shape
    A_mat = A.reshape(chi_L, d * chi_R)
    A_conj_mat = A.conj().reshape(chi_L, d * chi_R)
    # norm_mat @ A^dag @ A -> (chi_L, chi_L) @ (chi_L, d*chi_R) then trace over physical
    # Actually, need to contract properly
    # <...|A|...> -> transfer matrix approach
    # T[i,j] = sum_s A[i,s,k] * A*[j,s,k] (summing over s and k gives next norm_mat)
    # Wait, that's not right either.
    
    # Let me do it properly:
    # norm_tensor = einsum('ac,bsc,dsd->bd', norm_mat, A, A.conj())
    # This contracts: norm_mat[a,c] * A[c,s,b] * A*[c,s,d] -> need to be careful with indices
    
    # For left-to-right transfer: 
    # new_norm[b,d] = sum_{a,s} old_norm[a,a] * A[a,s,b] * A*[a,s,d]
    # But norm_mat might not be diagonal...
    
    # Let me use einsum:
    # norm_mat: (chi_L, chi_L) where first index is bra, second is ket
    # A: (chi_L, d, chi_R) is ket
    # A*: (chi_L, d, chi_R) is bra
    # new_norm[b,d] = norm_mat[a,c] * A*[a,s,b] * A[c,s,d]
    new_norm = np.einsum('ac,asb,csd->bd', norm_mat, A.conj(), A)
    norm_mat = new_norm
    print(f"  After site {i}: norm_mat shape={norm_mat.shape}, trace={np.trace(norm_mat):.10f}")

print(f"\nFinal norm = {np.trace(norm_mat):.10f}")

# Now do QR sweep and check norm
print("\n=== After QR sweep on site 0 ===")
mps_arrays_qr = [a.copy() for a in mps_arrays]
A0 = mps_arrays_qr[0]
chi_L, d, chi_R = A0.shape
M = A0.reshape(chi_L * d, chi_R)
Q, R = np.linalg.qr(M)
mps_arrays_qr[0] = Q.reshape(chi_L, d, -1)
mps_arrays_qr[1] = np.tensordot(R, mps_arrays_qr[1], axes=(1, 0))

print(f"QR'd MPS shapes: {[a.shape for a in mps_arrays_qr]}")

# Check norm after QR
norm_mat = np.eye(1)
for i, A in enumerate(mps_arrays_qr):
    new_norm = np.einsum('ac,asb,csd->bd', norm_mat, A.conj(), A)
    norm_mat = new_norm
    print(f"  After site {i}: norm_mat shape={norm_mat.shape}, trace={np.trace(norm_mat):.10f}")

print(f"\nFinal norm after QR = {np.trace(norm_mat):.10f}")

# Check theta norm
theta = np.einsum('ijk,klm->ijlm', mps_arrays_qr[1], mps_arrays_qr[2])
print(f"\n<theta|theta> = {np.vdot(theta.ravel(), theta.ravel()):.10f}")
