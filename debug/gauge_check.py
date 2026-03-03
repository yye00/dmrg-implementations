#!/usr/bin/env python3
"""Check MPS gauge after serial warmup"""

import sys
import numpy as np
import quimb.tensor as qtn

sys.path.insert(0, '/home/captain/clawd/work/dmrg-implementations/pdmrg')

from pdmrg.dmrg import serial_warmup
from pdmrg.mps.canonical import get_mpo_tensor_data

L = 12
bond_dim = 20
mpo = qtn.MPO_ham_heis(L=L, j=1.0, bz=0.0, cyclic=False)
mpo_arrays = [get_mpo_tensor_data(mpo, i) for i in range(L)]

# Get warmup state
mps_arrays, _, warmup_energy = serial_warmup(mpo, L, bond_dim_warmup=bond_dim, n_warmup_sweeps=5)
print(f"Warmup energy: {warmup_energy:.15f}")

# Check gauge: is each site left-canonical or right-canonical?
# Left-canonical: A^dag A = I (on the left-right bond)
# Right-canonical: A A^dag = I (on the left-right bond)

print("\nGauge check for each site:")
for i in range(L):
    A = mps_arrays[i]
    chi_L, d, chi_R = A.shape
    
    # Check left-canonical: reshape to (chi_L * d, chi_R), compute A^dag A
    A_mat = A.reshape(chi_L * d, chi_R)
    AdA = A_mat.conj().T @ A_mat
    left_err = np.linalg.norm(AdA - np.eye(chi_R))
    
    # Check right-canonical: reshape to (chi_L, d * chi_R), compute A A^dag
    A_mat2 = A.reshape(chi_L, d * chi_R)
    AAd = A_mat2 @ A_mat2.conj().T
    right_err = np.linalg.norm(AAd - np.eye(chi_L))
    
    if left_err < 1e-10:
        gauge = "LEFT-canonical"
    elif right_err < 1e-10:
        gauge = "RIGHT-canonical"
    else:
        gauge = "MIXED (OC?)"
    
    print(f"  Site {i:2d}: {gauge:16s} (left_err={left_err:.2e}, right_err={right_err:.2e})")

# After QR sweep on sites 0-4:
print("\nAfter QR sweep on sites 0-4:")
for j in range(5):
    A = mps_arrays[j]
    chi_L, d, chi_R = A.shape
    M = A.reshape(chi_L * d, chi_R)
    Q, R = np.linalg.qr(M)
    mps_arrays[j] = Q.reshape(chi_L, d, -1)
    mps_arrays[j + 1] = np.tensordot(R, mps_arrays[j + 1], axes=(1, 0))

for i in range(L):
    A = mps_arrays[i]
    chi_L, d, chi_R = A.shape
    
    A_mat = A.reshape(chi_L * d, chi_R)
    AdA = A_mat.conj().T @ A_mat
    left_err = np.linalg.norm(AdA - np.eye(chi_R))
    
    A_mat2 = A.reshape(chi_L, d * chi_R)
    AAd = A_mat2 @ A_mat2.conj().T
    right_err = np.linalg.norm(AAd - np.eye(chi_L))
    
    if left_err < 1e-10:
        gauge = "LEFT-canonical"
    elif right_err < 1e-10:
        gauge = "RIGHT-canonical"
    else:
        gauge = "MIXED (OC?)"
    
    print(f"  Site {i:2d}: {gauge:16s} (left_err={left_err:.2e}, right_err={right_err:.2e})")
