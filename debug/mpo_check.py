#!/usr/bin/env python3
"""Check MPO tensor extraction from quimb"""

import numpy as np
import quimb.tensor as qtn

L = 4  # Small system for debugging
mpo = qtn.MPO_ham_heis(L=L, j=1.0, bz=0.0, cyclic=False)

print("quimb MPO structure:")
for i in range(L):
    t = mpo[i]
    print(f"  Site {i}: shape={t.shape}, inds={t.inds}")

print("\nExpected MPO shape: (mpo_left, mpo_right, phys_up, phys_down)")
print("quimb indices:")
print(f"  upper_ind_id: {mpo.upper_ind_id}")
print(f"  lower_ind_id: {mpo.lower_ind_id}")

# Extract using PDMRG's function
import sys
sys.path.insert(0, '/home/captain/clawd/work/dmrg-implementations/pdmrg')
from pdmrg.mps.canonical import get_mpo_tensor_data

print("\nPDMRG extraction:")
for i in range(L):
    W = get_mpo_tensor_data(mpo, i)
    print(f"  Site {i}: shape={W.shape}")

# Check index ordering manually for site 1 (bulk)
print("\n=== Manual check for site 1 ===")
t = mpo[1]
print(f"tensor shape: {t.shape}")
print(f"tensor inds: {t.inds}")

# Find indices
upper_name = mpo.upper_ind_id.format(1)  # 'k1'
lower_name = mpo.lower_ind_id.format(1)  # 'b1'
print(f"upper (ket) index: {upper_name}")
print(f"lower (bra) index: {lower_name}")

# Find positions
upper_pos = list(t.inds).index(upper_name)
lower_pos = list(t.inds).index(lower_name)
bond_positions = [i for i in range(4) if i not in [upper_pos, lower_pos]]
print(f"upper_pos={upper_pos}, lower_pos={lower_pos}, bond_positions={bond_positions}")

# What are the bond indices?
for i, idx in enumerate(t.inds):
    if i not in [upper_pos, lower_pos]:
        print(f"  Bond index at pos {i}: {idx}")

# The issue: we need to know which bond is LEFT and which is RIGHT
# quimb uses systematic bond naming
