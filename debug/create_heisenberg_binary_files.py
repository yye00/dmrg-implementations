#!/usr/bin/env python3
"""
Create binary MPS and MPO files for Heisenberg chain.
These will be used by both CPU and GPU PDMRG implementations.
"""

import sys
sys.path.insert(0, '/home/captain/clawd/work/dmrg-implementations/pdmrg')

import numpy as np
import quimb.tensor as qtn
from pdmrg.mps.canonical import get_tensor_data, get_mpo_tensor_data

def create_heisenberg_files(L=8, bond_dim=32):
    """Create initial MPS and MPO files for Heisenberg chain."""

    # Build MPO
    mpo = qtn.MPO_ham_heis(L=L, j=1.0, bz=0.0, cyclic=False)

    # Create initial MPS using DMRG warmup
    bond_ramp = [10, 20, bond_dim]
    dmrg = qtn.DMRG2(mpo, bond_dims=bond_ramp, cutoffs=1e-14)
    dmrg.solve(tol=1e-12, max_sweeps=10, verbosity=1)

    mps = dmrg.state
    mps.canonize(0)  # Right-canonical form

    print(f"Warmup energy: {dmrg.energy:.15f}")

    # Extract MPS tensors using canonical format
    mps_arrays = []
    for i in range(L):
        A = get_tensor_data(mps, i)
        print(f"MPS[{i}].shape = {A.shape}")
        mps_arrays.append(A)

    # Extract MPO tensors using canonical format
    mpo_arrays = []
    for i in range(L):
        W = get_mpo_tensor_data(mpo, i)
        print(f"MPO[{i}].shape = {W.shape}")
        mpo_arrays.append(W)

    # Check if MPS is real
    is_real = all(np.allclose(A.imag, 0) for A in mps_arrays)
    print(f"MPS is real: {is_real}")

    if is_real:
        print("Converting to float64")
        mps_arrays = [A.real.astype(np.float64) for A in mps_arrays]
        mpo_arrays = [W.real.astype(np.float64) for W in mpo_arrays]

    # Save MPS to binary
    mps_file = f"/tmp/heisenberg_L{L}_mps_initial.bin"
    with open(mps_file, 'wb') as f:
        for A in mps_arrays:
            # Save as complex128 for compatibility
            if A.dtype == np.float64:
                A = A.astype(np.complex128)
            A.tofile(f)
    print(f"Saved MPS to {mps_file}")

    # Save MPO to binary
    mpo_file = f"/tmp/heisenberg_L{L}_mpo.bin"
    with open(mpo_file, 'wb') as f:
        for W in mpo_arrays:
            # Save as complex128 for compatibility
            if W.dtype == np.float64:
                W = W.astype(np.complex128)
            W.tofile(f)
    print(f"Saved MPO to {mpo_file}")

    # Print bond dimensions
    bond_dims = [mps_arrays[0].shape[0]]  # Left boundary
    for A in mps_arrays:
        bond_dims.append(A.shape[2])
    print(f"Bond dimensions: {bond_dims}")

    return dmrg.energy

if __name__ == '__main__':
    import sys
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    bond_dim = int(sys.argv[2]) if len(sys.argv) > 2 else 32

    energy = create_heisenberg_files(L, bond_dim)
    print(f"Reference energy: {energy:.15f}")
