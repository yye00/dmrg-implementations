#!/usr/bin/env python3
"""Generate MPO files for Heisenberg and Josephson models"""

import numpy as np
from pathlib import Path
import struct

def generate_heisenberg_mpo(L, output_dir="benchmark_data"):
    """Generate Heisenberg Hamiltonian MPO"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    d = 2  # spin-1/2
    D_mpo = 5  # MPO bond dimension for Heisenberg

    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)

    mpo_file = output_path / f"heisenberg_L{L}_mpo.bin"

    # Build Heisenberg MPO tensors
    tensors = []
    for i in range(L):
        if i == 0:
            # Left boundary: (1, d, d, D_mpo)
            W = np.zeros((1, d, d, D_mpo), dtype=np.complex128)
            W[0, :, :, 0] = I
            W[0, :, :, 1] = sx
            W[0, :, :, 2] = sy
            W[0, :, :, 3] = sz
            W[0, :, :, 4] = -sz/2  # -h*Sz/2 term
        elif i == L - 1:
            # Right boundary: (D_mpo, d, d, 1)
            W = np.zeros((D_mpo, d, d, 1), dtype=np.complex128)
            W[0, :, :, 0] = -sz/2  # -h*Sz/2 term
            W[1, :, :, 0] = sx/2  # Sx term
            W[2, :, :, 0] = sy/2  # Sy term
            W[3, :, :, 0] = sz/2  # Sz term
            W[4, :, :, 0] = I
        else:
            # Bulk: (D_mpo, d, d, D_mpo)
            W = np.zeros((D_mpo, d, d, D_mpo), dtype=np.complex128)
            W[0, :, :, 0] = I
            W[0, :, :, 1] = sx
            W[0, :, :, 2] = sy
            W[0, :, :, 3] = sz
            W[0, :, :, 4] = -sz/2
            W[1, :, :, 4] = sx/2
            W[2, :, :, 4] = sy/2
            W[3, :, :, 4] = sz/2
            W[4, :, :, 4] = I

        tensors.append(W)

    # Save MPO
    with open(mpo_file, "wb") as f:
        # Header: L, d, D_mpo
        f.write(struct.pack("<III", L, d, D_mpo))

        # MPO tensors
        for tensor in tensors:
            f.write(tensor.tobytes())

    print(f"  ✓ Saved: {mpo_file}")
    return mpo_file


def generate_josephson_mpo(L, n_max=2, E_J=1.0, E_C=1.0, output_dir="benchmark_data"):
    """Generate Josephson junction array MPO"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    d = 2 * n_max + 1
    D_mpo = 5  # MPO bond dimension

    # Number operator basis: |n> with n = -n_max, ..., n_max
    n_values = np.arange(-n_max, n_max + 1)

    # Operators
    n_op = np.diag(n_values.astype(complex))  # Number operator
    n2_op = np.diag((n_values**2).astype(complex))  # n^2 operator

    # Phase operators (approximate for simplicity)
    cos_phi = np.zeros((d, d), dtype=complex)
    sin_phi = np.zeros((d, d), dtype=complex)
    for i in range(d-1):
        cos_phi[i, i+1] = cos_phi[i+1, i] = 0.5
        sin_phi[i, i+1] = -0.5j
        sin_phi[i+1, i] = 0.5j

    I = np.eye(d, dtype=complex)

    mpo_file = output_path / f"josephson_L{L}_n{n_max}_mpo.bin"

    # Build Josephson MPO tensors
    tensors = []
    for i in range(L):
        if i == 0:
            W = np.zeros((1, d, d, D_mpo), dtype=np.complex128)
            W[0, :, :, 0] = I
            W[0, :, :, 1] = cos_phi
            W[0, :, :, 2] = sin_phi
            W[0, :, :, 3] = n_op
            W[0, :, :, 4] = E_C * n2_op / 2
        elif i == L - 1:
            W = np.zeros((D_mpo, d, d, 1), dtype=np.complex128)
            W[0, :, :, 0] = E_C * n2_op / 2
            W[1, :, :, 0] = -E_J * cos_phi / 2
            W[2, :, :, 0] = -E_J * sin_phi / 2
            W[3, :, :, 0] = 0  # No direct n-n coupling
            W[4, :, :, 0] = I
        else:
            W = np.zeros((D_mpo, d, d, D_mpo), dtype=np.complex128)
            W[0, :, :, 0] = I
            W[0, :, :, 1] = cos_phi
            W[0, :, :, 2] = sin_phi
            W[0, :, :, 3] = n_op
            W[0, :, :, 4] = E_C * n2_op / 2
            W[1, :, :, 4] = -E_J * cos_phi / 2
            W[2, :, :, 4] = -E_J * sin_phi / 2
            W[4, :, :, 4] = I

        tensors.append(W)

    # Save MPO
    with open(mpo_file, "wb") as f:
        # Header: L, d, D_mpo
        f.write(struct.pack("<III", L, d, D_mpo))

        # MPO tensors
        for tensor in tensors:
            f.write(tensor.tobytes())

    print(f"  ✓ Saved: {mpo_file}")
    return mpo_file


if __name__ == "__main__":
    print("="*70)
    print("MPO File Generator - Priority 1")
    print("="*70)
    print()

    # Generate Heisenberg MPOs
    print("Heisenberg MPOs:")
    for L in [8, 16, 24]:
        generate_heisenberg_mpo(L)

    print()

    # Generate Josephson MPOs
    print("Josephson MPOs:")
    for L in [10, 14]:
        generate_josephson_mpo(L, n_max=2)

    print()
    print("="*70)
    print("MPO generation complete!")
    print("="*70)
