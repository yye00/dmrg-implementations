#!/usr/bin/env python3
"""Simple benchmark file generator using Quimb"""

import numpy as np
import quimb.tensor as qtn
from pathlib import Path
import struct
import sys

def generate_heisenberg_mps_mpo(L, chi, seed=42, output_dir="benchmark_data"):
    """Generate Heisenberg MPS and MPO files"""
    np.random.seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    d = 2  # Physical dimension for spin-1/2

    # Generate random MPS
    print(f"Generating Heisenberg L={L}, chi={chi}...")
    mps = qtn.MPS_rand_state(L, bond_dim=chi, phys_dim=d, dtype=complex)
    mps.normalize()

    # Save MPS
    mps_file = output_path / f"heisenberg_L{L}_chi{chi}_mps.bin"
    with open(mps_file, "wb") as f:
        # Header: L, d, chi
        f.write(struct.pack("<III", L, d, chi))

        # Bond dimensions
        for i in range(L + 1):
            if i == 0 or i == L:
                bond = 1
            else:
                bond = min(chi, d**min(i, L-i))
            f.write(struct.pack("<I", bond))

        # MPS tensors
        for i in range(L):
            tensor = mps[i].data
            tensor_c128 = np.asarray(tensor, dtype=np.complex128)
            f.write(tensor_c128.tobytes())

    print(f"  ✓ Saved: {mps_file}")

    # Check if MPO already exists
    mpo_file = output_path / f"heisenberg_L{L}_mpo.bin"
    if not mpo_file.exists():
        print(f"  ! MPO not generated (use existing or create separately)")
    else:
        print(f"  ✓ MPO exists: {mpo_file}")

    return mps_file


def generate_josephson_mps_mpo(L, chi, n_max=2, seed=42, output_dir="benchmark_data"):
    """Generate Josephson MPS and MPO files"""
    np.random.seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    d = 2 * n_max + 1  # Physical dimension

    # Generate random MPS
    print(f"Generating Josephson L={L}, chi={chi}, n_max={n_max}...")
    mps = qtn.MPS_rand_state(L, bond_dim=chi, phys_dim=d, dtype=complex)
    mps.normalize()

    # Save MPS
    mps_file = output_path / f"josephson_L{L}_n{n_max}_chi{chi}_mps.bin"
    with open(mps_file, "wb") as f:
        # Header: L, d, chi
        f.write(struct.pack("<III", L, d, chi))

        # Bond dimensions
        for i in range(L + 1):
            if i == 0 or i == L:
                bond = 1
            else:
                bond = min(chi, d**min(i, L-i))
            f.write(struct.pack("<I", bond))

        # MPS tensors
        for i in range(L):
            tensor = mps[i].data
            tensor_c128 = np.asarray(tensor, dtype=np.complex128)
            f.write(tensor_c128.tobytes())

    print(f"  ✓ Saved: {mps_file}")

    # Check if MPO already exists
    mpo_file = output_path / f"josephson_L{L}_n{n_max}_mpo.bin"
    if not mpo_file.exists():
        print(f"  ! MPO not generated (use existing or create separately)")
    else:
        print(f"  ✓ MPO exists: {mpo_file}")

    return mps_file


if __name__ == "__main__":
    print("="*70)
    print("Benchmark File Generator - Priority 1")
    print("="*70)
    print()

    # Priority 1: Core benchmarks
    configs = [
        ("heisenberg", 8, 100),
        ("heisenberg", 16, 150),
        ("heisenberg", 24, 200),
        ("josephson", 10, 100),
        ("josephson", 14, 150),
    ]

    for model, L, chi in configs:
        try:
            if model == "heisenberg":
                generate_heisenberg_mps_mpo(L, chi, seed=42)
            else:
                generate_josephson_mps_mpo(L, chi, n_max=2, seed=42)
            print()
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            print()

    print("="*70)
    print("Generation complete!")
    print("="*70)
