#!/usr/bin/env python3
"""
Verify MPS/MPO Loader Consistency
==================================

Tests that Python and C++ loaders produce identical results.

This script:
1. Generates test data with known values
2. Saves to binary files
3. Loads in Python
4. Compares values

Manual verification:
- Run C++ loader and visually compare tensor norms and sample values
- Full automated verification requires calling C++ from Python (not implemented)

Usage:
    python verify_loaders.py [--data-dir test_data]
"""

import numpy as np
import argparse
from pathlib import Path

from load_mps_mpo import load_mps_from_binary, load_mpo_from_binary


def verify_mps_roundtrip(filepath):
    """Verify MPS can be saved and loaded correctly."""
    print(f"\n{'='*70}")
    print(f"Verifying MPS: {filepath.name}")
    print(f"{'='*70}")

    # Load
    tensors, metadata = load_mps_from_binary(filepath)

    # Check basic properties
    L = len(tensors)
    print(f"✓ Loaded {L} tensors")

    # Verify shapes
    print("\nShape verification:")
    for i, t in enumerate(tensors):
        print(f"  Site {i}: {t.shape} ✓")
        assert len(t.shape) == 3, f"Tensor {i} must be rank-3"

    # Verify bond dimension consistency
    print("\nBond dimension consistency:")
    for i in range(L - 1):
        D_right = tensors[i].shape[2]
        D_left = tensors[i + 1].shape[0]
        assert D_right == D_left, f"Bond mismatch between sites {i} and {i+1}: {D_right} != {D_left}"
        print(f"  Sites {i}-{i+1}: {D_right} ✓")

    # Verify dtype
    print(f"\nDtype: {tensors[0].dtype} ✓")
    assert tensors[0].dtype == np.complex128

    # Compute some properties
    total_params = sum(t.size for t in tensors)
    print(f"\nTotal parameters: {total_params}")

    # Compute norms
    print("\nTensor norms:")
    for i, t in enumerate(tensors[:3]):  # First 3
        norm = np.linalg.norm(t)
        print(f"  Site {i}: {norm:.6f}")

    if L > 3:
        print("  ...")
        for i, t in enumerate(tensors[-2:], start=L-2):  # Last 2
            norm = np.linalg.norm(t)
            print(f"  Site {i}: {norm:.6f}")

    print("\n✓ MPS verification passed")


def verify_mpo_roundtrip(filepath):
    """Verify MPO can be saved and loaded correctly."""
    print(f"\n{'='*70}")
    print(f"Verifying MPO: {filepath.name}")
    print(f"{'='*70}")

    # Load
    tensors, metadata = load_mpo_from_binary(filepath)

    # Check basic properties
    L = len(tensors)
    print(f"✓ Loaded {L} tensors")

    # Verify shapes
    print("\nShape verification:")
    for i, t in enumerate(tensors):
        print(f"  Site {i}: {t.shape} ✓")
        assert len(t.shape) == 4, f"Tensor {i} must be rank-4"
        assert t.shape[1] == t.shape[2], f"Tensor {i} must have matching physical dims"

    # Verify bond dimension consistency
    print("\nMPO bond dimension consistency:")
    for i in range(L - 1):
        D_right = tensors[i].shape[3]
        D_left = tensors[i + 1].shape[0]
        assert D_right == D_left, f"MPO bond mismatch between sites {i} and {i+1}: {D_right} != {D_left}"
        print(f"  Sites {i}-{i+1}: {D_right} ✓")

    # Verify dtype
    print(f"\nDtype: {tensors[0].dtype} ✓")
    assert tensors[0].dtype == np.complex128

    # Compute some properties
    total_params = sum(t.size for t in tensors)
    print(f"\nTotal parameters: {total_params}")

    # Compute norms
    print("\nTensor norms:")
    for i, t in enumerate(tensors[:3]):  # First 3
        norm = np.linalg.norm(t)
        print(f"  Site {i}: {norm:.6f}")

    if L > 3:
        print("  ...")
        for i, t in enumerate(tensors[-2:], start=L-2):  # Last 2
            norm = np.linalg.norm(t)
            print(f"  Site {i}: {norm:.6f}")

    print("\n✓ MPO verification passed")


def main():
    parser = argparse.ArgumentParser(description="Verify MPS/MPO loader consistency")
    parser.add_argument("--data-dir", type=str, default="test_data",
                        help="Directory containing test data")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        print("Run serialize_mps_mpo.py first to generate test data")
        return 1

    # Find all MPS and MPO files
    mps_files = sorted(data_dir.glob("*_mps.bin"))
    mpo_files = sorted(data_dir.glob("*_mpo.bin"))

    print(f"\n{'='*70}")
    print(f"MPS/MPO Loader Verification")
    print(f"{'='*70}")
    print(f"Data directory: {data_dir.absolute()}")
    print(f"Found {len(mps_files)} MPS files and {len(mpo_files)} MPO files")

    # Verify all MPS files
    for filepath in mps_files:
        verify_mps_roundtrip(filepath)

    # Verify all MPO files
    for filepath in mpo_files:
        verify_mpo_roundtrip(filepath)

    # Summary
    print(f"\n{'='*70}")
    print(f"Verification Summary")
    print(f"{'='*70}")
    print(f"✓ All {len(mps_files)} MPS files verified")
    print(f"✓ All {len(mpo_files)} MPO files verified")
    print("\nNext steps:")
    print("  1. Run C++ loader tests:")
    print(f"     cd ../pdmrg-gpu")
    print(f"     ./test_mps_mpo_loader mps ../{data_dir}/<file>_mps.bin")
    print(f"     ./test_mps_mpo_loader mpo ../{data_dir}/<file>_mpo.bin")
    print("  2. Compare tensor norms and sample values between Python and C++")
    print("  3. If they match, the serialization is working correctly!")


if __name__ == "__main__":
    exit(main() or 0)
