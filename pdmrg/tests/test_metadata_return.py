#!/usr/bin/env python3
"""
Test that pdmrg_main() metadata return works correctly.

Tests all three execution paths:
1. np=1 with warmup (early return)
2. np=1 with random init (serial PDMRG)
3. np>1 (parallel PDMRG)
"""

import sys
import numpy as np
from pathlib import Path

pdmrg_root = Path(__file__).parent.parent
sys.path.insert(0, str(pdmrg_root))

from mpi4py import MPI
import quimb.tensor as qtn
from pdmrg.dmrg import pdmrg_main


def test_metadata(test_name, **kwargs):
    """Run pdmrg_main with return_metadata=True and print results."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    L = 8
    mpo = qtn.MPO_ham_heis(L=L, j=1.0, bz=0.0, cyclic=False)

    if rank == 0:
        print(f"\n{'='*70}")
        print(f"TEST: {test_name}")
        print(f"{'='*70}")

    energy, pmps, metadata = pdmrg_main(
        L=L,
        mpo=mpo,
        max_sweeps=5,
        bond_dim=10,
        bond_dim_warmup=10,
        n_warmup_sweeps=3,
        tol=1e-8,
        comm=comm,
        verbose=False,
        return_metadata=True,
        **kwargs
    )

    if rank == 0:
        print(f"\nEnergy: {energy:.12f}")
        print(f"\nMetadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")

    return energy, metadata


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_procs = comm.Get_size()

    if rank == 0:
        print("\n" + "="*70)
        print("PDMRG METADATA RETURN TEST")
        print("="*70)
        print(f"Running with n_procs={n_procs}")

    # Test 1: np=1 with warmup (early return path)
    if n_procs == 1:
        energy1, meta1 = test_metadata(
            "np=1 with warmup (early return)",
            random_init_flag=False
        )

        # Verify metadata
        assert meta1["early_return"] == True, "Expected early_return=True"
        assert meta1["algorithm_executed"] == "quimb DMRG2 warmup (early return)"
        assert meta1["warmup_used"] == True
        assert meta1["skip_opt"] is None  # Not applicable for np=1

        if rank == 0:
            print("\n✓ Test 1 PASSED: Early return metadata correct")

        # Test 2: np=1 with random init (serial PDMRG path)
        energy2, meta2 = test_metadata(
            "np=1 with random init (serial PDMRG)",
            random_init_flag=True
        )

        # Verify metadata
        assert meta2["early_return"] == False, "Expected early_return=False"
        assert meta2["algorithm_executed"] == "PDMRG serial sweeps"
        assert meta2["warmup_used"] == False
        assert meta2["random_init"] == True
        assert meta2["skip_opt"] is None  # Not applicable for np=1

        if rank == 0:
            print("\n✓ Test 2 PASSED: Serial PDMRG metadata correct")

    elif n_procs >= 2:
        # Test 3: np>1 (parallel PDMRG path)
        energy3, meta3 = test_metadata(
            "np>1 parallel PDMRG",
            random_init_flag=False
        )

        # Verify metadata
        assert meta3["early_return"] == False, "Expected early_return=False"
        assert meta3["algorithm_executed"] == "PDMRG parallel sweeps"
        assert meta3["warmup_used"] == True
        assert meta3["skip_opt"] == True  # Always True for multi-rank
        assert meta3["np"] == n_procs

        if rank == 0:
            print("\n✓ Test 3 PASSED: Parallel PDMRG metadata correct")

    if rank == 0:
        print("\n" + "="*70)
        print("ALL TESTS PASSED")
        print("="*70)


if __name__ == '__main__':
    main()
