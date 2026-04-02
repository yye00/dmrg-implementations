"""Tests for numpy TT-SVD compression."""
import numpy as np
import pytest
import quimb.tensor as qtn


def test_compress_preserves_state():
    """Compressed MPS must have high overlap with original."""
    from a2dmrg.numerics.numpy_utils import extract_mps_arrays
    from a2dmrg.numerics.compression import tt_svd_compress
    from a2dmrg.numerics.observables import compute_overlap_numpy

    L = 8
    mps = qtn.MPS_rand_state(L, bond_dim=30, dtype='float64')
    mps /= mps.norm()
    arrays = extract_mps_arrays(mps)

    compressed = tt_svd_compress(arrays, max_bond=20)
    overlap = compute_overlap_numpy(arrays, compressed)
    # Random MPS compressed from chi=30 to chi=20 retains most weight;
    # exact fidelity depends on singular value spectrum of random state.
    assert abs(overlap) > 0.95


def test_compress_respects_max_bond():
    """All bonds in compressed MPS must be <= max_bond."""
    from a2dmrg.numerics.numpy_utils import extract_mps_arrays
    from a2dmrg.numerics.compression import tt_svd_compress

    L = 10
    mps = qtn.MPS_rand_state(L, bond_dim=50, dtype='float64')
    arrays = extract_mps_arrays(mps)
    max_bond = 20

    compressed = tt_svd_compress(arrays, max_bond=max_bond)
    for i in range(L - 1):
        chi = compressed[i].shape[2]
        assert chi <= max_bond, f"Bond {i}-{i+1}: chi={chi} > max_bond={max_bond}"


def test_compress_normalizes():
    """Compressed MPS must be normalized."""
    from a2dmrg.numerics.numpy_utils import extract_mps_arrays
    from a2dmrg.numerics.compression import tt_svd_compress
    from a2dmrg.numerics.observables import compute_overlap_numpy

    L = 8
    mps = qtn.MPS_rand_state(L, bond_dim=30, dtype='float64')
    arrays = extract_mps_arrays(mps)

    compressed = tt_svd_compress(arrays, max_bond=20, normalize=True)
    norm = compute_overlap_numpy(compressed, compressed)
    assert abs(norm - 1.0) < 1e-10
