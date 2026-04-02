"""Tests for numpy-based coarse-space computation."""
import numpy as np
import pytest
import quimb.tensor as qtn

from a2dmrg.hamiltonians.heisenberg import heisenberg_mpo


def test_numpy_energy_matches_quimb():
    """Energy from numpy arrays must match observables.compute_energy."""
    from a2dmrg.numerics.numpy_utils import extract_mps_arrays, extract_mpo_arrays
    from a2dmrg.numerics.observables import compute_energy, compute_energy_numpy

    L = 8
    mpo = heisenberg_mpo(L)
    dmrg = qtn.DMRG2(mpo, bond_dims=20, cutoffs=1e-14)
    dmrg.solve(max_sweeps=10, tol=1e-12, verbosity=0)

    E_quimb = compute_energy(dmrg._k, mpo)
    mps_arrays = extract_mps_arrays(dmrg._k)
    mpo_arrays = extract_mpo_arrays(mpo)
    E_numpy = compute_energy_numpy(mps_arrays, mpo_arrays)

    assert abs(E_quimb - E_numpy) < 1e-12


def test_numpy_overlap():
    """Overlap of MPS with itself must be ~1 (after normalization)."""
    from a2dmrg.numerics.numpy_utils import extract_mps_arrays
    from a2dmrg.numerics.observables import compute_overlap_numpy

    L = 6
    mps = qtn.MPS_rand_state(L, bond_dim=10, dtype='float64')
    mps /= mps.norm()
    arrays = extract_mps_arrays(mps)

    overlap = compute_overlap_numpy(arrays, arrays)
    assert abs(overlap - 1.0) < 1e-10
