"""Tests for numpy MPS/MPO extraction utilities."""
import numpy as np
import pytest
import quimb.tensor as qtn

from a2dmrg.hamiltonians.heisenberg import heisenberg_mpo


def test_extract_mps_shapes():
    """Extracted MPS arrays have shape (chi_L, d, chi_R) with chi_L[0]=1, chi_R[-1]=1."""
    from a2dmrg.numerics.numpy_utils import extract_mps_arrays

    L, chi = 8, 10
    mps = qtn.MPS_rand_state(L, bond_dim=chi, dtype='float64')
    arrays = extract_mps_arrays(mps)

    assert len(arrays) == L
    assert arrays[0].shape[0] == 1, "First site must have chi_L=1"
    assert arrays[-1].shape[2] == 1, "Last site must have chi_R=1"
    for i in range(L):
        assert arrays[i].ndim == 3, f"Site {i} must be 3D (chi_L, d, chi_R)"
        assert arrays[i].shape[1] == 2, f"Physical dim must be 2 (spin-1/2)"
    # Bond dimensions must match between neighbors
    for i in range(L - 1):
        assert arrays[i].shape[2] == arrays[i + 1].shape[0], \
            f"Bond mismatch at ({i},{i+1}): {arrays[i].shape[2]} != {arrays[i+1].shape[0]}"


def test_extract_mpo_shapes():
    """Extracted MPO arrays have shape (D_L, D_R, d, d) with D_L[0]=1, D_R[-1]=1."""
    from a2dmrg.numerics.numpy_utils import extract_mpo_arrays

    L = 8
    mpo = heisenberg_mpo(L)
    arrays = extract_mpo_arrays(mpo)

    assert len(arrays) == L
    assert arrays[0].shape[0] == 1, "First MPO site must have D_L=1"
    assert arrays[-1].shape[1] == 1, "Last MPO site must have D_R=1"
    for i in range(L):
        assert arrays[i].ndim == 4, f"MPO site {i} must be 4D"
        # Convention: (mpo_L, mpo_R, d_up, d_down)
        d = arrays[i].shape[2]
        assert arrays[i].shape[3] == d, "Physical dims must match"
    # Bond dimensions must match between neighbors
    for i in range(L - 1):
        assert arrays[i].shape[1] == arrays[i + 1].shape[0], \
            f"MPO bond mismatch at ({i},{i+1}): {arrays[i].shape[1]} != {arrays[i+1].shape[0]}"


def test_extract_mps_preserves_energy():
    """Energy computed from extracted arrays must match quimb's energy."""
    from a2dmrg.numerics.numpy_utils import extract_mps_arrays, extract_mpo_arrays
    from a2dmrg.numerics.observables import compute_energy

    L = 8
    mpo = heisenberg_mpo(L)
    dmrg = qtn.DMRG2(mpo, bond_dims=20, cutoffs=1e-14)
    dmrg.solve(max_sweeps=10, tol=1e-12, verbosity=0)
    mps = dmrg._k

    E_quimb = float(np.real(dmrg.energy))
    E_obs = compute_energy(mps, mpo)

    assert abs(E_quimb - E_obs) < 1e-10, \
        f"Energy mismatch: quimb={E_quimb}, observables={E_obs}"


def test_extract_complex_mps():
    """Complex MPS extraction works for Josephson-type models."""
    from a2dmrg.numerics.numpy_utils import extract_mps_arrays

    L = 6
    mps = qtn.MPS_rand_state(L, bond_dim=8, dtype='complex128')
    arrays = extract_mps_arrays(mps)

    for i in range(L):
        assert arrays[i].dtype == np.complex128
