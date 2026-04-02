"""Tests for fast micro-steps using cached environments."""
import numpy as np
import pytest
import quimb.tensor as qtn

from a2dmrg.hamiltonians.heisenberg import heisenberg_mpo


def test_fast_microstep_1site_energy():
    """Fast 1-site micro-step must produce lower energy than input."""
    from a2dmrg.numerics.numpy_utils import extract_mps_arrays, extract_mpo_arrays
    from a2dmrg.environments.environment import build_environments_incremental
    from a2dmrg.numerics.local_microstep import fast_microstep_1site
    from a2dmrg.numerics.observables import compute_energy

    L = 8
    mpo = heisenberg_mpo(L)
    dmrg = qtn.DMRG2(mpo, bond_dims=20, cutoffs=1e-14)
    dmrg.solve(max_sweeps=5, tol=1e-8, verbosity=0)
    mps = dmrg._k

    E_before = compute_energy(mps, mpo)
    mps_arrays = extract_mps_arrays(mps)
    mpo_arrays = extract_mpo_arrays(mpo)
    L_envs, R_envs, canon_arrays, lc, rc = build_environments_incremental(mps_arrays, mpo_arrays)

    # Optimize site 4
    site = 4
    optimized_tensor, eigval = fast_microstep_1site(
        canon_arrays[site], mpo_arrays[site],
        L_envs[site], R_envs[site + 1],
        site, L,
    )
    assert optimized_tensor.shape == canon_arrays[site].shape
    # Eigenvalue should be <= current energy (variational principle)
    assert eigval <= E_before + 1e-10


def test_fast_microstep_2site_energy():
    """Fast 2-site micro-step must produce lower energy than input."""
    from a2dmrg.numerics.numpy_utils import extract_mps_arrays, extract_mpo_arrays
    from a2dmrg.environments.environment import build_environments_incremental
    from a2dmrg.numerics.local_microstep import fast_microstep_2site
    from a2dmrg.numerics.observables import compute_energy

    L = 8
    mpo = heisenberg_mpo(L)
    dmrg = qtn.DMRG2(mpo, bond_dims=20, cutoffs=1e-14)
    dmrg.solve(max_sweeps=5, tol=1e-8, verbosity=0)
    mps = dmrg._k

    E_before = compute_energy(mps, mpo)
    mps_arrays = extract_mps_arrays(mps)
    mpo_arrays = extract_mpo_arrays(mpo)
    L_envs, R_envs, canon_arrays, lc, rc = build_environments_incremental(mps_arrays, mpo_arrays)

    site = 3
    U, SVh, eigval = fast_microstep_2site(
        canon_arrays[site], rc[site + 1],
        mpo_arrays[site], mpo_arrays[site + 1],
        L_envs[site], R_envs[site + 2],
        site, L, max_bond=20,
    )
    assert eigval <= E_before + 1e-10
