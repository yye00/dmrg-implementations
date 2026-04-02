"""Tests for incremental environment building."""
import numpy as np
import pytest
import quimb.tensor as qtn

from a2dmrg.hamiltonians.heisenberg import heisenberg_mpo


def _brute_force_energy_at_site(mps_arrays, mpo_arrays, L_env, R_env, site):
    """Compute <psi|H|psi>/<psi|psi> using environments and center tensor.

    Does NOT use build_effective_hamiltonian_1site (which would double-reshape).
    Contracts L @ A @ W @ R @ A* inline.
    """
    A = mps_arrays[site]
    W = mpo_arrays[site]  # (D_L, D_R, d_up, d_down)
    A_conj = A.conj()

    X = np.tensordot(L_env, A, axes=(2, 0))
    Y = np.tensordot(X, W, axes=([1, 2], [0, 2]))
    Z = np.tensordot(Y, R_env, axes=([1, 2], [2, 1]))
    Hv_dot_v = np.tensordot(Z, A_conj, axes=([0, 1, 2], [0, 1, 2]))
    energy = float(np.real(Hv_dot_v))
    norm_sq = float(np.real(np.dot(A.ravel().conj(), A.ravel())))
    return energy / norm_sq


def test_incremental_envs_match_full_energy():
    """Environments from incremental builder must give correct site energies."""
    from a2dmrg.numerics.numpy_utils import extract_mps_arrays, extract_mpo_arrays
    from a2dmrg.environments.environment import build_environments_incremental
    from a2dmrg.numerics.observables import compute_energy

    L = 8
    mpo = heisenberg_mpo(L)
    dmrg = qtn.DMRG2(mpo, bond_dims=20, cutoffs=1e-14)
    dmrg.solve(max_sweeps=10, tol=1e-12, verbosity=0)
    mps = dmrg._k

    E_ref = compute_energy(mps, mpo)
    mps_arrays = extract_mps_arrays(mps)
    mpo_arrays = extract_mpo_arrays(mpo)

    L_envs, R_envs, canon_arrays, lc, rc = build_environments_incremental(mps_arrays, mpo_arrays)

    for i in range(L):
        E_site = _brute_force_energy_at_site(
            canon_arrays, mpo_arrays, L_envs[i], R_envs[i + 1], i
        )
        assert abs(E_site - E_ref) < 1e-8, \
            f"Site {i}: E_site={E_site:.10f} vs E_ref={E_ref:.10f}, diff={abs(E_site-E_ref):.2e}"


def test_incremental_envs_boundary_shapes():
    """Boundary environments must be 1x1x1."""
    from a2dmrg.numerics.numpy_utils import extract_mps_arrays, extract_mpo_arrays
    from a2dmrg.environments.environment import build_environments_incremental

    L = 6
    mpo = heisenberg_mpo(L)
    mps = qtn.MPS_rand_state(L, bond_dim=10, dtype='float64')
    mps_arrays = extract_mps_arrays(mps)
    mpo_arrays = extract_mpo_arrays(mpo)

    L_envs, R_envs, *_ = build_environments_incremental(mps_arrays, mpo_arrays)

    assert L_envs[0].shape == (1, 1, 1)
    assert R_envs[L].shape == (1, 1, 1)
    assert len(L_envs) == L + 1
    assert len(R_envs) == L + 1


def test_incremental_envs_complex():
    """Works with complex128 dtype."""
    from a2dmrg.numerics.numpy_utils import extract_mps_arrays, extract_mpo_arrays
    from a2dmrg.environments.environment import build_environments_incremental

    L = 6
    mpo = heisenberg_mpo(L)
    mps = qtn.MPS_rand_state(L, bond_dim=10, dtype='complex128')
    mps_arrays = extract_mps_arrays(mps)
    mpo_arrays = extract_mpo_arrays(mpo)

    L_envs, R_envs, canon_arrays, lc, rc = build_environments_incremental(mps_arrays, mpo_arrays)

    for i in range(L + 1):
        assert L_envs[i].dtype == np.complex128
        assert R_envs[i].dtype == np.complex128
