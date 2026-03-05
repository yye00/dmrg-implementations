"""Unit tests for numerical routines: SVD, effective Hamiltonian, eigensolver."""

import numpy as np
import pytest
import quimb.tensor as qtn

from pdmrg.numerics.accurate_svd import accurate_svd, compute_v_from_svd, truncated_svd
from pdmrg.numerics.effective_ham import apply_heff, build_heff_operator
from pdmrg.numerics.eigensolver import optimize_two_site
from pdmrg.mps.canonical import get_tensor_data, get_mpo_tensor_data
from pdmrg.environments.update import (
    update_left_env, update_right_env,
    init_left_env, init_right_env,
)


class TestAccurateSVD:
    def test_basic_svd(self):
        """accurate_svd reproduces standard SVD for well-conditioned matrices."""
        np.random.seed(42)
        M = np.random.randn(10, 8)
        U, S, Vh = accurate_svd(M)
        assert np.allclose(U @ np.diag(S) @ Vh, M, atol=1e-12)

    def test_small_singular_values(self):
        """accurate_svd improves accuracy for small singular values."""
        np.random.seed(42)
        # Create matrix with large condition number
        U0 = np.linalg.qr(np.random.randn(20, 20))[0]
        V0 = np.linalg.qr(np.random.randn(20, 20))[0]
        S0 = np.array([1.0] * 5 + [1e-6] * 5 + [1e-12] * 5 + [1e-15] * 5)
        M = U0 @ np.diag(S0) @ V0

        U, S, Vh = accurate_svd(M)
        M_reconstructed = U @ np.diag(S) @ Vh
        assert np.allclose(M_reconstructed, M, atol=1e-14)

    def test_complex_matrix(self):
        """accurate_svd works with complex matrices."""
        np.random.seed(42)
        M = np.random.randn(8, 6) + 1j * np.random.randn(8, 6)
        U, S, Vh = accurate_svd(M)
        assert np.allclose(U @ np.diag(S) @ Vh, M, atol=1e-12)

    def test_compute_v(self):
        """compute_v_from_svd returns regularized inverse."""
        S = np.array([1.0, 0.5, 1e-15])
        V = compute_v_from_svd(S)
        assert V[0] == pytest.approx(1.0)
        assert V[1] == pytest.approx(2.0)
        assert V[2] == pytest.approx(1.0 / 1e-12)  # regularized


class TestTruncatedSVD:
    def test_truncation(self):
        np.random.seed(42)
        M = np.random.randn(10, 8)
        U, S, Vh, trunc_err = truncated_svd(M, max_bond=3)
        assert U.shape[1] == 3
        assert len(S) == 3
        assert Vh.shape[0] == 3
        assert trunc_err > 0

    def test_no_truncation(self):
        np.random.seed(42)
        M = np.random.randn(4, 3)
        U, S, Vh, trunc_err = truncated_svd(M, max_bond=100)
        assert len(S) == 3
        assert trunc_err == pytest.approx(0.0)


class TestEffectiveHamiltonian:
    def test_heff_matches_exact_4site(self):
        """H_eff with identity environments reproduces exact Hamiltonian."""
        L = 4
        H = qtn.MPO_ham_heis(L=L, j=1.0, bz=0.0, cyclic=False)
        W = [get_mpo_tensor_data(H, i) for i in range(L)]

        # Build exact H
        W0s = W[0][0, :, :, :]
        W3s = W[3][:, 0, :, :]
        H_exact = np.einsum('aij,abkl,bcmn,cop->ikmojlnp',
                             W0s, W[1], W[2], W3s).reshape(16, 16)
        evals_exact = np.linalg.eigvalsh(H_exact)

        # Build identity MPS for sites 2,3
        A2 = np.zeros((4, 2, 2))
        for s in range(2):
            for b in range(2):
                A2[2 * s + b, s, b] = 1.0
        A3 = np.zeros((2, 2, 1))
        for s in range(2):
            A3[s, s, 0] = 1.0

        R_init = init_right_env(1, 1)
        R3 = update_right_env(R_init, A3, W[3])
        R2 = update_right_env(R3, A2, W[2])
        L0 = init_left_env(1, 1)

        H_eff, _ = build_heff_operator(L0, R2, W[0], W[1])
        dim = H_eff.shape[0]
        H_dense = np.zeros((dim, dim))
        for j in range(dim):
            e = np.zeros(dim)
            e[j] = 1.0
            H_dense[:, j] = H_eff @ e

        evals_eff = np.sort(np.linalg.eigvalsh(H_dense))
        evals_ref = np.sort(evals_exact)
        assert np.allclose(evals_eff, evals_ref, atol=1e-10)

    def test_heff_complex(self):
        """H_eff works with complex environments."""
        np.random.seed(42)
        chi_L, chi_R, d, D = 3, 4, 2, 5
        L_env = np.random.randn(chi_L, D, chi_L) + 0.1j * np.random.randn(chi_L, D, chi_L)
        R_env = np.random.randn(chi_R, D, chi_R) + 0.1j * np.random.randn(chi_R, D, chi_R)
        W_L = np.random.randn(D, D, d, d)
        W_R = np.random.randn(D, D, d, d)
        theta = np.random.randn(chi_L, d, d, chi_R) + 0.1j * np.random.randn(chi_L, d, d, chi_R)

        result = apply_heff(L_env, R_env, W_L, W_R, theta)
        ref = np.einsum('awc,csre,wmps,mnqr,fne->apqf',
                         L_env, theta, W_L, W_R, R_env, optimize=True)
        assert np.allclose(result, ref, atol=1e-10)


class TestEigensolver:
    def test_ground_state_energy(self):
        """Eigensolver finds correct ground state for 4-site Heisenberg."""
        L = 4
        H = qtn.MPO_ham_heis(L=L, j=1.0, bz=0.0, cyclic=False)
        W = [get_mpo_tensor_data(H, i) for i in range(L)]

        # Exact GS
        W0s = W[0][0, :, :, :]
        W3s = W[3][:, 0, :, :]
        H_exact = np.einsum('aij,abkl,bcmn,cop->ikmojlnp',
                             W0s, W[1], W[2], W3s).reshape(16, 16)
        E_exact = np.linalg.eigvalsh(H_exact)[0]

        # Identity MPS
        A2 = np.zeros((4, 2, 2))
        for s in range(2):
            for b in range(2):
                A2[2 * s + b, s, b] = 1.0
        A3 = np.zeros((2, 2, 1))
        for s in range(2):
            A3[s, s, 0] = 1.0

        R_init = init_right_env(1, 1)
        R3 = update_right_env(R_init, A3, W[3])
        R2 = update_right_env(R3, A2, W[2])
        L0 = init_left_env(1, 1)

        theta = np.random.RandomState(42).randn(1, 2, 2, 4)
        E, theta_opt = optimize_two_site(L0, R2, W[0], W[1], theta)
        assert E == pytest.approx(E_exact, abs=1e-8)


class TestEnvironments:
    def test_left_env_update(self):
        """Left environment update matches reference einsum."""
        np.random.seed(42)
        chi, D, d = 3, 5, 2
        chi2, D2 = 4, 5

        L = np.random.randn(chi, D, chi)
        A = np.random.randn(chi, d, chi2)
        W = np.random.randn(D, D2, d, d)

        L_new = update_left_env(L, A, W)
        A_conj = A.conj()
        L_ref = np.einsum('awc,atp,wmts,csq->pmq',
                           L, A_conj, W, A, optimize=True)
        assert np.allclose(L_new, L_ref, atol=1e-12)

    def test_right_env_update(self):
        """Right environment update matches reference einsum."""
        np.random.seed(42)
        chi, D, d = 3, 5, 2
        chi2, D2 = 4, 5

        R = np.random.randn(chi, D, chi)
        B = np.random.randn(chi2, d, chi)
        W = np.random.randn(D2, D, d, d)

        R_new = update_right_env(R, B, W)
        B_conj = B.conj()
        R_ref = np.einsum('awc,pta,mwts,qsc->pmq',
                           R, B_conj, W, B, optimize=True)
        assert np.allclose(R_new, R_ref, atol=1e-12)

    def test_full_energy_contraction(self):
        """Full left-environment sweep gives correct energy."""
        L_sites = 6
        H = qtn.MPO_ham_heis(L=L_sites, j=1.0, bz=0.0, cyclic=False)
        dmrg_ref = qtn.DMRG2(H, bond_dims=[20], cutoffs=1e-12)
        dmrg_ref.solve(tol=1e-12, verbosity=0)
        mps = dmrg_ref.state.copy()
        mps.canonize(0)

        A = [get_tensor_data(mps, i) for i in range(L_sites)]
        W = [get_mpo_tensor_data(H, i) for i in range(L_sites)]

        L_env = init_left_env(A[0].shape[0], W[0].shape[0])
        for i in range(L_sites):
            L_env = update_left_env(L_env, A[i], W[i])

        assert L_env[0, 0, 0] == pytest.approx(dmrg_ref.energy, abs=1e-10)

    def test_product_state_energy(self):
        """Energy of all-up product state is correct for Heisenberg."""
        L_sites = 4
        H = qtn.MPO_ham_heis(L=L_sites, j=1.0, bz=0.0, cyclic=False)
        W = [get_mpo_tensor_data(H, i) for i in range(L_sites)]

        # All spin-up: |0000>
        A = [np.array([[[1.0], [0.0]]]) for _ in range(L_sites)]
        L_env = init_left_env(1, 1)
        for i in range(L_sites):
            L_env = update_left_env(L_env, A[i], W[i])

        # E = (L-1) * 1/4 for ferromagnetic alignment
        E_expected = (L_sites - 1) * 0.25
        assert L_env[0, 0, 0] == pytest.approx(E_expected, abs=1e-12)


class TestSerialDMRG:
    def test_serial_dmrg_converges(self):
        """Serial two-site DMRG converges to reference energy."""
        L_sites = 10
        max_bond = 30
        n_sweeps = 8

        H = qtn.MPO_ham_heis(L=L_sites, j=1.0, bz=0.0, cyclic=False)
        dmrg_ref = qtn.DMRG2(H, bond_dims=[30, 50], cutoffs=1e-12)
        dmrg_ref.solve(tol=1e-12, verbosity=0)
        E_ref = dmrg_ref.energy

        mps = qtn.MPS_rand_state(L=L_sites, bond_dim=4, dtype='float64')
        mps.canonize(0)

        A = [get_tensor_data(mps, i) for i in range(L_sites)]
        W = [get_mpo_tensor_data(H, i) for i in range(L_sites)]

        R_envs = {}
        R_envs[L_sites - 1] = init_right_env(A[-1].shape[2], W[-1].shape[1])
        for i in range(L_sites - 2, 0, -1):
            R_envs[i] = update_right_env(R_envs[i + 1], A[i + 1], W[i + 1])

        L_envs = {0: init_left_env(A[0].shape[0], W[0].shape[0])}

        E = 0.0
        for sweep in range(n_sweeps):
            for i in range(L_sites - 1):
                theta = np.einsum('ijk,klm->ijlm', A[i], A[i + 1])
                E, theta_opt = optimize_two_site(
                    L_envs[i],
                    R_envs.get(i + 1, init_right_env(
                        A[i + 1].shape[2], W[i + 1].shape[1])),
                    W[i], W[i + 1], theta, max_iter=50, tol=1e-12
                )
                chi_L, d_L, d_R, chi_R = theta_opt.shape
                M = theta_opt.reshape(chi_L * d_L, d_R * chi_R)
                U, S, Vh, _ = truncated_svd(M, max_bond)
                A[i] = U.reshape(chi_L, d_L, -1)
                A[i + 1] = (np.diag(S) @ Vh).reshape(-1, d_R, chi_R)
                L_envs[i + 1] = update_left_env(L_envs[i], A[i], W[i])

            R_envs[L_sites - 1] = init_right_env(
                A[-1].shape[2], W[-1].shape[1])
            for i in range(L_sites - 2, -1, -1):
                theta = np.einsum('ijk,klm->ijlm', A[i], A[i + 1])
                E, theta_opt = optimize_two_site(
                    L_envs[i], R_envs[i + 1],
                    W[i], W[i + 1], theta, max_iter=50, tol=1e-12
                )
                chi_L, d_L, d_R, chi_R = theta_opt.shape
                M = theta_opt.reshape(chi_L * d_L, d_R * chi_R)
                U, S, Vh, _ = truncated_svd(M, max_bond)
                A[i + 1] = Vh.reshape(-1, d_R, chi_R)
                A[i] = (U @ np.diag(S)).reshape(chi_L, d_L, -1)
                R_envs[i] = update_right_env(
                    R_envs[i + 1], A[i + 1], W[i + 1])

        assert abs(E - E_ref) < 1e-6
