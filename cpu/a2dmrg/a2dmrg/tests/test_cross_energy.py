"""Tests for compute_cross_energy."""
import numpy as np
import pytest
import quimb.tensor as qtn
from a2dmrg.numerics.observables import compute_cross_energy, compute_energy, compute_overlap


def make_heisenberg_mpo(L):
    """Build Heisenberg XXX MPO using quimb's built-in."""
    return qtn.MPO_ham_heis(L)


def make_random_mps(L, D, phys_dim=2, seed=42):
    """Make a random normalized MPS."""
    return qtn.MPS_rand_state(L, D, phys_dim=phys_dim, seed=seed)


class TestCrossEnergyDiagonal:
    """Test that compute_cross_energy(psi, mpo, psi) == compute_energy unnormalized."""

    def test_cross_energy_diagonal_matches_compute_energy(self):
        """When bra=ket, compute_cross_energy should equal compute_energy (unnormalized)."""
        L = 6
        D = 4
        mpo = make_heisenberg_mpo(L)
        psi = make_random_mps(L, D, seed=42)

        # compute_energy with normalize=False
        energy_direct = compute_energy(psi, mpo, normalize=False)

        # compute_cross_energy (returns unnormalized ⟨ψ|H|ψ⟩)
        energy_cross = compute_cross_energy(psi, mpo, psi)

        np.testing.assert_allclose(
            np.real(energy_cross), energy_direct, atol=1e-10,
            err_msg="Diagonal cross-energy should match compute_energy(normalize=False)"
        )

    def test_cross_energy_diagonal_is_real(self):
        """Diagonal cross-energy should be real for Hermitian H."""
        L = 6
        D = 4
        mpo = make_heisenberg_mpo(L)
        psi = make_random_mps(L, D, seed=7)

        energy_cross = compute_cross_energy(psi, mpo, psi)

        # Imaginary part should be negligible
        assert abs(energy_cross.imag) < 1e-10, (
            f"Diagonal energy should be real, got imag={energy_cross.imag:.2e}"
        )

    def test_cross_energy_diagonal_different_seeds(self):
        """Diagonal test for multiple states."""
        L = 6
        D = 3
        mpo = make_heisenberg_mpo(L)

        for seed in [1, 2, 3, 4, 5]:
            psi = make_random_mps(L, D, seed=seed)
            energy_direct = compute_energy(psi, mpo, normalize=False)
            energy_cross = compute_cross_energy(psi, mpo, psi)

            np.testing.assert_allclose(
                np.real(energy_cross), energy_direct, atol=1e-10,
                err_msg=f"Failed for seed={seed}"
            )


class TestCrossEnergyHermitian:
    """Test Hermitian symmetry: ⟨bra|H|ket⟩* = ⟨ket|H|bra⟩."""

    def test_cross_energy_hermitian_real_states(self):
        """For real states with Hermitian H: ⟨phi|H|psi⟩ = ⟨psi|H|phi⟩."""
        L = 6
        D = 4
        mpo = make_heisenberg_mpo(L)
        phi = make_random_mps(L, D, seed=1)
        psi = make_random_mps(L, D, seed=2)

        hij = compute_cross_energy(phi, mpo, psi)
        hji = compute_cross_energy(psi, mpo, phi)

        np.testing.assert_allclose(
            hij.real, hji.real, atol=1e-10,
            err_msg="Real part should satisfy Hermitian symmetry"
        )
        np.testing.assert_allclose(
            hij.imag, -hji.imag, atol=1e-10,
            err_msg="Imaginary part should be negated by conjugation"
        )

    def test_cross_energy_hermitian_conjugate(self):
        """compute_cross_energy(phi, mpo, psi).conj() == compute_cross_energy(psi, mpo, phi)."""
        L = 6
        D = 4
        mpo = make_heisenberg_mpo(L)
        phi = make_random_mps(L, D, seed=10)
        psi = make_random_mps(L, D, seed=20)

        hij = compute_cross_energy(phi, mpo, psi)
        hji = compute_cross_energy(psi, mpo, phi)

        np.testing.assert_allclose(
            np.conj(hij), hji, atol=1e-10,
            err_msg="⟨phi|H|psi⟩* should equal ⟨psi|H|phi⟩"
        )

    def test_cross_energy_hermitian_multiple_pairs(self):
        """Hermitian symmetry should hold for multiple state pairs."""
        L = 6
        D = 3
        mpo = make_heisenberg_mpo(L)

        for seed_a, seed_b in [(1, 2), (3, 4), (5, 6), (7, 8)]:
            phi = make_random_mps(L, D, seed=seed_a)
            psi = make_random_mps(L, D, seed=seed_b)

            hij = compute_cross_energy(phi, mpo, psi)
            hji = compute_cross_energy(psi, mpo, phi)

            np.testing.assert_allclose(
                np.conj(hij), hji, atol=1e-10,
                err_msg=f"Failed for seeds ({seed_a}, {seed_b})"
            )


class TestCrossEnergyVsQuimbTN:
    """Test correctness by comparing against quimb's own TN contraction."""

    def _quimb_cross_energy(self, bra, mpo, ket):
        """Compute ⟨bra|H|ket⟩ using quimb's TN contraction (reference)."""
        L = bra.L
        bra_reindexed = bra.H
        bra_reindexed.reindex_({f'k{idx}': f'b{idx}' for idx in range(L)})
        tn = bra_reindexed & mpo & ket
        result = tn.contract(all, optimize='auto')
        if hasattr(result, 'data'):
            data = result.data
        else:
            data = result
        data_array = np.asarray(data)
        if data_array.ndim == 0:
            return complex(data_array.item())
        else:
            return complex(data_array.ravel()[0])

    def test_cross_energy_vs_quimb_tn_off_diagonal(self):
        """compare compute_cross_energy(phi, mpo, psi) vs quimb TN contraction."""
        L = 6
        D = 4
        mpo = make_heisenberg_mpo(L)
        phi = make_random_mps(L, D, seed=100)
        psi = make_random_mps(L, D, seed=200)

        our_result = compute_cross_energy(phi, mpo, psi)
        ref_result = self._quimb_cross_energy(phi, mpo, psi)

        np.testing.assert_allclose(
            our_result, ref_result, atol=1e-10,
            err_msg="compute_cross_energy should match quimb TN contraction"
        )

    def test_cross_energy_vs_quimb_tn_diagonal(self):
        """For bra=ket, should also match quimb TN contraction."""
        L = 6
        D = 4
        mpo = make_heisenberg_mpo(L)
        psi = make_random_mps(L, D, seed=42)

        our_result = compute_cross_energy(psi, mpo, psi)
        ref_result = self._quimb_cross_energy(psi, mpo, psi)

        np.testing.assert_allclose(
            our_result, ref_result, atol=1e-10,
            err_msg="Diagonal compute_cross_energy should match quimb TN"
        )

    def test_cross_energy_vs_quimb_multiple_pairs(self):
        """Multiple off-diagonal pairs vs quimb reference."""
        L = 6
        D = 3
        mpo = make_heisenberg_mpo(L)

        for seed_a, seed_b in [(11, 22), (33, 44), (55, 66)]:
            phi = make_random_mps(L, D, seed=seed_a)
            psi = make_random_mps(L, D, seed=seed_b)

            our_result = compute_cross_energy(phi, mpo, psi)
            ref_result = self._quimb_cross_energy(phi, mpo, psi)

            np.testing.assert_allclose(
                our_result, ref_result, atol=1e-10,
                err_msg=f"Failed for seeds ({seed_a}, {seed_b})"
            )

    def test_cross_energy_returns_complex(self):
        """compute_cross_energy should return complex."""
        L = 6
        D = 4
        mpo = make_heisenberg_mpo(L)
        phi = make_random_mps(L, D, seed=1)
        psi = make_random_mps(L, D, seed=2)

        result = compute_cross_energy(phi, mpo, psi)
        assert isinstance(result, complex), (
            f"Expected complex, got {type(result)}"
        )

    def test_cross_energy_different_bond_dims(self):
        """compute_cross_energy should work when bra and ket have different bond dims."""
        L = 6
        mpo = make_heisenberg_mpo(L)
        # bra has D=3, ket has D=5
        phi = make_random_mps(L, D=3, seed=7)
        psi = make_random_mps(L, D=5, seed=8)

        our_result = compute_cross_energy(phi, mpo, psi)
        ref_result = self._quimb_cross_energy(phi, mpo, psi)

        np.testing.assert_allclose(
            our_result, ref_result, atol=1e-10,
            err_msg="Should work for different bond dimensions"
        )
