"""
Tests for coarse-space matrix construction (Features 23-25).

Tests verify:
- H_coarse construction and properties (Hermitian, diagonal=energies)
- S_coarse construction and properties (Hermitian, PSD, diagonal=norms)
- Parallel mode with MockComm
- Complex dtype support
- Edge cases
"""

import pytest
import numpy as np
from quimb.tensor import MPO_ham_heis

from a2dmrg.parallel.coarse_space import (
    build_coarse_matrices,
    verify_hermitian,
    verify_positive_semidefinite,
)
from a2dmrg.mps.mps_utils import create_random_mps
from a2dmrg.numerics.numpy_utils import extract_mps_arrays, extract_mpo_arrays
from a2dmrg.numerics.observables import compute_energy_numpy, compute_overlap_numpy


class MockComm:
    """Mock MPI communicator for serial testing."""

    def __init__(self, rank=0, size=1):
        self.rank = rank
        self.size = size

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return self.size

    def Allreduce(self, sendbuf, recvbuf, op=None):
        """Mock Allreduce: copy sendbuf to recvbuf (no actual reduction in serial)."""
        recvbuf[:] = sendbuf

    def allreduce(self, data, op=None):
        """In serial mode, allreduce is identity (lowercase version)."""
        return data

    def barrier(self):
        """No-op in serial mode."""
        pass


def _make_states(L, bond_dim, n, dtype="float64"):
    """Create n random MPS as numpy array lists + MPO arrays."""
    quimb_states = [create_random_mps(L, bond_dim, phys_dim=2, dtype=dtype, canonical="left")
                    for _ in range(n)]
    states = [extract_mps_arrays(s) for s in quimb_states]
    return states


def _make_mpo(L):
    """Create Heisenberg MPO as numpy arrays."""
    H_quimb = MPO_ham_heis(L, 1.0, 0.0)
    return extract_mpo_arrays(H_quimb)


class TestHCoarseConstruction:
    """Test H_coarse matrix construction and properties."""

    def test_h_coarse_hermitian(self):
        """H_coarse should be Hermitian."""
        L = 6
        states = _make_states(L, 4, 5)
        mpo_arrays = _make_mpo(L)

        H_coarse, S_coarse = build_coarse_matrices(states, mpo_arrays)

        assert verify_hermitian(H_coarse)
        np.testing.assert_allclose(H_coarse, H_coarse.conj().T, atol=1e-12)

    def test_h_coarse_diagonal_energies(self):
        """H_coarse diagonal should equal state energies."""
        L = 6
        states = _make_states(L, 4, 4)
        mpo_arrays = _make_mpo(L)

        H_coarse, S_coarse = build_coarse_matrices(states, mpo_arrays)

        # Check diagonal elements
        for i, state in enumerate(states):
            energy = compute_energy_numpy(state, mpo_arrays)
            np.testing.assert_allclose(H_coarse[i, i], energy, rtol=1e-10)

    def test_h_coarse_single_state(self):
        """H_coarse with single state should be 1x1 matrix."""
        L = 4
        states = _make_states(L, 3, 1)
        mpo_arrays = _make_mpo(L)

        H_coarse, S_coarse = build_coarse_matrices(states, mpo_arrays)

        assert H_coarse.shape == (1, 1)
        energy = compute_energy_numpy(states[0], mpo_arrays)
        np.testing.assert_allclose(H_coarse[0, 0], energy, rtol=1e-10)

    def test_h_coarse_off_diagonal(self):
        """H_coarse off-diagonal elements should be symmetric."""
        L = 6
        states = _make_states(L, 4, 3)
        mpo_arrays = _make_mpo(L)

        H_coarse, S_coarse = build_coarse_matrices(states, mpo_arrays)

        # Check symmetry of off-diagonal
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                np.testing.assert_allclose(H_coarse[i, j], H_coarse[j, i], atol=1e-12)


class TestSCoarseConstruction:
    """Test S_coarse matrix construction and properties."""

    def test_s_coarse_hermitian(self):
        """S_coarse should be Hermitian."""
        L = 6
        states = _make_states(L, 4, 5)
        mpo_arrays = _make_mpo(L)

        H_coarse, S_coarse = build_coarse_matrices(states, mpo_arrays)

        assert verify_hermitian(S_coarse)
        np.testing.assert_allclose(S_coarse, S_coarse.conj().T, atol=1e-12)

    def test_s_coarse_positive_semidefinite(self):
        """S_coarse should be positive semi-definite."""
        L = 6
        states = _make_states(L, 4, 5)
        mpo_arrays = _make_mpo(L)

        H_coarse, S_coarse = build_coarse_matrices(states, mpo_arrays)

        assert verify_positive_semidefinite(S_coarse)

        # Check eigenvalues are non-negative
        eigvals = np.linalg.eigvalsh(S_coarse)
        assert np.all(eigvals >= -1e-12)

    def test_s_coarse_diagonal_norms(self):
        """S_coarse diagonal should equal squared norms (should be 1 for normalized states)."""
        L = 6
        states = _make_states(L, 4, 4)
        mpo_arrays = _make_mpo(L)

        H_coarse, S_coarse = build_coarse_matrices(states, mpo_arrays)

        # For canonical MPS, diagonal should be close to 1
        for i in range(len(states)):
            np.testing.assert_allclose(S_coarse[i, i], 1.0, rtol=1e-10)

    def test_s_coarse_off_diagonal_overlaps(self):
        """S_coarse off-diagonal should equal state overlaps."""
        L = 6
        states = _make_states(L, 4, 3)
        mpo_arrays = _make_mpo(L)

        H_coarse, S_coarse = build_coarse_matrices(states, mpo_arrays)

        # Check off-diagonal overlaps
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                overlap = compute_overlap_numpy(states[i], states[j])
                np.testing.assert_allclose(S_coarse[i, j], overlap, rtol=1e-10)
                np.testing.assert_allclose(S_coarse[j, i], np.conj(overlap), rtol=1e-10)


class TestComplexDtype:
    """Test complex dtype support."""

    def test_h_coarse_complex_hermitian(self):
        """H_coarse should be Hermitian with complex states."""
        L = 6
        states = _make_states(L, 4, 5, dtype="complex128")
        mpo_arrays = _make_mpo(L)

        H_coarse, S_coarse = build_coarse_matrices(states, mpo_arrays)

        assert verify_hermitian(H_coarse)
        assert H_coarse.dtype == np.complex128

    def test_s_coarse_complex_psd(self):
        """S_coarse should be PSD with complex states."""
        L = 6
        states = _make_states(L, 4, 5, dtype="complex128")
        mpo_arrays = _make_mpo(L)

        H_coarse, S_coarse = build_coarse_matrices(states, mpo_arrays)

        assert verify_hermitian(S_coarse)
        assert verify_positive_semidefinite(S_coarse)
        assert S_coarse.dtype == np.complex128

    def test_complex_diagonal_values(self):
        """Diagonal values should be real even for complex matrices."""
        L = 6
        states = _make_states(L, 4, 4, dtype="complex128")
        mpo_arrays = _make_mpo(L)

        H_coarse, S_coarse = build_coarse_matrices(states, mpo_arrays)

        # Diagonal should be real
        np.testing.assert_allclose(H_coarse.diagonal().imag, 0, atol=1e-12)
        np.testing.assert_allclose(S_coarse.diagonal().imag, 0, atol=1e-12)


class TestParallelMode:
    """Test parallel mode with MockComm."""

    def test_serial_mode_single_rank(self):
        """Serial mode with single rank should work."""
        L = 6
        states = _make_states(L, 4, 4)
        mpo_arrays = _make_mpo(L)
        comm = MockComm(rank=0, size=1)

        H_coarse, S_coarse = build_coarse_matrices(states, mpo_arrays, comm=comm)

        assert verify_hermitian(H_coarse)
        assert verify_positive_semidefinite(S_coarse)

    def test_parallel_consistency(self):
        """Results should be consistent with/without comm."""
        L = 6
        states = _make_states(L, 4, 4)
        mpo_arrays = _make_mpo(L)

        # Without comm
        H_serial, S_serial = build_coarse_matrices(states, mpo_arrays)

        # With mock comm
        comm = MockComm(rank=0, size=1)
        H_parallel, S_parallel = build_coarse_matrices(states, mpo_arrays, comm=comm)

        np.testing.assert_allclose(H_serial, H_parallel, atol=1e-12)
        np.testing.assert_allclose(S_serial, S_parallel, atol=1e-12)


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_state_list(self):
        """Empty state list should raise error or return empty matrices."""
        L = 4
        mpo_arrays = _make_mpo(L)

        with pytest.raises((ValueError, IndexError)):
            build_coarse_matrices([], mpo_arrays)

    def test_large_coarse_space(self):
        """Should handle larger coarse spaces efficiently."""
        L = 6
        states = _make_states(L, 3, 20)
        mpo_arrays = _make_mpo(L)

        H_coarse, S_coarse = build_coarse_matrices(states, mpo_arrays)

        assert H_coarse.shape == (20, 20)
        assert S_coarse.shape == (20, 20)
        assert verify_hermitian(H_coarse)
        assert verify_positive_semidefinite(S_coarse)

    def test_identical_states(self):
        """Identical states should produce rank-deficient S_coarse."""
        L = 4
        states = _make_states(L, 3, 1)
        mpo_arrays = _make_mpo(L)

        # Three copies of same state
        states_dup = [states[0]] * 3

        H_coarse, S_coarse = build_coarse_matrices(states_dup, mpo_arrays)

        # S_coarse should be rank-1 (all states identical)
        eigvals = np.linalg.eigvalsh(S_coarse)
        rank = np.sum(eigvals > 1e-10)
        assert rank == 1
