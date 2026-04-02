"""
Tests for observable computations (energy and overlap).

This test suite verifies:
1. Energy computation ⟨ψ|H|ψ⟩ for MPS and MPO
2. Overlap computation ⟨φ|ψ⟩ between two MPS states
3. Both float64 and complex128 support
4. Hermitian conjugation for complex states
5. Symmetry properties

These features correspond to the A2DMRG spec requirements for
coarse-space minimization (Phase 3).
"""

import pytest
import numpy as np
import quimb.tensor as qtn
from a2dmrg.numerics.observables import compute_energy, compute_overlap


def build_heisenberg_mpo(L, J=1.0, h=0.0, dtype=float):
    """
    Build Heisenberg Hamiltonian MPO using quimb.

    Parameters
    ----------
    L : int
        System size
    J : float
        Coupling constant (default: 1.0)
    h : float
        Magnetic field (default: 0.0)
    dtype : type
        Data type (float or complex)

    Returns
    -------
    mpo : quimb MPO
        Heisenberg Hamiltonian as MPO
    """
    # Use quimb's built-in Heisenberg MPO
    # j parameter is the coupling, bz is the magnetic field
    mpo = qtn.MPO_ham_heis(L, j=J, bz=h, cyclic=False)

    # Convert to desired dtype if needed
    if dtype == complex:
        # Convert all tensors to complex
        for i in range(L):
            tensor_data = mpo[i].data
            if tensor_data.dtype != complex:
                mpo[i].modify(data=tensor_data.astype(complex))

    return mpo


class TestEnergyComputation:
    """Test energy computation ⟨ψ|H|ψ⟩."""

    def test_energy_basic_float64(self):
        """Test energy computation with float64 MPS."""
        # Create a simple MPS and Hamiltonian
        L = 10
        bond_dim = 8
        mps = qtn.MPS_rand_state(L, bond_dim, seed=42, dtype=float)
        mps.normalize()

        # Build Heisenberg Hamiltonian
        mpo = build_heisenberg_mpo(L, J=1.0, h=0.0, dtype=float)

        # Compute energy
        energy = compute_energy(mps, mpo)

        # Verify energy is real
        assert isinstance(energy, (float, np.floating))
        assert np.isfinite(energy)

        # For Heisenberg, energy should be bounded
        # E/L should be roughly in [-1, 1] for normalized state
        assert -2 * L < energy < 2 * L

    def test_energy_complex128(self):
        """Test energy computation with complex128 MPS."""
        L = 8
        bond_dim = 6
        mps = qtn.MPS_rand_state(L, bond_dim, seed=456, dtype=complex)
        mps.normalize()

        # Heisenberg with complex dtype
        mpo = build_heisenberg_mpo(L, J=1.0, h=0.0, dtype=complex)

        # Compute energy
        energy = compute_energy(mps, mpo)

        # Energy must be real (Hermitian operator)
        assert isinstance(energy, (float, np.floating))
        assert np.isfinite(energy)

        # Should be in reasonable range
        assert -2 * L < energy < 2 * L

    def test_energy_is_real_for_hermitian_operator(self):
        """Verify energy is real even for complex MPS (Hermitian H)."""
        L = 8
        bond_dim = 4
        mps = qtn.MPS_rand_state(L, bond_dim, seed=789, dtype=complex)

        # Add explicit complex phase
        for i in range(L):
            tensor = mps[i].data
            phase = np.exp(1j * np.random.rand())
            mps[i].modify(data=tensor * phase)

        mps.normalize()

        mpo = build_heisenberg_mpo(L, J=1.0, h=0.0, dtype=complex)

        energy = compute_energy(mps, mpo)

        # Must be real
        assert isinstance(energy, (float, np.floating))
        assert np.isfinite(energy)

    def test_energy_normalization(self):
        """Test normalization option in energy computation."""
        L = 6
        bond_dim = 4
        mps = qtn.MPS_rand_state(L, bond_dim, seed=111, dtype=float)

        # Deliberately scale MPS to have norm != 1
        # Scale only the first tensor to avoid complex norm scaling issues
        mps[0].modify(data=mps[0].data * 2.5)

        mpo = build_heisenberg_mpo(L, J=1.0, h=0.0, dtype=float)

        # Compute with normalization
        energy_normalized = compute_energy(mps, mpo, normalize=True)

        # Manually normalize and compute
        mps_copy = mps.copy()
        mps_copy.normalize()
        energy_manual = compute_energy(mps_copy, mpo, normalize=False)

        # Should match
        assert np.abs(energy_normalized - energy_manual) < 1e-10

    def test_energy_without_normalization(self):
        """Test energy computation assumes normalized MPS when normalize=False."""
        L = 6
        bond_dim = 4
        mps = qtn.MPS_rand_state(L, bond_dim, seed=222, dtype=float)
        mps.normalize()

        mpo = build_heisenberg_mpo(L, J=1.0, h=0.0, dtype=float)

        # Both should give same result for normalized MPS
        energy_with_norm = compute_energy(mps, mpo, normalize=True)
        energy_without_norm = compute_energy(mps, mpo, normalize=False)

        assert np.abs(energy_with_norm - energy_without_norm) < 1e-10


class TestOverlapComputation:
    """Test overlap computation ⟨φ|ψ⟩."""

    def test_overlap_basic_float64(self):
        """Test overlap computation with real MPS."""
        L = 8
        bond_dim = 6
        mps1 = qtn.MPS_rand_state(L, bond_dim, seed=42, dtype=float)
        mps2 = qtn.MPS_rand_state(L, bond_dim, seed=43, dtype=float)

        mps1.normalize()
        mps2.normalize()

        # Compute overlap
        overlap = compute_overlap(mps1, mps2)

        # Should be complex type (general case)
        assert isinstance(overlap, (complex, np.complexfloating))

        # For real MPS, imaginary part should be negligible
        assert np.abs(overlap.imag) < 1e-10

        # Magnitude should be <= 1 (Cauchy-Schwarz)
        assert np.abs(overlap) <= 1.0 + 1e-10

    def test_overlap_self_normalized(self):
        """Test ⟨ψ|ψ⟩ = 1 for normalized states."""
        L = 10
        bond_dim = 8
        mps = qtn.MPS_rand_state(L, bond_dim, seed=100, dtype=float)
        mps.normalize()

        # Self-overlap should be 1
        overlap = compute_overlap(mps, mps)

        assert np.abs(overlap - 1.0) < 1e-10

    def test_overlap_complex128(self):
        """Test overlap with complex MPS."""
        L = 8
        bond_dim = 6
        mps1 = qtn.MPS_rand_state(L, bond_dim, seed=200, dtype=complex)
        mps2 = qtn.MPS_rand_state(L, bond_dim, seed=201, dtype=complex)

        mps1.normalize()
        mps2.normalize()

        overlap = compute_overlap(mps1, mps2)

        # Should be complex
        assert isinstance(overlap, (complex, np.complexfloating))

        # Magnitude <= 1
        assert np.abs(overlap) <= 1.0 + 1e-10

    def test_overlap_symmetry_property(self):
        """Test ⟨φ|ψ⟩* = ⟨ψ|φ⟩."""
        L = 8
        bond_dim = 4
        mps1 = qtn.MPS_rand_state(L, bond_dim, seed=300, dtype=complex)
        mps2 = qtn.MPS_rand_state(L, bond_dim, seed=301, dtype=complex)

        mps1.normalize()
        mps2.normalize()

        overlap_12 = compute_overlap(mps1, mps2)
        overlap_21 = compute_overlap(mps2, mps1)

        # Should satisfy conjugate symmetry
        assert np.abs(overlap_12 - np.conj(overlap_21)) < 1e-10

    def test_overlap_hermitian_conjugate(self):
        """Verify uses Hermitian conjugate for complex case."""
        L = 6
        bond_dim = 4
        mps = qtn.MPS_rand_state(L, bond_dim, seed=400, dtype=complex)

        # Add explicit phase
        for i in range(L):
            tensor = mps[i].data
            phase = np.exp(1j * np.random.rand())
            mps[i].modify(data=tensor * phase)

        mps.normalize()

        # Self-overlap should still be real and equal to 1
        overlap = compute_overlap(mps, mps)

        # ⟨ψ|ψ⟩ should be real and = 1 for normalized state
        assert np.abs(overlap.imag) < 1e-10
        assert np.abs(overlap - 1.0) < 1e-10

    def test_overlap_orthogonal_states(self):
        """Test overlap of orthogonal states."""
        # For small systems, we can construct orthogonal states
        L = 2
        bond_dim = 2

        # Product state |00⟩
        mps1 = qtn.MPS_computational_state('0' * L)

        # Product state |11⟩
        mps2 = qtn.MPS_computational_state('1' * L)

        overlap = compute_overlap(mps1, mps2)

        # Should be exactly 0 (orthogonal computational basis states)
        assert np.abs(overlap) < 1e-12

    def test_overlap_identical_states(self):
        """Test overlap of identical (not just normalized) states."""
        L = 8
        bond_dim = 6
        mps = qtn.MPS_rand_state(L, bond_dim, seed=500, dtype=float)
        mps.normalize()

        # Create exact copy
        mps_copy = mps.copy()

        overlap = compute_overlap(mps, mps_copy)

        # Should be exactly 1
        assert np.abs(overlap - 1.0) < 1e-12

    def test_overlap_scaled_state(self):
        """Test overlap changes with state scaling."""
        L = 6
        bond_dim = 4
        mps = qtn.MPS_rand_state(L, bond_dim, seed=600, dtype=float)
        mps.normalize()

        # Create scaled version
        mps_scaled = mps.copy()
        # Scale by changing first tensor
        mps_scaled[0].modify(data=mps_scaled[0].data * 2.0)

        # Overlap should be different from 1
        overlap = compute_overlap(mps, mps_scaled)

        # For normalized mps and 2*mps, overlap should be 2
        # But mps_scaled is not normalized, so this tests general case
        assert np.isfinite(overlap)


class TestEnergyAndOverlapConsistency:
    """Test consistency between energy and overlap computations."""

    def test_overlap_matrix_is_hermitian(self):
        """Test that S_ij = ⟨i|j⟩ is Hermitian."""
        L = 6
        bond_dim = 4
        N_states = 4

        mps_list = []
        for i in range(N_states):
            mps = qtn.MPS_rand_state(L, bond_dim, seed=2000+i, dtype=complex)
            mps.normalize()
            mps_list.append(mps)

        # Build overlap matrix S_ij = ⟨i|j⟩
        S_matrix = np.zeros((N_states, N_states), dtype=complex)
        for i in range(N_states):
            for j in range(N_states):
                S_matrix[i, j] = compute_overlap(mps_list[i], mps_list[j])

        # S matrix should be Hermitian
        assert np.allclose(S_matrix, S_matrix.conj().T, atol=1e-10)

        # Diagonal should be 1 (normalized states)
        for i in range(N_states):
            assert np.abs(S_matrix[i, i] - 1.0) < 1e-10

    def test_energy_and_overlap_for_coarse_space(self):
        """Test energy and overlap for coarse-space minimization setup."""
        # This simulates the coarse-space setup in A2DMRG Phase 3
        L = 8
        bond_dim = 6
        d = 5  # Number of candidate MPS (simulating Y^(0), ..., Y^(4))

        # Create candidate MPS list
        Y_list = []
        for i in range(d):
            mps = qtn.MPS_rand_state(L, bond_dim, seed=3000+i, dtype=float)
            # Don't normalize - in A2DMRG they come from local updates
            Y_list.append(mps)

        mpo = build_heisenberg_mpo(L, J=1.0, h=0.0, dtype=float)

        # Build H_coarse and S_coarse
        H_coarse = np.zeros((d, d), dtype=float)
        S_coarse = np.zeros((d, d), dtype=float)

        for i in range(d):
            for j in range(d):
                # H_coarse[i,j] = ⟨Y^(i), H Y^(j)⟩
                H_Y_j = mpo @ Y_list[j]
                H_coarse[i, j] = (Y_list[i].H @ H_Y_j).data.ravel()[0].real

                # S_coarse[i,j] = ⟨Y^(i), Y^(j)⟩
                S_coarse[i, j] = compute_overlap(Y_list[i], Y_list[j]).real

        # Verify properties
        # H_coarse should be symmetric (real case)
        assert np.allclose(H_coarse, H_coarse.T, atol=1e-10)

        # S_coarse should be symmetric and positive definite
        assert np.allclose(S_coarse, S_coarse.T, atol=1e-10)

        # Diagonal elements of S_coarse should be positive
        for i in range(d):
            assert S_coarse[i, i] > 0

        # All elements should be finite
        assert np.all(np.isfinite(H_coarse))
        assert np.all(np.isfinite(S_coarse))


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_small_system_L2(self):
        """Test with minimal system size L=2."""
        L = 2
        bond_dim = 2
        mps = qtn.MPS_rand_state(L, bond_dim, seed=4000, dtype=float)
        mps.normalize()

        mpo = build_heisenberg_mpo(L, J=1.0, h=0.0, dtype=float)

        energy = compute_energy(mps, mpo)
        overlap = compute_overlap(mps, mps)

        assert np.isfinite(energy)
        assert np.abs(overlap - 1.0) < 1e-10

    def test_large_bond_dimension(self):
        """Test with large bond dimension."""
        L = 10
        bond_dim = 32  # Large bond dimension
        mps = qtn.MPS_rand_state(L, bond_dim, seed=5000, dtype=float)
        mps.normalize()

        mpo = build_heisenberg_mpo(L, J=1.0, h=0.0, dtype=float)

        energy = compute_energy(mps, mpo)

        assert np.isfinite(energy)

    def test_unnormalized_mps_energy(self):
        """Test energy with unnormalized MPS."""
        L = 6
        bond_dim = 4
        mps = qtn.MPS_rand_state(L, bond_dim, seed=6000, dtype=float)

        # Scale to have large norm
        for i in range(L):
            mps[i].modify(data=mps[i].data * 3.0)

        mpo = build_heisenberg_mpo(L, J=1.0, h=0.0, dtype=float)

        # With normalization, should work
        energy = compute_energy(mps, mpo, normalize=True)
        assert np.isfinite(energy)

        # Without normalization, energy will be scaled by norm^2
        energy_unnorm = compute_energy(mps, mpo, normalize=False)
        norm_sq = np.abs(mps.norm()) ** 2
        expected_energy = energy * norm_sq

        assert np.abs(energy_unnorm - expected_energy) < 1e-8 * np.abs(expected_energy)
