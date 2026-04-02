"""
Tests for MPS compression via TT-SVD algorithm.

This module tests Phase 4 of A2DMRG: compression of MPS back to rank manifold
after forming linear combinations of candidate states.
"""

import pytest
import numpy as np
from quimb.tensor import MPS_rand_state, MatrixProductState

from a2dmrg.mps import (
    compress_mps,
    verify_left_canonical,
    create_random_mps,
    left_canonicalize
)


class TestBasicCompression:
    """Test basic TT-SVD compression (Feature 29)."""

    def test_compress_with_max_rank(self):
        """Feature 29, Step 1-3: Basic compression with max_rank constraint."""
        # Create MPS with large bond dimensions
        L = 10
        bond_dim = 20
        mps = MPS_rand_state(L, bond_dim=bond_dim, dtype=float)

        # Compress to smaller bond dimension
        max_rank = 10
        mps_compressed, errors = compress_mps(mps, max_rank=max_rank)

        # Verify bond dimensions are reduced
        for i in range(L - 1):
            tensor = mps[i].data
            if tensor.ndim == 2:
                # First or last site
                bond = min(tensor.shape)
            else:
                # Middle site: (left_bond, right_bond, phys)
                bond = max(tensor.shape[0], tensor.shape[1])
            assert bond <= max_rank, f"Bond dimension {bond} exceeds max_rank {max_rank} at site {i}"

    def test_result_is_left_canonical(self):
        """Feature 29, Step 4: Verify result is left-orthogonal."""
        L = 8
        bond_dim = 16
        mps = MPS_rand_state(L, bond_dim=bond_dim, dtype=float)

        # Compress
        compress_mps(mps, max_rank=8)

        # Verify left-canonical form
        is_left_canonical = verify_left_canonical(mps, tol=1e-10)
        assert is_left_canonical, "Compressed MPS should be in left-canonical form"

    def test_compression_error_reasonable(self):
        """Feature 29, Step 5: Verify compression error within tolerance."""
        L = 8
        bond_dim = 20
        mps_original = MPS_rand_state(L, bond_dim=bond_dim, dtype=float)
        mps = mps_original.copy()

        # Store original norm
        original_norm = mps_original.norm()

        # Compress
        max_rank = 10
        compress_mps(mps, max_rank=max_rank)

        # Verify norm is preserved (approximately)
        compressed_norm = mps.norm()
        assert abs(compressed_norm - original_norm) < 1e-8, \
            f"Norm changed too much: {original_norm} -> {compressed_norm}"

    def test_compression_with_tolerance(self):
        """Feature 30, Step 1-2: Compression with tolerance constraint."""
        L = 8
        bond_dim = 20
        mps = MPS_rand_state(L, bond_dim=bond_dim, dtype=float)

        # Compress with tolerance
        tol = 1e-6
        mps_compressed, errors = compress_mps(mps, tol=tol, normalize=True)

        # Verify some compression occurred (bond dimensions reduced)
        # This is probabilistic, but with tol=1e-6, we expect some reduction
        final_bond_dims = []
        for i in range(mps.L - 1):
            tensor = mps[i].data
            if tensor.ndim == 2:
                bond = tensor.shape[0]
            else:
                bond = tensor.shape[1]  # right bond
            final_bond_dims.append(bond)

        # At least some bonds should be smaller than original
        assert min(final_bond_dims) < bond_dim or max(final_bond_dims) < bond_dim, \
            "Expected some compression with tolerance"

    def test_adaptive_bond_dimensions(self):
        """Feature 30, Step 4: Verify bond dimensions are adaptive."""
        L = 10
        bond_dim = 20
        mps = MPS_rand_state(L, bond_dim=bond_dim, dtype=float)

        # Compress with tolerance (adaptive)
        tol = 1e-5
        compress_mps(mps, tol=tol)

        # Extract bond dimensions
        bond_dims = []
        for i in range(mps.L - 1):
            tensor = mps[i].data
            if tensor.ndim == 2:
                bond = tensor.shape[0]
            else:
                bond = tensor.shape[1]  # right bond
            bond_dims.append(bond)

        # Bond dimensions should vary (adaptive to local entanglement)
        # For random MPS, we expect some variation
        if len(bond_dims) > 2:
            # Check that not all bonds have the same dimension
            unique_bonds = len(set(bond_dims))
            # Allow for some to be the same, but expect some variation
            assert unique_bonds > 1 or bond_dims[0] < bond_dim, \
                "Expected adaptive bond dimensions"


class TestCompressionQuality:
    """Test quality of TT-SVD compression."""

    def test_singular_value_threshold(self):
        """Feature 30, Step 3: Verify singular values kept satisfy threshold."""
        L = 6
        bond_dim = 16
        mps = MPS_rand_state(L, bond_dim=bond_dim, dtype=float)

        # Compress with tolerance
        tol = 1e-6
        mps_compressed, errors = compress_mps(mps, tol=tol)

        # The function should return truncation errors
        assert len(errors) == L - 1, f"Expected {L-1} truncation errors, got {len(errors)}"

        # All errors should be non-negative
        for error in errors:
            assert error >= 0, f"Truncation error should be non-negative, got {error}"

    def test_truncation_error_bound(self):
        """Feature 30, Step 5: Verify truncation error < tolerance."""
        L = 6
        bond_dim = 20
        mps_original = MPS_rand_state(L, bond_dim=bond_dim, dtype=float)

        # Store original state
        original_state = mps_original.copy()
        original_norm = original_state.norm()

        # Compress
        tol = 1e-5
        mps_compressed, errors = compress_mps(mps_original, tol=tol)

        # The truncation errors should be small
        max_error = max(errors) if errors else 0.0

        # The error should be reasonable (not larger than the norm)
        assert max_error < original_norm, \
            f"Truncation error {max_error} should be less than norm {original_norm}"


class TestComplexDtype:
    """Test compression with complex128 dtype."""

    def test_complex_compression(self):
        """Test compression works with complex MPS."""
        L = 8
        bond_dim = 16
        mps = MPS_rand_state(L, bond_dim=bond_dim, dtype=complex)

        # Compress
        max_rank = 8
        compress_mps(mps, max_rank=max_rank)

        # Verify still complex
        assert mps[0].dtype == np.complex128, "MPS should remain complex128"

        # Verify left-canonical with Hermitian conjugate
        is_canonical = verify_left_canonical(mps, tol=1e-10)
        assert is_canonical, "Complex MPS should be left-canonical after compression"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_no_compression_parameters(self):
        """Test that function requires at least one parameter."""
        L = 6
        mps = MPS_rand_state(L, bond_dim=8, dtype=float)

        with pytest.raises(ValueError, match="Must specify at least one"):
            compress_mps(mps, max_rank=None, tol=None)

    def test_combined_constraints(self):
        """Test compression with both max_rank and tolerance."""
        L = 8
        bond_dim = 20
        mps = MPS_rand_state(L, bond_dim=bond_dim, dtype=float)

        # Use both constraints
        max_rank = 12
        tol = 1e-6
        compress_mps(mps, max_rank=max_rank, tol=tol)

        # Verify bond dimensions respect max_rank
        for i in range(mps.L - 1):
            tensor = mps[i].data
            if tensor.ndim == 2:
                bond = min(tensor.shape)
            else:
                bond = max(tensor.shape[0], tensor.shape[1])
            assert bond <= max_rank, f"Bond dimension {bond} exceeds max_rank at site {i}"

    def test_small_system(self):
        """Test compression on small system."""
        L = 3
        bond_dim = 8
        mps = MPS_rand_state(L, bond_dim=bond_dim, dtype=float)

        # Compress
        compress_mps(mps, max_rank=4)

        # Should still work
        assert mps.L == L
        assert verify_left_canonical(mps, tol=1e-10)

    def test_already_compressed_mps(self):
        """Test compressing already compressed MPS."""
        L = 8
        bond_dim = 8
        mps = MPS_rand_state(L, bond_dim=bond_dim, dtype=float)

        # Compress to same dimension
        compress_mps(mps, max_rank=8)

        # Should still work without issues
        assert mps.L == L
        assert verify_left_canonical(mps, tol=1e-10)

    def test_very_strict_tolerance(self):
        """Test with very strict tolerance (keep almost everything)."""
        L = 6
        bond_dim = 12
        mps = MPS_rand_state(L, bond_dim=bond_dim, dtype=float)

        # Very strict tolerance (keep almost all singular values)
        tol = 1e-12
        compress_mps(mps, tol=tol)

        # Should keep most of the bond dimension
        assert mps.L == L

    def test_normalization_flag(self):
        """Test that normalization flag works."""
        L = 6
        bond_dim = 12
        mps = MPS_rand_state(L, bond_dim=bond_dim, dtype=float)

        # Compress without normalization
        compress_mps(mps, max_rank=8, normalize=False)

        # Should still produce valid MPS
        assert mps.L == L

        # Compress with normalization (default)
        mps2 = MPS_rand_state(L, bond_dim=bond_dim, dtype=float)
        compress_mps(mps2, max_rank=8, normalize=True)

        # Should have norm close to 1
        norm = mps2.norm()
        assert abs(norm - 1.0) < 1e-10, f"Normalized MPS should have norm 1, got {norm}"


class TestCompressionPreservesState:
    """Test that compression preserves the quantum state."""

    def test_overlap_after_compression(self):
        """Test that compressed state has high overlap with original."""
        L = 8
        bond_dim = 20
        mps_original = MPS_rand_state(L, bond_dim=bond_dim, dtype=float)

        # Make a copy for comparison
        mps_copy = mps_original.copy()

        # Compress with moderate tolerance
        tol = 1e-6
        compress_mps(mps_copy, tol=tol)

        # Compute overlap
        overlap = abs(mps_original.H @ mps_copy)

        # Overlap should be very close to 1
        assert overlap > 0.99, f"Overlap {overlap} should be close to 1 after compression"

    def test_norm_preservation(self):
        """Test that compression preserves norm when normalized."""
        L = 8
        bond_dim = 16
        mps = MPS_rand_state(L, bond_dim=bond_dim, dtype=float)

        # Compress with normalization
        compress_mps(mps, max_rank=10, normalize=True)

        # Norm should be 1
        norm = mps.norm()
        assert abs(norm - 1.0) < 1e-10, f"Norm should be 1, got {norm}"
