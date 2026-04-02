"""
Tests for truncated SVD functionality.

Covers features 6-8 from feature_list.json:
- Truncation by tolerance
- Truncation by maximum rank
- Combined tolerance and rank constraints
"""

import numpy as np
import pytest
from a2dmrg.numerics import truncated_svd, reconstruct_from_svd, truncation_error_bound


class TestTruncatedSVDTolerance:
    """Test truncation by tolerance (feature 6)."""

    def test_tolerance_truncation_simple(self):
        """Test basic tolerance-based truncation with known singular values."""
        # Create test matrix with known singular values
        # Use diagonal matrix for simplicity
        singular_values = np.array([10.0, 5.0, 1.0, 0.1, 0.01, 0.001])
        n = len(singular_values)

        # Create random orthogonal matrices
        np.random.seed(42)
        U_full = np.linalg.qr(np.random.randn(n, n))[0]
        Vh_full = np.linalg.qr(np.random.randn(n, n))[0]

        # Construct matrix with known singular values
        M = U_full @ np.diag(singular_values) @ Vh_full

        # Apply truncated SVD with tol=1e-6
        # sigma_max = 10.0, threshold = 10.0 * 1e-6 = 1e-5
        # Should keep: 10.0, 5.0, 1.0, 0.1, 0.01 (all >= 1e-5)
        # Should discard: 0.001 (< 1e-5)
        tol = 1e-6
        U, S, Vh = truncated_svd(M, tol=tol)

        # Verify correct number of singular values
        sigma_max = singular_values[0]
        threshold = sigma_max * tol
        expected_kept = np.sum(singular_values >= threshold)
        assert len(S) == expected_kept, f"Expected {expected_kept} singular values, got {len(S)}"

        # Verify singular values are correct (within numerical precision)
        np.testing.assert_allclose(S, singular_values[:len(S)], rtol=1e-10)

        # Verify reconstruction
        M_reconstructed = reconstruct_from_svd(U, S, Vh)
        reconstruction_error = np.linalg.norm(M - M_reconstructed, 'fro')

        # Error should be approximately equal to discarded singular values
        expected_error = truncation_error_bound(singular_values, len(S))
        np.testing.assert_allclose(reconstruction_error, expected_error, rtol=1e-10, atol=1e-14)

    def test_tolerance_truncation_strict(self):
        """Test that values < sigma_max * tol are removed."""
        # Create matrix with controlled singular values
        singular_values = np.array([100.0, 50.0, 1.0, 0.5, 0.01])
        n = len(singular_values)

        np.random.seed(123)
        U_full = np.linalg.qr(np.random.randn(n, n))[0]
        Vh_full = np.linalg.qr(np.random.randn(n, n))[0]
        M = U_full @ np.diag(singular_values) @ Vh_full

        # With tol=0.02, threshold = 100.0 * 0.02 = 2.0
        # Should keep: 100.0, 50.0 (>= 2.0)
        # Should discard: 1.0, 0.5, 0.01 (< 2.0)
        tol = 0.02
        U, S, Vh = truncated_svd(M, tol=tol)

        assert len(S) == 2, f"Expected 2 singular values, got {len(S)}"
        np.testing.assert_allclose(S, [100.0, 50.0], rtol=1e-10)

    def test_tolerance_zero_keeps_all(self):
        """Test that tol=0 keeps all non-zero singular values."""
        M = np.random.randn(5, 5)
        U, S, Vh = truncated_svd(M, tol=0.0)

        # Should keep all singular values
        S_full = np.linalg.svd(M, compute_uv=False)
        assert len(S) == len(S_full)
        np.testing.assert_allclose(S, S_full, rtol=1e-10)

    def test_svd_decomposition_property(self):
        """Test that U, S, Vh satisfy M ≈ U @ diag(S) @ Vh."""
        np.random.seed(456)
        M = np.random.randn(10, 8)

        tol = 1e-6
        U, S, Vh = truncated_svd(M, tol=tol)

        # Reconstruct matrix
        M_reconstructed = U @ np.diag(S) @ Vh

        # Verify shapes
        assert U.shape[0] == M.shape[0], "U has wrong number of rows"
        assert Vh.shape[1] == M.shape[1], "Vh has wrong number of columns"
        assert U.shape[1] == len(S), "U columns don't match S length"
        assert Vh.shape[0] == len(S), "Vh rows don't match S length"

        # Verify reconstruction is close to original
        reconstruction_error = np.linalg.norm(M - M_reconstructed, 'fro')
        original_norm = np.linalg.norm(M, 'fro')
        assert reconstruction_error < tol * original_norm * 10  # Allow some numerical error


class TestTruncatedSVDRank:
    """Test truncation by maximum rank (feature 7)."""

    def test_max_rank_truncation(self):
        """Test truncation by maximum rank."""
        # Create matrix with 50 singular values
        n = 50
        singular_values = np.logspace(2, -2, n)  # 100 down to 0.01

        np.random.seed(789)
        U_full = np.linalg.qr(np.random.randn(n, n))[0]
        Vh_full = np.linalg.qr(np.random.randn(n, n))[0]
        M = U_full @ np.diag(singular_values) @ Vh_full

        # Apply truncated SVD with max_rank=10
        max_rank = 10
        U, S, Vh = truncated_svd(M, max_rank=max_rank)

        # Verify exactly 10 singular values kept
        assert len(S) == max_rank, f"Expected {max_rank} singular values, got {len(S)}"

        # Verify largest 10 singular values are kept
        np.testing.assert_allclose(S, singular_values[:max_rank], rtol=1e-10)

        # Verify dimensions
        assert U.shape == (n, max_rank), f"U has wrong shape: {U.shape}"
        assert S.shape == (max_rank,), f"S has wrong shape: {S.shape}"
        assert Vh.shape == (max_rank, n), f"Vh has wrong shape: {Vh.shape}"

    def test_max_rank_larger_than_matrix(self):
        """Test that max_rank larger than matrix size keeps all singular values."""
        M = np.random.randn(5, 8)
        max_rank = 100  # Larger than min(5, 8) = 5

        U, S, Vh = truncated_svd(M, max_rank=max_rank)

        # Should keep all 5 singular values
        assert len(S) == 5

    def test_max_rank_one(self):
        """Test rank-1 approximation."""
        np.random.seed(101)
        M = np.random.randn(10, 10)

        U, S, Vh = truncated_svd(M, max_rank=1)

        assert len(S) == 1
        assert U.shape == (10, 1)
        assert Vh.shape == (1, 10)

        # Verify we get the largest singular value
        S_full = np.linalg.svd(M, compute_uv=False)
        np.testing.assert_allclose(S[0], S_full[0], rtol=1e-10)


class TestTruncatedSVDCombined:
    """Test combined tolerance and rank constraints (feature 8)."""

    def test_combined_rank_more_restrictive(self):
        """Test when rank constraint is more restrictive than tolerance."""
        # Create matrix where tolerance would keep 20 values but rank limits to 10
        singular_values = np.logspace(2, -1, 50)  # 100 down to 0.1
        n = len(singular_values)

        np.random.seed(202)
        U_full = np.linalg.qr(np.random.randn(n, n))[0]
        Vh_full = np.linalg.qr(np.random.randn(n, n))[0]
        M = U_full @ np.diag(singular_values) @ Vh_full

        # tol = 1e-3 would keep all values >= 100 * 1e-3 = 0.1 (many values)
        # max_rank = 10 limits to 10 values
        # Should use minimum: 10 values
        tol = 1e-3
        max_rank = 10
        U, S, Vh = truncated_svd(M, max_rank=max_rank, tol=tol)

        assert len(S) == 10, f"Expected 10 (rank limit), got {len(S)}"
        np.testing.assert_allclose(S, singular_values[:10], rtol=1e-10)

    def test_combined_tolerance_more_restrictive(self):
        """Test when tolerance is more restrictive than rank."""
        # Create matrix where rank would keep 20 values but tolerance limits to 5
        singular_values = np.array([100.0, 80.0, 60.0, 40.0, 20.0,
                                     0.5, 0.4, 0.3, 0.2, 0.1])
        n = len(singular_values)

        np.random.seed(303)
        U_full = np.linalg.qr(np.random.randn(n, n))[0]
        Vh_full = np.linalg.qr(np.random.randn(n, n))[0]
        M = U_full @ np.diag(singular_values) @ Vh_full

        # tol = 0.01 means threshold = 100 * 0.01 = 1.0
        # Keeps values >= 1.0: [100, 80, 60, 40, 20] (5 values)
        # max_rank = 20 would allow 10 values
        # Should use minimum: 5 values
        tol = 0.01
        max_rank = 20
        U, S, Vh = truncated_svd(M, max_rank=max_rank, tol=tol)

        assert len(S) == 5, f"Expected 5 (tolerance limit), got {len(S)}"
        np.testing.assert_allclose(S, singular_values[:5], rtol=1e-10)

    def test_combined_equal_constraints(self):
        """Test when both constraints give same result."""
        singular_values = np.array([10.0, 5.0, 1.0, 0.1, 0.01])
        n = len(singular_values)

        np.random.seed(404)
        U_full = np.linalg.qr(np.random.randn(n, n))[0]
        Vh_full = np.linalg.qr(np.random.randn(n, n))[0]
        M = U_full @ np.diag(singular_values) @ Vh_full

        # tol = 0.05 means threshold = 10 * 0.05 = 0.5
        # Keeps: 10, 5, 1 (3 values)
        # max_rank = 3 also keeps 3 values
        tol = 0.05
        max_rank = 3
        U, S, Vh = truncated_svd(M, max_rank=max_rank, tol=tol)

        assert len(S) == 3
        np.testing.assert_allclose(S, singular_values[:3], rtol=1e-10)


class TestTruncatedSVDComplex:
    """Test truncated SVD with complex matrices."""

    def test_complex_matrix_truncation(self):
        """Test that truncated SVD works with complex128."""
        np.random.seed(505)
        M_real = np.random.randn(10, 8)
        M_imag = np.random.randn(10, 8)
        M = M_real + 1j * M_imag

        U, S, Vh = truncated_svd(M, max_rank=5)

        # Verify dtypes
        assert U.dtype == np.complex128
        assert S.dtype == np.float64  # Singular values are always real
        assert Vh.dtype == np.complex128

        # Verify reconstruction is a valid low-rank approximation
        M_reconstructed = U @ np.diag(S) @ Vh

        # Since we truncated to rank 5, reconstruction won't be exact
        # But it should be close enough (keeping 5 out of 8 singular values)
        reconstruction_error = np.linalg.norm(M - M_reconstructed, 'fro')
        original_norm = np.linalg.norm(M, 'fro')

        # Verify error is reasonable (should be small relative to original)
        assert reconstruction_error < original_norm, "Reconstruction error too large"

        # Verify correct shape
        assert M_reconstructed.shape == M.shape

    def test_complex_tolerance_truncation(self):
        """Test tolerance-based truncation for complex matrices."""
        # Create complex matrix with known singular values
        singular_values = np.array([50.0, 10.0, 1.0, 0.1])
        n = len(singular_values)

        np.random.seed(606)
        # Create complex unitary matrices
        U_temp = np.random.randn(n, n) + 1j * np.random.randn(n, n)
        U_full, _ = np.linalg.qr(U_temp)
        V_temp = np.random.randn(n, n) + 1j * np.random.randn(n, n)
        Vh_full, _ = np.linalg.qr(V_temp)

        M = U_full @ np.diag(singular_values) @ Vh_full

        tol = 0.05  # threshold = 50 * 0.05 = 2.5, keeps 50 and 10
        U, S, Vh = truncated_svd(M, tol=tol)

        assert len(S) == 2
        np.testing.assert_allclose(S, [50.0, 10.0], rtol=1e-10)


class TestTruncatedSVDEdgeCases:
    """Test edge cases and error handling."""

    def test_rectangular_matrices(self):
        """Test with various rectangular matrices."""
        # Tall matrix
        M_tall = np.random.randn(20, 5)
        U, S, Vh = truncated_svd(M_tall, max_rank=3)
        assert U.shape == (20, 3)
        assert len(S) == 3
        assert Vh.shape == (3, 5)

        # Wide matrix
        M_wide = np.random.randn(5, 20)
        U, S, Vh = truncated_svd(M_wide, max_rank=3)
        assert U.shape == (5, 3)
        assert len(S) == 3
        assert Vh.shape == (3, 20)

    def test_none_parameters(self):
        """Test with no truncation (all parameters None)."""
        M = np.random.randn(8, 6)
        U, S, Vh = truncated_svd(M, max_rank=None, tol=None)

        # Should keep all 6 singular values
        assert len(S) == 6

    def test_return_truncation_error(self):
        """Test optional truncation error return."""
        singular_values = np.array([10.0, 5.0, 1.0, 0.5, 0.1])
        n = len(singular_values)

        np.random.seed(707)
        U_full = np.linalg.qr(np.random.randn(n, n))[0]
        Vh_full = np.linalg.qr(np.random.randn(n, n))[0]
        M = U_full @ np.diag(singular_values) @ Vh_full

        U, S, Vh, error = truncated_svd(M, max_rank=3, return_truncation_error=True)

        # Error should be norm of discarded singular values [0.5, 0.1]
        expected_error = np.linalg.norm([0.5, 0.1])
        np.testing.assert_allclose(error, expected_error, rtol=1e-10)

    def test_error_on_negative_tolerance(self):
        """Test that negative tolerance raises error."""
        M = np.random.randn(5, 5)
        with pytest.raises(ValueError, match="Tolerance must be non-negative"):
            truncated_svd(M, tol=-0.1)

    def test_error_on_negative_rank(self):
        """Test that negative max_rank raises error."""
        M = np.random.randn(5, 5)
        with pytest.raises(ValueError, match="max_rank must be non-negative"):
            truncated_svd(M, max_rank=-1)

    def test_empty_matrix_error(self):
        """Test that empty matrix raises error."""
        M = np.array([]).reshape(0, 0)
        with pytest.raises(ValueError, match="Cannot perform SVD on empty matrix"):
            truncated_svd(M)


class TestHelperFunctions:
    """Test helper functions."""

    def test_reconstruct_from_svd(self):
        """Test matrix reconstruction from SVD components."""
        M = np.random.randn(8, 6)
        U, S, Vh = truncated_svd(M, max_rank=4)

        M_reconstructed = reconstruct_from_svd(U, S, Vh)

        assert M_reconstructed.shape == (8, 6)
        # Check it's a valid approximation
        error = np.linalg.norm(M - M_reconstructed, 'fro')
        assert error < np.linalg.norm(M, 'fro')

    def test_truncation_error_bound(self):
        """Test theoretical error bound calculation."""
        S_full = np.array([10.0, 5.0, 1.0, 0.5, 0.1])

        # Keep first 3 values, discard last 2
        error = truncation_error_bound(S_full, keep_rank=3)
        expected = np.linalg.norm([0.5, 0.1])
        np.testing.assert_allclose(error, expected, rtol=1e-10)

        # Keep all values
        error_all = truncation_error_bound(S_full, keep_rank=5)
        assert error_all == 0.0

        # Keep none
        error_none = truncation_error_bound(S_full, keep_rank=0)
        expected_none = np.linalg.norm(S_full)
        np.testing.assert_allclose(error_none, expected_none, rtol=1e-10)
