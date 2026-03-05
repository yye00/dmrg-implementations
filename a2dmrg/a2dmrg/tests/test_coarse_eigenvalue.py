"""
Tests for coarse-space eigenvalue problem solver (Features 26-27).

This module tests the solution of the generalized eigenvalue problem
that arises in Phase 3 (Coarse-Space Minimization) of A2DMRG.
"""

import numpy as np
import pytest
from a2dmrg.numerics import (
    solve_coarse_eigenvalue_problem,
    verify_solution,
    estimate_condition_number
)


class TestBasicSolution:
    """Test basic eigenvalue problem solving (Feature 26, Steps 1-6)."""

    def test_simple_2x2_problem(self):
        """Test on a simple 2x2 eigenvalue problem."""
        # Create simple test matrices
        H_coarse = np.array([[2.0, 0.5],
                             [0.5, 3.0]], dtype=np.float64)
        S_coarse = np.eye(2, dtype=np.float64)

        # Solve
        energy, coefficients = solve_coarse_eigenvalue_problem(H_coarse, S_coarse)

        # Verify solution
        is_valid, residual = verify_solution(H_coarse, S_coarse, energy, coefficients)

        assert is_valid, f"Solution is not valid, residual = {residual}"
        assert residual < 1e-10, f"Residual too large: {residual}"
        assert np.abs(np.linalg.norm(coefficients) - 1.0) < 1e-10, "Coefficients not normalized"

        # For this problem, we can compute exact eigenvalues
        # det(H - λI) = 0 => (2-λ)(3-λ) - 0.25 = 0
        # => λ^2 - 5λ + 5.75 = 0
        # => λ = (5 ± √(25 - 23)) / 2 = (5 ± √2) / 2
        expected_min_eigenvalue = (5.0 - np.sqrt(2.0)) / 2.0
        assert np.abs(energy - expected_min_eigenvalue) < 1e-10

    def test_diagonal_problem(self):
        """Test on a diagonal eigenvalue problem."""
        n = 5
        H_coarse = np.diag([5.0, 3.0, 7.0, 2.0, 8.0])
        S_coarse = np.eye(n)

        energy, coefficients = solve_coarse_eigenvalue_problem(H_coarse, S_coarse)

        # Minimum eigenvalue should be 2.0
        assert np.abs(energy - 2.0) < 1e-10

        # Eigenvector should be e_3 = [0, 0, 0, 1, 0]
        expected = np.zeros(n)
        expected[3] = 1.0
        assert np.allclose(np.abs(coefficients), np.abs(expected), atol=1e-10)

        # Verify solution
        is_valid, residual = verify_solution(H_coarse, S_coarse, energy, coefficients)
        assert is_valid

    def test_svd_based_regularization(self):
        """Test SVD-based regularization (Feature 26, Step 2)."""
        # Create a well-conditioned problem
        H_coarse = np.array([[1.0, 0.2],
                             [0.2, 2.0]])
        S_coarse = np.array([[1.0, 0.1],
                             [0.1, 1.0]])

        # Should work with default regularization
        energy, coefficients = solve_coarse_eigenvalue_problem(
            H_coarse, S_coarse, regularization=1e-10
        )

        is_valid, residual = verify_solution(H_coarse, S_coarse, energy, coefficients)
        assert is_valid

    def test_standard_eigenvalue_transform(self):
        """Test transformation to standard eigenvalue problem (Feature 26, Step 3)."""
        # Non-trivial mass matrix
        H_coarse = np.array([[2.0, 1.0, 0.0],
                             [1.0, 3.0, 0.5],
                             [0.0, 0.5, 4.0]])
        S_coarse = np.array([[1.0, 0.2, 0.0],
                             [0.2, 1.0, 0.1],
                             [0.0, 0.1, 1.0]])

        energy, coefficients = solve_coarse_eigenvalue_problem(H_coarse, S_coarse)

        # Verify the solution satisfies the generalized eigenvalue equation
        is_valid, residual = verify_solution(H_coarse, S_coarse, energy, coefficients)
        assert is_valid
        assert residual < 1e-8

    def test_scipy_eigh_integration(self):
        """Test integration with scipy.linalg.eigh (Feature 26, Step 4)."""
        H_coarse = np.random.RandomState(42).randn(4, 4)
        H_coarse = 0.5 * (H_coarse + H_coarse.T)  # Make Hermitian
        S_coarse = np.eye(4)

        energy, coefficients = solve_coarse_eigenvalue_problem(H_coarse, S_coarse)

        # Should match scipy.linalg.eigh for identity S
        from scipy.linalg import eigh
        expected_eigenvalues, expected_eigenvectors = eigh(H_coarse)

        assert np.abs(energy - expected_eigenvalues[0]) < 1e-10

    def test_transform_back(self):
        """Test transformation back c* = S^(-1/2) c_tilde (Feature 26, Step 5)."""
        # Use a non-trivial but well-conditioned mass matrix
        H_coarse = np.array([[3.0, 1.0],
                             [1.0, 2.0]])
        S_coarse = np.array([[2.0, 0.5],
                             [0.5, 1.0]])

        energy, coefficients = solve_coarse_eigenvalue_problem(H_coarse, S_coarse)

        # Verify the solution
        is_valid, residual = verify_solution(H_coarse, S_coarse, energy, coefficients)
        assert is_valid


class TestIllConditioned:
    """Test handling of ill-conditioned mass matrix (Feature 27)."""

    def test_near_zero_singular_values(self):
        """Test S_coarse with near-zero singular values (Feature 27, Step 1)."""
        # Create ill-conditioned mass matrix
        S_coarse = np.array([[1.0, 0.99, 0.98],
                             [0.99, 1.0, 0.99],
                             [0.98, 0.99, 1.0]])
        H_coarse = np.diag([2.0, 3.0, 4.0])

        # Estimate condition number
        cond = estimate_condition_number(S_coarse)
        assert cond > 100, f"Expected ill-conditioned matrix, got condition number {cond}"

        # Should still work with regularization
        energy, coefficients = solve_coarse_eigenvalue_problem(
            H_coarse, S_coarse, regularization=1e-10
        )

        # Verify solution is reasonable
        is_valid, residual = verify_solution(H_coarse, S_coarse, energy, coefficients)
        assert is_valid

    def test_regularization_threshold(self):
        """Test regularization threshold parameter (Feature 27, Step 2)."""
        # Create matrix with known singular values
        U = np.eye(3)
        s = np.array([1.0, 1e-5, 1e-12])  # One very small singular value
        S_coarse = U @ np.diag(s) @ U.T
        H_coarse = np.eye(3)

        # With strict regularization, should truncate very small values
        energy, coefficients = solve_coarse_eigenvalue_problem(
            H_coarse, S_coarse, regularization=1e-8
        )

        # Solution should still be finite and valid
        assert np.isfinite(energy)
        assert np.all(np.isfinite(coefficients))
        assert not np.any(np.isnan(coefficients))

    def test_truncate_small_singular_values(self):
        """Test truncation of small singular values in pseudoinverse (Feature 27, Step 3)."""
        # Create rank-deficient mass matrix
        # S has rank 2, smallest singular value is exactly zero
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        S_coarse = np.outer(v1, v1) + np.outer(v2, v2)  # Rank 2 matrix

        H_coarse = np.eye(3)

        # Should handle rank deficiency via regularization
        energy, coefficients = solve_coarse_eigenvalue_problem(
            H_coarse, S_coarse, regularization=1e-10
        )

        # Solution should be finite
        assert np.isfinite(energy)
        assert np.all(np.isfinite(coefficients))

    def test_solution_stability(self):
        """Test that solution is stable for ill-conditioned matrix (Feature 27, Step 4)."""
        # Create ill-conditioned but not singular matrix
        H_coarse = np.array([[1.0, 0.5],
                             [0.5, 2.0]])
        S_coarse = np.array([[1.0, 0.9999],
                             [0.9999, 1.0]])

        energy1, coefficients1 = solve_coarse_eigenvalue_problem(
            H_coarse, S_coarse, regularization=1e-10
        )
        energy2, coefficients2 = solve_coarse_eigenvalue_problem(
            H_coarse, S_coarse, regularization=1e-9
        )

        # Solutions should be similar
        assert np.abs(energy1 - energy2) < 1e-6
        # Eigenvectors may differ by sign
        assert np.allclose(coefficients1, coefficients2, atol=1e-4) or \
               np.allclose(coefficients1, -coefficients2, atol=1e-4)

    def test_physically_meaningful_eigenvalue(self):
        """Test that eigenvalue is physically meaningful (Feature 27, Step 5)."""
        # For a positive-definite H and S, eigenvalue should be positive
        H_coarse = np.array([[2.0, 0.1],
                             [0.1, 3.0]])
        S_coarse = np.array([[1.0, 0.05],
                             [0.05, 1.0]])

        energy, coefficients = solve_coarse_eigenvalue_problem(H_coarse, S_coarse)

        # Energy should be positive
        assert energy > 0, f"Expected positive energy, got {energy}"

        # Energy should be less than or equal to the minimum diagonal element
        # (Variational principle)
        min_H_diag = np.min(np.diag(H_coarse))
        assert energy <= min_H_diag + 1e-8, \
            f"Energy {energy} exceeds minimum diagonal {min_H_diag}"


class TestComplex128:
    """Test complex-valued matrices."""

    def test_complex_hermitian_matrices(self):
        """Test with complex Hermitian matrices."""
        # Create complex Hermitian H and S
        H_coarse = np.array([[2.0, 1.0 + 0.5j],
                             [1.0 - 0.5j, 3.0]], dtype=np.complex128)
        S_coarse = np.array([[1.0, 0.1 + 0.05j],
                             [0.1 - 0.05j, 1.0]], dtype=np.complex128)

        energy, coefficients = solve_coarse_eigenvalue_problem(H_coarse, S_coarse)

        # Eigenvalue should be real for Hermitian matrices
        assert np.abs(np.imag(energy)) < 1e-10, f"Eigenvalue should be real, got {energy}"

        # Coefficients may be complex
        assert coefficients.dtype == np.complex128

        # Verify solution
        is_valid, residual = verify_solution(H_coarse, S_coarse, energy, coefficients)
        assert is_valid, f"Solution is not valid, residual = {residual}"

    def test_complex_normalization(self):
        """Test that complex eigenvectors are properly normalized."""
        H_coarse = np.array([[1.0, 0.5j],
                             [-0.5j, 2.0]], dtype=np.complex128)
        S_coarse = np.eye(2, dtype=np.complex128)

        energy, coefficients = solve_coarse_eigenvalue_problem(H_coarse, S_coarse)

        # Check normalization
        norm = np.linalg.norm(coefficients)
        assert np.abs(norm - 1.0) < 1e-10, f"Coefficients not normalized: norm = {norm}"


class TestReturnAll:
    """Test return_all parameter."""

    def test_return_all_eigenvalues(self):
        """Test that return_all returns all eigenvalues and eigenvectors."""
        H_coarse = np.diag([3.0, 1.0, 4.0, 2.0])
        S_coarse = np.eye(4)

        energies, coefficients_matrix = solve_coarse_eigenvalue_problem(
            H_coarse, S_coarse, return_all=True
        )

        # Should return all 4 eigenvalues
        assert energies.shape == (4,)
        assert coefficients_matrix.shape == (4, 4)

        # Eigenvalues should be sorted
        assert np.all(energies[:-1] <= energies[1:])

        # Should match diagonal elements (for this case)
        expected = np.sort([3.0, 1.0, 4.0, 2.0])
        assert np.allclose(energies, expected, atol=1e-10)

    def test_all_eigenvectors_orthonormal(self):
        """Test that all returned eigenvectors are orthonormal."""
        H_coarse = np.random.RandomState(123).randn(5, 5)
        H_coarse = 0.5 * (H_coarse + H_coarse.T)
        S_coarse = np.eye(5)

        energies, coefficients_matrix = solve_coarse_eigenvalue_problem(
            H_coarse, S_coarse, return_all=True
        )

        # Check orthonormality
        overlap = coefficients_matrix.T @ coefficients_matrix
        assert np.allclose(overlap, np.eye(5), atol=1e-10)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_dimension(self):
        """Test 1x1 matrices."""
        H_coarse = np.array([[2.5]])
        S_coarse = np.array([[1.0]])

        energy, coefficients = solve_coarse_eigenvalue_problem(H_coarse, S_coarse)

        assert np.abs(energy - 2.5) < 1e-10
        assert np.abs(coefficients[0] - 1.0) < 1e-10 or \
               np.abs(coefficients[0] + 1.0) < 1e-10

    def test_large_matrix(self):
        """Test with larger matrix (n=10)."""
        n = 10
        np.random.seed(456)
        H_coarse = np.random.randn(n, n)
        H_coarse = 0.5 * (H_coarse + H_coarse.T)
        S_coarse = np.eye(n) + 0.1 * np.random.randn(n, n)
        S_coarse = 0.5 * (S_coarse + S_coarse.T)
        S_coarse += n * np.eye(n)  # Make positive definite

        energy, coefficients = solve_coarse_eigenvalue_problem(H_coarse, S_coarse)

        is_valid, residual = verify_solution(H_coarse, S_coarse, energy, coefficients)
        assert is_valid

    def test_mismatched_shapes(self):
        """Test error handling for mismatched matrix shapes."""
        H_coarse = np.eye(3)
        S_coarse = np.eye(4)

        with pytest.raises(ValueError, match="same shape"):
            solve_coarse_eigenvalue_problem(H_coarse, S_coarse)

    def test_non_square_matrices(self):
        """Test error handling for non-square matrices."""
        H_coarse = np.random.randn(3, 4)
        S_coarse = np.random.randn(3, 4)

        with pytest.raises(ValueError, match="square"):
            solve_coarse_eigenvalue_problem(H_coarse, S_coarse)

    def test_empty_matrices(self):
        """Test error handling for empty matrices."""
        H_coarse = np.array([]).reshape(0, 0)
        S_coarse = np.array([]).reshape(0, 0)

        with pytest.raises(ValueError, match="empty"):
            solve_coarse_eigenvalue_problem(H_coarse, S_coarse)

    def test_negative_regularization(self):
        """Test error handling for negative regularization."""
        H_coarse = np.eye(2)
        S_coarse = np.eye(2)

        with pytest.raises(ValueError, match="non-negative"):
            solve_coarse_eigenvalue_problem(H_coarse, S_coarse, regularization=-1e-5)


class TestConditionNumber:
    """Test condition number estimation."""

    def test_well_conditioned_matrix(self):
        """Test condition number for well-conditioned matrix."""
        S_coarse = np.eye(3)
        cond = estimate_condition_number(S_coarse)
        assert np.abs(cond - 1.0) < 1e-10

    def test_ill_conditioned_matrix(self):
        """Test condition number for ill-conditioned matrix."""
        # Create matrix with large condition number
        U = np.eye(3)
        s = np.array([1.0, 1e-3, 1e-6])
        S_coarse = U @ np.diag(s) @ U.T

        cond = estimate_condition_number(S_coarse)
        assert cond > 1e5

    def test_singular_matrix(self):
        """Test condition number for singular matrix."""
        # Rank-deficient matrix
        S_coarse = np.array([[1.0, 1.0],
                             [1.0, 1.0]])
        cond = estimate_condition_number(S_coarse)
        assert np.isinf(cond)


class TestVerifySolution:
    """Test solution verification function."""

    def test_valid_solution(self):
        """Test verification of a valid solution."""
        H_coarse = np.array([[2.0, 0.5],
                             [0.5, 3.0]])
        S_coarse = np.eye(2)

        energy, coefficients = solve_coarse_eigenvalue_problem(H_coarse, S_coarse)

        is_valid, residual = verify_solution(H_coarse, S_coarse, energy, coefficients)

        assert is_valid
        assert residual < 1e-10

    def test_invalid_solution(self):
        """Test verification of an invalid solution."""
        H_coarse = np.array([[2.0, 0.5],
                             [0.5, 3.0]])
        S_coarse = np.eye(2)

        # Use wrong eigenvalue and eigenvector
        energy = 10.0
        coefficients = np.array([1.0, 0.0])

        is_valid, residual = verify_solution(H_coarse, S_coarse, energy, coefficients)

        assert not is_valid
        assert residual > 0.1


def test_return_diagnostics():
    """solve_coarse_eigenvalue_problem with return_diagnostics=True returns n_effective."""
    import numpy as np
    from a2dmrg.numerics.coarse_eigenvalue import solve_coarse_eigenvalue_problem

    H = np.array([[1.0, 0.1, 0.0],
                  [0.1, 2.0, 0.1],
                  [0.0, 0.1, 3.0]])
    S = np.array([[1.0, 0.5, 0.0],
                  [0.5, 1.0, 0.5],
                  [0.0, 0.5, 1.0]])

    energy, coeffs, diag = solve_coarse_eigenvalue_problem(
        H, S, regularization=1e-10, return_diagnostics=True
    )

    assert "n_effective" in diag
    assert isinstance(diag["n_effective"], int)
    assert 1 <= diag["n_effective"] <= 3
    assert np.isfinite(energy)
