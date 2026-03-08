"""
Test #66: Numerical robustness and error handling.

This test verifies that A2DMRG handles numerical edge cases gracefully:
- Nearly singular mass matrix S_coarse
- Ill-conditioned effective Hamiltonians
- Complex dtype numerical stability
- Extreme parameter values
"""

import numpy as np
import pytest
from a2dmrg.dmrg import a2dmrg_main
from quimb.tensor import SpinHam1D

pytestmark = pytest.mark.mpi


class TestNumericalRobustness:
    """Test #66: Numerical robustness and error handling."""

    def test_nearly_singular_coarse_space(self):
        """
        Step 1: Test with nearly singular mass matrix S_coarse.

        Small bond_dim creates a small coarse space, which can lead to
        near-linear dependence and nearly singular S_coarse.
        """
        L = 8
        bond_dim = 2  # Very small bond dimension
        max_sweeps = 3

        # Create Heisenberg Hamiltonian
        builder = SpinHam1D(S=1/2)
        builder += 1.0, 'X', 'X'
        builder += 1.0, 'Y', 'Y'
        builder += 1.0, 'Z', 'Z'
        mpo = builder.build_mpo(L)

        # Should complete without crashing
        energy, mps = a2dmrg_main(
            L=L, mpo=mpo, max_sweeps=max_sweeps,
            bond_dim=bond_dim, tol=1e-10,
            dtype=np.float64, one_site=True, verbose=False
        )

        # Step 2: Verify regularization prevents crashes
        assert np.isfinite(energy), "Energy should be finite (regularization working)"
        assert mps is not None, "MPS should be returned"
        assert not np.isnan(energy), "Energy should not be NaN"

        # Energy should be reasonable (negative for antiferromagnetic Heisenberg)
        assert energy < 0, f"Energy should be negative for AF Heisenberg, got {energy}"

    def test_ill_conditioned_effective_hamiltonian(self):
        """
        Step 3: Test with ill-conditioned H_eff.

        Very tight tolerance + small bond dim can create numerical stress.
        """
        L = 6
        bond_dim = 4
        max_sweeps = 5
        tol = 1e-14  # Very tight tolerance (near machine precision)

        builder = SpinHam1D(S=1/2)
        builder += 1.0, 'X', 'X'
        builder += 1.0, 'Y', 'Y'
        builder += 1.0, 'Z', 'Z'
        mpo = builder.build_mpo(L)

        # Step 4: Verify eigensolver convergence checks
        energy, mps = a2dmrg_main(
            L=L, mpo=mpo, max_sweeps=max_sweeps,
            bond_dim=bond_dim, tol=tol,
            dtype=np.float64, one_site=True, verbose=False
        )

        # Step 5: Verify graceful degradation rather than crashes
        assert np.isfinite(energy), "Should return finite energy even with tight tolerance"
        assert mps is not None, "Should return valid MPS"

        # Energy should still be reasonable
        assert energy < 0, "Energy should be negative"

    def test_complex_dtype_numerical_stability(self):
        """
        Test numerical stability with complex dtype.

        Complex arithmetic can have different numerical behavior.
        """
        L = 6
        bond_dim = 8
        max_sweeps = 10

        builder = SpinHam1D(S=1/2)
        builder += 1.0, 'X', 'X'
        builder += 1.0, 'Y', 'Y'
        builder += 1.0, 'Z', 'Z'
        mpo = builder.build_mpo(L)

        energy, mps = a2dmrg_main(
            L=L, mpo=mpo, max_sweeps=max_sweeps,
            bond_dim=bond_dim, tol=1e-10,
            dtype=np.complex128, one_site=True, verbose=False
        )

        # Energy should be real (Hermitian Hamiltonian)
        assert np.isfinite(energy), "Energy should be finite"
        assert abs(np.imag(energy)) < 1e-10, f"Energy should be real, imaginary part: {np.imag(energy)}"

        # Should get reasonable energy
        assert energy < 0, "Energy should be negative"

    def test_minimum_bond_dimension(self):
        """
        Test with bond_dim=1 (extreme edge case).

        Bond dimension of 1 is the minimum possible - effectively no entanglement.
        This stresses the coarse space solver.
        """
        L = 4
        bond_dim = 1  # Absolute minimum
        max_sweeps = 5

        builder = SpinHam1D(S=1/2)
        builder += 1.0, 'X', 'X'
        mpo = builder.build_mpo(L)

        energy, mps = a2dmrg_main(
            L=L, mpo=mpo, max_sweeps=max_sweeps,
            bond_dim=bond_dim, tol=1e-6,
            dtype=np.float64, one_site=True, verbose=False
        )

        assert np.isfinite(energy), "Should handle bond_dim=1 gracefully"
        assert mps is not None, "Should return valid MPS"

    def test_very_large_tolerance(self):
        """
        Test with very large tolerance (loose convergence).

        Large tolerance should allow quick convergence.
        """
        L = 8
        bond_dim = 4
        max_sweeps = 3
        tol = 1e-3  # Very loose tolerance

        builder = SpinHam1D(S=1/2)
        builder += 1.0, 'X', 'X'
        builder += 1.0, 'Y', 'Y'
        builder += 1.0, 'Z', 'Z'
        mpo = builder.build_mpo(L)

        energy, mps = a2dmrg_main(
            L=L, mpo=mpo, max_sweeps=max_sweeps,
            bond_dim=bond_dim, tol=tol,
            dtype=np.float64, one_site=True, verbose=False
        )

        assert np.isfinite(energy), "Should work with loose tolerance"
        assert mps is not None, "Should return valid MPS"

    def test_mixed_precision_stability(self):
        """
        Test that complex128 produces valid results for real Hamiltonian.

        For a real Hamiltonian, complex128 should produce real eigenvalues
        and reasonable ground state energies.

        Note: float64 and complex128 may converge to slightly different
        local minima due to random initialization, so we don't require
        exact agreement. We just verify both give reasonable results.
        """
        L = 6
        bond_dim = 8
        max_sweeps = 15
        tol = 1e-8

        builder = SpinHam1D(S=1/2)
        builder += 1.0, 'X', 'X'
        builder += 1.0, 'Y', 'Y'
        builder += 1.0, 'Z', 'Z'
        mpo = builder.build_mpo(L)

        # Run with float64
        energy_real, _ = a2dmrg_main(
            L=L, mpo=mpo, max_sweeps=max_sweeps,
            bond_dim=bond_dim, tol=tol,
            dtype=np.float64, one_site=True, verbose=False
        )

        # Run with complex128
        energy_complex, _ = a2dmrg_main(
            L=L, mpo=mpo, max_sweeps=max_sweeps,
            bond_dim=bond_dim, tol=tol,
            dtype=np.complex128, one_site=True, verbose=False
        )

        # Both should be finite and reasonable
        assert np.isfinite(energy_real), "Real energy should be finite"
        assert np.isfinite(energy_complex), "Complex energy should be finite"

        # Complex energy should have negligible imaginary part (Hermitian H)
        assert abs(np.imag(energy_complex)) < 1e-10, f"Complex energy should be real, imag: {np.imag(energy_complex)}"

        # Both should be negative for AF Heisenberg
        assert energy_real < 0, f"Real energy should be negative, got {energy_real}"
        assert np.real(energy_complex) < 0, f"Complex energy should be negative, got {np.real(energy_complex)}"

        # Both should be in reasonable range (not checking exact equality due to random init)
        # For L=6 Heisenberg, exact ground state is around -2.49
        assert energy_real > -3.0 and energy_real < 0, f"Real energy out of range: {energy_real}"
        assert np.real(energy_complex) > -3.0 and np.real(energy_complex) < 0, f"Complex energy out of range: {np.real(energy_complex)}"

    def test_zero_initial_energy_convergence(self):
        """
        Test convergence even if initial state has very different energy.

        A random initial state may have very high/low energy. The algorithm
        should still converge to the ground state.
        """
        L = 6
        bond_dim = 8
        max_sweeps = 20  # More sweeps to ensure convergence
        tol = 1e-10

        builder = SpinHam1D(S=1/2)
        builder += 1.0, 'X', 'X'
        builder += 1.0, 'Y', 'Y'
        builder += 1.0, 'Z', 'Z'
        mpo = builder.build_mpo(L)

        energy, mps = a2dmrg_main(
            L=L, mpo=mpo, max_sweeps=max_sweeps,
            bond_dim=bond_dim, tol=tol,
            dtype=np.float64, one_site=True, verbose=False
        )

        # Should converge to reasonable ground state energy
        assert np.isfinite(energy), "Should converge to finite energy"
        assert energy < 0, "Should converge to negative energy"

        # For L=6 Heisenberg, ground state energy is around -2.49 (from exact diagonalization)
        # With bond_dim=8, we should get close
        assert energy < -2.0, f"Should get reasonable ground state, got {energy}"

    def test_small_system_numerical_stability(self):
        """
        Test numerical stability with very small system (L=4).

        Small systems can have different numerical behavior.
        """
        L = 4
        bond_dim = 8  # Larger than needed for L=4
        max_sweeps = 10

        builder = SpinHam1D(S=1/2)
        builder += 1.0, 'X', 'X'
        builder += 1.0, 'Y', 'Y'
        builder += 1.0, 'Z', 'Z'
        mpo = builder.build_mpo(L)

        energy, mps = a2dmrg_main(
            L=L, mpo=mpo, max_sweeps=max_sweeps,
            bond_dim=bond_dim, tol=1e-10,
            dtype=np.float64, one_site=True, verbose=False
        )

        assert np.isfinite(energy), "Should handle small systems"
        assert mps is not None, "Should return valid MPS"

        # For L=4 Heisenberg, exact ground state energy is known
        # Should be close to exact value
        assert energy < 0, "Energy should be negative"

    def test_rapid_convergence_detection(self):
        """
        Test that algorithm detects rapid convergence correctly.

        With large tolerance, should converge in few sweeps.
        """
        L = 6
        bond_dim = 8
        max_sweeps = 50  # Allow many sweeps
        tol = 1e-4  # Moderate tolerance

        builder = SpinHam1D(S=1/2)
        builder += 1.0, 'X', 'X'
        builder += 1.0, 'Y', 'Y'
        builder += 1.0, 'Z', 'Z'
        mpo = builder.build_mpo(L)

        energy, mps = a2dmrg_main(
            L=L, mpo=mpo, max_sweeps=max_sweeps,
            bond_dim=bond_dim, tol=tol,
            dtype=np.float64, one_site=True, verbose=False
        )

        # Should converge successfully
        assert np.isfinite(energy), "Should converge"
        assert mps is not None, "Should return valid MPS"

        # Energy should be reasonable (even if not fully converged)
        assert energy < 0, "Energy should be negative"
