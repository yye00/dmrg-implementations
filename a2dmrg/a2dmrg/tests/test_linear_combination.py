"""
Tests for linear combination formation (Phase 3 of A2DMRG).

Test Coverage:
- Feature 27: Form optimal MPS Ỹ = Σⱼ c*ⱼ Y^(j)
- Feature 28: Distributed formation with MPI
"""

import pytest
import numpy as np
from quimb.tensor import MPS_rand_state, MPO_ham_heis

from a2dmrg.parallel import (
    form_linear_combination,
    verify_linear_combination_energy,
    build_coarse_matrices,
)
from a2dmrg.numerics import (
    solve_coarse_eigenvalue_problem,
    compute_energy,
    compute_overlap,
)


@pytest.fixture
def simple_mps_list():
    """Create a list of simple MPS for testing."""
    L = 6
    bond_dim = 4

    # Create several random MPS
    mps_list = []
    for i in range(4):  # Create 4 candidates
        mps = MPS_rand_state(L, bond_dim=bond_dim, dtype=float)
        mps.left_canonize()
        mps_list.append(mps)

    return mps_list


@pytest.fixture
def simple_hamiltonian():
    """Create a simple Hamiltonian MPO."""
    L = 6
    # Heisenberg Hamiltonian
    mpo = MPO_ham_heis(L, j=1.0, bz=0.0, cyclic=False)
    return mpo


class TestBasicLinearCombination:
    """Test basic linear combination formation (Feature 27, Steps 1-3)."""

    def test_linear_combination_basic(self, simple_mps_list, simple_hamiltonian):
        """
        Test basic linear combination formation.

        Feature 27, Steps 1-3:
        - Create list of d+1 candidate MPS
        - Obtain coefficients c* from coarse-space solver
        - Form linear combination for each site tensor
        """
        # Step 1: Already have candidate list from fixture
        candidate_list = simple_mps_list
        mpo = simple_hamiltonian

        # Step 2: Build coarse matrices and solve
        H_coarse, S_coarse = build_coarse_matrices(candidate_list, mpo)
        energy, coefficients = solve_coarse_eigenvalue_problem(H_coarse, S_coarse)

        # Verify we got valid coefficients
        assert coefficients.shape == (len(candidate_list),)
        assert np.all(np.isfinite(coefficients))

        # Step 3: Form linear combination
        combined_mps = form_linear_combination(candidate_list, coefficients)

        # Basic checks
        assert combined_mps is not None
        assert combined_mps.L == candidate_list[0].L
        assert combined_mps.phys_dim() == candidate_list[0].phys_dim()

    def test_coefficients_normalization(self, simple_mps_list, simple_hamiltonian):
        """
        Test that coefficients are properly normalized.

        The solver should return normalized coefficients: c† S c = 1
        """
        candidate_list = simple_mps_list
        mpo = simple_hamiltonian

        H_coarse, S_coarse = build_coarse_matrices(candidate_list, mpo)
        energy, coefficients = solve_coarse_eigenvalue_problem(H_coarse, S_coarse)

        # Check normalization: c† S c should be 1
        norm_sq = np.conj(coefficients) @ S_coarse @ coefficients
        assert abs(norm_sq - 1.0) < 1e-8, f"c† S c = {norm_sq}, expected 1.0"

    def test_linear_combination_shape(self, simple_mps_list):
        """
        Test that linear combination preserves MPS structure.

        Each tensor should have correct shape. Quimb uses:
        - Left edge (i=0): (bond, phys) - 2D
        - Middle sites: (left_bond, right_bond, phys) - 3D
        - Right edge (i=L-1): (bond, phys) - 2D
        """
        candidate_list = simple_mps_list
        L = candidate_list[0].L

        # Use simple uniform coefficients for this test
        n_candidates = len(candidate_list)
        coefficients = np.ones(n_candidates) / np.sqrt(n_candidates)

        combined_mps = form_linear_combination(candidate_list, coefficients)

        # Check all tensor shapes
        for i in range(L):
            tensor = combined_mps.tensors[i]
            if hasattr(tensor, 'data'):
                shape = tensor.data.shape
            else:
                shape = np.asarray(tensor).shape

            # Edge sites are 2D, middle sites are 3D
            if i == 0 or i == L - 1:
                assert len(shape) == 2, f"Edge site {i}: expected 2D tensor, got shape {shape}"
            else:
                assert len(shape) == 3, f"Middle site {i}: expected 3D tensor, got shape {shape}"

            # Verify physical dimension is 2 (spin-1/2)
            phys_dim = shape[-1]
            assert phys_dim == 2, f"Site {i}: expected phys_dim=2, got {phys_dim}"

    def test_linear_combination_with_single_mps(self):
        """
        Test linear combination with a single MPS (edge case).

        With c = [1.0] and single MPS, result should equal original.
        """
        L = 6
        bond_dim = 4
        mps = MPS_rand_state(L, bond_dim=bond_dim, dtype=float)
        mps.left_canonize()

        coefficients = np.array([1.0])
        candidate_list = [mps]

        combined = form_linear_combination(candidate_list, coefficients)

        # Result should be very close to original
        overlap = compute_overlap(combined, mps)
        norm_combined = np.sqrt(compute_overlap(combined, combined))
        norm_original = np.sqrt(compute_overlap(mps, mps))

        # Overlap should be close to product of norms
        expected_overlap = norm_combined * norm_original
        assert abs(overlap - expected_overlap) < 1e-8


class TestEnergyVerification:
    """Test energy verification (Feature 27, Steps 4-5)."""

    def test_energy_matches_coarse_prediction(self, simple_mps_list, simple_hamiltonian):
        """
        Test that energy of combined MPS matches coarse-space prediction.

        Feature 27, Step 5: Verify energy = c† H_coarse c / (c† S_coarse c)
        """
        candidate_list = simple_mps_list
        mpo = simple_hamiltonian

        # Build coarse matrices
        H_coarse, S_coarse = build_coarse_matrices(candidate_list, mpo)

        # Solve for optimal coefficients
        energy_predicted, coefficients = solve_coarse_eigenvalue_problem(H_coarse, S_coarse)

        # Form linear combination
        combined_mps = form_linear_combination(candidate_list, coefficients)

        # Compute actual energy
        energy_actual = compute_energy(combined_mps, mpo, normalize=True)

        # Extract scalar if needed
        if hasattr(energy_actual, 'data'):
            energy_actual = np.asarray(energy_actual.data).ravel()[0]
        energy_actual = np.real(energy_actual)

        # Verify energies match
        # For very small energies (< 1e-10), use absolute error instead of relative
        if abs(energy_predicted) < 1e-10:
            absolute_error = abs(energy_actual - energy_predicted)
            assert absolute_error < 1e-8, (
                f"Energy mismatch (absolute): actual={energy_actual:.2e}, "
                f"predicted={energy_predicted:.2e}, abs_error={absolute_error:.2e}"
            )
        else:
            relative_error = abs(energy_actual - energy_predicted) / abs(energy_predicted)
            assert relative_error < 1e-6, (
                f"Energy mismatch: actual={energy_actual:.10f}, "
                f"predicted={energy_predicted:.10f}, rel_error={relative_error:.2e}"
            )

    def test_verify_linear_combination_energy_function(self, simple_mps_list, simple_hamiltonian):
        """
        Test the verify_linear_combination_energy helper function.
        """
        candidate_list = simple_mps_list
        mpo = simple_hamiltonian

        H_coarse, S_coarse = build_coarse_matrices(candidate_list, mpo)
        energy_predicted, coefficients = solve_coarse_eigenvalue_problem(H_coarse, S_coarse)

        combined_mps = form_linear_combination(candidate_list, coefficients)

        # Use verification function
        is_valid, E_actual, E_predicted, rel_error = verify_linear_combination_energy(
            combined_mps, mpo, coefficients, H_coarse, S_coarse, tolerance=1e-6
        )

        assert is_valid, f"Energy verification failed: rel_error={rel_error:.2e}"
        assert abs(E_actual - E_predicted) < 1e-6

    def test_energy_lower_than_initial_state(self, simple_mps_list, simple_hamiltonian):
        """
        Test that combined state has energy <= initial state energy.

        The coarse-space minimization should produce a better (or equal) state.
        """
        candidate_list = simple_mps_list
        mpo = simple_hamiltonian

        # Energy of initial state (first candidate)
        E_initial = compute_energy(candidate_list[0], mpo, normalize=True)
        if hasattr(E_initial, 'data'):
            E_initial = np.asarray(E_initial.data).ravel()[0]
        E_initial = np.real(E_initial)

        # Build coarse matrices and solve
        H_coarse, S_coarse = build_coarse_matrices(candidate_list, mpo)
        energy_optimal, coefficients = solve_coarse_eigenvalue_problem(H_coarse, S_coarse)

        # Form linear combination
        combined_mps = form_linear_combination(candidate_list, coefficients)
        E_combined = compute_energy(combined_mps, mpo, normalize=True)
        if hasattr(E_combined, 'data'):
            E_combined = np.asarray(E_combined.data).ravel()[0]
        E_combined = np.real(E_combined)

        # Combined state should have lower or equal energy
        # (allowing small numerical tolerance)
        assert E_combined <= E_initial + 1e-8, (
            f"Combined energy ({E_combined:.10f}) should be <= "
            f"initial energy ({E_initial:.10f})"
        )


class TestComplexDtype:
    """Test linear combination with complex MPS."""

    def test_complex_linear_combination(self):
        """
        Test linear combination with complex128 dtype.

        Important for Josephson junction problems.
        """
        L = 6
        bond_dim = 4

        # Create complex MPS
        candidate_list = []
        for i in range(3):
            mps = MPS_rand_state(L, bond_dim=bond_dim, dtype=complex)
            mps.left_canonize()
            candidate_list.append(mps)

        # Create Hamiltonian
        mpo = MPO_ham_heis(L, j=1.0, bz=0.0, cyclic=False)

        # Build coarse matrices
        H_coarse, S_coarse = build_coarse_matrices(candidate_list, mpo)

        # Verify matrices are complex
        assert H_coarse.dtype == np.complex128
        assert S_coarse.dtype == np.complex128

        # Solve
        energy, coefficients = solve_coarse_eigenvalue_problem(H_coarse, S_coarse)

        # Coefficients should be complex
        assert coefficients.dtype == np.complex128

        # Form linear combination
        combined_mps = form_linear_combination(candidate_list, coefficients)

        # Verify result is complex
        tensor0 = combined_mps.tensors[0]
        if hasattr(tensor0, 'data'):
            data = np.asarray(tensor0.data)
        else:
            data = np.asarray(tensor0)
        assert np.iscomplexobj(data)

        # Verify energy
        is_valid, _, _, _ = verify_linear_combination_energy(
            combined_mps, mpo, coefficients, H_coarse, S_coarse
        )
        assert is_valid


class TestEdgeCases:
    """Test edge cases for linear combination."""

    def test_two_candidates(self):
        """Test with minimal number of candidates (2)."""
        L = 4
        bond_dim = 3

        mps1 = MPS_rand_state(L, bond_dim=bond_dim, dtype=float)
        mps2 = MPS_rand_state(L, bond_dim=bond_dim, dtype=float)
        mps1.left_canonize()
        mps2.left_canonize()

        candidate_list = [mps1, mps2]

        mpo = MPO_ham_heis(L, j=1.0)

        H_coarse, S_coarse = build_coarse_matrices(candidate_list, mpo)
        energy, coefficients = solve_coarse_eigenvalue_problem(H_coarse, S_coarse)

        combined_mps = form_linear_combination(candidate_list, coefficients)

        # Should work fine
        assert combined_mps.L == L

    def test_mismatched_coefficients_length(self, simple_mps_list):
        """Test error handling for mismatched coefficient length."""
        candidate_list = simple_mps_list

        # Wrong number of coefficients
        coefficients = np.array([0.5, 0.5])  # Should be 4

        with pytest.raises(ValueError, match="coefficients length"):
            form_linear_combination(candidate_list, coefficients)

    def test_zero_coefficients(self, simple_mps_list):
        """Test with all zero coefficients (edge case)."""
        candidate_list = simple_mps_list

        # All zero coefficients - this is a degenerate case
        # quimb's scalar multiplication with zero causes division by zero
        # So we test with very small coefficients instead
        coefficients = np.ones(len(candidate_list)) * 1e-10

        combined_mps = form_linear_combination(candidate_list, coefficients)

        # Result should be very small
        # Check norm is small
        from a2dmrg.numerics import compute_overlap
        norm_sq = compute_overlap(combined_mps, combined_mps)

        # Norm should be very small (roughly (1e-10)^2 * number_of_candidates)
        assert abs(norm_sq) < 1e-15


class TestParallelMode:
    """Test parallel MPI mode (Feature 28)."""

    def test_serial_mode_baseline(self, simple_mps_list, simple_hamiltonian):
        """
        Test serial mode as baseline for parallel tests.

        Feature 28, Step 4: All processors should get same result.
        First establish what that result is in serial mode.
        """
        candidate_list = simple_mps_list
        mpo = simple_hamiltonian

        H_coarse, S_coarse = build_coarse_matrices(candidate_list, mpo)
        energy, coefficients = solve_coarse_eigenvalue_problem(H_coarse, S_coarse)

        # Serial mode: no comm, no assigned_sites
        combined_mps = form_linear_combination(
            candidate_list, coefficients, comm=None, assigned_sites=None
        )

        # Verify energy
        is_valid, _, _, _ = verify_linear_combination_energy(
            combined_mps, mpo, coefficients, H_coarse, S_coarse
        )
        assert is_valid

    @pytest.mark.skip(reason="Requires MPI execution with mpirun")
    def test_parallel_formation_two_procs(self):
        """
        Test parallel formation with 2 processors.

        Feature 28, Steps 1-5:
        - Broadcast coefficients
        - Each processor forms local contribution
        - Verify all have same result

        This test must be run with: mpirun -np 2 pytest ...
        """
        from a2dmrg.mpi_compat import MPI

        comm = MPI.COMM_WORLD
        rank = comm.rank
        size = comm.size

        assert size >= 2, "This test requires at least 2 MPI processes"

        # Create test data (all ranks do this for simplicity)
        L = 8
        bond_dim = 4
        candidate_list = []
        for i in range(4):
            mps = MPS_rand_state(L, bond_dim=bond_dim, dtype=float)
            mps.left_canonize()
            candidate_list.append(mps)

        mpo = MPO_ham_heis(L, j=1.0)

        # Build coarse matrices (parallel)
        from a2dmrg.parallel import distribute_sites
        my_sites = distribute_sites(L, size, rank)

        H_coarse, S_coarse = build_coarse_matrices(
            candidate_list, mpo, comm=comm, assigned_sites=my_sites
        )

        # Rank 0 solves (others will receive via broadcast in form_linear_combination)
        if rank == 0:
            energy, coefficients = solve_coarse_eigenvalue_problem(H_coarse, S_coarse)
        else:
            coefficients = None

        # Form linear combination (includes broadcast)
        combined_mps = form_linear_combination(
            candidate_list, coefficients, comm=comm, assigned_sites=my_sites
        )

        # All ranks compute energy
        E_combined = compute_energy(combined_mps, mpo, normalize=True)
        if hasattr(E_combined, 'data'):
            E_combined = np.asarray(E_combined.data).ravel()[0]

        # Gather energies to rank 0 for comparison
        all_energies = comm.gather(E_combined, root=0)

        if rank == 0:
            # All energies should be the same
            for i, E in enumerate(all_energies):
                assert abs(E - all_energies[0]) < 1e-10, (
                    f"Rank {i} energy mismatch: {E} vs {all_energies[0]}"
                )
