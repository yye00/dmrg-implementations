"""
Tests for linear combination formation (Phase 3 of A2DMRG).

Test Coverage:
- Feature 27: Form optimal MPS Y_tilde = Sum_j c*_j Y^(j)
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
)
from a2dmrg.numerics.observables import (
    compute_energy_numpy,
    compute_overlap_numpy,
)
from a2dmrg.numerics.numpy_utils import extract_mps_arrays, extract_mpo_arrays, arrays_to_quimb_mps


@pytest.fixture
def simple_arrays_list():
    """Create a list of simple MPS as numpy array lists for testing."""
    L = 6
    bond_dim = 4

    arrays_list = []
    for i in range(4):  # Create 4 candidates
        mps = MPS_rand_state(L, bond_dim=bond_dim, dtype=float)
        mps.left_canonize()
        arrays_list.append(extract_mps_arrays(mps))

    return arrays_list


@pytest.fixture
def simple_mpo_arrays():
    """Create a simple Hamiltonian MPO as numpy arrays."""
    L = 6
    mpo = MPO_ham_heis(L, j=1.0, bz=0.0, cyclic=False)
    return extract_mpo_arrays(mpo)


@pytest.fixture
def simple_hamiltonian():
    """Create a simple Hamiltonian MPO (quimb, for verify_linear_combination_energy)."""
    L = 6
    return MPO_ham_heis(L, j=1.0, bz=0.0, cyclic=False)


class TestBasicLinearCombination:
    """Test basic linear combination formation (Feature 27, Steps 1-3)."""

    def test_linear_combination_basic(self, simple_arrays_list, simple_mpo_arrays):
        """
        Test basic linear combination formation.

        Feature 27, Steps 1-3:
        - Create list of d+1 candidate MPS
        - Obtain coefficients c* from coarse-space solver
        - Form linear combination for each site tensor
        """
        candidate_list = simple_arrays_list
        mpo_arrays = simple_mpo_arrays

        # Step 2: Build coarse matrices and solve
        H_coarse, S_coarse = build_coarse_matrices(candidate_list, mpo_arrays)
        energy, coefficients = solve_coarse_eigenvalue_problem(H_coarse, S_coarse)

        # Verify we got valid coefficients
        assert coefficients.shape == (len(candidate_list),)
        assert np.all(np.isfinite(coefficients))

        # Step 3: Form linear combination
        combined_arrays = form_linear_combination(candidate_list, coefficients)

        # Basic checks
        assert combined_arrays is not None
        assert len(combined_arrays) == len(candidate_list[0])
        # Physical dimension check on first site
        assert combined_arrays[0].shape[1] == candidate_list[0][0].shape[1]

    def test_coefficients_normalization(self, simple_arrays_list, simple_mpo_arrays):
        """
        Test that coefficients are properly normalized.

        The solver should return normalized coefficients: c^dag S c = 1
        """
        candidate_list = simple_arrays_list
        mpo_arrays = simple_mpo_arrays

        H_coarse, S_coarse = build_coarse_matrices(candidate_list, mpo_arrays)
        energy, coefficients = solve_coarse_eigenvalue_problem(H_coarse, S_coarse)

        # Check normalization: c^dag S c should be 1
        norm_sq = np.conj(coefficients) @ S_coarse @ coefficients
        assert abs(norm_sq - 1.0) < 1e-8, f"c^dag S c = {norm_sq}, expected 1.0"

    def test_linear_combination_shape(self, simple_arrays_list):
        """
        Test that linear combination preserves MPS structure.

        Each tensor should have shape (chi_L, d, chi_R) -- always 3D.
        """
        candidate_list = simple_arrays_list
        L = len(candidate_list[0])

        # Use simple uniform coefficients for this test
        n_candidates = len(candidate_list)
        coefficients = np.ones(n_candidates) / np.sqrt(n_candidates)

        combined_arrays = form_linear_combination(candidate_list, coefficients)

        # Check all tensor shapes
        for i in range(L):
            arr = combined_arrays[i]
            # All tensors in our convention are 3D: (chi_L, d, chi_R)
            assert arr.ndim == 3, f"Site {i}: expected 3D tensor, got shape {arr.shape}"

            # Verify physical dimension is 2 (spin-1/2)
            phys_dim = arr.shape[1]
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
        mps_arrays = extract_mps_arrays(mps)

        coefficients = np.array([1.0])
        candidate_list = [mps_arrays]

        combined = form_linear_combination(candidate_list, coefficients)

        # Result should be very close to original
        overlap = compute_overlap_numpy(combined, mps_arrays)
        norm_combined = np.sqrt(abs(compute_overlap_numpy(combined, combined)))
        norm_original = np.sqrt(abs(compute_overlap_numpy(mps_arrays, mps_arrays)))

        # Overlap should be close to product of norms
        expected_overlap = norm_combined * norm_original
        assert abs(abs(overlap) - expected_overlap) < 1e-8


class TestEnergyVerification:
    """Test energy verification (Feature 27, Steps 4-5)."""

    def test_energy_matches_coarse_prediction(self, simple_arrays_list, simple_mpo_arrays):
        """
        Test that energy of combined MPS matches coarse-space prediction.

        Feature 27, Step 5: Verify energy = c^dag H_coarse c / (c^dag S_coarse c)
        """
        candidate_list = simple_arrays_list
        mpo_arrays = simple_mpo_arrays

        # Build coarse matrices
        H_coarse, S_coarse = build_coarse_matrices(candidate_list, mpo_arrays)

        # Solve for optimal coefficients
        energy_predicted, coefficients = solve_coarse_eigenvalue_problem(H_coarse, S_coarse)

        # Form linear combination
        combined_arrays = form_linear_combination(candidate_list, coefficients)

        # Compute actual energy using numpy
        energy_actual = compute_energy_numpy(combined_arrays, mpo_arrays, normalize=True)
        energy_actual = np.real(energy_actual)

        # Verify energies match
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

    def test_verify_linear_combination_energy_function(
        self, simple_arrays_list, simple_mpo_arrays, simple_hamiltonian
    ):
        """
        Test the verify_linear_combination_energy helper function.
        """
        candidate_list = simple_arrays_list
        mpo_arrays = simple_mpo_arrays
        mpo_quimb = simple_hamiltonian

        H_coarse, S_coarse = build_coarse_matrices(candidate_list, mpo_arrays)
        energy_predicted, coefficients = solve_coarse_eigenvalue_problem(H_coarse, S_coarse)

        combined_arrays = form_linear_combination(candidate_list, coefficients)

        # verify_linear_combination_energy still takes quimb objects
        combined_quimb = arrays_to_quimb_mps(combined_arrays)

        is_valid, E_actual, E_predicted, rel_error = verify_linear_combination_energy(
            combined_quimb, mpo_quimb, coefficients, H_coarse, S_coarse, tolerance=1e-6
        )

        assert is_valid, f"Energy verification failed: rel_error={rel_error:.2e}"
        assert abs(E_actual - E_predicted) < 1e-6

    def test_energy_lower_than_initial_state(self, simple_arrays_list, simple_mpo_arrays):
        """
        Test that combined state has energy <= initial state energy.

        The coarse-space minimization should produce a better (or equal) state.
        """
        candidate_list = simple_arrays_list
        mpo_arrays = simple_mpo_arrays

        # Energy of initial state (first candidate)
        E_initial = compute_energy_numpy(candidate_list[0], mpo_arrays, normalize=True)
        E_initial = np.real(E_initial)

        # Build coarse matrices and solve
        H_coarse, S_coarse = build_coarse_matrices(candidate_list, mpo_arrays)
        energy_optimal, coefficients = solve_coarse_eigenvalue_problem(H_coarse, S_coarse)

        # Form linear combination
        combined_arrays = form_linear_combination(candidate_list, coefficients)
        E_combined = compute_energy_numpy(combined_arrays, mpo_arrays, normalize=True)
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

        # Create complex MPS as numpy arrays
        candidate_list = []
        for i in range(3):
            mps = MPS_rand_state(L, bond_dim=bond_dim, dtype=complex)
            mps.left_canonize()
            candidate_list.append(extract_mps_arrays(mps))

        # Create Hamiltonian
        mpo_quimb = MPO_ham_heis(L, j=1.0, bz=0.0, cyclic=False)
        mpo_arrays = extract_mpo_arrays(mpo_quimb)

        # Build coarse matrices
        H_coarse, S_coarse = build_coarse_matrices(candidate_list, mpo_arrays)

        # Verify matrices are complex
        assert H_coarse.dtype == np.complex128
        assert S_coarse.dtype == np.complex128

        # Solve
        energy, coefficients = solve_coarse_eigenvalue_problem(H_coarse, S_coarse)

        # Coefficients should be complex
        assert coefficients.dtype == np.complex128

        # Form linear combination
        combined_arrays = form_linear_combination(candidate_list, coefficients)

        # Verify result is complex
        assert np.iscomplexobj(combined_arrays[0])

        # Verify energy using numpy
        combined_quimb = arrays_to_quimb_mps(combined_arrays)
        is_valid, _, _, _ = verify_linear_combination_energy(
            combined_quimb, mpo_quimb, coefficients, H_coarse, S_coarse
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

        candidate_list = [extract_mps_arrays(mps1), extract_mps_arrays(mps2)]

        mpo_quimb = MPO_ham_heis(L, j=1.0)
        mpo_arrays = extract_mpo_arrays(mpo_quimb)

        H_coarse, S_coarse = build_coarse_matrices(candidate_list, mpo_arrays)
        energy, coefficients = solve_coarse_eigenvalue_problem(H_coarse, S_coarse)

        combined_arrays = form_linear_combination(candidate_list, coefficients)

        # Should work fine
        assert len(combined_arrays) == L

    def test_mismatched_coefficients_length(self, simple_arrays_list):
        """Test error handling for mismatched coefficient length."""
        candidate_list = simple_arrays_list

        # Wrong number of coefficients
        coefficients = np.array([0.5, 0.5])  # Should be 4

        with pytest.raises(ValueError, match="coefficients length"):
            form_linear_combination(candidate_list, coefficients)

    def test_zero_coefficients(self, simple_arrays_list):
        """Test with all zero coefficients (edge case)."""
        candidate_list = simple_arrays_list

        # All near-zero coefficients
        coefficients = np.ones(len(candidate_list)) * 1e-10

        combined_arrays = form_linear_combination(candidate_list, coefficients)

        # Result should be very small
        norm_sq = compute_overlap_numpy(combined_arrays, combined_arrays)

        # Norm should be very small (roughly (1e-10)^2 * number_of_candidates)
        assert abs(norm_sq) < 1e-15


class TestParallelMode:
    """Test parallel MPI mode (Feature 28)."""

    def test_serial_mode_baseline(self, simple_arrays_list, simple_mpo_arrays, simple_hamiltonian):
        """
        Test serial mode as baseline for parallel tests.

        Feature 28, Step 4: All processors should get same result.
        First establish what that result is in serial mode.
        """
        candidate_list = simple_arrays_list
        mpo_arrays = simple_mpo_arrays
        mpo_quimb = simple_hamiltonian

        H_coarse, S_coarse = build_coarse_matrices(candidate_list, mpo_arrays)
        energy, coefficients = solve_coarse_eigenvalue_problem(H_coarse, S_coarse)

        # Serial mode: no comm, no assigned_sites
        combined_arrays = form_linear_combination(
            candidate_list, coefficients, comm=None, assigned_sites=None
        )

        # Verify energy via quimb-based verify function
        combined_quimb = arrays_to_quimb_mps(combined_arrays)
        is_valid, _, _, _ = verify_linear_combination_energy(
            combined_quimb, mpo_quimb, coefficients, H_coarse, S_coarse
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
            candidate_list.append(extract_mps_arrays(mps))

        mpo_quimb = MPO_ham_heis(L, j=1.0)
        mpo_arrays = extract_mpo_arrays(mpo_quimb)

        # Build coarse matrices (parallel)
        from a2dmrg.parallel import distribute_sites
        my_sites = distribute_sites(L, size, rank)

        H_coarse, S_coarse = build_coarse_matrices(
            candidate_list, mpo_arrays, comm=comm, assigned_sites=my_sites
        )

        # Rank 0 solves (others will receive via broadcast in form_linear_combination)
        if rank == 0:
            energy, coefficients = solve_coarse_eigenvalue_problem(H_coarse, S_coarse)
        else:
            coefficients = None

        # Form linear combination (includes broadcast)
        combined_arrays = form_linear_combination(
            candidate_list, coefficients, comm=comm, assigned_sites=my_sites
        )

        # All ranks compute energy
        E_combined = compute_energy_numpy(combined_arrays, mpo_arrays, normalize=True)
        E_combined = np.real(E_combined)

        # Gather energies to rank 0 for comparison
        all_energies = comm.gather(E_combined, root=0)

        if rank == 0:
            # All energies should be the same
            for i, E in enumerate(all_energies):
                assert abs(E - all_energies[0]) < 1e-10, (
                    f"Rank {i} energy mismatch: {E} vs {all_energies[0]}"
                )
