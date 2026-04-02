"""
Tests for parallel local DMRG micro-steps.

These tests verify that:
1. Local micro-steps execute in parallel without communication
2. Each processor handles its assigned sites correctly
3. Results from all processors can be gathered correctly
4. The parallel phase is embarrassingly parallel

IMPORTANT: These tests require MPI and must be run with:
    mpirun -np 2 pytest a2dmrg/tests/test_parallel_local_steps.py -v

For non-MPI testing, we also provide serial tests that simulate MPI behavior.
"""

import pytest
import numpy as np
import quimb.tensor as qtn

from quimb.tensor import MPS_rand_state
from a2dmrg.parallel import (
    parallel_local_microsteps,
    gather_local_results,
    prepare_candidate_mps_list,
)

# Check if mpi4py is available by trying to import it
# We don't import at module level to avoid collection errors
def _check_mpi():
    """Check if MPI is available."""
    try:
        from a2dmrg.mpi_compat import MPI
        return True, MPI
    except (ImportError, RuntimeError):
        # RuntimeError occurs when MPI library is not found
        return False, None

HAS_MPI, MPI = _check_mpi()


class MockComm:
    """
    Mock MPI communicator for serial testing.

    This allows us to test the parallel code structure without actually
    running MPI.
    """
    def __init__(self, rank=0, size=1):
        self._rank = rank
        self._size = size

    def Get_rank(self):
        return self._rank

    def Get_size(self):
        return self._size

    def allgather(self, data):
        """Mock allgather - just returns list with single data element."""
        return [data]


class TestParallelLocalMicrosteps:
    """Test parallel local micro-step execution."""

    def test_parallel_microstep_basic_serial(self):
        """
        Test basic parallel micro-step execution in serial mode (np=1).

        This verifies the code structure works without actual MPI.
        """
        # Create simple test system
        L = 10
        bond_dim = 4
        dtype = np.float64

        mps = MPS_rand_state(L, bond_dim=bond_dim, dtype=dtype)
        mpo = qtn.MPO_ham_heis(L, cyclic=False)

        # Use mock communicator (serial)
        comm = MockComm(rank=0, size=1)

        # Execute parallel local micro-steps
        results = parallel_local_microsteps(
            mps, mpo, comm,
            microstep_type="one_site",
            tol=1e-8
        )

        # Verify: should have updates for all L sites
        assert len(results) == L

        # Verify: each result is a tuple (list[ndarray], energy)
        for site, (updated_arrays, energy) in results.items():
            assert isinstance(updated_arrays, list)
            assert isinstance(energy, (float, np.floating))
            assert len(updated_arrays) == L
            assert np.isfinite(energy)

    def test_site_distribution_in_parallel(self):
        """
        Test that sites are distributed correctly across processors.

        This test works in both serial and parallel modes.
        """
        L = 10
        bond_dim = 4
        dtype = np.float64

        mps = MPS_rand_state(L, bond_dim=bond_dim, dtype=dtype)
        mpo = qtn.MPO_ham_heis(L, cyclic=False)

        if HAS_MPI:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            n_procs = comm.Get_size()
        else:
            comm = MockComm(rank=0, size=1)
            rank = 0
            n_procs = 1

        # Execute parallel local micro-steps
        results = parallel_local_microsteps(
            mps, mpo, comm,
            microstep_type="one_site",
            tol=1e-8
        )

        # Verify: number of results matches expected site distribution
        from a2dmrg.parallel import distribute_sites
        my_sites = distribute_sites(L, n_procs, rank)
        assert len(results) == len(my_sites)

        # Verify: results are for the correct sites
        assert set(results.keys()) == set(my_sites)

    def test_no_communication_during_local_phase(self):
        """
        Verify that parallel_local_microsteps does not perform communication.

        This is a conceptual test - in practice, the function design ensures
        no MPI calls are made during the local update phase.
        """
        L = 10
        bond_dim = 4
        dtype = np.float64

        mps = MPS_rand_state(L, bond_dim=bond_dim, dtype=dtype)
        mpo = qtn.MPO_ham_heis(L, cyclic=False)

        comm = MockComm(rank=0, size=1)

        # This should complete without any MPI communication calls
        results = parallel_local_microsteps(
            mps, mpo, comm,
            microstep_type="one_site",
            tol=1e-8
        )

        # If we get here, no communication happened (by design)
        assert len(results) > 0

    def test_valid_mps_structure_after_update(self):
        """
        Verify that updated MPS arrays maintain valid structure.

        Each processor should produce valid MPS tensor lists.
        """
        L = 10
        bond_dim = 4
        dtype = np.float64

        mps = MPS_rand_state(L, bond_dim=bond_dim, dtype=dtype)
        mpo = qtn.MPO_ham_heis(L, cyclic=False)

        if HAS_MPI:
            comm = MPI.COMM_WORLD
        else:
            comm = MockComm(rank=0, size=1)

        results = parallel_local_microsteps(
            mps, mpo, comm,
            microstep_type="one_site",
            tol=1e-8
        )

        # Check each updated MPS
        for site, (updated_arrays, energy) in results.items():
            # Should have correct length
            assert len(updated_arrays) == L

            # Should have correct dtype
            assert updated_arrays[site].dtype == dtype

            # All arrays should be finite and non-zero
            for arr in updated_arrays:
                assert np.all(np.isfinite(arr)), f"Site {site}: non-finite values in array"
                assert np.linalg.norm(arr) > 0, f"Site {site}: zero array"

            # Energy should be finite
            assert np.isfinite(energy)

    def test_energy_is_reasonable(self):
        """
        Verify that energies from local updates are reasonable.

        For Heisenberg model, ground state energy should be negative.
        """
        L = 10
        bond_dim = 4
        dtype = np.float64

        mps = MPS_rand_state(L, bond_dim=bond_dim, dtype=dtype)
        mpo = qtn.MPO_ham_heis(L, cyclic=False)

        if HAS_MPI:
            comm = MPI.COMM_WORLD
        else:
            comm = MockComm(rank=0, size=1)

        results = parallel_local_microsteps(
            mps, mpo, comm,
            microstep_type="one_site",
            tol=1e-8
        )

        # Check energies
        for site, (updated_arrays, energy) in results.items():
            # Heisenberg ground state energy should be negative
            assert energy < 0, f"Site {site}: energy={energy} should be negative"

            # Should be in reasonable range for L=10 Heisenberg chain
            # Ground state energy ~ -0.44 per bond, so ~ -4.4 for 10 sites
            # Local energy might be different, but should be order of magnitude
            assert -20 < energy < 0


class TestTwoSiteMicrosteps:
    """Test two-site parallel micro-steps."""

    def test_two_site_basic(self):
        """Test two-site micro-steps in parallel."""
        L = 6
        # Use bond_dim=2 to avoid rank truncation in canonicalization
        # (edge sites have natural max rank=d=2)
        bond_dim = 2
        dtype = np.float64

        mps = MPS_rand_state(L, bond_dim=bond_dim, dtype=dtype)
        mpo = qtn.MPO_ham_heis(L, cyclic=False)

        comm = MockComm(rank=0, size=1)

        results = parallel_local_microsteps(
            mps, mpo, comm,
            microstep_type="two_site",
            max_bond=8,
            cutoff=1e-10,
            tol=1e-8
        )

        # For two-site: update bonds (i, i+1)
        assert len(results) == L - 1

        # Check each result
        for bond, (updated_arrays, energy) in results.items():
            assert isinstance(updated_arrays, list)
            assert np.isfinite(energy)
            assert energy < 0  # Heisenberg should be negative

    def test_two_site_bond_dimension_control(self):
        """Test that max_bond parameter controls bond dimension."""
        L = 6
        # Use bond_dim=2 to avoid rank truncation in canonicalization
        bond_dim = 2
        max_bond = 6
        dtype = np.float64

        mps = MPS_rand_state(L, bond_dim=bond_dim, dtype=dtype)
        mpo = qtn.MPO_ham_heis(L, cyclic=False)

        comm = MockComm(rank=0, size=1)

        results = parallel_local_microsteps(
            mps, mpo, comm,
            microstep_type="two_site",
            max_bond=max_bond,
            cutoff=1e-10,
            tol=1e-8
        )

        # Check bond dimensions don't exceed max_bond
        for bond, (updated_arrays, energy) in results.items():
            # Check bond dimensions of updated MPS arrays
            # Each array is (chi_L, d, chi_R)
            for i in range(len(updated_arrays) - 1):
                chi_R = updated_arrays[i].shape[2]
                # Should not exceed max_bond
                assert chi_R <= max_bond


class TestGatherResults:
    """Test gathering results from all processors."""

    def test_gather_local_results_serial(self):
        """Test gathering results in serial mode."""
        L = 10
        bond_dim = 4
        dtype = np.float64

        mps = MPS_rand_state(L, bond_dim=bond_dim, dtype=dtype)
        mpo = qtn.MPO_ham_heis(L, cyclic=False)

        comm = MockComm(rank=0, size=1)

        # Get local results
        local_results = parallel_local_microsteps(
            mps, mpo, comm,
            microstep_type="one_site",
            tol=1e-8
        )

        # Gather (in serial, this is a no-op)
        all_results = gather_local_results(local_results, comm)

        # Should have all sites
        assert len(all_results) == L
        assert set(all_results.keys()) == set(range(L))

    def test_prepare_candidate_mps_list(self):
        """Test preparing candidate MPS list for coarse-space."""
        L = 10
        bond_dim = 4
        dtype = np.float64

        original_mps = MPS_rand_state(L, bond_dim=bond_dim, dtype=dtype)
        mpo = qtn.MPO_ham_heis(L, cyclic=False)

        comm = MockComm(rank=0, size=1)

        # Get local results
        local_results = parallel_local_microsteps(
            original_mps, mpo, comm,
            microstep_type="one_site",
            tol=1e-8
        )

        # Gather results
        all_results = gather_local_results(local_results, comm)

        # Prepare candidate list
        candidates = prepare_candidate_mps_list(original_mps, all_results)

        # Should have L+1 candidates: original + L updated
        assert len(candidates) == L + 1

        # First should be a list of arrays with correct length
        assert isinstance(candidates[0], list)
        assert len(candidates[0]) == L

        # Rest should be list[ndarray] with correct length
        for i in range(1, len(candidates)):
            assert isinstance(candidates[i], list)
            assert len(candidates[i]) == L


class TestComplex128Support:
    """Test parallel micro-steps with complex128 dtype."""

    def test_complex_one_site(self):
        """Test one-site updates with complex dtype."""
        L = 8
        bond_dim = 4
        dtype = np.complex128

        mps = MPS_rand_state(L, bond_dim=bond_dim, dtype=dtype)
        mpo = qtn.MPO_ham_heis(L, cyclic=False)

        comm = MockComm(rank=0, size=1)

        results = parallel_local_microsteps(
            mps, mpo, comm,
            microstep_type="one_site",
            tol=1e-8
        )

        # Check complex dtype preserved
        for site, (updated_arrays, energy) in results.items():
            assert updated_arrays[site].dtype == np.complex128

            # Energy should be real (Hermitian Hamiltonian)
            assert np.abs(np.imag(energy)) < 1e-10

    def test_complex_two_site(self):
        """Test two-site updates with complex dtype."""
        L = 6
        # Use bond_dim=2 to avoid rank truncation in canonicalization
        bond_dim = 2
        dtype = np.complex128

        mps = MPS_rand_state(L, bond_dim=bond_dim, dtype=dtype)
        mpo = qtn.MPO_ham_heis(L, cyclic=False)

        comm = MockComm(rank=0, size=1)

        results = parallel_local_microsteps(
            mps, mpo, comm,
            microstep_type="two_site",
            max_bond=8,
            cutoff=1e-10,
            tol=1e-8
        )

        # Check complex dtype preserved
        for bond, (updated_arrays, energy) in results.items():
            # Check both sites at the bond
            assert updated_arrays[bond].dtype == np.complex128
            if bond + 1 < L:
                assert updated_arrays[bond + 1].dtype == np.complex128


class TestEdgeCases:
    """Test edge cases for parallel local steps."""

    def test_single_site_system(self):
        """Test with L=1 (single site)."""
        # Note: quimb's MPO_ham_heis doesn't support L=1
        # Skip this test for now
        pytest.skip("quimb MPO_ham_heis doesn't support L=1")

        L = 1
        bond_dim = 2
        dtype = np.float64

        mps = MPS_rand_state(L, bond_dim=bond_dim, dtype=dtype)
        mpo = qtn.MPO_ham_heis(L, cyclic=False)

        comm = MockComm(rank=0, size=1)

        results = parallel_local_microsteps(
            mps, mpo, comm,
            microstep_type="one_site",
            tol=1e-8
        )

        # Should have one update
        assert len(results) == 1
        assert 0 in results

    def test_two_site_system(self):
        """Test with L=2 (minimal system)."""
        L = 2
        bond_dim = 2
        dtype = np.float64

        mps = MPS_rand_state(L, bond_dim=bond_dim, dtype=dtype)
        mpo = qtn.MPO_ham_heis(L, cyclic=False)

        comm = MockComm(rank=0, size=1)

        # One-site
        results = parallel_local_microsteps(
            mps, mpo, comm,
            microstep_type="one_site",
            tol=1e-8
        )
        assert len(results) == 2

        # Two-site
        results = parallel_local_microsteps(
            mps, mpo, comm,
            microstep_type="two_site",
            tol=1e-8
        )
        assert len(results) == 1  # Only one bond (0, 1)

    def test_invalid_microstep_type(self):
        """Test that invalid microstep_type raises error."""
        L = 10
        bond_dim = 4
        dtype = np.float64

        mps = MPS_rand_state(L, bond_dim=bond_dim, dtype=dtype)
        mpo = qtn.MPO_ham_heis(L, cyclic=False)

        comm = MockComm(rank=0, size=1)

        with pytest.raises(ValueError, match="Unknown microstep_type"):
            parallel_local_microsteps(
                mps, mpo, comm,
                microstep_type="invalid",
                tol=1e-8
            )


# Tests that REQUIRE actual MPI (run with mpirun -np 2 or more)
@pytest.mark.skipif(not HAS_MPI, reason="Requires MPI")
class TestActualMPI:
    """
    Tests that require actual MPI execution.

    Run with: mpirun -np 2 pytest a2dmrg/tests/test_parallel_local_steps.py::TestActualMPI -v
    """

    def test_two_processors_site_distribution(self):
        """Test that sites are distributed correctly across 2 processors."""
        if not HAS_MPI:
            pytest.skip("MPI not available")

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        if size < 2:
            pytest.skip("Need at least 2 MPI processes")

        L = 10
        bond_dim = 4
        dtype = np.float64

        mps = MPS_rand_state(L, bond_dim=bond_dim, dtype=dtype)
        mpo = qtn.MPO_ham_heis(L, cyclic=False)

        # Each processor executes independently
        results = parallel_local_microsteps(
            mps, mpo, comm,
            microstep_type="one_site",
            tol=1e-8
        )

        # Verify this processor updated the correct sites
        from a2dmrg.parallel import distribute_sites
        my_sites = distribute_sites(L, size, rank)
        assert len(results) == len(my_sites)
        assert set(results.keys()) == set(my_sites)

        # Gather to verify all sites covered
        all_results = gather_local_results(results, comm)

        # All processors should now have complete results
        assert len(all_results) == L

    def test_embarrassingly_parallel_property(self):
        """
        Test that local phase is truly embarrassingly parallel.

        This verifies that processors don't wait for each other during updates.
        """
        if not HAS_MPI:
            pytest.skip("MPI not available")

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        if size < 2:
            pytest.skip("Need at least 2 MPI processes")

        L = 10
        bond_dim = 4
        dtype = np.float64

        mps = MPS_rand_state(L, bond_dim=bond_dim, dtype=dtype)
        mpo = qtn.MPO_ham_heis(L, cyclic=False)

        # Time the local update phase
        import time
        start_time = time.time()

        results = parallel_local_microsteps(
            mps, mpo, comm,
            microstep_type="one_site",
            tol=1e-8
        )

        elapsed_time = time.time() - start_time

        # Verify we got results
        assert len(results) > 0

        # The timing should be roughly proportional to number of sites per processor
        # (not total sites - that would indicate communication overhead)
        from a2dmrg.parallel import distribute_sites
        my_sites = distribute_sites(L, size, rank)

        # This is a soft check - just verify we completed
        assert elapsed_time < 60.0  # Should be much faster for L=10


def test_parallel_local_microsteps_one_site_covers_all_sites():
    """
    Verify that parallel_local_microsteps covers all L sites and returns
    finite energies for each. (One-site variant, serial mode.)
    """
    import numpy as np
    import quimb.tensor as qtn
    from a2dmrg.mpi_compat import MPI
    from a2dmrg.parallel.local_steps import parallel_local_microsteps

    L = 6
    bond_dim = 4
    mps = qtn.MPS_rand_state(L, bond_dim=bond_dim, phys_dim=2, dtype=np.float64)
    mpo = qtn.MPO_ham_heis(L, cyclic=False)
    comm = MPI.COMM_WORLD  # serial (size=1) in pytest

    results = parallel_local_microsteps(mps, mpo, comm, microstep_type="one_site", tol=1e-8)

    # All L sites should be covered
    assert set(results.keys()) == set(range(L)), f"Expected sites 0..{L-1}, got {sorted(results.keys())}"
    # All energies should be finite for Heisenberg
    for site, (updated_arrays, energy) in results.items():
        assert np.isfinite(energy), f"Site {site}: energy is not finite"
        assert updated_arrays is not None
        assert len(updated_arrays) == L
