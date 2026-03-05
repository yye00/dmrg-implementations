"""
Tests for site distribution across processors.

These tests verify features 19-20:
- Feature 19: Basic site distribution algorithm
- Feature 20: Load balancing for various L and P combinations
"""

import pytest
import numpy as np
from a2dmrg.parallel import distribute_sites, verify_distribution


class TestBasicDistribution:
    """Test basic site distribution functionality (Feature 19)."""

    def test_distribute_sites_L10_P3(self):
        """
        Feature 19, Step 1-6: Test with L=10 sites, P=3 processors.

        Expected distribution:
        - rank 0: [0, 1, 2, 3] (4 sites, has remainder)
        - rank 1: [4, 5, 6]    (3 sites)
        - rank 2: [7, 8, 9]    (3 sites)
        """
        L, n_procs = 10, 3

        # Step 2: Call distribute_sites for each rank
        sites_rank0 = distribute_sites(L, n_procs, rank=0)
        sites_rank1 = distribute_sites(L, n_procs, rank=1)
        sites_rank2 = distribute_sites(L, n_procs, rank=2)

        # Step 3: Verify rank 0 gets sites [0,1,2,3]
        assert sites_rank0 == [0, 1, 2, 3], \
            f"Rank 0 should get [0,1,2,3], got {sites_rank0}"

        # Step 4: Verify rank 1 gets sites [4,5,6]
        assert sites_rank1 == [4, 5, 6], \
            f"Rank 1 should get [4,5,6], got {sites_rank1}"

        # Step 5: Verify rank 2 gets sites [7,8,9]
        assert sites_rank2 == [7, 8, 9], \
            f"Rank 2 should get [7,8,9], got {sites_rank2}"

        # Step 6: Verify all sites covered exactly once
        all_sites = sites_rank0 + sites_rank1 + sites_rank2
        assert sorted(all_sites) == list(range(L)), \
            "All sites should be covered exactly once"
        assert len(all_sites) == len(set(all_sites)), \
            "No site should appear twice"

    def test_distribute_sites_returns_list(self):
        """Verify distribute_sites returns a list of integers."""
        sites = distribute_sites(10, 3, 0)
        assert isinstance(sites, list), "Should return a list"
        assert all(isinstance(s, int) for s in sites), "All elements should be integers"

    def test_distribute_sites_single_processor(self):
        """Test with single processor (all sites go to rank 0)."""
        L = 20
        sites = distribute_sites(L, n_procs=1, rank=0)
        assert sites == list(range(L)), \
            "Single processor should get all sites"

    def test_distribute_sites_equal_division(self):
        """Test case where L is evenly divisible by n_procs."""
        L, n_procs = 12, 3  # 12 / 3 = 4 exactly

        sites_rank0 = distribute_sites(L, n_procs, 0)
        sites_rank1 = distribute_sites(L, n_procs, 1)
        sites_rank2 = distribute_sites(L, n_procs, 2)

        # All ranks should get exactly 4 sites
        assert len(sites_rank0) == 4
        assert len(sites_rank1) == 4
        assert len(sites_rank2) == 4

        # Verify partitioning
        assert sites_rank0 == [0, 1, 2, 3]
        assert sites_rank1 == [4, 5, 6, 7]
        assert sites_rank2 == [8, 9, 10, 11]

    def test_distribute_sites_more_procs_than_sites(self):
        """Test with more processors than sites."""
        L, n_procs = 3, 5

        # First 3 ranks get 1 site each
        assert distribute_sites(L, n_procs, 0) == [0]
        assert distribute_sites(L, n_procs, 1) == [1]
        assert distribute_sites(L, n_procs, 2) == [2]

        # Last 2 ranks get no sites
        assert distribute_sites(L, n_procs, 3) == []
        assert distribute_sites(L, n_procs, 4) == []


class TestLoadBalancing:
    """Test load balancing for various L and P combinations (Feature 20)."""

    def test_L40_P8_even_division(self):
        """
        Feature 20, Step 1: Test L=40, P=8 (even division).

        40 / 8 = 5 sites per processor exactly.
        """
        L, n_procs = 40, 8

        site_counts = []
        all_sites = []

        for rank in range(n_procs):
            sites = distribute_sites(L, n_procs, rank)
            site_counts.append(len(sites))
            all_sites.extend(sites)

        # Each rank should get exactly 5 sites
        assert all(count == 5 for count in site_counts), \
            f"All ranks should get 5 sites, got {site_counts}"

        # Verify all sites covered
        assert sorted(all_sites) == list(range(L))

    def test_L41_P8_remainder_1(self):
        """
        Feature 20, Step 2: Test L=41, P=8 (remainder 1).

        41 = 8 * 5 + 1
        - Rank 0: 6 sites (gets the extra)
        - Ranks 1-7: 5 sites each
        """
        L, n_procs = 41, 8

        site_counts = []
        all_sites = []

        for rank in range(n_procs):
            sites = distribute_sites(L, n_procs, rank)
            site_counts.append(len(sites))
            all_sites.extend(sites)

        # Step 4: Verify load balancing (max - min <= 1)
        assert max(site_counts) - min(site_counts) <= 1, \
            f"Load imbalance too large: {site_counts}"

        # Rank 0 should have 6, others should have 5
        assert site_counts[0] == 6, f"Rank 0 should have 6 sites, got {site_counts[0]}"
        assert all(count == 5 for count in site_counts[1:]), \
            f"Ranks 1-7 should have 5 sites each, got {site_counts[1:]}"

        # Step 5: Verify no site duplicated or missed
        assert sorted(all_sites) == list(range(L))
        assert len(all_sites) == len(set(all_sites))

    def test_L47_P8_remainder_7(self):
        """
        Feature 20, Step 3: Test L=47, P=8 (remainder 7).

        47 = 8 * 5 + 7
        - Ranks 0-6: 6 sites each (7 ranks get extra)
        - Rank 7: 5 sites
        """
        L, n_procs = 47, 8

        site_counts = []
        all_sites = []

        for rank in range(n_procs):
            sites = distribute_sites(L, n_procs, rank)
            site_counts.append(len(sites))
            all_sites.extend(sites)

        # Step 4: Verify load balancing
        assert max(site_counts) - min(site_counts) <= 1, \
            f"Load imbalance too large: {site_counts}"

        # First 7 ranks should have 6, last rank should have 5
        assert all(count == 6 for count in site_counts[:7]), \
            f"Ranks 0-6 should have 6 sites each, got {site_counts[:7]}"
        assert site_counts[7] == 5, f"Rank 7 should have 5 sites, got {site_counts[7]}"

        # Step 5: Verify no site duplicated or missed
        assert sorted(all_sites) == list(range(L))
        assert len(all_sites) == len(set(all_sites))

    def test_verify_distribution_helper(self):
        """Test the verify_distribution helper function."""
        # Should pass for valid distributions
        assert verify_distribution(10, 3) is True
        assert verify_distribution(40, 8) is True
        assert verify_distribution(41, 8) is True
        assert verify_distribution(47, 8) is True
        assert verify_distribution(100, 16) is True

    def test_load_balancing_property(self):
        """
        Test load balancing property for multiple configurations.

        For any L and P, the difference between max and min sites
        per processor should be at most 1.
        """
        test_cases = [
            (10, 3), (40, 8), (41, 8), (47, 8),
            (100, 7), (100, 13), (100, 16),
            (5, 10),  # More procs than sites
            (1, 5),   # Single site
        ]

        for L, n_procs in test_cases:
            site_counts = [len(distribute_sites(L, n_procs, rank))
                           for rank in range(n_procs)]

            max_load = max(site_counts)
            min_load = min(site_counts)

            assert max_load - min_load <= 1, \
                f"Load imbalance for L={L}, P={n_procs}: " \
                f"max={max_load}, min={min_load}"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_site_single_proc(self):
        """Test minimal case: 1 site, 1 processor."""
        sites = distribute_sites(1, 1, 0)
        assert sites == [0]

    def test_two_sites_two_procs(self):
        """Test: 2 sites, 2 processors."""
        assert distribute_sites(2, 2, 0) == [0]
        assert distribute_sites(2, 2, 1) == [1]

    def test_invalid_L_negative(self):
        """Test error handling for negative L."""
        with pytest.raises(ValueError, match="Number of sites L must be positive"):
            distribute_sites(-1, 4, 0)

    def test_invalid_L_zero(self):
        """Test error handling for L=0."""
        with pytest.raises(ValueError, match="Number of sites L must be positive"):
            distribute_sites(0, 4, 0)

    def test_invalid_n_procs_negative(self):
        """Test error handling for negative n_procs."""
        with pytest.raises(ValueError, match="Number of processors must be positive"):
            distribute_sites(10, -1, 0)

    def test_invalid_n_procs_zero(self):
        """Test error handling for n_procs=0."""
        with pytest.raises(ValueError, match="Number of processors must be positive"):
            distribute_sites(10, 0, 0)

    def test_invalid_rank_negative(self):
        """Test error handling for negative rank."""
        with pytest.raises(ValueError, match="Rank must be in"):
            distribute_sites(10, 3, -1)

    def test_invalid_rank_too_large(self):
        """Test error handling for rank >= n_procs."""
        with pytest.raises(ValueError, match="Rank must be in"):
            distribute_sites(10, 3, 3)

        with pytest.raises(ValueError, match="Rank must be in"):
            distribute_sites(10, 3, 10)


class TestContiguity:
    """Test that site assignments are contiguous ranges."""

    def test_sites_are_contiguous(self):
        """Verify each processor gets a contiguous range of sites."""
        L, n_procs = 47, 8

        for rank in range(n_procs):
            sites = distribute_sites(L, n_procs, rank)

            if len(sites) > 0:
                # Sites should form a contiguous range
                expected = list(range(sites[0], sites[-1] + 1))
                assert sites == expected, \
                    f"Rank {rank} sites {sites} are not contiguous"

    def test_ranges_are_ordered(self):
        """Verify processor ranges are in ascending order without overlap."""
        L, n_procs = 40, 7

        prev_max = -1
        for rank in range(n_procs):
            sites = distribute_sites(L, n_procs, rank)

            if len(sites) > 0:
                # This rank's minimum should be exactly prev_max + 1
                assert sites[0] == prev_max + 1, \
                    f"Rank {rank} starts at {sites[0]}, " \
                    f"expected {prev_max + 1}"

                prev_max = sites[-1]

        # Final check: last site should be L-1
        assert prev_max == L - 1, \
            f"Last assigned site is {prev_max}, expected {L-1}"


class TestNumericalProperties:
    """Test numerical properties of the distribution."""

    def test_total_sites_equals_L(self):
        """Verify sum of all site counts equals L."""
        test_cases = [(10, 3), (40, 8), (47, 8), (100, 17)]

        for L, n_procs in test_cases:
            total_sites = sum(len(distribute_sites(L, n_procs, rank))
                              for rank in range(n_procs))

            assert total_sites == L, \
                f"Total sites {total_sites} != L={L} for P={n_procs}"

    def test_formula_consistency(self):
        """
        Verify the mathematical formula for distribution.

        For L = n_procs * q + r (0 <= r < n_procs):
        - First r processors get (q+1) sites each
        - Remaining (n_procs - r) processors get q sites each
        """
        test_cases = [(10, 3), (40, 8), (47, 8), (100, 17)]

        for L, n_procs in test_cases:
            q = L // n_procs
            r = L % n_procs

            for rank in range(n_procs):
                sites = distribute_sites(L, n_procs, rank)

                if rank < r:
                    expected_count = q + 1
                else:
                    expected_count = q

                assert len(sites) == expected_count, \
                    f"L={L}, P={n_procs}, rank={rank}: " \
                    f"expected {expected_count} sites, got {len(sites)}"
