"""
Site distribution for parallel A2DMRG.

This module handles the distribution of MPS sites across processors for
parallel execution of local DMRG micro-steps.
"""

from typing import List


def distribute_sites(L: int, n_procs: int, rank: int) -> List[int]:
    """
    Distribute L sites across n_procs processors using balanced partitioning.

    Uses a load-balanced distribution where processors with lower ranks handle
    one extra site if L is not evenly divisible by n_procs. This ensures that
    the difference in workload between any two processors is at most 1 site.

    Algorithm:
    - If L = n_procs * q + r (where 0 <= r < n_procs)
    - Ranks 0 to r-1 each get (q+1) sites
    - Ranks r to n_procs-1 each get q sites

    Example:
        L=10, n_procs=3:
        - rank 0: [0, 1, 2, 3]  (4 sites)
        - rank 1: [4, 5, 6]     (3 sites)
        - rank 2: [7, 8, 9]     (3 sites)

    Parameters
    ----------
    L : int
        Total number of MPS sites to distribute.
    n_procs : int
        Number of MPI processors.
    rank : int
        MPI rank of current processor (0 <= rank < n_procs).

    Returns
    -------
    List[int]
        List of site indices assigned to this processor.

    Raises
    ------
    ValueError
        If L <= 0, n_procs <= 0, or rank is out of range.

    Notes
    -----
    This function does NOT require MPI to be initialized - it's a pure
    computational function that can be tested without mpirun.

    References
    ----------
    Grigori & Hassan (2025), Algorithm 2: A2DMRG with site distribution.
    """
    # Validate inputs
    if L <= 0:
        raise ValueError(f"Number of sites L must be positive, got {L}")
    if n_procs <= 0:
        raise ValueError(f"Number of processors must be positive, got {n_procs}")
    if rank < 0 or rank >= n_procs:
        raise ValueError(f"Rank must be in [0, {n_procs}), got {rank}")

    # Compute base number of sites per processor and remainder
    sites_per_proc = L // n_procs
    remainder = L % n_procs

    # Compute starting index and count for this rank
    if rank < remainder:
        # First 'remainder' ranks get one extra site
        start = rank * (sites_per_proc + 1)
        count = sites_per_proc + 1
    else:
        # Remaining ranks get base number of sites
        start = remainder * (sites_per_proc + 1) + (rank - remainder) * sites_per_proc
        count = sites_per_proc

    # Return list of site indices
    return list(range(start, start + count))


def verify_distribution(L: int, n_procs: int) -> bool:
    """
    Verify that site distribution covers all sites exactly once.

    This is a helper function for testing that checks:
    - All sites 0 to L-1 are covered
    - No site appears more than once
    - Load is balanced (max_sites - min_sites <= 1)

    Parameters
    ----------
    L : int
        Total number of sites.
    n_procs : int
        Number of processors.

    Returns
    -------
    bool
        True if distribution is valid and balanced, False otherwise.

    Examples
    --------
    >>> verify_distribution(10, 3)
    True
    >>> verify_distribution(40, 8)
    True
    """
    # Get all site assignments
    all_sites = []
    site_counts = []

    for rank in range(n_procs):
        sites = distribute_sites(L, n_procs, rank)
        all_sites.extend(sites)
        site_counts.append(len(sites))

    # Check 1: All sites covered
    all_sites_sorted = sorted(all_sites)
    expected_sites = list(range(L))
    if all_sites_sorted != expected_sites:
        return False

    # Check 2: Load balanced (difference at most 1)
    if max(site_counts) - min(site_counts) > 1:
        return False

    return True
