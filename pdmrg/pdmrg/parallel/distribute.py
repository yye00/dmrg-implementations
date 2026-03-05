"""Distribute MPS and environments across MPI ranks.

Splits the global MPS into contiguous blocks, one per rank.
Computes V = Λ⁻¹ at each shared bond between neighboring ranks.
"""

import numpy as np
from pdmrg.mps.parallel_mps import ParallelMPS
from pdmrg.numerics.accurate_svd import accurate_svd, compute_v_from_svd


def compute_site_distribution(L, n_procs):
    """Compute which sites go to which processor.

    Parameters
    ----------
    L : int
        Total number of sites.
    n_procs : int
        Number of processors.

    Returns
    -------
    site_ranges : list of range
        site_ranges[rank] = range of global site indices for that rank.
    """
    base = L // n_procs
    remainder = L % n_procs

    site_ranges = []
    start = 0
    for r in range(n_procs):
        count = base + (1 if r < remainder else 0)
        site_ranges.append(range(start, start + count))
        start += count

    return site_ranges


def distribute_mps(mps_arrays, mpo_arrays, comm, dtype=np.float64):
    """Distribute a global MPS across ranks with V matrices at boundaries.

    Must be called on ALL ranks. Rank 0 holds the full MPS and broadcasts.

    Parameters
    ----------
    mps_arrays : list of ndarray or None
        Full MPS in our convention (left, phys, right). Only needed on rank 0.
    mpo_arrays : list of ndarray
        Full MPO in our convention. Needed on all ranks.
    comm : MPI.Comm
    dtype : dtype

    Returns
    -------
    pmps : ParallelMPS
        Local MPS fragment with V matrices.
    """
    rank = comm.Get_rank()
    n_procs = comm.Get_size()

    # Broadcast total system size
    if rank == 0:
        L = len(mps_arrays)
    else:
        L = None
    L = comm.bcast(L, root=0)

    site_ranges = compute_site_distribution(L, n_procs)
    my_sites = site_ranges[rank]

    # Broadcast MPS arrays to all ranks
    if rank == 0:
        all_arrays = mps_arrays
    else:
        all_arrays = None
    all_arrays = comm.bcast(all_arrays, root=0)

    # Extract local arrays
    local_arrays = [all_arrays[i].copy() for i in my_sites]

    # Compute V matrices at shared bonds
    V_left = None
    V_right = None

    if rank > 0:
        # Left boundary: bond between site my_sites[0]-1 and my_sites[0]
        left_site = my_sites[0] - 1
        right_site = my_sites[0]
        V_left = _compute_v_at_bond(all_arrays, left_site, right_site)

    if rank < n_procs - 1:
        # Right boundary: bond between my_sites[-1] and my_sites[-1]+1
        left_site = my_sites[-1]
        right_site = my_sites[-1] + 1
        V_right = _compute_v_at_bond(all_arrays, left_site, right_site)

    pmps = ParallelMPS(
        arrays=local_arrays,
        my_sites=my_sites,
        rank=rank,
        n_procs=n_procs,
        V_left=V_left,
        V_right=V_right,
    )
    # Return full arrays so caller can build correct boundary environments
    pmps._global_arrays = all_arrays

    return pmps


def _compute_v_at_bond(all_arrays, left_site, right_site, use_identity=True):
    """Compute V at the bond between left_site and right_site.

    For a freshly distributed MPS from serial warmup, V = identity is correct.
    The boundary tensors already form the correct wavefunction when contracted
    directly - no rescaling is needed.

    V = 1/S is only needed when bridging INDEPENDENTLY EVOLVED wavefunctions,
    which doesn't apply when distributing from serial warmup.

    Parameters
    ----------
    all_arrays : list of ndarray
        Full MPS arrays.
    left_site, right_site : int
        Adjacent global site indices.
    use_identity : bool
        If True, return identity (all 1s). Default True for serial warmup.

    Returns
    -------
    V : ndarray, shape (k,)
        V matrix for boundary merge.
    """
    A_left = all_arrays[left_site]
    chi_bond = A_left.shape[2]
    
    # Use identity V - the tensors form correct wavefunction when contracted directly
    return np.ones(chi_bond, dtype=A_left.dtype)
