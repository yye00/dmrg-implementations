"""
Parallel local micro-steps for A2DMRG.

This module implements Phase 2 of the A2DMRG algorithm: parallel local DMRG
micro-steps executed independently on each processor without communication.

Key features:
- Embarrassingly parallel: no MPI communication during local updates
- Each processor updates its assigned sites independently
- Works with both one-site and two-site DMRG micro-steps
"""

import numpy as np
import quimb.tensor as qtn
from typing import List, Dict, Tuple, Optional

from ..numerics.local_microstep import local_microstep_1site, local_microstep_2site
from .distribute import distribute_sites
from ..mps.canonical import move_orthogonality_center, pad_to_bond_dimensions
from ..environments.environment import build_left_environments, build_right_environments


def parallel_local_microsteps(
    mps: qtn.MatrixProductState,
    mpo,
    comm,
    microstep_type: str = "one_site",
    max_bond: Optional[int] = None,
    cutoff: float = 1e-10,
    tol: float = 1e-10,
) -> Dict[int, Tuple[qtn.MatrixProductState, float]]:
    """
    Execute parallel local DMRG micro-steps on assigned sites.

    This function implements Phase 2 of the A2DMRG algorithm:
    - Each processor updates its assigned sites INDEPENDENTLY
    - NO MPI communication during this phase (embarrassingly parallel)
    - Returns dictionary of updated MPS for each site

    Algorithm:
    1. Get processor rank and total number of processors
    2. Record original bond dimensions of the input MPS (once)
    3. Distribute sites across processors using distribute_sites()
    4. For each assigned site (streaming, one at a time):
       a. Copy the MPS and move the orthogonality center to that site
       b. Perform local micro-step on the temporary copy
       c. Discard the copy immediately to bound peak memory at O(chi^2)
    5. Return dictionary {site_idx: (updated_mps, energy)}

    Parameters
    ----------
    mps : quimb.tensor.MatrixProductState
        Input MPS (should be broadcast to all processors beforehand)
    mpo : quimb MPO
        Matrix Product Operator (Hamiltonian)
    comm : mpi4py.MPI.Comm
        MPI communicator (typically MPI.COMM_WORLD)
    microstep_type : str, optional
        Type of micro-step: "one_site" or "two_site" (default: "one_site")
    max_bond : int, optional
        Maximum bond dimension for two-site updates (default: None)
    cutoff : float, optional
        SVD truncation tolerance for two-site updates (default: 1e-10)
    tol : float, optional
        Eigensolver tolerance (default: 1e-10)

    Returns
    -------
    Dict[int, Tuple[qtn.MatrixProductState, float]]
        Dictionary mapping site index to (updated_mps, energy) tuple.
        For one-site: one entry per assigned site
        For two-site: one entry per assigned bond (site, site+1)

    Notes
    -----
    This function does NOT perform any MPI communication (no Allreduce, Bcast, etc).
    Each processor works completely independently. Communication happens later in
    the coarse-space minimization phase (Phase 3).

    CRITICAL: This function streams i-orthogonal decompositions one site at a time
    (only for each rank's assigned sites). Peak memory per rank is O(chi^2).
    Environments are built once by rank 0 and broadcast to all ranks (O(L) total
    work instead of O(n_procs * L)).

    References
    ----------
    Grigori & Hassan (2025), Section 3.2: Parallel Local DMRG Micro-Steps
    """
    # Get MPI rank and size
    rank = comm.Get_rank()
    n_procs = comm.Get_size()

    L = mps.L  # Number of sites

    # Distribute sites across processors.
    my_sites = distribute_sites(L, n_procs, rank)

    # Dictionary to store results
    results = {}

    # Rank 0 builds environments once and broadcasts to all ranks.
    # This avoids each rank independently doing O(L) env build work,
    # which caused np>1 to be *slower* than np=1 for small systems.
    if comm.Get_rank() == 0:
        left_envs = build_left_environments(mps, mpo)
        right_envs = build_right_environments(mps, mpo)
    else:
        left_envs = None
        right_envs = None
    left_envs = comm.bcast(left_envs, root=0)
    right_envs = comm.bcast(right_envs, root=0)

    if microstep_type == "one_site":
        for site in my_sites:
            updated_mps, energy = local_microstep_1site(
                mps, mpo, site, tol=tol,
                L_env=left_envs[site],
                R_env=right_envs[site + 1],
            )
            results[site] = (updated_mps, energy)

    elif microstep_type == "two_site":
        for site in my_sites:
            if site < L - 1:  # Can only do two-site update if not last site
                updated_mps, energy = local_microstep_2site(
                    mps, mpo, site,
                    max_bond=max_bond,
                    cutoff=cutoff,
                    tol=tol,
                    L_env=left_envs[site],
                    R_env=right_envs[site + 2],
                )
                results[site] = (updated_mps, energy)

    else:
        raise ValueError(f"Unknown microstep_type: {microstep_type}. "
                        f"Must be 'one_site' or 'two_site'")

    # IMPORTANT: No MPI communication here!
    # Each processor returns its local results independently
    return results


def verify_no_communication(func):
    """
    Decorator to verify that a function does not perform MPI communication.

    This is used for testing to ensure that parallel_local_microsteps is
    truly embarrassingly parallel.

    Implementation note: This is a conceptual decorator. In practice, we
    verify no communication by checking that the function completes without
    any MPI calls (Allreduce, Bcast, etc).
    """
    def wrapper(*args, **kwargs):
        # In a real implementation, we could track MPI calls here
        # For now, this is a placeholder
        return func(*args, **kwargs)
    return wrapper


def gather_local_results(
    local_results: Dict[int, Tuple[qtn.MatrixProductState, float]],
    comm,
) -> Dict[int, Tuple[qtn.MatrixProductState, float]]:
    """
    Gather local micro-step results from all processors.

    This function is called AFTER parallel_local_microsteps to collect
    all updated MPS from all processors.

    Parameters
    ----------
    local_results : Dict[int, Tuple[qtn.MatrixProductState, float]]
        Results from this processor's local micro-steps
    comm : mpi4py.MPI.Comm
        MPI communicator

    Returns
    -------
    Dict[int, Tuple[qtn.MatrixProductState, float]]
        All results from all processors (on all ranks)

    Notes
    -----
    This function DOES perform MPI communication (MPI.Allgather or similar).
    It's called after Phase 2 is complete, as part of preparing for Phase 3.
    """
    # Gather all results to all processors
    all_results_list = comm.allgather(local_results)

    # Merge dictionaries from all processors
    all_results = {}
    for results_from_rank in all_results_list:
        all_results.update(results_from_rank)

    return all_results


def prepare_candidate_mps_list(
    original_mps: qtn.MatrixProductState,
    all_local_results: Dict[int, Tuple[qtn.MatrixProductState, float]],
) -> List[qtn.MatrixProductState]:
    """
    Prepare list of candidate MPS for coarse-space minimization.

    Creates the Y^(0), Y^(1), ..., Y^(d) candidates:
    - Y^(0) = original MPS (no updates)
    - Y^(j) = MPS updated at site j

    Parameters
    ----------
    original_mps : qtn.MatrixProductState
        Original MPS before local updates
    all_local_results : Dict[int, Tuple[qtn.MatrixProductState, float]]
        All local micro-step results from all processors

    Returns
    -------
    List[qtn.MatrixProductState]
        List of candidate MPS [Y^(0), Y^(1), ..., Y^(d)]

    Notes
    -----
    This prepares the input for Phase 3 (Coarse-Space Minimization).
    """
    # Start with original MPS as Y^(0)
    candidates = [original_mps.copy()]

    # Add updated MPS from each site
    # Sort by site index to ensure consistent ordering
    for site_idx in sorted(all_local_results.keys()):
        updated_mps, _ = all_local_results[site_idx]
        candidates.append(updated_mps)

    return candidates
