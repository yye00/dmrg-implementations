"""
Parallel local micro-steps for A2DMRG.

This module implements Phase 2 of the A2DMRG algorithm: parallel local DMRG
micro-steps executed independently on each processor without communication.

Key features:
- Embarrassingly parallel: no MPI communication during local updates
- Each processor updates its assigned sites independently
- Works with both one-site and two-site DMRG micro-steps
- O(L) environment building (replicated on all ranks) instead of O(L²) per site
"""

import numpy as np
from typing import List, Dict, Tuple, Optional

from .distribute import distribute_sites


def parallel_local_microsteps(
    mps,
    mpo,
    comm,
    microstep_type="two_site",
    max_bond=None,
    cutoff=0.0,
    tol=1e-10,
):
    """Phase 2: parallel local micro-steps with O(L) environment caching.

    All ranks build environments from the same MPS (replicated, O(L) work).
    Each rank then solves its assigned sites using cached environments.
    No MPS copies, no per-site canonicalization.

    Parameters
    ----------
    mps : quimb.tensor.MatrixProductState
        Input MPS (broadcast to all processors beforehand).
    mpo : quimb MPO
        Matrix Product Operator (Hamiltonian).
    comm : MPI communicator
    microstep_type : str
        "one_site" or "two_site"
    max_bond : int or None
    cutoff : float
    tol : float

    Returns
    -------
    Dict[int, Tuple[list[ndarray], float]]
        Dictionary mapping site index to (candidate_arrays, energy).
        candidate_arrays is a list of numpy arrays in (chi_L, d, chi_R) format.
    """
    from a2dmrg.numerics.numpy_utils import extract_mps_arrays, extract_mpo_arrays
    from a2dmrg.environments.environment import build_environments_incremental
    from a2dmrg.numerics.local_microstep import fast_microstep_1site, fast_microstep_2site

    rank = comm.Get_rank()
    n_procs = comm.Get_size()
    L = mps.L

    # Extract to numpy (all ranks do this — MPS is already broadcast)
    mps_arrays = extract_mps_arrays(mps)
    mpo_arrays = extract_mpo_arrays(mpo)

    # Build all environments in O(L) — all ranks, replicated
    L_envs, R_envs, canon_arrays = build_environments_incremental(mps_arrays, mpo_arrays)

    # Distribute sites
    my_sites = distribute_sites(L, n_procs, rank)

    results = {}

    if microstep_type == "one_site":
        for site in my_sites:
            opt_tensor, eigval = fast_microstep_1site(
                canon_arrays[site], mpo_arrays[site],
                L_envs[site], R_envs[site + 1],
                site, L, tol=tol,
            )
            # Build candidate: use canonicalized arrays (not original mps_arrays!)
            candidate_arrays = [a.copy() for a in canon_arrays]
            candidate_arrays[site] = opt_tensor
            results[site] = (candidate_arrays, eigval)

    elif microstep_type == "two_site":
        for site in my_sites:
            if site >= L - 1:
                continue
            U, SVh, eigval = fast_microstep_2site(
                canon_arrays[site], canon_arrays[site + 1],
                mpo_arrays[site], mpo_arrays[site + 1],
                L_envs[site], R_envs[site + 2],
                site, L, max_bond=max_bond, cutoff=cutoff, tol=tol,
            )
            # Use canon_arrays as base (gauge consistency)
            candidate_arrays = [a.copy() for a in canon_arrays]
            candidate_arrays[site] = U
            candidate_arrays[site + 1] = SVh
            results[site] = (candidate_arrays, eigval)
    else:
        raise ValueError(f"Unknown microstep_type: {microstep_type}")

    return results


def gather_local_results(local_results, comm):
    """Gather results from all ranks. Results are (numpy_arrays, energy) tuples."""
    all_results_list = comm.allgather(local_results)
    all_results = {}
    for results_from_rank in all_results_list:
        all_results.update(results_from_rank)
    return all_results


def prepare_candidate_mps_list(mps, all_local_results):
    """Prepare candidate list as numpy array lists.

    Returns list of list[ndarray] where first is the original.
    """
    from a2dmrg.numerics.numpy_utils import extract_mps_arrays
    original_arrays = extract_mps_arrays(mps)
    candidates = [original_arrays]

    for site_idx in sorted(all_local_results.keys()):
        candidate_arrays, _ = all_local_results[site_idx]
        candidates.append(candidate_arrays)

    return candidates
