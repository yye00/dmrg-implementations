"""
Main A2DMRG algorithm entry point.

This module implements the complete Additive Two-Level Parallel DMRG algorithm
with 5 phases per iteration:

1. Initialization: Start with left-orthogonal tensor train
2. Prepare Orthogonal Decompositions: Create d i-orthogonal forms
3. Parallel Local Micro-Steps: Independent site updates (embarrassingly parallel)
4. Coarse-Space Minimization: Find optimal linear combination
5. Compression: Project back to target rank manifold

Usage:
    from mpi4py import MPI
    from a2dmrg.dmrg import a2dmrg_main
    from a2dmrg.hamiltonians import heisenberg_mpo

    comm = MPI.COMM_WORLD
    mpo = heisenberg_mpo(L=40, J=1.0, cyclic=False)
    energy, mps = a2dmrg_main(L=40, mpo=mpo, max_sweeps=20,
                              bond_dim=100, tol=1e-10, comm=comm)
"""

import argparse
import sys
import os
import json
import time
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import quimb.tensor as qtn
from a2dmrg.mpi_compat import MPI

from a2dmrg.mps.canonical import (
    left_canonicalize,
    prepare_orthogonal_decompositions,
    compress_mps,
)
from a2dmrg.mps.mps_utils import create_random_mps, create_neel_state, create_product_state_mps
from a2dmrg.parallel.local_steps import (
    parallel_local_microsteps,
    gather_local_results,
    prepare_candidate_mps_list,
)
from a2dmrg.parallel.coarse_space import build_coarse_matrices
from a2dmrg.numerics.coarse_eigenvalue import solve_coarse_eigenvalue_problem
from a2dmrg.parallel.linear_combination import form_linear_combination
from a2dmrg.numerics.observables import compute_energy


def a2dmrg_main(
    L: int,
    mpo,
    max_sweeps: int = 20,
    bond_dim: int = 100,
    tol: float = 1e-10,
    comm: Optional[MPI.Comm] = None,
    dtype=np.float64,
    one_site: bool = False,
    verbose: bool = True,
    initial_mps: Optional[qtn.MatrixProductState] = None,
    warmup_sweeps: int = 2,
    finalize_sweeps: int = 0,
    timing_report: bool = True,
    timing_dir: str = "reports",
    max_candidates: Optional[int] = None,
    coarse_reduction_tol: float = 1e-8,
    overlap_threshold: float = 0.99,
    experimental_nonpaper: bool = False,
    return_metadata: bool = False,
) -> Tuple[float, qtn.MatrixProductState]:
    """
    Main A2DMRG algorithm.

    DEFAULT CONFIGURATION (2026-03-07):
    - warmup_sweeps=2 for practical convergence
    - Matches PDMRG/PDMRG2 warmup configuration
    - Serial warmup improves initial state quality and convergence rate
    - Paper-faithful mode (warmup_sweeps=0) available via experimental_nonpaper flag

    The Grigori & Hassan paper uses random initialization with very small rank,
    relying on Lanczos reuse. For production use, 2 warmup sweeps provide better
    initial states without significantly increasing runtime.

    Parameters
    ----------
    L : int
        Number of sites
    mpo : quimb MPO
        Matrix Product Operator representing the Hamiltonian
    max_sweeps : int, optional
        Maximum number of A2DMRG sweeps (default: 20)
    bond_dim : int, optional
        Maximum bond dimension (default: 100)
    tol : float, optional
        Energy convergence tolerance (default: 1e-10)
    comm : MPI.Comm, optional
        MPI communicator (default: MPI.COMM_WORLD)
    dtype : numpy dtype, optional
        Data type (float64 or complex128, default: float64)
    one_site : bool, optional
        Use one-site updates instead of two-site (default: False)
    verbose : bool, optional
        Print progress information (default: True)
    initial_mps : quimb.tensor.MatrixProductState, optional
        Initial MPS to start from (default: None, creates product state).
        If provided, this MPS is used instead of creating a new one.
        Useful for warm-starting with a few serial DMRG sweeps.
    warmup_sweeps : int, optional
        Number of standard DMRG sweeps to run before A2DMRG (default: 0).
        Paper-faithful mode uses warmup_sweeps=0 (random init, no serial warmup).
        Allowed range: 0-2 (hard cap for benchmark hygiene).
        Values > 2 require experimental_nonpaper=True.
        WARNING: Serial warmup is a non-paper convenience mode.
    finalize_sweeps : int, optional
        Number of standard DMRG sweeps to run after A2DMRG (default: 0).
        These "polishing" sweeps can achieve machine precision accuracy.
        Workflow: warmup → A2DMRG (parallel) → finalize (serial, tight tol)
    timing_report : bool, optional
        If True, collect per-phase timings and write a JSON report on rank 0.
    timing_dir : str, optional
        Directory to write timing reports into (default: "reports").
    max_candidates : int, optional
        If set, keep at most this many coarse-space candidates (including Y^(0)).
        This reduces coarse-space cost from O(d^2) to O(k^2).
    experimental_nonpaper : bool, optional
        If True, allow warmup_sweeps > 2 (default: False).
        This explicitly marks non-paper-faithful execution mode.
        Use only for experimental comparison, not for benchmark results.

    Returns
    -------
    energy : float
        Ground state energy
    mps : quimb.tensor.MatrixProductState
        Ground state MPS

    Raises
    ------
    ValueError
        If input parameters are invalid (negative or zero values)
    """
    # Input validation
    if L <= 0:
        raise ValueError(f"L must be positive, got {L}")
    if bond_dim <= 0:
        raise ValueError(f"bond_dim must be positive, got {bond_dim}")
    if max_sweeps <= 0:
        raise ValueError(f"max_sweeps must be positive, got {max_sweeps}")
    if tol <= 0:
        raise ValueError(f"tol must be positive, got {tol}")
    if dtype not in (np.float64, np.complex128):
        raise ValueError(f"dtype must be np.float64 or np.complex128, got {dtype}")

    if comm is None:
        comm = MPI.COMM_WORLD

    rank = comm.rank
    size = comm.size

    # Validate np >= 2 (A2DMRG is a parallel algorithm)
    if size < 2:
        raise ValueError(
            f"A2DMRG requires at least 2 MPI ranks (got np={size}). "
            "A2DMRG is a parallel algorithm based on additive subspace correction. "
            "For serial execution, use quimb.DMRG2 instead."
        )

    # Validate warmup policy (2026-03-07: warmup_sweeps=2 default)
    if warmup_sweeps < 0:
        raise ValueError(f"warmup_sweeps must be non-negative, got {warmup_sweeps}")

    if warmup_sweeps > 5 and not experimental_nonpaper:
        raise ValueError(
            f"warmup_sweeps={warmup_sweeps} exceeds recommended bound (≤5). "
            "For very high warmup_sweeps, set experimental_nonpaper=True to acknowledge "
            "this is outside typical usage."
        )

    # Optional verbose notifications
    if rank == 0 and verbose:
        if warmup_sweeps == 0:
            print(f"⚠️  Paper-faithful mode: warmup_sweeps=0 (random initialization)")
        elif warmup_sweeps > 5:
            print(f"⚠️  Experimental mode: warmup_sweeps={warmup_sweeps} (experimental_nonpaper=True)")

    # Timing collection (rank-local; reduced when reporting)
    timing_enabled = bool(timing_report)
    # Determine initialization mode for metadata
    if initial_mps is not None:
        init_mode = "provided_initial_mps"
        paper_faithful = False  # Custom init is not paper-faithful
    elif warmup_sweeps == 0:
        init_mode = "paper_faithful_random"
        paper_faithful = True
    else:
        init_mode = f"warmup_{warmup_sweeps}_sweeps"
        paper_faithful = (warmup_sweeps <= 2)  # 1-2 sweeps considered reasonable

    timing = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "L": L,
            "bond_dim": bond_dim,
            "max_sweeps": max_sweeps,
            "tol": tol,
            "dtype": str(dtype),
            "np": size,
            "one_site": bool(one_site),
            "warmup_sweeps": warmup_sweeps,
            "finalize_sweeps": finalize_sweeps,
            "max_candidates": max_candidates,
            "coarse_reduction_tol": coarse_reduction_tol,
            "overlap_threshold": overlap_threshold,
            "initialization_mode": init_mode,
            "paper_faithful_mode": paper_faithful,
            "experimental_nonpaper": experimental_nonpaper,
        },
        "sweeps": [],
    }

    if rank == 0 and verbose:
        print(f"A2DMRG: L={L}, bond_dim={bond_dim}, np={size}, dtype={dtype.__name__}", flush=True)
        print(f"Algorithm: {'one-site' if one_site else 'two-site'} updates", flush=True)
        print(f"Target convergence: {tol}", flush=True)

    # Phase 0: Initialization - Create or use provided MPS
    # Using product state instead of random for better convergence
    # (recommended in paper: "warm start with a few serial DMRG sweeps or product state")
    if rank == 0 and verbose:
        print("\n=== Phase 0: Initialization ===", flush=True)

    # Create initial MPS on rank 0 and broadcast to all
    if rank == 0:
        # Use provided initial_mps if available, otherwise create new MPS
        if initial_mps is not None:
            mps = initial_mps.copy()
            if verbose:
                print(f"Using provided initial MPS (warm start)", flush=True)
                print(f"MPS: L={mps.L}, bond_dims={[mps[i].shape for i in range(mps.L)]}", flush=True)
        else:
            # Detect physical dimension from MPO
            first_tensor = mpo.tensors[0]
            if first_tensor.ndim == 3:
                # Edge site: (bond, ket, bra)
                phys_dim = first_tensor.shape[1]
            elif first_tensor.ndim == 4:
                # Middle site: (left_bond, right_bond, ket, bra)
                phys_dim = first_tensor.shape[2]
            else:
                # Fallback: assume spin-1/2
                phys_dim = 2

            # For spin-1/2 (phys_dim=2), use Neel state which has negative energy for Heisenberg
            # For other systems (bosons, etc.), use vacuum state |0,0,...,0⟩
            if phys_dim == 2:
                mps = create_neel_state(L, bond_dim=bond_dim, dtype=dtype)
                if verbose:
                    print(f"Created Neel state MPS with L={L}, bond_dim={bond_dim}", flush=True)
                    print(f"Neel state: |↑↓↑↓...⟩ (alternating spins)", flush=True)
            else:
                # Use vacuum state for bosonic systems
                mps = create_product_state_mps(L, bond_dim=bond_dim, state_index=0,
                                              phys_dim=phys_dim, dtype=dtype)
                if verbose:
                    print(f"Created product state MPS with L={L}, bond_dim={bond_dim}", flush=True)
                    print(f"Physical dimension: {phys_dim}", flush=True)
                    print(f"Product state: |0,0,...,0⟩ (vacuum state)", flush=True)

            # Product state is already a valid state with uniform bond dimensions
            # after padding inside create function
            if verbose:
                print(f"MPS is in product state form with uniform bond dimensions", flush=True)
    else:
        mps = None

    # Broadcast MPS to all ranks
    # Note: quimb MPS needs manual broadcast of tensor data
    if size > 1:
        if rank == 0:
            # Prepare data for broadcast: list of (tensor data, shape, inds, tags)
            mps_data = []
            site_tag_id = mps.site_tag_id
            for i in range(L):
                t = mps.tensors[i]
                mps_data.append((t.data, t.shape, t.inds, list(t.tags)))
            bcast_info = (mps_data, site_tag_id)
        else:
            bcast_info = None

        bcast_info = comm.bcast(bcast_info, root=0)
        mps_data, site_tag_id = bcast_info

        if rank != 0:
            # Reconstruct MPS from broadcasted data with proper index ordering and tags
            tensors = []
            for data, shape, inds, tags in mps_data:
                tensors.append(qtn.Tensor(data=data, inds=inds, tags=tags))
            tn = qtn.TensorNetwork(tensors)
            mps = qtn.MatrixProductState.from_TN(
                tn, 
                site_tag_id=site_tag_id,
                site_ind_id='k{}',
                cyclic=False,
                L=L
            )

    # Compute initial energy (before warm-up)
    energy_init = compute_energy(mps, mpo)
    if rank == 0 and verbose:
        print(f"Initial energy: {energy_init:.12f}", flush=True)

    # Warm-up phase: Run standard DMRG sweeps to establish entanglement structure
    # This is CRITICAL for A2DMRG to work correctly.
    # Per Grigori & Hassan (arXiv:2505.23429), the algorithm requires initialization
    # "sufficiently close to the true minimizer". Product states (e.g., Neel) don't
    # satisfy this and cause the local eigenvectors to be orthogonal to the MPS.
    if warmup_sweeps > 0 and initial_mps is None:
        if rank == 0:
            if verbose:
                print(f"\n=== Warm-up Phase: {warmup_sweeps} standard DMRG sweeps ===", flush=True)
            
            # Use quimb's DMRG2 for warm-up (two-site for entanglement growth)
            from quimb.tensor import DMRG2
            dmrg_warmup = DMRG2(mpo, bond_dims=bond_dim, cutoffs=1e-14)
            
            # Run warmup sweeps with tight tolerance for accurate initialization
            # A2DMRG requires starting close to the true minimum
            dmrg_warmup.solve(max_sweeps=warmup_sweeps, tol=1e-12, verbosity=1 if verbose else 0)
            
            # Get the warmed-up MPS
            mps = dmrg_warmup._k.copy()
            energy_warmup = float(np.real(dmrg_warmup.energy))
            
            if verbose:
                print(f"Warm-up complete: E = {energy_warmup:.12f}", flush=True)
                print(f"Energy improvement: {energy_init - energy_warmup:.6f}", flush=True)
        
        # Broadcast warmed-up MPS to all ranks
        if size > 1:
            if rank == 0:
                mps_data = []
                site_tag_id = mps.site_tag_id
                for i in range(L):
                    t = mps.tensors[i]
                    mps_data.append((t.data, t.shape, t.inds, list(t.tags)))
                bcast_info = (mps_data, site_tag_id)
            else:
                bcast_info = None
            
            bcast_info = comm.bcast(bcast_info, root=0)
            mps_data, site_tag_id = bcast_info
            
            if rank != 0:
                # Reconstruct MPS with proper index ordering and original tags
                tensors = []
                for data, shape, inds, tags in mps_data:
                    tensors.append(qtn.Tensor(data=data, inds=inds, tags=tags))
                tn = qtn.TensorNetwork(tensors)
                mps = qtn.MatrixProductState.from_TN(
                    tn, 
                    site_tag_id=site_tag_id,
                    site_ind_id='k{}',
                    cyclic=False,
                    L=L
                )
    
    # Compute energy after warm-up (or initial energy if no warm-up)
    energy_prev = compute_energy(mps, mpo)
    if rank == 0 and verbose:
        if warmup_sweeps > 0 and initial_mps is None:
            print(f"\nStarting A2DMRG from warm-up state: E = {energy_prev:.12f}", flush=True)
        else:
            print(f"Starting A2DMRG: E = {energy_prev:.12f}", flush=True)

    # Metadata tracking
    start_time = time.time()
    converged_flag = False
    final_sweep_num = 0

    # Main iteration loop
    for sweep in range(max_sweeps):
        final_sweep_num = sweep + 1
        sweep_t0 = time.perf_counter()
        phase_times = {}

        def _tmark(name: str, t0: float):
            phase_times[name] = time.perf_counter() - t0

        if rank == 0 and verbose:
            print(f"\n=== Sweep {sweep + 1}/{max_sweeps} ===", flush=True)

        # Phase 1: Ensure MPS is left-canonical
        # NOTE: For A2DMRG, we do NOT left-canonicalize at each sweep!
        # Canonicalization can reduce bond dimensions, making MPS incompatible
        # for linear combination in the coarse-space method.
        # We rely on the initial MPS being properly initialized and
        # the compression phase (Phase 4) to maintain numerical stability.
        if rank == 0 and verbose:
            print("Phase 1: Skipping canonicalization (not needed for A2DMRG)...", flush=True)

        # left_canonicalize(mps, normalize=True)  # DISABLED for A2DMRG

        # Extract MPO arrays once per sweep for numpy pipeline
        from a2dmrg.numerics.numpy_utils import extract_mpo_arrays
        mpo_arrays = extract_mpo_arrays(mpo)

        # Phase 2: Parallel local micro-steps
        if rank == 0 and verbose:
            print("Phase 2: Performing parallel local micro-steps...", flush=True)

        # Perform local updates (embarrassingly parallel)
        microstep_t0 = time.perf_counter()
        microstep_type = "one_site" if one_site else "two_site"
        local_results = parallel_local_microsteps(
            mps,
            mpo,
            comm,
            microstep_type=microstep_type,
            max_bond=bond_dim,
            cutoff=tol,
            tol=tol
        )

        # Gather results from all processors
        all_results = gather_local_results(local_results, comm)
        _tmark("phase2_local_microsteps", microstep_t0)

        if rank == 0 and verbose:
            print(f"Completed {len(all_results)} local micro-steps", flush=True)

        # Phase 3: Coarse-space minimization
        if rank == 0 and verbose:
            print("Phase 3: Coarse-space minimization...", flush=True)

        # Prepare candidate MPS list: [original MPS] + [updated MPS from each site]
        cand_t0 = time.perf_counter()

        # Candidate reduction (CPU scaling): keep only the best-k local updates by microstep energy.
        # Always keep Y^(0) (the current state).
        if max_candidates is not None and max_candidates >= 2:
            # all_results: site -> (updated_mps, energy)
            # Select k-1 lowest-energy updates.
            sorted_sites = sorted(all_results.keys(), key=lambda s: all_results[s][1])
            keep_sites = sorted_sites[: max_candidates - 1]
            # Broadcast keep list for determinism.
            if size > 1:
                keep_sites = comm.bcast(keep_sites if rank == 0 else None, root=0)
            reduced_results = {s: all_results[s] for s in keep_sites}
            candidate_mps_list = prepare_candidate_mps_list(mps, reduced_results)
            if timing_enabled and rank == 0:
                timing.setdefault("candidate_selection", []).append({
                    "sweep": sweep + 1,
                    "total_local_updates": int(len(all_results)),
                    "kept_local_updates": int(len(keep_sites)),
                    "coarse_dim": int(len(candidate_mps_list)),
                })
        else:
            candidate_mps_list = prepare_candidate_mps_list(mps, all_results)

        _tmark("phase3_prepare_candidates", cand_t0)

        # Build coarse-space matrices
        #
        # Obvious performance issue (fixed): previously every rank computed the *entire* coarse matrix,
        # duplicating work. We now:
        #   1) Filter redundant candidates on rank 0 and broadcast the retained indices
        #   2) Split coarse-matrix *rows* across ranks and Allreduce
        coarse_t0 = time.perf_counter()
        if size > 1:
            from a2dmrg.parallel.coarse_space import filter_redundant_candidates

            if rank == 0:
                original_count = len(candidate_mps_list)
                filtered_candidates, retained_indices = filter_redundant_candidates(
                    candidate_mps_list,
                    overlap_threshold=overlap_threshold,
                )
            else:
                retained_indices = None

            retained_indices = comm.bcast(retained_indices, root=0)
            candidate_mps_list = [candidate_mps_list[i] for i in retained_indices]

            if rank == 0 and verbose and len(filtered_candidates) < original_count:
                print(
                    f"  Filtered redundant candidates: {original_count} → {len(filtered_candidates)}",
                    flush=True,
                )

            # Distribute coarse-matrix rows (excluding row 0, which rank 0 always does)
            d_plus_1 = len(candidate_mps_list)
            assigned_sites = [
                (i - 1)
                for i in range(1, d_plus_1)
                if (i % size) == rank
            ]

            H_coarse, S_coarse = build_coarse_matrices(
                candidate_mps_list,
                mpo_arrays,
                comm=comm,
                assigned_sites=assigned_sites,
                filter_redundant=False,
            )
        else:
            # Serial: filter then build matrices
            from a2dmrg.parallel.coarse_space import filter_redundant_candidates

            original_count = len(candidate_mps_list)
            candidate_mps_list, _ = filter_redundant_candidates(
                candidate_mps_list,
                overlap_threshold=overlap_threshold,
            )
            if verbose and len(candidate_mps_list) < original_count:
                print(
                    f"  Filtered redundant candidates: {original_count} → {len(candidate_mps_list)}",
                    flush=True,
                )

            H_coarse, S_coarse = build_coarse_matrices(
                candidate_mps_list,
                mpo_arrays,
                comm=None,
                filter_redundant=False,
            )

        _tmark("phase3_build_coarse_matrices", coarse_t0)

        # Solve coarse eigenvalue problem on rank 0
        n_effective_coarse = None
        coarse_solve_t0 = time.perf_counter()
        if rank == 0:
            energy_new, coeffs, coarse_diag = solve_coarse_eigenvalue_problem(
                H_coarse,
                S_coarse,
                return_all=False,
                regularization=coarse_reduction_tol,
                return_diagnostics=True,
            )
            n_effective_coarse = coarse_diag["n_effective"]
            if verbose:
                print(f"Coarse-space energy: {energy_new:.12f}", flush=True)
                print(f"Energy change: {energy_new - energy_prev:.3e}", flush=True)
                print(f"  Coarse dim: {len(candidate_mps_list)} candidates → {n_effective_coarse} effective (S-spectrum)", flush=True)
        else:
            coeffs = None
            energy_new = None

        # Broadcast coefficients to all ranks
        if size > 1:
            coeffs = comm.bcast(coeffs, root=0)
            energy_new = comm.bcast(energy_new, root=0)
            n_effective_coarse = comm.bcast(n_effective_coarse, root=0)

        _tmark("phase3_solve_and_bcast", coarse_solve_t0)

        # Phase 4: Form linear combination
        if rank == 0 and verbose:
            print("Phase 4: Forming linear combination...", flush=True)

        lincomb_t0 = time.perf_counter()
        combined_mps = form_linear_combination(candidate_mps_list, coeffs)
        _tmark("phase4_form_linear_combination", lincomb_t0)

        # Phase 5: Compression
        if rank == 0 and verbose:
            print("Phase 5: Compressing MPS...", flush=True)

        # Phase 5: Compression using numpy TT-SVD
        compress_t0 = time.perf_counter()
        from a2dmrg.numerics.compression import tt_svd_compress
        from a2dmrg.numerics.observables import compute_energy_numpy
        from a2dmrg.numerics.numpy_utils import arrays_to_quimb_mps

        compressed_arrays = tt_svd_compress(combined_mps, max_bond=bond_dim, normalize=True)
        _tmark("phase5_compress", compress_t0)

        # Compute actual energy after compression
        ecomp_t0 = time.perf_counter()
        energy_after_compression = compute_energy_numpy(compressed_arrays, mpo_arrays)
        _tmark("phase5_energy_after_compression", ecomp_t0)

        # Convert back to quimb MPS for next sweep iteration
        mps = arrays_to_quimb_mps(compressed_arrays)

        # Check convergence using ACTUAL compressed energies, not coarse-space energies
        # This is critical because compression changes the energy!
        energy_diff = abs(energy_after_compression - energy_prev)

        if rank == 0 and verbose:
            print(f"Energy after compression: {energy_after_compression:.12f}", flush=True)
            print(f"Convergence check: |ΔE| = {energy_diff:.3e} (target: {tol:.3e})", flush=True)
            print(f"Coarse-space energy was: {energy_new:.12f}", flush=True)
            print(f"Compression changed energy by: {energy_after_compression - energy_new:+.3e}", flush=True)

        if energy_diff < tol:
            # Record timings for final sweep as well
            _tmark("sweep_total", sweep_t0)
            if timing_enabled:
                phase_summary = {"sweep": sweep + 1}
                for k, v in phase_times.items():
                    if size > 1:
                        t_sum = comm.allreduce(v, op=MPI.SUM)
                        t_max = comm.allreduce(v, op=MPI.MAX)
                        phase_summary[k] = {"mean": t_sum / size, "max": t_max}
                    else:
                        phase_summary[k] = {"mean": v, "max": v}
                if rank == 0 and n_effective_coarse is not None:
                    phase_summary["n_coarse_candidates"] = int(len(candidate_mps_list))
                    phase_summary["n_effective_coarse"] = int(n_effective_coarse)
                if rank == 0:
                    timing["sweeps"].append(phase_summary)

            if rank == 0 and verbose:
                print(f"\nConverged after {sweep + 1} sweeps!", flush=True)
            converged_flag = True
            break

        # Update energy_prev to the ACTUAL compressed energy (not coarse-space energy)
        energy_prev = energy_after_compression

        # Record timings (reduce across ranks so rank 0 can report max/mean)
        _tmark("sweep_total", sweep_t0)
        if timing_enabled:
            # reduce each phase time
            phase_summary = {"sweep": sweep + 1}
            for k, v in phase_times.items():
                if size > 1:
                    t_sum = comm.allreduce(v, op=MPI.SUM)
                    t_max = comm.allreduce(v, op=MPI.MAX)
                    phase_summary[k] = {"mean": t_sum / size, "max": t_max}
                else:
                    phase_summary[k] = {"mean": v, "max": v}
            if rank == 0 and n_effective_coarse is not None:
                phase_summary["n_coarse_candidates"] = int(len(candidate_mps_list))
                phase_summary["n_effective_coarse"] = int(n_effective_coarse)
            if rank == 0:
                timing["sweeps"].append(phase_summary)

    # Final compression to target bond dimension
    if rank == 0 and verbose:
        print(f"\n=== Final Compression to bond_dim={bond_dim} ===", flush=True)

    # Final compression using numpy TT-SVD (consistent with sweep compression)
    from a2dmrg.numerics.numpy_utils import extract_mps_arrays, arrays_to_quimb_mps
    from a2dmrg.numerics.compression import tt_svd_compress
    final_arrays = extract_mps_arrays(mps)
    final_arrays = tt_svd_compress(final_arrays, max_bond=bond_dim, normalize=True)
    mps = arrays_to_quimb_mps(final_arrays)

    # Compute energy after A2DMRG (before finalization)
    energy_after_a2dmrg = compute_energy(mps, mpo)

    if rank == 0 and verbose:
        print(f"Energy after A2DMRG: {energy_after_a2dmrg:.12f}", flush=True)

    # Finalization phase: Run standard DMRG sweeps to polish to machine precision
    # This is done on rank 0 only (serial DMRG)
    if finalize_sweeps > 0:
        if rank == 0:
            if verbose:
                print(f"\n=== Finalization Phase: {finalize_sweeps} standard DMRG sweeps ===", flush=True)
            
            from quimb.tensor import DMRG2
            # Use p0 parameter to pass our MPS as the initial state.
            # This ensures proper index alignment between MPS and MPO
            # (replaces the broken _k/_b replacement approach).
            dmrg_finalize = DMRG2(mpo, bond_dims=bond_dim, p0=mps)
            dmrg_finalize.solve(max_sweeps=finalize_sweeps, tol=1e-12, verbosity=1 if verbose else 0)
            
            mps = dmrg_finalize._k.copy()
            final_energy = float(np.real(dmrg_finalize.energy))
            
            if verbose:
                print(f"Finalization complete: E = {final_energy:.12f}", flush=True)
                print(f"Improvement from finalization: {energy_after_a2dmrg - final_energy:.2e}", flush=True)
        
        # Broadcast finalized MPS to all ranks
        if size > 1:
            if rank == 0:
                mps_data = []
                site_tag_id = mps.site_tag_id
                for i in range(L):
                    t = mps.tensors[i]
                    mps_data.append((t.data, t.shape, t.inds, list(t.tags)))
                bcast_info = (mps_data, site_tag_id, final_energy)
            else:
                bcast_info = None
            
            bcast_info = comm.bcast(bcast_info, root=0)
            mps_data, site_tag_id, final_energy = bcast_info
            
            if rank != 0:
                tensors = []
                for data, shape, inds, tags in mps_data:
                    tensors.append(qtn.Tensor(data=data, inds=inds, tags=tags))
                tn = qtn.TensorNetwork(tensors)
                mps = qtn.MatrixProductState.from_TN(
                    tn, 
                    site_tag_id=site_tag_id,
                    site_ind_id='k{}',
                    cyclic=False,
                    L=L
                )
    else:
        # No finalization, just compute final energy
        final_energy = energy_after_a2dmrg

    if rank == 0 and verbose:
        print(f"\n=== A2DMRG Complete ===", flush=True)
        print(f"Final energy: {final_energy:.12f}", flush=True)
        print(f"Energy per site: {final_energy/L:.12f}", flush=True)

    # Write timing report (rank 0)
    if timing_enabled and rank == 0:
        os.makedirs(timing_dir, exist_ok=True)
        timing["meta"]["final_energy"] = float(final_energy)
        timing["meta"]["completed_sweeps"] = len(timing["sweeps"])
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        path = os.path.join(timing_dir, f"a2dmrg_timing_{stamp}.json")
        with open(path, "w") as f:
            json.dump(timing, f, indent=2)
        if verbose:
            print(f"Timing report written: {path}", flush=True)

    # Metadata for benchmarking and reproducibility
    metadata = {
        "algorithm_executed": "A2DMRG additive two-level parallel",
        "warmup_method": "quimb DMRG2 serial" if warmup_sweeps > 0 else None,
        "warmup_sweeps": warmup_sweeps,
        "experimental_nonpaper": experimental_nonpaper,
        "initialization_mode": init_mode,
        "paper_faithful": paper_faithful,
        "converged": converged_flag,
        "final_sweep": final_sweep_num,
        "np": size,
        "max_sweeps": max_sweeps,
        "bond_dim": bond_dim,
        "tol": tol,
        "total_time": time.time() - start_time,
    }

    if return_metadata:
        return final_energy, mps, metadata
    return final_energy, mps


def main():
    """Command-line interface for A2DMRG."""
    parser = argparse.ArgumentParser(
        description="A2DMRG: Additive Two-Level Parallel DMRG",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--sites", type=int, default=40,
                        help="Number of sites")
    parser.add_argument("--bond-dim", type=int, default=100,
                        help="Maximum bond dimension")
    parser.add_argument("--sweeps", type=int, default=20,
                        help="Maximum number of sweeps")
    parser.add_argument("--tol", type=float, default=1e-10,
                        help="Energy convergence tolerance")
    parser.add_argument("--dtype", choices=["float64", "complex128"],
                        default="float64",
                        help="Data type")
    parser.add_argument("--model", choices=["heisenberg", "josephson"],
                        default="heisenberg",
                        help="Hamiltonian model")
    parser.add_argument("--one-site", action="store_true",
                        help="Use one-site updates instead of two-site")
    parser.add_argument("--timing", action="store_true",
                        help="Report timing information")

    args = parser.parse_args()

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.rank

    # Convert dtype string to numpy dtype
    dtype = np.float64 if args.dtype == "float64" else np.complex128

    # Create Hamiltonian (placeholder)
    if rank == 0:
        print(f"Creating {args.model} Hamiltonian...")

    # TODO: Import and use actual Hamiltonian constructors
    # from a2dmrg.hamiltonians import heisenberg_mpo, bose_hubbard_mpo
    # if args.model == "heisenberg":
    #     mpo = heisenberg_mpo(L=args.sites, J=1.0, cyclic=False)
    # else:
    #     mpo = bose_hubbard_mpo(L=args.sites, t=1.0, U=4.0, mu=2.0, dtype=dtype)

    if rank == 0:
        print("Error: Hamiltonians not yet implemented")
        sys.exit(1)

    # Run A2DMRG
    # energy, mps = a2dmrg_main(
    #     L=args.sites,
    #     mpo=mpo,
    #     max_sweeps=args.sweeps,
    #     bond_dim=args.bond_dim,
    #     tol=args.tol,
    #     comm=comm,
    #     dtype=dtype,
    #     one_site=args.one_site,
    #     verbose=True,
    # )

    # if rank == 0:
    #     print(f"\nGround state energy: {energy:.12f}")
    #     print(f"Energy per site: {energy/args.sites:.12f}")


if __name__ == "__main__":
    main()
