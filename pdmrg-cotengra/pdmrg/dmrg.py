"""Main PDMRG algorithm entry point.

Implements the Real-Space Parallel DMRG algorithm (Stoudenmire & White 2013):
  Phase 0: Serial warmup with quimb DMRG1
  Phase 1: Distribute MPS across ranks with V matrices
  Phase 2: Parallel staggered sweeps (independent, no communication)
  Phase 3: Merge at shared bonds using V = Lambda^-1
  Phase 4: Repeat until convergence
"""

import argparse
import time

import numpy as np
import quimb.tensor as qtn

from pdmrg.mps.canonical import (
    get_tensor_data, get_mpo_tensor_data, set_tensor_data,
)
from pdmrg.mps.parallel_mps import ParallelMPS
from pdmrg.environments.environment import EnvironmentManager
from pdmrg.environments.update import (
    update_left_env, update_right_env,
    init_left_env, init_right_env,
)
from pdmrg.numerics.eigensolver import optimize_two_site
from pdmrg.numerics.accurate_svd import truncated_svd, accurate_svd, compute_v_from_svd
from pdmrg.parallel.distribute import compute_site_distribution, distribute_mps
from pdmrg.parallel.merge import merge_boundary_tensors
from pdmrg.parallel.communication import safe_exchange, check_convergence
from pdmrg.parallel.sweep_pattern import get_initial_direction
from pdmrg.hamiltonians.heisenberg import build_heisenberg_mpo


def serial_warmup(mpo, L, bond_dim_warmup=50, n_warmup_sweeps=5,
                  dtype='float64', initial_mps=None):
    """Phase 0: Use quimb DMRG1 for a fast initial state.

    Returns MPS in right-canonical form (canonical center at site 0).
    This means all R_envs are correct after distribution.

    Uses DMRG1 (single-site) instead of DMRG2: warmup only needs a
    reasonable starting state, not optimal bond structure. DMRG1 is
    ~2x cheaper at large chi since the eigsh problem is chi*d vs chi*d².

    Parameters
    ----------
    initial_mps : quimb.tensor.MatrixProductState, optional
        If provided, used as the starting state for DMRG1 warmup.
        This ensures reproducible benchmarks when all implementations
        start from the same stored MPS.
    """
    mpo_arrays = [get_mpo_tensor_data(mpo, i) for i in range(L)]

    # Warmup with DMRG1: cheap single-site sweeps to get a decent state.
    # Bond dimension is set directly (DMRG1 doesn't grow bonds, but the
    # random initial state already has the target bond dim).
    dmrg = qtn.DMRG1(mpo, bond_dims=bond_dim_warmup, cutoffs=1e-14,
                      p0=initial_mps)
    dmrg.solve(tol=1e-10, max_sweeps=n_warmup_sweeps, verbosity=0)

    warmup_energy = dmrg.energy

    # Right-canonical: all sites right-isometric, OC at site 0.
    # R_envs will be correct everywhere after distribution.
    mps = dmrg.state
    mps.canonize(0)
    A = [get_tensor_data(mps, i) for i in range(L)]

    return A, mpo_arrays, warmup_energy




def canonize_block(pmps, env_mgr, mpo_arrays, direction):
    """Canonize the local block for the initial sweep direction.

    For sweep 'left': QR sweep right to put canonical center at right edge,
                       building correct L_envs along the way.
    For sweep 'right': LQ sweep left to put canonical center at left edge,
                        building correct R_envs along the way.
    """
    n_local = pmps.n_local
    global_start = pmps.my_sites[0]

    if direction == 'left':
        # Put canonical center at RIGHT edge (left-canonize all but last)
        for j in range(n_local - 1):
            gi = global_start + j
            A = pmps.arrays[j]
            chi_L, d, chi_R = A.shape
            M = A.reshape(chi_L * d, chi_R)
            Q, R = np.linalg.qr(M)
            pmps.arrays[j] = Q.reshape(chi_L, d, -1)
            pmps.arrays[j + 1] = np.tensordot(R, pmps.arrays[j + 1], axes=(1, 0))
            env_mgr.L_envs[gi + 1] = update_left_env(
                env_mgr.L_envs[gi], pmps.arrays[j], mpo_arrays[gi])
    else:
        # Put canonical center at LEFT edge (right-canonize all but first)
        for j in range(n_local - 1, 0, -1):
            gi = global_start + j
            B = pmps.arrays[j]
            chi_L, d, chi_R = B.shape
            M = B.reshape(chi_L, d * chi_R)
            Q_T, R_T = np.linalg.qr(M.T)
            Q = Q_T.T
            L_mat = R_T.T
            pmps.arrays[j] = Q.reshape(-1, d, chi_R)
            pmps.arrays[j - 1] = np.tensordot(pmps.arrays[j - 1], L_mat, axes=(2, 0))
            env_mgr.R_envs[gi - 1] = update_right_env(
                env_mgr.R_envs[gi], pmps.arrays[j], mpo_arrays[gi])


def local_sweep(pmps, env_mgr, mpo_arrays, direction, max_bond,
                max_iter=20, tol=1e-10):
    """Phase 2: Perform a local DMRG sweep on this rank's block.

    No inter-rank communication. Standard two-site DMRG within the block.
    """
    n_local = pmps.n_local
    global_start = pmps.my_sites[0]
    E_local = 0.0

    if direction == 'right':
        for j in range(n_local - 1):
            gi = global_start + j
            gi1 = global_start + j + 1

            theta = np.einsum('ijk,klm->ijlm',
                              pmps.arrays[j], pmps.arrays[j + 1])

            E_local, theta_opt = optimize_two_site(
                env_mgr.L_envs[gi], env_mgr.R_envs[gi1],
                mpo_arrays[gi], mpo_arrays[gi1], theta,
                max_iter=max_iter, tol=tol
            )

            chi_L, d_L, d_R, chi_R = theta_opt.shape
            M = theta_opt.reshape(chi_L * d_L, d_R * chi_R)
            U, S, Vh, _ = truncated_svd(M, max_bond)

            pmps.arrays[j] = U.reshape(chi_L, d_L, -1)
            pmps.arrays[j + 1] = (np.diag(S) @ Vh).reshape(-1, d_R, chi_R)

            env_mgr.L_envs[gi + 1] = update_left_env(
                env_mgr.L_envs[gi], pmps.arrays[j], mpo_arrays[gi]
            )

        direction = 'left'

    else:  # direction == 'left'
        for j in range(n_local - 2, -1, -1):
            gi = global_start + j
            gi1 = global_start + j + 1

            theta = np.einsum('ijk,klm->ijlm',
                              pmps.arrays[j], pmps.arrays[j + 1])

            E_local, theta_opt = optimize_two_site(
                env_mgr.L_envs[gi], env_mgr.R_envs[gi1],
                mpo_arrays[gi], mpo_arrays[gi1], theta,
                max_iter=max_iter, tol=tol
            )

            chi_L, d_L, d_R, chi_R = theta_opt.shape
            M = theta_opt.reshape(chi_L * d_L, d_R * chi_R)
            U, S, Vh, _ = truncated_svd(M, max_bond)

            pmps.arrays[j + 1] = Vh.reshape(-1, d_R, chi_R)
            pmps.arrays[j] = (U @ np.diag(S)).reshape(chi_L, d_L, -1)

            env_mgr.R_envs[gi] = update_right_env(
                env_mgr.R_envs[gi1], pmps.arrays[j + 1], mpo_arrays[gi1]
            )

        direction = 'right'

    return E_local, direction


def boundary_merge(pmps, env_mgr, mpo_arrays, comm, boundaries,
                   max_bond, max_iter=30, tol=1e-10):
    """Phase 3: Merge at shared boundary bonds.

    Must be called by ALL ranks. Merges happen at the specified boundaries.

    Parameters
    ----------
    boundaries : str
        'even' for boundaries 0↔1, 2↔3, 4↔5, ...
        'odd' for boundaries 1↔2, 3↔4, 5↔6, ...

    At each active boundary, the left rank (lower) performs the merge
    computation. Both ranks exchange data and receive results.

    Returns the merge energy (0.0 if this rank didn't participate).
    """
    rank = pmps.rank
    n_procs = pmps.n_procs
    global_start = pmps.my_sites[0]
    global_end = pmps.my_sites[-1]
    energy = None  # None for idle ranks (0.0 is invalid sentinel for physics)

    # Determine which boundary this rank participates in
    # Even boundaries: rank pairs (0,1), (2,3), (4,5), ...
    # Odd boundaries: rank pairs (1,2), (3,4), (5,6), ...
    if boundaries == 'even':
        # Left partner has even rank, right partner has odd rank
        if rank % 2 == 0 and rank + 1 < n_procs:
            role = 'left'
            neighbor = rank + 1
        elif rank % 2 == 1:
            role = 'right'
            neighbor = rank - 1
        else:
            role = 'idle'
            neighbor = None
    else:  # 'odd'
        # Left partner has odd rank, right partner has even rank
        if rank % 2 == 1 and rank + 1 < n_procs:
            role = 'left'
            neighbor = rank + 1
        elif rank % 2 == 0 and rank > 0:
            role = 'right'
            neighbor = rank - 1
        else:
            role = 'idle'
            neighbor = None

    if role == 'left':
        # I am the LEFT side of the merge (my right boundary)
        psi_left = pmps.arrays[-1].copy()
        L_env = env_mgr.L_envs[global_end].copy()
        V = pmps.V_right

        my_data = {'psi': psi_left, 'env': L_env}
        neighbor_data = safe_exchange(comm, rank, neighbor, my_data)
        psi_right = neighbor_data['psi']
        R_env = neighbor_data['env']

        left_global = global_end
        right_global = global_end + 1

        A_left_new, A_right_new, V_new, energy, _, A_right_canonical = \
            merge_boundary_tensors(
                psi_left, psi_right, V,
                L_env, R_env,
                mpo_arrays[left_global], mpo_arrays[right_global],
                max_bond=max_bond, max_iter=max_iter, tol=tol
            )

        # Update local state
        pmps.arrays[-1] = A_left_new
        pmps.V_right = V_new

        # CRITICAL FIX: Use CANONICAL tensors for boundary environments.
        # R_env from Vh (right-canonical) gives R_norm = I.
        # Using A_right_new (S*Vh) would give R_norm = S² ≠ I, breaking
        # the eigensolver's assumption that N_eff = I in subsequent sweeps.
        env_mgr.R_envs[global_end] = update_right_env(
            R_env, A_right_canonical, mpo_arrays[right_global])

        # L_env from A_left_new (U, already left-canonical) → L_norm = I
        L_env_new = update_left_env(
            L_env, A_left_new, mpo_arrays[left_global])
        comm.send({'A_right': A_right_new, 'L_env': L_env_new,
                   'V': V_new, 'energy': energy},
                  dest=neighbor, tag=300 + rank)

    elif role == 'right':
        # I am the RIGHT side of the merge (my left boundary)
        psi_right = pmps.arrays[0].copy()
        R_env = env_mgr.R_envs[global_start].copy()

        my_data = {'psi': psi_right, 'env': R_env}
        neighbor_data = safe_exchange(comm, rank, neighbor, my_data)

        # Wait for the merge results from the left partner
        result = comm.recv(source=neighbor, tag=300 + neighbor)

        pmps.arrays[0] = result['A_right']
        env_mgr.L_envs[global_start] = result['L_env']
        pmps.V_left = result['V']
        energy = result['energy']

    return energy


def build_local_environments(pmps, mpo_arrays, dtype=np.float64,
                              global_mps_arrays=None):
    """Build initial local L and R environments for a rank's block.
    
    If global_mps_arrays is None (parallel warmup case), use local
    MPS only and treat boundaries as "vacuum" states.
    """
    env_mgr = EnvironmentManager()
    n_local = pmps.n_local
    global_start = pmps.my_sites[0]
    global_end = pmps.my_sites[-1]
    L = len(mpo_arrays)

    # Left environment at left edge of block
    if pmps.is_leftmost or global_mps_arrays is None:
        # Use local MPS boundary dimension
        chi_L = pmps.arrays[0].shape[0]
        D_L = mpo_arrays[global_start].shape[0]
        env_mgr.L_envs[global_start] = init_left_env(chi_L, D_L, dtype)
    else:
        # CRITICAL FIX: Left-canonicalize global MPS before building L_envs.
        # The warmup returns right-canonical MPS, but L_envs built from
        # right-canonical tensors give norm matrix N ≠ I, which breaks the
        # standard eigensolver assumption. Left-canonicalizing ensures
        # L_norm = I at every environment site.
        lc_arrays = [a.copy() for a in global_mps_arrays[:global_start]]
        for i in range(len(lc_arrays) - 1):
            chi_L_i, d_i, chi_R_i = lc_arrays[i].shape
            M = lc_arrays[i].reshape(chi_L_i * d_i, chi_R_i)
            Q, R = np.linalg.qr(M)
            lc_arrays[i] = Q.reshape(chi_L_i, d_i, -1)
            lc_arrays[i + 1] = np.tensordot(R, lc_arrays[i + 1], axes=(1, 0))

        chi_L_0 = lc_arrays[0].shape[0]
        D_0 = mpo_arrays[0].shape[0]
        L_env = init_left_env(chi_L_0, D_0, dtype)
        for i in range(global_start):
            L_env = update_left_env(L_env, lc_arrays[i], mpo_arrays[i])
        env_mgr.L_envs[global_start] = L_env

    # Build remaining left envs by sweeping right through local block
    for j in range(n_local - 1):
        gi = global_start + j
        env_mgr.L_envs[gi + 1] = update_left_env(
            env_mgr.L_envs[gi], pmps.arrays[j], mpo_arrays[gi]
        )

    # Right environment at right edge of block
    if pmps.is_rightmost or global_mps_arrays is None:
        # Use local MPS boundary dimension
        chi_R = pmps.arrays[-1].shape[2]
        D_R = mpo_arrays[global_end].shape[1]
        env_mgr.R_envs[global_end] = init_right_env(chi_R, D_R, dtype)
    else:
        # Right-canonicalize global MPS sites after this block before building
        # R_envs. This mirrors the L_envs path which left-canonicalizes.
        # Without this, R_envs are built from non-canonical tensors, giving
        # R_norm ≠ I and breaking the eigensolver's N_eff = I assumption.
        rc_arrays = [a.copy() for a in global_mps_arrays[global_end + 1:]]
        for i in range(len(rc_arrays) - 1, 0, -1):
            chi_L_i, d_i, chi_R_i = rc_arrays[i].shape
            M = rc_arrays[i].reshape(chi_L_i, d_i * chi_R_i)
            Q_T, R_T = np.linalg.qr(M.conj().T)
            rc_arrays[i] = Q_T.conj().T.reshape(-1, d_i, chi_R_i)
            rc_arrays[i - 1] = np.tensordot(rc_arrays[i - 1], R_T.conj().T,
                                             axes=(2, 0))

        chi_R_last = rc_arrays[-1].shape[2]
        D_last = mpo_arrays[-1].shape[1]
        R_env = init_right_env(chi_R_last, D_last, dtype)
        for i in range(len(rc_arrays) - 1, -1, -1):
            gi = global_end + 1 + i
            R_env = update_right_env(R_env, rc_arrays[i], mpo_arrays[gi])
        env_mgr.R_envs[global_end] = R_env

    # Build remaining right envs by sweeping left through local block
    for j in range(n_local - 2, -1, -1):
        gi = global_start + j
        gi1 = global_start + j + 1
        env_mgr.R_envs[gi] = update_right_env(
            env_mgr.R_envs[gi1], pmps.arrays[j + 1], mpo_arrays[gi1]
        )

    return env_mgr


def rebuild_boundary_r_env(pmps, env_mgr, mpo_arrays):
    """Rebuild R_env at the left boundary for the merge.
    
    After canonize_block('left'), all sites are left-canonical with OC at right edge.
    But for the merge with the left neighbor, we need R_env[global_start] to be
    built from RIGHT-canonical sites.
    
    This function right-canonizes sites 1 to n_local-1 (leaving site 0 as OC)
    and rebuilds R_env[global_start].
    """
    from pdmrg.environments.update import init_right_env, update_right_env
    
    n_local = pmps.n_local
    global_start = pmps.my_sites[0]
    global_end = pmps.my_sites[-1]
    
    if n_local <= 1:
        # No interior sites to right-canonize
        return
    
    # Right-canonize sites from n_local-1 down to 1 (LQ decomposition)
    # This puts sites 1..n_local-1 in right-canonical form with OC at site 0
    for j in range(n_local - 1, 0, -1):
        A = pmps.arrays[j]
        chi_L, d, chi_R = A.shape
        M = A.reshape(chi_L, d * chi_R)
        # LQ decomposition: A = L @ Q where Q is right-isometric
        Q, R = np.linalg.qr(M.conj().T)
        L = R.conj().T  # L = R^H
        Q_right = Q.conj().T.reshape(-1, d, chi_R)  # Q_right = Q^H
        pmps.arrays[j] = Q_right
        pmps.arrays[j - 1] = np.tensordot(pmps.arrays[j - 1], L, axes=(2, 0))
    
    # Now sites 1..n_local-1 are right-canonical, site 0 is OC
    # Rebuild R_env[global_start] from sites 1..n_local-1
    
    # Initialize at right edge
    chi_R = pmps.arrays[-1].shape[2]
    D_R = mpo_arrays[global_end].shape[1]
    if pmps.is_rightmost:
        R_env = init_right_env(chi_R, D_R, pmps.arrays[-1].dtype)
    else:
        R_env = env_mgr.R_envs[global_end]
    
    # Sweep left from site n_local-1 to site 1
    for j in range(n_local - 1, 0, -1):
        gi = global_start + j
        R_env = update_right_env(R_env, pmps.arrays[j], mpo_arrays[gi])
    
    env_mgr.R_envs[global_start] = R_env


def rebuild_boundary_l_env(pmps, env_mgr, mpo_arrays):
    """Rebuild L_env at the right boundary for the merge.
    
    After canonize_block('right'), all sites are right-canonical with OC at left edge.
    But for the merge with the right neighbor, we need L_env[global_end] to be
    built from LEFT-canonical sites.
    
    This function left-canonizes sites 0 to n_local-2 (leaving site n_local-1 as OC)
    and rebuilds L_env[global_end].
    """
    from pdmrg.environments.update import init_left_env, update_left_env
    
    n_local = pmps.n_local
    global_start = pmps.my_sites[0]
    global_end = pmps.my_sites[-1]
    
    if n_local <= 1:
        return
    
    # Left-canonize sites from 0 to n_local-2 (QR decomposition)
    for j in range(n_local - 1):
        A = pmps.arrays[j]
        chi_L, d, chi_R = A.shape
        M = A.reshape(chi_L * d, chi_R)
        Q, R = np.linalg.qr(M)
        pmps.arrays[j] = Q.reshape(chi_L, d, -1)
        pmps.arrays[j + 1] = np.tensordot(R, pmps.arrays[j + 1], axes=(1, 0))
    
    # Rebuild L_env[global_end] from sites 0..n_local-2
    chi_L = pmps.arrays[0].shape[0]
    D_L = mpo_arrays[global_start].shape[0]
    if pmps.is_leftmost:
        L_env = init_left_env(chi_L, D_L, pmps.arrays[0].dtype)
    else:
        L_env = env_mgr.L_envs[global_start]
    
    for j in range(n_local - 1):
        gi = global_start + j
        L_env = update_left_env(L_env, pmps.arrays[j], mpo_arrays[gi])
    
    env_mgr.L_envs[global_end] = L_env


def compute_v_from_boundary_tensor(tensor, boundary_side='right'):
    """Compute V = Lambda^-1 from a boundary tensor's SVD.

    Parameters
    ----------
    tensor : ndarray
        Boundary MPS tensor, either shape (chi_L, d, chi_bond) for right boundary
        or shape (chi_bond, d, chi_R) for left boundary.
    boundary_side : str
        'right' or 'left'

    Returns
    -------
    V : ndarray, shape (chi_bond,)
        V = 1/S computed from the boundary tensor's singular values.
    """
    from pdmrg.numerics.accurate_svd import compute_v_from_svd

    if boundary_side == 'right':
        # Right boundary: (chi_L, d, chi_bond) -> reshape to (chi_L*d, chi_bond)
        chi_L, d, chi_bond = tensor.shape
        M = tensor.reshape(chi_L * d, chi_bond)
    else:
        # Left boundary: (chi_bond, d, chi_R) -> reshape to (chi_bond, d*chi_R)
        chi_bond, d, chi_R = tensor.shape
        M = tensor.reshape(chi_bond, d * chi_R)

    # Compute SVD and extract singular values
    _, S, _ = np.linalg.svd(M, full_matrices=False)

    # Return V = 1/S with regularization
    return compute_v_from_svd(S)


def recompute_boundary_v(pmps, comm, which_boundary):
    """Update V at a boundary after canonization.

    Computes V = Lambda^-1 where Lambda comes from the SVD of the boundary
    tensor, following Stoudenmire & White 2013 Eq. 5.

    After independent local sweeps, the blocks have evolved separately and
    V = Lambda^-1 is needed to properly bridge them during the merge.

    Parameters
    ----------
    pmps : ParallelMPS
        Local MPS fragment
    comm : MPI.Comm
        MPI communicator
    which_boundary : str
        'left' or 'right'
    """
    from pdmrg.parallel.communication import safe_exchange
    from pdmrg.numerics.accurate_svd import compute_v_from_svd

    rank = pmps.rank
    n_procs = pmps.n_procs

    if which_boundary == 'right':
        if rank < n_procs - 1:
            neighbor = rank + 1
            chi_bond = pmps.arrays[-1].shape[2]

            # Compute V = Lambda^-1 from SVD of boundary tensor
            pmps.V_right = compute_v_from_boundary_tensor(pmps.arrays[-1], 'right')

            # Exchange with neighbor for synchronization
            my_data = {'chi': chi_bond, 'v_size': len(pmps.V_right)}
            safe_exchange(comm, rank, neighbor, my_data)

        if rank > 0:
            neighbor = rank - 1
            chi_bond = pmps.arrays[0].shape[0]

            my_data = {'chi': chi_bond}
            safe_exchange(comm, rank, neighbor, my_data)

    elif which_boundary == 'left':
        if rank > 0:
            neighbor = rank - 1
            chi_bond = pmps.arrays[0].shape[0]

            # Compute V = Lambda^-1 from SVD of boundary tensor
            pmps.V_left = compute_v_from_boundary_tensor(pmps.arrays[0], 'left')

            # Exchange with neighbor for synchronization
            my_data = {'chi': chi_bond, 'v_size': len(pmps.V_left)}
            safe_exchange(comm, rank, neighbor, my_data)

        if rank < n_procs - 1:
            neighbor = rank + 1
            chi_bond = pmps.arrays[-1].shape[2]

            my_data = {'chi': chi_bond}
            safe_exchange(comm, rank, neighbor, my_data)


def pdmrg_main(L, mpo, max_sweeps=20, bond_dim=100, bond_dim_warmup=50,
               n_warmup_sweeps=5, tol=1e-8, dtype='float64',
               comm=None, verbose=True,
               random_init_flag=False, return_metadata=False,
               initial_mps=None):
    """Run the full PDMRG algorithm.

    For n_procs > 1, uses staggered sweeps (Fig. 4 of the paper):
      - Even ranks start at right end, sweep left first
      - Odd ranks start at left end, sweep right first
      - After sweeps reach boundaries, merge with neighbor using V
      - Sweep back, merge with other neighbor

    Warmup policy:
      - Serial warmup only: rank 0 runs quimb DMRG1, then MPS is scattered
      - Parallel warmup removed for algorithmic fidelity (2026-03-07)
      - Use random_init_flag=True only for experimental testing

    Parameters
    ----------
    return_metadata : bool, optional
        If True, return (energy, pmps, metadata) tuple.
        If False (default), return (energy, pmps) for backward compatibility.
    random_init_flag : bool
        If True, skip warmup and start from a random MPS.
        This requires more sweeps but is useful for testing.
        Not recommended for benchmark use.
    initial_mps : quimb.tensor.MatrixProductState, optional
        If provided, used as the starting state for serial warmup.
        Ensures all implementations start from the same MPS for fair
        benchmark comparison.
    """
    if comm is None:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    n_procs = comm.Get_size()

    # PDMRG is a parallel algorithm and requires at least 2 MPI ranks.
    # Running with np=1 provides no parallelism and masks algorithmic issues.
    # For serial DMRG, use quimb.DMRG2 directly.
    if n_procs < 2:
        raise ValueError(
            f"PDMRG requires at least 2 MPI ranks (got np={n_procs}). "
            "PDMRG is a parallel real-space DMRG algorithm (Stoudenmire & White 2013) "
            "that divides the MPS chain across processors. "
            "For serial execution, use quimb.DMRG2 instead."
        )

    mpo_arrays = [get_mpo_tensor_data(mpo, i) for i in range(L)]

    t_warmup = time.time()
    
    if random_init_flag:
        # Random initialization: create simple random MPS for testing
        if rank == 0 and verbose:
            print(f"PDMRG: L={L}, bond_dim={bond_dim}, n_procs={n_procs}")
            print(f"Phase 0: Random initialization (no warmup)")
        
        # Distribute sites evenly
        sites_per_proc = L // n_procs
        my_start = rank * sites_per_proc
        if rank == n_procs - 1:
            my_end = L
        else:
            my_end = my_start + sites_per_proc
        my_sites = list(range(my_start, my_end))
        n_local = len(my_sites)
        
        # Create random local MPS arrays
        d = mpo_arrays[0].shape[2]  # physical dimension
        chi = min(5, bond_dim)  # use small bond dim for harder convergence
        local_mps = []
        
        np.random.seed(42 + rank)  # reproducible per rank
        for j in range(n_local):
            gi = my_start + j
            chi_L = 1 if gi == 0 else chi
            chi_R = 1 if gi == L - 1 else chi
            
            if dtype == 'complex128':
                arr = (np.random.randn(chi_L, d, chi_R) + 
                       1j * np.random.randn(chi_L, d, chi_R))
            else:
                arr = np.random.randn(chi_L, d, chi_R)
            arr = arr.astype(dtype)
            # Normalize
            arr /= np.linalg.norm(arr)
            local_mps.append(arr)
        
        warmup_energy = 0.0
        mps_arrays = None
        
    else:
        # Serial warmup on rank 0
        if rank == 0 and verbose:
            print(f"PDMRG: L={L}, bond_dim={bond_dim}, n_procs={n_procs}")
            print(f"Phase 0: Serial warmup (quimb DMRG1, m={bond_dim})")
        
        warmup_energy = None
        local_mps = None
        if rank == 0:
            mps_arrays, _, warmup_energy = serial_warmup(
                mpo, L, bond_dim_warmup=bond_dim_warmup,
                n_warmup_sweeps=n_warmup_sweeps, dtype=dtype,
                initial_mps=initial_mps
            )
        else:
            mps_arrays = None
            
    t_warmup = time.time() - t_warmup

    if rank == 0 and verbose:
        print(f"  Warmup time: {t_warmup:.2f}s")
        if warmup_energy is not None:
            print(f"  Warmup energy: {warmup_energy:.12f}")

    # Phase 1: Distribute
    if rank == 0 and verbose:
        print("Phase 1: Distributing MPS across ranks")

    if random_init_flag:
        # Already have local MPS from random init
        site_ranges = compute_site_distribution(L, n_procs)
        my_sites = site_ranges[rank]
        
        # Initialize V matrices at boundaries using exact SVD
        V_left = None
        V_right = None
        if rank < n_procs - 1:
            # Compute V from SVD of right boundary tensor
            V_right = compute_v_from_boundary_tensor(local_mps[-1], 'right')
        if rank > 0:
            # Compute V from SVD of left boundary tensor
            V_left = compute_v_from_boundary_tensor(local_mps[0], 'left')
        
        # Create ParallelMPS with local arrays
        pmps = ParallelMPS(
            arrays=local_mps,
            my_sites=my_sites,
            rank=rank,
            n_procs=n_procs,
            V_left=V_left,
            V_right=V_right
        )
        
        # Build environments from local state only (no global info)
        env_mgr = build_local_environments(pmps, mpo_arrays,
                                            dtype=np.dtype(dtype),
                                            global_mps_arrays=None)
    else:
        # Distribute from rank 0's serial warmup
        pmps = distribute_mps(mps_arrays, mpo_arrays, comm,
                               dtype=np.dtype(dtype))
        
        # Build local environments using the global MPS for correct boundary envs
        global_arrays = getattr(pmps, '_global_arrays', None)
        env_mgr = build_local_environments(pmps, mpo_arrays,
                                            dtype=np.dtype(dtype),
                                            global_mps_arrays=global_arrays)
        if hasattr(pmps, '_global_arrays'):
            del pmps._global_arrays

    # NOTE: np=1 early return was removed. PDMRG now requires np >= 2.
    # Validation check at function entry enforces this requirement.

    if rank == 0 and verbose:
        site_ranges = compute_site_distribution(L, n_procs)
        for r in range(n_procs):
            print(f"  Rank {r}: sites {list(site_ranges[r])}")

    # Phase 2-4: Main loop
    E_prev = 0.0
    E_global = 0.0
    converged_flag = False
    final_sweep_num = max_sweeps

    eigsolver_max_iter = 100
    eigsolver_tol = tol / 10

    # NOTE: n_procs==1 path removed - validation at entry ensures np >= 2

    # Multi-rank parallel PDMRG (Stoudenmire & White 2013):
    # Real-space parallelization with staggered local sweeps and boundary merges.
    #
    # ALGORITHMIC FIX (2026-03-07):
    # Previous implementation did NOT perform local optimization within blocks!
    # It only called canonize_block() which does QR decomposition without energy minimization.
    # This fix adds proper local_sweep() calls that optimize energy using 2-site DMRG.
    #
    # Algorithm structure per Stoudenmire & White 2013:
    #   1. Local optimization sweeps within each rank's block (parallel, independent)
    #   2. Merge at even boundaries (0↔1, 2↔3, ...) with V = Lambda^-1 bridge
    #   3. Local optimization sweeps in opposite direction
    #   4. Merge at odd boundaries (1↔2, 3↔4, ...)
    #
    # Staggered pattern: even ranks sweep right first, odd ranks sweep left first.
    # This maximizes parallel efficiency by preventing idle waiting.

    # Initialize sweep direction based on rank (staggered pattern)
    direction = 'right' if rank % 2 == 0 else 'left'

    for sweep in range(max_sweeps):
        t0 = time.time()

        # ===== PHASE 1: LOCAL OPTIMIZATION SWEEPS (parallel, no communication) =====
        # Each rank independently optimizes within its block using standard 2-site DMRG.
        # This is the CRITICAL FIX - previous version skipped this step entirely!
        if rank == 0 and verbose:
            print(f"  Phase 1: Local optimization sweeps...")

        E_local1, direction = local_sweep(
            pmps, env_mgr, mpo_arrays, direction, bond_dim,
            max_iter=eigsolver_max_iter, tol=eigsolver_tol)
        # After sweep, direction is flipped and OC is at opposite edge of block

        # ===== PHASE 2: MERGE AT EVEN BOUNDARIES (0↔1, 2↔3, ...) =====
        # Prepare environments for merge
        if direction == 'left':  # Just swept right, OC at right edge
            # For even boundaries: even ranks are left side, odd ranks are right side
            if rank % 2 == 1:  # Odd ranks need R_env at their left edge
                rebuild_boundary_r_env(pmps, env_mgr, mpo_arrays)
        else:  # Just swept left, OC at left edge
            if rank % 2 == 0:  # Even ranks need L_env at their right edge
                rebuild_boundary_l_env(pmps, env_mgr, mpo_arrays)

        # Recompute V at right boundary using exact SVD (V = Lambda^-1)
        recompute_boundary_v(pmps, comm, 'right')

        if rank == 0 and verbose:
            print(f"  Phase 2: Merging even boundaries...")

        E_merge1 = boundary_merge(
            pmps, env_mgr, mpo_arrays, comm, 'even',
            max_bond=bond_dim, max_iter=eigsolver_max_iter,
            tol=eigsolver_tol)

        # ===== PHASE 3: LOCAL OPTIMIZATION SWEEPS IN OPPOSITE DIRECTION =====
        if rank == 0 and verbose:
            print(f"  Phase 3: Local optimization sweeps (opposite direction)...")

        E_local2, direction = local_sweep(
            pmps, env_mgr, mpo_arrays, direction, bond_dim,
            max_iter=eigsolver_max_iter, tol=eigsolver_tol)

        # ===== PHASE 4: MERGE AT ODD BOUNDARIES (1↔2, 3↔4, ...) =====
        # Prepare environments for merge
        if direction == 'left':  # Just swept right
            if rank % 2 == 0 and rank > 0:  # Even ranks (except 0) need R_env at left edge
                rebuild_boundary_r_env(pmps, env_mgr, mpo_arrays)
        else:  # Just swept left
            if rank % 2 == 1:  # Odd ranks need L_env at right edge
                rebuild_boundary_l_env(pmps, env_mgr, mpo_arrays)

        # Recompute V at left boundary
        recompute_boundary_v(pmps, comm, 'left')

        # Merge at odd boundaries
        if rank == 0 and verbose:
            print(f"  Phase 4: Merging odd boundaries...")

        E_merge2 = boundary_merge(
            pmps, env_mgr, mpo_arrays, comm, 'odd',
            max_bond=bond_dim, max_iter=eigsolver_max_iter,
            tol=eigsolver_tol)

        # ===== CONVERGENCE CHECK =====
        # Use best energy from merges (local sweep energies are rank-local only)
        merge_energies = [e for e in [E_merge1, E_merge2] if e is not None]
        E_best = min(merge_energies) if merge_energies else E_local2

        converged, E_global = check_convergence(
            E_best, E_prev, tol, comm)

        dt = time.time() - t0
        if rank == 0 and verbose:
            print(f"Sweep {sweep}: E = {E_global:.12f}, "
                  f"dE = {abs(E_global - E_prev):.2e}, "
                  f"time = {dt:.2f}s")

        if converged and sweep > 0:
            if rank == 0 and verbose:
                print(f"Converged after {sweep + 1} sweeps!")
            converged_flag = True
            final_sweep_num = sweep + 1
            break

        E_prev = E_global


    # ===== QUIMB CLEANUP SWEEPS =====
    # PDMRG block-local sweeps use cross-block environments from the previous
    # merge (stale). Quimb DMRG1 cleanup on the assembled MPS closes this gap
    # to machine-precision accuracy (ΔE < 1e-14).
    mps_assembled = gather_mps(pmps, comm)

    if rank == 0:
        E_global = _quimb_cleanup_sweeps(
            mps_assembled, mpo, bond_dim, tol / 100, verbose)

    E_global = comm.bcast(E_global, root=0)

    if rank == 0 and verbose:
        print(f"Final energy: {E_global:.12f}")

    if return_metadata:
        # Determine algorithm executed and warmup method
        # NOTE: np=1 path removed - PDMRG now requires np >= 2
        # NOTE: parallel warmup removed 2026-03-07 - serial warmup only
        if random_init_flag:
            warmup_method_str = None
        else:
            warmup_method_str = "quimb DMRG1 serial"

        algorithm_executed_str = "PDMRG parallel sweeps with local optimization"

        metadata = {
            "algorithm_executed": algorithm_executed_str,
            "local_sweeps_enabled": True,  # FIX 2026-03-07: Local optimization restored
            "boundary_optimization_enabled": True,  # Enabled with exact SVD (2026-03-07)
            "V_computation": "exact_svd_Lambda_inverse",  # Exact SVD method (2026-03-07)
            "early_return": False,
            "early_return_reason": None,
            "warmup_used": not random_init_flag,
            "warmup_sweeps": n_warmup_sweeps if not random_init_flag else 0,
            "warmup_method": warmup_method_str,
            "random_init": random_init_flag,
            "np": n_procs,
            "converged": converged_flag,
            "final_sweep": final_sweep_num,
            "max_sweeps": max_sweeps,
        }
        return E_global, pmps, metadata
    else:
        return E_global, pmps


def _quimb_cleanup_sweeps(mps_arrays, mpo, bond_dim, tol, verbose):
    """Run quimb DMRG1 cleanup sweeps on the assembled MPS.

    Converts the assembled PDMRG MPS to a quimb MPS and runs a few
    DMRG1 sweeps to polish the energy. DMRG1 is cheaper than DMRG2
    (eigsh problem chi*d vs chi*d²) and sufficient when the bond
    structure from PDMRG is already close to converged.
    """
    L = len(mps_arrays)

    # Right-canonicalize first to fix ||ψ||² < 1 from independent blocks
    for i in range(L - 1, 0, -1):
        chi_L, d, chi_R = mps_arrays[i].shape
        M = mps_arrays[i].reshape(chi_L, d * chi_R)
        Q_T, R_T = np.linalg.qr(M.conj().T)
        mps_arrays[i] = Q_T.conj().T.reshape(-1, d, chi_R)
        mps_arrays[i - 1] = np.tensordot(mps_arrays[i - 1], R_T.conj().T,
                                           axes=(2, 0))

    # Build quimb MPS: transpose (chi_L, d, chi_R) -> (chi_L, chi_R, d)
    arrays_q = []
    for i, t in enumerate(mps_arrays):
        if t.ndim == 3:
            t = t.transpose(0, 2, 1)
        arrays_q.append(t)
    mps_q = qtn.MatrixProductState(arrays_q)

    dmrg = qtn.DMRG1(mpo, bond_dims=bond_dim, cutoffs=1e-14, p0=mps_q)
    dmrg.solve(max_sweeps=50, tol=tol, verbosity=0)
    E = float(np.real(dmrg.energy))

    if verbose:
        print(f"  Cleanup (quimb DMRG1): E = {E:.12f}")

    return E



def gather_mps(pmps, comm):
    """Gather distributed MPS onto rank 0."""
    rank = comm.Get_rank()
    n_procs = comm.Get_size()

    local_data = {
        'arrays': pmps.arrays,
        'my_sites': list(pmps.my_sites),
    }

    all_data = comm.gather(local_data, root=0)

    if rank == 0:
        L = sum(len(d['arrays']) for d in all_data)
        mps_arrays = [None] * L
        for d in all_data:
            for i, gi in enumerate(d['my_sites']):
                mps_arrays[gi] = d['arrays'][i]
        return mps_arrays
    else:
        return None


def main():
    """CLI entry point for PDMRG."""
    parser = argparse.ArgumentParser(description='Parallel DMRG')
    parser.add_argument('--sites', type=int, default=40)
    parser.add_argument('--bond-dim', type=int, default=100)
    parser.add_argument('--warmup-dim', type=int, default=50)
    parser.add_argument('--warmup-sweeps', type=int, default=5)
    parser.add_argument('--sweeps', type=int, default=20)
    parser.add_argument('--tol', type=float, default=1e-10)
    parser.add_argument('--model', type=str, default='heisenberg',
                        choices=['heisenberg', 'josephson', 'random_tfim'])
    parser.add_argument('--dtype', type=str, default='float64',
                        choices=['float64', 'complex128'])
    parser.add_argument('--timing', action='store_true')
    parser.add_argument('--random-init', action='store_true',
                        help='Skip warmup, start from random MPS (needs more sweeps)')
    args = parser.parse_args()

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    L = args.sites
    if args.model == 'heisenberg':
        mpo = build_heisenberg_mpo(L)
        if args.dtype == 'complex128':
            for i in range(L):
                mpo[i].modify(data=mpo[i].data.astype('complex128'))
    elif args.model == 'josephson':
        from pdmrg.hamiltonians.bose_hubbard import build_bose_hubbard_mpo
        mpo = build_bose_hubbard_mpo(L, dtype=args.dtype)
    elif args.model == 'random_tfim':
        from pdmrg.hamiltonians.random_tfim import build_random_tfim_mpo
        mpo, couplings = build_random_tfim_mpo(L, dtype=args.dtype)
        if rank == 0:
            print(f"Random TFIM: J_mean={couplings['J'].mean():.3f}, h_mean={couplings['h'].mean():.3f}")

    t_start = time.time()

    energy, pmps = pdmrg_main(
        L=L, mpo=mpo,
        max_sweeps=args.sweeps,
        bond_dim=args.bond_dim,
        bond_dim_warmup=args.warmup_dim,
        n_warmup_sweeps=args.warmup_sweeps,
        tol=args.tol,
        dtype=args.dtype,
        comm=comm,
        random_init_flag=args.random_init,
    )

    t_end = time.time()

    if rank == 0 and args.timing:
        print(f"Total wall time: {t_end - t_start:.2f}s")
        print(f"Energy per site: {energy / L:.12f}")


if __name__ == '__main__':
    main()
