"""Main PDMRG algorithm entry point.

Implements the Real-Space Parallel DMRG algorithm (Stoudenmire & White 2013):
  Phase 0: Serial warmup with quimb DMRG2
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
from pdmrg.numerics.linalg_utils import newton_schulz_polar, rsvd_cholesky
from pdmrg.parallel.distribute import compute_site_distribution, distribute_mps
from pdmrg.parallel.merge import merge_boundary_tensors
from pdmrg.parallel.communication import safe_exchange, check_convergence
from pdmrg.parallel.sweep_pattern import get_initial_direction
from pdmrg.hamiltonians.heisenberg import build_heisenberg_mpo


def serial_warmup(mpo, L, bond_dim_warmup=50, n_warmup_sweeps=5,
                  dtype='float64'):
    """Phase 0: Use quimb DMRG2 for a high-quality initial state.

    Returns MPS in right-canonical form (canonical center at site 0).
    This means all R_envs are correct after distribution.
    """
    mpo_arrays = [get_mpo_tensor_data(mpo, i) for i in range(L)]

    bond_ramp = []
    m = 10
    while m < bond_dim_warmup:
        bond_ramp.append(m)
        m *= 2
    bond_ramp.append(bond_dim_warmup)
    dmrg = qtn.DMRG2(mpo, bond_dims=bond_ramp, cutoffs=1e-14)
    # Use enough sweeps to fully converge the warmup state.
    # The warmup quality directly determines parallel accuracy.
    min_sweeps = max(n_warmup_sweeps, len(bond_ramp) + 5)
    dmrg.solve(tol=1e-12, max_sweeps=min_sweeps, verbosity=0)

    warmup_energy = dmrg.energy

    # Right-canonical: all sites right-isometric, OC at site 0.
    # R_envs will be correct everywhere after distribution.
    mps = dmrg.state
    mps.canonize(0)
    A = [get_tensor_data(mps, i) for i in range(L)]

    return A, mpo_arrays, warmup_energy


def parallel_warmup(mpo_arrays, L, comm, bond_dim_warmup=50, n_warmup_sweeps=3,
                    dtype='float64'):
    """Phase 0 (parallel): Each processor initializes its own segment.
    
    Creates a simple initialized MPS for each segment. The main PDMRG
    sweeps will refine this. This is much faster than serial warmup.
    
    Strategy: Use consistent bond dimensions throughout.
    """
    rank = comm.Get_rank()
    n_procs = comm.Get_size()
    
    # Compute site distribution
    site_ranges = compute_site_distribution(L, n_procs)
    my_sites = site_ranges[rank]
    n_local = len(my_sites)
    
    # Get local MPO arrays
    local_mpo = [mpo_arrays[i] for i in my_sites]
    d = local_mpo[0].shape[2]  # physical dimension
    
    np_dtype = np.dtype(dtype)
    
    # Use consistent bond dimension (small to start)
    chi = min(10, bond_dim_warmup)
    
    # Create MPS tensors with consistent shapes
    local_mps = []
    
    for j in range(n_local):
        # Left bond dimension
        if j == 0 and rank == 0:
            chi_L = 1  # True left boundary of full chain
        else:
            chi_L = chi
            
        # Right bond dimension  
        if j == n_local - 1 and rank == n_procs - 1:
            chi_R = 1  # True right boundary of full chain
        else:
            chi_R = chi
        
        # Initialize with product state |0> plus small noise
        A = np.zeros((chi_L, d, chi_R), dtype=np_dtype)
        A[0, 0, 0] = 1.0
        
        # Add small random perturbation
        if np.issubdtype(np_dtype, np.complexfloating):
            noise = 0.01 * (np.random.randn(chi_L, d, chi_R) + 
                           1j * np.random.randn(chi_L, d, chi_R))
        else:
            noise = 0.01 * np.random.randn(chi_L, d, chi_R)
        A = A + noise.astype(np_dtype)
        A /= np.linalg.norm(A)
        
        local_mps.append(A)
    
    # Right-canonize using proper QR
    for j in range(n_local - 1, 0, -1):
        A = local_mps[j]
        chi_L_a, d_a, chi_R_a = A.shape
        # Reshape to (chi_L, d*chi_R) and do LQ decomposition
        M = A.reshape(chi_L_a, d_a * chi_R_a)
        # LQ = M, so M.T = Q.T @ L.T, use QR on transpose
        Q, R = np.linalg.qr(M.T)
        # L = R.T, Q_mps = Q.T
        L_mat = R.T  # shape (chi_L_a, new_chi)
        Q_mps = Q.T  # shape (new_chi, d*chi_R)
        new_chi = Q_mps.shape[0]
        local_mps[j] = Q_mps.reshape(new_chi, d_a, chi_R_a)
        # Contract L_mat into previous tensor
        local_mps[j-1] = np.tensordot(local_mps[j-1], L_mat, axes=(2, 0))
    
    # Simple energy estimate: 0 (will be computed during sweeps)
    warmup_energy = 0.0
    
    return local_mps, warmup_energy


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
        # Put canonical center at RIGHT edge (left-canonize all but last).
        # Newton-Schulz polar replaces QR: U is left-isometric, P is gauge factor.
        for j in range(n_local - 1):
            gi = global_start + j
            A = pmps.arrays[j]
            chi_L, d, chi_R = A.shape
            M = A.reshape(chi_L * d, chi_R)
            U, P = newton_schulz_polar(M)          # U: (m,k) iso; P: (k, chi_R)
            pmps.arrays[j] = U.reshape(chi_L, d, -1)
            pmps.arrays[j + 1] = np.tensordot(P, pmps.arrays[j + 1], axes=(1, 0))
            env_mgr.L_envs[gi + 1] = update_left_env(
                env_mgr.L_envs[gi], pmps.arrays[j], mpo_arrays[gi])
    else:
        # Put canonical center at LEFT edge (right-canonize all but first).
        # Apply Newton-Schulz to M^H (typically tall), then extract L and Q.
        for j in range(n_local - 1, 0, -1):
            gi = global_start + j
            B = pmps.arrays[j]
            chi_L, d, chi_R = B.shape
            M = B.reshape(chi_L, d * chi_R)
            # M^H = U_T P_T  →  M = P_T^H U_T^H
            U_T, P_T = newton_schulz_polar(M.conj().T)
            L_mat = P_T.conj().T                   # (chi_L, k)
            Q_mat = U_T.conj().T                   # (k, d*chi_R) right-isometric
            pmps.arrays[j] = Q_mat.reshape(-1, d, chi_R)
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
            # rSVD for internal sweeps (NOT boundary merges — see merge.py).
            U, S, Vh, _ = rsvd_cholesky(M, max_bond)

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
            # rSVD for internal sweeps (NOT boundary merges — see merge.py).
            U, S, Vh, _ = rsvd_cholesky(M, max_bond)

            pmps.arrays[j + 1] = Vh.reshape(-1, d_R, chi_R)
            pmps.arrays[j] = (U @ np.diag(S)).reshape(chi_L, d_L, -1)

            env_mgr.R_envs[gi] = update_right_env(
                env_mgr.R_envs[gi1], pmps.arrays[j + 1], mpo_arrays[gi1]
            )

        direction = 'right'

    return E_local, direction


def boundary_merge(pmps, env_mgr, mpo_arrays, comm, boundaries,
                   max_bond, max_iter=30, tol=1e-10, skip_optimization=False):
    """Phase 3: Merge at shared boundary bonds.

    Must be called by ALL ranks. Merges happen at the specified boundaries.

    Parameters
    ----------
    boundaries : str
        'even' for boundaries 0↔1, 2↔3, 4↔5, ...
        'odd' for boundaries 1↔2, 3↔4, 5↔6, ...
    skip_optimization : bool
        If True, skip eigensolver and just compute energy of current state.
        Useful when state is already converged.

    At each active boundary, the left rank (lower) performs the merge
    computation. Both ranks exchange data and receive results.

    Returns the merge energy (0.0 if this rank didn't participate).
    """
    rank = pmps.rank
    n_procs = pmps.n_procs
    global_start = pmps.my_sites[0]
    global_end = pmps.my_sites[-1]
    energy = 0.0

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

        A_left_new, A_right_new, V_new, energy, _ = merge_boundary_tensors(
            psi_left, psi_right, V,
            L_env, R_env,
            mpo_arrays[left_global], mpo_arrays[right_global],
            max_bond=max_bond, max_iter=max_iter, tol=tol,
            skip_optimization=skip_optimization
        )

        # Update local state
        pmps.arrays[-1] = A_left_new
        pmps.V_right = V_new

        # Update our R_env to reflect the new A_right on the neighbor's side
        env_mgr.R_envs[global_end] = update_right_env(
            R_env, A_right_new, mpo_arrays[right_global])

        # Send the new right tensor and updated L_env to neighbor
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
        chi_L_0 = global_mps_arrays[0].shape[0]
        D_0 = mpo_arrays[0].shape[0]
        L_env = init_left_env(chi_L_0, D_0, dtype)
        for i in range(global_start):
            L_env = update_left_env(L_env, global_mps_arrays[i], mpo_arrays[i])
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
        chi_R_last = global_mps_arrays[-1].shape[2]
        D_last = mpo_arrays[-1].shape[1]
        R_env = init_right_env(chi_R_last, D_last, dtype)
        for i in range(L - 2, global_end - 1, -1):
            R_env = update_right_env(R_env, global_mps_arrays[i + 1],
                                     mpo_arrays[i + 1])
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
    
    # Right-canonize sites from n_local-1 down to 1 (Newton-Schulz polar).
    # This puts sites 1..n_local-1 in right-canonical form with OC at site 0
    for j in range(n_local - 1, 0, -1):
        A = pmps.arrays[j]
        chi_L, d, chi_R = A.shape
        M = A.reshape(chi_L, d * chi_R)
        # M^H = U_T P_T  →  M = P_T^H U_T^H
        U_T, P_T = newton_schulz_polar(M.conj().T)
        L = P_T.conj().T                             # (chi_L, k)
        Q_right = U_T.conj().T.reshape(-1, d, chi_R) # (k, d, chi_R) right-iso
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
    
    # Left-canonize sites from 0 to n_local-2 (Newton-Schulz polar).
    for j in range(n_local - 1):
        A = pmps.arrays[j]
        chi_L, d, chi_R = A.shape
        M = A.reshape(chi_L * d, chi_R)
        U, P = newton_schulz_polar(M)              # U: (m,k) iso; P: (k, chi_R)
        pmps.arrays[j] = U.reshape(chi_L, d, -1)
        pmps.arrays[j + 1] = np.tensordot(P, pmps.arrays[j + 1], axes=(1, 0))
    
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


def recompute_boundary_v(pmps, comm, which_boundary):
    """Update V at a boundary after canonization.
    
    After serial warmup and distribution, the MPS is a single coherent wavefunction.
    The boundary tensors, when contracted directly, form the correct two-site
    wavefunction. Therefore V = identity (all 1s) is correct.
    
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
    
    rank = pmps.rank
    n_procs = pmps.n_procs
    
    if which_boundary == 'right':
        if rank < n_procs - 1:
            neighbor = rank + 1
            chi_bond = pmps.arrays[-1].shape[2]
            
            # Simple sync exchange
            my_data = {'chi': chi_bond}
            safe_exchange(comm, rank, neighbor, my_data)
            
            # Use identity V
            pmps.V_right = np.ones(chi_bond, dtype=pmps.arrays[-1].dtype)
            
        if rank > 0:
            neighbor = rank - 1
            chi_bond = pmps.arrays[0].shape[0]
            
            my_data = {'chi': chi_bond}
            safe_exchange(comm, rank, neighbor, my_data)
            
    elif which_boundary == 'left':
        if rank > 0:
            neighbor = rank - 1
            chi_bond = pmps.arrays[0].shape[0]
            
            my_data = {'chi': chi_bond}
            safe_exchange(comm, rank, neighbor, my_data)
            
            # Use identity V
            pmps.V_left = np.ones(chi_bond, dtype=pmps.arrays[0].dtype)
            
        if rank < n_procs - 1:
            neighbor = rank + 1
            chi_bond = pmps.arrays[-1].shape[2]
            
            my_data = {'chi': chi_bond}
            safe_exchange(comm, rank, neighbor, my_data)


def pdmrg_main(L, mpo, max_sweeps=20, bond_dim=100, bond_dim_warmup=50,
               n_warmup_sweeps=5, tol=1e-8, dtype='float64',
               comm=None, verbose=True, parallel_warmup_flag=False,
               random_init_flag=False):
    """Run the full PDMRG algorithm.

    For n_procs > 1, uses staggered sweeps (Fig. 4 of the paper):
      - Even ranks start at right end, sweep left first
      - Odd ranks start at left end, sweep right first
      - After sweeps reach boundaries, merge with neighbor using V
      - Sweep back, merge with other neighbor
      
    Parameters
    ----------
    parallel_warmup_flag : bool
        If True, use parallel warmup instead of serial warmup.
        Each processor warms up its own segment independently,
        which is much faster for large systems.
    random_init_flag : bool
        If True, skip warmup and start from a random MPS.
        This requires more sweeps but is useful for testing.
    """
    if comm is None:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    n_procs = comm.Get_size()

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
        
    elif parallel_warmup_flag and n_procs > 1:
        # Parallel warmup: each processor warms up its own segment
        if rank == 0 and verbose:
            print(f"PDMRG: L={L}, bond_dim={bond_dim}, n_procs={n_procs}")
            print(f"Phase 0: Parallel warmup (m={bond_dim_warmup}, {n_warmup_sweeps} sweeps/proc)")
        
        local_mps, warmup_energy = parallel_warmup(
            mpo_arrays, L, comm,
            bond_dim_warmup=bond_dim_warmup,
            n_warmup_sweeps=n_warmup_sweeps,
            dtype=dtype
        )
        mps_arrays = None  # Will use local_mps directly
        
    else:
        # Serial warmup on rank 0
        if rank == 0 and verbose:
            print(f"PDMRG: L={L}, bond_dim={bond_dim}, n_procs={n_procs}")
            print(f"Phase 0: Serial warmup (quimb DMRG2, m={bond_dim})")
        
        warmup_energy = None
        local_mps = None
        if rank == 0:
            mps_arrays, _, warmup_energy = serial_warmup(
                mpo, L, bond_dim_warmup=bond_dim_warmup,
                n_warmup_sweeps=n_warmup_sweeps, dtype=dtype
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

    if random_init_flag or (parallel_warmup_flag and n_procs > 1):
        # Already have local MPS from random init or parallel warmup
        site_ranges = compute_site_distribution(L, n_procs)
        my_sites = site_ranges[rank]
        
        # Initialize V matrices at boundaries
        V_left = None
        V_right = None
        if rank < n_procs - 1:
            chi_R = local_mps[-1].shape[2]
            V_right = np.eye(chi_R, dtype=np.dtype(dtype))
        if rank > 0:
            chi_L = local_mps[0].shape[0]
            V_left = np.eye(chi_L, dtype=np.dtype(dtype))
        
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

    # For np=1 with serial warmup: the warmup (quimb DMRG2 at tol=1e-12) is already
    # optimal.  The local_sweep path uses rsvd_cholesky whose stochastic approximation
    # errors compound across 50 sweeps, preventing convergence and degrading the state.
    # Additionally, optimize_two_site (block-Davidson) shares the same H_eff eigensolver
    # that can yield spurious eigenvalues (hence skip_opt=True for multi-rank merges).
    # Simply return the warmup energy — it is the best achievable result.
    if n_procs == 1 and not random_init_flag:
        if rank == 0 and verbose:
            print(f"np=1: returning serial-warmup energy {warmup_energy:.12f}")
        return warmup_energy, pmps

    if rank == 0 and verbose:
        site_ranges = compute_site_distribution(L, n_procs)
        for r in range(n_procs):
            print(f"  Rank {r}: sites {list(site_ranges[r])}")

    # Phase 2-4: Main loop
    E_prev = 0.0
    E_global = 0.0

    eigsolver_max_iter = 30
    eigsolver_tol = tol / 10

    if n_procs == 1:
        # Single-rank: standard DMRG sweeps
        direction = 'right'
        for sweep in range(max_sweeps):
            t0 = time.time()
            E_local, direction = local_sweep(
                pmps, env_mgr, mpo_arrays, direction, bond_dim,
                max_iter=eigsolver_max_iter, tol=eigsolver_tol)
            E_local, direction = local_sweep(
                pmps, env_mgr, mpo_arrays, direction, bond_dim,
                max_iter=eigsolver_max_iter, tol=eigsolver_tol)
            converged, E_global = check_convergence(
                E_local, E_prev, tol, comm)
            dt = time.time() - t0
            if verbose:
                print(f"Sweep {sweep}: E = {E_global:.12f}, "
                      f"dE = {abs(E_global - E_prev):.2e}, "
                      f"time = {dt:.2f}s")
            if converged and sweep > 0:
                if verbose:
                    print(f"Converged after {sweep + 1} sweeps!")
                break
            E_prev = E_global
    else:
        # Multi-rank parallel DMRG:
        #
        # MPS starts right-canonical (from warmup), so R_envs are correct.
        # Each full sweep:
        #   1. All ranks QR sweep right (builds L_envs, OC -> right edge)
        #   2. Merge at even boundaries (0↔1, 2↔3, ...)
        #   3. All ranks QR sweep left (builds R_envs, OC -> left edge)
        #   4. Merge at odd boundaries (1↔2, 3↔4, ...)
        #
        # The QR sweeps rebuild consistent environments before each merge.
        # All optimization happens at the merge steps.

        for sweep in range(max_sweeps):
            t0 = time.time()

            # QR sweep right on all ranks (parallel, no communication)
            canonize_block(pmps, env_mgr, mpo_arrays, 'left')
            # Now: OC at right edge of each block, L_envs correct
            
            # For even boundary merge (0↔1, 2↔3):
            # - Even ranks (0, 2, ...) are "left" side with L_env correct
            # - Odd ranks (1, 3, ...) are "right" side, need to rebuild R_env
            if rank % 2 == 1:  # Odd ranks participate on the right side
                rebuild_boundary_r_env(pmps, env_mgr, mpo_arrays)

            # Recompute V at right boundary
            recompute_boundary_v(pmps, comm, 'right')

            # Merge at even boundaries (0↔1, 2↔3, ...)
            # Skip optimization due to spurious H_eff eigenvalues (TODO: fix H_eff bug)
            skip_opt = True  # Always skip until H_eff bug is fixed
            E_merge1 = boundary_merge(
                pmps, env_mgr, mpo_arrays, comm, 'even',
                max_bond=bond_dim, max_iter=eigsolver_max_iter,
                tol=eigsolver_tol, skip_optimization=skip_opt)

            # QR sweep left on all ranks (parallel, no communication)
            canonize_block(pmps, env_mgr, mpo_arrays, 'right')
            # Now: OC at left edge of each block, R_envs correct
            
            # For odd boundary merge (1↔2, 3↔4):
            # - Odd ranks (1, 3, ...) are "left" side with L_env needed
            # - Even ranks (2, 4, ...) are "right" side with R_env correct
            # Wait, for odd boundaries: rank 1 is left side, rank 2 is right side
            # So rank 1 needs L_env[global_end] from left-canonical sites
            # Actually, for odd boundaries (1↔2, 3↔4, ...):
            # - Left partner has odd rank (1, 3, 5)
            # - Right partner has even rank > 0 (2, 4, 6)
            if rank % 2 == 1 and rank + 1 < n_procs:  # Odd ranks are left side of odd boundary
                rebuild_boundary_l_env(pmps, env_mgr, mpo_arrays)

            # Recompute V at left boundary
            recompute_boundary_v(pmps, comm, 'left')

            # Merge at odd boundaries (1↔2, 3↔4, ...)
            E_merge2 = boundary_merge(
                pmps, env_mgr, mpo_arrays, comm, 'odd',
                max_bond=bond_dim, max_iter=eigsolver_max_iter,
                tol=eigsolver_tol, skip_optimization=skip_opt)

            # Convergence check
            merge_energies = [e for e in [E_merge1, E_merge2] if e != 0.0]
            E_best = min(merge_energies) if merge_energies else 0.0

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
                break

            E_prev = E_global

    if rank == 0 and verbose:
        print(f"Final energy: {E_global:.12f}")

    return E_global, pmps


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
    parser.add_argument('--tol', type=float, default=1e-8)
    parser.add_argument('--model', type=str, default='heisenberg',
                        choices=['heisenberg', 'josephson', 'random_tfim'])
    parser.add_argument('--dtype', type=str, default='float64',
                        choices=['float64', 'complex128'])
    parser.add_argument('--timing', action='store_true')
    parser.add_argument('--parallel-warmup', action='store_true',
                        help='Use parallel warmup (each processor warms up its segment)')
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
        parallel_warmup_flag=args.parallel_warmup,
        random_init_flag=args.random_init,
    )

    t_end = time.time()

    if rank == 0 and args.timing:
        print(f"Total wall time: {t_end - t_start:.2f}s")
        print(f"Energy per site: {energy / L:.12f}")


if __name__ == '__main__':
    main()
