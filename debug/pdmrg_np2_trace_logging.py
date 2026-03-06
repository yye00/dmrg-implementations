#!/usr/bin/env python3
"""
PDMRG np=2 with detailed trace logging for GPU comparison.

This script runs PDMRG with np=2 using the same serialized MPS/MPO as the GPU version,
and logs detailed information about tensor contractions for step-by-step comparison.
"""

import sys
import time
import json
import numpy as np
import quimb.tensor as qtn

sys.path.insert(0, '/home/captain/clawd/work/dmrg-implementations/pdmrg')

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_procs = comm.Get_size()

# Configuration
L = 8
bond_dim = 32
tol = 1e-10

if rank == 0:
    print("="*80)
    print(f"PDMRG np=2 TRACE LOGGING FOR GPU COMPARISON")
    print("="*80)
    print(f"L={L}, bond_dim={bond_dim}, n_procs={n_procs}")
    print()

# Load serialized MPS and MPO
mps_file = f"/tmp/heisenberg_L{L}_mps_initial.bin"
mpo_file = f"/tmp/heisenberg_L{L}_mpo.bin"

def load_mps_from_binary(filename, L):
    """Load MPS from binary file."""
    # Bond dimensions for L=8: [1, 2, 4, 8, 16, 8, 4, 2, 1]
    bond_dims = [1, 2, 4, 8, 16, 8, 4, 2, 1]

    with open(filename, 'rb') as f:
        arrays = []
        for i in range(L):
            chi_L = bond_dims[i]
            chi_R = bond_dims[i+1]
            d = 2  # spin-1/2
            size = chi_L * d * chi_R
            data = np.fromfile(f, dtype=np.complex128, count=size)
            A = data.reshape((chi_L, d, chi_R))
            arrays.append(A)
    return arrays

def load_mpo_from_binary(filename, L):
    """Load MPO from binary file."""
    with open(filename, 'rb') as f:
        arrays = []
        for i in range(L):
            if i == 0:
                D_L, D_R = 1, 5
            elif i == L - 1:
                D_L, D_R = 5, 1
            else:
                D_L, D_R = 5, 5
            d = 2
            size = D_L * D_R * d * d
            data = np.fromfile(f, dtype=np.complex128, count=size)
            W = data.reshape((D_L, D_R, d, d))
            arrays.append(W)
    return arrays

if rank == 0:
    print(f"Loading MPS from {mps_file}")
    print(f"Loading MPO from {mpo_file}")
    print()

mps_arrays = load_mps_from_binary(mps_file, L)
mpo_arrays = load_mpo_from_binary(mpo_file, L)

# Convert to real if they are real
if all(np.allclose(A.imag, 0) for A in mps_arrays):
    if rank == 0:
        print("MPS is real, converting to float64")
    mps_arrays = [A.real.astype(np.float64) for A in mps_arrays]
    mpo_arrays = [W.real.astype(np.float64) for W in mpo_arrays]
    dtype = np.float64
else:
    dtype = np.complex128

# Import PDMRG components
from pdmrg.dmrg import distribute_mps, build_local_environments
from pdmrg.dmrg import canonize_block, boundary_merge
from pdmrg.parallel.distribute import compute_site_distribution

# Import environment functions for logging
from pdmrg.environments.update import update_left_env, update_right_env

# Monkey-patch environment update functions to add logging
original_update_left_env = update_left_env
original_update_right_env = update_right_env

def logged_update_left_env(L_env, A, W):
    """Logged version of update_left_env."""
    print(f"  [Rank {rank}] update_left_env:")
    print(f"    L_env.shape = {L_env.shape}")
    print(f"    A.shape = {A.shape}")
    print(f"    W.shape = {W.shape}")
    result = original_update_left_env(L_env, A, W)
    print(f"    result.shape = {result.shape}")
    return result

def logged_update_right_env(R_env, B, W):
    """Logged version of update_right_env."""
    print(f"  [Rank {rank}] update_right_env:")
    print(f"    R_env.shape = {R_env.shape}")
    print(f"    B.shape = {B.shape}")
    print(f"    W.shape = {W.shape}")
    result = original_update_right_env(R_env, B, W)
    print(f"    result.shape = {result.shape}")
    return result

# Patch the modules
import pdmrg.environments.update
pdmrg.environments.update.update_left_env = logged_update_left_env
pdmrg.environments.update.update_right_env = logged_update_right_env

# Also patch in dmrg module
import pdmrg.dmrg
pdmrg.dmrg.update_left_env = logged_update_left_env
pdmrg.dmrg.update_right_env = logged_update_right_env

# Phase 1: Distribute
if rank == 0:
    print("="*80)
    print("PHASE 1: Distributing MPS")
    print("="*80)

pmps = distribute_mps(mps_arrays, mpo_arrays, comm, dtype=dtype)

global_arrays = getattr(pmps, '_global_arrays', None)
env_mgr = build_local_environments(pmps, mpo_arrays, dtype=dtype,
                                    global_mps_arrays=global_arrays)
if hasattr(pmps, '_global_arrays'):
    del pmps._global_arrays

site_ranges = compute_site_distribution(L, n_procs)
print(f"[Rank {rank}] sites {list(site_ranges[rank])}, shapes: {[a.shape for a in pmps.arrays]}")

comm.Barrier()

# Phase 2: First sweep (QR right)
if rank == 0:
    print()
    print("="*80)
    print("PHASE 2: QR sweep right (canonize_block 'left')")
    print("="*80)

canonize_block(pmps, env_mgr, mpo_arrays, 'left')

comm.Barrier()

# Print environment states after canonization
print(f"[Rank {rank}] After QR sweep right:")
for gi in pmps.my_sites:
    if gi in env_mgr.L_envs:
        print(f"  L_env[{gi}].shape = {env_mgr.L_envs[gi].shape}")
    if gi in env_mgr.R_envs:
        print(f"  R_env[{gi}].shape = {env_mgr.R_envs[gi].shape}")

comm.Barrier()

# Phase 3: rebuild_boundary_r_env for odd ranks
if rank == 0:
    print()
    print("="*80)
    print("PHASE 3: Rebuild boundary R_env for odd ranks")
    print("="*80)

from pdmrg.dmrg import rebuild_boundary_r_env

if rank % 2 == 1:
    print(f"[Rank {rank}] Rebuilding R_env at left boundary")
    rebuild_boundary_r_env(pmps, env_mgr, mpo_arrays)
    print(f"[Rank {rank}] After rebuild: R_env[{pmps.my_sites[0]}].shape = {env_mgr.R_envs[pmps.my_sites[0]].shape}")

comm.Barrier()

# Phase 4: Even boundary merge
if rank == 0:
    print()
    print("="*80)
    print("PHASE 4: Even boundary merge (0↔1)")
    print("="*80)

from pdmrg.dmrg import recompute_boundary_v

recompute_boundary_v(pmps, comm, 'right')

print(f"[Rank {rank}] Before merge:")
print(f"  pmps.arrays[-1].shape = {pmps.arrays[-1].shape if pmps.arrays else 'N/A'}")
if hasattr(pmps, 'V_right') and pmps.V_right is not None:
    print(f"  V_right.shape = {pmps.V_right.shape}")

E_merge1 = boundary_merge(
    pmps, env_mgr, mpo_arrays, comm, 'even',
    max_bond=bond_dim, max_iter=30, tol=tol/10, skip_optimization=True
)

print(f"[Rank {rank}] After merge:")
print(f"  Energy = {E_merge1:.15f}")
print(f"  pmps.arrays[-1].shape = {pmps.arrays[-1].shape if pmps.arrays else 'N/A'}")

comm.Barrier()

if rank == 0:
    print()
    print("="*80)
    print("RESULTS")
    print("="*80)
    print(f"Even merge energy: {E_merge1:.15f}")
    print()
