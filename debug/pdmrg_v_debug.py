#!/usr/bin/env python3
"""Debug V matrix in PDMRG"""

import sys
import numpy as np
import quimb.tensor as qtn

sys.path.insert(0, '/home/captain/clawd/work/dmrg-implementations/pdmrg')

from mpi4py import MPI
from pdmrg.dmrg import serial_warmup, distribute_mps
from pdmrg.mps.canonical import get_mpo_tensor_data

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

L = 12
mpo = qtn.MPO_ham_heis(L=L, j=1.0, bz=0.0, cyclic=False)
mpo_arrays = [get_mpo_tensor_data(mpo, i) for i in range(L)]

if rank == 0:
    mps_arrays, _, warmup_energy = serial_warmup(mpo, L, bond_dim_warmup=20, n_warmup_sweeps=5)
    print(f"Warmup energy: {warmup_energy:.15f}")
    
    # Check bond at site 5-6
    theta = np.einsum('ijk,klm->ijlm', mps_arrays[5], mps_arrays[6])
    M = theta.reshape(theta.shape[0]*theta.shape[1], theta.shape[2]*theta.shape[3])
    U, S, Vh = np.linalg.svd(M, full_matrices=False)
    print(f"\nSingular values at bond 5-6:")
    print(f"  S = {S[:5]}...")
    print(f"  V = 1/S = {1/S[:5]}...")
else:
    mps_arrays = None

pmps = distribute_mps(mps_arrays, mpo_arrays, comm, dtype=np.float64)

if rank == 0:
    print(f"\nRank {rank}: V_right = {pmps.V_right[:5]}...")
    
    # What happens if we form theta with V vs without?
    # Get rank 1's left tensor
    comm.send({'request': 'tensor'}, dest=1)
    data = comm.recv(source=1)
    neighbor_tensor = data['tensor']
    
    my_tensor = pmps.arrays[-1]
    V = pmps.V_right
    
    # With V
    V_psi = V[:, None, None] * neighbor_tensor
    theta_with_V = np.einsum('ijk,klm->ijlm', my_tensor, V_psi)
    
    # Without V (direct)
    theta_direct = np.einsum('ijk,klm->ijlm', my_tensor, neighbor_tensor)
    
    # Compute energies
    from pdmrg.environments.update import init_left_env, init_right_env, update_left_env, update_right_env
    
    # Build environments
    chi_L0 = mps_arrays[0].shape[0]
    D_0 = mpo_arrays[0].shape[0]
    L_env = init_left_env(chi_L0, D_0, np.float64)
    for i in range(5):
        L_env = update_left_env(L_env, mps_arrays[i], mpo_arrays[i])
    L_env = update_left_env(L_env, my_tensor, mpo_arrays[5])
    
    chi_R_last = mps_arrays[-1].shape[2]
    D_last = mpo_arrays[-1].shape[1]
    R_env = init_right_env(chi_R_last, D_last, np.float64)
    for i in range(L-1, 6, -1):
        R_env = update_right_env(R_env, mps_arrays[i], mpo_arrays[i])
    R_env = update_right_env(R_env, neighbor_tensor, mpo_arrays[6])
    
    # Actually, to compare properly, let's compute <theta|H|theta>/<theta|theta>
    # This requires the effective Hamiltonian...
    
    # Simpler: just check if theta_with_V == theta_direct (up to normalization)
    norm_with_V = np.linalg.norm(theta_with_V)
    norm_direct = np.linalg.norm(theta_direct)
    
    theta_with_V_normalized = theta_with_V / norm_with_V
    theta_direct_normalized = theta_direct / norm_direct
    
    diff = np.linalg.norm(theta_with_V_normalized - theta_direct_normalized)
    print(f"\n||theta_with_V - theta_direct|| = {diff:.6e}")
    print(f"norm(theta_with_V) = {norm_with_V:.6f}")
    print(f"norm(theta_direct) = {norm_direct:.6f}")
    print(f"ratio = {norm_with_V/norm_direct:.6f}")
    
elif rank == 1:
    data = comm.recv(source=0)
    comm.send({'tensor': pmps.arrays[0].copy()}, dest=0)
