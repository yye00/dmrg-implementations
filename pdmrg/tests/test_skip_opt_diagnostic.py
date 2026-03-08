#!/usr/bin/env python3
"""
Diagnostic test for PDMRG skip_opt / H_eff spurious eigenvalue problem.

This test demonstrates the failure mode when boundary merge optimization
is enabled (skip_optimization=False).

Expected behavior:
- With skip_opt=True: merge completes successfully, energy is reasonable
- With skip_opt=False: eigensolver may fail or produce spurious eigenvalues

Root cause hypothesis:
1. V = Lambda^-1 amplifies small singular values → large V entries
2. theta = psi_left · diag(V) · psi_right becomes poorly conditioned
3. H_eff eigensolver finds spurious eigenvalues due to gauge inconsistency
"""

import sys
import numpy as np
from pathlib import Path

# Add pdmrg to path
pdmrg_root = Path(__file__).parent.parent
sys.path.insert(0, str(pdmrg_root))

from mpi4py import MPI
import quimb.tensor as qtn

from pdmrg.dmrg import build_local_environments, boundary_merge
from pdmrg.parallel.distribute import distribute_mps
from pdmrg.mps.canonical import get_tensor_data, get_mpo_tensor_data


def run_diagnostic(skip_opt, L=12, bond_dim=10, n_warmup_sweeps=5):
    """
    Run a minimal PDMRG test with 2 ranks to isolate the skip_opt issue.

    Parameters
    ----------
    skip_opt : bool
        If True, skip the eigensolver optimization (current safe path)
        If False, run eigensolver (triggers spurious eigenvalue problem)
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_procs = comm.Get_size()

    if n_procs != 2:
        if rank == 0:
            print("ERROR: This test requires exactly 2 MPI processes")
            print(f"Usage: mpirun -np 2 python {__file__}")
        sys.exit(1)

    # Build Heisenberg MPO
    mpo = qtn.MPO_ham_heis(L=L, j=1.0, bz=0.0, cyclic=False)
    mpo_arrays = [get_mpo_tensor_data(mpo, i) for i in range(L)]

    # Serial warmup on rank 0
    if rank == 0:
        print("=" * 70)
        print(f"SKIP_OPT DIAGNOSTIC TEST: skip_optimization={skip_opt}")
        print("=" * 70)
        print(f"System: L={L}, bond_dim={bond_dim}, n_procs={n_procs}")
        print(f"Running serial warmup (quimb DMRG2)...")

        dmrg = qtn.DMRG2(mpo, bond_dims=bond_dim, cutoffs=1e-14)
        dmrg.solve(tol=1e-12, max_sweeps=n_warmup_sweeps, verbosity=0)
        warmup_energy = float(np.real(dmrg.energy))
        print(f"Warmup energy: {warmup_energy:.12f}")

        mps = dmrg.state
        mps.canonize(L - 1)  # Left-canonical form
        mps_arrays = [get_tensor_data(mps, i) for i in range(L)]
    else:
        mps_arrays = None
        warmup_energy = None

    warmup_energy = comm.bcast(warmup_energy, root=0)

    # Distribute MPS to 2 ranks
    pmps = distribute_mps(mps_arrays, mpo_arrays, comm)

    # Build environments
    global_arrays = getattr(pmps, '_global_arrays', None)
    env_mgr = build_local_environments(pmps, mpo_arrays,
                                        global_mps_arrays=global_arrays)
    if hasattr(pmps, '_global_arrays'):
        del pmps._global_arrays

    # Diagnostic: Check V matrix condition
    if rank == 0:
        V = pmps.V_right
        S = 1.0 / V  # Singular values
        print(f"\nRank 0 boundary V matrix diagnostics:")
        print(f"  V shape: {V.shape}")
        print(f"  V range: [{V.min():.3e}, {V.max():.3e}]")
        print(f"  S range: [{S.min():.3e}, {S.max():.3e}]")
        print(f"  Condition number (max(V)/min(V)): {V.max()/V.min():.3e}")
        print(f"  V[:5] = {V[:5]}")
        print(f"  S[:5] = {S[:5]}")

    comm.Barrier()

    # Attempt boundary merge with specified skip_opt
    if rank == 0:
        print(f"\nAttempting boundary merge with skip_optimization={skip_opt}...")

    try:
        E_merge = boundary_merge(
            pmps, env_mgr, mpo_arrays, comm, 'even',
            max_bond=bond_dim, max_iter=30, tol=1e-10,
            skip_optimization=skip_opt
        )

        if rank == 0:
            dE = E_merge - warmup_energy
            print(f"\nRESULT: SUCCESS")
            print(f"  Merge energy:  {E_merge:.12f}")
            print(f"  Warmup energy: {warmup_energy:.12f}")
            print(f"  ΔE = {dE:.3e}")

            # Check if energy is reasonable (should be close to warmup)
            if abs(dE) > 1e-6:
                print(f"  WARNING: Energy changed significantly!")
                print(f"  This may indicate a spurious eigenvalue problem.")
            else:
                print(f"  Energy change is reasonable.")

    except Exception as e:
        if rank == 0:
            print(f"\nRESULT: FAILED")
            print(f"  Exception: {type(e).__name__}")
            print(f"  Message: {str(e)}")
        raise

    comm.Barrier()

    if rank == 0:
        print("=" * 70)

    return E_merge


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Diagnose PDMRG skip_opt issue')
    parser.add_argument('--skip-opt', action='store_true',
                        help='Run with skip_optimization=True (safe path)')
    parser.add_argument('--no-skip-opt', action='store_true',
                        help='Run with skip_optimization=False (triggers issue)')
    parser.add_argument('--both', action='store_true',
                        help='Run both modes for comparison')
    parser.add_argument('--L', type=int, default=12,
                        help='System size')
    parser.add_argument('--bond-dim', type=int, default=10,
                        help='Bond dimension')

    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if args.both:
        # Run both modes
        if rank == 0:
            print("\n" + "=" * 70)
            print("RUNNING BOTH MODES FOR COMPARISON")
            print("=" * 70 + "\n")

        run_diagnostic(skip_opt=True, L=args.L, bond_dim=args.bond_dim)

        comm.Barrier()
        if rank == 0:
            print("\n\n")

        run_diagnostic(skip_opt=False, L=args.L, bond_dim=args.bond_dim)

    elif args.no_skip_opt:
        run_diagnostic(skip_opt=False, L=args.L, bond_dim=args.bond_dim)
    else:
        # Default: skip_opt=True (safe)
        run_diagnostic(skip_opt=True, L=args.L, bond_dim=args.bond_dim)
