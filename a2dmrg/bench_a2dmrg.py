"""Benchmark A2DMRG accuracy and performance vs quimb DMRG1/DMRG2.

Usage:
    mpirun -np 2 python bench_a2dmrg.py

Measures:
1. Accuracy: |E_a2dmrg - E_ref| for each (L, chi) pair
2. Performance: wall time for each solver
"""

import time
import numpy as np
import quimb.tensor as qtn
from a2dmrg.hamiltonians.heisenberg import heisenberg_mpo
from a2dmrg.dmrg import a2dmrg_main
from a2dmrg.mpi_compat import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Test cases: (L, chi)
cases = [
    (8, 20),
    (10, 20),
    (12, 20),
    (12, 50),
    (16, 20),
    (16, 50),
    (20, 20),
    (20, 50),
]


def run_quimb_dmrg1(mpo, chi):
    dmrg = qtn.DMRG1(mpo, bond_dims=chi, cutoffs=1e-14)
    dmrg.solve(max_sweeps=40, tol=1e-13, verbosity=0)
    return float(np.real(dmrg.energy))


def run_quimb_dmrg2(mpo, chi):
    dmrg = qtn.DMRG2(mpo, bond_dims=chi, cutoffs=1e-14)
    dmrg.solve(max_sweeps=40, tol=1e-13, verbosity=0)
    return float(np.real(dmrg.energy))


def run_a2dmrg(L, mpo, chi):
    energy, mps = a2dmrg_main(
        L=L, mpo=mpo, bond_dim=chi, max_sweeps=20, tol=1e-12,
        warmup_sweeps=2, finalize_sweeps=0,
        comm=comm, verbose=False, one_site=False,
        timing_report=False,
    )
    return energy


if rank == 0:
    print(f"{'L':>4} {'chi':>4} | {'E_DMRG1':>18} {'t1':>6} | {'E_DMRG2':>18} {'t2':>6} | {'E_A2DMRG':>18} {'ta2':>6} | {'|A2-D1|':>10} {'|A2-D2|':>10}")
    print("-" * 130)

for L, chi in cases:
    mpo = heisenberg_mpo(L)

    # quimb DMRG1 (rank 0 only)
    if rank == 0:
        t0 = time.perf_counter()
        E_dmrg1 = run_quimb_dmrg1(mpo, chi)
        t_dmrg1 = time.perf_counter() - t0
    else:
        E_dmrg1 = None
        t_dmrg1 = None

    # quimb DMRG2 (rank 0 only)
    if rank == 0:
        t0 = time.perf_counter()
        E_dmrg2 = run_quimb_dmrg2(mpo, chi)
        t_dmrg2 = time.perf_counter() - t0
    else:
        E_dmrg2 = None
        t_dmrg2 = None

    # A2DMRG (all ranks)
    comm.Barrier()
    t0 = time.perf_counter()
    E_a2dmrg = run_a2dmrg(L, mpo, chi)
    comm.Barrier()
    t_a2dmrg = time.perf_counter() - t0

    if rank == 0:
        diff1 = abs(E_a2dmrg - E_dmrg1)
        diff2 = abs(E_a2dmrg - E_dmrg2)
        status1 = "PASS" if diff1 < 1e-10 else ("WARN" if diff1 < 1e-8 else "FAIL")
        status2 = "PASS" if diff2 < 1e-10 else ("WARN" if diff2 < 1e-8 else "FAIL")
        print(
            f"{L:4d} {chi:4d} | {E_dmrg1:18.12f} {t_dmrg1:5.1f}s | {E_dmrg2:18.12f} {t_dmrg2:5.1f}s | "
            f"{E_a2dmrg:18.12f} {t_a2dmrg:5.1f}s | {diff1:10.2e} {status1:4s} {diff2:10.2e} {status2:4s}"
        )

if rank == 0:
    print("\nTarget: |diff| < 1e-10 = PASS, < 1e-8 = WARN, else FAIL")
