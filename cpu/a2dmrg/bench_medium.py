"""Medium-scale benchmark: A2DMRG (np=2) vs quimb DMRG1/DMRG2.

Tests:
1. Heisenberg spin-1/2 chain (real, OBC): L=32,48,64 chi=50,100
2. Josephson junction / Bose-Hubbard (complex128, OBC): L=16,24,32 chi=30,50

Usage:
    mpirun -np 2 python bench_medium.py
"""

import time
import numpy as np
import quimb.tensor as qtn
from a2dmrg.hamiltonians.heisenberg import heisenberg_mpo
from a2dmrg.hamiltonians.bose_hubbard import bose_hubbard_mpo
from a2dmrg.dmrg import a2dmrg_main
from a2dmrg.mpi_compat import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def run_dmrg1(mpo, chi, max_sw=50):
    dmrg = qtn.DMRG1(mpo, bond_dims=chi, cutoffs=1e-14)
    dmrg.solve(max_sweeps=max_sw, tol=1e-13, verbosity=0)
    return float(np.real(dmrg.energy))


def run_dmrg2(mpo, chi, max_sw=50):
    dmrg = qtn.DMRG2(mpo, bond_dims=chi, cutoffs=1e-14)
    dmrg.solve(max_sweeps=max_sw, tol=1e-13, verbosity=0)
    return float(np.real(dmrg.energy))


def run_a2dmrg(L, mpo, chi, dtype=np.float64):
    energy, mps = a2dmrg_main(
        L=L, mpo=mpo, bond_dim=chi, max_sweeps=20, tol=1e-12,
        warmup_sweeps=2, finalize_sweeps=0,
        comm=comm, verbose=False, one_site=False,
        timing_report=False, dtype=dtype,
    )
    return energy


def bench_row(label, L, chi, mpo, dtype=np.float64):
    """Run all 3 solvers and print comparison."""
    # DMRG1 on rank 0
    if rank == 0:
        t0 = time.perf_counter()
        E1 = run_dmrg1(mpo, chi)
        t1 = time.perf_counter() - t0
    else:
        E1, t1 = None, None

    # DMRG2 on rank 0
    if rank == 0:
        t0 = time.perf_counter()
        E2 = run_dmrg2(mpo, chi)
        t2 = time.perf_counter() - t0
    else:
        E2, t2 = None, None

    # A2DMRG on all ranks
    comm.Barrier()
    t0 = time.perf_counter()
    Ea = run_a2dmrg(L, mpo, chi, dtype)
    comm.Barrier()
    ta = time.perf_counter() - t0

    if rank == 0:
        d1 = abs(Ea - E1)
        d2 = abs(Ea - E2)
        s1 = "PASS" if d1 < 1e-10 else ("WARN" if d1 < 1e-8 else "FAIL")
        s2 = "PASS" if d2 < 1e-10 else ("WARN" if d2 < 1e-8 else "FAIL")
        print(
            f"{label:>20s} L={L:<3d} chi={chi:<4d} | "
            f"DMRG1 {E1:16.10f} {t1:6.1f}s | "
            f"DMRG2 {E2:16.10f} {t2:6.1f}s | "
            f"A2DMRG {Ea:16.10f} {ta:6.1f}s | "
            f"d1={d1:.1e} {s1} d2={d2:.1e} {s2}"
        )


# ==============================
# Heisenberg (real, medium)
# ==============================
if rank == 0:
    print("=" * 160)
    print("HEISENBERG SPIN-1/2 CHAIN (real, OBC)")
    print("=" * 160)

for L, chi in [(32, 50), (32, 100), (48, 50), (48, 100), (64, 50), (64, 100)]:
    mpo = heisenberg_mpo(L)
    bench_row("Heisenberg", L, chi, mpo, np.float64)

# ==============================
# Josephson / Bose-Hubbard (complex128)
# ==============================
if rank == 0:
    print()
    print("=" * 160)
    print("BOSE-HUBBARD / JOSEPHSON (complex128, t=1+0.5j, U=4, mu=2, nmax=3, OBC)")
    print("=" * 160)

for L, chi in [(16, 30), (16, 50), (24, 30), (24, 50), (32, 30), (32, 50)]:
    mpo = bose_hubbard_mpo(L, t=1.0 + 0.5j, U=4.0, mu=2.0, nmax=3)
    bench_row("Josephson", L, chi, mpo, np.complex128)

if rank == 0:
    print()
    print("Target: |diff| < 1e-10 = PASS, < 1e-8 = WARN, else FAIL")
