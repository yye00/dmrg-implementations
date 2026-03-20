"""Paper-faithful benchmark: reproduce Grigori-Hassan arXiv:2505.23429 experiments.

Tests A2DMRG vs standard DMRG2 on the H6 linear hydrogen chain (d=12 spin orbitals,
STO-3G basis, 1 Angstrom spacing) — the smallest system in the paper.

Paper settings:
- Two-site mode
- No warmup (random initialization)
- Tolerance: 1e-6 (relative energy difference)
- Bond dimensions: r = 16, 32, 48
- Reference: PySCF FCI energy

Usage:
    mpirun -np 2 python bench_paper_faithful.py
    mpirun -np 4 python bench_paper_faithful.py
    mpirun -np 11 python bench_paper_faithful.py  # d-1 = 11, paper's ideal
"""

import time
import numpy as np
import quimb.tensor as qtn
from pyscf import gto, scf, fci
from openfermion import MolecularData, jordan_wigner, get_sparse_operator
from openfermionpyscf import run_pyscf
from scipy.sparse.linalg import eigsh

from a2dmrg.dmrg import a2dmrg_main
from a2dmrg.mpi_compat import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def build_h6_system():
    """Build H6 linear chain Hamiltonian (paper's smallest test case).

    Returns (mpo, e_exact, n_qubits).
    """
    geometry = [('H', (0, 0, i * 1.0)) for i in range(6)]
    mol_data = MolecularData(geometry, 'sto-3g', 1, 0)
    mol_data = run_pyscf(mol_data, run_fci=True)

    ham_ferm = mol_data.get_molecular_hamiltonian()
    ham_jw = jordan_wigner(ham_ferm)

    n_qubits = mol_data.n_qubits  # 12
    ham_sparse = get_sparse_operator(ham_jw, n_qubits=n_qubits)

    # Exact ground state
    e_exact, _ = eigsh(ham_sparse, k=1, which='SA')
    e_exact = float(e_exact[0])

    # Build MPO from dense (feasible: 4096x4096 = 134 MB)
    # Force real — H is Hermitian with negligible imaginary parts from numerics
    ham_dense = np.real(ham_sparse.toarray())
    mpo = qtn.MatrixProductOperator.from_dense(
        ham_dense, dims=[2] * n_qubits,
        upper_ind_id='k{}', lower_ind_id='b{}'
    )
    mpo.compress(cutoff=1e-12)

    return mpo, e_exact, n_qubits


def run_dmrg2(mpo, chi, tol=1e-6, max_sw=20):
    """Run quimb DMRG2 (serial, reference)."""
    dmrg = qtn.DMRG2(mpo, bond_dims=chi, cutoffs=tol)
    dmrg.solve(max_sweeps=max_sw, tol=tol, verbosity=0)
    return float(np.real(dmrg.energy))


def run_a2dmrg_paper(L, mpo, chi, comm, warmup=0, max_sw=20, tol=1e-6):
    """Run A2DMRG with paper-faithful settings."""
    energy, mps = a2dmrg_main(
        L=L, mpo=mpo, bond_dim=chi,
        max_sweeps=max_sw,
        tol=tol,
        warmup_sweeps=warmup,
        finalize_sweeps=0,
        comm=comm,
        verbose=(rank == 0),
        one_site=False,  # two-site (paper default)
        timing_report=False,
    )
    return energy


def run_a2dmrg_practical(L, mpo, chi, comm, warmup=2, max_sw=20, tol=1e-10):
    """Run A2DMRG with our practical defaults."""
    energy, mps = a2dmrg_main(
        L=L, mpo=mpo, bond_dim=chi,
        max_sweeps=max_sw,
        tol=tol,
        warmup_sweeps=warmup,
        finalize_sweeps=0,
        comm=comm,
        verbose=False,
        one_site=False,
        timing_report=False,
    )
    return energy


# ============================================================
# Build system
# ============================================================
if rank == 0:
    print("=" * 120)
    print("PAPER-FAITHFUL BENCHMARK: H6 Linear Chain (d=12, STO-3G, 1 Å)")
    print(f"Reproducing Grigori-Hassan arXiv:2505.23429, Figure 1")
    print(f"MPI ranks: {size}")
    print("=" * 120)
    print("\nBuilding H6 quantum chemistry MPO...")

t0 = time.perf_counter()
mpo, e_exact, L = build_h6_system()
t_build = time.perf_counter() - t0

if rank == 0:
    print(f"  FCI exact energy: {e_exact:.12f}")
    print(f"  d (sites): {L}")
    print(f"  MPO bond dims: {mpo.bond_sizes()}")
    print(f"  Build time: {t_build:.1f}s")

# ============================================================
# Benchmark: Paper bond dimensions (r = 16, 32, 48)
# ============================================================
if rank == 0:
    print(f"\n{'='*120}")
    print(f"{'Method':>25s} {'r':>4s} | {'Energy':>20s} {'Rel Error':>12s} {'Abs Error':>12s} {'Time':>8s} | {'Status'}")
    print(f"{'-'*120}")

for chi in [16, 32, 48]:
    # --- quimb DMRG2 (rank 0 only) ---
    if rank == 0:
        t0 = time.perf_counter()
        e_dmrg2 = run_dmrg2(mpo, chi, tol=1e-6)
        t_dmrg2 = time.perf_counter() - t0
        rel_d2 = abs(e_dmrg2 - e_exact) / abs(e_exact)
        abs_d2 = abs(e_dmrg2 - e_exact)
        s_d2 = "PASS" if rel_d2 < 1e-4 else ("WARN" if rel_d2 < 1e-2 else "FAIL")
        print(f"{'DMRG2 (quimb)':>25s} r={chi:<3d} | {e_dmrg2:20.12f} {rel_d2:12.2e} {abs_d2:12.2e} {t_dmrg2:7.1f}s | {s_d2}")
    else:
        e_dmrg2, t_dmrg2 = None, None

    # --- A2DMRG paper-faithful (warmup=0) ---
    comm.Barrier()
    t0 = time.perf_counter()
    e_a2_paper = run_a2dmrg_paper(L, mpo, chi, comm, warmup=0, tol=1e-6)
    comm.Barrier()
    t_a2_paper = time.perf_counter() - t0

    if rank == 0:
        rel_a2p = abs(e_a2_paper - e_exact) / abs(e_exact)
        abs_a2p = abs(e_a2_paper - e_exact)
        s_a2p = "PASS" if rel_a2p < 1e-4 else ("WARN" if rel_a2p < 1e-2 else "FAIL")
        print(f"{'A2DMRG paper (wu=0)':>25s} r={chi:<3d} | {e_a2_paper:20.12f} {rel_a2p:12.2e} {abs_a2p:12.2e} {t_a2_paper:7.1f}s | {s_a2p}")

    # --- A2DMRG practical (warmup=2) ---
    comm.Barrier()
    t0 = time.perf_counter()
    e_a2_prac = run_a2dmrg_practical(L, mpo, chi, comm, warmup=2, tol=1e-10)
    comm.Barrier()
    t_a2_prac = time.perf_counter() - t0

    if rank == 0:
        rel_a2r = abs(e_a2_prac - e_exact) / abs(e_exact)
        abs_a2r = abs(e_a2_prac - e_exact)
        s_a2r = "PASS" if rel_a2r < 1e-4 else ("WARN" if rel_a2r < 1e-2 else "FAIL")
        print(f"{'A2DMRG practical (wu=2)':>25s} r={chi:<3d} | {e_a2_prac:20.12f} {rel_a2r:12.2e} {abs_a2r:12.2e} {t_a2_prac:7.1f}s | {s_a2r}")

    if rank == 0:
        print()

if rank == 0:
    print("=" * 120)
    print("Paper thresholds: rel_err < 1e-4 = PASS, < 1e-2 = WARN, else FAIL")
    print(f"Paper uses tolerance 1e-6 (relative), we test both paper-faithful and practical modes")
    print(f"E_exact (FCI) = {e_exact:.12f}")
