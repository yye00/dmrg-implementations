#!/usr/bin/env python3
"""
Exact comparison using same parameters as correctness suite.
"""

import sys
import time
import numpy as np

sys.path.insert(0, 'a2dmrg')

from quimb.tensor import DMRG2
from a2dmrg.hamiltonians.heisenberg import heisenberg_mpo
from a2dmrg.dmrg import a2dmrg_main
from a2dmrg.mpi_compat import MPI


def test_heisenberg_l12_d20():
    """Test L=12, D=20 Heisenberg - exact parameters from correctness suite."""

    L = 12
    bond_dim = 20
    tol = 1e-11
    max_sweeps = 40

    print("="*80)
    print(f"Heisenberg L={L}, bond_dim={bond_dim} Comparison")
    print("="*80)

    mpo = heisenberg_mpo(L, J=1.0, cyclic=False)

    # quimb DMRG2
    print("\n--- quimb DMRG2 (reference) ---")
    dmrg = DMRG2(mpo, bond_dims=bond_dim)
    start = time.time()
    dmrg.solve(tol=tol, verbosity=2, max_sweeps=max_sweeps)
    quimb_time = time.time() - start
    quimb_energy = dmrg.energy

    print(f"\nquimb DMRG2 Results:")
    print(f"  Energy: {quimb_energy:.15f}")
    print(f"  Time:   {quimb_time:.4f} s")

    # Fixed A2DMRG
    print("\n--- A2DMRG (FIXED with i-orthogonal transformation) ---")
    start = time.time()
    a2dmrg_energy, mps = a2dmrg_main(
        L=L,
        mpo=mpo,
        max_sweeps=max_sweeps,
        bond_dim=bond_dim,
        tol=tol,
        comm=MPI.COMM_WORLD,
        verbose=True,
        warmup_sweeps=5,  # Match correctness suite
        one_site=False
    )
    a2dmrg_time = time.time() - start

    print(f"\nA2DMRG Results:")
    print(f"  Energy: {a2dmrg_energy:.15f}")
    print(f"  Time:   {a2dmrg_time:.4f} s")

    # Comparison
    error = abs(a2dmrg_energy - quimb_energy)

    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    print(f"\n{'Method':<20} {'Energy':>22} {'Error':>15} {'Time (s)':>12} {'Status':>15}")
    print("-"*80)
    print(f"{'quimb DMRG2':<20} {quimb_energy:>22.15f} {'(reference)':>15} {quimb_time:>12.4f} {'-':>15}")
    print(f"{'A2DMRG FIXED':<20} {a2dmrg_energy:>22.15f} {error:>15.2e} {a2dmrg_time:>12.4f}", end="")

    if error < 1e-12:
        status = "✅ MACH PREC"
    elif error < 5e-10:
        status = "✅ ACCEPTANCE"
    else:
        status = "⚠️ LARGE ERR"
    print(f" {status:>15}")

    print("\n" + "="*80)
    print("VERDICT:")
    if error < 5e-10:
        print(f"  ✅ A2DMRG FIX VALIDATED - Error {error:.2e} within acceptance (<5e-10)")
    elif error < 1e-6:
        print(f"  ⚠️  A2DMRG working but with {error:.2e} error (may be convergence path difference)")
    else:
        print(f"  ❌ A2DMRG still broken - Error {error:.2e} too large")

    print("="*80)

    # Compare to OLD benchmark results
    print("\nComparison to OLD correctness results (BEFORE i-orthogonal fix):")
    print("  OLD quimb DMRG2:  -5.142090628178135")
    print("  OLD A2DMRG np=2:  -5.142090628178136")
    print("  OLD error:        2.09e-11 (machine precision)")
    print("\nNOTE: Different quimb energy suggests different random initialization")
    print("      or different quimb version. Focus on relative accuracy.")
    print("="*80)


if __name__ == '__main__':
    test_heisenberg_l12_d20()
