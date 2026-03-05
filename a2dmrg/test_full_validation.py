#!/usr/bin/env python3
import sys
if "pytest" in sys.modules:
    import pytest
    pytest.skip("dev script (not a unit test)", allow_module_level=True)

"""
Full A2DMRG Validation Suite
============================
1. Compare A2DMRG (np=2) against quimb DMRG for various system sizes
2. Verify energies match to machine precision
3. Scalability study for different np values
"""

import sys
sys.path.insert(0, '.')

# Disable numba caching to avoid MPI conflicts
try:
    import numba
    from numba.core.dispatcher import Dispatcher
    from numba.np.ufunc import ufuncbuilder
    Dispatcher.enable_caching = lambda self: None
    ufuncbuilder.UFuncDispatcher.enable_caching = lambda self: None
except Exception:
    pass

import numpy as np
import time
from a2dmrg.mpi_compat import MPI, HAS_MPI
from quimb.tensor import SpinHam1D, DMRG2
from a2dmrg.dmrg import a2dmrg_main

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def run_quimb_dmrg(L, bond_dim, tol=1e-10):
    """Run reference quimb DMRG."""
    builder = SpinHam1D(S=1/2)
    builder += 1.0, "X", "X"
    builder += 1.0, "Y", "Y"
    builder += 1.0, "Z", "Z"
    mpo = builder.build_mpo(L)
    
    dmrg = DMRG2(mpo, bond_dims=bond_dim)
    start = time.time()
    dmrg.solve(tol=tol, verbosity=0)
    elapsed = time.time() - start
    return dmrg.energy, elapsed, mpo


def run_a2dmrg(L, mpo, bond_dim, comm, tol=1e-10, max_sweeps=30):
    """Run A2DMRG."""
    start = time.time()
    energy, mps = a2dmrg_main(
        L=L,
        mpo=mpo,
        max_sweeps=max_sweeps,
        bond_dim=bond_dim,
        tol=tol,
        comm=comm,
        dtype=np.float64,
        one_site=True,
        verbose=False
    )
    elapsed = time.time() - start
    return energy, elapsed


def test_accuracy():
    """Test A2DMRG accuracy against quimb for various system sizes."""
    if rank == 0:
        print("=" * 70)
        print(f"A2DMRG vs Quimb DMRG Accuracy Test (np={size})")
        print("=" * 70)
        print()
    
    # Test cases: (L, bond_dim)
    test_cases = [
        (6, 16),
        (10, 32),
        (20, 50),
        (40, 80),
    ]
    
    results = []
    
    for L, bond_dim in test_cases:
        if rank == 0:
            print(f"\n--- L={L}, bond_dim={bond_dim} ---")
            
            # Run quimb (only on rank 0)
            E_quimb, t_quimb, mpo = run_quimb_dmrg(L, bond_dim)
            print(f"Quimb DMRG:  E = {E_quimb:.15f}  (t={t_quimb:.2f}s)")
        else:
            E_quimb = None
            mpo = None
            t_quimb = None
        
        # Broadcast MPO and quimb energy to all ranks
        mpo = comm.bcast(mpo, root=0)
        E_quimb = comm.bcast(E_quimb, root=0)
        
        # Run A2DMRG on all ranks
        E_a2dmrg, t_a2dmrg = run_a2dmrg(L, mpo, bond_dim, comm)
        
        if rank == 0:
            diff = abs(E_a2dmrg - E_quimb)
            status = "✓ PASS" if diff < 1e-8 else "✗ FAIL"
            print(f"A2DMRG:      E = {E_a2dmrg:.15f}  (t={t_a2dmrg:.2f}s)")
            print(f"Difference:  {diff:.3e}  {status}")
            
            results.append({
                'L': L,
                'bond_dim': bond_dim,
                'E_quimb': E_quimb,
                'E_a2dmrg': E_a2dmrg,
                'diff': diff,
                't_quimb': t_quimb,
                't_a2dmrg': t_a2dmrg,
                'pass': diff < 1e-8
            })
    
    if rank == 0:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"{'L':>4} {'χ':>4} {'E_quimb':>20} {'E_a2dmrg':>20} {'Δ':>12} {'Status':>8}")
        print("-" * 70)
        
        all_pass = True
        for r in results:
            status = "PASS" if r['pass'] else "FAIL"
            if not r['pass']:
                all_pass = False
            print(f"{r['L']:>4} {r['bond_dim']:>4} {r['E_quimb']:>20.12f} {r['E_a2dmrg']:>20.12f} {r['diff']:>12.3e} {status:>8}")
        
        print("-" * 70)
        if all_pass:
            print("ALL TESTS PASSED ✓")
        else:
            print("SOME TESTS FAILED ✗")
        print()
        
        return all_pass
    return None


def main():
    all_pass = test_accuracy()
    
    if rank == 0 and all_pass is not None:
        if not all_pass:
            sys.exit(1)


if __name__ == "__main__":
    main()
