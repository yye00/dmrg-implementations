#!/usr/bin/env python3
import sys
if "pytest" in sys.modules:
    import pytest
    pytest.skip("dev script (not a unit test)", allow_module_level=True)

"""
Test different numbers of warm-up sweeps to find the optimal value.

This script tests warmup_sweeps = 0, 1, 2, 3, 5 and measures:
1. Whether the algorithm converges correctly
2. Final energy accuracy vs quimb DMRG
3. Total time (warmup + A2DMRG sweeps)
"""

import sys
sys.path.insert(0, '.')

try:
    import numba
    from numba.core.dispatcher import Dispatcher
    Dispatcher.enable_caching = lambda self: None
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

# Test parameters
L = 10
bond_dim = 20
a2dmrg_sweeps = 5
tol = 1e-6

# Build Hamiltonian
builder = SpinHam1D(S=1/2)
builder += 1.0, "X", "X"
builder += 1.0, "Y", "Y"
builder += 1.0, "Z", "Z"
mpo = builder.build_mpo(L)

# Reference energy from quimb
if rank == 0:
    print("=" * 60)
    print(f"Testing warm-up sweeps for A2DMRG")
    print(f"L={L}, bond_dim={bond_dim}, np={size}")
    print("=" * 60)
    print()
    
    dmrg_ref = DMRG2(mpo, bond_dims=bond_dim)
    dmrg_ref.solve(tol=1e-10, verbosity=0)
    E_ref = dmrg_ref.energy.real
    print(f"Reference (quimb DMRG): E = {E_ref:.12f}")
    print()
else:
    E_ref = None

E_ref = comm.bcast(E_ref, root=0)

# Test different warm-up values
warmup_values = [0, 1, 2, 3, 5]
results = []

for warmup in warmup_values:
    if rank == 0:
        print(f"--- Testing warmup_sweeps = {warmup} ---")
    
    start = time.time()
    
    try:
        energy, mps = a2dmrg_main(
            L=L,
            mpo=mpo,
            max_sweeps=a2dmrg_sweeps,
            bond_dim=bond_dim,
            tol=tol,
            comm=comm,
            dtype=np.float64,
            one_site=True,
            verbose=False,
            warmup_sweeps=warmup
        )
        
        elapsed = time.time() - start
        diff = abs(energy - E_ref)
        success = diff < 1e-6
        
        if rank == 0:
            status = "✓ PASS" if success else "✗ FAIL"
            print(f"  Energy: {energy:.12f}")
            print(f"  Diff from ref: {diff:.3e} {status}")
            print(f"  Time: {elapsed:.1f}s")
            print()
            
            results.append({
                'warmup': warmup,
                'energy': energy,
                'diff': diff,
                'time': elapsed,
                'success': success
            })
    
    except Exception as e:
        elapsed = time.time() - start
        if rank == 0:
            print(f"  ERROR: {e}")
            print(f"  Time: {elapsed:.1f}s")
            print()
            results.append({
                'warmup': warmup,
                'energy': None,
                'diff': None,
                'time': elapsed,
                'success': False,
                'error': str(e)
            })

# Summary
if rank == 0:
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Warmup':>8} {'Energy':>18} {'Diff':>12} {'Time':>8} {'Status':>8}")
    print("-" * 60)
    
    for r in results:
        if r['energy'] is not None:
            status = "PASS" if r['success'] else "FAIL"
            print(f"{r['warmup']:>8} {r['energy']:>18.12f} {r['diff']:>12.3e} {r['time']:>8.1f}s {status:>8}")
        else:
            print(f"{r['warmup']:>8} {'ERROR':>18} {'-':>12} {r['time']:>8.1f}s {'FAIL':>8}")
    
    print("-" * 60)
    
    # Find optimal
    successful = [r for r in results if r['success']]
    if successful:
        # Prefer minimum warmup that works
        optimal = min(successful, key=lambda x: x['warmup'])
        print(f"\nRECOMMENDATION: warmup_sweeps = {optimal['warmup']}")
        print(f"  (Minimum sweeps that achieves < 1e-6 accuracy)")
    else:
        print("\nNo configuration achieved target accuracy!")
