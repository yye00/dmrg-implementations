#!/usr/bin/env python3
"""
Quick benchmark to validate the A2DMRG i-orthogonal transformation fix.

Compares quimb DMRG2 (reference) vs fixed A2DMRG on Heisenberg chain.
"""

import sys
import time
import numpy as np

# Add paths
sys.path.insert(0, 'a2dmrg')

from quimb.tensor import DMRG2
import quimb.tensor as qtn

from a2dmrg.hamiltonians.heisenberg import heisenberg_mpo
from a2dmrg.dmrg import a2dmrg_main
from a2dmrg.mpi_compat import MPI


def run_quimb_dmrg2(L, bond_dim, tol=1e-10, max_sweeps=20):
    """Run quimb DMRG2 as reference."""
    print(f"\n{'='*80}")
    print(f"Running quimb DMRG2 (reference) - L={L}, chi={bond_dim}")
    print(f"{'='*80}")

    # Build Hamiltonian
    mpo = heisenberg_mpo(L, J=1.0, cyclic=False)

    # Initialize
    dmrg = DMRG2(mpo, bond_dims=bond_dim)

    # Run
    start = time.time()
    dmrg.solve(tol=tol, verbosity=1, max_sweeps=max_sweeps)
    elapsed = time.time() - start

    energy = dmrg.energy
    converged = dmrg.opts.get('converged', True)
    sweeps = getattr(dmrg.state, 'sweep_number', 0)

    print(f"\nResults:")
    print(f"  Energy:    {energy:.15f}")
    print(f"  Time:      {elapsed:.4f} s")
    print(f"  Sweeps:    {sweeps}")
    print(f"  Converged: {converged}")

    return {
        'energy': energy,
        'time': elapsed,
        'sweeps': sweeps,
        'converged': converged
    }


def run_a2dmrg(L, bond_dim, tol=1e-10, max_sweeps=10, warmup_sweeps=2):
    """Run fixed A2DMRG implementation."""
    print(f"\n{'='*80}")
    print(f"Running A2DMRG (FIXED) - L={L}, chi={bond_dim}")
    print(f"{'='*80}")

    # Build Hamiltonian
    mpo = heisenberg_mpo(L, J=1.0, cyclic=False)

    # Run (np=1 will use warmup path)
    comm = MPI.COMM_WORLD

    start = time.time()
    energy, mps = a2dmrg_main(
        L=L,
        mpo=mpo,
        max_sweeps=max_sweeps,
        bond_dim=bond_dim,
        tol=tol,
        comm=comm,
        verbose=True,
        warmup_sweeps=warmup_sweeps,
        one_site=False  # two-site for comparison
    )
    elapsed = time.time() - start

    print(f"\nResults:")
    print(f"  Energy:    {energy:.15f}")
    print(f"  Time:      {elapsed:.4f} s")

    return {
        'energy': energy,
        'time': elapsed
    }


def compare_results(quimb_result, a2dmrg_result, system_name):
    """Compare and display results."""
    print(f"\n{'='*80}")
    print(f"COMPARISON: {system_name}")
    print(f"{'='*80}")

    quimb_E = quimb_result['energy']
    a2dmrg_E = a2dmrg_result['energy']

    delta_E = abs(a2dmrg_E - quimb_E)

    print(f"\n{'Method':<20} {'Energy':>22} {'Error vs quimb':>20} {'Time (s)':>12}")
    print(f"{'-'*80}")
    print(f"{'quimb DMRG2':<20} {quimb_E:>22.15f} {'(reference)':>20} {quimb_result['time']:>12.4f}")
    print(f"{'A2DMRG (FIXED)':<20} {a2dmrg_E:>22.15f} {delta_E:>20.2e} {a2dmrg_result['time']:>12.4f}")

    # Accuracy assessment
    print(f"\n{'Accuracy Assessment:'}")
    if delta_E < 1e-12:
        status = "✅ MACHINE PRECISION"
        print(f"  {status} (error < 1e-12)")
    elif delta_E < 5e-10:
        status = "✅ ACCEPTANCE"
        print(f"  {status} (error < 5e-10)")
    else:
        status = "❌ FAILED"
        print(f"  {status} (error >= 5e-10)")

    print(f"  |Error| = {delta_E:.2e}")

    return delta_E


def main():
    """Run validation benchmarks."""
    print("\n" + "="*80)
    print(" "*20 + "A2DMRG FIX VALIDATION BENCHMARK")
    print("="*80)
    print("\nTesting i-orthogonal transformation fix (Definition 6, Grigori & Hassan)")
    print("Comparing quimb DMRG2 (reference) vs fixed A2DMRG implementation")

    test_cases = [
        {'L': 8, 'bond_dim': 32, 'name': 'Heisenberg L=8'},
        {'L': 12, 'bond_dim': 64, 'name': 'Heisenberg L=12'},
    ]

    results = []

    for test in test_cases:
        L = test['L']
        bond_dim = test['bond_dim']
        name = test['name']

        # Run quimb DMRG2
        quimb_result = run_quimb_dmrg2(L, bond_dim, tol=1e-10, max_sweeps=20)

        # Run fixed A2DMRG
        a2dmrg_result = run_a2dmrg(L, bond_dim, tol=1e-10, max_sweeps=10, warmup_sweeps=2)

        # Compare
        error = compare_results(quimb_result, a2dmrg_result, name)

        results.append({
            'name': name,
            'L': L,
            'quimb_energy': quimb_result['energy'],
            'a2dmrg_energy': a2dmrg_result['energy'],
            'error': error,
            'quimb_time': quimb_result['time'],
            'a2dmrg_time': a2dmrg_result['time']
        })

    # Final summary
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"\n{'System':<20} {'|Error|':>15} {'Status':>20} {'A2DMRG/quimb time':>20}")
    print(f"{'-'*80}")

    for r in results:
        if r['error'] < 1e-12:
            status = "✅ MACHINE PREC"
        elif r['error'] < 5e-10:
            status = "✅ ACCEPTANCE"
        else:
            status = "❌ FAILED"

        time_ratio = r['a2dmrg_time'] / r['quimb_time']

        print(f"{r['name']:<20} {r['error']:>15.2e} {status:>20} {time_ratio:>20.2f}×")

    print(f"\n{'='*80}")
    print("FIX VALIDATION: ", end="")

    all_pass = all(r['error'] < 5e-10 for r in results)
    if all_pass:
        print("✅ PASSED - A2DMRG produces accurate results!")
    else:
        print("❌ FAILED - A2DMRG still has accuracy issues")

    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
