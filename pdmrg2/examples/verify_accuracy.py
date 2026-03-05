#!/usr/bin/env python
"""
Verify PDMRG accuracy against quimb DMRG2 reference.

This script compares PDMRG results to quimb's DMRG2 implementation
across all models to verify numerical accuracy.

Usage:
    python examples/verify_accuracy.py
"""

import subprocess
import sys
import time
import quimb.tensor as qtn

# Import our Hamiltonians
from pdmrg.hamiltonians.heisenberg import build_heisenberg_mpo
from pdmrg.hamiltonians.random_tfim import build_random_tfim_mpo
from pdmrg.hamiltonians.bose_hubbard import build_bose_hubbard_mpo


def run_quimb_dmrg(mpo, m, tol=1e-10, max_sweeps=20):
    """Run quimb DMRG2 as reference."""
    t0 = time.time()
    dmrg = qtn.DMRG2(mpo, bond_dims=[10, 20, m])
    dmrg.solve(tol=tol, max_sweeps=max_sweeps, verbosity=0)
    t1 = time.time()
    return dmrg.energy, t1 - t0


def run_pdmrg(model, L, m, np, tol=1e-10, sweeps=20):
    """Run PDMRG via subprocess."""
    cmd = f"mpirun --oversubscribe -np {np} python -m pdmrg " \
          f"--sites {L} --bond-dim {m} --warmup-dim {m} " \
          f"--model {model} --sweeps {sweeps} --tol {tol}"
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
    output = result.stdout + result.stderr
    
    energy = None
    for line in output.split('\n'):
        if 'Final energy:' in line:
            energy = float(line.split(':')[1].strip())
    
    return energy


def main():
    print("=" * 70)
    print("PDMRG Accuracy Verification")
    print("Comparing to quimb DMRG2 reference")
    print("=" * 70)
    
    # Test configurations
    tests = [
        ('heisenberg', 20, 30),
        ('heisenberg', 40, 50),
        ('random_tfim', 20, 30),
        ('random_tfim', 40, 50),
        ('josephson', 12, 30),  # Smaller L for Bose-Hubbard
    ]
    
    all_passed = True
    
    for model, L, m in tests:
        print(f"\n--- {model.upper()}: L={L}, m={m} ---")
        
        # Build MPO
        if model == 'heisenberg':
            mpo = build_heisenberg_mpo(L)
        elif model == 'random_tfim':
            mpo, _ = build_random_tfim_mpo(L, seed=42)
        else:
            mpo = build_bose_hubbard_mpo(L)
        
        # Run quimb reference
        print("  Running quimb DMRG2...", end=" ", flush=True)
        E_ref, t_ref = run_quimb_dmrg(mpo, m)
        print(f"E = {E_ref:.12f} ({t_ref:.2f}s)")
        
        # Test PDMRG with different processor counts
        for np in [1, 2, 4]:
            print(f"  PDMRG np={np}...", end=" ", flush=True)
            E_pdmrg = run_pdmrg(model, L, m, np)
            
            if E_pdmrg is None:
                print("FAILED (no output)")
                all_passed = False
                continue
            
            diff = abs(E_pdmrg - E_ref)
            rel_diff = diff / abs(E_ref)
            
            # Check accuracy (should be ~1e-10 or better)
            if rel_diff < 1e-8:
                status = "✓ PASS"
            elif rel_diff < 1e-6:
                status = "⚠ WARN"
            else:
                status = "✗ FAIL"
                all_passed = False
            
            print(f"E = {E_pdmrg:.12f}, ΔE = {diff:.2e}, rel = {rel_diff:.2e} {status}")
    
    print("\n" + "=" * 70)
    if all_passed:
        print("All tests PASSED ✓")
    else:
        print("Some tests FAILED ✗")
    print("=" * 70)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
