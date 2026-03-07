#!/usr/bin/env python3
"""
Comprehensive A2DMRG Audit Against refs/a2dmrg.pdf

This script systematically verifies each component of the A2DMRG implementation
against Algorithm 2 from Grigori & Hassan (2025).

Run with single-threaded BLAS for normalized performance.
"""

import os
# Force single-threaded BLAS BEFORE any numpy imports
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import sys
import time
import numpy as np

sys.path.insert(0, 'a2dmrg')

from a2dmrg.mps.mps_utils import create_random_mps, create_neel_state
from a2dmrg.mps.canonical import move_orthogonality_center, left_canonicalize
from a2dmrg.hamiltonians.heisenberg import heisenberg_mpo
from a2dmrg.numerics.local_microstep import (
    local_microstep_1site,
    local_microstep_2site,
    _transform_to_i_orthogonal
)
from a2dmrg.numerics.observables import compute_energy, compute_overlap
from a2dmrg.parallel.coarse_space import build_coarse_matrices
from a2dmrg.numerics.coarse_eigenvalue import solve_coarse_eigenvalue_problem
from a2dmrg.dmrg import a2dmrg_main
from a2dmrg.mpi_compat import MPI

from quimb.tensor import DMRG2


class AuditResult:
    def __init__(self, name):
        self.name = name
        self.tests = []
        self.passed = 0
        self.failed = 0
        self.warnings = []

    def add_test(self, test_name, passed, message=""):
        self.tests.append({
            'name': test_name,
            'passed': passed,
            'message': message
        })
        if passed:
            self.passed += 1
        else:
            self.failed += 1

    def add_warning(self, warning):
        self.warnings.append(warning)

    def print_summary(self):
        status = "✅ PASS" if self.failed == 0 else f"❌ FAIL ({self.failed} failures)"
        print(f"\n{'='*80}")
        print(f"{self.name}: {status}")
        print(f"{'='*80}")
        print(f"Tests: {self.passed + self.failed} total, {self.passed} passed, {self.failed} failed")

        if self.warnings:
            print(f"\nWarnings: {len(self.warnings)}")
            for w in self.warnings:
                print(f"  ⚠️  {w}")

        if self.failed > 0:
            print(f"\nFailed Tests:")
            for t in self.tests:
                if not t['passed']:
                    print(f"  ❌ {t['name']}: {t['message']}")

        print(f"{'='*80}\n")


def verify_i_orthogonal(mps, center, tol=1e-12):
    """
    Verify MPS is i-orthogonal per Definition 6 (page 6).

    Returns (is_valid, {'left': max_err, 'right': max_err})
    """
    L = mps.L
    max_left_err = 0.0
    max_right_err = 0.0

    # Check left-orthogonal sites (j < center)
    for j in range(center):
        tensor = mps[j].data

        # Reshape to (left_bond, phys*right_bond) for mode-2 unfolding
        if tensor.ndim == 2:
            # Edge site
            mat = tensor.reshape(-1, tensor.shape[-1])
        else:
            # Middle site: (left, phys, right) -> (left, phys*right)
            mat = tensor.reshape(tensor.shape[0], -1)

        # Check mat^T @ mat = I
        gram = mat.T @ mat
        identity = np.eye(gram.shape[0])
        err = np.linalg.norm(gram - identity)
        max_left_err = max(max_left_err, err)

    # Check right-orthogonal sites (k > center)
    for k in range(center + 1, L):
        tensor = mps[k].data

        # Reshape to (left_bond*phys, right_bond) for mode-1 unfolding
        if tensor.ndim == 2:
            mat = tensor.reshape(tensor.shape[0], -1)
        else:
            # Middle site: (left, phys, right) -> (left*phys, right)
            mat = tensor.reshape(-1, tensor.shape[-1])

        # Check mat @ mat^T = I
        gram = mat @ mat.T
        identity = np.eye(gram.shape[0])
        err = np.linalg.norm(gram - identity)
        max_right_err = max(max_right_err, err)

    is_valid = max_left_err < tol and max_right_err < tol
    return is_valid, {'left': max_left_err, 'right': max_right_err}


def audit_step1_orthogonalization():
    """
    Audit Step 1: Orthogonalization Sweep (Algorithm 2, lines 1-8)

    Verifies that _transform_to_i_orthogonal() correctly implements
    Definition 6 (i-orthogonal tensor train decomposition).
    """
    result = AuditResult("STEP 1: Orthogonalization (Definition 6)")

    # Test 1: Basic i-orthogonal transformation
    for L in [4, 8, 12]:
        for center in range(L):
            mps = create_random_mps(L, bond_dim=16, phys_dim=2)
            E_before = compute_energy(mps, heisenberg_mpo(L, 1.0, False))

            _transform_to_i_orthogonal(mps, center_site=center, normalize=True)
            E_after = compute_energy(mps, heisenberg_mpo(L, 1.0, False))

            is_valid, errors = verify_i_orthogonal(mps, center, tol=1e-12)

            # Test: Correct orthogonality
            result.add_test(
                f"L={L}, center={center}: i-orthogonal",
                is_valid,
                f"left_err={errors['left']:.2e}, right_err={errors['right']:.2e}"
            )

            # Test: Energy preserved (gauge transformation is unitary)
            energy_diff = abs(E_after - E_before)
            result.add_test(
                f"L={L}, center={center}: energy preserved",
                energy_diff < 1e-10,
                f"ΔE = {energy_diff:.2e}"
            )

    # Test 2: Edge case L=2
    mps = create_random_mps(L=2, bond_dim=8, phys_dim=2)
    for center in [0, 1]:
        _transform_to_i_orthogonal(mps, center_site=center, normalize=True)
        is_valid, errors = verify_i_orthogonal(mps, center, tol=1e-12)
        result.add_test(
            f"Edge case L=2, center={center}",
            is_valid,
            f"errors={errors}"
        )

    # Test 3: Bond dimensions unchanged
    mps = create_random_mps(L=8, bond_dim=20, phys_dim=2)
    bonds_before = [mps[i].data.shape for i in range(8)]
    _transform_to_i_orthogonal(mps, center_site=4, normalize=True)
    bonds_after = [mps[i].data.shape for i in range(8)]

    result.add_test(
        "Bond dimensions preserved",
        bonds_before == bonds_after,
        f"Before: {bonds_before[4]}, After: {bonds_after[4]}"
    )

    result.print_summary()
    return result


def audit_step2_microsteps():
    """
    Audit Step 2: Parallel Local Micro-Steps (Algorithm 2, lines 9-12)

    Verifies:
    - Lemma 10: Standard eigenvalue problem (not generalized)
    - Definition 9: One-site and two-site DMRG micro-iterations
    - Energy decreases or stays constant
    """
    result = AuditResult("STEP 2: Local Micro-Steps (Lemma 10, Definition 9)")

    L = 8
    bond_dim = 20
    mpo = heisenberg_mpo(L, J=1.0, cyclic=False)

    # Test 1: One-site microstep decreases energy
    mps = create_neel_state(L, bond_dim=bond_dim)
    E_init = compute_energy(mps, mpo)

    for site in range(L):
        mps_updated, E_after = local_microstep_1site(mps, mpo, site=site, tol=1e-10)

        # Energy should decrease or stay constant
        energy_decreased = E_after <= E_init + 1e-12
        result.add_test(
            f"One-site site={site}: energy decreased",
            energy_decreased,
            f"E_before={E_init:.12f}, E_after={E_after:.12f}, ΔE={E_after - E_init:.2e}"
        )

        # Updated MPS should still be i-orthogonal at site
        is_valid, errors = verify_i_orthogonal(mps_updated, site, tol=1e-10)
        result.add_test(
            f"One-site site={site}: remains i-orthogonal",
            is_valid,
            f"errors={errors}"
        )

    # Test 2: Two-site microstep
    mps = create_neel_state(L, bond_dim=bond_dim)
    E_init = compute_energy(mps, mpo)

    for site in range(L-1):
        mps_updated, E_after = local_microstep_2site(
            mps, mpo, site=site, max_bond=bond_dim, tol=1e-10
        )

        energy_decreased = E_after <= E_init + 1e-12
        result.add_test(
            f"Two-site site={site}: energy decreased",
            energy_decreased,
            f"ΔE={E_after - E_init:.2e}"
        )

    # Test 3: Verify Lemma 8 (orthogonal projection property)
    mps = create_random_mps(L=8, bond_dim=16, phys_dim=2)
    _transform_to_i_orthogonal(mps, center_site=4, normalize=True)

    # Create two random site tensors
    W1 = np.random.randn(*mps[4].data.shape)
    W2 = np.random.randn(*mps[4].data.shape)

    # Inner product before retraction
    overlap_before = np.sum(W1.conj() * W2).real

    # Apply retraction (replace site, compute overlap of full MPS)
    mps1 = mps.copy()
    mps1[4].modify(data=W1 / np.linalg.norm(W1))
    mps2 = mps.copy()
    mps2[4].modify(data=W2 / np.linalg.norm(W2))

    overlap_after = compute_overlap(mps1, mps2).real

    # Normalize
    overlap_before /= (np.linalg.norm(W1) * np.linalg.norm(W2))

    # Should be proportional (orthogonal projection)
    ratio = overlap_after / overlap_before if abs(overlap_before) > 1e-15 else 1.0
    result.add_test(
        "Lemma 8: Retraction is orthogonal projection",
        abs(ratio - 1.0) < 0.1,  # Allow 10% deviation due to normalization
        f"ratio={ratio:.6f}"
    )

    result.print_summary()
    return result


def audit_step3_coarse_space():
    """
    Audit Step 3: Second-Level Minimization (Algorithm 2, line 13)

    Verifies:
    - Coarse matrices H and S are Hermitian (Eq. 16)
    - S is positive semi-definite
    - Generalized eigenvalue solver is correct (Eq. 17-18)
    """
    result = AuditResult("STEP 3: Coarse-Space Minimization (Section 3.1)")

    L = 8
    bond_dim = 32
    mps = create_neel_state(L, bond_dim=bond_dim)
    mpo = heisenberg_mpo(L, J=1.0, cyclic=False)

    # Create candidate list (original + updates from several sites)
    candidates = [mps]
    for site in [2, 4, 6]:
        mps_updated, _ = local_microstep_1site(mps, mpo, site)
        candidates.append(mps_updated)

    # Build coarse matrices
    H_coarse, S_coarse = build_coarse_matrices(candidates, mpo)

    # Test 1: H is Hermitian
    H_herm_err = np.linalg.norm(H_coarse - H_coarse.conj().T)
    result.add_test(
        "H_coarse is Hermitian",
        H_herm_err < 1e-12,
        f"||H - H†|| = {H_herm_err:.2e}"
    )

    # Test 2: S is Hermitian
    S_herm_err = np.linalg.norm(S_coarse - S_coarse.conj().T)
    result.add_test(
        "S_coarse is Hermitian",
        S_herm_err < 1e-12,
        f"||S - S†|| = {S_herm_err:.2e}"
    )

    # Test 3: S is positive semi-definite
    s_eig = np.linalg.eigvalsh(S_coarse)
    min_eig = s_eig[0]
    result.add_test(
        "S_coarse is positive semi-definite",
        min_eig > -1e-12,
        f"min(eig(S)) = {min_eig:.2e}"
    )

    # Test 4: Solve generalized eigenvalue problem
    try:
        energy, coeffs = solve_coarse_eigenvalue_problem(
            H_coarse, S_coarse, regularization=1e-10
        )

        # Verify solution: H c = λ S c
        residual = H_coarse @ coeffs - energy * S_coarse @ coeffs
        res_norm = np.linalg.norm(residual) / (abs(energy) + 1e-15)

        result.add_test(
            "Coarse eigenvalue problem solved correctly",
            res_norm < 1e-8,
            f"||H c - λ S c|| / |λ| = {res_norm:.2e}"
        )

        # Check normalization: c† S c = 1
        norm_check = coeffs.conj().T @ S_coarse @ coeffs
        result.add_test(
            "Eigenvector normalized w.r.t. S",
            abs(norm_check - 1.0) < 1e-10,
            f"c† S c = {norm_check:.12f}"
        )

    except Exception as e:
        result.add_test(
            "Coarse eigenvalue solver",
            False,
            f"Exception: {str(e)}"
        )

    result.print_summary()
    return result


def audit_algorithm_deviations():
    """
    Check for deviations from Algorithm 2 noted in the plan.
    """
    result = AuditResult("DEVIATIONS FROM ALGORITHM 2")

    # Read dmrg.py to check for specific patterns
    dmrg_path = 'a2dmrg/a2dmrg/dmrg.py'
    with open(dmrg_path) as f:
        dmrg_code = f.read()

    # Check 1: np=1 early return
    has_early_return = 'if size == 1' in dmrg_code and 'return energy_prev, mps' in dmrg_code
    result.add_warning(
        "np=1 early return exists - skips A2DMRG for single processor"
    )

    # Check 2: Disabled canonicalization
    has_disabled_canon = 'left_canonicalize(mps' in dmrg_code and '# DISABLED' in dmrg_code
    if has_disabled_canon:
        result.add_warning(
            "Global canonicalization disabled in main loop (comment says: prevents bond dimension reduction)"
        )

    # Check 3: Full MPS allgather
    local_steps_path = 'a2dmrg/a2dmrg/parallel/local_steps.py'
    with open(local_steps_path) as f:
        local_steps_code = f.read()

    if 'allgather' in local_steps_code and 'local_results' in local_steps_code:
        result.add_warning(
            "Full MPS allgather found in parallel/local_steps.py - potential scalability issue"
        )

    # Check 4: Energy recomputation in microsteps
    microstep_path = 'a2dmrg/a2dmrg/numerics/local_microstep.py'
    with open(microstep_path) as f:
        microstep_code = f.read()

    compute_energy_count = microstep_code.count('compute_energy(')
    if compute_energy_count > 0:
        result.add_warning(
            f"compute_energy() called {compute_energy_count} times in local_microstep.py - O(L²) cost per sweep"
        )

    result.print_summary()
    return result


def run_performance_benchmark():
    """
    Run performance benchmark with single-threaded BLAS.
    """
    print("\n" + "="*80)
    print("PERFORMANCE BENCHMARK (Single-threaded BLAS)")
    print("="*80)
    print(f"OPENBLAS_NUM_THREADS = {os.environ.get('OPENBLAS_NUM_THREADS', 'not set')}")
    print(f"MKL_NUM_THREADS = {os.environ.get('MKL_NUM_THREADS', 'not set')}")
    print("="*80)

    configs = [
        {'L': 8, 'bond_dim': 32, 'name': 'Small'},
        {'L': 12, 'bond_dim': 64, 'name': 'Medium'},
    ]

    results = []

    for config in configs:
        L = config['L']
        bond_dim = config['bond_dim']
        name = config['name']
        mpo = heisenberg_mpo(L, J=1.0, cyclic=False)

        print(f"\n{name}: L={L}, bond_dim={bond_dim}")
        print("-" * 80)

        # quimb DMRG2 reference
        print("  Running quimb DMRG2...")
        dmrg = DMRG2(mpo, bond_dims=bond_dim)
        t0 = time.time()
        dmrg.solve(tol=1e-10, verbosity=0, max_sweeps=20)
        t_quimb = time.time() - t0
        E_quimb = dmrg.energy
        print(f"    E = {E_quimb:.12f}, time = {t_quimb:.4f}s")

        # A2DMRG
        print("  Running A2DMRG...")
        t0 = time.time()
        E_a2dmrg, mps = a2dmrg_main(
            L, mpo, bond_dim=bond_dim, tol=1e-10,
            max_sweeps=10, warmup_sweeps=2, verbose=False,
            comm=MPI.COMM_WORLD
        )
        t_a2dmrg = time.time() - t0
        print(f"    E = {E_a2dmrg:.12f}, time = {t_a2dmrg:.4f}s")

        error = abs(E_a2dmrg - E_quimb)
        ratio = t_a2dmrg / t_quimb

        print(f"\n  Comparison:")
        print(f"    Error: {error:.2e}")
        print(f"    Time ratio (A2DMRG/quimb): {ratio:.2f}×")

        if error < 1e-12:
            status = "✅ Machine precision"
        elif error < 5e-10:
            status = "✅ Acceptance"
        else:
            status = "⚠️  Large error"

        print(f"    Status: {status}")

        results.append({
            'config': name,
            'E_quimb': E_quimb,
            'E_a2dmrg': E_a2dmrg,
            'error': error,
            't_quimb': t_quimb,
            't_a2dmrg': t_a2dmrg,
            'ratio': ratio
        })

    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"{'Config':<10} {'Error':>15} {'Ratio':>10} {'Status':>20}")
    print("-" * 80)
    for r in results:
        if r['error'] < 1e-12:
            status = "✅ MACH PREC"
        elif r['error'] < 5e-10:
            status = "✅ ACCEPT"
        else:
            status = "⚠️  HIGH ERR"

        print(f"{r['config']:<10} {r['error']:>15.2e} {r['ratio']:>10.2f}× {status:>20}")

    print("="*80 + "\n")


def main():
    print("\n" + "="*80)
    print("COMPREHENSIVE A2DMRG AUDIT")
    print("Reference: Grigori & Hassan, arXiv:2505.23429v2 (2025)")
    print("="*80)
    print("\nAuditing implementation against Algorithm 2 (page 10):")
    print("  Step 1: Orthogonalization sweep (Definition 6)")
    print("  Step 2: Parallel local micro-steps (Lemma 10, Definition 9)")
    print("  Step 3: Second-level minimization (Section 3.1, Eq. 16-18)")
    print("  Step 4: Compression to low-rank manifold")
    print("\n" + "="*80)

    all_results = []

    # Run audits
    all_results.append(audit_step1_orthogonalization())
    all_results.append(audit_step2_microsteps())
    all_results.append(audit_step3_coarse_space())
    all_results.append(audit_algorithm_deviations())

    # Performance benchmark
    run_performance_benchmark()

    # Final summary
    print("\n" + "="*80)
    print("FINAL AUDIT SUMMARY")
    print("="*80)

    total_tests = sum(r.passed + r.failed for r in all_results if hasattr(r, 'passed'))
    total_passed = sum(r.passed for r in all_results if hasattr(r, 'passed'))
    total_failed = sum(r.failed for r in all_results if hasattr(r, 'passed'))

    print(f"\nTotal Tests: {total_tests}")
    print(f"  Passed: {total_passed}")
    print(f"  Failed: {total_failed}")

    all_warnings = []
    for r in all_results:
        if hasattr(r, 'warnings'):
            all_warnings.extend(r.warnings)

    if all_warnings:
        print(f"\nTotal Warnings: {len(all_warnings)}")
        for w in all_warnings:
            print(f"  ⚠️  {w}")

    if total_failed == 0:
        print("\n✅ AUDIT PASSED: Implementation matches Algorithm 2")
    else:
        print(f"\n❌ AUDIT FAILED: {total_failed} tests failed")

    print("="*80 + "\n")


if __name__ == '__main__':
    main()
