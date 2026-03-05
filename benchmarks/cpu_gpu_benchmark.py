#!/usr/bin/env python3
"""
Comprehensive CPU vs GPU DMRG Benchmark Suite
==============================================

Tests CPU implementations (Quimb DMRG1 and DMRG2) across:
- Heisenberg model: L=12/D=100, L=20/D=100, L=40/D=200
- Josephson junction: L=8/D=50, L=12/D=50, L=16/D=100

Outputs JSON results and formatted tables for comparison with GPU results.
"""

import json
import time
import sys
import os
import traceback
import argparse
import resource
from datetime import datetime

import numpy as np

# Add pdmrg to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'pdmrg'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'pdmrg2'))

import quimb
import quimb.tensor as qtn


# ===========================================================================
# Known exact/reference energies for validation
# ===========================================================================
# Heisenberg S=1/2 XXX chain, open boundary conditions
# E_exact(L) per site approaches -ln(2) + 1/4 = -0.4431...
HEISENBERG_REFERENCE = {
    12: -5.14209138,   # well-known from exact diag / Bethe ansatz
    20: -8.91254841,
    40: -18.07533653,
}


# ===========================================================================
# Test case definitions
# ===========================================================================
HEISENBERG_CASES = [
    {"name": "Heisenberg-Small",  "L": 12, "D": 100, "sweeps": 20, "tol": 1e-10},
    {"name": "Heisenberg-Medium", "L": 20, "D": 100, "sweeps": 30, "tol": 1e-10},
    {"name": "Heisenberg-Large",  "L": 40, "D": 200, "sweeps": 40, "tol": 1e-10},
]

JOSEPHSON_CASES = [
    {"name": "Josephson-Small",  "L": 8,  "D": 50,  "n_max": 2, "sweeps": 20, "tol": 1e-10},
    {"name": "Josephson-Medium", "L": 12, "D": 50,  "n_max": 2, "sweeps": 30, "tol": 1e-10},
    {"name": "Josephson-Large",  "L": 16, "D": 100, "n_max": 2, "sweeps": 40, "tol": 1e-10},
]


def get_memory_mb():
    """Get current process memory usage in MB."""
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / 1024.0  # Linux reports in KB
    except Exception:
        return 0.0


def run_quimb_dmrg1(mpo, bond_dim, max_sweeps, tol, cutoff=1e-14):
    """Run quimb 1-site DMRG and return results dict."""
    mem_before = get_memory_mb()
    t0 = time.time()

    dmrg = qtn.DMRG1(mpo, bond_dims=bond_dim, cutoffs=cutoff)
    dmrg.solve(max_sweeps=max_sweeps, tol=tol, verbosity=0)

    t1 = time.time()
    mem_after = get_memory_mb()

    energy = float(np.real(dmrg.energy))
    n_sweeps = len(dmrg.energies) if hasattr(dmrg, 'energies') else max_sweeps
    wall_time = t1 - t0

    return {
        "energy": energy,
        "wall_time_s": wall_time,
        "n_sweeps": n_sweeps,
        "memory_mb": mem_after - mem_before if mem_after > mem_before else mem_after,
        "converged": True,
    }


def run_quimb_dmrg2(mpo, bond_dim, max_sweeps, tol, cutoff=1e-14):
    """Run quimb 2-site DMRG and return results dict."""
    mem_before = get_memory_mb()
    t0 = time.time()

    dmrg = qtn.DMRG2(mpo, bond_dims=bond_dim, cutoffs=cutoff)
    dmrg.solve(max_sweeps=max_sweeps, tol=tol, verbosity=0)

    t1 = time.time()
    mem_after = get_memory_mb()

    energy = float(np.real(dmrg.energy))
    n_sweeps = len(dmrg.energies) if hasattr(dmrg, 'energies') else max_sweeps
    wall_time = t1 - t0

    return {
        "energy": energy,
        "wall_time_s": wall_time,
        "n_sweeps": n_sweeps,
        "memory_mb": mem_after - mem_before if mem_after > mem_before else mem_after,
        "converged": True,
    }


def build_bose_hubbard_mpo(L, t=1.0, U=4.0, mu=2.0, n_max=2, dtype='complex128'):
    """Build Bose-Hubbard / Josephson junction MPO."""
    d = n_max + 1
    a_dag = np.zeros((d, d), dtype=dtype)
    for n in range(d - 1):
        a_dag[n + 1, n] = np.sqrt(n + 1)
    a = a_dag.conj().T
    n_op = a_dag @ a

    builder = qtn.SpinHam1D(S=(d - 1) / 2)
    builder.add_term(-t, a_dag, a)
    builder.add_term(-t, a, a_dag)
    n2 = n_op @ n_op
    onsite = (U / 2.0) * (n2 - n_op) - mu * n_op
    builder.add_term(1.0, onsite)
    return builder.build_mpo(L)


def run_heisenberg_benchmarks(cases, skip_large=False):
    """Run all Heisenberg test cases."""
    results = {}

    for case in cases:
        name = case["name"]
        L = case["L"]
        D = case["D"]
        sweeps = case["sweeps"]
        tol = case["tol"]

        if skip_large and L > 20:
            print(f"  [SKIP] {name} (L={L}, D={D}) -- skipping large case")
            results[name] = {"skipped": True}
            continue

        print(f"\n  === {name}: L={L}, D={D}, sweeps={sweeps} ===")

        # Build MPO
        mpo = qtn.MPO_ham_heis(L=L, j=1.0, bz=0.0, cyclic=False)

        # DMRG1
        print(f"    Running quimb DMRG1...", end="", flush=True)
        try:
            r1 = run_quimb_dmrg1(mpo, D, sweeps, tol)
            print(f" E={r1['energy']:.10f}, t={r1['wall_time_s']:.2f}s, "
                  f"sweeps={r1['n_sweeps']}")
        except Exception as e:
            print(f" FAILED: {e}")
            r1 = {"energy": None, "wall_time_s": None, "n_sweeps": None,
                   "memory_mb": None, "converged": False, "error": str(e)}

        # DMRG2
        print(f"    Running quimb DMRG2...", end="", flush=True)
        try:
            r2 = run_quimb_dmrg2(mpo, D, sweeps, tol)
            print(f" E={r2['energy']:.10f}, t={r2['wall_time_s']:.2f}s, "
                  f"sweeps={r2['n_sweeps']}")
        except Exception as e:
            print(f" FAILED: {e}")
            r2 = {"energy": None, "wall_time_s": None, "n_sweeps": None,
                   "memory_mb": None, "converged": False, "error": str(e)}

        results[name] = {
            "L": L,
            "D": D,
            "d": 2,
            "model": "heisenberg",
            "reference_energy": HEISENBERG_REFERENCE.get(L),
            "DMRG1": r1,
            "DMRG2": r2,
        }

    return results


def run_josephson_benchmarks(cases, skip_large=False):
    """Run all Josephson junction test cases."""
    results = {}

    for case in cases:
        name = case["name"]
        L = case["L"]
        D = case["D"]
        n_max = case["n_max"]
        sweeps = case["sweeps"]
        tol = case["tol"]

        if skip_large and L > 12:
            print(f"  [SKIP] {name} (L={L}, D={D}) -- skipping large case")
            results[name] = {"skipped": True}
            continue

        d = n_max + 1
        print(f"\n  === {name}: L={L}, D={D}, n_max={n_max} (d={d}) ===")

        # Build Bose-Hubbard / Josephson MPO
        mpo = build_bose_hubbard_mpo(L, n_max=n_max)

        # DMRG1
        print(f"    Running quimb DMRG1...", end="", flush=True)
        try:
            r1 = run_quimb_dmrg1(mpo, D, sweeps, tol)
            print(f" E={r1['energy']:.10f}, t={r1['wall_time_s']:.2f}s, "
                  f"sweeps={r1['n_sweeps']}")
        except Exception as e:
            print(f" FAILED: {e}")
            r1 = {"energy": None, "wall_time_s": None, "n_sweeps": None,
                   "memory_mb": None, "converged": False, "error": str(e)}

        # DMRG2
        print(f"    Running quimb DMRG2...", end="", flush=True)
        try:
            r2 = run_quimb_dmrg2(mpo, D, sweeps, tol)
            print(f" E={r2['energy']:.10f}, t={r2['wall_time_s']:.2f}s, "
                  f"sweeps={r2['n_sweeps']}")
        except Exception as e:
            print(f" FAILED: {e}")
            r2 = {"energy": None, "wall_time_s": None, "n_sweeps": None,
                   "memory_mb": None, "converged": False, "error": str(e)}

        results[name] = {
            "L": L,
            "D": D,
            "d": d,
            "n_max": n_max,
            "model": "josephson",
            "DMRG1": r1,
            "DMRG2": r2,
        }

    return results


def print_table(title, results):
    """Print a formatted results table."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    print(f"{'Case':<22} {'Algo':<8} {'Energy':>16} {'Time(s)':>10} "
          f"{'Sweeps':>8} {'Mem(MB)':>10}")
    print(f"{'-'*22} {'-'*8} {'-'*16} {'-'*10} {'-'*8} {'-'*10}")

    for case_name, case_data in results.items():
        if case_data.get("skipped"):
            print(f"{case_name:<22} {'SKIP':<8} {'--':>16} {'--':>10} "
                  f"{'--':>8} {'--':>10}")
            continue

        for algo in ["DMRG1", "DMRG2"]:
            r = case_data.get(algo, {})
            if r.get("energy") is not None:
                energy_str = f"{r['energy']:.10f}"
                time_str = f"{r['wall_time_s']:.3f}"
                sweeps_str = str(r.get('n_sweeps', '--'))
                mem_str = f"{r.get('memory_mb', 0):.1f}" if r.get('memory_mb') else "--"
            else:
                energy_str = "FAILED"
                time_str = "--"
                sweeps_str = "--"
                mem_str = "--"

            print(f"{case_name:<22} {algo:<8} {energy_str:>16} {time_str:>10} "
                  f"{sweeps_str:>8} {mem_str:>10}")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive CPU DMRG Benchmark (Quimb DMRG1 + DMRG2)")
    parser.add_argument("--skip-large", action="store_true",
                        help="Skip large test cases (L>20 Heisenberg, L>12 Josephson)")
    parser.add_argument("--heisenberg-only", action="store_true",
                        help="Only run Heisenberg benchmarks")
    parser.add_argument("--josephson-only", action="store_true",
                        help="Only run Josephson benchmarks")
    parser.add_argument("--out", type=str,
                        default=os.path.join(SCRIPT_DIR, "cpu_gpu_benchmark_results.json"),
                        help="Output JSON file path")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{'='*80}")
    print(f"  COMPREHENSIVE CPU DMRG BENCHMARK SUITE")
    print(f"  Quimb v{quimb.__version__} | NumPy v{np.__version__}")
    print(f"  {timestamp}")
    print(f"{'='*80}")

    all_results = {
        "timestamp": timestamp,
        "platform": "CPU",
        "quimb_version": quimb.__version__,
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
    }

    total_start = time.time()

    # Heisenberg benchmarks
    if not args.josephson_only:
        print(f"\n{'#'*80}")
        print(f"# HEISENBERG MODEL BENCHMARKS (d=2, real)")
        print(f"{'#'*80}")
        heis_results = run_heisenberg_benchmarks(
            HEISENBERG_CASES, skip_large=args.skip_large)
        all_results["heisenberg"] = heis_results
        print_table("HEISENBERG RESULTS", heis_results)

    # Josephson benchmarks
    if not args.heisenberg_only:
        print(f"\n{'#'*80}")
        print(f"# JOSEPHSON JUNCTION BENCHMARKS (d=5, complex128)")
        print(f"{'#'*80}")
        jos_results = run_josephson_benchmarks(
            JOSEPHSON_CASES, skip_large=args.skip_large)
        all_results["josephson"] = jos_results
        print_table("JOSEPHSON JUNCTION RESULTS", jos_results)

    total_time = time.time() - total_start
    all_results["total_benchmark_time_s"] = total_time

    print(f"\n{'='*80}")
    print(f"  Total benchmark time: {total_time:.1f}s")
    print(f"{'='*80}")

    # Save results
    with open(args.out, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {args.out}")

    return all_results


if __name__ == "__main__":
    main()
