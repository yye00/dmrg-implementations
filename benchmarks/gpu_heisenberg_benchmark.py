#!/usr/bin/env python3
"""
GPU Multi-Stream Heisenberg Benchmark

Tests GPU DMRG implementation with varying number of streams:
- num_streams = 1, 2, 4, 8

Compares against quimb DMRG2 reference with 1e-10 accuracy target.

Reports:
- Energy accuracy (vs quimb DMRG2)
- Timing per stream count
- Linear scalability metrics
- Parallel efficiency

Usage:
    python gpu_heisenberg_benchmark.py --L 8 --chi 32 --streams 1,2,4,8
"""

import json
import time
import subprocess
import sys
import os
import argparse
import numpy as np
from pathlib import Path

# Configuration defaults
L = 8
CHI_MAX = 32
MAX_ITERATIONS = 20
TOL = 1e-10  # Target accuracy
PASS_TOL = 1e-10  # Same as solver tolerance

# GPU executable path
GPU_PORT_DIR = Path(__file__).parent.parent / "gpu-port"
BUILD_DIR = GPU_PORT_DIR / "build"


def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="GPU Multi-Stream Heisenberg Benchmark")
    p.add_argument("--L", type=int, default=L, help="Chain length (must be even for multi-stream)")
    p.add_argument("--chi", type=int, default=CHI_MAX, help="Maximum bond dimension")
    p.add_argument("--max-iter", type=int, default=MAX_ITERATIONS, help="Max DMRG iterations")
    p.add_argument("--tol", type=float, default=TOL, help="Energy convergence tolerance")
    p.add_argument("--pass-tol", type=float, default=PASS_TOL, help="Pass/fail threshold vs quimb")
    p.add_argument("--streams", type=str, default="1,2,4,8", help="Comma-separated stream counts")
    p.add_argument("--out", type=str, default=None, help="Write JSON results to this path")
    p.add_argument("--gpu-exe", type=str, default=None, help="Path to GPU test executable")
    p.add_argument("--check-speedup", action="store_true", help="Require linear scalability")
    p.add_argument("--min-efficiency", type=float, default=0.70, help="Minimum parallel efficiency")
    return p.parse_args()


def run_quimb_reference(L, chi_max, max_sweeps, tol):
    """Run quimb DMRG2 as reference."""
    import quimb.tensor as qtn

    print(f"\n{'='*70}")
    print(f"Running Quimb DMRG2 Reference (L={L}, chi={chi_max})")
    print(f"{'='*70}")

    mpo = qtn.MPO_ham_heis(L=L, j=1.0, bz=0.0, cyclic=False)

    t0 = time.time()
    dmrg2 = qtn.DMRG2(mpo, bond_dims=chi_max, cutoffs=1e-14)
    dmrg2.solve(max_sweeps=max_sweeps, tol=tol, verbosity=1)
    t1 = time.time()

    energy = float(np.real(dmrg2.energy))
    elapsed = t1 - t0
    sweeps = len(dmrg2.energies) if hasattr(dmrg2, 'energies') else 'N/A'

    print(f"\n✓ Quimb DMRG2: E = {energy:.12f}")
    print(f"  Time: {elapsed:.3f}s, Sweeps: {sweeps}")

    result = {
        'implementation': 'quimb_DMRG2',
        'num_streams': 'N/A',
        'energy': energy,
        'time': elapsed,
        'sweeps': sweeps,
        'delta_E': 0.0,
        'passed': True
    }

    return result, energy


def run_gpu_dmrg(num_streams, L, chi_max, max_iter, ref_energy, gpu_exe):
    """Run GPU DMRG with specified number of streams."""

    print(f"\n{'='*70}")
    print(f"Running GPU DMRG (L={L}, chi={chi_max}, streams={num_streams})")
    print(f"{'='*70}")

    # Check executable exists
    if not os.path.exists(gpu_exe):
        raise FileNotFoundError(f"GPU executable not found: {gpu_exe}")

    # Build command
    # Assume test_heisenberg_multistream takes: L chi_max num_streams max_iterations
    cmd = [
        str(gpu_exe),
        str(L),
        str(chi_max),
        str(num_streams),
        str(max_iter)
    ]

    print(f"Command: {' '.join(cmd)}")

    try:
        t0 = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            cwd=BUILD_DIR
        )
        t1 = time.time()
        elapsed = t1 - t0

        if result.returncode != 0:
            print(f"❌ GPU DMRG failed with return code {result.returncode}")
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            return {
                'implementation': f'GPU_DMRG',
                'num_streams': num_streams,
                'energy': None,
                'time': elapsed,
                'error': result.stderr,
                'delta_E': None,
                'passed': False
            }

        # Parse energy from output
        # Expected format: "Final Energy: -X.XXXXXXXXX" or "Energy: -X.XXXXXXXXX"
        energy = None
        iterations = None

        for line in result.stdout.split('\n'):
            if 'Final Energy:' in line or 'Energy:' in line:
                try:
                    energy = float(line.split(':')[-1].strip())
                except:
                    pass
            if 'Iteration' in line and 'Energy:' in line:
                # Count iterations
                if iterations is None:
                    iterations = 0
                iterations += 1

        if energy is None:
            # Try to find last "Energy: X.XXX" pattern
            import re
            matches = re.findall(r'Energy:\s*([-+]?\d+\.\d+)', result.stdout)
            if matches:
                energy = float(matches[-1])

        if energy is None:
            print(f"❌ Could not parse energy from output")
            print(f"STDOUT:\n{result.stdout}")
            return {
                'implementation': f'GPU_DMRG',
                'num_streams': num_streams,
                'energy': None,
                'time': elapsed,
                'iterations': iterations,
                'error': 'Could not parse energy',
                'delta_E': None,
                'passed': False
            }

        # Compute accuracy
        delta_E = energy - ref_energy
        passed = abs(delta_E) < PASS_TOL

        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"\n{status} GPU DMRG (streams={num_streams}): E = {energy:.12f}")
        print(f"  ΔE = {delta_E:.2e}, Time: {elapsed:.3f}s, Iterations: {iterations}")

        return {
            'implementation': f'GPU_DMRG',
            'num_streams': num_streams,
            'energy': energy,
            'time': elapsed,
            'iterations': iterations,
            'delta_E': delta_E,
            'passed': passed
        }

    except subprocess.TimeoutExpired:
        print(f"❌ GPU DMRG timed out (>300s)")
        return {
            'implementation': f'GPU_DMRG',
            'num_streams': num_streams,
            'energy': None,
            'time': 300.0,
            'error': 'Timeout',
            'delta_E': None,
            'passed': False
        }
    except Exception as e:
        print(f"❌ GPU DMRG error: {e}")
        return {
            'implementation': f'GPU_DMRG',
            'num_streams': num_streams,
            'energy': None,
            'time': None,
            'error': str(e),
            'delta_E': None,
            'passed': False
        }


def compute_scalability_metrics(results, stream_counts):
    """Compute parallel efficiency and speedup metrics."""

    # Find baseline (single stream) time
    baseline = None
    for r in results:
        if r['num_streams'] == 1 and r['time'] is not None:
            baseline = r['time']
            break

    if baseline is None:
        print("⚠️  No baseline (1-stream) timing available for scalability analysis")
        return {}

    metrics = {
        'baseline_time': baseline,
        'speedups': {},
        'efficiencies': {}
    }

    print(f"\n{'='*70}")
    print(f"Scalability Analysis (baseline: {baseline:.3f}s @ 1 stream)")
    print(f"{'='*70}")
    print(f"{'Streams':<10} {'Time (s)':<12} {'Speedup':<12} {'Efficiency':<12} {'Status'}")
    print(f"{'-'*70}")

    for n_streams in stream_counts:
        # Find result for this stream count
        result = None
        for r in results:
            if r['num_streams'] == n_streams and r['time'] is not None:
                result = r
                break

        if result is None:
            print(f"{n_streams:<10} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'❌ Missing'}")
            continue

        t = result['time']
        speedup = baseline / t
        efficiency = speedup / n_streams

        metrics['speedups'][n_streams] = speedup
        metrics['efficiencies'][n_streams] = efficiency

        # Status based on efficiency
        if efficiency >= 0.80:
            status = "✓ Excellent"
        elif efficiency >= 0.60:
            status = "✓ Good"
        elif efficiency >= 0.40:
            status = "⚠️  Fair"
        else:
            status = "❌ Poor"

        print(f"{n_streams:<10} {t:<12.3f} {speedup:<12.2f} {efficiency:<12.1%} {status}")

    print(f"{'-'*70}")

    return metrics


def main():
    args = parse_args()

    # Parse stream counts
    stream_counts = [int(x.strip()) for x in args.streams.split(',')]

    # Determine GPU executable
    if args.gpu_exe:
        gpu_exe = Path(args.gpu_exe)
    else:
        gpu_exe = BUILD_DIR / "test_heisenberg_multistream"

    if not gpu_exe.exists():
        print(f"❌ GPU executable not found: {gpu_exe}")
        print(f"   Build with: cd {GPU_PORT_DIR} && mkdir -p build && cd build && cmake .. && make -j16")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"GPU Multi-Stream Heisenberg Benchmark")
    print(f"{'='*70}")
    print(f"System: L={args.L}, chi_max={args.chi}, max_iter={args.max_iter}")
    print(f"Stream counts: {stream_counts}")
    print(f"Pass tolerance: {args.pass_tol}")
    print(f"GPU executable: {gpu_exe}")

    # Run quimb reference
    try:
        quimb_result, ref_energy = run_quimb_reference(args.L, args.chi, args.max_iter, args.tol)
    except ImportError:
        print("❌ Quimb not available. Using hardcoded reference energy.")
        # For L=8, chi=32, exact Heisenberg energy
        ref_energy = -3.374931816815  # Known exact value for L=8
        quimb_result = {
            'implementation': 'quimb_DMRG2',
            'num_streams': 'N/A',
            'energy': ref_energy,
            'time': None,
            'sweeps': None,
            'delta_E': 0.0,
            'passed': True
        }

    all_results = [quimb_result]

    # Run GPU DMRG with different stream counts
    for num_streams in stream_counts:
        result = run_gpu_dmrg(num_streams, args.L, args.chi, args.max_iter, ref_energy, gpu_exe)
        all_results.append(result)

    # Compute scalability metrics
    scalability = compute_scalability_metrics(
        [r for r in all_results if r['implementation'] == 'GPU_DMRG'],
        stream_counts
    )

    # Summary
    print(f"\n{'='*70}")
    print(f"Summary")
    print(f"{'='*70}")

    passed_count = sum(1 for r in all_results if r.get('passed', False))
    total_count = len(all_results)

    print(f"Tests passed: {passed_count}/{total_count}")
    print(f"Reference energy: {ref_energy:.12f}")

    for r in all_results:
        if r['implementation'] == 'GPU_DMRG':
            status = "✓" if r.get('passed', False) else "❌"
            streams = r['num_streams']
            energy = r.get('energy', 'N/A')
            delta_E = r.get('delta_E', 'N/A')
            if isinstance(delta_E, float):
                print(f"  {status} GPU (streams={streams}): E={energy:.12f}, ΔE={delta_E:.2e}")
            else:
                print(f"  {status} GPU (streams={streams}): {delta_E}")

    # Check scalability requirements
    if args.check_speedup and scalability.get('efficiencies'):
        print(f"\nScalability Check (min efficiency: {args.min_efficiency:.0%})")
        failed_streams = []
        for n_streams, eff in scalability['efficiencies'].items():
            if n_streams > 1 and eff < args.min_efficiency:
                failed_streams.append((n_streams, eff))
                print(f"  ❌ {n_streams} streams: {eff:.1%} < {args.min_efficiency:.0%}")

        if failed_streams:
            print(f"\n❌ Scalability requirements not met")
            sys.exit(1)
        else:
            print(f"  ✓ All stream counts meet efficiency target")

    # Save results to JSON
    output_data = {
        'config': {
            'L': args.L,
            'chi_max': args.chi,
            'max_iterations': args.max_iter,
            'tolerance': args.tol,
            'pass_tolerance': args.pass_tol,
            'stream_counts': stream_counts
        },
        'reference_energy': ref_energy,
        'results': all_results,
        'scalability': scalability
    }

    if args.out:
        with open(args.out, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n✓ Results saved to {args.out}")
    else:
        # Default output filename
        default_out = f"gpu_heisenberg_L{args.L}_chi{args.chi}_results.json"
        with open(default_out, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n✓ Results saved to {default_out}")

    # Exit code based on all tests passing
    if passed_count == total_count:
        print(f"\n✓ All tests PASSED")
        sys.exit(0)
    else:
        print(f"\n❌ Some tests FAILED")
        sys.exit(1)


if __name__ == '__main__':
    main()
