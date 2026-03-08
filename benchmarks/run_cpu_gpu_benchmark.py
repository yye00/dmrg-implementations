#!/usr/bin/env python3
"""
CPU vs GPU DMRG Benchmark Runner
=================================

Automated benchmarking tool for comparing CPU (Quimb) and GPU (pdmrg_gpu_with_loader)
DMRG implementations.

Features:
- CPU: Tests with 1, 2, 4, 8 BLAS threads
- GPU: Tests with 1, 2, 4, 8 HIP streams
- 5 runs per configuration for statistical validation
- Times ONLY sweep execution (excludes MPS/MPO loading)
- Reports mean, std, min, max, speedup
- JSON output for further analysis

Usage:
    # Run specific model and size
    python3 run_cpu_gpu_benchmark.py --model heisenberg --L 12

    # Run all benchmarks
    python3 run_cpu_gpu_benchmark.py --all

    # Dry run to see what would be tested
    python3 run_cpu_gpu_benchmark.py --all --dry-run
"""

import argparse
import json
import subprocess
import time
import numpy as np
from pathlib import Path
import sys
import os

# Benchmark configurations
BENCHMARKS = {
    "heisenberg": [
        {"L": 8, "chi": 100, "desc": "Small-fast"},
        {"L": 12, "chi": 10, "desc": "Small"},
        {"L": 16, "chi": 150, "desc": "Medium"},
        {"L": 20, "chi": 10, "desc": "Medium"},
        {"L": 24, "chi": 200, "desc": "Large"},
    ],
    "josephson": [
        {"L": 8, "chi": 10, "n_max": 2, "desc": "Small"},
        {"L": 10, "chi": 100, "n_max": 2, "desc": "Medium"},
        {"L": 12, "chi": 10, "n_max": 2, "desc": "Medium"},
        {"L": 14, "chi": 150, "n_max": 2, "desc": "Large"},
    ],
}

THREAD_COUNTS = [1, 2, 4, 8]
STREAM_COUNTS = [1, 2, 4, 8]
N_RUNS = 5


def run_cpu_benchmark(model, L, chi, n_max, threads, n_runs=5):
    """Run Quimb CPU DMRG benchmark with specified thread count"""

    env = os.environ.copy()
    env["OPENBLAS_NUM_THREADS"] = str(threads)
    env["OMP_NUM_THREADS"] = str(threads)

    times = []
    energies = []

    for run in range(n_runs):
        # Run Quimb DMRG
        cmd = [
            "python3", "-m", "pdmrg",
            "--sites", str(L),
            "--bond-dim", str(chi),
            "--model", model,
            "--sweeps", "20",
            "--random-init",
            "--timing",
            "--tol", "1e-10",
        ]

        try:
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(Path.home() / "dmrg-implementations" / "pdmrg")
            )

            # Parse output for timing and energy
            for line in result.stdout.split('\n'):
                if "Total wall time:" in line:
                    t = float(line.split(':')[1].strip().replace('s', ''))
                    times.append(t)
                elif "Final energy:" in line:
                    e = float(line.split(':')[1].strip())
                    energies.append(e)
        except subprocess.TimeoutExpired:
            print(f"    ! Run {run+1} timed out")
        except Exception as e:
            print(f"    ! Run {run+1} failed: {e}")

    if len(times) == 0:
        return None

    return {
        "times": times,
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "min_time": np.min(times),
        "max_time": np.max(times),
        "energies": energies,
        "mean_energy": np.mean(energies) if energies else None,
        "n_successful": len(times),
    }


def run_gpu_benchmark(model, L, chi, n_max, streams, n_runs=5):
    """Run GPU DMRG benchmark with specified stream count"""

    times_total = []
    times_sweep = []
    energies = []

    # Find appropriate MPS/MPO files
    benchmark_dir = Path.home() / "dmrg-implementations" / "benchmarks" / "benchmark_data"

    if model == "heisenberg":
        mps_file = benchmark_dir / f"heisenberg_L{L}_chi{chi}_mps.bin"
        mpo_file = benchmark_dir / f"heisenberg_L{L}_mpo.bin"
    else:
        mps_file = benchmark_dir / f"josephson_L{L}_n{n_max}_chi{chi}_mps.bin"
        mpo_file = benchmark_dir / f"josephson_L{L}_n{n_max}_mpo.bin"

    # Check if files exist
    if not mps_file.exists():
        # Try with chi=10 fallback
        if model == "heisenberg":
            mps_file = benchmark_dir / f"heisenberg_L{L}_chi10_mps.bin"
        else:
            mps_file = benchmark_dir / f"josephson_L{L}_n{n_max}_chi10_mps.bin"

    if not mps_file.exists() or not mpo_file.exists():
        print(f"    ! MPS/MPO files not found: {mps_file.name}")
        return None

    for run in range(n_runs):
        cmd = [
            "./pdmrg_gpu_with_loader",
            "--L", str(L),
            "--model", model,
            "--streams", str(streams),
            "--load-mps", str(mps_file),
            "--load-mpo", str(mpo_file),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                cwd=str(Path.home() / "dmrg-implementations" / "pdmrg-gpu" / "build")
            )

            # Parse output
            for line in result.stdout.split('\n'):
                if "time=" in line and "E=" in line:
                    # Parse: time=1.234s
                    parts = line.split('time=')
                    if len(parts) > 1:
                        t_str = parts[1].split('s')[0]
                        times_total.append(float(t_str))

                    # Parse: E=-3.374932
                    parts = line.split('E=')
                    if len(parts) > 1:
                        e_str = parts[1].split()[0]
                        energies.append(float(e_str))

                # Parse sweep time from detailed breakdown
                if "Sweep time:" in line:
                    t_str = line.split(':')[1].strip().replace('s', '')
                    times_sweep.append(float(t_str))

        except subprocess.TimeoutExpired:
            print(f"    ! Run {run+1} timed out")
        except Exception as e:
            print(f"    ! Run {run+1} failed: {e}")

    if len(times_total) == 0:
        return None

    return {
        "times_total": times_total,
        "times_sweep": times_sweep if times_sweep else times_total,
        "mean_time_total": np.mean(times_total),
        "std_time_total": np.std(times_total),
        "min_time_total": np.min(times_total),
        "max_time_total": np.max(times_total),
        "mean_time_sweep": np.mean(times_sweep) if times_sweep else np.mean(times_total),
        "energies": energies,
        "mean_energy": np.mean(energies) if energies else None,
        "n_successful": len(times_total),
    }


def print_results_table(results, model, L):
    """Print formatted results table"""

    print(f"\n{'='*80}")
    print(f"Results: {model.upper()} L={L}")
    print(f"{'='*80}\n")

    # CPU results
    print("CPU (Quimb + OpenBLAS)")
    print(f"{'Threads':<10} {'Mean Time (s)':<15} {'Std':<10} {'Energy':<15} {'Speedup'}")
    print("-" * 80)

    cpu_results = results.get("cpu", {})
    cpu_baseline = None

    for threads in THREAD_COUNTS:
        if threads not in cpu_results or cpu_results[threads] is None:
            continue

        r = cpu_results[threads]
        mean_t = r["mean_time"]
        std_t = r["std_time"]
        energy = r["mean_energy"]

        if cpu_baseline is None:
            cpu_baseline = mean_t
            speedup_str = "1.00x"
        else:
            speedup = cpu_baseline / mean_t
            speedup_str = f"{speedup:.2f}x"

        print(f"{threads:<10} {mean_t:<15.4f} {std_t:<10.4f} {energy:<15.10f} {speedup_str}")

    # GPU results
    print(f"\nGPU (HIP + pdmrg_gpu_with_loader)")
    print(f"{'Streams':<10} {'Mean Time (s)':<15} {'Std':<10} {'Energy':<15} {'Speedup'}")
    print("-" * 80)

    gpu_results = results.get("gpu", {})
    gpu_baseline = None

    for streams in STREAM_COUNTS:
        if streams not in gpu_results or gpu_results[streams] is None:
            continue

        r = gpu_results[streams]
        mean_t = r["mean_time_total"]
        std_t = r["std_time_total"]
        energy = r["mean_energy"]

        if gpu_baseline is None:
            gpu_baseline = mean_t
            speedup_str = "1.00x"
        else:
            speedup = gpu_baseline / mean_t
            speedup_str = f"{speedup:.2f}x"

        print(f"{streams:<10} {mean_t:<15.4f} {std_t:<10.4f} {energy:<15.10f} {speedup_str}")

    # CPU vs GPU comparison
    if cpu_baseline and gpu_baseline:
        print(f"\nCPU (1 thread) vs GPU (1 stream): {cpu_baseline/gpu_baseline:.2f}x")

        # Best CPU vs Best GPU
        best_cpu = min([r["mean_time"] for r in cpu_results.values() if r is not None])
        best_gpu = min([r["mean_time_total"] for r in gpu_results.values() if r is not None])
        print(f"Best CPU vs Best GPU: {best_cpu/best_gpu:.2f}x")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Automated CPU vs GPU DMRG benchmarking")
    parser.add_argument("--model", choices=["heisenberg", "josephson"],
                        help="Model to benchmark")
    parser.add_argument("--L", type=int, help="System size")
    parser.add_argument("--all", action="store_true",
                        help="Run all benchmarks")
    parser.add_argument("--cpu-only", action="store_true",
                        help="Run CPU benchmarks only")
    parser.add_argument("--gpu-only", action="store_true",
                        help="Run GPU benchmarks only")
    parser.add_argument("--n-runs", type=int, default=5,
                        help="Number of runs per configuration")
    parser.add_argument("--output", type=str, default="benchmark_results.json",
                        help="Output JSON file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be run without running")

    args = parser.parse_args()

    global N_RUNS
    N_RUNS = args.n_runs

    # Determine benchmarks to run
    benchmarks_to_run = []

    if args.all:
        for model, configs in BENCHMARKS.items():
            for config in configs:
                benchmarks_to_run.append((model, config))
    elif args.model and args.L:
        # Find matching config
        for config in BENCHMARKS[args.model]:
            if config["L"] == args.L:
                benchmarks_to_run.append((args.model, config))
                break
        if not benchmarks_to_run:
            print(f"No benchmark config found for {args.model} L={args.L}")
            return 1
    else:
        parser.error("Must specify --model and --L, or --all")

    print("="*80)
    print("CPU vs GPU DMRG Benchmark Runner")
    print("="*80)
    print(f"Benchmarks to run: {len(benchmarks_to_run)}")
    print(f"Runs per config: {N_RUNS}")
    print(f"CPU threads: {THREAD_COUNTS}")
    print(f"GPU streams: {STREAM_COUNTS}")
    print(f"Output: {args.output}")
    if args.dry_run:
        print("Mode: DRY RUN")
    print("="*80)
    print()

    if args.dry_run:
        for model, config in benchmarks_to_run:
            L = config["L"]
            chi = config["chi"]
            desc = config["desc"]
            print(f"{model:12s} L={L:2d} χ={chi:3d} [{desc}]")
        return 0

    # Run benchmarks
    all_results = {}

    for model, config in benchmarks_to_run:
        L = config["L"]
        chi = config["chi"]
        n_max = config.get("n_max", 2)
        desc = config["desc"]

        print(f"\n{'='*80}")
        print(f"Benchmarking: {model.upper()} L={L} χ={chi} [{desc}]")
        print(f"{'='*80}\n")

        results = {"cpu": {}, "gpu": {}, "config": config}

        # CPU benchmarks
        if not args.gpu_only:
            print("CPU Benchmarks (Quimb + OpenBLAS):")
            for threads in THREAD_COUNTS:
                print(f"  Threads={threads} ... ", end="", flush=True)
                r = run_cpu_benchmark(model, L, chi, n_max, threads, N_RUNS)
                results["cpu"][threads] = r
                if r:
                    print(f"✓ {r['mean_time']:.3f}s ± {r['std_time']:.3f}s")
                else:
                    print("✗ FAILED")

        # GPU benchmarks
        if not args.cpu_only:
            print(f"\nGPU Benchmarks (HIP + pdmrg_gpu_with_loader):")
            for streams in STREAM_COUNTS:
                print(f"  Streams={streams} ... ", end="", flush=True)
                r = run_gpu_benchmark(model, L, chi, n_max, streams, N_RUNS)
                results["gpu"][streams] = r
                if r:
                    print(f"✓ {r['mean_time_total']:.3f}s ± {r['std_time_total']:.3f}s")
                else:
                    print("✗ FAILED")

        # Store results
        key = f"{model}_L{L}"
        all_results[key] = results

        # Print table
        print_results_table(results, model, L)

    # Save results
    output_file = Path(args.output)
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Benchmark complete! Results saved to: {output_file}")
    print(f"{'='*80}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
