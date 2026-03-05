#!/usr/bin/env python3
"""
MPI Scaling Benchmark for A2DMRG

Run with: mpirun -np N python -m a2dmrg.tests.test_scaling_mpi

This generates timing data for np=1,2,4,8 and creates scaling efficiency plots.
"""

import json
import time
import sys
import os
import numpy as np

from a2dmrg.mpi_compat import MPI, HAS_MPI


from a2dmrg.dmrg import a2dmrg_main
from quimb.tensor import SpinHam1D
import quimb.tensor as qtn


def run_benchmark(L, bond_dim, max_sweeps, warmup_sweeps, comm, verbose=False):
    """Run A2DMRG benchmark and return timing info."""
    rank = comm.Get_rank() if comm else 0
    size = comm.Get_size() if comm else 1
    
    # Create Heisenberg MPO
    builder = SpinHam1D(S=1/2)
    builder += 1.0, 'X', 'X'
    builder += 1.0, 'Y', 'Y'
    builder += 1.0, 'Z', 'Z'
    mpo = builder.build_mpo(L)
    
    # Synchronize before timing
    if comm:
        comm.Barrier()
    
    start_total = time.time()
    
    # Run A2DMRG
    energy, mps = a2dmrg_main(
        L=L,
        mpo=mpo,
        max_sweeps=max_sweeps,
        bond_dim=bond_dim,
        tol=1e-12,
        comm=comm if comm else MPI.COMM_SELF,
        warmup_sweeps=warmup_sweeps,
        one_site=True,
        verbose=verbose and rank == 0,
    )
    
    if comm:
        comm.Barrier()
    
    elapsed_total = time.time() - start_total
    
    return {
        'np': size,
        'L': L,
        'bond_dim': bond_dim,
        'energy': float(np.real(energy)),
        'time_total': elapsed_total,
    }


def run_scaling_study(L=20, bond_dim=50, max_sweeps=5, warmup_sweeps=2):
    """Run scaling study and save results."""
    if not HAS_MPI:
        print("mpi4py not available")
        return
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print(f"Running A2DMRG scaling benchmark: L={L}, bond_dim={bond_dim}")
        print(f"  np={size}, max_sweeps={max_sweeps}, warmup_sweeps={warmup_sweeps}")
    
    # Run benchmark
    result = run_benchmark(L, bond_dim, max_sweeps, warmup_sweeps, comm, verbose=True)
    
    if rank == 0:
        # Save result
        results_file = f"scaling_np{size}_L{L}_D{bond_dim}.json"
        with open(results_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {results_file}")
        print(f"  Energy: {result['energy']:.12f}")
        print(f"  Time: {result['time_total']:.2f}s")
    
    return result


def generate_plots(results_dir="."):
    """Generate scaling efficiency plots from saved results."""
    import glob
    
    # Load all results
    results = []
    for f in glob.glob(os.path.join(results_dir, "scaling_np*.json")):
        with open(f) as fp:
            results.append(json.load(fp))
    
    if not results:
        print("No results found!")
        return
    
    # Sort by np
    results.sort(key=lambda x: x['np'])
    
    # Extract data
    nps = [r['np'] for r in results]
    times = [r['time_total'] for r in results]
    energies = [r['energy'] for r in results]
    
    # Calculate speedup and efficiency
    t1 = times[0]  # Time with np=1
    speedups = [t1 / t for t in times]
    efficiencies = [s / n for s, n in zip(speedups, nps)]
    
    print("\nScaling Results:")
    print("-" * 60)
    print(f"{'np':>4} {'Time (s)':>10} {'Speedup':>10} {'Efficiency':>12} {'Energy':>16}")
    print("-" * 60)
    for i, r in enumerate(results):
        print(f"{r['np']:>4} {times[i]:>10.2f} {speedups[i]:>10.2f} {efficiencies[i]*100:>11.1f}% {energies[i]:>16.10f}")
    
    # Generate plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: Time vs np
        axes[0].plot(nps, times, 'bo-', linewidth=2, markersize=10)
        axes[0].set_xlabel('Number of Processors (np)', fontsize=12)
        axes[0].set_ylabel('Time (seconds)', fontsize=12)
        axes[0].set_title('Execution Time', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xticks(nps)
        
        # Plot 2: Speedup
        axes[1].plot(nps, speedups, 'go-', linewidth=2, markersize=10, label='Actual')
        axes[1].plot(nps, nps, 'r--', linewidth=2, label='Ideal (linear)')
        axes[1].set_xlabel('Number of Processors (np)', fontsize=12)
        axes[1].set_ylabel('Speedup (T₁/Tₙ)', fontsize=12)
        axes[1].set_title('Parallel Speedup', fontsize=14)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xticks(nps)
        
        # Plot 3: Efficiency
        axes[2].bar(range(len(nps)), [e*100 for e in efficiencies], tick_label=[str(n) for n in nps])
        axes[2].axhline(y=100, color='r', linestyle='--', linewidth=2, label='100% efficiency')
        axes[2].set_xlabel('Number of Processors (np)', fontsize=12)
        axes[2].set_ylabel('Efficiency (%)', fontsize=12)
        axes[2].set_title('Parallel Efficiency', fontsize=14)
        axes[2].set_ylim(0, 110)
        axes[2].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('scaling_plot.png', dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to scaling_plot.png")
        
    except ImportError:
        print("\nmatplotlib not available, skipping plot generation")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='A2DMRG MPI Scaling Benchmark')
    parser.add_argument('--L', type=int, default=20, help='Chain length')
    parser.add_argument('--bond-dim', type=int, default=50, help='Bond dimension')
    parser.add_argument('--max-sweeps', type=int, default=3, help='Max A2DMRG sweeps')
    parser.add_argument('--warmup-sweeps', type=int, default=2, help='Warmup sweeps')
    parser.add_argument('--plot', action='store_true', help='Generate plots from saved results')
    
    args = parser.parse_args()
    
    if args.plot:
        generate_plots()
    else:
        run_scaling_study(
            L=args.L,
            bond_dim=args.bond_dim,
            max_sweeps=args.max_sweeps,
            warmup_sweeps=args.warmup_sweeps
        )
