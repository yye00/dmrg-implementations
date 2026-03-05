#!/usr/bin/env python3
"""
Benchmark for parallel local microsteps in A2DMRG.

This measures ONLY the parallelizable part of A2DMRG: the local micro-iterations.
According to the paper, this should show linear scaling with number of processors.

Run with: mpirun -np N python benchmark_local_microsteps.py --L 100 --bond-dim 100
"""

import json
import time
import sys
import os
import numpy as np

try:    from a2dmrg.mpi_compat import MPI, HAS_MPI
    HAS_MPI = True
except ImportError:
    HAS_MPI = False

import quimb.tensor as qtn
from quimb.tensor import SpinHam1D, DMRG2

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from a2dmrg.parallel.local_steps import parallel_local_microsteps, gather_local_results
from a2dmrg.mps.mps_utils import create_neel_state


def create_heisenberg_mpo(L):
    """Create Heisenberg chain MPO."""
    builder = SpinHam1D(S=1/2)
    builder += 1.0, 'X', 'X'
    builder += 1.0, 'Y', 'Y'
    builder += 1.0, 'Z', 'Z'
    return builder.build_mpo(L)


def broadcast_mps(mps, comm, L):
    """Broadcast MPS from rank 0 to all ranks."""
    rank = comm.Get_rank()
    
    if rank == 0:
        mps_data = []
        site_tag_id = mps.site_tag_id
        for i in range(L):
            t = mps.tensors[i]
            mps_data.append((t.data, t.shape, t.inds, list(t.tags)))
        bcast_info = (mps_data, site_tag_id)
    else:
        bcast_info = None
    
    bcast_info = comm.bcast(bcast_info, root=0)
    mps_data, site_tag_id = bcast_info
    
    if rank != 0:
        tensors = []
        for data, shape, inds, tags in mps_data:
            tensors.append(qtn.Tensor(data=data, inds=inds, tags=tags))
        tn = qtn.TensorNetwork(tensors)
        mps = qtn.MatrixProductState.from_TN(
            tn, 
            site_tag_id=site_tag_id,
            site_ind_id='k{}',
            cyclic=False,
            L=L
        )
    
    return mps


def run_benchmark(L, bond_dim, num_repeats=3, comm=None):
    """
    Benchmark the parallel local microsteps.
    
    Returns timing statistics for the parallel portion only.
    """
    if comm is None:
        comm = MPI.COMM_WORLD
    
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Create MPO on all ranks
    mpo = create_heisenberg_mpo(L)
    
    # Rank 0 creates and warms up MPS
    if rank == 0:
        print(f"Setting up benchmark: L={L}, bond_dim={bond_dim}, np={size}", flush=True)
        print(f"Creating initial MPS and running warmup...", flush=True)
        mps = create_neel_state(L, bond_dim=bond_dim, dtype=np.float64)
        dmrg = DMRG2(mpo, bond_dims=bond_dim)
        dmrg.solve(max_sweeps=3, tol=1e-6, verbosity=0)
        mps = dmrg._k.copy()
        print(f"Warmup complete: E = {dmrg.energy:.10f}", flush=True)
    else:
        mps = None
    
    # Broadcast MPS
    mps = broadcast_mps(mps, comm, L)
    comm.Barrier()
    
    if rank == 0:
        print(f"\nBenchmarking parallel local microsteps ({num_repeats} repeats)...", flush=True)
    
    times = []
    for rep in range(num_repeats):
        comm.Barrier()
        start = time.time()
        
        # Run parallel local microsteps - THIS IS WHAT WE'RE BENCHMARKING
        local_results = parallel_local_microsteps(
            mps, mpo, comm,
            microstep_type='one_site',
            max_bond=bond_dim,
            cutoff=1e-10,
            tol=1e-10
        )
        
        # Gather results
        all_results = gather_local_results(local_results, comm)
        
        comm.Barrier()
        elapsed = time.time() - start
        times.append(elapsed)
        
        if rank == 0:
            print(f"  Rep {rep+1}: {elapsed:.3f}s ({len(all_results)} microsteps)", flush=True)
    
    # Compute statistics
    times = np.array(times)
    mean_time = np.mean(times)
    std_time = np.std(times)
    
    result = {
        'np': size,
        'L': L,
        'bond_dim': bond_dim,
        'num_microsteps': L,  # One per site for one-site DMRG
        'mean_time': mean_time,
        'std_time': std_time,
        'times': times.tolist(),
    }
    
    return result


def run_scaling_study(L=100, bond_dim=100, num_repeats=5):
    """Run scaling study and save results."""
    if not HAS_MPI:
        print("mpi4py not available")
        return
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Run benchmark
    result = run_benchmark(L, bond_dim, num_repeats, comm)
    
    if rank == 0:
        # Save result
        results_file = f"microstep_scaling_np{size}_L{L}_D{bond_dim}.json"
        with open(results_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {results_file}")
        print(f"  Mean time: {result['mean_time']:.3f} ± {result['std_time']:.3f}s")
    
    return result


def generate_plots(results_dir="."):
    """Generate scaling efficiency plots from saved results."""
    import glob
    
    # Load all results
    results = []
    for f in glob.glob(os.path.join(results_dir, "microstep_scaling_np*.json")):
        with open(f) as fp:
            results.append(json.load(fp))
    
    if not results:
        print("No results found!")
        return
    
    # Sort by np
    results.sort(key=lambda x: x['np'])
    
    # Extract data
    nps = [r['np'] for r in results]
    times = [r['mean_time'] for r in results]
    stds = [r['std_time'] for r in results]
    
    # Calculate speedup and efficiency
    t1 = times[0]  # Time with np=1
    speedups = [t1 / t for t in times]
    efficiencies = [s / n for s, n in zip(speedups, nps)]
    
    print("\nParallel Local Microsteps - Scaling Results:")
    print("=" * 70)
    print(f"{'np':>4} {'Time (s)':>12} {'Speedup':>10} {'Efficiency':>12}")
    print("-" * 70)
    for i, r in enumerate(results):
        print(f"{r['np']:>4} {times[i]:>10.3f}±{stds[i]:.3f} {speedups[i]:>10.2f} {efficiencies[i]*100:>11.1f}%")
    print("=" * 70)
    
    # Generate plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: Time vs np
        axes[0].errorbar(nps, times, yerr=stds, fmt='bo-', linewidth=2, markersize=10, capsize=5)
        axes[0].set_xlabel('Number of Processors (np)', fontsize=12)
        axes[0].set_ylabel('Time (seconds)', fontsize=12)
        axes[0].set_title('Local Microsteps: Execution Time', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xticks(nps)
        
        # Plot 2: Speedup
        axes[1].plot(nps, speedups, 'go-', linewidth=2, markersize=10, label='Actual')
        axes[1].plot(nps, nps, 'r--', linewidth=2, label='Ideal (linear)')
        axes[1].set_xlabel('Number of Processors (np)', fontsize=12)
        axes[1].set_ylabel('Speedup (T₁/Tₙ)', fontsize=12)
        axes[1].set_title('Local Microsteps: Parallel Speedup', fontsize=14)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xticks(nps)
        
        # Plot 3: Efficiency
        colors = ['green' if e > 0.8 else 'orange' if e > 0.5 else 'red' for e in efficiencies]
        axes[2].bar(range(len(nps)), [e*100 for e in efficiencies], tick_label=[str(n) for n in nps], color=colors)
        axes[2].axhline(y=100, color='r', linestyle='--', linewidth=2, label='100% efficiency')
        axes[2].axhline(y=80, color='g', linestyle=':', linewidth=1, alpha=0.7, label='80% efficiency')
        axes[2].set_xlabel('Number of Processors (np)', fontsize=12)
        axes[2].set_ylabel('Efficiency (%)', fontsize=12)
        axes[2].set_title('Local Microsteps: Parallel Efficiency', fontsize=14)
        axes[2].set_ylim(0, 110)
        axes[2].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f"A2DMRG Parallel Scaling (L={results[0]['L']}, D={results[0]['bond_dim']})", fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig('microstep_scaling_plot.png', dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to microstep_scaling_plot.png")
        
    except ImportError:
        print("\nmatplotlib not available, skipping plot generation")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='A2DMRG Local Microsteps Scaling Benchmark')
    parser.add_argument('--L', type=int, default=100, help='Chain length')
    parser.add_argument('--bond-dim', type=int, default=100, help='Bond dimension')
    parser.add_argument('--repeats', type=int, default=5, help='Number of timing repeats')
    parser.add_argument('--plot', action='store_true', help='Generate plots from saved results')
    
    args = parser.parse_args()
    
    if args.plot:
        generate_plots()
    else:
        run_scaling_study(
            L=args.L,
            bond_dim=args.bond_dim,
            num_repeats=args.repeats
        )
