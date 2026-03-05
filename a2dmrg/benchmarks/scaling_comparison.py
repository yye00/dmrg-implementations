#!/usr/bin/env python3
"""
Scalability comparison: PDMRG vs A2DMRG

This script runs both algorithms with np=1,2,4,8 and generates comparison plots.
"""

import subprocess
import time
import json
import os
import sys
import numpy as np

# Results storage
RESULTS_FILE = 'benchmarks/scaling_comparison_results.json'


def run_a2dmrg_benchmark(L, bond_dim, np_list, warmup_sweeps=2, max_sweeps=3):
    """Run A2DMRG scaling benchmark."""
    results = []
    
    for np in np_list:
        print(f"  A2DMRG np={np}...", end=" ", flush=True)
        
        cmd = f"""
source /etc/profile.d/modules.sh && module load mpi/openmpi-x86_64 && \
cd /home/captain/clawd/work/dmrg-implementations/a2dmrg && \
mpirun -np {np} --oversubscribe --mca btl tcp,self --mca btl_tcp_if_include lo \
python3 -c "
import time
import sys
sys.path.insert(0, '.')
from a2dmrg.mpi_compat import MPI, HAS_MPI
from a2dmrg.dmrg import a2dmrg_main
from quimb.tensor import SpinHam1D

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Create Heisenberg MPO
builder = SpinHam1D(S=1/2)
builder += 1.0, 'X', 'X'
builder += 1.0, 'Y', 'Y'
builder += 1.0, 'Z', 'Z'
mpo = builder.build_mpo({L})

t0 = time.time()
energy, mps = a2dmrg_main(
    L={L}, mpo=mpo, max_sweeps={max_sweeps}, bond_dim={bond_dim},
    tol=1e-10, comm=comm, warmup_sweeps={warmup_sweeps}, one_site=True, verbose=False
)
t1 = time.time()

if rank == 0:
    print(f'RESULT:{{energy}}:{{t1-t0}}')
"
"""
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
            output = result.stdout + result.stderr
            
            # Parse result
            for line in output.split('\n'):
                if 'RESULT:' in line:
                    parts = line.split(':')
                    energy = float(parts[1])
                    elapsed = float(parts[2])
                    print(f"E={energy:.8f}, t={elapsed:.2f}s")
                    results.append({'np': np, 'energy': energy, 'time': elapsed})
                    break
            else:
                print("FAILED (no result)")
                results.append({'np': np, 'energy': None, 'time': None})
        except Exception as e:
            print(f"FAILED: {e}")
            results.append({'np': np, 'energy': None, 'time': None})
    
    return results


def run_pdmrg_benchmark(L, bond_dim, np_list, sweeps=10):
    """Run PDMRG scaling benchmark."""
    results = []
    
    for np in np_list:
        print(f"  PDMRG np={np}...", end=" ", flush=True)
        
        cmd = f"""
source /etc/profile.d/modules.sh && module load mpi/openmpi-x86_64 && \
cd /home/captain/clawd/work/dmrg-implementations/pdmrg && \
mpirun -np {np} --oversubscribe --mca btl tcp,self --mca btl_tcp_if_include lo \
python -m pdmrg --sites {L} --bond-dim {bond_dim} --model heisenberg --sweeps {sweeps} --tol 1e-10 --timing 2>&1
"""
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
            output = result.stdout + result.stderr
            
            # Parse PDMRG output
            energy = None
            total_time = None
            
            for line in output.split('\n'):
                if 'Final energy:' in line:
                    energy = float(line.split(':')[1].strip())
                if 'Total wall time:' in line:
                    total_time = float(line.split(':')[1].strip().replace('s', ''))
            
            if energy is not None and total_time is not None:
                print(f"E={energy:.8f}, t={total_time:.2f}s")
                results.append({'np': np, 'energy': energy, 'time': total_time})
            else:
                print("FAILED (no energy/time found)")
                # Try to extract any timing info
                results.append({'np': np, 'energy': energy, 'time': total_time})
        except Exception as e:
            print(f"FAILED: {e}")
            results.append({'np': np, 'energy': None, 'time': None})
    
    return results


def generate_comparison_plot(a2dmrg_results, pdmrg_results, L, bond_dim):
    """Generate comparison plot."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # Extract data
    a2_nps = [r['np'] for r in a2dmrg_results if r['time'] is not None]
    a2_times = [r['time'] for r in a2dmrg_results if r['time'] is not None]
    
    pd_nps = [r['np'] for r in pdmrg_results if r['time'] is not None]
    pd_times = [r['time'] for r in pdmrg_results if r['time'] is not None]
    
    if not a2_times or not pd_times:
        print("Not enough data to generate plot")
        return
    
    # Calculate speedups
    a2_t1 = a2_times[0] if a2_nps[0] == 1 else a2_times[a2_nps.index(1)] if 1 in a2_nps else a2_times[0]
    pd_t1 = pd_times[0] if pd_nps[0] == 1 else pd_times[pd_nps.index(1)] if 1 in pd_nps else pd_times[0]
    
    a2_speedups = [a2_t1 / t for t in a2_times]
    pd_speedups = [pd_t1 / t for t in pd_times]
    
    a2_efficiencies = [s / n for s, n in zip(a2_speedups, a2_nps)]
    pd_efficiencies = [s / n for s, n in zip(pd_speedups, pd_nps)]
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Execution Time
    axes[0].plot(a2_nps, a2_times, 'bo-', lw=2, ms=10, label='A2DMRG')
    axes[0].plot(pd_nps, pd_times, 'rs-', lw=2, ms=10, label='PDMRG')
    axes[0].set_xlabel('Number of Processors', fontsize=12)
    axes[0].set_ylabel('Time (seconds)', fontsize=12)
    axes[0].set_title('Execution Time', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(sorted(set(a2_nps + pd_nps)))
    
    # Plot 2: Speedup
    max_np = max(max(a2_nps), max(pd_nps))
    ideal_nps = list(range(1, max_np + 1))
    axes[1].plot(a2_nps, a2_speedups, 'bo-', lw=2, ms=10, label='A2DMRG')
    axes[1].plot(pd_nps, pd_speedups, 'rs-', lw=2, ms=10, label='PDMRG')
    axes[1].plot(ideal_nps, ideal_nps, 'k--', lw=2, label='Ideal (linear)')
    axes[1].set_xlabel('Number of Processors', fontsize=12)
    axes[1].set_ylabel('Speedup', fontsize=12)
    axes[1].set_title('Parallel Speedup', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(sorted(set(a2_nps + pd_nps)))
    
    # Plot 3: Efficiency
    x_a2 = np.arange(len(a2_nps))
    x_pd = np.arange(len(pd_nps))
    width = 0.35
    
    axes[2].bar(x_a2 - width/2, [e*100 for e in a2_efficiencies], width, label='A2DMRG', color='blue', alpha=0.7)
    axes[2].bar(x_pd + width/2, [e*100 for e in pd_efficiencies], width, label='PDMRG', color='red', alpha=0.7)
    axes[2].axhline(y=100, color='k', linestyle='--', lw=1)
    axes[2].set_xlabel('Number of Processors', fontsize=12)
    axes[2].set_ylabel('Efficiency (%)', fontsize=12)
    axes[2].set_title('Parallel Efficiency', fontsize=14)
    axes[2].set_xticks(x_a2)
    axes[2].set_xticklabels([str(n) for n in a2_nps])
    axes[2].legend(fontsize=11)
    axes[2].set_ylim(0, 120)
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'PDMRG vs A2DMRG Scalability (L={L}, D={bond_dim})', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('benchmarks/pdmrg_vs_a2dmrg_scaling.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to benchmarks/pdmrg_vs_a2dmrg_scaling.png")


def main():
    L = 40
    bond_dim = 50
    np_list = [1, 2, 4, 8]
    
    print("=" * 60)
    print(f"PDMRG vs A2DMRG Scalability Comparison")
    print(f"L={L}, bond_dim={bond_dim}")
    print("=" * 60)
    
    print("\n--- A2DMRG Benchmark ---")
    a2dmrg_results = run_a2dmrg_benchmark(L, bond_dim, np_list)
    
    print("\n--- PDMRG Benchmark ---")
    pdmrg_results = run_pdmrg_benchmark(L, bond_dim, np_list)
    
    # Save results
    all_results = {
        'L': L,
        'bond_dim': bond_dim,
        'a2dmrg': a2dmrg_results,
        'pdmrg': pdmrg_results
    }
    
    os.makedirs('benchmarks', exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {RESULTS_FILE}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'np':>4} {'A2DMRG Time':>15} {'PDMRG Time':>15}")
    print("-" * 40)
    for i, np in enumerate(np_list):
        a2_t = a2dmrg_results[i]['time'] if i < len(a2dmrg_results) and a2dmrg_results[i]['time'] else 'N/A'
        pd_t = pdmrg_results[i]['time'] if i < len(pdmrg_results) and pdmrg_results[i]['time'] else 'N/A'
        a2_str = f"{a2_t:.2f}s" if isinstance(a2_t, float) else a2_t
        pd_str = f"{pd_t:.2f}s" if isinstance(pd_t, float) else pd_t
        print(f"{np:>4} {a2_str:>15} {pd_str:>15}")
    
    # Generate plot
    generate_comparison_plot(a2dmrg_results, pdmrg_results, L, bond_dim)


if __name__ == '__main__':
    main()
