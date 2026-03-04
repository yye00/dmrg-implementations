#!/usr/bin/env python3
"""
Publication-Ready DMRG Benchmark Suite

Runs all DMRG implementations with tight convergence criteria and
produces clean, publication-quality results.

Methods tested:
  - quimb DMRG1 (single-site, serial)
  - quimb DMRG2 (two-site, serial)
  - PDMRG np=1,2,4,8 (Real-Space Parallel DMRG)
  - A2DMRG np=1,2,4,8 (Additive Two-Level Parallel DMRG)

Model: Josephson Junction Array (complex128)
  - Physically realistic for superconducting quantum computing
  - Requires complex128 for proper phase physics

Target: All energies within 1e-10 of quimb DMRG2 reference
"""

import subprocess
import time
import json
import os
import sys
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, List

# Configuration
CONFIG = {
    'L': 20,           # Number of sites
    'bond_dim': 50,    # Maximum bond dimension
    'n_max': 2,        # Charge truncation (d = 2*n_max + 1 = 5)
    'max_sweeps': 30,  # Maximum sweeps
    'tol': 1e-12,      # Convergence tolerance (tight!)
    'match_tol': 1e-10, # Required agreement between implementations
}

NP_LIST = [1, 2, 4, 8]

# Get script directory and construct relative paths to virtual environments
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PDMRG_VENV = os.path.join(SCRIPT_DIR, 'pdmrg', 'venv', 'bin', 'python')
A2DMRG_VENV = os.path.join(SCRIPT_DIR, 'a2dmrg', 'venv', 'bin', 'python')


@dataclass
class BenchmarkResult:
    method: str
    energy: Optional[float] = None
    time: Optional[float] = None
    np: int = 1
    success: bool = False
    error: Optional[str] = None
    sweeps: Optional[int] = None


def run_quimb_dmrg(method='DMRG2'):
    """Run quimb DMRG1 or DMRG2 as reference."""
    sys.path.insert(0, '/home/captain/clawd/work/dmrg-implementations/a2dmrg')
    from benchmarks.josephson_junction import build_josephson_mpo
    import quimb.tensor as qtn
    import numpy as np
    
    mpo = build_josephson_mpo(
        CONFIG['L'], E_J=1.0, E_C=0.5, 
        n_max=CONFIG['n_max'], with_flux=True
    )
    
    t0 = time.time()
    if method == 'DMRG1':
        dmrg = qtn.DMRG1(mpo, bond_dims=CONFIG['bond_dim'])
    else:
        dmrg = qtn.DMRG2(mpo, bond_dims=CONFIG['bond_dim'])
    
    dmrg.solve(max_sweeps=CONFIG['max_sweeps'], tol=CONFIG['tol'], verbosity=0)
    t1 = time.time()
    
    return BenchmarkResult(
        method=f'quimb_{method}',
        energy=float(np.real(dmrg.energy)),
        time=t1 - t0,
        np=1,
        success=True
    )


def run_pdmrg(np_count):
    """Run PDMRG with specified number of processes."""
    script = f'''
import sys, time
import numpy as np
sys.path.insert(0, '/home/captain/clawd/work/dmrg-implementations/a2dmrg')
from benchmarks.josephson_junction import build_josephson_mpo
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

mpo = build_josephson_mpo({CONFIG['L']}, E_J=1.0, E_C=0.5, n_max={CONFIG['n_max']}, with_flux=True)

sys.path.insert(0, '/home/captain/clawd/work/dmrg-implementations/pdmrg')
from pdmrg.dmrg import pdmrg_main

t0 = time.time()
energy, mps = pdmrg_main(
    L={CONFIG['L']},
    mpo=mpo,
    bond_dim={CONFIG['bond_dim']},
    max_sweeps={CONFIG['max_sweeps']},
    tol={CONFIG['tol']},
    dtype='complex128',
    comm=comm,
    verbose=False
)
t1 = time.time()

if rank == 0:
    print(f"RESULT:{{np.real(energy):.15f}}:{{t1-t0:.4f}}")
'''
    
    script_path = '/tmp/pdmrg_pub_benchmark.py'
    with open(script_path, 'w') as f:
        f.write(script)
    
    cmd = f'''
source /etc/profile.d/modules.sh && module load mpi/openmpi-x86_64 && \
mpirun -np {np_count} --oversubscribe --mca btl tcp,self --mca btl_tcp_if_include lo \
{PDMRG_VENV} -u {script_path} 2>&1
'''
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=3600)
        output = result.stdout + result.stderr
        
        for line in output.split('\n'):
            if line.startswith('RESULT:'):
                parts = line.split(':')
                return BenchmarkResult(
                    method='PDMRG',
                    energy=float(parts[1]),
                    time=float(parts[2]),
                    np=np_count,
                    success=True
                )
        
        return BenchmarkResult(
            method='PDMRG',
            np=np_count,
            success=False,
            error=output[-500:] if len(output) > 500 else output
        )
    except subprocess.TimeoutExpired:
        return BenchmarkResult(
            method='PDMRG',
            np=np_count,
            success=False,
            error='Timeout after 3600s'
        )
    except Exception as e:
        return BenchmarkResult(
            method='PDMRG',
            np=np_count,
            success=False,
            error=str(e)
        )


def run_a2dmrg(np_count):
    """Run A2DMRG with specified number of processes."""
    script = f'''
import sys, time
import numpy as np
sys.path.insert(0, '/home/captain/clawd/work/dmrg-implementations/a2dmrg')
from benchmarks.josephson_junction import build_josephson_mpo
from a2dmrg.dmrg import a2dmrg_main
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

mpo = build_josephson_mpo({CONFIG['L']}, E_J=1.0, E_C=0.5, n_max={CONFIG['n_max']}, with_flux=True)

t0 = time.time()
energy, mps = a2dmrg_main(
    L={CONFIG['L']},
    mpo=mpo,
    max_sweeps={CONFIG['max_sweeps']},
    bond_dim={CONFIG['bond_dim']},
    tol={CONFIG['tol']},
    dtype=np.complex128,
    comm=comm,
    warmup_sweeps=3,
    one_site=True,
    verbose=False
)
t1 = time.time()

if rank == 0:
    print(f"RESULT:{{np.real(energy):.15f}}:{{t1-t0:.4f}}")
'''
    
    script_path = '/tmp/a2dmrg_pub_benchmark.py'
    with open(script_path, 'w') as f:
        f.write(script)
    
    cmd = f'''
source /etc/profile.d/modules.sh && module load mpi/openmpi-x86_64 && \
mpirun -np {np_count} --oversubscribe --mca btl tcp,self --mca btl_tcp_if_include lo \
{A2DMRG_VENV} -u {script_path} 2>&1
'''
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=3600)
        output = result.stdout + result.stderr
        
        for line in output.split('\n'):
            if line.startswith('RESULT:'):
                parts = line.split(':')
                return BenchmarkResult(
                    method='A2DMRG',
                    energy=float(parts[1]),
                    time=float(parts[2]),
                    np=np_count,
                    success=True
                )
        
        return BenchmarkResult(
            method='A2DMRG',
            np=np_count,
            success=False,
            error=output[-500:] if len(output) > 500 else output
        )
    except subprocess.TimeoutExpired:
        return BenchmarkResult(
            method='A2DMRG',
            np=np_count,
            success=False,
            error='Timeout after 3600s'
        )
    except Exception as e:
        return BenchmarkResult(
            method='A2DMRG',
            np=np_count,
            success=False,
            error=str(e)
        )


def print_results_table(results: List[BenchmarkResult], E_ref: float):
    """Print results in publication-ready table format."""
    print()
    print("=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Model: Josephson Junction Array")
    print(f"Parameters: L={CONFIG['L']}, D={CONFIG['bond_dim']}, n_max={CONFIG['n_max']}, tol={CONFIG['tol']}")
    print(f"Reference energy: {E_ref:.12f}")
    print()
    
    print(f"{'Method':<20} | {'np':>4} | {'Energy':>18} | {'ΔE':>12} | {'Time (s)':>10} | {'Status':<10}")
    print("-" * 80)
    
    all_pass = True
    for r in results:
        if r.success:
            delta_E = abs(r.energy - E_ref)
            status = "✓ PASS" if delta_E < CONFIG['match_tol'] else "✗ FAIL"
            if delta_E >= CONFIG['match_tol']:
                all_pass = False
            print(f"{r.method:<20} | {r.np:>4} | {r.energy:>18.12f} | {delta_E:>12.2e} | {r.time:>10.1f} | {status}")
        else:
            all_pass = False
            print(f"{r.method:<20} | {r.np:>4} | {'N/A':>18} | {'N/A':>12} | {'N/A':>10} | ✗ FAILED")
    
    print("-" * 80)
    print()
    
    if all_pass:
        print("✓ ALL IMPLEMENTATIONS PASS - Results within machine precision!")
    else:
        n_pass = sum(1 for r in results if r.success and abs(r.energy - E_ref) < CONFIG['match_tol'])
        print(f"✗ {n_pass}/{len(results)} implementations within tolerance")
    
    print("=" * 80)


def main():
    print("=" * 80)
    print("DMRG PUBLICATION BENCHMARK")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Configuration: {CONFIG}")
    print("=" * 80)
    
    results = []
    
    # 1. Serial references
    print("\n[1/10] quimb DMRG1 (single-site, serial)...")
    r = run_quimb_dmrg('DMRG1')
    results.append(r)
    if r.success:
        print(f"  Energy: {r.energy:.12f}, Time: {r.time:.1f}s")
    else:
        print(f"  FAILED: {r.error}")
    
    print("\n[2/10] quimb DMRG2 (two-site, serial)...")
    r = run_quimb_dmrg('DMRG2')
    results.append(r)
    E_ref = r.energy if r.success else None
    if r.success:
        print(f"  Energy: {r.energy:.12f}, Time: {r.time:.1f}s")
        print(f"  → Using as reference energy")
    else:
        print(f"  FAILED: {r.error}")
        return 1
    
    # 2. PDMRG
    for i, np_count in enumerate(NP_LIST):
        print(f"\n[{3+i}/10] PDMRG np={np_count}...")
        r = run_pdmrg(np_count)
        results.append(r)
        if r.success:
            print(f"  Energy: {r.energy:.12f}, ΔE: {abs(r.energy - E_ref):.2e}, Time: {r.time:.1f}s")
        else:
            print(f"  FAILED: {r.error[:200] if r.error else 'Unknown'}")
    
    # 3. A2DMRG
    for i, np_count in enumerate(NP_LIST):
        print(f"\n[{7+i}/10] A2DMRG np={np_count}...")
        r = run_a2dmrg(np_count)
        results.append(r)
        if r.success:
            print(f"  Energy: {r.energy:.12f}, ΔE: {abs(r.energy - E_ref):.2e}, Time: {r.time:.1f}s")
        else:
            print(f"  FAILED: {r.error[:200] if r.error else 'Unknown'}")
    
    # Print summary
    print_results_table(results, E_ref)
    
    # Save results
    output = {
        'benchmark': 'publication_dmrg',
        'timestamp': datetime.now().isoformat(),
        'config': CONFIG,
        'reference_energy': E_ref,
        'results': [asdict(r) for r in results]
    }
    
    output_file = '/home/captain/clawd/work/dmrg-implementations/publication_benchmark_results.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    # Return exit code
    all_pass = all(
        r.success and abs(r.energy - E_ref) < CONFIG['match_tol'] 
        for r in results
    )
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
