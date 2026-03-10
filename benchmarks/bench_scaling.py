#!/usr/bin/env python3
"""Strong-scaling benchmark: PDMRG sweep phase across np=2,4,8,16.

Tests Stoudenmire's claim of near-linear sweep speedup.
Reports sweep-only timings (excludes warmup and cleanup).
"""

import sys
import os
import json
import time
import subprocess
import shutil
from pathlib import Path

repo_root = Path(__file__).parent.parent
MPIRUN = shutil.which('mpirun')
TOL = 1e-11


def run_pdmrg(model, case, np_count, pdmrg_dir='pdmrg', timeout=600):
    """Run PDMRG and extract per-sweep timings."""
    script = f"""
import sys, os, time, json
import numpy as np
sys.path.insert(0, '{repo_root / pdmrg_dir}')
sys.path.insert(0, '{repo_root / "benchmarks"}')
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from mpi4py import MPI
from pdmrg.dmrg import pdmrg_main
from benchmark_data_loader import (
    load_benchmark_case, convert_tensors_to_quimb_mpo, convert_tensors_to_quimb_mps)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

data = load_benchmark_case('{model}', '{case}')
mpo = convert_tensors_to_quimb_mpo(data['mpo_tensors'])
manifest = data['manifest']
initial_mps = None
if data['mps_tensors'] is not None:
    initial_mps = convert_tensors_to_quimb_mps(data['mps_tensors'])

t0 = time.time()
result = pdmrg_main(
    L=manifest['L'], mpo=mpo, max_sweeps=20,
    bond_dim=manifest['bond_dim'], bond_dim_warmup=manifest['bond_dim'],
    n_warmup_sweeps=3, tol={TOL}, dtype=manifest['dtype'],
    comm=comm, verbose=(rank==0), return_metadata=True,
    initial_mps=initial_mps)
energy, _, metadata = result
wall = time.time() - t0

if rank == 0:
    print(json.dumps({{'energy': float(np.real(energy)), 'time': wall,
                       'sweeps': metadata.get('final_sweep', -1),
                       'converged': metadata.get('converged', False)}}))
"""
    script_path = f'/tmp/bench_scale_{model}_{case}_np{np_count}.py'
    with open(script_path, 'w') as f:
        f.write(script)

    venv_python = repo_root / 'pdmrg' / 'venv' / 'bin' / 'python'
    cmd = [
        MPIRUN, '-np', str(np_count), '--oversubscribe',
        '--mca', 'btl', 'tcp,self', '--mca', 'btl_tcp_if_include', 'lo',
        str(venv_python), script_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            return None, (result.stderr or result.stdout or 'unknown')[-500:]

        lines = result.stdout.strip().split('\n')
        sweep_times = []
        warmup_time = None
        for line in lines:
            if 'Warmup time:' in line:
                warmup_time = float(line.split('Warmup time:')[1].strip().rstrip('s'))
            if 'time =' in line and 'Sweep' in line:
                t = float(line.split('time =')[1].strip().rstrip('s'))
                sweep_times.append(t)

        for line in lines:
            if line.startswith('{'):
                d = json.loads(line)
                total = d['time']
                energy = d['energy']
                cleanup_time = total - (warmup_time or 0) - sum(sweep_times)
                return {
                    'energy': energy,
                    'total': total,
                    'warmup': warmup_time or 0,
                    'sweep_times': sweep_times,
                    'sweep_total': sum(sweep_times),
                    'cleanup': cleanup_time,
                    'n_sweeps': len(sweep_times),
                    'converged': d.get('converged', False),
                }, None
        return None, 'no JSON output'
    except subprocess.TimeoutExpired:
        return None, f'KILLED (>{timeout:.0f}s)'


def main():
    # Test cases: pick ones with enough sites for scaling
    cases = [
        ('heisenberg', 'L48_D20'),
        ('heisenberg', 'L64_D20'),
        ('heisenberg', 'L96_D20'),
        ('heisenberg', 'L128_D20'),
    ]

    np_counts = [2, 4, 8, 16]

    print("=" * 90)
    print("PDMRG STRONG SCALING: sweep-phase timing across np=2,4,8,16")
    print("=" * 90)
    print(f"Machine: 16 physical cores, 32 logical cores")
    print(f"Each rank pinned to OMP_NUM_THREADS=1")
    print()

    all_results = {}

    for model, case in cases:
        print(f"\n{'='*90}")
        print(f"  {model}/{case}")
        print(f"{'='*90}")

        results_for_case = {}
        for np_count in np_counts:
            label = f"np={np_count}"
            print(f"\n  {label}:", flush=True)
            r, err = run_pdmrg(model, case, np_count, timeout=1800)
            if err:
                print(f"    FAIL: {err}")
                continue

            avg_sweep = (sum(r['sweep_times'][1:]) / len(r['sweep_times'][1:])
                         if len(r['sweep_times']) > 1
                         else r['sweep_times'][0] if r['sweep_times'] else 0)

            print(f"    sweeps={r['n_sweeps']}  "
                  f"warmup={r['warmup']:.2f}s  "
                  f"sweeps_total={r['sweep_total']:.2f}s  "
                  f"cleanup={r['cleanup']:.2f}s  "
                  f"total={r['total']:.2f}s")
            print(f"    avg_sweep={avg_sweep:.3f}s  "
                  f"E={r['energy']:.12f}  "
                  f"converged={r['converged']}")
            results_for_case[np_count] = {**r, 'avg_sweep': avg_sweep}

        if len(results_for_case) >= 2:
            print(f"\n  {'─'*70}")
            print(f"  SCALING SUMMARY (1 PDMRG sweep = 2 serial sweep directions)")
            print(f"  {'np':>4s}  {'sites/rank':>10s}  {'avg sweep':>10s}  "
                  f"{'vs np=2':>10s}  {'ideal':>10s}  {'efficiency':>10s}")

            # Get L from case name
            import re
            L_match = re.search(r'L(\d+)', case)
            L = int(L_match.group(1)) if L_match else 0

            base = results_for_case.get(2)
            for np_count in np_counts:
                if np_count not in results_for_case:
                    continue
                r = results_for_case[np_count]
                sites_per_rank = L // np_count
                if base:
                    speedup = base['avg_sweep'] / r['avg_sweep']
                    ideal = np_count / 2  # relative to np=2
                    efficiency = speedup / ideal * 100
                    print(f"  {np_count:4d}  {sites_per_rank:10d}  "
                          f"{r['avg_sweep']:10.3f}s  "
                          f"{speedup:10.2f}x  "
                          f"{ideal:10.2f}x  "
                          f"{efficiency:9.1f}%")
                else:
                    print(f"  {np_count:4d}  {sites_per_rank:10d}  "
                          f"{r['avg_sweep']:10.3f}s")

        all_results[(model, case)] = results_for_case

    # Final summary table
    print(f"\n\n{'='*90}")
    print("FINAL SCALING TABLE")
    print(f"{'='*90}")
    print(f"{'Case':30s}", end='')
    for np_count in np_counts:
        print(f"  {'np='+str(np_count):>10s}", end='')
    print()

    for model, case in cases:
        key = (model, case)
        if key not in all_results or not all_results[key]:
            continue
        print(f"{model}/{case:20s}", end='')
        base_time = all_results[key].get(2, {}).get('avg_sweep', None)
        for np_count in np_counts:
            r = all_results[key].get(np_count)
            if r and base_time:
                speedup = base_time / r['avg_sweep']
                print(f"  {r['avg_sweep']:.3f}s({speedup:.1f}x)", end='')
            elif r:
                print(f"  {r['avg_sweep']:.3f}s     ", end='')
            else:
                print(f"  {'FAIL':>10s}", end='')
        print()


if __name__ == '__main__':
    main()
