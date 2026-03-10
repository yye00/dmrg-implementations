#!/usr/bin/env python3
"""Head-to-head benchmark: pdmrg vs pdmrg-cotengra vs a2dmrg.

Compares wall-clock time (the metric that matters for scaling studies)
across all production-ready parallel DMRG implementations.

All runs use np=2 (minimum for parallel) and np=4.
Reports total wall time, sweep time, and energy accuracy.
"""

import sys
import os
import json
import time
import subprocess
import shutil
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np
import quimb.tensor as qtn
from benchmark_data_loader import (
    list_available_benchmarks, load_benchmark_case,
    convert_tensors_to_quimb_mpo, convert_tensors_to_quimb_mps,
)

MPIRUN = shutil.which('mpirun')
TOL = 1e-11


def run_quimb_dmrg2(mpo, mps, bond_dim):
    """Run quimb DMRG2 as the reference serial baseline."""
    t0 = time.time()
    dmrg = qtn.DMRG2(mpo, bond_dims=bond_dim, cutoffs=1e-14,
                       p0=mps.copy() if mps else None)
    dmrg.solve(max_sweeps=50, tol=TOL, verbosity=0)
    wall = time.time() - t0
    return float(np.real(dmrg.energy)), wall


def run_mpi_impl(model, case, np_count, impl_dir, impl_module, timeout=600):
    """Run an MPI-based DMRG implementation and extract timings.

    Returns (result_dict, error_string). result_dict has:
      energy, total, sweep_times, warmup, n_sweeps, converged
    """
    if impl_dir == 'a2dmrg':
        venv_python = repo_root / impl_dir / 'venv' / 'bin' / 'python'
        script = f"""
import sys, os, time, json
import numpy as np
sys.path.insert(0, '{repo_root / impl_dir}')
sys.path.insert(0, '{repo_root / "benchmarks"}')
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from mpi4py import MPI
from a2dmrg.dmrg import a2dmrg_main
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
result = a2dmrg_main(
    L=manifest['L'], mpo=mpo, max_sweeps=20,
    bond_dim=manifest['bond_dim'],
    tol={TOL}, dtype=np.dtype(manifest['dtype']),
    comm=comm, verbose=(rank==0), return_metadata=True,
    initial_mps=initial_mps)
energy, _, metadata = result
wall = time.time() - t0

if rank == 0:
    print(json.dumps({{'energy': float(np.real(energy)), 'time': wall,
                       'sweeps': metadata.get('final_sweep', -1),
                       'converged': metadata.get('converged', False)}}))
"""
    else:
        venv_python = repo_root / 'pdmrg' / 'venv' / 'bin' / 'python'
        script = f"""
import sys, os, time, json
import numpy as np
sys.path.insert(0, '{repo_root / impl_dir}')
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

    script_path = f'/tmp/bench_h2h_{impl_dir}_{model}_{case}_np{np_count}.py'
    with open(script_path, 'w') as f:
        f.write(script)

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
            if 'time =' in line and ('Sweep' in line or 'Iter' in line):
                t = float(line.split('time =')[1].strip().rstrip('s'))
                sweep_times.append(t)

        for line in lines:
            if line.startswith('{'):
                d = json.loads(line)
                total = d['time']
                energy = d['energy']
                return {
                    'energy': energy,
                    'total': total,
                    'warmup': warmup_time or 0,
                    'sweep_times': sweep_times,
                    'sweep_total': sum(sweep_times),
                    'n_sweeps': len(sweep_times) or d.get('sweeps', -1),
                    'converged': d.get('converged', False),
                }, None
        return None, 'no JSON output'
    except subprocess.TimeoutExpired:
        return None, f'KILLED (>{timeout:.0f}s)'


def main():
    # Benchmark cases spanning small to large
    cases = [
        ('heisenberg', 'L32_D20'),
        ('heisenberg', 'L48_D20'),
        ('heisenberg', 'L64_D20'),
        ('heisenberg', 'L96_D20'),
        ('heisenberg', 'L128_D20'),
    ]

    implementations = [
        ('pdmrg', 'pdmrg', 'pdmrg.dmrg'),
        ('pdmrg-cotengra', 'pdmrg-cotengra', 'pdmrg.dmrg'),
        ('a2dmrg', 'a2dmrg', 'a2dmrg.dmrg'),
    ]

    np_counts = [2, 4]

    print("=" * 100)
    print("HEAD-TO-HEAD: pdmrg vs pdmrg-cotengra vs a2dmrg  (wall-clock time)")
    print("=" * 100)
    print(f"Baseline: quimb DMRG2 (serial, single-threaded)")
    print(f"All MPI runs: OMP_NUM_THREADS=1, --oversubscribe, loopback network")
    print()

    all_results = {}

    for model, case in cases:
        print(f"\n{'='*100}")
        print(f"  {model}/{case}")
        print(f"{'='*100}")

        data = load_benchmark_case(model, case)
        mpo = convert_tensors_to_quimb_mpo(data['mpo_tensors'])
        mps = convert_tensors_to_quimb_mps(data['mps_tensors']) if data['mps_tensors'] else None
        manifest = data['manifest']
        bond_dim = manifest['bond_dim']
        L = manifest['L']

        # quimb DMRG2 baseline
        print(f"\n  quimb DMRG2 (serial) ...", end='', flush=True)
        E_ref, t_ref = run_quimb_dmrg2(mpo, mps, bond_dim)
        print(f"  E={E_ref:.12f}  t={t_ref:.2f}s")

        case_results = {'quimb': {'energy': E_ref, 'total': t_ref}}

        for impl_name, impl_dir, impl_module in implementations:
            for np_count in np_counts:
                label = f"{impl_name} np={np_count}"
                print(f"\n  {label} ...", end='', flush=True)
                r, err = run_mpi_impl(model, case, np_count, impl_dir, impl_module,
                                       timeout=1800)
                if err:
                    print(f"  FAIL: {err[:200]}")
                    case_results[label] = None
                else:
                    dE = abs(r['energy'] - E_ref)
                    speedup = t_ref / r['total']
                    print(f"  E={r['energy']:.12f}  ΔE={dE:.2e}  "
                          f"t={r['total']:.2f}s  ({speedup:.2f}x vs quimb)  "
                          f"sweeps={r['n_sweeps']}  conv={r['converged']}")
                    case_results[label] = {**r, 'dE': dE, 'speedup_vs_quimb': speedup}

        # Summary table for this case
        print(f"\n  {'─'*80}")
        print(f"  SUMMARY: {model}/{case} (L={L}, D={bond_dim})")
        print(f"  {'Method':<25s} {'Energy':>18s} {'ΔE':>10s} {'Wall(s)':>10s} {'vs quimb':>10s}")
        print(f"  {'quimb DMRG2 (serial)':<25s} {E_ref:18.12f} {'—':>10s} {t_ref:10.2f} {'1.00x':>10s}")
        for impl_name, _, _ in implementations:
            for np_count in np_counts:
                label = f"{impl_name} np={np_count}"
                r = case_results.get(label)
                if r:
                    print(f"  {label:<25s} {r['energy']:18.12f} {r['dE']:10.2e} "
                          f"{r['total']:10.2f} {r['speedup_vs_quimb']:9.2f}x")
                else:
                    print(f"  {label:<25s} {'FAIL':>18s}")

        all_results[(model, case)] = case_results

    # Final comparison table
    print(f"\n\n{'='*100}")
    print("FINAL COMPARISON: Wall-clock speedup vs quimb DMRG2")
    print(f"{'='*100}")
    header = f"{'Case':<25s}"
    impl_labels = []
    for impl_name, _, _ in implementations:
        for np_count in np_counts:
            label = f"{impl_name} np={np_count}"
            impl_labels.append(label)
            header += f"  {label:>20s}"
    print(header)

    for model, case in cases:
        key = (model, case)
        if key not in all_results:
            continue
        row = f"{model}/{case:<20s}"
        for label in impl_labels:
            r = all_results[key].get(label)
            if r:
                row += f"  {r['total']:6.2f}s ({r['speedup_vs_quimb']:.2f}x)"
            else:
                row += f"  {'FAIL':>20s}"
        print(row)


if __name__ == '__main__':
    main()
