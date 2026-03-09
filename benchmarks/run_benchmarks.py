#!/usr/bin/env python3
"""Run regular-tier benchmarks: quimb DMRG1/2, PDMRG, PDMRG2, A2DMRG.

Single-threaded BLAS, np=2,4,8 for parallel methods.
A2DMRG killed if it exceeds 10x the quimb reference time.
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

from benchmark_data_loader import (
    list_available_benchmarks, load_benchmark_case,
    convert_tensors_to_quimb_mpo, convert_tensors_to_quimb_mps,
)
import numpy as np
import quimb.tensor as qtn

MPIRUN = shutil.which('mpirun')
TOL = 1e-11


def run_quimb(mpo, mps, bond_dim, method='dmrg2'):
    dmrg_class = qtn.DMRG2 if method == 'dmrg2' else qtn.DMRG1
    t0 = time.time()
    dmrg = dmrg_class(mpo, bond_dims=bond_dim, cutoffs=1e-14,
                       p0=mps.copy() if mps else None)
    dmrg.solve(max_sweeps=50, tol=TOL, verbosity=0)
    t1 = time.time()
    return float(np.real(dmrg.energy)), t1 - t0


def run_mpi_method(model, case, np_count, implementation, timeout):
    """Run a parallel DMRG method via subprocess."""
    if implementation in ('pdmrg', 'pdmrg2'):
        func_import = 'from pdmrg.dmrg import pdmrg_main'
        call = f"""result = pdmrg_main(
    L=manifest['L'], mpo=mpo, max_sweeps=15,
    bond_dim=manifest['bond_dim'], bond_dim_warmup=manifest['bond_dim'],
    n_warmup_sweeps=3, tol={TOL}, dtype=manifest['dtype'],
    comm=comm, verbose=False, return_metadata=False,
    initial_mps=initial_mps)
energy, _ = result"""
    else:  # a2dmrg
        func_import = 'from a2dmrg.dmrg import a2dmrg_main'
        call = f"""dtype = np.complex128 if manifest['dtype'] == 'complex128' else np.float64
energy, _ = a2dmrg_main(
    L=manifest['L'], mpo=mpo, max_sweeps=40,
    bond_dim=manifest['bond_dim'], tol={TOL}, dtype=dtype,
    comm=comm, warmup_sweeps=5, verbose=False,
    initial_mps=initial_mps)"""

    script = f"""
import sys, os, time, json
import numpy as np
sys.path.insert(0, '{repo_root / implementation}')
sys.path.insert(0, '{repo_root / "benchmarks"}')
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from mpi4py import MPI
{func_import}
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
{call}
wall = time.time() - t0

if rank == 0:
    print(json.dumps({{'energy': float(np.real(energy)), 'time': wall}}))
"""
    script_path = f'/tmp/bench_{implementation}_{model}_{case}_np{np_count}.py'
    with open(script_path, 'w') as f:
        f.write(script)

    venv_python = repo_root / implementation / 'venv' / 'bin' / 'python'
    cmd = [
        MPIRUN, '-np', str(np_count), '--oversubscribe',
        '--mca', 'btl', 'tcp,self', '--mca', 'btl_tcp_if_include', 'lo',
        str(venv_python), script_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=timeout)
        if result.returncode != 0:
            err = (result.stderr or result.stdout or 'unknown')[-200:]
            return None, None, f'exit {result.returncode}: {err}'
        for line in result.stdout.strip().split('\n'):
            if line.startswith('{'):
                d = json.loads(line)
                return d['energy'], d['time'], None
        return None, None, 'no JSON output'
    except subprocess.TimeoutExpired:
        return None, None, f'KILLED (>{timeout:.0f}s)'
    except Exception as e:
        return None, None, str(e)


def main():
    benchmarks = list_available_benchmarks(tier='regular')
    print("=" * 90)
    print("DMRG BENCHMARK SUITE — Regular Tier")
    print(f"Single-threaded BLAS | np=2,4,8 | A2DMRG timeout: 10x quimb")
    print("=" * 90)

    all_results = {}

    for model in sorted(benchmarks):
        for case in sorted(benchmarks[model]):
            print(f"\n{'='*90}")
            print(f"  {model}/{case}")
            print(f"{'='*90}")

            data = load_benchmark_case(model, case)
            if data['golden_results'] is None:
                print("  (no golden results, skipping)")
                continue

            mpo = convert_tensors_to_quimb_mpo(data['mpo_tensors'])
            mps = convert_tensors_to_quimb_mps(data['mps_tensors']) if data['mps_tensors'] else None
            manifest = data['manifest']
            E_golden = data['golden_results']['quimb_dmrg2']['energy']
            bond_dim = manifest['bond_dim']

            results = {}
            quimb_max_time = 0

            # --- quimb DMRG1 ---
            print(f"\n  quimb DMRG1 ...", end='', flush=True)
            E, t = run_quimb(mpo, mps, bond_dim, 'dmrg1')
            dE = abs(E - E_golden)
            quimb_max_time = max(quimb_max_time, t)
            results['quimb_DMRG1'] = {'energy': E, 'time': t, 'dE': dE}
            print(f"  E={E:.12f}  ΔE={dE:.2e}  t={t:.2f}s")

            # --- quimb DMRG2 ---
            print(f"  quimb DMRG2 ...", end='', flush=True)
            E, t = run_quimb(mpo, mps, bond_dim, 'dmrg2')
            dE = abs(E - E_golden)
            quimb_max_time = max(quimb_max_time, t)
            results['quimb_DMRG2'] = {'energy': E, 'time': t, 'dE': dE}
            print(f"  E={E:.12f}  ΔE={dE:.2e}  t={t:.2f}s")

            a2dmrg_timeout = max(quimb_max_time * 10, 60)  # at least 60s

            # --- Parallel methods ---
            for impl in ('pdmrg', 'pdmrg2', 'a2dmrg'):
                for np_count in (2, 4, 8):
                    label = f"{impl.upper()} np={np_count}"
                    timeout = a2dmrg_timeout if impl == 'a2dmrg' else 600
                    print(f"  {label:20s} ...", end='', flush=True)

                    E, t, err = run_mpi_method(model, case, np_count, impl, timeout)
                    key = f'{impl.upper()}_np{np_count}'

                    if err:
                        results[key] = {'error': err}
                        print(f"  FAIL: {err}")
                    else:
                        dE = abs(E - E_golden)
                        results[key] = {'energy': E, 'time': t, 'dE': dE}
                        print(f"  E={E:.12f}  ΔE={dE:.2e}  t={t:.2f}s")

            all_results[f"{model}/{case}"] = results

    # --- Summary table ---
    print(f"\n\n{'='*90}")
    print("SUMMARY")
    print(f"{'='*90}")

    for test_case, results in all_results.items():
        print(f"\n  {test_case}:")
        print(f"  {'Method':24s} {'Energy':>20s} {'ΔE':>12s} {'Time':>10s}")
        print(f"  {'-'*66}")
        for method, r in results.items():
            if 'error' in r:
                print(f"  {method:24s} {'FAIL':>20s} {r['error']:>22s}")
            else:
                tag = '✓' if r['dE'] < 5e-10 else '✗'
                print(f"  {method:24s} {r['energy']:20.12f} {r['dE']:12.2e} {r['time']:9.2f}s {tag}")

    # Save
    out = repo_root / 'benchmarks' / 'benchmark_results.json'
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {out}")


if __name__ == '__main__':
    main()
