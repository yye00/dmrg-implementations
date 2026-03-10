#!/usr/bin/env python3
"""Focused benchmark: quimb DMRG1/2 and PDMRG np=2 only."""

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


def run_pdmrg(model, case, np_count=2, timeout=300, pdmrg_dir='pdmrg'):
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
    L=manifest['L'], mpo=mpo, max_sweeps=15,
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
    script_path = f'/tmp/bench_pdmrg_{model}_{case}.py'
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
            err = (result.stderr or result.stdout or 'unknown')[-500:]
            return None, None, f'exit {result.returncode}: {err}'
        # Print verbose output
        for line in result.stdout.strip().split('\n'):
            if not line.startswith('{'):
                print(f"    {line}")
        for line in result.stdout.strip().split('\n'):
            if line.startswith('{'):
                d = json.loads(line)
                return d['energy'], d['time'], d
        return None, None, 'no JSON output'
    except subprocess.TimeoutExpired:
        return None, None, f'KILLED (>{timeout:.0f}s)'
    except Exception as e:
        return None, None, str(e)


def main():
    benchmarks = list_available_benchmarks(tier='regular')
    print("=" * 80)
    print("FOCUSED BENCHMARK: quimb DMRG1/2 vs PDMRG np=2")
    print("=" * 80)

    for model in sorted(benchmarks):
        for case in sorted(benchmarks[model]):
            print(f"\n{'='*80}")
            print(f"  {model}/{case}")
            print(f"{'='*80}")

            data = load_benchmark_case(model, case)
            if data['golden_results'] is None:
                print("  (no golden results, skipping)")
                continue

            mpo = convert_tensors_to_quimb_mpo(data['mpo_tensors'])
            mps = convert_tensors_to_quimb_mps(data['mps_tensors']) if data['mps_tensors'] else None
            manifest = data['manifest']
            E_golden = data['golden_results']['quimb_dmrg2']['energy']
            bond_dim = manifest['bond_dim']

            # quimb DMRG1
            print(f"\n  quimb DMRG1 ...", end='', flush=True)
            E1, t1 = run_quimb(mpo, mps, bond_dim, 'dmrg1')
            dE1 = abs(E1 - E_golden)
            print(f"  E={E1:.12f}  ΔE={dE1:.2e}  t={t1:.2f}s")

            # quimb DMRG2
            print(f"  quimb DMRG2 ...", end='', flush=True)
            E2, t2 = run_quimb(mpo, mps, bond_dim, 'dmrg2')
            dE2 = abs(E2 - E_golden)
            print(f"  E={E2:.12f}  ΔE={dE2:.2e}  t={t2:.2f}s")

            # PDMRG np=2
            print(f"\n  PDMRG np=2:")
            E_p, t_p, meta = run_pdmrg(model, case, np_count=2)
            if isinstance(meta, str):
                print(f"  FAIL: {meta}")
            else:
                dE_p = abs(E_p - E_golden)
                sweeps = meta.get('sweeps', '?') if isinstance(meta, dict) else '?'
                conv = meta.get('converged', '?') if isinstance(meta, dict) else '?'
                print(f"  E={E_p:.12f}  ΔE={dE_p:.2e}  t={t_p:.2f}s  sweeps={sweeps}  converged={conv}")

            # PDMRG-cotengra np=2
            print(f"\n  PDMRG-cotengra np=2:")
            E_pc, t_pc, meta_c = run_pdmrg(model, case, np_count=2,
                                             pdmrg_dir='pdmrg-cotengra')
            if isinstance(meta_c, str):
                print(f"  FAIL: {meta_c}")
            else:
                dE_pc = abs(E_pc - E_golden)
                sweeps_c = meta_c.get('sweeps', '?') if isinstance(meta_c, dict) else '?'
                conv_c = meta_c.get('converged', '?') if isinstance(meta_c, dict) else '?'
                print(f"  E={E_pc:.12f}  ΔE={dE_pc:.2e}  t={t_pc:.2f}s  sweeps={sweeps_c}  converged={conv_c}")

            # Summary
            print(f"\n  --- Summary ---")
            print(f"  {'Method':20s} {'ΔE':>12s} {'Time':>10s} {'Speedup':>10s}")
            if E_p is not None:
                print(f"  {'quimb DMRG1':20s} {dE1:12.2e} {t1:10.2f}s {'1.00x':>10s}")
                print(f"  {'quimb DMRG2':20s} {dE2:12.2e} {t2:10.2f}s {t1/t2:10.2f}x")
                print(f"  {'PDMRG np=2':20s} {dE_p:12.2e} {t_p:10.2f}s {t1/t_p:10.2f}x")
            if E_pc is not None and not isinstance(meta_c, str):
                print(f"  {'PDMRG-cotengra np=2':20s} {dE_pc:12.2e} {t_pc:10.2f}s {t1/t_pc:10.2f}x")


if __name__ == '__main__':
    main()
