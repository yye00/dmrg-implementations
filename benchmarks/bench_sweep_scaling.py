#!/usr/bin/env python3
"""Measure PDMRG sweep-phase scaling: np=1 (serial) vs np=2.

This isolates the sweep phase cost to test Stoudenmire's claim that
sweep time scales as ~1/np. We exclude warmup and cleanup overhead.

The fair comparison is:
  - Our serial 2-site DMRG sweep (full chain, L sites)
  - Our PDMRG sweep (np=2, each processor sweeps L/2 sites + merge)

Both use the SAME eigensolver, SAME environment code, SAME SVD.
"""

import sys
import os
import time
import json
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
from pdmrg.mps.canonical import get_tensor_data, get_mpo_tensor_data
from pdmrg.numerics.eigensolver import optimize_two_site
from pdmrg.numerics.accurate_svd import truncated_svd
from pdmrg.environments.update import (
    update_left_env, update_right_env, init_left_env, init_right_env,
)

MPIRUN = shutil.which('mpirun')
TOL = 1e-11


def serial_sweep_time(mps_arrays, mpo_arrays, bond_dim, dtype, n_sweeps=4):
    """Time our serial 2-site DMRG sweeps (same code as PDMRG uses internally)."""
    L = len(mps_arrays)
    arrays = [a.copy() for a in mps_arrays]

    # Build environments
    L_envs = {0: init_left_env(arrays[0].shape[0], mpo_arrays[0].shape[0], dtype)}
    for i in range(L - 1):
        L_envs[i + 1] = update_left_env(L_envs[i], arrays[i], mpo_arrays[i])
    R_envs = {L - 1: init_right_env(arrays[-1].shape[2], mpo_arrays[-1].shape[1], dtype)}
    for i in range(L - 2, -1, -1):
        R_envs[i] = update_right_env(R_envs[i + 1], arrays[i + 1], mpo_arrays[i + 1])

    direction = 'right'
    sweep_times = []

    for sweep in range(n_sweeps):
        t0 = time.time()
        if direction == 'right':
            for j in range(L - 1):
                theta = np.einsum('ijk,klm->ijlm', arrays[j], arrays[j + 1])
                E, theta_opt = optimize_two_site(
                    L_envs[j], R_envs[j + 1], mpo_arrays[j], mpo_arrays[j + 1],
                    theta, max_iter=100, tol=1e-12)
                chi_L, d_L, d_R, chi_R = theta_opt.shape
                M = theta_opt.reshape(chi_L * d_L, d_R * chi_R)
                U, S, Vh, _ = truncated_svd(M, bond_dim)
                arrays[j] = U.reshape(chi_L, d_L, -1)
                arrays[j + 1] = (np.diag(S) @ Vh).reshape(-1, d_R, chi_R)
                L_envs[j + 1] = update_left_env(L_envs[j], arrays[j], mpo_arrays[j])
            direction = 'left'
        else:
            for j in range(L - 2, -1, -1):
                theta = np.einsum('ijk,klm->ijlm', arrays[j], arrays[j + 1])
                E, theta_opt = optimize_two_site(
                    L_envs[j], R_envs[j + 1], mpo_arrays[j], mpo_arrays[j + 1],
                    theta, max_iter=100, tol=1e-12)
                chi_L, d_L, d_R, chi_R = theta_opt.shape
                M = theta_opt.reshape(chi_L * d_L, d_R * chi_R)
                U, S, Vh, _ = truncated_svd(M, bond_dim)
                arrays[j + 1] = Vh.reshape(-1, d_R, chi_R)
                arrays[j] = (U @ np.diag(S)).reshape(chi_L, d_L, -1)
                R_envs[j] = update_right_env(R_envs[j + 1], arrays[j + 1], mpo_arrays[j + 1])
            direction = 'right'
        dt = time.time() - t0
        sweep_times.append(dt)
        print(f"    Serial sweep {sweep}: E={float(np.real(E)):.12f}  t={dt:.3f}s")

    return sweep_times


def run_pdmrg_sweep_only(model, case, np_count=2, pdmrg_dir='pdmrg', timeout=300):
    """Run PDMRG and extract per-sweep timings (excluding warmup/cleanup)."""
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
    script_path = f'/tmp/bench_sweep_{model}_{case}_{pdmrg_dir}.py'
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
        # Extract per-sweep times from verbose output
        sweep_times = []
        warmup_time = None
        for line in lines:
            if 'Warmup time:' in line:
                warmup_time = float(line.split('Warmup time:')[1].strip().rstrip('s'))
            if 'time =' in line and 'Sweep' in line:
                t = float(line.split('time =')[1].strip().rstrip('s'))
                sweep_times.append(t)
            if not line.startswith('{'):
                print(f"    {line}")

        # Get total time and energy from JSON
        for line in lines:
            if line.startswith('{'):
                d = json.loads(line)
                total = d['time']
                energy = d['energy']
                cleanup_time = total - (warmup_time or 0) - sum(sweep_times)
                return {
                    'energy': energy,
                    'total': total,
                    'warmup': warmup_time,
                    'sweep_times': sweep_times,
                    'sweep_total': sum(sweep_times),
                    'cleanup': cleanup_time,
                    'n_sweeps': len(sweep_times),
                }, None
        return None, 'no JSON output'
    except subprocess.TimeoutExpired:
        return None, f'KILLED (>{timeout:.0f}s)'


def main():
    benchmarks = list_available_benchmarks(tier='regular')

    print("=" * 80)
    print("SWEEP-PHASE SCALING: serial (our code) vs PDMRG np=2")
    print("=" * 80)
    print()
    print("This tests Stoudenmire's claim: sweep time ~ 1/np")
    print("We compare OUR serial sweep cost vs PDMRG sweep cost,")
    print("NOT quimb vs PDMRG (different implementations).")
    print()

    for model in sorted(benchmarks):
        for case in sorted(benchmarks[model]):
            print(f"\n{'='*80}")
            print(f"  {model}/{case}")
            print(f"{'='*80}")

            data = load_benchmark_case(model, case)
            if data['golden_results'] is None:
                print("  (no golden results, skipping)")
                continue

            mpo_q = convert_tensors_to_quimb_mpo(data['mpo_tensors'])
            mps_q = convert_tensors_to_quimb_mps(data['mps_tensors']) \
                if data['mps_tensors'] else None
            manifest = data['manifest']
            L = manifest['L']
            bond_dim = manifest['bond_dim']
            dtype = np.dtype(manifest['dtype'])

            # Get converged MPS from quimb (same starting point for both)
            dmrg2 = qtn.DMRG2(mpo_q, bond_dims=bond_dim, cutoffs=1e-14,
                               p0=mps_q.copy() if mps_q else None)
            dmrg2.solve(max_sweeps=50, tol=TOL, verbosity=0)
            E_ref = float(np.real(dmrg2.energy))
            print(f"  E_ref = {E_ref:.12f}")

            mps_ref = dmrg2.state.copy()
            mpo_arrays = [get_mpo_tensor_data(mpo_q, i) for i in range(L)]
            mps_ref.canonize(0)
            mps_arrays = [get_tensor_data(mps_ref, i) for i in range(L)]

            # Serial sweep timing (our code, full chain)
            print(f"\n  Serial sweeps (our eigensolver, L={L}):")
            serial_times = serial_sweep_time(
                mps_arrays, mpo_arrays, bond_dim, dtype, n_sweeps=4)

            # PDMRG np=2
            print(f"\n  PDMRG np=2:")
            pdmrg_result, err = run_pdmrg_sweep_only(model, case, np_count=2)
            if err:
                print(f"  FAIL: {err}")
                continue

            # PDMRG-cotengra np=2
            print(f"\n  PDMRG-cotengra np=2:")
            ctg_result, err = run_pdmrg_sweep_only(
                model, case, np_count=2, pdmrg_dir='pdmrg-cotengra')
            if err:
                print(f"  FAIL: {err}")
                ctg_result = None

            # Analysis
            avg_serial = np.mean(serial_times[1:])  # skip first (warmup effects)
            avg_pdmrg = np.mean(pdmrg_result['sweep_times'][1:]) \
                if len(pdmrg_result['sweep_times']) > 1 else pdmrg_result['sweep_times'][0]

            print(f"\n  {'─'*60}")
            print(f"  TIMING BREAKDOWN (PDMRG):")
            print(f"    Warmup (quimb serial):  {pdmrg_result['warmup']:.2f}s")
            print(f"    Sweeps ({pdmrg_result['n_sweeps']}×):          "
                  f"{pdmrg_result['sweep_total']:.2f}s  "
                  f"(avg {avg_pdmrg:.3f}s/sweep)")
            print(f"    Cleanup (quimb serial): {pdmrg_result['cleanup']:.2f}s")
            print(f"    Total:                  {pdmrg_result['total']:.2f}s")

            print(f"\n  SWEEP SCALING (Stoudenmire's claim):")
            print(f"    Serial sweep (L={L}):          {avg_serial:.3f}s/sweep")
            print(f"    PDMRG sweep (np=2, L/2={L//2}): {avg_pdmrg:.3f}s/sweep")
            ideal = avg_serial / 2
            print(f"    Ideal np=2 (serial/2):          {ideal:.3f}s/sweep")
            print(f"    Actual speedup:                  {avg_serial/avg_pdmrg:.2f}x "
                  f"(ideal=2.00x)")
            overhead = avg_pdmrg - ideal
            print(f"    Merge overhead:                  {overhead:.3f}s/sweep")

            if ctg_result:
                avg_ctg = np.mean(ctg_result['sweep_times'][1:]) \
                    if len(ctg_result['sweep_times']) > 1 else ctg_result['sweep_times'][0]
                print(f"    PDMRG-cotengra sweep:            {avg_ctg:.3f}s/sweep "
                      f"({avg_serial/avg_ctg:.2f}x)")


if __name__ == '__main__':
    main()
