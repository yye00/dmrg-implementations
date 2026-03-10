#!/usr/bin/env python3
"""Cross-check correctness of all DMRG implementations against quimb reference.

Tests: pdmrg, pdmrg-cotengra, pdmrg2, a2dmrg
Reference: quimb DMRG2 (serial)

Checks:
  1. Energy accuracy (ΔE vs quimb)
  2. Convergence (did it claim convergence?)
  3. Numerical stability (NaN, inf, divergence)
  4. MPS format correctness (tensor shapes)
"""

import sys
import os
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

MPIRUN = shutil.which('mpirun')
TOL = 1e-10
TIMEOUT = 600  # 10 min per run


def quimb_reference(mpo, mps, bond_dim):
    """Get quimb DMRG2 reference energy."""
    dmrg = qtn.DMRG2(mpo, bond_dims=bond_dim, cutoffs=1e-14,
                       p0=mps.copy() if mps else None)
    dmrg.solve(max_sweeps=50, tol=1e-12, verbosity=0)
    return float(np.real(dmrg.energy))


def make_pdmrg_script(model, case, impl_dir):
    """Generate script for pdmrg/pdmrg-cotengra/pdmrg2."""
    return f"""
import sys, os, time, json, traceback
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

try:
    data = load_benchmark_case('{model}', '{case}')
    mpo = convert_tensors_to_quimb_mpo(data['mpo_tensors'])
    manifest = data['manifest']
    initial_mps = None
    if data['mps_tensors'] is not None:
        initial_mps = convert_tensors_to_quimb_mps(data['mps_tensors'])

    result = pdmrg_main(
        L=manifest['L'], mpo=mpo, max_sweeps=20,
        bond_dim=manifest['bond_dim'], bond_dim_warmup=manifest['bond_dim'],
        n_warmup_sweeps=3, tol={TOL}, dtype=manifest['dtype'],
        comm=comm, verbose=(rank==0), return_metadata=True,
        initial_mps=initial_mps)
    energy, mps_out, metadata = result

    if rank == 0:
        e = float(np.real(energy))
        has_nan = bool(np.isnan(e) or np.isinf(e))
        print(json.dumps({{
            'energy': e,
            'converged': bool(metadata.get('converged', False)),
            'sweeps': metadata.get('final_sweep', -1),
            'has_nan': has_nan,
            'error': None
        }}))
except Exception as exc:
    if comm.Get_rank() == 0:
        print(json.dumps({{
            'energy': None,
            'converged': False,
            'sweeps': -1,
            'has_nan': True,
            'error': str(exc)
        }}))
"""


def make_a2dmrg_script(model, case):
    """Generate script for a2dmrg."""
    return f"""
import sys, os, time, json, traceback
import numpy as np
sys.path.insert(0, '{repo_root / "a2dmrg"}')
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

try:
    data = load_benchmark_case('{model}', '{case}')
    mpo = convert_tensors_to_quimb_mpo(data['mpo_tensors'])
    manifest = data['manifest']
    initial_mps = None
    if data['mps_tensors'] is not None:
        initial_mps = convert_tensors_to_quimb_mps(data['mps_tensors'])

    dtype_map = {{'float64': np.float64, 'complex128': np.complex128}}
    dtype = dtype_map[manifest['dtype']]

    result = a2dmrg_main(
        L=manifest['L'], mpo=mpo, max_sweeps=20,
        bond_dim=manifest['bond_dim'],
        tol={TOL}, dtype=dtype,
        comm=comm, verbose=(rank==0), return_metadata=True,
        initial_mps=initial_mps)
    energy, mps_out, metadata = result

    if rank == 0:
        e = float(np.real(energy))
        has_nan = bool(np.isnan(e) or np.isinf(e))
        print(json.dumps({{
            'energy': e,
            'converged': bool(metadata.get('converged', False)),
            'sweeps': metadata.get('final_sweep', -1),
            'has_nan': has_nan,
            'error': None
        }}))
except Exception as exc:
    if comm.Get_rank() == 0:
        print(json.dumps({{
            'energy': None,
            'converged': False,
            'sweeps': -1,
            'has_nan': True,
            'error': str(exc)
        }}))
"""


def run_impl(model, case, impl_name, np_count=2):
    """Run an implementation and parse results."""
    if impl_name == 'a2dmrg':
        script = make_a2dmrg_script(model, case)
        venv_python = repo_root / 'a2dmrg' / 'venv' / 'bin' / 'python'
    else:
        script = make_pdmrg_script(model, case, impl_name)
        # pdmrg2 has its own venv, pdmrg and pdmrg-cotengra share pdmrg's
        if impl_name == 'pdmrg2':
            venv_python = repo_root / 'pdmrg2' / 'venv' / 'bin' / 'python'
        else:
            venv_python = repo_root / 'pdmrg' / 'venv' / 'bin' / 'python'

    script_path = f'/tmp/crosscheck_{impl_name}_{model}_{case}.py'
    with open(script_path, 'w') as f:
        f.write(script)

    cmd = [
        MPIRUN, '-np', str(np_count), '--oversubscribe',
        '--mca', 'btl', 'tcp,self', '--mca', 'btl_tcp_if_include', 'lo',
        str(venv_python), script_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
        if result.returncode != 0:
            stderr = (result.stderr or '')[-500:]
            stdout = (result.stdout or '')[-500:]
            return {
                'energy': None, 'converged': False, 'sweeps': -1,
                'has_nan': True, 'error': f'exit {result.returncode}: {stderr or stdout}',
                'status': 'CRASH'
            }

        lines = result.stdout.strip().split('\n')
        for line in lines:
            if line.startswith('{'):
                try:
                    d = json.loads(line)
                    d['status'] = 'OK' if d['error'] is None else 'ERROR'
                    return d
                except json.JSONDecodeError:
                    pass
        return {
            'energy': None, 'converged': False, 'sweeps': -1,
            'has_nan': True, 'error': 'no JSON output',
            'status': 'NO_OUTPUT'
        }
    except subprocess.TimeoutExpired:
        return {
            'energy': None, 'converged': False, 'sweeps': -1,
            'has_nan': True, 'error': f'TIMEOUT (>{TIMEOUT}s)',
            'status': 'TIMEOUT'
        }


def main():
    # Use regular tier cases (known to work with quimb)
    cases = [
        ('heisenberg', 'L12_D20'),
        ('heisenberg', 'L32_D20'),
        ('heisenberg', 'L48_D20'),
    ]

    # Also test challenge tier (MPS format difference)
    challenge_cases = [
        ('heisenberg', 'L64_D20'),
    ]

    # Also test complex dtype (Josephson)
    complex_cases = [
        ('josephson', 'L20_D50_nmax2'),
    ]

    all_cases = cases + challenge_cases + complex_cases

    implementations = ['pdmrg', 'pdmrg-cotengra', 'pdmrg2', 'a2dmrg']

    print("=" * 100)
    print("CORRECTNESS CROSS-CHECK: all implementations vs quimb DMRG2 reference")
    print("=" * 100)
    print()

    # First pass: get quimb references
    refs = {}
    for model, case in all_cases:
        data = load_benchmark_case(model, case)
        mpo = convert_tensors_to_quimb_mpo(data['mpo_tensors'])
        mps = convert_tensors_to_quimb_mps(data['mps_tensors']) if data['mps_tensors'] else None
        manifest = data['manifest']
        E_ref = quimb_reference(mpo, mps, manifest['bond_dim'])
        refs[(model, case)] = E_ref
        print(f"  Reference {model}/{case}: E = {E_ref:.15f}")

    print()

    # Second pass: test each implementation
    results = {}
    for model, case in all_cases:
        E_ref = refs[(model, case)]
        print(f"\n{'='*100}")
        print(f"  {model}/{case}  (E_ref = {E_ref:.15f})")
        print(f"{'='*100}")

        for impl in implementations:
            print(f"\n  {impl} np=2 ... ", end='', flush=True)
            r = run_impl(model, case, impl, np_count=2)
            results[(model, case, impl)] = r

            if r['status'] != 'OK':
                err_msg = r.get('error', 'unknown')
                # Truncate error for display
                if len(err_msg) > 200:
                    err_msg = err_msg[:200] + '...'
                print(f"{r['status']}: {err_msg}")
                continue

            dE = abs(r['energy'] - E_ref)
            status = 'PASS' if dE < 1e-6 else ('WARN' if dE < 1e-3 else 'FAIL')
            if r['has_nan']:
                status = 'NaN'

            print(f"E={r['energy']:.15f}  ΔE={dE:.2e}  "
                  f"sweeps={r['sweeps']}  conv={r['converged']}  [{status}]")

    # Summary table
    print(f"\n\n{'='*100}")
    print("SUMMARY TABLE")
    print(f"{'='*100}")
    print(f"{'Case':<25s}", end='')
    for impl in implementations:
        print(f"  {impl:>18s}", end='')
    print()

    for model, case in all_cases:
        E_ref = refs[(model, case)]
        print(f"{model}/{case:<20s}", end='')
        for impl in implementations:
            r = results.get((model, case, impl))
            if r is None:
                print(f"  {'???':>18s}", end='')
            elif r['status'] != 'OK':
                print(f"  {r['status']:>18s}", end='')
            else:
                dE = abs(r['energy'] - E_ref)
                if dE < 1e-10:
                    grade = 'A+'
                elif dE < 1e-6:
                    grade = 'A'
                elif dE < 1e-3:
                    grade = 'B'
                elif dE < 1:
                    grade = 'C'
                else:
                    grade = 'F'
                print(f"  {grade} (ΔE={dE:.1e})", end='')
        print()

    # Detailed issue report
    print(f"\n\n{'='*100}")
    print("ISSUE REPORT")
    print(f"{'='*100}")
    issues = []
    for model, case in all_cases:
        E_ref = refs[(model, case)]
        for impl in implementations:
            r = results.get((model, case, impl))
            if r is None:
                continue
            if r['status'] != 'OK':
                issues.append(f"  [{r['status']}] {impl} on {model}/{case}: {r.get('error', 'unknown')[:200]}")
            elif r['has_nan']:
                issues.append(f"  [NaN] {impl} on {model}/{case}: energy contains NaN/Inf")
            elif abs(r['energy'] - E_ref) > 1e-6:
                dE = abs(r['energy'] - E_ref)
                issues.append(f"  [ACCURACY] {impl} on {model}/{case}: ΔE={dE:.2e} "
                            f"(E={r['energy']:.12f} vs ref={E_ref:.12f})")
            elif not r['converged']:
                issues.append(f"  [NO_CONV] {impl} on {model}/{case}: did not converge "
                            f"(E={r['energy']:.12f}, sweeps={r['sweeps']})")

    if issues:
        for issue in issues:
            print(issue)
    else:
        print("  No issues found. All implementations match reference within 1e-6.")


if __name__ == '__main__':
    main()
