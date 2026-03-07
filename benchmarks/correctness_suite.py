#!/usr/bin/env python3
"""
Correctness Benchmark Suite

Tests all DMRG implementations against golden quimb DMRG2 references.
Uses static benchmark data to ensure fair comparison.

Methods tested:
- quimb DMRG1 (serial reference)
- quimb DMRG2 (serial reference, golden standard)
- PDMRG (np=1,2,4,8)
- PDMRG2 (np=1,2,4,8)
- A2DMRG (np=1,2,4,8)

Validation thresholds:
- Machine precision target: |ΔE| < 1e-12
- Acceptance threshold: |ΔE| < 5e-10
"""

import sys
import os
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
import numpy as np

# Validation thresholds
MACHINE_PRECISION_THRESHOLD = 1e-12  # Gold standard
ACCEPTANCE_THRESHOLD = 5e-10  # Acceptable agreement (order 1e-10)

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

# Force single-threaded
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from benchmark_data_loader import list_available_benchmarks, load_benchmark_case, convert_tensors_to_quimb_mpo
import quimb.tensor as qtn


def run_quimb_dmrg(model, case, method='dmrg2'):
    """Run quimb DMRG1 or DMRG2."""
    data = load_benchmark_case(model, case)
    mpo = convert_tensors_to_quimb_mpo(data['mpo_tensors'])
    manifest = data['manifest']

    dmrg_class = qtn.DMRG2 if method == 'dmrg2' else qtn.DMRG1

    t0 = time.time()
    dmrg = dmrg_class(mpo, bond_dims=manifest['bond_dim'], cutoffs=1e-14)
    dmrg.solve(max_sweeps=50, tol=ACCEPTANCE_THRESHOLD, verbosity=0)
    t1 = time.time()

    return {
        'method': f'quimb_{method.upper()}',
        'energy': float(np.real(dmrg.energy)),
        'time': t1 - t0,
        'sweeps': len(dmrg.energies) if hasattr(dmrg, 'energies') else 50,
        'success': True
    }


def run_pdmrg(model, case, np_count, implementation='pdmrg'):
    """Run PDMRG or PDMRG2 with MPI."""
    data = load_benchmark_case(model, case)
    manifest = data['manifest']

    # Create temp script
    script = f'''
import sys
import os
import numpy as np
sys.path.insert(0, '{repo_root / implementation}')
sys.path.insert(0, '{repo_root}')

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from mpi4py import MPI
from pdmrg.dmrg import pdmrg_main
from benchmark_data_loader import load_benchmark_case, convert_tensors_to_quimb_mpo

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

data = load_benchmark_case('{model}', '{case}')
mpo = convert_tensors_to_quimb_mpo(data['mpo_tensors'])
manifest = data['manifest']

import time
t0 = time.time()
result = pdmrg_main(
    L=manifest['L'],
    mpo=mpo,
    max_sweeps=30,
    bond_dim=manifest['bond_dim'],
    bond_dim_warmup=manifest['bond_dim'],
    n_warmup_sweeps=5,
    tol={ACCEPTANCE_THRESHOLD},
    dtype=manifest['dtype'],
    comm=comm,
    verbose=False,
    return_metadata=True
)
t1 = time.time()

if len(result) == 3:
    energy, pmps, metadata = result
else:
    energy, pmps = result
    metadata = None

if rank == 0:
    import json
    print(json.dumps({{
        'energy': float(np.real(energy)),
        'time': t1 - t0,
        'metadata': metadata
    }}, default=str))
'''

    script_path = f'/tmp/{implementation}_benchmark_{model}_{case}_np{np_count}.py'
    with open(script_path, 'w') as f:
        f.write(script)

    # Run with MPI
    venv_python = repo_root / implementation / 'venv' / 'bin' / 'python'

    cmd = [
        '/usr/lib64/openmpi/bin/mpirun',
        '-np', str(np_count),
        '--oversubscribe',
        '--mca', 'btl', 'tcp,self',
        '--mca', 'btl_tcp_if_include', 'lo',
        str(venv_python),
        script_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                               env={**os.environ, 'PATH': '/usr/lib64/openmpi/bin:' + os.environ.get('PATH', '')})

        if result.returncode != 0:
            return {
                'method': f'{implementation.upper()}_np{np_count}',
                'success': False,
                'error': result.stderr[-500:] if result.stderr else result.stdout[-500:]
            }

        # Parse JSON output
        for line in result.stdout.strip().split('\n'):
            if line.startswith('{'):
                data = json.loads(line)
                return {
                    'method': f'{implementation.upper()}_np{np_count}',
                    'energy': data['energy'],
                    'time': data['time'],
                    'metadata': data.get('metadata'),
                    'success': True
                }

        return {
            'method': f'{implementation.upper()}_np{np_count}',
            'success': False,
            'error': 'No JSON output found'
        }

    except subprocess.TimeoutExpired:
        return {
            'method': f'{implementation.upper()}_np{np_count}',
            'success': False,
            'error': 'Timeout (600s)'
        }
    except Exception as e:
        return {
            'method': f'{implementation.upper()}_np{np_count}',
            'success': False,
            'error': str(e)
        }


def run_a2dmrg(model, case, np_count):
    """Run A2DMRG with MPI."""
    data = load_benchmark_case(model, case)
    manifest = data['manifest']

    script = f'''
import sys
import os
import numpy as np
sys.path.insert(0, '{repo_root / "a2dmrg"}')
sys.path.insert(0, '{repo_root}')

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from mpi4py import MPI
from a2dmrg.dmrg import a2dmrg_main
from benchmark_data_loader import load_benchmark_case, convert_tensors_to_quimb_mpo

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

data = load_benchmark_case('{model}', '{case}')
mpo = convert_tensors_to_quimb_mpo(data['mpo_tensors'])
manifest = data['manifest']

import time
t0 = time.time()
dtype = np.complex128 if manifest['dtype'] == 'complex128' else np.float64
energy, mps = a2dmrg_main(
    L=manifest['L'],
    mpo=mpo,
    max_sweeps=30,
    bond_dim=manifest['bond_dim'],
    tol={ACCEPTANCE_THRESHOLD},
    dtype=dtype,
    comm=comm,
    warmup_sweeps=5,
    verbose=False
)
t1 = time.time()

if rank == 0:
    import json
    print(json.dumps({{
        'energy': float(np.real(energy)),
        'time': t1 - t0
    }}, default=str))
'''

    script_path = f'/tmp/a2dmrg_benchmark_{model}_{case}_np{np_count}.py'
    with open(script_path, 'w') as f:
        f.write(script)

    venv_python = repo_root / 'a2dmrg' / 'venv' / 'bin' / 'python'

    cmd = [
        '/usr/lib64/openmpi/bin/mpirun',
        '-np', str(np_count),
        '--oversubscribe',
        '--mca', 'btl', 'tcp,self',
        '--mca', 'btl_tcp_if_include', 'lo',
        str(venv_python),
        script_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                               env={**os.environ, 'PATH': '/usr/lib64/openmpi/bin:' + os.environ.get('PATH', '')})

        if result.returncode != 0:
            return {
                'method': f'A2DMRG_np{np_count}',
                'success': False,
                'error': result.stderr[-500:] if result.stderr else result.stdout[-500:]
            }

        for line in result.stdout.strip().split('\n'):
            if line.startswith('{'):
                data = json.loads(line)
                return {
                    'method': f'A2DMRG_np{np_count}',
                    'energy': data['energy'],
                    'time': data['time'],
                    'success': True
                }

        return {
            'method': f'A2DMRG_np{np_count}',
            'success': False,
            'error': 'No JSON output found'
        }

    except subprocess.TimeoutExpired:
        return {
            'method': f'A2DMRG_np{np_count}',
            'success': False,
            'error': 'Timeout (600s)'
        }
    except Exception as e:
        return {
            'method': f'A2DMRG_np{np_count}',
            'success': False,
            'error': str(e)
        }


def run_correctness_suite(tier='regular'):
    """Run full correctness test suite.

    Parameters
    ----------
    tier : str
        Benchmark tier to test: 'regular', 'challenge', or 'all'
    """
    print("="*80)
    print("DMRG CORRECTNESS BENCHMARK SUITE")
    print("="*80)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Tier: {tier}")
    print(f"Threads: OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
    print()

    # Get available benchmarks for the specified tier
    if tier == 'all':
        benchmarks = list_available_benchmarks()
    else:
        benchmarks = list_available_benchmarks(tier=tier)

    if not benchmarks:
        print("ERROR: No benchmark data found!")
        print("Run: python scripts/generate_benchmark_data.py --all")
        return 1

    all_results = {}

    for model, cases in sorted(benchmarks.items()):
        for case in sorted(cases):
            print(f"\n{'='*80}")
            print(f"TEST CASE: {model}/{case}")
            print(f"{'='*80}")

            # Load golden reference
            data = load_benchmark_case(model, case)
            golden = data['golden_results']
            E_golden = golden['quimb_dmrg2']['energy']

            print(f"\nGolden Reference (quimb DMRG2): {E_golden:.15f}")

            case_results = {}

            # Test quimb DMRG1
            print(f"\n[1/14] Running quimb DMRG1...")
            result = run_quimb_dmrg(model, case, 'dmrg1')
            case_results['quimb_DMRG1'] = result
            if result['success']:
                dE = result['energy'] - E_golden
                result['delta_E'] = dE
                result['machine_precision_pass'] = abs(dE) < MACHINE_PRECISION_THRESHOLD
                result['acceptance_pass'] = abs(dE) < ACCEPTANCE_THRESHOLD
                status = "✓" if result['acceptance_pass'] else "✗"
                print(f"  {status} E = {result['energy']:.15f}, ΔE = {dE:.3e}, t = {result['time']:.2f}s")

            # Test quimb DMRG2 (should match golden exactly)
            print(f"\n[2/14] Running quimb DMRG2...")
            result = run_quimb_dmrg(model, case, 'dmrg2')
            case_results['quimb_DMRG2'] = result
            if result['success']:
                dE = result['energy'] - E_golden
                result['delta_E'] = dE
                result['machine_precision_pass'] = abs(dE) < MACHINE_PRECISION_THRESHOLD
                result['acceptance_pass'] = abs(dE) < ACCEPTANCE_THRESHOLD
                status = "✓" if result['machine_precision_pass'] else "✗"
                print(f"  {status} E = {result['energy']:.15f}, ΔE = {dE:.3e}, t = {result['time']:.2f}s")

            # Test PDMRG
            for idx, np_count in enumerate([1, 2, 4, 8], start=3):
                print(f"\n[{idx}/14] Running PDMRG np={np_count}...")
                result = run_pdmrg(model, case, np_count, 'pdmrg')
                case_results[f'PDMRG_np{np_count}'] = result

                if result['success']:
                    dE = result['energy'] - E_golden
                    result['delta_E'] = dE
                    result['machine_precision_pass'] = abs(dE) < MACHINE_PRECISION_THRESHOLD
                    result['acceptance_pass'] = abs(dE) < ACCEPTANCE_THRESHOLD
                    status = "✓" if result['acceptance_pass'] else "✗"
                    print(f"  {status} E = {result['energy']:.15f}, ΔE = {dE:.3e}, t = {result['time']:.2f}s")
                else:
                    print(f"  ✗ FAILED: {result.get('error', 'Unknown')[:100]}")

            # Test PDMRG2
            for idx, np_count in enumerate([1, 2, 4, 8], start=7):
                print(f"\n[{idx}/14] Running PDMRG2 np={np_count}...")
                result = run_pdmrg(model, case, np_count, 'pdmrg2')
                case_results[f'PDMRG2_np{np_count}'] = result

                if result['success']:
                    dE = result['energy'] - E_golden
                    result['delta_E'] = dE
                    result['machine_precision_pass'] = abs(dE) < MACHINE_PRECISION_THRESHOLD
                    result['acceptance_pass'] = abs(dE) < ACCEPTANCE_THRESHOLD
                    status = "✓" if result['acceptance_pass'] else "✗"
                    print(f"  {status} E = {result['energy']:.15f}, ΔE = {dE:.3e}, t = {result['time']:.2f}s")
                else:
                    print(f"  ✗ FAILED: {result.get('error', 'Unknown')[:100]}")

            # Test A2DMRG
            for idx, np_count in enumerate([2, 4, 8], start=11):
                print(f"\n[{idx}/14] Running A2DMRG np={np_count}...")
                result = run_a2dmrg(model, case, np_count)
                case_results[f'A2DMRG_np{np_count}'] = result

                if result['success']:
                    dE = result['energy'] - E_golden
                    result['delta_E'] = dE
                    result['machine_precision_pass'] = abs(dE) < MACHINE_PRECISION_THRESHOLD
                    result['acceptance_pass'] = abs(dE) < ACCEPTANCE_THRESHOLD
                    status = "✓" if result['acceptance_pass'] else "✗"
                    print(f"  {status} E = {result['energy']:.15f}, ΔE = {dE:.3e}, t = {result['time']:.2f}s")
                else:
                    print(f"  ✗ FAILED: {result.get('error', 'Unknown')[:100]}")

            all_results[f"{model}/{case}"] = case_results

    # Summary
    print(f"\n{'='*80}")
    print("CORRECTNESS SUMMARY")
    print(f"{'='*80}")

    machine_precision_count = 0
    acceptance_count = 0
    fail_count = 0
    total_count = 0

    for test_case, case_results in all_results.items():
        print(f"\n{test_case}:")
        golden = load_benchmark_case(*test_case.split('/'))['golden_results']['quimb_dmrg2']['energy']

        for method, result in case_results.items():
            total_count += 1
            if result.get('success'):
                dE = abs(result['energy'] - golden)
                if dE < MACHINE_PRECISION_THRESHOLD:
                    print(f"  ✓✓ {method}: ΔE = {dE:.3e} (MACHINE PRECISION)")
                    machine_precision_count += 1
                elif dE < ACCEPTANCE_THRESHOLD:
                    print(f"  ✓  {method}: ΔE = {dE:.3e} (ACCEPTED)")
                    acceptance_count += 1
                else:
                    print(f"  ✗  {method}: ΔE = {dE:.3e} (FAIL)")
                    fail_count += 1
            else:
                print(f"  ✗  {method}: FAILED")
                fail_count += 1

    print(f"\n{'='*80}")
    print(f"TOTALS:")
    print(f"  Machine precision (ΔE < 1e-12): {machine_precision_count}/{total_count}")
    print(f"  Acceptance (ΔE < 5e-10):        {acceptance_count}/{total_count}")
    print(f"  Failures:                       {fail_count}/{total_count}")
    print(f"{'='*80}")

    # Save results
    output_file = repo_root / 'benchmarks' / 'correctness_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': all_results,
            'summary': {
                'machine_precision': machine_precision_count,
                'acceptance': acceptance_count,
                'failures': fail_count,
                'total': total_count
            }
        }, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    return 0 if fail_count == 0 else 1


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run DMRG correctness benchmark suite')
    parser.add_argument('--tier', choices=['regular', 'challenge', 'all'], default='regular',
                        help='Benchmark tier to test (default: regular)')
    args = parser.parse_args()

    sys.exit(run_correctness_suite(tier=args.tier))
