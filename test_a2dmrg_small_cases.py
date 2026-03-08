#!/usr/bin/env python3
"""
Test A2DMRG on small Heisenberg and Josephson cases to validate convergence.

Tests:
- Heisenberg L12_D20 (smallest Heisenberg case)
- Josephson L20_D50_nmax2 (smallest Josephson case)

Compares against golden quimb DMRG2 reference.
"""

import sys
import os
import json
import time
import subprocess
import shutil
from pathlib import Path
import numpy as np

# Force single-threaded BLAS
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

from benchmark_data_loader import load_benchmark_case, convert_tensors_to_quimb_mpo
import quimb.tensor as qtn

# Validation thresholds
GOLDEN_TOLERANCE = 1e-11
MACHINE_PRECISION_THRESHOLD = 1e-12
ACCEPTANCE_THRESHOLD = 5e-10


def run_quimb_dmrg2(model, case):
    """Run quimb DMRG2 to generate/verify golden reference."""
    print(f"  Running quimb DMRG2...")
    data = load_benchmark_case(model, case)
    mpo = convert_tensors_to_quimb_mpo(data['mpo_tensors'])
    manifest = data['manifest']

    t0 = time.time()
    dmrg = qtn.DMRG2(mpo, bond_dims=manifest['bond_dim'], cutoffs=1e-14)
    dmrg.solve(max_sweeps=50, tol=GOLDEN_TOLERANCE, verbosity=0)
    t1 = time.time()

    return float(np.real(dmrg.energy)), t1 - t0


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
    max_sweeps=40,
    bond_dim=manifest['bond_dim'],
    tol={GOLDEN_TOLERANCE},
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

    script_path = f'/tmp/a2dmrg_test_{model}_{case}_np{np_count}.py'
    with open(script_path, 'w') as f:
        f.write(script)

    venv_python = repo_root / 'a2dmrg' / 'venv' / 'bin' / 'python'
    mpirun = shutil.which('mpirun')

    if not mpirun:
        return None, None, 'mpirun not found'

    cmd = [
        mpirun, '-np', str(np_count),
        '--oversubscribe',
        '--mca', 'btl', 'tcp,self',
        '--mca', 'btl_tcp_if_include', 'lo',
        str(venv_python),
        script_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            return None, None, result.stderr[-500:]

        for line in result.stdout.strip().split('\n'):
            if line.startswith('{'):
                data = json.loads(line)
                return data['energy'], data['time'], None

        return None, None, 'No JSON output'

    except subprocess.TimeoutExpired:
        return None, None, 'Timeout (600s)'
    except Exception as e:
        return None, None, str(e)


def test_case(model, case):
    """Test a single benchmark case."""
    print(f"\n{'='*80}")
    print(f"TEST CASE: {model}/{case}")
    print(f"{'='*80}\n")

    # Load golden reference
    data = load_benchmark_case(model, case)
    manifest = data['manifest']
    print(f"Configuration: L={manifest['L']}, D={manifest['bond_dim']}, dtype={manifest['dtype']}")

    # Get golden energy
    golden_data = data['golden_results']
    E_golden = golden_data['quimb_dmrg2']['energy']
    print(f"\nGolden Reference: {E_golden:.15f}")

    # Verify with fresh quimb run
    print()
    E_quimb, t_quimb = run_quimb_dmrg2(model, case)
    dE_quimb = E_quimb - E_golden
    print(f"  ✓ quimb DMRG2: E = {E_quimb:.15f}, ΔE = {dE_quimb:.3e}, t = {t_quimb:.3f}s")

    # Test A2DMRG with different np counts
    results = {}

    for np_count in [2, 4]:
        print(f"\n  Running A2DMRG np={np_count}...")
        E_a2dmrg, t_a2dmrg, error = run_a2dmrg(model, case, np_count)

        if error:
            print(f"    ✗ FAILED: {error[:100]}")
            results[f'np{np_count}'] = {'success': False, 'error': error}
        else:
            dE = E_a2dmrg - E_golden
            machine_prec = abs(dE) < MACHINE_PRECISION_THRESHOLD
            accepted = abs(dE) < ACCEPTANCE_THRESHOLD

            if machine_prec:
                status = "✓✓ MACHINE PRECISION"
            elif accepted:
                status = "✓  ACCEPTED"
            else:
                status = "✗  FAILED"

            print(f"    {status}")
            print(f"    E = {E_a2dmrg:.15f}")
            print(f"    ΔE = {dE:.3e}")
            print(f"    t = {t_a2dmrg:.3f}s")

            results[f'np{np_count}'] = {
                'success': True,
                'energy': E_a2dmrg,
                'delta_E': dE,
                'time': t_a2dmrg,
                'machine_precision': machine_prec,
                'accepted': accepted
            }

    return {
        'golden': E_golden,
        'quimb_verify': {'energy': E_quimb, 'delta_E': dE_quimb, 'time': t_quimb},
        'a2dmrg': results
    }


def main():
    print("="*80)
    print("A2DMRG VALIDATION TEST - SMALL CASES")
    print("="*80)
    print("Testing fixed A2DMRG implementation on smallest benchmark cases")
    print()
    print(f"Environment:")
    print(f"  OMP_NUM_THREADS = {os.environ.get('OMP_NUM_THREADS')}")
    print(f"  OPENBLAS_NUM_THREADS = {os.environ.get('OPENBLAS_NUM_THREADS')}")
    print()

    all_results = {}

    # Test Heisenberg L12
    heisenberg_result = test_case('heisenberg', 'L12_D20')
    all_results['heisenberg/L12_D20'] = heisenberg_result

    # Test Josephson L20
    josephson_result = test_case('josephson', 'L20_D50_nmax2')
    all_results['josephson/L20_D50_nmax2'] = josephson_result

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")

    total_tests = 0
    machine_precision_count = 0
    accepted_count = 0
    failed_count = 0

    for case_name, case_data in all_results.items():
        print(f"{case_name}:")
        for np_name, np_data in case_data['a2dmrg'].items():
            total_tests += 1
            if np_data.get('success'):
                if np_data['machine_precision']:
                    print(f"  ✓✓ A2DMRG {np_name}: ΔE = {np_data['delta_E']:.3e} (MACHINE PRECISION)")
                    machine_precision_count += 1
                elif np_data['accepted']:
                    print(f"  ✓  A2DMRG {np_name}: ΔE = {np_data['delta_E']:.3e} (ACCEPTED)")
                    accepted_count += 1
                else:
                    print(f"  ✗  A2DMRG {np_name}: ΔE = {np_data['delta_E']:.3e} (FAILED)")
                    failed_count += 1
            else:
                print(f"  ✗  A2DMRG {np_name}: FAILED")
                failed_count += 1

    print(f"\n{'='*80}")
    print(f"RESULTS:")
    print(f"  Machine precision (ΔE < 1e-12): {machine_precision_count}/{total_tests}")
    print(f"  Accepted (ΔE < 5e-10):          {accepted_count}/{total_tests}")
    print(f"  Failed:                         {failed_count}/{total_tests}")
    print(f"{'='*80}\n")

    # Save results
    output_file = repo_root / 'a2dmrg_small_cases_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to: {output_file}\n")

    return 0 if failed_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
