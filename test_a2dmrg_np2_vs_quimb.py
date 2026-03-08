#!/usr/bin/env python3
"""
Test A2DMRG np=2 vs serial quimb DMRG2 on small benchmark cases.

This runs the FULL A2DMRG algorithm (not just warmup):
- Parallel microsteps across 2 processes
- Coarse-space minimization
- Additive corrections

Compares against serial quimb DMRG2 reference.
"""

import sys
import os
from pathlib import Path
import time
import json
import subprocess
import numpy as np

# Force single-threaded BLAS
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# Set up OpenMPI environment
os.environ['PATH'] = '/usr/lib64/openmpi/bin:' + os.environ.get('PATH', '')
os.environ['LD_LIBRARY_PATH'] = '/usr/lib64/openmpi/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['OMPI_PRTERUN'] = '/usr/lib64/openmpi/bin/prterun'

repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

from benchmark_data_loader import load_benchmark_case, convert_tensors_to_quimb_mpo
import quimb.tensor as qtn

# Validation thresholds
GOLDEN_TOLERANCE = 1e-11
MACHINE_PRECISION_THRESHOLD = 1e-12
ACCEPTANCE_THRESHOLD = 5e-10

# MPI executable (now in PATH)
MPIRUN = 'mpirun'


def run_quimb_dmrg2(model, case):
    """Run serial quimb DMRG2 reference."""
    print(f"  Running quimb DMRG2 (serial)...")
    data = load_benchmark_case(model, case)
    mpo = convert_tensors_to_quimb_mpo(data['mpo_tensors'])
    manifest = data['manifest']

    t0 = time.time()
    dmrg = qtn.DMRG2(mpo, bond_dims=manifest['bond_dim'], cutoffs=1e-14)
    dmrg.solve(max_sweeps=50, tol=GOLDEN_TOLERANCE, verbosity=0)
    t1 = time.time()

    return {
        'energy': float(np.real(dmrg.energy)),
        'time': t1 - t0,
        'sweeps': len(dmrg.energies) if hasattr(dmrg, 'energies') else 50
    }


def run_a2dmrg_np2(model, case):
    """Run A2DMRG with np=2 (full parallel algorithm)."""
    print(f"  Running A2DMRG np=2 (full algorithm)...")

    data = load_benchmark_case(model, case)
    manifest = data['manifest']

    # Create MPI script
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
# Use more warmup sweeps for complex systems (Josephson needs ~20 sweeps)
warmup = 20 if manifest['dtype'] == 'complex128' else 5
energy, mps = a2dmrg_main(
    L=manifest['L'],
    mpo=mpo,
    max_sweeps=40,
    bond_dim=manifest['bond_dim'],
    tol={GOLDEN_TOLERANCE},
    dtype=dtype,
    comm=comm,
    warmup_sweeps=warmup,
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

    script_path = f'/tmp/a2dmrg_np2_{model}_{case}.py'
    with open(script_path, 'w') as f:
        f.write(script)

    venv_python = repo_root / 'a2dmrg' / 'venv' / 'bin' / 'python'

    cmd = [
        MPIRUN,
        '-np', '2',
        '--oversubscribe',
        '--mca', 'btl', 'tcp,self',
        '--mca', 'btl_tcp_if_include', 'lo',
        str(venv_python),
        script_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            return {
                'success': False,
                'error': result.stderr[-500:] if result.stderr else result.stdout[-500:]
            }

        # Parse JSON output from rank 0
        for line in result.stdout.strip().split('\n'):
            if line.startswith('{'):
                data = json.loads(line)
                return {
                    'success': True,
                    'energy': data['energy'],
                    'time': data['time']
                }

        return {
            'success': False,
            'error': 'No JSON output found'
        }

    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'error': 'Timeout (600s)'
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


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

    # Run quimb DMRG2 (serial)
    print()
    quimb_result = run_quimb_dmrg2(model, case)
    E_quimb = quimb_result['energy']
    dE_quimb = E_quimb - E_golden
    print(f"    E = {E_quimb:.15f}")
    print(f"    ΔE = {dE_quimb:.3e}")
    print(f"    t = {quimb_result['time']:.3f}s")
    print(f"    sweeps = {quimb_result['sweeps']}")

    # Run A2DMRG np=2
    print()
    a2dmrg_result = run_a2dmrg_np2(model, case)

    if not a2dmrg_result.get('success'):
        print(f"    ✗ FAILED: {a2dmrg_result.get('error', 'Unknown')[:200]}")
        return {
            'golden': E_golden,
            'quimb': quimb_result,
            'a2dmrg_np2': a2dmrg_result
        }

    E_a2dmrg = a2dmrg_result['energy']
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
    print(f"    t = {a2dmrg_result['time']:.3f}s")

    if quimb_result['time'] > 0:
        speedup = quimb_result['time'] / a2dmrg_result['time']
        speedup_str = f"{speedup:.2f}×" if speedup >= 1 else f"{1/speedup:.2f}× slower"
        print(f"    Speedup vs quimb: {speedup_str}")

    a2dmrg_result['delta_E'] = dE
    a2dmrg_result['machine_precision'] = machine_prec
    a2dmrg_result['accepted'] = accepted

    return {
        'golden': E_golden,
        'quimb': quimb_result,
        'a2dmrg_np2': a2dmrg_result
    }


def main():
    print("="*80)
    print("A2DMRG np=2 vs QUIMB DMRG2 COMPARISON")
    print("="*80)
    print("Testing FULL A2DMRG algorithm (parallel microsteps + coarse-space)")
    print()
    print(f"Environment:")
    print(f"  OMP_NUM_THREADS = {os.environ.get('OMP_NUM_THREADS')}")
    print(f"  OPENBLAS_NUM_THREADS = {os.environ.get('OPENBLAS_NUM_THREADS')}")
    print(f"  MPI executable = {MPIRUN}")
    print(f"  A2DMRG processes = 2")
    print()

    results = {}

    # Test Heisenberg L12
    print("\n" + "="*80)
    print("HEISENBERG CHAIN L=12")
    print("="*80)
    heisenberg_result = test_case('heisenberg', 'L12_D20')
    results['heisenberg/L12_D20'] = heisenberg_result

    # Test Josephson L20
    print("\n" + "="*80)
    print("JOSEPHSON JUNCTION L=20")
    print("="*80)
    josephson_result = test_case('josephson', 'L20_D50_nmax2')
    results['josephson/L20_D50_nmax2'] = josephson_result

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY: A2DMRG np=2 vs QUIMB DMRG2")
    print(f"{'='*80}\n")

    all_passed = True
    for case_name, case_data in results.items():
        a2dmrg = case_data['a2dmrg_np2']
        quimb = case_data['quimb']

        if not a2dmrg.get('success'):
            print(f"✗ {case_name}: A2DMRG FAILED - {a2dmrg.get('error', 'Unknown')[:100]}")
            all_passed = False
            continue

        print(f"\n{case_name}:")
        print(f"  quimb DMRG2:   E = {quimb['energy']:.15f}, t = {quimb['time']:.2f}s")
        print(f"  A2DMRG np=2:   E = {a2dmrg['energy']:.15f}, t = {a2dmrg['time']:.2f}s")
        print(f"  ΔE (A2DMRG):   {a2dmrg['delta_E']:.3e}")

        speedup = quimb['time'] / a2dmrg['time']
        print(f"  Speedup:       {speedup:.2f}×")

        if a2dmrg['machine_precision']:
            print(f"  Status:        ✓✓ MACHINE PRECISION")
        elif a2dmrg['accepted']:
            print(f"  Status:        ✓  ACCEPTED")
        else:
            print(f"  Status:        ✗  FAILED (error too large)")
            all_passed = False

    print(f"\n{'='*80}")
    if all_passed:
        print("✓✓ ALL TESTS PASSED")
        print("="*80)
        print("\nA2DMRG np=2 converges correctly on all test cases!")
    else:
        print("✗ SOME TESTS FAILED")
        print("="*80)

    # Save results
    output_file = repo_root / 'a2dmrg_np2_vs_quimb_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")

    print()
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
