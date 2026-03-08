#!/usr/bin/env python3
"""
Debug A2DMRG on Josephson L20 with verbose output.
Check convergence behavior and see if tolerance/max_sweeps is the issue.
"""

import sys
import os
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['PATH'] = '/usr/lib64/openmpi/bin:' + os.environ.get('PATH', '')
os.environ['LD_LIBRARY_PATH'] = '/usr/lib64/openmpi/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['OMPI_PRTERUN'] = '/usr/lib64/openmpi/bin/prterun'

repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

from benchmark_data_loader import load_benchmark_case, convert_tensors_to_quimb_mpo
import json
import subprocess

# Load Josephson case
data = load_benchmark_case('josephson', 'L20_D50_nmax2')
manifest = data['manifest']
E_golden = data['golden_results']['quimb_dmrg2']['energy']

print("="*80)
print("A2DMRG JOSEPHSON L20 CONVERGENCE DEBUG")
print("="*80)
print(f"Golden Reference: {E_golden:.15f}")
print()

# Test different configurations
configs = [
    {'tol': 1e-11, 'max_sweeps': 40, 'warmup': 5, 'name': 'Current (tol=1e-11, 40 sweeps)'},
    {'tol': 1e-12, 'max_sweeps': 40, 'warmup': 5, 'name': 'Tighter tol (1e-12, 40 sweeps)'},
    {'tol': 1e-11, 'max_sweeps': 80, 'warmup': 5, 'name': 'More sweeps (tol=1e-11, 80 sweeps)'},
    {'tol': 1e-12, 'max_sweeps': 80, 'warmup': 5, 'name': 'Both (tol=1e-12, 80 sweeps)'},
]

for config in configs:
    print(f"\n{'='*80}")
    print(f"CONFIG: {config['name']}")
    print(f"{'='*80}")

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

data = load_benchmark_case('josephson', 'L20_D50_nmax2')
mpo = convert_tensors_to_quimb_mpo(data['mpo_tensors'])
manifest = data['manifest']

import time
t0 = time.time()
energy, mps = a2dmrg_main(
    L=manifest['L'],
    mpo=mpo,
    max_sweeps={config['max_sweeps']},
    bond_dim=manifest['bond_dim'],
    tol={config['tol']},
    dtype=np.complex128,
    comm=comm,
    warmup_sweeps={config['warmup']},
    verbose=True  # VERBOSE OUTPUT
)
t1 = time.time()

if rank == 0:
    import json
    print("\\n" + "="*80)
    print(json.dumps({{
        'energy': float(np.real(energy)),
        'time': t1 - t0,
        'tol': {config['tol']},
        'max_sweeps': {config['max_sweeps']}
    }}, default=str))
'''

    script_path = f'/tmp/debug_a2dmrg_{config["tol"]}_{config["max_sweeps"]}.py'
    with open(script_path, 'w') as f:
        f.write(script)

    venv_python = repo_root / 'a2dmrg' / 'venv' / 'bin' / 'python'

    cmd = [
        'mpirun',
        '-np', '2',
        '--oversubscribe',
        '--mca', 'btl', 'tcp,self',
        '--mca', 'btl_tcp_if_include', 'lo',
        str(venv_python),
        script_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        # Show all output for debugging
        if result.stdout:
            print(result.stdout)

        if result.returncode != 0:
            print(f"\n✗ FAILED")
            if result.stderr:
                print("STDERR:", result.stderr[-1000:])
            continue

        # Parse final JSON
        for line in result.stdout.strip().split('\n'):
            if line.startswith('{') and 'energy' in line:
                data = json.loads(line)
                E = data['energy']
                dE = E - E_golden
                t = data['time']

                print(f"\nRESULT:")
                print(f"  Energy: {E:.15f}")
                print(f"  ΔE: {dE:.3e}")
                print(f"  Time: {t:.2f}s")

                if abs(dE) < 1e-12:
                    print(f"  Status: ✓✓ MACHINE PRECISION")
                elif abs(dE) < 5e-10:
                    print(f"  Status: ✓ ACCEPTED")
                else:
                    print(f"  Status: ✗ FAILED (above threshold)")

    except subprocess.TimeoutExpired:
        print(f"\n✗ TIMEOUT (600s)")
    except Exception as e:
        print(f"\n✗ ERROR: {e}")

print(f"\n{'='*80}")
print("DEBUG COMPLETE")
print(f"{'='*80}")
