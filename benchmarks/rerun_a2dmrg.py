#!/usr/bin/env python3
"""
Re-run A2DMRG Josephson benchmark for specific np values.

Usage: python3 benchmarks/rerun_a2dmrg.py [np1 np2 ...]
       python3 benchmarks/rerun_a2dmrg.py 1 2      # re-run np=1 and np=2
       python3 benchmarks/rerun_a2dmrg.py 1        # re-run only np=1

Appends results to josephson_full_results.json.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'a2dmrg'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pdmrg'))

import quimb.tensor as qtn

L = 20
E_C = 1.0
E_J = 2.0
N_MAX = 2
PHI_EXT = np.pi / 4
BOND_DIM = 50
MAX_SWEEPS = 200
TOL = 1e-10
CUTOFF = 1e-14

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_PATH = os.path.join(os.path.dirname(__file__), 'josephson_full_results.json')


def run_a2dmrg(np_count):
    print(f"Running A2DMRG np={np_count}...", flush=True)

    script = f'''
import sys
sys.path.insert(0, '{BASE_DIR}/a2dmrg')
import numpy as np
import quimb.tensor as qtn
from mpi4py import MPI
import time

n_max = {N_MAX}
d = 2 * n_max + 1
S = (d - 1) / 2

charges = np.arange(-n_max, n_max + 1, dtype='complex128')
n_op = np.diag(charges)
exp_iphi = np.zeros((d, d), dtype='complex128')
for i in range(d - 1):
    exp_iphi[i + 1, i] = 1.0
exp_miphi = exp_iphi.conj().T

builder = qtn.SpinHam1D(S=S)
flux_phase = np.exp(1j * {PHI_EXT})
builder.add_term(-{E_J} / 2 * flux_phase, exp_iphi, exp_miphi)
builder.add_term(-{E_J} / 2 * np.conj(flux_phase), exp_miphi, exp_iphi)
n_squared = n_op @ n_op
builder.add_term({E_C}, n_squared)
mpo = builder.build_mpo({L})

from a2dmrg.dmrg import a2dmrg_main
t0 = time.time()
energy, mps = a2dmrg_main(
    L={L},
    mpo=mpo,
    max_sweeps={MAX_SWEEPS},
    bond_dim={BOND_DIM},
    tol={TOL},
    warmup_sweeps=5,
    dtype=np.complex128,
    comm=MPI.COMM_WORLD,
    verbose=False
)
t1 = time.time()

if MPI.COMM_WORLD.Get_rank() == 0:
    print(f"RESULT:energy={{energy.real:.15f}},time={{t1-t0:.2f}}")
'''

    script_path = f'/tmp/a2dmrg_jj_np{np_count}_rerun.py'
    with open(script_path, 'w') as f:
        f.write(script)

    env = os.environ.copy()
    env['PATH'] = '/usr/lib64/openmpi/bin:' + env.get('PATH', '')
    env['LD_LIBRARY_PATH'] = '/usr/lib64/openmpi/lib:' + env.get('LD_LIBRARY_PATH', '')

    result = subprocess.run(
        ['mpirun', '-np', str(np_count), '--oversubscribe',
         '--mca', 'btl', 'tcp,self', '--mca', 'btl_tcp_if_include', 'lo',
         sys.executable, script_path],
        capture_output=True, text=True, env=env
    )

    for line in result.stdout.split('\n'):
        if line.startswith('RESULT:'):
            parts = line[7:].split(',')
            energy = float(parts[0].split('=')[1])
            time_s = float(parts[1].split('=')[1])
            return {'energy': energy, 'time': time_s}

    print(f"A2DMRG np={np_count} FAILED:")
    print(result.stderr if result.stderr else "No stderr")
    return None


def main():
    np_list = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [1, 2]

    # Load existing results
    results = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            results = json.load(f)

    ref_energy = results.get('quimb2', {}).get('energy')
    if ref_energy is None:
        print("ERROR: No DMRG2 reference energy found in results file.")
        sys.exit(1)

    print(f"Re-running A2DMRG for np={np_list}")
    print(f"Reference energy (DMRG2): {ref_energy:.12f}")
    print(f"Started: {datetime.now().isoformat()}")
    print()

    if 'a2dmrg' not in results:
        results['a2dmrg'] = {}

    for np_count in np_list:
        res = run_a2dmrg(np_count)
        if res:
            delta = res['energy'] - ref_energy
            results['a2dmrg'][f'np{np_count}'] = res
            results['a2dmrg'][f'np{np_count}']['delta_e'] = delta
            print(f"  np={np_count}: E = {res['energy']:.12f}, ΔE = {delta:.2e}, time = {res['time']:.2f}s")
        else:
            print(f"  np={np_count}: FAILED")

    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {RESULTS_PATH}")
    print(f"Finished: {datetime.now().isoformat()}")


if __name__ == '__main__':
    main()
