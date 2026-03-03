#!/usr/bin/env python3
"""
PDMRG vs PDMRG2 Scaling Benchmark
===================================
Compares the original PDMRG (Lanczos + QR + exact SVD) against PDMRG2
(Block-Davidson + Newton-Schulz + rSVD/Cholesky-QR2) across np = 1, 2, 4.

Metrics reported
----------------
- Ground-state energy accuracy vs quimb DMRG2 reference
- Wall-clock time (total, warmup, sweep)
- Speedup over serial PDMRG (np=1) for each variant
- Pass/fail: |ΔE| < tol

Critical-exception check
-------------------------
PDMRG2 boundary merges continue to use accurate_svd (exact recursive SVD)
for the V = Λ⁻¹ inversion step.  This is enforced in merge.py and is NOT
changed by the GEMM-optimization patch.
"""

import json
import os
import subprocess
import sys
import time

import numpy as np

# ── Configuration ──────────────────────────────────────────────────────────────
L          = 16          # chain length  (small enough for fast CI, large enough to exercise parallellism)
BOND_DIM   = 30          # bond dimension
MAX_SWEEPS = 20          # max PDMRG sweeps
TOL        = 1e-8        # solver convergence tolerance
PASS_TOL   = 1e-6        # energy accuracy pass/fail threshold vs quimb DMRG2
NPS        = [1, 2, 4]  # process counts to benchmark

BASE   = '/home/captain/clawd/work/dmrg-implementations'
PDMRG_PY  = f'{BASE}/pdmrg/venv/bin/python'
PDMRG2_PY = f'{BASE}/pdmrg2/venv/bin/python'

MPI_OPTS = [
    '--oversubscribe',
    '--mca', 'btl', 'tcp,self',
    '--mca', 'btl_tcp_if_include', 'lo',
]

ENV_PATCH = {
    'PATH': '/usr/lib64/openmpi/bin:' + os.environ.get('PATH', ''),
    'LD_LIBRARY_PATH': '/usr/lib64/openmpi/lib:' + os.environ.get('LD_LIBRARY_PATH', ''),
}


# ── Quimb reference ────────────────────────────────────────────────────────────

def run_quimb_reference():
    """Run quimb DMRG2 as ground-truth reference."""
    import quimb.tensor as qtn
    t0 = time.time()
    mpo = qtn.MPO_ham_heis(L=L, j=1.0, bz=0.0, cyclic=False)
    dmrg = qtn.DMRG2(mpo, bond_dims=BOND_DIM, cutoffs=1e-14)
    dmrg.solve(max_sweeps=30, tol=1e-12, verbosity=0)
    elapsed = time.time() - t0
    return {'energy': float(np.real(dmrg.energy)), 'time': elapsed}


# ── Generic MPI runner ─────────────────────────────────────────────────────────

def _run_mpi(python_bin, src_path, script, np_count, timeout=600):
    """Write *script* to a temp file and run it with mpirun -np *np_count*."""
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py',
                                     delete=False, prefix='bench_') as f:
        f.write(script)
        script_path = f.name

    env = os.environ.copy()
    env.update(ENV_PATCH)

    cmd = ['mpirun', '-np', str(np_count)] + MPI_OPTS + [python_bin, script_path]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout, env=env)
        os.unlink(script_path)
        if res.returncode != 0:
            return {'error': (res.stderr or res.stdout)[:300]}
        lines = [l for l in res.stdout.strip().split('\n') if l.startswith('{')]
        if not lines:
            return {'error': f'No JSON output. stdout={res.stdout[:200]}'}
        return json.loads(lines[-1])
    except subprocess.TimeoutExpired:
        os.unlink(script_path)
        return {'error': f'Timeout ({timeout}s)'}
    except Exception as exc:
        os.unlink(script_path)
        return {'error': str(exc)}


# ── PDMRG runner ───────────────────────────────────────────────────────────────

def run_pdmrg(np_count):
    script = f'''
import sys, time, json
import numpy as np
import quimb.tensor as qtn
sys.path.insert(0, '{BASE}/pdmrg')

from mpi4py import MPI
from pdmrg.dmrg import pdmrg_main
from pdmrg.mps.canonical import get_mpo_tensor_data

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

mpo = qtn.MPO_ham_heis(L={L}, j=1.0, bz=0.0, cyclic=False)

t0 = time.time()
energy, pmps = pdmrg_main(
    L={L}, mpo=mpo,
    max_sweeps={MAX_SWEEPS},
    bond_dim={BOND_DIM},
    bond_dim_warmup={BOND_DIM},
    n_warmup_sweeps=5,
    tol={TOL},
    comm=comm,
    verbose=False,
)
elapsed = time.time() - t0

if rank == 0:
    print(json.dumps({{"energy": float(energy), "time": elapsed}}))
'''
    return _run_mpi(PDMRG_PY, f'{BASE}/pdmrg', script, np_count)


# ── PDMRG2 runner ──────────────────────────────────────────────────────────────

def run_pdmrg2(np_count):
    script = f'''
import sys, time, json
import numpy as np
import quimb.tensor as qtn
sys.path.insert(0, '{BASE}/pdmrg2')

from mpi4py import MPI
from pdmrg.dmrg import pdmrg_main
from pdmrg.mps.canonical import get_mpo_tensor_data

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

mpo = qtn.MPO_ham_heis(L={L}, j=1.0, bz=0.0, cyclic=False)

t0 = time.time()
energy, pmps = pdmrg_main(
    L={L}, mpo=mpo,
    max_sweeps={MAX_SWEEPS},
    bond_dim={BOND_DIM},
    bond_dim_warmup={BOND_DIM},
    n_warmup_sweeps=5,
    tol={TOL},
    comm=comm,
    verbose=False,
)
elapsed = time.time() - t0

if rank == 0:
    print(json.dumps({{"energy": float(energy), "time": elapsed}}))
'''
    return _run_mpi(PDMRG2_PY, f'{BASE}/pdmrg2', script, np_count)


# ── Reporting helpers ──────────────────────────────────────────────────────────

def fmt_dE(dE):
    return f'{dE:+.3e}' if dE is not None else '   N/A   '

def status(dE):
    if dE is None:
        return '?'
    return '✓' if abs(dE) < PASS_TOL else '✗'

def speedup(t_ref, t):
    if t_ref is None or t is None or t <= 0:
        return None
    return t_ref / t


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print('=' * 72)
    print('PDMRG vs PDMRG2 — Scaling Benchmark')
    print('=' * 72)
    print(f'L={L}  bond_dim={BOND_DIM}  max_sweeps={MAX_SWEEPS}  '
          f'tol={TOL}  pass_tol={PASS_TOL}')
    print(f'np list: {NPS}')
    print()

    # ── Quimb reference ──────────────────────────────────────────────────────
    print('Running quimb DMRG2 reference...', end=' ', flush=True)
    ref = run_quimb_reference()
    E_ref = ref['energy']
    print(f'E = {E_ref:.12f}  ({ref["time"]:.2f}s)')
    print()

    results = {}

    # ── PDMRG (original) ─────────────────────────────────────────────────────
    print('PDMRG (original: Lanczos + QR + exact SVD)')
    print('-' * 52)
    t1_pdmrg = None
    for np_count in NPS:
        print(f'  np={np_count}...', end=' ', flush=True)
        r = run_pdmrg(np_count)
        key = f'pdmrg_np{np_count}'
        results[key] = r
        if 'error' in r:
            print(f'ERROR: {r["error"][:60]}')
        else:
            dE = r['energy'] - E_ref
            if np_count == 1:
                t1_pdmrg = r['time']
            sp = speedup(t1_pdmrg, r['time'])
            sp_str = f'{sp:.2f}×' if sp is not None else ' N/A '
            print(f'E={r["energy"]:.12f}  ΔE={fmt_dE(dE)}  '
                  f't={r["time"]:.2f}s  speedup={sp_str}  {status(dE)}')
    print()

    # ── PDMRG2 (upgraded) ────────────────────────────────────────────────────
    print('PDMRG2 (upgraded: Block-Davidson + Newton-Schulz + rSVD/Chol-QR2)')
    print('  [boundary merges still use exact recursive SVD — critical exception]')
    print('-' * 52)
    t1_pdmrg2 = None
    for np_count in NPS:
        print(f'  np={np_count}...', end=' ', flush=True)
        r = run_pdmrg2(np_count)
        key = f'pdmrg2_np{np_count}'
        results[key] = r
        if 'error' in r:
            print(f'ERROR: {r["error"][:60]}')
        else:
            dE = r['energy'] - E_ref
            if np_count == 1:
                t1_pdmrg2 = r['time']
            sp = speedup(t1_pdmrg2, r['time'])
            sp_str = f'{sp:.2f}×' if sp is not None else ' N/A '
            print(f'E={r["energy"]:.12f}  ΔE={fmt_dE(dE)}  '
                  f't={r["time"]:.2f}s  speedup={sp_str}  {status(dE)}')
    print()

    # ── Comparison table ─────────────────────────────────────────────────────
    print('=' * 72)
    print('COMPARISON TABLE  (vs quimb DMRG2 reference)')
    print('=' * 72)
    hdr = f'{"Method":<22}{"np":>4}{"Energy":>18}{"ΔE":>12}{"Time(s)":>10}{"Speedup":>10}{"Pass"}'
    print(hdr)
    print('-' * 72)

    for variant, label in [('pdmrg', 'PDMRG'), ('pdmrg2', 'PDMRG2')]:
        t1 = results.get(f'{variant}_np1', {}).get('time')
        for np_count in NPS:
            key = f'{variant}_np{np_count}'
            r = results.get(key, {})
            if 'error' in r:
                print(f'{label:<22}{np_count:>4}{"ERROR":>18}{"—":>12}{"—":>10}{"—":>10}  ✗')
            else:
                dE = r.get('energy', None)
                dE_str = fmt_dE(dE - E_ref if dE is not None else None)
                e_str = f'{r["energy"]:.12f}' if 'energy' in r else 'N/A'
                t = r.get('time')
                t_str = f'{t:.2f}' if t is not None else 'N/A'
                sp = speedup(t1, t)
                sp_str = f'{sp:.2f}×' if sp is not None else 'N/A'
                ok = status(dE - E_ref if dE is not None else None)
                print(f'{label:<22}{np_count:>4}{e_str:>18}{dE_str:>12}{t_str:>10}{sp_str:>10}  {ok}')
        print()

    # ── PDMRG2 vs PDMRG single-process speedup ───────────────────────────────
    print('PDMRG2 vs PDMRG wall-clock comparison (np=1)')
    t_p1  = results.get('pdmrg_np1',  {}).get('time')
    t_p21 = results.get('pdmrg2_np1', {}).get('time')
    if t_p1 and t_p21:
        ratio = t_p1 / t_p21
        direction = 'faster' if ratio > 1 else 'slower'
        print(f'  PDMRG  np=1: {t_p1:.2f}s')
        print(f'  PDMRG2 np=1: {t_p21:.2f}s  →  PDMRG2 is {abs(ratio):.2f}× {direction}')
    print()

    # ── Save results ─────────────────────────────────────────────────────────
    out_path = f'{BASE}/benchmarks/pdmrg_vs_pdmrg2_results.json'
    payload = {
        'config': {'L': L, 'bond_dim': BOND_DIM, 'max_sweeps': MAX_SWEEPS,
                   'tol': TOL, 'pass_tol': PASS_TOL, 'nps': NPS},
        'reference': {'quimb_DMRG2': ref},
        'results': results,
    }
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'Results saved → {out_path}')


if __name__ == '__main__':
    main()
