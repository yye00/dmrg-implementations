#!/usr/bin/env python3
"""
Josephson Junction Array Benchmark — Complex128 Test

Short benchmark (L=10) verifying that all three DMRG implementations
handle complex128 arithmetic correctly on a 1D Josephson junction array
with non-zero external flux (PHI_EXT = π/4).

Physical model:
    H = E_C Σ_i (n_i - n_g)² - E_J/2 Σ_<ij> [e^{i(φ_i-φ_j+Φ_ext)} + h.c.]

Charge-basis representation:
    - Local dim d = 2*n_max + 1 = 5  (charges: -2, -1, 0, 1, 2)
    - PHI_EXT ≠ 0 breaks time-reversal → complex128 mandatory
    - E_J/E_C = 2  (intermediate coupling, significant entanglement)

Tests:
    - quimb DMRG2 (reference, complex128)
    - PDMRG  : np = 1, 2, 4, 8
    - PDMRG2 : np = 1, 2, 4, 8
    - A2DMRG : np = 1, 2, 4, 8
"""

import json
import time
import subprocess
import sys
import os
import numpy as np

# ── Configuration ──────────────────────────────────────────────────────────────
L          = 10       # Number of junctions (short benchmark)
BOND_DIM   = 20       # Bond dimension
MAX_SWEEPS = 30
TOL        = 1e-10
CUTOFF     = 1e-14
PASS_TOL   = 1e-5     # Pass/fail threshold vs quimb DMRG2
# NOTE: bond_dim=20 with d=5 (Josephson) has truncation error ~1e-7, so
# PASS_TOL is set looser than TOL.  This benchmark tests complex128
# *functionality*, not sub-1e-10 variational accuracy.

# Josephson parameters
E_C     = 1.0        # Charging energy (sets energy scale)
E_J     = 2.0        # Josephson coupling  (E_J/E_C = 2)
N_G     = 0.0        # Gate charge offset
N_MAX   = 2          # Max charge per site: n ∈ {-2,-1,0,1,2}  →  d=5
PHI_EXT = np.pi / 4  # External flux (requires complex128)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── Helper: build the Josephson MPO (in-process, for quimb reference) ─────────

def _build_mpo():
    import quimb.tensor as qtn
    d         = 2 * N_MAX + 1
    dtype     = 'complex128'
    charges   = np.arange(-N_MAX, N_MAX + 1, dtype=dtype)
    n_op      = np.diag(charges)
    exp_iphi  = np.zeros((d, d), dtype=dtype)
    for i in range(d - 1):
        exp_iphi[i + 1, i] = 1.0
    exp_miphi = exp_iphi.conj().T
    n_squared = n_op @ n_op
    S         = (d - 1) / 2
    builder   = qtn.SpinHam1D(S=S)
    flux_phase = np.exp(1j * PHI_EXT)
    builder.add_term(-E_J / 2 * flux_phase,          exp_iphi,  exp_miphi)
    builder.add_term(-E_J / 2 * np.conj(flux_phase), exp_miphi, exp_iphi)
    builder.add_term(E_C, n_squared)
    return builder.build_mpo(L)


# ── Argument parsing ───────────────────────────────────────────────────────────

def _parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description='Josephson junction array benchmark — complex128 test')
    p.add_argument('--L',           type=int,   default=L)
    p.add_argument('--bond-dim',    type=int,   default=BOND_DIM)
    p.add_argument('--max-sweeps',  type=int,   default=MAX_SWEEPS)
    p.add_argument('--tol',         type=float, default=TOL)
    p.add_argument('--cutoff',      type=float, default=CUTOFF)
    p.add_argument('--pass-tol',    type=float, default=PASS_TOL)
    p.add_argument('--out',         type=str,   default=None)
    p.add_argument('--nps',         type=str,   default='1,2,4,8',
                   help='Comma-separated list of MPI process counts')
    return p.parse_args()


# ── Reference runner ───────────────────────────────────────────────────────────

def run_quimb_reference():
    """Run quimb DMRG2 as complex128 reference."""
    import quimb.tensor as qtn
    mpo    = _build_mpo()
    t0     = time.time()
    dmrg2  = qtn.DMRG2(mpo, bond_dims=BOND_DIM, cutoffs=CUTOFF)
    dmrg2.solve(max_sweeps=MAX_SWEEPS, tol=TOL, verbosity=0)
    t1     = time.time()
    return {
        'quimb_DMRG2': {
            'energy': float(np.real(dmrg2.energy)),
            'time':   t1 - t0,
        }
    }


# ── Inline MPO build code (embedded in subprocess scripts) ────────────────────

def _mpo_build_code():
    """Return Python source that rebuilds the Josephson MPO inside a subprocess."""
    return f"""
import numpy as np
import quimb.tensor as qtn

_n_max      = {N_MAX}
_d          = 2 * _n_max + 1
_dtype      = 'complex128'
_charges    = np.arange(-_n_max, _n_max + 1, dtype=_dtype)
_n_op       = np.diag(_charges)
_exp_iphi   = np.zeros((_d, _d), dtype=_dtype)
for _i in range(_d - 1):
    _exp_iphi[_i + 1, _i] = 1.0
_exp_miphi  = _exp_iphi.conj().T
_n_squared  = _n_op @ _n_op
_S          = (_d - 1) / 2
_builder    = qtn.SpinHam1D(S=_S)
_flux_phase = np.exp(1j * {PHI_EXT})
_builder.add_term(-{E_J} / 2 * _flux_phase,           _exp_iphi,  _exp_miphi)
_builder.add_term(-{E_J} / 2 * np.conj(_flux_phase),  _exp_miphi, _exp_iphi)
_builder.add_term({E_C}, _n_squared)
mpo = _builder.build_mpo({L})
"""


# ── MPI subprocess runner ─────────────────────────────────────────────────────

def _run_mpi(script, script_path, python_exe, np_count, timeout=300):
    with open(script_path, 'w') as f:
        f.write(script)
    env = os.environ.copy()
    env['PATH']            = '/usr/lib64/openmpi/bin:' + env.get('PATH', '')
    env['LD_LIBRARY_PATH'] = '/usr/lib64/openmpi/lib:' + env.get('LD_LIBRARY_PATH', '')
    cmd = [
        'mpirun', '-np', str(np_count),
        '--oversubscribe', '--mca', 'btl', 'tcp,self',
        '--mca', 'btl_tcp_if_include', 'lo',
        python_exe, script_path,
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout, env=env)
        if res.returncode != 0:
            return {'error': res.stderr, 'energy': None, 'time': None}
        lines = [l for l in res.stdout.strip().split('\n') if l.startswith('{')]
        if lines:
            return json.loads(lines[-1])
        return {'error': 'No JSON output',
                'stdout': res.stdout, 'stderr': res.stderr}
    except subprocess.TimeoutExpired:
        return {'error': f'Timeout ({timeout}s)', 'energy': None, 'time': None}
    except Exception as e:
        return {'error': str(e), 'energy': None, 'time': None}


# ── PDMRG runner ──────────────────────────────────────────────────────────────

def run_pdmrg(np_count, ref_energy):
    """Run PDMRG with given number of processes (complex128)."""
    script = f"""
import sys, time, json, numpy as np
sys.path.insert(0, '{BASE_DIR}/pdmrg')
from mpi4py import MPI
from pdmrg.dmrg import pdmrg_main
{_mpo_build_code()}
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
t0 = time.time()
energy, pmps = pdmrg_main(
    L={L}, mpo=mpo,
    max_sweeps={MAX_SWEEPS}, bond_dim={BOND_DIM}, bond_dim_warmup={BOND_DIM},
    n_warmup_sweeps=5, tol={TOL}, dtype='complex128',
    comm=comm, verbose=False
)
t1 = time.time()
if rank == 0:
    print(json.dumps({{'energy': float(np.real(energy)), 'time': t1 - t0}}))
"""
    return _run_mpi(script, '/tmp/pdmrg_jj_bench.py',
                    f'{BASE_DIR}/pdmrg/venv/bin/python', np_count)


# ── PDMRG2 runner ─────────────────────────────────────────────────────────────

def run_pdmrg2(np_count, ref_energy):
    """Run PDMRG2 with given number of processes (complex128)."""
    script = f"""
import sys, time, json, numpy as np
sys.path.insert(0, '{BASE_DIR}/pdmrg2')
from mpi4py import MPI
from pdmrg.dmrg import pdmrg_main
{_mpo_build_code()}
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
t0 = time.time()
energy, pmps = pdmrg_main(
    L={L}, mpo=mpo,
    max_sweeps={MAX_SWEEPS}, bond_dim={BOND_DIM}, bond_dim_warmup={BOND_DIM},
    n_warmup_sweeps=5, tol={TOL}, dtype='complex128',
    comm=comm, verbose=False
)
t1 = time.time()
if rank == 0:
    print(json.dumps({{'energy': float(np.real(energy)), 'time': t1 - t0}}))
"""
    return _run_mpi(script, '/tmp/pdmrg2_jj_bench.py',
                    f'{BASE_DIR}/pdmrg2/venv/bin/python', np_count)


# ── A2DMRG runner ─────────────────────────────────────────────────────────────

def run_a2dmrg(np_count, ref_energy):
    """Run A2DMRG with given number of processes (complex128)."""
    script = f"""
import sys, time, json, numpy as np
sys.path.insert(0, '{BASE_DIR}/a2dmrg')
from mpi4py import MPI
from a2dmrg.dmrg import a2dmrg_main
{_mpo_build_code()}
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
t0 = time.time()
energy, mps = a2dmrg_main(
    L={L}, mpo=mpo,
    max_sweeps={MAX_SWEEPS}, bond_dim={BOND_DIM}, tol={TOL},
    warmup_sweeps=5, dtype=np.complex128,
    comm=comm, verbose=False
)
t1 = time.time()
if rank == 0:
    print(json.dumps({{'energy': float(np.real(energy)), 'time': t1 - t0}}))
"""
    return _run_mpi(script, '/tmp/a2dmrg_jj_bench.py',
                    f'{BASE_DIR}/a2dmrg/venv/bin/python', np_count, timeout=90)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    global L, BOND_DIM, MAX_SWEEPS, TOL, CUTOFF, PASS_TOL

    args       = _parse_args()
    L          = args.L
    BOND_DIM   = args.bond_dim
    MAX_SWEEPS = args.max_sweeps
    TOL        = args.tol
    CUTOFF     = args.cutoff
    PASS_TOL   = args.pass_tol
    nps        = [int(x.strip()) for x in args.nps.split(',') if x.strip()]

    print('=' * 70)
    print('JOSEPHSON JUNCTION ARRAY BENCHMARK — complex128 test')
    print('=' * 70)
    print(f'  H = E_C Σ(n-n_g)² - E_J/2 Σ [e^{{i(φᵢ-φⱼ+Φ)}} + h.c.]')
    print(f'  L={L}, d={2*N_MAX+1} (n_max={N_MAX}), '
          f'E_J/E_C={E_J/E_C:.1f}, Φ_ext=π/{int(round(np.pi/PHI_EXT))}')
    print(f'  bond_dim={BOND_DIM}, max_sweeps={MAX_SWEEPS}, '
          f'tol={TOL}, pass_tol={PASS_TOL}')
    print(f'  np list: {nps}')
    print()

    all_results = {}
    issues      = []

    # ── Reference ──────────────────────────────────────────────────────────────
    print('Running quimb DMRG2 reference (complex128)...')
    ref_results = run_quimb_reference()
    all_results.update(ref_results)
    E_ref = ref_results['quimb_DMRG2']['energy']
    print(f'  quimb DMRG2: E = {E_ref:.15f}  '
          f'({ref_results["quimb_DMRG2"]["time"]:.2f}s)  [REFERENCE]')
    print()

    # ── PDMRG ──────────────────────────────────────────────────────────────────
    print('Running PDMRG tests (complex128)...')
    for np_count in nps:
        print(f'  PDMRG np={np_count}...', end=' ', flush=True)
        result = run_pdmrg(np_count, E_ref)
        all_results[f'PDMRG_np{np_count}'] = result
        if 'error' in result:
            print(f'ERROR: {str(result["error"])[:60]}')
            issues.append(f'PDMRG np={np_count}: {result["error"]}')
        else:
            dE = result['energy'] - E_ref
            st = '✓ PASS' if abs(dE) < PASS_TOL else '✗ FAIL'
            print(f'E = {result["energy"]:.15f}, ΔE = {dE:.2e}, '
                  f't = {result["time"]:.2f}s  {st}')
            if abs(dE) >= PASS_TOL:
                issues.append(f'PDMRG np={np_count}: ΔE = {dE:.2e}')
    print()

    # ── PDMRG2 ─────────────────────────────────────────────────────────────────
    print('Running PDMRG2 tests (complex128)...')
    for np_count in nps:
        print(f'  PDMRG2 np={np_count}...', end=' ', flush=True)
        result = run_pdmrg2(np_count, E_ref)
        all_results[f'PDMRG2_np{np_count}'] = result
        if 'error' in result:
            print(f'ERROR: {str(result["error"])[:60]}')
            issues.append(f'PDMRG2 np={np_count}: {result["error"]}')
        else:
            dE = result['energy'] - E_ref
            st = '✓ PASS' if abs(dE) < PASS_TOL else '✗ FAIL'
            print(f'E = {result["energy"]:.15f}, ΔE = {dE:.2e}, '
                  f't = {result["time"]:.2f}s  {st}')
            if abs(dE) >= PASS_TOL:
                issues.append(f'PDMRG2 np={np_count}: ΔE = {dE:.2e}')
    print()

    # ── A2DMRG ─────────────────────────────────────────────────────────────────
    print('Running A2DMRG tests (complex128)...')
    for np_count in nps:
        print(f'  A2DMRG np={np_count}...', end=' ', flush=True)
        result = run_a2dmrg(np_count, E_ref)
        all_results[f'A2DMRG_np{np_count}'] = result
        if 'error' in result:
            print(f'ERROR: {str(result["error"])[:60]}')
            issues.append(f'A2DMRG np={np_count}: {result["error"]}')
        else:
            dE = result['energy'] - E_ref
            st = '✓ PASS' if abs(dE) < PASS_TOL else '✗ FAIL'
            print(f'E = {result["energy"]:.15f}, ΔE = {dE:.2e}, '
                  f't = {result["time"]:.2f}s  {st}')
            if abs(dE) >= PASS_TOL:
                issues.append(f'A2DMRG np={np_count}: ΔE = {dE:.2e}')

    # ── Summary table ────────────────────────────────────────────────────────
    print()
    print('=' * 70)
    print('SUMMARY TABLE')
    print('=' * 70)
    print(f'{"Method":<22} {"Energy":<22} {"ΔE":<12} {"Time (s)":<10} Status')
    print('-' * 70)
    for name, result in all_results.items():
        if 'error' in result:
            print(f'{name:<22} {"ERROR":<22} {"-":<12} {"-":<10} ✗')
        else:
            dE = result['energy'] - E_ref if name != 'quimb_DMRG2' else 0.0
            st = '✓' if abs(dE) < PASS_TOL else '✗'
            print(f'{name:<22} {result["energy"]:<22.15f} '
                  f'{dE:<12.2e} {result["time"]:<10.2f} {st}')

    # ── Issues summary ───────────────────────────────────────────────────────
    print()
    print('=' * 70)
    print('ISSUES DETECTED')
    print('=' * 70)
    if issues:
        for issue in issues:
            print(f'  ⚠ {issue}')
    else:
        print('  ✓ No issues — all complex128 tests passed!')

    # ── Save results ─────────────────────────────────────────────────────────
    output_path = args.out or os.path.join(
        os.path.dirname(__file__), 'josephson_benchmark_results.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nResults saved to: {output_path}')


if __name__ == '__main__':
    main()
