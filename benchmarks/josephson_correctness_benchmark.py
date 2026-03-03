#!/usr/bin/env python3
"""
Josephson Junction Array — Correctness / Regression Benchmark (complex128)

A small, numerically EXACT problem designed to verify that all three DMRG
implementations handle complex128 arithmetic correctly.

Key design choices vs. josephson_benchmark.py (the performance/scaling test):
  - L=6, n_max=1  →  d=3 local states  (charges: -1, 0, +1)
  - bond_dim=30 > d^(L/2) = 27  →  ZERO truncation error, exact ground state
  - PASS_TOL=1e-9  (vs. 1e-5 in the performance benchmark — 4 orders tighter)
  - PHI_EXT=π/3   (different from the performance benchmark's π/4)
  - np=1,2,4,8  (correctness at all standard process counts)
  - Separate output file: josephson_correctness_results.json

Physical model:
    H = E_C Σ_i (n_i - n_g)²  -  E_J/2 Σ_<ij> [e^{i Φ_ext}(φ⁺_i φ⁻_j) + h.c.]

where φ⁺ = exp(+iφ) raises the charge by 1, φ⁻ lowers it,
and with d=3 (n_max=1) the local basis is |n=-1⟩, |n=0⟩, |n=+1⟩.

Because bond_dim ≥ max bond dimension, all methods must reproduce quimb
DMRG2's energy to within their optimizer convergence (not truncation error).
Any ΔE > 1e-9 against the reference signals a complex128 implementation bug.
"""

import json
import time
import subprocess
import sys
import os
import numpy as np

# ── Configuration ──────────────────────────────────────────────────────────────
L          = 6        # Number of junctions  (small for exact solution)
BOND_DIM   = 30       # > d^(L/2) = 3^3 = 27  →  zero truncation error
MAX_SWEEPS = 20
TOL        = 1e-12    # Tight convergence (exact problem → optimizer-limited)
CUTOFF     = 1e-16    # Essentially no SVD cutoff
PASS_TOL   = 1e-9     # Correctness threshold (no truncation excuse)

# Josephson parameters (different from josephson_benchmark.py)
E_C     = 1.0          # Charging energy (sets energy scale)
E_J     = 1.5          # Josephson coupling  (E_J / E_C = 1.5)
N_G     = 0.0          # Gate charge offset
N_MAX   = 1            # Charges: -1, 0, +1  →  d = 3
PHI_EXT = np.pi / 3    # External flux = π/3  (≠ 0, requires complex128)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Note: d^(L/2) = 3^3 = 27 < BOND_DIM = 30.
# With open boundaries, the maximum Schmidt rank across the centre bond is
# exactly 27, so BOND_DIM=30 achieves zero truncation: the MPS representation
# is exact within the d=3 Hilbert space.

# ── Helper: build the Josephson MPO (in-process, for quimb reference) ─────────

def _build_mpo(L_=None):
    import quimb.tensor as qtn
    n  = L_ if L_ is not None else L
    d         = 2 * N_MAX + 1                          # = 3
    dtype     = 'complex128'
    charges   = np.arange(-N_MAX, N_MAX + 1, dtype=dtype)
    n_op      = np.diag(charges)
    # φ⁺ shifts charge up by 1  (lower-triangular: row n+1, col n)
    exp_iphi  = np.zeros((d, d), dtype=dtype)
    for i in range(d - 1):
        exp_iphi[i + 1, i] = 1.0
    exp_miphi = exp_iphi.conj().T                      # φ⁻ = (φ⁺)†
    # Charging energy: E_C * (n - n_g)^2
    n_minus_ng = n_op - N_G * np.eye(d, dtype=dtype)
    charging   = E_C * (n_minus_ng @ n_minus_ng)
    S          = (d - 1) / 2                           # = 1.0  (spin-1 proxy)
    builder    = qtn.SpinHam1D(S=S)
    flux_phase = np.exp(1j * PHI_EXT)
    # -E_J/2 * e^{iΦ} φ⁺_i φ⁻_j  +  h.c.
    builder.add_term(-E_J / 2 * flux_phase,           exp_iphi,  exp_miphi)
    builder.add_term(-E_J / 2 * np.conj(flux_phase),  exp_miphi, exp_iphi)
    # On-site charging energy
    builder.add_term(1.0, charging)
    return builder.build_mpo(n)


# ── Argument parsing ───────────────────────────────────────────────────────────

def _parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description='Josephson junction correctness benchmark — complex128')
    p.add_argument('--L',          type=int,   default=L)
    p.add_argument('--bond-dim',   type=int,   default=BOND_DIM)
    p.add_argument('--max-sweeps', type=int,   default=MAX_SWEEPS)
    p.add_argument('--tol',        type=float, default=TOL)
    p.add_argument('--cutoff',     type=float, default=CUTOFF)
    p.add_argument('--pass-tol',   type=float, default=PASS_TOL)
    p.add_argument('--out',        type=str,   default=None)
    p.add_argument('--nps',        type=str,   default='1,2,4,8',
                   help='Comma-separated list of MPI process counts')
    return p.parse_args()


# ── Reference runner ───────────────────────────────────────────────────────────

def run_quimb_reference():
    """Run quimb DMRG2 as the complex128 reference (exact, zero truncation)."""
    import quimb.tensor as qtn
    mpo   = _build_mpo()
    t0    = time.time()
    dmrg2 = qtn.DMRG2(mpo, bond_dims=BOND_DIM, cutoffs=CUTOFF)
    dmrg2.solve(max_sweeps=MAX_SWEEPS, tol=TOL, verbosity=0)
    t1    = time.time()
    return {
        'quimb_DMRG2': {
            'energy': float(np.real(dmrg2.energy)),
            'time':   t1 - t0,
        }
    }


# ── Inline MPO build code (embedded verbatim in subprocess scripts) ────────────

def _mpo_build_code():
    """Return Python source that reconstructs the Josephson MPO in a subprocess.

    Uses the same parameters (E_C, E_J, N_MAX, PHI_EXT, L) as the parent
    process, baked in at format-string time so no import of this file is needed.
    """
    return f"""
import numpy as np
import quimb.tensor as qtn

_n_max      = {N_MAX}
_d          = 2 * _n_max + 1                              # = 3
_dtype      = 'complex128'
_charges    = np.arange(-_n_max, _n_max + 1, dtype=_dtype)
_n_op       = np.diag(_charges)
_exp_iphi   = np.zeros((_d, _d), dtype=_dtype)
for _i in range(_d - 1):
    _exp_iphi[_i + 1, _i] = 1.0
_exp_miphi  = _exp_iphi.conj().T
_n_minus_ng = _n_op - {N_G} * np.eye(_d, dtype=_dtype)
_charging   = {E_C} * (_n_minus_ng @ _n_minus_ng)
_S          = (_d - 1) / 2
_builder    = qtn.SpinHam1D(S=_S)
_flux_phase = np.exp(1j * {PHI_EXT})
_builder.add_term(-{E_J} / 2 * _flux_phase,           _exp_iphi,  _exp_miphi)
_builder.add_term(-{E_J} / 2 * np.conj(_flux_phase),  _exp_miphi, _exp_iphi)
_builder.add_term(1.0, _charging)
mpo = _builder.build_mpo({L})
"""


# ── Generic MPI subprocess runner ─────────────────────────────────────────────

def _run_mpi(script, script_path, python_exe, np_count, timeout=120):
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
            return {'error': res.stderr[-2000:], 'energy': None, 'time': None}
        lines = [l for l in res.stdout.strip().split('\n') if l.startswith('{')]
        if lines:
            return json.loads(lines[-1])
        return {'error': 'No JSON output',
                'stdout': res.stdout[-1000:], 'stderr': res.stderr[-1000:]}
    except subprocess.TimeoutExpired:
        return {'error': f'Timeout ({timeout}s)', 'energy': None, 'time': None}
    except Exception as e:
        return {'error': str(e), 'energy': None, 'time': None}


# ── PDMRG runner ──────────────────────────────────────────────────────────────

def run_pdmrg(np_count, ref_energy):
    """Run PDMRG with given number of processes (complex128, correctness test)."""
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
    return _run_mpi(script, '/tmp/pdmrg_jj_correct.py',
                    f'{BASE_DIR}/pdmrg/venv/bin/python', np_count)


# ── PDMRG2 runner ─────────────────────────────────────────────────────────────

def run_pdmrg2(np_count, ref_energy):
    """Run PDMRG2 with given number of processes (complex128, correctness test)."""
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
    return _run_mpi(script, '/tmp/pdmrg2_jj_correct.py',
                    f'{BASE_DIR}/pdmrg2/venv/bin/python', np_count)


# ── A2DMRG runner ─────────────────────────────────────────────────────────────

def run_a2dmrg(np_count, ref_energy):
    """Run A2DMRG with given number of processes (complex128, correctness test)."""
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
    return _run_mpi(script, '/tmp/a2dmrg_jj_correct.py',
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

    d        = 2 * N_MAX + 1
    max_bond = d ** (L // 2)

    print('=' * 72)
    print('JOSEPHSON JUNCTION — CORRECTNESS BENCHMARK  (complex128)')
    print('=' * 72)
    print(f'  H = E_C Σ(n-n_g)²  -  E_J/2 Σ [e^{{iΦ}} φ⁺φ⁻ + h.c.]')
    print(f'  L={L}, d={d} (n_max={N_MAX}), '
          f'E_J/E_C={E_J/E_C:.2f}, Φ_ext=π/{int(round(np.pi/PHI_EXT))}')
    print(f'  bond_dim={BOND_DIM}  (max exact bond = d^(L/2) = {d}^{L//2} = {max_bond})')
    print(f'  {"EXACT (zero truncation)" if BOND_DIM >= max_bond else "TRUNCATED"}')
    print(f'  max_sweeps={MAX_SWEEPS}, tol={TOL}, pass_tol={PASS_TOL}')
    print(f'  np list: {nps}')
    print()

    all_results = {}
    issues      = []

    # ── Reference ──────────────────────────────────────────────────────────────
    print('Running quimb DMRG2 reference (complex128, exact)...')
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
        if 'error' in result and result.get('energy') is None:
            print(f'ERROR: {str(result["error"])[:80]}')
            issues.append(f'PDMRG np={np_count}: {result["error"]}')
        else:
            dE = result['energy'] - E_ref
            st = '✓ PASS' if abs(dE) < PASS_TOL else '✗ FAIL'
            print(f'E = {result["energy"]:.15f},  ΔE = {dE:+.3e},  '
                  f't = {result["time"]:.2f}s  {st}')
            if abs(dE) >= PASS_TOL:
                issues.append(f'PDMRG np={np_count}: ΔE = {dE:.3e}')
    print()

    # ── PDMRG2 ─────────────────────────────────────────────────────────────────
    print('Running PDMRG2 tests (complex128)...')
    for np_count in nps:
        print(f'  PDMRG2 np={np_count}...', end=' ', flush=True)
        result = run_pdmrg2(np_count, E_ref)
        all_results[f'PDMRG2_np{np_count}'] = result
        if 'error' in result and result.get('energy') is None:
            print(f'ERROR: {str(result["error"])[:80]}')
            issues.append(f'PDMRG2 np={np_count}: {result["error"]}')
        else:
            dE = result['energy'] - E_ref
            st = '✓ PASS' if abs(dE) < PASS_TOL else '✗ FAIL'
            print(f'E = {result["energy"]:.15f},  ΔE = {dE:+.3e},  '
                  f't = {result["time"]:.2f}s  {st}')
            if abs(dE) >= PASS_TOL:
                issues.append(f'PDMRG2 np={np_count}: ΔE = {dE:.3e}')
    print()

    # ── A2DMRG ─────────────────────────────────────────────────────────────────
    print('Running A2DMRG tests (complex128)...')
    for np_count in nps:
        print(f'  A2DMRG np={np_count}...', end=' ', flush=True)
        result = run_a2dmrg(np_count, E_ref)
        all_results[f'A2DMRG_np{np_count}'] = result
        if 'error' in result and result.get('energy') is None:
            print(f'ERROR: {str(result["error"])[:80]}')
            issues.append(f'A2DMRG np={np_count}: {result["error"]}')
        else:
            dE = result['energy'] - E_ref
            st = '✓ PASS' if abs(dE) < PASS_TOL else '✗ FAIL'
            print(f'E = {result["energy"]:.15f},  ΔE = {dE:+.3e},  '
                  f't = {result["time"]:.2f}s  {st}')
            if abs(dE) >= PASS_TOL:
                issues.append(f'A2DMRG np={np_count}: ΔE = {dE:.3e}')

    # ── Summary table ─────────────────────────────────────────────────────────
    print()
    print('=' * 72)
    print('SUMMARY TABLE')
    print('=' * 72)
    print(f'{"Method":<22} {"Energy":<24} {"ΔE":<14} {"Time (s)":<10} Status')
    print('-' * 72)
    for name, result in all_results.items():
        if 'error' in result and result.get('energy') is None:
            print(f'{name:<22} {"ERROR":<24} {"-":<14} {"-":<10} ✗')
        else:
            dE = result['energy'] - E_ref if name != 'quimb_DMRG2' else 0.0
            st = '✓' if abs(dE) < PASS_TOL else '✗'
            print(f'{name:<22} {result["energy"]:<24.15f} '
                  f'{dE:<+14.3e} {result["time"]:<10.2f} {st}')

    # ── Issues / verdict ──────────────────────────────────────────────────────
    print()
    print('=' * 72)
    if issues:
        print(f'ISSUES DETECTED  ({len(issues)} failure(s))')
        print('=' * 72)
        for issue in issues:
            print(f'  ⚠  {issue}')
        print()
        print('  NOTE: pass_tol =', PASS_TOL,
              '— failures indicate complex128 implementation bugs,')
        print('  NOT truncation error (bond_dim >= max exact bond).')
    else:
        print('ALL TESTS PASSED')
        print('=' * 72)
        print(f'  ✓  All {len(all_results)-1} complex128 implementations'
              f' match quimb DMRG2 within {PASS_TOL:.0e}')
        print(f'  ✓  bond_dim={BOND_DIM} ≥ max bond {max_bond}: result is exact')

    # ── Save results ──────────────────────────────────────────────────────────
    output_path = args.out or os.path.join(
        os.path.dirname(__file__), 'josephson_correctness_results.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nResults saved to: {output_path}')


if __name__ == '__main__':
    main()
