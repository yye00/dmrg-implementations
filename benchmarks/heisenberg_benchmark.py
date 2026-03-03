#!/usr/bin/env python3
"""
Comprehensive Heisenberg Model Benchmark

Tests all implementations against quimb reference:
- quimb DMRG1 (reference)
- quimb DMRG2 (reference)
- PDMRG: np=1, 2, 4, 8
- PDMRG2: np=1, 2, 4, 8
- A2DMRG: np=1, 2, 4, 8

Reports:
- Energy accuracy (vs quimb DMRG2 as ground truth)
- Timing
- Number of sweeps/iterations
- Any issues (convergence, scaling, etc.)
"""

import json
import time
import subprocess
import sys
import os
import numpy as np

# Configuration (defaults; can be overridden by CLI args)
L = 12  # System size
BOND_DIM = 20
MAX_SWEEPS = 30
# Single tolerance criterion used everywhere (convergence + pass/fail)
TOL = 1e-10
CUTOFF = 1e-14

# Pass/fail threshold is the same as solver tol
PASS_TOL = TOL


def _parse_args():
    import argparse

    p = argparse.ArgumentParser(description="Comprehensive Heisenberg benchmark (quimb vs PDMRG vs A2DMRG)")
    p.add_argument("--L", type=int, default=L, help="Chain length")
    p.add_argument("--bond-dim", type=int, default=BOND_DIM, help="Bond dimension")
    p.add_argument("--max-sweeps", type=int, default=MAX_SWEEPS, help="Max sweeps")
    p.add_argument("--tol", type=float, default=TOL, help="Solver tolerance")
    p.add_argument("--cutoff", type=float, default=CUTOFF, help="SVD cutoff")
    p.add_argument("--pass-tol", type=float, default=PASS_TOL, help="Pass/fail threshold vs quimb DMRG2")
    p.add_argument("--out", type=str, default=None, help="Write JSON results to this path")
    p.add_argument("--check-speedup", action="store_true", help="Fail nonzero exit if speedup thresholds not met")
    p.add_argument("--min-efficiency", type=float, default=0.70, help="Minimum parallel efficiency to require when --check-speedup")
    p.add_argument("--nps", type=str, default="1,2,4,8", help="Comma-separated np list")
    return p.parse_args()

def run_quimb_reference():
    """Run quimb DMRG1 and DMRG2 as reference."""
    import quimb.tensor as qtn
    
    mpo = qtn.MPO_ham_heis(L=L, j=1.0, bz=0.0, cyclic=False)
    results = {}
    
    # DMRG1
    t0 = time.time()
    dmrg1 = qtn.DMRG1(mpo, bond_dims=BOND_DIM, cutoffs=CUTOFF)
    dmrg1.solve(max_sweeps=MAX_SWEEPS, tol=TOL, verbosity=0)
    t1 = time.time()
    results['quimb_DMRG1'] = {
        'energy': float(np.real(dmrg1.energy)),
        'time': t1 - t0,
        'sweeps': len(dmrg1.energies) if hasattr(dmrg1, 'energies') else 'N/A'
    }
    
    # DMRG2
    t0 = time.time()
    dmrg2 = qtn.DMRG2(mpo, bond_dims=BOND_DIM, cutoffs=CUTOFF)
    dmrg2.solve(max_sweeps=MAX_SWEEPS, tol=TOL, verbosity=0)
    t1 = time.time()
    results['quimb_DMRG2'] = {
        'energy': float(np.real(dmrg2.energy)),
        'time': t1 - t0,
        'sweeps': len(dmrg2.energies) if hasattr(dmrg2, 'energies') else 'N/A'
    }
    
    return results

def run_pdmrg(np_count, ref_energy):
    """Run PDMRG with given number of processes."""
    script = f'''
import sys
import time
import json
import numpy as np
import quimb.tensor as qtn

sys.path.insert(0, '/home/captain/clawd/work/dmrg-implementations/pdmrg')

from mpi4py import MPI
from pdmrg.dmrg import pdmrg_main
from pdmrg.mps.canonical import get_mpo_tensor_data

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

L = {L}
mpo = qtn.MPO_ham_heis(L=L, j=1.0, bz=0.0, cyclic=False)
mpo_arrays = [get_mpo_tensor_data(mpo, i) for i in range(L)]

t0 = time.time()
energy, pmps = pdmrg_main(
    L=L,
    mpo=mpo,
    max_sweeps={MAX_SWEEPS},
    bond_dim={BOND_DIM},
    bond_dim_warmup={BOND_DIM},
    n_warmup_sweeps=5,
    tol={TOL},
    comm=comm,
    verbose=False
)
t1 = time.time()

if rank == 0:
    result = {{
        'energy': float(energy),
        'time': t1 - t0,
        'sweeps': 'N/A'
    }}
    print(json.dumps(result))
'''
    
    # Write temp script
    script_path = '/tmp/pdmrg_bench.py'
    with open(script_path, 'w') as f:
        f.write(script)
    
    # Run with MPI
    cmd = [
        'mpirun', '-np', str(np_count),
        '--oversubscribe', '--mca', 'btl', 'tcp,self',
        '--mca', 'btl_tcp_if_include', 'lo',
        '/home/captain/clawd/work/dmrg-implementations/pdmrg/venv/bin/python',
        script_path
    ]
    
    env = os.environ.copy()
    # Load MPI module
    env['PATH'] = '/usr/lib64/openmpi/bin:' + env.get('PATH', '')
    env['LD_LIBRARY_PATH'] = '/usr/lib64/openmpi/lib:' + env.get('LD_LIBRARY_PATH', '')
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
        if result.returncode != 0:
            return {'error': result.stderr, 'energy': None, 'time': None}
        
        # Parse JSON from last line of stdout
        lines = [l for l in result.stdout.strip().split('\n') if l.startswith('{')]
        if lines:
            return json.loads(lines[-1])
        else:
            return {'error': 'No output', 'stdout': result.stdout, 'stderr': result.stderr}
    except subprocess.TimeoutExpired:
        return {'error': 'Timeout (300s)', 'energy': None, 'time': None}
    except Exception as e:
        return {'error': str(e), 'energy': None, 'time': None}

def run_pdmrg2(np_count, ref_energy):
    """Run PDMRG2 with given number of processes."""
    script = f'''
import sys
import time
import json
import numpy as np
import quimb.tensor as qtn

sys.path.insert(0, '/home/captain/clawd/work/dmrg-implementations/pdmrg2')

from mpi4py import MPI
from pdmrg.dmrg import pdmrg_main
from pdmrg.mps.canonical import get_mpo_tensor_data

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

L = {L}
mpo = qtn.MPO_ham_heis(L=L, j=1.0, bz=0.0, cyclic=False)
mpo_arrays = [get_mpo_tensor_data(mpo, i) for i in range(L)]

t0 = time.time()
energy, pmps = pdmrg_main(
    L=L,
    mpo=mpo,
    max_sweeps={MAX_SWEEPS},
    bond_dim={BOND_DIM},
    bond_dim_warmup={BOND_DIM},
    n_warmup_sweeps=5,
    tol={TOL},
    comm=comm,
    verbose=False
)
t1 = time.time()

if rank == 0:
    result = {{
        'energy': float(energy),
        'time': t1 - t0,
        'sweeps': 'N/A'
    }}
    print(json.dumps(result))
'''

    script_path = '/tmp/pdmrg2_bench.py'
    with open(script_path, 'w') as f:
        f.write(script)

    cmd = [
        'mpirun', '-np', str(np_count),
        '--oversubscribe', '--mca', 'btl', 'tcp,self',
        '--mca', 'btl_tcp_if_include', 'lo',
        '/home/captain/clawd/work/dmrg-implementations/pdmrg2/venv/bin/python',
        script_path
    ]

    env = os.environ.copy()
    env['PATH'] = '/usr/lib64/openmpi/bin:' + env.get('PATH', '')
    env['LD_LIBRARY_PATH'] = '/usr/lib64/openmpi/lib:' + env.get('LD_LIBRARY_PATH', '')

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
        if result.returncode != 0:
            return {'error': result.stderr, 'energy': None, 'time': None}

        lines = [l for l in result.stdout.strip().split('\n') if l.startswith('{')]
        if lines:
            return json.loads(lines[-1])
        else:
            return {'error': 'No output', 'stdout': result.stdout, 'stderr': result.stderr}
    except subprocess.TimeoutExpired:
        return {'error': 'Timeout (300s)', 'energy': None, 'time': None}
    except Exception as e:
        return {'error': str(e), 'energy': None, 'time': None}

def run_a2dmrg(np_count, ref_energy):
    """Run A2DMRG with given number of processes."""
    script = f'''
import sys
import time
import json
import numpy as np
import quimb.tensor as qtn

sys.path.insert(0, '/home/captain/clawd/work/dmrg-implementations/a2dmrg')

from mpi4py import MPI
from a2dmrg.dmrg import a2dmrg_main

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

L = {L}
mpo = qtn.MPO_ham_heis(L=L, j=1.0, bz=0.0, cyclic=False)

t0 = time.time()
energy, mps = a2dmrg_main(
    L=L,
    mpo=mpo,
    max_sweeps={MAX_SWEEPS},
    bond_dim={BOND_DIM},
    tol={TOL},
    comm=comm,
    warmup_sweeps=5,
    verbose=False
)
t1 = time.time()

if rank == 0:
    result = {{
        'energy': float(np.real(energy)),
        'time': t1 - t0,
        'sweeps': {MAX_SWEEPS}
    }}
    print(json.dumps(result))
'''
    
    script_path = '/tmp/a2dmrg_bench.py'
    with open(script_path, 'w') as f:
        f.write(script)
    
    cmd = [
        'mpirun', '-np', str(np_count),
        '--oversubscribe', '--mca', 'btl', 'tcp,self',
        '--mca', 'btl_tcp_if_include', 'lo',
        '/home/captain/clawd/work/dmrg-implementations/a2dmrg/venv/bin/python',
        script_path
    ]
    
    env = os.environ.copy()
    env['PATH'] = '/usr/lib64/openmpi/bin:' + env.get('PATH', '')
    env['LD_LIBRARY_PATH'] = '/usr/lib64/openmpi/lib:' + env.get('LD_LIBRARY_PATH', '')
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
        if result.returncode != 0:
            return {'error': result.stderr, 'energy': None, 'time': None}
        
        lines = [l for l in result.stdout.strip().split('\n') if l.startswith('{')]
        if lines:
            return json.loads(lines[-1])
        else:
            return {'error': 'No output', 'stdout': result.stdout, 'stderr': result.stderr}
    except subprocess.TimeoutExpired:
        return {'error': 'Timeout (300s)', 'energy': None, 'time': None}
    except Exception as e:
        return {'error': str(e), 'energy': None, 'time': None}

def main():
    global L, BOND_DIM, MAX_SWEEPS, TOL, CUTOFF, PASS_TOL

    args = _parse_args()
    L = args.L
    BOND_DIM = args.bond_dim
    MAX_SWEEPS = args.max_sweeps
    TOL = args.tol
    CUTOFF = args.cutoff
    PASS_TOL = args.pass_tol
    nps = [int(x.strip()) for x in args.nps.split(',') if x.strip()]

    print("=" * 70)
    print("HEISENBERG MODEL BENCHMARK - Comprehensive Test")
    print("=" * 70)
    print(f"Configuration: L={L}, bond_dim={BOND_DIM}, max_sweeps={MAX_SWEEPS}")
    print(f"Tolerance: {TOL}, Cutoff: {CUTOFF}")
    print(f"Pass threshold: |ΔE| < {PASS_TOL}")
    print(f"np list: {nps}")
    print()

    all_results = {}
    issues = []
    
    # Run quimb references
    print("Running quimb references...")
    ref_results = run_quimb_reference()
    all_results.update(ref_results)

    E_ref = ref_results['quimb_DMRG2']['energy']
    # Annotate reference entries with delta_E / passed
    ref_results['quimb_DMRG2']['delta_E'] = 0.0
    ref_results['quimb_DMRG2']['passed'] = True
    dE_dmrg1 = ref_results['quimb_DMRG1']['energy'] - E_ref
    ref_results['quimb_DMRG1']['delta_E'] = dE_dmrg1
    ref_results['quimb_DMRG1']['passed'] = bool(abs(dE_dmrg1) < PASS_TOL)
    print(f"  quimb DMRG1: E = {ref_results['quimb_DMRG1']['energy']:.15f} ({ref_results['quimb_DMRG1']['time']:.2f}s)")
    print(f"  quimb DMRG2: E = {E_ref:.15f} ({ref_results['quimb_DMRG2']['time']:.2f}s) [REFERENCE]")
    print()
    
    # Run PDMRG tests
    print("Running PDMRG tests...")
    for np_count in nps:
        print(f"  PDMRG np={np_count}...", end=" ", flush=True)
        result = run_pdmrg(np_count, E_ref)
        all_results[f'PDMRG_np{np_count}'] = result

        if 'error' in result:
            result['delta_E'] = None
            result['passed'] = False
            print(f"ERROR: {result['error'][:50]}...")
            issues.append(f"PDMRG np={np_count}: {result['error']}")
        else:
            dE = result['energy'] - E_ref
            result['delta_E'] = dE
            result['passed'] = bool(abs(dE) < PASS_TOL)
            status = "✓ PASS" if result['passed'] else "✗ FAIL"
            print(f"E = {result['energy']:.15f}, ΔE = {dE:.2e}, t = {result['time']:.2f}s {status}")
            if not result['passed']:
                issues.append(f"PDMRG np={np_count}: ΔE = {dE:.2e} exceeds threshold")
    print()

    # Run PDMRG2 tests
    print("Running PDMRG2 tests...")
    for np_count in nps:
        print(f"  PDMRG2 np={np_count}...", end=" ", flush=True)
        result = run_pdmrg2(np_count, E_ref)
        all_results[f'PDMRG2_np{np_count}'] = result

        if 'error' in result:
            result['delta_E'] = None
            result['passed'] = False
            print(f"ERROR: {result['error'][:50]}...")
            issues.append(f"PDMRG2 np={np_count}: {result['error']}")
        else:
            dE = result['energy'] - E_ref
            result['delta_E'] = dE
            result['passed'] = bool(abs(dE) < PASS_TOL)
            status = "✓ PASS" if result['passed'] else "✗ FAIL"
            print(f"E = {result['energy']:.15f}, ΔE = {dE:.2e}, t = {result['time']:.2f}s {status}")
            if not result['passed']:
                issues.append(f"PDMRG2 np={np_count}: ΔE = {dE:.2e} exceeds threshold")
    print()

    # Run A2DMRG tests
    print("Running A2DMRG tests...")
    for np_count in nps:
        print(f"  A2DMRG np={np_count}...", end=" ", flush=True)
        result = run_a2dmrg(np_count, E_ref)
        all_results[f'A2DMRG_np{np_count}'] = result

        if 'error' in result:
            result['delta_E'] = None
            result['passed'] = False
            print(f"ERROR: {result['error'][:50]}...")
            issues.append(f"A2DMRG np={np_count}: {result['error']}")
        else:
            dE = result['energy'] - E_ref
            result['delta_E'] = dE
            result['passed'] = bool(abs(dE) < PASS_TOL)
            status = "✓ PASS" if result['passed'] else "✗ FAIL"
            print(f"E = {result['energy']:.15f}, ΔE = {dE:.2e}, t = {result['time']:.2f}s {status}")
            if not result['passed']:
                issues.append(f"A2DMRG np={np_count}: ΔE = {dE:.2e} exceeds threshold")
    
    # Summary table
    print()
    print("=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Method':<20} {'Energy':<20} {'ΔE':<12} {'Time (s)':<10} {'Status'}")
    print("-" * 70)
    
    for name, result in all_results.items():
        if 'error' in result:
            print(f"{name:<20} {'ERROR':<20} {'-':<12} {'-':<10} ✗")
        else:
            dE = result['energy'] - E_ref if name != 'quimb_DMRG2' else 0.0
            status = "✓" if abs(dE) < PASS_TOL else "✗"
            print(f"{name:<20} {result['energy']:<20.15f} {dE:<12.2e} {result['time']:<10.2f} {status}")
    
    # Issues summary
    print()
    print("=" * 70)
    print("ISSUES DETECTED")
    print("=" * 70)
    if issues:
        for issue in issues:
            print(f"  ⚠ {issue}")
    else:
        print("  ✓ No issues detected - all tests passed!")
    
    # Speedup / efficiency summary (A2DMRG only)
    a2_times = {n: all_results.get(f"A2DMRG_np{n}", {}).get("time") for n in nps}
    if nps and nps[0] in a2_times and isinstance(a2_times[nps[0]], (int, float)):
        t1 = a2_times[nps[0]]
        print("\nA2DMRG SPEEDUP")
        print("-" * 70)
        print(f"{'np':>4} {'time(s)':>10} {'speedup':>10} {'efficiency':>12}")
        for n in nps:
            t = a2_times.get(n)
            if not isinstance(t, (int, float)) or t <= 0:
                continue
            s = t1 / t
            eff = s / n
            print(f"{n:>4} {t:>10.2f} {s:>10.2f} {eff:>11.1%}")
            if args.check_speedup and n > 1 and eff < args.min_efficiency:
                issues.append(f"A2DMRG np={n}: efficiency {eff:.1%} < {args.min_efficiency:.0%}")

    # Save results
    output_path = args.out or '/home/captain/clawd/work/dmrg-implementations/benchmarks/heisenberg_benchmark_results.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Exit non-zero if requested checks failed
    if args.check_speedup and issues:
        raise SystemExit(2)

if __name__ == '__main__':
    main()
