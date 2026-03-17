#!/usr/bin/env python3
"""
Comprehensive DMRG benchmark suite for paper results.

Runs all implementations across Heisenberg and Josephson Junction models
at multiple system sizes, bond dimensions, and parallelism levels.

Results are saved as JSON for reproducibility and analysis.

Usage (on MI300X remote host):
    tmux new-session -d -s paper_bench 'cd ~/dmrg-implementations && python3 benchmarks/paper_benchmark.py 2>&1 | tee benchmarks/paper_results/benchmark.log'
"""

import subprocess
import time
import re
import sys
import os
import json
import tempfile
from datetime import datetime
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO, 'benchmarks', 'paper_results')
PG_BIN = os.path.join(REPO, 'pdmrg-gpu', 'build', 'pdmrg_gpu')
P2G_BIN = os.path.join(REPO, 'pdmrg2-gpu', 'build', 'pdmrg2_gpu')

TIMEOUT = 1800  # 30 minutes per run

# Models and sizes
HEISENBERG_SIZES = [
    # (L, chi, sweeps)
    (12, 20, 30),
    (12, 50, 30),
    (12, 128, 40),
    (20, 20, 30),
    (20, 50, 30),
    (20, 128, 40),
    (32, 20, 30),
    (32, 50, 30),
    (32, 128, 40),
    (64, 50, 40),
    (64, 128, 50),
    (100, 50, 50),
    (100, 128, 60),
]

JOSEPHSON_SIZES = [
    # (L, chi, sweeps, nmax)
    (8, 20, 30, 2),
    (8, 50, 30, 2),
    (8, 128, 40, 2),
    (16, 20, 30, 2),
    (16, 50, 30, 2),
    (16, 128, 40, 2),
    (32, 50, 40, 2),
    (32, 128, 50, 2),
    (48, 50, 50, 2),
    (48, 128, 60, 2),
    (64, 50, 50, 2),
    (64, 128, 60, 2),
]

# Parallelism sweeps
QUIMB_THREADS = [1, 2, 4, 8, 12]
MPI_NP = [2, 4, 8]
GPU_SEGMENTS = [2, 4, 8, 16]


# ============================================================================
# Helpers
# ============================================================================

def run_cmd(cmd, env_extra=None, timeout=TIMEOUT):
    """Run a command with timeout, return (stdout+stderr, walltime, success)."""
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    t0 = time.time()
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout, env=env)
        t1 = time.time()
        return r.stdout + r.stderr, t1 - t0, r.returncode == 0
    except subprocess.TimeoutExpired:
        return 'TIMEOUT', timeout, False


def run_python_script(script_content, env_extra=None, timeout=TIMEOUT, mpi_np=None):
    """Write script to temp file and run it, avoiding shell quoting issues."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir='/tmp') as f:
        f.write(script_content)
        script_path = f.name

    try:
        if mpi_np:
            cmd = (f'mpirun --allow-run-as-root --oversubscribe '
                   f'-np {mpi_np} python3 {script_path}')
        else:
            cmd = f'python3 {script_path}'
        return run_cmd(cmd, env_extra=env_extra, timeout=timeout)
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass


def extract(text, pattern):
    """Extract first match from text."""
    m = re.search(pattern, text)
    return m.group(1) if m else None


def save_results(results, filename='results.json'):
    """Save results to JSON file."""
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  [saved {len(results)} results to {path}]")


def log(msg):
    """Print with timestamp."""
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] {msg}")
    sys.stdout.flush()


def max_segments_for_L(L):
    """Max segments allowed: need >=3 sites per segment for 2-site DMRG."""
    return L // 3


# ============================================================================
# quimb DMRG1 / DMRG2 runners
# ============================================================================

def run_quimb_heisenberg(L, chi, sweeps, threads, dmrg_type='DMRG2'):
    """Run quimb DMRG1 or DMRG2 for Heisenberg model."""
    env = {'OPENBLAS_NUM_THREADS': str(threads), 'OMP_NUM_THREADS': str(threads),
           'MKL_NUM_THREADS': str(threads)}
    script = f"""\
import time
import quimb.tensor as qtn

dmrg = qtn.{dmrg_type}(
    qtn.MPO_ham_heis(L={L}, j=1.0, bz=0.0, cyclic=False),
    bond_dims=[{chi}], cutoffs=[1e-12]
)
t0 = time.time()
dmrg.solve(max_sweeps={sweeps}, tol=1e-10, verbosity=0)
t1 = time.time()
e = dmrg.energy
if hasattr(e, 'real'):
    e = e.real
print(f"ENERGY={{e:.15f}}")
print(f"TIME={{t1-t0:.3f}}")
"""
    out, wall, ok = run_python_script(script, env_extra=env)
    energy = extract(out, r'ENERGY=([-\d.eE+]+)')
    solve_time = extract(out, r'TIME=([\d.]+)')
    return {
        'impl': f'quimb-{dmrg_type.lower()}',
        'model': 'heisenberg',
        'L': L, 'chi': chi, 'sweeps': sweeps,
        'threads': threads,
        'energy': float(energy) if energy else None,
        'solve_time': float(solve_time) if solve_time else None,
        'wall_time': wall,
        'success': ok and energy is not None,
        'raw_output': out[:2000] if not ok else '',
    }


def run_quimb_josephson(L, chi, sweeps, nmax, threads, dmrg_type='DMRG2'):
    """Run quimb DMRG1 or DMRG2 for Josephson Junction model."""
    env = {'OPENBLAS_NUM_THREADS': str(threads), 'OMP_NUM_THREADS': str(threads),
           'MKL_NUM_THREADS': str(threads)}
    script = f"""\
import time
import sys
sys.path.insert(0, "{REPO}")
from benchmarks.lib.models import build_josephson_mpo
import quimb.tensor as qtn

mpo = build_josephson_mpo({L}, n_max={nmax})
dmrg = qtn.{dmrg_type}(mpo, bond_dims=[{chi}], cutoffs=[1e-12])
t0 = time.time()
dmrg.solve(max_sweeps={sweeps}, tol=1e-10, verbosity=0)
t1 = time.time()
e = dmrg.energy
if hasattr(e, 'real'):
    e = e.real
print(f"ENERGY={{e:.15f}}")
print(f"TIME={{t1-t0:.3f}}")
"""
    out, wall, ok = run_python_script(script, env_extra=env)
    energy = extract(out, r'ENERGY=([-\d.eE+]+)')
    solve_time = extract(out, r'TIME=([\d.]+)')
    return {
        'impl': f'quimb-{dmrg_type.lower()}',
        'model': 'josephson',
        'L': L, 'chi': chi, 'sweeps': sweeps, 'nmax': nmax,
        'threads': threads,
        'energy': float(energy) if energy else None,
        'solve_time': float(solve_time) if solve_time else None,
        'wall_time': wall,
        'success': ok and energy is not None,
        'raw_output': out[:2000] if not ok else '',
    }


# ============================================================================
# MPI PDMRG / PDMRG2 runners
# ============================================================================

def run_mpi_pdmrg(L, chi, sweeps, np_val, model='heisenberg', nmax=2, impl='pdmrg'):
    """Run MPI-based pdmrg or pdmrg2."""
    pkg_dir = 'pdmrg' if impl == 'pdmrg' else 'pdmrg2'
    env = {'PYTHONPATH': f'{REPO}/{pkg_dir}:{REPO}:' + os.environ.get('PYTHONPATH', ''),
           'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}

    if model == 'heisenberg':
        mpo_setup = f"""\
from pdmrg.hamiltonians.heisenberg import build_heisenberg_mpo
mpo = build_heisenberg_mpo({L})
"""
    else:
        mpo_setup = f"""\
import sys
sys.path.insert(0, "{REPO}")
from benchmarks.lib.models import build_josephson_mpo
mpo = build_josephson_mpo({L}, n_max={nmax})
"""

    script = f"""\
import sys
import time
sys.path.insert(0, "{REPO}/{pkg_dir}")
{mpo_setup}
from pdmrg.dmrg import pdmrg_main
from mpi4py import MPI

t0 = time.time()
energy, pmps = pdmrg_main({L}, mpo, max_sweeps={sweeps}, bond_dim={chi}, tol=1e-10, verbose=False)
t1 = time.time()

if MPI.COMM_WORLD.Get_rank() == 0:
    print(f"ENERGY={{energy:.15f}}")
    print(f"TIME={{t1-t0:.3f}}")
"""
    out, wall, ok = run_python_script(script, env_extra=env, mpi_np=np_val)
    energy = extract(out, r'ENERGY=([-\d.eE+]+)')
    solve_time = extract(out, r'TIME=([\d.]+)')

    result = {
        'impl': impl,
        'model': model,
        'L': L, 'chi': chi, 'sweeps': sweeps,
        'np': np_val,
        'energy': float(energy) if energy else None,
        'solve_time': float(solve_time) if solve_time else None,
        'wall_time': wall,
        'success': ok and energy is not None,
        'raw_output': out[:2000] if not ok else '',
    }
    if model == 'josephson':
        result['nmax'] = nmax
    return result


# ============================================================================
# GPU PDMRG / PDMRG2 runners
# ============================================================================

def run_gpu_pdmrg(L, chi, sweeps, segments, model='heisenberg', nmax=2):
    """Run pdmrg-gpu."""
    extra = f'--josephson --nmax {nmax}' if model == 'josephson' else ''
    cmd = f'{PG_BIN} {L} {chi} {sweeps} --segments {segments} {extra}'
    out, wall, ok = run_cmd(cmd)
    energy = extract(out, r'Final energy:\s*([-\d.eE+]+)')
    wtime = extract(out, r'Total wall time:\s*([\d.]+)')
    polish = extract(out, r'Polish converged after (\d+) sweeps')
    outer = extract(out, r'Converged after (\d+) outer')

    result = {
        'impl': 'pdmrg-gpu',
        'model': model,
        'L': L, 'chi': chi, 'sweeps': sweeps,
        'segments': segments,
        'energy': float(energy) if energy else None,
        'wall_time': float(wtime) if wtime else wall,
        'outer_iters': int(outer) if outer else None,
        'polish_sweeps': int(polish) if polish else None,
        'success': ok and energy is not None,
        'raw_output': out[:2000] if not ok else '',
    }
    if model == 'josephson':
        result['nmax'] = nmax
    return result


def run_gpu_pdmrg2(L, chi, sweeps, segments, model='heisenberg', nmax=2):
    """Run pdmrg2-gpu."""
    extra = f'--josephson --nmax {nmax}' if model == 'josephson' else ''
    cmd = f'{P2G_BIN} {L} {chi} {sweeps} --segments {segments} {extra}'
    out, wall, ok = run_cmd(cmd)
    energy = extract(out, r'Final energy:\s*([-\d.eE+]+)')
    wtime = extract(out, r'Total wall time:\s*([\d.]+)')
    polish = extract(out, r'Polish converged after (\d+) sweeps')
    outer = extract(out, r'Converged after (\d+) outer')

    result = {
        'impl': 'pdmrg2-gpu',
        'model': model,
        'L': L, 'chi': chi, 'sweeps': sweeps,
        'segments': segments,
        'energy': float(energy) if energy else None,
        'wall_time': float(wtime) if wtime else wall,
        'outer_iters': int(outer) if outer else None,
        'polish_sweeps': int(polish) if polish else None,
        'success': ok and energy is not None,
        'raw_output': out[:2000] if not ok else '',
    }
    if model == 'josephson':
        result['nmax'] = nmax
    return result


# ============================================================================
# Main benchmark orchestrator
# ============================================================================

def run_all_benchmarks():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = []
    start_time = time.time()

    def save():
        save_results(all_results)

    # ------------------------------------------------------------------
    # Phase 1: quimb reference (DMRG1 + DMRG2, all thread counts)
    # ------------------------------------------------------------------
    log("=" * 80)
    log("PHASE 1: quimb CPU reference (DMRG1 + DMRG2)")
    log("=" * 80)

    for dmrg_type in ['DMRG1', 'DMRG2']:
        # Heisenberg
        for L, chi, sweeps in HEISENBERG_SIZES:
            for threads in QUIMB_THREADS:
                log(f"  quimb-{dmrg_type.lower()} heisenberg L={L} chi={chi} threads={threads}")
                r = run_quimb_heisenberg(L, chi, sweeps, threads, dmrg_type)
                all_results.append(r)
                e_str = f"{r['energy']:.12f}" if r['energy'] else 'FAIL'
                t_str = f"{r['solve_time']:.1f}s" if r['solve_time'] else f"{r['wall_time']:.1f}s(wall)"
                log(f"    E={e_str}  t={t_str}  {'OK' if r['success'] else 'FAIL'}")
                save()

        # Josephson
        for L, chi, sweeps, nmax in JOSEPHSON_SIZES:
            for threads in QUIMB_THREADS:
                log(f"  quimb-{dmrg_type.lower()} josephson L={L} chi={chi} threads={threads}")
                r = run_quimb_josephson(L, chi, sweeps, nmax, threads, dmrg_type)
                all_results.append(r)
                e_str = f"{r['energy']:.12f}" if r['energy'] else 'FAIL'
                t_str = f"{r['solve_time']:.1f}s" if r['solve_time'] else f"{r['wall_time']:.1f}s(wall)"
                log(f"    E={e_str}  t={t_str}  {'OK' if r['success'] else 'FAIL'}")
                save()

    # ------------------------------------------------------------------
    # Phase 2: MPI pdmrg + pdmrg2
    # ------------------------------------------------------------------
    log("=" * 80)
    log("PHASE 2: MPI parallel DMRG (pdmrg + pdmrg2)")
    log("=" * 80)

    for impl in ['pdmrg', 'pdmrg2']:
        # Heisenberg
        for L, chi, sweeps in HEISENBERG_SIZES:
            for np_val in MPI_NP:
                if np_val > L // 2:
                    log(f"  SKIP {impl} heisenberg L={L} np={np_val} (too many procs)")
                    continue
                log(f"  {impl} heisenberg L={L} chi={chi} np={np_val}")
                r = run_mpi_pdmrg(L, chi, sweeps, np_val, 'heisenberg', impl=impl)
                all_results.append(r)
                e_str = f"{r['energy']:.12f}" if r['energy'] else 'FAIL'
                t_str = f"{r['solve_time']:.1f}s" if r['solve_time'] else f"{r['wall_time']:.1f}s(wall)"
                status = 'TIMEOUT' if 'TIMEOUT' in str(r.get('raw_output', '')) else ('OK' if r['success'] else 'FAIL')
                log(f"    E={e_str}  t={t_str}  {status}")
                save()

        # Josephson
        for L, chi, sweeps, nmax in JOSEPHSON_SIZES:
            for np_val in MPI_NP:
                if np_val > L // 2:
                    log(f"  SKIP {impl} josephson L={L} np={np_val} (too many procs)")
                    continue
                log(f"  {impl} josephson L={L} chi={chi} np={np_val}")
                r = run_mpi_pdmrg(L, chi, sweeps, np_val, 'josephson', nmax, impl=impl)
                all_results.append(r)
                e_str = f"{r['energy']:.12f}" if r['energy'] else 'FAIL'
                t_str = f"{r['solve_time']:.1f}s" if r['solve_time'] else f"{r['wall_time']:.1f}s(wall)"
                status = 'TIMEOUT' if 'TIMEOUT' in str(r.get('raw_output', '')) else ('OK' if r['success'] else 'FAIL')
                log(f"    E={e_str}  t={t_str}  {status}")
                save()

    # ------------------------------------------------------------------
    # Phase 3: GPU pdmrg-gpu + pdmrg2-gpu
    # ------------------------------------------------------------------
    log("=" * 80)
    log("PHASE 3: GPU parallel DMRG (pdmrg-gpu + pdmrg2-gpu)")
    log("=" * 80)

    for gpu_runner, impl_name in [(run_gpu_pdmrg, 'pdmrg-gpu'), (run_gpu_pdmrg2, 'pdmrg2-gpu')]:
        # Heisenberg
        for L, chi, sweeps in HEISENBERG_SIZES:
            for seg in GPU_SEGMENTS:
                max_seg = max_segments_for_L(L)
                if seg > max_seg:
                    log(f"  SKIP {impl_name} heisenberg L={L} seg={seg} (need >=3 sites/seg)")
                    continue
                log(f"  {impl_name} heisenberg L={L} chi={chi} segments={seg}")
                r = gpu_runner(L, chi, sweeps, seg, 'heisenberg')
                all_results.append(r)
                e_str = f"{r['energy']:.12f}" if r['energy'] else 'FAIL'
                t_str = f"{r['wall_time']:.1f}s" if isinstance(r['wall_time'], (int, float)) else str(r['wall_time'])
                log(f"    E={e_str}  t={t_str}  {'OK' if r['success'] else 'FAIL'}")
                save()

        # Josephson
        for L, chi, sweeps, nmax in JOSEPHSON_SIZES:
            for seg in GPU_SEGMENTS:
                max_seg = max_segments_for_L(L)
                if seg > max_seg:
                    log(f"  SKIP {impl_name} josephson L={L} seg={seg} (need >=3 sites/seg)")
                    continue
                log(f"  {impl_name} josephson L={L} chi={chi} segments={seg}")
                r = gpu_runner(L, chi, sweeps, seg, 'josephson', nmax)
                all_results.append(r)
                e_str = f"{r['energy']:.12f}" if r['energy'] else 'FAIL'
                t_str = f"{r['wall_time']:.1f}s" if isinstance(r['wall_time'], (int, float)) else str(r['wall_time'])
                log(f"    E={e_str}  t={t_str}  {'OK' if r['success'] else 'FAIL'}")
                save()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - start_time
    log("=" * 80)
    log(f"BENCHMARK COMPLETE — {len(all_results)} data points in {elapsed/3600:.1f} hours")
    log("=" * 80)

    # Generate summary CSV
    csv_path = os.path.join(RESULTS_DIR, 'summary.csv')
    with open(csv_path, 'w') as f:
        f.write('impl,model,L,chi,threads_or_np_or_seg,energy,solve_time,wall_time,success\n')
        for r in all_results:
            parallelism = r.get('threads', r.get('np', r.get('segments', '')))
            e = r.get('energy', '')
            st = r.get('solve_time', '')
            wt = r.get('wall_time', '')
            f.write(f"{r['impl']},{r['model']},{r['L']},{r['chi']},{parallelism},"
                    f"{e},{st},{wt},{r['success']}\n")
    log(f"Summary CSV: {csv_path}")

    save()
    return all_results


if __name__ == '__main__':
    run_all_benchmarks()
