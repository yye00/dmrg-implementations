#!/usr/bin/env python3
"""
Re-run all GPU and MPI benchmarks with corrected timing (sweep-to-convergence only).

Keeps quimb results intact. Removes and re-runs:
  - dmrg-gpu, dmrg2-gpu (serial GPU)
  - pdmrg-gpu, pdmrg-gpu-opt (parallel GPU)
  - pdmrg, pdmrg-opt (MPI, now reports SWEEP_TIME)

Usage (on MI300X):
    cd ~/dmrg-implementations
    python3 benchmarks/rerun_timing_fix.py 2>&1 | tee benchmarks/paper_results/rerun_timing.log
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

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO, 'benchmarks', 'paper_results')

# GPU binary paths
DG_BIN = os.path.join(REPO, 'dmrg-gpu', 'build', 'dmrg_gpu')
D2G_BIN = os.path.join(REPO, 'dmrg2-gpu', 'build', 'dmrg2_gpu')
PG_BIN = os.path.join(REPO, 'pdmrg-gpu', 'build', 'pdmrg_gpu')
PGO_BIN = os.path.join(REPO, 'pdmrg-gpu-opt', 'build', 'pdmrg_gpu_opt')

TIMEOUT = 1800

HEISENBERG_SIZES = [
    (12, 20, 30), (12, 50, 30), (12, 128, 40),
    (20, 20, 30), (20, 50, 30), (20, 128, 40),
    (32, 20, 30), (32, 50, 30), (32, 128, 40),
    (64, 50, 40), (64, 128, 50),
    (100, 50, 50), (100, 128, 60),
]

JOSEPHSON_SIZES = [
    (8, 20, 30, 2), (8, 50, 30, 2), (8, 128, 40, 2),
    (16, 20, 30, 2), (16, 50, 30, 2), (16, 128, 40, 2),
    (32, 50, 40, 2), (32, 128, 50, 2),
    (48, 50, 50, 2), (48, 128, 60, 2),
    (64, 50, 50, 2), (64, 128, 60, 2),
]

TFIM_SIZES = [
    (12, 20, 30), (12, 50, 30), (12, 128, 40),
    (20, 20, 30), (20, 50, 30), (20, 128, 40),
    (32, 20, 30), (32, 50, 30), (32, 128, 40),
    (64, 50, 40), (64, 128, 50),
    (100, 50, 50), (100, 128, 60),
]

MPI_NP = [2, 4, 8]
GPU_SEGMENTS = [2, 4, 8, 16]

OOM_PATTERNS = [
    'out of memory', 'OOM', 'MemoryError', 'std::bad_alloc',
    'hipErrorOutOfMemory', 'HIP out of memory', 'cannot allocate memory',
    'Killed',
]

IMPLS_TO_RERUN = {'dmrg-gpu', 'dmrg2-gpu', 'pdmrg-gpu', 'pdmrg-gpu-opt', 'pdmrg', 'pdmrg-opt'}

GIT_PUSH_INTERVAL = 10


def log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] {msg}")
    sys.stdout.flush()


def run_cmd(cmd, env_extra=None, timeout=TIMEOUT):
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
    m = re.search(pattern, text)
    return m.group(1) if m else None


def is_oom(output):
    lower = output.lower()
    return any(pat.lower() in lower for pat in OOM_PATTERNS)


def save_results(results, filename='results.json'):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log(f"  [saved {len(results)} results to {path}]")


def git_push_results():
    try:
        subprocess.run(
            'cd {} && git add benchmarks/paper_results/ && '
            'git diff --cached --quiet || '
            'git commit -m "data: timing-fix benchmark results ({})" && '
            'git push'.format(REPO, datetime.now().strftime('%Y-%m-%d %H:%M')),
            shell=True, capture_output=True, timeout=60
        )
        log("  [git push: results saved to GitHub]")
    except Exception as e:
        log(f"  [git push failed: {e}]")


def max_segments_for_L(L):
    return L // 3


# ============================================================================
# GPU runners
# ============================================================================

def run_gpu_dmrg(L, chi, sweeps, model='heisenberg', nmax=2):
    if model == 'josephson':
        extra = f'--josephson --nmax {nmax}'
    elif model == 'tfim':
        extra = '--tfim --hfield 1.0'
    else:
        extra = ''
    cmd = f'{DG_BIN} {L} {chi} {sweeps} --gpu-svd {extra}'
    out, wall, ok = run_cmd(cmd)
    energy = extract(out, r'Final energy:\s*([-\d.eE+]+)')
    wtime = extract(out, r'Total wall time:\s*([\d.]+)')
    result = {
        'impl': 'dmrg-gpu', 'model': model,
        'L': L, 'chi': chi, 'sweeps': sweeps,
        'energy': float(energy) if energy else None,
        'solve_time': float(wtime) if wtime else None,
        'wall_time': float(wtime) if wtime else wall,
        'success': ok and energy is not None,
        'raw_output': out[:2000] if not ok else '',
        'note': 'timing-fix',
    }
    if model == 'josephson':
        result['nmax'] = nmax
    return result


def run_gpu_dmrg2(L, chi, sweeps, model='heisenberg', nmax=2):
    if model == 'josephson':
        extra = f'--josephson --nmax {nmax}'
    elif model == 'tfim':
        extra = '--tfim --hfield 1.0'
    else:
        extra = ''
    cmd = f'{D2G_BIN} {L} {chi} {sweeps} {extra}'
    out, wall, ok = run_cmd(cmd)
    energy = extract(out, r'Final energy:\s*([-\d.eE+]+)')
    wtime = extract(out, r'Total wall time:\s*([\d.]+)')
    result = {
        'impl': 'dmrg2-gpu', 'model': model,
        'L': L, 'chi': chi, 'sweeps': sweeps,
        'energy': float(energy) if energy else None,
        'solve_time': float(wtime) if wtime else None,
        'wall_time': float(wtime) if wtime else wall,
        'success': ok and energy is not None,
        'raw_output': out[:2000] if not ok else '',
        'note': 'timing-fix',
    }
    if model == 'josephson':
        result['nmax'] = nmax
    return result


def run_gpu_pdmrg(L, chi, sweeps, segments, model='heisenberg', nmax=2):
    if model == 'josephson':
        extra = f'--josephson --nmax {nmax}'
    elif model == 'tfim':
        extra = '--tfim --hfield 1.0'
    else:
        extra = ''
    cmd = f'{PG_BIN} {L} {chi} {sweeps} --segments {segments} {extra}'
    out, wall, ok = run_cmd(cmd)
    energy = extract(out, r'Final energy:\s*([-\d.eE+]+)')
    wtime = extract(out, r'Total wall time:\s*([\d.]+)')
    polish = extract(out, r'Polish converged after (\d+) sweeps')
    outer = extract(out, r'Converged after (\d+) outer')
    result = {
        'impl': 'pdmrg-gpu', 'model': model,
        'L': L, 'chi': chi, 'sweeps': sweeps,
        'segments': segments,
        'energy': float(energy) if energy else None,
        'wall_time': float(wtime) if wtime else wall,
        'outer_iters': int(outer) if outer else None,
        'polish_sweeps': int(polish) if polish else None,
        'success': ok and energy is not None,
        'raw_output': out[:2000] if not ok else '',
        'note': 'timing-fix',
    }
    if model == 'josephson':
        result['nmax'] = nmax
    return result


def run_gpu_pdmrg_opt(L, chi, sweeps, segments, model='heisenberg', nmax=2):
    if model == 'josephson':
        extra = f'--josephson --nmax {nmax}'
    elif model == 'tfim':
        extra = '--tfim --hfield 1.0'
    else:
        extra = ''
    cmd = f'{PGO_BIN} {L} {chi} {sweeps} --segments {segments} {extra}'
    out, wall, ok = run_cmd(cmd)
    energy = extract(out, r'Final energy:\s*([-\d.eE+]+)')
    wtime = extract(out, r'Total wall time:\s*([\d.]+)')
    polish = extract(out, r'Polish converged after (\d+) sweeps')
    outer = extract(out, r'Converged after (\d+) outer')
    result = {
        'impl': 'pdmrg-gpu-opt', 'model': model,
        'L': L, 'chi': chi, 'sweeps': sweeps,
        'segments': segments,
        'energy': float(energy) if energy else None,
        'wall_time': float(wtime) if wtime else wall,
        'outer_iters': int(outer) if outer else None,
        'polish_sweeps': int(polish) if polish else None,
        'success': ok and energy is not None,
        'raw_output': out[:2000] if not ok else '',
        'note': 'timing-fix',
    }
    if model == 'josephson':
        result['nmax'] = nmax
    return result


# ============================================================================
# MPI PDMRG runner
# ============================================================================

def run_mpi_pdmrg(L, chi, sweeps, np_val, model='heisenberg', nmax=2, impl='pdmrg', threads=1):
    pkg_dir = 'pdmrg' if impl == 'pdmrg' else 'pdmrg-opt'
    env = {'PYTHONPATH': f'{REPO}/{pkg_dir}:{REPO}:' + os.environ.get('PYTHONPATH', ''),
           'OPENBLAS_NUM_THREADS': str(threads), 'OMP_NUM_THREADS': str(threads)}

    if model == 'heisenberg':
        mpo_setup = f"import quimb.tensor as qtn\nmpo = qtn.MPO_ham_heis(L={L}, j=1.0, bz=0.0, cyclic=False)"
    elif model == 'josephson':
        mpo_setup = (f"import sys; sys.path.insert(0, '{REPO}')\n"
                     f"from benchmarks.lib.models import build_josephson_mpo\n"
                     f"mpo = build_josephson_mpo({L}, n_max={nmax})")
    else:
        mpo_setup = (f"import sys; sys.path.insert(0, '{REPO}')\n"
                     f"from benchmarks.lib.models import build_tfim_mpo\n"
                     f"mpo = build_tfim_mpo({L}, J=1.0, h=1.0)")

    script = f"""\
import sys, time, json
import numpy as np
sys.path.insert(0, '{REPO}/{pkg_dir}')
sys.path.insert(0, '{REPO}')
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
    sweep_time = extract(out, r'SWEEP_TIME=([\d.]+)')

    result = {
        'impl': impl, 'model': model,
        'L': L, 'chi': chi, 'sweeps': sweeps,
        'np': np_val, 'threads': threads,
        'energy': float(energy) if energy else None,
        'solve_time': float(sweep_time) if sweep_time else (float(solve_time) if solve_time else None),
        'wall_time': wall,
        'success': ok and energy is not None,
        'raw_output': out[:2000] if not ok else '',
        'note': 'timing-fix',
    }
    if model == 'josephson':
        result['nmax'] = nmax
    return result


# ============================================================================
# Main
# ============================================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, 'results.json')

    # Load existing results
    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
        log(f"Loaded {len(all_results)} existing results")
    else:
        all_results = []

    # Remove old GPU + MPI results (keep quimb)
    old_count = len(all_results)
    all_results = [r for r in all_results if r['impl'] not in IMPLS_TO_RERUN]
    removed = old_count - len(all_results)
    log(f"Removed {removed} old GPU/MPI results, {len(all_results)} quimb results kept")
    save_results(all_results)

    # Failure tracking
    timeout_records = []
    oom_records = []
    n_run = 0
    results_since_push = 0

    def should_skip(impl, model, L, chi, records):
        for key, fL, fchi in records:
            if key[0] != impl or key[1] != model:
                continue
            if L >= fL and chi >= fchi:
                return True
        return False

    def record_failure(impl, model, L, chi, output, wall_time=0):
        if wall_time >= TIMEOUT * 0.9:
            timeout_records.append(((impl, model), L, chi))
        if is_oom(output):
            oom_records.append(((impl, model), L, chi))

    def log_result(r):
        e_str = f"{r['energy']:.12f}" if r['energy'] else 'FAIL'
        t_key = 'solve_time' if r.get('solve_time') else 'wall_time'
        t_val = r.get(t_key)
        t_str = f"{t_val:.1f}s" if isinstance(t_val, (int, float)) else str(t_val)
        status = 'OK' if r['success'] else 'FAIL'
        if not r['success'] and r.get('raw_output') and is_oom(r['raw_output']):
            status = 'OOM'
        log(f"    E={e_str}  t={t_str}  {status}")

    def handle_result(r, impl, model, L, chi):
        nonlocal n_run, results_since_push
        all_results.append(r)
        n_run += 1
        if not r['success']:
            record_failure(impl, model, L, chi,
                           r.get('raw_output', ''), r.get('wall_time', 0))
        log_result(r)
        if n_run % 5 == 0:
            save_results(all_results)
        results_since_push += 1
        if results_since_push >= GIT_PUSH_INTERVAL:
            git_push_results()
            results_since_push = 0

    # ------------------------------------------------------------------
    # Phase 0: Serial GPU (dmrg-gpu, dmrg2-gpu) — fastest
    # ------------------------------------------------------------------
    log("=" * 80)
    log("PHASE 0: Serial GPU (dmrg-gpu + dmrg2-gpu) — timing fix")
    log("=" * 80)

    for gpu_runner, impl_name in [(run_gpu_dmrg, 'dmrg-gpu'), (run_gpu_dmrg2, 'dmrg2-gpu')]:
        for model_name, sizes in [('heisenberg', HEISENBERG_SIZES),
                                   ('josephson', JOSEPHSON_SIZES),
                                   ('tfim', TFIM_SIZES)]:
            for size_tuple in sizes:
                if model_name == 'josephson':
                    L, chi, sweeps, nmax = size_tuple
                else:
                    L, chi, sweeps = size_tuple
                    nmax = 2

                if should_skip(impl_name, model_name, L, chi, timeout_records):
                    log(f"  SKIP {impl_name} {model_name} L={L} chi={chi} (timeout)")
                    continue
                if should_skip(impl_name, model_name, L, chi, oom_records):
                    log(f"  SKIP {impl_name} {model_name} L={L} chi={chi} (OOM)")
                    continue

                log(f"  {impl_name} {model_name} L={L} chi={chi}")
                r = gpu_runner(L, chi, sweeps, model_name, nmax)
                handle_result(r, impl_name, model_name, L, chi)

    save_results(all_results)
    git_push_results()

    # ------------------------------------------------------------------
    # Phase 1: Parallel GPU (pdmrg-gpu, pdmrg-gpu-opt)
    # ------------------------------------------------------------------
    log("=" * 80)
    log("PHASE 1: Parallel GPU (pdmrg-gpu + pdmrg-gpu-opt) — timing fix")
    log("=" * 80)

    for gpu_runner, impl_name in [(run_gpu_pdmrg, 'pdmrg-gpu'), (run_gpu_pdmrg_opt, 'pdmrg-gpu-opt')]:
        for model_name, sizes in [('heisenberg', HEISENBERG_SIZES),
                                   ('josephson', JOSEPHSON_SIZES),
                                   ('tfim', TFIM_SIZES)]:
            for size_tuple in sizes:
                if model_name == 'josephson':
                    L, chi, sweeps, nmax = size_tuple
                else:
                    L, chi, sweeps = size_tuple
                    nmax = 2

                for segments in GPU_SEGMENTS:
                    if segments > max_segments_for_L(L):
                        continue
                    if should_skip(impl_name, model_name, L, chi, timeout_records):
                        log(f"  SKIP {impl_name} {model_name} L={L} chi={chi} seg={segments} (timeout)")
                        continue
                    if should_skip(impl_name, model_name, L, chi, oom_records):
                        log(f"  SKIP {impl_name} {model_name} L={L} chi={chi} seg={segments} (OOM)")
                        continue

                    log(f"  {impl_name} {model_name} L={L} chi={chi} seg={segments}")
                    r = gpu_runner(L, chi, sweeps, segments, model_name, nmax)
                    handle_result(r, impl_name, model_name, L, chi)

    save_results(all_results)
    git_push_results()

    # ------------------------------------------------------------------
    # Phase 2: MPI PDMRG (pdmrg, pdmrg-opt) — slowest
    # ------------------------------------------------------------------
    log("=" * 80)
    log("PHASE 2: MPI PDMRG (pdmrg + pdmrg-opt) — timing fix (SWEEP_TIME)")
    log("=" * 80)

    for impl in ['pdmrg', 'pdmrg-opt']:
        for model_name, sizes in [('heisenberg', HEISENBERG_SIZES),
                                   ('josephson', JOSEPHSON_SIZES),
                                   ('tfim', TFIM_SIZES)]:
            for size_tuple in sizes:
                if model_name == 'josephson':
                    L, chi, sweeps, nmax = size_tuple
                else:
                    L, chi, sweeps = size_tuple
                    nmax = 2

                for np_val in MPI_NP:
                    if np_val > L // 2:
                        continue
                    if should_skip(impl, model_name, L, chi, timeout_records):
                        log(f"  SKIP {impl} {model_name} L={L} chi={chi} np={np_val} (timeout)")
                        continue

                    log(f"  {impl} {model_name} L={L} chi={chi} np={np_val}")
                    r = run_mpi_pdmrg(L, chi, sweeps, np_val, model_name, nmax, impl)
                    handle_result(r, impl, model_name, L, chi)

    save_results(all_results)
    git_push_results()
    log(f"Done! Ran {n_run} benchmarks total.")


if __name__ == '__main__':
    main()
