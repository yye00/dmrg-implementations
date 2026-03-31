#!/usr/bin/env python3
"""
Re-run GPU benchmarks after zero-sync Lanczos changes.
Only runs: dmrg-gpu, dmrg2-gpu, pdmrg-gpu (NOT pdmrg-gpu-opt or MPI).

Usage (on MI300X):
    cd ~/dmrg-implementations
    python3 benchmarks/rerun_zero_sync.py 2>&1 | tee benchmarks/paper_results/rerun_zero_sync.log
"""

import subprocess
import time
import re
import sys
import os
import json
from datetime import datetime
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO, 'benchmarks', 'paper_results')

DG_BIN = os.path.join(REPO, 'dmrg-gpu', 'build', 'dmrg_gpu')
D2G_BIN = os.path.join(REPO, 'dmrg2-gpu', 'build', 'dmrg2_gpu')
PG_BIN = os.path.join(REPO, 'pdmrg-gpu', 'build', 'pdmrg_gpu')

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

GPU_SEGMENTS = [2, 4, 8, 16]

IMPLS_TO_RERUN = {'dmrg-gpu', 'dmrg2-gpu', 'pdmrg-gpu'}

OOM_PATTERNS = [
    'out of memory', 'OOM', 'MemoryError', 'std::bad_alloc',
    'hipErrorOutOfMemory', 'HIP out of memory', 'cannot allocate memory',
    'Killed',
]


def log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] {msg}")
    sys.stdout.flush()


def run_cmd(cmd, timeout=TIMEOUT):
    t0 = time.time()
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        t1 = time.time()
        return r.stdout + r.stderr, t1 - t0, r.returncode == 0
    except subprocess.TimeoutExpired:
        return 'TIMEOUT', timeout, False


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


def max_segments_for_L(L):
    return L // 3


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
        'note': 'zero-sync',
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
        'note': 'zero-sync',
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
        'note': 'zero-sync',
    }
    if model == 'josephson':
        result['nmax'] = nmax
    return result


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, 'results.json')

    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
        log(f"Loaded {len(all_results)} existing results")
    else:
        all_results = []

    # Remove old results for the 3 GPU impls we're re-running
    old_count = len(all_results)
    all_results = [r for r in all_results if r['impl'] not in IMPLS_TO_RERUN]
    removed = old_count - len(all_results)
    log(f"Removed {removed} old results for {IMPLS_TO_RERUN}")
    log(f"Kept {len(all_results)} results (quimb, pdmrg-gpu-opt, MPI, etc)")
    save_results(all_results)

    timeout_records = []
    oom_records = []
    n_run = 0

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
        nonlocal n_run
        all_results.append(r)
        n_run += 1
        if not r['success']:
            record_failure(impl, model, L, chi,
                           r.get('raw_output', ''), r.get('wall_time', 0))
        log_result(r)
        if n_run % 5 == 0:
            save_results(all_results)

    # Phase 0: Serial GPU
    log("=" * 80)
    log("PHASE 0: Serial GPU (dmrg-gpu + dmrg2-gpu) — zero-sync Lanczos")
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

    # Phase 1: Parallel GPU (pdmrg-gpu only)
    log("=" * 80)
    log("PHASE 1: Parallel GPU (pdmrg-gpu) — zero-sync Lanczos")
    log("=" * 80)

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
                if should_skip('pdmrg-gpu', model_name, L, chi, timeout_records):
                    log(f"  SKIP pdmrg-gpu {model_name} L={L} chi={chi} seg={segments} (timeout)")
                    continue
                if should_skip('pdmrg-gpu', model_name, L, chi, oom_records):
                    log(f"  SKIP pdmrg-gpu {model_name} L={L} chi={chi} seg={segments} (OOM)")
                    continue

                log(f"  pdmrg-gpu {model_name} L={L} chi={chi} seg={segments}")
                r = run_gpu_pdmrg(L, chi, sweeps, segments, model_name, nmax)
                handle_result(r, 'pdmrg-gpu', model_name, L, chi)

    save_results(all_results)
    log(f"Done! Ran {n_run} benchmarks total.")


if __name__ == '__main__':
    main()
