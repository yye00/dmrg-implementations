#!/usr/bin/env python3
"""
Re-run pdmrg-gpu-opt benchmarks with DMRG1 warmup+polish (consistency with pdmrg-gpu).

1. Removes all old pdmrg-gpu-opt results from results.json
2. Re-runs with the rebuilt binary (now uses dmrg1 warmup/polish)
3. Saves updated results.json

Usage (on MI300X):
    cd ~/dmrg-implementations
    python3 benchmarks/rerun_gpu_opt.py 2>&1 | tee benchmarks/paper_results/rerun_gpu_opt.log
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

PG_OPT_BIN = os.path.join(REPO, 'pdmrg-gpu-opt', 'build', 'pdmrg_gpu_opt')

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


def run_gpu_pdmrg_opt(L, chi, sweeps, segments, model='heisenberg', nmax=2):
    if model == 'josephson':
        extra = f'--josephson --nmax {nmax}'
    elif model == 'tfim':
        extra = '--tfim --hfield 1.0'
    else:
        extra = ''
    cmd = f'{PG_OPT_BIN} {L} {chi} {sweeps} --segments {segments} {extra}'
    out, wall, ok = run_cmd(cmd)
    energy = extract(out, r'Final energy:\s*([-\d.eE+]+)')
    wtime = extract(out, r'Total wall time:\s*([\d.]+)')
    polish = extract(out, r'Polish converged after (\d+) sweeps')
    outer = extract(out, r'Converged after (\d+) outer')

    result = {
        'impl': 'pdmrg-gpu-opt',
        'model': model,
        'L': L, 'chi': chi, 'sweeps': sweeps,
        'segments': segments,
        'energy': float(energy) if energy else None,
        'wall_time': float(wtime) if wtime else wall,
        'outer_iters': int(outer) if outer else None,
        'polish_sweeps': int(polish) if polish else None,
        'success': ok and energy is not None,
        'raw_output': out[:2000] if not ok else '',
        'note': 'dmrg1-warmup-polish',
    }
    if model == 'josephson':
        result['nmax'] = nmax
    return result


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

    # Remove old pdmrg-gpu-opt results
    old_count = len(all_results)
    all_results = [r for r in all_results if r['impl'] != 'pdmrg-gpu-opt']
    removed = old_count - len(all_results)
    log(f"Removed {removed} old pdmrg-gpu-opt results, {len(all_results)} remaining")
    save_results(all_results)

    # Track timeouts/OOM for skip logic
    timeout_L_chi = set()

    def should_skip(L, chi):
        for fL, fchi in timeout_L_chi:
            if L >= fL and chi >= fchi:
                return True
        return False

    def record_failure(L, chi, output, wall_time=0):
        if wall_time >= TIMEOUT * 0.9 or is_oom(output):
            timeout_L_chi.add((L, chi))

    n_run = 0

    log("=" * 80)
    log("Re-running pdmrg-gpu-opt with DMRG1 warmup+polish")
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
                if should_skip(L, chi):
                    log(f"  SKIP pdmrg-gpu-opt {model_name} L={L} chi={chi} seg={segments}")
                    continue

                log(f"  pdmrg-gpu-opt {model_name} L={L} chi={chi} seg={segments}")
                r = run_gpu_pdmrg_opt(L, chi, sweeps, segments, model_name, nmax)
                all_results.append(r)
                n_run += 1

                e_str = f"{r['energy']:.12f}" if r['energy'] else 'FAIL'
                t_str = f"{r['wall_time']:.1f}s" if isinstance(r['wall_time'], (int, float)) else str(r['wall_time'])
                status = 'OK' if r['success'] else 'FAIL'
                if not r['success'] and r.get('raw_output') and is_oom(r['raw_output']):
                    status = 'OOM'
                log(f"    E={e_str}  t={t_str}  {status}")

                if not r['success']:
                    record_failure(L, chi, r.get('raw_output', ''), r.get('wall_time', 0))

                if n_run % 5 == 0:
                    save_results(all_results)

    save_results(all_results)
    log(f"Done! Ran {n_run} benchmarks total.")


if __name__ == '__main__':
    main()
