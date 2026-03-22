#!/usr/bin/env python3
"""
Benchmark dmrg-gpu-opt and dmrg2-gpu-opt vs baselines (dmrg-gpu, dmrg2-gpu).

Runs Heisenberg, Josephson Junction, and TFIM models at multiple (L, chi).
Smart timeout: 300s per run. Convergence target: 1e-10.
Computes win rates for opt vs baseline wall time.

Usage (on MI300X remote):
    python3 bench_opt.py 2>&1 | tee bench_opt.log
"""

import subprocess
import time
import re
import sys
import os
import json
import csv
from datetime import datetime

# Binary paths (relative to home dir)
HOME = os.path.expanduser('~')
BINS = {
    'dmrg-gpu':      os.path.join(HOME, 'dmrg-gpu', 'build', 'dmrg_gpu'),
    'dmrg-gpu-opt':  os.path.join(HOME, 'dmrg-gpu-opt', 'build', 'dmrg_gpu_opt'),
    'dmrg2-gpu':     os.path.join(HOME, 'dmrg2-gpu', 'build', 'dmrg2_gpu'),
    'dmrg2-gpu-opt': os.path.join(HOME, 'dmrg2-gpu-opt', 'build', 'dmrg2_gpu_opt'),
}

TIMEOUT = 300  # 5 min per run
RESULTS_FILE = os.path.join(HOME, 'bench_opt_results.json')
CSV_FILE = os.path.join(HOME, 'bench_opt_results.csv')

# Benchmark configurations: (L, chi, sweeps)
HEISENBERG_SIZES = [
    (16, 20, 30), (16, 50, 20), (16, 128, 15), (16, 256, 10),
    (32, 20, 30), (32, 50, 20), (32, 128, 15), (32, 256, 10),
    (64, 20, 30), (64, 50, 20), (64, 128, 15), (64, 256, 10),
    (128, 20, 30), (128, 50, 20), (128, 128, 15), (128, 256, 10),
]

# (L, chi, sweeps, nmax)
JOSEPHSON_SIZES = [
    (8, 20, 30, 1), (8, 50, 20, 1), (8, 128, 15, 1), (8, 256, 10, 1),
    (16, 20, 30, 1), (16, 50, 20, 1), (16, 128, 15, 1), (16, 256, 10, 1),
    (32, 20, 30, 1), (32, 50, 20, 1), (32, 128, 15, 1), (32, 256, 10, 1),
    (64, 20, 30, 1), (64, 50, 20, 1), (64, 128, 15, 1), (64, 256, 10, 1),
]

TFIM_SIZES = [
    (16, 20, 30), (16, 50, 20), (16, 128, 15), (16, 256, 10),
    (32, 20, 30), (32, 50, 20), (32, 128, 15), (32, 256, 10),
    (64, 20, 30), (64, 50, 20), (64, 128, 15), (64, 256, 10),
    (128, 20, 30), (128, 50, 20), (128, 128, 15), (128, 256, 10),
]

# Implementations to benchmark
IMPLS_1SITE = ['dmrg-gpu', 'dmrg-gpu-opt']
IMPLS_2SITE = ['dmrg2-gpu', 'dmrg2-gpu-opt']
ALL_IMPLS = IMPLS_1SITE + IMPLS_2SITE


def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)


def run_benchmark(impl, L, chi, sweeps, model='heisenberg', nmax=1):
    """Run a single benchmark configuration."""
    binary = BINS[impl]
    if not os.path.exists(binary):
        return None

    if model == 'josephson':
        extra = f'--josephson --nmax {nmax}'
    elif model == 'tfim':
        extra = '--tfim --hfield 1.0'
    else:
        extra = ''

    # baselines always use GPU SVD (rocsolver); opt versions use Newton-Schulz (GPU GEMMs)
    cmd = f'{binary} {L} {chi} {sweeps} {extra}'

    t0 = time.time()
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=TIMEOUT)
        wall = time.time() - t0
        output = r.stdout + r.stderr
        ok = r.returncode == 0
    except subprocess.TimeoutExpired:
        wall = TIMEOUT
        output = 'TIMEOUT'
        ok = False

    energy = None
    solve_time = None
    sweeps_done = None
    converged = False

    if ok:
        m = re.search(r'Final energy:\s*([-\d.eE+]+)', output)
        if m:
            energy = float(m.group(1))
        m = re.search(r'Total wall time:\s*([\d.]+)', output)
        if m:
            solve_time = float(m.group(1))
            wall = solve_time  # use reported wall time
        m = re.search(r'Converged after (\d+) sweeps', output)
        if m:
            converged = True
            sweeps_done = int(m.group(1))

    return {
        'impl': impl,
        'model': model,
        'L': L,
        'chi': chi,
        'energy': energy,
        'wall_time': round(wall, 3),
        'sweeps': sweeps_done if sweeps_done else sweeps,
        'converged': converged,
    }


def compute_win_rates(results):
    """Compute win rates: opt vs baseline for matching (model, L, chi) configs."""
    # Group by (model, L, chi, site_type)
    groups = {}
    for r in results:
        if r is None or r['energy'] is None:
            continue
        impl = r['impl']
        site_type = '1site' if 'dmrg-gpu' in impl and 'dmrg2' not in impl else '2site'
        key = (r['model'], r['L'], r['chi'], site_type)
        if key not in groups:
            groups[key] = {}
        groups[key][impl] = r

    wins = {'dmrg-gpu-opt': 0, 'dmrg-gpu': 0, 'tie': 0,
            'dmrg2-gpu-opt': 0, 'dmrg2-gpu': 0, 'tie2': 0}
    comparisons_1site = 0
    comparisons_2site = 0
    details = []

    for key, impls in sorted(groups.items()):
        model, L, chi, site_type = key
        if site_type == '1site':
            base = impls.get('dmrg-gpu')
            opt = impls.get('dmrg-gpu-opt')
            if base and opt and base['converged'] and opt['converged']:
                comparisons_1site += 1
                speedup = base['wall_time'] / opt['wall_time'] if opt['wall_time'] > 0 else 999
                if speedup > 1.05:
                    wins['dmrg-gpu-opt'] += 1
                    winner = 'opt'
                elif speedup < 0.95:
                    wins['dmrg-gpu'] += 1
                    winner = 'base'
                else:
                    wins['tie'] += 1
                    winner = 'tie'
                details.append({
                    'model': model, 'L': L, 'chi': chi, 'type': '1-site',
                    'base_time': base['wall_time'], 'opt_time': opt['wall_time'],
                    'speedup': round(speedup, 2), 'winner': winner
                })
        else:
            base = impls.get('dmrg2-gpu')
            opt = impls.get('dmrg2-gpu-opt')
            if base and opt and base['converged'] and opt['converged']:
                comparisons_2site += 1
                speedup = base['wall_time'] / opt['wall_time'] if opt['wall_time'] > 0 else 999
                if speedup > 1.05:
                    wins['dmrg2-gpu-opt'] += 1
                    winner = 'opt'
                elif speedup < 0.95:
                    wins['dmrg2-gpu'] += 1
                    winner = 'base'
                else:
                    wins['tie2'] += 1
                    winner = 'tie'
                details.append({
                    'model': model, 'L': L, 'chi': chi, 'type': '2-site',
                    'base_time': base['wall_time'], 'opt_time': opt['wall_time'],
                    'speedup': round(speedup, 2), 'winner': winner
                })

    return wins, comparisons_1site, comparisons_2site, details


def main():
    log("=== GPU-OPT Benchmark Suite ===")
    log(f"Timeout: {TIMEOUT}s per run")

    # Check binaries
    for impl, path in BINS.items():
        exists = os.path.exists(path)
        log(f"  {impl}: {'OK' if exists else 'MISSING'} ({path})")
        if not exists and 'opt' in impl:
            log(f"ERROR: {impl} binary not found!")
            return

    results = []
    total_configs = (len(HEISENBERG_SIZES) + len(JOSEPHSON_SIZES) + len(TFIM_SIZES)) * len(ALL_IMPLS)
    run_count = 0

    # --- Heisenberg ---
    log("\n=== HEISENBERG (d=2, real) ===")
    for L, chi, sweeps in HEISENBERG_SIZES:
        for impl in ALL_IMPLS:
            run_count += 1
            log(f"[{run_count}/{total_configs}] {impl} Heisenberg L={L} chi={chi}...")
            r = run_benchmark(impl, L, chi, sweeps, 'heisenberg')
            if r:
                results.append(r)
                status = f"E={r['energy']:.6f} t={r['wall_time']:.1f}s" if r['energy'] else "FAIL"
                if not r['converged'] and r['energy']:
                    status += " (NOT CONVERGED)"
                log(f"  -> {status}")
            else:
                log(f"  -> SKIP (binary missing)")

    # --- Josephson ---
    log("\n=== JOSEPHSON (d=3, complex) ===")
    for L, chi, sweeps, nmax in JOSEPHSON_SIZES:
        for impl in ALL_IMPLS:
            run_count += 1
            log(f"[{run_count}/{total_configs}] {impl} Josephson L={L} chi={chi}...")
            r = run_benchmark(impl, L, chi, sweeps, 'josephson', nmax)
            if r:
                results.append(r)
                status = f"E={r['energy']:.6f} t={r['wall_time']:.1f}s" if r['energy'] else "FAIL"
                if not r['converged'] and r['energy']:
                    status += " (NOT CONVERGED)"
                log(f"  -> {status}")
            else:
                log(f"  -> SKIP (binary missing)")

    # --- TFIM ---
    log("\n=== TFIM (d=2, real, h/J=1.0 critical) ===")
    for L, chi, sweeps in TFIM_SIZES:
        for impl in ALL_IMPLS:
            run_count += 1
            log(f"[{run_count}/{total_configs}] {impl} TFIM L={L} chi={chi}...")
            r = run_benchmark(impl, L, chi, sweeps, 'tfim')
            if r:
                results.append(r)
                status = f"E={r['energy']:.6f} t={r['wall_time']:.1f}s" if r['energy'] else "FAIL"
                if not r['converged'] and r['energy']:
                    status += " (NOT CONVERGED)"
                log(f"  -> {status}")
            else:
                log(f"  -> SKIP (binary missing)")

    # Save results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    log(f"\nSaved {len(results)} results to {RESULTS_FILE}")

    # Save CSV
    if results:
        keys = ['model', 'impl', 'L', 'chi', 'energy', 'wall_time', 'sweeps', 'converged']
        with open(CSV_FILE, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
            w.writeheader()
            w.writerows(results)
        log(f"Saved CSV to {CSV_FILE}")

    # Win rate analysis
    log("\n" + "=" * 60)
    log("WIN RATE ANALYSIS (opt vs baseline, converged runs only)")
    log("=" * 60)

    wins, n1, n2, details = compute_win_rates(results)

    log(f"\n--- 1-site: dmrg-gpu-opt vs dmrg-gpu ({n1} comparisons) ---")
    if n1 > 0:
        log(f"  opt wins: {wins['dmrg-gpu-opt']}/{n1} ({100*wins['dmrg-gpu-opt']/n1:.0f}%)")
        log(f"  base wins: {wins['dmrg-gpu']}/{n1} ({100*wins['dmrg-gpu']/n1:.0f}%)")
        log(f"  ties (<5%): {wins['tie']}/{n1}")

    log(f"\n--- 2-site: dmrg2-gpu-opt vs dmrg2-gpu ({n2} comparisons) ---")
    if n2 > 0:
        log(f"  opt wins: {wins['dmrg2-gpu-opt']}/{n2} ({100*wins['dmrg2-gpu-opt']/n2:.0f}%)")
        log(f"  base wins: {wins['dmrg2-gpu']}/{n2} ({100*wins['dmrg2-gpu']/n2:.0f}%)")
        log(f"  ties (<5%): {wins['tie2']}/{n2}")

    log(f"\n--- Detailed comparisons ---")
    log(f"{'Model':<12} {'L':>4} {'chi':>4} {'Type':<7} {'Base(s)':>8} {'Opt(s)':>8} {'Speedup':>8} {'Winner'}")
    log("-" * 65)
    for d in details:
        log(f"{d['model']:<12} {d['L']:>4} {d['chi']:>4} {d['type']:<7} "
            f"{d['base_time']:>8.1f} {d['opt_time']:>8.1f} {d['speedup']:>7.2f}x {d['winner']}")

    log("\n=== BENCHMARK COMPLETE ===")


if __name__ == '__main__':
    main()
