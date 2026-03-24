#!/usr/bin/env python3
"""
Re-run pdmrg-gpu and pdmrg-gpu-opt benchmarks after Stoudenmire V=Λ⁻¹ fix.

1. Removes all old pdmrg-gpu / pdmrg-gpu-opt results from results.json
2. Re-runs Phase 3 (GPU parallel) benchmarks for both implementations
3. Saves updated results.json and summary.csv

Usage (on MI300X):
    cd ~/dmrg-implementations
    python3 benchmarks/rerun_pdmrg_gpu.py 2>&1 | tee benchmarks/paper_results/rerun.log
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

PG_BIN = os.path.join(REPO, 'pdmrg-gpu', 'build', 'pdmrg_gpu')
P2G_BIN = os.path.join(REPO, 'pdmrg-gpu-opt', 'build', 'pdmrg_gpu_opt')

TIMEOUT = 1800

# Same sizes as paper_benchmark.py
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


def max_segments_for_L(L):
    return L // 3


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


def run_gpu_pdmrg_opt(L, chi, sweeps, segments, model='heisenberg', nmax=2):
    if model == 'josephson':
        extra = f'--josephson --nmax {nmax}'
    elif model == 'tfim':
        extra = '--tfim --hfield 1.0'
    else:
        extra = ''
    cmd = f'{P2G_BIN} {L} {chi} {sweeps} --segments {segments} {extra}'
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
    }
    if model == 'josephson':
        result['nmax'] = nmax
    return result


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Step 1: Load existing results
    # On first run: remove old pdmrg-gpu/pdmrg-gpu-opt entries
    # On resume: keep new entries (already cleaned in prior run)
    results_path = os.path.join(RESULTS_DIR, 'results.json')
    marker_path = os.path.join(RESULTS_DIR, '.rerun_cleaned')
    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
        if not os.path.exists(marker_path):
            # First run: strip old broken results
            n_before = len(all_results)
            all_results = [r for r in all_results
                           if r['impl'] not in ('pdmrg-gpu', 'pdmrg-gpu-opt')]
            n_removed = n_before - len(all_results)
            log(f"Loaded {n_before} results, removed {n_removed} old pdmrg-gpu/pdmrg-gpu-opt entries")
            with open(results_path, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            with open(marker_path, 'w') as f:
                f.write(datetime.now().isoformat())
        else:
            log(f"Resuming: {len(all_results)} results (already cleaned)")
    else:
        all_results = []
        log("No existing results.json found, starting fresh")

    log(f"Base results: {len(all_results)}")

    # Step 2: Re-run all pdmrg-gpu and pdmrg-gpu-opt benchmarks
    # Build completed set from any already-run results (for resume)
    completed = set()
    for r in all_results:
        if r['impl'] in ('pdmrg-gpu', 'pdmrg-gpu-opt') and r.get('success'):
            seg = r.get('segments', '')
            completed.add(f"{r['impl']}|{r['model']}|{r['L']}|{r['chi']}|{seg}")

    new_results = []
    timeout_records = []
    oom_records = []
    start_time = time.time()
    results_since_push = 0

    def should_skip(impl, model, L, chi, records):
        for key, fL, fchi in records:
            if key[0] != impl or key[1] != model:
                continue
            if fchi == chi and L >= fL:
                return True
            if fL == L and chi >= fchi:
                return True
            if L >= fL and chi >= fchi:
                return True
        return False

    def save():
        merged = all_results + new_results
        with open(results_path, 'w') as f:
            json.dump(merged, f, indent=2, default=str)

    def git_push():
        # Skip git push — no auth configured on this remote VM.
        # Results are saved locally; pull them after benchmark completes.
        pass

    def maybe_push():
        nonlocal results_since_push
        results_since_push += 1
        if results_since_push >= 10:
            git_push()
            results_since_push = 0

    for gpu_runner, impl_name in [(run_gpu_pdmrg, 'pdmrg-gpu'),
                                   (run_gpu_pdmrg_opt, 'pdmrg-gpu-opt')]:
        log("=" * 70)
        log(f"Running {impl_name}")
        log("=" * 70)

        # Heisenberg
        for L, chi, sweeps in HEISENBERG_SIZES:
            for seg in GPU_SEGMENTS:
                if seg > max_segments_for_L(L):
                    continue
                key = f"{impl_name}|heisenberg|{L}|{chi}|{seg}"
                if key in completed:
                    continue
                if should_skip(impl_name, 'heisenberg', L, chi, timeout_records):
                    log(f"  SKIP {impl_name} heisenberg L={L} chi={chi} seg={seg} (timeout)")
                    continue
                if should_skip(impl_name, 'heisenberg', L, chi, oom_records):
                    log(f"  SKIP {impl_name} heisenberg L={L} chi={chi} seg={seg} (OOM)")
                    continue

                log(f"  {impl_name} heisenberg L={L} chi={chi} seg={seg}")
                r = gpu_runner(L, chi, sweeps, seg, 'heisenberg')
                new_results.append(r)

                e_str = f"{r['energy']:.12f}" if r['energy'] else 'FAIL'
                wt = r.get('wall_time', 0)
                t_str = f"{wt:.1f}s" if isinstance(wt, (int, float)) else str(wt)
                status = 'OK' if r['success'] else 'FAIL'
                if not r['success'] and r.get('raw_output') and is_oom(r['raw_output']):
                    status = 'OOM'
                    oom_records.append(((impl_name, 'heisenberg'), L, chi))
                if not r['success'] and isinstance(wt, (int, float)) and wt >= TIMEOUT * 0.9:
                    status = 'TIMEOUT'
                    timeout_records.append(((impl_name, 'heisenberg'), L, chi))
                log(f"    E={e_str}  t={t_str}  {status}")
                save()
                maybe_push()

        # Josephson
        for L, chi, sweeps, nmax in JOSEPHSON_SIZES:
            for seg in GPU_SEGMENTS:
                if seg > max_segments_for_L(L):
                    continue
                key = f"{impl_name}|josephson|{L}|{chi}|{seg}"
                if key in completed:
                    continue
                if should_skip(impl_name, 'josephson', L, chi, timeout_records):
                    log(f"  SKIP {impl_name} josephson L={L} chi={chi} seg={seg} (timeout)")
                    continue
                if should_skip(impl_name, 'josephson', L, chi, oom_records):
                    log(f"  SKIP {impl_name} josephson L={L} chi={chi} seg={seg} (OOM)")
                    continue

                log(f"  {impl_name} josephson L={L} chi={chi} seg={seg}")
                r = gpu_runner(L, chi, sweeps, seg, 'josephson', nmax)
                new_results.append(r)

                e_str = f"{r['energy']:.12f}" if r['energy'] else 'FAIL'
                wt = r.get('wall_time', 0)
                t_str = f"{wt:.1f}s" if isinstance(wt, (int, float)) else str(wt)
                status = 'OK' if r['success'] else 'FAIL'
                if not r['success'] and r.get('raw_output') and is_oom(r['raw_output']):
                    status = 'OOM'
                    oom_records.append(((impl_name, 'josephson'), L, chi))
                if not r['success'] and isinstance(wt, (int, float)) and wt >= TIMEOUT * 0.9:
                    timeout_records.append(((impl_name, 'josephson'), L, chi))
                log(f"    E={e_str}  t={t_str}  {status}")
                save()
                maybe_push()

        # TFIM
        for L, chi, sweeps in TFIM_SIZES:
            for seg in GPU_SEGMENTS:
                if seg > max_segments_for_L(L):
                    continue
                key = f"{impl_name}|tfim|{L}|{chi}|{seg}"
                if key in completed:
                    continue
                if should_skip(impl_name, 'tfim', L, chi, timeout_records):
                    log(f"  SKIP {impl_name} tfim L={L} chi={chi} seg={seg} (timeout)")
                    continue
                if should_skip(impl_name, 'tfim', L, chi, oom_records):
                    log(f"  SKIP {impl_name} tfim L={L} chi={chi} seg={seg} (OOM)")
                    continue

                log(f"  {impl_name} tfim L={L} chi={chi} seg={seg}")
                r = gpu_runner(L, chi, sweeps, seg, 'tfim')
                new_results.append(r)

                e_str = f"{r['energy']:.12f}" if r['energy'] else 'FAIL'
                wt = r.get('wall_time', 0)
                t_str = f"{wt:.1f}s" if isinstance(wt, (int, float)) else str(wt)
                status = 'OK' if r['success'] else 'FAIL'
                if not r['success'] and r.get('raw_output') and is_oom(r['raw_output']):
                    status = 'OOM'
                    oom_records.append(((impl_name, 'tfim'), L, chi))
                if not r['success'] and isinstance(wt, (int, float)) and wt >= TIMEOUT * 0.9:
                    timeout_records.append(((impl_name, 'tfim'), L, chi))
                log(f"    E={e_str}  t={t_str}  {status}")
                save()
                maybe_push()

    # Final save
    elapsed = time.time() - start_time
    log("=" * 70)
    log(f"COMPLETE — {len(new_results)} new results in {elapsed/3600:.1f} hours")
    log(f"  Successes: {sum(1 for r in new_results if r['success'])}")
    log(f"  Failures: {sum(1 for r in new_results if not r['success'])}")
    log("=" * 70)

    save()

    # Generate summary CSV
    merged = all_results + new_results
    csv_path = os.path.join(RESULTS_DIR, 'summary.csv')
    with open(csv_path, 'w') as f:
        f.write('impl,model,L,chi,threads_or_np_or_seg,energy,solve_time,wall_time,success\n')
        for r in merged:
            parallelism = r.get('threads', r.get('np', r.get('segments', '')))
            e = r.get('energy', '')
            st = r.get('solve_time', '')
            wt = r.get('wall_time', '')
            ok = r.get('success', False)
            f.write(f"{r['impl']},{r['model']},{r['L']},{r['chi']},{parallelism},"
                    f"{e},{st},{wt},{ok}\n")
    log(f"Summary CSV saved to {csv_path}")

    git_push()


if __name__ == '__main__':
    main()
