#!/usr/bin/env python3
"""
Comprehensive DMRG benchmark suite for paper results.

Runs all implementations across Heisenberg, Josephson Junction, and TFIM models
at multiple system sizes, bond dimensions, and parallelism levels.

Results are saved as JSON for reproducibility and analysis.
Periodic git push protects against data loss on ephemeral VMs.

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
sys.path.insert(0, os.path.join(REPO, "benchmarks", "lib"))
from provenance import provenance_block, binary_info
RESULTS_DIR = os.path.join(REPO, 'benchmarks', 'paper_results')

# GPU binary paths
DG_BIN = os.path.join(REPO, 'dmrg-gpu', 'build', 'dmrg_gpu')
D2G_BIN = os.path.join(REPO, 'dmrg2-gpu', 'build', 'dmrg2_gpu')
PG_BIN = os.path.join(REPO, 'pdmrg-gpu', 'build', 'pdmrg_gpu')
P2G_BIN = os.path.join(REPO, 'pdmrg-gpu-opt', 'build', 'pdmrg_gpu_opt')

TIMEOUT = 1800  # 30 minutes per run

# Push results to GitHub every N results
GIT_PUSH_INTERVAL = 10

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

TFIM_SIZES = [
    # (L, chi, sweeps) — h/J = 1.0 (critical point)
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

# Parallelism sweeps
# Note: OpenBLAS thread counts >4 cause severe anti-scaling for quimb,
# especially with complex-valued Josephson (80-100x slowdown at 12 threads).
# Root cause: OpenBLAS pthread overhead per BLAS call dominates for small matrices.
# Cap at 4 threads — beyond that, use MPI parallelism instead.
QUIMB_THREADS = [1, 2, 4]
MPI_NP = [2, 4, 8]
GPU_SEGMENTS = [2, 4, 8, 16]

# OOM detection patterns in stderr
OOM_PATTERNS = [
    'out of memory', 'OOM', 'MemoryError', 'std::bad_alloc',
    'hipErrorOutOfMemory', 'HIP out of memory', 'cannot allocate memory',
    'Killed',  # Linux OOM killer
]


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


def is_oom(output):
    """Detect out-of-memory errors in command output."""
    lower = output.lower()
    return any(pat.lower() in lower for pat in OOM_PATTERNS)


_PROVENANCE_CACHE = None

def _get_provenance():
    """Lazy-capture provenance once per process run."""
    global _PROVENANCE_CACHE
    if _PROVENANCE_CACHE is None:
        _PROVENANCE_CACHE = {
            "provenance": provenance_block(repo_root=REPO, script_argv=sys.argv[1:]),
            "binaries":   {
                "dmrg-gpu":       binary_info(DG_BIN),
                "dmrg2-gpu":      binary_info(D2G_BIN),
                "pdmrg-gpu":      binary_info(PG_BIN),
                "pdmrg-gpu-opt":  binary_info(P2G_BIN),
            },
        }
    return _PROVENANCE_CACHE


def save_results(results, filename='results.json'):
    """Save results to JSON file (wrapped in a provenance envelope)."""
    path = os.path.join(RESULTS_DIR, filename)
    envelope = {
        "provenance":  _get_provenance()["provenance"],
        "binaries":    _get_provenance()["binaries"],
        "results":     results,
    }
    with open(path, 'w') as f:
        json.dump(envelope, f, indent=2, default=str)
    print(f"  [saved {len(results)} results to {path}]")


def git_push_results():
    """Commit and push results to GitHub for data safety."""
    try:
        subprocess.run(
            'cd {} && git add benchmarks/paper_results/ && '
            'git diff --cached --quiet || '
            'git commit -m "auto: benchmark results update ({})" && '
            'git push'.format(REPO, datetime.now().strftime('%Y-%m-%d %H:%M')),
            shell=True, capture_output=True, timeout=60
        )
        log("  [git push: results saved to GitHub]")
    except Exception as e:
        log(f"  [git push failed: {e}]")


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


def run_quimb_tfim(L, chi, sweeps, threads, dmrg_type='DMRG2'):
    """Run quimb DMRG1 or DMRG2 for transverse-field Ising model."""
    env = {'OPENBLAS_NUM_THREADS': str(threads), 'OMP_NUM_THREADS': str(threads),
           'MKL_NUM_THREADS': str(threads)}
    script = f"""\
import time
import sys
sys.path.insert(0, "{REPO}")
from benchmarks.lib.models import build_tfim_mpo
import quimb.tensor as qtn

mpo = build_tfim_mpo({L}, J=1.0, h=1.0)
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
        'model': 'tfim',
        'L': L, 'chi': chi, 'sweeps': sweeps,
        'threads': threads,
        'energy': float(energy) if energy else None,
        'solve_time': float(solve_time) if solve_time else None,
        'wall_time': wall,
        'success': ok and energy is not None,
        'raw_output': out[:2000] if not ok else '',
    }


# ============================================================================
# MPI PDMRG / PDMRG-OPT runners
# ============================================================================

def run_mpi_pdmrg(L, chi, sweeps, np_val, model='heisenberg', nmax=2, impl='pdmrg', threads=1):
    """Run MPI-based pdmrg or pdmrg-opt."""
    pkg_dir = 'pdmrg' if impl == 'pdmrg' else 'pdmrg-opt'
    env = {'PYTHONPATH': f'{REPO}/{pkg_dir}:{REPO}:' + os.environ.get('PYTHONPATH', ''),
           'OPENBLAS_NUM_THREADS': str(threads), 'OMP_NUM_THREADS': str(threads)}

    if model == 'heisenberg':
        mpo_setup = f"""\
from pdmrg.hamiltonians.heisenberg import build_heisenberg_mpo
mpo = build_heisenberg_mpo({L})
"""
    elif model == 'josephson':
        mpo_setup = f"""\
import sys
sys.path.insert(0, "{REPO}")
from benchmarks.lib.models import build_josephson_mpo
mpo = build_josephson_mpo({L}, n_max={nmax})
"""
    else:  # tfim
        mpo_setup = f"""\
import sys
sys.path.insert(0, "{REPO}")
from benchmarks.lib.models import build_tfim_mpo
mpo = build_tfim_mpo({L}, J=1.0, h=1.0)
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
    sweep_time = extract(out, r'SWEEP_TIME=([\d.]+)')

    result = {
        'impl': impl,
        'model': model,
        'L': L, 'chi': chi, 'sweeps': sweeps,
        'np': np_val,
        'threads': threads,
        'energy': float(energy) if energy else None,
        'solve_time': float(sweep_time) if sweep_time else (float(solve_time) if solve_time else None),
        'wall_time': wall,
        'success': ok and energy is not None,
        'raw_output': out[:2000] if not ok else '',
    }
    if model == 'josephson':
        result['nmax'] = nmax
    return result


# ============================================================================
# Single-GPU DMRG runners (dmrg-gpu, dmrg2-gpu)
# ============================================================================

def run_gpu_dmrg(L, chi, sweeps, model='heisenberg', nmax=2):
    """Run single-GPU dmrg-gpu (1-site DMRG).

    Note: dmrg-gpu defaults to CPU SVD — must pass --gpu-svd explicitly.
    """
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
        'impl': 'dmrg-gpu',
        'model': model,
        'L': L, 'chi': chi, 'sweeps': sweeps,
        'energy': float(energy) if energy else None,
        'solve_time': float(wtime) if wtime else None,
        'wall_time': float(wtime) if wtime else wall,
        'success': ok and energy is not None,
        'raw_output': out[:2000] if not ok else '',
    }
    if model == 'josephson':
        result['nmax'] = nmax
    return result


def run_gpu_dmrg2(L, chi, sweeps, model='heisenberg', nmax=2):
    """Run single-GPU dmrg2-gpu (2-site DMRG).

    Note: dmrg2-gpu defaults to GPU SVD — no extra flag needed.
    """
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
        'impl': 'dmrg2-gpu',
        'model': model,
        'L': L, 'chi': chi, 'sweeps': sweeps,
        'energy': float(energy) if energy else None,
        'solve_time': float(wtime) if wtime else None,
        'wall_time': float(wtime) if wtime else wall,
        'success': ok and energy is not None,
        'raw_output': out[:2000] if not ok else '',
    }
    if model == 'josephson':
        result['nmax'] = nmax
    return result


# ============================================================================
# GPU PDMRG / PDMRG-OPT runners (multi-segment parallel)
# ============================================================================

def run_gpu_pdmrg(L, chi, sweeps, segments, model='heisenberg', nmax=2):
    """Run pdmrg-gpu (multi-segment parallel, GPU SVD default)."""
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
    """Run pdmrg-gpu-opt (multi-segment parallel, GPU SVD default)."""
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


# ============================================================================
# Main benchmark orchestrator
# ============================================================================

def make_result_key(r):
    """Create a unique key for a result to detect duplicates."""
    impl = r['impl']
    model = r['model']
    L = r['L']
    chi = r['chi']
    # For MPI results with both np and threads, include both
    np_val = r.get('np')
    threads = r.get('threads')
    if np_val is not None and threads is not None:
        p = f"{np_val}x{threads}"
    else:
        p = r.get('threads', r.get('np', r.get('segments', '')))
    return f"{impl}|{model}|{L}|{chi}|{p}"


def load_existing_results():
    """Load previously completed results for resume support."""
    path = os.path.join(RESULTS_DIR, 'results.json')
    if os.path.exists(path):
        with open(path) as f:
            results = json.load(f)
        log(f"Loaded {len(results)} existing results for resume")
        return results
    return []


def run_all_benchmarks():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Resume: load existing results and build lookup of completed runs
    # Only count successes and real timeouts as completed; retry config errors
    all_results = load_existing_results()
    completed = set(make_result_key(r) for r in all_results
                    if r['success'] or r.get('wall_time', 0) >= TIMEOUT * 0.9)

    # Timeout/OOM tracking: aggressive skip logic
    timeout_records = []  # list of ((impl, model), L, chi) for cross-parameter skipping
    oom_records = []      # same format for OOM failures
    for r in all_results:
        if not r['success']:
            if r.get('wall_time', 0) >= TIMEOUT * 0.9:
                timeout_records.append(((r['impl'], r['model']), r['L'], r['chi']))
            raw = r.get('raw_output', '')
            if raw and is_oom(raw):
                oom_records.append(((r['impl'], r['model']), r['L'], r['chi']))

    start_time = time.time()
    results_since_push = 0

    def save():
        save_results(all_results)

    def maybe_push():
        nonlocal results_since_push
        results_since_push += 1
        if results_since_push >= GIT_PUSH_INTERVAL:
            git_push_results()
            results_since_push = 0

    def should_skip(impl, model, L, chi, records):
        """Skip if we can infer this will fail (timeout or OOM).

        Skip when ANY of these hold:
        1. A smaller L failed at same (impl, model, chi) with ANY parallelism
        2. A smaller chi failed at same (impl, model, L) with ANY parallelism
        3. Same (impl, model, L, chi) failed with ANY parallelism value
        """
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

    def should_skip_timeout(impl, model, L, chi, parallelism):
        return should_skip(impl, model, L, chi, timeout_records)

    def should_skip_oom(impl, model, L, chi):
        return should_skip(impl, model, L, chi, oom_records)

    def record_failure(impl, model, L, chi, output, wall_time=0):
        """Record timeout or OOM for aggressive skip logic."""
        if wall_time >= TIMEOUT * 0.9:
            timeout_records.append(((impl, model), L, chi))
        if is_oom(output):
            oom_records.append(((impl, model), L, chi))

    def is_completed(key):
        return key in completed

    def log_result(r):
        e_str = f"{r['energy']:.12f}" if r['energy'] else 'FAIL'
        t_str = f"{r['solve_time']:.1f}s" if r.get('solve_time') else (
            f"{r['wall_time']:.1f}s" if isinstance(r['wall_time'], (int, float)) else str(r['wall_time']))
        status = 'OK' if r['success'] else 'TIMEOUT/FAIL'
        if not r['success'] and r.get('raw_output') and is_oom(r['raw_output']):
            status = 'OOM'
        log(f"    E={e_str}  t={t_str}  {status}")

    def run_phase_result(r, impl, model, L, chi, parallelism_key):
        """Common handler: record result, check failures, save, maybe push."""
        all_results.append(r)
        completed.add(parallelism_key)
        if not r['success']:
            record_failure(impl, model, L, chi,
                           r.get('raw_output', ''), r.get('wall_time', 0))
        log_result(r)
        save()
        maybe_push()

    # ------------------------------------------------------------------
    # Phase 0: Single-GPU DMRG (dmrg-gpu + dmrg2-gpu) — paper core
    # ------------------------------------------------------------------
    log("=" * 80)
    log("PHASE 0: Single-GPU DMRG (dmrg-gpu + dmrg2-gpu)")
    log("=" * 80)

    for gpu_runner, impl_name in [(run_gpu_dmrg, 'dmrg-gpu'), (run_gpu_dmrg2, 'dmrg2-gpu')]:
        # Heisenberg
        for L, chi, sweeps in HEISENBERG_SIZES:
            key = f"{impl_name}|heisenberg|{L}|{chi}|"
            if is_completed(key):
                continue
            if should_skip_timeout(impl_name, 'heisenberg', L, chi, ''):
                log(f"  SKIP {impl_name} heisenberg L={L} chi={chi} (timeout)")
                continue
            if should_skip_oom(impl_name, 'heisenberg', L, chi):
                log(f"  SKIP {impl_name} heisenberg L={L} chi={chi} (OOM)")
                continue
            log(f"  {impl_name} heisenberg L={L} chi={chi}")
            r = gpu_runner(L, chi, sweeps, 'heisenberg')
            run_phase_result(r, impl_name, 'heisenberg', L, chi, key)

        # Josephson
        for L, chi, sweeps, nmax in JOSEPHSON_SIZES:
            key = f"{impl_name}|josephson|{L}|{chi}|"
            if is_completed(key):
                continue
            if should_skip_timeout(impl_name, 'josephson', L, chi, ''):
                log(f"  SKIP {impl_name} josephson L={L} chi={chi} (timeout)")
                continue
            if should_skip_oom(impl_name, 'josephson', L, chi):
                log(f"  SKIP {impl_name} josephson L={L} chi={chi} (OOM)")
                continue
            log(f"  {impl_name} josephson L={L} chi={chi}")
            r = gpu_runner(L, chi, sweeps, 'josephson', nmax)
            run_phase_result(r, impl_name, 'josephson', L, chi, key)

        # TFIM
        for L, chi, sweeps in TFIM_SIZES:
            key = f"{impl_name}|tfim|{L}|{chi}|"
            if is_completed(key):
                continue
            if should_skip_timeout(impl_name, 'tfim', L, chi, ''):
                log(f"  SKIP {impl_name} tfim L={L} chi={chi} (timeout)")
                continue
            if should_skip_oom(impl_name, 'tfim', L, chi):
                log(f"  SKIP {impl_name} tfim L={L} chi={chi} (OOM)")
                continue
            log(f"  {impl_name} tfim L={L} chi={chi}")
            r = gpu_runner(L, chi, sweeps, 'tfim')
            run_phase_result(r, impl_name, 'tfim', L, chi, key)

    # ------------------------------------------------------------------
    # Phase 1: quimb reference (DMRG1 + DMRG2, all thread counts)
    # ------------------------------------------------------------------
    log("=" * 80)
    log("PHASE 1: quimb CPU reference (DMRG1 + DMRG2)")
    log("=" * 80)

    for dmrg_type in ['DMRG1', 'DMRG2']:
        impl_name = f'quimb-{dmrg_type.lower()}'
        # Heisenberg
        for L, chi, sweeps in HEISENBERG_SIZES:
            for threads in QUIMB_THREADS:
                key = f"{impl_name}|heisenberg|{L}|{chi}|{threads}"
                if is_completed(key):
                    continue
                if should_skip_timeout(impl_name, 'heisenberg', L, chi, threads):
                    log(f"  SKIP {impl_name} heisenberg L={L} chi={chi} threads={threads} (timeout)")
                    continue
                log(f"  {impl_name} heisenberg L={L} chi={chi} threads={threads}")
                r = run_quimb_heisenberg(L, chi, sweeps, threads, dmrg_type)
                run_phase_result(r, impl_name, 'heisenberg', L, chi, key)

        # Josephson
        for L, chi, sweeps, nmax in JOSEPHSON_SIZES:
            for threads in QUIMB_THREADS:
                key = f"{impl_name}|josephson|{L}|{chi}|{threads}"
                if is_completed(key):
                    continue
                if should_skip_timeout(impl_name, 'josephson', L, chi, threads):
                    log(f"  SKIP {impl_name} josephson L={L} chi={chi} threads={threads} (timeout)")
                    continue
                log(f"  {impl_name} josephson L={L} chi={chi} threads={threads}")
                r = run_quimb_josephson(L, chi, sweeps, nmax, threads, dmrg_type)
                run_phase_result(r, impl_name, 'josephson', L, chi, key)

        # TFIM
        for L, chi, sweeps in TFIM_SIZES:
            for threads in QUIMB_THREADS:
                key = f"{impl_name}|tfim|{L}|{chi}|{threads}"
                if is_completed(key):
                    continue
                if should_skip_timeout(impl_name, 'tfim', L, chi, threads):
                    log(f"  SKIP {impl_name} tfim L={L} chi={chi} threads={threads} (timeout)")
                    continue
                log(f"  {impl_name} tfim L={L} chi={chi} threads={threads}")
                r = run_quimb_tfim(L, chi, sweeps, threads, dmrg_type)
                run_phase_result(r, impl_name, 'tfim', L, chi, key)

    # ------------------------------------------------------------------
    # Phase 2: MPI pdmrg + pdmrg-opt
    # ------------------------------------------------------------------
    log("=" * 80)
    log("PHASE 2: MPI parallel DMRG (pdmrg + pdmrg-opt)")
    log("=" * 80)

    for impl in ['pdmrg', 'pdmrg-opt']:
        # Heisenberg
        for L, chi, sweeps in HEISENBERG_SIZES:
            for np_val in MPI_NP:
                if np_val > L // 2:
                    continue
                key = f"{impl}|heisenberg|{L}|{chi}|{np_val}"
                if is_completed(key):
                    continue
                if should_skip_timeout(impl, 'heisenberg', L, chi, np_val):
                    log(f"  SKIP {impl} heisenberg L={L} chi={chi} np={np_val} (timeout)")
                    continue
                log(f"  {impl} heisenberg L={L} chi={chi} np={np_val}")
                r = run_mpi_pdmrg(L, chi, sweeps, np_val, 'heisenberg', impl=impl)
                run_phase_result(r, impl, 'heisenberg', L, chi, key)

        # Josephson
        for L, chi, sweeps, nmax in JOSEPHSON_SIZES:
            for np_val in MPI_NP:
                if np_val > L // 2:
                    continue
                key = f"{impl}|josephson|{L}|{chi}|{np_val}"
                if is_completed(key):
                    continue
                if should_skip_timeout(impl, 'josephson', L, chi, np_val):
                    log(f"  SKIP {impl} josephson L={L} chi={chi} np={np_val} (timeout)")
                    continue
                log(f"  {impl} josephson L={L} chi={chi} np={np_val}")
                r = run_mpi_pdmrg(L, chi, sweeps, np_val, 'josephson', nmax, impl=impl)
                run_phase_result(r, impl, 'josephson', L, chi, key)

        # TFIM
        for L, chi, sweeps in TFIM_SIZES:
            for np_val in MPI_NP:
                if np_val > L // 2:
                    continue
                key = f"{impl}|tfim|{L}|{chi}|{np_val}"
                if is_completed(key):
                    continue
                if should_skip_timeout(impl, 'tfim', L, chi, np_val):
                    log(f"  SKIP {impl} tfim L={L} chi={chi} np={np_val} (timeout)")
                    continue
                log(f"  {impl} tfim L={L} chi={chi} np={np_val}")
                r = run_mpi_pdmrg(L, chi, sweeps, np_val, 'tfim', impl=impl)
                run_phase_result(r, impl, 'tfim', L, chi, key)

    # ------------------------------------------------------------------
    # Phase 3: GPU pdmrg-gpu + pdmrg-gpu-opt (multi-segment parallel)
    # ------------------------------------------------------------------
    log("=" * 80)
    log("PHASE 3: GPU parallel DMRG (pdmrg-gpu + pdmrg-gpu-opt)")
    log("=" * 80)

    for gpu_runner, impl_name in [(run_gpu_pdmrg, 'pdmrg-gpu'), (run_gpu_pdmrg_opt, 'pdmrg-gpu-opt')]:
        # Heisenberg
        for L, chi, sweeps in HEISENBERG_SIZES:
            for seg in GPU_SEGMENTS:
                max_seg = max_segments_for_L(L)
                if seg > max_seg:
                    continue
                key = f"{impl_name}|heisenberg|{L}|{chi}|{seg}"
                if is_completed(key):
                    continue
                if should_skip_timeout(impl_name, 'heisenberg', L, chi, seg):
                    log(f"  SKIP {impl_name} heisenberg L={L} chi={chi} seg={seg} (timeout)")
                    continue
                if should_skip_oom(impl_name, 'heisenberg', L, chi):
                    log(f"  SKIP {impl_name} heisenberg L={L} chi={chi} seg={seg} (OOM)")
                    continue
                log(f"  {impl_name} heisenberg L={L} chi={chi} segments={seg}")
                r = gpu_runner(L, chi, sweeps, seg, 'heisenberg')
                run_phase_result(r, impl_name, 'heisenberg', L, chi, key)

        # Josephson
        for L, chi, sweeps, nmax in JOSEPHSON_SIZES:
            for seg in GPU_SEGMENTS:
                max_seg = max_segments_for_L(L)
                if seg > max_seg:
                    continue
                key = f"{impl_name}|josephson|{L}|{chi}|{seg}"
                if is_completed(key):
                    continue
                if should_skip_timeout(impl_name, 'josephson', L, chi, seg):
                    log(f"  SKIP {impl_name} josephson L={L} chi={chi} seg={seg} (timeout)")
                    continue
                if should_skip_oom(impl_name, 'josephson', L, chi):
                    log(f"  SKIP {impl_name} josephson L={L} chi={chi} seg={seg} (OOM)")
                    continue
                log(f"  {impl_name} josephson L={L} chi={chi} segments={seg}")
                r = gpu_runner(L, chi, sweeps, seg, 'josephson', nmax)
                run_phase_result(r, impl_name, 'josephson', L, chi, key)

        # TFIM
        for L, chi, sweeps in TFIM_SIZES:
            for seg in GPU_SEGMENTS:
                max_seg = max_segments_for_L(L)
                if seg > max_seg:
                    continue
                key = f"{impl_name}|tfim|{L}|{chi}|{seg}"
                if is_completed(key):
                    continue
                if should_skip_timeout(impl_name, 'tfim', L, chi, seg):
                    log(f"  SKIP {impl_name} tfim L={L} chi={chi} seg={seg} (timeout)")
                    continue
                if should_skip_oom(impl_name, 'tfim', L, chi):
                    log(f"  SKIP {impl_name} tfim L={L} chi={chi} seg={seg} (OOM)")
                    continue
                log(f"  {impl_name} tfim L={L} chi={chi} segments={seg}")
                r = gpu_runner(L, chi, sweeps, seg, 'tfim')
                run_phase_result(r, impl_name, 'tfim', L, chi, key)

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
    # Final push to GitHub
    git_push_results()
    return all_results


if __name__ == '__main__':
    run_all_benchmarks()
