#!/usr/bin/env python3
"""
Comprehensive DMRG Benchmark: PDMRG-GPU (2,4,8 streams) vs dmrg2-gpu vs dmrg-gpu vs quimb DMRG2

Runs:
  - quimb DMRG2: single-thread CPU (local, OMP_NUM_THREADS=1)
  - dmrg-gpu:    single-site GPU DMRG (remote MI300X)
  - dmrg2-gpu:   two-site GPU DMRG (remote MI300X)
  - pdmrg-gpu:   stream-parallel DMRG with P=2,4,8 (remote MI300X)
"""

import os
import sys
import time
import json
import subprocess
import numpy as np

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

import quimb.tensor as qtn

REMOTE = "hotaisle@23.183.40.79"
REMOTE_DIR = "/home/hotaisle/dmrg-implementations"


# ============================================================================
# Josephson MPO for quimb
# ============================================================================
def build_josephson_mpo_quimb(L, n_max=1, E_J=1.0, E_C=0.5, phi_ext=np.pi/4):
    d = 2 * n_max + 1
    D = 4
    eye = np.eye(d, dtype=complex)
    exp_iphi = np.zeros((d, d), dtype=complex)
    exp_miphi = np.zeros((d, d), dtype=complex)
    H_onsite = np.zeros((d, d), dtype=complex)
    for i in range(d):
        charge = i - n_max
        H_onsite[i, i] = E_C * charge**2
    for i in range(d - 1):
        exp_iphi[i + 1, i] = 1.0
        exp_miphi[i, i + 1] = 1.0
    alpha = -E_J / 2.0 * np.exp(1j * phi_ext)
    alpha_conj = np.conj(alpha)

    def build_W(site):
        W = np.zeros((D, D, d, d), dtype=complex)
        if site == 0:
            W[0, 0] = eye; W[0, 1] = alpha * exp_iphi
            W[0, 2] = alpha_conj * exp_miphi; W[0, 3] = H_onsite
        elif site == L - 1:
            W[0, 3] = H_onsite; W[1, 3] = exp_miphi
            W[2, 3] = exp_iphi; W[3, 3] = eye
        else:
            W[0, 0] = eye; W[0, 1] = alpha * exp_iphi
            W[0, 2] = alpha_conj * exp_miphi; W[0, 3] = H_onsite
            W[1, 3] = exp_miphi; W[2, 3] = exp_iphi; W[3, 3] = eye
        return W

    arrays = []
    for site in range(L):
        W = build_W(site)
        if site == 0:
            arrays.append(W[0, :, :, :])
        elif site == L - 1:
            arrays.append(W[:, D-1, :, :])
        else:
            arrays.append(W)
    return qtn.MatrixProductOperator(arrays)


# ============================================================================
# CPU benchmark: quimb DMRG2
# ============================================================================
def run_quimb(L, chi, model='heisenberg', max_sweeps=50, timeout=300, **kwargs):
    """Run quimb DMRG2 with a signal-based timeout."""
    import signal

    class TimeoutError(Exception):
        pass

    def _handler(signum, frame):
        raise TimeoutError()

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout)
    try:
        if model == 'heisenberg':
            mpo = qtn.MPO_ham_heis(L=L, j=1.0, bz=0.0, cyclic=False)
        else:
            mpo = build_josephson_mpo_quimb(L, **kwargs)
        dmrg = qtn.DMRG2(mpo, bond_dims=chi, cutoffs=1e-14)
        t0 = time.perf_counter()
        dmrg.solve(max_sweeps=max_sweeps, tol=1e-12, verbosity=0)
        t1 = time.perf_counter()
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        return float(np.real(dmrg.energy)), t1 - t0
    except TimeoutError:
        signal.signal(signal.SIGALRM, old_handler)
        return None, None
    except Exception:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        return None, None


# ============================================================================
# GPU benchmark: parse output for energy and wall time
# ============================================================================
def parse_gpu_output(output):
    energy = None
    wall_time = None
    for line in output.split('\n'):
        if 'Final energy:' in line:
            energy = float(line.split(':')[1].strip())
        if 'Total wall time:' in line:
            wall_time = float(line.split(':')[1].strip().replace('s', '').strip())
    return energy, wall_time


def run_gpu(exe_dir, exe_name, L, chi, n_sweeps, extra_args="", timeout=600):
    cmd = f"ssh {REMOTE} 'cd {REMOTE_DIR}/{exe_dir}/build && ./{exe_name} {L} {chi} {n_sweeps} {extra_args}'"
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        output = result.stdout + result.stderr
        energy, wall_time = parse_gpu_output(output)
        return energy, wall_time
    except subprocess.TimeoutExpired:
        return None, None


def run_dmrg_gpu(L, chi, n_sweeps, extra_args=""):
    return run_gpu('dmrg-gpu', 'dmrg_gpu', L, chi, n_sweeps, extra_args)


def run_dmrg2_gpu(L, chi, n_sweeps, extra_args=""):
    return run_gpu('dmrg2-gpu', 'dmrg2_gpu', L, chi, n_sweeps, extra_args)


def run_pdmrg_gpu(L, chi, n_outer, n_segments, extra_args="", warmup=3, local_sweeps=2):
    args = f"--segments {n_segments} --warmup {warmup} --local-sweeps {local_sweeps} {extra_args}"
    return run_gpu('pdmrg-gpu', 'pdmrg_gpu', L, chi, n_outer, args)


# ============================================================================
# Main benchmark
# ============================================================================
def main():
    print("=" * 120)
    print("COMPREHENSIVE DMRG BENCHMARK")
    print("  CPU: quimb DMRG2 (single-thread, OMP_NUM_THREADS=1)")
    print("  GPU: AMD Instinct MI300X (gfx942) via rocBLAS GEMM")
    print("=" * 120)

    # Heisenberg test cases: (L, chi, n_sweeps_gpu)
    heisenberg_cases = [
        (32,  64,  10),
        (32,  128, 10),
        (64,  64,  10),
        (64,  128, 10),
        (64,  256, 8),
    ]

    # Josephson test cases: (L, chi, n_sweeps_gpu)
    josephson_cases = [
        (12, 50,  20),
        (24, 50,  15),
        (24, 100, 10),
    ]

    all_results = []

    # ============================================================
    # Heisenberg benchmarks
    # ============================================================
    print("\n" + "=" * 120)
    print("HEISENBERG CHAIN (d=2, D_mpo=5, OBC)")
    print("=" * 120)

    header = f"{'L':>4} {'chi':>5} | {'quimb':>10} {'dmrg-gpu':>10} {'dmrg2-gpu':>10} | {'pdmrg P=2':>10} {'pdmrg P=4':>10} {'pdmrg P=8':>10} | {'best GPU':>10} {'speedup':>8}"
    print(header)
    print("-" * len(header))

    for L, chi, n_sw in heisenberg_cases:
        sys.stdout.write(f"  Heisenberg L={L} chi={chi}... ")
        sys.stdout.flush()

        row = {'model': 'Heisenberg', 'L': L, 'chi': chi}

        # quimb DMRG2 (CPU single-thread, 5min timeout)
        e_cpu, t_cpu = run_quimb(L, chi, model='heisenberg', max_sweeps=50, timeout=300)
        row['quimb_E'] = e_cpu
        row['quimb_t'] = t_cpu

        # dmrg-gpu (single-site)
        e_g1, t_g1 = run_dmrg_gpu(L, chi, n_sw)
        row['dmrg_gpu_E'] = e_g1
        row['dmrg_gpu_t'] = t_g1

        # dmrg2-gpu (two-site)
        e_g2, t_g2 = run_dmrg2_gpu(L, chi, n_sw)
        row['dmrg2_gpu_E'] = e_g2
        row['dmrg2_gpu_t'] = t_g2

        # pdmrg-gpu with P=2,4,8 segments
        pdmrg_times = {}
        for P in [2, 4, 8]:
            sites_per_seg = L // P
            if sites_per_seg < 4:
                row[f'pdmrg_P{P}_E'] = None
                row[f'pdmrg_P{P}_t'] = None
                continue
            e_p, t_p = run_pdmrg_gpu(L, chi, n_sw, P)
            row[f'pdmrg_P{P}_E'] = e_p
            row[f'pdmrg_P{P}_t'] = t_p
            if t_p is not None:
                pdmrg_times[P] = t_p

        # Find best GPU time
        gpu_times = {}
        if t_g1 is not None: gpu_times['dmrg-gpu'] = t_g1
        if t_g2 is not None: gpu_times['dmrg2-gpu'] = t_g2
        for P, t in pdmrg_times.items():
            gpu_times[f'pdmrg P={P}'] = t

        best_gpu = min(gpu_times, key=gpu_times.get) if gpu_times else None
        best_t = gpu_times[best_gpu] if best_gpu else None

        cpu_t = row.get('quimb_t')
        speedup = f"{cpu_t/best_t:.1f}x" if (cpu_t and best_t) else "N/A"

        fmt = lambda t: f"{t:.2f}s" if t else "---"
        print(f"\r{L:4d} {chi:5d} | {fmt(row.get('quimb_t')):>10} {fmt(t_g1):>10} {fmt(t_g2):>10} | "
              f"{fmt(row.get('pdmrg_P2_t')):>10} {fmt(row.get('pdmrg_P4_t')):>10} {fmt(row.get('pdmrg_P8_t')):>10} | "
              f"{(best_gpu or '---'):>10} {speedup:>8}")

        all_results.append(row)

    # ============================================================
    # Josephson benchmarks
    # ============================================================
    print("\n" + "=" * 120)
    print("JOSEPHSON JUNCTION (d=3, D_mpo=4, n_max=1, E_J=1, E_C=0.5, phi=pi/4)")
    print("=" * 120)

    header = f"{'L':>4} {'chi':>5} | {'quimb':>10} {'dmrg-gpu':>10} {'dmrg2-gpu':>10} | {'pdmrg P=2':>10} {'pdmrg P=4':>10} {'pdmrg P=8':>10} | {'best GPU':>10} {'speedup':>8}"
    print(header)
    print("-" * len(header))

    for L, chi, n_sw in josephson_cases:
        sys.stdout.write(f"  Josephson L={L} chi={chi}... ")
        sys.stdout.flush()

        row = {'model': 'Josephson', 'L': L, 'chi': chi}

        # quimb DMRG2 (CPU, 5min timeout)
        e_cpu, t_cpu = run_quimb(L, chi, model='josephson', max_sweeps=50, timeout=300)
        row['quimb_E'] = e_cpu
        row['quimb_t'] = t_cpu

        # dmrg-gpu (single-site, complex)
        e_g1, t_g1 = run_dmrg_gpu(L, chi, n_sw, '--josephson')
        row['dmrg_gpu_E'] = e_g1
        row['dmrg_gpu_t'] = t_g1

        # dmrg2-gpu (two-site, complex)
        e_g2, t_g2 = run_dmrg2_gpu(L, chi, n_sw, '--josephson')
        row['dmrg2_gpu_E'] = e_g2
        row['dmrg2_gpu_t'] = t_g2

        # pdmrg-gpu with P=2,4,8 (complex)
        pdmrg_times = {}
        for P in [2, 4, 8]:
            sites_per_seg = L // P
            if sites_per_seg < 4:
                row[f'pdmrg_P{P}_E'] = None
                row[f'pdmrg_P{P}_t'] = None
                continue
            e_p, t_p = run_pdmrg_gpu(L, chi, n_sw, P, extra_args='--josephson')
            row[f'pdmrg_P{P}_E'] = e_p
            row[f'pdmrg_P{P}_t'] = t_p
            if t_p is not None:
                pdmrg_times[P] = t_p

        gpu_times = {}
        if t_g1 is not None: gpu_times['dmrg-gpu'] = t_g1
        if t_g2 is not None: gpu_times['dmrg2-gpu'] = t_g2
        for P, t in pdmrg_times.items():
            gpu_times[f'pdmrg P={P}'] = t

        best_gpu = min(gpu_times, key=gpu_times.get) if gpu_times else None
        best_t = gpu_times[best_gpu] if best_gpu else None

        cpu_t = row.get('quimb_t')
        speedup = f"{cpu_t/best_t:.1f}x" if (cpu_t and best_t) else "N/A"

        fmt = lambda t: f"{t:.2f}s" if t else "---"
        print(f"\r{L:4d} {chi:5d} | {fmt(row.get('quimb_t')):>10} {fmt(t_g1):>10} {fmt(t_g2):>10} | "
              f"{fmt(row.get('pdmrg_P2_t')):>10} {fmt(row.get('pdmrg_P4_t')):>10} {fmt(row.get('pdmrg_P8_t')):>10} | "
              f"{(best_gpu or '---'):>10} {speedup:>8}")

        all_results.append(row)

    # ============================================================
    # Energy accuracy table
    # ============================================================
    print("\n" + "=" * 120)
    print("ENERGY ACCURACY (|E - E_quimb|)")
    print("=" * 120)
    print(f"{'Model':>12} {'L':>4} {'chi':>5} | {'E_quimb':>18} | {'dmrg-gpu':>10} {'dmrg2-gpu':>10} {'pdmrg P=2':>10} {'pdmrg P=4':>10} {'pdmrg P=8':>10}")
    print("-" * 100)

    for row in all_results:
        e_ref = row.get('quimb_E')
        def de(key):
            e = row.get(key)
            if e is not None and e_ref is not None:
                return f"{abs(e - e_ref):.1e}"
            return "---"

        e_str = f"{e_ref:.12f}" if e_ref is not None else "---"
        print(f"{row['model']:>12} {row['L']:4d} {row['chi']:5d} | {e_str:>18} | "
              f"{de('dmrg_gpu_E'):>10} {de('dmrg2_gpu_E'):>10} "
              f"{de('pdmrg_P2_E'):>10} {de('pdmrg_P4_E'):>10} {de('pdmrg_P8_E'):>10}")

    # ============================================================
    # Save results as JSON
    # ============================================================
    results_file = os.path.join(os.path.dirname(__file__), 'benchmark_all_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_file}")

    print("\nDone!")


if __name__ == '__main__':
    main()
