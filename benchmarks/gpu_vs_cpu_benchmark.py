#!/usr/bin/env python3
"""
Benchmark: GPU DMRG (single-site & two-site) vs single-thread CPU DMRG (quimb)

Runs quimb DMRG1/DMRG2 locally with OMP_NUM_THREADS=1,
then SSHes to remote MI300X to run dmrg_gpu and dmrg2_gpu.
"""

import os
import sys
import time
import subprocess
import numpy as np

# Force single-thread for CPU baseline
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

import quimb.tensor as qtn
import quimb as qu

REMOTE = "hotaisle@23.183.40.79"
REMOTE_DIR = "/home/hotaisle/dmrg-implementations"

# ============================================================================
# Josephson MPO builder for quimb
# ============================================================================
def build_josephson_mpo_quimb(L, n_max=1, E_J=1.0, E_C=0.5, n_g=0.0, phi_ext=np.pi/4):
    """Build Josephson junction chain as explicit quimb MPO."""
    d = 2 * n_max + 1
    D = 4  # MPO bond dim

    eye = np.eye(d, dtype=complex)
    exp_iphi = np.zeros((d, d), dtype=complex)
    exp_miphi = np.zeros((d, d), dtype=complex)
    H_onsite = np.zeros((d, d), dtype=complex)

    for i in range(d):
        charge = i - n_max
        H_onsite[i, i] = E_C * charge**2 - n_g * charge
    for i in range(d - 1):
        exp_iphi[i + 1, i] = 1.0
        exp_miphi[i, i + 1] = 1.0

    alpha = -E_J / 2.0 * np.exp(1j * phi_ext)
    alpha_conj = np.conj(alpha)

    # Transfer matrix (D=4, upper triangular):
    #   Row 0: [I,  a*phi+,  a'*phi-,  H_onsite]
    #   Row 1: [0,  0,       0,        phi-    ]
    #   Row 2: [0,  0,       0,        phi+    ]
    #   Row 3: [0,  0,       0,        I       ]
    def build_W(site):
        W = np.zeros((D, D, d, d), dtype=complex)  # (D_left, D_right, ket, bra)
        if site == 0:
            W[0, 0] = eye
            W[0, 1] = alpha * exp_iphi
            W[0, 2] = alpha_conj * exp_miphi
            W[0, 3] = H_onsite
        elif site == L - 1:
            W[0, 3] = H_onsite
            W[1, 3] = exp_miphi
            W[2, 3] = exp_iphi
            W[3, 3] = eye
        else:
            W[0, 0] = eye
            W[0, 1] = alpha * exp_iphi
            W[0, 2] = alpha_conj * exp_miphi
            W[0, 3] = H_onsite
            W[1, 3] = exp_miphi
            W[2, 3] = exp_iphi
            W[3, 3] = eye
        return W

    arrays = []
    for site in range(L):
        W = build_W(site)
        if site == 0:
            arrays.append(W[0, :, :, :])   # (D_right, d, d)
        elif site == L - 1:
            arrays.append(W[:, D-1, :, :])  # (D_left, d, d)
        else:
            arrays.append(W)
    return qtn.MatrixProductOperator(arrays)


# ============================================================================
# CPU benchmark runners
# ============================================================================
def run_quimb_heisenberg(L, chi_max, max_sweeps=30, method='DMRG2'):
    """Run quimb DMRG on Heisenberg chain, return (energy, time_seconds, n_sweeps)."""
    mpo = qtn.MPO_ham_heis(L=L, j=1.0, bz=0.0, cyclic=False)

    if method == 'DMRG2':
        dmrg = qtn.DMRG2(mpo, bond_dims=chi_max, cutoffs=1e-14)
    else:
        dmrg = qtn.DMRG1(mpo, bond_dims=chi_max, cutoffs=1e-14)

    t0 = time.perf_counter()
    dmrg.solve(max_sweeps=max_sweeps, tol=1e-12, verbosity=0)
    t1 = time.perf_counter()

    return dmrg.energy, t1 - t0


def run_quimb_josephson(L, chi_max, max_sweeps=30, method='DMRG2',
                        n_max=1, E_J=1.0, E_C=0.5, phi_ext=np.pi/4):
    """Run quimb DMRG on Josephson junction chain."""
    mpo = build_josephson_mpo_quimb(L, n_max=n_max, E_J=E_J, E_C=E_C, phi_ext=phi_ext)

    if method == 'DMRG2':
        dmrg = qtn.DMRG2(mpo, bond_dims=chi_max, cutoffs=1e-14)
    else:
        dmrg = qtn.DMRG1(mpo, bond_dims=chi_max, cutoffs=1e-14)

    t0 = time.perf_counter()
    dmrg.solve(max_sweeps=max_sweeps, tol=1e-12, verbosity=0)
    t1 = time.perf_counter()

    energy = np.real(dmrg.energy) if np.iscomplex(dmrg.energy) else dmrg.energy
    return float(energy), t1 - t0


# ============================================================================
# GPU benchmark runner (via SSH)
# ============================================================================
def run_gpu_dmrg(exe_dir, exe_name, L, chi_max, n_sweeps, extra_args=""):
    """Run GPU DMRG on remote machine, parse energy and time."""
    cmd = f"ssh {REMOTE} 'cd {REMOTE_DIR}/{exe_dir}/build && ./{exe_name} {L} {chi_max} {n_sweeps} {extra_args}'"
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return None, None, output

    # Parse energy and time from output
    energy = None
    wall_time = None
    for line in output.split('\n'):
        if 'Final energy:' in line:
            energy = float(line.split(':')[1].strip())
        if 'Total wall time:' in line:
            wall_time = float(line.split(':')[1].strip().replace('s', '').strip())

    return energy, wall_time, output


# ============================================================================
# Main benchmark
# ============================================================================
def main():
    print("=" * 80)
    print("GPU vs Single-Thread CPU DMRG Benchmark")
    print("=" * 80)
    print(f"CPU: quimb DMRG (OMP_NUM_THREADS=1)")
    print(f"GPU: MI300X (gfx942) via rocBLAS GEMM")
    print()

    # Define test cases: (L, chi_max, n_sweeps_gpu)
    # Keep CPU cases manageable (<60s each at single-thread)
    heisenberg_cases = [
        (8, 20, 20),
        (16, 20, 20),
        (16, 50, 20),
        (32, 50, 20),
        (32, 100, 15),
        (64, 100, 15),
        (64, 200, 10),
    ]

    josephson_cases = [
        (6, 20, 20),
        (12, 20, 20),
        (12, 50, 20),
        (24, 50, 20),
        (24, 100, 15),
    ]

    results = []

    # ============================================================
    # Heisenberg benchmarks
    # ============================================================
    print("\n" + "=" * 80)
    print("HEISENBERG CHAIN (d=2, D_mpo=5, OBC)")
    print("=" * 80)
    print(f"{'L':>4} {'chi':>5} | {'quimb DMRG1':>14} {'quimb DMRG2':>14} | {'GPU 1-site':>14} {'GPU 2-site':>14} | {'Speedup(1s)':>12} {'Speedup(2s)':>12}")
    print("-" * 110)

    for L, chi, n_sw in heisenberg_cases:
        row = {'model': 'Heisenberg', 'L': L, 'chi': chi}
        sys.stdout.write(f"  Running Heisenberg L={L} chi={chi}...")
        sys.stdout.flush()

        # CPU: quimb DMRG2 (standard baseline)
        try:
            e2, t2 = run_quimb_heisenberg(L, chi, max_sweeps=50, method='DMRG2')
            row['cpu_dmrg2_E'] = e2
            row['cpu_dmrg2_t'] = t2
        except Exception as ex:
            row['cpu_dmrg2_E'] = None
            row['cpu_dmrg2_t'] = None

        # CPU: quimb DMRG1 (skip for large cases)
        if L * chi <= 3200:
            try:
                e1, t1 = run_quimb_heisenberg(L, chi, max_sweeps=50, method='DMRG1')
                row['cpu_dmrg1_E'] = e1
                row['cpu_dmrg1_t'] = t1
            except Exception as ex:
                row['cpu_dmrg1_E'] = None
                row['cpu_dmrg1_t'] = None
        else:
            row['cpu_dmrg1_E'] = None
            row['cpu_dmrg1_t'] = None

        # GPU: single-site
        eg1, tg1, _ = run_gpu_dmrg('dmrg-gpu', 'dmrg_gpu', L, chi, n_sw)
        row['gpu_1site_E'] = eg1
        row['gpu_1site_t'] = tg1

        # GPU: two-site
        eg2, tg2, _ = run_gpu_dmrg('dmrg2-gpu', 'dmrg2_gpu', L, chi, n_sw)
        row['gpu_2site_E'] = eg2
        row['gpu_2site_t'] = tg2

        # Compute speedups vs quimb DMRG2 (standard baseline)
        cpu_ref = row.get('cpu_dmrg2_t')
        sp1 = f"{cpu_ref/tg1:.1f}x" if (cpu_ref and tg1) else "N/A"
        sp2 = f"{cpu_ref/tg2:.1f}x" if (cpu_ref and tg2) else "N/A"

        fmt_t = lambda t: f"{t:.3f}s" if t else "N/A"

        print(f"\r{L:4d} {chi:5d} | {fmt_t(row.get('cpu_dmrg1_t')):>14} {fmt_t(row.get('cpu_dmrg2_t')):>14} | {fmt_t(tg1):>14} {fmt_t(tg2):>14} | {sp1:>12} {sp2:>12}")

        results.append(row)

    # ============================================================
    # Josephson benchmarks
    # ============================================================
    print("\n" + "=" * 80)
    print("JOSEPHSON JUNCTION (d=3, D_mpo=4, n_max=1, E_J=1, E_C=0.5, phi=pi/4)")
    print("=" * 80)
    print(f"{'L':>4} {'chi':>5} | {'quimb DMRG1':>14} {'quimb DMRG2':>14} | {'GPU 1-site':>14} {'GPU 2-site':>14} | {'Speedup(1s)':>12} {'Speedup(2s)':>12}")
    print("-" * 110)

    for L, chi, n_sw in josephson_cases:
        row = {'model': 'Josephson', 'L': L, 'chi': chi}
        sys.stdout.write(f"  Running Josephson L={L} chi={chi}...")
        sys.stdout.flush()

        # CPU: quimb DMRG2 (standard baseline)
        try:
            e2, t2 = run_quimb_josephson(L, chi, max_sweeps=50, method='DMRG2')
            row['cpu_dmrg2_E'] = e2
            row['cpu_dmrg2_t'] = t2
        except Exception as ex:
            row['cpu_dmrg2_E'] = None
            row['cpu_dmrg2_t'] = None

        # CPU: quimb DMRG1 (skip for large cases)
        if L * chi <= 1200:
            try:
                e1, t1 = run_quimb_josephson(L, chi, max_sweeps=50, method='DMRG1')
                row['cpu_dmrg1_E'] = e1
                row['cpu_dmrg1_t'] = t1
            except Exception as ex:
                row['cpu_dmrg1_E'] = None
                row['cpu_dmrg1_t'] = None
        else:
            row['cpu_dmrg1_E'] = None
            row['cpu_dmrg1_t'] = None

        # GPU: single-site (complex)
        eg1, tg1, _ = run_gpu_dmrg('dmrg-gpu', 'dmrg_gpu', L, chi, n_sw, '--josephson')
        row['gpu_1site_E'] = eg1
        row['gpu_1site_t'] = tg1

        # GPU: two-site (complex)
        eg2, tg2, _ = run_gpu_dmrg('dmrg2-gpu', 'dmrg2_gpu', L, chi, n_sw, '--josephson')
        row['gpu_2site_E'] = eg2
        row['gpu_2site_t'] = tg2

        cpu_ref = row.get('cpu_dmrg2_t')
        sp1 = f"{cpu_ref/tg1:.1f}x" if (cpu_ref and tg1) else "N/A"
        sp2 = f"{cpu_ref/tg2:.1f}x" if (cpu_ref and tg2) else "N/A"

        fmt_t2 = lambda t: f"{t:.3f}s" if t else "N/A"

        print(f"\r{L:4d} {chi:5d} | {fmt_t2(row.get('cpu_dmrg1_t')):>14} {fmt_t2(row.get('cpu_dmrg2_t')):>14} | {fmt_t2(tg1):>14} {fmt_t2(tg2):>14} | {sp1:>12} {sp2:>12}")

        results.append(row)

    # ============================================================
    # Energy comparison table
    # ============================================================
    print("\n" + "=" * 80)
    print("ENERGY COMPARISON")
    print("=" * 80)
    print(f"{'Model':>12} {'L':>4} {'chi':>5} | {'quimb DMRG2':>18} {'GPU 1-site':>18} {'GPU 2-site':>18} | {'|dE| 1s':>10} {'|dE| 2s':>10}")
    print("-" * 110)
    for row in results:
        e_ref = row.get('cpu_dmrg2_E')
        e_g1 = row.get('gpu_1site_E')
        e_g2 = row.get('gpu_2site_E')
        de1 = f"{abs(e_ref - e_g1):.1e}" if (e_ref is not None and e_g1 is not None) else "N/A"
        de2 = f"{abs(e_ref - e_g2):.1e}" if (e_ref is not None and e_g2 is not None) else "N/A"
        def fmt_e(e):
            return f"{e:.12f}" if e is not None else "N/A"
        print(f"{row['model']:>12} {row['L']:4d} {row['chi']:5d} | {fmt_e(e_ref):>18} {fmt_e(e_g1):>18} {fmt_e(e_g2):>18} | {de1:>10} {de2:>10}")

    print("\nDone!")


if __name__ == '__main__':
    main()
