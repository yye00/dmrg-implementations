#!/usr/bin/env python3
"""
MI300X Challenge Benchmarks — Realistic quantum computing scale.

44 configs across 3 models, targeting chi=64-512 and L=16-500.
GPU warmup before each timed run. All 6 ROCm GPU implementations + quimb CPU.

Usage:
    python3 benchmarks/run_mi300x_challenge.py                    # all impls, all configs
    python3 benchmarks/run_mi300x_challenge.py --impl dmrg-gpu    # single impl
    python3 benchmarks/run_mi300x_challenge.py --model heisenberg # single model
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─── Challenge Size Configurations ───────────────────────────────────────────

CHALLENGE_SIZES = {
    "heisenberg": [
        # L,   chi, sweeps
        (50,   64,  20),
        (50,  128,  20),
        (50,  256,  15),
        (50,  512,  10),
        (100,  64,  20),
        (100, 128,  20),
        (100, 256,  15),
        (100, 512,  10),
        (200,  64,  15),
        (200, 128,  15),
        (200, 256,  10),
        (200, 512,   8),
        (500, 128,  10),
        (500, 256,   8),
    ],
    "josephson": [
        (16,   64,  20),
        (16,  128,  20),
        (16,  256,  15),
        (32,   64,  20),
        (32,  128,  20),
        (32,  256,  15),
        (48,   64,  20),
        (48,  128,  15),
        (48,  256,  10),
        (64,   64,  20),
        (64,  128,  15),
        (64,  256,  10),
        (100,  64,  15),
        (100, 128,  10),
        (100, 256,   8),
    ],
    "tfim": [
        (50,   64,  20),
        (50,  128,  20),
        (50,  256,  15),
        (50,  512,  10),
        (100,  64,  20),
        (100, 128,  20),
        (100, 256,  15),
        (100, 512,  10),
        (200,  64,  20),
        (200, 128,  15),
        (200, 256,  10),
        (200, 512,   8),
        (500, 128,  10),
        (500, 256,   8),
        (500, 512,   6),
    ],
}

# ─── GPU executables ─────────────────────────────────────────────────────────

GPU_IMPLS = {
    "dmrg-gpu":          "gpu-rocm/dmrg-gpu/build/dmrg_gpu",
    "dmrg-gpu-opt":      "gpu-rocm/dmrg-gpu-opt/build/dmrg_gpu_opt",
    "dmrg2-gpu":         "gpu-rocm/dmrg2-gpu/build/dmrg2_gpu",
    "dmrg2-gpu-opt":     "gpu-rocm/dmrg2-gpu-opt/build/dmrg2_gpu_opt",
    "pdmrg-gpu":         "gpu-rocm/pdmrg-gpu/build/pdmrg_gpu",
    "pdmrg-gpu-opt":     "gpu-rocm/pdmrg-gpu-opt/build/pdmrg_gpu_opt",
    "pdmrg-multi-gpu":   "gpu-rocm/pdmrg-multi-gpu/build/pdmrg_multi_gpu",
}

CPU_IMPLS = {"quimb-dmrg1", "quimb-dmrg2"}

ALL_IMPLS = sorted(GPU_IMPLS.keys()) + sorted(CPU_IMPLS)

# Josephson parameters (must match GPU defaults)
JOSEPHSON_NMAX = 2


# ─── GPU runner ──────────────────────────────────────────────────────────────

def _parse_gpu_output(stdout):
    energy, wall_time = None, None
    for line in stdout.split("\n"):
        m = re.search(r"(?:Final|Ground state)\s+energy:\s+([-\d.eE+]+)", line)
        if m:
            energy = float(m.group(1))
        m = re.search(r"(?:Total\s+)?[Ww]all\s+time:\s+([-\d.eE+]+)", line)
        if m:
            wall_time = float(m.group(1))
    return energy, wall_time


def _build_gpu_cmd(impl, model, L, chi, sweeps):
    exe = os.path.join(REPO_ROOT, GPU_IMPLS[impl])
    if not os.path.exists(exe):
        return None, f"Not found: {exe}"
    cmd = [exe, str(L), str(chi), str(sweeps)]
    if model == "josephson":
        cmd += ["--josephson", "--nmax", str(JOSEPHSON_NMAX)]
    elif model == "tfim":
        cmd += ["--tfim"]
    if impl == "pdmrg-multi-gpu":
        cmd += ["--devices", "4"]
    return cmd, None


def run_gpu(impl, model, L, chi, sweeps, warmup=True, timeout=None):
    """Run GPU benchmark with optional warmup."""
    # Warmup: same exe, tiny problem
    if warmup:
        warmup_cmd, _ = _build_gpu_cmd(impl, model, 4, 4, 2)
        if warmup_cmd:
            try:
                subprocess.run(warmup_cmd, capture_output=True, text=True, timeout=30)
            except Exception:
                pass

    cmd, err = _build_gpu_cmd(impl, model, L, chi, sweeps)
    if err:
        return {"energy": None, "time": None, "success": False, "error": err}

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            return {"energy": None, "time": None, "success": False,
                    "error": (result.stderr or f"rc={result.returncode}")[-500:]}
        energy, wt = _parse_gpu_output(result.stdout)
        if energy is None:
            return {"energy": None, "time": None, "success": False,
                    "error": f"No energy: {result.stdout[-300:]}"}
        return {"energy": energy, "time": wt, "success": True}
    except subprocess.TimeoutExpired:
        return {"energy": None, "time": None, "success": False, "error": f"Timeout ({timeout}s)"}


# ─── CPU (quimb) runner ─────────────────────────────────────────────────────

def run_quimb(impl, model, L, chi, sweeps, threads=1, tol=1e-11):
    thread_str = str(threads)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = thread_str

    try:
        import numpy as np
        import quimb.tensor as qtn
    except ImportError as e:
        return {"energy": None, "time": None, "success": False, "error": str(e)}

    algorithm = "dmrg2" if "dmrg2" in impl else "dmrg1"

    try:
        if model == "heisenberg":
            mpo = qtn.MPO_ham_heis(L=L, j=1.0, bz=0.0, cyclic=False)
        elif model == "josephson":
            mpo = _build_josephson_mpo(L, JOSEPHSON_NMAX, qtn, np)
        elif model == "tfim":
            mpo = _build_tfim_mpo(L, qtn)
        else:
            return {"energy": None, "time": None, "success": False, "error": f"Unknown: {model}"}

        dmrg_cls = qtn.DMRG2 if algorithm == "dmrg2" else qtn.DMRG1
        dmrg = dmrg_cls(mpo, bond_dims=chi, cutoffs=1e-14)

        t0 = time.perf_counter()
        dmrg.solve(max_sweeps=sweeps, tol=tol, verbosity=0)
        elapsed = time.perf_counter() - t0

        return {"energy": float(np.real(dmrg.energy)), "time": elapsed, "success": True}
    except Exception as e:
        return {"energy": None, "time": None, "success": False, "error": str(e)[:200]}


def _build_josephson_mpo(L, n_max, qtn, np):
    d = 2 * n_max + 1
    charges = np.arange(-n_max, n_max + 1, dtype=np.float64)
    n_op = np.diag(charges.astype(np.complex128))
    exp_iphi = np.zeros((d, d), dtype=np.complex128)
    for i in range(d - 1):
        exp_iphi[i + 1, i] = 1.0 + 0j
    exp_miphi = exp_iphi.conj().T
    E_J, E_C = 1.0, 0.5
    phi_ext = np.pi / 4
    flux_phase = np.exp(1j * phi_ext)
    S = (d - 1) / 2
    builder = qtn.SpinHam1D(S=S)
    builder.add_term(-E_J / 2 * flux_phase, exp_iphi, exp_miphi)
    builder.add_term(-E_J / 2 * np.conj(flux_phase), exp_miphi, exp_iphi)
    builder.add_term(E_C, n_op @ n_op)
    return builder.build_mpo(L)


def _build_tfim_mpo(L, qtn):
    builder = qtn.SpinHam1D(S=0.5)
    builder.add_term(-4.0, 'Z', 'Z')
    builder.add_term(-2.0, 'X')
    return builder.build_mpo(L)


# ─── Main benchmark loop ────────────────────────────────────────────────────

def run_all(impl_names, models, do_warmup=True):
    results = []

    # Build flat config list
    configs = []
    for impl in impl_names:
        for model in models:
            if model not in CHALLENGE_SIZES:
                continue
            for L, chi, sweeps in CHALLENGE_SIZES[model]:
                configs.append((impl, model, L, chi, sweeps))

    total = len(configs)
    current_group = None

    for i, (impl, model, L, chi, sweeps) in enumerate(configs):
        group = (impl, model)
        if group != current_group:
            current_group = group
            print(f"\n{'='*65}")
            print(f"  {impl} — {model}")
            print(f"{'='*65}")

        label = f"L={L:3d} chi={chi:3d} sw={sweeps:2d}"
        print(f"  [{i+1:3d}/{total}] {label:25s}", end="", flush=True)

        is_gpu = impl in GPU_IMPLS
        if is_gpu:
            r = run_gpu(impl, model, L, chi, sweeps, warmup=do_warmup)
        else:
            r = run_quimb(impl, model, L, chi, sweeps, threads=1)

        r.update({
            "impl": impl, "model": model, "L": L, "chi": chi,
            "sweeps": sweeps, "arch": "mi300x", "benchmark": "challenge",
            "threads": 1, "np": 1,
        })
        if model == "josephson":
            r["nmax"] = JOSEPHSON_NMAX

        if r["success"]:
            t_str = f"{r['time']:.3f}s" if r.get("time") else "?"
            print(f"  E={r['energy']:.10f}  t={t_str}")
        else:
            print(f"  FAIL: {r.get('error', '?')[:50]}")

        results.append(r)

    return results


def save_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    by_impl = {}
    for r in results:
        by_impl.setdefault(r["impl"], []).append(r)

    for impl, impl_results in by_impl.items():
        path = os.path.join(output_dir, f"{impl}_mi300x_challenge_{ts}.json")
        with open(path, "w") as f:
            json.dump({"implementation": impl, "architecture": "mi300x",
                        "benchmark": "challenge", "timestamp": ts,
                        "results": impl_results}, f, indent=2)
        print(f"  Saved: {path}")

    path = os.path.join(output_dir, f"challenge_mi300x_{ts}.json")
    with open(path, "w") as f:
        json.dump({"architecture": "mi300x", "benchmark": "challenge",
                    "timestamp": ts, "results": results}, f, indent=2)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="MI300X Challenge Benchmarks")
    parser.add_argument("--impl", type=str, default=None,
                        help="Comma-separated impls (default: all 6 GPU + quimb)")
    parser.add_argument("--model", type=str, default=None,
                        help="heisenberg, josephson, tfim (default: all)")
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(REPO_ROOT, "benchmarks", "paper_results", "mi300x", "challenge"))
    args = parser.parse_args()

    if args.impl:
        impl_names = [x.strip() for x in args.impl.split(",")]
    else:
        # All 6 GPU impls + quimb-dmrg1 (single-thread CPU reference)
        impl_names = sorted(GPU_IMPLS.keys()) + ["quimb-dmrg1"]

    models = list(CHALLENGE_SIZES.keys())
    if args.model:
        models = [x.strip() for x in args.model.split(",")]

    # Count configs
    n_configs = sum(len(CHALLENGE_SIZES[m]) for m in models) * len(impl_names)

    print(f"\n{'#'*65}")
    print(f"  MI300X Challenge Benchmarks — Quantum Computing Scale")
    print(f"  Implementations: {', '.join(impl_names)}")
    print(f"  Models: {', '.join(models)}")
    print(f"  Total configs: {n_configs}")
    print(f"  GPU warmup: {'OFF' if args.skip_warmup else 'ON'}")
    print(f"  Output: {args.output_dir}")
    print(f"{'#'*65}")

    results = run_all(impl_names, models, do_warmup=not args.skip_warmup)

    print(f"\n{'='*65}")
    print(f"  Saving results...")
    print(f"{'='*65}")
    save_results(results, args.output_dir)

    ok = sum(1 for r in results if r["success"])
    fail = len(results) - ok
    print(f"\n  Done: {len(results)} runs, {ok} passed, {fail} failed")


if __name__ == "__main__":
    main()
