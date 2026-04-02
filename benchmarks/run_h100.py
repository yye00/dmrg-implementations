#!/usr/bin/env python3
"""
H100 Benchmark Script — Self-contained runner for CUDA GPU + quimb CPU benchmarks.

Runs on the H100 machine directly. Results saved to benchmarks/paper_results/h100/.

Usage:
    python3 benchmarks/run_h100.py --impl quimb-dmrg1,dmrg-gpu
    python3 benchmarks/run_h100.py --impl all
    python3 benchmarks/run_h100.py --impl quimb-dmrg1 --model heisenberg --size small
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

# ─── Configuration ───────────────────────────────────────────────────────────

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SIZES = {
    "heisenberg": {
        "small":  {"L": 12, "chi": 20,  "max_sweeps": 30},
        "medium": {"L": 20, "chi": 50,  "max_sweeps": 30},
        "large":  {"L": 40, "chi": 100, "max_sweeps": 50},
    },
    "josephson": {
        "small":  {"L": 8,  "chi": 20,  "n_max": 2, "max_sweeps": 30},
        "medium": {"L": 12, "chi": 50,  "n_max": 2, "max_sweeps": 40},
        "large":  {"L": 16, "chi": 100, "n_max": 2, "max_sweeps": 50},
    },
}

# CUDA GPU executables (relative to repo root)
GPU_IMPLS = {
    "dmrg-gpu": {
        "executable": "gpu-cuda/dmrg-gpu/build/dmrg_gpu",
        "type": "serial",
        "description": "Single-site DMRG, Lanczos + SVD (CUDA)",
    },
    "dmrg-gpu-opt": {
        "executable": "gpu-cuda/dmrg-gpu-opt/build/dmrg_gpu_opt",
        "type": "serial",
        "description": "Single-site DMRG, Block-Davidson + Newton-Schulz (CUDA)",
    },
    "dmrg2-gpu": {
        "executable": "gpu-cuda/dmrg2-gpu/build/dmrg2_gpu",
        "type": "serial",
        "description": "Two-site DMRG, Lanczos + SVD (CUDA)",
    },
    "dmrg2-gpu-opt": {
        "executable": "gpu-cuda/dmrg2-gpu-opt/build/dmrg2_gpu_opt",
        "type": "serial",
        "description": "Two-site DMRG, Block-Davidson + Newton-Schulz (CUDA)",
    },
    "pdmrg-gpu": {
        "executable": "gpu-cuda/pdmrg-gpu/build/pdmrg_gpu",
        "type": "parallel",
        "description": "Parallel two-site DMRG, Lanczos + SVD (CUDA)",
    },
    "pdmrg-gpu-opt": {
        "executable": "gpu-cuda/pdmrg-gpu-opt/build/pdmrg_gpu_opt",
        "type": "parallel",
        "description": "Parallel two-site DMRG, Block-Davidson + Newton-Schulz (CUDA)",
    },
}

CPU_IMPLS = {
    "quimb-dmrg1": {"algorithm": "dmrg1", "description": "Quimb DMRG single-site (CPU)"},
    "quimb-dmrg2": {"algorithm": "dmrg2", "description": "Quimb DMRG two-site (CPU)"},
}

ALL_IMPLS = list(CPU_IMPLS.keys()) + list(GPU_IMPLS.keys())


# ─── GPU runner ──────────────────────────────────────────────────────────────

def run_gpu(impl_name, model, L, chi, max_sweeps, n_max=2, np_count=None):
    """Run a CUDA GPU executable and parse output."""
    cfg = GPU_IMPLS[impl_name]
    exe = os.path.join(REPO_ROOT, cfg["executable"])

    if not os.path.exists(exe):
        return {"energy": None, "time": None, "success": False,
                "error": f"Executable not found: {exe}"}

    cmd = [exe, str(L), str(chi), str(max_sweeps)]
    if model == "josephson":
        cmd.append("--josephson")
    if cfg["type"] == "parallel" and np_count and np_count > 1:
        cmd.extend(["--segments", str(np_count)])
        cmd.extend(["--warmup", "3"])
        cmd.extend(["--local-sweeps", "2"])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            return {"energy": None, "time": None, "success": False,
                    "error": (result.stderr or f"Exit code {result.returncode}")[-500:]}

        energy, wall_time = None, None
        for line in result.stdout.split("\n"):
            m = re.search(r"(?:Final|Ground state)\s+energy:\s+([-\d.eE+]+)", line)
            if m:
                energy = float(m.group(1))
            m = re.search(r"(?:Total\s+)?[Ww]all\s+time:\s+([-\d.eE+]+)", line)
            if m:
                wall_time = float(m.group(1))

        if energy is None:
            return {"energy": None, "time": None, "success": False,
                    "error": f"No energy in output: {result.stdout[-300:]}"}

        return {"energy": energy, "time": wall_time, "success": True}

    except subprocess.TimeoutExpired:
        return {"energy": None, "time": None, "success": False, "error": "Timeout (600s)"}


# ─── CPU (quimb) runner ─────────────────────────────────────────────────────

def run_quimb(impl_name, model, L, chi, max_sweeps, tol=1e-11, n_max=2, threads=1):
    """Run quimb DMRG in-process."""
    # Set thread counts before importing numpy/quimb
    thread_str = str(threads)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = thread_str

    try:
        import numpy as np
        import quimb.tensor as qtn
    except ImportError as e:
        return {"energy": None, "time": None, "success": False,
                "error": f"Import error: {e}"}

    algorithm = CPU_IMPLS[impl_name]["algorithm"]

    # Build MPO
    if model == "heisenberg":
        mpo = qtn.MPO_ham_heis(L=L, j=1.0, bz=0.0, cyclic=False)
    elif model == "josephson":
        mpo = build_josephson_mpo_quimb(L, n_max=n_max)
    else:
        return {"energy": None, "time": None, "success": False,
                "error": f"Unknown model: {model}"}

    dmrg_class = qtn.DMRG2 if algorithm == "dmrg2" else qtn.DMRG1
    dmrg = dmrg_class(mpo, bond_dims=chi, cutoffs=1e-14)

    t0 = time.perf_counter()
    dmrg.solve(max_sweeps=max_sweeps, tol=tol, verbosity=0)
    elapsed = time.perf_counter() - t0

    return {"energy": float(np.real(dmrg.energy)), "time": elapsed, "success": True}


def build_josephson_mpo_quimb(L, n_max=2):
    """Build Josephson junction array MPO for quimb."""
    import numpy as np
    import quimb.tensor as qtn

    d = 2 * n_max + 1
    charges = np.arange(-n_max, n_max + 1)

    # Operators
    n_op = np.diag(charges).astype(float)
    n2_op = np.diag(charges ** 2).astype(float)
    cos_op = 0.5 * (np.diag(np.ones(d - 1), 1) + np.diag(np.ones(d - 1), -1))

    # Parameters
    Ec = 1.0
    Ej = 1.0

    # Build as MPO
    # H = Ec * sum_i n_i^2 - Ej * sum_<ij> cos(phi_i - phi_j)
    # cos(phi_i - phi_j) = cos(phi_i)cos(phi_j) + sin(phi_i)sin(phi_j)
    # Using number-phase: cos(phi_i - phi_j) ≈ (raise_i lower_j + lower_i raise_j)/2

    # W-matrix dimension: 4 (I, cos, cos†, H_local)
    W_dim = 4
    I_d = np.eye(d)

    # Build site tensors
    arrays = []
    for i in range(L):
        W = np.zeros((W_dim, W_dim, d, d))
        # Row 0 (left boundary feeds)
        W[0, 0] = I_d                       # I ⊗ I
        W[0, 1] = cos_op                    # Start cos pair
        W[0, 3] = Ec * n2_op               # On-site charging energy
        # Row 1 (continue cos pair)
        W[1, 3] = -Ej * cos_op             # Complete cos pair: -Ej * cos_i cos_j
        # Row 3 (right boundary collects)
        W[3, 3] = I_d                       # I ⊗ I

        if i == 0:
            arr = W[0:1, :, :, :]           # (1, W_dim, d, d)
        elif i == L - 1:
            arr = W[:, 3:4, :, :]           # (W_dim, 1, d, d)
        else:
            arr = W                          # (W_dim, W_dim, d, d)

        # quimb expects (left_bond, right_bond, phys_bra, phys_ket) = (bl, br, k, b)
        arrays.append(arr)

    return qtn.MatrixProductOperator(arrays)


# ─── Main benchmark loop ────────────────────────────────────────────────────

def run_benchmark(impl_names, models, sizes, np_values=None, thread_values=None):
    """Run benchmarks and return results list."""
    if np_values is None:
        np_values = [2, 4]
    if thread_values is None:
        thread_values = [1]

    results = []

    for model in models:
        for size_name in sizes:
            params = SIZES[model][size_name]
            L, chi = params["L"], params["chi"]
            max_sweeps = params["max_sweeps"]
            n_max = params.get("n_max", 2)

            print(f"\n{'='*60}")
            print(f"  {model.upper()} {size_name} — L={L}, chi={chi}")
            print(f"{'='*60}")

            for impl_name in impl_names:
                is_gpu = impl_name in GPU_IMPLS
                is_parallel = is_gpu and GPU_IMPLS[impl_name]["type"] == "parallel"

                if is_parallel:
                    configs = [(np_val, 1) for np_val in np_values]
                elif is_gpu:
                    configs = [(1, 1)]
                else:
                    configs = [(1, t) for t in thread_values]

                for np_val, threads in configs:
                    label = impl_name
                    if is_parallel:
                        label += f" (np={np_val})"
                    elif threads > 1:
                        label += f" (t={threads})"

                    print(f"  {label:40s}", end="", flush=True)

                    if is_gpu:
                        r = run_gpu(impl_name, model, L, chi, max_sweeps,
                                    n_max=n_max, np_count=np_val)
                    else:
                        r = run_quimb(impl_name, model, L, chi, max_sweeps,
                                      n_max=n_max, threads=threads)

                    r["impl"] = impl_name
                    r["model"] = model
                    r["size"] = size_name
                    r["L"] = L
                    r["chi"] = chi
                    r["np"] = np_val
                    r["threads"] = threads
                    r["arch"] = "h100"

                    if r["success"]:
                        t_str = f"{r['time']:.2f}s" if r.get("time") else "?"
                        print(f"  E={r['energy']:.10f}  t={t_str}")
                    else:
                        print(f"  FAIL: {r.get('error', '?')[:50]}")

                    results.append(r)

    return results


def save_results(results, output_dir):
    """Save results as JSON files in the output directory."""
    os.makedirs(output_dir, exist_ok=True)

    # Group by implementation
    by_impl = {}
    for r in results:
        impl = r["impl"]
        if impl not in by_impl:
            by_impl[impl] = []
        by_impl[impl].append(r)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    for impl, impl_results in by_impl.items():
        fname = f"{impl}_h100_{timestamp}.json"
        path = os.path.join(output_dir, fname)
        with open(path, "w") as f:
            json.dump({
                "implementation": impl,
                "architecture": "h100",
                "timestamp": timestamp,
                "results": impl_results,
            }, f, indent=2)
        print(f"  Saved: {path}")

    # Also save combined results
    combined_path = os.path.join(output_dir, f"combined_h100_{timestamp}.json")
    with open(combined_path, "w") as f:
        json.dump({
            "architecture": "h100",
            "timestamp": timestamp,
            "results": results,
        }, f, indent=2)
    print(f"  Saved: {combined_path}")


def main():
    parser = argparse.ArgumentParser(description="H100 DMRG Benchmarks")
    parser.add_argument("--impl", type=str, default=None,
                        help="Comma-separated impls (or 'all', 'gpu', 'cpu')")
    parser.add_argument("--model", type=str, default=None,
                        help="heisenberg, josephson, or both (default: both)")
    parser.add_argument("--size", type=str, default=None,
                        help="small, medium, large (comma-separated, default: all)")
    parser.add_argument("--np", type=str, default="2,4", dest="np_values",
                        help="np values for parallel impls (default: 2,4)")
    parser.add_argument("--threads", type=str, default="1", dest="thread_values",
                        help="Thread counts for CPU impls (default: 1)")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(REPO_ROOT, "benchmarks", "paper_results", "h100"),
                        help="Output directory (default: benchmarks/paper_results/h100/)")
    args = parser.parse_args()

    # Parse impl list
    if args.impl is None or args.impl == "all":
        impl_names = ALL_IMPLS
    elif args.impl == "gpu":
        impl_names = list(GPU_IMPLS.keys())
    elif args.impl == "cpu":
        impl_names = list(CPU_IMPLS.keys())
    else:
        impl_names = [x.strip() for x in args.impl.split(",")]

    # Validate impl names
    for name in impl_names:
        if name not in GPU_IMPLS and name not in CPU_IMPLS:
            print(f"Error: Unknown implementation '{name}'")
            print(f"Available: {', '.join(ALL_IMPLS)}")
            sys.exit(1)

    models = ["heisenberg", "josephson"]
    if args.model:
        models = [x.strip() for x in args.model.split(",")]

    sizes = ["small", "medium", "large"]
    if args.size:
        sizes = [x.strip() for x in args.size.split(",")]

    np_values = [int(x) for x in args.np_values.split(",")]
    thread_values = [int(x) for x in args.thread_values.split(",")]

    print(f"\n{'#'*60}")
    print(f"  H100 DMRG Benchmark Suite")
    print(f"  Implementations: {', '.join(impl_names)}")
    print(f"  Models: {', '.join(models)}")
    print(f"  Sizes: {', '.join(sizes)}")
    print(f"  Output: {args.output_dir}")
    print(f"{'#'*60}")

    results = run_benchmark(impl_names, models, sizes, np_values, thread_values)

    print(f"\n{'='*60}")
    print(f"  Saving results...")
    print(f"{'='*60}")
    save_results(results, args.output_dir)

    # Summary
    successes = sum(1 for r in results if r["success"])
    failures = sum(1 for r in results if not r["success"])
    print(f"\n  Total: {len(results)} runs, {successes} passed, {failures} failed")


if __name__ == "__main__":
    main()
