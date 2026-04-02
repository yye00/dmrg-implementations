#!/usr/bin/env python3
"""
H100 Benchmark Script — Reproduces MI300X benchmark configs on NVIDIA H100.

Reads configs from MI300X results.json and runs the same (impl, model, L, chi,
sweeps, threads, np, nmax) tuples using CUDA GPU executables and quimb CPU.

GPU runs include a warmup run (discarded) before the timed run to eliminate
CUDA JIT and context initialization overhead.

Usage:
    python3 benchmarks/run_h100.py                          # all reproducible configs
    python3 benchmarks/run_h100.py --impl dmrg-gpu          # single impl
    python3 benchmarks/run_h100.py --impl quimb-dmrg1,dmrg-gpu --model heisenberg
    python3 benchmarks/run_h100.py --skip-warmup             # no warmup (faster, less accurate)
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

# ─── Paths ───────────────────────────────────────────────────────────────────

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MI300X_RESULTS = os.path.join(REPO_ROOT, "benchmarks", "paper_results", "mi300x", "results.json")

# ─── CUDA GPU executables (relative to repo root) ───────────────────────────

GPU_EXECUTABLES = {
    "dmrg-gpu":      "gpu-cuda/dmrg-gpu/build/dmrg_gpu",
    "dmrg-gpu-opt":  "gpu-cuda/dmrg-gpu-opt/build/dmrg_gpu_opt",
    "dmrg2-gpu":     "gpu-cuda/dmrg2-gpu/build/dmrg2_gpu",
    "dmrg2-gpu-opt": "gpu-cuda/dmrg2-gpu-opt/build/dmrg2_gpu_opt",
    "pdmrg-gpu":     "gpu-cuda/pdmrg-gpu/build/pdmrg_gpu",
    "pdmrg-gpu-opt": "gpu-cuda/pdmrg-gpu-opt/build/pdmrg_gpu_opt",
}

# Implementations we can run on H100 (no MPI-based CPU pdmrg)
RUNNABLE = set(GPU_EXECUTABLES.keys()) | {"quimb-dmrg1", "quimb-dmrg2"}


# ─── Load MI300X configs ────────────────────────────────────────────────────

def load_mi300x_configs(impl_filter=None, model_filter=None):
    """Load benchmark configs from MI300X results.json.

    Returns list of dicts with keys: impl, model, L, chi, sweeps, threads, nmax, np
    """
    with open(MI300X_RESULTS) as f:
        data = json.load(f)

    configs = []
    seen = set()
    for r in data:
        impl = r["impl"]
        if impl not in RUNNABLE:
            continue
        if impl_filter and impl not in impl_filter:
            continue

        model = r["model"]
        if model_filter and model not in model_filter:
            continue

        # Skip failed runs
        if not r.get("success", True):
            continue
        if r.get("energy") is None:
            continue

        key = (impl, model, r["L"], r["chi"],
               r.get("sweeps", 30), r.get("threads", 1),
               r.get("nmax", ""), r.get("np", 1))
        if key in seen:
            continue
        seen.add(key)

        configs.append({
            "impl": impl,
            "model": model,
            "L": r["L"],
            "chi": r["chi"],
            "sweeps": r.get("sweeps", 30),
            "threads": r.get("threads", 1),
            "nmax": r.get("nmax", 2),
            "np": r.get("np", 1),
            "mi300x_energy": r.get("energy"),
            "mi300x_time": r.get("solve_time", r.get("wall_time")),
        })

    # Sort: impl, model, L, chi, threads
    configs.sort(key=lambda c: (c["impl"], c["model"], c["L"], c["chi"], c["threads"]))
    return configs


# ─── GPU runner ──────────────────────────────────────────────────────────────

def _parse_gpu_output(stdout):
    """Parse energy and wall time from GPU executable stdout."""
    energy, wall_time = None, None
    for line in stdout.split("\n"):
        m = re.search(r"(?:Final|Ground state)\s+energy:\s+([-\d.eE+]+)", line)
        if m:
            energy = float(m.group(1))
        m = re.search(r"(?:Total\s+)?[Ww]all\s+time:\s+([-\d.eE+]+)", line)
        if m:
            wall_time = float(m.group(1))
    return energy, wall_time


def _build_gpu_cmd(impl, model, L, chi, sweeps, nmax=2, np_count=1):
    """Build command line for a GPU executable."""
    exe = os.path.join(REPO_ROOT, GPU_EXECUTABLES[impl])
    if not os.path.exists(exe):
        return None, f"Executable not found: {exe}"

    cmd = [exe, str(L), str(chi), str(sweeps)]

    if model == "josephson":
        cmd.append("--josephson")
        cmd.extend(["--nmax", str(nmax)])
    elif model == "tfim":
        cmd.append("--tfim")

    if "pdmrg" in impl and np_count and np_count > 1:
        cmd.extend(["--segments", str(np_count)])
        cmd.extend(["--warmup", "3"])
        cmd.extend(["--local-sweeps", "2"])

    return cmd, None


def run_gpu(impl, model, L, chi, sweeps, nmax=2, np_count=1, timeout=600):
    """Run a single GPU benchmark. Returns result dict."""
    cmd, err = _build_gpu_cmd(impl, model, L, chi, sweeps, nmax, np_count)
    if err:
        return {"energy": None, "time": None, "success": False, "error": err}

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            return {"energy": None, "time": None, "success": False,
                    "error": (result.stderr or f"Exit code {result.returncode}")[-500:]}

        energy, wall_time = _parse_gpu_output(result.stdout)
        if energy is None:
            return {"energy": None, "time": None, "success": False,
                    "error": f"No energy in output: {result.stdout[-300:]}"}

        return {"energy": energy, "time": wall_time, "success": True}

    except subprocess.TimeoutExpired:
        return {"energy": None, "time": None, "success": False, "error": f"Timeout ({timeout}s)"}


def run_gpu_with_warmup(impl, model, L, chi, sweeps, nmax=2, np_count=1, timeout=600):
    """Run GPU benchmark with a warmup run first (discarded)."""
    # Warmup: small problem to initialize CUDA context + JIT
    # Use same executable but L=4 chi=4 sweeps=2 for minimal cost
    warmup_cmd, _ = _build_gpu_cmd(impl, model, 4, 4, 2, nmax, 1)
    if warmup_cmd:
        try:
            subprocess.run(warmup_cmd, capture_output=True, text=True, timeout=30)
        except Exception:
            pass  # warmup failure is non-fatal

    # Actual run
    return run_gpu(impl, model, L, chi, sweeps, nmax, np_count, timeout)


# ─── CPU (quimb) runner ─────────────────────────────────────────────────────

def run_quimb(impl, model, L, chi, sweeps, nmax=2, threads=1, tol=1e-11):
    """Run quimb DMRG in-process."""
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

    algorithm = "dmrg2" if "dmrg2" in impl else "dmrg1"

    try:
        if model == "heisenberg":
            mpo = qtn.MPO_ham_heis(L=L, j=1.0, bz=0.0, cyclic=False)
        elif model == "josephson":
            mpo = _build_josephson_mpo(L, nmax, qtn, np)
        elif model == "tfim":
            mpo = _build_tfim_mpo(L, qtn)
        else:
            return {"energy": None, "time": None, "success": False,
                    "error": f"Unknown model: {model}"}

        dmrg_class = qtn.DMRG2 if algorithm == "dmrg2" else qtn.DMRG1
        dmrg = dmrg_class(mpo, bond_dims=chi, cutoffs=1e-14)

        t0 = time.perf_counter()
        dmrg.solve(max_sweeps=sweeps, tol=tol, verbosity=0)
        elapsed = time.perf_counter() - t0

        return {"energy": float(np.real(dmrg.energy)), "time": elapsed, "success": True}

    except Exception as e:
        return {"energy": None, "time": None, "success": False, "error": str(e)[:200]}


def _build_josephson_mpo(L, n_max, qtn, np):
    """Build Josephson MPO matching GPU parameters: E_J=1.0, E_C=0.5, phi_ext=π/4."""
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
    """Build TFIM MPO: H = -J σz⊗σz - h σx, J=h=1.0 (critical point)."""
    J, h = 1.0, 1.0
    builder = qtn.SpinHam1D(S=0.5)
    builder.add_term(-4 * J, 'Z', 'Z')
    builder.add_term(-2 * h, 'X')
    return builder.build_mpo(L)


# ─── Main benchmark loop ────────────────────────────────────────────────────

def run_all(configs, do_warmup=True):
    """Run all configs, return results list."""
    results = []
    total = len(configs)

    current_group = None
    for i, cfg in enumerate(configs):
        group = (cfg["impl"], cfg["model"])
        if group != current_group:
            current_group = group
            print(f"\n{'='*65}")
            print(f"  {cfg['impl']} — {cfg['model']}")
            print(f"{'='*65}")

        is_gpu = cfg["impl"] in GPU_EXECUTABLES

        # Build label
        label = f"L={cfg['L']:3d} chi={cfg['chi']:3d} sw={cfg['sweeps']:2d}"
        if not is_gpu and cfg["threads"] > 1:
            label += f" t={cfg['threads']}"
        if cfg["np"] > 1:
            label += f" np={cfg['np']}"

        print(f"  [{i+1:3d}/{total}] {label:35s}", end="", flush=True)

        if is_gpu:
            runner = run_gpu_with_warmup if do_warmup else run_gpu
            r = runner(cfg["impl"], cfg["model"], cfg["L"], cfg["chi"],
                       cfg["sweeps"], nmax=cfg["nmax"], np_count=cfg["np"])
        else:
            r = run_quimb(cfg["impl"], cfg["model"], cfg["L"], cfg["chi"],
                          cfg["sweeps"], nmax=cfg["nmax"], threads=cfg["threads"])

        # Annotate result
        r["impl"] = cfg["impl"]
        r["model"] = cfg["model"]
        r["L"] = cfg["L"]
        r["chi"] = cfg["chi"]
        r["sweeps"] = cfg["sweeps"]
        r["threads"] = cfg["threads"]
        r["nmax"] = cfg["nmax"]
        r["np"] = cfg["np"]
        r["arch"] = "h100"
        r["mi300x_energy"] = cfg.get("mi300x_energy")
        r["mi300x_time"] = cfg.get("mi300x_time")

        if r["success"]:
            t_str = f"{r['time']:.3f}s" if r.get("time") else "?"
            mi_t = cfg.get("mi300x_time")
            speedup = ""
            if mi_t and r.get("time") and r["time"] > 0:
                ratio = mi_t / r["time"]
                speedup = f"  ({ratio:.1f}x vs MI300X)"
            print(f"  E={r['energy']:.10f}  t={t_str}{speedup}")
        else:
            print(f"  FAIL: {r.get('error', '?')[:50]}")

        results.append(r)

    return results


def save_results(results, output_dir):
    """Save results as JSON."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Per-implementation files
    by_impl = {}
    for r in results:
        by_impl.setdefault(r["impl"], []).append(r)

    for impl, impl_results in by_impl.items():
        path = os.path.join(output_dir, f"{impl}_h100_{timestamp}.json")
        with open(path, "w") as f:
            json.dump({"implementation": impl, "architecture": "h100",
                        "timestamp": timestamp, "results": impl_results}, f, indent=2)
        print(f"  Saved: {path}")

    # Combined file
    path = os.path.join(output_dir, f"combined_h100_{timestamp}.json")
    with open(path, "w") as f:
        json.dump({"architecture": "h100", "timestamp": timestamp,
                    "results": results}, f, indent=2)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="H100 Benchmarks (MI300X config reproduction)")
    parser.add_argument("--impl", type=str, default=None,
                        help="Comma-separated impls to run (default: all runnable)")
    parser.add_argument("--model", type=str, default=None,
                        help="Filter by model: heisenberg, josephson, tfim")
    parser.add_argument("--skip-warmup", action="store_true",
                        help="Skip GPU warmup runs")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(REPO_ROOT, "benchmarks", "paper_results", "h100"))
    args = parser.parse_args()

    impl_filter = None
    if args.impl:
        impl_filter = set(x.strip() for x in args.impl.split(","))

    model_filter = None
    if args.model:
        model_filter = set(x.strip() for x in args.model.split(","))

    configs = load_mi300x_configs(impl_filter, model_filter)

    if not configs:
        print("No matching configs found.")
        sys.exit(1)

    # Summary
    impls = sorted(set(c["impl"] for c in configs))
    models = sorted(set(c["model"] for c in configs))
    print(f"\n{'#'*65}")
    print(f"  H100 Benchmark — Reproducing MI300X configs")
    print(f"  Implementations: {', '.join(impls)}")
    print(f"  Models: {', '.join(models)}")
    print(f"  Total configs: {len(configs)}")
    print(f"  GPU warmup: {'OFF' if args.skip_warmup else 'ON'}")
    print(f"  Output: {args.output_dir}")
    print(f"{'#'*65}")

    results = run_all(configs, do_warmup=not args.skip_warmup)

    print(f"\n{'='*65}")
    print(f"  Saving results...")
    print(f"{'='*65}")
    save_results(results, args.output_dir)

    successes = sum(1 for r in results if r["success"])
    failures = sum(1 for r in results if not r["success"])
    print(f"\n  Done: {len(results)} runs, {successes} passed, {failures} failed")


if __name__ == "__main__":
    main()
