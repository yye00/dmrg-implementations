#!/usr/bin/env python3
"""
Comprehensive GPU benchmark: all -gpu and -gpu-opt implementations.
Includes pdmrg-gpu-opt cross-segment batched GEMM scaling study.

Run on remote MI300X via SSH or directly on GPU host.
"""
import subprocess
import json
import time
import os
import re
import sys
from datetime import datetime

REMOTE = os.environ.get("BENCH_REMOTE", "")  # e.g. "hotaisle@23.183.40.79"
BASE_DIR = os.environ.get("BENCH_BASE", "/home/hotaisle/dmrg-implementations")
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "paper_results", "gpu_opt_bench.json")
TIMEOUT = 600  # 10 min per run

# ============================================================
# GPU binaries and their CLI patterns
# ============================================================
BINARIES = {
    "dmrg-gpu":      f"{BASE_DIR}/dmrg-gpu/build/dmrg_gpu",
    "dmrg2-gpu":     f"{BASE_DIR}/dmrg2-gpu/build/dmrg2_gpu",
    "dmrg-gpu-opt":  f"{BASE_DIR}/dmrg-gpu-opt/build/dmrg_gpu_opt",
    "dmrg2-gpu-opt": f"{BASE_DIR}/dmrg2-gpu-opt/build/dmrg2_gpu_opt",
    "pdmrg-gpu":     f"{BASE_DIR}/pdmrg-gpu/build/pdmrg_gpu",
    "pdmrg-gpu-opt": f"{BASE_DIR}/pdmrg-gpu-opt/build/pdmrg_gpu_opt",
}

def run_cmd(cmd_str, timeout=TIMEOUT):
    """Run command, optionally via SSH."""
    if REMOTE:
        full_cmd = ["ssh", REMOTE, cmd_str]
    else:
        full_cmd = ["bash", "-c", cmd_str]
    try:
        result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)
        return result.stdout + result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "TIMEOUT", -1

def parse_output(output):
    """Extract energy and wall time from GPU binary output."""
    energy = None
    wall_time = None
    success = False

    m = re.search(r'Final energy:\s*([-\d.eE+]+)', output)
    if m:
        energy = float(m.group(1))
    m = re.search(r'Total wall time:\s*([\d.]+)\s*s', output)
    if m:
        wall_time = float(m.group(1))
    if "PASS" in output:
        success = True
    m = re.search(r'Absolute error:\s*([\d.eE+-]+)', output)
    error = float(m.group(1)) if m else None

    return {"energy": energy, "wall_time": wall_time, "success": success, "error": error}

def build_cmd(impl, model, L, chi, sweeps, segments=None, extra_flags=""):
    """Build command line for a GPU binary."""
    exe = BINARIES[impl]

    if "pdmrg" in impl:
        # pdmrg: L chi outer --segments N
        cmd = f"{exe} {L} {chi} {sweeps}"
        if segments:
            cmd += f" --segments {segments}"
    else:
        # dmrg/dmrg2: L chi sweeps
        cmd = f"{exe} {L} {chi} {sweeps}"

    if model == "josephson":
        cmd += " --josephson --nmax 2"
    elif model == "tfim":
        cmd += " --tfim"

    if extra_flags:
        cmd += f" {extra_flags}"

    return cmd

def run_bench(impl, model, L, chi, sweeps, segments=None, extra_flags="", label=""):
    """Run a single benchmark and return results dict."""
    cmd = build_cmd(impl, model, L, chi, sweeps, segments, extra_flags)
    desc = f"{impl} {model} L={L} chi={chi}"
    if segments:
        desc += f" seg={segments}"
    if label:
        desc += f" [{label}]"

    print(f"  Running: {desc} ... ", end="", flush=True)
    t0 = time.time()
    output, rc = run_cmd(cmd)
    elapsed = time.time() - t0

    if rc == -1:
        print(f"TIMEOUT ({TIMEOUT}s)")
        return {
            "impl": impl, "model": model, "L": L, "chi": chi, "sweeps": sweeps,
            "segments": segments, "extra_flags": extra_flags, "label": label,
            "wall_time": None, "energy": None, "success": False, "error": None,
            "status": "timeout", "elapsed": elapsed,
            "timestamp": datetime.now().isoformat()
        }

    parsed = parse_output(output)
    status = "pass" if parsed["success"] else ("completed" if parsed["energy"] else "error")

    wt = parsed["wall_time"]
    print(f"{wt:.2f}s" if wt else "N/A", end="")
    if parsed["error"]:
        print(f"  err={parsed['error']:.1e}", end="")
    print(f"  [{status}]")

    return {
        "impl": impl, "model": model, "L": L, "chi": chi, "sweeps": sweeps,
        "segments": segments, "extra_flags": extra_flags, "label": label,
        "wall_time": parsed["wall_time"], "energy": parsed["energy"],
        "success": parsed["success"], "error": parsed["error"],
        "status": status, "elapsed": elapsed,
        "timestamp": datetime.now().isoformat()
    }

def save_results(results):
    """Append results to JSON file."""
    existing = []
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            existing = json.load(f)
    existing.extend(results)
    with open(RESULTS_FILE, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"  Saved {len(results)} results to {RESULTS_FILE}")

def main():
    results = []

    # ============================================================
    # Phase 1: All -gpu implementations (serial DMRG)
    # ============================================================
    print("=" * 60)
    print("PHASE 1: Serial GPU implementations (dmrg-gpu, dmrg2-gpu)")
    print("=" * 60)

    serial_configs = [
        # (model, L, chi, sweeps)
        ("heisenberg", 8,   32,  30),
        ("heisenberg", 12,  20,  30),
        ("heisenberg", 20,  50,  40),
        ("heisenberg", 32,  64,  40),
        ("heisenberg", 32,  128, 50),
        ("heisenberg", 64,  128, 50),
        ("heisenberg", 100, 128, 60),
        ("tfim",       20,  50,  40),
        ("tfim",       32,  128, 50),
        ("josephson",  8,   32,  30),
        ("josephson",  16,  50,  40),
        ("josephson",  32,  128, 50),
    ]

    for impl in ["dmrg-gpu", "dmrg2-gpu", "dmrg-gpu-opt", "dmrg2-gpu-opt"]:
        print(f"\n--- {impl} ---")
        for model, L, chi, sweeps in serial_configs:
            r = run_bench(impl, model, L, chi, sweeps)
            results.append(r)
            if r["status"] == "timeout":
                print(f"    Skipping larger sizes for {impl} {model}")
                break
        save_results(results[-len(serial_configs):])

    # ============================================================
    # Phase 2: Parallel GPU (pdmrg-gpu, pdmrg-gpu-opt baseline)
    # ============================================================
    print("\n" + "=" * 60)
    print("PHASE 2: Parallel GPU (pdmrg-gpu, pdmrg-gpu-opt)")
    print("=" * 60)

    parallel_configs = [
        # (model, L, chi, sweeps, segments)
        ("heisenberg", 8,   32,  20, 2),
        ("heisenberg", 12,  20,  20, 2),
        ("heisenberg", 20,  50,  20, 2),
        ("heisenberg", 32,  64,  20, 2),
        ("heisenberg", 32,  128, 20, 2),
        ("heisenberg", 64,  128, 20, 2),
        ("heisenberg", 64,  128, 20, 4),
        ("heisenberg", 100, 128, 20, 2),
        ("heisenberg", 100, 128, 20, 4),
        ("tfim",       32,  128, 20, 2),
        ("josephson",  16,  50,  20, 2),
        ("josephson",  32,  128, 20, 2),
    ]

    for impl in ["pdmrg-gpu", "pdmrg-gpu-opt"]:
        print(f"\n--- {impl} ---")
        batch = []
        for model, L, chi, sweeps, seg in parallel_configs:
            r = run_bench(impl, model, L, chi, sweeps, segments=seg)
            results.append(r)
            batch.append(r)
        save_results(batch)

    # ============================================================
    # Phase 3: pdmrg-gpu-opt batched sweep scaling study
    # ============================================================
    print("\n" + "=" * 60)
    print("PHASE 3: pdmrg-gpu-opt batched-sweep scaling study")
    print("  Testing if cross-segment batched GEMM helps at scale")
    print("=" * 60)

    # Test matrix: vary L, chi, segments with and without batched sweep
    batched_configs = [
        # Small: should be worse (known from Item 4 testing)
        ("heisenberg", 8,   32,  20, 2),
        # Medium
        ("heisenberg", 20,  50,  20, 2),
        ("heisenberg", 20,  50,  20, 4),
        ("heisenberg", 32,  64,  20, 2),
        ("heisenberg", 32,  64,  20, 4),
        # Large: where batching might help
        ("heisenberg", 32,  128, 20, 2),
        ("heisenberg", 32,  128, 20, 4),
        ("heisenberg", 32,  128, 20, 8),
        ("heisenberg", 64,  128, 20, 2),
        ("heisenberg", 64,  128, 20, 4),
        ("heisenberg", 64,  128, 20, 8),
        ("heisenberg", 64,  128, 20, 16),
        ("heisenberg", 100, 128, 20, 2),
        ("heisenberg", 100, 128, 20, 4),
        ("heisenberg", 100, 128, 20, 8),
        # Very large chi
        ("heisenberg", 32,  256, 20, 2),
        ("heisenberg", 32,  256, 20, 4),
        ("heisenberg", 64,  256, 20, 4),
        ("heisenberg", 64,  256, 20, 8),
        # Josephson (complex, d=5 -> larger theta)
        ("josephson",  16,  50,  20, 2),
        ("josephson",  16,  50,  20, 4),
        ("josephson",  32,  128, 20, 2),
        ("josephson",  32,  128, 20, 4),
    ]

    batch = []
    for model, L, chi, sweeps, seg in batched_configs:
        # Baseline: no batched sweep (default)
        r = run_bench("pdmrg-gpu-opt", model, L, chi, sweeps, segments=seg,
                       extra_flags="--no-batched-sweep", label="baseline")
        results.append(r)
        batch.append(r)

        # Batched sweep
        r = run_bench("pdmrg-gpu-opt", model, L, chi, sweeps, segments=seg,
                       extra_flags="--batched-sweep", label="batched")
        results.append(r)
        batch.append(r)

    save_results(batch)

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)

    # Print batched vs baseline comparison
    print("\n--- Batched Sweep Comparison ---")
    print(f"{'Config':<45} {'Baseline':>10} {'Batched':>10} {'Speedup':>10}")
    print("-" * 80)

    # Pair up baseline/batched results from Phase 3
    phase3 = [r for r in results if r.get("label") in ("baseline", "batched")]
    for i in range(0, len(phase3), 2):
        if i + 1 >= len(phase3):
            break
        base, batc = phase3[i], phase3[i + 1]
        if base["label"] != "baseline" or batc["label"] != "batched":
            continue
        desc = f"{base['model']} L={base['L']} chi={base['chi']} seg={base['segments']}"
        bt = base["wall_time"]
        at = batc["wall_time"]
        if bt and at:
            speedup = bt / at
            print(f"  {desc:<43} {bt:>8.2f}s {at:>8.2f}s {speedup:>8.2f}x")
        else:
            print(f"  {desc:<43} {'N/A':>10} {'N/A':>10} {'N/A':>10}")

    total = len(results)
    passed = sum(1 for r in results if r["success"])
    failed = sum(1 for r in results if r["status"] == "error")
    timeouts = sum(1 for r in results if r["status"] == "timeout")
    print(f"\nTotal: {total} runs, {passed} passed, {failed} failed, {timeouts} timeouts")


if __name__ == "__main__":
    main()
