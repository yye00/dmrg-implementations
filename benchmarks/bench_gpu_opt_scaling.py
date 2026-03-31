#!/usr/bin/env python3
"""
Phase 2+3: pdmrg-gpu-opt baseline + cross-segment batched GEMM scaling study.
Continuation of bench_gpu_all.py (Phase 1 + pdmrg-gpu already done).
"""
import subprocess
import json
import time
import os
import re
import sys
from datetime import datetime

BASE_DIR = os.environ.get("BENCH_BASE", "/home/hotaisle/dmrg-implementations")
RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "paper_results", "gpu_opt_bench.json")
TIMEOUT = 300  # 5 min per run (reduced from 10)

def run_cmd(cmd_str, timeout=TIMEOUT):
    try:
        result = subprocess.run(["bash", "-c", cmd_str],
                                capture_output=True, text=True, timeout=timeout)
        return result.stdout + result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "TIMEOUT", -1

def parse_output(output):
    energy = None
    wall_time = None
    success = "PASS" in output
    m = re.search(r'Final energy:\s*([-\d.eE+]+)', output)
    if m: energy = float(m.group(1))
    m = re.search(r'Total wall time:\s*([\d.]+)\s*s', output)
    if m: wall_time = float(m.group(1))
    m = re.search(r'Absolute error:\s*([\d.eE+-]+)', output)
    error = float(m.group(1)) if m else None
    m = re.search(r'Converged after (\d+) outer', output)
    outer_iters = int(m.group(1)) if m else None
    return {"energy": energy, "wall_time": wall_time, "success": success,
            "error": error, "outer_iters": outer_iters}

def run_bench(impl, model, L, chi, sweeps, segments=None, extra_flags="", label=""):
    exe = f"{BASE_DIR}/{impl.replace('-', '-', 1)}/build/{impl.replace('-', '_')}"
    # Fix: pdmrg-gpu-opt -> pdmrg_gpu_opt in pdmrg-gpu-opt/build/pdmrg_gpu_opt
    if "pdmrg-gpu-opt" in impl:
        exe = f"{BASE_DIR}/pdmrg-gpu-opt/build/pdmrg_gpu_opt"
    elif "pdmrg-gpu" in impl:
        exe = f"{BASE_DIR}/pdmrg-gpu/build/pdmrg_gpu"

    cmd = f"{exe} {L} {chi} {sweeps}"
    if segments:
        cmd += f" --segments {segments}"
    if model == "josephson":
        cmd += " --josephson --nmax 2"
    elif model == "tfim":
        cmd += " --tfim"
    if extra_flags:
        cmd += f" {extra_flags}"

    desc = f"{impl} {model} L={L} chi={chi}"
    if segments: desc += f" seg={segments}"
    if label: desc += f" [{label}]"
    print(f"  {desc} ... ", end="", flush=True)

    t0 = time.time()
    output, rc = run_cmd(cmd)
    elapsed = time.time() - t0

    if rc == -1:
        print(f"TIMEOUT ({TIMEOUT}s)")
        return {"impl": impl, "model": model, "L": L, "chi": chi, "sweeps": sweeps,
                "segments": segments, "extra_flags": extra_flags, "label": label,
                "wall_time": None, "energy": None, "success": False, "error": None,
                "outer_iters": None, "status": "timeout", "elapsed": elapsed,
                "timestamp": datetime.now().isoformat()}

    p = parse_output(output)
    status = "pass" if p["success"] else ("completed" if p["energy"] else "error")
    wt = p["wall_time"]
    print(f"{wt:.2f}s" if wt else "N/A", end="")
    if p["error"]: print(f"  err={p['error']:.1e}", end="")
    if p["outer_iters"]: print(f"  iters={p['outer_iters']}", end="")
    print(f"  [{status}]")

    return {"impl": impl, "model": model, "L": L, "chi": chi, "sweeps": sweeps,
            "segments": segments, "extra_flags": extra_flags, "label": label,
            "wall_time": wt, "energy": p["energy"], "success": p["success"],
            "error": p["error"], "outer_iters": p["outer_iters"],
            "status": status, "elapsed": elapsed,
            "timestamp": datetime.now().isoformat()}

def save_results(results):
    existing = []
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            existing = json.load(f)
    existing.extend(results)
    with open(RESULTS_FILE, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"  [saved {len(results)} results, total {len(existing)}]")

def main():
    all_results = []

    # ============================================================
    # Phase 2: pdmrg-gpu-opt baseline (no batched sweep)
    # ============================================================
    print("=" * 60)
    print("PHASE 2: pdmrg-gpu-opt baseline")
    print("=" * 60)

    p2_configs = [
        ("heisenberg", 8,   32,  20, 2),
        ("heisenberg", 20,  50,  20, 2),
        ("heisenberg", 32,  64,  20, 2),
        ("heisenberg", 32,  128, 20, 2),
        ("heisenberg", 64,  128, 20, 2),
        ("heisenberg", 64,  128, 20, 4),
        ("heisenberg", 100, 128, 20, 2),
        ("heisenberg", 100, 128, 20, 4),
        ("tfim",       32,  128, 20, 2),
        ("josephson",  16,  50,  20, 2),
    ]

    batch = []
    for model, L, chi, sweeps, seg in p2_configs:
        r = run_bench("pdmrg-gpu-opt", model, L, chi, sweeps, segments=seg)
        all_results.append(r)
        batch.append(r)
    save_results(batch)

    # ============================================================
    # Phase 3: Batched sweep scaling study
    # ============================================================
    print("\n" + "=" * 60)
    print("PHASE 3: Cross-segment batched GEMM scaling")
    print("  Comparing baseline (thread-per-segment) vs batched sweep")
    print("=" * 60)

    # Key question: does batching help at higher segment counts or larger problems?
    batched_configs = [
        # (model, L, chi, sweeps, segments)
        # Small baseline
        ("heisenberg", 20,  50,  20, 2),
        ("heisenberg", 20,  50,  20, 4),
        # Medium
        ("heisenberg", 32,  64,  20, 2),
        ("heisenberg", 32,  64,  20, 4),
        ("heisenberg", 32,  128, 20, 2),
        ("heisenberg", 32,  128, 20, 4),
        ("heisenberg", 32,  128, 20, 8),
        # Large L
        ("heisenberg", 64,  128, 20, 2),
        ("heisenberg", 64,  128, 20, 4),
        ("heisenberg", 64,  128, 20, 8),
        ("heisenberg", 64,  128, 20, 16),
        # Very large L
        ("heisenberg", 100, 128, 20, 2),
        ("heisenberg", 100, 128, 20, 4),
        ("heisenberg", 100, 128, 20, 8),
        # Large chi
        ("heisenberg", 32,  256, 20, 2),
        ("heisenberg", 32,  256, 20, 4),
        ("heisenberg", 64,  256, 20, 4),
        # Josephson (complex, d=5)
        ("josephson",  16,  50,  20, 2),
        ("josephson",  16,  50,  20, 4),
    ]

    batch = []
    for model, L, chi, sweeps, seg in batched_configs:
        # Baseline
        r = run_bench("pdmrg-gpu-opt", model, L, chi, sweeps, segments=seg,
                       extra_flags="--no-batched-sweep", label="baseline")
        all_results.append(r)
        batch.append(r)
        # Batched
        r = run_bench("pdmrg-gpu-opt", model, L, chi, sweeps, segments=seg,
                       extra_flags="--batched-sweep", label="batched")
        all_results.append(r)
        batch.append(r)
        save_results(batch[-2:])  # save pair immediately

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    # Phase 3 comparison
    print("\n--- Batched Sweep Comparison ---")
    print("%-45s %10s %10s %10s" % ("Config", "Baseline", "Batched", "Speedup"))
    print("-" * 80)

    phase3 = [r for r in all_results if r.get("label") in ("baseline", "batched")]
    for i in range(0, len(phase3), 2):
        if i + 1 >= len(phase3):
            break
        base, batc = phase3[i], phase3[i + 1]
        if base["label"] != "baseline" or batc["label"] != "batched":
            continue
        desc = "%s L=%d chi=%d seg=%s" % (base["model"], base["L"], base["chi"], base["segments"])
        bt = base["wall_time"]
        at = batc["wall_time"]
        if bt and at:
            speedup = bt / at
            marker = " <--" if speedup > 1.05 else (" SLOWER" if speedup < 0.95 else "")
            print("  %-43s %8.2fs %8.2fs %8.2fx%s" % (desc, bt, at, speedup, marker))
        else:
            bt_s = "%.2fs" % bt if bt else "TIMEOUT"
            at_s = "%.2fs" % at if at else "TIMEOUT"
            print("  %-43s %10s %10s %10s" % (desc, bt_s, at_s, "N/A"))

    print("\nDone! Results in:", RESULTS_FILE)

if __name__ == "__main__":
    main()
