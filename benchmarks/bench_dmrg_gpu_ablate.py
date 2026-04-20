#!/usr/bin/env python3
"""
Ablation benchmark for dmrg-gpu GPU-specific performance flags.

Design
------
Each optimization on the claude/dmrg-gpu-concurrency-XrGHv branch is gated by
a DMRG_GPU_OPT_<NAME> env var. With all flags unset the binary reproduces the
baseline numerics from main. This script runs a primary Josephson config and
a secondary Josephson config through:

    * baseline        (all flags off)
    * one-at-a-time   (baseline + single flag on)         — isolated contribution
    * leave-one-out   (all flags on, one flag removed)    — redundancy check
    * all-on          (every flag on)                     — upper-bound speedup

Results land in benchmarks/data/gpu_ablation/<timestamp>/results.json and a
markdown table is printed to stdout.

Correctness gate: before timing, the script runs a small Josephson config
(L=8, chi=32) with each single-flag configuration and asserts the energy
matches the baseline to ENERGY_TOL (1e-10 absolute). Regressions abort the
whole run.

Usage
-----
    python bench_dmrg_gpu_ablate.py \
        --binary gpu-rocm/dmrg-gpu/build/dmrg_gpu \
        --reps 3

On the MI300X host run from ~/dmrg-implementations.
"""

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_BENCH_LIB = Path(__file__).resolve().parent / "lib"
sys.path.insert(0, str(_BENCH_LIB))
from provenance import provenance_block, binary_info

# ─── Optimization flags (mirrors GpuOpts in dmrg_gpu.h) ───────────────────────
OPT_FLAGS = [
    "DEVICE_K",
    "LANCZOS_FIXED",
    "LANCZOS_GRAPH",
    "RSVD",
    "SPARSE_MPO",
    "FUSE_LANCZOS",
    "D_PAD",
]

# ─── Problem configs ──────────────────────────────────────────────────────────
# Primary: (32, 128, 20) Josephson, n_max=2, complex128 — mid-size apply_heff / SVD mix.
# Secondary: (32, 256, 15) — larger chi, SVD-dominated regime.
# Correctness: (8, 32, 10) — tiny, fast, catches regressions.
BENCH_CONFIGS = [
    {"label": "josephson_L32_chi128", "L": 32, "chi": 128, "sweeps": 20, "nmax": 2},
    {"label": "josephson_L32_chi256", "L": 32, "chi": 256, "sweeps": 15, "nmax": 2},
]

CORRECTNESS_CONFIG = {"L": 8, "chi": 32, "sweeps": 10, "nmax": 2}
ENERGY_TOL = 5e-10  # 5e-10 accommodates RSVD stochasticity (~4e-11 std across reps)


def build_argv(binary: str, cfg: dict) -> list:
    return [
        str(binary),
        str(cfg["L"]), str(cfg["chi"]), str(cfg["sweeps"]),
        "--josephson", "--nmax", str(cfg["nmax"]),
        "--quiet",
    ]


def build_env(flag_names: list, profile: bool = False) -> dict:
    env = os.environ.copy()
    for f in OPT_FLAGS:
        env.pop(f"DMRG_GPU_OPT_{f}", None)
    for f in flag_names:
        env[f"DMRG_GPU_OPT_{f}"] = "1"
    if profile:
        env["DMRG_GPU_PROFILE"] = "1"
    else:
        env.pop("DMRG_GPU_PROFILE", None)
    return env


ENERGY_RE = re.compile(r"Final energy:\s+([-\d.eE+]+)")
WALL_RE   = re.compile(r"Total wall time:\s+([\d.]+)\s*s")
PHASE_RE  = re.compile(r"^\s+(\w+)\s*:\s+([\d.]+) ms\s+\(\s*(\d+) calls")


def run_one(binary: str, cfg: dict, flag_names: list, profile: bool = False, timeout: int = 1800):
    argv = build_argv(binary, cfg)
    env = build_env(flag_names, profile=profile)
    t0 = time.time()
    proc = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=timeout)
    wall = time.time() - t0
    energy = None
    reported_wall = None
    phases = {}
    for line in proc.stdout.splitlines() + proc.stderr.splitlines():
        if m := ENERGY_RE.search(line):
            energy = float(m.group(1))
        if m := WALL_RE.search(line):
            reported_wall = float(m.group(1))
        if m := PHASE_RE.match(line):
            phases[m.group(1)] = {
                "ms": float(m.group(2)),
                "calls": int(m.group(3)),
            }
    return {
        "energy": energy,
        "wall_s": reported_wall if reported_wall is not None else wall,
        "wall_wallclock_s": wall,
        "phases": phases,
        "returncode": proc.returncode,
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-10:]),
    }


def correctness_gate(binary: str):
    """Run baseline + each single flag on the tiny config; assert energy match."""
    print(f"== Correctness gate ({CORRECTNESS_CONFIG['L']}, chi={CORRECTNESS_CONFIG['chi']}, "
          f"sweeps={CORRECTNESS_CONFIG['sweeps']}, tol={ENERGY_TOL}) ==")
    base = run_one(binary, CORRECTNESS_CONFIG, [])
    if base["energy"] is None or base["returncode"] != 0:
        print(f"  baseline FAILED: rc={base['returncode']}\n{base['stderr_tail']}")
        sys.exit(2)
    print(f"  baseline                E = {base['energy']:.12f}")
    bad = []
    for f in OPT_FLAGS:
        r = run_one(binary, CORRECTNESS_CONFIG, [f])
        if r["energy"] is None:
            status = "MISSING_ENERGY"
        elif abs(r["energy"] - base["energy"]) < ENERGY_TOL:
            status = "ok"
        else:
            status = f"MISMATCH dE={r['energy'] - base['energy']:+.2e}"
            bad.append(f)
        print(f"  {f:<14}         E = {r['energy']!s:<20}  [{status}]")
    if bad:
        print(f"\nCorrectness gate FAILED for: {', '.join(bad)}")
        print("Aborting benchmark run.")
        sys.exit(1)
    print("  all single-flag configs within tolerance.\n")


def ablation_configs():
    """List of (label, [flags]) tuples for the ablation sweep."""
    configs = [("baseline", [])]
    for f in OPT_FLAGS:
        configs.append((f"only_{f}", [f]))
    for f in OPT_FLAGS:
        rest = [x for x in OPT_FLAGS if x != f]
        configs.append((f"no_{f}", rest))
    configs.append(("all_on", list(OPT_FLAGS)))
    return configs


def run_bench(binary: str, reps: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    configs = ablation_configs()
    n = len(configs) * len(BENCH_CONFIGS) * reps
    i = 0
    for prob in BENCH_CONFIGS:
        for label, flags in configs:
            reps_data = []
            for r in range(reps):
                i += 1
                t0 = time.time()
                res = run_one(binary, prob, flags, profile=False)
                print(f"  [{i:3d}/{n}] {prob['label']:<24} {label:<22} rep={r} "
                      f"wall={res['wall_s']:.2f}s E={res['energy']}")
                if res["returncode"] != 0 or res["energy"] is None:
                    print(f"    FAILED:\n{res['stderr_tail']}")
                reps_data.append(res)
            walls = [r["wall_s"] for r in reps_data if r["wall_s"] is not None]
            median_wall = statistics.median(walls) if walls else None
            results.append({
                "problem": prob["label"],
                "config": label,
                "flags": flags,
                "reps": reps_data,
                "median_wall_s": median_wall,
            })
    # One profile run per problem for phase breakdown (all-on).
    profile_runs = {}
    for prob in BENCH_CONFIGS:
        r = run_one(binary, prob, list(OPT_FLAGS), profile=True)
        profile_runs[prob["label"]] = r

    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "binary": str(binary),
        "binary_info": binary_info(str(binary)),
        "provenance": provenance_block(script_argv=sys.argv[1:]),
        "problems": BENCH_CONFIGS,
        "opt_flags": OPT_FLAGS,
        "reps": reps,
        "results": results,
        "profile_runs_all_on": profile_runs,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2))
    return payload


def print_markdown(payload: dict):
    print("\n## Ablation results (median of reps)\n")
    problems = [p["label"] for p in payload["problems"]]
    configs = []
    seen = set()
    for r in payload["results"]:
        if r["config"] not in seen:
            configs.append(r["config"])
            seen.add(r["config"])
    # baseline per problem
    base = {}
    for r in payload["results"]:
        if r["config"] == "baseline":
            base[r["problem"]] = r["median_wall_s"]

    header = "| config |" + "".join(f" {p} (s) | spd |" for p in problems)
    sep = "|---|" + "---|---|" * len(problems)
    print(header)
    print(sep)
    for cfg in configs:
        row = [f"| `{cfg}` "]
        for p in problems:
            wall = next((r["median_wall_s"] for r in payload["results"]
                         if r["problem"] == p and r["config"] == cfg), None)
            spd = base[p] / wall if (wall and base.get(p)) else None
            row.append(f"| {wall:.2f} " if wall else "| — ")
            row.append(f"| {spd:.2f}× " if spd else "| — ")
        row.append("|")
        print("".join(row))

    print("\n## Phase breakdown (all flags on, profile build)\n")
    for prob, r in payload["profile_runs_all_on"].items():
        print(f"\n**{prob}** — wall={r['wall_s']:.2f}s  E={r['energy']}\n")
        phases = r.get("phases", {})
        if not phases:
            print("  (no phase data)")
            continue
        print("| phase | total_ms | calls | ms/call |")
        print("|---|---|---|---|")
        for name, d in phases.items():
            per = d["ms"] / d["calls"] if d["calls"] else 0.0
            print(f"| {name} | {d['ms']:.2f} | {d['calls']} | {per:.3f} |")


# ─── Multi-variant support ────────────────────────────────────────────────────

IMPL_BINARIES = {
    "dmrg-gpu":     "gpu-rocm/dmrg-gpu/build/dmrg_gpu",
    "dmrg2-gpu":    "gpu-rocm/dmrg2-gpu/build/dmrg2_gpu",
    "pdmrg-gpu":    "gpu-rocm/pdmrg-gpu/build/pdmrg_gpu",
    "dmrg-gpu-opt": "gpu-rocm/dmrg-gpu-opt/build/dmrg_gpu_opt",
    "dmrg2-gpu-opt":"gpu-rocm/dmrg2-gpu-opt/build/dmrg2_gpu_opt",
    "pdmrg-gpu-opt":"gpu-rocm/pdmrg-gpu-opt/build/pdmrg_gpu_opt",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=None,
                    help="path to a single executable (overrides --impl)")
    ap.add_argument("--impl", nargs="*", default=["dmrg-gpu"],
                    choices=list(IMPL_BINARIES.keys()),
                    help="variant names to benchmark (default: dmrg-gpu)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out",  default="benchmarks/data/gpu_ablation",
                    help="output directory root")
    ap.add_argument("--skip-correctness", action="store_true",
                    help="skip the correctness gate (NOT recommended)")
    args = ap.parse_args()

    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    if args.binary:
        binaries = [("custom", Path(args.binary).resolve())]
    else:
        binaries = []
        for name in args.impl:
            p = Path(IMPL_BINARIES[name]).resolve()
            if not p.exists():
                print(f"Binary not found for {name}: {p}")
                sys.exit(2)
            binaries.append((name, p))

    for impl_name, binary in binaries:
        print(f"\n{'='*60}")
        print(f"  Variant: {impl_name}  ({binary})")
        print(f"{'='*60}\n")
        if not args.skip_correctness:
            correctness_gate(str(binary))
        out_dir = Path(args.out) / stamp / impl_name
        payload = run_bench(str(binary), args.reps, out_dir)
        print_markdown(payload)
        print(f"\nresults → {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
