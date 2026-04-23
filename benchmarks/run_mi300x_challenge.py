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
sys.path.insert(0, os.path.join(REPO_ROOT, "benchmarks", "lib"))
from provenance import provenance_block, binary_info

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

# ─── Ultra-trim grid: 18 configs total for R3 regression + F1/F2/size-gate validation
# Covers the meaningful GPU-wins regime. Chi in {64,128,256}, strategic L.
ULTRA_TRIM_SIZES = {
    "heisenberg": [
        ( 50,  64, 20),
        ( 50, 128, 20),
        ( 50, 256, 15),
        (100,  64, 20),
        (100, 128, 20),
        (100, 256, 15),
    ],
    "josephson": [
        (16,  64, 20),
        (16, 128, 20),
        (16, 256, 15),
        (32,  64, 20),
        (32, 128, 20),
        (32, 256, 15),
    ],
    "tfim": [
        ( 50,  64, 20),
        ( 50, 128, 20),
        ( 50, 256, 15),
        (100,  64, 20),
        (100, 128, 20),
        (100, 256, 15),
    ],
}

# ─── GPU executables ─────────────────────────────────────────────────────────

GPU_IMPLS = {
    "dmrg-gpu":          "gpu-rocm/dmrg-gpu/build/dmrg_gpu",
    "dmrg-gpu-base":     "gpu-rocm/dmrg-gpu-base/build/dmrg_gpu_base",
    "dmrg-gpu-opt":      "gpu-rocm/dmrg-gpu-opt/build/dmrg_gpu_opt",
    "dmrg2-gpu":         "gpu-rocm/dmrg2-gpu/build/dmrg2_gpu",
    "dmrg2-gpu-base":    "gpu-rocm/dmrg2-gpu-base/build/dmrg2_gpu_base",
    "dmrg2-gpu-opt":     "gpu-rocm/dmrg2-gpu-opt/build/dmrg2_gpu_opt",
    "pdmrg-gpu":         "gpu-rocm/pdmrg-gpu/build/pdmrg_gpu",
    "pdmrg-gpu-base":    "gpu-rocm/pdmrg-gpu-base/build/pdmrg_gpu_base",
    "pdmrg-gpu-opt":     "gpu-rocm/pdmrg-gpu-opt/build/pdmrg_gpu_opt",
    "pdmrg-multi-gpu":   "gpu-rocm/pdmrg-multi-gpu/build/pdmrg_multi_gpu",
}

CPU_IMPLS = {"quimb-dmrg1", "quimb-dmrg2"}

ALL_IMPLS = sorted(GPU_IMPLS.keys()) + sorted(CPU_IMPLS)

# Josephson parameters (must match GPU defaults)
JOSEPHSON_NMAX = 2


# ─── rocm-smi pinning harness ────────────────────────────────────────────────

def _rocm_smi_capture():
    """Capture GPU clock/thermal/process state via rocm-smi.  Returns a dict.
    On VF the clock-pin commands will no-op; we log their return codes so
    downstream analysis can detect non-pinned runs."""
    result = {}
    for flag, key in [
        ('--showclocks --json',                    'clocks'),
        ('--showtemp --showpower --showuse --json', 'thermals'),
        ('--showpids --json',                       'pids'),
    ]:
        try:
            r = subprocess.run(['rocm-smi'] + flag.split(),
                               capture_output=True, text=True, timeout=10)
            try:
                result[key] = json.loads(r.stdout)
            except json.JSONDecodeError:
                result[key] = r.stdout.strip()
        except Exception as exc:
            result[key] = {'error': str(exc)}
    return result


def _rocm_smi_pin():
    """Attempt to pin MI300X clocks (will silently no-op on VF).
    Returns dict mapping command -> return code for auditability."""
    pin_cmds = [
        ['rocm-smi', '--setperfdeterminism', '1900'],
        ['rocm-smi', '--setperflevel', 'manual'],
        ['rocm-smi', '--setpowercap', '750'],
    ]
    rc_log = {}
    for cmd in pin_cmds:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            rc_log[' '.join(cmd)] = r.returncode
        except Exception as exc:
            rc_log[' '.join(cmd)] = str(exc)
    return rc_log


def _rocm_smi_rep_preamble():
    """Pre-rep: capture baseline state, attempt clock pin.
    Returns (pre_state_dict, pin_rc_dict)."""
    pre = _rocm_smi_capture()
    pin_rc = _rocm_smi_pin()
    return pre, pin_rc


def _rocm_smi_rep_postamble(pre, pinned_sclk_mhz=1900):
    """Post-rep: capture final state, compute throttled + cotenant_changed flags.
    Returns (post_state_dict, throttled_flag, cotenant_changed_flag)."""
    post = _rocm_smi_capture()

    # throttled: check if average observed SCLK < 97% of pinned target
    throttled = False
    try:
        clocks_data = post.get('clocks', {})
        if isinstance(clocks_data, dict):
            for device_key, device_data in clocks_data.items():
                if isinstance(device_data, dict):
                    sclk_val = device_data.get('sclk', device_data.get('SCLK', ''))
                    if isinstance(sclk_val, str) and 'Mhz' in sclk_val:
                        obs_mhz = float(sclk_val.replace('Mhz','').strip())
                        if obs_mhz < 0.97 * pinned_sclk_mhz:
                            throttled = True
                            break
    except Exception:
        throttled = None  # indeterminate

    # cotenant_changed: compare PID sets
    cotenant_changed = False
    try:
        pre_pids  = set(str(pre.get('pids',  {}) or {}).split())
        post_pids = set(str(post.get('pids', {}) or {}).split())
        cotenant_changed = (pre_pids != post_pids)
    except Exception:
        cotenant_changed = None  # indeterminate

    return post, throttled, cotenant_changed



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


def _build_gpu_cmd(impl, model, L, chi, sweeps,
                   pdmrg_warmup=None, pdmrg_polish=None, pdmrg_local=None,
                   pdmrg_recal=None):
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
    # PDMRG variants: always pass --warmup, --polish, --local-sweeps explicitly (CLAUDE.md rule)
    if impl.startswith("pdmrg"):
        if pdmrg_warmup is not None:
            cmd += ["--warmup", str(pdmrg_warmup)]
        if pdmrg_polish is not None:
            cmd += ["--polish", str(pdmrg_polish)]
        if pdmrg_local is not None:
            cmd += ["--local-sweeps", str(pdmrg_local)]
        if pdmrg_recal is not None:
            cmd += ["--recal", str(pdmrg_recal)]
    return cmd, None


def run_gpu(impl, model, L, chi, sweeps, warmup=True, timeout=None,
            pdmrg_warmup=None, pdmrg_polish=None, pdmrg_local=None,
            pdmrg_recal=None):
    """Run GPU benchmark with optional warmup."""
    # Warmup: same exe, tiny problem
    if warmup:
        warmup_cmd, _ = _build_gpu_cmd(impl, model, 4, 4, 2,
                                        pdmrg_warmup=pdmrg_warmup,
                                        pdmrg_polish=pdmrg_polish,
                                        pdmrg_local=pdmrg_local,
                                        pdmrg_recal=pdmrg_recal)
        if warmup_cmd:
            try:
                subprocess.run(warmup_cmd, capture_output=True, text=True, timeout=30)
            except Exception:
                pass

    cmd, err = _build_gpu_cmd(impl, model, L, chi, sweeps,
                               pdmrg_warmup=pdmrg_warmup,
                               pdmrg_polish=pdmrg_polish,
                               pdmrg_local=pdmrg_local,
                               pdmrg_recal=pdmrg_recal)
    if err:
        return {"energy": None, "time": None, "success": False, "error": err,
                "cmd": None}

    # cmd_rel: binary path relative to REPO_ROOT so rows are portable
    cmd_rel = list(cmd)
    if cmd_rel and cmd_rel[0].startswith(REPO_ROOT):
        cmd_rel[0] = os.path.relpath(cmd_rel[0], REPO_ROOT)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            return {"energy": None, "time": None, "success": False,
                    "error": (result.stderr or f"rc={result.returncode}")[-500:],
                    "cmd": cmd_rel}
        energy, wt = _parse_gpu_output(result.stdout)
        if energy is None:
            return {"energy": None, "time": None, "success": False,
                    "error": f"No energy: {result.stdout[-300:]}",
                    "cmd": cmd_rel}
        return {"energy": energy, "time": wt, "success": True, "cmd": cmd_rel}
    except subprocess.TimeoutExpired:
        return {"energy": None, "time": None, "success": False,
                "error": f"Timeout ({timeout}s)", "cmd": cmd_rel}


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

def _flush_impl_results(output_path, impl, arch, benchmark, ts, impl_results,
                        provenance=None, run_config=None, binary_meta=None):
    """Atomic-ish incremental write: per-impl JSON file, overwrite-in-place.

    Embeds a `provenance` dict (git commit, host, GPU, env vars) and a
    `run_config` dict (script-level args + PDMRG overrides) so results
    are reproducible to the exact code + configuration that produced them.
    `binary_meta` captures the binary's SHA256 / mtime so stale builds are
    detectable.
    """
    tmp = output_path + ".tmp"
    payload = {
        "implementation": impl,
        "architecture": arch,
        "benchmark": benchmark,
        "timestamp": ts,
        "results": impl_results,
    }
    if provenance is not None:
        payload["provenance"] = provenance
    if run_config is not None:
        payload["run_config"] = run_config
    if binary_meta is not None:
        payload["binary"] = binary_meta
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, output_path)


def run_all(impl_names, models, sizes_dict, do_warmup=True, repeats=1,
            output_dir=None, timestamp=None,
            pdmrg_warmup=None, pdmrg_polish=None, pdmrg_local=None,
            pdmrg_recal=None,
            provenance=None, run_config=None):
    """Run (impl × model × config) × repeats, flushing per-impl JSON after each config.

    Each result row stores a list `times` with all successful repeat timings plus
    `time_median` so downstream analysis can use robust statistics.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Build flat config list
    configs = []
    for impl in impl_names:
        for model in models:
            if model not in sizes_dict:
                continue
            for L, chi, sweeps in sizes_dict[model]:
                configs.append((impl, model, L, chi, sweeps))

    total = len(configs)
    current_group = None
    by_impl = {}  # impl -> list of result dicts
    per_impl_path = {}  # impl -> output JSON path
    per_impl_binary = {}  # impl -> binary_info dict (captured once per impl)

    for i, (impl, model, L, chi, sweeps) in enumerate(configs):
        group = (impl, model)
        if group != current_group:
            current_group = group
            print(f"\n{'='*65}")
            print(f"  {impl} — {model}")
            print(f"{'='*65}")

        if impl not in per_impl_path:
            per_impl_path[impl] = os.path.join(
                output_dir, f"{impl}_mi300x_challenge_{timestamp}.json")
            by_impl[impl] = []
            # Capture binary metadata once per impl for provenance
            if impl in GPU_IMPLS:
                per_impl_binary[impl] = binary_info(
                    os.path.join(REPO_ROOT, GPU_IMPLS[impl]))
            else:
                per_impl_binary[impl] = {"path": impl, "exists": None}

        label = f"L={L:3d} chi={chi:3d} sw={sweeps:2d}"
        print(f"  [{i+1:3d}/{total}] {label:25s}", flush=True)

        is_gpu = impl in GPU_IMPLS
        times = []
        energies = []
        rocm_states = []   # per-rep {pre, post, pin_rc, throttled, cotenant_changed}
        last_err = None
        last_cmd = None
        for rep in range(repeats):
            # rocm-smi pre-rep: capture state and attempt clock pin (no-ops on VF)
            rocm_pre, pin_rc = _rocm_smi_rep_preamble() if is_gpu else ({}, {})

            if is_gpu:
                # Warmup on the first rep of each config only (saves time)
                r = run_gpu(impl, model, L, chi, sweeps,
                            warmup=(do_warmup and rep == 0),
                            pdmrg_warmup=pdmrg_warmup,
                            pdmrg_polish=pdmrg_polish,
                            pdmrg_local=pdmrg_local,
                            pdmrg_recal=pdmrg_recal)
            else:
                r = run_quimb(impl, model, L, chi, sweeps, threads=1)

            # rocm-smi post-rep: capture state, compute throttled + cotenant_changed
            if is_gpu:
                rocm_post, throttled, cotenant_changed = _rocm_smi_rep_postamble(rocm_pre)
                r["rocm_pre"]          = rocm_pre
                r["rocm_post"]         = rocm_post
                r["pin_rc"]            = pin_rc
                r["throttled"]         = throttled
                r["cotenant_changed"]  = cotenant_changed

            if r.get("cmd") is not None:
                last_cmd = r["cmd"]
            if r.get("success"):
                times.append(r["time"])
                energies.append(r["energy"])
                if is_gpu:
                    rocm_states.append({
                        "rep": rep,
                        "pre":              r.get("rocm_pre"),
                        "post":             r.get("rocm_post"),
                        "pin_rc":           r.get("pin_rc"),
                        "throttled":        r.get("throttled"),
                        "cotenant_changed": r.get("cotenant_changed"),
                    })
                throttle_tag = " THROTTLED" if r.get("throttled") else ""
                cotenant_tag = " COTENANT_CHG" if r.get("cotenant_changed") else ""
                print(f"        rep {rep+1}/{repeats}: t={r['time']:.3f}s  E={r['energy']:.10f}{throttle_tag}{cotenant_tag}",
                      flush=True)
            else:
                last_err = r.get("error", "?")
                print(f"        rep {rep+1}/{repeats}: FAIL {last_err[:60]}",
                      flush=True)

        if times:
            stimes = sorted(times)
            t_med = stimes[len(stimes) // 2]
            t_min = stimes[0]
            row = {
                "impl": impl, "model": model, "L": L, "chi": chi,
                "sweeps": sweeps, "arch": "mi300x", "benchmark": "challenge",
                "threads": 1, "np": 1,
                "success": True,
                "time": t_med,          # retain legacy field (median)
                "time_median": t_med,
                "time_min": t_min,
                "times": times,
                "energy": energies[-1],
                "energies": energies,
                "repeats": repeats,
                "successful_reps": len(times),
                "cmd": last_cmd,        # exact argv that produced this row
                "rocm_states": rocm_states if is_gpu else None,
                "any_throttled": any(s.get("throttled") for s in rocm_states) if rocm_states else None,
                "any_cotenant_changed": any(s.get("cotenant_changed") for s in rocm_states) if rocm_states else None,
            }
        else:
            row = {
                "impl": impl, "model": model, "L": L, "chi": chi,
                "sweeps": sweeps, "arch": "mi300x", "benchmark": "challenge",
                "threads": 1, "np": 1,
                "success": False,
                "time": None, "time_median": None, "time_min": None,
                "times": [], "energy": None, "energies": [],
                "repeats": repeats, "successful_reps": 0,
                "error": last_err,
                "cmd": last_cmd,
            }
        if model == "josephson":
            row["nmax"] = JOSEPHSON_NMAX

        by_impl[impl].append(row)

        # Flush immediately so a crash loses at most the current config
        _flush_impl_results(per_impl_path[impl], impl, "mi300x", "challenge",
                            timestamp, by_impl[impl],
                            provenance=provenance,
                            run_config=run_config,
                            binary_meta=per_impl_binary.get(impl))

    return by_impl, per_impl_path


def main():
    parser = argparse.ArgumentParser(description="MI300X Challenge Benchmarks")
    parser.add_argument("--impl", type=str, default=None,
                        help="Comma-separated impls (default: all 6 GPU + quimb)")
    parser.add_argument("--model", type=str, default=None,
                        help="heisenberg, josephson, tfim (default: all)")
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Number of repeat timings per config (default: 1)")
    parser.add_argument("--trim", action="store_true",
                        help="Use ULTRA_TRIM_SIZES (18 configs) instead of full 44")
    parser.add_argument("--pdmrg-warmup", type=int, default=None,
                        help="Override --warmup N for pdmrg impls (MUST be explicit)")
    parser.add_argument("--pdmrg-polish", type=int, default=None,
                        help="Override --polish N for pdmrg impls (MUST be explicit)")
    parser.add_argument("--pdmrg-local", type=int, default=None,
                        help="Override --local-sweeps N for pdmrg impls")
    parser.add_argument("--pdmrg-recal", type=int, default=None,
                        help="Override --recal N for pdmrg impls (serial recalibration every N outer iters)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Optional suffix for output filenames")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(REPO_ROOT, "benchmarks", "paper_results", "mi300x", "challenge"))
    args = parser.parse_args()

    sizes_dict = ULTRA_TRIM_SIZES if args.trim else CHALLENGE_SIZES
    grid_label = "ULTRA_TRIM (18 configs)" if args.trim else "CHALLENGE (44 configs)"

    if args.impl:
        impl_names = [x.strip() for x in args.impl.split(",")]
    else:
        impl_names = sorted(GPU_IMPLS.keys()) + ["quimb-dmrg1"]

    models = list(sizes_dict.keys())
    if args.model:
        models = [x.strip() for x in args.model.split(",")]

    # Timestamp: fixed for this invocation so one --repeats run produces one JSON
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if args.tag:
        ts = f"{ts}_{args.tag}"

    n_configs = sum(len(sizes_dict[m]) for m in models) * len(impl_names)

    print(f"\n{'#'*65}")
    print(f"  MI300X Challenge Benchmarks")
    print(f"  Grid:             {grid_label}")
    print(f"  Implementations:  {', '.join(impl_names)}")
    print(f"  Models:           {', '.join(models)}")
    print(f"  Configs:          {n_configs}  (× {args.repeats} repeats)")
    pdmrg_warmup = args.pdmrg_warmup
    pdmrg_polish = args.pdmrg_polish
    pdmrg_local = args.pdmrg_local
    pdmrg_recal = args.pdmrg_recal

    print(f"  GPU warmup:       {'OFF' if args.skip_warmup else 'ON'}")
    if any(v is not None for v in [pdmrg_warmup, pdmrg_polish, pdmrg_local, pdmrg_recal]):
        print(f"  PDMRG overrides:  warmup={pdmrg_warmup}, polish={pdmrg_polish}, local={pdmrg_local}, recal={pdmrg_recal}")
    print(f"  Timestamp:        {ts}")
    print(f"  Output:           {args.output_dir}")
    print(f"{'#'*65}")

    provenance = provenance_block(repo_root=REPO_ROOT, script_argv=sys.argv[1:])
    run_config = {
        "grid":            grid_label,
        "impl_names":      impl_names,
        "models":          models,
        "repeats":         args.repeats,
        "skip_warmup":     args.skip_warmup,
        "trim":            args.trim,
        "tag":             args.tag,
        "pdmrg_warmup":    pdmrg_warmup,
        "pdmrg_polish":    pdmrg_polish,
        "pdmrg_local":     pdmrg_local,
        "pdmrg_recal":     pdmrg_recal,
        "josephson_nmax":  JOSEPHSON_NMAX,
    }
    if provenance["git"].get("dirty"):
        print(f"  WARNING:          git repo is dirty "
              f"({len(provenance['git'].get('dirty_files', []))} files modified)")
    print(f"  Git commit:       {provenance['git'].get('commit_short', '?')}"
          f" ({provenance['git'].get('branch', '?')})")

    by_impl, per_impl_path = run_all(
        impl_names, models, sizes_dict,
        do_warmup=not args.skip_warmup, repeats=args.repeats,
        output_dir=args.output_dir, timestamp=ts,
        pdmrg_warmup=pdmrg_warmup, pdmrg_polish=pdmrg_polish,
        pdmrg_local=pdmrg_local, pdmrg_recal=pdmrg_recal,
        provenance=provenance, run_config=run_config)

    print(f"\n{'='*65}")
    print(f"  Final per-impl files:")
    for impl, path in per_impl_path.items():
        print(f"    {impl}: {path}")
    print(f"{'='*65}")

    total_rows = sum(len(v) for v in by_impl.values())
    ok = sum(1 for v in by_impl.values() for r in v if r["success"])
    fail = total_rows - ok
    print(f"\n  Done: {total_rows} configs, {ok} passed, {fail} failed")


if __name__ == "__main__":
    main()
