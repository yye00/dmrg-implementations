#!/usr/bin/env python3
"""
Full validation + timing benchmark across all DMRG implementations.
Orchestrates from local, all execution happens on remote MI300X host via SSH.

Usage: python3 benchmarks/run_full_validation.py
"""
import subprocess, time, json, sys, os, re

REMOTE = "hotaisle@23.183.40.81"
REMOTE_REPO = "/home/hotaisle/dmrg-implementations"

# ============================================================================
# Test matrix
# ============================================================================
MODELS = {
    "heisenberg": {
        "small":  {"L": 12, "chi": 20,  "sweeps": 30},
        "medium": {"L": 20, "chi": 50,  "sweeps": 30},
    },
    "josephson": {
        "small":  {"L": 8,  "chi": 20,  "sweeps": 30, "nmax": 2},
        "medium": {"L": 12, "chi": 50,  "sweeps": 40, "nmax": 2},
    },
}

SERIAL_CPU = ["quimb-dmrg1", "quimb-dmrg2"]
PARALLEL_CPU = ["pdmrg", "pdmrg2"]
SERIAL_GPU = ["dmrg-gpu", "dmrg2-gpu"]
PARALLEL_GPU = ["pdmrg-gpu", "pdmrg2-gpu"]
NP_VALUES = [2, 4]
BLAS_THREADS = [1, 2, 4, 8]
TOTAL_CORES = 13

# ============================================================================
# SSH helper
# ============================================================================

def _blas_env_prefix(blas_threads=None):
    """Return shell env var prefix for BLAS/OMP thread control, or empty string."""
    if blas_threads is None:
        return ""
    return f"OPENBLAS_NUM_THREADS={blas_threads} OMP_NUM_THREADS={blas_threads} "

def _mpi_env_flags(blas_threads=None):
    """Return mpirun -x flags for BLAS/OMP thread control, or empty string."""
    if blas_threads is None:
        return ""
    return f"-x OPENBLAS_NUM_THREADS={blas_threads} -x OMP_NUM_THREADS={blas_threads} "

def ssh_run(cmd, timeout=600, blas_threads=None):
    """Run command on remote host, return CompletedProcess."""
    full_cmd = _blas_env_prefix(blas_threads) + cmd
    return subprocess.run(
        ["ssh", REMOTE, full_cmd],
        capture_output=True, text=True, timeout=timeout
    )

def ssh_run_script(script, timeout=600, blas_threads=None):
    """Write script to remote /tmp and run it."""
    import hashlib
    h = hashlib.md5(script.encode()).hexdigest()[:8]
    tmpfile = f"/tmp/_bench_{h}.py"
    # Escape for shell
    subprocess.run(
        ["ssh", REMOTE, f"cat > {tmpfile} << 'HEREDOC_END'\n{script}\nHEREDOC_END"],
        capture_output=True, timeout=10
    )
    return ssh_run(f"cd {REMOTE_REPO} && python3 {tmpfile}", timeout=timeout, blas_threads=blas_threads)

def ssh_run_mpi_script(script, np_val, timeout=600, blas_threads=None):
    """Write script to remote /tmp and run with mpirun."""
    import hashlib
    h = hashlib.md5(script.encode()).hexdigest()[:8]
    tmpfile = f"/tmp/_bench_{h}.py"
    subprocess.run(
        ["ssh", REMOTE, f"cat > {tmpfile} << 'HEREDOC_END'\n{script}\nHEREDOC_END"],
        capture_output=True, timeout=10
    )
    env_flags = _mpi_env_flags(blas_threads)
    return ssh_run(
        f"cd {REMOTE_REPO} && mpirun --allow-run-as-root --oversubscribe {env_flags}-np {np_val} python3 {tmpfile} 2>/dev/null",
        timeout=timeout
    )

# ============================================================================
# Runners
# ============================================================================

def run_quimb(model, size, algo, blas_threads=None):
    p = MODELS[model][size]
    L, chi, sweeps = p["L"], p["chi"], p["sweeps"]

    if model == "heisenberg":
        script = f"""
import time, quimb.tensor as qtn
dmrg = qtn.DMRG{'1' if algo == 'dmrg1' else '2'}(qtn.MPO_ham_heis(L={L}, j=1.0, bz=0.0, cyclic=False), bond_dims=[{chi}], cutoffs=[1e-12])
t0 = time.time()
dmrg.solve(max_sweeps={sweeps}, tol=1e-10, verbosity=0)
t1 = time.time()
print(f"ENERGY={{dmrg.energy:.15f}}")
print(f"TIME={{t1-t0:.3f}}")
"""
    else:
        nmax = p["nmax"]
        script = f"""
import time, sys
sys.path.insert(0, '{REMOTE_REPO}')
from benchmarks.lib.models import build_josephson_mpo
import quimb.tensor as qtn
mpo = build_josephson_mpo({L}, n_max={nmax})
dmrg = qtn.DMRG{'1' if algo == 'dmrg1' else '2'}(mpo, bond_dims=[{chi}], cutoffs=[1e-12])
t0 = time.time()
dmrg.solve(max_sweeps={sweeps}, tol=1e-10, verbosity=0)
t1 = time.time()
e = dmrg.energy
if hasattr(e, 'imag') and abs(e.imag) > 1e-10:
    print(f"ERROR: complex energy with Im(E)={{e.imag:.2e}} — non-Hermitian MPO?")
else:
    e = e.real if hasattr(e, 'real') else e
    print(f"ENERGY={{e:.15f}}")
    print(f"TIME={{t1-t0:.3f}}")
"""
    result = ssh_run_script(script, blas_threads=blas_threads)
    return parse_python_output(result)


def run_pdmrg(impl, model, size, np_val, blas_threads=None):
    p = MODELS[model][size]
    L, chi, sweeps = p["L"], p["chi"], p["sweeps"]
    package_dir = f"{REMOTE_REPO}/{impl}"

    if model == "heisenberg":
        model_setup = f"""
import quimb.tensor as qtn
mpo = qtn.MPO_ham_heis(L={L}, j=1.0, bz=0.0, cyclic=False)
"""
    else:
        nmax = p["nmax"]
        model_setup = f"""
import sys
sys.path.insert(0, '{REMOTE_REPO}')
from benchmarks.lib.models import build_josephson_mpo
mpo = build_josephson_mpo({L}, n_max={nmax})
"""

    script = f"""
import sys, time
sys.path.insert(0, '{package_dir}')
sys.path.insert(0, '{REMOTE_REPO}')
from mpi4py import MPI
{model_setup}
from pdmrg.dmrg import pdmrg_main
t0 = time.time()
energy, pmps = pdmrg_main({L}, mpo, max_sweeps={sweeps}, bond_dim={chi}, tol=1e-10, verbose=False)
t1 = time.time()
if MPI.COMM_WORLD.Get_rank() == 0:
    print(f"ENERGY={{energy:.15f}}")
    print(f"TIME={{t1-t0:.3f}}")
"""
    # For MPI, cap BLAS threads to floor(TOTAL_CORES / np) to avoid oversubscription
    effective_threads = blas_threads
    if effective_threads is not None:
        max_threads_per_rank = TOTAL_CORES // np_val
        effective_threads = min(effective_threads, max(1, max_threads_per_rank))
    timeout = 900 if np_val > 2 else 600
    result = ssh_run_mpi_script(script, np_val, timeout=timeout, blas_threads=effective_threads)
    return parse_python_output(result)


def run_gpu_serial(impl, model, size):
    p = MODELS[model][size]
    L, chi, sweeps = p["L"], p["chi"], p["sweeps"]

    exe_map = {
        "dmrg-gpu": f"{REMOTE_REPO}/dmrg-gpu/build/dmrg_gpu",
        "dmrg2-gpu": f"{REMOTE_REPO}/dmrg2-gpu/build/dmrg2_gpu",
    }
    exe = exe_map[impl]

    cmd = f"{exe} {L} {chi} {sweeps}"
    if model == "josephson":
        cmd += f" --josephson --nmax {p['nmax']}"

    result = ssh_run(cmd)
    return parse_gpu_output(result)


def run_gpu_parallel(impl, model, size, n_streams):
    p = MODELS[model][size]
    L, chi, sweeps = p["L"], p["chi"], p["sweeps"]

    exe_map = {
        "pdmrg-gpu": f"{REMOTE_REPO}/pdmrg-gpu/build/pdmrg_gpu",
        "pdmrg2-gpu": f"{REMOTE_REPO}/pdmrg2-gpu/build/pdmrg2_gpu",
    }
    exe = exe_map[impl]

    cmd = f"{exe} {L} {chi} {sweeps} --segments {n_streams}"
    if model == "josephson":
        cmd += f" --josephson --nmax {p['nmax']}"

    result = ssh_run(cmd)
    return parse_gpu_output(result)


# ============================================================================
# Parsers
# ============================================================================

def parse_python_output(result):
    out = result.stdout + result.stderr
    energy = extract_float(out, r"ENERGY=([-\d.]+)")
    wall = extract_float(out, r"TIME=([\d.]+)")
    if energy is None:
        return {"error": (out[-500:] if len(out) > 500 else out).strip()}
    return {"energy": energy, "time": wall}


def parse_gpu_output(result):
    out = result.stdout + result.stderr
    energy = extract_float(out, r"Final energy:\s*([-\d.]+)")
    wall = extract_float(out, r"Total wall time:\s*([\d.]+)")
    error = extract_float(out, r"Absolute error:\s*([\d.e+-]+)")
    passed = "PASS" in out
    if energy is None:
        return {"error": (out[-500:] if len(out) > 500 else out).strip()}
    return {"energy": energy, "time": wall, "abs_error": error, "passed": passed}


def extract_float(text, pattern):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


# ============================================================================
# Main
# ============================================================================

def main():
    results = {}

    for model in ["heisenberg", "josephson"]:
        for size in ["small", "medium"]:
            key = f"{model}-{size}"
            params = MODELS[model][size]
            print(f"\n{'='*70}")
            print(f"  {model.upper()} {size.upper()} — L={params['L']}, chi={params['chi']}")
            print(f"{'='*70}")
            results[key] = {}

            # --- Serial CPU (sweep over BLAS thread counts) ---
            for t in BLAS_THREADS:
                for impl in SERIAL_CPU:
                    algo = "dmrg1" if "dmrg1" in impl else "dmrg2"
                    label = f"{impl} t={t}"
                    print(f"  {label:25s} ... ", end="", flush=True)
                    try:
                        r = run_quimb(model, size, algo, blas_threads=t)
                        results[key][label] = r
                        if "error" in r:
                            print(f"ERROR: {r['error'][:80]}")
                        else:
                            print(f"E={r['energy']:.10f}  t={r['time']:.1f}s")
                    except Exception as e:
                        print(f"EXCEPTION: {e}")
                        results[key][label] = {"error": str(e)}

            # --- Parallel CPU (sweep over BLAS thread counts) ---
            for t in BLAS_THREADS:
                for impl in PARALLEL_CPU:
                    for np_val in NP_VALUES:
                        label = f"{impl} np={np_val} t={t}"
                        print(f"  {label:25s} ... ", end="", flush=True)
                        try:
                            r = run_pdmrg(impl, model, size, np_val, blas_threads=t)
                            results[key][label] = r
                            if "error" in r:
                                print(f"ERROR: {r['error'][:80]}")
                            else:
                                print(f"E={r['energy']:.10f}  t={r['time']:.1f}s")
                        except Exception as e:
                            print(f"EXCEPTION: {e}")
                            results[key][label] = {"error": str(e)}

            # --- Serial GPU ---
            for impl in SERIAL_GPU:
                print(f"  {impl:25s} ... ", end="", flush=True)
                try:
                    r = run_gpu_serial(impl, model, size)
                    results[key][impl] = r
                    if "error" in r:
                        print(f"ERROR: {r['error'][:80]}")
                    else:
                        status = "PASS" if r.get("passed") else f"err={r.get('abs_error', '?')}"
                        print(f"E={r['energy']:.10f}  t={r['time']:.1f}s  {status}")
                except Exception as e:
                    print(f"EXCEPTION: {e}")
                    results[key][impl] = {"error": str(e)}

            # --- Parallel GPU ---
            for impl in PARALLEL_GPU:
                for ns in NP_VALUES:
                    label = f"{impl} s={ns}"
                    print(f"  {label:25s} ... ", end="", flush=True)
                    try:
                        r = run_gpu_parallel(impl, model, size, ns)
                        results[key][label] = r
                        if "error" in r:
                            print(f"ERROR: {r['error'][:80]}")
                        else:
                            status = "PASS" if r.get("passed") else f"err={r.get('abs_error', '?')}"
                            print(f"E={r['energy']:.10f}  t={r['time']:.1f}s  {status}")
                    except Exception as e:
                        print(f"EXCEPTION: {e}")
                        results[key][label] = {"error": str(e)}

    # Save results
    local_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outfile = os.path.join(local_repo, "benchmarks/results/full_validation.json")
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {outfile}")

    # Print summary table
    print_summary(results)


def print_summary(results):
    print(f"\n{'='*90}")
    print(f"  ACCURACY SUMMARY")
    print(f"{'='*90}")

    for key, impls in results.items():
        print(f"\n  {key}")
        # Find a reference energy: prefer quimb-dmrg2 t=1, fall back to any quimb-dmrg2
        ref = None
        for rlabel in sorted(impls.keys()):
            if rlabel.startswith("quimb-dmrg2"):
                ref = impls[rlabel].get("energy")
                if ref is not None:
                    break
        print(f"  {'Implementation':<30s} {'Energy':>18s} {'ΔE vs ref':>12s} {'Time':>8s}")
        print(f"  {'-'*30} {'-'*18} {'-'*12} {'-'*8}")
        for label, r in impls.items():
            if "error" in r:
                print(f"  {label:<30s} {'ERROR':>18s}")
                continue
            e = r["energy"]
            t = r.get("time", 0) or 0
            if ref is not None:
                de = abs(e - ref)
                de_str = f"{de:.2e}"
            else:
                de_str = "N/A"
            print(f"  {label:<30s} {e:>18.10f} {de_str:>12s} {t:>7.1f}s")


if __name__ == "__main__":
    main()
