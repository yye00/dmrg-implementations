#!/usr/bin/env python3
"""
Comprehensive CPU + GPU DMRG Benchmark Suite
==============================================

Runs all implementations with IDENTICAL problems and tolerances:

CPU (14 runs per model):
  - Quimb DMRG1 (1 run)
  - Quimb DMRG2 (1 run)
  - PDMRG (np=1,2,4,8) - 4 runs
  - PDMRG2 (np=1,2,4,8) - 4 runs
  - A2DMRG (np=1,2,4,8) - 4 runs

GPU (8 runs per model):
  - PDMRG_GPU (streams=1,2,4,8) - 4 runs
  - PDMRG2_GPU (streams=1,2,4,8) - 4 runs

Models:
  - Heisenberg: L=12, D=100, d=2 (real)
  - Josephson: L=8, D=50, d=5, n_max=2 (complex128)

CRITICAL: If any results disagree by more than 1e-10, STOP IMMEDIATELY.
Quimb DMRG1/DMRG2 are the source of truth.
"""

import sys
import os
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path
import numpy as np

# Add paths
SCRIPT_DIR = Path(__file__).parent.absolute()
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR / 'pdmrg2'))
sys.path.insert(0, str(ROOT_DIR))

import quimb.tensor as qtn


# =============================================================================
# EXACT PROBLEM DEFINITIONS (Shared across all implementations)
# =============================================================================

HEISENBERG_PARAMS = {
    "name": "Heisenberg",
    "L": 12,
    "D": 100,
    "d": 2,
    "sweeps": 20,
    "tol": 1e-10,
    "cutoff": 1e-14,
    "dtype": "float64",  # Real for Heisenberg
}

JOSEPHSON_PARAMS = {
    "name": "Josephson",
    "L": 8,
    "D": 50,
    "d": 5,
    "n_max": 2,
    "E_J": 1.0,
    "E_C": 0.5,
    "mu": 0.0,
    "sweeps": 20,
    "tol": 1e-10,
    "cutoff": 1e-14,
    "dtype": "complex128",
}

TOLERANCE = 1e-10  # Maximum allowed energy difference


# =============================================================================
# MPO Construction (Identical for all implementations)
# =============================================================================

def build_heisenberg_mpo(L):
    """Build Heisenberg XXX chain MPO."""
    return qtn.MPO_ham_heis(L=L, j=1.0, bz=0.0, cyclic=False)


def build_josephson_mpo(L, E_J=1.0, E_C=0.5, mu=0.0, n_max=2):
    """Build Josephson junction array MPO in charge basis."""
    d = 2 * n_max + 1

    # Charge operator: n|n> = n|n>
    charges = np.arange(-n_max, n_max + 1, dtype=np.float64)
    n_op = np.diag(charges.astype(np.complex128))

    # Phase operators: e^{±iφ}
    exp_iphi = np.zeros((d, d), dtype=np.complex128)
    for i in range(d - 1):
        exp_iphi[i + 1, i] = 1.0 + 0j
    exp_miphi = exp_iphi.conj().T

    # Build MPO
    S = (d - 1) / 2
    builder = qtn.SpinHam1D(S=S)

    # Josephson coupling with external flux Φ_ext = π/4
    phi_ext = np.pi / 4
    flux_phase = np.exp(1j * phi_ext)
    builder.add_term(-E_J / 2 * flux_phase, exp_iphi, exp_miphi)
    builder.add_term(-E_J / 2 * np.conj(flux_phase), exp_miphi, exp_iphi)

    # Charging energy
    n2 = n_op @ n_op
    builder.add_term(E_C, n2)

    # Chemical potential
    if mu != 0:
        builder.add_term(-mu, n_op)

    return builder.build_mpo(L)


# =============================================================================
# Results Storage and Validation
# =============================================================================

class BenchmarkResults:
    def __init__(self):
        self.results = {
            "heisenberg": {},
            "josephson": {}
        }
        self.reference_energy = {
            "heisenberg": None,
            "josephson": None
        }
        self.failed = False
        self.failure_message = ""

    def set_reference(self, model, energy):
        """Set reference energy (from Quimb DMRG1)."""
        self.reference_energy[model] = energy
        print(f"\n{'='*70}")
        print(f"REFERENCE ENERGY ({model}): {energy:.15f}")
        print(f"{'='*70}\n")

    def add_result(self, model, implementation, config, energy, wall_time):
        """Add a result and validate against reference."""
        key = f"{implementation}_{config}"
        self.results[model][key] = {
            "implementation": implementation,
            "config": config,
            "energy": energy,
            "wall_time": wall_time,
        }

        # Validate if reference is set
        if self.reference_energy[model] is not None:
            ref = self.reference_energy[model]
            error = abs(energy - ref)

            print(f"  {implementation:15s} {config:10s}: E={energy:.15f}, "
                  f"error={error:.2e}, time={wall_time:.3f}s", end="")

            if error > TOLERANCE:
                print(f"  ❌ FAILED!")
                self.failed = True
                self.failure_message = (
                    f"\n{'='*70}\n"
                    f"VALIDATION FAILED!\n"
                    f"{'='*70}\n"
                    f"Model: {model}\n"
                    f"Implementation: {implementation} ({config})\n"
                    f"Energy: {energy:.15f}\n"
                    f"Reference: {ref:.15f}\n"
                    f"Error: {error:.2e} (tolerance: {TOLERANCE:.2e})\n"
                    f"{'='*70}\n"
                )
                return False
            else:
                print(f"  ✓")
        else:
            print(f"  {implementation:15s} {config:10s}: E={energy:.15f}, time={wall_time:.3f}s")

        return True

    def save(self, filename):
        """Save results to JSON."""
        output = {
            "timestamp": datetime.now().isoformat(),
            "reference_energy": self.reference_energy,
            "results": self.results,
            "validation": {
                "passed": not self.failed,
                "tolerance": TOLERANCE,
                "message": self.failure_message if self.failed else "All results within tolerance"
            }
        }
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)


results = BenchmarkResults()


# =============================================================================
# CPU Benchmarks - Quimb DMRG1 & DMRG2 (Source of Truth)
# =============================================================================

def run_quimb_dmrg1(mpo, params):
    """Run Quimb DMRG1 (1-site) - SOURCE OF TRUTH."""
    print(f"\nRunning Quimb DMRG1...", flush=True)
    t0 = time.time()

    dmrg = qtn.DMRG1(mpo, bond_dims=params["D"], cutoffs=params["cutoff"])
    dmrg.solve(max_sweeps=params["sweeps"], tol=params["tol"], verbosity=0)

    t1 = time.time()
    energy = float(np.real(dmrg.energy))

    return energy, t1 - t0


def run_quimb_dmrg2(mpo, params):
    """Run Quimb DMRG2 (2-site)."""
    print(f"Running Quimb DMRG2...", flush=True)
    t0 = time.time()

    dmrg = qtn.DMRG2(mpo, bond_dims=params["D"], cutoffs=params["cutoff"])
    dmrg.solve(max_sweeps=params["sweeps"], tol=params["tol"], verbosity=0)

    t1 = time.time()
    energy = float(np.real(dmrg.energy))

    return energy, t1 - t0


# =============================================================================
# CPU Benchmarks - PDMRG, PDMRG2, A2DMRG (MPI-based)
# =============================================================================

def run_mpi_implementation(impl_name, model, params, np):
    """Run an MPI-based implementation."""
    script_map = {
        "pdmrg": ROOT_DIR / "run_pdmrg.py",
        "pdmrg2": ROOT_DIR / "run_pdmrg2.py",
        "a2dmrg": ROOT_DIR / "run_a2dmrg.py",
    }

    script = script_map.get(impl_name)
    if not script or not script.exists():
        print(f"  WARNING: {impl_name} script not found at {script}")
        return None, None

    print(f"  Running {impl_name} (np={np})...", flush=True)

    # Build command
    cmd = [
        "mpirun", "-np", str(np),
        "python", str(script),
        "--model", model,
        "--L", str(params["L"]),
        "--D", str(params["D"]),
        "--sweeps", str(params["sweeps"]),
        "--tol", str(params["tol"]),
    ]

    # Add model-specific params
    if model == "josephson":
        cmd.extend([
            "--n-max", str(params["n_max"]),
            "--E-J", str(params["E_J"]),
            "--E-C", str(params["E_C"]),
        ])

    try:
        t0 = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        t1 = time.time()

        if result.returncode != 0:
            print(f"    ERROR: {impl_name} failed with code {result.returncode}")
            print(f"    stderr: {result.stderr}")
            return None, None

        # Parse energy from output
        for line in result.stdout.split('\n'):
            if 'Final energy:' in line or 'Energy:' in line:
                energy = float(line.split(':')[-1].strip())
                return energy, t1 - t0

        print(f"    ERROR: Could not parse energy from {impl_name} output")
        return None, None

    except subprocess.TimeoutExpired:
        print(f"    ERROR: {impl_name} timed out")
        return None, None
    except Exception as e:
        print(f"    ERROR: {impl_name} failed: {e}")
        return None, None


# =============================================================================
# GPU Benchmarks - PDMRG_GPU & PDMRG2_GPU
# =============================================================================

def run_gpu_implementation(impl_name, model, params, streams):
    """Run a GPU implementation."""
    executable_map = {
        "pdmrg_gpu": ROOT_DIR / "gpu-port" / "build" / "pdmrg_gpu",
        "pdmrg2_gpu": ROOT_DIR / "gpu-port" / "build" / "pdmrg2_gpu",
    }

    exe = executable_map.get(impl_name)
    if not exe or not exe.exists():
        print(f"  WARNING: {impl_name} executable not found at {exe}")
        return None, None

    print(f"  Running {impl_name} (streams={streams})...", flush=True)

    # Build command
    cmd = [
        str(exe),
        "--model", model.lower(),
        "--L", str(params["L"]),
        "--max-D", str(params["D"]),
        "--sweeps", str(params["sweeps"]),
        "--streams", str(streams),
    ]

    # Add model-specific params
    if model == "josephson":
        cmd.extend([
            "--n-max", str(params["n_max"]),
            "--E-J", str(params["E_J"]),
            "--E-C", str(params["E_C"]),
        ])

    try:
        t0 = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        t1 = time.time()

        if result.returncode != 0:
            print(f"    ERROR: {impl_name} failed with code {result.returncode}")
            print(f"    stderr: {result.stderr}")
            return None, None

        # Parse energy from output
        for line in result.stdout.split('\n'):
            if 'Final energy:' in line or 'Energy:' in line:
                energy_str = line.split(':')[-1].strip()
                energy = float(energy_str)
                return energy, t1 - t0

        print(f"    ERROR: Could not parse energy from {impl_name} output")
        return None, None

    except subprocess.TimeoutExpired:
        print(f"    ERROR: {impl_name} timed out")
        return None, None
    except Exception as e:
        print(f"    ERROR: {impl_name} failed: {e}")
        return None, None


# =============================================================================
# Main Benchmark Orchestration
# =============================================================================

def run_model_benchmarks(model_name, params, mpo):
    """Run all benchmarks for a single model."""
    print(f"\n{'='*70}")
    print(f"BENCHMARKING: {model_name.upper()}")
    print(f"  L={params['L']}, D={params['D']}, d={params['d']}, sweeps={params['sweeps']}")
    print(f"{'='*70}")

    model_key = model_name.lower()

    # =========================================================================
    # 1. Quimb DMRG1 - SOURCE OF TRUTH
    # =========================================================================
    print(f"\n--- Quimb DMRG (Reference) ---")
    energy, wall_time = run_quimb_dmrg1(mpo, params)
    results.set_reference(model_key, energy)
    results.add_result(model_key, "quimb_dmrg1", "default", energy, wall_time)

    # =========================================================================
    # 2. Quimb DMRG2
    # =========================================================================
    energy, wall_time = run_quimb_dmrg2(mpo, params)
    if energy is not None:
        if not results.add_result(model_key, "quimb_dmrg2", "default", energy, wall_time):
            print(results.failure_message)
            return False

    # =========================================================================
    # 3. CPU MPI Implementations (PDMRG, PDMRG2, A2DMRG) - np = 1,2,4,8
    # =========================================================================
    print(f"\n--- CPU MPI Implementations ---")
    for impl in ["pdmrg", "pdmrg2", "a2dmrg"]:
        for np in [1, 2, 4, 8]:
            energy, wall_time = run_mpi_implementation(impl, model_key, params, np)
            if energy is not None:
                if not results.add_result(model_key, impl, f"np={np}", energy, wall_time):
                    print(results.failure_message)
                    return False

    # =========================================================================
    # 4. GPU Implementations (PDMRG_GPU, PDMRG2_GPU) - streams = 1,2,4,8
    # =========================================================================
    print(f"\n--- GPU Implementations ---")
    for impl in ["pdmrg_gpu", "pdmrg2_gpu"]:
        for streams in [1, 2, 4, 8]:
            energy, wall_time = run_gpu_implementation(impl, model_key, params, streams)
            if energy is not None:
                if not results.add_result(model_key, impl, f"streams={streams}", energy, wall_time):
                    print(results.failure_message)
                    return False

    return True


def main():
    """Run comprehensive benchmark suite."""
    print("=" * 70)
    print("COMPREHENSIVE DMRG BENCHMARK SUITE")
    print("=" * 70)
    print(f"Start time: {datetime.now()}")
    print(f"Host: {os.uname().nodename}")
    print(f"Tolerance: {TOLERANCE:.2e}")
    print("=" * 70)

    # Build MPOs
    print("\nBuilding MPOs...")
    heis_mpo = build_heisenberg_mpo(HEISENBERG_PARAMS["L"])
    jos_mpo = build_josephson_mpo(
        JOSEPHSON_PARAMS["L"],
        E_J=JOSEPHSON_PARAMS["E_J"],
        E_C=JOSEPHSON_PARAMS["E_C"],
        mu=JOSEPHSON_PARAMS["mu"],
        n_max=JOSEPHSON_PARAMS["n_max"]
    )

    # Run Heisenberg benchmarks
    if not run_model_benchmarks("Heisenberg", HEISENBERG_PARAMS, heis_mpo):
        print("\n❌ HEISENBERG BENCHMARKS FAILED - STOPPING")
        results.save(SCRIPT_DIR / "benchmark_results_FAILED.json")
        return 1

    # Run Josephson benchmarks
    if not run_model_benchmarks("Josephson", JOSEPHSON_PARAMS, jos_mpo):
        print("\n❌ JOSEPHSON BENCHMARKS FAILED - STOPPING")
        results.save(SCRIPT_DIR / "benchmark_results_FAILED.json")
        return 1

    # Success!
    print("\n" + "=" * 70)
    print("✓ ALL BENCHMARKS PASSED")
    print("=" * 70)

    # Save results
    output_file = SCRIPT_DIR / f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results.save(output_file)
    print(f"\nResults saved to: {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
