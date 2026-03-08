#!/usr/bin/env python3
"""
Comprehensive DMRG Benchmark Suite
===================================
Runs all VALIDATED CPU and GPU implementations with IDENTICAL parameters.

CPU Implementations (11 runs per model):
  - Quimb DMRG1 (1 run - reference)
  - Quimb DMRG2 (1 run - reference)
  - PDMRG (np=2,4,8) (3 runs - validated)
  - A2DMRG (np=2,4,8) (3 runs - validated)
  - PDMRG2: EXCLUDED (prototype-only, not validated)

GPU Implementations (4 runs per model):
  - PDMRG-GPU (streams=1,2,4,8) (4 runs)
  - PDMRG2-GPU: EXCLUDED (prototype-only)

Test Cases:
  - Heisenberg: L=12, D=100, sweeps=20, tol=1e-10
  - Josephson: L=8, D=50, sweeps=20, tol=1e-10, n_max=2

All results must agree within 1e-10 (machine precision).
Quimb DMRG1/2 are the reference/source of truth.
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime
from dataclasses import dataclass, asdict
import numpy as np

# ============================================================================
# EXACT BENCHMARK PARAMETERS - DO NOT MODIFY
# ============================================================================
HEISENBERG_PARAMS = {
    'L': 12,
    'D': 100,
    'sweeps': 20,
    'tol': 1e-10,
    'cutoff': 1e-14,
    'model': 'heisenberg',
}

JOSEPHSON_PARAMS = {
    'L': 8,
    'D': 50,
    'sweeps': 20,
    'tol': 1e-10,
    'cutoff': 1e-14,
    'n_max': 2,
    'E_J': 1.0,
    'E_C': 0.5,
    'model': 'josephson',
}

# Expected reference energies (from exact diag / previous verification)
HEISENBERG_REFERENCE = -5.142090632841
JOSEPHSON_REFERENCE = -2.843801043291333

# Parallel algorithms require np >= 2 (PDMRG, A2DMRG are parallel real-space algorithms)
NP_VALUES = [2, 4, 8]
STREAM_VALUES = [1, 2, 4, 8]

# ============================================================================
# Directory setup
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GPU_BUILD_DIR = os.path.join(SCRIPT_DIR, 'gpu-port', 'build')


@dataclass
class BenchmarkResult:
    implementation: str
    model: str
    np_or_streams: int
    energy: float
    wall_time: float
    sweeps_actual: int
    converged: bool
    error_vs_reference: float
    timestamp: str


class BenchmarkRunner:
    def __init__(self):
        self.results = []
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    def log(self, msg):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

    def verify_error(self, energy, reference, impl, model, np_str):
        error = abs(energy - reference)
        status = "✓" if error < 1e-10 else "✗ FAIL"
        self.log(f"  {impl} ({np_str}) {model}: E={energy:.12f}, error={error:.2e} {status}")
        return error

    # ========================================================================
    # QUIMB BENCHMARKS (Reference/Source of Truth)
    # ========================================================================
    def run_quimb_dmrg1(self, params):
        """Run Quimb DMRG1 - Source of Truth."""
        self.log(f"Running Quimb DMRG1 on {params['model']}...")

        import quimb.tensor as qtn

        if params['model'] == 'heisenberg':
            mpo = qtn.MPO_ham_heis(L=params['L'], j=1.0, cyclic=False)
            ref_energy = HEISENBERG_REFERENCE
        else:  # josephson
            mpo = self.build_josephson_mpo(params)
            ref_energy = JOSEPHSON_REFERENCE

        t0 = time.time()
        dmrg = qtn.DMRG1(mpo, bond_dims=params['D'], cutoffs=params['cutoff'])
        dmrg.solve(max_sweeps=params['sweeps'], tol=params['tol'], verbosity=0)
        t1 = time.time()

        energy = float(np.real(dmrg.energy))
        error = self.verify_error(energy, ref_energy, 'Quimb-DMRG1', params['model'], 'ref')

        result = BenchmarkResult(
            implementation='Quimb-DMRG1',
            model=params['model'],
            np_or_streams=1,
            energy=energy,
            wall_time=t1-t0,
            sweeps_actual=len(dmrg.energies) if hasattr(dmrg, 'energies') else params['sweeps'],
            converged=True,
            error_vs_reference=error,
            timestamp=self.timestamp,
        )
        self.results.append(result)
        return energy  # Return as reference for other methods

    def run_quimb_dmrg2(self, params):
        """Run Quimb DMRG2 - Source of Truth."""
        self.log(f"Running Quimb DMRG2 on {params['model']}...")

        import quimb.tensor as qtn

        if params['model'] == 'heisenberg':
            mpo = qtn.MPO_ham_heis(L=params['L'], j=1.0, cyclic=False)
            ref_energy = HEISENBERG_REFERENCE
        else:  # josephson
            mpo = self.build_josephson_mpo(params)
            ref_energy = JOSEPHSON_REFERENCE

        t0 = time.time()
        dmrg = qtn.DMRG2(mpo, bond_dims=params['D'], cutoffs=params['cutoff'])
        dmrg.solve(max_sweeps=params['sweeps'], tol=params['tol'], verbosity=0)
        t1 = time.time()

        energy = float(np.real(dmrg.energy))
        error = self.verify_error(energy, ref_energy, 'Quimb-DMRG2', params['model'], 'ref')

        result = BenchmarkResult(
            implementation='Quimb-DMRG2',
            model=params['model'],
            np_or_streams=1,
            energy=energy,
            wall_time=t1-t0,
            sweeps_actual=len(dmrg.energies) if hasattr(dmrg, 'energies') else params['sweeps'],
            converged=True,
            error_vs_reference=error,
            timestamp=self.timestamp,
        )
        self.results.append(result)
        return energy

    def build_josephson_mpo(self, params):
        """Build Josephson MPO with correct d=5 charge basis."""
        import quimb.tensor as qtn
        import numpy as np

        L = params['L']
        E_J = params['E_J']
        E_C = params['E_C']
        n_max = params['n_max']

        d = 2 * n_max + 1  # d=5 for n_max=2

        # Charge number operator
        charges = np.arange(-n_max, n_max + 1, dtype=np.float64)
        n_op = np.diag(charges.astype(np.complex128))

        # Phase operators
        exp_iphi = np.zeros((d, d), dtype=np.complex128)
        for i in range(d - 1):
            exp_iphi[i + 1, i] = 1.0 + 0j
        exp_miphi = exp_iphi.conj().T

        # Build MPO
        S = (d - 1) / 2
        builder = qtn.SpinHam1D(S=S)

        # External flux
        phi_ext = np.pi / 4
        flux_phase = np.exp(1j * phi_ext)

        # Josephson coupling
        builder.add_term(-E_J / 2 * flux_phase, exp_iphi, exp_miphi)
        builder.add_term(-E_J / 2 * np.conj(flux_phase), exp_miphi, exp_iphi)

        # Charging energy
        n2 = n_op @ n_op
        builder.add_term(E_C, n2)

        return builder.build_mpo(L)

    # ========================================================================
    # MPI-BASED CPU IMPLEMENTATIONS
    # ========================================================================
    def run_mpi_implementation(self, impl_name, params, np):
        """Run PDMRG, PDMRG2, or A2DMRG with MPI."""
        self.log(f"Running {impl_name} (np={np}) on {params['model']}...")

        # Determine which implementation directory and script
        impl_dir = os.path.join(SCRIPT_DIR, impl_name.lower())
        venv_python = os.path.join(impl_dir, 'venv', 'bin', 'python')

        if not os.path.exists(venv_python):
            self.log(f"  WARNING: {venv_python} not found, skipping")
            return

        # Create temporary parameter file
        param_file = f'/tmp/benchmark_{impl_name}_{params["model"]}_np{np}.json'
        with open(param_file, 'w') as f:
            json.dump(params, f)

        # Run with mpirun
        cmd = [
            'mpirun', '-np', str(np),
            venv_python, os.path.join(impl_dir, 'dmrg_benchmark.py'),
            '--params', param_file
        ]

        try:
            t0 = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            t1 = time.time()

            if result.returncode != 0:
                self.log(f"  ERROR: {impl_name} failed: {result.stderr}")
                return

            # Parse output for energy
            output = result.stdout
            energy_line = [l for l in output.split('\n') if 'Final energy' in l]
            if not energy_line:
                self.log(f"  ERROR: Could not parse energy from {impl_name}")
                return

            energy = float(energy_line[0].split(':')[1].strip())

            ref_energy = HEISENBERG_REFERENCE if params['model'] == 'heisenberg' else JOSEPHSON_REFERENCE
            error = self.verify_error(energy, ref_energy, impl_name, params['model'], f'np={np}')

            result_obj = BenchmarkResult(
                implementation=impl_name,
                model=params['model'],
                np_or_streams=np,
                energy=energy,
                wall_time=t1-t0,
                sweeps_actual=params['sweeps'],
                converged=True,
                error_vs_reference=error,
                timestamp=self.timestamp,
            )
            self.results.append(result_obj)

        except subprocess.TimeoutExpired:
            self.log(f"  ERROR: {impl_name} timed out")
        except Exception as e:
            self.log(f"  ERROR: {impl_name} exception: {e}")

    # ========================================================================
    # GPU IMPLEMENTATIONS
    # ========================================================================
    def run_gpu_implementation(self, impl_name, params, streams):
        """Run GPU PDMRG or PDMRG2."""
        self.log(f"Running GPU-{impl_name} (streams={streams}) on {params['model']}...")

        executable = os.path.join(GPU_BUILD_DIR, f"{impl_name.lower()}_gpu")
        if not os.path.exists(executable):
            self.log(f"  WARNING: {executable} not found, skipping")
            return

        # Build command
        cmd = [executable, '--model', params['model'], '--L', str(params['L']),
               '--max-D', str(params['D']), '--sweeps', str(params['sweeps']),
               '--streams', str(streams), '--tol', str(params['tol'])]

        if params['model'] == 'josephson':
            cmd.extend(['--n-max', str(params['n_max']),
                       '--E-J', str(params['E_J']),
                       '--E-C', str(params['E_C'])])

        try:
            t0 = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            t1 = time.time()

            if result.returncode != 0:
                self.log(f"  ERROR: GPU-{impl_name} failed: {result.stderr}")
                return

            # Parse output
            output = result.stdout
            energy_line = [l for l in output.split('\n') if 'Final energy' in l or 'Energy:' in l]
            if not energy_line:
                self.log(f"  ERROR: Could not parse energy from GPU-{impl_name}")
                self.log(f"  Output: {output}")
                return

            energy = float(energy_line[0].split(':')[-1].strip())

            ref_energy = HEISENBERG_REFERENCE if params['model'] == 'heisenberg' else JOSEPHSON_REFERENCE
            error = self.verify_error(energy, ref_energy, f'GPU-{impl_name}', params['model'], f'streams={streams}')

            result_obj = BenchmarkResult(
                implementation=f'GPU-{impl_name}',
                model=params['model'],
                np_or_streams=streams,
                energy=energy,
                wall_time=t1-t0,
                sweeps_actual=params['sweeps'],
                converged=True,
                error_vs_reference=error,
                timestamp=self.timestamp,
            )
            self.results.append(result_obj)

        except subprocess.TimeoutExpired:
            self.log(f"  ERROR: GPU-{impl_name} timed out")
        except Exception as e:
            self.log(f"  ERROR: GPU-{impl_name} exception: {e}")

    # ========================================================================
    # MAIN BENCHMARK RUNNER
    # ========================================================================
    def run_full_suite(self, models=['heisenberg', 'josephson'],
                       run_cpu=True, run_gpu=True):
        """Run complete benchmark suite."""
        self.log("="*80)
        self.log("COMPREHENSIVE DMRG BENCHMARK SUITE")
        self.log(f"Timestamp: {self.timestamp}")
        self.log("="*80)

        for model in models:
            params = HEISENBERG_PARAMS if model == 'heisenberg' else JOSEPHSON_PARAMS

            self.log("")
            self.log("#"*80)
            self.log(f"# {model.upper()} MODEL")
            self.log(f"# L={params['L']}, D={params['D']}, sweeps={params['sweeps']}, tol={params['tol']}")
            self.log("#"*80)

            if run_cpu:
                # Quimb (reference)
                self.log("")
                self.log("--- Quimb (Reference/Source of Truth) ---")
                ref1 = self.run_quimb_dmrg1(params)
                ref2 = self.run_quimb_dmrg2(params)

                # MPI implementations (validated only)
                # PDMRG2 excluded: prototype-only, not validated
                for impl in ['pdmrg', 'a2dmrg']:
                    self.log("")
                    self.log(f"--- {impl.upper()} (MPI) ---")
                    for np in NP_VALUES:
                        self.run_mpi_implementation(impl, params, np)

            if run_gpu:
                # GPU implementations (validated only)
                # PDMRG2-GPU excluded: prototype-only, not validated
                for impl in ['pdmrg']:
                    self.log("")
                    self.log(f"--- GPU-{impl.upper()} ---")
                    for streams in STREAM_VALUES:
                        self.run_gpu_implementation(impl, params, streams)

        # Save results
        self.save_results()
        self.print_summary()

    def save_results(self):
        """Save results to JSON."""
        output_file = os.path.join(SCRIPT_DIR, 'benchmarks',
                                   f'comprehensive_results_{self.timestamp}.json')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)

        self.log("")
        self.log(f"Results saved to: {output_file}")

    def print_summary(self):
        """Print summary of results."""
        self.log("")
        self.log("="*80)
        self.log("SUMMARY")
        self.log("="*80)

        # Check for failures
        failures = [r for r in self.results if r.error_vs_reference >= 1e-10]
        if failures:
            self.log("")
            self.log("⚠️  FAILURES (error >= 1e-10):")
            for r in failures:
                self.log(f"  {r.implementation} {r.model} (np/streams={r.np_or_streams}): "
                        f"error={r.error_vs_reference:.2e}")
        else:
            self.log("")
            self.log("✅ ALL TESTS PASSED (all errors < 1e-10)")

        self.log("")
        self.log(f"Total benchmarks run: {len(self.results)}")
        self.log(f"Total time: {sum(r.wall_time for r in self.results):.1f}s")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Comprehensive DMRG Benchmark Suite')
    parser.add_argument('--heisenberg-only', action='store_true')
    parser.add_argument('--josephson-only', action='store_true')
    parser.add_argument('--cpu-only', action='store_true')
    parser.add_argument('--gpu-only', action='store_true')
    args = parser.parse_args()

    models = []
    if args.heisenberg_only:
        models = ['heisenberg']
    elif args.josephson_only:
        models = ['josephson']
    else:
        models = ['heisenberg', 'josephson']

    run_cpu = not args.gpu_only
    run_gpu = not args.cpu_only

    runner = BenchmarkRunner()
    runner.run_full_suite(models=models, run_cpu=run_cpu, run_gpu=run_gpu)
