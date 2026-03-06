#!/usr/bin/env python3
"""Phase 3: GPU Benchmarks (PDMRG_GPU and PDMRG2_GPU)"""
import json
import subprocess
import sys
import os
import re
import numpy as np
from datetime import datetime

REPO_ROOT = os.path.expanduser("~/dmrg-implementations")
GPU_BUILD = os.path.join(REPO_ROOT, "gpu-port/build")

VALIDATION_TOL = 1e-10
N_REPETITIONS = 5

# Reference energies (from completed benchmarks)
HEISENBERG_REF = -5.142090632840  # Phase 1
JOSEPHSON_REF = -2.843801043139   # Phase 2 (from Quimb DMRG2 output)

def run_gpu_benchmark(executable, model, L, max_D, sweeps, streams, n_max=None):
    """Run GPU benchmark and parse output."""
    cmd = [
        f"./{executable}",
        "--model", model,
        "--L", str(L),
        "--max-D", str(max_D),
        "--sweeps", str(sweeps),
        "--streams", str(streams)
    ]
    if n_max is not None:
        cmd.extend(["--n-max", str(n_max)])
    
    try:
        result = subprocess.run(cmd, cwd=GPU_BUILD, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            return None, None, result.stderr
        
        for line in result.stdout.split('\n'):
            if line.startswith('>>'):
                match_e = re.search(r'E=([-0-9.]+)', line)
                match_t = re.search(r'time=([0-9.]+)s', line)
                if match_e and match_t:
                    energy = float(match_e.group(1))
                    time_s = float(match_t.group(1))
                    return energy, time_s, None
        return None, None, "Could not parse output"
    except Exception as e:
        return None, None, str(e)

def validate_energy(test_energy, reference_energy, test_name):
    """Validate energy."""
    diff = abs(test_energy - reference_energy)
    if diff > VALIDATION_TOL:
        print(f"\n{'='*80}")
        print(f"❌ GPU VALIDATION FAILURE!")
        print(f"Test: {test_name}")
        print(f"Reference: {reference_energy:.15f}")
        print(f"GPU: {test_energy:.15f}")
        print(f"Difference: {diff:.2e}")
        print(f"{'='*80}")
        return False
    return True

def run_model_gpu_benchmarks(model_name, model, L, D, sweeps, reference_energy, n_max=None):
    """Run GPU benchmarks for one model."""
    print(f"\n{'='*80}")
    print(f"  {model_name} (L={L}, D={D})")
    print(f"  Reference: {reference_energy:.12f}")
    print(f"{'='*80}")
    
    results = {}
    
    for impl in ['pdmrg_gpu', 'pdmrg2_gpu']:
        impl_name = impl.replace('_gpu', '').upper() + '_GPU'
        for streams in [1, 2, 4, 8]:
            print(f"\n[{impl_name} streams={streams}]...")
            energies, times = [], []
            
            for rep in range(N_REPETITIONS):
                energy, time_s, error = run_gpu_benchmark(
                    impl, model, L, D, sweeps, streams, n_max=n_max
                )
                if error:
                    print(f"  ❌ Error: {error}")
                    return None
                
                energies.append(energy)
                times.append(time_s)
                print(f"  Rep {rep+1}/{N_REPETITIONS}: E={energy:.12f}, time={time_s:.4f}s")
                
                if not validate_energy(energy, reference_energy, f"{impl_name} streams={streams}"):
                    return None
            
            results[f'{impl}_streams{streams}'] = {
                'energies': energies,
                'times': times,
                'energy_mean': np.mean(energies),
                'time_mean': np.mean(times),
                'time_std': np.std(times)
            }
            print(f"  ✓ Validated! Avg: {np.mean(times):.4f}s ± {np.std(times):.4f}s")
    
    return results

def main():
    print(f"{'#'*80}")
    print(f"# PHASE 3: GPU BENCHMARKS")
    print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*80}")
    print(f"\nReference energies:")
    print(f"  Heisenberg: {HEISENBERG_REF:.12f}")
    print(f"  Josephson:  {JOSEPHSON_REF:.12f}")
    
    all_results = {
        'metadata': {
            'start_time': datetime.now().isoformat(),
            'validation_tol': VALIDATION_TOL,
            'n_repetitions': N_REPETITIONS,
            'heisenberg_ref': HEISENBERG_REF,
            'josephson_ref': JOSEPHSON_REF
        }
    }
    
    # Heisenberg GPU
    print(f"\n{'#'*80}")
    print(f"# HEISENBERG GPU")
    print(f"{'#'*80}")
    heis_results = run_model_gpu_benchmarks(
        "Heisenberg-Short", "heisenberg", 12, 100, 20, HEISENBERG_REF
    )
    if heis_results is None:
        return 1
    all_results['heisenberg_short_gpu'] = heis_results
    
    # Josephson GPU
    print(f"\n{'#'*80}")
    print(f"# JOSEPHSON GPU")
    print(f"{'#'*80}")
    jos_results = run_model_gpu_benchmarks(
        "Josephson-Short", "josephson", 8, 50, 20, JOSEPHSON_REF, n_max=2
    )
    if jos_results is None:
        return 1
    all_results['josephson_short_gpu'] = jos_results
    
    with open('benchmark_results_phase3_gpu.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'#'*80}")
    print(f"# PHASE 3 GPU COMPLETE!")
    print(f"{'#'*80}\n")
    return 0

if __name__ == '__main__':
    sys.exit(main())
