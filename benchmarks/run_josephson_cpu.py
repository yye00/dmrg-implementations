#!/usr/bin/env python3
"""Quick Josephson-only CPU benchmark to complete the benchmark suite."""

import sys
sys.path.insert(0, '/home/captain/clawd/work/dmrg-implementations/benchmarks')

from cpu_gpu_benchmark import build_josephson_mpo
import quimb.tensor as qtn
import time
import tracemalloc
import json

# Run only Josephson benchmarks
test_cases = {
    'Josephson-Small': {
        'L': 8, 'D': 50, 'd': 5, 'n_max': 2,
        'model': 'josephson', 'sweeps': 20,
        'reference_energy': -2.843801043291333
    },
    'Josephson-Medium': {
        'L': 12, 'D': 50, 'd': 5, 'n_max': 2,
        'model': 'josephson', 'sweeps': 30,
        'reference_energy': None  # Will be computed
    },
    'Josephson-Large': {
        'L': 16, 'D': 100, 'd': 5, 'n_max': 2,
        'model': 'josephson', 'sweeps': 50,
        'reference_energy': None  # Will be computed
    }
}

print("="*80)
print("  JOSEPHSON CPU BENCHMARKS (d=5, charge basis)")
print("="*80)
print()

results = {}
for name, config in test_cases.items():
    print(f"  === {name}: L={config['L']}, D={config['D']}, sweeps={config['sweeps']} ===")

    results[name] = config.copy()

    # DMRG1
    print(f"    Running quimb DMRG1...", end=" ", flush=True)

    H = build_josephson_mpo(config['L'], n_max=config['n_max'])

    tracemalloc.start()
    start = time.perf_counter()

    dmrg1 = qtn.DMRG1(H, bond_dims=[config['D']])
    dmrg1.solve(tol=1e-10, verbosity=0)

    wall_time = time.perf_counter() - start
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    energy = dmrg1.energy
    converged = dmrg1.state.H @ dmrg1.state is not None

    print(f"E={energy:.10f}, t={wall_time:.2f}s, sweeps={dmrg1.sweep}")

    results[name]['DMRG1'] = {
        'energy': energy,
        'wall_time_s': wall_time,
        'n_sweeps': dmrg1.sweep,
        'memory_mb': peak_mem / 1024**2,
        'converged': converged
    }

    # DMRG2
    print(f"    Running quimb DMRG2...", end=" ", flush=True)

    H = build_josephson_mpo(config['L'], n_max=config['n_max'])

    tracemalloc.start()
    start = time.perf_counter()

    dmrg2 = qtn.DMRG2(H, bond_dims=[config['D']])
    dmrg2.solve(tol=1e-10, verbosity=0)

    wall_time = time.perf_counter() - start
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    energy = dmrg2.energy
    converged = dmrg2.state.H @ dmrg2.state is not None

    print(f"E={energy:.10f}, t={wall_time:.2f}s, sweeps={dmrg2.sweep}")
    print()

    results[name]['DMRG2'] = {
        'energy': energy,
        'wall_time_s': wall_time,
        'n_sweeps': dmrg2.sweep,
        'memory_mb': peak_mem / 1024**2,
        'converged': converged
    }

# Save results
output_file = 'josephson_cpu_results.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print("="*80)
print(f"Results saved to {output_file}")
print("="*80)
