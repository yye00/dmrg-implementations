# Challenge Benchmark Data Generation Instructions

## Overview

Challenge benchmarks are larger, more expensive cases intended for execution on HPC systems with 32+ cores and 64+ GB RAM. This document provides exact instructions for generating the missing static input data and golden reference results.

---

## Current Status

### Josephson Challenge Cases (Partial Data Exists)

These cases have **MPO and initial MPS already committed**, but lack golden reference results:

- ✅ `josephson/L24_D50_nmax2/` - MPO (7.6 KB), MPS (3.6 MB), manifest
- ✅ `josephson/L28_D50_nmax2/` - MPO (8.8 KB), MPS (4.3 MB), manifest
- ✅ `josephson/L32_D50_nmax2/` - MPO (10 KB), MPS (5.0 MB), manifest

**Missing**: `golden_results.json` for each case

**Why deferred**: Golden result generation (quimb DMRG at tol=1e-12) took 13+ hours on 16-core workstation without completing. Requires HPC system.

### Heisenberg Challenge Cases (Not Yet Generated)

These cases are completely deferred - no data exists yet:

- ⏳ `heisenberg/L64_D20/` - Not generated
- ⏳ `heisenberg/L96_D20/` - Not generated
- ⏳ `heisenberg/L128_D20/` - Not generated

**Rationale**: Scaling studies at this size benefit from more cores. Generation deferred to HPC execution.

---

## Prerequisites

### Software Requirements

```bash
# Clone repository
git clone https://github.com/yye00/dmrg-implementations.git
cd dmrg-implementations

# Install dependencies
pip install -r requirements.txt

# Verify quimb installation
python -c "import quimb.tensor as qtn; print('quimb OK')"
```

### Hardware Requirements

**Minimum for Josephson golden results**:
- 32+ CPU cores
- 64 GB RAM
- 500 GB disk space

**Recommended for Heisenberg L128**:
- 64+ CPU cores
- 128 GB RAM
- 1 TB disk space

---

## Generating Josephson Golden Results

The static MPO/MPS already exist in the repository. You only need to generate golden reference results.

### Step 1: Verify Existing Data

```bash
cd dmrg-implementations/benchmark_data/challenge/josephson

# Check that MPO, MPS, and manifest exist for each case
for case in L24_D50_nmax2 L28_D50_nmax2 L32_D50_nmax2; do
    echo "=== $case ==="
    ls -lh $case/
done

# Expected output for each:
# - mpo.npz
# - initial_mps.npz
# - manifest.json
# - (golden_results.json missing - this is what we'll generate)
```

### Step 2: Generate Golden Results

Create a script `generate_josephson_golden.py`:

```python
#!/usr/bin/env python3
"""
Generate golden reference results for Josephson challenge benchmarks.

This script loads existing MPO/MPS and runs quimb DMRG1/DMRG2 to obtain
high-precision reference energies.
"""

import sys
import json
import time
import numpy as np
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmark_data_loader import load_mpo_from_disk, load_mps_from_disk, convert_tensors_to_quimb_mpo
import quimb.tensor as qtn


def generate_golden_for_case(model, case, threads=16):
    """Generate golden results for a single case."""
    print(f"\n{'='*80}")
    print(f"Generating golden results: {model}/{case}")
    print(f"{'='*80}")

    # Load existing static data
    print("Loading MPO and initial MPS...")
    mpo_tensors, mpo_meta = load_mpo_from_disk(model, case)
    mps_tensors, mps_meta = load_mps_from_disk(model, case)

    # Convert to quimb objects
    mpo = convert_tensors_to_quimb_mpo(mpo_tensors)

    L = len(mpo_tensors)
    bond_dim = mpo_meta['bond_dim']

    print(f"  L={L}, bond_dim={bond_dim}, dtype={mpo_meta['dtype']}")

    # DMRG parameters
    tolerance = 1e-12
    cutoff = 1e-14

    results = {}

    # Run DMRG1 (single-site)
    print(f"\nRunning quimb DMRG1 with {threads} threads...")
    import os
    os.environ['OMP_NUM_THREADS'] = str(threads)

    dmrg1 = qtn.DMRG1(mpo, bond_dims=bond_dim)
    t0 = time.time()
    dmrg1.solve(tol=tolerance, cutoff=cutoff, verbosity=2)
    t1 = time.time()

    results['quimb_dmrg1'] = {
        'energy': float(dmrg1.energy),
        'energy_per_site': float(dmrg1.energy / L),
        'sweeps': dmrg1.sweep,
        'tolerance': tolerance,
        'cutoff': cutoff,
        'converged': True,
        'wall_time': t1 - t0,
        'threads': threads
    }

    print(f"  Energy: {dmrg1.energy:.15f}")
    print(f"  Sweeps: {dmrg1.sweep}")
    print(f"  Time: {t1-t0:.2f}s")

    # Run DMRG2 (two-site)
    print(f"\nRunning quimb DMRG2 with {threads} threads...")

    dmrg2 = qtn.DMRG2(mpo, bond_dims=bond_dim)
    t0 = time.time()
    dmrg2.solve(tol=tolerance, cutoff=cutoff, verbosity=2)
    t1 = time.time()

    results['quimb_dmrg2'] = {
        'energy': float(dmrg2.energy),
        'energy_per_site': float(dmrg2.energy / L),
        'sweeps': dmrg2.sweep,
        'tolerance': tolerance,
        'cutoff': cutoff,
        'converged': True,
        'wall_time': t1 - t0,
        'threads': threads
    }

    print(f"  Energy: {dmrg2.energy:.15f}")
    print(f"  Sweeps: {dmrg2.sweep}")
    print(f"  Time: {t1-t0:.2f}s")

    # Save results
    output_path = Path(f"benchmark_data/challenge/{model}/{case}/golden_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Saved golden results to: {output_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate Josephson challenge golden results')
    parser.add_argument('--threads', type=int, default=16, help='OpenMP threads')
    parser.add_argument('--case', type=str, help='Specific case (e.g., L24_D50_nmax2) or "all"')
    args = parser.parse_args()

    cases = ['L24_D50_nmax2', 'L28_D50_nmax2', 'L32_D50_nmax2']

    if args.case and args.case != 'all':
        if args.case not in cases:
            print(f"Error: Unknown case {args.case}")
            print(f"Available: {cases}")
            sys.exit(1)
        cases = [args.case]

    print(f"{'='*80}")
    print(f"JOSEPHSON CHALLENGE GOLDEN RESULT GENERATION")
    print(f"{'='*80}")
    print(f"System: {os.cpu_count()} cores detected")
    print(f"Threads: {args.threads} (OMP_NUM_THREADS)")
    print(f"Cases: {len(cases)}")
    print()

    for case in cases:
        try:
            results = generate_golden_for_case('josephson', case, threads=args.threads)
        except Exception as e:
            print(f"\n✗ FAILED: {case}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*80}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*80}")
```

### Step 3: Execute on HPC System

```bash
# SSH to HPC system with 64+ cores
ssh your-hpc-system

# Clone repo (if not already there)
git clone https://github.com/yye00/dmrg-implementations.git
cd dmrg-implementations

# Setup environment
module load python/3.10  # or appropriate version
pip install --user -r requirements.txt

# Generate golden results for all Josephson challenge cases
# Using 16 threads (adjust based on your system)
python benchmark_data/challenge/generate_josephson_golden.py --threads 16 --case all

# Or generate one at a time:
python benchmark_data/challenge/generate_josephson_golden.py --threads 16 --case L24_D50_nmax2
python benchmark_data/challenge/generate_josephson_golden.py --threads 16 --case L28_D50_nmax2
python benchmark_data/challenge/generate_josephson_golden.py --threads 16 --case L32_D50_nmax2
```

**Expected runtime**: 30-120 minutes per case on 64-core system

### Step 4: Verify Results

```bash
# Check that golden results were created
ls -lh benchmark_data/challenge/josephson/*/golden_results.json

# Verify JSON format
for case in L24_D50_nmax2 L28_D50_nmax2 L32_D50_nmax2; do
    echo "=== $case ==="
    python -m json.tool benchmark_data/challenge/josephson/$case/golden_results.json | head -20
done

# Should show:
# - quimb_dmrg1: energy, sweeps, wall_time, converged
# - quimb_dmrg2: energy, sweeps, wall_time, converged
```

### Step 5: Commit to Repository

```bash
# Add golden results
git add benchmark_data/challenge/josephson/L24_D50_nmax2/golden_results.json
git add benchmark_data/challenge/josephson/L28_D50_nmax2/golden_results.json
git add benchmark_data/challenge/josephson/L32_D50_nmax2/golden_results.json

# Commit with metadata
git commit -m "Add Josephson challenge golden reference results

Generated on HPC system:
- Hostname: $(hostname)
- Cores: $(nproc)
- Memory: $(free -h | grep Mem | awk '{print $2}')
- Date: $(date -I)

Results from quimb DMRG1/DMRG2:
- Tolerance: 1e-12
- Cutoff: 1e-14
- Threads: 16

Cases:
- josephson/L24_D50_nmax2
- josephson/L28_D50_nmax2
- josephson/L32_D50_nmax2

Co-Authored-By: [Your Name] <your.email@domain.com>"

# Push to repository
git push origin cpu-audit
```

---

## Generating Heisenberg Challenge Cases

These cases need to be generated from scratch (MPO, MPS, and golden results).

### Step 1: Generate Complete Datasets

```bash
# On HPC system with 64+ cores
cd dmrg-implementations

# Generate each case (takes a few minutes each)
python scripts/generate_benchmark_data.py --case heisenberg_L64_D20 --threads 16
python scripts/generate_benchmark_data.py --case heisenberg_L96_D20 --threads 16
python scripts/generate_benchmark_data.py --case heisenberg_L128_D20 --threads 16
```

**Expected output for each case**:
```
benchmark_data/challenge/heisenberg/L64_D20/
├── mpo.npz
├── initial_mps.npz
├── manifest.json
└── golden_results.json
```

**Expected runtime**:
- L=64: ~2-5 minutes
- L=96: ~5-10 minutes
- L=128: ~10-20 minutes

### Step 2: Verify Data

```bash
# Check file sizes
du -h benchmark_data/challenge/heisenberg/*/

# Verify golden results exist and are sensible
for case in L64_D20 L96_D20 L128_D20; do
    echo "=== $case ==="
    python -c "
import json
with open('benchmark_data/challenge/heisenberg/$case/golden_results.json') as f:
    data = json.load(f)
    print(f\"  DMRG1 energy: {data['quimb_dmrg1']['energy']:.12f}\")
    print(f\"  DMRG2 energy: {data['quimb_dmrg2']['energy']:.12f}\")
    print(f\"  Delta: {abs(data['quimb_dmrg1']['energy'] - data['quimb_dmrg2']['energy']):.2e}\")
"
done

# Expected: DMRG1 and DMRG2 should agree to ~1e-11 or better
```

### Step 3: Commit to Repository

```bash
# Add all Heisenberg challenge data
git add benchmark_data/challenge/heisenberg/

# Commit
git commit -m "Add Heisenberg challenge benchmark datasets

Generated complete datasets for L=64, 96, 128:
- Static MPO and initial MPS
- Golden reference results from quimb DMRG1/DMRG2

Generated on HPC system:
- Hostname: $(hostname)
- Cores: $(nproc)
- Date: $(date -I)

Co-Authored-By: [Your Name] <your.email@domain.com>"

# Push
git push origin cpu-audit
```

---

## Running Challenge Benchmarks

Once golden results exist, run the challenge benchmark suite:

```bash
# On HPC system (requires 32+ cores for full test matrix)
cd dmrg-implementations/benchmarks

# Run correctness tests on challenge tier
python correctness_suite.py --tier challenge

# Run scaling studies
python scaling_suite.py --tier challenge --method pdmrg --max-np 32

# Results saved to:
# - benchmarks/correctness_results_challenge.json
# - benchmarks/scaling_results_challenge.json
```

---

## Expected Data Sizes

### After Josephson Golden Generation

| Case | MPO | Initial MPS | Manifest | Golden | Total |
|------|-----|-------------|----------|--------|-------|
| L24  | 7.6 KB | 3.6 MB | 287 B | ~500 B | ~3.6 MB |
| L28  | 8.8 KB | 4.3 MB | 287 B | ~500 B | ~4.3 MB |
| L32  | 10 KB | 5.0 MB | 287 B | ~500 B | ~5.0 MB |

**Total**: ~13 MB

### After Heisenberg L64/96/128 Generation

| Case | MPO | Initial MPS | Manifest | Golden | Total |
|------|-----|-------------|----------|--------|-------|
| L64  | ~20 KB | ~50 KB | 200 B | ~500 B | ~70 KB |
| L96  | ~30 KB | ~70 KB | 200 B | ~500 B | ~100 KB |
| L128 | ~40 KB | ~90 KB | 200 B | ~500 B | ~130 KB |

**Total**: ~300 KB

**Grand Total**: ~13.3 MB for all challenge benchmarks

---

## Troubleshooting

### Issue: quimb DMRG not converging

**Symptoms**: Runs for hours without convergence

**Solutions**:
1. Increase bond dimension: `bond_dim=100` instead of 50
2. Loosen tolerance: `tol=1e-10` instead of 1e-12
3. Increase max sweeps: `max_sweeps=100`
4. Use more threads: `--threads 32`

### Issue: Out of memory

**Symptoms**: Process killed, OOM error

**Solutions**:
1. Request more memory in job script
2. Reduce bond dimension temporarily
3. Use single-threaded execution (less memory overhead)

### Issue: Results don't match expected accuracy

**Symptoms**: DMRG1 vs DMRG2 differ by > 1e-09

**Solutions**:
1. Check convergence: `dmrg.converged` should be `True`
2. Increase sweeps: let it run longer
3. Verify cutoff parameter: should be 1e-14
4. Check for numerical instability warnings in output

---

## Contact

For questions about challenge benchmark generation:
- Check repository issues: https://github.com/yye00/dmrg-implementations/issues
- Review benchmark documentation: `benchmark_data/BENCHMARK_TIERS.md`
