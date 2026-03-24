# DMRG Benchmark Tier System

## Overview

The benchmark suite is organized into two tiers to support both routine validation and future large-scale performance studies:

- **REGULAR**: Fast benchmarks suitable for routine correctness checks and scaling studies on typical workstations
- **CHALLENGE**: Larger, more expensive benchmarks intended for execution on high-performance systems

## Directory Structure

```
benchmark_data/
├── regular/
│   ├── heisenberg/
│   │   ├── L12_D20/     # Fast correctness check
│   │   ├── L32_D20/     # Intermediate scaling
│   │   └── L48_D20/     # Larger regular case
│   └── josephson/
│       └── L20_D50_nmax2/  # Complex wavefunction baseline
├── challenge/
│   ├── heisenberg/
│   │   └── (L64+, to be defined)
│   └── josephson/
│       ├── L24_D50_nmax2/  # Partial: MPO/MPS exist, no golden results
│       ├── L28_D50_nmax2/  # Partial: MPO/MPS exist, no golden results
│       └── L32_D50_nmax2/  # Partial: MPO/MPS exist, no golden results
└── BENCHMARK_TIERS.md  # This file
```

---

## REGULAR BENCHMARKS

### Purpose
- **Routine validation**: Verify correctness after code changes
- **Performance regression**: Detect performance degradations
- **Scaling studies**: Show parallel efficiency across np=[1,2,4,8]
- **CI/CD integration**: Can run in automated testing pipelines

### Hardware Requirements
- **Minimum**: 8 CPU cores
- **Recommended**: 16+ CPU cores
- **Runtime per case**: 1-60 seconds for most methods
- **Total suite runtime**: ~10-30 minutes

### Heisenberg Cases (Real-valued, float64)

| Case | L | D | Purpose | Expected Runtime |
|------|---|---|---------|-----------------|
| **L12_D20** | 12 | 20 | Fast correctness check | 1-5s per method |
| **L32_D20** | 32 | 20 | Intermediate scaling | 2-10s per method |
| **L48_D20** | 48 | 20 | Larger regular case | 5-60s per method |

**Model**: 1D Heisenberg spin chain with open boundary conditions
```python
H = Σᵢ (Sˣᵢ Sˣᵢ₊₁ + Sʸᵢ Sʸᵢ₊₁ + Sᶻᵢ Sᶻᵢ₊₁)
```

**Status**: ✅ **COMPLETE** - Static MPO/MPS and golden results committed

**Validation Results**:
- PDMRG: Achieves machine precision (<1e-12) on all cases with np=1,2,4,8
- quimb DMRG2: Machine precision reference
- A2DMRG: Machine precision on L=12, degrades to ~1e-10 on L=32, ~1e-09 on L=48

### Josephson Cases (Complex-valued, complex128)

| Case | L | D | n_max | Purpose | Expected Runtime |
|------|---|---|-------|---------|-----------------|
| **L20_D50_nmax2** | 20 | 50 | 2 | Complex baseline | 20-800s per method |

**Model**: 1D Josephson junction array with magnetic flux
```python
H = -EJ Σᵢ cos(φᵢ - φᵢ₊₁ - A) + (4EC/2) Σᵢ nᵢ²
```
- EJ = Josephson energy
- EC = Charging energy
- A = Magnetic flux per plaquette
- nᵢ = Cooper pair number operator (truncated to |n| ≤ n_max)

**Status**: ✅ **COMPLETE** - Static MPO/MPS and golden results committed

**Validation Results**:
- PDMRG: Mixed results - only np=4 achieves machine precision (3.8e-11)
  - np=1: ΔE = 3.6e-10 (marginal)
  - np=2: ΔE = 1.3e-09 (fails acceptance)
  - np=8: ΔE = 4.9e-10 (fails acceptance)
- quimb DMRG2: ΔE = 1.3e-10 (marginal)
- A2DMRG: Timeout (>600s)

**Known Issues**:
- PDMRG load balancing may be suboptimal for complex MPOs
- Boundary merge stability varies with processor count
- Convergence criteria may need tuning for complex systems

---

## CHALLENGE BENCHMARKS

### Purpose
- **Large-scale performance**: Test scalability on HPC systems
- **Production workloads**: Realistic quantum simulation sizes
- **Hardware evaluation**: Compare different architectures
- **Future optimization**: Identify bottlenecks at scale

### Hardware Requirements
- **Minimum**: 32+ CPU cores (64+ recommended)
- **Memory**: 64+ GB RAM
- **Runtime per case**: Minutes to hours depending on method and hardware
- **Total suite runtime**: Hours to days

### Heisenberg Cases (Planned)

| Case | L | D | Status | Notes |
|------|---|---|--------|-------|
| L64_D20 | 64 | 20 | PLANNED | MPO/MPS generation deferred |
| L96_D20 | 96 | 20 | PLANNED | MPO/MPS generation deferred |
| L128_D20 | 128 | 20 | PLANNED | MPO/MPS generation deferred |

**Status**: ⏳ **PLANNED** - Static input generation deferred to HPC system

### Josephson Cases (Partial)

| Case | L | D | n_max | Status | Notes |
|------|---|---|-------|--------|-------|
| **L24_D50_nmax2** | 24 | 50 | 2 | PARTIAL | MPO/MPS exist, golden results missing |
| **L28_D50_nmax2** | 28 | 50 | 2 | PARTIAL | MPO/MPS exist, golden results missing |
| **L32_D50_nmax2** | 32 | 50 | 2 | PARTIAL | MPO/MPS exist, golden results missing |

**Status**: ⚠️ **PARTIAL** - Static MPO/MPS exist but golden results not generated

**Why Deferred**:
- Golden result generation took 13+ hours on 16-core workstation without completing
- quimb DMRG with high precision (tol=1e-12) is extremely expensive for large complex systems
- Better suited for 64+ core HPC system with more memory

**What Exists**:
- ✅ MPO serialized to .npz
- ✅ Initial MPS from quick warmup
- ✅ Manifest with system parameters
- ❌ Golden results from quimb DMRG1/DMRG2

---

## Static Input Data Format

Each benchmark case directory contains:

### Required Files

1. **mpo.npz** - Serialized MPO tensors
   ```python
   np.savez(path, **{f'W_{i}': W[i] for i in range(L)})
   ```

2. **initial_mps.npz** - Initial state for optimization
   ```python
   np.savez(path, **{f'A_{i}': A[i] for i in range(L)})
   ```

3. **manifest.json** - System parameters and metadata
   ```json
   {
     "model": "heisenberg",
     "L": 48,
     "bond_dim": 20,
     "physical_dim": 2,
     "dtype": "float64",
     "generated_date": "2026-03-06T20:30:00",
     "model_params": { ... }
   }
   ```

4. **golden_results.json** - Reference energies from quimb (if available)
   ```json
   {
     "quimb_dmrg1": {
       "energy": -21.085911022739744,
       "sweeps": 20,
       "tolerance": 1e-12,
       "wall_time": 3.36,
       "converged": true
     },
     "quimb_dmrg2": { ... }
   }
   ```

### File Sizes (Approximate)

**Regular benchmarks**:
- Heisenberg L=12: ~10 KB total
- Heisenberg L=48: ~40 KB total
- Josephson L=20: ~1.5 MB total

**Challenge benchmarks**:
- Josephson L=24: ~3.6 MB (MPS dominates)
- Josephson L=32: ~5.0 MB (MPS dominates)

---

## Hardware-Aware Execution

### Runtime Configuration

Benchmarks automatically detect available CPU cores and adapt thread/rank combinations:

```python
from benchmarks.hardware_config import generate_run_matrix

# Auto-detect and generate valid configurations
run_matrix = generate_run_matrix()

# Example output on 16-core system:
# Serial methods: threads in [1, 2, 4, 8]
# MPI methods: (np, threads) combinations where np × threads ≤ 16
```

### Thread/Rank Constraints

**Policy**: `np × threads_per_rank ≤ detected_core_count`

**Example on 16-core system**:
- ✅ np=4, threads=2 → total=8 (allowed)
- ✅ np=8, threads=2 → total=16 (allowed)
- ❌ np=4, threads=8 → total=32 (exceeds cores, skipped)
- ❌ np=8, threads=4 → total=32 (exceeds cores, skipped)

**Preferred thread counts**: [1, 2, 4, 8]
**Preferred np values**: [1, 2, 4, 8]

Only combinations satisfying the constraint are executed. Skipped combinations are logged with reason.

---

## Generating Challenge Data Later

### For Josephson L=24/28/32 Golden Results

The static MPO/MPS already exist. To generate golden results on an HPC system:

```bash
# On 64+ core system
cd /path/to/dmrg-implementations

# Generate golden results only (MPO/MPS already committed)
python scripts/generate_benchmark_data.py \
  --case josephson_L24_D50_nmax2 \
  --golden-only \
  --threads 16

# This will create: benchmark_data/challenge/josephson/L24_D50_nmax2/golden_results.json
```

Expected runtime: 30-120 minutes per case on 64-core system with 128GB RAM.

### For Future Heisenberg Challenge Cases

To generate L=64/96/128 cases:

```bash
# Generate complete dataset (MPO + MPS + golden)
python scripts/generate_benchmark_data.py \
  --case heisenberg_L64_D20 \
  --threads 16

# Creates: benchmark_data/challenge/heisenberg/L64_D20/*
```

### Committing Challenge Data

After generation:

```bash
git add benchmark_data/challenge/josephson/L24_D50_nmax2/golden_results.json
git add benchmark_data/challenge/heisenberg/L64_D20/
git commit -m "Add challenge benchmark golden results from HPC system

Generated on: [system description]
Cores: [N]
Memory: [X] GB
Runtime: [Y] minutes

Co-Authored-By: [your name] <email>"
```

---

## Benchmark Execution

### Correctness Suite (Regular Tier)

Run full correctness validation on regular benchmarks:

```bash
cd benchmarks
python correctness_suite.py --tier regular

# Tests all methods vs golden references:
# - quimb DMRG1, DMRG2
# - PDMRG (np=1,2,4,8)
# - PDMRG-OPT (np=1,2,4,8)
# - A2DMRG (np=2,4,8)
#
# Reports machine precision (<1e-12) vs acceptance (<1e-10)
```

### Scaling Suite (Regular Tier)

Measure parallel scaling efficiency:

```bash
cd benchmarks
python scaling_suite.py --tier regular --method pdmrg

# Runs strong scaling study:
# - Fixed problem size
# - Varying np in [1, 2, 4, 8]
# - Measures wall time, speedup, efficiency
```

### Challenge Execution (Future)

On HPC system with 64+ cores:

```bash
# Generate missing golden results first
python scripts/generate_benchmark_data.py --golden-only --case josephson_L24_D50_nmax2

# Run challenge benchmarks
python benchmarks/correctness_suite.py --tier challenge
python benchmarks/scaling_suite.py --tier challenge --method pdmrg --max-np 32
```

---

## Validation Thresholds

**Machine Precision**: |ΔE| < 1e-12
- Target accuracy for production correctness

**Acceptance Threshold**: |ΔE| < 1e-10
- Minimum acceptable accuracy (chemical accuracy ~ 1 kcal/mol ~ 1e-10 Ha)

**Marginal**: 1e-10 < |ΔE| < 1e-06
- May be acceptable depending on application, but warrants investigation

**Failure**: |ΔE| ≥ 1e-06
- Unacceptable for quantum chemistry/physics applications

---

## Current Status Summary

### Regular Benchmarks: VALIDATED ✅

**Heisenberg (real-valued)**:
- ✅ PDMRG achieves machine precision (<1e-12) on all cases
- ✅ All static inputs and golden results committed
- ✅ Tested with np=1,2,4,8 on 16-core system

**Josephson L=20 (complex)**:
- ⚠️ PDMRG partially validated (only np=4 achieves machine precision)
- ✅ Static inputs and golden results committed
- ⚠️ Load balancing and boundary stability issues identified

### Challenge Benchmarks: DEFERRED ⏳

**Josephson L=24/28/32**:
- ✅ Static MPO/MPS committed
- ❌ Golden results not generated (13+ hours without completion on 16-core)
- 📋 Requires 64+ core HPC system

**Heisenberg L=64+**:
- ❌ Not generated (entirely deferred)
- 📋 Generation scripts ready
- 📋 Requires HPC system

### Known Issues

1. **PDMRG-OPT**: Complete API incompatibility - all tests fail with TypeError
2. **A2DMRG**: Accuracy degrades on larger systems (1e-09 errors on L=48)
3. **PDMRG Complex**: Load balancing inconsistent across np values
4. **Golden Generation**: Extremely expensive for large complex systems

---

## References

- Benchmark infrastructure: `benchmarks/`
- Hardware config: `benchmarks/hardware_config.py`
- Data loader: `benchmark_data_loader.py`
- Generation script: `scripts/generate_benchmark_data.py`
- Correctness suite: `benchmarks/correctness_suite.py`

For questions or issues: see repository documentation.
