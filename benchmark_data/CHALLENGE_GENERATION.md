# Challenge Benchmark Data Generation

This document provides exact instructions for generating challenge benchmark data on HPC systems.

## Overview

Challenge benchmarks are larger, more expensive cases intended for execution on systems with 32+ cores and 64+ GB RAM. Some challenge cases have partial data (MPO/MPS exist but golden results missing), while others need to be generated entirely.

## Current Challenge Benchmark Status

### Josephson - PARTIAL (Golden Results Needed)

**STATUS**: MPO and initial MPS already committed, golden results missing

```
benchmark_data/challenge/josephson/
├── L24_D50_nmax2/  ✅ MPO, ✅ MPS, ❌ Golden
├── L28_D50_nmax2/  ✅ MPO, ✅ MPS, ❌ Golden
└── L32_D50_nmax2/  ✅ MPO, ✅ MPS, ❌ Golden
```

**Why deferred**: Golden reference generation took 13+ hours on 16-core system without completing.

### Heisenberg - PLANNED (Not Yet Generated)

**STATUS**: Not yet generated

```
benchmark_data/challenge/heisenberg/
├── L64_D20/   ❌ Not generated
├── L96_D20/   ❌ Not generated
└── L128_D20/  ❌ Not generated
```

---

## Generating Golden Results for Existing Challenge Cases

For Josephson L=24/28/32 where MPO/MPS already exist.

### System Requirements

- **CPU**: 32+ cores recommended (64+ ideal)
- **Memory**: 64+ GB RAM
- **Runtime**: 30-120 minutes per case on 64-core system
- **Python**: 3.8+ with quimb installed

### Instructions

1. **Clone repository and install dependencies**:
   ```bash
   git clone https://github.com/yye00/dmrg-implementations.git
   cd dmrg-implementations
   pip install quimb numpy scipy
   ```

2. **Generate golden results for Josephson L=24**:
   ```bash
   python scripts/generate_golden_for_challenge.py \
       --model josephson \
       --case L24_D50_nmax2 \
       --threads 16 \
       --tolerance 1e-12 \
       --cutoff 1e-14
   ```

   This will:
   - Load existing MPO/MPS from `benchmark_data/challenge/josephson/L24_D50_nmax2/`
   - Run quimb DMRG1 and DMRG2 with specified parameters
   - Save results to `benchmark_data/challenge/josephson/L24_D50_nmax2/golden_results.json`

3. **Repeat for L=28 and L=32**:
   ```bash
   python scripts/generate_golden_for_challenge.py \
       --model josephson --case L28_D50_nmax2 --threads 16

   python scripts/generate_golden_for_challenge.py \
       --model josephson --case L32_D50_nmax2 --threads 16
   ```

### Expected Output

After successful generation, each case directory should contain:

```
L24_D50_nmax2/
├── mpo.npz               (already exists)
├── initial_mps.npz       (already exists)
├── manifest.json         (already exists)
└── golden_results.json   (newly generated)
```

The `golden_results.json` format:
```json
{
  "quimb_dmrg1": {
    "energy": -9.123456789012345,
    "energy_per_site": -0.380143616208848,
    "sweeps": 50,
    "tolerance": 1e-12,
    "cutoff": 1e-14,
    "converged": true,
    "wall_time": 1234.56,
    "threads": 16
  },
  "quimb_dmrg2": {
    "energy": -9.123456789012340,
    "energy_per_site": -0.380143616208848,
    "sweeps": 25,
    "tolerance": 1e-12,
    "cutoff": 1e-14,
    "converged": true,
    "wall_time": 2345.67,
    "threads": 16
  },
  "generation_info": {
    "date": "2026-03-07T10:30:00",
    "system": "HPC Cluster XYZ",
    "cores": 64,
    "memory_gb": 128
  }
}
```

---

## Generating Complete New Challenge Cases

For Heisenberg L=64/96/128 that don't exist yet.

### System Requirements

- **CPU**: 32+ cores recommended
- **Memory**: 32+ GB RAM (64+ for L=128)
- **Runtime**: 5-30 minutes per case
- **Python**: 3.8+ with quimb installed

### Instructions

1. **Generate complete Heisenberg L=64 dataset**:
   ```bash
   python scripts/generate_benchmark_data.py \
       --case heisenberg_L64_D20 \
       --output-tier challenge \
       --threads 16
   ```

   This will create:
   ```
   benchmark_data/challenge/heisenberg/L64_D20/
   ├── mpo.npz
   ├── initial_mps.npz
   ├── manifest.json
   └── golden_results.json
   ```

2. **Repeat for L=96 and L=128**:
   ```bash
   python scripts/generate_benchmark_data.py \
       --case heisenberg_L96_D20 \
       --output-tier challenge \
       --threads 16

   python scripts/generate_benchmark_data.py \
       --case heisenberg_L128_D20 \
       --output-tier challenge \
       --threads 16
   ```

### Expected Runtimes (64-core system)

| Case | MPO Generation | Golden DMRG1 | Golden DMRG2 | Total |
|------|----------------|--------------|--------------|-------|
| L=64 | ~5s | ~30s | ~15s | ~1 min |
| L=96 | ~10s | ~90s | ~45s | ~3 min |
| L=128 | ~15s | ~300s | ~180s | ~10 min |

*Actual times depend on hardware and bond dimension convergence.*

---

## Committing Challenge Data to Repository

After generating challenge data on HPC system:

### 1. Verify Data Integrity

```bash
# Check all required files exist
cd benchmark_data/challenge
find . -name "manifest.json" -o -name "mpo.npz" -o -name "initial_mps.npz" -o -name "golden_results.json" | sort

# Verify golden results are valid JSON and contain expected fields
python -c "
import json
from pathlib import Path

for p in Path('josephson').rglob('golden_results.json'):
    data = json.load(open(p))
    assert 'quimb_dmrg1' in data
    assert 'quimb_dmrg2' in data
    assert 'energy' in data['quimb_dmrg1']
    print(f'✓ {p}')
"
```

### 2. Stage and Commit

For **golden results only** (Josephson L=24/28/32):
```bash
git add benchmark_data/challenge/josephson/L24_D50_nmax2/golden_results.json
git add benchmark_data/challenge/josephson/L28_D50_nmax2/golden_results.json
git add benchmark_data/challenge/josephson/L32_D50_nmax2/golden_results.json

git commit -m "Add challenge Josephson golden results (L=24,28,32)

Generated on HPC system with 64 cores, 128GB RAM
- quimb DMRG1/DMRG2 with tol=1e-12, cutoff=1e-14
- Each case converged in 30-120 minutes
- Golden energies validated to machine precision

Co-Authored-By: <your-name> <your-email>"
```

For **complete new cases** (Heisenberg L=64/96/128):
```bash
git add benchmark_data/challenge/heisenberg/

git commit -m "Add challenge Heisenberg benchmarks (L=64,96,128)

Generated on HPC system with 64 cores
- Complete MPO/MPS and golden results
- D=20, open boundary conditions
- Runtime: 1-10 minutes per case

Golden energies:
- L=64: [energy value]
- L=96: [energy value]
- L=128: [energy value]

Co-Authored-By: <your-name> <your-email>"
```

### 3. Update Documentation

After committing, update `BENCHMARK_TIERS.md`:

Change:
```markdown
**Status**: ⏳ **PLANNED** - Static input generation deferred to HPC system
```

To:
```markdown
**Status**: ✅ **COMPLETE** - Generated on [system description] on [date]
```

---

## Validation After Generation

Run correctness suite on challenge benchmarks:

```bash
cd benchmarks

# Test challenge tier (requires data to exist)
python correctness_suite.py --tier challenge

# Or test specific case
python correctness_suite.py --model josephson --case L24_D50_nmax2
```

Expected results:
- PDMRG should achieve machine precision (<1e-12) on Heisenberg
- Josephson may show load-balancing variance across np values
- A2DMRG may degrade on larger systems

---

## Troubleshooting

### Issue: Golden generation times out or runs out of memory

**Solution**: Reduce precision or use iterative approach
```bash
# Relax tolerance slightly
--tolerance 1e-10 \
--cutoff 1e-12

# Or generate with fewer DMRG sweeps and accept lower precision
--max-sweeps 50
```

### Issue: quimb not installed or import errors

**Solution**: Install with conda for better numeric library support
```bash
conda install -c conda-forge quimb
```

### Issue: Generated energies don't match expected values

**Solution**: Check convergence and compare against published benchmarks
```python
# Verify convergence
with open('golden_results.json') as f:
    data = json.load(f)
    assert data['quimb_dmrg2']['converged'] == True
    print(f"Energy: {data['quimb_dmrg2']['energy']}")
    print(f"Sweeps: {data['quimb_dmrg2']['sweeps']}")
```

---

## Contact

For questions about challenge benchmark generation, see repository issues or documentation.

Last updated: 2026-03-06
