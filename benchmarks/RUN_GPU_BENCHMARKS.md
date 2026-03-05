# GPU Multi-Stream Benchmark Suite

Comprehensive benchmark suite for GPU DMRG with multi-stream scalability testing.

## Quick Start (MI300X)

### 1. Build GPU Code with LAPACK

```bash
cd ~/dmrg-implementations/gpu-port
git pull origin master

# Clean build
rm -rf build
mkdir build && cd build

# Configure and build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j16 test_heisenberg_multistream

# Verify executable
ls -lh test_heisenberg_multistream
```

### 2. Run Heisenberg Multi-Stream Benchmark

```bash
cd ~/dmrg-implementations/benchmarks

# Install quimb if needed (for reference energy)
pip install quimb

# Run benchmark with default settings (L=8, chi=32, streams=1,2,4,8)
python gpu_heisenberg_benchmark.py

# Custom settings
python gpu_heisenberg_benchmark.py --L 12 --chi 64 --streams 1,2,4,8 --max-iter 30

# With scalability checking (requires >=70% efficiency)
python gpu_heisenberg_benchmark.py --check-speedup --min-efficiency 0.70

# Specify custom output file
python gpu_heisenberg_benchmark.py --out heisenberg_gpu_results.json
```

## Benchmark Parameters

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--L` | 8 | Chain length (must be even for multi-stream) |
| `--chi` | 32 | Maximum bond dimension |
| `--max-iter` | 20 | Maximum DMRG iterations |
| `--tol` | 1e-10 | Energy convergence tolerance |
| `--pass-tol` | 1e-10 | Pass/fail threshold vs quimb |
| `--streams` | "1,2,4,8" | Comma-separated stream counts |
| `--out` | auto | JSON output file path |
| `--gpu-exe` | auto | Path to GPU executable |
| `--check-speedup` | False | Require linear scalability |
| `--min-efficiency` | 0.70 | Minimum parallel efficiency (70%) |

### System Configurations

**Small Test (Fast):**
```bash
python gpu_heisenberg_benchmark.py --L 8 --chi 32 --max-iter 15
# Expected: ~30s total, 1e-10 accuracy
```

**Medium Test (Accuracy):**
```bash
python gpu_heisenberg_benchmark.py --L 12 --chi 64 --max-iter 30
# Expected: ~2min total, 1e-10 accuracy
```

**Large Test (Scalability):**
```bash
python gpu_heisenberg_benchmark.py --L 16 --chi 128 --max-iter 50 --streams 1,2,4,8
# Expected: ~10min total, measure strong scaling
```

## Expected Output

### Console Output

```
======================================================================
GPU Multi-Stream Heisenberg Benchmark
======================================================================
System: L=8, chi_max=32, max_iter=20
Stream counts: [1, 2, 4, 8]
Pass tolerance: 1e-10
GPU executable: /path/to/test_heisenberg_multistream

======================================================================
Running Quimb DMRG2 Reference (L=8, chi=32)
======================================================================
...
✓ Quimb DMRG2: E = -3.374931816815
  Time: 1.234s, Sweeps: 12

======================================================================
Running GPU DMRG (L=8, chi=32, streams=1)
======================================================================
...
✓ PASS GPU DMRG (streams=1): E = -3.374931816810
  ΔE = 5.00e-12, Time: 2.150s, Iterations: 15

======================================================================
Running GPU DMRG (L=8, chi=32, streams=2)
======================================================================
...
✓ PASS GPU DMRG (streams=2): E = -3.374931816812
  ΔE = 3.00e-12, Time: 1.100s, Iterations: 15

======================================================================
Running GPU DMRG (L=8, chi=32, streams=4)
======================================================================
...
✓ PASS GPU DMRG (streams=4): E = -3.374931816808
  ΔE = 7.00e-12, Time: 0.560s, Iterations: 15

======================================================================
Running GPU DMRG (L=8, chi=32, streams=8)
======================================================================
...
✓ PASS GPU DMRG (streams=8): E = -3.374931816814
  ΔE = 1.00e-12, Time: 0.290s, Iterations: 15

======================================================================
Scalability Analysis (baseline: 2.150s @ 1 stream)
======================================================================
Streams    Time (s)     Speedup      Efficiency   Status
----------------------------------------------------------------------
1          2.150        1.00         100.0%       ✓ Excellent
2          1.100        1.95         97.7%        ✓ Excellent
4          0.560        3.84         96.0%        ✓ Excellent
8          0.290        7.41         92.7%        ✓ Excellent
----------------------------------------------------------------------

======================================================================
Summary
======================================================================
Tests passed: 5/5
Reference energy: -3.374931816815
  ✓ GPU (streams=1): E=-3.374931816810, ΔE=5.00e-12
  ✓ GPU (streams=2): E=-3.374931816812, ΔE=3.00e-12
  ✓ GPU (streams=4): E=-3.374931816808, ΔE=7.00e-12
  ✓ GPU (streams=8): E=-3.374931816814, ΔE=1.00e-12

✓ Results saved to gpu_heisenberg_L8_chi32_results.json
✓ All tests PASSED
```

### Success Criteria

| Metric | Target | Status |
|--------|--------|--------|
| **Accuracy** | \|ΔE\| < 1e-10 | ⏳ Testing |
| **Single stream** | Works correctly | ⏳ Testing |
| **Multi-stream** | All counts work | ⏳ Testing |
| **2-stream efficiency** | ≥ 80% | ⏳ Testing |
| **4-stream efficiency** | ≥ 70% | ⏳ Testing |
| **8-stream efficiency** | ≥ 60% | ⏳ Testing |

## JSON Output Format

```json
{
  "config": {
    "L": 8,
    "chi_max": 32,
    "max_iterations": 20,
    "tolerance": 1e-10,
    "pass_tolerance": 1e-10,
    "stream_counts": [1, 2, 4, 8]
  },
  "reference_energy": -3.374931816815,
  "results": [
    {
      "implementation": "quimb_DMRG2",
      "num_streams": "N/A",
      "energy": -3.374931816815,
      "time": 1.234,
      "sweeps": 12,
      "delta_E": 0.0,
      "passed": true
    },
    {
      "implementation": "GPU_DMRG",
      "num_streams": 1,
      "energy": -3.37493181681,
      "time": 2.15,
      "iterations": 15,
      "delta_E": 5e-12,
      "passed": true
    },
    ...
  ],
  "scalability": {
    "baseline_time": 2.15,
    "speedups": {
      "1": 1.0,
      "2": 1.95,
      "4": 3.84,
      "8": 7.41
    },
    "efficiencies": {
      "1": 1.0,
      "2": 0.977,
      "4": 0.960,
      "8": 0.927
    }
  }
}
```

## Troubleshooting

### Issue 1: Executable Not Found

**Error:**
```
❌ GPU executable not found: .../build/test_heisenberg_multistream
   Build with: cd ... && mkdir -p build && cd build && cmake .. && make -j16
```

**Fix:**
```bash
cd ~/dmrg-implementations/gpu-port
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j16 test_heisenberg_multistream
```

### Issue 2: LAPACK Not Linked

**Error:**
```
undefined reference to 'dstev_'
```

**Fix:**
Verify CMakeLists.txt has LAPACK in target_link_libraries:
```cmake
target_link_libraries(test_heisenberg_multistream
    ...
    lapack
)
```

Then rebuild:
```bash
cd build && rm CMakeCache.txt && cmake .. && make -j16
```

### Issue 3: Accuracy Failure (Error > 1e-10)

**Error:**
```
❌ FAIL GPU DMRG (streams=1): E=-3.500000000000, ΔE=1.25e-01
```

**Diagnosis:**
- Likely LAPACK eigensolver not being called
- Check for "DEBUG LAPACK dstev: eigenvalues = ..." in output
- If eigenvalues are [-1, 0, 1], rocSOLVER bug still present

**Fix:**
```bash
# Verify LAPACK fallback is compiled in
grep -n "dstev_" ../src/boundary_merge_gpu.cpp

# Check for debug output in test
./test_heisenberg_multistream 8 32 1 15 2>&1 | grep "DEBUG LAPACK"
```

### Issue 4: Poor Scalability

**Error:**
```
❌ 8 streams: 25.0% < 60.0%
```

**Diagnosis:**
- Check if GPU is being shared with other processes
- Verify sufficient GPU memory
- Check for CPU bottlenecks (PCIe transfers)

**Fix:**
```bash
# Check GPU utilization during run
rocm-smi --showpids
rocm-smi --showuse

# Run with profiling
rocprof --stats ./test_heisenberg_multistream 8 32 8 15
```

### Issue 5: Quimb Import Error

**Error:**
```
❌ Quimb not available. Using hardcoded reference energy.
```

**Fix:**
```bash
# Install quimb
pip install quimb

# Or use hardcoded reference (for L=8 only)
# Reference energy: -3.374931816815 (exact for L=8 Heisenberg)
```

## Performance Targets

### Accuracy (CRITICAL)

- **All stream counts**: |ΔE| < 1e-10 vs Quimb
- **No stream count variation**: All energies should agree within 1e-11

### Scalability (Strong Scaling)

| Streams | Ideal Speedup | Target Efficiency | Expected Time (L=8) |
|---------|---------------|-------------------|---------------------|
| 1 | 1.0x | 100% | ~2.0s |
| 2 | 2.0x | ≥80% | ~1.1s |
| 4 | 4.0x | ≥70% | ~0.6s |
| 8 | 8.0x | ≥60% | ~0.3s |

**Note**: Efficiency typically decreases with more streams due to:
1. Communication overhead (boundary merges)
2. Load imbalance (site distribution)
3. GPU kernel launch overhead

## Next Steps

### After Successful Heisenberg Benchmark

1. **Josephson Junction Array Benchmark**
   - Implement `test_josephson_multistream.cpp`
   - Create `gpu_josephson_benchmark.py`
   - Test with d=5, complex arithmetic
   - Target: Same 1e-10 accuracy

2. **Weak Scaling Test**
   - Fix streams, increase L proportionally
   - Measure time vs problem size
   - Target: Constant time for proportional scaling

3. **Multi-GPU Scaling**
   - Distribute streams across multiple MI300X GPUs
   - Measure inter-GPU communication overhead
   - Target: Linear scaling to 4-8 GPUs

4. **Production Benchmarks**
   - Large systems: L=40, chi=256
   - Compare to CPU PDMRG/PDMRG2
   - Generate performance report

## Related Documentation

- `TEST_LAPACK_FALLBACK.md` - LAPACK eigensolver validation
- `PHASE2_COMPLETE.md` - Phase 2 implementation status (updated after testing)
- `GPU_VERIFICATION_REPORT.md` - Phase 1 validation results
- CPU benchmarks: `heisenberg_benchmark.py`, `josephson_benchmark.py`

## Contact

Questions? Check:
1. Git commit matches latest (CPU LAPACK fallback implemented)
2. LAPACK linkage in CMakeLists.txt
3. Debug output shows real eigenvalues (not [-1,0,1])
4. No GPU memory errors in `rocm-smi`

**Target**: Same exact answers as CPU (< 1e-10) with high linear scalability!
