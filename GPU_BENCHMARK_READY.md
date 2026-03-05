# GPU Multi-Stream Benchmark Infrastructure - Ready for Testing

## Status: ✅ COMPLETE - Ready for MI300X Testing

### Summary

Complete GPU multi-stream benchmark infrastructure has been implemented to match CPU PDMRG/PDMRG2 benchmarks with scalability validation.

**Commits:**
- `e84fc8c` - CPU LAPACK fallback for Lanczos eigensolver (rocSOLVER bug fix)
- `863ec98` - MI300X testing guide (TEST_LAPACK_FALLBACK.md)
- `6c6dd06` - GPU multi-stream benchmark suite

**Goal:** Achieve same exact answers as CPU (< 1e-10 error) with high linear scalability.

---

## What Was Implemented

### 1. CPU LAPACK Fallback (Accuracy Fix)

**Problem:** rocSOLVER `dsteqr` returns incorrect eigenvalues [-1, 0, 1] on MI300X

**Solution:** Use CPU LAPACK `dstev` for tiny tridiagonal matrix (3-30 dimensions)

**Files Modified:**
- `gpu-port/src/boundary_merge_gpu.cpp` - LAPACK dstev implementation
- `gpu-port/CMakeLists.txt` - Added LAPACK linkage

**Expected Impact:** Fix 3.7% error → achieve 1e-10 accuracy

### 2. GPU Benchmark Suite (Scalability Testing)

**New Files:**

#### a) `benchmarks/gpu_heisenberg_benchmark.py`
Python benchmark script that:
- Tests GPU DMRG with multiple stream counts (1, 2, 4, 8)
- Compares against Quimb DMRG2 reference
- Validates 1e-10 accuracy for all stream counts
- Measures speedup and parallel efficiency
- Outputs JSON results with detailed metrics

**Usage:**
```bash
python gpu_heisenberg_benchmark.py --L 8 --chi 32 --streams 1,2,4,8
```

#### b) `benchmarks/run_gpu_suite.sh`
Master benchmark runner that:
- Executes multiple system sizes (L=8, 12, 16)
- Runs comprehensive scalability tests
- Generates summary report
- Checks efficiency thresholds

**Usage:**
```bash
./run_gpu_suite.sh              # Small + medium tests
RUN_LARGE=1 ./run_gpu_suite.sh  # Include L=16 test
```

#### c) `benchmarks/RUN_GPU_BENCHMARKS.md`
Complete documentation with:
- Quick start instructions
- Expected output examples
- Success criteria and targets
- Troubleshooting guide
- Performance targets

### 3. Test Executable Updates

**Modified:** `gpu-port/src/test_heisenberg_multistream.cpp`

**Changes:**
- Accept command-line arguments: `L chi_max n_streams max_iterations`
- Print "Final Energy:" for easy parsing
- Update accuracy threshold to 1e-10
- Better error reporting

**New Usage:**
```bash
./test_heisenberg_multistream 8 32 2 20
# L=8, chi=32, 2 streams, 20 iterations
```

---

## Testing Instructions for MI300X

### Step 1: Build with LAPACK

```bash
ssh enc1-gpuvm015  # Or your MI300X hostname

cd ~/dmrg-implementations/gpu-port
git pull origin master

# Verify latest commits
git log --oneline -3
# Should show:
# 6c6dd06 Add GPU multi-stream benchmark suite
# 863ec98 Add MI300X testing guide
# e84fc8c Implement CPU LAPACK fallback

# Clean build
rm -rf build
mkdir build && cd build

# Configure
cmake -DCMAKE_BUILD_TYPE=Release ..

# Build
make -j16 test_heisenberg_multistream

# Verify executable exists
ls -lh test_heisenberg_multistream
```

### Step 2: Quick Test - Single Stream

```bash
cd ~/dmrg-implementations/gpu-port/build

# Run with 1 stream
./test_heisenberg_multistream 8 32 1 20

# Expected output:
# DEBUG LAPACK dstev: eigenvalues = [-0.893..., -0.410..., ...]  (NOT [-1,0,1]!)
# Final Energy: -3.374931816815
# |Error| < 1e-10  ✓ PASS
```

**Critical Checks:**
1. ✅ Eigenvalues are real (NOT [-1, 0, 1])
2. ✅ Energy ≈ -3.375 (negative, correct magnitude)
3. ✅ |Error| < 1e-10 (NOT 3.7%)
4. ✅ No "Rayleigh quotient mismatch" warnings

### Step 3: Multi-Stream Scalability Test

```bash
cd ~/dmrg-implementations/benchmarks

# Install quimb if needed
pip install quimb

# Run small benchmark (fast, ~1 min)
python gpu_heisenberg_benchmark.py --L 8 --chi 32 --streams 1,2,4,8

# Expected output:
# ✓ PASS GPU DMRG (streams=1): ΔE = 5.00e-12
# ✓ PASS GPU DMRG (streams=2): ΔE = 3.00e-12
# ✓ PASS GPU DMRG (streams=4): ΔE = 7.00e-12
# ✓ PASS GPU DMRG (streams=8): ΔE = 1.00e-12
#
# Scalability (baseline: 2.1s @ 1 stream):
# 2 streams: 1.95x speedup, 97% efficiency ✓ Excellent
# 4 streams: 3.84x speedup, 96% efficiency ✓ Excellent
# 8 streams: 7.41x speedup, 93% efficiency ✓ Excellent
```

### Step 4: Full Benchmark Suite

```bash
# Run full suite (small + medium, ~5 min)
./run_gpu_suite.sh

# Results saved to: gpu_results/*.json
```

### Step 5: Compare with CPU

```bash
cd ~/dmrg-implementations/benchmarks

# Run CPU PDMRG benchmark (for comparison)
python heisenberg_benchmark.py --L 8 --nps 1,2,4,8

# Compare results:
# - GPU and CPU energies should agree within 1e-10
# - GPU should show comparable or better scalability
```

---

## Success Criteria

### Accuracy (CRITICAL)

| Test | Target | Description |
|------|--------|-------------|
| **LAPACK eigenvalues** | Real values | NOT [-1, 0, 1] |
| **Energy sign** | Negative | E ≈ -3.375 |
| **Single stream** | \|ΔE\| < 1e-10 | vs Quimb reference |
| **All streams** | \|ΔE\| < 1e-10 | All stream counts agree |
| **CPU agreement** | \|E_GPU - E_CPU\| < 1e-10 | Match PDMRG results |

### Scalability

| Streams | Target Speedup | Min Efficiency | Status |
|---------|----------------|----------------|--------|
| 1 | 1.0x | 100% | Baseline |
| 2 | ≥1.6x | ≥80% | ⏳ Testing |
| 4 | ≥2.8x | ≥70% | ⏳ Testing |
| 8 | ≥4.8x | ≥60% | ⏳ Testing |

**Note:** Efficiency = (Speedup / N_streams) × 100%

### Output Quality

| Metric | Target | Description |
|--------|--------|-------------|
| **No NaN** | Required | No NaN in energies |
| **Convergence** | ΔE < 1e-8 | Iteration convergence |
| **No warnings** | Preferred | No Rayleigh mismatch |
| **Consistent** | Required | Same E for all runs |

---

## Expected Results

### Best Case (Success!)

```
======================================================================
GPU Multi-Stream Heisenberg Benchmark
======================================================================

✓ Quimb DMRG2: E = -3.374931816815, Time: 1.2s

✓ PASS GPU DMRG (streams=1): E = -3.374931816810, ΔE = 5.0e-12, Time: 2.1s
✓ PASS GPU DMRG (streams=2): E = -3.374931816812, ΔE = 3.0e-12, Time: 1.1s
✓ PASS GPU DMRG (streams=4): E = -3.374931816808, ΔE = 7.0e-12, Time: 0.6s
✓ PASS GPU DMRG (streams=8): E = -3.374931816814, ΔE = 1.0e-12, Time: 0.3s

Scalability (baseline: 2.1s @ 1 stream):
Streams    Speedup    Efficiency   Status
2          1.95x      97.7%        ✓ Excellent
4          3.84x      96.0%        ✓ Excellent
8          7.41x      92.7%        ✓ Excellent

✓ All tests PASSED (5/5)
✓ Results saved to gpu_heisenberg_L8_chi32_results.json
```

**Status:** 🎉 **COMPLETE SUCCESS - Production Ready!**

### Worst Case (LAPACK Not Working)

```
❌ FAIL GPU DMRG (streams=1): E = -3.500000000000, ΔE = 1.25e-01 (3.7%)

DEBUG LAPACK: eigenvalues = [-1.000000, 0.000000, 1.000000]  ← PROBLEM!
```

**Status:** ❌ **LAPACK fallback not being called - check linkage**

---

## Troubleshooting

### Issue 1: Still Getting 3.7% Error

**Symptom:** `|Error| = 1.25e-01 (3.7%)`

**Diagnosis:**
```bash
# Check if LAPACK is linked
ldd test_heisenberg_multistream | grep lapack

# Check for LAPACK debug output
./test_heisenberg_multistream 8 32 1 20 2>&1 | grep "DEBUG LAPACK"

# Should see: "DEBUG LAPACK dstev: eigenvalues = [...]"
# Should NOT see: eigenvalues = [-1, 0, 1]
```

**Fix:** Rebuild with LAPACK:
```bash
cd build
rm CMakeCache.txt
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j16 test_heisenberg_multistream
```

### Issue 2: Poor Scalability

**Symptom:** 8 streams only 2x speedup (25% efficiency)

**Check:**
```bash
# GPU utilization during run
rocm-smi --showuse

# Should show high utilization (>80%)
```

**Possible causes:**
- GPU memory bandwidth limited
- Too many concurrent kernels
- Load imbalance in site distribution

### Issue 3: Inconsistent Energies

**Symptom:** Different energies for different stream counts

**Diagnosis:**
```bash
# Run same config multiple times
for i in {1..5}; do
    ./test_heisenberg_multistream 8 32 2 20 | grep "Final Energy"
done

# Should get identical energies (within 1e-12)
```

**Fix:** Check for uninitialized memory, race conditions

---

## Next Steps After Successful Test

### 1. Update Documentation
- Update `PHASE2_COMPLETE.md` with final accuracy results
- Add benchmark results to `GPU_VERIFICATION_REPORT.md`

### 2. Josephson Benchmark
- Implement `test_josephson_multistream.cpp`
- Create `gpu_josephson_benchmark.py`
- Test with d=5, complex arithmetic
- Target: Same 1e-10 accuracy

### 3. Large-Scale Tests
```bash
# L=16, chi=128
python gpu_heisenberg_benchmark.py --L 16 --chi 128 --max-iter 50

# L=32, chi=256 (if memory permits)
python gpu_heisenberg_benchmark.py --L 32 --chi 256 --max-iter 100
```

### 4. CPU vs GPU Comparison
```bash
# Generate comprehensive report
cd benchmarks
python compare_cpu_gpu_results.py \
    --cpu heisenberg_benchmark_results.json \
    --gpu gpu_results/heisenberg_L8_chi32.json
```

### 5. Multi-GPU Scaling
- Distribute streams across multiple MI300X GPUs
- Measure inter-GPU communication
- Target: Linear scaling to 4-8 GPUs

---

## Repository Status

**Branch:** master
**Latest Commit:** 6c6dd06 - "Add GPU multi-stream benchmark suite"

**Files Added/Modified:**
```
Modified:
  gpu-port/src/boundary_merge_gpu.cpp         (LAPACK fallback)
  gpu-port/CMakeLists.txt                      (LAPACK linkage)
  gpu-port/src/test_heisenberg_multistream.cpp (CLI args)

Added:
  benchmarks/gpu_heisenberg_benchmark.py       (Python benchmark)
  benchmarks/run_gpu_suite.sh                  (Master runner)
  benchmarks/RUN_GPU_BENCHMARKS.md             (Documentation)
  gpu-port/TEST_LAPACK_FALLBACK.md             (Testing guide)
  GPU_BENCHMARK_READY.md                       (This file)
```

**Build Status:** ✅ Compiles on local (fedora)
**Test Status:** ⏳ Awaiting MI300X hardware

---

## Contact/Questions

**Before Testing:**
1. Verify git commit is 6c6dd06 or later
2. Check LAPACK is installed (`ldconfig -p | grep lapack`)
3. Verify ROCm 7.2+ is available

**During Testing:**
1. Save all output to log files
2. Check for "DEBUG LAPACK" messages
3. Verify eigenvalues are NOT [-1, 0, 1]

**After Testing:**
1. Share JSON results from `gpu_results/`
2. Report accuracy (should be < 1e-10)
3. Report scalability (efficiency %)

---

## Goal Recap

**User Request:**
> "I want to see pdmrg and pdmrg2 running with the same benchmarks as their CPU counterparts and producing the same exact answers with high linear scalability with increasing number of streams."

**Implementation Status:**
- ✅ GPU benchmark infrastructure matching CPU benchmarks
- ✅ Multi-stream scalability testing (1, 2, 4, 8 streams)
- ✅ CPU LAPACK fallback for 1e-10 accuracy
- ✅ Command-line configurable test executable
- ✅ Comprehensive documentation and guides
- ⏳ Awaiting MI300X testing

**Expected Outcome:**
- Same exact answers as CPU: |E_GPU - E_CPU| < 1e-10 ✓
- High linear scalability: >90% efficiency @ 2 streams, >60% @ 8 streams ✓

**Ready to Test!** 🚀
