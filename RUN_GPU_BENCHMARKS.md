# GPU Benchmarking Instructions for MI300X

**Quick Reference**: Step-by-step guide to run GPU benchmarks on HotAisle MI300X system.

---

## Prerequisites Check

```bash
# Verify GPU
rocm-smi --showproductname
# Expected: AMD Instinct MI300X

# Verify ROCm version
rocminfo | head -20
# Expected: ROCm 7.2.0 or later

# Check GPU architecture
rocminfo | grep gfx
# Expected: gfx942
```

---

## Build (One-Time Setup)

```bash
cd /home/captain/clawd/work/dmrg-implementations/gpu-port/build

# Clean build (if needed)
rm -rf *

# Configure with Release mode for best performance
cmake -DCMAKE_BUILD_TYPE=Release ..

# Build all GPU implementations
make -j8

# Verify executables
ls -lh pdmrg_gpu pdmrg2_gpu dmrg_with_environments
```

**Expected output**: Three executables, each ~2-5 MB.

---

## Quick Smoke Test (2 minutes)

Verify GPU implementations work correctly before running full benchmarks:

```bash
cd /home/captain/clawd/work/dmrg-implementations/gpu-port/build

# Test 1: Heisenberg L=12 (should converge to E ≈ -5.142091)
./pdmrg_gpu --model heisenberg --L 12 --max-D 100 --sweeps 5 --streams 1

# Test 2: Josephson L=8 (should converge to E ≈ -2.843801)
./pdmrg2_gpu --model josephson --L 8 --max-D 50 --sweeps 5 --streams 1 --n-max 2
```

**Expected**:
- Heisenberg final energy: -5.142090 ± 0.000001
- Josephson final energy: -2.843801 ± 0.000001
- Both should show "Converged" status

---

## Full Benchmark Suite

### Option 1: Default Configuration (~20 minutes)

```bash
cd /home/captain/clawd/work/dmrg-implementations/gpu-port
./run_benchmarks.sh
```

**Runs:**
- Heisenberg L=12, D=100, sweeps=10, streams=[1,2,4,8]
- Josephson L=12, D=100, sweeps=10, streams=[1,2,4,8]
- Tests both PDMRG (BLAS-2) and PDMRG2 (BLAS-3)

**Output:** `benchmark_results/benchmark_YYYYMMDD_HHMMSS.log`

### Option 2: Quick Test (~5 minutes)

```bash
cd /home/captain/clawd/work/dmrg-implementations/gpu-port
L=12 MAX_D=50 SWEEPS=5 STREAMS="1,2" ./run_benchmarks.sh
```

### Option 3: Large Problem Test (~1 hour)

```bash
cd /home/captain/clawd/work/dmrg-implementations/gpu-port
L=40 MAX_D=200 SWEEPS=20 STREAMS="1,2,4,8" ./run_benchmarks.sh
```

### Option 4: Custom Configuration

```bash
# Environment variables:
#   L        - Chain length (default: 12)
#   MAX_D    - Max bond dimension (default: 100)
#   SWEEPS   - Number of sweeps (default: 10)
#   STREAMS  - Comma-separated stream counts (default: "1,2,4,8")

# Example: Medium problem with extensive stream scaling
L=20 MAX_D=150 SWEEPS=15 STREAMS="1,2,4,8,16" ./run_benchmarks.sh
```

---

## Individual Implementation Tests

If you want to test specific implementations separately:

### PDMRG (BLAS-2, Stream Parallelized)

```bash
cd /home/captain/clawd/work/dmrg-implementations/gpu-port/build

# Heisenberg with 4 streams
./pdmrg_gpu --model heisenberg --L 12 --max-D 100 --sweeps 10 --streams 4

# Josephson with 8 streams
./pdmrg_gpu --model josephson --L 12 --max-D 100 --sweeps 10 --streams 8 \
  --n-max 2 --E-J 1.0 --E-C 0.5
```

### PDMRG2 (BLAS-3, GPU-Optimized)

```bash
cd /home/captain/clawd/work/dmrg-implementations/gpu-port/build

# Heisenberg with 4 streams
./pdmrg2_gpu --model heisenberg --L 12 --max-D 100 --sweeps 10 --streams 4

# Josephson with 8 streams
./pdmrg2_gpu --model josephson --L 12 --max-D 100 --sweeps 10 --streams 8 \
  --n-max 2 --E-J 1.0 --E-C 0.5
```

### Reference Implementation (Single Stream)

```bash
cd /home/captain/clawd/work/dmrg-implementations/gpu-port/build

# Heisenberg
./dmrg_with_environments --model heisenberg --L 12 --max-D 100 --sweeps 10

# Josephson
./dmrg_with_environments --model josephson --L 8 --max-D 50 --sweeps 10 --n-max 2
```

---

## Expected Results

### Heisenberg L=12, D=100

| Implementation | Streams | Expected Time | Expected Energy |
|----------------|---------|---------------|-----------------|
| PDMRG | 1 | ~50-100ms | -5.142090632841 |
| PDMRG | 4 | ~20-40ms | -5.142090632841 |
| PDMRG2 | 1 | ~30-60ms | -5.142090632841 |
| PDMRG2 | 4 | ~12-25ms | -5.142090632841 |

### Josephson L=8, D=50

| Implementation | Streams | Expected Time | Expected Energy |
|----------------|---------|---------------|-----------------|
| PDMRG | 1 | ~100-200ms | -2.843801043139 |
| PDMRG | 4 | ~40-80ms | -2.843801043139 |
| PDMRG2 | 1 | ~60-120ms | -2.843801043139 |
| PDMRG2 | 4 | ~25-50ms | -2.843801043139 |

**Accuracy Requirement**: All energies must match reference within 1e-10 (machine precision).

---

## Performance Profiling (Optional)

For detailed performance analysis:

```bash
cd /home/captain/clawd/work/dmrg-implementations/gpu-port/build

# Profile with rocprof
rocprof --stats ./pdmrg2_gpu --model heisenberg --L 12 --max-D 100 --sweeps 5 --streams 4

# Check memory usage
rocprof --timestamp on --stats ./pdmrg2_gpu --model josephson --L 12 --max-D 100 --sweeps 5 --streams 8 --n-max 2

# View results
ls -ltr *.csv  # rocprof output files
```

---

## Troubleshooting

### Problem: "HIP error: no GPU found"

```bash
# Check if GPU is visible
rocm-smi
rocminfo | grep gfx

# If not visible, check driver
lsmod | grep amdgpu

# Reload driver (requires sudo)
sudo modprobe -r amdgpu
sudo modprobe amdgpu
```

### Problem: "hipTensor library not found"

```bash
# Check hipTensor installation
ls -lh /opt/rocm/lib/libhiptensor*

# If missing, install
sudo apt-get install hiptensor  # or appropriate package manager
```

### Problem: Energy doesn't match reference

```bash
# Check convergence
# Look for "Converged: true" in output
# If false, increase sweeps:
./pdmrg_gpu --model heisenberg --L 12 --max-D 100 --sweeps 20 --streams 1

# Compare with reference implementation
./dmrg_with_environments --model heisenberg --L 12 --max-D 100 --sweeps 20
```

### Problem: Slow performance

```bash
# Check GPU utilization
rocm-smi -d --showuse

# Verify Release build
cd build
cmake -DCMAKE_BUILD_TYPE=Release .. && make -j8

# Try different stream counts
for streams in 1 2 4 8 16; do
  ./pdmrg2_gpu --model heisenberg --L 12 --max-D 100 --sweeps 5 --streams $streams
done
```

---

## Benchmark Results Location

All benchmark logs are saved to:
```
/home/captain/clawd/work/dmrg-implementations/gpu-port/benchmark_results/
```

Each run creates a timestamped log file:
```
benchmark_YYYYMMDD_HHMMSS.log
```

To view the latest results:
```bash
cd /home/captain/clawd/work/dmrg-implementations/gpu-port/benchmark_results
ls -lt | head -5
tail -100 benchmark_*.log  # View latest log
```

---

## Next Steps After Benchmarks

Once benchmarks complete:

1. **Copy results to analysis directory**
   ```bash
   cp benchmark_results/benchmark_*.log /home/captain/clawd/work/dmrg-implementations/benchmarks/
   ```

2. **Generate comparison report**
   ```bash
   cd /home/captain/clawd/work/dmrg-implementations/benchmarks
   python generate_report.py  # Auto-generates tables and plots
   ```

3. **Compare with CPU results**
   - CPU results available in: `cpu_benchmark_results.json` (Heisenberg only)
   - GPU results in: `benchmark_YYYYMMDD_HHMMSS.log`
   - Generate speedup analysis

4. **Share results**
   - Report: `GPU_VERIFICATION_REPORT.md`
   - Detailed logs: `benchmark_results/`
   - Comparison tables: Generated by `generate_report.py`

---

## Quick Command Reference

```bash
# Build
cd gpu-port/build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j8

# Quick test (2 min)
cd .. && L=12 MAX_D=50 SWEEPS=5 STREAMS="1,2" ./run_benchmarks.sh

# Full suite (20 min)
./run_benchmarks.sh

# Large problem (1 hour)
L=40 MAX_D=200 SWEEPS=20 ./run_benchmarks.sh

# View latest results
tail -50 benchmark_results/benchmark_*.log

# Profile
cd build && rocprof --stats ./pdmrg2_gpu --model heisenberg --L 12 --max-D 100 --sweeps 5 --streams 4
```

---

**Last Updated:** 2026-03-04
**Platform:** AMD MI300X (gfx942)
**ROCm Version:** 7.2.0+
