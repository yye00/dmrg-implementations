# F1A Remote Machine Setup Instructions

## Pulling the Reproducible Benchmark System

### Step 1: Pull Latest Changes from GitHub

```bash
# On f1a machine
cd /path/to/dmrg-implementations

# Export GitHub token from .zshrc
export GITHUB_TOKEN=<your_github_token_here>

# Pull latest changes
git pull https://${GITHUB_TOKEN}@github.com/yye00/dmrg-implementations.git master
```

### Step 2: Verify Files

```bash
# Check benchmark data files are present
ls -lh benchmarks/benchmark_data/*.bin

# Should see:
# heisenberg_L12_chi10_mps.bin + mpo.bin   (Small - focus on this)
# heisenberg_L20_chi10_mps.bin + mpo.bin   (Medium - later)
# josephson_L8_n2_chi10_mps.bin + mpo.bin  (Small - focus on this)
# josephson_L12_n2_chi10_mps.bin + mpo.bin (Medium - later)
```

### Step 3: Build the GPU Benchmark Executable

```bash
cd pdmrg-gpu
mkdir -p build && cd build

# Configure with CMake
cmake ..

# Build the new benchmark executable
make pdmrg_benchmark_loaded -j16

# Verify it was built
ls -lh pdmrg_benchmark_loaded
```

### Step 4: Test with Small Heisenberg Benchmark

```bash
# From pdmrg-gpu/build directory

# Run Heisenberg L=12 benchmark
./pdmrg_benchmark_loaded \
    ../../benchmarks/benchmark_data/heisenberg_L12_chi10_mps.bin \
    ../../benchmarks/benchmark_data/heisenberg_L12_mpo.bin \
    100 20 3 1

# Arguments:
#   100 = chi_max (max bond dimension)
#   20  = max_sweeps
#   3   = warmup_sweeps (single stream)
#   1   = num_streams (for parallel phase)
```

Expected output:
```
================================================================================
  PDMRG GPU Benchmark with Loaded Data
================================================================================

Configuration:
  MPS file:      ../../benchmarks/benchmark_data/heisenberg_L12_chi10_mps.bin
  MPO file:      ../../benchmarks/benchmark_data/heisenberg_L12_mpo.bin
  Max bond dim:  100
  Max sweeps:    20
  Warm-up:       3 sweeps (single stream)
  Parallel:      1 stream(s)

...

FINAL RESULTS:
  Final energy: -5.1420906328 Ha  (target)
  Error:        < 1e-10           (required)
  Status:       ✅ PASS
```

### Step 5: Test with Small Josephson Benchmark

```bash
./pdmrg_benchmark_loaded \
    ../../benchmarks/benchmark_data/josephson_L8_n2_chi10_mps.bin \
    ../../benchmarks/benchmark_data/josephson_L8_n2_mpo.bin \
    50 20 3 1
```

Target energy: **-2.8438010431 Ha**

---

## Integration with Your PDMRG GPU Implementation

### Current Status

The executable `pdmrg_benchmark_loaded` is a **template** with placeholders:

```cpp
// Line ~150: PLACEHOLDER for single-stream warm-up
WarmupResult run_single_stream_warmup(...) {
    // TODO: Call your actual PDMRG single-stream implementation
    // Replace with: run_pdmrg_single_stream(mps_gpu, mpo_gpu, warmup_sweeps, tol)
}

// Line ~200: PLACEHOLDER for multi-stream DMRG
DMRGResult run_multistream_dmrg(...) {
    // TODO: Call your actual PDMRG multi-stream implementation
    // Replace with: run_pdmrg_multistream(mps_gpu, mpo_gpu, chi_max, max_sweeps, num_streams, tol)
}
```

### Integration Steps

1. **Identify your PDMRG functions:**
   - Single-stream DMRG function
   - Multi-stream DMRG function

2. **Replace placeholders:**
   - Edit `pdmrg-gpu/src/pdmrg_benchmark_loaded.cpp`
   - Replace `run_single_stream_warmup` body with actual implementation
   - Replace `run_multistream_dmrg` body with actual implementation

3. **Key integration points:**
   ```cpp
   // You have access to:
   std::vector<GPUTensor3D*> mps_gpu;  // MPS on GPU
   std::vector<GPUTensor4D*> mpo_gpu;  // MPO on GPU

   // Convert to your format if needed:
   YourGPUMPS my_mps = convert_to_your_format(mps_gpu);
   YourGPUMPO my_mpo = convert_to_your_format(mpo_gpu);

   // Then call your DMRG:
   double energy = your_dmrg_function(my_mps, my_mpo, chi_max, sweeps, streams);
   ```

4. **Rebuild and test:**
   ```bash
   cd build
   make pdmrg_benchmark_loaded -j16
   ./pdmrg_benchmark_loaded [args...]
   ```

---

## Validation Criteria

### Energy Accuracy
- **Heisenberg L=12:** `|E_GPU - (-5.1420906328)| < 1e-10`
- **Josephson L=8:** `|E_GPU - (-2.8438010431)| < 1e-10`

### Convergence
- Should converge within similar number of sweeps as CPU (~3-5 for these small cases)
- Energy should decrease monotonically

### Performance
- GPU should be faster than CPU (goal: >2x speedup)
- Single-stream warm-up: ~few seconds
- Multi-stream phase: faster with more streams

---

## Troubleshooting

### Problem: "File not found"
```bash
# Check you're in the right directory
pwd  # Should be in pdmrg-gpu/build

# Check files exist
ls ../../benchmarks/benchmark_data/*.bin
```

### Problem: "Cannot open MPS file"
```bash
# Check file permissions
ls -l ../../benchmarks/benchmark_data/*.bin

# Should be readable (r-- in permissions)
```

### Problem: "HIP error" or "GPU allocation failed"
```bash
# Check GPU is available
rocm-smi

# Check GPU memory
rocm-smi --showmeminfo
```

### Problem: Energies don't match
- Check that you're using complex128 (not complex64)
- Verify tensor layout (C-contiguous, row-major)
- Check index ordering matches expected convention

---

## Next Steps

1. **Pull code on f1a** ✓
2. **Build `pdmrg_benchmark_loaded`** ✓
3. **Test with placeholders** (should run but give dummy energies)
4. **Integrate your PDMRG implementation**
5. **Validate against gold standard**
6. **Run performance benchmarks**

---

## Files Checklist

On f1a, you should have:

```
dmrg-implementations/
├── benchmarks/
│   ├── benchmark_data/
│   │   ├── heisenberg_L12_chi10_mps.bin     ← Small (FOCUS)
│   │   ├── heisenberg_L12_mpo.bin           ← Small (FOCUS)
│   │   ├── josephson_L8_n2_chi10_mps.bin    ← Small (FOCUS)
│   │   ├── josephson_L8_n2_mpo.bin          ← Small (FOCUS)
│   │   └── ... (medium benchmarks)
│   ├── cpu_gold_standard_results.json       ← Gold standard energies
│   └── CPU_GOLD_STANDARD.md                 ← Human-readable results
├── pdmrg-gpu/
│   ├── include/mps_mpo_loader.hpp           ← C++ loader
│   ├── src/pdmrg_benchmark_loaded.cpp       ← Main benchmark executable
│   ├── build/pdmrg_benchmark_loaded         ← Compiled executable
│   └── GPU_BENCHMARK_INTEGRATION.md         ← Integration guide
└── F1A_SETUP_INSTRUCTIONS.md                ← This file
```

---

## Quick Command Reference

```bash
# Pull latest
cd dmrg-implementations
export GITHUB_TOKEN=<your_github_token_here>
git pull https://${GITHUB_TOKEN}@github.com/yye00/dmrg-implementations.git master

# Build
cd pdmrg-gpu/build
cmake .. && make pdmrg_benchmark_loaded -j16

# Run Heisenberg small
./pdmrg_benchmark_loaded \
    ../../benchmarks/benchmark_data/heisenberg_L12_chi10_mps.bin \
    ../../benchmarks/benchmark_data/heisenberg_L12_mpo.bin \
    100 20 3 1

# Run Josephson small
./pdmrg_benchmark_loaded \
    ../../benchmarks/benchmark_data/josephson_L8_n2_chi10_mps.bin \
    ../../benchmarks/benchmark_data/josephson_L8_n2_mpo.bin \
    50 20 3 1
```

---

**Ready to proceed on f1a!** 🚀
