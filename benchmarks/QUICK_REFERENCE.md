# Quick Reference: Reproducible CPU/GPU Benchmarks

## 🎯 Goal Achieved
Run identical benchmarks on CPU and GPU with **same exact MPS and MPO data**.

## ✅ What You Have

### 1. Serialized Data (seed=42)
```bash
benchmarks/benchmark_data/
├── heisenberg_L12_chi10_mps.bin + mpo.bin    # 50 KB
├── heisenberg_L20_chi10_mps.bin + mpo.bin    # 88 KB
├── heisenberg_L40_chi20_mps.bin + mpo.bin    # 540 KB
├── josephson_L8_n2_chi10_mps.bin + mpo.bin   # 91 KB
├── josephson_L12_n2_chi10_mps.bin + mpo.bin  # 148 KB
└── josephson_L16_n2_chi20_mps.bin + mpo.bin  # 534 KB
```

### 2. CPU Gold Standard Results
```bash
benchmarks/cpu_gold_standard_results.json
```

**Energies**:
- Heisenberg L=12: -5.1420906328 Ha
- Heisenberg L=20: -8.6824733344 Ha
- Josephson L=8:   -2.8438010431 Ha
- Josephson L=12:  -4.5070608947 Ha

### 3. Loaders
- **Python**: `load_mps_mpo.py`
- **C++**: `include/mps_mpo_loader.hpp`
- **Test**: `test_mps_mpo_loader` (compiled)

---

## 🚀 Use in GPU Benchmarks

### Step 1: Include header
```cpp
#include "mps_mpo_loader.hpp"
```

### Step 2: Load data
```cpp
// Load exact same data as CPU benchmark
auto mps = MPSLoader::load("benchmark_data/heisenberg_L12_chi10_mps.bin");
auto mpo = MPOLoader::load("benchmark_data/heisenberg_L12_mpo.bin");
```

### Step 3: Access tensors
```cpp
// MPS tensors
for (size_t i = 0; i < mps.size(); ++i) {
    std::cout << "Site " << i << ": "
              << "(" << mps[i].D_left << ", "
              << mps[i].d << ", "
              << mps[i].D_right << ")\n";

    // Access data
    Complex value = mps[i](0, 0, 0);  // (left_bond, phys, right_bond)
}

// MPO tensors
for (size_t i = 0; i < mpo.size(); ++i) {
    std::cout << "Site " << i << ": "
              << "(" << mpo[i].D_mpo_left << ", "
              << mpo[i].d_bra << ", "
              << mpo[i].d_ket << ", "
              << mpo[i].D_mpo_right << ")\n";

    // Access data
    Complex value = mpo[i](0, 0, 0, 0);  // (left, bra, ket, right)
}
```

### Step 4: Convert to your format
```cpp
// Example: Copy to GPU-friendly array
std::vector<Complex> flat_mps;
for (const auto& tensor : mps) {
    flat_mps.insert(flat_mps.end(),
                    tensor.data.begin(),
                    tensor.data.end());
}

// Or copy directly to GPU
hipMemcpy(d_mps, mps[i].data.data(),
          mps[i].total_elements() * sizeof(Complex),
          hipMemcpyHostToDevice);
```

### Step 5: Run and compare
```cpp
// Run your GPU DMRG
double gpu_energy = run_dmrg_gpu(mps, mpo, chi_max, sweeps);

// Compare against CPU gold standard
double cpu_energy = -5.1420906328;  // From results file
double error = std::abs(gpu_energy - cpu_energy);

if (error < 1e-10) {
    std::cout << "✓ GPU matches CPU to high precision!\n";
} else {
    std::cout << "⚠ Energy mismatch: " << error << "\n";
}
```

---

## 📋 Test Cases Available

| Model | L | D_max | d | File Prefix |
|-------|---|-------|---|-------------|
| Heisenberg | 12 | 100 | 2 | heisenberg_L12 |
| Heisenberg | 20 | 100 | 2 | heisenberg_L20 |
| Heisenberg | 40 | 200 | 2 | heisenberg_L40 |
| Josephson | 8 | 50 | 5 | josephson_L8_n2 |
| Josephson | 12 | 50 | 5 | josephson_L12_n2 |
| Josephson | 16 | 100 | 5 | josephson_L16_n2 |

---

## 🔍 Verification Commands

### Python
```bash
cd benchmarks
python3 load_mps_mpo.py benchmark_data/heisenberg_L12_chi10_mps.bin --type mps
python3 verify_loaders.py --data-dir benchmark_data
```

### C++
```bash
cd pdmrg-gpu
./test_mps_mpo_loader mps ../benchmarks/benchmark_data/heisenberg_L12_chi10_mps.bin
./test_mps_mpo_loader mpo ../benchmarks/benchmark_data/heisenberg_L12_mpo.bin
```

---

## 📖 Documentation

- **Quick start**: `QUICKSTART.md`
- **Full spec**: `REPRODUCIBLE_BENCHMARKS.md`
- **Results**: `RESULTS_SUMMARY.md`
- **Status**: `STATUS.md`
- **This file**: `QUICK_REFERENCE.md`

---

## 🎯 Expected Results

When GPU implementation uses same data, it should match CPU energies:

| Case | Expected Energy | Tolerance |
|------|----------------|-----------|
| Heisenberg L=12 | -5.1420906328 | < 1e-10 |
| Heisenberg L=20 | -8.6824733344 | < 1e-10 |
| Josephson L=8 | -2.8438010431 | < 1e-10 |
| Josephson L=12 | -4.5070608947 | < 1e-10 |

---

## 💡 Tips

1. **Reproducibility**: Always use seed=42 data for fair comparison
2. **Validation**: Compare GPU energy against CPU gold standard
3. **Debugging**: Check tensor norms match between Python and C++
4. **Performance**: Time your GPU implementation vs CPU times in results
5. **Scaling**: Test with L=12, then L=20, then larger cases

---

## 🐛 Troubleshooting

**Problem**: File not found
```bash
# Use absolute path
./test_mps_mpo_loader mps $(pwd)/benchmark_data/file.bin
```

**Problem**: Energy doesn't match
- Check you loaded correct file
- Verify tensor shapes with test_mps_mpo_loader
- Compare tensor norms with Python loader
- Check numerical precision in GPU code

**Problem**: Segfault when loading
- Check file is complete (not truncated)
- Verify file size matches expectations
- Run test_mps_mpo_loader first to validate

---

## ✨ Key Achievement

**Before**: Random initial states → Can't compare fairly

**After**: Same MPS/MPO → Fair comparison! ✅

---

Ready to benchmark! 🚀
