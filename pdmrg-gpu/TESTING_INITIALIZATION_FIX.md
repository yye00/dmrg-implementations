# Testing Guide: Initialization Fix for PDMRG-GPU

## Summary

Fixed critical initialization bugs that caused wrong energies:
- **Root cause**: Bond dimensions used exponential growth, leading to chi_bond=1 at boundaries
- **Impact**: Hilbert space only 16-dim → LAPACK returned trivial eigenvalues [-1,0,1] → wrong energy
- **Fix**: Use full chi_max for all internal bonds + random Gaussian initialization with seed matching CPU

## Changes Made (Commit 5eb2a11)

### 1. Bond Dimension Pattern (stream_segment.cpp:202-217)

**Before**:
```cpp
int chi_left = (i == 0) ? 1 : std::min(chi_max_, (int)std::pow(d_, i));
int chi_right = (i == num_sites_ - 1) ? 1 : std::min(chi_max_, (int)std::pow(d_, num_sites_ - 1 - i));
```

**After**:
```cpp
int global_site = start_site_ + i;
int chi_left = (global_site == 0) ? 1 : chi_max_;  // Full chi_max for internal bonds
int chi_right = (global_site == chain_length - 1) ? 1 : chi_max_;  // chi=1 only at chain edges
```

### 2. Random Initialization (stream_segment.cpp:223-242)

**Added**:
```cpp
std::mt19937 rng(42 + id_);  // Match CPU seed (42 + rank)
std::normal_distribution<double> dist(0.0, 1.0);  // Standard normal

for (int i = 0; i < num_sites_; i++) {
    int tensor_size = mps_chi_left_[i] * d_ * mps_chi_right_[i];
    std::vector<double> h_tensor(tensor_size);
    for (int j = 0; j < tensor_size; j++) {
        h_tensor[j] = dist(rng);
    }
    HIP_CHECK(hipMemcpy(d_mps_tensors_[i], h_tensor.data(),
                        tensor_size * sizeof(double), hipMemcpyHostToDevice));
}
```

## Testing Protocol on MI300X

### Prerequisites
```bash
# SSH to MI300X
ssh enc1-gpuvm015  # or your MI300X hostname

# Navigate to project
cd /data/yye00/dmrg-implementations

# Pull latest changes
git pull origin master
```

### Build with LAPACK
```bash
cd pdmrg-gpu
rm -rf build
mkdir build && cd build

# Configure with Release build and LAPACK
cmake -DCMAKE_BUILD_TYPE=Release ..

# Build the test executable
make -j16 test_heisenberg_multistream
```

### Critical Test: 2-Stream Case

This is the MOST IMPORTANT test because:
- Single stream has no boundary merges (can't detect chi_bond=1 bug)
- 2 streams has 1 boundary at center → tests the initialization fix directly

```bash
# Run: L=8, chi=32, n_streams=2, max_iter=20
./test_heisenberg_multistream 8 32 2 20
```

### Expected Output (SUCCESS ✅)

Look for these specific indicators:

1. **Bond Dimension at Boundary**:
```
DEBUG merge: chi_L=32, chi_R=32, chi_bond=32
```
- **chi_bond MUST be 32** (NOT 1!)
- If chi_bond=1, the fix didn't work

2. **Hilbert Space Dimension**:
```
DEBUG LAPACK: chi_L=32, chi_R=32, chi_bond=32, Hilbert space: 2048-dim
```
- Should be 2048-dim = 32 × 2 × 32
- If 16-dim, bond dimensions are still wrong

3. **LAPACK Eigenvalues**:
```
DEBUG LAPACK input:
  D[0..2] = [0.375, 0.147, -0.522]  (example values, will vary)
  E[0..1] = [0.599, 0.650]
```
- **D values should NOT be [-1.0, 0.0, 1.0]**
- Should be real numbers from Hamiltonian spectrum

4. **Final Energy**:
```
Iteration 19: Energy = -3.374931...
```
- Should be close to **-3.375** (exact value for L=8 Heisenberg)
- Tolerance: within 1e-10 (target) or at least 1e-6 (acceptable for now)

### Failure Indicators (❌)

If you see ANY of these, the fix did NOT work:

```
DEBUG merge: chi_L=2, chi_R=2, chi_bond=1              ❌ Wrong bond dimension
Hilbert space: 16-dim                                  ❌ Too small
D[0..2] = [-1.000000, 0.000000, 1.000000]             ❌ Trivial eigenvalues
Energy: 0.0                                            ❌ Zero (no optimization)
Energy: -7.0                                           ❌ Wrong operator
```

### Multi-Stream Scalability Tests

If the 2-stream test PASSES, run full scalability suite:

```bash
# All should give SAME energy (within 1e-10)
./test_heisenberg_multistream 8 32 1 20   # 1 stream
./test_heisenberg_multistream 8 32 2 20   # 2 streams (already tested)
./test_heisenberg_multistream 8 32 4 20   # 4 streams
./test_heisenberg_multistream 8 32 8 20   # 8 streams
```

**Success criteria**:
- All energies should match within numerical precision
- No degradation with increasing stream count
- Consistent convergence behavior

## Debugging Failed Tests

### If chi_bond is still 1:

Check that the code was actually rebuilt:
```bash
# Verify git commit
git log -1 --oneline
# Should show: 5eb2a11 Fix PDMRG-GPU initialization to match CPU PDMRG exactly

# Verify source code
grep "global_site == chain_length - 1" ../src/stream_segment.cpp
# Should find the line with chi_right assignment
```

If code is correct but chi_bond=1 still appears:
- Check that StreamCoordinator is creating segments correctly
- Verify `end_site_` is set to actual chain length - 1 (not segment end)

### If eigenvalues are still [-1,0,1]:

This means the Hamiltonian matrix is not being constructed correctly:
- Check MPO initialization in test_heisenberg_multistream.cpp
- Verify environment contractions in boundary_merge_gpu.cpp
- Enable additional debug output in BoundaryMergeGPU::merge()

### If energy is wrong but eigenvalues look reasonable:

Could be convergence issue:
- Increase max_iterations: `./test_heisenberg_multistream 8 32 2 50`
- Check for energy convergence pattern in output
- Compare against CPU PDMRG with same parameters

## Reference Values

### Heisenberg L=8, chi=32

**Exact ground state energy**: -3.374931816815

**CPU PDMRG2 (from benchmarks/)**:
```bash
cd /data/yye00/dmrg-implementations/benchmarks
pdmrg2/venv/bin/python heisenberg_benchmark.py
```

Should produce:
```
PDMRG2 np=1: E = -3.374931..., ΔE < 1e-10
```

**GPU Target**: Match CPU within 1e-10

## Next Steps After Successful Tests

1. ✅ If 2-stream test passes → Test scalability (1,2,4,8 streams)
2. ✅ If scalability tests pass → Run full GPU benchmark suite:
   ```bash
   cd /data/yye00/dmrg-implementations/benchmarks
   ./run_gpu_suite.sh
   ```
3. ✅ If benchmark passes → Test Josephson model
4. ✅ If all tests pass → Generate final CPU vs GPU comparison report

## Commit Info

- **Commit**: 5eb2a11
- **Branch**: master
- **Date**: 2026-03-05
- **Files Modified**: pdmrg-gpu/src/stream_segment.cpp (+33, -8 lines)

## Contact

If tests fail with unexpected errors not covered in this guide, capture:
1. Full terminal output from build + test
2. `git log -1` output (verify correct commit)
3. `hipinfo` output (verify MI300X GPU detected)
4. Return to session for further debugging
