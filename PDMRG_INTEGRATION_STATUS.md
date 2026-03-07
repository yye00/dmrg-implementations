# PDMRG Integration Status Report
**Date:** 2026-03-05
**Location:** f1a GPU machine (AMD MI300X)

---

## Current Status: ⚠️ Partial Success

### ✅ What Works

1. **MPS/MPO Loading Infrastructure**
   - Binary file loaders work correctly (`mps_mpo_loader.hpp`)
   - Files load successfully from disk
   - Type compatibility resolved (`HostComplex` for host, `Complex` for GPU)
   - LoadedMPO class created and compiles

2. **Original PDMRG_GPU**
   - **Verified working** on f1a with random initialization
   - Test: `./pdmrg_gpu --model heisenberg --L 12 --max-D 100 --sweeps 10 --streams 1`
   - Result: E = -5.142090632841 Ha
   - Gold standard: E = -5.1420906328 Ha
   - **Error: < 1e-10** ✅
   - Converged in 2 sweeps, 1.08s total

3. **Build System**
   - CMakeLists.txt updated
   - Both executables compile successfully
   - All dependencies resolved (hipTensor, rocBLAS, rocSOLVER)

### ❌ What Doesn't Work

1. **pdmrg_benchmark_loaded_v2.cpp**
   - Builds successfully but **produces wrong energies**
   - Test result: E = -1.773659921668 Ha
   - Expected: E = -5.142090632800 Ha
   - **Error: 3.37 Ha** ❌ (way too large)
   - Energy oscillates wildly (negative → positive → negative)
   - Indicates bug in H_eff application or tensor contractions

---

## Root Cause Analysis

The auto-generated `pdmrg_benchmark_loaded_v2.cpp` was created by copying PDMRG_GPU class structure from `pdmrg_gpu.cpp`, but contains implementation bugs:

**Likely Issues:**
1. **H_eff Application**: The 8-loop contraction (lines 829-887) may have index ordering errors
2. **Tensor Layout Mismatch**: Loaded MPS tensors may not match expected layout
3. **Bond Dimension Updates**: Bond dims extracted from loaded MPS may not propagate correctly
4. **Environment Initialization**: Environments may not initialize correctly with loaded data

**Evidence:**
- Original `pdmrg_gpu.cpp` with random MPS: ✅ Correct (-5.142 Ha)
- Auto-generated with loaded MPS: ❌ Wrong (-1.774 Ha)
- Both use same DMRG algorithm structure
- **Conclusion:** Bug introduced during code generation/adaptation

---

## Recommended Fix Strategy

### Option A: Minimal Modification (Recommended)

**Modify existing `pdmrg_gpu.cpp` to accept loaded MPS/MPO**

**Steps:**
1. Add constructor overload to PDMRG_GPU class:
   ```cpp
   PDMRG_GPU(MPOBase* mpo_in,
             const std::vector<MPSTensor>& mps_loaded,  // NEW
             int max_bond, int sweeps, int num_streams,
             const std::string& model, bool debug = false)
   ```

2. In new constructor:
   - Extract bond_dims from `mps_loaded` instead of computing
   - Copy loaded MPS data to GPU (convert HostComplex → hipDoubleComplex)
   - Skip random initialization
   - Continue with existing canonicalization and env init

3. Create simple wrapper executable `pdmrg_benchmark_final.cpp`:
   - Load MPS/MPO using loaders
   - Create LoadedMPO instance
   - Call new PDMRG_GPU constructor
   - Run and validate

**Advantages:**
- Uses proven, working DMRG implementation
- Minimal code changes
- High confidence of correctness

**Estimated Time:** 30-60 minutes

### Option B: Debug Generated Code

**Debug pdmrg_benchmark_loaded_v2.cpp**

**Challenges:**
- Complex 1600+ line file
- Multiple potential bug locations
- Hard to verify without extensive testing

**Not Recommended:** Time-consuming with uncertain outcome

---

## Test Results Summary

| Executable | MPS Source | Energy (Ha) | Error | Status |
|------------|-----------|-------------|-------|--------|
| `pdmrg_gpu` | Random (seed=42) | -5.142090632841 | < 1e-10 | ✅ PASS |
| `pdmrg_benchmark_loaded_v2` | Loaded from file | -1.773659921668 | 3.37 | ❌ FAIL |
| CPU (Quimb DMRG1) | Random (seed=42) | -5.142090632840528 | N/A | Gold Standard |

---

## Files Status

### ✅ Completed
- `/pdmrg-gpu/include/mps_mpo_loader.hpp` - MPS/MPO binary loaders (type-safe with HostComplex)
- `/pdmrg-gpu/include/loaded_mpo.hpp` - MPOBase wrapper for loaded MPO
- `/pdmrg-gpu/CMakeLists.txt` - Build configuration
- `/benchmarks/benchmark_data/` - All test data files (6 cases)
- `/benchmarks/cpu_gold_standard_results.json` - CPU reference energies

### ⚠️ Needs Fix
- `/pdmrg-gpu/src/pdmrg_benchmark_loaded_v2.cpp` - Has bugs, produces wrong results

### 📝 Recommended New File
- `/pdmrg-gpu/src/pdmrg_benchmark_final.cpp` - Clean implementation using modified pdmrg_gpu.cpp

---

## Next Steps (Option A - Recommended)

1. **Modify `pdmrg_gpu.cpp`** (locally, then copy to f1a):
   - Add constructor accepting `std::vector<MPSTensor>& mps_loaded`
   - Extract bond dimensions from loaded data
   - Convert and copy MPS to GPU

2. **Create `pdmrg_benchmark_final.cpp`**:
   ```cpp
   // Load MPS/MPO
   auto mps_host = MPSLoader::load(mps_file);
   auto mpo_host = MPOLoader::load(mpo_file);

   // Create MPO wrapper
   auto* loaded_mpo = new LoadedMPO(mpo_host);

   // Run DMRG with loaded data
   PDMRG_GPU dmrg(loaded_mpo, mps_host, chi_max, sweeps, num_streams, model);
   double energy = dmrg.run();

   // Validate
   validate_against_gold_standard(energy, benchmark_name);
   ```

3. **Update CMakeLists.txt**:
   - Add `pdmrg_benchmark_final` executable
   - Link required libraries

4. **Build and Test on f1a**:
   ```bash
   cd ~/dmrg-implementations/pdmrg-gpu/build
   cmake ..
   make pdmrg_benchmark_final -j16

   # Test Heisenberg L=12
   ./pdmrg_benchmark_final \
       ../../benchmarks/benchmark_data/heisenberg_L12_chi10_mps.bin \
       ../../benchmarks/benchmark_data/heisenberg_L12_mpo.bin \
       100 20 3 1

   # Expected: E ≈ -5.142 Ha, Status: ✅ PASS
   ```

5. **Run All Benchmarks**:
   - Heisenberg L=12 (small)
   - Josephson L=8 (small)
   - Heisenberg L=20 (medium) - optional
   - Josephson L=12 (medium) - optional

6. **Document Results**:
   - GPU vs CPU energy comparison
   - Convergence behavior
   - Performance metrics
   - Validation status

---

## Key Learnings

1. **Code Generation Risk**: Auto-generated code (even from working templates) can introduce subtle bugs
2. **Incremental Testing**: Should have tested loaded vs random MPS with original pdmrg_gpu first
3. **Type Safety**: HostComplex vs Complex distinction prevented many issues
4. **Validation Critical**: Gold standard comparison immediately identified the bug

---

##Success Criteria

For the final implementation to be considered successful:

- ✅ Loads MPS/MPO from binary files (same as CPU)
- ✅ Heisenberg L=12: |E_GPU - E_CPU| < 1e-10
- ✅ Josephson L=8: |E_GPU - E_CPU| < 1e-10
- ✅ No GPU memory errors or crashes
- ✅ Reasonable convergence (within 2x of CPU sweeps)
- ✅ Clean, maintainable code

---

**Status:** Ready to implement Option A
**Next Action:** Modify pdmrg_gpu.cpp to accept loaded MPS
**ETA:** 30-60 minutes to working solution
