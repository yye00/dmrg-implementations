# GPU Port Confidence Analysis & Validation Plan

**Date:** 2026-03-03
**Purpose:** Rigorous validation of every assumption in GPU_PORT_GAMEPLAN.md

**Confidence Levels:**
- 🟢 **HIGH (80-100%)** - Well-documented, standard practice, or verified
- 🟡 **MEDIUM (50-80%)** - Reasonable assumption, needs verification
- 🔴 **LOW (20-50%)** - Uncertain, may need alternatives
- ⚫ **UNKNOWN (<20%)** - Critical gap, must verify immediately

---

## 1. HotAisle Platform Specifics

### What We Don't Know About HotAisle

| Aspect | Confidence | Status | Day 1 Validation |
|--------|-----------|--------|------------------|
| **HotAisle exists as GPU provider** | ⚫ **UNKNOWN** | Cannot verify without access | Check if HotAisle is the actual provider name |
| **MI300X availability** | ⚫ **UNKNOWN** | Assuming based on user statement | Run `rocminfo` and verify gfx942 |
| **ROCm version (5.7, 6.0, 6.1?)** | ⚫ **UNKNOWN** | Critical for hipTensor | `rocminfo \| grep "Runtime Version"` |
| **Pre-installed libraries** | ⚫ **UNKNOWN** | May need manual installation | Check `/opt/rocm/lib/` for libraries |
| **Docker support** | 🟡 **MEDIUM** | Most cloud providers support it | Test `docker run hello-world` |
| **Root/sudo access** | 🔴 **LOW** | May be restricted | Try `sudo apt-get update` |
| **Persistent storage** | 🟡 **MEDIUM** | Standard for cloud | Check mounted volumes |
| **Network bandwidth** | 🟡 **MEDIUM** | Not critical for single GPU | Not blocking |

**Critical Day 1 Questions:**
1. Is this actually HotAisle or a different provider?
2. What's the actual hostname/provider?
3. Do we have bare metal or container access?
4. Can we install software (sudo)?

---

## 2. ROCm Software Stack

### 2.1 Core ROCm Components

| Component | Confidence | Rationale | Verification |
|-----------|-----------|-----------|--------------|
| **HIP runtime** | 🟢 **HIGH** | Always included in ROCm | `hipcc --version` |
| **rocminfo** | 🟢 **HIGH** | Core diagnostic tool | `which rocminfo` |
| **rocm-smi** | 🟢 **HIGH** | GPU monitoring | `rocm-smi` |
| **hipcc compiler** | 🟢 **HIGH** | Standard HIP compiler | `hipcc test.cpp` |

### 2.2 Math Libraries

| Library | Confidence | Reason for Uncertainty | Fallback |
|---------|-----------|------------------------|----------|
| **hipBLAS** | 🟢 **HIGH** | Core library, always available | `ls /opt/rocm/lib/libhipblas.so` |
| **rocBLAS** | 🟢 **HIGH** | Backend for hipBLAS | `ls /opt/rocm/lib/librocblas.so` |
| **rocSOLVER** | 🟢 **HIGH** | Standard in ROCm 5.0+ | `ls /opt/rocm/lib/librocsolver.so` |
| **hipTensor** | 🔴 **LOW** | **May not be installed by default!** | Manual install or build from source |

**CRITICAL UNKNOWN: hipTensor Availability**

**Why Low Confidence:**
- hipTensor is relatively new (2022-2023)
- May not be in default ROCm installation
- Documentation sparse compared to hipBLAS/rocBLAS
- May need to build from source

**Day 1 Validation:**
```bash
# Check if hipTensor exists
ls /opt/rocm/lib/libhiptensor.so
ls /opt/rocm/include/hiptensor.h

# Check package manager
apt search hiptensor
dnf search hiptensor  # if RHEL-based

# Check ROCm version and included libraries
rocm-smi --showdriverversion
ls /opt/rocm/lib/ | grep tensor
```

**Fallback Plan if hipTensor Missing:**
1. **Option A:** Install from ROCm repos
   ```bash
   apt-get install hiptensor
   # or
   dnf install hiptensor
   ```

2. **Option B:** Build from source
   ```bash
   git clone https://github.com/ROCmSoftwarePlatform/hipTensor
   cd hipTensor
   mkdir build && cd build
   cmake ..
   make -j$(nproc)
   sudo make install
   ```

3. **Option C:** Use rocBLAS directly (slower but works)
   - Implement tensor contractions as sequences of GEMM operations
   - Manual tensor reshaping and permutation
   - Lose some optimization but guaranteed to work

### 2.3 Eigen Decomposition Options

| Method | Library | Confidence | Notes |
|--------|---------|-----------|-------|
| **Dense eigensolvers** | rocSOLVER | 🟢 **HIGH** | `rocsolver_zheevd` for complex hermitian |
| **Sparse eigensolvers** | ??? | 🔴 **LOW** | **No clear AMD equivalent to ARPACK/cuSPARSE eigsh** |
| **Block Davidson** | Custom | 🟡 **MEDIUM** | Build on rocBLAS GEMM |
| **Lanczos** | Custom | 🟡 **MEDIUM** | Matrix-vector products with rocBLAS |

**CRITICAL GAP: Sparse Eigensolvers**

DMRG needs eigenvalues of **sparse** effective Hamiltonian (size D² × D², typically D=50-500).

**Problem:**
- rocSOLVER only has dense eigensolvers
- No direct AMD equivalent to `scipy.sparse.linalg.eigsh` or cuSPARSE `eigsh`
- Dense eigensolver for D=500 matrix (250k × 250k) is IMPRACTICAL

**Solutions (in order of preference):**

1. **Iterative Eigensolver (Lanczos/Davidson) - REQUIRED**
   - Implement using rocBLAS for matrix-vector products
   - H_eff is applied via tensor contractions, not stored densely
   - This is standard DMRG approach
   - **Confidence: 🟡 MEDIUM** (need to implement ourselves)

2. **Use dense solver only for validation (small D)**
   - For D < 32, can use `rocsolver_zheevd`
   - **Confidence: 🟢 HIGH**

**Implementation Path:**
```cpp
// We'll need to implement Lanczos ourselves
class LanczosEigensolver {
    // Uses hipBLAS for vector operations
    // H_eff applied via tensor contraction kernel
    void solve(TensorNetwork& H_eff, ...);
};
```

---

## 3. Tensor Operations - The Critical Path

### 3.1 hipTensor API - BIGGEST UNKNOWN

| Capability | Confidence | Reason | Validation |
|------------|-----------|--------|------------|
| **Library exists** | 🔴 **LOW** | May not be installed | `ls /opt/rocm/lib/libhiptensor.so` |
| **Arbitrary einsum support** | ⚫ **UNKNOWN** | Docs unclear | Test `'ijk,klm->ijlm'` contraction |
| **Complex128 support** | 🟡 **MEDIUM** | Should work, needs verification | Test with `hipDoubleComplex` |
| **Batch operations** | 🔴 **LOW** | Feature may not exist | Check API documentation |
| **Stream support** | 🟡 **MEDIUM** | Standard HIP feature | Test async contraction |
| **Performance vs manual GEMM** | ⚫ **UNKNOWN** | Needs benchmarking | Profile both approaches |

**CRITICAL TEST: Can hipTensor Do What We Need?**

**Day 1 Test Program:**
```cpp
#include <hiptensor/hiptensor.h>  // May not exist!
#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>

int main() {
    // Test basic 2-site DMRG contraction
    // einsum('ijk,klm->ijlm', A, B) where:
    //   A: D × d × D  (left MPS + MPO site)
    //   B: D × d × D  (right MPS + MPO site)
    //   Output: D × d × d × D

    const int D = 50;   // Bond dimension
    const int d = 5;    // Physical dimension (n_max=2)

    // Allocate tensors
    hipDoubleComplex *d_A, *d_B, *d_C;
    hipMalloc(&d_A, D * d * D * sizeof(hipDoubleComplex));
    hipMalloc(&d_B, D * d * D * sizeof(hipDoubleComplex));
    hipMalloc(&d_C, D * d * d * D * sizeof(hipDoubleComplex));

    // Try to create hipTensor contraction plan
    // THIS IS WHERE WE'LL DISCOVER IF API WORKS
    hiptensorHandle_t handle;
    hiptensorStatus_t status = hiptensorCreate(&handle);

    if (status != HIPTENSOR_STATUS_SUCCESS) {
        printf("FAIL: hipTensor initialization failed\n");
        return 1;
    }

    // Define tensor descriptors
    // (API syntax may be different - need to check docs)

    // Try contraction
    // ...

    printf("SUCCESS: hipTensor basic test passed\n");
    return 0;
}
```

**Compilation Test:**
```bash
hipcc -o test_hiptensor test_hiptensor.cpp \
    -I/opt/rocm/include \
    -L/opt/rocm/lib \
    -lhiptensor \
    --offload-arch=gfx942

./test_hiptensor
```

**If This Fails:** We fall back to manual GEMM implementation.

### 3.2 Fallback: Manual Tensor Contraction with rocBLAS

**Confidence: 🟢 HIGH** (we can ALWAYS do this)

Any tensor contraction can be rewritten as:
1. Reshape tensors to 2D matrices
2. Call `rocblas_zgemm` (complex GEMM)
3. Reshape result back

**Example: `'ijk,klm->ijlm'` becomes:**
```cpp
// Reshape A(i,j,k) → A'(ij, k)
// Reshape B(k,l,m) → B'(k, lm)
// GEMM: C'(ij, lm) = A'(ij, k) × B'(k, lm)
// Reshape C'(ij, lm) → C(i,j,l,m)

rocblas_zgemm(handle,
    ROCBLAS_OP_N, ROCBLAS_OP_N,
    i*j, l*m, k,  // Matrix dimensions
    &alpha,
    d_A, i*j,     // Leading dimension
    d_B, k,
    &beta,
    d_C, i*j
);
```

**Trade-off:**
- ✅ Guaranteed to work
- ✅ rocBLAS is highly optimized
- ❌ Less optimal than native tensor operations
- ❌ Extra memory for reshaping
- ❌ More code complexity

**Performance Estimate:**
- rocBLAS GEMM: ~80-90% of peak TFLOPS
- hipTensor (if optimal): ~90-95% of peak TFLOPS
- **Difference: ~10%** - acceptable!

---

## 4. SVD and Matrix Factorizations

### 4.1 rocSOLVER Capabilities

| Operation | Function | Confidence | Verification |
|-----------|----------|-----------|--------------|
| **Real SVD (double)** | `rocsolver_dgesvd` | 🟢 **HIGH** | Documented, standard |
| **Complex SVD (complex128)** | `rocsolver_zgesvd` | 🟢 **HIGH** | Documented, standard |
| **QR factorization** | `rocsolver_zgeqrf` | 🟢 **HIGH** | For gauge fixing |
| **Eigenvalues (dense)** | `rocsolver_zheevd` | 🟢 **HIGH** | Small matrices only |

**Day 1 Validation:**
```cpp
#include <rocsolver/rocsolver.h>
#include <hip/hip_runtime.h>

int main() {
    // Test complex SVD for DMRG bond truncation
    const int M = 100;  // D × d
    const int N = 100;  // D × d

    hipDoubleComplex *d_A;
    double *d_S;  // Singular values
    hipDoubleComplex *d_U, *d_Vt;

    hipMalloc(&d_A, M * N * sizeof(hipDoubleComplex));
    hipMalloc(&d_S, std::min(M, N) * sizeof(double));
    hipMalloc(&d_U, M * M * sizeof(hipDoubleComplex));
    hipMalloc(&d_Vt, N * N * sizeof(hipDoubleComplex));

    rocblas_handle handle;
    rocblas_create_handle(&handle);

    // Test SVD
    rocsolver_zgesvd(
        handle,
        rocblas_svect_all,  // Compute all left vectors
        rocblas_svect_all,  // Compute all right vectors
        M, N,
        d_A, M,
        d_S,
        d_U, M,
        d_Vt, N,
        nullptr, 0,  // No extra work array (may need!)
        nullptr      // Info
    );

    // Verify result
    double S_max;
    hipMemcpy(&S_max, d_S, sizeof(double), hipMemcpyDeviceToHost);
    printf("Largest singular value: %e\n", S_max);

    return 0;
}
```

**Compilation:**
```bash
hipcc -o test_svd test_svd.cpp \
    -lrocsolver \
    -lrocblas \
    --offload-arch=gfx942
```

**POTENTIAL ISSUE: Workspace Requirements**

rocSOLVER SVD may require workspace allocation. Need to call:
```cpp
rocsolver_zgesvd_bufferSize(...)  // Get required workspace size
hipMalloc(&workspace, workspace_size);
rocsolver_zgesvd(..., workspace, ...)
```

**Confidence: 🟢 HIGH** - This is standard, will work.

---

## 5. Compilation Strategy

### 5.1 CMake + HIP

| Aspect | Confidence | Notes |
|--------|-----------|-------|
| **CMake finds HIP** | 🟢 **HIGH** | `find_package(hip)` is standard |
| **Linking rocBLAS** | 🟢 **HIGH** | Well-documented |
| **Linking rocSOLVER** | 🟢 **HIGH** | Standard CMake targets |
| **Linking hipTensor** | 🔴 **LOW** | May need manual path specification |
| **gfx942 target** | 🟡 **MEDIUM** | `--offload-arch=gfx942` should work |
| **Complex number support** | 🟢 **HIGH** | `hipDoubleComplex` is standard |

**Potential CMake Issues:**

1. **hipTensor not in CMake package registry**
   ```cmake
   # May need manual specification:
   find_library(HIPTENSOR_LIBRARY
       NAMES hiptensor
       PATHS /opt/rocm/lib /opt/rocm/lib64 /usr/local/lib
       REQUIRED
   )
   ```

2. **Architecture detection**
   ```cmake
   # Verify gfx942 is recognized
   set(CMAKE_HIP_ARCHITECTURES "gfx942")
   # May need: set(GPU_TARGETS "gfx942")
   ```

**Test Compilation:**
```bash
mkdir test_build && cd test_build

cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_HIP_ARCHITECTURES=gfx942

make VERBOSE=1  # See actual compilation commands
```

### 5.2 Compilation Flags - Numerical Precision

| Flag | Effect | Confidence | Recommendation |
|------|--------|-----------|----------------|
| `-O3` | Aggressive optimization | 🟢 **HIGH** | Use in Release |
| `-ffast-math` | Fast math (breaks IEEE) | 🔴 **LOW** | **AVOID initially** |
| `-march=native` | CPU optimizations | 🟢 **HIGH** | Safe |
| `--offload-arch=gfx942` | GPU target | 🟡 **MEDIUM** | Should work |

**CRITICAL: -ffast-math and DMRG Convergence**

**Problem:** `-ffast-math` enables:
- Associativity changes: `(a + b) + c ≠ a + (b + c)`
- Contracted multiplications
- Reciprocal approximations

**Risk for DMRG:**
- Energy convergence may fail
- Iterative eigensolvers sensitive to numerics
- SVD truncation errors accumulate

**Strategy:**
1. Phase 1 (Correctness): `-O3` only, NO fast-math
2. Phase 2 (Optimization): Test with `-ffast-math`, validate ΔE < 1e-12 still holds
3. If fast-math breaks convergence: stay with `-O3`

**Confidence: 🟢 HIGH** - We can always fall back to safe flags.

---

## 6. HIP Streams and Async Execution

| Feature | Confidence | Verification |
|---------|-----------|--------------|
| **Stream creation** | 🟢 **HIGH** | `hipStreamCreate` is standard |
| **Async kernel launch** | 🟢 **HIGH** | `kernel<<<grid, block, 0, stream>>>` |
| **Async memcpy** | 🟢 **HIGH** | `hipMemcpyAsync` |
| **Stream synchronization** | 🟢 **HIGH** | `hipStreamSynchronize` |
| **Overlap compute+transfer** | 🟡 **MEDIUM** | Hardware-dependent |
| **Multi-stream execution** | 🟡 **MEDIUM** | Needs profiling to verify benefit |

**Test Stream Concurrency:**
```cpp
hipStream_t s1, s2, s3;
hipStreamCreate(&s1);
hipStreamCreate(&s2);
hipStreamCreate(&s3);

// Launch kernels on different streams
contract_left<<<grid, block, 0, s1>>>(...);
contract_right<<<grid, block, 0, s2>>>(...);
eigensolver_step<<<grid, block, 0, s3>>>(...);

// Should execute concurrently (verify with rocprof)
hipStreamSynchronize(s1);
hipStreamSynchronize(s2);
hipStreamSynchronize(s3);
```

**Profiling:**
```bash
rocprof --hip-trace ./pdmrg_gpu

# Check timeline for concurrent execution
# Look for overlapping kernel launches
```

**Confidence: 🟢 HIGH** - Streams will work, benefit needs measurement.

---

## 7. Memory Management

### 7.1 HIP Memory Allocation

| Operation | Confidence | Notes |
|-----------|-----------|-------|
| **hipMalloc** | 🟢 **HIGH** | Standard allocation |
| **hipMallocManaged** | 🟡 **MEDIUM** | Unified memory, may be slower |
| **Large allocations (>100GB)** | 🟡 **MEDIUM** | Should work with 192GB, needs testing |
| **Pinned host memory** | 🟢 **HIGH** | `hipMallocHost` for fast transfers |

**Test Large Allocation:**
```cpp
void* d_large;
size_t size = 150ULL * 1024 * 1024 * 1024;  // 150 GB
hipError_t err = hipMalloc(&d_large, size);

if (err != hipSuccess) {
    printf("FAIL: Cannot allocate 150GB\n");
    // Check actual available memory
    size_t free, total;
    hipMemGetInfo(&free, &total);
    printf("Free: %zu GB, Total: %zu GB\n", free>>30, total>>30);
}
```

### 7.2 Memory Layout for DMRG

**Decision: Row-major vs Column-major**

| Layout | Pros | Cons | Confidence |
|--------|------|------|-----------|
| **Row-major (C-style)** | Natural for C++ | May need transposes for BLAS | 🟡 **MEDIUM** |
| **Column-major (Fortran)** | BLAS-friendly | Unnatural indexing | 🟡 **MEDIUM** |

**Test Both:**
```cpp
// Test GEMM performance with different layouts
// Measure: TFLOPS, memory bandwidth utilization
```

**Confidence: 🟡 MEDIUM** - Need to benchmark both.

---

## 8. Performance Estimation

### 8.1 Expected Speedup Analysis

**CPU Baseline (48 cores):**
- Peak TFLOPS: ~2 TFLOPS (FP64)
- Memory BW: ~200 GB/s (DDR4/DDR5)
- DMRG bottleneck: Memory-bound tensor contractions

**MI300X:**
- Peak TFLOPS: 325 TFLOPS (FP64)
- Memory BW: 8 TB/s (HBM3)
- DMRG bottleneck: Initially API overhead, then compute

**Theoretical Speedup:**
- Compute-bound: 325 / 2 = **162x**
- Memory-bound: 8000 / 200 = **40x**
- **Realistic (60% efficiency): 25-60x**

**Our Target: 50-100x**

**Confidence: 🟡 MEDIUM**

**Why Medium:**
- ✅ Hardware capable of 40-162x
- ❌ Software efficiency unknown
- ❌ hipTensor overhead unknown
- ❌ Eigensolver GPU implementation quality TBD

**Validation Plan:**
1. Week 1: Simple GEMM benchmark (should get >150 TFLOPS)
2. Week 2: Single tensor contraction (target >100 TFLOPS)
3. Week 4: Full DMRG sweep (target 25x vs CPU)
4. Week 8: Optimized version (target 50-100x)

---

## 9. Critical Path Dependencies

### What Must Work on Day 1

| Component | Criticality | Confidence | Blocker if Missing? |
|-----------|-------------|-----------|---------------------|
| ROCm 5.7+ | ⚫ CRITICAL | ⚫ **UNKNOWN** | YES - can't use hipTensor |
| hipcc compiler | ⚫ CRITICAL | 🟢 **HIGH** | YES - can't compile |
| rocBLAS | ⚫ CRITICAL | 🟢 **HIGH** | YES - no GEMM |
| rocSOLVER | ⚫ CRITICAL | 🟢 **HIGH** | YES - no SVD |
| hipTensor | 🔴 HIGH | 🔴 **LOW** | NO - can use rocBLAS fallback |
| HIP streams | 🟡 MEDIUM | 🟢 **HIGH** | NO - can run serially first |
| CMake 3.20+ | 🟡 MEDIUM | 🟢 **HIGH** | NO - can use manual compilation |

### Dependency Graph

```
Day 1 Must-Haves (Blocking):
├── ROCm installation (any version 5.0+)
├── hipcc compiler
├── rocBLAS
└── rocSOLVER

Week 1 Needs (High Priority):
├── hipTensor OR ability to install it
├── CMake 3.20+
└── Docker (if using containerized approach)

Week 2+ Needs (Medium Priority):
├── Profiling tools (rocprof)
├── Debugger (rocgdb)
└── Performance monitoring (rocm-smi)
```

---

## 10. Risk Mitigation Matrix

### High-Risk Items

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **hipTensor not available** | 🔴 MEDIUM | 🔴 HIGH | Use rocBLAS GEMM fallback |
| **hipTensor API incompatible** | 🔴 MEDIUM | 🟡 MEDIUM | Manual tensor reshaping |
| **No sparse eigensolver** | 🟢 LOW | 🔴 HIGH | Implement Lanczos (planned) |
| **ROCm version too old (<5.7)** | 🟡 MEDIUM | 🔴 HIGH | Request upgrade or use fallbacks |
| **Insufficient permissions** | 🟡 MEDIUM | 🟡 MEDIUM | Request sudo or use containers |
| **Numerical precision issues** | 🟡 MEDIUM | 🔴 HIGH | Disable fast-math, validate frequently |

### Low-Risk Items

| Aspect | Confidence | Why Low Risk |
|--------|-----------|--------------|
| HIP compilation | 🟢 **HIGH** | Well-established toolchain |
| rocBLAS GEMM | 🟢 **HIGH** | Mature, heavily used |
| rocSOLVER SVD | 🟢 **HIGH** | Standard linear algebra |
| Stream creation | 🟢 **HIGH** | Core HIP feature |
| Memory allocation | 🟢 **HIGH** | Basic GPU operation |

---

## 11. Day 1 Validation Checklist

### Environment Discovery Script

```bash
#!/bin/bash
# Save as: day1_validation.sh

echo "=== DMRG GPU Port Day 1 Validation ==="
echo ""

# 1. Provider identification
echo "[1] Provider & System Info"
hostname
cat /etc/os-release | grep PRETTY_NAME
uname -r
echo ""

# 2. ROCm installation
echo "[2] ROCm Version"
rocminfo | grep "Runtime Version" || echo "FAIL: rocminfo not found"
rocm-smi --showproductname || echo "FAIL: rocm-smi not found"
echo ""

# 3. GPU architecture
echo "[3] GPU Architecture"
rocminfo | grep -A 3 "Name:" | grep "gfx" || echo "FAIL: Cannot detect GPU arch"
echo "REQUIRED: gfx942 (MI300X)"
echo ""

# 4. HIP compiler
echo "[4] HIP Compiler"
hipcc --version || echo "FAIL: hipcc not found"
echo ""

# 5. Math libraries
echo "[5] Math Libraries"
ls /opt/rocm/lib/libhipblas.so && echo "✓ hipBLAS found" || echo "✗ hipBLAS missing"
ls /opt/rocm/lib/librocblas.so && echo "✓ rocBLAS found" || echo "✗ rocBLAS missing"
ls /opt/rocm/lib/librocsolver.so && echo "✓ rocSOLVER found" || echo "✗ rocSOLVER missing"
ls /opt/rocm/lib/libhiptensor.so && echo "✓ hipTensor found" || echo "✗ hipTensor MISSING (need fallback)"
echo ""

# 6. hipTensor in package manager
echo "[6] hipTensor Package Availability"
apt search hiptensor 2>/dev/null | grep hiptensor || echo "Not in apt"
dnf search hiptensor 2>/dev/null | grep hiptensor || echo "Not in dnf"
echo ""

# 7. CMake
echo "[7] CMake Version"
cmake --version | head -1 || echo "FAIL: cmake not found"
echo "REQUIRED: >= 3.20"
echo ""

# 8. Compiler
echo "[8] C++ Compiler"
g++ --version | head -1 || clang++ --version | head -1 || echo "FAIL: No C++ compiler"
echo ""

# 9. Memory available
echo "[9] GPU Memory"
rocm-smi --showmeminfo vram | grep "Total Memory" || echo "Cannot query"
echo "REQUIRED: ~192 GB for MI300X"
echo ""

# 10. Permissions
echo "[10] Permissions Check"
sudo -n true 2>/dev/null && echo "✓ Have sudo" || echo "✗ No sudo (may need container workaround)"
echo ""

# 11. Docker
echo "[11] Docker Availability"
docker --version 2>/dev/null && echo "✓ Docker available" || echo "✗ Docker not available"
echo ""

# Summary
echo "==================================="
echo "CRITICAL CHECKS:"
echo "  [ ] ROCm >= 5.7"
echo "  [ ] gfx942 architecture"
echo "  [ ] hipBLAS present"
echo "  [ ] rocSOLVER present"
echo "  [ ] rocBLAS present"
echo ""
echo "HIGH PRIORITY:"
echo "  [ ] hipTensor (or plan to install)"
echo "  [ ] CMake >= 3.20"
echo "  [ ] sudo access or Docker"
echo "==================================="
```

**Run on Day 1:**
```bash
chmod +x day1_validation.sh
./day1_validation.sh | tee day1_report.txt
# Send day1_report.txt to planning team
```

---

## 12. Alternative Paths if Components Missing

### Scenario A: hipTensor Missing, Cannot Install

**Workaround:** Manual GEMM implementation

```cpp
// All tensor contractions via rocBLAS
class TensorContractionGEMM {
    rocblas_handle handle_;

    void contract_ijk_klm_to_ijlm(
        const complex<double>* A,  // i × j × k
        const complex<double>* B,  // k × l × m
        complex<double>* C,        // i × j × l × m
        int i, int j, int k, int l, int m
    ) {
        // Reshape A: (i×j, k)
        // Reshape B: (k, l×m)
        // GEMM: C' = A' × B'  (i×j, l×m)
        // Reshape C': (i, j, l, m)

        rocblas_zgemm(handle_,
            ROCBLAS_OP_N, ROCBLAS_OP_N,
            i*j, l*m, k,
            &alpha, A, i*j, B, k,
            &beta, C, i*j
        );

        // Note: May need explicit reshape if strides don't match
    }
};
```

**Confidence: 🟢 HIGH** - Always works, ~10% slower than hipTensor.

### Scenario B: ROCm < 5.7, Cannot Upgrade

**Workarounds:**
1. Use rocBLAS-only implementation (see Scenario A)
2. Build hipTensor from source (may work on older ROCm)
3. Stick with CPU implementation (not ideal)

**Confidence: 🟡 MEDIUM** - Possible but requires more manual work.

### Scenario C: No Sudo, No Docker

**Workarounds:**
1. Install libraries to `$HOME/.local`
   ```bash
   cmake -DCMAKE_INSTALL_PREFIX=$HOME/.local ..
   export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH
   ```

2. Use Spack package manager (no root needed)
   ```bash
   git clone https://github.com/spack/spack.git
   . spack/share/spack/setup-env.sh
   spack install rocblas rocsolver
   ```

**Confidence: 🟡 MEDIUM** - More setup time, but feasible.

---

## 13. Communication vs Computation (Latency Hiding)

### 13.1 Data Movement Analysis

| Operation | Size (D=100, L=40, complex128) | Time Estimate | Strategy |
|-----------|-------------------------------|---------------|----------|
| **Host→Device (MPO)** | ~40 MB | 0.01 ms @ 8TB/s | Do once at startup |
| **Device→Host (final E)** | 8 bytes | <0.001 ms | Do once at end |
| **Within-device tensor** | ~6 MB per site | 0.001 ms | Negligible |

**Key Insight:** With 192GB HBM3, we have ZERO host-device transfers during sweeps!

**Latency Hiding Strategy:**

```cpp
// Traditional (bad):
for (site = 0; site < L; site++) {
    contract_tensors();      // GPU work
    hipDeviceSynchronize();  // WAIT (wasted time!)
    solve_eigenproblem();    // GPU work
    hipDeviceSynchronize();  // WAIT (wasted time!)
    update_state();          // GPU work
}

// Pipelined (good):
hipStream_t s_contract, s_solve, s_update;

for (site = 0; site < L; site++) {
    // These run concurrently:
    contract_for_next_site<<<..., s_contract>>>(site+1);  // Prep ahead
    solve_current_site<<<..., s_solve>>>(site);           // Main work
    update_previous_site<<<..., s_update>>>(site-1);      // Finalize

    // Only sync when needed (e.g., dependencies)
}
// Single sync at end of sweep
hipDeviceSynchronize();
```

**Potential Speedup from Pipelining:** 1.2-1.5x (if kernels are small)

**Confidence: 🟡 MEDIUM**
- Streams definitely work
- Whether there's independent work to overlap is uncertain
- May not matter if kernels are large (saturate GPU anyway)

### 13.2 Kernel Launch Overhead

**HIP kernel launch:** ~5-10 μs per kernel

**DMRG sweep:** L × N_sweeps × kernels_per_site launches
- Example: 40 sites × 20 sweeps × 10 kernels = 8,000 launches
- Overhead: 8,000 × 10 μs = 80 ms

**Mitigation:**
1. **Fuse kernels** where possible
2. **Persistent kernels** - launch once, process all sites
3. **Accept the overhead** - 80ms is negligible vs minutes of compute

**Confidence: 🟢 HIGH** - Not a major concern.

---

## 14. Testing & Validation Strategy

### Phase-by-Phase Validation

#### Phase 0: Environment (Week 1)
- [ ] Run day1_validation.sh
- [ ] Compile hello_streams.hip
- [ ] Test rocBLAS GEMM (measure TFLOPS)
- [ ] Test rocSOLVER SVD (measure time)
- [ ] Test hipTensor (if available) or design GEMM fallback

**Success Criteria:**
- All libraries found or installable
- GEMM achieves >100 TFLOPS (30% peak)
- SVD completes without errors

#### Phase 1: Single Tensor Contraction (Week 2)
- [ ] Implement 2-site contraction (hipTensor OR GEMM)
- [ ] Benchmark D=50, d=5 contraction
- [ ] Compare to CPU numpy.einsum time

**Success Criteria:**
- Contraction correctness: ||Result - CPU_result|| < 1e-12
- Performance: >1000x faster than CPU for this operation

#### Phase 2: Eigensolver (Week 3)
- [ ] Implement Lanczos on GPU
- [ ] Test on small H_eff (D=32)
- [ ] Validate eigenvalue vs dense solver

**Success Criteria:**
- Eigenvalue accuracy: |E_Lanczos - E_dense| < 1e-10
- Convergence in <20 iterations

#### Phase 3: Full DMRG Sweep (Week 4)
- [ ] Implement L=12 Heisenberg
- [ ] Compare energy to CPU PDMRG

**Success Criteria:**
- Energy: |E_GPU - E_CPU| < 1e-12
- Time: <10 seconds for L=12 (CPU: ~60s)

#### Phase 4: Optimization (Week 5-7)
- [ ] Add stream pipelining
- [ ] Profile with rocprof
- [ ] Optimize hotspots

**Success Criteria:**
- L=40 in <30 seconds (CPU: ~30 min) = 60x speedup

#### Phase 5: Production (Week 8)
- [ ] L=100, D=500 benchmark
- [ ] Josephson junction (complex128)
- [ ] Documentation

**Success Criteria:**
- L=100 completes (impossible on CPU)
- 50-100x speedup on L=40 benchmark

---

## 15. Confidence Summary

### High Confidence (Will Work) 🟢

| Component | Why |
|-----------|-----|
| HIP compilation | Industry standard, well-documented |
| rocBLAS GEMM | Mature library, extensively used |
| rocSOLVER SVD/QR | Standard linear algebra, reliable |
| HIP streams | Core GPU feature, guaranteed |
| Manual tensor contractions | Always works via GEMM |
| 25-40x speedup | Conservative estimate, achievable |

### Medium Confidence (Should Work) 🟡

| Component | Uncertainty |
|-----------|-------------|
| hipTensor availability | May need manual install |
| hipTensor performance | Unknown vs manual GEMM |
| 50-100x speedup | Optimistic, depends on efficiency |
| Stream pipelining benefit | Depends on kernel granularity |
| Complex128 performance | Should be fine, needs verification |

### Low Confidence (Risky) 🔴

| Component | Risk |
|-----------|------|
| hipTensor in default install | Probably not included |
| Sparse eigensolver | Must implement ourselves |
| Fast-math compatibility | May break convergence |

### Unknown (Must Verify Day 1) ⚫

| Component | Critical? |
|-----------|-----------|
| ROCm version | YES |
| MI300X architecture (gfx942) | YES |
| HotAisle provider specifics | YES |
| Actual available memory | YES |

---

## 16. Final Recommendations

### Before Starting Development

1. **Run day1_validation.sh immediately** upon HotAisle access
2. **Report findings** - identify blockers early
3. **Test hipTensor** - if missing, switch to GEMM fallback plan
4. **Verify ROCm >= 5.7** - critical for compatibility

### Development Strategy

1. **Start with rocBLAS-only** - guaranteed to work
2. **Add hipTensor later** - if available and beneficial
3. **Implement Lanczos eigensolver** - don't rely on finding sparse solver
4. **Validate frequently** - compare to CPU PDMRG after every change
5. **Profile early** - use rocprof from Week 1

### Success Metrics (Revised for Realism)

**Minimum Viable Product (MVP):**
- L=12 Heisenberg correct to 1e-12
- Any measurable speedup vs CPU

**Good Success:**
- L=40 with 25x speedup
- Josephson junction (complex128) working

**Excellent Success:**
- L=100 demonstration
- 50-100x speedup on L=40

**Confidence in MVP: 🟢 HIGH (80%)**
**Confidence in Good Success: 🟡 MEDIUM (60%)**
**Confidence in Excellent Success: 🟡 MEDIUM (50%)**

---

## Summary

**What We're Sure About:**
- MI300X has enough memory (192GB >> DMRG needs)
- rocBLAS/rocSOLVER will work for core math
- We can implement everything via GEMM if needed
- Single-GPU strategy is sound

**What We're Uncertain About:**
- hipTensor availability and API
- Exact ROCm version and setup
- HotAisle provider specifics
- Optimal performance tuning

**What We Must Do on Day 1:**
- Run diagnostic script
- Test hipTensor or commit to GEMM fallback
- Verify gfx942 architecture
- Check ROCm version

**Bottom Line:**
Even in worst case (no hipTensor, old ROCm), we can build a working GPU DMRG using rocBLAS. Target 50-100x speedup is achievable but not guaranteed - conservative 25x is highly likely.
