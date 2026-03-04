# GPU Port Research Checklist - Unknowns to Verify

**Purpose:** Systematic discovery of critical unknowns before development starts

**How to use:** Run these checks on Day 1 when HotAisle access is obtained

---

## CRITICAL UNKNOWN #1: HotAisle Provider & ROCm Version

### What We Need to Know:
1. Is "HotAisle" the actual provider name?
2. What ROCm version is pre-installed?
3. Can we upgrade ROCm if needed?
4. What's the base OS (Ubuntu 22.04, RHEL, etc.)?

### Commands to Run:

```bash
# 1. Provider identification
echo "=== Provider & System Info ==="
hostname
cat /etc/os-release
uname -a

# 2. ROCm version (CRITICAL!)
rocminfo | grep "Runtime Version"
# Expected output: "Runtime Version: 6.0.x" or similar
# REQUIRED: >= 5.7 for hipTensor compatibility

# 3. ROCm installation path
echo $ROCM_PATH
ls -la /opt/rocm/
ls -la /opt/rocm-*/  # Multiple versions installed?

# 4. Detailed ROCm info
/opt/rocm/bin/rocm-smi --showdriverversion
dpkg -l | grep rocm  # On Debian/Ubuntu
rpm -qa | grep rocm  # On RHEL/CentOS

# 5. Check if we can upgrade
sudo apt-cache policy rocm-dev  # Shows available versions
# or
sudo dnf list available | grep rocm
```

### Decision Tree:

```
ROCm Version Found:
├─ 6.0+ → ✅ EXCELLENT - Proceed with confidence
├─ 5.7-5.9 → ✅ GOOD - Should work, may need manual hipTensor install
├─ 5.0-5.6 → ⚠️ RISKY - Try manual hipTensor build, or request upgrade
└─ <5.0 → ❌ BLOCKER - Must upgrade or reconsider GPU approach
```

### Research Action:
- [ ] Record exact ROCm version: _____________
- [ ] Check if upgrade possible: Yes / No
- [ ] Verify sudo access: Yes / No
- [ ] Document base OS: _____________

---

## CRITICAL UNKNOWN #2: hipTensor Availability

### What We Need to Know:
1. Is hipTensor installed by default?
2. What version of hipTensor (if any)?
3. Can we install it via package manager?
4. Do we need to build from source?

### Commands to Run:

```bash
# 1. Check for hipTensor library
echo "=== hipTensor Detection ==="
ls -lh /opt/rocm/lib/libhiptensor.so
ls -lh /opt/rocm/lib64/libhiptensor.so
find /opt/rocm -name "libhiptensor*" 2>/dev/null

# 2. Check for hipTensor headers
ls -la /opt/rocm/include/hiptensor/
ls -la /opt/rocm/include/hiptensor.h

# 3. Check package availability
apt-cache search hiptensor
apt-cache policy hiptensor
# or
dnf search hiptensor

# 4. Check installed ROCm packages
dpkg -l | grep -i tensor
rpm -qa | grep -i tensor

# 5. Try to compile a test program
cat > /tmp/test_hiptensor_exists.cpp << 'EOF'
#include <iostream>
#include <hiptensor/hiptensor.h>

int main() {
    std::cout << "hipTensor header found!" << std::endl;
    return 0;
}
EOF

hipcc -o /tmp/test_hiptensor /tmp/test_hiptensor_exists.cpp \
    -I/opt/rocm/include \
    -L/opt/rocm/lib \
    -lhiptensor 2>&1 | tee /tmp/hiptensor_compile.log

# If compilation succeeds:
if [ -f /tmp/test_hiptensor ]; then
    echo "✅ hipTensor is available and linkable"
    ./test_hiptensor
else
    echo "❌ hipTensor not available - check compile log"
    cat /tmp/hiptensor_compile.log
fi
```

### Fallback Options (if hipTensor missing):

#### Option A: Install from ROCm repos
```bash
# Try package manager first
sudo apt-get update
sudo apt-get install hiptensor hiptensor-dev
# or
sudo dnf install hiptensor hiptensor-devel

# Verify installation
ls /opt/rocm/lib/libhiptensor.so
```

#### Option B: Build from source
```bash
# Clone hipTensor repository
git clone https://github.com/ROCmSoftwarePlatform/hipTensor.git
cd hipTensor

# Check README for build instructions
cat README.md

# Typical build process:
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/opt/rocm \
      -DCMAKE_BUILD_TYPE=Release \
      ..
make -j$(nproc)
sudo make install

# Verify
ls /opt/rocm/lib/libhiptensor.so
```

#### Option C: Use rocBLAS fallback (guaranteed to work)
```bash
# If hipTensor unavailable and cannot install:
# → Switch to rocBLAS-only implementation
# → Document this decision in day1_report.txt
# → Expect ~10% performance penalty vs hipTensor
```

### Decision Matrix:

| hipTensor Status | Action | Confidence |
|------------------|--------|------------|
| Pre-installed | Use directly | 🟢 HIGH |
| In package manager | `apt/dnf install` | 🟢 HIGH |
| Must build from source | Follow Option B | 🟡 MEDIUM |
| Cannot install (no sudo) | Use rocBLAS fallback | 🟢 HIGH |
| ROCm too old (<5.7) | Request upgrade or fallback | 🔴 LOW |

### Research Action:
- [ ] hipTensor found: Yes / No / Version: _______
- [ ] Installation method: Pre-installed / Package / Source / None
- [ ] Decision: Use hipTensor / Use rocBLAS fallback

---

## CRITICAL UNKNOWN #3: hipTensor API Capabilities

**This is only relevant if hipTensor is available**

### What We Need to Know:
1. Does hipTensor support arbitrary einsum contractions?
2. What's the actual API syntax?
3. Does it support complex128 (hipDoubleComplex)?
4. Are batch operations available?
5. Can operations be launched on streams?

### Test Program #1: Basic Contraction

```cpp
// File: test_hiptensor_basic.cpp
// Test if hipTensor can do what we need

#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include <hiptensor/hiptensor.h>
#include <iostream>
#include <vector>

int main() {
    std::cout << "Testing hipTensor for DMRG contractions\n";
    std::cout << "=========================================\n\n";

    // Test 1: Initialize hipTensor
    hiptensorHandle_t handle;
    hiptensorStatus_t status = hiptensorCreate(&handle);

    if (status != HIPTENSOR_STATUS_SUCCESS) {
        std::cerr << "FAIL: Cannot create hipTensor handle\n";
        std::cerr << "Error code: " << status << "\n";
        return 1;
    }
    std::cout << "✓ hipTensor handle created\n";

    // Test 2: Define a DMRG 2-site contraction
    // einsum('ijk,klm->ijlm', A, B)
    // A: D × d × D (left MPS tensor)
    // B: D × d × D (right MPS tensor)
    // C: D × d × d × D (output)

    const int D = 50;   // Bond dimension
    const int d = 5;    // Physical dimension

    // Allocate tensors on GPU
    hipDoubleComplex *d_A, *d_B, *d_C;
    size_t size_A = D * d * D * sizeof(hipDoubleComplex);
    size_t size_B = D * d * D * sizeof(hipDoubleComplex);
    size_t size_C = D * d * d * D * sizeof(hipDoubleComplex);

    hipMalloc(&d_A, size_A);
    hipMalloc(&d_B, size_B);
    hipMalloc(&d_C, size_C);

    std::cout << "✓ GPU memory allocated\n";
    std::cout << "  A: " << D << "×" << d << "×" << D << " = " << size_A/1024 << " KB\n";
    std::cout << "  B: " << D << "×" << d << "×" << D << " = " << size_B/1024 << " KB\n";
    std::cout << "  C: " << D << "×" << d << "×" << d << "×" << D << " = " << size_C/1024 << " KB\n\n";

    // Test 3: Try to create tensor descriptors
    // NOTE: Actual API may differ - check documentation!
    // This is speculative based on similar APIs (cuTENSOR)

    std::cout << "Attempting tensor contraction setup...\n";

    // Tensor mode extents (dimensions)
    std::vector<int64_t> extent_A = {D, d, D};  // i, j, k
    std::vector<int64_t> extent_B = {D, d, D};  // k, l, m
    std::vector<int64_t> extent_C = {D, d, d, D};  // i, j, l, m

    // Tensor mode labels (Einstein notation)
    std::vector<int32_t> mode_A = {'i', 'j', 'k'};
    std::vector<int32_t> mode_B = {'k', 'l', 'm'};
    std::vector<int32_t> mode_C = {'i', 'j', 'l', 'm'};

    // TODO: Fill in actual hipTensor API calls
    // The following is PSEUDOCODE - actual API may be different:

    /*
    hiptensorTensorDescriptor_t desc_A, desc_B, desc_C;
    hiptensorInitTensorDescriptor(
        &desc_A,
        3,                           // number of modes
        extent_A.data(),
        nullptr,                     // strides (use default)
        HIPTENSOR_R_64F,            // complex double (check actual enum!)
        HIPTENSOR_OP_IDENTITY
    );

    // Similar for desc_B and desc_C

    hiptensorContractionDescriptor_t contraction;
    hiptensorInitContractionDescriptor(
        &contraction,
        &desc_A,
        mode_A.data(),
        &desc_B,
        mode_B.data(),
        &desc_C,
        mode_C.data(),
        HIPTENSOR_COMPUTE_64F
    );

    // Execute contraction
    const hipDoubleComplex alpha = make_hipDoubleComplex(1.0, 0.0);
    const hipDoubleComplex beta = make_hipDoubleComplex(0.0, 0.0);

    hiptensorContraction(
        handle,
        &contraction,
        &alpha,
        d_A,
        d_B,
        &beta,
        d_C,
        d_C,
        nullptr,  // workspace
        0,        // workspace size
        nullptr   // stream
    );

    hipDeviceSynchronize();
    */

    std::cout << "\n";
    std::cout << "⚠️  API test incomplete - need actual hipTensor docs\n";
    std::cout << "Next steps:\n";
    std::cout << "1. Check /opt/rocm/include/hiptensor/ for headers\n";
    std::cout << "2. Find hipTensor examples or documentation\n";
    std::cout << "3. Verify einsum notation support\n";
    std::cout << "4. Test with actual contraction\n";

    // Cleanup
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    hiptensorDestroy(handle);

    std::cout << "\n✓ Test completed without crashes\n";
    return 0;
}
```

**Compile and run:**
```bash
hipcc -o test_hiptensor_basic test_hiptensor_basic.cpp \
    -I/opt/rocm/include \
    -L/opt/rocm/lib \
    -lhiptensor \
    --offload-arch=gfx942

./test_hiptensor_basic
```

### Test Program #2: Stream Support

```cpp
// File: test_hiptensor_streams.cpp
#include <hip/hip_runtime.h>
#include <hiptensor/hiptensor.h>
#include <iostream>

int main() {
    std::cout << "Testing hipTensor with HIP streams\n\n";

    hiptensorHandle_t handle;
    hiptensorCreate(&handle);

    // Create HIP streams
    hipStream_t stream1, stream2;
    hipStreamCreate(&stream1);
    hipStreamCreate(&stream2);

    std::cout << "✓ Streams created\n";

    // Check if hipTensor can use streams
    // API: hiptensorSetStream(handle, stream1);
    // or similar

    std::cout << "⚠️  Need to test if hipTensor supports stream assignment\n";
    std::cout << "Check documentation for hiptensorSetStream or equivalent\n";

    hipStreamDestroy(stream1);
    hipStreamDestroy(stream2);
    hiptensorDestroy(handle);

    return 0;
}
```

### Information to Gather:

From `/opt/rocm/include/hiptensor/`:
```bash
# Find and read header files
ls -la /opt/rocm/include/hiptensor/
cat /opt/rocm/include/hiptensor/hiptensor.h | grep -A 5 "typedef\|enum\|struct"

# Look for examples
find /opt/rocm -name "*hiptensor*example*" -o -name "*hiptensor*sample*"
ls /opt/rocm/share/hiptensor/

# Check for documentation
find /opt/rocm -name "*hiptensor*.md" -o -name "*hiptensor*.rst"
```

### Research Action:
- [ ] hipTensor API documented: Yes / No / Where: __________
- [ ] Einsum support confirmed: Yes / No / How: __________
- [ ] Complex128 support: Yes / No
- [ ] Stream support: Yes / No
- [ ] Example code found: Yes / No / Path: __________

---

## CRITICAL UNKNOWN #4: rocSOLVER & Eigensolvers

### What We Need to Know:
1. Is rocSOLVER definitely available?
2. What eigensolver functions exist?
3. Is there ANY sparse eigensolver in AMD ecosystem?
4. What's the performance of dense SVD for DMRG sizes?

### Commands to Run:

```bash
# 1. Verify rocSOLVER
ls -lh /opt/rocm/lib/librocsolver.so
ls -lh /opt/rocm/lib64/librocsolver.so

# 2. Check version
strings /opt/rocm/lib/librocsolver.so | grep -i version | head -5

# 3. List available functions
nm -D /opt/rocm/lib/librocsolver.so | grep solver | grep -i "svd\|eig\|heev"

# Expect to see:
# rocsolver_zgesvd - Complex SVD ✓
# rocsolver_zheevd - Complex Hermitian eigenvalues ✓
# rocsolver_zgeev  - General eigenvalues (may exist)

# 4. Check for sparse linear algebra
ls /opt/rocm/lib/librocsp* 2>/dev/null
# rocSPARSE exists for sparse BLAS, but no sparse eigensolvers!

# 5. Search for any eigensolver libraries
find /opt/rocm -name "*eig*" -o -name "*lanczos*" -o -name "*arnoldi*"
```

### Test SVD Performance:

```cpp
// File: test_rocsolver_svd.cpp
#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include <rocsolver/rocsolver.h>
#include <rocblas/rocblas.h>
#include <iostream>
#include <chrono>

int main() {
    const int M = 250;  // D × d = 50 × 5
    const int N = 250;
    const int K = std::min(M, N);

    std::cout << "Testing rocSOLVER SVD performance\n";
    std::cout << "Matrix size: " << M << " × " << N << "\n\n";

    // Create rocBLAS handle
    rocblas_handle handle;
    rocblas_create_handle(&handle);

    // Allocate GPU memory
    hipDoubleComplex *d_A;
    double *d_S;
    hipDoubleComplex *d_U, *d_Vt;

    hipMalloc(&d_A, M * N * sizeof(hipDoubleComplex));
    hipMalloc(&d_S, K * sizeof(double));
    hipMalloc(&d_U, M * M * sizeof(hipDoubleComplex));
    hipMalloc(&d_Vt, N * N * sizeof(hipDoubleComplex));

    // Initialize A with random data
    std::vector<hipDoubleComplex> h_A(M * N);
    for (auto& x : h_A) {
        x = make_hipDoubleComplex(rand() / double(RAND_MAX),
                                   rand() / double(RAND_MAX));
    }
    hipMemcpy(d_A, h_A.data(), M * N * sizeof(hipDoubleComplex),
              hipMemcpyHostToDevice);

    // Query workspace size
    rocblas_int *d_info;
    hipMalloc(&d_info, sizeof(rocblas_int));

    // Time the SVD
    auto start = std::chrono::high_resolution_clock::now();

    rocsolver_zgesvd(
        handle,
        rocblas_svect_all,      // Compute all left singular vectors
        rocblas_svect_all,      // Compute all right singular vectors
        M, N,
        d_A, M,                 // Input matrix (destroyed on output)
        d_S,                    // Singular values
        d_U, M,                 // Left singular vectors
        d_Vt, N,                // Right singular vectors
        nullptr, 0,             // Work array (query first!)
        d_info                  // Convergence info
    );

    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Check convergence
    int info;
    hipMemcpy(&info, d_info, sizeof(int), hipMemcpyDeviceToHost);

    std::cout << "SVD completed in " << duration.count() << " ms\n";
    std::cout << "Convergence info: " << info << " (0 = success)\n";

    // Get largest singular value
    double S_max;
    hipMemcpy(&S_max, d_S, sizeof(double), hipMemcpyDeviceToHost);
    std::cout << "Largest singular value: " << S_max << "\n";

    // Cleanup
    hipFree(d_A);
    hipFree(d_S);
    hipFree(d_U);
    hipFree(d_Vt);
    hipFree(d_info);
    rocblas_destroy_handle(handle);

    std::cout << "\n✓ rocSOLVER SVD works correctly\n";
    return 0;
}
```

**Compile and run:**
```bash
hipcc -o test_svd test_rocsolver_svd.cpp \
    -lrocsolver \
    -lrocblas \
    --offload-arch=gfx942

./test_svd
```

### Sparse Eigensolver Decision:

```
AMD Sparse Eigensolver Available?
├─ YES (unlikely) → Use it directly
└─ NO (expected) → Implement Lanczos ourselves
    ├─ Use rocBLAS for BLAS operations
    ├─ Implement Lanczos iteration in custom kernel
    └─ This is standard DMRG approach anyway
```

### Research Action:
- [ ] rocSOLVER available: Yes / No
- [ ] SVD performance acceptable: Yes / No / Time: _____ ms
- [ ] Sparse eigensolver found: Yes / No / Name: __________
- [ ] Decision: Implement Lanczos / Use existing library

---

## UNKNOWN #5: MI300X Architecture & Memory

### What We Need to Know:
1. Verify this is actually gfx942 (MI300X)
2. Confirm 192GB HBM3 memory
3. Check actual available memory
4. Verify we can allocate large chunks (>100GB)

### Commands to Run:

```bash
# 1. GPU architecture
rocminfo | grep -A 10 "Name:" | grep -i "gfx\|arch"
# Expected: gfx942

# 2. GPU memory
rocm-smi --showmeminfo vram
# Expected: ~192 GB total

# 3. Detailed device info
rocminfo | grep -i "mem\|size" | head -20

# 4. Check multiple GPUs
rocm-smi --showid
# How many MI300X devices?

# 5. Get exact memory
rocm-smi --showmeminfo vram --json | grep -i total
```

### Test Large Allocation:

```cpp
// File: test_large_allocation.cpp
#include <hip/hip_runtime.h>
#include <iostream>

int main() {
    std::cout << "Testing large GPU memory allocations\n\n";

    // Get device properties
    int device;
    hipGetDevice(&device);

    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, device);

    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Architecture: " << prop.gcnArchName << "\n";
    std::cout << "Total Memory: " << prop.totalGlobalMem / (1024ULL*1024*1024) << " GB\n\n";

    if (std::string(prop.gcnArchName).find("gfx942") == std::string::npos) {
        std::cout << "⚠️  WARNING: Not gfx942 (MI300X)!\n";
        std::cout << "  Expected: gfx942\n";
        std::cout << "  Got: " << prop.gcnArchName << "\n";
    }

    // Check available memory
    size_t free, total;
    hipMemGetInfo(&free, &total);

    std::cout << "Memory Status:\n";
    std::cout << "  Total: " << total / (1024ULL*1024*1024) << " GB\n";
    std::cout << "  Free:  " << free / (1024ULL*1024*1024) << " GB\n";
    std::cout << "  Used:  " << (total - free) / (1024ULL*1024*1024) << " GB\n\n";

    // Test allocations of increasing size
    std::vector<size_t> test_sizes = {
        1ULL << 30,      // 1 GB
        10ULL << 30,     // 10 GB
        50ULL << 30,     // 50 GB
        100ULL << 30,    // 100 GB
        150ULL << 30     // 150 GB
    };

    for (size_t size : test_sizes) {
        void* d_ptr = nullptr;
        hipError_t err = hipMalloc(&d_ptr, size);

        if (err == hipSuccess) {
            std::cout << "✓ Successfully allocated " << size / (1024ULL*1024*1024) << " GB\n";
            hipFree(d_ptr);
        } else {
            std::cout << "✗ Failed to allocate " << size / (1024ULL*1024*1024) << " GB\n";
            std::cout << "  Error: " << hipGetErrorString(err) << "\n";
            break;
        }
    }

    std::cout << "\n";

    // Report maximum allocation
    hipMemGetInfo(&free, &total);
    std::cout << "Maximum safe allocation: ~" << (free * 0.9) / (1024ULL*1024*1024) << " GB\n";

    return 0;
}
```

**Compile and run:**
```bash
hipcc -o test_memory test_large_allocation.cpp --offload-arch=gfx942
./test_memory
```

### Research Action:
- [ ] Architecture confirmed: _________ (need gfx942)
- [ ] Total memory: _________ GB (need ~192 GB)
- [ ] Max allocation tested: _________ GB
- [ ] Multiple GPUs present: Yes / No / Count: _____

---

## UNKNOWN #6: Compilation & Build Environment

### What We Need to Know:
1. CMake version
2. C++ compiler and version
3. Can we use C++17 features?
4. Build system preferences

### Commands to Run:

```bash
# 1. CMake
cmake --version
# Need: >= 3.20

# 2. C++ compiler
g++ --version
clang++ --version
hipcc --version

# 3. Test C++17
cat > /tmp/test_cpp17.cpp << 'EOF'
#include <iostream>
#include <optional>
#include <variant>

int main() {
    std::optional<int> opt = 42;
    std::variant<int, double> var = 3.14;
    std::cout << "C++17 features work!\n";
    return 0;
}
EOF

hipcc -std=c++17 -o /tmp/test_cpp17 /tmp/test_cpp17.cpp
./tmp/test_cpp17

# 4. Check available tools
which make
which ninja
which ccache  # Build cache for faster recompilation
```

### Research Action:
- [ ] CMake version: _________
- [ ] C++ compiler: _________ version _________
- [ ] C++17 support: Yes / No
- [ ] Build tools available: make / ninja / other: _________

---

## Summary Template (Fill in after running all checks)

```
=================================================
GPU PORT ENVIRONMENT VALIDATION SUMMARY
=================================================

Provider: ________________
OS: ________________
ROCm Version: ________________

CRITICAL COMPONENTS:
[ ] GPU Architecture: _________ (need gfx942)
[ ] GPU Memory: _________ GB (need ~192 GB)
[ ] rocBLAS: _________ (version)
[ ] rocSOLVER: _________ (version)
[ ] hipTensor: _________ (installed/missing/version)

BUILD ENVIRONMENT:
[ ] CMake: _________ (need >= 3.20)
[ ] hipcc: _________ (version)
[ ] C++17: Supported / Not supported

DECISIONS MADE:
[ ] hipTensor strategy: Use native / Use rocBLAS fallback
[ ] Eigensolver: Implement Lanczos / Found library: _______
[ ] Development approach: Docker / Native

CONFIDENCE UPDATES:
Components with HIGH confidence:
-
-

Components with LOW confidence (need alternatives):
-
-

BLOCKERS IDENTIFIED:
-
-

READY TO PROCEED: YES / NO / CONDITIONAL
If conditional, what needs to change:
-
-

=================================================
```

---

## Next Steps After Research

1. **Share findings** - Send completed summary to planning team
2. **Update CONFIDENCE_ANALYSIS.md** - Revise confidence ratings based on facts
3. **Make go/no-go decision** - Can we proceed with GPU development?
4. **Choose primary path** - hipTensor vs rocBLAS fallback
5. **Begin Week 1 development** - Start with what we confirmed works

---

## Emergency Fallback Decision Tree

```
Can we do GPU DMRG?
│
├─ MI300X confirmed + rocBLAS + rocSOLVER?
│  └─ YES → Proceed with rocBLAS-only implementation
│     Confidence: HIGH, Expected speedup: 25-40x
│
├─ MI300X confirmed + hipTensor working?
│  └─ YES → Proceed with optimal implementation
│     Confidence: HIGH, Expected speedup: 50-100x
│
├─ Wrong GPU (not MI300X) but AMD + ROCm?
│  └─ Adjust arch target, reduce memory expectations
│     Confidence: MEDIUM, Expected speedup: 10-25x
│
└─ No ROCm, wrong vendor, or <5.0 ROCm?
   └─ STOP → Reconsider GPU approach or request different instance
      Consider: CPU optimization, cloud migration, etc.
```

