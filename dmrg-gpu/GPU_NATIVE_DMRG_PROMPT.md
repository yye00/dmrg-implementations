# GPU-Native Reference DMRG: Eliminate All CPU Contractions

## Objective

The current `dmrg-gpu` implementation is **25x slower than single-threaded CPU numpy** because
every tensor contraction copies data to CPU, runs nested for-loops, and copies back. Fix this
so that **everything runs on GPU** from start to finish. The only acceptable CPU work is program
startup, printing results, and trivial control flow (loop counters, convergence checks on single
scalars).

**Benchmark target:** L=32, chi=64, 5 sweeps must be faster than 4.7s (the single-threaded
CPU numpy time). Realistically, with proper GPU contractions on MI300X, this should be **well
under 1 second**.

## Current Architecture (what's broken)

All files in `/home/hotaisle/dmrg-implementations/dmrg-gpu/src/`.

### Five CPU bottlenecks to eliminate:

| # | Function | What it does on CPU | Calls per run (L=32, chi=64, 5 sweeps) |
|---|----------|--------------------|-----------------------------------------|
| 1 | `apply_heff()` | 3-step tensor contraction via nested for-loops | ~15,000+ (Lanczos matvec, called per iteration per site per sweep) |
| 2 | `update_left_env()` | 3-step tensor contraction via nested for-loops | ~155 (once per site per L→R half-sweep) |
| 3 | `update_right_env()` | 3-step tensor contraction via nested for-loops | ~155 (once per site per R→L half-sweep + initial build) |
| 4 | `svd_and_update_mps()` | S*Vh matrix multiply + neighbor absorption | ~310 (once per site per sweep) |
| 5 | `lanczos_eigensolver()` tridiagonal solve | LAPACK `dstev_` on CPU | ~310 |

**#1 is the dominant bottleneck by far** — it accounts for >95% of runtime.

### What already works on GPU:
- MPS/MPO/environment storage (all `hipMalloc`)
- Lanczos BLAS-1 ops: `rocblas_dnrm2`, `rocblas_dscal`, `rocblas_ddot`, `rocblas_daxpy`
- Lanczos eigenvector reconstruction: `rocblas_dgemv`
- SVD: `AccurateSVD_GPU::decompose()` via `rocsolver_dgesvd`

## Existing GPU Infrastructure to Reuse

From `pdmrg-gpu/` — these are already written and tested:

### 1. hipTensor H_eff Contraction: `heff_optimized_gpu.h/.cpp`

**Location:** `/home/hotaisle/dmrg-implementations/pdmrg-gpu/src/heff_optimized_gpu.h` and `.cpp`

`OptimizedHeff` class uses hipTensor with pre-compiled contraction plans and workspace caching.
Data type: `double` (real FP64) — **matches dmrg-gpu exactly**.

Current implementation does 2-site (4 steps). For 1-site DMRG, adapt to 3 steps:

```
Step 1: T1[w, a', s, b]   = L[a, w, a']  * theta[a, s, b]     — contract over 'a'
Step 2: T2[a', s', w', b]  = W[w, s, s', w'] * T1[w, a', s, b] — contract over 'w','s'
Step 3: result[a', s', b'] = R[b, w', b'] * T2[a', s', w', b]  — contract over 'w','b'
```

Each step is a single `hiptensorContract()` call. Plans get created once and reused.

### 2. GPU-Native Lanczos: `lanczos_eigensolver_gpu_native.hpp`

**Location:** `/home/hotaisle/dmrg-implementations/pdmrg-gpu/include/lanczos_eigensolver_gpu_native.hpp`

Fully GPU-native Lanczos using `rocsolver_dsyev` for tridiagonal solve (replaces CPU `dstev_`).
Currently uses complex types — adapt to real (replace `z` prefix BLAS calls with `d` prefix).

### 3. GPU Memory Utilities: `gpu_memory.hpp`

**Location:** `/home/hotaisle/dmrg-implementations/pdmrg-gpu/include/gpu_memory.hpp`

RAII `GPUBuffer<T>` with move semantics, resize, zero-fill. Use this instead of raw hipMalloc.

## Implementation Plan

### Phase 1: GPU `apply_heff()` — THE CRITICAL FIX

This is the **only phase that matters for the 25x speedup**. Everything else is gravy.

**Replace the CPU for-loop `apply_heff()` with hipTensor contractions.**

Create a new class `HeffGPU1Site` (adapted from `OptimizedHeff`):

```cpp
class HeffGPU1Site {
public:
    // Constructor: create hipTensor descriptors and plans for given dimensions
    HeffGPU1Site(int chi_L, int chi_R, int d, int D_mpo);
    ~HeffGPU1Site();

    // Apply H_eff to theta entirely on GPU
    // theta[a, s, b] -> result[a', s', b']
    void apply(const double* d_L_env,      // (chi_L, D_mpo, chi_L)
               const double* d_R_env,      // (chi_R, D_mpo, chi_R)
               const double* d_W,          // (D_mpo, d, d, D_mpo)
               const double* d_theta,      // (chi_L, d, chi_R)
               double* d_result,           // (chi_L, d, chi_R)
               hipStream_t stream = 0);

private:
    // Pre-allocated intermediates
    double* d_T1_;  // (D_mpo, chi_L, d, chi_R)  — after step 1
    double* d_T2_;  // (chi_L, d, D_mpo, chi_R)  — after step 2

    // hipTensor handles and plans (created once, reused every call)
    hiptensorHandle_t handle_;
    hiptensorContractionPlan_t plan_step1_, plan_step2_, plan_step3_;
    // workspace buffers
    double* d_workspace1_, *d_workspace2_, *d_workspace3_;
    size_t ws_size1_, ws_size2_, ws_size3_;

    int chi_L_, chi_R_, d_, D_mpo_;
};
```

**Implementation notes:**

- Use the ROCm 7.2 hipTensor API (same as `heff_optimized_gpu.cpp` uses)
- Create tensor descriptors with `hiptensorCreateTensorDescriptor`
- Create contraction with `hiptensorCreateContraction` specifying mode labels
- Create plan with `hiptensorCreatePlan` + `hiptensorCreatePlanPreference`
- Execute with `hiptensorContract`
- **Plans are created once in constructor and reused for every `apply()` call**
- Intermediates `d_T1_` and `d_T2_` are pre-allocated in constructor
- For varying bond dimensions across sites, either:
  - (Simple) Create one `HeffGPU1Site` per unique `(chi_L, chi_R)` pair, cache them
  - (Simpler) Allocate for max dimensions, use actual dimensions in descriptors

**Key hipTensor contraction setup (Step 1 example):**

```cpp
// Step 1: T1[w, a', s, b] = L[a, w, a'] * theta[a, s, b]
// Contract over index 'a'
// L modes: {a=0, w=1, a'=2}, theta modes: {a=0, s=3, b=4}
// T1 modes: {w=1, a'=2, s=3, b=4}

int64_t extL[] = {chi_L, D_mpo, chi_L};
int64_t extTheta[] = {chi_L, d, chi_R};
int64_t extT1[] = {D_mpo, chi_L, d, chi_R};

int32_t modesL[] = {'a', 'w', 'p'};      // a, w, a'
int32_t modesTheta[] = {'a', 's', 'b'};  // a, s, b
int32_t modesT1[] = {'w', 'p', 's', 'b'}; // w, a', s, b
```

### Phase 2: GPU Environment Updates

Replace `update_left_env()` and `update_right_env()` with hipTensor contractions.
Same 3-step pattern as H_eff but different index structure.

**Left environment update:**
```
L_new[b, w', b'] = sum_{a,s,w} L[a, w, a'] * A[a, s, b] * W[w, s, s', w'] * A*[a', s', b']
```
3 hipTensor contraction steps (same pattern as apply_heff with different indices).

**Right environment update:**
```
R_new[a, w, a'] = sum_{b,s',w'} A[a, s, b] * W[w, s, s', w'] * R[b, w', b'] * A*[a', s', b']
```

These are called much less frequently than `apply_heff`, so they're lower priority.
But they still involve `hipMemcpy` round-trips that add up.

### Phase 3: GPU SVD Absorption

Replace the CPU matrix multiply in `svd_and_update_mps()` with `rocblas_dgemm`:

```cpp
// After SVD: theta = U * S * Vh
// Direction 'R': A[site] = U, absorb S*Vh into A[site+1]
//   SV = diag(S) @ Vh  -- use rocblas_dgemm or rocblas_ddgmm
//   A[site+1] = SV @ A[site+1]  -- use rocblas_dgemm

// Direction 'L': A[site] = Vh, absorb U*S into A[site-1]
//   US = U @ diag(S)  -- use rocblas_dgemm or rocblas_ddgmm
//   A[site-1] = A[site-1] @ US  -- use rocblas_dgemm
```

### Phase 4: GPU Tridiagonal Eigensolver

Replace `dstev_` (CPU LAPACK) with `rocsolver_dstev` or `rocsolver_dsyev`:

```cpp
// Build tridiagonal matrix T on GPU from alpha/beta Lanczos coefficients
// T is only ~20x20 to 50x50, so this isn't a bottleneck
// But it eliminates the last CPU computation

// Option A: rocsolver_dstev (tridiagonal eigensolve, if available)
// Option B: Build full symmetric T matrix, use rocsolver_dsyev
rocsolver_dsyev(rocblas_h_, rocblas_evect_original, rocblas_fill_upper,
                niter, d_T, niter, d_eigenvalues, d_E, d_info);
```

**Note:** The Lanczos alpha/beta values are currently computed as individual scalars on host
(from `rocblas_ddot` results). This is fine — they're single doubles needed for control flow.
Just accumulate them on GPU into the tridiagonal matrix directly.

## Memory Layout Convention

All tensors use column-major (Fortran) order, matching rocBLAS and hipTensor defaults:

```
MPS:  A[a, s, b]     stored as a + s*chi_L + b*chi_L*d
MPO:  W[w, s, s', w'] stored as w + s*D + s'*D*d + w'*D*d*d
L_env: L[a, w, a']    stored as a + w*chi_L + a'*chi_L*D
R_env: R[b, w', b']   stored as b + w'*chi_R + b'*chi_R*D
```

## Build Configuration

The `dmrg-gpu/CMakeLists.txt` needs to add hipTensor:

```cmake
find_library(HIPTENSOR_LIBRARY NAMES hiptensor
    PATHS ${ROCM_PATH}/lib ${ROCM_PATH}/lib64)
find_path(HIPTENSOR_INCLUDE_DIR hiptensor/hiptensor.h
    PATHS ${ROCM_PATH}/include)
include_directories(${HIPTENSOR_INCLUDE_DIR})
target_link_libraries(dmrg_gpu ... ${HIPTENSOR_LIBRARY})
```

## Verification

After each phase, verify correctness:

```bash
cd /home/hotaisle/dmrg-implementations/dmrg-gpu/build
cmake .. -DGPU_TARGETS=gfx942 && make -j8

# Accuracy check
./dmrg_gpu 4 16 10    # L=4: expect -1.616025403784, error < 1e-10
./dmrg_gpu 8 32 10    # L=8: expect -3.374932598688, error < 1e-10

# Performance benchmark
./dmrg_gpu 32 64 5    # Must beat 4.7s (CPU numpy baseline)
```

## What NOT to Do

- Do NOT use `hipMemcpy(..., hipMemcpyDeviceToHost)` for any tensor data during the algorithm.
  The only acceptable D→H transfers are single scalars (energy, norms, convergence checks).
- Do NOT fall back to CPU for any contraction. If hipTensor has issues, use rocBLAS GEMM with
  explicit reshape/permute kernels.
- Do NOT add unnecessary synchronization. Use async operations and streams where possible.
- Do NOT change the algorithm. Single-site DMRG with Lanczos eigensolver and SVD truncation.
  Just move the computation to GPU.

## Files to Modify

1. **`src/dmrg_gpu.cpp`** — Replace CPU contraction functions with GPU calls
2. **`src/dmrg_gpu.h`** — Add HeffGPU1Site member, remove CPU-specific fields
3. **`src/test_dmrg_gpu.cpp`** — Already has timing, no changes needed
4. **`CMakeLists.txt`** — Add hipTensor dependency

## Files to Create

1. **`src/heff_gpu_1site.h`** — HeffGPU1Site class declaration
2. **`src/heff_gpu_1site.cpp`** — hipTensor-based 1-site H_eff implementation
3. **`src/env_update_gpu.h`** — GPU environment update declarations
4. **`src/env_update_gpu.cpp`** — hipTensor-based environment updates

Or alternatively, put everything in `dmrg_gpu.cpp` if that's simpler.

## Reference: hipTensor API (ROCm 7.2)

```cpp
#include <hiptensor/hiptensor.h>

// 1. Create handle (once)
hiptensorHandle_t handle;
hiptensorCreate(&handle);

// 2. Create tensor descriptors
hiptensorTensorDescriptor_t descA, descB, descC;
int64_t extA[] = {m, k};
int64_t strA[] = {1, m};  // column-major
hiptensorCreateTensorDescriptor(handle, &descA, 2, extA, strA, HIP_R_64F, HIPTENSOR_OP_IDENTITY);

// 3. Create contraction descriptor
int32_t modesA[] = {'i', 'k'};
int32_t modesB[] = {'k', 'j'};
int32_t modesC[] = {'i', 'j'};
hiptensorContractionDescriptor_t desc;
hiptensorCreateContraction(handle, &desc,
    descA, modesA, HIPTENSOR_OP_IDENTITY,
    descB, modesB, HIPTENSOR_OP_IDENTITY,
    descC, modesC, HIPTENSOR_OP_IDENTITY,
    descC, modesC, HIP_R_64F);

// 4. Create plan
hiptensorContractionFind_t find;
hiptensorCreateContractionFind(handle, &find, HIPTENSOR_ALGO_DEFAULT);
size_t ws_size;
hiptensorContractionGetWorkspaceSize(handle, desc, find, HIPTENSOR_WORKSPACE_RECOMMENDED, &ws_size);
void* d_workspace;
hipMalloc(&d_workspace, ws_size);
hiptensorContractionPlan_t plan;
hiptensorCreateContractionPlan(handle, &plan, desc, find, ws_size);

// 5. Execute (reuse plan for many calls)
double alpha = 1.0, beta = 0.0;
hiptensorContract(handle, plan, &alpha, d_A, d_B, &beta, d_C, d_C, d_workspace, ws_size, stream);

// 6. Cleanup
hiptensorDestroyPlan(plan);
hiptensorDestroyContractionFind(find);
hiptensorDestroyContraction(desc);
hiptensorDestroyTensorDescriptor(descA);
hiptensorDestroy(handle);
```

**IMPORTANT:** Check the actual API in `/opt/rocm/include/hiptensor/hiptensor.h` on the remote
machine — the ROCm 7.2 API may differ slightly from the above. The `heff_optimized_gpu.cpp`
file in pdmrg-gpu is the **ground truth** for working hipTensor calls on this system.
