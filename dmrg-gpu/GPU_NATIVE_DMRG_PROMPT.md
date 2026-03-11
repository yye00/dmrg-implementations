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

## Machine and Build Environment

- **Remote host:** `ssh hotaisle@23.183.40.79` (passwordless)
- **GPU:** AMD Instinct MI300X (gfx942)
- **ROCm:** 7.2 at `/opt/rocm`
- **Project path:** `/home/hotaisle/dmrg-implementations/dmrg-gpu/`
- **hipTensor header:** `/opt/rocm/include/hiptensor/hiptensor.h` (confirmed present)
- **hipTensor library:** `/opt/rocm/lib/libhiptensor.so` (confirmed present)

## Current Architecture (what's broken)

All source files in `/home/hotaisle/dmrg-implementations/dmrg-gpu/src/`.

The code is a **working, verified** single-site DMRG that produces correct results:
- L=4: error 4.4e-13
- L=8: error 1.0e-13

But it's catastrophically slow because the hot path runs on CPU.

### Five CPU bottlenecks to eliminate:

| # | Function | What it does on CPU | Calls per run (L=32, chi=64, 5 sweeps) |
|---|----------|--------------------|-----------------------------------------|
| 1 | `apply_heff()` | 3-step tensor contraction via nested for-loops with D↔H copies | ~15,000+ (Lanczos matvec) |
| 2 | `update_left_env()` | 3-step tensor contraction via nested for-loops with D↔H copies | ~155 |
| 3 | `update_right_env()` | 3-step tensor contraction via nested for-loops with D↔H copies | ~155 |
| 4 | `svd_and_update_mps()` | CPU matrix multiply for S*Vh and neighbor absorption | ~310 |
| 5 | `lanczos_eigensolver()` | LAPACK `dstev_` tridiagonal solve on CPU | ~310 |

**#1 is the dominant bottleneck** — >95% of runtime.

### What already works on GPU:
- MPS/MPO/environment storage (all `hipMalloc`)
- Lanczos BLAS-1: `rocblas_dnrm2`, `rocblas_dscal`, `rocblas_ddot`, `rocblas_daxpy`
- Lanczos eigenvector reconstruction: `rocblas_dgemv`
- SVD: `AccurateSVD_GPU::decompose()` via `rocsolver_dgesvd`

## Implementation Plan

All new code goes in `dmrg-gpu/src/`. Do NOT reference or copy from `pdmrg-gpu/` — that
code has unresolved correctness issues.

### Phase 1: GPU `apply_heff()` — THE CRITICAL FIX (>95% of speedup)

**Replace the CPU for-loop `apply_heff()` with hipTensor contractions.**

The single-site H_eff contraction is:

```
result[a', s', b'] = Σ_{a,s,b,w,w'} L[a, w, a'] * W[w, s, s', w'] * R[b, w', b'] * theta[a, s, b]
```

Split into 3 hipTensor steps:

```
Step 1: T1[w, a', s, b]    = Σ_a   L[a, w, a']       * theta[a, s, b]
Step 2: T2[a', s', w', b]  = Σ_{w,s} W[w, s, s', w'] * T1[w, a', s, b]
Step 3: result[a', s', b'] = Σ_{w',b} R[b, w', b']   * T2[a', s', w', b]
```

Create a class `HeffGPU1Site`:

```cpp
class HeffGPU1Site {
public:
    HeffGPU1Site(int chi_L, int chi_R, int d, int D_mpo);
    ~HeffGPU1Site();

    void apply(const double* d_L_env,   // (chi_L, D_mpo, chi_L)
               const double* d_R_env,   // (chi_R, D_mpo, chi_R)
               const double* d_W,       // (D_mpo, d, d, D_mpo)
               const double* d_theta,   // (chi_L, d, chi_R)
               double* d_result,        // (chi_L, d, chi_R)
               hipStream_t stream = 0);

private:
    double* d_T1_;  // intermediate after step 1
    double* d_T2_;  // intermediate after step 2

    hiptensorHandle_t handle_;
    hiptensorPlan_t plan1_, plan2_, plan3_;
    void* d_ws1_; void* d_ws2_; void* d_ws3_;
    uint64_t ws_size1_, ws_size2_, ws_size3_;

    int chi_L_, chi_R_, d_, D_mpo_;
};
```

**Key:** Plans are created once in constructor and reused for every `apply()` call.
For varying bond dimensions across sites, create one `HeffGPU1Site` per unique
`(chi_L, chi_R)` pair and cache them in an `std::map`.

### Phase 2: GPU Environment Updates

Replace `update_left_env()` and `update_right_env()` with hipTensor contractions.

**Left environment update (3 steps):**
```
T1[w, a', s, b]   = Σ_a     L[a, w, a']       * A[a, s, b]
T2[a', s', w', b]  = Σ_{w,s} W[w, s, s', w']  * T1[w, a', s, b]
L_new[b, w', b']   = Σ_{a',s'} A*[a', s', b'] * T2[a', s', w', b]
```

**Right environment update (3 steps):**
```
T1[a, s, w', b']   = Σ_b     A[a, s, b]        * R[b, w', b']
T2[a, s', w, b']   = Σ_{w',s} W[w, s, s', w']  * T1[a, s, w', b']
R_new[a, w, a']    = Σ_{s',b'} A*[a', s', b']  * T2[a, s', w, b']
```

Same hipTensor pattern: create descriptors, plans once, reuse.

### Phase 3: GPU SVD Absorption

Replace the CPU matrix multiply in `svd_and_update_mps()` with `rocblas_dgemm`:

```cpp
// Direction 'R': A[site] = U, absorb S*Vh into next site
// 1. Scale rows of Vh by S:  SV[i,j] = S[i] * Vh[i,j]
//    Use rocblas_ddgmm(side_left, k, n, Vh, k, S, 1, SV, k)
// 2. Multiply into neighbor: A[site+1] = SV @ A_old[site+1]
//    Use rocblas_dgemm(N, N, k, cols_next, old_chi, 1.0, SV, k, A_old, old_chi, 0.0, A_new, k)

// Direction 'L': A[site] = Vh, absorb U*S into previous site
// 1. Scale columns of U by S: US[i,j] = U[i,j] * S[j]
//    Use rocblas_ddgmm(side_right, m, k, U, m, S, 1, US, m)
// 2. Multiply into neighbor: A[site-1] = A_old[site-1] @ US
//    Use rocblas_dgemm(N, N, rows_prev, k, old_chi, 1.0, A_old, rows_prev, US, old_chi, 0.0, A_new, rows_prev)
```

### Phase 4: GPU Tridiagonal Eigensolver

Replace CPU LAPACK `dstev_` with `rocsolver_dsyev`:

```cpp
// Build full symmetric tridiagonal matrix T on GPU
// T[i,i] = alpha[i], T[i,i+1] = T[i+1,i] = beta[i]
// Solve with rocsolver_dsyev
rocsolver_dsyev(rocblas_h_, rocblas_evect_original, rocblas_fill_upper,
                niter, d_T, niter, d_eigenvalues, d_E, d_info);
```

The alpha/beta values from Lanczos are single scalars returned by `rocblas_ddot` / `rocblas_dnrm2`
— these host-side scalars are fine. Build the tridiagonal matrix on GPU from them using a
small HIP kernel or `hipMemcpy` of individual elements.

## Memory Layout Convention

All tensors are column-major, matching rocBLAS and hipTensor:

```
MPS:   A[a, s, b]       stored as  a + s*chi_L + b*chi_L*d
MPO:   W[w, s, s', w']  stored as  w + s*D + s'*D*d + w'*D*d*d
L_env: L[a, w, a']      stored as  a + w*chi_L + a'*chi_L*D
R_env: R[b, w', b']     stored as  b + w'*chi_R + b'*chi_R*D
```

## hipTensor API Reference (verified on this machine, ROCm 7.2)

```cpp
#include <hiptensor/hiptensor.h>
#include <hiptensor/hiptensor_types.h>

// Handle (create once for entire program)
hiptensorHandle_t handle;
hiptensorCreate(&handle);

// Tensor descriptors (strides=NULL means column-major default)
hiptensorTensorDescriptor_t descA;
int64_t extA[] = {chi_L, D_mpo, chi_L};
hiptensorCreateTensorDescriptor(handle, &descA,
    3,          // numModes
    extA,       // lens
    NULL,       // strides (NULL = column-major)
    HIP_R_64F,  // dataType
    0);         // alignmentRequirement (0 = auto)

// Contraction descriptor
hiptensorOperationDescriptor_t opDesc;
int32_t modesA[] = {'a', 'w', 'p'};
int32_t modesB[] = {'a', 's', 'b'};
int32_t modesC[] = {'w', 'p', 's', 'b'};
hiptensorCreateContraction(handle, &opDesc,
    descA, modesA, HIPTENSOR_OP_IDENTITY,  // A
    descB, modesB, HIPTENSOR_OP_IDENTITY,  // B
    descC, modesC, HIPTENSOR_OP_IDENTITY,  // C (output for beta)
    descC, modesC,                         // D (output)
    HIPTENSOR_COMPUTE_DESC_64F);           // compute type

// Plan preference
hiptensorPlanPreference_t pref;
hiptensorCreatePlanPreference(handle, &pref, HIPTENSOR_ALGO_DEFAULT);

// Plan (encapsulates workspace requirements)
hiptensorPlan_t plan;
hiptensorCreatePlan(handle, &plan, opDesc, pref, 0);  // 0 = no workspace limit

// Query workspace size
uint64_t ws_size = 0;
hiptensorPlanGetAttribute(handle, plan,
    HIPTENSOR_PLAN_REQUIRED_WORKSPACE, &ws_size, sizeof(ws_size));
void* d_workspace = nullptr;
if (ws_size > 0) hipMalloc(&d_workspace, ws_size);

// Execute contraction (reuse plan for every call with same dimensions)
double alpha = 1.0, beta = 0.0;
hiptensorContract(handle, plan,
    &alpha, d_A, d_B,
    &beta,  d_C, d_D,
    d_workspace, ws_size, stream);

// Cleanup
hiptensorDestroyPlan(plan);
hiptensorDestroyPlanPreference(pref);
hiptensorDestroyOperationDescriptor(opDesc);
hiptensorDestroyTensorDescriptor(descA);
hiptensorDestroy(handle);
```

**IMPORTANT:** The above API is verified against `/opt/rocm/include/hiptensor/hiptensor.h`
on the remote machine. Key differences from some online docs:
- `hiptensorCreateContraction` takes `hiptensorComputeDescriptor_t` (use `HIPTENSOR_COMPUTE_DESC_64F`)
- `hiptensorCreatePlan` takes `(handle, plan*, opDesc, pref, workspaceSizeLimit)`
- Strides `NULL` = column-major (natural for our layout)

## Build Configuration

Update `dmrg-gpu/CMakeLists.txt` to add hipTensor:

```cmake
# After find_package(rocsolver REQUIRED), add:
find_library(HIPTENSOR_LIBRARY NAMES hiptensor
    PATHS ${ROCM_PATH}/lib ${ROCM_PATH}/lib64)
find_path(HIPTENSOR_INCLUDE_DIR hiptensor/hiptensor.h
    PATHS ${ROCM_PATH}/include)
if(NOT HIPTENSOR_LIBRARY)
    message(FATAL_ERROR "hipTensor not found")
endif()
message(STATUS "hipTensor: ${HIPTENSOR_LIBRARY}")
include_directories(${HIPTENSOR_INCLUDE_DIR})

# Add to target_link_libraries:
target_link_libraries(dmrg_gpu
    hip::device
    roc::rocblas
    roc::rocsolver
    ${HIPTENSOR_LIBRARY}
    ${LAPACK_LIBRARIES}
)
```

## Verification

After each phase, rebuild and verify:

```bash
cd /home/hotaisle/dmrg-implementations/dmrg-gpu/build
rm -rf * && cmake .. -DGPU_TARGETS=gfx942 && make -j8

# Accuracy (must match CPU results)
./dmrg_gpu 4 16 10    # L=4: expect -1.616025403784, error < 1e-10
./dmrg_gpu 8 32 10    # L=8: expect -3.374932598688, error < 1e-10

# Performance (must beat 4.7s CPU baseline)
./dmrg_gpu 32 64 5    # Current: 120s. Target: <4.7s. Aspirational: <1s.
```

## Rules

- Do NOT use `hipMemcpy(..., hipMemcpyDeviceToHost)` for tensor data during the algorithm.
  Only acceptable D→H transfers: single scalars (energy, norm, convergence check).
- Do NOT fall back to CPU for any contraction or matrix multiply.
- Do NOT reference or copy code from `pdmrg-gpu/`. Build fresh from the hipTensor API.
- Do NOT change the algorithm. Single-site DMRG with Lanczos + SVD. Just move computation to GPU.
- Do NOT add unnecessary synchronization. Use async ops and streams where possible.
- **Preserve correctness.** If accuracy degrades below 1e-10, the change is wrong.

## Files to Modify

1. `src/dmrg_gpu.h` — Add GPU contraction members, remove CPU-only fields
2. `src/dmrg_gpu.cpp` — Replace all CPU contraction functions with GPU calls
3. `CMakeLists.txt` — Add hipTensor dependency

## Files to Create

1. `src/heff_gpu_1site.h` — `HeffGPU1Site` class for GPU H_eff application
2. `src/heff_gpu_1site.cpp` — hipTensor-based implementation (3-step contraction)
3. `src/env_update_gpu.h` — GPU environment update class
4. `src/env_update_gpu.cpp` — hipTensor-based environment updates (3-step each)

Or put everything inline in `dmrg_gpu.cpp` if simpler — correctness and performance matter
more than code organization.
