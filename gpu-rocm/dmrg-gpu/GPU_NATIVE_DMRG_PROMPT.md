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

## Remote Machine — ALL Work Happens Here

**All compilation, editing, and testing MUST happen on the remote GPU machine.**
The local machine has no GPU and cannot compile HIP code.

See `CLAUDE.md` in the repository root for full remote access instructions. Quick reference:

- **SSH:** `ssh hotaisle@23.183.40.79` (passwordless, no password prompt)
- **tmux session:** `tmux attach -t test_remote` (persistent session, always running)
- **Send commands to tmux:** `tmux send-keys -t test_remote 'command here' Enter`
- **Capture tmux output:** `tmux capture-pane -t test_remote -p | tail -20`
- **Remote repo path:** `/home/hotaisle/dmrg-implementations/`
- **GPU code path:** `/home/hotaisle/dmrg-implementations/dmrg-gpu/`

### Remote Machine Details

- **GPU:** AMD Instinct MI300X (gfx942)
- **ROCm:** 7.2 at `/opt/rocm`
- **hipTensor header:** `/opt/rocm/include/hiptensor/hiptensor.h` (confirmed present)
- **hipTensor library:** `/opt/rocm/lib/libhiptensor.so` (confirmed present)
- **GitHub token:** Configured on remote for commits/pushes
- **Git user:** Configured on remote

### Workflow

1. SSH into the remote machine (or use tmux session)
2. Edit files on the remote machine
3. Build on the remote machine: `cd /home/hotaisle/dmrg-implementations/dmrg-gpu/build && cmake .. -DGPU_TARGETS=gfx942 && make -j8`
4. Test on the remote machine: `./dmrg_gpu 4 16 10`
5. Commit and push from the remote machine (or pull locally and push)

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

### SVD: Use plain rocSOLVER, NOT AccurateSVD

The `AccurateSVD_GPU` class (recursive refinement SVD) exists in the codebase and is verified
working. **However, it is NOT needed for standard single-site DMRG.** AccurateSVD is specifically
needed for PDMRG boundary exchange where singular value inversion (`V = Λ⁻¹`) amplifies errors.

For standard DMRG, use plain `rocsolver_dgesvd` directly:

```cpp
// Standard SVD for DMRG truncation — no recursive refinement needed
rocsolver_dgesvd(rocblas_h_,
    rocblas_svect_all,   // left singular vectors (U)
    rocblas_svect_all,   // right singular vectors (Vh)
    m, n,
    d_A, m,              // input matrix (overwritten)
    d_S,                 // singular values
    d_U, m,              // left singular vectors
    d_Vh, n,             // right singular vectors
    d_E,                 // superdiagonal workspace
    d_info);             // error info
```

Keep `accurate_svd_gpu.h/.cpp` in the codebase for future PDMRG use, but the DMRG pipeline
should call `rocsolver_dgesvd` directly. This is simpler and avoids unnecessary overhead.

## Implementation Plan

All new code goes in `dmrg-gpu/src/`. Do NOT reference or copy code from `pdmrg-gpu/` — that
code has unresolved correctness issues. Build fresh from the hipTensor and rocBLAS APIs.

### Phase 1: GPU `apply_heff()` — THE CRITICAL FIX (>95% of speedup)

**Replace the CPU for-loop `apply_heff()` with hipTensor contractions.**

The single-site H_eff contraction is:

```
result[a', s', b'] = Σ_{a,s,b,w,w'} L[a, w, a'] * W[w, s, s', w'] * R[b, w', b'] * theta[a, s, b]
```

Split into 3 hipTensor steps:

```
Step 1: T1[w, a', s, b]    = Σ_a     L[a, w, a']       * theta[a, s, b]
Step 2: T2[a', s', w', b]  = Σ_{w,s} W[w, s, s', w']   * T1[w, a', s, b]
Step 3: result[a', s', b'] = Σ_{w',b} R[b, w', b']     * T2[a', s', w', b]
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

**Contraction order note:** Our analysis showed the optimal contraction order for 2-site
DMRG starts from the right (theta×R first), giving 2.3-3.8× fewer FLOPs than L-first.
For 1-site with 3 steps, the order matters less, but benchmark both L-first and R-first
if performance is unexpectedly slow.

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

### Phase 3: GPU SVD and Absorption

Use plain `rocsolver_dgesvd` directly (not AccurateSVD). After SVD, absorb singular values
into the neighbor site entirely on GPU using `rocblas_dgemm`:

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

**CRITICAL:** The SVD absorption step (multiplying S into neighbor) is **not optional**.
Without it, the wavefunction norm is silently destroyed at each site optimization,
causing the energy to diverge. This was a major bug we already found and fixed.

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

## CRITICAL: Lessons Learned from Previous Bugs

We spent days debugging these bugs. Every one of them silently produced wrong answers
without crashing. **Read this section carefully before writing any code.**

### Bug 1: Boundary Conditions Must Match the MPO

There is **no shape or structure requirement on the MPO** — it can be upper-triangular,
lower-triangular, dense, or anything else. The only requirement is that the boundary
environments are consistent with the MPO.

The boundary environments select which MPO bond indices "start" and "end" the Hamiltonian:
- Left boundary:  `L[0, w_start, 0] = 1.0`
- Right boundary: `R[0, w_end,   0] = 1.0`

The MPO must have a non-zero path from `w_start` to `w_end` across the chain.

**The bug we hit:** The MPO used one convention but the boundaries used another. There
was no valid path through the MPO, causing R environments to be identically zero and
energy = 0.0 for L > 2. The fix was making the boundaries match the MPO.

**Current GPU code** uses `L[0,0,0]=1` and `R[0,D-1,0]=1`. The Heisenberg MPO is
written as upper-triangular to match (signal flows w=0 → w=D-1). If you change one,
change the other.

### Bug 2: Physical Index Convention — `<sp|O|s>` Not `<s|O|sp>`

The MPO operator elements must use:
```cpp
int idx = sp * 2 + s;   // CORRECT: gives <sp|O|s>
// NOT:
int idx = s * 2 + sp;   // WRONG: gives <s|O|sp> = O^T
```

Getting this backwards computes `O^T * theta` instead of `O * theta`. The energy may
still converge to something, but it will be wrong. For Heisenberg (symmetric H) the
error is subtle — it shows up as slightly wrong energies at larger system sizes.

### Bug 3: SVD Singular Value Absorption Is Mandatory

After SVD decomposition `theta = U * S * Vh`, you MUST absorb S into a neighbor:
- Right sweep: `A[site] = U`, then `A[site+1] = (S * Vh) @ A[site+1]`
- Left sweep: `A[site] = Vh`, then `A[site-1] = A[site-1] @ (U * S)`

**If you skip this step**, the wavefunction norm is silently destroyed at every site
optimization. The energy will appear to converge but to the wrong value, or oscillate.
This was one of our hardest bugs to find because there's no crash or obvious error.

### Bug 4: Bond Dimension Tracking — Use a Single Array

Tracking bond dimensions with separate `chi_left[]` and `chi_right[]` arrays causes
consistency bugs when SVD changes a bond dimension — you must update both arrays and
keep them synchronized across adjacent sites.

**Solution:** Use a single `bond_dims_[]` array where `bond_dims_[i]` is the dimension
of the bond between site `i-1` and site `i`. Then:
- `chi_L(site) = bond_dims_[site]`
- `chi_R(site) = bond_dims_[site + 1]`

This makes it impossible for adjacent sites to disagree about a shared bond dimension.

### Bug 5: Environment Buffer Allocation

Environments change size as bond dimensions grow during SVD truncation. If you allocate
environment buffers based on initial bond dimensions, later SVD truncations can produce
tensors that don't fit.

**Solution:** Allocate all interior environment buffers at `chi_max` size. This wastes
some memory but avoids reallocation bugs. Only boundary environments (site 0 and L) are
size 1.

### Bug 6: Lanczos Needs Full Reorthogonalization

Standard Lanczos (3-term recurrence only) suffers from loss of orthogonality in finite
precision. This creates "ghost eigenvalues" — spurious copies of already-converged
eigenvalues that corrupt the ground state.

**Solution:** After each Lanczos step, reorthogonalize the new vector against ALL previous
Lanczos vectors using Gram-Schmidt:

```cpp
for (int j = 0; j <= iter; j++) {
    double overlap;
    rocblas_ddot(handle, n, d_vj, 1, d_w, 1, &overlap);
    double neg = -overlap;
    rocblas_daxpy(handle, n, &neg, d_vj, 1, d_w, 1);
}
```

This costs O(k*n) extra per iteration but prevents ghost eigenvalues that cause
convergence to wrong energies.

### Bug 7: Contraction Order Affects FLOP Count by 2-4×

For the 2-site H_eff (5-tensor contraction), the contraction order matters enormously:
- **L-first order** (L×theta, then W_L, then W_R, then R): 2.3-3.8× more FLOPs
- **R-first order** (theta×R, then W_R, then W_L, then L): optimal

For the 1-site H_eff (4-tensor, 3-step contraction), the difference is smaller but still
present. When benchmarking, if performance is surprisingly slow, check whether a different
contraction order reduces FLOP count.

### Bug 8: hipTensor Mode Labels Must Be Consistent

In hipTensor contractions, mode labels identify which indices to contract. If you use
inconsistent labels between tensors, hipTensor will silently compute the wrong contraction
or produce transposed results.

**Example of correct mode labeling:**
```cpp
// Step 1: T1[w, a', s, b] = L[a, w, a'] * theta[a, s, b]
// The 'a' mode appears in both L and theta → it gets contracted
int32_t modesL[]     = {'a', 'w', 'p'};     // L[a, w, a']
int32_t modesTheta[] = {'a', 's', 'b'};     // theta[a, s, b]
int32_t modesT1[]    = {'w', 'p', 's', 'b'}; // T1[w, a', s, b]  — 'a' is gone
```

**Rules:**
- Modes that appear in BOTH input tensors get contracted (summed over)
- Modes that appear in only one input must appear in the output
- Every output mode must come from exactly one of the two inputs
- Mode labels are just integers — using chars like 'a', 'w' is just for readability

### Bug 9: Verify Contraction Results Before Scaling Up

Before deploying a hipTensor contraction in the hot path, verify it against a CPU reference
for a small test case (e.g., chi=2, d=2, D=3). Compare element-by-element. This catches:
- Transposed indices
- Wrong contraction (mode labels wrong)
- Memory layout mismatches
- Off-by-one in dimensions

### Bug 10: L=8 Exact Energy Reference

The correct exact energy for the L=8 Heisenberg chain is:
```
E_exact(L=8) = -3.374932598688   (NOT -3.374931816815 which appears in some references)
```

Verified by exact diagonalization. Using the wrong reference energy will make you chase
phantom accuracy problems.

### General Principle

**Every bug we found was a silent correctness bug** — the code compiled, ran without
crashes, and produced numbers that looked plausible but were wrong. Always:
1. Test against known exact results (L=4, L=8 Heisenberg)
2. Compare CPU and GPU results element-by-element for small test cases
3. Check that energy is variational (E_DMRG ≥ E_exact)
4. Verify convergence behavior (energy should decrease monotonically within a sweep)

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
    extA,       // lens (dimensions)
    NULL,       // strides (NULL = column-major)
    HIP_R_64F,  // dataType (real double)
    0);         // alignmentRequirement (0 = auto)

// Contraction descriptor
hiptensorOperationDescriptor_t opDesc;
int32_t modesA[] = {'a', 'w', 'p'};
int32_t modesB[] = {'a', 's', 'b'};
int32_t modesC[] = {'w', 'p', 's', 'b'};
hiptensorCreateContraction(handle, &opDesc,
    descA, modesA, HIPTENSOR_OP_IDENTITY,          // A
    descB, modesB, HIPTENSOR_OP_IDENTITY,          // B
    descC, modesC, HIPTENSOR_OP_IDENTITY,          // C (for beta*C term)
    descC, modesC,                                 // D (output)
    HIPTENSOR_COMPUTE_DESC_64F);                   // compute type: FP64

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
    &beta,  d_C, d_D,       // C=D for in-place, or separate
    d_workspace, ws_size,
    stream);                 // hipStream_t, 0 for default

// Cleanup
hiptensorDestroyPlan(plan);
hiptensorDestroyPlanPreference(pref);
hiptensorDestroyOperationDescriptor(opDesc);
hiptensorDestroyTensorDescriptor(descA);
hiptensorDestroy(handle);
```

**IMPORTANT:** The above API is verified against `/opt/rocm/include/hiptensor/hiptensor.h`
on the remote machine. Key details:
- `hiptensorCreateContraction` last arg is `hiptensorComputeDescriptor_t` — use `HIPTENSOR_COMPUTE_DESC_64F`
- `hiptensorCreatePlan` takes `(handle, plan*, opDesc, pref, workspaceSizeLimit)`
- Strides `NULL` = column-major (matches our memory layout)
- `hiptensorContract` signature: `(handle, plan, alpha, A, B, beta, C, D, workspace, ws_size, stream)`

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

# Accuracy (must match CPU results exactly)
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
- Do NOT use AccurateSVD for the DMRG pipeline. Use plain `rocsolver_dgesvd` directly.
- Do NOT add unnecessary synchronization. Use async ops and streams where possible.
- **Preserve correctness.** If accuracy degrades below 1e-10, the change is wrong.
- **Verify each contraction step** against CPU output for a small test case before trusting it.

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

## Exact Reference Energies (Heisenberg Chain)

```
L=4:  E = -1.616025403784
L=8:  E = -3.374932598688   ← Note: NOT -3.374931816815
L=16: E = -7.142296361       (requires chi > 64 to converge)
L=32: E ≈ -14.0              (chi=64 gives -13.9973, not converged)
```
