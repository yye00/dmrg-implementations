# GPU DMRG Optimization Design Document

**Date:** 2026-03-05
**Target:** Optimize PDMRG_GPU and PDMRG2_GPU implementations
**Hardware:** AMD MI300X GPU with ROCm/hipTensor

## Executive Summary

This document outlines the design for optimizing GPU-based parallel DMRG implementations (PDMRG_GPU and PDMRG2_GPU) to achieve correctness and performance parity with CPU implementations while exploiting GPU parallelism.

**Primary Objectives:**
1. Implement true stream parallelization (replacing MPI workers)
2. Preserve exact SVD boundary reconciliation pattern from CPU
3. Move all H_eff operations fully to GPU
4. Optimize hipTensor contraction performance
5. Achieve linear scaling with stream counts
6. Validate correctness against Quimb (< 1e-10 tolerance)

---

## 1. Architecture Overview

### 1.1 Stream-Based Parallelization

**Design Pattern: MPI Workers → GPU Streams**

```
CPU (MPI):                      GPU (Streams):
┌──────────────────┐           ┌──────────────────┐
│ Worker 0 (MPI)   │           │ Stream 0         │
│   Segment 0      │           │   Segment 0      │
├──────────────────┤           ├──────────────────┤
│ Worker 1 (MPI)   │    →      │ Stream 1         │
│   Segment 1      │           │   Segment 1      │
├──────────────────┤           ├──────────────────┤
│ Worker 2 (MPI)   │           │ Stream 2         │
│   Segment 2      │           │   Segment 2      │
└──────────────────┘           └──────────────────┘
     ↓ MPI Comm                     ↓ Event Sync
  Boundary Merge               Boundary Merge
```

**Implementation:**
- Create `n_streams` HIP streams at initialization
- Each stream owns one MPS segment (contiguous site range)
- Segments evolve independently using DMRG sweeps
- Boundary synchronization via HIP events + accurate SVD merge

### 1.2 Segment Management

```cpp
struct MPSSegment {
    int start_site;           // First site in segment
    int end_site;             // Last site in segment
    hipStream_t stream;       // Dedicated GPU stream

    // Device memory for this segment
    double* d_tensors;        // MPS tensors [start_site:end_site]
    double* d_L_envs;         // Left environments
    double* d_R_envs;         // Right environments

    // Boundary bridge matrices
    double* d_V_left;         // Bridge to left neighbor (Lambda^-1)
    double* d_V_right;        // Bridge to right neighbor (Lambda^-1)

    int left_boundary_site;   // Site where left merge occurs
    int right_boundary_site;  // Site where right merge occurs
};

struct ParallelDMRG_GPU {
    int n_streams;
    std::vector<MPSSegment> segments;
    std::vector<hipEvent_t> boundary_events;

    // Global data (shared across streams)
    MPO* d_mpo;               // Hamiltonian MPO
    HipTensorContractor* ht;  // Optimized tensor contractor
};
```

---

## 2. Exact SVD Boundary Reconciliation

### 2.1 CPU Algorithm (Reference)

From `pdmrg/parallel/merge.py`:

```python
def merge_boundary_tensors(psi_left, psi_right, V, L_env, R_env, mpo, ...):
    """
    Reconciles independently-evolved segments at boundary.

    Args:
        psi_left: Left segment boundary tensor [chi_L, d_L, k_old]
        psi_right: Right segment boundary tensor [k_old, d_R, chi_R]
        V: Bridge matrix (Lambda^-1) [k_old]
        L_env, R_env: DMRG environments
        mpo: Two-site MPO at boundary

    Returns:
        A_left, A_right: New boundary tensors
        V_new: Updated bridge matrix
        energy: Ground state energy at boundary
        trunc_err: Truncation error
    """
    # Step 1: Form two-site wavefunction with V bridge
    # Θ = ψ_left · diag(V) · ψ_right
    theta = np.einsum('ija,akl->ijkl', psi_left, V[:, None, None] * psi_right)

    # Step 2: Optimize with Lanczos eigensolver
    # Θ_opt = argmin <Θ| H_eff |Θ>
    theta_opt, energy = optimize_two_site(theta, H_eff)

    # Step 3: Accurate SVD with recursive refinement
    # Θ_opt ≈ U · Σ · V†  (with improved small singular values)
    U, S, Vh = accurate_svd(theta_opt.reshape(chi_L * d_L, d_R * chi_R))

    # Step 4: Compute new bridge matrix
    # V_new = 1 / Σ  (with clipping to prevent overflow)
    V_new = 1.0 / np.clip(S[:k], 1e-12, None)

    # Step 5: Extract new boundary tensors
    A_left = U[:, :k].reshape(chi_L, d_L, k)
    A_right = (np.diag(S[:k]) @ Vh[:k, :]).reshape(k, d_R, chi_R)

    return A_left, A_right, V_new, energy, trunc_err
```

**Key Properties:**
- V = Λ^-1 acts as "bridge" across segment boundary
- Stoudenmire & White prescription: merged wavefunction = ψ_L · diag(V) · ψ_R
- Accurate SVD ensures small singular values are refined (prevents V blowup)
- New V_new = 1/S from optimized boundary

### 2.2 GPU Implementation Design

**File:** `gpu-port/src/boundary_merge_gpu.cpp`

```cpp
struct BoundaryMergeResult {
    double energy;
    double trunc_err;
    int new_bond_dim;
};

class BoundaryMerger {
private:
    HipTensorContractor* ht_contractor;
    LanczosSolver* lanczos;
    AccurateSVD_GPU* accurate_svd;

public:
    BoundaryMergeResult merge_boundary(
        MPSSegment* left_seg,
        MPSSegment* right_seg,
        const MPO* mpo,
        int boundary_site,
        int max_bond_dim,
        double cutoff
    ) {
        // Step 1: Form two-site theta with V bridge
        // theta[i,j,k,l] = psi_left[i,j,a] * V[a] * psi_right[a,k,l]
        hipTensorContraction theta_contraction = build_theta_with_bridge(
            left_seg->d_tensors + offset_left,
            right_seg->d_V_left,  // V = Lambda^-1
            right_seg->d_tensors + offset_right
        );

        double* d_theta;
        HIP_CHECK(hipMalloc(&d_theta, chi_L * d_L * d_R * chi_R * sizeof(double)));
        ht_contractor->contract(theta_contraction, d_theta);

        // Step 2: Lanczos optimization on GPU
        // E_0, |theta_opt> = min <theta| H_eff |theta>
        double* d_theta_opt;
        double energy = lanczos->solve(
            d_theta,
            [&](double* in, double* out) {
                apply_H_eff_gpu(in, out, left_seg, right_seg, mpo, boundary_site);
            },
            &d_theta_opt
        );

        // Step 3: Accurate SVD on GPU
        // theta_opt = U * S * Vh  (with recursive refinement)
        AccurateSVDResult svd_result = accurate_svd->decompose(
            d_theta_opt,
            chi_L * d_L,
            d_R * chi_R,
            1e-4  // epsilon threshold
        );

        // Step 4: Truncate to max_bond_dim
        int k = compute_truncation_dim(svd_result.S, max_bond_dim, cutoff);
        double trunc_err = compute_truncation_error(svd_result.S, k);

        // Step 5: Compute new V = 1/S
        double* d_V_new;
        HIP_CHECK(hipMalloc(&d_V_new, k * sizeof(double)));
        launch_invert_with_clipping<<<...>>>(svd_result.d_S, d_V_new, k, 1e-12);

        // Step 6: Extract new boundary tensors
        // A_left = U[:, :k].reshape(chi_L, d_L, k)
        // A_right = (diag(S[:k]) @ Vh[:k, :]).reshape(k, d_R, chi_R)
        extract_boundary_tensors(
            svd_result.d_U, svd_result.d_S, svd_result.d_Vh,
            left_seg->d_tensors + offset_left,
            right_seg->d_tensors + offset_right,
            k
        );

        // Step 7: Update bridge matrices
        HIP_CHECK(hipMemcpy(right_seg->d_V_left, d_V_new, k * sizeof(double), ...));
        HIP_CHECK(hipMemcpy(left_seg->d_V_right, d_V_new, k * sizeof(double), ...));

        return {energy, trunc_err, k};
    }
};
```

---

## 3. Accurate SVD on GPU

### 3.1 CPU Algorithm (Reference)

From `pdmrg/numerics/accurate_svd.py`:

```python
def accurate_svd(M, epsilon=1e-4):
    """
    SVD with recursive refinement for small singular values.

    Standard SVD loses accuracy for singular values < epsilon * sigma_max.
    This function recursively refines those values by re-orthogonalizing
    the inaccurate subspace.
    """
    U, S, Vh = np.linalg.svd(M, full_matrices=False)

    # Find accuracy degradation threshold
    p = None
    for i in range(len(S)):
        if S[i] / S[0] < epsilon:
            p = i
            break

    # Recursively refine inaccurate subspace
    if p is not None:
        # Project M onto inaccurate subspace
        X = U[:, p:].conj().T @ M @ Vh[p:, :].conj().T

        # Recursively compute accurate SVD of subspace
        U_sub, S_sub, Vh_sub = accurate_svd(X, epsilon)

        # Update original decomposition
        U[:, p:] = U[:, p:] @ U_sub
        Vh[p:, :] = Vh_sub @ Vh[p:, :]
        S[p:] = S_sub

    return U, S, Vh
```

### 3.2 GPU Implementation Design

**File:** `gpu-port/src/accurate_svd_gpu.cpp`

```cpp
struct AccurateSVDResult {
    double* d_U;     // Left singular vectors [m, k]
    double* d_S;     // Singular values [k]
    double* d_Vh;    // Right singular vectors [k, n]
    int rank;
};

class AccurateSVD_GPU {
private:
    rocblas_handle rocblas_h;
    rocsolver_handle rocsolver_h;
    double epsilon;
    int max_recursion_depth;

    void find_degradation_threshold(double* d_S, int n, int* h_p) {
        // Find index p where S[p]/S[0] < epsilon
        std::vector<double> h_S(n);
        HIP_CHECK(hipMemcpy(h_S.data(), d_S, n * sizeof(double), hipMemcpyDeviceToHost));

        double sigma_max = h_S[0];
        *h_p = -1;
        for (int i = 0; i < n; i++) {
            if (h_S[i] / sigma_max < epsilon) {
                *h_p = i;
                break;
            }
        }
    }

public:
    AccurateSVD_GPU(double eps = 1e-4, int max_depth = 5)
        : epsilon(eps), max_recursion_depth(max_depth) {
        rocblas_create_handle(&rocblas_h);
        rocsolver_create_handle(&rocsolver_h);
    }

    AccurateSVDResult decompose(double* d_M, int m, int n, int depth = 0) {
        // Base case: standard rocsolver SVD
        int k = std::min(m, n);
        double *d_U, *d_S, *d_Vh;
        HIP_CHECK(hipMalloc(&d_U, m * k * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_S, k * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_Vh, k * n * sizeof(double)));

        // Initial SVD
        rocsolver_zgesvd(
            rocsolver_h,
            rocblas_svect_singular,  // Compute U
            rocblas_svect_singular,  // Compute Vh
            m, n,
            d_M, m,
            d_S,
            d_U, m,
            d_Vh, k,
            nullptr, 1,  // No V (we use Vh)
            rocblas_workmode_device,
            nullptr  // Info
        );

        // Check if refinement needed
        if (depth >= max_recursion_depth) {
            return {d_U, d_S, d_Vh, k};
        }

        int p;
        find_degradation_threshold(d_S, k, &p);

        if (p == -1) {
            // No refinement needed
            return {d_U, d_S, d_Vh, k};
        }

        // Recursive refinement
        // X = U[:, p:]^H @ M @ Vh[p:, :]^H
        int m_sub = m;
        int n_sub = n;
        int k_sub = k - p;

        double* d_X;
        HIP_CHECK(hipMalloc(&d_X, k_sub * k_sub * sizeof(double)));

        // X = U_sub^H @ M
        rocblas_zgemm(
            rocblas_h,
            rocblas_operation_conjugate_transpose,
            rocblas_operation_none,
            k_sub, n, m,
            &alpha,
            d_U + p * m, m,  // U[:, p:]
            d_M, m,
            &beta,
            d_X, k_sub
        );

        // X = X @ Vh_sub^H
        double* d_X_temp;
        HIP_CHECK(hipMalloc(&d_X_temp, k_sub * k_sub * sizeof(double)));
        rocblas_zgemm(
            rocblas_h,
            rocblas_operation_none,
            rocblas_operation_conjugate_transpose,
            k_sub, k_sub, n,
            &alpha,
            d_X, k_sub,
            d_Vh + p * n, k,  // Vh[p:, :]
            &beta,
            d_X_temp, k_sub
        );

        // Recursively compute accurate SVD of X
        AccurateSVDResult sub_result = decompose(d_X_temp, k_sub, k_sub, depth + 1);

        // Update U[:, p:] = U[:, p:] @ U_sub
        double* d_U_new;
        HIP_CHECK(hipMalloc(&d_U_new, m * k_sub * sizeof(double)));
        rocblas_zgemm(
            rocblas_h,
            rocblas_operation_none,
            rocblas_operation_none,
            m, k_sub, k_sub,
            &alpha,
            d_U + p * m, m,
            sub_result.d_U, k_sub,
            &beta,
            d_U_new, m
        );
        HIP_CHECK(hipMemcpy(d_U + p * m, d_U_new, m * k_sub * sizeof(double), ...));

        // Update Vh[p:, :] = Vh_sub @ Vh[p:, :]
        double* d_Vh_new;
        HIP_CHECK(hipMalloc(&d_Vh_new, k_sub * n * sizeof(double)));
        rocblas_zgemm(
            rocblas_h,
            rocblas_operation_none,
            rocblas_operation_none,
            k_sub, n, k_sub,
            &alpha,
            sub_result.d_Vh, k_sub,
            d_Vh + p * n, k,
            &beta,
            d_Vh_new, k_sub
        );
        HIP_CHECK(hipMemcpy(d_Vh + p * n, d_Vh_new, k_sub * n * sizeof(double), ...));

        // Update S[p:]
        HIP_CHECK(hipMemcpy(d_S + p, sub_result.d_S, k_sub * sizeof(double), ...));

        // Cleanup
        HIP_CHECK(hipFree(d_X));
        HIP_CHECK(hipFree(d_X_temp));
        HIP_CHECK(hipFree(d_U_new));
        HIP_CHECK(hipFree(d_Vh_new));
        HIP_CHECK(hipFree(sub_result.d_U));
        HIP_CHECK(hipFree(sub_result.d_S));
        HIP_CHECK(hipFree(sub_result.d_Vh));

        return {d_U, d_S, d_Vh, k};
    }
};
```

---

## 4. H_eff Optimization

### 4.1 Current Bottleneck Analysis

**PDMRG_GPU (CPU-based H_eff):**
- Downloads L, R, W1, W2, theta to CPU
- Performs contraction with nested loops
- Uploads result back to GPU
- ~1980 transfers per sweep (~99 sites × 2 directions × 10 sweeps)

**PDMRG2_GPU (Hybrid H_eff):**
- Uses hipTensor for 4-step contraction
- But uses HIPTENSOR_ALGO_DEFAULT (no optimization)
- No workspace caching
- No contraction path selection

**Performance Impact:**
- H_eff called 15,000-30,000 times per DMRG run
- Each application: ~480M FLOPs
- Total: 7-14 TFLOPs computational cost
- Current performance: 50-70% of MI300X peak
- Target: 70-90% of peak

### 4.2 Optimized H_eff Implementation

**Strategy:**
1. Fully GPU-resident (no CPU transfers)
2. Workspace caching (allocate once, reuse)
3. Optimal contraction path selection
4. Memory layout optimization (column-major for rocBLAS/hipTensor)
5. MPO product precomputation where beneficial

**File:** `gpu-port/src/heff_optimized_gpu.cpp`

```cpp
class OptimizedHeff {
private:
    // Workspace management
    struct WorkspaceCache {
        size_t workspace_size;
        void* d_workspace;
        bool is_allocated;
    };

    std::map<std::string, WorkspaceCache> workspace_cache;

    // hipTensor handles
    hiptensorHandle_t handle;

    // Contraction descriptors (reusable)
    hiptensorTensorDescriptor_t desc_L;
    hiptensorTensorDescriptor_t desc_R;
    hiptensorTensorDescriptor_t desc_W1;
    hiptensorTensorDescriptor_t desc_W2;
    hiptensorTensorDescriptor_t desc_theta;
    hiptensorTensorDescriptor_t desc_T1;
    hiptensorTensorDescriptor_t desc_T2;
    hiptensorTensorDescriptor_t desc_T3;
    hiptensorTensorDescriptor_t desc_result;

    // Contraction plans (pre-optimized)
    hiptensorPlan_t plan_L_theta;
    hiptensorPlan_t plan_W1_T1;
    hiptensorPlan_t plan_W2_T2;
    hiptensorPlan_t plan_T3_R;

    void initialize_descriptors(int chi_L, int chi_R, int d, int D_mpo) {
        // L[w, ap, a]: Environment from left
        std::vector<int64_t> extent_L = {D_mpo, chi_L, chi_L};
        std::vector<int> mode_L = {'w', 'p', 'a'};
        hiptensorCreateTensorDescriptor(
            handle, &desc_L,
            extent_L.size(), extent_L.data(),
            nullptr,  // Strides (null = column-major)
            HIPTENSOR_R_64F, HIPTENSOR_OP_IDENTITY
        );

        // theta[a, s1, s2, b]: Two-site wavefunction
        std::vector<int64_t> extent_theta = {chi_L, d, d, chi_R};
        std::vector<int> mode_theta = {'a', 's', 't', 'b'};
        hiptensorCreateTensorDescriptor(
            handle, &desc_theta,
            extent_theta.size(), extent_theta.data(),
            nullptr, HIPTENSOR_R_64F, HIPTENSOR_OP_IDENTITY
        );

        // W1[w, s, sp, x], W2[x, t, tp, y]
        std::vector<int64_t> extent_W1 = {D_mpo, d, d, D_mpo};
        std::vector<int> mode_W1 = {'w', 's', 'p', 'x'};
        hiptensorCreateTensorDescriptor(
            handle, &desc_W1,
            extent_W1.size(), extent_W1.data(),
            nullptr, HIPTENSOR_R_64F, HIPTENSOR_OP_IDENTITY
        );

        // Similar for W2, T1, T2, T3, R, result...
    }

    void create_optimized_plans() {
        // Plan 1: T1[w, p, s, t, b] = L[w, p, a] * theta[a, s, t, b]
        hiptensorContractionDescriptor_t desc_1;
        hiptensorCreateContraction(
            handle, &desc_1,
            desc_L, {'w', 'p', 'a'},
            desc_theta, {'a', 's', 't', 'b'},
            desc_T1, {'w', 'p', 's', 't', 'b'},
            desc_T1, {'w', 'p', 's', 't', 'b'},
            HIPTENSOR_R_64F
        );

        hiptensorPlanPreference_t pref_1;
        hiptensorCreatePlanPreference(
            handle, &pref_1,
            HIPTENSOR_ALGO_GREEDY,  // Use greedy path optimization
            HIPTENSOR_JIT_MODE_NONE
        );

        size_t workspace_size_1 = 0;
        hiptensorEstimateWorkspaceSize(
            handle, desc_1, pref_1,
            HIPTENSOR_WORKSPACE_RECOMMENDED,
            &workspace_size_1
        );

        // Allocate workspace and create plan
        WorkspaceCache& cache_1 = workspace_cache["L_theta"];
        cache_1.workspace_size = workspace_size_1;
        HIP_CHECK(hipMalloc(&cache_1.d_workspace, workspace_size_1));
        cache_1.is_allocated = true;

        hiptensorCreatePlan(
            handle, &plan_L_theta,
            desc_1, pref_1,
            workspace_size_1
        );

        // Similar for other 3 contractions...
    }

public:
    OptimizedHeff(int chi_L, int chi_R, int d, int D_mpo) {
        hiptensorCreate(&handle);
        initialize_descriptors(chi_L, chi_R, d, D_mpo);
        create_optimized_plans();
    }

    void apply(
        double* d_theta,       // Input wavefunction
        double* d_result,      // Output H_eff * theta
        double* d_L,           // Left environment
        double* d_R,           // Right environment
        double* d_W1,          // Left MPO tensor
        double* d_W2,          // Right MPO tensor
        hipStream_t stream
    ) {
        double alpha = 1.0;
        double beta = 0.0;

        // All data stays on GPU, no transfers!

        // Step 1: T1 = L × theta
        double* d_T1 = (double*)workspace_cache["T1"].d_workspace;
        hiptensorContract(
            handle, plan_L_theta,
            &alpha, d_L, d_theta,
            &beta, d_T1, d_T1,
            workspace_cache["L_theta"].d_workspace,
            workspace_cache["L_theta"].workspace_size,
            stream
        );

        // Step 2: T2 = W1 × T1
        double* d_T2 = (double*)workspace_cache["T2"].d_workspace;
        hiptensorContract(
            handle, plan_W1_T1,
            &alpha, d_W1, d_T1,
            &beta, d_T2, d_T2,
            workspace_cache["W1_T1"].d_workspace,
            workspace_cache["W1_T1"].workspace_size,
            stream
        );

        // Step 3: T3 = W2 × T2
        double* d_T3 = (double*)workspace_cache["T3"].d_workspace;
        hiptensorContract(
            handle, plan_W2_T2,
            &alpha, d_W2, d_T2,
            &beta, d_T3, d_T3,
            workspace_cache["W2_T2"].d_workspace,
            workspace_cache["W2_T2"].workspace_size,
            stream
        );

        // Step 4: result = T3 × R
        hiptensorContract(
            handle, plan_T3_R,
            &alpha, d_T3, d_R,
            &beta, d_result, d_result,
            workspace_cache["T3_R"].d_workspace,
            workspace_cache["T3_R"].workspace_size,
            stream
        );
    }

    ~OptimizedHeff() {
        for (auto& [key, cache] : workspace_cache) {
            if (cache.is_allocated) {
                HIP_CHECK(hipFree(cache.d_workspace));
            }
        }
        hiptensorDestroy(handle);
    }
};
```

**Expected Performance Gains:**
- Workspace caching: 10-20% improvement (eliminates repeated allocation)
- Path optimization: 5-15% improvement (better contraction ordering)
- No CPU transfers: 20-40% improvement (eliminates PCIe bottleneck in PDMRG)
- **Total expected: 35-75% improvement over current PDMRG, 15-35% over PDMRG2**

---

## 5. Main DMRG Loop with Stream Parallelization

### 5.1 Overall Algorithm

```cpp
void parallel_dmrg_gpu(
    MPS* mps,
    MPO* mpo,
    int n_streams,
    int max_sweeps,
    double tol
) {
    // Initialize segments
    std::vector<MPSSegment> segments(n_streams);
    int sites_per_segment = mps->n_sites / n_streams;

    for (int i = 0; i < n_streams; i++) {
        segments[i].start_site = i * sites_per_segment;
        segments[i].end_site = (i + 1) * sites_per_segment - 1;
        HIP_CHECK(hipStreamCreate(&segments[i].stream));

        // Allocate device memory for segment
        allocate_segment_memory(&segments[i], mps, mpo);

        // Initialize V bridges to identity (first iteration)
        initialize_bridge_matrices(&segments[i]);
    }

    // Initialize boundary merging
    BoundaryMerger merger(n_streams);

    // Main sweep loop
    double energy = 0.0;
    for (int sweep = 0; sweep < max_sweeps; sweep++) {
        bool forward = (sweep % 2 == 0);

        // Phase 1: Independent segment sweeps
        for (int i = 0; i < n_streams; i++) {
            launch_segment_sweep(
                &segments[i],
                mpo,
                forward,
                segments[i].stream
            );
        }

        // Synchronize all streams before boundary merge
        for (int i = 0; i < n_streams; i++) {
            HIP_CHECK(hipStreamSynchronize(segments[i].stream));
        }

        // Phase 2: Boundary reconciliation
        // Staggered merge pattern (even boundaries, then odd)
        energy = 0.0;

        // Even boundaries: 0-1, 2-3, 4-5, ...
        for (int i = 0; i < n_streams - 1; i += 2) {
            BoundaryMergeResult result = merger.merge_boundary(
                &segments[i],
                &segments[i + 1],
                mpo,
                segments[i].right_boundary_site,
                mps->max_bond_dim,
                mps->cutoff
            );
            energy += result.energy;
        }

        // Odd boundaries: 1-2, 3-4, 5-6, ...
        for (int i = 1; i < n_streams - 1; i += 2) {
            BoundaryMergeResult result = merger.merge_boundary(
                &segments[i],
                &segments[i + 1],
                mpo,
                segments[i].right_boundary_site,
                mps->max_bond_dim,
                mps->cutoff
            );
            energy += result.energy;
        }

        // Check convergence
        if (sweep > 0 && std::abs(energy - prev_energy) < tol) {
            printf("Converged at sweep %d: E = %.12f\n", sweep, energy);
            break;
        }
        prev_energy = energy;
    }

    // Cleanup
    for (int i = 0; i < n_streams; i++) {
        free_segment_memory(&segments[i]);
        HIP_CHECK(hipStreamDestroy(segments[i].stream));
    }
}
```

### 5.2 Segment Sweep Implementation

```cpp
void launch_segment_sweep(
    MPSSegment* segment,
    MPO* mpo,
    bool forward,
    hipStream_t stream
) {
    OptimizedHeff heff(
        segment->max_chi,
        segment->max_chi,
        mpo->phys_dim,
        mpo->bond_dim
    );

    LanczosSolver lanczos(stream);

    if (forward) {
        // Left-to-right sweep within segment
        for (int site = segment->start_site; site < segment->end_site; site++) {
            // Two-site optimization
            double* d_theta = get_two_site_wavefunction(segment, site);
            double* d_theta_opt;

            double site_energy = lanczos.solve(
                d_theta,
                [&](double* in, double* out) {
                    heff.apply(
                        in, out,
                        segment->d_L_envs + offset_L(site),
                        segment->d_R_envs + offset_R(site + 1),
                        mpo->d_tensors + offset_mpo(site),
                        mpo->d_tensors + offset_mpo(site + 1),
                        stream
                    );
                },
                &d_theta_opt
            );

            // SVD and truncate
            svd_and_update_mps(segment, site, d_theta_opt, stream);

            // Update environments
            update_left_environment(segment, site, mpo, stream);
        }
    } else {
        // Right-to-left sweep within segment
        for (int site = segment->end_site - 1; site >= segment->start_site; site--) {
            // Similar but moving right-to-left
            // ...
        }
    }
}
```

---

## 6. Performance Targets and Validation

### 6.1 Correctness Validation

**Acceptance Criteria:**
- Energy difference vs Quimb DMRG1/2: < 1e-10 (absolute)
- All test problems (Heisenberg, Josephson): Pass validation
- Boundary V matrices: Physically reasonable (no NaN, no overflow)
- MPS gauge: Proper normalization maintained

**Validation Strategy:**
1. Unit tests for each component:
   - accurate_svd_gpu: Compare with CPU reference
   - boundary_merge: Compare energy with CPU merge
   - H_eff optimization: Compare output with CPU H_eff

2. Integration tests:
   - Full DMRG run: Compare final energy with Quimb
   - Convergence trajectory: Similar to CPU implementation
   - Multiple stream counts: All produce same final energy

3. Numerical stability tests:
   - Long chain (L=100): Check no degradation
   - High bond dimension (chi=256): Check memory stability
   - Many sweeps (100+): Check long-term accuracy

### 6.2 Performance Targets

**Baseline (Current State):**
- PDMRG_GPU: ~300-500s per run (Josephson L=20, chi=64)
- PDMRG2_GPU: ~150-250s per run (same problem)
- CPU PDMRG np=2: ~30-60s per run

**Target Performance:**
- Optimized GPU (1 stream): 20-40s per run (2-3x faster than CPU np=2)
- Optimized GPU (2 streams): 10-20s per run (linear scaling)
- Optimized GPU (4 streams): 5-10s per run (linear scaling)
- Optimized GPU (8 streams): 3-6s per run (near-linear scaling)

**Scaling Metrics:**
- Stream efficiency: > 90% for 2 streams, > 80% for 4 streams, > 70% for 8 streams
- H_eff performance: 70-90% of MI300X peak FP64 (vs current 50-70%)
- Memory bandwidth utilization: > 60% of peak

---

## 7. Implementation Phases

### Phase 1: Foundation (Week 1)
**Goal:** Accurate SVD and improved H_eff

**Tasks:**
1. Implement `AccurateSVD_GPU` class with recursive refinement
   - Unit tests vs CPU accurate_svd
   - Validate on small matrices (100x100, 1000x1000)
   - Benchmark recursion overhead

2. Implement `OptimizedHeff` class with workspace caching
   - Replace CPU H_eff in PDMRG
   - Add contraction path optimization
   - Benchmark vs current PDMRG2 implementation

3. Validation:
   - Run single-site DMRG with new components
   - Verify energy matches Quimb (< 1e-10)

**Success Criteria:**
- Accurate SVD matches CPU to 1e-12 precision
- Optimized H_eff achieves 70-90% of peak
- Single-stream DMRG validates correctly

### Phase 2: Segmentation (Week 2)
**Goal:** Multi-stream infrastructure without boundary merging

**Tasks:**
1. Implement `MPSSegment` data structure
   - Memory allocation for segments
   - Stream creation and management
   - Copy initial MPS to segments

2. Implement segment-local DMRG sweeps
   - Adapt existing sweep code to segment boundaries
   - Test convergence on single segment (should match single-stream)

3. Validation:
   - Multi-segment without merging (segments drift apart)
   - Verify each segment internally consistent
   - Check memory footprint scales linearly

**Success Criteria:**
- N independent segments run without interference
- Each segment produces valid local energy
- Memory usage = N × single_segment_size

### Phase 3: Boundary Merging (Week 3)
**Goal:** Exact SVD boundary reconciliation

**Tasks:**
1. Implement `BoundaryMerger` class
   - Two-site theta formation with V bridge
   - Lanczos optimization at boundary
   - V = 1/S update logic

2. Implement staggered merge pattern
   - Even/odd boundary ordering
   - Stream synchronization with hipEvents
   - Energy accumulation across boundaries

3. Validation:
   - Compare boundary energies with CPU merge
   - Verify global energy matches Quimb (< 1e-10)
   - Test multiple stream counts (1, 2, 4, 8)

**Success Criteria:**
- All stream counts produce same final energy
- Energy matches Quimb to 1e-10 precision
- Boundary V matrices remain stable (no overflow)

### Phase 4: Optimization and Scaling (Week 4)
**Goal:** Performance tuning and benchmarking

**Tasks:**
1. Profile and optimize critical paths
   - H_eff: Further tuning (memory layout, precomputation)
   - Boundary merge: Minimize synchronization overhead
   - Memory transfers: Prefetching, pinned memory

2. Implement dynamic load balancing
   - Adjust segment sizes if imbalanced
   - Detect and handle slow segments

3. Comprehensive benchmarking
   - Weak scaling: Fixed problem per stream
   - Strong scaling: Fixed total problem size
   - Comparison with CPU baseline

**Success Criteria:**
- 2 streams: > 90% efficiency vs 1 stream
- 4 streams: > 80% efficiency vs 1 stream
- 8 streams: > 70% efficiency vs 1 stream
- Faster than CPU np=2 baseline for all stream counts

---

## 8. Risk Mitigation

### Technical Risks

**Risk 1: Accurate SVD recursion depth**
- Issue: Deep recursion may cause stack overflow or poor GPU utilization
- Mitigation:
  - Limit max_recursion_depth = 5
  - Iterative implementation if recursion problematic
  - Profile recursion frequency (expect depth 1-2 typical)

**Risk 2: Stream synchronization overhead**
- Issue: Boundary merging requires global synchronization (Amdahl's law)
- Mitigation:
  - Minimize merge frequency (sweep-level, not site-level)
  - Overlap segment sweeps with asynchronous merges where possible
  - Use hipEvents instead of hipStreamSynchronize where feasible

**Risk 3: Memory footprint explosion**
- Issue: N streams × segment_memory + intermediate buffers
- Mitigation:
  - Shared workspace pools across streams
  - Just-in-time allocation for merges
  - Memory profiling and bounds checking

**Risk 4: Numerical instability in V = 1/S**
- Issue: Small singular values cause V overflow
- Mitigation:
  - Clipping: S_clipped = max(S, 1e-12)
  - Accurate SVD reduces small value errors
  - Monitor V norms during debugging

**Risk 5: hipTensor performance not meeting targets**
- Issue: Contraction path optimization insufficient
- Mitigation:
  - Fallback to manually-optimized GEMM sequences
  - Static precomputation of optimal paths
  - Consider alternative libraries (cuTENSOR via AMD compatibility)

### Schedule Risks

**Risk: Recursive SVD too complex**
- Backup plan: Use standard rocsolver_zgesvd with tighter tolerance
- Impact: Slight accuracy degradation (1e-9 vs 1e-10), but may still pass validation

**Risk: Boundary merging bugs extend timeline**
- Backup plan: Simplify to global merge (1 boundary per sweep) first
- Impact: Reduced parallelism but correct algorithm

---

## 9. Success Metrics

### Correctness (MANDATORY)
- ✅ All test problems validate against Quimb (< 1e-10)
- ✅ No numerical instabilities (NaN, Inf) in production runs
- ✅ Reproducible results across runs

### Performance (TARGETS)
- ✅ 1 stream: 2-3x faster than CPU np=2
- ✅ 2 streams: > 90% scaling efficiency
- ✅ 4 streams: > 80% scaling efficiency
- ✅ H_eff: 70-90% of MI300X peak FP64

### Code Quality
- ✅ Clear separation of concerns (H_eff, SVD, merge as reusable classes)
- ✅ Comprehensive unit tests
- ✅ Memory leak free (Valgrind clean)
- ✅ Well-documented public APIs

---

## 10. Next Steps

**Immediate Actions:**
1. Review this design document for completeness
2. Set up development branch: `feature/gpu-optimization`
3. Create skeleton implementations of key classes:
   - `AccurateSVD_GPU`
   - `OptimizedHeff`
   - `BoundaryMerger`
   - `MPSSegment`
4. Implement Phase 1 (AccurateSVD + OptimizedHeff)
5. Unit test and validate Phase 1 components

**Decision Points:**
- After Phase 1: Evaluate H_eff performance gains, adjust strategy if needed
- After Phase 2: Assess memory footprint, consider optimization
- After Phase 3: If validation fails, revisit boundary merge logic
- After Phase 4: If scaling suboptimal, profile and re-optimize

---

## Appendix A: Key Equations

**Boundary Merge Prescription (Stoudenmire & White):**
```
Θ = ψ_left · diag(V) · ψ_right
  = A_L[i, j, a] × V[a] × A_R[a, k, l]
  → Θ[i, j, k, l]
```

**Bridge Matrix Update:**
```
After SVD: Θ_opt = U · Σ · V†
New bridge: V_new[α] = 1 / Σ[α]  (for α in kept bond dimensions)
```

**H_eff Contraction Sequence:**
```
Step 1: T1[w, ap, s1, s2, b] = L[w, ap, a] × theta[a, s1, s2, b]
Step 2: T2[ap, s1p, s2, b, y] = W1[w, s1, s1p, x] × T1[w, ap, s1, s2, b]
Step 3: T3[ap, s1p, s2p, b, y] = W2[x, s2, s2p, y] × T2[ap, s1p, s2, b, x]
Step 4: result[a, s1p, s2p, bp] = T3[a, s1p, s2p, b, y] × R[y, b, bp]
```

**Accurate SVD Recursion:**
```
Standard: M = U · Σ · V†
If Σ[p] / Σ[0] < ε:
    X = U[:, p:]† · M · V[p:, :]†
    X = U_sub · Σ_sub · V_sub†  (recursive call)
    U[:, p:] ← U[:, p:] · U_sub
    V[p:, :] ← V_sub · V[p:, :]
    Σ[p:] ← Σ_sub
```

---

## Appendix B: File Structure

```
gpu-port/
├── src/
│   ├── accurate_svd_gpu.cpp         [NEW]
│   ├── accurate_svd_gpu.h           [NEW]
│   ├── boundary_merge_gpu.cpp       [NEW]
│   ├── boundary_merge_gpu.h         [NEW]
│   ├── heff_optimized_gpu.cpp       [NEW]
│   ├── heff_optimized_gpu.h         [NEW]
│   ├── segment_manager.cpp          [NEW]
│   ├── segment_manager.h            [NEW]
│   ├── pdmrg_gpu.cpp                [REFACTOR]
│   ├── pdmrg2_gpu.cpp               [REFACTOR]
│   └── ...
├── tests/
│   ├── test_accurate_svd.cpp        [NEW]
│   ├── test_boundary_merge.cpp      [NEW]
│   ├── test_heff_optimized.cpp      [NEW]
│   ├── test_segment_manager.cpp     [NEW]
│   └── test_integration.cpp         [NEW]
└── benchmarks/
    ├── benchmark_heff.cpp           [NEW]
    ├── benchmark_scaling.cpp        [NEW]
    └── ...
```

---

**End of Design Document**
