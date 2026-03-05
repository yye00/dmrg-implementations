# Phase 2 Design: Multi-Stream Segmentation Infrastructure

## Overview

Map CPU PDMRG's MPI worker pattern to GPU streams for domain-decomposed parallel DMRG.

**Critical**: Preserve exact SVD boundary reconciliation from CPU implementation.

---

## CPU PDMRG Pattern Analysis

### Three-Phase Algorithm

1. **Independent Sweeps** (parallel across MPI ranks)
   - Each rank does local DMRG sweeps on its segment
   - No inter-rank communication during sweeps

2. **Boundary Merge** (synchronization points)
   - Ranks meet at shared boundaries
   - Merge using **V = Lambda^-1** prescription (Stoudenmire & White Eq. 5):
     ```
     theta = psi_left . diag(V) . psi_right
     ```
   - Optimize two-site wavefunction with Lanczos
   - **Exact SVD** to split and compute new V = 1/S

3. **Alternating Boundaries**
   - Even iteration: merge boundaries (0↔1, 2↔3, ...)
   - Odd iteration: merge boundaries (1↔2, 3↔4, ...)

### Key Functions (from CPU)

```python
# pdmrg/parallel/merge.py
def merge_boundary_tensors(psi_left, psi_right, V,
                           L_env, R_env, W_left, W_right,
                           max_bond, max_iter=30, tol=1e-10,
                           skip_optimization=False):
    # Step 1: Form theta = psi_left . diag(V) . psi_right
    V_psi_right = V[:, None, None] * psi_right
    theta = np.einsum('ija,akl->ijkl', psi_left, V_psi_right)

    # Step 2: Optimize with Lanczos
    energy, theta_opt = optimize_two_site(L_env, R_env, W_left, W_right, theta, ...)

    # Step 3: Exact SVD
    U, S, Vh = accurate_svd(M)

    # Step 4: Compute new V = 1/S (with regularization)
    V_new = compute_v_from_svd(S)  # = 1 / clip(S, 1e-12, None)

    return A_left_new, A_right_new, V_new, energy, trunc_err

# pdmrg/numerics/accurate_svd.py
def compute_v_from_svd(S, regularization=1e-12):
    return 1.0 / np.clip(S, regularization, None)
```

---

## GPU Mapping Strategy

### MPI Ranks → HIP Streams

| CPU (MPI)                    | GPU (HIP Streams)           |
|------------------------------|-----------------------------|
| MPI rank                     | HIP stream                  |
| MPI_Send/Recv                | hipMemcpyAsync + Events     |
| MPI_Barrier                  | hipStreamSynchronize        |
| Each rank owns segment       | Each stream owns segment    |
| Rank boundaries              | Stream boundaries           |

### Stream-Based Segmentation

```
Chain: [0] [1] [2] [3] [4] [5] [6] [7]

4 Streams:
  Stream 0: [0] [1]           boundary_right
  Stream 1:     [2] [3]       boundary_left, boundary_right
  Stream 2:         [4] [5]   boundary_left, boundary_right
  Stream 3:             [6] [7]   boundary_left
```

### Synchronization Pattern

```cpp
// Even iteration: merge (0↔1, 2↔3)
for (int i = 0; i < n_streams - 1; i += 2) {
    hipStreamSynchronize(streams[i]);
    hipStreamSynchronize(streams[i+1]);

    merge_boundary_gpu(segments[i], segments[i+1], boundaries[i]);

    // Broadcast results back to both streams
}

// Odd iteration: merge (1↔2, 3↔4)
for (int i = 1; i < n_streams - 1; i += 2) {
    hipStreamSynchronize(streams[i]);
    hipStreamSynchronize(streams[i+1]);

    merge_boundary_gpu(segments[i], segments[i+1], boundaries[i]);
}
```

---

## Phase 2 Implementation Plan

### 2.1: Data Structures

**File**: `gpu-port/src/stream_segment.h`

```cpp
struct BoundaryData {
    // MPS tensors at boundary
    double* d_psi_left;   // (chi_L, d, chi_bond) - right edge of left segment
    double* d_psi_right;  // (chi_bond, d, chi_R) - left edge of right segment

    // V = Lambda^-1 bridge matrix
    double* d_V;          // (chi_bond,) - current V values

    // Environments for two-site optimization
    double* d_L_env;      // (chi_L, D_mpo, chi_L)
    double* d_R_env;      // (chi_R, D_mpo, chi_R)

    // MPO tensors at boundary sites
    double* d_W_left;     // (D_mpo, d, d, D_mpo)
    double* d_W_right;    // (D_mpo, d, d, D_mpo)

    // Dimensions
    int chi_L, chi_R, chi_bond;
    int d, D_mpo;
};

class StreamSegment {
public:
    StreamSegment(int segment_id, int start_site, int end_site,
                  int chi_max, int d, int D_mpo, hipStream_t stream);

    // Local DMRG sweep (runs independently on this stream)
    void sweep_left_to_right();
    void sweep_right_to_left();

    // Boundary data access
    BoundaryData* get_left_boundary();
    BoundaryData* get_right_boundary();

    // Rebuild environments at boundaries (before merge)
    void rebuild_left_boundary_env();
    void rebuild_right_boundary_env();

    // Update V after canonization
    void recompute_boundary_v(bool left_boundary);

private:
    int id_;
    int start_site_, end_site_;
    hipStream_t stream_;

    // Local MPS, environments, MPO
    double* d_mps_tensors_;
    double* d_L_envs_;
    double* d_R_envs_;
    double* d_mpo_tensors_;

    BoundaryData left_boundary_;
    BoundaryData right_boundary_;

    // Use OptimizedHeff for local sweeps
    OptimizedHeff* heff_;
};
```

### 2.2: Boundary Merge (Exact SVD)

**File**: `gpu-port/src/boundary_merge_gpu.h`

```cpp
class BoundaryMergeGPU {
public:
    BoundaryMergeGPU(int max_bond, int max_iter, double tol);

    // Main merge function (mirrors CPU merge_boundary_tensors)
    void merge(BoundaryData* left, BoundaryData* right,
               double& energy, double& trunc_err,
               bool skip_optimization = false,
               hipStream_t stream = 0);

private:
    // Step 1: Form theta = psi_left . diag(V) . psi_right
    void form_theta_from_boundary(
        const double* d_psi_left,
        const double* d_psi_right,
        const double* d_V,
        double* d_theta,
        int chi_L, int d_L, int d_R, int chi_R, int chi_bond,
        hipStream_t stream
    );

    // Step 2: Optimize with Lanczos (or compute energy if skip)
    void optimize_two_site_gpu(
        const double* d_L_env,
        const double* d_R_env,
        const double* d_W_left,
        const double* d_W_right,
        double* d_theta,  // in/out
        double& energy,
        int chi_L, int d_L, int d_R, int chi_R, int D_mpo,
        hipStream_t stream
    );

    // Step 3: Exact SVD (use AccurateSVD_GPU from Phase 1)
    void split_with_svd(
        const double* d_theta_opt,
        double* d_A_left_new,
        double* d_A_right_new,
        double* d_V_new,
        double& trunc_err,
        int chi_L, int d_L, int d_R, int chi_R,
        int max_bond,
        hipStream_t stream
    );

    // Step 4: Compute V = 1/S with regularization
    void compute_v_from_s(
        const double* d_S,
        double* d_V,
        int k,
        double regularization = 1e-12,
        hipStream_t stream = 0
    );

    int max_bond_;
    int max_iter_;
    double tol_;

    AccurateSVD_GPU* svd_;
    OptimizedHeff* heff_;

    // Workspace for intermediate tensors
    double* d_theta_;
    double* d_theta_opt_;
    double* d_M_;  // Reshaped theta for SVD
};
```

### 2.3: Stream Coordinator

**File**: `gpu-port/src/stream_coordinator.h`

```cpp
class StreamCoordinator {
public:
    StreamCoordinator(int n_streams, int chain_length,
                      int chi_max, int d, int D_mpo);
    ~StreamCoordinator();

    // Main DMRG iteration
    void run_iteration(int iter);

    // Get total energy
    double get_energy() const;

private:
    // Sweep phase: all streams work independently
    void sweep_phase_forward();
    void sweep_phase_backward();

    // Merge phase: synchronize and merge boundaries
    void merge_even_boundaries(int iter);  // 0↔1, 2↔3, ...
    void merge_odd_boundaries(int iter);   // 1↔2, 3↔4, ...

    int n_streams_;
    int chain_length_;

    std::vector<hipStream_t> streams_;
    std::vector<StreamSegment*> segments_;
    std::vector<BoundaryMergeGPU*> mergers_;

    // Synchronization events
    std::vector<hipEvent_t> boundary_events_;
};
```

---

## Implementation Steps

### Step 1: StreamSegment Class
- [ ] Implement StreamSegment constructor with memory allocation
- [ ] Implement local sweep methods (using OptimizedHeff from Phase 1)
- [ ] Implement boundary data extraction
- [ ] Implement environment rebuilding at boundaries
- [ ] Implement V recomputation after canonization

### Step 2: BoundaryMergeGPU Class
- [ ] Implement form_theta_from_boundary (tensor contraction with diag(V))
- [ ] Implement optimize_two_site_gpu (Lanczos eigensolver)
- [ ] Implement split_with_svd (use AccurateSVD_GPU from Phase 1)
- [ ] Implement compute_v_from_s (V = 1/S with regularization)
- [ ] Integrate all steps into merge() method

### Step 3: StreamCoordinator Class
- [ ] Implement stream creation and segment distribution
- [ ] Implement sweep_phase (launch independent sweeps)
- [ ] Implement merge_even/odd_boundaries (synchronization + merge)
- [ ] Implement energy collection across streams

### Step 4: Testing
- [ ] Test single stream vs Quimb (should match)
- [ ] Test 2-stream vs Quimb (validate boundary merge)
- [ ] Test 4-stream, 8-stream scaling
- [ ] Verify exact SVD boundary reconciliation (< 1e-10)

---

## Success Criteria (Phase 2)

1. **Correctness**: Multi-stream results match Quimb (< 1e-10 tolerance)
2. **Scaling**: 2-stream achieves ~1.5x speedup over 1-stream
3. **Exact SVD**: Boundary reconciliation uses accurate_svd (not rSVD)
4. **Stream Independence**: Each stream can sweep without blocking others
5. **Synchronization**: Clean merge at boundaries with proper event handling

---

## Key Differences from CPU

| Aspect              | CPU (MPI)                | GPU (Streams)           |
|---------------------|--------------------------|-------------------------|
| Parallelism         | Process-level            | Stream-level            |
| Communication       | MPI_Send/Recv            | hipMemcpyAsync + Events |
| Memory              | Distributed              | Shared GPU memory       |
| Synchronization     | MPI_Barrier              | hipStreamSynchronize    |
| Overhead            | ~ms (network)            | ~μs (on-device)         |

**Advantage**: GPU streams have much lower synchronization overhead than MPI!

---

## Notes

- **Exact SVD is non-negotiable**: Use AccurateSVD_GPU from Phase 1, not rSVD
- **V = Lambda^-1 must be preserved**: Critical for boundary reconciliation accuracy
- **Regularization**: Clip S to >= 1e-12 before inversion (prevents numerical blowup)
- **Environment rebuilding**: Must rebuild L_env/R_env at boundaries before each merge
- **Two-site optimization**: Use Lanczos eigensolver at boundary merge points

---

## References

- Stoudenmire & White, "Minimally Entangled Typical Thermal States" (Eq. 5)
- CPU implementation: `pdmrg/parallel/merge.py`
- CPU accurate_svd: `pdmrg/numerics/accurate_svd.py`
- Phase 1: `AccurateSVD_GPU` (already implemented and validated)
