# PDMRG-GPU Implementation Prompt

## Overview

Implement a **Parallel DMRG (PDMRG) on GPU** using HIP streams for parallelism, based on the working `dmrg2-gpu` two-site DMRG implementation. This replaces MPI inter-process communication with concurrent HIP streams operating on independent chain segments within a single GPU. Boundary exchange between segments uses **accurate SVD** (V = Λ⁻¹ from singular values).

**Location**: `pdmrg-gpu/src/` (new directory, ignore the broken `pdmrg-gpu/` if it exists)

**Base code**: Copy architecture from `dmrg2-gpu/src/` — same scalar_traits.h, same rocBLAS GEMM patterns, same Lanczos eigensolver, same fused MPO (WW) approach.

---

## Algorithm: Stream-Parallel DMRG

### Chain Partitioning

Partition an L-site chain into P segments (P = number of HIP streams, default 2-4):

```
Segment 0: sites [0, b1)
Segment 1: sites [b1, b2)
...
Segment P-1: sites [b_{P-1}, L)
```

Where `b_k = k * (L / P)` with remainder distributed to first segments. Each segment is assigned to one HIP stream. Segments share **boundary bonds** — the bond between the last site of segment k and the first site of segment k+1.

### Data Ownership

Each stream/segment owns:
- **Local MPS tensors**: `MPS[first..last]` for its sites
- **Local left environments**: `L_env[first..last+1]`
- **Local right environments**: `R_env[first..last+1]`
- **Local fused MPO**: `WW[first..last-1]` (two-site fused W tensors)
- **Boundary V-matrix**: `V_left` and `V_right` (diagonal matrices = 1/singular_values at segment boundaries)

The MPO is global (read-only, shared across all streams without contention).

### Algorithm Phases

#### Phase 0: Initialization (Serial on default stream)

1. Allocate all MPS tensors on GPU with random initialization (or product state)
2. Set MPO tensors (shared, one copy on GPU)
3. Precompute fused MPO tensors WW[bond] for all bonds (same as dmrg2-gpu)
4. **Warmup**: Run standard dmrg2-gpu sweeps on the full chain for `n_warmup` sweeps (e.g., 3-5 sweeps at target chi). This produces a good initial MPS in right-canonical form.
5. Partition chain: compute segment boundaries
6. Compute initial V-matrices at each boundary bond via SVD:
   ```
   For boundary bond at site b between segment k and k+1:
     theta = MPS[b] * MPS[b+1]        // form two-site tensor
     U, S, Vh = SVD(theta)             // accurate SVD
     V_boundary[k] = diag(1.0 / S)    // V = Λ⁻¹ (regularize small S < 1e-14)
     MPS[b] = U                        // left-canonical
     MPS[b+1] = diag(S) * Vh           // right part gets weight
   ```
7. Build all left and right environments for each segment independently

#### Phase 1: Independent Segment Sweeps (Parallel — one HIP stream per segment)

Each segment performs `n_local_sweeps` (typically 2-4) standard two-site DMRG sweeps **within its segment boundaries only**. This is identical to dmrg2-gpu sweep logic but restricted to local sites:

```
For segment k owning sites [first, last]:
  Stream[k]:
    sweep_left_to_right():
      for bond in [first, first+1, ..., last-1]:
        optimize_bond(bond, 'R')    // Lanczos + SVD split
        update_left_env(bond)
    sweep_right_to_left():
      for bond in [last-1, last-2, ..., first]:
        optimize_bond(bond, 'L')
        update_right_env(bond+1)
```

**Boundary environments** at segment edges use the V-matrix to properly weight the boundary:
- Segment k's rightmost R_env incorporates the V-matrix from boundary k
- Segment k's leftmost L_env incorporates the V-matrix from boundary k-1

All P streams run concurrently. No synchronization during this phase.

**Stream assignment**: Each segment's GEMM/BLAS calls are issued on its own `hipStream_t`. rocBLAS handle per stream (via `rocblas_set_stream`). This gives the GPU scheduler freedom to overlap work from different segments.

#### Phase 2: Boundary Merge (Sequential across boundaries, but each merge is GPU-accelerated)

After all streams complete their local sweeps (synchronize all streams), perform boundary merges:

**Even boundaries first** (0↔1, 2↔3, ...), then **odd boundaries** (1↔2, 3↔4, ...):

```
For each active boundary between segment k and k+1 at bond site b:
  1. Form boundary theta:
     theta = MPS[b] * diag(V_boundary[k]) * MPS[b+1]
     // MPS[b] is (cL, d, chi_mid) from segment k's right edge
     // MPS[b+1] is (chi_mid, d, cR) from segment k+1's left edge
     // V inserts the boundary coupling

  2. Two-site optimization:
     Apply H_eff using L_env from segment k and R_env from segment k+1
     Lanczos eigensolver → ground state theta

  3. SVD split:
     U, S, Vh = SVD(theta)
     Truncate to chi_max
     V_boundary_new[k] = diag(1.0 / S)
     MPS[b] = U                    // left-canonical
     MPS[b+1] = diag(S) * Vh      // right part

  4. Update environments:
     Update L_env[b+1] from L_env[b] and new MPS[b]
     Update R_env[b] from R_env[b+1] and new MPS[b+1]
```

Each boundary merge can run on its own stream (even boundaries in parallel, then odd boundaries in parallel).

#### Phase 3: Convergence Check

```
E_global = sum of segment energies (from last Lanczos eigenvalue at each bond)
           OR recompute <psi|H|psi> at boundary bonds
dE = |E_global - E_prev|
converged = dE < tol (e.g., 1e-10)
```

If not converged, go back to Phase 1. Alternate sweep direction each outer iteration (even sweeps start segments L→R, odd sweeps start R→L, using the staggered pattern from the Python PDMRG).

#### Phase 4: Final Assembly & Optional Cleanup

After convergence:
1. MPS tensors are already in GPU memory in correct global order
2. Optionally run 1-2 full-chain dmrg2-gpu sweeps to polish cross-boundary accuracy

---

## Implementation Architecture

### File Structure

```
pdmrg-gpu/
├── CMakeLists.txt
└── src/
    ├── scalar_traits.h          // Copy from dmrg2-gpu (identical)
    ├── pdmrg_gpu.h              // Class declaration
    ├── pdmrg_gpu_impl.h         // Full implementation
    ├── pdmrg_gpu.cpp            // Explicit template instantiations
    └── test_pdmrg_gpu.cpp       // Tests: Heisenberg + Josephson
```

### Class Design

```cpp
template<typename Scalar>
class PDMRGGPU {
public:
    // Constructor: L sites, d physical dim, chi_max bond dim, D_mpo, P segments
    PDMRGGPU(int L, int d, int chi_max, int D_mpo, int n_segments, double tol=1e-10);
    ~PDMRGGPU();

    // Setup
    void set_mpo(const std::vector<Scalar*>& h_mpo);
    void initialize_mps_random();

    // Run
    double run(int n_outer_sweeps, int n_local_sweeps=2, int n_warmup=3);

    // Results
    double get_energy() const;
    void get_mps(std::vector<std::vector<Scalar>>& h_mps) const;
    void set_cpu_svd(bool use_cpu);

private:
    // === Chain partitioning ===
    int n_segments_;
    std::vector<int> seg_first_;    // seg_first_[k] = first site of segment k
    std::vector<int> seg_last_;     // seg_last_[k] = last site of segment k (inclusive)
    std::vector<int> boundary_bonds_; // bonds between segments

    // === HIP streams (one per segment + extras for merges) ===
    std::vector<hipStream_t> streams_;
    std::vector<rocblas_handle> handles_;  // one rocBLAS handle per stream

    // === Global GPU data (shared across streams) ===
    std::vector<Scalar*> d_mps_tensors_;    // [L] MPS tensors
    std::vector<Scalar*> d_mpo_tensors_;    // [L] MPO tensors (read-only)
    std::vector<Scalar*> d_WW_;             // [L-1] fused two-site MPO
    std::vector<Scalar*> d_W_matrices_;     // [L] single-site W for env updates
    std::vector<int> bond_dims_;            // [L+1] bond dimensions

    // === Per-segment environments ===
    std::vector<Scalar*> d_L_envs_;         // [L+1] left environments
    std::vector<Scalar*> d_R_envs_;         // [L+1] right environments

    // === Boundary V-matrices (diagonal, stored as vectors of singular values) ===
    std::vector<double*> d_V_boundary_;     // [n_segments-1] = 1/S at each boundary

    // === Per-stream workspaces ===
    // Each stream needs independent workspace for Lanczos, apply_heff, SVD
    struct StreamWorkspace {
        Scalar* d_theta;
        Scalar* d_heff_result;
        Scalar* d_T1;              // intermediate for apply_heff step 1
        Scalar* d_T2;              // intermediate for apply_heff step 2
        Scalar* d_lanczos_v;       // Lanczos vector storage
        Scalar* d_ritz_coeffs;
        // Batched GEMM pointer arrays
        Scalar** d_batch_A;
        Scalar** d_batch_B;
        Scalar** d_batch_C;
        // SVD workspace
        Scalar* d_svd_U;
        double* d_svd_S;
        Scalar* d_svd_Vh;
    };
    std::vector<StreamWorkspace> workspaces_;

    // === Core methods (operate on specified stream) ===
    void warmup_sweeps(int n_warmup);
    void compute_boundary_V(int boundary_idx);
    void partition_chain();

    // Two-site operations (stream-aware versions of dmrg2-gpu methods)
    void form_theta_two_site(int site, int stream_idx);
    void apply_heff_two_site(int site, const Scalar* d_in, Scalar* d_out, int stream_idx);
    void lanczos_eigensolver(int site, Scalar* d_theta, int theta_size, int stream_idx);
    void svd_split(int site, Scalar* d_theta, char direction, int stream_idx);
    void optimize_bond(int site, char direction, int stream_idx);

    // Environment updates (stream-aware)
    void update_left_env(int site, int stream_idx);
    void update_right_env(int site, int stream_idx);
    void build_segment_environments(int seg_idx);

    // Sweep methods
    void segment_sweep_LR(int seg_idx);    // left-to-right within segment
    void segment_sweep_RL(int seg_idx);    // right-to-left within segment

    // Boundary operations
    void merge_boundary(int boundary_idx, int stream_idx);
    void form_boundary_theta(int boundary_idx, Scalar* d_theta, int stream_idx);

    // Fused MPO
    void precompute_fused_mpo(const std::vector<Scalar*>& h_mpo_tensors);

    // Memory management
    void allocate_stream_workspaces();
    void free_gpu_resources();
};
```

### Key Differences from dmrg2-gpu

| Aspect | dmrg2-gpu | pdmrg-gpu |
|--------|-----------|-----------|
| Parallelism | None (serial sweeps) | P concurrent HIP streams |
| Sweep scope | Full chain [0, L-1] | Per-segment [first, last] |
| rocBLAS handles | 1 global | P handles (one per stream) |
| Workspaces | 1 set | P sets (one per stream) |
| Boundary coupling | N/A | V-matrix (accurate SVD) |
| Environments | Global L/R | Per-segment with boundary conditions |
| Warmup | N/A | Full-chain dmrg2 warmup before partitioning |

---

## Detailed Implementation Notes

### HIP Stream Management

```cpp
// Create P streams + handles
for (int k = 0; k < n_segments_; k++) {
    hipStreamCreate(&streams_[k]);
    rocblas_create_handle(&handles_[k]);
    rocblas_set_stream(handles_[k], streams_[k]);
}
```

All rocBLAS calls within a segment's sweep use that segment's handle, which is bound to that segment's stream. This ensures operations within a segment are serialized (correct dependency order) while operations across segments can overlap.

### Per-Stream Workspace Allocation

Each stream needs its own scratch buffers to avoid data races. Size each workspace for the maximum theta size that stream's segment can produce:

```
max_theta_size = chi_max * d * d * chi_max  // two-site theta
T1/T2 size = D * d^2 * chi_max * chi_max    // apply_heff intermediates
lanczos_v = max_lanczos_iter * max_theta_size
```

This multiplies GPU memory by ~P. For P=2 on MI300X (192 GB HBM3), this is fine up to chi~500.

### Boundary Theta Formation with V-Matrix

The V-matrix is diagonal (stored as a vector of `1/S` values). Forming the boundary theta:

```
Step 1: Scale MPS[b]'s right index by V
  // MPS[b] has shape (cL, d, chi_mid), view as (cL*d, chi_mid)
  // V is diagonal (chi_mid,)
  // scaled = MPS[b] * diag(V)  →  column-scale by V
  For column j: scaled[:, j] = MPS[b][:, j] * V[j]

Step 2: Contract with MPS[b+1]
  theta = scaled @ MPS[b+1]  // (cL*d, chi_mid) @ (chi_mid, d*cR) → (cL*d, d*cR)
```

The column-scaling is a simple element-wise multiply (one HIP kernel or `rocblas_*dgmm` with diagonal).

### Boundary Environment Construction

At segment boundaries, the environment must account for the rest of the chain **outside** the segment. During warmup, the full-chain environments are computed. After partitioning:

- Segment k's `L_env[first]` = full-chain `L_env[first]` (from warmup or prior iteration)
- Segment k's `R_env[last+1]` = full-chain `R_env[last+1]` (from warmup or prior iteration)

During local sweeps, only interior environments are updated. Boundary environments stay fixed until the next merge phase updates them.

### Staggered Sweep Pattern

To ensure segments meet at boundaries in compatible canonical forms:

```
Outer iteration (even):
  All segments: sweep L→R, then R→L
  Even boundary merges, then odd boundary merges

Outer iteration (odd):
  All segments: sweep R→L, then L→R
  Odd boundary merges, then even boundary merges
```

This ensures:
- After L→R sweep: segment's right-edge MPS[last] is freshly optimized, left-canonical
- After R→L sweep: segment's left-edge MPS[first] is freshly optimized, right-canonical
- Merge sees correctly canonicalized tensors at boundaries

### Energy Computation

Track energy from the Lanczos eigenvalue at each optimized bond. The global energy estimate comes from the merge step's Lanczos eigenvalue (which sees the full effective Hamiltonian across the boundary). Report the minimum energy seen across all boundary merges as the global estimate.

### SVD at Boundaries (Accurate, Not Approximate)

The boundary merge SVD must be **full accurate SVD** (not randomized or truncated iteratively). This is the same CPU LAPACK SVD used in dmrg2-gpu's `svd_split`:

```cpp
// CPU SVD (default, faster for chi < 200)
// Copy theta from GPU to host
// Call dgesvd_ / zgesvd_
// Truncate: keep singular values > 1e-14, cap at chi_max
// V_new = 1.0 / S[0:new_chi]
// Copy U, S*Vh back to GPU
```

The V-matrix update is simply `V[i] = 1.0 / S[i]` with regularization:
```cpp
for (int i = 0; i < new_chi; i++) {
    V[i] = (S[i] > 1e-14) ? 1.0 / S[i] : 0.0;
}
```

---

## Build System

```cmake
cmake_minimum_required(VERSION 3.21)
project(pdmrg_gpu LANGUAGES CXX HIP)

# Same as dmrg2-gpu:
find_package(hip REQUIRED)
find_package(rocblas REQUIRED)
find_package(rocsolver REQUIRED)
find_package(LAPACK REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_HIP_ARCHITECTURES gfx942)

add_executable(pdmrg_gpu
    src/pdmrg_gpu.cpp
    src/test_pdmrg_gpu.cpp
)
target_link_libraries(pdmrg_gpu hip::device roc::rocblas roc::rocsolver ${LAPACK_LIBRARIES})
target_compile_options(pdmrg_gpu PRIVATE -O3)
```

---

## Test Cases

### Test 1: Heisenberg Chain (Real, d=2)

Same Heisenberg MPO builder as dmrg2-gpu. Test configurations:

| L | chi | P (segments) | Expected energy | Tolerance |
|---|-----|-------------|-----------------|-----------|
| 8 | 20 | 2 | -3.374932598688 | 1e-8 |
| 16 | 30 | 2 | -6.911737... | 1e-8 |
| 32 | 50 | 2 | -13.997315... | 1e-8 |
| 32 | 50 | 4 | -13.997315... | 1e-8 |
| 64 | 100 | 4 | -28.175424... | 1e-6 |

### Test 2: Josephson Junction (Complex, d=3)

Same Josephson MPO builder as dmrg2-gpu. Test configurations:

| L | chi | P (segments) | Expected energy | Tolerance |
|---|-----|-------------|-----------------|-----------|
| 6 | 20 | 2 | -1.748843... | 1e-8 |
| 12 | 30 | 2 | -3.843385... | 1e-8 |
| 24 | 50 | 2 | -8.038883... | 1e-8 |

### Test 3: Speedup Measurement

Compare wall time of pdmrg-gpu (P=2,4) vs dmrg2-gpu (P=1) at L=64, chi=100:
- Report time per sweep
- Report total time to convergence
- Speedup should be >1.3x for P=2 at chi>=100 (GPU has enough work per stream)

### CLI Interface

```
Usage: pdmrg_gpu [L] [chi_max] [n_sweeps] [options]

Options:
  --segments P       Number of parallel segments (default: 2)
  --local-sweeps N   Local sweeps per outer iteration (default: 2)
  --warmup N         Warmup sweeps before partitioning (default: 3)
  --gpu-svd          Use GPU SVD instead of CPU LAPACK
  --josephson        Run Josephson junction model
  --nmax N           Charge truncation for Josephson (default: 1)
  --ej EJ            Josephson energy (default: 1.0)
  --ec EC            Charging energy (default: 0.5)
  --phi PHI          External flux (default: pi/4)
```

---

## Memory Budget (MI300X, 192 GB HBM3)

For L=64, chi=200, d=2, D=5, P=4 segments:

| Component | Size | Total |
|-----------|------|-------|
| MPS tensors (64) | 200×2×200×8B = 640 KB each | 40 MB |
| MPO tensors (64) | 5×2×2×5×8B = 1.6 KB each | 0.1 MB |
| Fused WW (63) | 20×20×8B = 3.2 KB each | 0.2 MB |
| L/R environments (130) | 200×5×200×8B = 1.6 MB each | 208 MB |
| Per-stream workspace (×4) | ~80 MB each (theta+T1+T2+Lanczos) | 320 MB |
| V-matrices (3 boundaries) | 200×8B each | ~5 KB |
| **Total** | | **~570 MB** |

Well within MI300X capacity. Can scale to chi=1000+ with P=4.

---

## Implementation Order

1. **Copy scalar_traits.h** from dmrg2-gpu (identical)
2. **Implement PDMRGGPU class** with stream management
3. **Port dmrg2-gpu core methods** (apply_heff, Lanczos, SVD, env updates) to accept stream_idx parameter — route all rocBLAS calls through `handles_[stream_idx]`
4. **Add chain partitioning** and boundary V-matrix computation
5. **Implement segment_sweep_LR/RL** (restricted-range sweeps)
6. **Implement merge_boundary** with V-weighted theta formation
7. **Implement run()** outer loop: warmup → partition → [local sweeps → merge] × n_outer
8. **Add tests** — verify energy matches dmrg2-gpu to within tolerance
9. **Benchmark** — measure stream overlap and speedup

### Critical Implementation Details to Get Right

1. **rocBLAS handle per stream**: Every BLAS call must use the correct handle for its stream. Passing the wrong handle silently produces wrong results or races.

2. **Workspace isolation**: Each stream's Lanczos vectors, theta buffer, T1/T2 intermediates must be separate allocations. Shared read-only data (MPO, WW) is fine.

3. **Stream synchronization points**:
   - `hipStreamSynchronize(streams_[k])` after each segment finishes local sweeps (before merge)
   - `hipDeviceSynchronize()` after all merges complete (before next local sweep phase)

4. **Boundary environment consistency**: After a merge updates MPS[b] and MPS[b+1], the adjacent segments must rebuild their edge environments before the next local sweep. Specifically:
   - Segment k must rebuild `R_env` at its right boundary from the new MPS[b]
   - Segment k+1 must rebuild `L_env` at its left boundary from the new MPS[b+1]

5. **Bond dimension changes at boundaries**: After SVD truncation at a boundary, the bond dimension may change. Update `bond_dims_[b+1]` and reallocate MPS tensors and environments if needed (same reallocation logic as dmrg2-gpu's `ensure_L_env_alloc` / `ensure_R_env_alloc`).

6. **Warmup reuse**: The warmup phase uses the same `apply_heff_two_site`, `lanczos_eigensolver`, `svd_split` methods as local sweeps — just on stream 0 with full-chain range. Factor this so warmup and local sweeps share code.

7. **Column-major layout**: All tensors follow the same column-major convention as dmrg2-gpu. The V-matrix scaling (diagonal multiply) operates on the rightmost index of MPS[b] viewed as (cL*d, chi_mid).
