# GPU-CPU Parity Fix: Complete Implementation Guide

**Goal**: Make `pdmrg-gpu` numerically and algorithmically identical to the CPU `pdmrg` implementation, achieving < 1e-10 accuracy.

**Date Created**: 2026-03-09
**CPU Reference**: Commit `6dbeabf` (includes boundary environment canonicalization fix)
**GPU Current Status**: Phase 2 complete but **12.5% energy error** (UNACCEPTABLE)

---

## Executive Summary

The GPU implementation has **critical accuracy failures** (12.5% vs target < 1e-10) despite claiming "production ready" status. This prompt provides step-by-step instructions to achieve exact numerical parity with the audited, validated CPU implementation.

**Key Principle**: The GPU code must implement the **EXACT SAME ALGORITHM** as CPU PDMRG, with only one difference: **GPU streams replace MPI ranks** for parallelization. Everything else must be identical.

---

## Critical Issues to Fix

### Issue #1: Accuracy Failure (CRITICAL)
**Current state**: 12.5% energy error on Heisenberg L=8 test
**Target**: < 1e-10 accuracy vs exact diagonalization
**Root cause**: Unknown - requires systematic debugging

### Issue #2: Algorithm Drift
**Problem**: GPU and CPU implementations have diverged
**Evidence**: CPU has recent canonicalization fixes (commit 6dbeabf) that GPU lacks
**Impact**: Numerical instability in boundary environments

### Issue #3: Code Quality
**Problem**: 77 TODOs, 6 backup files, unclear production vs experimental code
**Impact**: Maintenance burden, unclear which code to trust

### Issue #4: Production Ambiguity
**Problem**: Multiple DMRG executables (`pdmrg_gpu.cpp`, `dmrg_gpu.cpp`, `dmrg_production.cpp`, etc.)
**Impact**: Unclear which is the canonical implementation

---

## CPU Reference Implementation (Ground Truth)

### Files to Match Exactly

**Core algorithm** (in `pdmrg/pdmrg/`):
1. `dmrg.py` - Main PDMRG algorithm with 4-phase structure
2. `parallel/merge.py` - Boundary merge with V = Λ⁻¹
3. `parallel/communication.py` - MPI coordination (GPU: use streams instead)
4. `parallel/distribute.py` - MPS distribution (GPU: data layout for streams)
5. `numerics/eigensolver.py` - Lanczos eigensolver
6. `numerics/accurate_svd.py` - Exact SVD with `compute_v_from_svd()`
7. `numerics/environments.py` - L_env/R_env building and updates

**Critical recent fixes** (commit 6dbeabf):
- **Canonicalization before environment building**: L_envs require left-canonical MPS, R_envs require right-canonical
- **Boundary merge returns canonical tensors**: Use `Vh` (right-canonical) for R_env updates, not `S*Vh`
- **Warmup parameter tuning**: `tol=1e-10`, lighter convergence

---

## Algorithmic Requirements (Must Match CPU)

### 1. Exact SVD with V = Λ⁻¹

**CPU reference**: `pdmrg/pdmrg/numerics/accurate_svd.py`

```python
def compute_v_from_svd(S, epsilon=1e-14):
    """V = 1/S with safeguards for small singular values."""
    V = np.zeros_like(S)
    mask = S > epsilon
    V[mask] = 1.0 / S[mask]
    V[~mask] = 0.0
    return V
```

**GPU requirements**:
- Use `rocSOLVER dgesvd` (full accuracy, no randomized approximations)
- Compute V = 1/S with epsilon = 1e-14 safeguard
- Apply at EVERY boundary merge (not just some)
- Store V as 1D vector (shape: chi_bond)

**Validation**:
```cpp
// After SVD: U, S, Vt = svd(M)
std::vector<double> V(k);
for (int i = 0; i < k; i++) {
    V[i] = (S[i] > 1e-14) ? (1.0 / S[i]) : 0.0;
}
```

---

### 2. Boundary Environment Canonicalization (NEW FIX)

**CPU reference**: `pdmrg/pdmrg/dmrg.py`, lines 310-368 (commit 6dbeabf)

**Critical insight**: Environments must be built from **canonical** MPS tensors to ensure norm matrices N = I.

#### Left Environments (L_env)

**Before building L_envs from global MPS**:
1. Left-canonicalize all MPS tensors via QR:
   ```python
   for i in range(len(mps) - 1):
       chi_L, d, chi_R = mps[i].shape
       M = mps[i].reshape(chi_L * d, chi_R)
       Q, R = np.linalg.qr(M)
       mps[i] = Q.reshape(chi_L, d, -1)  # Left-canonical
       mps[i+1] = np.tensordot(R, mps[i+1], axes=(1, 0))
   ```

2. Build L_envs using **left-canonical** tensors:
   ```python
   L_env = init_left_env(chi_L, D_L)
   for i in range(global_start):
       L_env = update_left_env(L_env, mps[i], mpo[i])
   ```

**Why**: Left-canonical tensors satisfy A†A = I, giving L_norm = I (identity norm matrix).

#### Right Environments (R_env)

**Before building R_envs from global MPS**:
1. Right-canonicalize all MPS tensors via LQ:
   ```python
   for i in range(len(mps)-1, 0, -1):
       chi_L, d, chi_R = mps[i].shape
       M = mps[i].reshape(chi_L, d * chi_R)
       L, Q = scipy.linalg.lq(M)
       mps[i] = Q.reshape(-1, d, chi_R)  # Right-canonical
       mps[i-1] = np.tensordot(mps[i-1], L, axes=(2, 0))
   ```

2. Build R_envs using **right-canonical** tensors.

**Why**: Right-canonical tensors satisfy AA† = I, giving R_norm = I.

#### Boundary Merge Returns Canonical Tensors

**CPU reference**: `pdmrg/pdmrg/parallel/merge.py`, lines 88-97 (commit 6dbeabf)

After boundary merge SVD: `theta → U, S, Vh`:
- Return `A_left = U` (already left-canonical)
- Return `A_right = S @ Vh` (mixed form for MPS storage)
- **NEW**: Also return `A_right_canonical = Vh` (for R_env updates)

**Usage**:
```python
# Update R_env using CANONICAL tensor (Vh), not mixed form (S*Vh)
R_env_new = update_right_env(R_env, A_right_canonical, mpo)

# But store mixed form in MPS
mps[site] = A_right  # S*Vh
```

**Critical**: Using `S*Vh` for R_env gives R_norm = S² ≠ I, breaking the eigensolver assumption.

**GPU implementation**:
```cpp
// After boundary merge SVD
double* U;      // Shape: (chi_L * d_L, k)
double* S;      // Shape: (k,)
double* Vt;     // Shape: (k, d_R * chi_R)

// Left tensor (left-canonical)
A_left = U.reshape(chi_L, d_L, k);

// Right tensor - RETURN BOTH FORMS
A_right_mixed = (diag(S) @ Vt).reshape(k, d_R, chi_R);  // For MPS storage
A_right_canonical = Vt.reshape(k, d_R, chi_R);          // For R_env update

return {A_left, A_right_mixed, V_new, energy, A_right_canonical};
```

---

### 3. Lanczos Eigensolver (Single-Vector)

**CPU reference**: `pdmrg/pdmrg/numerics/eigensolver.py`

**Requirements**:
- Single-vector Krylov method (NOT block methods)
- Tridiagonalization: α = ⟨v|H|v⟩, β = ||H|v⟩ - α|v⟩||
- LAPACK `dstev` for tridiagonal eigensolve (CPU fallback is CORRECT)
- Convergence: When |E_new - E_old| < tol and |β| < tol

**Validation checks**:
```cpp
// After eigensolver
assert(eigenvalue < 0.0);  // Antiferromagnetic Heisenberg
assert(eigenvalue > -10.0);  // Reasonable magnitude
assert(!std::isnan(eigenvalue));
assert(std::abs(eigenvalue_imag) < 1e-12);  // Must be real
```

**Known rocSOLVER bug**: CPU LAPACK fallback is the CORRECT workaround (see CPU code).

---

### 4. H_eff Application (4-Step Contraction)

**CPU reference**: `pdmrg/pdmrg/numerics/eigensolver.py`, `apply_heff()`

**Contraction order** (CRITICAL for correctness):
```python
def apply_heff(theta, L_env, R_env, W_left, W_right):
    # theta: (chi_L, d_L, d_R, chi_R)
    # L_env: (chi_L, D_L, chi_L)
    # R_env: (chi_R, D_R, chi_R)
    # W_left: (D_L, D_M, d_L, d_L)
    # W_right: (D_M, D_R, d_R, d_R)

    # Step 1: Apply L_env
    out = np.tensordot(L_env, theta, axes=([2], [0]))
    # out: (chi_L, D_L, d_L, d_R, chi_R)

    # Step 2: Apply W_left
    out = np.tensordot(out, W_left, axes=([1, 2], [0, 3]))
    # out: (chi_L, d_R, chi_R, D_M, d_L)

    # Step 3: Apply W_right
    out = np.tensordot(out, W_right, axes=([3, 1], [0, 3]))
    # out: (chi_L, chi_R, d_L, D_R, d_R)

    # Step 4: Apply R_env
    out = np.tensordot(out, R_env, axes=([1, 3], [0, 1]))
    # out: (chi_L, d_L, d_R, chi_R)

    return out
```

**GPU implementation**:
- Use hipTensor for contractions (correct choice)
- Match EXACT contraction order above
- Double-check axis indices (off-by-one errors are common)
- Validate: `||H_eff(theta)||` should be O(1), not O(1e10) or O(1e-10)

---

### 5. Environment Updates

**CPU reference**: `pdmrg/pdmrg/numerics/environments.py`

**Left environment update**:
```python
def update_left_env(L_env, A, W):
    # L_env: (chi_L, D_L, chi_L)
    # A: (chi_L, d, chi_R)  [must be LEFT-CANONICAL]
    # W: (D_L, D_R, d, d)

    # Contract A with L_env
    temp = np.tensordot(A, L_env, axes=([0], [0]))
    # temp: (d, chi_R, D_L, chi_L)

    # Contract with W
    temp = np.tensordot(temp, W, axes=([0, 2], [2, 0]))
    # temp: (chi_R, chi_L, D_R, d)

    # Contract with A† (conjugate transpose)
    L_new = np.tensordot(temp, A.conj(), axes=([1, 3], [0, 1]))
    # L_new: (chi_R, D_R, chi_R)

    return L_new
```

**Right environment update** (mirror of left):
```python
def update_right_env(R_env, A, W):
    # R_env: (chi_R, D_R, chi_R)
    # A: (chi_L, d, chi_R)  [must be RIGHT-CANONICAL]
    # W: (D_L, D_R, d, d)

    # Similar contractions in reverse order
    # CRITICAL: A must be right-canonical (AA† = I)
```

**GPU validation**:
```cpp
// After environment update
double norm = frobenius_norm(L_env);
assert(norm > 0.1 && norm < 100.0);  // Should be O(1)
assert(!std::isnan(norm));
```

---

### 6. Four-Phase PDMRG Structure

**CPU reference**: `pdmrg/pdmrg/dmrg.py`, `pdmrg_main()`

```python
for sweep in range(max_sweeps):
    # PHASE 1: Local sweeps
    local_sweep(direction='right')
    local_sweep(direction='left')

    # PHASE 2: Even boundary merges (streams 0-1, 2-3, ...)
    boundary_merge(boundaries='even')

    # PHASE 3: Local sweeps again
    local_sweep(direction='right')
    local_sweep(direction='left')

    # PHASE 4: Odd boundary merges (streams 1-2, 3-4, ...)
    boundary_merge(boundaries='odd')

    # Energy convergence check
    if abs(E_new - E_old) < tol:
        break
```

**GPU streams replace MPI ranks**:
- CPU: Each MPI rank owns a segment
- GPU: Each HIP stream owns a segment
- Synchronization: MPI barriers → `hipStreamSynchronize()`
- Communication: MPI send/recv → direct memory access (shared GPU memory)

---

### 7. Convergence Criterion

**CPU reference**: `pdmrg/pdmrg/parallel/communication.py`, lines 72-82

```python
# Gather energies from all ranks (GPU: all streams)
all_E = comm.allgather(E_local)

# Filter out None sentinels (idle ranks with no merge)
merge_energies = [e for e in all_E if e is not None]

if merge_energies:
    E_global = min(merge_energies)
else:
    E_global = E_local if E_local is not None else 0.0

# Convergence: |dE| < atol + rtol * |E|
converged = abs(E_new - E_old) < (1e-10 + 1e-12 * abs(E_new))
```

**GPU implementation**:
```cpp
// Collect energies from all streams
std::vector<double> stream_energies;
for (auto& stream : streams) {
    if (stream.has_merge) {
        stream_energies.push_back(stream.boundary_energy);
    }
}

// Take minimum
double E_global = *std::min_element(stream_energies.begin(), stream_energies.end());

// Convergence check
bool converged = std::abs(E_new - E_old) < (1e-10 + 1e-12 * std::abs(E_new));
```

---

## MPS/MPO Loading from Disk

**Requirement**: GPU must load binary MPS/MPO files saved by CPU benchmarks for fair comparison.

### CPU Format

**File structure** (from `benchmarks/benchmark_data_loader.py`):
```python
# MPS format: List of numpy arrays
# - Bulk tensors: (chi_L, d, chi_R)
# - Left boundary: (d, chi_R) OR (chi_L, d, chi_R) with chi_L=1
# - Right boundary: (chi_L, d) OR (chi_L, d, chi_R) with chi_R=1

# MPO format: List of numpy arrays
# - Bulk tensors: (D_L, D_R, d, d)
# - Left boundary: (1, D_R, d, d) or (D_R, d, d)
# - Right boundary: (D_L, 1, d, d) or (D_L, d, d)

# Saved as .npy (single array) or .npz (multiple arrays)
```

### GPU Loader Requirements

**Files**: `pdmrg-gpu/include/mps_mpo_loader.hpp`, `pdmrg-gpu/src/loaded_mpo.hpp`

**Must handle**:
1. Variable bond dimensions (chi changes along chain)
2. Boundary tensor shape ambiguity (2D vs 3D with size-1 dimension)
3. Dtype conversion (Python float64 → C++ double)
4. Memory layout (row-major NumPy → GPU column-major?)

**Auto-detection** (from CPU fix in commit 6dbeabf):
```cpp
// Detect if MPS is in internal format (chi_L, d, chi_R)
// or quimb format (chi_L, chi_R, d)
int phys_dim = 2;  // Assume spin-1/2
bool needs_transpose = true;

// Check a bulk tensor (avoid boundary ambiguity)
for (int i = L/3; i < 2*L/3; i++) {
    if (tensor[i].ndim == 3 && tensor[i].shape[1] != tensor[i].shape[2]) {
        // Unambiguous: check if axis 1 or axis 2 is physical dim
        if (tensor[i].shape[2] == phys_dim && tensor[i].shape[1] != phys_dim) {
            needs_transpose = false;  // Already in quimb format
        }
        break;
    }
}

// Apply transpose if needed
if (needs_transpose) {
    for (auto& t : mps_tensors) {
        if (t.ndim == 3) {
            t = t.transpose(0, 2, 1);  // (chi_L, d, chi_R) -> (chi_L, chi_R, d)
        }
    }
}
```

**Validation**:
```cpp
// After loading
assert(mps_tensors.size() == L);
assert(mpo_tensors.size() == L);
assert(mps_tensors[0].shape[0] == 1);  // Left boundary
assert(mps_tensors[L-1].shape[2] == 1);  // Right boundary
```

---

## Code Cleanup Requirements

### Remove Dead Code
- [ ] Delete all `.backup*` files in `src/`
- [ ] Remove deprecated implementations:
  - `dmrg_working.cpp` (working prototype → merge into production)
  - `dmrg_gpu.cpp`, `dmrg_gpu_native.cpp` (keep only ONE canonical version)
  - `pdmrg_complete.cpp`, `pdmrg-opt_complete.cpp` (merge into `pdmrg_gpu.cpp` and `pdmrg_gpu_opt.cpp`)

### Consolidate Implementations

**Decision tree**:
- **Production PDMRG GPU**: `test_heisenberg_multistream.cpp` + `stream_coordinator.cpp/.h` + `stream_segment.cpp/.h`
- **Benchmark suite**: `pdmrg_benchmark_loaded.cpp` (uses `mps_mpo_loader.hpp`)
- **Everything else**: Archive to `pdmrg-gpu/archive/` or delete

**After cleanup, should have**:
```
src/
├── stream_coordinator.cpp/.h       # Multi-stream orchestration
├── stream_segment.cpp/.h           # Stream-local operations
├── boundary_merge_gpu.cpp/.h       # Boundary reconciliation
├── accurate_svd_gpu.cpp/.h         # Exact SVD
├── heff_optimized_gpu.cpp/.h       # H_eff contractions
├── heisenberg_mpo_real.cpp/.h      # MPO builders
├── josephson_mpo.cpp               # MPO builders
├── mps_mpo_loader.hpp              # Data loading
├── test_heisenberg_multistream.cpp # Main PDMRG test
├── test_boundary_merge.cpp         # Unit test
├── test_stream_coordinator.cpp     # Unit test
└── test_phase1.cpp                 # Unit test
```

**Total**: ~12 files (down from 50+)

### Resolve TODOs

**Process**:
1. Audit all 77 TODOs
2. Classify:
   - **Fix now**: Critical for correctness (e.g., "TODO: Use actual Lanczos instead of placeholder")
   - **Document**: Design decisions (e.g., "TODO: Consider block-Davidson for PDMRG-OPT")
   - **Delete**: Obsolete or already fixed
3. Fix or remove ALL TODOs in production code path
4. Move design TODOs to `FUTURE_WORK.md`

**Zero TODOs in production files**:
- `stream_coordinator.cpp/.h`
- `stream_segment.cpp/.h`
- `boundary_merge_gpu.cpp/.h`
- `accurate_svd_gpu.cpp/.h`
- `heff_optimized_gpu.cpp/.h`

---

## Testing & Validation Strategy

### Phase 1: Unit Tests

**Objective**: Verify individual components match CPU behavior.

#### Test 1.1: Exact SVD + V Computation
```bash
cd pdmrg-gpu/build
./test_phase1
```

**Pass criteria**:
- [ ] SVD: `||A - U @ diag(S) @ Vt|| < 1e-12`
- [ ] V computation: `V[i] = 1.0 / S[i]` for `S[i] > 1e-14`
- [ ] V computation: `V[i] = 0.0` for `S[i] <= 1e-14`

#### Test 1.2: Boundary Merge
```bash
./test_boundary_merge
```

**Pass criteria**:
- [ ] Merge energy converges (|dE| < 1e-10)
- [ ] Returns 5 values: `(A_left, A_right, V_new, energy, A_right_canonical)`
- [ ] A_left is left-canonical: `||A_left† @ A_left - I|| < 1e-12`
- [ ] A_right_canonical is right-canonical: `||A_right_canonical @ A_right_canonical† - I|| < 1e-12`

#### Test 1.3: Environment Updates
```bash
./test_stream_segment
```

**Pass criteria**:
- [ ] L_env norm: `0.1 < ||L_env|| < 100.0` (should be O(1))
- [ ] R_env norm: `0.1 < ||R_env|| < 100.0`
- [ ] No NaN or Inf values

---

### Phase 2: Heisenberg Accuracy Test

**Objective**: Match exact diagonalization to < 1e-10.

#### Test 2.1: Single-Stream (No Parallelism)
```bash
./test_heisenberg_multistream 8 32 1 30
```

**Expected output**:
```
Final Energy: -3.374931816815
|Error| < 1e-10  ✓ PASS
```

**Pass criteria**:
- [ ] `|E_GPU - E_exact| < 1e-10`
- [ ] Energy is negative (antiferromagnetic)
- [ ] No "eigenvalue mismatch" warnings

**If failed**: Debug H_eff, eigensolver, environment updates (no parallelism to complicate)

#### Test 2.2: Multi-Stream Consistency
```bash
for n in 1 2 4 8; do
    ./test_heisenberg_multistream 8 32 $n 30
done
```

**Pass criteria**:
- [ ] ALL stream counts: `|E_GPU - E_exact| < 1e-10`
- [ ] Energy variation: `|E(n=1) - E(n=2)| < 1e-11`
- [ ] Energy variation: `|E(n=1) - E(n=4)| < 1e-11`
- [ ] Energy variation: `|E(n=1) - E(n=8)| < 1e-11`

**If failed**: Debug boundary merge, V computation, stream coordination

---

### Phase 3: Cross-Validation with CPU

**Objective**: GPU and CPU produce identical results on same inputs.

#### Test 3.1: Load CPU Warmup MPS
```bash
# CPU: Save warmup MPS
cd ../benchmarks
python -c "
import numpy as np
from benchmark_data_loader import build_heisenberg_model
from pdmrg.dmrg import serial_warmup

manifest = build_heisenberg_model(L=12, bond_dim=50)
mps_warmup = serial_warmup(
    manifest['mpo'], L=12, bond_dim_warmup=50,
    n_warmup_sweeps=5, dtype=np.float64
)
np.savez('warmup_L12_chi50.npz', *[a.data for a in mps_warmup])
"

# GPU: Load and compute energy
cd ../pdmrg-gpu/build
./test_mps_energy_only ../benchmarks/warmup_L12_chi50.npz
```

**Pass criteria**:
- [ ] GPU loads MPS without errors
- [ ] `|E_GPU - E_CPU| < 1e-13` (near machine precision)

#### Test 3.2: Full PDMRG Benchmark
```bash
# Run CPU benchmark
cd ../benchmarks
mpirun -np 2 python run_focused_bench.py --model heisenberg --case basic

# Extract CPU energy
E_CPU=$(jq '.heisenberg.basic.pdmrg_np2.energy' benchmark_results.json)

# Run GPU benchmark (2 streams)
cd ../pdmrg-gpu/build
./pdmrg_benchmark_loaded \
    --mpo ../benchmarks/heisenberg_L12_chi50.npz \
    --streams 2 \
    --max_sweeps 15 \
    --tol 1e-10

# Compare
echo "CPU: $E_CPU"
echo "GPU: $E_GPU"
echo "Difference: $(python -c "print(abs($E_CPU - $E_GPU))")"
```

**Pass criteria**:
- [ ] `|E_CPU - E_GPU| < 1e-10`
- [ ] Both converge in similar number of sweeps (±2 sweeps)

---

### Phase 4: Larger Systems

#### Test 4.1: Heisenberg Challenge Benchmarks
```bash
# L=64, chi=20 (from benchmarks/benchmark_data_loader.py)
./test_heisenberg_multistream 64 20 4 20

# L=96, chi=20
./test_heisenberg_multistream 96 20 4 20

# L=128, chi=20
./test_heisenberg_multistream 128 20 8 20
```

**Pass criteria**:
- [ ] Converges to stable energy (|dE| < 1e-10)
- [ ] No crashes, no NaN/Inf
- [ ] Speedup scales reasonably with streams

#### Test 4.2: Josephson Junction
```bash
# Run CPU benchmark first for reference
cd ../benchmarks
mpirun -np 2 python run_focused_bench.py --model josephson --case challenge

E_CPU=$(jq '.josephson.challenge.pdmrg_np2.energy' benchmark_results.json)

# Run GPU
cd ../pdmrg-gpu/build
./pdmrg_benchmark_loaded \
    --mpo ../benchmarks/josephson_L100_chi50.npz \
    --streams 2 \
    --max_sweeps 20

# Compare
```

**Pass criteria**:
- [ ] `|E_CPU - E_GPU| < 1e-10`

---

## Implementation Checklist

### Step 1: Audit Current GPU Code
- [ ] Read `stream_segment.cpp` and compare to `pdmrg/dmrg.py`
- [ ] Read `boundary_merge_gpu.cpp` and compare to `pdmrg/parallel/merge.py`
- [ ] Read `heff_optimized_gpu.cpp` and compare to `pdmrg/numerics/eigensolver.py` (`apply_heff`)
- [ ] Identify ALL differences (document in `GPU_CPU_DIFFERENCES.md`)

### Step 2: Fix Canonicalization (Highest Priority)
- [ ] Implement left-canonicalization before building L_envs (QR sweeps)
- [ ] Implement right-canonicalization before building R_envs (LQ sweeps)
- [ ] Modify `boundary_merge_gpu.cpp` to return `A_right_canonical` (6th return value)
- [ ] Update stream coordinator to use canonical tensor for R_env updates
- [ ] Test: `./test_boundary_merge` (unit test)

### Step 3: Fix V Computation
- [ ] Verify V = 1/S computation uses epsilon = 1e-14
- [ ] Verify V is applied at EVERY boundary merge
- [ ] Check V is stored as 1D vector (not matrix)
- [ ] Test: Print V values, ensure they're reasonable (not all 0 or 1)

### Step 4: Fix H_eff Application
- [ ] Verify contraction order matches CPU exactly (4-step sequence)
- [ ] Check hipTensor axis indices (common source of errors)
- [ ] Add validation: `||H_eff(theta)|| ~ ||theta||` (should be similar magnitude)
- [ ] Test: Compare GPU H_eff output to CPU on same input tensors

### Step 5: Fix Eigensolver
- [ ] Verify single-vector Lanczos (not block)
- [ ] Verify tridiagonalization: α = ⟨v|H|v⟩, β = ||H|v⟩ - α|v⟩||
- [ ] Verify LAPACK dstev CPU fallback is used
- [ ] Add validation: eigenvalue must be real, negative, reasonable magnitude
- [ ] Test: `./test_boundary_merge` should show correct energies

### Step 6: Fix MPS/MPO Loading
- [ ] Implement auto-detection of format (internal vs quimb)
- [ ] Handle 2D vs 3D boundary tensors
- [ ] Validate loaded tensors (shapes, dtypes, no NaN)
- [ ] Test: Load CPU-saved MPS, compute energy, compare to CPU

### Step 7: Clean Up Code
- [ ] Delete backup files
- [ ] Archive/delete deprecated implementations
- [ ] Fix or remove all 77 TODOs
- [ ] Add comments explaining critical sections

### Step 8: Validate End-to-End
- [ ] Run Phase 1 unit tests (all pass)
- [ ] Run Phase 2 Heisenberg accuracy tests (< 1e-10 error)
- [ ] Run Phase 3 cross-validation (CPU-GPU parity)
- [ ] Run Phase 4 larger systems (no crashes, stable convergence)

### Step 9: Performance Benchmarking
- [ ] Measure GPU vs CPU wall time (AFTER correctness validated)
- [ ] Measure multi-stream scalability (efficiency ≥ 60% @ 8 streams)
- [ ] Profile: Identify bottlenecks (eigensolver? H_eff? SVD?)

### Step 10: Documentation
- [ ] Update README.md with current accuracy status
- [ ] Create VALIDATION_REPORT.md showing < 1e-10 accuracy on all tests
- [ ] Document any remaining differences vs CPU (if any)
- [ ] Archive PHASE2_COMPLETE.md (outdated, shows 12.5% error)

---

## Critical Success Criteria

### Must Pass (BLOCKING)
1. ✅ **Heisenberg L=8, chi=32, 1 stream**: `|E_GPU - E_exact| < 1e-10`
2. ✅ **Multi-stream consistency**: `|E(n=1) - E(n=8)| < 1e-11`
3. ✅ **CPU-GPU parity**: `|E_CPU - E_GPU| < 1e-10` on same inputs
4. ✅ **Zero TODOs** in production files
5. ✅ **Zero backup files** in `src/`

### Should Pass (HIGH PRIORITY)
6. ✅ **Heisenberg L=64, chi=20**: Converges without crashes
7. ✅ **Josephson benchmark**: Matches CPU energy
8. ✅ **MPS loading**: Handles CPU-saved files correctly

### Nice to Have (MEDIUM PRIORITY)
9. 🎯 **Performance**: GPU ≥ 2x faster than CPU (single stream vs single MPI rank)
10. 🎯 **Scalability**: 8 streams ≥ 60% parallel efficiency

---

## Common Pitfalls

### 1. Off-by-One Axis Indices
**Problem**: GPU tensor libraries use different axis conventions.
**Solution**: Print tensor shapes at every contraction, verify against CPU.

### 2. Non-Canonical Environments
**Problem**: Building L_env from right-canonical tensors (or vice versa).
**Solution**: Follow CPU fix in commit 6dbeabf EXACTLY.

### 3. V = 1 Approximation
**Problem**: Setting V = ones(chi) instead of V = 1/S.
**Solution**: Verify V computation, print V values to check.

### 4. Mixed vs Canonical Tensors
**Problem**: Using A_right = S @ Vh for R_env update.
**Solution**: Return and use A_right_canonical = Vh separately.

### 5. rocSOLVER Bug Workaround Incomplete
**Problem**: Trusting rocSOLVER dsteqr output (returns [-1, 0, 1]).
**Solution**: Use CPU LAPACK dstev fallback (already correct in GPU code).

### 6. Transposed MPS Tensors
**Problem**: Loading (chi_L, chi_R, d) when expecting (chi_L, d, chi_R).
**Solution**: Use auto-detection from commit 6dbeabf.

---

## Reference: CPU Commit 6dbeabf Summary

**Commit message**: "Fix PDMRG boundary environment norm matrices via canonicalization"

**Key changes**:
1. **build_local_environments()**: Left/right-canonicalize global MPS before building L_envs/R_envs
2. **boundary_merge()**: Use canonical tensor (Vh) for R_env updates instead of S*Vh
3. **merge_boundary_tensors()**: Return right-canonical tensor separately
4. **Warmup tuning**: Reduced sweeps, lighter tolerance
5. **MPS loader**: Auto-detect format (internal vs quimb)

**Files modified**:
- `pdmrg/pdmrg/dmrg.py` (120 insertions, environment building)
- `pdmrg/pdmrg/parallel/merge.py` (15 insertions, canonical tensor return)
- `pdmrg/pdmrg/numerics/eigensolver.py` (8 insertions, fallback tuning)
- `benchmarks/benchmark_data_loader.py` (34 insertions, format detection)

**GPU must implement ALL of these fixes**.

---

## Questions to Resolve During Implementation

1. **Memory layout**: Does hipTensor expect row-major or column-major? (Check with small test)
2. **Stream synchronization**: Do we need explicit `hipStreamSynchronize()` between phases? (Profile)
3. **Shared memory**: Can streams directly access each other's tensors, or need explicit copy? (Check docs)
4. **LAPACK fallback**: Measure CPU fallback overhead (should be < 1% of iteration time)
5. **Optimal stream count**: What's the best stream count for MI300X? (Benchmark)

---

## Success Definition

**Before**: 12.5% energy error, "production ready" (FALSE)
**After**: < 1e-10 energy error on ALL tests, PROVEN production ready

**Deliverable**: `VALIDATION_REPORT.md` showing:
- ✅ All unit tests pass
- ✅ Heisenberg L=8: |Error| = 3.2e-11 (example)
- ✅ Multi-stream consistency: |E(n=1) - E(n=8)| = 5.7e-12
- ✅ CPU-GPU parity: |E_CPU - E_GPU| = 1.4e-11
- ✅ Heisenberg L=64, L=96, L=128: All converge, |Error| < 1e-10
- ✅ Josephson junction: |E_CPU - E_GPU| < 1e-10
- ✅ Zero TODOs in production code
- ✅ Performance: GPU 3.2x faster than CPU (example)

**Timeline estimate**: 3-5 days (assuming full-time work)

---

## Final Notes

- **Priority**: Correctness first, performance second
- **Testing**: Test after EVERY change (unit tests are fast)
- **CPU reference**: When in doubt, check CPU code in `pdmrg/`
- **Validation**: < 1e-10 accuracy is NON-NEGOTIABLE for scientific use
- **Documentation**: Update docs to reflect actual accuracy, remove "production ready" claims until validated

Good luck! 🚀
