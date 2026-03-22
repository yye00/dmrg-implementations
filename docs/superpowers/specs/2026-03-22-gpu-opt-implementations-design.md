# GPU-Optimized DMRG Implementations

## Goal

Create GPU-optimized variants of dmrg-gpu (single-site) and dmrg2-gpu (two-site) by porting Newton-Schulz polar decomposition and Block-Davidson eigensolver from pdmrg2-gpu. Rename pdmrg2-gpu to pdmrg-gpu-opt for naming consistency.

## Motivation

Profiling shows SVD dominates wall time at high bond dimension:
- dmrg-gpu L=64 chi=256: SVD = 97% of wall time
- dmrg2-gpu L=64 chi=256: SVD = 99% of wall time

Newton-Schulz replaces sequential gesvd with ~12-16 GPU GEMMs. Block-Davidson replaces BLAS-2-heavy Lanczos with BLAS-3-heavy block subspace iteration. Both are already proven in pdmrg2-gpu.

## Architecture

The project has 4 GPU DMRG implementations:
- `dmrg-gpu/` — single-site baseline (Lanczos + CPU SVD)
- `dmrg2-gpu/` — two-site baseline (Lanczos + CPU SVD)
- `pdmrg-gpu/` — parallel single-site baseline (Lanczos + GPU SVD, recently fixed)
- `pdmrg2-gpu/` — parallel two-site with Newton-Schulz + Block-Davidson

After this work:
- `dmrg-gpu/` — single-site baseline (unchanged)
- `dmrg-gpu-opt/` — single-site with Newton-Schulz + Block-Davidson
- `dmrg2-gpu/` — two-site baseline (unchanged)
- `dmrg2-gpu-opt/` — two-site with Newton-Schulz + Block-Davidson
- `pdmrg-gpu/` — parallel single-site baseline (unchanged)
- `pdmrg-gpu-opt/` — renamed from pdmrg2-gpu (parallel two-site, already has NS + Davidson)

## Deliverables

### 1. dmrg-gpu-opt (new)

Copy `dmrg-gpu/` → `dmrg-gpu-opt/`. Rename class `DMRGGPU` → `DMRGGPUOpt`.

**Files:**
- `dmrg_gpu_opt.h` — class declaration with new workspace members
- `dmrg_gpu_opt_impl.h` — implementation with NS + Davidson
- `dmrg_gpu_opt.cpp` — explicit template instantiations
- `scalar_traits.h` — extended with NS/scaling GPU kernels
- `test_dmrg_gpu_opt.cpp` — test driver
- `CMakeLists.txt` — build config

**New algorithms added:**
- `newton_schulz_left(d_A, m, n, d_U, d_P, tol, max_iter)` — polar decomposition via iterative GEMM
- `ns_svd_and_update_mps(site, d_theta, direction)` — replaces `svd_and_update_mps`; uses NS polar + eigendecomp of P^H P to get singular values/vectors; falls back to CPU SVD for k ≤ 4 or convergence failure
- `block_davidson_eigensolver(site, d_theta)` — replaces `lanczos_eigensolver`; block size b=4, max subspace 32; falls back to Lanczos for dim ≤ 2*b

**Single-site adaptation notes:**
- Direction 'R': theta is (chi_L*d, chi_R) — tall matrix, NS-left works directly
- Direction 'L': theta is (chi_L, d*chi_R) — wide matrix when chi_L < d*chi_R. Fall back to CPU SVD for the wide case (matching pdmrg2-gpu behavior which also falls back for wide matrices in ns_split)
- After NS factorization: U → MPS[site], S*Vh absorbed into adjacent MPS tensor via GEMM (same as current SVD path, just different factorization method)
- Block-Davidson calls single-site `apply_heff()` not `apply_heff_two_site()`

**Workspace sizing for single-site:**
- NS buffers: `d_ns_U`, `d_ns_U_new` are (max_m, max_n) where max_m = chi_max*d, max_n = chi_max; `d_ns_gram` is (max_n, max_n) = (chi_max, chi_max); `d_ns_P` is (max_n, max_n)
- Davidson buffers: `d_dav_V`, `d_dav_AV` are (theta_size_max, max_sub) where theta_size_max = chi_max * d * chi_max, max_sub = min(32, theta_size_max); `d_dav_work`, `d_dav_work2` are (theta_size_max, b) where b=4

**New GPU workspace:**
- Newton-Schulz: `d_ns_U`, `d_ns_U_new`, `d_ns_gram`, `d_ns_P` (GPU); `h_ns_PtP`, `h_ns_eigvals`, `h_ns_syev_work` (host)
- Block-Davidson: `d_dav_V`, `d_dav_AV`, `d_dav_work`, `d_dav_work2` (GPU); `h_dav_H_proj`, `h_dav_eigvals`, `h_dav_eigvecs` (host)

**scalar_traits.h additions** (ported from pdmrg2-gpu's scalar_traits.h):

GPU kernels:
- `scaled_identity_minus_double` / `scaled_identity_minus_complex` — computes `A[i,j] = alpha*I[i,j] - A[i,j]` in-place
- `scale_columns_by_real_kernel` / `scale_rows_by_real_kernel` — column/row scaling by real diagonal
- Wrapper functions: `launch_scaled_identity_minus()`, `scale_columns_by_real()`, `scale_rows_by_real()`

LAPACK additions (not in baseline scalar_traits.h):
- `extern "C"` declarations for `dsyev_` and `zheev_`
- `ScalarTraits<double>::lapack_syev()` and `ScalarTraits<hipDoubleComplex>::lapack_syev()` static methods
- `syev_rwork_size()` for complex workspace sizing

Note: the baseline scalar_traits.h already has `dsyev_`/`zheev_` declarations and `lapack_syev` — these only need to be added if missing. The dead-code kernels (`compute_3I_minus_A`, templated `scaled_identity_minus_kernel`) from pdmrg2-gpu should NOT be ported.

**Removed from baseline:**
- `--cpu-svd`, `--gpu-svd`, `--rsvd` command-line flags and associated code paths
- rSVD workspace and implementation (NS replaces it)

### 2. dmrg2-gpu-opt (new)

Copy `dmrg2-gpu/` → `dmrg2-gpu-opt/`. Rename class `DMRG2GPU` → `DMRG2GPUOpt`.

**Files:**
- `dmrg2_gpu_opt.h` — class declaration
- `dmrg2_gpu_opt_impl.h` — implementation
- `dmrg2_gpu_opt.cpp` — explicit instantiations
- `scalar_traits.h` — extended with NS/scaling GPU kernels
- `test_dmrg2_gpu_opt.cpp` — test driver
- `CMakeLists.txt` — build config

**New algorithms added:**
- `newton_schulz_left(d_A, m, n, d_U, d_P, tol, max_iter)` — same as dmrg-gpu-opt
- `ns_split(site, d_theta, direction)` — nearly direct port from pdmrg2-gpu; factors (chi_L*d, d*chi_R) two-site theta; removes `si` stream parameter
- `block_davidson_eigensolver(site, d_theta, theta_size)` — direct port from pdmrg2-gpu; calls `apply_heff_two_site()`; removes `si` stream parameter

**Two-site notes:**
- Geometry matches pdmrg2-gpu exactly (both are two-site)
- The port removes the `si` stream parameter and flattens `StreamWorkspace` members into class members directly (matching dmrg2-gpu's flat member style). E.g. `ws.d_ns_U` → `d_ns_U_`, `ws.d_dav_V` → `d_dav_V_`
- Uses single `stream_` and `rocblas_h_` instead of per-stream vectors

**Workspace sizing for two-site:**
- NS buffers: `d_ns_U`, `d_ns_U_new` are (max_m, max_n) where max_m = max_n = chi_max*d; `d_ns_gram`, `d_ns_P` are (chi_max*d, chi_max*d)
- Davidson buffers: `d_dav_V`, `d_dav_AV` are (theta_size_max, max_sub) where theta_size_max = chi_max * d² * chi_max, max_sub = min(32, theta_size_max)

**Same scalar_traits.h additions as dmrg-gpu-opt.**

**Removed from baseline:**
- Same removals as dmrg-gpu-opt (SVD toggle flags, rSVD)

### 3. pdmrg-gpu-opt (rename of pdmrg2-gpu)

Rename directory and all identifiers. No algorithmic changes.

**Renames:**
- Directory: `pdmrg2-gpu/` → `pdmrg-gpu-opt/`
- Header: `pdmrg2_gpu.h` → `pdmrg_gpu_opt.h`
- Implementation: `pdmrg2_gpu_impl.h` → `pdmrg_gpu_opt_impl.h`
- Instantiation: `pdmrg2_gpu.cpp` → `pdmrg_gpu_opt.cpp`
- Test: `test_pdmrg2_gpu.cpp` → `test_pdmrg_gpu_opt.cpp`
- Class: `PDMRG2GPU` → `PDMRGGPUOpt`
- CMake target: `pdmrg2_gpu` → `pdmrg_gpu_opt`

### 4. Benchmark data update

Rename `pdmrg2-gpu` → `pdmrg-gpu-opt` in all benchmark result files:
- `benchmarks/paper_results/gpu_4way_results.csv` — impl column
- `benchmarks/paper_results/summary.csv` — impl column
- `benchmarks/paper_results/results.json` — impl field

## Internal fallback behavior

Newton-Schulz falls back to CPU SVD when:
- Matrix too small (k ≤ 4 or m < 2 or n < 2)
- Orthogonality check fails (||U^H U - I||_F ≥ 1e-10 after max iterations)

Block-Davidson falls back to Lanczos when:
- Subspace dimension ≤ 2*b (too small for block method)
- Projected eigendecomp fails (dsyev returns info ≠ 0)

These are algorithmic safety nets, not user-facing options.

## Profiling output

Rename profiling variables to reflect actual algorithms:
- `prof_svd_ms` → `prof_ns_ms` (Newton-Schulz time)
- `prof_lanczos_ms` → `prof_davidson_ms` (Block-Davidson time)
- Print NS iteration count and Davidson iteration count per sweep

## Implementation order

1. **pdmrg-gpu-opt** (rename) — simplest, validates rename process
2. **Benchmark data update** — rename pdmrg2-gpu in CSVs
3. **dmrg2-gpu-opt** (new) — mechanical port, geometry matches pdmrg2-gpu
4. **dmrg-gpu-opt** (new) — requires single-site NS adaptation

## Test plan

Each opt implementation must pass the same correctness tests as its baseline:
- Heisenberg L=4, L=8: energy error < 1e-10 vs exact
- Josephson L=4, L=6 (complex): energy error < 1e-10 vs exact
- TFIM L=8 (J=1.0, h=1.0 critical): energy error < 1e-10

Additional: verify NS and Davidson are actually being used (not silently falling back) by checking iteration counts in output.

## Files to copy/exclude

When copying dmrg-gpu → dmrg-gpu-opt:
- Copy: `dmrg_gpu.h`, `dmrg_gpu_impl.h`, `dmrg_gpu.cpp`, `scalar_traits.h`, `test_dmrg_gpu.cpp`, `CMakeLists.txt`
- Exclude: `accurate_svd_gpu.h`, `accurate_svd_gpu.cpp`, `test_svd_bug.cpp`, `test_svd_fix.cpp` (not used by main DMRG)

When copying dmrg2-gpu → dmrg2-gpu-opt:
- Copy: `dmrg2_gpu.h`, `dmrg2_gpu_impl.h`, `dmrg2_gpu.cpp`, `scalar_traits.h`, `test_dmrg2_gpu.cpp`, `CMakeLists.txt`
- Exclude: `test_hipgraph.cpp`, `test_hipgraph2.cpp` (experimental, not used)

## What stays unchanged

- `dmrg-gpu/`, `dmrg2-gpu/`, `pdmrg-gpu/` — untouched baselines
- `pdmrg2-gpu/` — deleted after pdmrg-gpu-opt is created and verified
- Benchmark scripts — not modified (new benchmarks are a separate task)
