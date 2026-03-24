# GPU-Optimized DMRG Implementations Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create GPU-optimized variants of dmrg-gpu and dmrg2-gpu with Newton-Schulz + Block-Davidson, and rename pdmrg-gpu-opt to pdmrg-gpu-opt.

**Architecture:** Copy each baseline implementation into a new `-opt` directory, replace SVD with Newton-Schulz polar decomposition + eigendecomp, replace Lanczos with Block-Davidson eigensolver, remove unused SVD toggle flags. Source of truth for the algorithms is pdmrg-gpu-opt.

**Tech Stack:** C++17, HIP, rocBLAS, rocSOLVER, LAPACK (OpenBLAS 0.3.28+), AMD MI300X (gfx942)

**Spec:** `docs/superpowers/specs/2026-03-22-gpu-opt-implementations-design.md`

**Remote build/test:** `ssh hotaisle@23.183.40.75` (passwordless), GPU code in `~/`

---

## File Map

### Task 1: pdmrg-gpu-opt (rename)
- Rename: `pdmrg-gpu-opt/` → `pdmrg-gpu-opt/`
- Rename: all `pdmrg_gpu_opt*` files → `pdmrg_gpu_opt*`
- Modify: all source files (class name `PDMRGGPUOpt` → `PDMRGGPUOpt`)
- Modify: `CMakeLists.txt` (target name)

### Task 2: Benchmark data rename
- Modify: `benchmarks/paper_results/gpu_4way_results.csv`
- Modify: `benchmarks/paper_results/summary.csv`
- Modify: `benchmarks/paper_results/results.json`

### Task 3: dmrg2-gpu-opt (new, two-site + NS + Davidson)
- Create: `dmrg2-gpu-opt/src/dmrg2_gpu_opt.h`
- Create: `dmrg2-gpu-opt/src/dmrg2_gpu_opt_impl.h`
- Create: `dmrg2-gpu-opt/src/dmrg2_gpu_opt.cpp`
- Create: `dmrg2-gpu-opt/src/scalar_traits.h`
- Create: `dmrg2-gpu-opt/src/test_dmrg2_gpu_opt.cpp`
- Create: `dmrg2-gpu-opt/CMakeLists.txt`

### Task 4: dmrg-gpu-opt (new, single-site + NS + Davidson)
- Create: `dmrg-gpu-opt/src/dmrg_gpu_opt.h`
- Create: `dmrg-gpu-opt/src/dmrg_gpu_opt_impl.h`
- Create: `dmrg-gpu-opt/src/dmrg_gpu_opt.cpp`
- Create: `dmrg-gpu-opt/src/scalar_traits.h`
- Create: `dmrg-gpu-opt/src/test_dmrg_gpu_opt.cpp`
- Create: `dmrg-gpu-opt/CMakeLists.txt`

### Task 5: Remote build and test all
- Build all three opt implementations on MI300X
- Run correctness tests
- Compare timing vs baselines

---

## Task 1: Rename pdmrg-gpu-opt → pdmrg-gpu-opt

**Files:**
- Rename: `pdmrg-gpu-opt/` → `pdmrg-gpu-opt/`
- Modify: all 5 source files + CMakeLists.txt

- [ ] **Step 1: Copy the directory with new name**

```bash
cp -r pdmrg-gpu-opt pdmrg-gpu-opt
```

- [ ] **Step 2: Rename the source files**

```bash
cd pdmrg-gpu-opt/src
mv pdmrg_gpu_opt.h pdmrg_gpu_opt.h
mv pdmrg_gpu_opt_impl.h pdmrg_gpu_opt_impl.h
mv pdmrg_gpu_opt.cpp pdmrg_gpu_opt.cpp
mv test_pdmrg_gpu_opt.cpp test_pdmrg_gpu_opt.cpp
```

- [ ] **Step 3: Rename class and identifiers in all source files**

In every `.h`, `.cpp` file in `pdmrg-gpu-opt/src/`:
- Replace `PDMRGGPUOpt` → `PDMRGGPUOpt` (class name)
- Replace `PDMRG-OPT_GPU_H` → `PDMRG_GPU_OPT_H` (include guard)
- Replace `pdmrg_gpu_opt.h` → `pdmrg_gpu_opt.h` (include directive)
- Replace `pdmrg_gpu_opt_impl.h` → `pdmrg_gpu_opt_impl.h` (include directive)
- Replace `PDMRG-OPT_GPU_IMPL_H` → `PDMRG_GPU_OPT_IMPL_H` (include guard)

In `pdmrg_gpu_opt.cpp`:
```cpp
#include "pdmrg_gpu_opt.h"

template class PDMRGGPUOpt<double>;
template class PDMRGGPUOpt<hipDoubleComplex>;
```

- [ ] **Step 4: Update CMakeLists.txt**

Change the target name and source files:
- `pdmrg_gpu_opt` → `pdmrg_gpu_opt` (target name)
- `src/test_pdmrg_gpu_opt.cpp` → `src/test_pdmrg_gpu_opt.cpp`
- `src/pdmrg_gpu_opt.cpp` → `src/pdmrg_gpu_opt.cpp`
- Update project name and status messages accordingly

- [ ] **Step 5: Update test driver output strings**

In `test_pdmrg_gpu_opt.cpp`, update the banner strings:
- `"PDMRG-OPT-GPU"` → `"PDMRG-GPU-OPT"` in all printf/cout statements
- `"pdmrg-gpu-opt"` → `"pdmrg-gpu-opt"` in result output

- [ ] **Step 6: Delete old pdmrg-gpu-opt directory**

```bash
rm -rf pdmrg-gpu-opt
```

- [ ] **Step 7: Verify build locally (syntax check)**

Read through the renamed files and verify all identifiers are consistent.

- [ ] **Step 8: Commit**

```bash
git add pdmrg-gpu-opt/ && git rm -r pdmrg-gpu-opt/
git commit -m "refactor: rename pdmrg-gpu-opt to pdmrg-gpu-opt"
```

---

## Task 2: Rename pdmrg-gpu-opt in benchmark data

**Files:**
- Modify: `benchmarks/paper_results/gpu_4way_results.csv`
- Modify: `benchmarks/paper_results/summary.csv`
- Modify: `benchmarks/paper_results/results.json`

- [ ] **Step 1: Replace in CSV files**

In `gpu_4way_results.csv` and `summary.csv`, replace all occurrences of `pdmrg-gpu-opt` with `pdmrg-gpu-opt` in the `impl` column.

- [ ] **Step 2: Replace in JSON file**

In `results.json`, replace all `"pdmrg-gpu-opt"` with `"pdmrg-gpu-opt"`.

- [ ] **Step 3: Verify counts**

Count occurrences before and after to ensure all replacements happened:
- `gpu_4way_results.csv`: 88 rows of pdmrg-gpu-opt → 88 rows of pdmrg-gpu-opt
- `summary.csv`: 57 rows
- `results.json`: 57 entries

- [ ] **Step 4: Commit**

```bash
git add benchmarks/paper_results/
git commit -m "refactor: rename pdmrg-gpu-opt to pdmrg-gpu-opt in benchmark data"
```

---

## Task 3: Create dmrg2-gpu-opt (two-site + NS + Davidson)

This is the largest task. The two-site geometry matches pdmrg-gpu-opt exactly, so Newton-Schulz and Block-Davidson port almost directly — only the multi-stream infrastructure is removed.

**Files:**
- Create: `dmrg2-gpu-opt/` directory with 6 files
- Reference: `dmrg2-gpu/src/` (baseline to copy) and `pdmrg-gpu-opt/src/` (algorithms to port, now at `pdmrg-gpu-opt/src/`)

### Step-by-step:

- [ ] **Step 1: Copy baseline dmrg2-gpu**

```bash
cp -r dmrg2-gpu dmrg2-gpu-opt
```

Remove files we don't need:
```bash
rm -f dmrg2-gpu-opt/src/test_hipgraph.cpp dmrg2-gpu-opt/src/test_hipgraph2.cpp
rm -rf dmrg2-gpu-opt/build
```

- [ ] **Step 2: Create scalar_traits.h with NS + Davidson kernels**

Start from `dmrg2-gpu/src/scalar_traits.h` (290 lines). Add from `pdmrg-gpu-opt/src/scalar_traits.h`:

**A) Add LAPACK extern declarations** (after the existing zgesvd_ declaration, before `template<typename T> struct ScalarTraits;`):

```cpp
extern "C" void dsyev_(const char* jobz, const char* uplo,
                       const int* n, double* a, const int* lda,
                       double* w, double* work, const int* lwork, int* info);

extern "C" void zheev_(const char* jobz, const char* uplo,
                       const int* n, hipDoubleComplex* a, const int* lda,
                       double* w, hipDoubleComplex* work, const int* lwork,
                       double* rwork, int* info);
```

**B) Add `lapack_syev` to `ScalarTraits<double>`** (after the existing `lapack_gesvd` method):

```cpp
    static void lapack_syev(const char* jobz, const char* uplo, const int* n,
            double* a, const int* lda, double* w,
            double* work, const int* lwork, double* /*rwork*/, int* info) {
        dsyev_(jobz, uplo, n, a, lda, w, work, lwork, info);
    }
    static int syev_rwork_size(int /*n*/) { return 0; }
```

**C) Add `lapack_syev` to `ScalarTraits<hipDoubleComplex>`** (after the existing `lapack_gesvd` method):

```cpp
    static void lapack_syev(const char* jobz, const char* uplo, const int* n,
            hipDoubleComplex* a, const int* lda, double* w,
            hipDoubleComplex* work, const int* lwork, double* rwork, int* info) {
        zheev_(jobz, uplo, n, a, lda, w, work, lwork, rwork, info);
    }
    static int syev_rwork_size(int n) { return std::max(1, 3*n - 2); }
```

**D) Add GPU kernels** (before `#endif`):

Copy from `pdmrg-gpu-opt/src/scalar_traits.h` lines 502-610:
- `scaled_identity_minus_double` kernel
- `scaled_identity_minus_complex` kernel
- `launch_scaled_identity_minus` overloads (double and complex)
- `scale_columns_by_real_kernel` template
- `scale_rows_by_real_kernel` template
- `scale_columns_by_real` helper
- `scale_rows_by_real` helper

Do NOT copy: `scaled_identity_minus_kernel` (dead code), `compute_3I_minus_A` (dead code), Lanczos device-pointer-mode kernels, batched GEMM pointer setup kernels (not needed — dmrg2-gpu uses host-side pointer setup).

- [ ] **Step 3: Create dmrg2_gpu_opt.h (class declaration)**

Start from `dmrg2-gpu/src/dmrg2_gpu.h`. Make these changes:

**A) Rename:**
- Include guard: `DMRG2_GPU_H` → `DMRG2_GPU_OPT_H`
- Class: `DMRG2GPU` → `DMRG2GPUOpt`
- Include: `dmrg2_gpu_impl.h` → `dmrg2_gpu_opt_impl.h`

**B) Remove SVD toggle members:**
- Remove: `void set_cpu_svd(bool use_cpu)`, `void set_rsvd(bool use_rsvd)`
- Remove: `bool use_cpu_svd_`, `bool use_rsvd_`, `int rsvd_oversampling_`
- Remove: all `d_rsvd_*` pointers and `h_rsvd_*` vectors
- Remove: `d_svd_E_`, `d_svd_info_`, `d_svd_work_` (GPU SVD workspace — not needed, we use NS)
- Keep: `d_svd_A_`, `d_svd_U_`, `d_svd_S_`, `d_svd_Vh_` (reused by NS as scratch)
- Keep: `h_svd_A_`, `h_svd_U_`, `h_svd_Vh_`, `h_svd_work_`, `h_svd_S_`, `h_svd_rwork_`, `h_svd_tmp_` (CPU SVD fallback)

**C) Add Newton-Schulz workspace members:**

```cpp
    // Newton-Schulz workspace
    Scalar* d_ns_U_;        // (chi_max*d, chi_max*d) NS iterate
    Scalar* d_ns_U_new_;    // (chi_max*d, chi_max*d) NS next iterate
    Scalar* d_ns_gram_;     // (chi_max*d, chi_max*d) U^H U
    Scalar* d_ns_P_;        // (chi_max*d, chi_max*d) PSD factor

    // NS eigendecomp host workspace
    std::vector<Scalar> h_ns_PtP_;
    std::vector<RealType> h_ns_eigvals_;
    std::vector<Scalar> h_ns_syev_work_;
    std::vector<RealType> h_ns_syev_rwork_;
```

**D) Add Block-Davidson workspace members:**

```cpp
    // Block-Davidson workspace
    int davidson_b_;         // block size (4)
    int davidson_max_sub_;   // max subspace size (32)

    Scalar* d_dav_V_;        // (theta_size_max, max_sub) basis
    Scalar* d_dav_AV_;       // (theta_size_max, max_sub) H-images
    Scalar* d_dav_work_;     // (theta_size_max, davidson_b) scratch
    Scalar* d_dav_work2_;    // (theta_size_max, davidson_b) scratch2

    // Davidson host workspace
    std::vector<Scalar> h_dav_H_proj_;
    std::vector<RealType> h_dav_eigvals_;
    std::vector<Scalar> h_dav_eigvecs_;
    std::vector<RealType> h_dav_syev_work_;
```

**E) Add new method declarations:**

```cpp
    // Newton-Schulz polar decomposition
    void newton_schulz_left(Scalar* d_A, int m, int n,
                            Scalar* d_U, Scalar* d_P,
                            double tol, int* out_iters);
    void ns_split(int site, Scalar* d_theta, char direction);

    // Block-Davidson eigensolver (replaces Lanczos)
    double block_davidson_eigensolver(int site, Scalar* d_theta, int theta_size);
```

**F) Keep but make private the SVD fallback:**

```cpp
    void svd_split_fallback(int site, Scalar* d_theta, char direction);
```

Remove `rsvd_split` declaration entirely.

- [ ] **Step 4: Create dmrg2_gpu_opt_impl.h**

Start from `dmrg2-gpu/src/dmrg2_gpu_impl.h`. The key changes:

**A) Rename class references:** `DMRG2GPU` → `DMRG2GPUOpt` everywhere.

**B) Constructor changes:**
- Remove GPU SVD workspace allocation (`d_svd_E_`, `d_svd_info_`, `d_svd_work_`)
- Remove rSVD workspace allocation (all `d_rsvd_*`)
- Remove `use_cpu_svd_` and `use_rsvd_` initialization
- Add Newton-Schulz GPU allocation:

```cpp
    // Newton-Schulz workspace
    int ns_max = chi_max_ * d_;
    HIP_CHECK(hipMalloc(&d_ns_U_,     (size_t)ns_max * ns_max * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_ns_U_new_, (size_t)ns_max * ns_max * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_ns_gram_,  (size_t)ns_max * ns_max * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_ns_P_,     (size_t)ns_max * ns_max * sizeof(Scalar)));

    h_ns_PtP_.resize(ns_max * ns_max);
    h_ns_eigvals_.resize(ns_max);
    // Query optimal syev workspace
    {
        int lwork_query = -1;
        Scalar work_opt;
        int info_q;
        Traits::lapack_syev("V", "U", &ns_max, h_ns_PtP_.data(), &ns_max,
                            h_ns_eigvals_.data(), &work_opt, &lwork_query,
                            nullptr, &info_q);
        int opt_lwork;
        if constexpr (Traits::is_complex) {
            opt_lwork = (int)Traits::real_part(work_opt) + 1;
        } else {
            opt_lwork = (int)work_opt + 1;
        }
        h_ns_syev_work_.resize(opt_lwork);
    }
    h_ns_syev_rwork_.resize(Traits::syev_rwork_size(ns_max));
```

- Add Block-Davidson GPU allocation:

```cpp
    // Block-Davidson workspace
    davidson_b_ = 4;
    davidson_max_sub_ = std::min(davidson_b_ * 8, theta_size_max_);

    HIP_CHECK(hipMalloc(&d_dav_V_,     (size_t)theta_size_max_ * davidson_max_sub_ * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_dav_AV_,    (size_t)theta_size_max_ * davidson_max_sub_ * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_dav_work_,  (size_t)theta_size_max_ * davidson_b_ * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_dav_work2_, (size_t)theta_size_max_ * davidson_b_ * sizeof(Scalar)));

    h_dav_H_proj_.resize(davidson_max_sub_ * davidson_max_sub_);
    h_dav_eigvals_.resize(davidson_max_sub_);
    h_dav_eigvecs_.resize(davidson_max_sub_ * davidson_max_sub_);
```

**C) Destructor changes:**
- Add `hipFree` for all new `d_ns_*` and `d_dav_*` buffers
- Remove `hipFree` for removed `d_rsvd_*` buffers

**D) Add newton_schulz_left function:**

Port from `pdmrg-gpu-opt/src/pdmrg_gpu_opt_impl.h` (was pdmrg_gpu_opt_impl.h lines 766-852). Changes:
- Remove `si` parameter → use `stream_` and `rocblas_h_` directly
- Remove `auto& ws = workspaces_[si]` → use `d_ns_gram_`, `d_ns_U_new_`, `d_heff_result_` directly
- Replace `handles_[si]` → `rocblas_h_`
- Replace `streams_[si]` → `stream_`
- Replace `ws.d_ns_gram` → `d_ns_gram_`
- Replace `ws.d_ns_U_new` → `d_ns_U_new_`
- Replace `ws.d_ns_U` → `d_ns_U_`
- Replace `ws.d_heff_result` → `d_heff_result_`

**E) Add ns_split function:**

Port from pdmrg-gpu-opt lines 951-1157. Changes:
- Remove `si` parameter and all `streams_[si]`/`handles_[si]` references
- Remove `auto& ws = workspaces_[si]` → use class members directly
- Replace all `ws.d_svd_Vh` → `d_svd_Vh_`, `ws.d_svd_U` → `d_svd_U_`, `ws.d_svd_work` → `d_svd_A_` (reuse as scratch), `ws.d_svd_S` → `d_svd_S_`, `ws.d_svd_A` → use `d_T2_` as scratch for U_full (or allocate d_svd_A_ large enough)
- Replace `ws.d_ns_U` → `d_ns_U_`, `ws.d_ns_P` → `d_ns_P_`, `ws.d_ns_gram` → `d_ns_gram_`
- Replace `ws.h_ns_*` → `h_ns_*_`
- Remove `ws.heff_cached_site = -1` line (dmrg2-gpu has no pointer caching)
- Fallback calls: `svd_split(site, d_theta, direction, si)` → `svd_split_fallback(site, d_theta, direction)`

**F) Add block_davidson_eigensolver function:**

Port from pdmrg-gpu-opt lines 1455-1779. Changes:
- Remove `si` parameter
- Replace `handles_[si]` → `rocblas_h_`, `streams_[si]` → `stream_`
- Replace `ws.d_dav_V` → `d_dav_V_`, `ws.d_dav_AV` → `d_dav_AV_`, etc.
- Replace `ws.d_dav_work` → `d_dav_work_`, `ws.d_dav_work2` → `d_dav_work2_`
- Replace `ws.d_heff_result` → `d_heff_result_`
- Replace `ws.h_dav_*` → `h_dav_*_`
- Fallback: `lanczos_eigensolver(site, d_theta, theta_size, si)` → `lanczos_eigensolver(site, d_theta, theta_size)` (existing dmrg2-gpu method, keep as-is)
- `apply_heff_two_site(site, v, out, si)` → `apply_heff_two_site(site, v, out)` (dmrg2-gpu's existing method has no `si` param)

**G) Modify optimize_bond to use new methods:**

Replace the body of `optimize_bond`:
```cpp
template<typename Scalar>
double DMRG2GPUOpt<Scalar>::optimize_bond(int site, char direction) {
    form_theta_two_site(site);

    int cL = chi_L(site);
    int cR = chi_R(site + 1);
    int theta_size = cL * d_ * d_ * cR;

    auto t0 = std::chrono::high_resolution_clock::now();
    double energy = block_davidson_eigensolver(site, d_theta_, theta_size);
    auto t1 = std::chrono::high_resolution_clock::now();
    prof_davidson_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

    t0 = std::chrono::high_resolution_clock::now();
    HIP_CHECK(hipStreamSynchronize(stream_));
    ns_split(site, d_theta_, direction);
    t1 = std::chrono::high_resolution_clock::now();
    prof_ns_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

    return energy;
}
```

**H) Update profiling variables:**
- Rename `prof_lanczos_ms` → `prof_davidson_ms`
- Rename `prof_svd_ms` → `prof_ns_ms`
- Update profiling output strings in `run()` to show "davidson" and "ns" labels

**I) Rename svd_split → svd_split_fallback:**
- Keep the existing svd_split implementation but rename it to `svd_split_fallback`
- This is only called from ns_split when NS fails
- Remove the GPU SVD path from it (only keep CPU LAPACK path since we don't allocate GPU SVD workspace)

**J) Remove rsvd_split entirely.**

- [ ] **Step 5: Create dmrg2_gpu_opt.cpp**

```cpp
#include "dmrg2_gpu_opt.h"

template class DMRG2GPUOpt<double>;
template class DMRG2GPUOpt<hipDoubleComplex>;
```

- [ ] **Step 6: Create test_dmrg2_gpu_opt.cpp**

Copy from `dmrg2-gpu/src/test_dmrg2_gpu.cpp`. Changes:
- Include: `#include "dmrg2_gpu_opt.h"` instead of `"dmrg2_gpu.h"`
- Replace class usage: `DMRG2GPU<double>` → `DMRG2GPUOpt<double>`, same for complex
- Remove `--cpu-svd`, `--gpu-svd`, `--rsvd` argument parsing
- Remove `dmrg.set_cpu_svd(...)` and `dmrg.set_rsvd(...)` calls
- Update banner: `"Two-Site DMRG-GPU-OPT (Newton-Schulz + Block-Davidson)"`
- Keep all MPO builders, test functions, and exact energy comparisons identical

- [ ] **Step 7: Update CMakeLists.txt**

Copy from `dmrg2-gpu/CMakeLists.txt`. Changes:
- Project: `DMRG2_GPU` → `DMRG2_GPU_OPT`
- Target: `dmrg2_gpu` → `dmrg2_gpu_opt`
- Source files: `src/test_dmrg2_gpu.cpp` → `src/test_dmrg2_gpu_opt.cpp`, `src/dmrg2_gpu.cpp` → `src/dmrg2_gpu_opt.cpp`
- Status messages: update accordingly

- [ ] **Step 8: Commit**

```bash
git add dmrg2-gpu-opt/
git commit -m "feat(dmrg2-gpu-opt): two-site DMRG with Newton-Schulz + Block-Davidson"
```

---

## Task 4: Create dmrg-gpu-opt (single-site + NS + Davidson)

Single-site requires adapting NS to different theta shapes:
- Direction 'R': theta is (chi_L*d, chi_R) — tall, NS-left works
- Direction 'L': theta is (chi_L, d*chi_R) — may be wide, fall back to CPU SVD

**Files:**
- Create: `dmrg-gpu-opt/` directory with 6 files
- Reference: `dmrg-gpu/src/` (baseline) and `pdmrg-gpu-opt/src/` (algorithms)

### Step-by-step:

- [ ] **Step 1: Copy baseline dmrg-gpu**

```bash
cp -r dmrg-gpu dmrg-gpu-opt
```

Remove files we don't need:
```bash
rm -f dmrg-gpu-opt/src/accurate_svd_gpu.h dmrg-gpu-opt/src/accurate_svd_gpu.cpp
rm -f dmrg-gpu-opt/src/test_svd_bug.cpp dmrg-gpu-opt/src/test_svd_fix.cpp
rm -rf dmrg-gpu-opt/build
```

- [ ] **Step 2: Create scalar_traits.h**

Identical to Task 3 Step 2 — same additions to the baseline scalar_traits.h:
- `dsyev_`/`zheev_` extern declarations
- `lapack_syev` and `syev_rwork_size` in both ScalarTraits specializations
- `scaled_identity_minus_*` kernels and wrappers
- `scale_columns_by_real` / `scale_rows_by_real` kernels and wrappers

- [ ] **Step 3: Create dmrg_gpu_opt.h (class declaration)**

Start from `dmrg-gpu/src/dmrg_gpu.h`. Changes:

**A) Rename:**
- Include guard: `DMRG_GPU_H` → `DMRG_GPU_OPT_H`
- Class: `DMRGGPU` → `DMRGGPUOpt`
- Include: `dmrg_gpu_impl.h` → `dmrg_gpu_opt_impl.h`

**B) Remove SVD toggle members** (same pattern as Task 3 Step 3B):
- Remove `set_cpu_svd`, `set_rsvd`, `use_cpu_svd_`, `use_rsvd_`, `rsvd_oversampling_`
- Remove all `d_rsvd_*` and `h_rsvd_*`
- Remove `d_svd_E_`, `d_svd_info_`, `d_svd_work_`
- Keep SVD host/device scratch for fallback

**C) Add Newton-Schulz workspace members:**

```cpp
    // Newton-Schulz workspace
    Scalar* d_ns_U_;        // (chi_max*d, chi_max) NS iterate
    Scalar* d_ns_U_new_;    // (chi_max*d, chi_max) NS next iterate
    Scalar* d_ns_gram_;     // (chi_max, chi_max) U^H U
    Scalar* d_ns_P_;        // (chi_max, chi_max) PSD factor

    // NS eigendecomp host workspace
    std::vector<Scalar> h_ns_PtP_;
    std::vector<RealType> h_ns_eigvals_;
    std::vector<Scalar> h_ns_syev_work_;
    std::vector<RealType> h_ns_syev_rwork_;
```

Note: for single-site, the NS operates on (chi_L*d, chi_R) matrices. The tall dimension is max chi_max*d, the short dimension is max chi_max. So gram and P are at most (chi_max, chi_max).

**D) Add Block-Davidson workspace members** (same as Task 3 Step 3D, but theta_size_max = chi_max * d * chi_max for single-site):

```cpp
    int davidson_b_;
    int davidson_max_sub_;

    Scalar* d_dav_V_;
    Scalar* d_dav_AV_;
    Scalar* d_dav_work_;
    Scalar* d_dav_work2_;

    std::vector<Scalar> h_dav_H_proj_;
    std::vector<RealType> h_dav_eigvals_;
    std::vector<Scalar> h_dav_eigvecs_;
    std::vector<RealType> h_dav_syev_work_;
```

**E) Add new method declarations:**

```cpp
    void newton_schulz_left(Scalar* d_A, int m, int n,
                            Scalar* d_U, Scalar* d_P,
                            double tol, int* out_iters);
    void ns_svd_and_update_mps(int site, Scalar* d_theta, char direction);
    double block_davidson_eigensolver(int site, Scalar* d_theta);
    void svd_fallback(int site, Scalar* d_theta, char direction);
```

Remove: `svd_and_update_mps`, `rsvd_and_update_mps`, `load_mps_from_file` declarations.

- [ ] **Step 4: Create dmrg_gpu_opt_impl.h**

Start from `dmrg-gpu/src/dmrg_gpu_impl.h`. Key changes:

**A) Rename all `DMRGGPU` → `DMRGGPUOpt`.**

**B) Constructor: same pattern as Task 3 Step 4B** — remove rSVD/GPU-SVD allocation, add NS + Davidson allocation. For single-site NS sizing:

```cpp
    // Newton-Schulz workspace (single-site: theta is at most chi_max*d × chi_max)
    int ns_max_m = chi_max_ * d_;
    int ns_max_n = chi_max_;
    int ns_max = std::max(ns_max_m, ns_max_n);
    HIP_CHECK(hipMalloc(&d_ns_U_,     (size_t)ns_max_m * ns_max_n * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_ns_U_new_, (size_t)ns_max_m * ns_max_n * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_ns_gram_,  (size_t)ns_max_n * ns_max_n * sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_ns_P_,     (size_t)ns_max_n * ns_max_n * sizeof(Scalar)));

    h_ns_PtP_.resize(ns_max_n * ns_max_n);
    h_ns_eigvals_.resize(ns_max_n);
    // ... syev workspace query as in Task 3 but with ns_max_n
```

**C) Add newton_schulz_left:** Same as Task 3 Step 4D — direct port from pdmrg-gpu-opt with `si` removed.

**D) Add ns_svd_and_update_mps:** This is the single-site adaptation of ns_split. The key difference from two-site ns_split:

```cpp
template<typename Scalar>
void DMRGGPUOpt<Scalar>::ns_svd_and_update_mps(int site, Scalar* d_theta, char direction) {
    int cL = chi_L(site);
    int cR = chi_R(site);

    int m, n_svd;
    if (direction == 'R') {
        m = cL * d_;    // tall dimension
        n_svd = cR;     // short dimension
    } else {
        m = cL;         // short dimension
        n_svd = d_ * cR; // tall dimension
    }

    int k = std::min(m, n_svd);

    // Fall back for small matrices or wide case in 'R' direction
    if (k <= 4 || m < 2 || n_svd < 2) {
        svd_fallback(site, d_theta, direction);
        return;
    }

    if (m >= n_svd) {
        // Tall/square: NS-left works (direction 'R' always, direction 'L' when chi_L >= d*chi_R)
        // ... same NS polar + eigendecomp as two-site ns_split ...
        // ... then store results differently:

        if (direction == 'R') {
            // U → MPS[site] (cL*d × new_k)
            // S*Vh needs to be absorbed into MPS[site+1]:
            //   new_tensor = (diag(S) @ Vh) @ MPS[site+1]
            //   First compute S_Vh (new_k × cR), then GEMM with MPS[site+1] (cR × d × chi_R_next)
            allocate_mps_tensor(site, cL, new_k);
            // copy U_full to MPS[site]
            HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site], /* U_full */,
                        m * new_k * sizeof(Scalar), hipMemcpyDeviceToDevice, stream_));

            // S_Vh = diag(S) @ Vh → (new_k, cR)
            // Scale rows of Vh by S on GPU
            scale_rows_by_real(/* Vh */, new_k, d_svd_S_, /* S_Vh scratch */, new_k, new_k, n_svd, stream_);

            // Absorb into MPS[site+1]: MPS'[site+1] = S_Vh @ MPS[site+1]
            // MPS[site+1] shape: (cR, d*chi_R_next) → result: (new_k, d*chi_R_next)
            int next_cols = d_ * chi_R(site + 1);
            Scalar* d_old_next = d_mps_tensors_[site + 1];
            Scalar* d_new_next;
            HIP_CHECK(hipMalloc(&d_new_next, (size_t)new_k * next_cols * sizeof(Scalar)));
            Scalar one = Traits::one(), zero_v = Traits::zero();
            ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
                rocblas_operation_none, rocblas_operation_none,
                new_k, next_cols, cR, &one, /* S_Vh */, new_k,
                d_old_next, cR, &zero_v, d_new_next, new_k));
            HIP_CHECK(hipFree(d_old_next));
            d_mps_tensors_[site + 1] = d_new_next;
            bond_dims_[site + 1] = new_k;

        } else { // direction == 'L', m >= n_svd means chi_L >= d*chi_R
            // Vh → MPS[site] (new_k × d × cR) = (new_k, d*cR)
            // U*S needs to be absorbed into MPS[site-1]:
            //   MPS'[site-1] = MPS[site-1] @ (U @ diag(S))
            allocate_mps_tensor(site, new_k, cR);
            // Copy Vh_trunc to MPS[site]
            HIP_CHECK(hipMemcpyAsync(d_mps_tensors_[site], /* Vh_trunc */,
                        new_k * n_svd * sizeof(Scalar), hipMemcpyDeviceToDevice, stream_));

            // U_S = U_full @ diag(S) → (m, new_k)
            scale_columns_by_real(/* U_full */, m, d_svd_S_, /* U_S scratch */, m, m, new_k, stream_);

            // Absorb into MPS[site-1]: MPS'[site-1] = MPS[site-1] @ U_S
            // MPS[site-1] shape: (chi_L_prev * d, cL) → result: (chi_L_prev * d, new_k)
            int prev_rows = chi_L(site - 1) * d_;
            Scalar* d_old_prev = d_mps_tensors_[site - 1];
            Scalar* d_new_prev;
            HIP_CHECK(hipMalloc(&d_new_prev, (size_t)prev_rows * new_k * sizeof(Scalar)));
            Scalar one = Traits::one(), zero_v = Traits::zero();
            ROCBLAS_CHECK(Traits::gemm(rocblas_h_,
                rocblas_operation_none, rocblas_operation_none,
                prev_rows, new_k, cL, &one, d_old_prev, prev_rows,
                /* U_S */, m, &zero_v, d_new_prev, prev_rows));
            HIP_CHECK(hipFree(d_old_prev));
            d_mps_tensors_[site - 1] = d_new_prev;
            bond_dims_[site] = new_k;
        }
    } else {
        // Wide case: fall back to CPU SVD
        svd_fallback(site, d_theta, direction);
    }
}
```

The actual implementation will fill in the scratch buffer names and complete the NS polar + eigendecomp section (identical math to ns_split in Task 3, just different post-factorization MPS update logic).

**E) Add block_davidson_eigensolver:** Port from pdmrg-gpu-opt, adapted for single-site:

```cpp
template<typename Scalar>
double DMRGGPUOpt<Scalar>::block_davidson_eigensolver(int site, Scalar* d_theta) {
    int dim = chi_L(site) * d_ * chi_R(site);
    // ... same Block-Davidson algorithm as pdmrg-gpu-opt ...
    // Key difference: calls apply_heff(site, v, out) instead of apply_heff_two_site(site, v, out, si)
    // Remove si parameter from all rocBLAS/HIP calls
}
```

The Lanczos fallback call becomes: `lanczos_eigensolver(site, d_theta)` (2 params, matching existing signature).

**F) Rename svd_and_update_mps → svd_fallback:** Keep CPU LAPACK SVD only, remove GPU SVD path.

**G) Remove rsvd_and_update_mps entirely.**

**H) Update optimize_site to use new methods:**

```cpp
template<typename Scalar>
double DMRGGPUOpt<Scalar>::optimize_site(int site, char direction) {
    form_theta(site, d_theta_);

    auto t0 = std::chrono::high_resolution_clock::now();
    double energy = block_davidson_eigensolver(site, d_theta_);
    auto t1 = std::chrono::high_resolution_clock::now();
    prof_davidson_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

    t0 = std::chrono::high_resolution_clock::now();
    HIP_CHECK(hipStreamSynchronize(stream_));
    ns_svd_and_update_mps(site, d_theta_, direction);
    t1 = std::chrono::high_resolution_clock::now();
    prof_ns_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

    return energy;
}
```

**I) Update profiling output** (same as Task 3 Step 4H).

- [ ] **Step 5: Create dmrg_gpu_opt.cpp**

```cpp
#include "dmrg_gpu_opt.h"

template class DMRGGPUOpt<double>;
template class DMRGGPUOpt<hipDoubleComplex>;
```

- [ ] **Step 6: Create test_dmrg_gpu_opt.cpp**

Copy from `dmrg-gpu/src/test_dmrg_gpu.cpp`. Changes:
- Include: `"dmrg_gpu_opt.h"` instead of `"dmrg_gpu.h"`
- Replace: `DMRGGPU<double>` → `DMRGGPUOpt<double>`, same for complex
- Remove: `--cpu-svd`, `--gpu-svd`, `--rsvd` flag parsing and `set_cpu_svd`/`set_rsvd` calls
- Remove: `load_mps_from_file` option if present
- Update banner: `"DMRG-GPU-OPT (Newton-Schulz + Block-Davidson)"`

- [ ] **Step 7: Update CMakeLists.txt**

Same pattern as Task 3 Step 7:
- Project: `DMRG_GPU` → `DMRG_GPU_OPT`
- Target: `dmrg_gpu` → `dmrg_gpu_opt`
- Sources: rename to `*_opt.*`

- [ ] **Step 8: Commit**

```bash
git add dmrg-gpu-opt/
git commit -m "feat(dmrg-gpu-opt): single-site DMRG with Newton-Schulz + Block-Davidson"
```

---

## Task 5: Remote build and correctness test

- [ ] **Step 1: Sync all three opt implementations to remote**

```bash
scp -r pdmrg-gpu-opt hotaisle@23.183.40.75:~/pdmrg-gpu-opt/
scp -r dmrg2-gpu-opt hotaisle@23.183.40.75:~/dmrg2-gpu-opt/
scp -r dmrg-gpu-opt hotaisle@23.183.40.75:~/dmrg-gpu-opt/
```

- [ ] **Step 2: Build pdmrg-gpu-opt on remote**

```bash
ssh hotaisle@23.183.40.75 'cd ~/pdmrg-gpu-opt && mkdir -p build && cd build && cmake .. && make -j4'
```

Expected: builds successfully, executable `pdmrg_gpu_opt` created.

- [ ] **Step 3: Test pdmrg-gpu-opt correctness**

```bash
ssh hotaisle@23.183.40.75 '~/pdmrg-gpu-opt/build/pdmrg_gpu_opt 8 50 3 --ns-split --davidson'
```

Expected: Heisenberg L=8 energy error < 1e-10. (This is just the rename, so should work identically to pdmrg-gpu-opt.)

- [ ] **Step 4: Build dmrg2-gpu-opt on remote**

```bash
ssh hotaisle@23.183.40.75 'cd ~/dmrg2-gpu-opt && mkdir -p build && cd build && cmake .. && make -j4'
```

Expected: compiles without errors, `dmrg2_gpu_opt` executable.

- [ ] **Step 5: Test dmrg2-gpu-opt correctness**

Run the standard tests:
```bash
ssh hotaisle@23.183.40.75 '~/dmrg2-gpu-opt/build/dmrg2_gpu_opt 8 50 10'
```

Expected: Heisenberg L=8 energy error < 1e-10.

Then complex test:
```bash
ssh hotaisle@23.183.40.75 '~/dmrg2-gpu-opt/build/dmrg2_gpu_opt 6 50 10 --josephson'
```

Expected: Josephson L=6 energy error < 1e-10.

And TFIM:
```bash
ssh hotaisle@23.183.40.75 '~/dmrg2-gpu-opt/build/dmrg2_gpu_opt 8 50 10 --tfim'
```

Expected: TFIM L=8 energy error < 1e-10.

- [ ] **Step 6: Build dmrg-gpu-opt on remote**

```bash
ssh hotaisle@23.183.40.75 'cd ~/dmrg-gpu-opt && mkdir -p build && cd build && cmake .. && make -j4'
```

- [ ] **Step 7: Test dmrg-gpu-opt correctness**

```bash
ssh hotaisle@23.183.40.75 '~/dmrg-gpu-opt/build/dmrg_gpu_opt 8 50 10'
```

Expected: Heisenberg L=8 energy error < 1e-10.

```bash
ssh hotaisle@23.183.40.75 '~/dmrg-gpu-opt/build/dmrg_gpu_opt 6 50 10 --josephson'
```

Expected: Josephson L=6 energy error < 1e-10.

- [ ] **Step 8: Quick timing comparison**

Run Heisenberg L=64 chi=128 on both baseline and opt to verify NS speedup:

```bash
# Baseline
ssh hotaisle@23.183.40.75 '~/dmrg2-gpu/build/dmrg2_gpu 64 128 4 --cpu-svd'
# Opt
ssh hotaisle@23.183.40.75 '~/dmrg2-gpu-opt/build/dmrg2_gpu_opt 64 128 4'
```

Expected: opt should show dramatically lower NS time vs SVD time at chi=128, ~2-4x overall speedup.

- [ ] **Step 9: Commit any fixes from testing**

If tests revealed issues, fix and recommit.

- [ ] **Step 10: Final commit**

```bash
git add -A
git commit -m "test: verify all GPU-opt implementations pass correctness tests"
```
