#ifndef ACCURATE_SVD_GPU_H
#define ACCURATE_SVD_GPU_H

/**
 * GPU-native accurate SVD with recursive refinement (Stoudenmire & White,
 * arXiv:1301.3494, Appendix). Templated on Scalar (double + hipDoubleComplex),
 * stream-aware, all data device-resident. Replaces the host-LAPACK accurate_svd.h
 * used by pdmrg-gpu prior to commit 8dbd1b8 → <this commit>.
 *
 * Why GPU-native: the host version did D2H of the full theta tensor (cL*d × d*cR
 * scalars, MBs at chi=256), then ran recursive LAPACK SVD + cpu_gemm on host,
 * then H2D'd U/Vh/S. Per boundary per sweep × n_segments-1 boundaries × tens of
 * sweeps, that PCIe traffic dominates segment-boundary cost. This port keeps
 * everything on device: rocsolver_gesvd for the standard SVD, rocBLAS gemm for
 * the projection and write-back, a tiny on-device kernel for the split-point
 * search. Per call only a single int (split-point + gesvd info merged) crosses
 * the PCIe boundary.
 *
 * Algorithm (per accurate_svd.py / accurate_svd.h, mirrored exactly):
 *   1. Standard SVD: M = U · S · Vh (via rocsolver_gesvd)
 *   2. Find split point p where S[p] / S[0] < epsilon (on-device kernel)
 *   3. If p < full_k:
 *        T   = U[:, p:]^H · M               # (sub_k × n)
 *        X   = T · Vh[p:, :]^H              # (sub_k × sub_k)
 *        sub_U, sub_S, sub_Vh = accurate_svd_gpu(X, depth+1)
 *        U[:, p:]  ← U[:, p:]  · sub_U      # write-back via temp + 2D copy
 *        Vh[p:, :] ← sub_Vh · Vh[p:, :]
 *        S[p:]     ← sub_S
 *
 * Workspace: pre-allocated arena per stream, sized for chi_max × d at depth 0,
 * and full_k × full_k for depths 1..MAX_DEPTH-1. Recursion depth is bounded by
 * AsvdScratch::MAX_DEPTH (default 5); deeper recursion is exceptionally rare
 * (would require log_eps(1) levels of nested condition-number stratification).
 */

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>
#include "scalar_traits.h"

#include <stdexcept>
#include <string>

// Local check macro — uses a unique name so it can't collide with the variant
// impls' HIP_CHECK macros (which throw via cerr; we want a self-contained
// throw-only path inside this header).
#define ASVD_HIP_CHECK(call) \
    do { \
        hipError_t _e = (call); \
        if (_e != hipSuccess) { \
            throw std::runtime_error(std::string("ASVD_HIP_CHECK ") + __FILE__ + ":" + \
                                     std::to_string(__LINE__) + " " #call " - " + \
                                     hipGetErrorString(_e)); \
        } \
    } while (0)

// RAII guard for rocBLAS pointer mode. Captures caller's mode at construction,
// installs the requested mode, restores caller's mode on destruction (including
// when destruction fires via stack-unwinding from a thrown exception). Replaces
// the previous inline `rocblas_set_pointer_mode(prev_pm)` calls scattered
// before every throw / return — those couldn't cover the implicit ASVD_HIP_CHECK
// throws inside step-6 hipMemcpyAsync blocks, leaking host-mode state into the
// caller's handle on rare HIP failures.
struct AsvdPointerModeGuard {
    rocblas_handle handle;
    rocblas_pointer_mode prev_mode;
    AsvdPointerModeGuard(rocblas_handle h, rocblas_pointer_mode new_mode) : handle(h) {
        rocblas_get_pointer_mode(h, &prev_mode);
        rocblas_set_pointer_mode(h, new_mode);
    }
    ~AsvdPointerModeGuard() {
        // Best-effort restore. Swallow errors: throwing from a destructor
        // during stack unwinding would call std::terminate.
        rocblas_set_pointer_mode(handle, prev_mode);
    }
    AsvdPointerModeGuard(const AsvdPointerModeGuard&) = delete;
    AsvdPointerModeGuard& operator=(const AsvdPointerModeGuard&) = delete;
};

// ============================================================================
// On-device split-point kernel.
// Returns the smallest index p such that S[p] < epsilon * S[0], or k if none.
// Single block, single thread — the array is tiny (k ≤ chi_max ~ 256) so the
// scan is ~µs. Avoids a D2H of S just to compute one int.
// ============================================================================
__device__ inline int asvd_split_point_impl(const double* d_S, int k, double epsilon) {
    if (k == 0) return 0;
    if (d_S[0] < 1e-30) return k;
    double thresh = epsilon * d_S[0];
    for (int i = 0; i < k; i++) {
        if (d_S[i] < thresh) return i;
    }
    return k;
}

__global__ inline void asvd_find_split_kernel(
    const double* d_S, int k, double epsilon, int* d_p_out)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    *d_p_out = asvd_split_point_impl(d_S, k, epsilon);
}

// ============================================================================
// Pre-allocated per-recursion-level scratch.
//
// Sized once at construction for the largest (m, n, full_k) the caller will
// ever pass. Subsequent calls reuse the buffers — zero per-call hipMalloc,
// zero per-call hipFree. One AsvdScratch lives per stream / per device.
// ============================================================================
template<typename Scalar>
struct AsvdScratch {
    using Traits = ScalarTraits<Scalar>;
    using RealType = typename Traits::RealType;
    // Recursion cap. Each level pre-allocates ~6×max_k² scalars; with
    // max_k=chi_max·d and complex scalars at chi=256/d=2, that's ~24 MB per
    // level. MAX_DEPTH=3 keeps the per-stream arena under ~100 MB while still
    // covering the spectral-degeneracy regime where Stoudenmire actually fires
    // (depth 1 already enough for typical PDMRG boundary tensors; depth 2 for
    // pathological condition numbers; depth 3 is reserved breathing room).
    // If recursion would exceed MAX_DEPTH the function gracefully returns the
    // standard rocsolver_gesvd output for that sub-problem, mirroring the
    // host accurate_svd's natural depth-limited behavior in the rare case
    // where condition number requires deeper splitting.
    static constexpr int MAX_DEPTH = 3;

    int max_m = 0, max_n = 0, max_k = 0;

    // Per-depth buffers. Level 0 sees the full (max_m × max_n); levels 1+ see
    // sub-problems at most (max_k × max_k). Sized accordingly to keep peak
    // VRAM modest (~30 MB per stream at chi=256, complex).
    Scalar*   d_M_work[MAX_DEPTH] = {};
    Scalar*   d_U[MAX_DEPTH] = {};
    RealType* d_S[MAX_DEPTH] = {};
    Scalar*   d_Vh[MAX_DEPTH] = {};
    RealType* d_E[MAX_DEPTH] = {};
    int*      d_info[MAX_DEPTH] = {};

    // Per-depth recursion temporaries
    Scalar*   d_T[MAX_DEPTH] = {};         // (sub_k × n) = U[:,p:]^H · M
    Scalar*   d_X[MAX_DEPTH] = {};         // (sub_k × sub_k) = T · Vh[p:,:]^H
    Scalar*   d_block[MAX_DEPTH] = {};     // (m × sub_k) or (sub_k × n) for write-back

    void allocate(int m_max, int n_max) {
        if (m_max <= max_m && n_max <= max_n) return;  // already big enough
        release();
        max_m = m_max;
        max_n = n_max;
        max_k = (m_max < n_max) ? m_max : n_max;

        for (int d = 0; d < MAX_DEPTH; d++) {
            // Level 0: full m×n.   Levels 1+: at most max_k×max_k (sub-problem).
            int dm = (d == 0) ? max_m : max_k;
            int dn = (d == 0) ? max_n : max_k;
            int dk = (dm < dn) ? dm : dn;

            size_t sz_mn = (size_t)dm * dn;
            size_t sz_mk = (size_t)dm * dk;
            size_t sz_kn = (size_t)dk * dn;
            size_t sz_kk = (size_t)dk * dk;

            HIP_CHECK_OR_THROW(hipMalloc(&d_M_work[d], sz_mn * sizeof(Scalar)));
            HIP_CHECK_OR_THROW(hipMalloc(&d_U[d],      sz_mk * sizeof(Scalar)));
            HIP_CHECK_OR_THROW(hipMalloc(&d_S[d],      dk    * sizeof(RealType)));
            HIP_CHECK_OR_THROW(hipMalloc(&d_Vh[d],     sz_kn * sizeof(Scalar)));
            HIP_CHECK_OR_THROW(hipMalloc(&d_E[d],      dk    * sizeof(RealType)));
            HIP_CHECK_OR_THROW(hipMalloc(&d_info[d],   sizeof(int)));

            HIP_CHECK_OR_THROW(hipMalloc(&d_T[d],      sz_kn * sizeof(Scalar)));
            HIP_CHECK_OR_THROW(hipMalloc(&d_X[d],      sz_kk * sizeof(Scalar)));
            HIP_CHECK_OR_THROW(hipMalloc(&d_block[d],
                ((sz_mk > sz_kn) ? sz_mk : sz_kn) * sizeof(Scalar)));
        }
    }

    void release() {
        for (int d = 0; d < MAX_DEPTH; d++) {
            if (d_M_work[d]) { hipFree(d_M_work[d]); d_M_work[d] = nullptr; }
            if (d_U[d])      { hipFree(d_U[d]);      d_U[d]      = nullptr; }
            if (d_S[d])      { hipFree(d_S[d]);      d_S[d]      = nullptr; }
            if (d_Vh[d])     { hipFree(d_Vh[d]);     d_Vh[d]     = nullptr; }
            if (d_E[d])      { hipFree(d_E[d]);      d_E[d]      = nullptr; }
            if (d_info[d])   { hipFree(d_info[d]);   d_info[d]   = nullptr; }
            if (d_T[d])      { hipFree(d_T[d]);      d_T[d]      = nullptr; }
            if (d_X[d])      { hipFree(d_X[d]);      d_X[d]      = nullptr; }
            if (d_block[d])  { hipFree(d_block[d]);  d_block[d]  = nullptr; }
        }
        max_m = max_n = max_k = 0;
    }

    ~AsvdScratch() { release(); }

    // Non-copyable; per-stream owner.
    AsvdScratch() = default;
    AsvdScratch(const AsvdScratch&) = delete;
    AsvdScratch& operator=(const AsvdScratch&) = delete;

private:
    static void HIP_CHECK_OR_THROW(hipError_t e) {
        if (e != hipSuccess) {
            throw std::runtime_error(std::string("AsvdScratch hipMalloc: ") +
                                     hipGetErrorString(e));
        }
    }
};

// ============================================================================
// Recursive on-device accurate SVD.
//
// Inputs/outputs are all DEVICE pointers. The input matrix d_M_in (m × n,
// column-major, lda=ldm) is read but not modified. Outputs U (m × full_k,
// lda=ldu), S (full_k), Vh (full_k × n, lda=ldvh) on device.
//
// `ws` provides per-depth scratch arenas. `depth` MUST start at 0 in the
// top-level call; recursive sub-calls bump it. Hard cap at AsvdScratch::MAX_DEPTH.
// ============================================================================
template<typename Scalar>
inline void accurate_svd_gpu(
    rocblas_handle handle, hipStream_t stream,
    int m, int n,
    const Scalar* d_M_in, int ldm,
    Scalar* d_U_out, int ldu,
    typename ScalarTraits<Scalar>::RealType* d_S_out,
    Scalar* d_Vh_out, int ldvh,
    AsvdScratch<Scalar>& ws,
    int depth = 0,
    double epsilon = 1e-4)
{
    using Traits = ScalarTraits<Scalar>;
    using RealType = typename Traits::RealType;

    int full_k = (m < n) ? m : n;
    if (full_k == 0) return;
    if (depth >= AsvdScratch<Scalar>::MAX_DEPTH) {
        // Defensive cap: the public-facing recursion gate at the
        // `if (depth + 1 < MAX_DEPTH)` check below prevents recursion past
        // depth = MAX_DEPTH-1 in normal operation, so this branch is dead
        // code today. Kept to prevent OOB on `ws.d_*[depth]` if a future
        // edit relaxes the gate without realloc'ing per-depth slots.
        return;
    }

    // Workspace-bounds check (round-4 M2): caller's allocate() sized the
    // arena for some (max_m, max_n); if a later call passes (m, n) beyond
    // those, we'd OOB on ws.d_M_work[depth] / ws.d_U[depth] / etc. Throw
    // a clear error rather than silently corrupting memory.
    if (depth == 0 && (m > ws.max_m || n > ws.max_n)) {
        throw std::runtime_error("accurate_svd_gpu: input (m=" + std::to_string(m)
                                 + ", n=" + std::to_string(n)
                                 + ") exceeds AsvdScratch capacity (max_m="
                                 + std::to_string(ws.max_m) + ", max_n="
                                 + std::to_string(ws.max_n)
                                 + "). Call ws.allocate() with larger sizes first.");
    }

    // --- Step 1: Standard SVD on device ---
    // rocsolver_gesvd is destructive on the input, so copy d_M_in → d_M_work.
    // Caller's d_M_in may have ldm != m; copy by 2D into a packed (m × n) buffer.
    if (ldm == m) {
        ASVD_HIP_CHECK(hipMemcpyAsync(ws.d_M_work[depth], d_M_in,
                                       (size_t)m * n * sizeof(Scalar),
                                       hipMemcpyDeviceToDevice, stream));
    } else {
        ASVD_HIP_CHECK(hipMemcpy2DAsync(
            ws.d_M_work[depth], m * sizeof(Scalar),
            d_M_in,             ldm * sizeof(Scalar),
            m * sizeof(Scalar), n,
            hipMemcpyDeviceToDevice, stream));
    }

    // RAII pointer-mode guard: installs host mode, restores caller's mode on
    // destruction — covers every exit path (return / throw / unwind) without
    // per-block boilerplate. Replaces the prior pattern of inline
    // `rocblas_set_pointer_mode(prev_pm); throw;` before each error site,
    // which couldn't cover the implicit ASVD_HIP_CHECK throws in step 6.
    AsvdPointerModeGuard pm_guard(handle, rocblas_pointer_mode_host);

    rocblas_status st = Traits::rocsolver_gesvd(
        handle,
        rocblas_svect_singular, rocblas_svect_singular,
        m, n,
        ws.d_M_work[depth], m,
        d_S_out,
        d_U_out, ldu,
        d_Vh_out, ldvh,
        ws.d_E[depth],
        rocblas_outofplace,
        ws.d_info[depth]);
    if (st != rocblas_status_success) {
        throw std::runtime_error("accurate_svd_gpu: rocsolver_gesvd failed status="
                                 + std::to_string((int)st));
    }

    // gesvd may report info != 0 (non-convergence on worst-case matrices);
    // mirror the host version's behavior — return the standard result rather
    // than throwing. The caller still gets a usable (if less accurate) SVD.
    int h_info = 0;
    ASVD_HIP_CHECK(hipMemcpyAsync(&h_info, ws.d_info[depth], sizeof(int),
                                   hipMemcpyDeviceToHost, stream));
    ASVD_HIP_CHECK(hipStreamSynchronize(stream));
    if (h_info != 0) {
        return;  // pm_guard restores caller's mode
    }

    // --- Step 2: Find split point p on device, read back the single int ---
    int h_p = full_k;
    if (depth + 1 < AsvdScratch<Scalar>::MAX_DEPTH) {
        // Reuse d_info[depth] slot for the split-point output.
        hipLaunchKernelGGL(asvd_find_split_kernel, dim3(1), dim3(1), 0, stream,
                           d_S_out, full_k, epsilon, ws.d_info[depth]);
        ASVD_HIP_CHECK(hipMemcpyAsync(&h_p, ws.d_info[depth], sizeof(int),
                                       hipMemcpyDeviceToHost, stream));
        ASVD_HIP_CHECK(hipStreamSynchronize(stream));
    }

    // --- Step 3: No split needed → standard SVD is good enough, done ---
    if (h_p >= full_k) {
        return;  // pm_guard restores caller's mode
    }

    int sub_k = full_k - h_p;
    int next_depth = depth + 1;

    // --- Step 4: Project M onto the inaccurate subspace ---
    // T = U[:, p:]^H · M    →   (sub_k × n)
    Scalar one = Traits::one();
    Scalar zero = Traits::zero();
    {
        // U[:, p:] is at d_U_out + p*ldu, shape (m × sub_k), lda=ldu.
        // op_h conjugate-transposes for complex; transposes for real.
        rocblas_status gst = Traits::gemm(
            handle,
            Traits::op_h, rocblas_operation_none,
            sub_k, n, m, &one,
            d_U_out + (size_t)h_p * ldu, ldu,
            d_M_in,                       ldm,
            &zero,
            ws.d_T[depth], sub_k);
        if (gst != rocblas_status_success) {
            throw std::runtime_error("accurate_svd_gpu: gemm T failed status="
                                     + std::to_string((int)gst));
        }
    }

    // X = T · Vh[p:, :]^H    →   (sub_k × sub_k)
    // Vh[p:, :] is at d_Vh_out + p, shape (sub_k × n), lda=ldvh.
    {
        rocblas_status gst = Traits::gemm(
            handle,
            rocblas_operation_none, Traits::op_h,
            sub_k, sub_k, n, &one,
            ws.d_T[depth],         sub_k,
            d_Vh_out + h_p,        ldvh,
            &zero,
            ws.d_X[depth], sub_k);
        if (gst != rocblas_status_success) {
            throw std::runtime_error("accurate_svd_gpu: gemm X failed status="
                                     + std::to_string((int)gst));
        }
    }

    // --- Step 5: Recursive accurate SVD of X (sub_k × sub_k) ---
    // Outputs go into the next-depth slot's U/S/Vh buffers. The recursive
    // call manages its own AsvdPointerModeGuard — it sees host mode on entry
    // (we're still in host mode here), captures it, and restores host on
    // exit. No need to flip pointer mode around the recursion.
    accurate_svd_gpu<Scalar>(
        handle, stream,
        sub_k, sub_k,
        ws.d_X[depth], sub_k,
        ws.d_U[next_depth],  sub_k,
        ws.d_S[next_depth],
        ws.d_Vh[next_depth], sub_k,
        ws, next_depth, epsilon);
    // sub_U lives at ws.d_U[next_depth] (sub_k × sub_k, lda=sub_k)
    // sub_S lives at ws.d_S[next_depth] (sub_k)
    // sub_Vh lives at ws.d_Vh[next_depth] (sub_k × sub_k, lda=sub_k)

    // --- Step 6a: U[:, p:] ← U[:, p:] · sub_U   (m × sub_k) ---
    // Compute new_cols = U[:,p:] · sub_U into ws.d_block, then 2D copy back.
    // (rocBLAS does not support in-place GEMM with overlapping A/C.)
    {
        rocblas_status gst = Traits::gemm(
            handle,
            rocblas_operation_none, rocblas_operation_none,
            m, sub_k, sub_k, &one,
            d_U_out + (size_t)h_p * ldu, ldu,
            ws.d_U[next_depth],          sub_k,
            &zero,
            ws.d_block[depth], m);
        if (gst != rocblas_status_success) {
            throw std::runtime_error("accurate_svd_gpu: gemm U-update failed status="
                                     + std::to_string((int)gst));
        }
        // Copy block (m × sub_k, lda=m) → U_out[:, p:] (lda=ldu).
        if (ldu == m) {
            ASVD_HIP_CHECK(hipMemcpyAsync(d_U_out + (size_t)h_p * ldu, ws.d_block[depth],
                                           (size_t)m * sub_k * sizeof(Scalar),
                                           hipMemcpyDeviceToDevice, stream));
        } else {
            ASVD_HIP_CHECK(hipMemcpy2DAsync(
                d_U_out + (size_t)h_p * ldu, ldu * sizeof(Scalar),
                ws.d_block[depth],            m   * sizeof(Scalar),
                m * sizeof(Scalar), sub_k,
                hipMemcpyDeviceToDevice, stream));
        }
    }

    // --- Step 6b: Vh[p:, :] ← sub_Vh · Vh[p:, :]   (sub_k × n) ---
    {
        rocblas_status gst = Traits::gemm(
            handle,
            rocblas_operation_none, rocblas_operation_none,
            sub_k, n, sub_k, &one,
            ws.d_Vh[next_depth], sub_k,
            d_Vh_out + h_p,      ldvh,
            &zero,
            ws.d_block[depth], sub_k);
        if (gst != rocblas_status_success) {
            throw std::runtime_error("accurate_svd_gpu: gemm Vh-update failed status="
                                     + std::to_string((int)gst));
        }
        // Copy block (sub_k × n, lda=sub_k) → Vh_out[p:, :] (lda=ldvh).
        ASVD_HIP_CHECK(hipMemcpy2DAsync(
            d_Vh_out + h_p,    ldvh  * sizeof(Scalar),
            ws.d_block[depth], sub_k * sizeof(Scalar),
            sub_k * sizeof(Scalar), n,
            hipMemcpyDeviceToDevice, stream));
    }

    // --- Step 6c: S[p:] ← sub_S   (D2D copy, sub_k doubles) ---
    ASVD_HIP_CHECK(hipMemcpyAsync(d_S_out + h_p, ws.d_S[next_depth],
                                   sub_k * sizeof(RealType),
                                   hipMemcpyDeviceToDevice, stream));

    // pm_guard destructor restores caller's mode.
}

#endif // ACCURATE_SVD_GPU_H
