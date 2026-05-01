// Shared GPU kernels for batched-GEMM pointer setup. Eliminates the
// per-call host loop + 3× hipMemcpyAsync H2D pattern that violates the
// "no host roundtrips per sweep" rule (CLAUDE.md 2026-04-27).
//
// Originally file-local in dmrg-gpu/src/dmrg_gpu_impl.h; promoted to a
// shared header in round-15 so dmrg-gpu-opt and other variants can
// reuse them without duplication.
#ifndef PMRG_BATCH_PTRS_KERNELS_H
#define PMRG_BATCH_PTRS_KERNELS_H

#include <hip/hip_runtime.h>

// Step 1/update_env: A[w*d+s] = base_A + w*strideA,
//                    B[w*d+s] = base_B + s*strideB,
//                    C[w*d+s] = base_C + (w*d+s)*strideC
template<typename Scalar>
__global__ void setup_batch_ptrs_wd(Scalar** A, Scalar** B, Scalar** C,
                                     Scalar* base_A, Scalar* base_B, Scalar* base_C,
                                     int d, int strideA, int strideB, int strideC) {
    int idx = threadIdx.x;  // idx = w*d + s
    int w = idx / d, s = idx % d;
    A[idx] = base_A + w * strideA;
    B[idx] = base_B + s * strideB;
    C[idx] = base_C + idx * strideC;
}

// Mirror of setup_batch_ptrs_wd with A/B index roles swapped:
// A[w*d+s] = base_A + s*strideA (s-indexed),
// B[w*d+s] = base_B + w*strideB (w-indexed),
// C[w*d+s] = base_C + (w*d+s)*strideC.
// Used by update_right_env Step 1 where the MPS A_s is s-strided and
// R_env_w is w-strided (opposite of update_left_env).
template<typename Scalar>
__global__ void setup_batch_ptrs_sw(Scalar** A, Scalar** B, Scalar** C,
                                     Scalar* base_A, Scalar* base_B, Scalar* base_C,
                                     int d, int strideA, int strideB, int strideC) {
    int idx = threadIdx.x;  // idx = w*d + s
    int w = idx / d, s = idx % d;
    A[idx] = base_A + s * strideA;
    B[idx] = base_B + w * strideB;
    C[idx] = base_C + idx * strideC;
}

// Step 3 (apply_heff): A[s] = base_A + (wp*d+s)*strideA,
//                      B[s] = base_B + wp*strideB,
//                      C[s] = base_C + s*strideC
template<typename Scalar>
__global__ void setup_batch_ptrs_step3(Scalar** A, Scalar** B, Scalar** C,
                                        Scalar* base_A, Scalar* base_B, Scalar* base_C,
                                        int wp, int d, int strideA, int strideB, int strideC) {
    int s = threadIdx.x;
    A[s] = base_A + (wp * d + s) * strideA;
    B[s] = base_B + wp * strideB;
    C[s] = base_C + s * strideC;
}

// Step 3 (R3-F1 full batched): single launch writes all D*d pointers.
// idx = wp*d + sp. Each wp writes to its own scratch slice; slices
// are summed afterwards by a single rocblas_gemv reduction.
//   A[idx] -> U[wp*d + sp]        (cL x cR block, lda = cL*cR)
//   B[idx] -> R[wp]               (cR x cR block, ldb = cR*D)
//   C[idx] -> scratch + wp*slice_stride + sp*strideC_tile
//             (cL x cR tile inside the (cL, d, cR) slice, ldc = cL*d)
template<typename Scalar>
__global__ void setup_batch_ptrs_step3_full(Scalar** A, Scalar** B, Scalar** C,
                                             Scalar* base_A, Scalar* base_B, Scalar* base_C_scratch,
                                             int d, int strideA, int strideB,
                                             int strideC_tile, int slice_stride) {
    int idx = threadIdx.x;  // wp*d + sp
    int wp = idx / d;
    int sp = idx % d;
    A[idx] = base_A + (wp * d + sp) * strideA;
    B[idx] = base_B + wp * strideB;
    C[idx] = base_C_scratch + wp * slice_stride + sp * strideC_tile;
}

// Step 3 (env update): A[w] = base_A + (w*d+sp)*strideA,
//                      B[w] = base_B + sp*strideB,
//                      C[w] = base_C + w*strideC
template<typename Scalar>
__global__ void setup_batch_ptrs_env3(Scalar** A, Scalar** B, Scalar** C,
                                       Scalar* base_A, Scalar* base_B, Scalar* base_C,
                                       int sp, int d, int strideA, int strideB, int strideC) {
    int w = threadIdx.x;
    A[w] = base_A + (w * d + sp) * strideA;
    B[w] = base_B + sp * strideB;
    C[w] = base_C + w * strideC;
}

// SPARSE_MPO variants: index into a precomputed list of nonzero (w, s)
// or (wp, sp) pairs of W_left (packed as ws = w*d + s), emitting only
// those batches. The destination C tiles are still in the full-layout
// slot, so the destination buffer must be zeroed beforehand if any
// subsequent step reads the skipped slots.

// Sparse Step 1: A[idx] <- base_A + w*strideA   where (w,s) = unpack(nnz_ws[idx])
//                B[idx] <- base_B + s*strideB
//                C[idx] <- base_C + ws*strideC
template<typename Scalar>
__global__ void setup_batch_ptrs_wd_sparse(Scalar** A, Scalar** B, Scalar** C,
                                           Scalar* base_A, Scalar* base_B, Scalar* base_C,
                                           const int* nnz_ws, int d,
                                           int strideA, int strideB, int strideC) {
    int idx = threadIdx.x;
    int ws = nnz_ws[idx];
    int w = ws / d;
    int s = ws % d;
    A[idx] = base_A + w * strideA;
    B[idx] = base_B + s * strideB;
    C[idx] = base_C + ws * strideC;
}

// Sparse Step 3-full: same layout as setup_batch_ptrs_step3_full but
// indexes only the nonzero (wp, sp) columns. The per-wp scratch slices
// must be zeroed beforehand for the reduction GEMV over D to not pick
// up stale values from skipped wp slices.
template<typename Scalar>
__global__ void setup_batch_ptrs_step3_full_sparse(Scalar** A, Scalar** B, Scalar** C,
                                                   Scalar* base_A, Scalar* base_B, Scalar* base_C_scratch,
                                                   const int* nnz_wpsp, int d,
                                                   int strideA, int strideB,
                                                   int strideC_tile, int slice_stride) {
    int idx = threadIdx.x;
    int wpsp = nnz_wpsp[idx];
    int wp = wpsp / d;
    int sp = wpsp % d;
    A[idx] = base_A + wpsp * strideA;
    B[idx] = base_B + wp * strideB;
    C[idx] = base_C_scratch + wp * slice_stride + sp * strideC_tile;
}

// pdmrg-gpu-opt cross-segment batched variant: same w*dd+ss layout but
// theta is indexed linearly by ss (no transpose) — mirrors the inner-
// segment apply_heff_two_site of pdmrg-gpu where theta is laid out
// (cL, dd, cR) with dd contiguous. Used by the cross-segment batched
// path that aggregates D*dd batches per segment.
template<typename Scalar>
__global__ void setup_batch_ptrs_wd_twosite_linear(Scalar** A, Scalar** B, Scalar** C,
                                                    Scalar* base_A, Scalar* base_B, Scalar* base_C,
                                                    int dd, int strideA, int strideB, int strideC) {
    int idx = threadIdx.x;
    int n = idx / dd;
    int ss = idx % dd;
    A[idx] = base_A + n * strideA;
    B[idx] = base_B + ss * strideB;
    C[idx] = base_C + idx * strideC;
}

// ============================================================================
// Two-site variants (dmrg2-* family). Pack three indices: idx = w*dd + s1*d + s2
// with dd = d*d. The B base is indexed by the transposed physical index
// (s1 + s2*d) so theta_{s1,s2} can be read column-major in s2.
// ============================================================================

template<typename Scalar>
__global__ void setup_batch_ptrs_wd_twosite(Scalar** A, Scalar** B, Scalar** C,
                                             Scalar* base_A, Scalar* base_B, Scalar* base_C,
                                             int d, int dd, int strideA, int strideB, int strideC) {
    int idx = threadIdx.x;
    int w = idx / dd;
    int ss = idx % dd;        // s1*d + s2
    int s1 = ss / d, s2 = ss % d;
    A[idx] = base_A + w * strideA;
    B[idx] = base_B + (s1 + s2 * d) * strideB;  // transposed physical index
    C[idx] = base_C + idx * strideC;
}

template<typename Scalar>
__global__ void setup_batch_ptrs_step3_twosite(Scalar** A, Scalar** B, Scalar** C,
                                                Scalar* base_A, Scalar* base_B, Scalar* base_C,
                                                int n, int d, int dd, int strideA, int strideB, int strideC) {
    int idx = threadIdx.x;  // s1p*d + s2p
    int s1p = idx / d, s2p = idx % d;
    A[idx] = base_A + (n * dd + idx) * strideA;
    B[idx] = base_B + n * strideB;
    C[idx] = base_C + (s1p + s2p * d) * strideC;  // transposed physical index
}

// Two-site step 3 (full batched): one launch sets up D*dd batch pointers.
template<typename Scalar>
__global__ void setup_batch_ptrs_step3_twosite_full(Scalar** A, Scalar** B, Scalar** C,
                                                     Scalar* base_A, Scalar* base_B, Scalar* base_C_scratch,
                                                     int d, int dd, int strideA, int strideB,
                                                     int strideC_tile, int slice_stride) {
    int idx = threadIdx.x;  // n*dd + s1p*d + s2p
    int n = idx / dd;
    int ss = idx % dd;
    int s1p = ss / d, s2p = ss % d;
    A[idx] = base_A + (n * dd + ss) * strideA;
    B[idx] = base_B + n * strideB;
    C[idx] = base_C_scratch + n * slice_stride + (s1p + s2p * d) * strideC_tile;
}

template<typename Scalar>
__global__ void setup_batch_ptrs_wd_twosite_sparse(Scalar** A, Scalar** B, Scalar** C,
                                                    Scalar* base_A, Scalar* base_B, Scalar* base_C,
                                                    const int* nnz_wss, int d, int dd,
                                                    int strideA, int strideB, int strideC) {
    int idx = threadIdx.x;
    int packed = nnz_wss[idx];
    int w = packed / dd;
    int ss = packed % dd;
    int s1 = ss / d, s2 = ss % d;
    A[idx] = base_A + w * strideA;
    B[idx] = base_B + (s1 + s2 * d) * strideB;
    C[idx] = base_C + packed * strideC;
}

template<typename Scalar>
__global__ void setup_batch_ptrs_step3_twosite_full_sparse(Scalar** A, Scalar** B, Scalar** C,
                                                           Scalar* base_A, Scalar* base_B, Scalar* base_C_scratch,
                                                           const int* nnz_nss, int d, int dd,
                                                           int strideA, int strideB,
                                                           int strideC_tile, int slice_stride) {
    int idx = threadIdx.x;
    int packed = nnz_nss[idx];
    int n = packed / dd;
    int ss = packed % dd;
    int s1p = ss / d, s2p = ss % d;
    A[idx] = base_A + packed * strideA;
    B[idx] = base_B + n * strideB;
    C[idx] = base_C_scratch + n * slice_stride + (s1p + s2p * d) * strideC_tile;
}

#endif // PMRG_BATCH_PTRS_KERNELS_H
