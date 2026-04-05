#ifndef SCALAR_TRAITS_H
#define SCALAR_TRAITS_H

#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>
#include <cmath>
#include <cstdlib>

// ============================================================================
// ScalarTraits: type-dependent dispatch for rocBLAS/LAPACK routines
// ============================================================================

// LAPACK extern declarations (SVD only — tridiagonal solve uses rocsolver_dsteqr)
extern "C" void dgesvd_(const char* jobu, const char* jobvt,
                        const int* m, const int* n, double* a, const int* lda,
                        double* s, double* u, const int* ldu,
                        double* vt, const int* ldvt,
                        double* work, const int* lwork, int* info);

extern "C" void zgesvd_(const char* jobu, const char* jobvt,
                        const int* m, const int* n, hipDoubleComplex* a, const int* lda,
                        double* s, hipDoubleComplex* u, const int* ldu,
                        hipDoubleComplex* vt, const int* ldvt,
                        hipDoubleComplex* work, const int* lwork,
                        double* rwork, int* info);

template<typename T> struct ScalarTraits;

// ============================================================================
// double specialization
// ============================================================================
template<>
struct ScalarTraits<double> {
    using Scalar = double;
    using RealType = double;
    static constexpr bool is_complex = false;

    static Scalar one()  { return 1.0; }
    static Scalar zero() { return 0.0; }
    static Scalar make_scalar(double re, double /*im*/ = 0.0) { return re; }
    static RealType real_part(Scalar x) { return x; }
    static Scalar neg(Scalar x) { return -x; }

    // Scale a scalar by a real value
    static __host__ __device__ Scalar scale_by_real(RealType s, Scalar v) { return s * v; }

    // Random value for initialization
    static Scalar random_val() { return 2.0 * rand() / RAND_MAX - 1.0; }

    // Transpose operations
    static constexpr rocblas_operation op_t = rocblas_operation_transpose;
    static constexpr rocblas_operation op_h = rocblas_operation_transpose;  // same for real

    // --- rocBLAS dispatch ---

    static rocblas_status gemm(rocblas_handle h, rocblas_operation opA, rocblas_operation opB,
            int m, int n, int k, const Scalar* alpha,
            const Scalar* A, int lda, const Scalar* B, int ldb,
            const Scalar* beta, Scalar* C, int ldc) {
        return rocblas_dgemm(h, opA, opB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    static rocblas_status gemm_batched(rocblas_handle h, rocblas_operation opA, rocblas_operation opB,
            int m, int n, int k, const Scalar* alpha,
            const Scalar* const* A, int lda, const Scalar* const* B, int ldb,
            const Scalar* beta, Scalar** C, int ldc, int batch_count) {
        return rocblas_dgemm_batched(h, opA, opB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batch_count);
    }

    static rocblas_status gemv(rocblas_handle h, rocblas_operation op,
            int m, int n, const Scalar* alpha, const Scalar* A, int lda,
            const Scalar* x, int incx, const Scalar* beta, Scalar* y, int incy) {
        return rocblas_dgemv(h, op, m, n, alpha, A, lda, x, incx, beta, y, incy);
    }

    static rocblas_status dot(rocblas_handle h, int n, const Scalar* x, int incx,
            const Scalar* y, int incy, Scalar* result) {
        return rocblas_ddot(h, n, x, incx, y, incy, result);
    }

    static rocblas_status nrm2(rocblas_handle h, int n, const Scalar* x, int incx, RealType* result) {
        return rocblas_dnrm2(h, n, x, incx, result);
    }

    static rocblas_status scal(rocblas_handle h, int n, const Scalar* alpha, Scalar* x, int incx) {
        return rocblas_dscal(h, n, alpha, x, incx);
    }

    static rocblas_status scal_real(rocblas_handle h, int n, const RealType* alpha, Scalar* x, int incx) {
        return rocblas_dscal(h, n, alpha, x, incx);
    }

    static rocblas_status axpy(rocblas_handle h, int n, const Scalar* alpha,
            const Scalar* x, int incx, Scalar* y, int incy) {
        return rocblas_daxpy(h, n, alpha, x, incx, y, incy);
    }

    // --- LAPACK SVD ---
    // rwork is unused for real; rwork_size returns 0
    static int svd_rwork_size(int /*m*/, int /*n*/) { return 0; }

    static void lapack_gesvd(const char* jobu, const char* jobvt,
            const int* m, const int* n, Scalar* a, const int* lda,
            RealType* s, Scalar* u, const int* ldu, Scalar* vt, const int* ldvt,
            Scalar* work, const int* lwork, RealType* /*rwork*/, int* info) {
        dgesvd_(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info);
    }

    // --- rocSOLVER SVD ---
    static rocblas_status rocsolver_gesvd(rocblas_handle h,
            rocblas_svect lu, rocblas_svect rv,
            int m, int n, Scalar* A, int lda, RealType* S,
            Scalar* U, int ldu, Scalar* Vh, int ldvh,
            RealType* E, rocblas_workmode wm, int* info) {
        return rocsolver_dgesvd(h, lu, rv, m, n, A, lda, S, U, ldu, Vh, ldvh, E, wm, info);
    }

    // --- rocSOLVER QR ---
    static rocblas_status rocsolver_geqrf(rocblas_handle h,
            int m, int n, Scalar* A, int lda, Scalar* ipiv) {
        return rocsolver_dgeqrf(h, m, n, A, lda, ipiv);
    }
    static rocblas_status rocsolver_orgqr(rocblas_handle h,
            int m, int n, int k, Scalar* A, int lda, Scalar* ipiv) {
        return rocsolver_dorgqr(h, m, n, k, A, lda, ipiv);
    }
};

// ============================================================================
// hipDoubleComplex specialization
// ============================================================================
template<>
struct ScalarTraits<hipDoubleComplex> {
    using Scalar = hipDoubleComplex;
    using RealType = double;
    static constexpr bool is_complex = true;

    static Scalar one()  { return make_hipDoubleComplex(1.0, 0.0); }
    static Scalar zero() { return make_hipDoubleComplex(0.0, 0.0); }
    static Scalar make_scalar(double re, double im = 0.0) { return make_hipDoubleComplex(re, im); }
    static RealType real_part(Scalar x) { return hipCreal(x); }
    static Scalar neg(Scalar x) { return make_hipDoubleComplex(-hipCreal(x), -hipCimag(x)); }

    static __host__ __device__ Scalar scale_by_real(RealType s, Scalar v) {
        return make_hipDoubleComplex(s * hipCreal(v), s * hipCimag(v));
    }

    static Scalar random_val() {
        return make_hipDoubleComplex(2.0 * rand() / RAND_MAX - 1.0,
                                     2.0 * rand() / RAND_MAX - 1.0);
    }

    static constexpr rocblas_operation op_t = rocblas_operation_transpose;
    static constexpr rocblas_operation op_h = rocblas_operation_conjugate_transpose;

    // --- rocBLAS dispatch ---

    static rocblas_status gemm(rocblas_handle h, rocblas_operation opA, rocblas_operation opB,
            int m, int n, int k, const Scalar* alpha,
            const Scalar* A, int lda, const Scalar* B, int ldb,
            const Scalar* beta, Scalar* C, int ldc) {
        return rocblas_zgemm(h, opA, opB, m, n, k,
            reinterpret_cast<const rocblas_double_complex*>(alpha),
            reinterpret_cast<const rocblas_double_complex*>(A), lda,
            reinterpret_cast<const rocblas_double_complex*>(B), ldb,
            reinterpret_cast<const rocblas_double_complex*>(beta),
            reinterpret_cast<rocblas_double_complex*>(C), ldc);
    }

    static rocblas_status gemm_batched(rocblas_handle h, rocblas_operation opA, rocblas_operation opB,
            int m, int n, int k, const Scalar* alpha,
            const Scalar* const* A, int lda, const Scalar* const* B, int ldb,
            const Scalar* beta, Scalar** C, int ldc, int batch_count) {
        return rocblas_zgemm_batched(h, opA, opB, m, n, k,
            reinterpret_cast<const rocblas_double_complex*>(alpha),
            reinterpret_cast<const rocblas_double_complex* const*>(A), lda,
            reinterpret_cast<const rocblas_double_complex* const*>(B), ldb,
            reinterpret_cast<const rocblas_double_complex*>(beta),
            reinterpret_cast<rocblas_double_complex**>(C), ldc, batch_count);
    }

    static rocblas_status gemv(rocblas_handle h, rocblas_operation op,
            int m, int n, const Scalar* alpha, const Scalar* A, int lda,
            const Scalar* x, int incx, const Scalar* beta, Scalar* y, int incy) {
        return rocblas_zgemv(h, op, m, n,
            reinterpret_cast<const rocblas_double_complex*>(alpha),
            reinterpret_cast<const rocblas_double_complex*>(A), lda,
            reinterpret_cast<const rocblas_double_complex*>(x), incx,
            reinterpret_cast<const rocblas_double_complex*>(beta),
            reinterpret_cast<rocblas_double_complex*>(y), incy);
    }

    static rocblas_status dot(rocblas_handle h, int n, const Scalar* x, int incx,
            const Scalar* y, int incy, Scalar* result) {
        return rocblas_zdotc(h, n,
            reinterpret_cast<const rocblas_double_complex*>(x), incx,
            reinterpret_cast<const rocblas_double_complex*>(y), incy,
            reinterpret_cast<rocblas_double_complex*>(result));
    }

    static rocblas_status nrm2(rocblas_handle h, int n, const Scalar* x, int incx, RealType* result) {
        return rocblas_dznrm2(h, n,
            reinterpret_cast<const rocblas_double_complex*>(x), incx, result);
    }

    static rocblas_status scal(rocblas_handle h, int n, const Scalar* alpha, Scalar* x, int incx) {
        return rocblas_zscal(h, n,
            reinterpret_cast<const rocblas_double_complex*>(alpha),
            reinterpret_cast<rocblas_double_complex*>(x), incx);
    }

    static rocblas_status scal_real(rocblas_handle h, int n, const RealType* alpha, Scalar* x, int incx) {
        return rocblas_zdscal(h, n, alpha,
            reinterpret_cast<rocblas_double_complex*>(x), incx);
    }

    static rocblas_status axpy(rocblas_handle h, int n, const Scalar* alpha,
            const Scalar* x, int incx, Scalar* y, int incy) {
        return rocblas_zaxpy(h, n,
            reinterpret_cast<const rocblas_double_complex*>(alpha),
            reinterpret_cast<const rocblas_double_complex*>(x), incx,
            reinterpret_cast<rocblas_double_complex*>(y), incy);
    }

    // --- LAPACK SVD ---
    static int svd_rwork_size(int m, int n) { return 5 * std::min(m, n); }

    static void lapack_gesvd(const char* jobu, const char* jobvt,
            const int* m, const int* n, Scalar* a, const int* lda,
            RealType* s, Scalar* u, const int* ldu, Scalar* vt, const int* ldvt,
            Scalar* work, const int* lwork, RealType* rwork, int* info) {
        zgesvd_(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, info);
    }

    // --- rocSOLVER SVD ---
    static rocblas_status rocsolver_gesvd(rocblas_handle h,
            rocblas_svect lu, rocblas_svect rv,
            int m, int n, Scalar* A, int lda, RealType* S,
            Scalar* U, int ldu, Scalar* Vh, int ldvh,
            RealType* E, rocblas_workmode wm, int* info) {
        return rocsolver_zgesvd(h, lu, rv, m, n,
            reinterpret_cast<rocblas_double_complex*>(A), lda, S,
            reinterpret_cast<rocblas_double_complex*>(U), ldu,
            reinterpret_cast<rocblas_double_complex*>(Vh), ldvh,
            E, wm, info);
    }

    // --- rocSOLVER QR ---
    static rocblas_status rocsolver_geqrf(rocblas_handle h,
            int m, int n, Scalar* A, int lda, Scalar* ipiv) {
        return rocsolver_zgeqrf(h, m, n,
            reinterpret_cast<rocblas_double_complex*>(A), lda,
            reinterpret_cast<rocblas_double_complex*>(ipiv));
    }
    static rocblas_status rocsolver_orgqr(rocblas_handle h,
            int m, int n, int k, Scalar* A, int lda, Scalar* ipiv) {
        return rocsolver_zungqr(h, m, n, k,
            reinterpret_cast<rocblas_double_complex*>(A), lda,
            reinterpret_cast<rocblas_double_complex*>(ipiv));
    }
};

// ============================================================================
// In-place conjugation of GPU arrays (no-op for real)
// ============================================================================
// Device helpers for Lanczos device-pointer-mode operations
// ============================================================================

__device__ inline double dev_real_part(double x) { return x; }
__device__ inline double dev_real_part(hipDoubleComplex x) { return hipCreal(x); }

__device__ inline double dev_make_neg_real_scalar(double, double neg_alpha) { return neg_alpha; }
__device__ inline hipDoubleComplex dev_make_neg_real_scalar(hipDoubleComplex, double neg_alpha) {
    return make_hipDoubleComplex(neg_alpha, 0.0);
}

template<typename Scalar>
__global__ void lanczos_process_alpha_kernel(const Scalar* dot_result, Scalar* neg_alpha_out,
                                              double* alpha_arr, int iter) {
    double alpha = dev_real_part(dot_result[0]);
    alpha_arr[iter] = alpha;
    neg_alpha_out[0] = dev_make_neg_real_scalar(dot_result[0], -alpha);
}

template<typename Scalar>
__global__ void lanczos_process_beta_kernel(const double* nrm2_result, double* inv_nrm_out,
                                             double* beta_arr, Scalar* neg_beta_scalars, int iter) {
    double beta = nrm2_result[0];
    beta_arr[iter] = beta;
    inv_nrm_out[0] = 1.0 / beta;
    neg_beta_scalars[iter] = dev_make_neg_real_scalar(Scalar{}, -beta);
}

__device__ inline double dev_negate(double x) { return -x; }
__device__ inline hipDoubleComplex dev_negate(hipDoubleComplex x) {
    return make_hipDoubleComplex(-hipCreal(x), -hipCimag(x));
}

template<typename Scalar>
__global__ void negate_scalar_kernel(const Scalar* in, Scalar* out) {
    out[0] = dev_negate(in[0]);
}

template<typename RealType>
__global__ void invert_nrm_kernel(const RealType* nrm, RealType* inv_nrm) {
    inv_nrm[0] = RealType(1.0) / nrm[0];
}

// ============================================================================
// SVD post-processing kernels: keep U/S/Vh on GPU, avoid D2H→CPU→H2D roundtrip
// ============================================================================

// Find truncation point: first index where S[i] < tol, clamped to [1, max_k]
// Writes result to d_new_k[0]
template<typename RealType>
__global__ void svd_truncate_kernel(const RealType* S, int max_k, double tol, int* d_new_k) {
    int new_k = max_k;
    for (int i = 0; i < max_k; i++) {
        if (S[i] < tol) { new_k = i; break; }
    }
    if (new_k == 0) new_k = 1;
    d_new_k[0] = new_k;
}

// Scale rows of A by diagonal S: C[i + j*ldc] = S[i] * A[i + j*lda]
// Used for S*Vh where S scales rows of Vh
template<typename Scalar, typename RealType>
__global__ void scale_rows_by_diag_kernel(const RealType* S, const Scalar* A, int lda,
                                           Scalar* C, int ldc, int nrows, int ncols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nrows * ncols;
    if (idx < total) {
        int i = idx % nrows;  // row
        int j = idx / nrows;  // col
        C[i + j * ldc] = ScalarTraits<Scalar>::scale_by_real(S[i], A[i + j * lda]);
    }
}

// Scale columns of A by diagonal S: C[i + j*ldc] = S[j] * A[i + j*lda]
// Used for U*S where S scales columns of U
template<typename Scalar, typename RealType>
__global__ void scale_cols_by_diag_kernel(const RealType* S, const Scalar* A, int lda,
                                           Scalar* C, int ldc, int nrows, int ncols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nrows * ncols;
    if (idx < total) {
        int i = idx % nrows;  // row
        int j = idx / nrows;  // col
        C[i + j * ldc] = ScalarTraits<Scalar>::scale_by_real(S[j], A[i + j * lda]);
    }
}

// Extract first new_k columns from (m, full_k) with lda=lda_in to contiguous (m, new_k)
template<typename Scalar>
__global__ void extract_cols_kernel(const Scalar* in, int lda_in,
                                     Scalar* out, int lda_out,
                                     int nrows, int ncols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nrows * ncols;
    if (idx < total) {
        int i = idx % nrows;
        int j = idx / nrows;
        out[i + j * lda_out] = in[i + j * lda_in];
    }
}

// ============================================================================

__global__ inline void conjugate_complex_kernel(hipDoubleComplex* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = hipConj(data[i]);
    }
}

inline void conjugate_inplace(double* /*data*/, int /*n*/, hipStream_t /*stream*/) {
    // no-op for real
}

inline void conjugate_inplace(hipDoubleComplex* data, int n, hipStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    hipLaunchKernelGGL(conjugate_complex_kernel, dim3(grid), dim3(block), 0, stream, data, n);
}

#endif // SCALAR_TRAITS_H

