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
// Extended with eigendecomposition for Block-Davidson + Newton-Schulz
// ============================================================================

// LAPACK extern declarations
extern "C" void dstev_(const char* jobz, const int* n, double* d, double* e,
                       double* z, const int* ldz, double* work, int* info);

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

// Symmetric/Hermitian eigendecomposition (for Block-Davidson projected H + NS split)
extern "C" void dsyev_(const char* jobz, const char* uplo, const int* n,
                       double* a, const int* lda, double* w,
                       double* work, const int* lwork, int* info);

extern "C" void zheev_(const char* jobz, const char* uplo, const int* n,
                       hipDoubleComplex* a, const int* lda, double* w,
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
    static Scalar scale_by_real(RealType s, Scalar v) { return s * v; }

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

    // --- LAPACK eigendecomposition ---
    static void lapack_syev(const char* jobz, const char* uplo, const int* n,
            double* a, const int* lda, double* w,
            double* work, const int* lwork, double* /*rwork*/, int* info) {
        dsyev_(jobz, uplo, n, a, lda, w, work, lwork, info);
    }
    static int syev_rwork_size(int /*n*/) { return 0; }

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

    static Scalar scale_by_real(RealType s, Scalar v) {
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

    // --- LAPACK eigendecomposition ---
    static void lapack_syev(const char* jobz, const char* uplo, const int* n,
            Scalar* a, const int* lda, RealType* w,
            Scalar* work, const int* lwork, RealType* rwork, int* info) {
        zheev_(jobz, uplo, n, a, lda, w, work, lwork, rwork, info);
    }
    static int syev_rwork_size(int n) { return std::max(1, 3*n - 2); }

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

// ============================================================================
// Newton-Schulz kernel: compute data[idx] = alpha*I[idx] - data[idx]
// ============================================================================

__global__ inline void scaled_identity_minus_double(double* data, int n, double alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * n) {
        int row = idx % n;
        int col = idx / n;
        double diag = (row == col) ? alpha : 0.0;
        data[idx] = diag - data[idx];
    }
}

__global__ inline void scaled_identity_minus_complex(hipDoubleComplex* data, int n, double alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * n) {
        int row = idx % n;
        int col = idx / n;
        double diag = (row == col) ? alpha : 0.0;
        data[idx] = make_hipDoubleComplex(diag - hipCreal(data[idx]), -hipCimag(data[idx]));
    }
}

inline void launch_scaled_identity_minus(double* data, int n, double alpha, hipStream_t stream) {
    int block = 256;
    int grid = (n * n + block - 1) / block;
    hipLaunchKernelGGL(scaled_identity_minus_double, dim3(grid), dim3(block), 0, stream, data, n, alpha);
}

inline void launch_scaled_identity_minus(hipDoubleComplex* data, int n, double alpha, hipStream_t stream) {
    int block = 256;
    int grid = (n * n + block - 1) / block;
    hipLaunchKernelGGL(scaled_identity_minus_complex, dim3(grid), dim3(block), 0, stream, data, n, alpha);
}

// ============================================================================
// GPU-side diagonal scaling kernels for SVD factor absorption
// ============================================================================

// Scale columns of A (m x n, column-major) by real diagonal d:
//   C[i, j] = A[i, j] * d[j]
template<typename Scalar>
__global__ void scale_columns_by_real_kernel(const Scalar* __restrict__ A, int lda,
                                              const double* __restrict__ d_diag,
                                              Scalar* __restrict__ C, int ldc,
                                              int m, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = m * n;
    if (idx < total) {
        int i = idx % m;   // row
        int j = idx / m;   // column
        double s = d_diag[j];
        Scalar a = A[i + j * lda];
        if constexpr (sizeof(Scalar) == sizeof(double)) {
            C[i + j * ldc] = s * a;
        } else {
            C[i + j * ldc] = make_hipDoubleComplex(s * hipCreal(a), s * hipCimag(a));
        }
    }
}

// Scale rows of A (m x n, column-major) by real diagonal d:
//   C[i, j] = d[i] * A[i, j]
template<typename Scalar>
__global__ void scale_rows_by_real_kernel(const Scalar* __restrict__ A, int lda,
                                           const double* __restrict__ d_diag,
                                           Scalar* __restrict__ C, int ldc,
                                           int m, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = m * n;
    if (idx < total) {
        int i = idx % m;   // row
        int j = idx / m;   // column
        double s = d_diag[i];
        Scalar a = A[i + j * lda];
        if constexpr (sizeof(Scalar) == sizeof(double)) {
            C[i + j * ldc] = s * a;
        } else {
            C[i + j * ldc] = make_hipDoubleComplex(s * hipCreal(a), s * hipCimag(a));
        }
    }
}

// Helper to launch column-scale: C = A * diag(d)
template<typename Scalar>
inline void scale_columns_by_real(const Scalar* A, int lda, const double* d_diag,
                                   Scalar* C, int ldc, int m, int n, hipStream_t stream) {
    int total = m * n;
    int block = 256;
    int grid = (total + block - 1) / block;
    hipLaunchKernelGGL(scale_columns_by_real_kernel<Scalar>,
                       dim3(grid), dim3(block), 0, stream,
                       A, lda, d_diag, C, ldc, m, n);
}

// Helper to launch row-scale: C = diag(d) * A
template<typename Scalar>
inline void scale_rows_by_real(const Scalar* A, int lda, const double* d_diag,
                                Scalar* C, int ldc, int m, int n, hipStream_t stream) {
    int total = m * n;
    int block = 256;
    int grid = (total + block - 1) / block;
    hipLaunchKernelGGL(scale_rows_by_real_kernel<Scalar>,
                       dim3(grid), dim3(block), 0, stream,
                       A, lda, d_diag, C, ldc, m, n);
}

#endif // SCALAR_TRAITS_H
