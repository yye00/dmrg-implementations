#ifndef SCALAR_TRAITS_H
#define SCALAR_TRAITS_H

#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>
#include <cmath>
#include <cstdlib>

// ============================================================================
// ScalarTraits — naive baseline (no custom kernels, CPU LAPACK for dstev,
// host-pointer rocBLAS throughout). Single-site + two-site DMRG share this.
// ============================================================================

// --- LAPACK extern declarations ---
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

// Tridiagonal symmetric eigensolver (CPU LAPACK) — used in the naive Lanczos
extern "C" void dstev_(const char* jobz, const int* n, double* d, double* e,
                       double* z, const int* ldz, double* work, int* info);


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

    static Scalar scale_by_real(RealType s, Scalar v) { return s * v; }

    static Scalar random_val() { return 2.0 * rand() / RAND_MAX - 1.0; }

    static constexpr rocblas_operation op_t = rocblas_operation_transpose;
    static constexpr rocblas_operation op_h = rocblas_operation_transpose;

    // --- rocBLAS dispatch (host pointer mode throughout) ---

    static rocblas_status gemm(rocblas_handle h, rocblas_operation opA, rocblas_operation opB,
            int m, int n, int k, const Scalar* alpha,
            const Scalar* A, int lda, const Scalar* B, int ldb,
            const Scalar* beta, Scalar* C, int ldc) {
        return rocblas_dgemm(h, opA, opB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
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

    // --- rocSOLVER SVD (naive baseline uses GPU SVD, no CPU fallback) ---
    static rocblas_status rocsolver_gesvd(rocblas_handle h,
            rocblas_svect lu, rocblas_svect rv,
            int m, int n, Scalar* A, int lda, RealType* S,
            Scalar* U, int ldu, Scalar* Vh, int ldvh,
            RealType* E, rocblas_workmode wm, int* info) {
        return rocsolver_dgesvd(h, lu, rv, m, n, A, lda, S, U, ldu, Vh, ldvh, E, wm, info);
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
};

// ============================================================================
// In-place conjugation of GPU arrays (for complex env updates)
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
