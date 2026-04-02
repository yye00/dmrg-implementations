#ifndef SCALAR_TRAITS_H
#define SCALAR_TRAITS_H

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cmath>
#include <cstdlib>

// ============================================================================
// ScalarTraits: type-dependent dispatch for cuBLAS/LAPACK routines
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
                        const int* m, const int* n, cuDoubleComplex* a, const int* lda,
                        double* s, cuDoubleComplex* u, const int* ldu,
                        cuDoubleComplex* vt, const int* ldvt,
                        cuDoubleComplex* work, const int* lwork,
                        double* rwork, int* info);

// Symmetric/Hermitian eigendecomposition (for Block-Davidson projected H + NS split)
extern "C" void dsyev_(const char* jobz, const char* uplo, const int* n,
                       double* a, const int* lda, double* w,
                       double* work, const int* lwork, int* info);

extern "C" void zheev_(const char* jobz, const char* uplo, const int* n,
                       cuDoubleComplex* a, const int* lda, double* w,
                       cuDoubleComplex* work, const int* lwork,
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
    static constexpr cublasOperation_t op_t = CUBLAS_OP_T;
    static constexpr cublasOperation_t op_h = CUBLAS_OP_T;  // same for real

    // --- cuBLAS dispatch ---

    static cublasStatus_t gemm(cublasHandle_t h, cublasOperation_t opA, cublasOperation_t opB,
            int m, int n, int k, const Scalar* alpha,
            const Scalar* A, int lda, const Scalar* B, int ldb,
            const Scalar* beta, Scalar* C, int ldc) {
        return cublasDgemm(h, opA, opB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    static cublasStatus_t gemm_batched(cublasHandle_t h, cublasOperation_t opA, cublasOperation_t opB,
            int m, int n, int k, const Scalar* alpha,
            const Scalar* const* A, int lda, const Scalar* const* B, int ldb,
            const Scalar* beta, Scalar** C, int ldc, int batch_count) {
        return cublasDgemmBatched(h, opA, opB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batch_count);
    }

    static cublasStatus_t gemm_strided_batched(cublasHandle_t h,
            cublasOperation_t opA, cublasOperation_t opB,
            int m, int n, int k, const Scalar* alpha,
            const Scalar* A, int lda, long long int strideA,
            const Scalar* B, int ldb, long long int strideB,
            const Scalar* beta, Scalar* C, int ldc, long long int strideC,
            int batch_count) {
        return cublasDgemmStridedBatched(h, opA, opB, m, n, k, alpha,
            A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batch_count);
    }

    static cublasStatus_t gemv(cublasHandle_t h, cublasOperation_t op,
            int m, int n, const Scalar* alpha, const Scalar* A, int lda,
            const Scalar* x, int incx, const Scalar* beta, Scalar* y, int incy) {
        return cublasDgemv(h, op, m, n, alpha, A, lda, x, incx, beta, y, incy);
    }

    static cublasStatus_t dot(cublasHandle_t h, int n, const Scalar* x, int incx,
            const Scalar* y, int incy, Scalar* result) {
        return cublasDdot(h, n, x, incx, y, incy, result);
    }

    static cublasStatus_t nrm2(cublasHandle_t h, int n, const Scalar* x, int incx, RealType* result) {
        return cublasDnrm2(h, n, x, incx, result);
    }

    static cublasStatus_t scal(cublasHandle_t h, int n, const Scalar* alpha, Scalar* x, int incx) {
        return cublasDscal(h, n, alpha, x, incx);
    }

    static cublasStatus_t scal_real(cublasHandle_t h, int n, const RealType* alpha, Scalar* x, int incx) {
        return cublasDscal(h, n, alpha, x, incx);
    }

    static cublasStatus_t axpy(cublasHandle_t h, int n, const Scalar* alpha,
            const Scalar* x, int incx, Scalar* y, int incy) {
        return cublasDaxpy(h, n, alpha, x, incx, y, incy);
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
};

// ============================================================================
// cuDoubleComplex specialization
// ============================================================================
template<>
struct ScalarTraits<cuDoubleComplex> {
    using Scalar = cuDoubleComplex;
    using RealType = double;
    static constexpr bool is_complex = true;

    static Scalar one()  { return make_cuDoubleComplex(1.0, 0.0); }
    static Scalar zero() { return make_cuDoubleComplex(0.0, 0.0); }
    static Scalar make_scalar(double re, double im = 0.0) { return make_cuDoubleComplex(re, im); }
    static RealType real_part(Scalar x) { return cuCreal(x); }
    static Scalar neg(Scalar x) { return make_cuDoubleComplex(-cuCreal(x), -cuCimag(x)); }

    static Scalar scale_by_real(RealType s, Scalar v) {
        return make_cuDoubleComplex(s * cuCreal(v), s * cuCimag(v));
    }

    static Scalar random_val() {
        return make_cuDoubleComplex(2.0 * rand() / RAND_MAX - 1.0,
                                     2.0 * rand() / RAND_MAX - 1.0);
    }

    static constexpr cublasOperation_t op_t = CUBLAS_OP_T;
    static constexpr cublasOperation_t op_h = CUBLAS_OP_C;

    // --- cuBLAS dispatch ---

    static cublasStatus_t gemm(cublasHandle_t h, cublasOperation_t opA, cublasOperation_t opB,
            int m, int n, int k, const Scalar* alpha,
            const Scalar* A, int lda, const Scalar* B, int ldb,
            const Scalar* beta, Scalar* C, int ldc) {
        return cublasZgemm(h, opA, opB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    static cublasStatus_t gemm_batched(cublasHandle_t h, cublasOperation_t opA, cublasOperation_t opB,
            int m, int n, int k, const Scalar* alpha,
            const Scalar* const* A, int lda, const Scalar* const* B, int ldb,
            const Scalar* beta, Scalar** C, int ldc, int batch_count) {
        return cublasZgemmBatched(h, opA, opB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batch_count);
    }

    static cublasStatus_t gemm_strided_batched(cublasHandle_t h,
            cublasOperation_t opA, cublasOperation_t opB,
            int m, int n, int k, const Scalar* alpha,
            const Scalar* A, int lda, long long int strideA,
            const Scalar* B, int ldb, long long int strideB,
            const Scalar* beta, Scalar* C, int ldc, long long int strideC,
            int batch_count) {
        return cublasZgemmStridedBatched(h, opA, opB, m, n, k, alpha,
            A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batch_count);
    }

    static cublasStatus_t gemv(cublasHandle_t h, cublasOperation_t op,
            int m, int n, const Scalar* alpha, const Scalar* A, int lda,
            const Scalar* x, int incx, const Scalar* beta, Scalar* y, int incy) {
        return cublasZgemv(h, op, m, n, alpha, A, lda, x, incx, beta, y, incy);
    }

    static cublasStatus_t dot(cublasHandle_t h, int n, const Scalar* x, int incx,
            const Scalar* y, int incy, Scalar* result) {
        return cublasZdotc(h, n, x, incx, y, incy, result);
    }

    static cublasStatus_t nrm2(cublasHandle_t h, int n, const Scalar* x, int incx, RealType* result) {
        return cublasDznrm2(h, n, x, incx, result);
    }

    static cublasStatus_t scal(cublasHandle_t h, int n, const Scalar* alpha, Scalar* x, int incx) {
        return cublasZscal(h, n, alpha, x, incx);
    }

    static cublasStatus_t scal_real(cublasHandle_t h, int n, const RealType* alpha, Scalar* x, int incx) {
        return cublasZdscal(h, n, alpha, x, incx);
    }

    static cublasStatus_t axpy(cublasHandle_t h, int n, const Scalar* alpha,
            const Scalar* x, int incx, Scalar* y, int incy) {
        return cublasZaxpy(h, n, alpha, x, incx, y, incy);
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
};

// ============================================================================
// Matrix transpose kernel: out(n,m) = in(m,n)^T  (column-major)
// ============================================================================

template<typename Scalar>
__global__ void transpose_kernel(const Scalar* __restrict__ in, int lda_in,
                                  Scalar* __restrict__ out, int lda_out,
                                  int rows_in, int cols_in) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows_in * cols_in;
    if (idx < total) {
        int i = idx % rows_in;  // row in input
        int j = idx / rows_in;  // col in input
        // out[j + i * lda_out] = in[i + j * lda_in]
        out[j + i * lda_out] = in[i + j * lda_in];
    }
}

// ============================================================================
// In-place conjugation of GPU arrays (no-op for real)
// ============================================================================

static __global__ void conjugate_complex_kernel(cuDoubleComplex* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = cuConj(data[i]);
    }
}

static inline void conjugate_inplace(double* /*data*/, int /*n*/, cudaStream_t /*stream*/) {
    // no-op for real
}

static inline void conjugate_inplace(cuDoubleComplex* data, int n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    conjugate_complex_kernel<<<dim3(grid), dim3(block), 0, stream>>>(data, n);
}

// ============================================================================
// Newton-Schulz kernel: compute data[idx] = alpha*I[idx] - data[idx]
// ============================================================================

static __global__ void scaled_identity_minus_double(double* data, int n, double alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * n) {
        int row = idx % n;
        int col = idx / n;
        double diag = (row == col) ? alpha : 0.0;
        data[idx] = diag - data[idx];
    }
}

static __global__ void scaled_identity_minus_complex(cuDoubleComplex* data, int n, double alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * n) {
        int row = idx % n;
        int col = idx / n;
        double diag = (row == col) ? alpha : 0.0;
        data[idx] = make_cuDoubleComplex(diag - cuCreal(data[idx]), -cuCimag(data[idx]));
    }
}

static inline void launch_scaled_identity_minus(double* data, int n, double alpha, cudaStream_t stream) {
    int block = 256;
    int grid = (n * n + block - 1) / block;
    scaled_identity_minus_double<<<dim3(grid), dim3(block), 0, stream>>>(data, n, alpha);
}

static inline void launch_scaled_identity_minus(cuDoubleComplex* data, int n, double alpha, cudaStream_t stream) {
    int block = 256;
    int grid = (n * n + block - 1) / block;
    scaled_identity_minus_complex<<<dim3(grid), dim3(block), 0, stream>>>(data, n, alpha);
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
            C[i + j * ldc] = make_cuDoubleComplex(s * cuCreal(a), s * cuCimag(a));
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
            C[i + j * ldc] = make_cuDoubleComplex(s * cuCreal(a), s * cuCimag(a));
        }
    }
}

// Helper to launch column-scale: C = A * diag(d)
template<typename Scalar>
inline void scale_columns_by_real(const Scalar* A, int lda, const double* d_diag,
                                   Scalar* C, int ldc, int m, int n, cudaStream_t stream) {
    int total = m * n;
    int block = 256;
    int grid = (total + block - 1) / block;
    scale_columns_by_real_kernel<Scalar><<<dim3(grid), dim3(block), 0, stream>>>(
                       A, lda, d_diag, C, ldc, m, n);
}

// Helper to launch row-scale: C = diag(d) * A
template<typename Scalar>
inline void scale_rows_by_real(const Scalar* A, int lda, const double* d_diag,
                                Scalar* C, int ldc, int m, int n, cudaStream_t stream) {
    int total = m * n;
    int block = 256;
    int grid = (total + block - 1) / block;
    scale_rows_by_real_kernel<Scalar><<<dim3(grid), dim3(block), 0, stream>>>(
                       A, lda, d_diag, C, ldc, m, n);
}

#endif // SCALAR_TRAITS_H
