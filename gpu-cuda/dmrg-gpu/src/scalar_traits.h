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

    // --- cuSOLVER SVD ---
    static cusolverStatus_t cusolver_gesvd_bufferSize(cusolverDnHandle_t h,
            int m, int n, int* lwork) {
        return cusolverDnDgesvd_bufferSize(h, m, n, lwork);
    }

    static cusolverStatus_t cusolver_gesvd(cusolverDnHandle_t h,
            signed char jobu, signed char jobvt,
            int m, int n, Scalar* A, int lda, RealType* S,
            Scalar* U, int ldu, Scalar* Vh, int ldvh,
            Scalar* work, int lwork, RealType* /*rwork*/, int* devInfo) {
        return cusolverDnDgesvd(h, jobu, jobvt, m, n, A, lda, S, U, ldu, Vh, ldvh,
                                work, lwork, nullptr, devInfo);
    }

    // --- cuSOLVER QR ---
    static cusolverStatus_t cusolver_geqrf_bufferSize(cusolverDnHandle_t h,
            int m, int n, Scalar* A, int lda, int* lwork) {
        return cusolverDnDgeqrf_bufferSize(h, m, n, A, lda, lwork);
    }

    static cusolverStatus_t cusolver_geqrf(cusolverDnHandle_t h,
            int m, int n, Scalar* A, int lda, Scalar* tau,
            Scalar* work, int lwork, int* devInfo) {
        return cusolverDnDgeqrf(h, m, n, A, lda, tau, work, lwork, devInfo);
    }

    static cusolverStatus_t cusolver_orgqr_bufferSize(cusolverDnHandle_t h,
            int m, int n, int k, Scalar* A, int lda, const Scalar* tau, int* lwork) {
        return cusolverDnDorgqr_bufferSize(h, m, n, k, A, lda, tau, lwork);
    }

    static cusolverStatus_t cusolver_orgqr(cusolverDnHandle_t h,
            int m, int n, int k, Scalar* A, int lda, const Scalar* tau,
            Scalar* work, int lwork, int* devInfo) {
        return cusolverDnDorgqr(h, m, n, k, A, lda, tau, work, lwork, devInfo);
    }
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

    static __host__ __device__ Scalar scale_by_real(RealType s, Scalar v) {
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

    // --- cuSOLVER SVD ---
    static cusolverStatus_t cusolver_gesvd_bufferSize(cusolverDnHandle_t h,
            int m, int n, int* lwork) {
        return cusolverDnZgesvd_bufferSize(h, m, n, lwork);
    }

    static cusolverStatus_t cusolver_gesvd(cusolverDnHandle_t h,
            signed char jobu, signed char jobvt,
            int m, int n, Scalar* A, int lda, RealType* S,
            Scalar* U, int ldu, Scalar* Vh, int ldvh,
            Scalar* work, int lwork, RealType* rwork, int* devInfo) {
        return cusolverDnZgesvd(h, jobu, jobvt, m, n, A, lda, S, U, ldu, Vh, ldvh,
                                work, lwork, rwork, devInfo);
    }

    // --- cuSOLVER QR ---
    static cusolverStatus_t cusolver_geqrf_bufferSize(cusolverDnHandle_t h,
            int m, int n, Scalar* A, int lda, int* lwork) {
        return cusolverDnZgeqrf_bufferSize(h, m, n, A, lda, lwork);
    }

    static cusolverStatus_t cusolver_geqrf(cusolverDnHandle_t h,
            int m, int n, Scalar* A, int lda, Scalar* tau,
            Scalar* work, int lwork, int* devInfo) {
        return cusolverDnZgeqrf(h, m, n, A, lda, tau, work, lwork, devInfo);
    }

    static cusolverStatus_t cusolver_orgqr_bufferSize(cusolverDnHandle_t h,
            int m, int n, int k, Scalar* A, int lda, const Scalar* tau, int* lwork) {
        return cusolverDnZungqr_bufferSize(h, m, n, k, A, lda, tau, lwork);
    }

    static cusolverStatus_t cusolver_orgqr(cusolverDnHandle_t h,
            int m, int n, int k, Scalar* A, int lda, const Scalar* tau,
            Scalar* work, int lwork, int* devInfo) {
        return cusolverDnZungqr(h, m, n, k, A, lda, tau, work, lwork, devInfo);
    }
};

// ============================================================================
// In-place conjugation of GPU arrays (no-op for real)
// ============================================================================
// Device helpers for Lanczos device-pointer-mode operations
// ============================================================================

__device__ inline double dev_real_part(double x) { return x; }
__device__ inline double dev_real_part(cuDoubleComplex x) { return cuCreal(x); }

__device__ inline double dev_make_neg_real_scalar(double, double neg_alpha) { return neg_alpha; }
__device__ inline cuDoubleComplex dev_make_neg_real_scalar(cuDoubleComplex, double neg_alpha) {
    return make_cuDoubleComplex(neg_alpha, 0.0);
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
__device__ inline cuDoubleComplex dev_negate(cuDoubleComplex x) {
    return make_cuDoubleComplex(-cuCreal(x), -cuCimag(x));
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
// SVD post-processing kernels: keep U/S/Vh on GPU, avoid D2H->CPU->H2D roundtrip
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
    conjugate_complex_kernel<<<grid, block, 0, stream>>>(data, n);
}

#endif // SCALAR_TRAITS_H
