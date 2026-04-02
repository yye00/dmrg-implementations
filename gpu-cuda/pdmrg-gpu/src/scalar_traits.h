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

    static Scalar scale_by_real(RealType s, Scalar v) { return s * v; }

    static Scalar random_val() { return 2.0 * rand() / RAND_MAX - 1.0; }

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
            int m, int n, int k, Scalar* A, int lda, Scalar* tau, int* lwork) {
        return cusolverDnDorgqr_bufferSize(h, m, n, k, A, lda, tau, lwork);
    }

    static cusolverStatus_t cusolver_orgqr(cusolverDnHandle_t h,
            int m, int n, int k, Scalar* A, int lda, Scalar* tau,
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
        return cublasZgemm(h, opA, opB, m, n, k,
            reinterpret_cast<const cuDoubleComplex*>(alpha),
            reinterpret_cast<const cuDoubleComplex*>(A), lda,
            reinterpret_cast<const cuDoubleComplex*>(B), ldb,
            reinterpret_cast<const cuDoubleComplex*>(beta),
            reinterpret_cast<cuDoubleComplex*>(C), ldc);
    }

    static cublasStatus_t gemm_batched(cublasHandle_t h, cublasOperation_t opA, cublasOperation_t opB,
            int m, int n, int k, const Scalar* alpha,
            const Scalar* const* A, int lda, const Scalar* const* B, int ldb,
            const Scalar* beta, Scalar** C, int ldc, int batch_count) {
        return cublasZgemmBatched(h, opA, opB, m, n, k,
            reinterpret_cast<const cuDoubleComplex*>(alpha),
            reinterpret_cast<const cuDoubleComplex* const*>(A), lda,
            reinterpret_cast<const cuDoubleComplex* const*>(B), ldb,
            reinterpret_cast<const cuDoubleComplex*>(beta),
            reinterpret_cast<cuDoubleComplex**>(C), ldc, batch_count);
    }

    static cublasStatus_t gemv(cublasHandle_t h, cublasOperation_t op,
            int m, int n, const Scalar* alpha, const Scalar* A, int lda,
            const Scalar* x, int incx, const Scalar* beta, Scalar* y, int incy) {
        return cublasZgemv(h, op, m, n,
            reinterpret_cast<const cuDoubleComplex*>(alpha),
            reinterpret_cast<const cuDoubleComplex*>(A), lda,
            reinterpret_cast<const cuDoubleComplex*>(x), incx,
            reinterpret_cast<const cuDoubleComplex*>(beta),
            reinterpret_cast<cuDoubleComplex*>(y), incy);
    }

    static cublasStatus_t dot(cublasHandle_t h, int n, const Scalar* x, int incx,
            const Scalar* y, int incy, Scalar* result) {
        return cublasZdotc(h, n,
            reinterpret_cast<const cuDoubleComplex*>(x), incx,
            reinterpret_cast<const cuDoubleComplex*>(y), incy,
            reinterpret_cast<cuDoubleComplex*>(result));
    }

    static cublasStatus_t nrm2(cublasHandle_t h, int n, const Scalar* x, int incx, RealType* result) {
        return cublasDznrm2(h, n,
            reinterpret_cast<const cuDoubleComplex*>(x), incx, result);
    }

    static cublasStatus_t scal(cublasHandle_t h, int n, const Scalar* alpha, Scalar* x, int incx) {
        return cublasZscal(h, n,
            reinterpret_cast<const cuDoubleComplex*>(alpha),
            reinterpret_cast<cuDoubleComplex*>(x), incx);
    }

    static cublasStatus_t scal_real(cublasHandle_t h, int n, const RealType* alpha, Scalar* x, int incx) {
        return cublasZdscal(h, n, alpha,
            reinterpret_cast<cuDoubleComplex*>(x), incx);
    }

    static cublasStatus_t axpy(cublasHandle_t h, int n, const Scalar* alpha,
            const Scalar* x, int incx, Scalar* y, int incy) {
        return cublasZaxpy(h, n,
            reinterpret_cast<const cuDoubleComplex*>(alpha),
            reinterpret_cast<const cuDoubleComplex*>(x), incx,
            reinterpret_cast<cuDoubleComplex*>(y), incy);
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
        return cusolverDnZgesvd(h, jobu, jobvt, m, n,
            reinterpret_cast<cuDoubleComplex*>(A), lda, S,
            reinterpret_cast<cuDoubleComplex*>(U), ldu,
            reinterpret_cast<cuDoubleComplex*>(Vh), ldvh,
            reinterpret_cast<cuDoubleComplex*>(work), lwork,
            rwork, devInfo);
    }

    // --- cuSOLVER QR ---
    static cusolverStatus_t cusolver_geqrf_bufferSize(cusolverDnHandle_t h,
            int m, int n, Scalar* A, int lda, int* lwork) {
        return cusolverDnZgeqrf_bufferSize(h, m, n,
            reinterpret_cast<cuDoubleComplex*>(A), lda, lwork);
    }

    static cusolverStatus_t cusolver_geqrf(cusolverDnHandle_t h,
            int m, int n, Scalar* A, int lda, Scalar* tau,
            Scalar* work, int lwork, int* devInfo) {
        return cusolverDnZgeqrf(h, m, n,
            reinterpret_cast<cuDoubleComplex*>(A), lda,
            reinterpret_cast<cuDoubleComplex*>(tau),
            reinterpret_cast<cuDoubleComplex*>(work), lwork, devInfo);
    }

    static cusolverStatus_t cusolver_orgqr_bufferSize(cusolverDnHandle_t h,
            int m, int n, int k, Scalar* A, int lda, Scalar* tau, int* lwork) {
        return cusolverDnZungqr_bufferSize(h, m, n, k,
            reinterpret_cast<cuDoubleComplex*>(A), lda,
            reinterpret_cast<cuDoubleComplex*>(tau), lwork);
    }

    static cusolverStatus_t cusolver_orgqr(cusolverDnHandle_t h,
            int m, int n, int k, Scalar* A, int lda, Scalar* tau,
            Scalar* work, int lwork, int* devInfo) {
        return cusolverDnZungqr(h, m, n, k,
            reinterpret_cast<cuDoubleComplex*>(A), lda,
            reinterpret_cast<cuDoubleComplex*>(tau),
            reinterpret_cast<cuDoubleComplex*>(work), lwork, devInfo);
    }
};

// ============================================================================
// Device helpers for Lanczos device-pointer-mode operations
// ============================================================================

__device__ inline double dev_real_part(double x) { return x; }
__device__ inline double dev_real_part(cuDoubleComplex x) { return cuCreal(x); }

__device__ inline double dev_make_neg_real_scalar(double, double neg_alpha) { return neg_alpha; }
__device__ inline cuDoubleComplex dev_make_neg_real_scalar(cuDoubleComplex, double neg_alpha) {
    return make_cuDoubleComplex(neg_alpha, 0.0);
}

// Process dot result for Lanczos alpha step:
// 1. Store real_part(dot_result) in alpha_arr[iter]
// 2. Compute neg_alpha = Scalar(-real_part(dot_result)) for axpy
template<typename Scalar>
__global__ void lanczos_process_alpha_kernel(const Scalar* dot_result, Scalar* neg_alpha_out,
                                              double* alpha_arr, int iter) {
    double alpha = dev_real_part(dot_result[0]);
    alpha_arr[iter] = alpha;
    neg_alpha_out[0] = dev_make_neg_real_scalar(dot_result[0], -alpha);
}

// Process nrm2 result for Lanczos beta step:
// 1. Store nrm2 result in beta_arr[iter]
// 2. Compute 1/nrm2 for normalization
// 3. Store -beta as Scalar for next iteration's axpy
template<typename Scalar>
__global__ void lanczos_process_beta_kernel(const double* nrm2_result, double* inv_nrm_out,
                                             double* beta_arr, Scalar* neg_beta_scalars, int iter) {
    double beta = nrm2_result[0];
    beta_arr[iter] = beta;
    inv_nrm_out[0] = 1.0 / beta;
    neg_beta_scalars[iter] = dev_make_neg_real_scalar(Scalar{}, -beta);
}

// Negate a scalar value (for overlap in reorthogonalization)
__device__ inline double dev_negate(double x) { return -x; }
__device__ inline cuDoubleComplex dev_negate(cuDoubleComplex x) {
    return make_cuDoubleComplex(-cuCreal(x), -cuCimag(x));
}

template<typename Scalar>
__global__ void negate_scalar_kernel(const Scalar* in, Scalar* out) {
    out[0] = dev_negate(in[0]);
}

// Compute 1/x for a single real value (initial normalization)
static __global__ void inv_real_kernel(const double* in, double* out) {
    out[0] = 1.0 / in[0];
}

// ============================================================================
// GPU-side batched GEMM pointer setup kernels
// Compute pointer arrays directly on GPU — no host->device DMA, no race conditions.
// ============================================================================

// apply_heff Step 1: A[idx] = L_env + w*cL, where w = idx/dd
template<typename Scalar>
__global__ void setup_heff_A_ptrs(Scalar** ptrs, Scalar* L_env, int cL, int dd, int n) {
    int i = threadIdx.x;
    if (i < n) ptrs[i] = L_env + (i / dd) * cL;
}

// apply_heff Step 1: B[idx] = theta + s1*cL + s2*cL*d, where s1=(idx%dd)/d, s2=idx%d
template<typename Scalar>
__global__ void setup_heff_B_ptrs(Scalar** ptrs, Scalar* theta, int cL, int d, int dd, int n) {
    int i = threadIdx.x;
    if (i < n) {
        int s1 = (i % dd) / d;
        int s2 = i % d;
        ptrs[i] = theta + s1 * cL + s2 * cL * d;
    }
}

// apply_heff Step 1: C[idx] = T1 + idx*cL*cR
template<typename Scalar>
__global__ void setup_heff_C_ptrs(Scalar** ptrs, Scalar* T1, int cL_cR, int n) {
    int i = threadIdx.x;
    if (i < n) ptrs[i] = T1 + i * cL_cR;
}

// update_left_env Step 1: A[w*d+s] = L_env + w*chi_in
// update_left_env Step 1: B[w*d+s] = A_mps + s*chi_in
// update_left_env Step 1: C[w*d+s] = V + (w*d+s)*chi_in*chi_out
template<typename Scalar>
__global__ void setup_lenv_ptrs(Scalar** d_A, Scalar** d_B, Scalar** d_C,
                                 Scalar* L_env, Scalar* A_mps, Scalar* V,
                                 int chi_in, int chi_out, int d, int n) {
    int i = threadIdx.x;
    if (i < n) {
        int w = i / d, s = i % d;
        d_A[i] = L_env + w * chi_in;
        d_B[i] = A_mps + s * chi_in;
        d_C[i] = V + i * chi_in * chi_out;
    }
}

// update_right_env Step 1: A[wp*d+s] = A_mps + s*chi_out
// update_right_env Step 1: B[wp*d+s] = R_env + wp*chi_in
// update_right_env Step 1: C[wp*d+s] = V + (wp*d+s)*chi_out*chi_in
template<typename Scalar>
__global__ void setup_renv_ptrs(Scalar** d_A, Scalar** d_B, Scalar** d_C,
                                 Scalar* A_mps, Scalar* R_env, Scalar* V,
                                 int chi_in, int chi_out, int d, int n) {
    int i = threadIdx.x;
    if (i < n) {
        int wp = i / d, s = i % d;
        d_A[i] = A_mps + s * chi_out;
        d_B[i] = R_env + wp * chi_in;
        d_C[i] = V + i * chi_out * chi_in;
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
    conjugate_complex_kernel<<<grid, block, 0, stream>>>(data, n);
}

// ============================================================================
// GPU-side diagonal scaling kernels for SVD factor absorption
// Scale columns or rows of a column-major matrix by a real diagonal vector.
// Replaces host-side scaling loops + H<->D round-trips.
// ============================================================================

// Find truncation point: first index where S[i] < tol, clamped to [1, max_k]
template<typename RealType>
__global__ void svd_truncate_kernel(const RealType* S, int max_k, double tol, int* d_new_k) {
    int new_k = max_k;
    for (int i = 0; i < max_k; i++) {
        if (S[i] < tol) { new_k = i; break; }
    }
    if (new_k == 0) new_k = 1;
    d_new_k[0] = new_k;
}

// Scale columns of A (m x n, column-major) by real diagonal d:
//   C[i, j] = A[i, j] * d[j]     (each column j multiplied by d[j])
// Used for U * diag(S): columns of U scaled by singular values.
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
            // cuDoubleComplex
            C[i + j * ldc] = make_cuDoubleComplex(s * cuCreal(a), s * cuCimag(a));
        }
    }
}

// Scale rows of A (m x n, column-major) by real diagonal d:
//   C[i, j] = d[i] * A[i, j]     (each row i multiplied by d[i])
// Used for diag(S) * Vh: rows of Vh scaled by singular values.
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

// Helper to launch column-scale: C = A * diag(d), where A is (m x n) column-major
template<typename Scalar>
inline void scale_columns_by_real(const Scalar* A, int lda, const double* d_diag,
                                   Scalar* C, int ldc, int m, int n, cudaStream_t stream) {
    int total = m * n;
    int block = 256;
    int grid = (total + block - 1) / block;
    scale_columns_by_real_kernel<Scalar><<<grid, block, 0, stream>>>(A, lda, d_diag, C, ldc, m, n);
}

// Helper to launch row-scale: C = diag(d) * A, where A is (m x n) column-major
template<typename Scalar>
inline void scale_rows_by_real(const Scalar* A, int lda, const double* d_diag,
                                Scalar* C, int ldc, int m, int n, cudaStream_t stream) {
    int total = m * n;
    int block = 256;
    int grid = (total + block - 1) / block;
    scale_rows_by_real_kernel<Scalar><<<grid, block, 0, stream>>>(A, lda, d_diag, C, ldc, m, n);
}

#endif // SCALAR_TRAITS_H
