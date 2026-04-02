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
    static int svd_rwork_size(int /*m*/, int /*n*/) { return 0; }

    static void lapack_gesvd(const char* jobu, const char* jobvt,
            const int* m, const int* n, Scalar* a, const int* lda,
            RealType* s, Scalar* u, const int* ldu, Scalar* vt, const int* ldvt,
            Scalar* work, const int* lwork, RealType* /*rwork*/, int* info) {
        dgesvd_(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info);
    }

    // --- LAPACK eigendecomposition ---
    static int syev_rwork_size(int /*n*/) { return 0; }

    static void lapack_syev(const char* jobz, const char* uplo, const int* n,
            Scalar* a, const int* lda, RealType* w,
            Scalar* work, const int* lwork, RealType* /*rwork*/, int* info) {
        dsyev_(jobz, uplo, n, a, lda, w, work, lwork, info);
    }

    // --- cuSOLVER SVD ---
    static cusolverStatus_t cusolver_gesvd(cusolverDnHandle_t h,
            signed char jobu, signed char jobvt,
            int m, int n, Scalar* A, int lda, RealType* S,
            Scalar* U, int ldu, Scalar* Vh, int ldvh,
            Scalar* work, int lwork, RealType* /*rwork*/, int* info) {
        return cusolverDnDgesvd(h, jobu, jobvt, m, n, A, lda, S, U, ldu, Vh, ldvh,
                                work, lwork, nullptr, info);
    }

    static cusolverStatus_t cusolver_gesvd_bufferSize(cusolverDnHandle_t h,
            int m, int n, int* lwork) {
        return cusolverDnDgesvd_bufferSize(h, m, n, lwork);
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

    static cublasStatus_t gemm_strided_batched(cublasHandle_t h,
            cublasOperation_t opA, cublasOperation_t opB,
            int m, int n, int k, const Scalar* alpha,
            const Scalar* A, int lda, long long int strideA,
            const Scalar* B, int ldb, long long int strideB,
            const Scalar* beta, Scalar* C, int ldc, long long int strideC,
            int batch_count) {
        return cublasZgemmStridedBatched(h, opA, opB, m, n, k,
            reinterpret_cast<const cuDoubleComplex*>(alpha),
            reinterpret_cast<const cuDoubleComplex*>(A), lda, strideA,
            reinterpret_cast<const cuDoubleComplex*>(B), ldb, strideB,
            reinterpret_cast<const cuDoubleComplex*>(beta),
            reinterpret_cast<cuDoubleComplex*>(C), ldc, strideC,
            batch_count);
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

    // --- LAPACK eigendecomposition ---
    static int syev_rwork_size(int n) { return std::max(1, 3*n - 2); }

    static void lapack_syev(const char* jobz, const char* uplo, const int* n,
            Scalar* a, const int* lda, RealType* w,
            Scalar* work, const int* lwork, RealType* rwork, int* info) {
        zheev_(jobz, uplo, n, a, lda, w, work, lwork, rwork, info);
    }

    // --- cuSOLVER SVD ---
    static cusolverStatus_t cusolver_gesvd(cusolverDnHandle_t h,
            signed char jobu, signed char jobvt,
            int m, int n, Scalar* A, int lda, RealType* S,
            Scalar* U, int ldu, Scalar* Vh, int ldvh,
            Scalar* work, int lwork, RealType* rwork, int* info) {
        return cusolverDnZgesvd(h, jobu, jobvt, m, n,
            reinterpret_cast<cuDoubleComplex*>(A), lda, S,
            reinterpret_cast<cuDoubleComplex*>(U), ldu,
            reinterpret_cast<cuDoubleComplex*>(Vh), ldvh,
            reinterpret_cast<cuDoubleComplex*>(work), lwork, rwork, info);
    }

    static cusolverStatus_t cusolver_gesvd_bufferSize(cusolverDnHandle_t h,
            int m, int n, int* lwork) {
        return cusolverDnZgesvd_bufferSize(h, m, n, lwork);
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
// Device helpers for Lanczos device-pointer-mode operations (fallback)
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

static __global__ void inv_real_kernel(const double* in, double* out) {
    out[0] = 1.0 / in[0];
}

// ============================================================================
// GPU-side batched GEMM pointer setup kernels
// ============================================================================

template<typename Scalar>
__global__ void setup_heff_A_ptrs(Scalar** ptrs, Scalar* L_env, int cL, int dd, int n) {
    int i = threadIdx.x;
    if (i < n) ptrs[i] = L_env + (i / dd) * cL;
}

template<typename Scalar>
__global__ void setup_heff_B_ptrs(Scalar** ptrs, Scalar* theta, int cL, int d, int dd, int n) {
    int i = threadIdx.x;
    if (i < n) {
        int s1 = (i % dd) / d;
        int s2 = i % d;
        ptrs[i] = theta + s1 * cL + s2 * cL * d;
    }
}

template<typename Scalar>
__global__ void setup_heff_C_ptrs(Scalar** ptrs, Scalar* T1, int cL_cR, int n) {
    int i = threadIdx.x;
    if (i < n) ptrs[i] = T1 + i * cL_cR;
}

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

// Setup pointer arrays for apply_heff Step 3 batched GEMM
// For a given MPO index n, sets up d^2 batched GEMMs:
//   A[b] = T2 + (n*dd + b) * cL_cR    (each a cL*cR slice)
//   B[b] = R_env + n * cR              (same R_env block for all)
//   C[b] = result + (b/d)*cL + (b%d)*cL*d  (output theta slices)
template<typename Scalar>
__global__ void setup_step3_ptrs(Scalar** d_A, Scalar** d_B, Scalar** d_C,
                                  Scalar* T2, Scalar* R_env, Scalar* result,
                                  int cL, int cR, int d, int dd, int n_mpo,
                                  int cL_cR, int batch_count) {
    int b = threadIdx.x;
    if (b < batch_count) {
        d_A[b] = T2 + ((size_t)n_mpo * dd + b) * cL_cR;
        d_B[b] = R_env + n_mpo * cR;
        int s1p = b / d, s2p = b % d;
        d_C[b] = result + s1p * cL + s2p * cL * d;
    }
}

// Setup pointer arrays for update_left_env Step 3 batched GEMM
// For a given physical index sp, sets up D batched GEMMs:
//   A[wp] = U + (wp*d + sp) * chi_in * chi_out    (U^H slice)
//   B[wp] = A_mps + sp * chi_in                     (same A slice for all wp)
//   C[wp] = L_new + wp * chi_out                    (L_env output slice)
template<typename Scalar>
__global__ void setup_lenv_step3_ptrs(Scalar** d_A, Scalar** d_B, Scalar** d_C,
                                       Scalar* U, Scalar* A_mps, Scalar* L_new,
                                       int chi_in, int chi_out, int d, int sp,
                                       int D) {
    int wp = threadIdx.x;
    if (wp < D) {
        d_A[wp] = U + (wp * d + sp) * chi_in * chi_out;
        d_B[wp] = A_mps + sp * chi_in;
        d_C[wp] = L_new + wp * chi_out;
    }
}

// Setup pointer arrays for update_right_env Step 3 batched GEMM
// For a given physical index sp, sets up D batched GEMMs:
//   A[w] = U + (w*d + sp) * chi_out * chi_in      (U slice)
//   B[w] = A_mps + sp * chi_out                     (same A^H slice for all w)
//   C[w] = R_new + w * chi_out                      (R_env output slice)
template<typename Scalar>
__global__ void setup_renv_step3_ptrs(Scalar** d_A, Scalar** d_B, Scalar** d_C,
                                       Scalar* U, Scalar* A_mps, Scalar* R_new,
                                       int chi_in, int chi_out, int d, int sp,
                                       int D) {
    int w = threadIdx.x;
    if (w < D) {
        d_A[w] = U + (w * d + sp) * chi_out * chi_in;
        d_B[w] = A_mps + sp * chi_out;
        d_C[w] = R_new + w * chi_out;
    }
}

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
    conjugate_complex_kernel<<<grid, block, 0, stream>>>(data, n);
}

// ============================================================================
// Newton-Schulz kernel: compute A = alpha*I - A  (for 3I - UtU step)
// ============================================================================

__device__ inline double dev_subtract(double a, double b) { return a - b; }
__device__ inline cuDoubleComplex dev_subtract(cuDoubleComplex a, cuDoubleComplex b) {
    return make_cuDoubleComplex(cuCreal(a) - cuCreal(b), cuCimag(a) - cuCimag(b));
}

template<typename Scalar>
__global__ void scaled_identity_minus_kernel(Scalar* A, int n, double alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * n) {
        int row = idx % n;
        int col = idx / n;
        double diag = (row == col) ? alpha : 0.0;
        // A[idx] = diag - A[idx]  (real part only matters for diagonal)
        A[idx] = dev_subtract(
            dev_make_neg_real_scalar(A[idx], diag),  // reuse helper to make Scalar from double
            A[idx]);
    }
}

// Simpler version that works correctly for both types
template<typename Scalar>
__global__ void compute_3I_minus_A(Scalar* A, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * n) {
        int row = idx % n;
        int col = idx / n;
        // result = 3*delta(row,col) - A[idx]
        double diag = (row == col) ? 3.0 : 0.0;
        double re = diag - dev_real_part(A[idx]);
        A[idx] = dev_make_neg_real_scalar(A[idx], re);
        // For complex: imaginary part should be negated too
    }
}

// Most general version: computes out[idx] = alpha*I[idx] - in[idx]
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
    scaled_identity_minus_double<<<grid, block, 0, stream>>>(data, n, alpha);
}

static inline void launch_scaled_identity_minus(cuDoubleComplex* data, int n, double alpha, cudaStream_t stream) {
    int block = 256;
    int grid = (n * n + block - 1) / block;
    scaled_identity_minus_complex<<<grid, block, 0, stream>>>(data, n, alpha);
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
