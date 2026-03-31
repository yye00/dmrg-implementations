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

    static Scalar scale_by_real(RealType s, Scalar v) { return s * v; }

    static Scalar random_val() { return 2.0 * rand() / RAND_MAX - 1.0; }

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

    static rocblas_status gemm_strided_batched(rocblas_handle h,
            rocblas_operation opA, rocblas_operation opB,
            int m, int n, int k, const Scalar* alpha,
            const Scalar* A, int lda, rocblas_stride strideA,
            const Scalar* B, int ldb, rocblas_stride strideB,
            const Scalar* beta, Scalar* C, int ldc, rocblas_stride strideC,
            int batch_count) {
        return rocblas_dgemm_strided_batched(h, opA, opB, m, n, k, alpha,
            A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batch_count);
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

    static rocblas_status gemm_strided_batched(rocblas_handle h,
            rocblas_operation opA, rocblas_operation opB,
            int m, int n, int k, const Scalar* alpha,
            const Scalar* A, int lda, rocblas_stride strideA,
            const Scalar* B, int ldb, rocblas_stride strideB,
            const Scalar* beta, Scalar* C, int ldc, rocblas_stride strideC,
            int batch_count) {
        return rocblas_zgemm_strided_batched(h, opA, opB, m, n, k,
            reinterpret_cast<const rocblas_double_complex*>(alpha),
            reinterpret_cast<const rocblas_double_complex*>(A), lda, strideA,
            reinterpret_cast<const rocblas_double_complex*>(B), ldb, strideB,
            reinterpret_cast<const rocblas_double_complex*>(beta),
            reinterpret_cast<rocblas_double_complex*>(C), ldc, strideC,
            batch_count);
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
    static int syev_rwork_size(int n) { return std::max(1, 3*n - 2); }

    static void lapack_syev(const char* jobz, const char* uplo, const int* n,
            Scalar* a, const int* lda, RealType* w,
            Scalar* work, const int* lwork, RealType* rwork, int* info) {
        zheev_(jobz, uplo, n, a, lda, w, work, lwork, rwork, info);
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
// Device helpers for Lanczos device-pointer-mode operations (fallback)
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

__global__ inline void inv_real_kernel(const double* in, double* out) {
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
// For a given MPO index n, sets up d² batched GEMMs:
//   A[b] = T2 + (n*dd + b) * cL_cR    (each a cL×cR slice)
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
// Newton-Schulz kernel: compute A = alpha*I - A  (for 3I - UtU step)
// ============================================================================

__device__ inline double dev_subtract(double a, double b) { return a - b; }
__device__ inline hipDoubleComplex dev_subtract(hipDoubleComplex a, hipDoubleComplex b) {
    return make_hipDoubleComplex(hipCreal(a) - hipCreal(b), hipCimag(a) - hipCimag(b));
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
// Scale columns or rows of a column-major matrix by a real diagonal vector.
// Replaces host-side scaling loops + H↔D round-trips.
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

// Scale columns of A (m × n, column-major) by real diagonal d:
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
            // hipDoubleComplex
            C[i + j * ldc] = make_hipDoubleComplex(s * hipCreal(a), s * hipCimag(a));
        }
    }
}

// Scale rows of A (m × n, column-major) by real diagonal d:
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
            C[i + j * ldc] = make_hipDoubleComplex(s * hipCreal(a), s * hipCimag(a));
        }
    }
}

// Helper to launch column-scale: C = A * diag(d), where A is (m × n) column-major
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

// Helper to launch row-scale: C = diag(d) * A, where A is (m × n) column-major
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
