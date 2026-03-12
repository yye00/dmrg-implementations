#ifndef ACCURATE_SVD_H
#define ACCURATE_SVD_H

/**
 * Accurate SVD algorithm from Stoudenmire & White (arXiv:1301.3494) Appendix.
 *
 * Standard LAPACK SVD (DGESVD/ZGESVD) has poor relative accuracy for small
 * singular values. When computing V = 1/sigma for the PDMRG boundary merge,
 * these errors get amplified by 1/sigma, producing huge absolute errors in V.
 *
 * This recursive algorithm achieves uniform relative accuracy for ALL singular
 * values by recursively refining the small ones on progressively smaller
 * submatrices. Cost remains ~mn^2 since each recursive call is smaller.
 */

#include "scalar_traits.h"
#include <vector>
#include <cmath>
#include <algorithm>

// ============================================================================
// BLAS gemm extern declarations (CPU-side matrix multiply for A†MB†)
// ============================================================================

extern "C" void dgemm_(const char* transa, const char* transb,
                        const int* m, const int* n, const int* k,
                        const double* alpha, const double* a, const int* lda,
                        const double* b, const int* ldb,
                        const double* beta, double* c, const int* ldc);

extern "C" void zgemm_(const char* transa, const char* transb,
                        const int* m, const int* n, const int* k,
                        const hipDoubleComplex* alpha,
                        const hipDoubleComplex* a, const int* lda,
                        const hipDoubleComplex* b, const int* ldb,
                        const hipDoubleComplex* beta,
                        hipDoubleComplex* c, const int* ldc);

// ============================================================================
// CPU GEMM dispatch (overloaded for double / hipDoubleComplex)
// ============================================================================

inline void cpu_gemm(const char* transa, const char* transb,
                     const int* m, const int* n, const int* k,
                     const double* alpha, const double* a, const int* lda,
                     const double* b, const int* ldb,
                     const double* beta, double* c, const int* ldc) {
    dgemm_(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

inline void cpu_gemm(const char* transa, const char* transb,
                     const int* m, const int* n, const int* k,
                     const hipDoubleComplex* alpha,
                     const hipDoubleComplex* a, const int* lda,
                     const hipDoubleComplex* b, const int* ldb,
                     const hipDoubleComplex* beta,
                     hipDoubleComplex* c, const int* ldc) {
    zgemm_(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

// ============================================================================
// Recursive accurate SVD
//
// Input:  M (m x n) column-major matrix, not modified
// Output: U (m x k), S (k), Vh (k x n) where k = min(m,n)
//         with uniform relative accuracy for all singular values
// ============================================================================

template<typename Scalar>
void accurate_svd(int m, int n,
                  const Scalar* M_in, int ldm,
                  Scalar* U_out, int ldu,
                  typename ScalarTraits<Scalar>::RealType* S_out,
                  Scalar* Vh_out, int ldvh,
                  double epsilon = 1e-4) {
    using Traits = ScalarTraits<Scalar>;
    using RealType = typename Traits::RealType;

    int full_k = std::min(m, n);
    if (full_k == 0) return;

    // --- Step 1: Standard SVD via LAPACK ---

    // Copy M to work buffer (SVD is destructive)
    std::vector<Scalar> svd_a(m * n);
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
            svd_a[i + j * m] = M_in[i + j * ldm];

    // Query optimal workspace
    std::vector<RealType> svd_rwork(Traits::svd_rwork_size(m, n));
    const char jobu = 'S', jobvt = 'S';
    int lwork_query = -1;
    Scalar work_opt;
    int info;

    Traits::lapack_gesvd(&jobu, &jobvt, &m, &n, svd_a.data(), &m,
            S_out, U_out, &ldu, Vh_out, &ldvh,
            &work_opt, &lwork_query,
            svd_rwork.empty() ? nullptr : svd_rwork.data(), &info);

    int lwork;
    if constexpr (Traits::is_complex) {
        lwork = (int)Traits::real_part(work_opt) + 1;
    } else {
        lwork = (int)work_opt + 1;
    }
    std::vector<Scalar> svd_work(lwork);

    // Re-copy M (workspace query may have modified svd_a)
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
            svd_a[i + j * m] = M_in[i + j * ldm];

    // Perform SVD: svd_a = U · diag(S) · Vh
    Traits::lapack_gesvd(&jobu, &jobvt, &m, &n, svd_a.data(), &m,
            S_out, U_out, &ldu, Vh_out, &ldvh,
            svd_work.data(), &lwork,
            svd_rwork.empty() ? nullptr : svd_rwork.data(), &info);

    if (info != 0) return;  // SVD failed, return standard result

    // --- Step 2: Find split point p ---
    // Smallest index where S[p]/S[0] < epsilon
    if (S_out[0] < 1e-30) return;  // matrix is essentially zero

    int p = full_k;
    for (int i = 0; i < full_k; i++) {
        if (S_out[i] / S_out[0] < epsilon) {
            p = i;
            break;
        }
    }

    // --- Step 3: If no split needed, done ---
    if (p >= full_k) return;

    int sub_k = full_k - p;

    // --- Step 4: Compute X = U[:,p:]^H · M · Vh[p:,:]^H ---
    // Result: X is (sub_k x sub_k)

    Scalar alpha_one = Traits::one();
    Scalar beta_zero = Traits::zero();

    // T = U[:,p:]^H · M_in → (sub_k x n)
    // U[:,p:] stored at U_out + p*ldu, shape (m x sub_k), lda=ldu
    // transa='C'/'T': op(A) = A^H is (sub_k x m)
    // transb='N': op(B) = M_in is (m x n)
    std::vector<Scalar> T(sub_k * n);
    {
        const char transa = Traits::is_complex ? 'C' : 'T';
        const char transb = 'N';
        cpu_gemm(&transa, &transb, &sub_k, &n, &m,
                 &alpha_one, U_out + p * ldu, &ldu,
                 M_in, &ldm,
                 &beta_zero, T.data(), &sub_k);
    }

    // X = T · Vh[p:,:]^H → (sub_k x sub_k)
    // T is (sub_k x n), lda=sub_k
    // Vh[p:,:] stored at Vh_out + p, shape (sub_k x n), lda=ldvh
    // transb='C'/'T': op(B) = Vh[p:,:]^H is (n x sub_k)
    std::vector<Scalar> X(sub_k * sub_k);
    {
        const char transa = 'N';
        const char transb = Traits::is_complex ? 'C' : 'T';
        cpu_gemm(&transa, &transb, &sub_k, &sub_k, &n,
                 &alpha_one, T.data(), &sub_k,
                 Vh_out + p, &ldvh,
                 &beta_zero, X.data(), &sub_k);
    }

    // --- Step 5: Recursively SVD X ---
    std::vector<Scalar> sub_U(sub_k * sub_k);
    std::vector<RealType> sub_S(sub_k);
    std::vector<Scalar> sub_Vh(sub_k * sub_k);

    accurate_svd<Scalar>(sub_k, sub_k, X.data(), sub_k,
                         sub_U.data(), sub_k, sub_S.data(),
                         sub_Vh.data(), sub_k, epsilon);

    // --- Step 6: Update U, S, Vh for indices p..full_k-1 ---

    // U_new[:,p:] = U[:,p:] · sub_U
    // (m x sub_k) = (m x sub_k) · (sub_k x sub_k)
    {
        std::vector<Scalar> new_cols(m * sub_k);
        const char transa = 'N', transb = 'N';
        cpu_gemm(&transa, &transb, &m, &sub_k, &sub_k,
                 &alpha_one, U_out + p * ldu, &ldu,
                 sub_U.data(), &sub_k,
                 &beta_zero, new_cols.data(), &m);
        for (int j = 0; j < sub_k; j++)
            for (int i = 0; i < m; i++)
                U_out[i + (p + j) * ldu] = new_cols[i + j * m];
    }

    // Vh_new[p:,:] = sub_Vh · Vh[p:,:]
    // (sub_k x n) = (sub_k x sub_k) · (sub_k x n)
    {
        std::vector<Scalar> new_rows(sub_k * n);
        const char transa = 'N', transb = 'N';
        cpu_gemm(&transa, &transb, &sub_k, &n, &sub_k,
                 &alpha_one, sub_Vh.data(), &sub_k,
                 Vh_out + p, &ldvh,
                 &beta_zero, new_rows.data(), &sub_k);
        for (int j = 0; j < n; j++)
            for (int i = 0; i < sub_k; i++)
                Vh_out[(p + i) + j * ldvh] = new_rows[i + j * sub_k];
    }

    // S_new[p:] = sub_S
    for (int i = 0; i < sub_k; i++) {
        S_out[p + i] = sub_S[i];
    }
}

#endif // ACCURATE_SVD_H
