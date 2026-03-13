/**
 * Test workarounds for OpenBLAS 0.3.20 SVD bug:
 * 1. Post-SVD Modified Gram-Schmidt re-orthogonalization of U
 * 2. QR-based re-orthogonalization via dgeqrf/dorgqr
 * 3. Recompute U from A, S, Vh: U = A * Vh^T * diag(1/S)
 *
 * Compile: g++ -O2 -o test_svd_fix test_svd_fix.cpp -llapack -lopenblas -lm
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>
#include <algorithm>

extern "C" {
    void dgesvd_(const char* jobu, const char* jobvt, const int* m, const int* n,
                 double* a, const int* lda, double* s, double* u, const int* ldu,
                 double* vt, const int* ldvt, double* work, const int* lwork,
                 int* info);
    void dgemm_(const char* transa, const char* transb, const int* m, const int* n,
                const int* k, const double* alpha, const double* a, const int* lda,
                const double* b, const int* ldb, const double* beta, double* c,
                const int* ldc);
    double ddot_(const int* n, const double* x, const int* incx, const double* y, const int* incy);
    void daxpy_(const int* n, const double* alpha, const double* x, const int* incx,
                double* y, const int* incy);
    double dnrm2_(const int* n, const double* x, const int* incx);
    void dscal_(const int* n, const double* alpha, double* x, const int* incx);
}

double check_orthonormality(const double* U, int m, int k) {
    double max_err = 0.0;
    for (int i = 0; i < k; i++) {
        for (int j = 0; j <= i; j++) {
            double dot = 0.0;
            for (int r = 0; r < m; r++) {
                dot += U[i * m + r] * U[j * m + r];
            }
            double expected = (i == j) ? 1.0 : 0.0;
            double err = fabs(dot - expected);
            if (err > max_err) max_err = err;
        }
    }
    return max_err;
}

// Check reconstruction error: ||A - U*diag(S)*Vh||_F / ||A||_F
double check_reconstruction(const double* A, const double* U, const double* S,
                            const double* Vh, int m, int n, int k) {
    // Compute U*diag(S)*Vh
    std::vector<double> USVh(m * n, 0.0);
    // First: US = U * diag(S)
    std::vector<double> US(m * k);
    for (int j = 0; j < k; j++) {
        for (int i = 0; i < m; i++) {
            US[j * m + i] = U[j * m + i] * S[j];
        }
    }
    // Then: USVh = US * Vh
    double one = 1.0, zero = 0.0;
    dgemm_("N", "N", &m, &n, &k, &one, US.data(), &m, Vh, &k, &zero, USVh.data(), &m);

    double err_sq = 0.0, nrm_sq = 0.0;
    for (int i = 0; i < m * n; i++) {
        double diff = A[i] - USVh[i];
        err_sq += diff * diff;
        nrm_sq += A[i] * A[i];
    }
    return sqrt(err_sq / nrm_sq);
}

// Workaround 1: Modified Gram-Schmidt on columns of U
void mgs_reorthogonalize(double* U, int m, int k) {
    int inc1 = 1;
    for (int j = 0; j < k; j++) {
        double* col_j = U + j * m;
        // Orthogonalize against all previous columns
        for (int i = 0; i < j; i++) {
            double* col_i = U + i * m;
            double dot = ddot_(&m, col_j, &inc1, col_i, &inc1);
            double neg_dot = -dot;
            daxpy_(&m, &neg_dot, col_i, &inc1, col_j, &inc1);
        }
        // Normalize
        double nrm = dnrm2_(&m, col_j, &inc1);
        if (nrm > 1e-15) {
            double inv_nrm = 1.0 / nrm;
            dscal_(&m, &inv_nrm, col_j, &inc1);
        }
    }
}

// Workaround 2: Recompute U = A * Vh^T * diag(1/S) (only for well-conditioned S)
void recompute_U(const double* A, const double* S, const double* Vh,
                 double* U, int m, int n, int k) {
    // U_new = A * Vh^T * diag(1/S)
    // Step 1: T = A * Vh^T  (m×k)
    double one = 1.0, zero = 0.0;
    dgemm_("N", "T", &m, &k, &n, &one, A, &m, Vh, &k, &zero, U, &m);
    // Step 2: Scale each column by 1/S[j]
    for (int j = 0; j < k; j++) {
        if (S[j] > 1e-15) {
            double inv_s = 1.0 / S[j];
            for (int i = 0; i < m; i++) {
                U[j * m + i] *= inv_s;
            }
        }
    }
}

int main() {
    srand(42);

    int test_cases[][2] = {
        {184, 92},
        {200, 100},
        {256, 128},
        {500, 250},
    };

    for (auto& tc : test_cases) {
        int m = tc[0], n = tc[1];
        int k = std::min(m, n);
        printf("\n=== m=%d, n=%d ===\n", m, n);

        // Generate random matrix
        std::vector<double> A(m * n);
        for (int i = 0; i < m * n; i++) A[i] = (double)rand() / RAND_MAX - 0.5;
        double nrm = 0;
        for (int i = 0; i < m * n; i++) nrm += A[i] * A[i];
        nrm = sqrt(nrm);
        for (int i = 0; i < m * n; i++) A[i] /= nrm;

        std::vector<double> A_copy(A);

        // Do SVD (broken)
        std::vector<double> S(k), U(m * k), Vt(k * n);
        int lwork_query = -1;
        double work_opt;
        int info;
        const char jobu = 'S', jobvt = 'S';
        dgesvd_(&jobu, &jobvt, &m, &n, A.data(), &m, S.data(),
                U.data(), &m, Vt.data(), &k, &work_opt, &lwork_query, &info);
        int opt_lwork = (int)work_opt + 1;
        std::vector<double> work(opt_lwork);

        memcpy(A.data(), A_copy.data(), m * n * sizeof(double));
        dgesvd_(&jobu, &jobvt, &m, &n, A.data(), &m, S.data(),
                U.data(), &m, Vt.data(), &k, work.data(), &opt_lwork, &info);

        double ortho_raw = check_orthonormality(U.data(), m, k);
        double recon_raw = check_reconstruction(A_copy.data(), U.data(), S.data(), Vt.data(), m, n, k);
        printf("  Raw SVD:        ortho_err=%.4e  recon_err=%.4e\n", ortho_raw, recon_raw);

        // Fix 1: MGS re-orthogonalization
        std::vector<double> U_mgs(U);
        mgs_reorthogonalize(U_mgs.data(), m, k);
        double ortho_mgs = check_orthonormality(U_mgs.data(), m, k);
        double recon_mgs = check_reconstruction(A_copy.data(), U_mgs.data(), S.data(), Vt.data(), m, n, k);
        printf("  MGS fix:        ortho_err=%.4e  recon_err=%.4e\n", ortho_mgs, recon_mgs);

        // Fix 2: Recompute U from A, S, Vh
        std::vector<double> U_recomp(m * k);
        recompute_U(A_copy.data(), S.data(), Vt.data(), U_recomp.data(), m, n, k);
        double ortho_recomp = check_orthonormality(U_recomp.data(), m, k);
        double recon_recomp = check_reconstruction(A_copy.data(), U_recomp.data(), S.data(), Vt.data(), m, n, k);
        printf("  Recompute U:    ortho_err=%.4e  recon_err=%.4e\n", ortho_recomp, recon_recomp);

        // Fix 3: Recompute U then MGS
        std::vector<double> U_both(U_recomp);
        mgs_reorthogonalize(U_both.data(), m, k);
        double ortho_both = check_orthonormality(U_both.data(), m, k);
        double recon_both = check_reconstruction(A_copy.data(), U_both.data(), S.data(), Vt.data(), m, n, k);
        printf("  Recomp+MGS:     ortho_err=%.4e  recon_err=%.4e\n", ortho_both, recon_both);
    }

    printf("\nDone.\n");
    return 0;
}
