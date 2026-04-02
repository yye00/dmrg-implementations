/**
 * Standalone test for OpenBLAS dgesvd bug with tall-skinny matrices.
 * Tests both dgesvd and dgesdd to see if the divide-and-conquer variant works.
 *
 * Compile: g++ -O2 -o test_svd_bug test_svd_bug.cpp -llapack -lopenblas -lm
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>

extern "C" {
    void dgesvd_(const char* jobu, const char* jobvt, const int* m, const int* n,
                 double* a, const int* lda, double* s, double* u, const int* ldu,
                 double* vt, const int* ldvt, double* work, const int* lwork,
                 int* info);
    void dgesdd_(const char* jobz, const int* m, const int* n,
                 double* a, const int* lda, double* s, double* u, const int* ldu,
                 double* vt, const int* ldvt, double* work, const int* lwork,
                 int* iwork, int* info);
}

// Check orthonormality of U: compute max|U^T*U - I|
double check_orthonormality(const double* U, int m, int k) {
    double max_diag_err = 0.0, max_off_err = 0.0;
    for (int i = 0; i < k; i++) {
        for (int j = 0; j <= i; j++) {
            double dot = 0.0;
            for (int r = 0; r < m; r++) {
                dot += U[i * m + r] * U[j * m + r];  // column-major
            }
            double expected = (i == j) ? 1.0 : 0.0;
            double err = fabs(dot - expected);
            if (i == j) {
                if (err > max_diag_err) max_diag_err = err;
            } else {
                if (err > max_off_err) max_off_err = err;
            }
        }
    }
    return max_diag_err + max_off_err;
}

void test_dgesvd(int m, int n, const double* A_orig) {
    int k = (m < n) ? m : n;

    // Query workspace
    int lwork_query = -1;
    double work_opt;
    int info;
    const char jobu = 'S', jobvt = 'S';
    std::vector<double> A(m * n);
    std::vector<double> S(k);
    std::vector<double> U(m * k);
    std::vector<double> Vt(k * n);

    memcpy(A.data(), A_orig, m * n * sizeof(double));

    dgesvd_(&jobu, &jobvt, &m, &n, A.data(), &m, S.data(),
            U.data(), &m, Vt.data(), &k, &work_opt, &lwork_query, &info);
    int opt_lwork = (int)work_opt + 1;

    // Test with exact optimal workspace
    std::vector<double> work(opt_lwork);
    memcpy(A.data(), A_orig, m * n * sizeof(double));
    dgesvd_(&jobu, &jobvt, &m, &n, A.data(), &m, S.data(),
            U.data(), &m, Vt.data(), &k, work.data(), &opt_lwork, &info);
    double err1 = check_orthonormality(U.data(), m, k);
    printf("  dgesvd (lwork=%d): info=%d, U ortho err=%.4e, S[0]=%.6f S[%d]=%.4e\n",
           opt_lwork, info, err1, S[0], k-1, S[k-1]);

    // Test with 2x workspace
    int big_lwork = opt_lwork * 2;
    std::vector<double> work2(big_lwork);
    memcpy(A.data(), A_orig, m * n * sizeof(double));
    std::fill(U.begin(), U.end(), 0.0);
    dgesvd_(&jobu, &jobvt, &m, &n, A.data(), &m, S.data(),
            U.data(), &m, Vt.data(), &k, work2.data(), &big_lwork, &info);
    double err2 = check_orthonormality(U.data(), m, k);
    printf("  dgesvd (lwork=%d): info=%d, U ortho err=%.4e\n", big_lwork, info, err2);

    // Test with square-queried workspace (the original bug)
    int sq = (m > n) ? m : n;
    lwork_query = -1;
    dgesvd_(&jobu, &jobvt, &sq, &sq, nullptr, &sq, nullptr,
            nullptr, &sq, nullptr, &sq, &work_opt, &lwork_query, &info);
    int sq_lwork = (int)work_opt + 1;
    if (sq_lwork < opt_lwork) {
        std::vector<double> work3(sq_lwork);
        memcpy(A.data(), A_orig, m * n * sizeof(double));
        std::fill(U.begin(), U.end(), 0.0);
        dgesvd_(&jobu, &jobvt, &m, &n, A.data(), &m, S.data(),
                U.data(), &m, Vt.data(), &k, work3.data(), &sq_lwork, &info);
        double err3 = check_orthonormality(U.data(), m, k);
        printf("  dgesvd (sq_lwork=%d, UNDERSIZED): info=%d, U ortho err=%.4e\n",
               sq_lwork, info, err3);
    } else {
        printf("  (square lwork %d >= optimal %d, no undersized test)\n", sq_lwork, opt_lwork);
    }
}

void test_dgesdd(int m, int n, const double* A_orig) {
    int k = (m < n) ? m : n;

    int lwork_query = -1;
    double work_opt;
    int info;
    const char jobz = 'S';
    std::vector<double> A(m * n);
    std::vector<double> S(k);
    std::vector<double> U(m * k);
    std::vector<double> Vt(k * n);
    std::vector<int> iwork(8 * k);

    memcpy(A.data(), A_orig, m * n * sizeof(double));
    dgesdd_(&jobz, &m, &n, A.data(), &m, S.data(),
            U.data(), &m, Vt.data(), &k, &work_opt, &lwork_query, iwork.data(), &info);
    int opt_lwork = (int)work_opt + 1;

    std::vector<double> work(opt_lwork);
    memcpy(A.data(), A_orig, m * n * sizeof(double));
    dgesdd_(&jobz, &m, &n, A.data(), &m, S.data(),
            U.data(), &m, Vt.data(), &k, work.data(), &opt_lwork, iwork.data(), &info);
    double err = check_orthonormality(U.data(), m, k);
    printf("  dgesdd (lwork=%d): info=%d, U ortho err=%.4e, S[0]=%.6f S[%d]=%.4e\n",
           opt_lwork, info, err, S[0], k-1, S[k-1]);
}

int main() {
    srand(42);

    // Test dimensions matching the DMRG failure case
    int test_cases[][2] = {
        {184, 92},   // chi=92, d=2: exact failure case
        {182, 91},   // chi=91, d=2: last working case
        {200, 100},  // round numbers
        {256, 128},  // power of 2
        {500, 250},  // larger
        {100, 100},  // square (should always work)
    };

    for (auto& tc : test_cases) {
        int m = tc[0], n = tc[1];
        printf("\n=== m=%d, n=%d ===\n", m, n);

        // Generate random matrix and normalize
        std::vector<double> A(m * n);
        for (int i = 0; i < m * n; i++) {
            A[i] = (double)rand() / RAND_MAX - 0.5;
        }
        // Normalize to unit Frobenius norm
        double nrm = 0;
        for (int i = 0; i < m * n; i++) nrm += A[i] * A[i];
        nrm = sqrt(nrm);
        for (int i = 0; i < m * n; i++) A[i] /= nrm;

        test_dgesvd(m, n, A.data());
        test_dgesdd(m, n, A.data());
    }

    printf("\nDone.\n");
    return 0;
}
