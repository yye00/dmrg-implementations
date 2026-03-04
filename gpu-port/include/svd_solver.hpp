#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include "gpu_memory.hpp"

// SVD implementations for DMRG tensor truncation
// 1. Standard SVD using rocSOLVER (PDMRG-GPU)
// 2. Randomized SVD with Cholesky-QR2 (PDMRG2-GPU)
//
// Both maintain complex128 precision for accuracy

#define ROCSOLVER_CHECK(call) \
    do { \
        rocblas_status status = call; \
        if (status != rocblas_status_success) { \
            throw std::runtime_error(std::string("rocSOLVER error at ") + __FILE__ + ":" + \
                                     std::to_string(__LINE__) + " - code " + std::to_string(status)); \
        } \
    } while(0)

// Standard SVD solver using rocSOLVER zgesvd
// Memory-bandwidth bound but straightforward and accurate
class StandardSVD {
private:
    rocblas_handle handle;

public:
    StandardSVD() {
        ROCBLAS_CHECK(rocblas_create_handle(&handle));
    }

    ~StandardSVD() {
        rocblas_destroy_handle(handle);
    }

    // Compute SVD: A = U * S * V†
    // A: (m × n) complex matrix
    // U: (m × k) left singular vectors
    // S: (k,) singular values
    // Vh: (k × n) right singular vectors (conjugate transposed)
    // k = min(m, n, max_bond)
    //
    // Returns: number of singular values computed
    int compute(const Complex* d_A, int m, int n,
                Complex* d_U, double* d_S, Complex* d_Vh,
                int max_bond, hipStream_t stream = 0);

private:
    // Query workspace size for zgesvd
    size_t query_workspace(int m, int n);
};

inline size_t StandardSVD::query_workspace(int m, int n) {
    // Query workspace for rocSOLVER zgesvd
    size_t lwork = 0;
    double* d_work = nullptr;

    // Dummy call to query workspace
    int info = 0;
    ROCSOLVER_CHECK(rocsolver_zgesvd_buffsize(handle,
                                               rocblas_svect_singular,  // Left vectors
                                               rocblas_svect_singular,  // Right vectors
                                               m, n, &lwork));
    return lwork;
}

inline int StandardSVD::compute(const Complex* d_A, int m, int n,
                                 Complex* d_U, double* d_S, Complex* d_Vh,
                                 int max_bond, hipStream_t stream) {
    ROCBLAS_CHECK(rocblas_set_stream(handle, stream));

    int k = std::min({m, n, max_bond});
    int lda = m;

    // Copy A to temporary buffer (zgesvd destroys input)
    GPUBuffer<Complex> d_A_copy(m * n);
    HIP_CHECK(hipMemcpyAsync(d_A_copy.data(), d_A, m * n * sizeof(Complex),
                            hipMemcpyDeviceToDevice, stream));

    // Query workspace
    size_t lwork = query_workspace(m, n);
    GPUBuffer<Complex> d_work(lwork);

    // Allocate full U and Vh (rocSOLVER requires full matrices)
    GPUBuffer<Complex> d_U_full(m * m);
    GPUBuffer<Complex> d_Vh_full(n * n);
    GPUBuffer<double> d_S_full(std::min(m, n));
    GPUBuffer<double> d_rwork(5 * std::min(m, n));
    GPUBuffer<int> d_info(1);

    // Call rocSOLVER zgesvd
    ROCSOLVER_CHECK(rocsolver_zgesvd(handle,
                                      rocblas_svect_singular,  // Compute U
                                      rocblas_svect_singular,  // Compute Vh
                                      m, n,
                                      d_A_copy.data(), lda,
                                      d_S_full.data(),
                                      d_U_full.data(), m,
                                      d_Vh_full.data(), n,
                                      d_rwork.data(),
                                      rocblas_outofplace,
                                      d_info.data()));

    // Check convergence
    int info_cpu;
    HIP_CHECK(hipMemcpyAsync(&info_cpu, d_info.data(), sizeof(int),
                            hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    if (info_cpu != 0) {
        throw std::runtime_error("SVD failed to converge, info = " + std::to_string(info_cpu));
    }

    // Truncate to max_bond
    // U: extract first k columns of U_full
    for (int j = 0; j < k; j++) {
        HIP_CHECK(hipMemcpyAsync(d_U + j * m,
                                d_U_full.data() + j * m,
                                m * sizeof(Complex),
                                hipMemcpyDeviceToDevice, stream));
    }

    // S: extract first k singular values
    HIP_CHECK(hipMemcpyAsync(d_S, d_S_full.data(), k * sizeof(double),
                            hipMemcpyDeviceToDevice, stream));

    // Vh: extract first k rows of Vh_full
    for (int i = 0; i < k; i++) {
        HIP_CHECK(hipMemcpyAsync(d_Vh + i * n,
                                d_Vh_full.data() + i * n,
                                n * sizeof(Complex),
                                hipMemcpyDeviceToDevice, stream));
    }

    HIP_CHECK(hipStreamSynchronize(stream));
    return k;
}


// Randomized SVD with Cholesky-QR2 orthogonalization (PDMRG2-GPU)
// More GEMM-heavy, better GPU utilization
// Uses oversampling for accuracy
class RandomizedSVD {
private:
    rocblas_handle handle;
    int oversample;

public:
    explicit RandomizedSVD(int p = 10) : oversample(p) {
        ROCBLAS_CHECK(rocblas_create_handle(&handle));
    }

    ~RandomizedSVD() {
        rocblas_destroy_handle(handle);
    }

    // Compute approximate SVD using randomized algorithm
    // A: (m × n) complex matrix
    // U: (m × k) left singular vectors
    // S: (k,) singular values
    // Vh: (k × n) right singular vectors
    // k = min(m, n, max_bond)
    int compute(const Complex* d_A, int m, int n,
                Complex* d_U, double* d_S, Complex* d_Vh,
                int max_bond, hipStream_t stream = 0);

private:
    // Cholesky-QR2 orthogonalization (GEMM-based)
    void cholesky_qr2(Complex* d_Q, int m, int k, hipStream_t stream);
};

inline void RandomizedSVD::cholesky_qr2(Complex* d_Q, int m, int k,
                                         hipStream_t stream) {
    // Orthogonalize Q using Cholesky QR twice for stability
    // Q is (m × k)
    //
    // Algorithm:
    //   1. Compute G = Q† Q  (GEMM)
    //   2. Cholesky: G = L L†
    //   3. Q = Q L^{-1}  (TRSM)
    //   4. Repeat for stability

    ROCBLAS_CHECK(rocblas_set_stream(handle, stream));

    for (int iter = 0; iter < 2; iter++) {  // Two iterations for QR2
        // G = Q† Q  (k × k Gram matrix)
        GPUBuffer<Complex> d_G(k * k);
        Complex alpha = make_complex(1.0, 0.0);
        Complex beta = make_complex(0.0, 0.0);

        ROCBLAS_CHECK(rocblas_zgemm(handle,
                                     rocblas_operation_conjugate_transpose,
                                     rocblas_operation_none,
                                     k, k, m,
                                     &alpha,
                                     d_Q, m,
                                     d_Q, m,
                                     &beta,
                                     d_G.data(), k));

        // Cholesky factorization: G = L L†
        GPUBuffer<int> d_info(1);
        ROCSOLVER_CHECK(rocsolver_zpotrf(handle,
                                          rocblas_fill_lower,
                                          k,
                                          d_G.data(), k,
                                          d_info.data()));

        // Q = Q L^{-1}  (TRSM: solve Q L† = Q for new Q)
        alpha = make_complex(1.0, 0.0);
        ROCBLAS_CHECK(rocblas_ztrsm(handle,
                                     rocblas_side_right,
                                     rocblas_fill_lower,
                                     rocblas_operation_conjugate_transpose,
                                     rocblas_diagonal_non_unit,
                                     m, k,
                                     &alpha,
                                     d_G.data(), k,
                                     d_Q, m));
    }
}

inline int RandomizedSVD::compute(const Complex* d_A, int m, int n,
                                   Complex* d_U, double* d_S, Complex* d_Vh,
                                   int max_bond, hipStream_t stream) {
    // Randomized SVD algorithm (Halko et al. 2011)
    //
    // 1. Form random matrix Omega: (n × l) where l = k + p
    // 2. Y = A * Omega  (range finder via GEMM)
    // 3. Orthogonalize Y → Q using Cholesky-QR2
    // 4. B = Q† * A  (project A onto range)
    // 5. Compute SVD of small B: B = U_B S V†
    // 6. U = Q * U_B

    ROCBLAS_CHECK(rocblas_set_stream(handle, stream));

    int k = std::min({m, n, max_bond});
    int l = std::min(k + oversample, std::min(m, n));  // Oversampled rank

    // 1. Generate random test matrix Omega (n × l)
    GPUBuffer<Complex> d_Omega(n * l);
    std::vector<Complex> h_Omega(n * l);
    for (int i = 0; i < n * l; i++) {
        // Deterministic random for reproducibility
        double real_part = std::sin(double(i)) / (1.0 + i);
        double imag_part = std::cos(double(i)) / (1.0 + i);
        h_Omega[i] = make_complex(real_part, imag_part);
    }
    d_Omega.copy_from_host(h_Omega, stream);

    // 2. Y = A * Omega  (m × l matrix, range finder)
    GPUBuffer<Complex> d_Y(m * l);
    Complex alpha = make_complex(1.0, 0.0);
    Complex beta = make_complex(0.0, 0.0);

    ROCBLAS_CHECK(rocblas_zgemm(handle,
                                 rocblas_operation_none,
                                 rocblas_operation_none,
                                 m, l, n,
                                 &alpha,
                                 d_A, m,
                                 d_Omega.data(), n,
                                 &beta,
                                 d_Y.data(), m));

    // 3. Orthogonalize Y → Q using Cholesky-QR2
    GPUBuffer<Complex> d_Q(m * l);
    HIP_CHECK(hipMemcpyAsync(d_Q.data(), d_Y.data(), m * l * sizeof(Complex),
                            hipMemcpyDeviceToDevice, stream));
    cholesky_qr2(d_Q.data(), m, l, stream);

    // 4. B = Q† * A  (l × n matrix)
    GPUBuffer<Complex> d_B(l * n);
    ROCBLAS_CHECK(rocblas_zgemm(handle,
                                 rocblas_operation_conjugate_transpose,
                                 rocblas_operation_none,
                                 l, n, m,
                                 &alpha,
                                 d_Q.data(), m,
                                 d_A, m,
                                 &beta,
                                 d_B.data(), l));

    // 5. Compute SVD of B (small matrix, use standard SVD)
    GPUBuffer<Complex> d_U_B(l * l);
    GPUBuffer<double> d_S_full(l);
    GPUBuffer<Complex> d_Vh_B(l * n);

    StandardSVD svd_small;
    svd_small.compute(d_B.data(), l, n,
                      d_U_B.data(), d_S_full.data(), d_Vh_B.data(),
                      l, stream);

    // 6. U = Q * U_B  (m × k matrix)
    // Extract first k columns
    GPUBuffer<Complex> d_U_B_truncated(m * k);
    for (int j = 0; j < k; j++) {
        HIP_CHECK(hipMemcpyAsync(d_U_B_truncated.data() + j * l,
                                d_U_B.data() + j * l,
                                l * sizeof(Complex),
                                hipMemcpyDeviceToDevice, stream));
    }

    ROCBLAS_CHECK(rocblas_zgemm(handle,
                                 rocblas_operation_none,
                                 rocblas_operation_none,
                                 m, k, l,
                                 &alpha,
                                 d_Q.data(), m,
                                 d_U_B_truncated.data(), l,
                                 &beta,
                                 d_U, m));

    // Copy S (first k singular values)
    HIP_CHECK(hipMemcpyAsync(d_S, d_S_full.data(), k * sizeof(double),
                            hipMemcpyDeviceToDevice, stream));

    // Copy Vh (first k rows)
    for (int i = 0; i < k; i++) {
        HIP_CHECK(hipMemcpyAsync(d_Vh + i * n,
                                d_Vh_B.data() + i * n,
                                n * sizeof(Complex),
                                hipMemcpyDeviceToDevice, stream));
    }

    HIP_CHECK(hipStreamSynchronize(stream));
    return k;
}
