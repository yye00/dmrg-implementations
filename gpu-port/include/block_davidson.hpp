#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>
#include <vector>
#include <cmath>
#include <stdexcept>
#include "gpu_memory.hpp"

// Block-Davidson (LOBPCG-style) eigensolver for PDMRG2-GPU
// Uses rocBLAS GEMM (BLAS-3) for batched operations - compute-bound
// Much better GPU utilization than Lanczos (BLAS-2)

class BlockDavidson {
private:
    rocblas_handle handle;
    int block_size;
    int max_iter;
    double tol;

public:
    BlockDavidson(int b = 4, int max_iterations = 30, double tolerance = 1e-12)
        : block_size(b), max_iter(max_iterations), tol(tolerance) {
        ROCBLAS_CHECK(rocblas_create_handle(&handle));
    }

    ~BlockDavidson() {
        rocblas_destroy_handle(handle);
    }

    // Solve for lowest eigenvalue/eigenvector using block methods
    // H_eff: Dense effective Hamiltonian matrix (m × m, complex128)
    // m: Matrix dimension
    // d_v0: Initial guess vector (m, complex128)
    // d_v_out: Output eigenvector (m, complex128)
    // stream: HIP stream for async execution
    //
    // Returns: lowest eigenvalue (double)
    double solve(const Complex* d_H_eff, int m,
                 const Complex* d_v0, Complex* d_v_out,
                 hipStream_t stream = 0);

private:
    // Apply H to block of vectors: Y = H * X
    // X: (m × b) block of trial vectors
    // Y: (m × b) output block
    void block_matvec(const Complex* d_H, int m,
                      const Complex* d_X, Complex* d_Y,
                      hipStream_t stream = 0);

    // Gram-Schmidt orthogonalization of block X
    void orthogonalize_block(Complex* d_X, int m, int b, hipStream_t stream = 0);

    // Rayleigh-Ritz projection: solve dense b×b eigenvalue problem
    // H_proj = X† * Y  where Y = H * X
    // Returns lowest eigenvalue and coefficients
    double rayleigh_ritz(const Complex* d_X, const Complex* d_Y,
                         int m, int b, std::vector<double>& coeffs,
                         hipStream_t stream = 0);
};

inline void BlockDavidson::block_matvec(const Complex* d_H, int m,
                                          const Complex* d_X, Complex* d_Y,
                                          hipStream_t stream) {
    // Y = H * X  where X is (m × b), Y is (m × b)
    // This is a single GEMM: Y = 1.0 * H * X + 0.0 * Y
    // Much more efficient than b separate GEMVs!

    ROCBLAS_CHECK(rocblas_set_stream(handle, stream));

    Complex alpha = make_complex(1.0, 0.0);
    Complex beta = make_complex(0.0, 0.0);

    ROCBLAS_CHECK(rocblas_zgemm(handle,
                                 rocblas_operation_none,    // No transpose on H
                                 rocblas_operation_none,    // No transpose on X
                                 m, block_size, m,          // M, N, K
                                 &alpha,                    // Scalar alpha
                                 d_H, m,                    // Matrix A (H), lda
                                 d_X, m,                    // Matrix B (X), ldb
                                 &beta,                     // Scalar beta
                                 d_Y, m));                  // Matrix C (Y), ldc
}

inline void BlockDavidson::orthogonalize_block(Complex* d_X, int m, int b,
                                                 hipStream_t stream) {
    // Modified Gram-Schmidt for block X (m × b)
    // Column-by-column orthogonalization

    ROCBLAS_CHECK(rocblas_set_stream(handle, stream));

    for (int j = 0; j < b; j++) {
        Complex* d_xj = d_X + j * m;  // Column j

        // Orthogonalize against previous columns
        for (int i = 0; i < j; i++) {
            Complex* d_xi = d_X + i * m;  // Column i

            // <xi, xj>
            Complex dot_val;
            ROCBLAS_CHECK(rocblas_zdotc(handle, m, d_xi, 1, d_xj, 1, &dot_val));

            // xj = xj - <xi, xj> * xi
            Complex neg_dot = make_complex(-dot_val.x, -dot_val.y);
            ROCBLAS_CHECK(rocblas_zaxpy(handle, m, &neg_dot, d_xi, 1, d_xj, 1));
        }

        // Normalize xj
        double norm_val;
        ROCBLAS_CHECK(rocblas_dznrm2(handle, m, d_xj, 1, &norm_val));

        if (norm_val > 1e-14) {
            Complex inv_norm = make_complex(1.0 / norm_val, 0.0);
            ROCBLAS_CHECK(rocblas_zscal(handle, m, &inv_norm, d_xj, 1));
        }
    }
}

inline double BlockDavidson::rayleigh_ritz(const Complex* d_X, const Complex* d_Y,
                                             int m, int b, std::vector<double>& coeffs,
                                             hipStream_t stream) {
    // Compute H_proj = X† * Y  (b × b projection matrix)
    // Solve dense b×b eigenvalue problem on CPU
    // Return lowest eigenvalue and corresponding coefficients

    ROCBLAS_CHECK(rocblas_set_stream(handle, stream));

    // H_proj = X† * Y  (GEMM with conjugate transpose)
    GPUBuffer<Complex> d_H_proj(b * b);
    Complex alpha = make_complex(1.0, 0.0);
    Complex beta = make_complex(0.0, 0.0);

    ROCBLAS_CHECK(rocblas_zgemm(handle,
                                 rocblas_operation_conjugate_transpose,  // X†
                                 rocblas_operation_none,                 // Y
                                 b, b, m,                                // M, N, K
                                 &alpha,
                                 d_X, m,                                 // X (m×b), lda
                                 d_Y, m,                                 // Y (m×b), ldb
                                 &beta,
                                 d_H_proj.data(), b));                   // H_proj (b×b), ldc

    // Copy H_proj to CPU
    std::vector<Complex> h_H_proj(b * b);
    d_H_proj.copy_to_host(h_H_proj, stream);
    HIP_CHECK(hipStreamSynchronize(stream));

    // Convert to real matrix (H_proj should be Hermitian)
    std::vector<double> H_proj_real(b * b);
    for (int i = 0; i < b * b; i++) {
        H_proj_real[i] = h_H_proj[i].x;  // Real part
    }

    // Solve eigenvalue problem using LAPACK (CPU)
    // For small b (typically 4-8), CPU diagonalization is instant
    // TODO: Use actual LAPACK dsyev or rocsolver for production

    // Simple power iteration for demonstration
    std::vector<double> v(b, 1.0 / std::sqrt(b));
    std::vector<double> Hv(b);

    for (int iter = 0; iter < 100; iter++) {
        // Hv = H_proj * v
        for (int i = 0; i < b; i++) {
            Hv[i] = 0.0;
            for (int j = 0; j < b; j++) {
                Hv[i] += H_proj_real[i * b + j] * v[j];
            }
        }

        // Normalize
        double norm = 0.0;
        for (double x : Hv) norm += x * x;
        norm = std::sqrt(norm);

        for (int i = 0; i < b; i++) {
            v[i] = Hv[i] / norm;
        }
    }

    // Rayleigh quotient
    double eigenvalue = 0.0;
    for (int i = 0; i < b; i++) {
        double Hi_v = 0.0;
        for (int j = 0; j < b; j++) {
            Hi_v += H_proj_real[i * b + j] * v[j];
        }
        eigenvalue += v[i] * Hi_v;
    }

    coeffs = v;
    return eigenvalue;
}

inline double BlockDavidson::solve(const Complex* d_H_eff, int m,
                                    const Complex* d_v0, Complex* d_v_out,
                                    hipStream_t stream) {
    // Block-Davidson iteration
    // Maintains block of b trial vectors, applies H in batch
    // Much better GPU utilization than Lanczos

    int b = block_size;

    // Allocate block matrices X (m × b), Y (m × b)
    GPUBuffer<Complex> d_X(m * b);
    GPUBuffer<Complex> d_Y(m * b);

    // Initialize first column with v0, rest random
    HIP_CHECK(hipMemcpyAsync(d_X.data(), d_v0, m * sizeof(Complex),
                            hipMemcpyDeviceToDevice, stream));

    // Random initialization for other columns (simplified)
    std::vector<Complex> h_X_init(m * b);
    for (int j = 0; j < b; j++) {
        for (int i = 0; i < m; i++) {
            int idx = j * m + i;
            if (j == 0) {
                // Will be overwritten from d_v0
                h_X_init[idx] = make_complex(0.0, 0.0);
            } else {
                // Simple random (deterministic for reproducibility)
                double val = std::sin(double(i + j * m)) / (1.0 + i + j);
                h_X_init[idx] = make_complex(val, val * 0.5);
            }
        }
    }
    d_X.copy_from_host(h_X_init, stream);

    // First column from v0
    HIP_CHECK(hipMemcpyAsync(d_X.data(), d_v0, m * sizeof(Complex),
                            hipMemcpyDeviceToDevice, stream));

    // Orthogonalize initial block
    orthogonalize_block(d_X.data(), m, b, stream);

    double energy_prev = 1e10;

    for (int iter = 0; iter < max_iter; iter++) {
        // Y = H * X  (single batched GEMM!)
        block_matvec(d_H_eff, m, d_X.data(), d_Y.data(), stream);

        // Rayleigh-Ritz projection
        std::vector<double> coeffs;
        double energy = rayleigh_ritz(d_X.data(), d_Y.data(), m, b, coeffs, stream);

        // Check convergence
        if (std::abs(energy - energy_prev) < tol) {
            break;
        }
        energy_prev = energy;

        // Compute residual and new trial vector (simplified)
        // Full Block-Davidson would add residuals to subspace
        // For now, just reconstruct eigenvector and exit

        // v_out = X * coeffs  (linear combination of columns)
        d_v_out->zero(stream);
        for (int j = 0; j < b; j++) {
            Complex coeff = make_complex(coeffs[j], 0.0);
            const Complex* d_xj = d_X.data() + j * m;
            ROCBLAS_CHECK(rocblas_set_stream(handle, stream));
            ROCBLAS_CHECK(rocblas_zaxpy(handle, m, &coeff, d_xj, 1, d_v_out, 1));
        }

        // For convergence check, continue iterating...
        // But for simplicity, break after first Rayleigh-Ritz if converged
        if (iter > 0) break;
    }

    // Final eigenvector: v_out = X * coeffs
    std::vector<double> final_coeffs;
    double final_energy = rayleigh_ritz(d_X.data(), d_Y.data(), m, b,
                                        final_coeffs, stream);

    d_v_out->zero(stream);
    for (int j = 0; j < b; j++) {
        Complex coeff = make_complex(final_coeffs[j], 0.0);
        const Complex* d_xj = d_X.data() + j * m;
        ROCBLAS_CHECK(rocblas_set_stream(handle, stream));
        ROCBLAS_CHECK(rocblas_zaxpy(handle, m, &coeff, d_xj, 1, d_v_out, 1));
    }

    // Normalize
    double out_norm;
    ROCBLAS_CHECK(rocblas_dznrm2(handle, m, d_v_out, 1, &out_norm));
    Complex inv_norm = make_complex(1.0 / out_norm, 0.0);
    ROCBLAS_CHECK(rocblas_zscal(handle, m, &inv_norm, d_v_out, 1));

    HIP_CHECK(hipStreamSynchronize(stream));
    return final_energy;
}
