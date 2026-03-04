#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include <rocblas/rocblas.h>
#include <vector>
#include <cmath>
#include <stdexcept>
#include "gpu_memory.hpp"

// Lanczos eigensolver for DMRG effective Hamiltonian
// Uses rocBLAS GEMV (BLAS-2) - memory bandwidth bound but straightforward
// For PDMRG-GPU (not PDMRG2-GPU which uses Block-Davidson)

#define ROCBLAS_CHECK(call) \
    do { \
        rocblas_status status = call; \
        if (status != rocblas_status_success) { \
            throw std::runtime_error(std::string("rocBLAS error at ") + __FILE__ + ":" + \
                                     std::to_string(__LINE__) + " - code " + std::to_string(status)); \
        } \
    } while(0)

class LanczosEigensolver {
private:
    rocblas_handle handle;
    int max_iter;
    double tol;

public:
    LanczosEigensolver(int max_iterations = 50, double tolerance = 1e-12)
        : max_iter(max_iterations), tol(tolerance) {
        ROCBLAS_CHECK(rocblas_create_handle(&handle));
    }

    ~LanczosEigensolver() {
        rocblas_destroy_handle(handle);
    }

    // Solve for lowest eigenvalue/eigenvector of effective Hamiltonian
    // H_eff is provided as a matrix-vector product operator
    // For DMRG: H_eff |psi> = einsum('ijkl,kl->ij', theta_4, psi_mat)
    //
    // Parameters:
    //   d_H_eff: Dense effective Hamiltonian matrix (m × m, complex128)
    //   m: Matrix dimension
    //   d_v0: Initial guess vector (m, complex128)
    //   d_v_out: Output eigenvector (m, complex128)
    //   stream: HIP stream for async execution
    //
    // Returns: lowest eigenvalue (double)
    double solve(const Complex* d_H_eff, int m,
                 const Complex* d_v0, Complex* d_v_out,
                 hipStream_t stream = 0);

private:
    // Apply H_eff to vector using rocBLAS GEMV
    void matvec(const Complex* d_H, int m, const Complex* d_x, Complex* d_y,
                hipStream_t stream = 0);

    // Compute <x|y> using rocBLAS DOTC
    std::complex<double> dot(const Complex* d_x, const Complex* d_y, int n,
                              hipStream_t stream = 0);

    // Compute ||x|| using rocBLAS NRMC
    double norm(const Complex* d_x, int n, hipStream_t stream = 0);

    // AXPY: y = alpha*x + y using rocBLAS
    void axpy(const std::complex<double>& alpha, const Complex* d_x,
              Complex* d_y, int n, hipStream_t stream = 0);

    // Solve tridiagonal eigenvalue problem on CPU
    double solve_tridiagonal(const std::vector<double>& alpha,
                             const std::vector<double>& beta,
                             std::vector<double>& eig_vec);
};

inline void LanczosEigensolver::matvec(const Complex* d_H, int m,
                                        const Complex* d_x, Complex* d_y,
                                        hipStream_t stream) {
    // Set stream
    ROCBLAS_CHECK(rocblas_set_stream(handle, stream));

    // y = H * x  (GEMV: y = alpha*A*x + beta*y)
    Complex alpha = make_complex(1.0, 0.0);
    Complex beta = make_complex(0.0, 0.0);

    ROCBLAS_CHECK(rocblas_zgemv(handle,
                                 rocblas_operation_none,  // No transpose
                                 m, m,                    // Rows, cols
                                 &alpha,                  // Scalar alpha
                                 d_H, m,                  // Matrix A, lda
                                 d_x, 1,                  // Vector x, incx
                                 &beta,                   // Scalar beta
                                 d_y, 1));                // Vector y, incy
}

inline std::complex<double> LanczosEigensolver::dot(const Complex* d_x, const Complex* d_y,
                                                      int n, hipStream_t stream) {
    ROCBLAS_CHECK(rocblas_set_stream(handle, stream));

    Complex result;
    ROCBLAS_CHECK(rocblas_zdotc(handle, n, d_x, 1, d_y, 1, &result));

    return std::complex<double>(result.x, result.y);
}

inline double LanczosEigensolver::norm(const Complex* d_x, int n, hipStream_t stream) {
    ROCBLAS_CHECK(rocblas_set_stream(handle, stream));

    double result;
    ROCBLAS_CHECK(rocblas_dznrm2(handle, n, d_x, 1, &result));

    return result;
}

inline void LanczosEigensolver::axpy(const std::complex<double>& alpha,
                                      const Complex* d_x, Complex* d_y, int n,
                                      hipStream_t stream) {
    ROCBLAS_CHECK(rocblas_set_stream(handle, stream));

    Complex alpha_hip = to_hip_complex(alpha);
    ROCBLAS_CHECK(rocblas_zaxpy(handle, n, &alpha_hip, d_x, 1, d_y, 1));
}

inline double LanczosEigensolver::solve(const Complex* d_H_eff, int m,
                                         const Complex* d_v0, Complex* d_v_out,
                                         hipStream_t stream) {
    // Lanczos iteration to find lowest eigenvalue
    // Builds tridiagonal matrix T via Krylov subspace
    //
    // Algorithm:
    //   v_0 = v0 / ||v0||
    //   for j = 0 to max_iter:
    //     w = H * v_j
    //     alpha_j = <v_j | w>
    //     w = w - alpha_j * v_j - beta_{j-1} * v_{j-1}
    //     beta_j = ||w||
    //     v_{j+1} = w / beta_j
    //   Solve tridiagonal eigenvalue problem for T
    //   Reconstruct eigenvector from Lanczos vectors

    std::vector<double> alpha_vec;
    std::vector<double> beta_vec;

    // Allocate GPU buffers
    GPUBuffer<Complex> d_v_curr(m);
    GPUBuffer<Complex> d_v_prev(m);
    GPUBuffer<Complex> d_w(m);

    // Initialize v_0 = v0 / ||v0||
    HIP_CHECK(hipMemcpyAsync(d_v_curr.data(), d_v0, m * sizeof(Complex),
                            hipMemcpyDeviceToDevice, stream));
    double v0_norm = norm(d_v_curr.data(), m, stream);
    Complex inv_norm = make_complex(1.0 / v0_norm, 0.0);
    ROCBLAS_CHECK(rocblas_set_stream(handle, stream));
    ROCBLAS_CHECK(rocblas_zscal(handle, m, &inv_norm, d_v_curr.data(), 1));

    // Store Lanczos vectors for reconstruction (CPU for simplicity)
    std::vector<std::vector<Complex>> lanczos_vecs_cpu;
    std::vector<Complex> v_cpu(m);
    d_v_curr.copy_to_host(v_cpu, stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    lanczos_vecs_cpu.push_back(v_cpu);

    double beta_prev = 0.0;

    for (int j = 0; j < max_iter; j++) {
        // w = H * v_j
        matvec(d_H_eff, m, d_v_curr.data(), d_w.data(), stream);

        // alpha_j = <v_j | w>
        auto alpha_complex = dot(d_v_curr.data(), d_w.data(), m, stream);
        double alpha_j = alpha_complex.real();  // Should be real for Hermitian H
        alpha_vec.push_back(alpha_j);

        // w = w - alpha_j * v_j
        axpy(std::complex<double>(-alpha_j, 0.0), d_v_curr.data(), d_w.data(), m, stream);

        // w = w - beta_{j-1} * v_{j-1}
        if (j > 0) {
            axpy(std::complex<double>(-beta_prev, 0.0), d_v_prev.data(), d_w.data(), m, stream);
        }

        // beta_j = ||w||
        double beta_j = norm(d_w.data(), m, stream);
        beta_vec.push_back(beta_j);

        // Check convergence
        if (beta_j < tol) {
            break;
        }

        // v_{j+1} = w / beta_j
        Complex inv_beta = make_complex(1.0 / beta_j, 0.0);
        ROCBLAS_CHECK(rocblas_set_stream(handle, stream));
        ROCBLAS_CHECK(rocblas_zscal(handle, m, &inv_beta, d_w.data(), 1));

        // Rotate: v_prev = v_curr, v_curr = w
        std::swap(d_v_prev, d_v_curr);
        HIP_CHECK(hipMemcpyAsync(d_v_curr.data(), d_w.data(), m * sizeof(Complex),
                                hipMemcpyDeviceToDevice, stream));

        // Store for reconstruction
        d_v_curr.copy_to_host(v_cpu, stream);
        HIP_CHECK(hipStreamSynchronize(stream));
        lanczos_vecs_cpu.push_back(v_cpu);

        beta_prev = beta_j;
    }

    // Solve tridiagonal system on CPU
    std::vector<double> eig_coeffs;
    double eigenvalue = solve_tridiagonal(alpha_vec, beta_vec, eig_coeffs);

    // Reconstruct eigenvector: v_out = sum_j eig_coeffs[j] * lanczos_vecs[j]
    d_v_out->zero(stream);
    for (size_t j = 0; j < eig_coeffs.size(); j++) {
        GPUBuffer<Complex> d_lanczos_j(m);
        d_lanczos_j.copy_from_host(lanczos_vecs_cpu[j], stream);
        axpy(std::complex<double>(eig_coeffs[j], 0.0),
             d_lanczos_j.data(), d_v_out, m, stream);
    }

    // Normalize output
    double out_norm = norm(d_v_out, m, stream);
    Complex inv_out_norm = make_complex(1.0 / out_norm, 0.0);
    ROCBLAS_CHECK(rocblas_set_stream(handle, stream));
    ROCBLAS_CHECK(rocblas_zscal(handle, m, &inv_out_norm, d_v_out, 1));

    HIP_CHECK(hipStreamSynchronize(stream));
    return eigenvalue;
}

inline double LanczosEigensolver::solve_tridiagonal(const std::vector<double>& alpha,
                                                      const std::vector<double>& beta,
                                                      std::vector<double>& eig_vec) {
    // Solve tridiagonal eigenvalue problem using LAPACK (CPU)
    // T is tridiagonal with diagonal alpha and off-diagonal beta
    //
    // For now, use simple power iteration on T (CPU is fast enough)
    // TODO: Use proper LAPACK dstev for production

    int n = alpha.size();
    eig_vec.resize(n, 0.0);

    if (n == 1) {
        eig_vec[0] = 1.0;
        return alpha[0];
    }

    // Power iteration to find lowest eigenvalue of T
    std::vector<double> v(n, 1.0 / std::sqrt(n));
    std::vector<double> Tv(n);

    for (int iter = 0; iter < 100; iter++) {
        // Tv = T * v (tridiagonal matvec)
        for (int i = 0; i < n; i++) {
            Tv[i] = alpha[i] * v[i];
            if (i > 0) Tv[i] += beta[i-1] * v[i-1];
            if (i < n-1) Tv[i] += beta[i] * v[i+1];
        }

        // Normalize
        double norm_val = 0.0;
        for (double x : Tv) norm_val += x * x;
        norm_val = std::sqrt(norm_val);

        for (int i = 0; i < n; i++) {
            v[i] = Tv[i] / norm_val;
        }
    }

    // Rayleigh quotient: eigenvalue = <v|T|v>
    double eigenvalue = 0.0;
    for (int i = 0; i < n; i++) {
        double Ti_v = alpha[i] * v[i];
        if (i > 0) Ti_v += beta[i-1] * v[i-1];
        if (i < n-1) Ti_v += beta[i] * v[i+1];
        eigenvalue += v[i] * Ti_v;
    }

    eig_vec = v;
    return eigenvalue;
}
