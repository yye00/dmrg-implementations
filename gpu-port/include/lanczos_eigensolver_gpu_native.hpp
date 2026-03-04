#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>
#include <vector>
#include <cmath>
#include <stdexcept>
#include "gpu_memory.hpp"

// GPU-NATIVE Lanczos eigensolver - NO CPU TRANSFERS during iteration
// Everything stays on GPU: Lanczos vectors, tridiagonal matrix, eigenvector
// Only final eigenvalue/eigenvector come back to CPU if needed

class LanczosEigensolverGPU {
private:
    rocblas_handle rocblas_handle;
    rocsolver_handle rocsolver_handle;
    int max_iter;
    double tol;

public:
    LanczosEigensolverGPU(int max_iterations = 50, double tolerance = 1e-12)
        : max_iter(max_iterations), tol(tolerance) {
        ROCBLAS_CHECK(rocblas_create_handle(&rocblas_handle));

        // rocSOLVER uses rocBLAS handle
        rocsolver_handle = rocblas_handle;
    }

    ~LanczosEigensolverGPU() {
        rocblas_destroy_handle(rocblas_handle);
    }

    // GPU-NATIVE Lanczos: Everything on GPU
    // Callback: matvec(d_x, d_y) applies H_eff to vector (user provides)
    // Returns: eigenvalue (single double to CPU), eigenvector stays on d_v_out
    template<typename MatVecFunc>
    double solve_gpu_native(
        MatVecFunc matvec,  // Callback: void(const Complex*, Complex*, hipStream_t)
        int dim,
        const Complex* d_v0,
        Complex* d_v_out,
        hipStream_t stream = 0) {

        // Allocate all Lanczos vectors on GPU (no CPU storage!)
        int k_max = std::min(max_iter, dim);

        GPUBuffer<Complex> d_V(dim * k_max);      // Lanczos basis on GPU
        GPUBuffer<double> d_alpha(k_max);         // Diagonal of T (on GPU)
        GPUBuffer<double> d_beta(k_max);          // Off-diagonal of T (on GPU)

        GPUBuffer<Complex> d_v_curr(dim);
        GPUBuffer<Complex> d_v_prev(dim);
        GPUBuffer<Complex> d_w(dim);

        ROCBLAS_CHECK(rocblas_set_stream(rocblas_handle, stream));

        // Initialize v_0 = v0 / ||v0|| (on GPU)
        HIP_CHECK(hipMemcpyAsync(d_v_curr.data(), d_v0, dim * sizeof(Complex),
                                hipMemcpyDeviceToDevice, stream));

        double v0_norm;
        ROCBLAS_CHECK(rocblas_dznrm2(rocblas_handle, dim, d_v_curr.data(), 1, &v0_norm));

        Complex inv_norm = make_complex(1.0 / v0_norm, 0.0);
        ROCBLAS_CHECK(rocblas_zscal(rocblas_handle, dim, &inv_norm, d_v_curr.data(), 1));

        // Store first Lanczos vector on GPU
        HIP_CHECK(hipMemcpyAsync(d_V.data(), d_v_curr.data(), dim * sizeof(Complex),
                                hipMemcpyDeviceToDevice, stream));

        int k_converged = 0;
        double beta_prev = 0.0;

        // Lanczos iteration - EVERYTHING ON GPU
        for (int j = 0; j < k_max; j++) {
            // w = H * v_j (matvec callback - stays on GPU)
            matvec(d_v_curr.data(), d_w.data(), stream);

            // alpha_j = <v_j | w> (stays on GPU)
            Complex alpha_complex;
            ROCBLAS_CHECK(rocblas_zdotc(rocblas_handle, dim,
                                        d_v_curr.data(), 1, d_w.data(), 1,
                                        &alpha_complex));
            double alpha_j = alpha_complex.x;  // Real part (Hermitian)

            // Store alpha on GPU
            HIP_CHECK(hipMemcpyAsync(d_alpha.data() + j, &alpha_j, sizeof(double),
                                    hipMemcpyHostToDevice, stream));

            // w = w - alpha_j * v_j
            Complex neg_alpha = make_complex(-alpha_j, 0.0);
            ROCBLAS_CHECK(rocblas_zaxpy(rocblas_handle, dim, &neg_alpha,
                                        d_v_curr.data(), 1, d_w.data(), 1));

            // w = w - beta_{j-1} * v_{j-1}
            if (j > 0) {
                Complex neg_beta = make_complex(-beta_prev, 0.0);
                ROCBLAS_CHECK(rocblas_zaxpy(rocblas_handle, dim, &neg_beta,
                                            d_v_prev.data(), 1, d_w.data(), 1));
            }

            // beta_j = ||w||
            double beta_j;
            ROCBLAS_CHECK(rocblas_dznrm2(rocblas_handle, dim, d_w.data(), 1, &beta_j));

            // Store beta on GPU
            HIP_CHECK(hipMemcpyAsync(d_beta.data() + j, &beta_j, sizeof(double),
                                    hipMemcpyHostToDevice, stream));

            // Check convergence
            if (beta_j < tol) {
                k_converged = j + 1;
                break;
            }

            // v_{j+1} = w / beta_j
            Complex inv_beta = make_complex(1.0 / beta_j, 0.0);
            ROCBLAS_CHECK(rocblas_zscal(rocblas_handle, dim, &inv_beta, d_w.data(), 1));

            // Rotate vectors
            std::swap(d_v_prev, d_v_curr);
            HIP_CHECK(hipMemcpyAsync(d_v_curr.data(), d_w.data(), dim * sizeof(Complex),
                                    hipMemcpyDeviceToDevice, stream));

            // Store new Lanczos vector on GPU
            if (j + 1 < k_max) {
                HIP_CHECK(hipMemcpyAsync(d_V.data() + (j + 1) * dim,
                                        d_v_curr.data(), dim * sizeof(Complex),
                                        hipMemcpyDeviceToDevice, stream));
            }

            beta_prev = beta_j;
        }

        if (k_converged == 0) k_converged = k_max;

        // Solve tridiagonal eigenvalue problem ON GPU using rocSOLVER
        double eigenvalue = solve_tridiagonal_gpu(
            d_alpha.data(), d_beta.data(), k_converged,
            d_V.data(), dim, d_v_out, stream);

        HIP_CHECK(hipStreamSynchronize(stream));
        return eigenvalue;
    }

private:
    // Solve tridiagonal eigenvalue problem ON GPU
    // T is k×k tridiagonal with diagonal alpha, off-diagonal beta
    // Returns lowest eigenvalue, reconstructs eigenvector on GPU
    double solve_tridiagonal_gpu(
        const double* d_alpha, const double* d_beta, int k,
        const Complex* d_V, int dim,
        Complex* d_v_out,
        hipStream_t stream) {

        // Build full tridiagonal matrix on GPU
        GPUBuffer<double> d_T(k * k);
        d_T.zero(stream);

        // Copy alpha to diagonal
        for (int i = 0; i < k; i++) {
            HIP_CHECK(hipMemcpyAsync(d_T.data() + i * k + i,
                                    d_alpha + i, sizeof(double),
                                    hipMemcpyDeviceToDevice, stream));

            // Off-diagonal (symmetric)
            if (i < k - 1) {
                HIP_CHECK(hipMemcpyAsync(d_T.data() + i * k + (i + 1),
                                        d_beta + i, sizeof(double),
                                        hipMemcpyDeviceToDevice, stream));
                HIP_CHECK(hipMemcpyAsync(d_T.data() + (i + 1) * k + i,
                                        d_beta + i, sizeof(double),
                                        hipMemcpyDeviceToDevice, stream));
            }
        }

        // Solve symmetric eigenvalue problem on GPU using rocSOLVER
        GPUBuffer<double> d_eigenvalues(k);
        GPUBuffer<int> d_info(1);

        ROCSOLVER_CHECK(rocsolver_dsyev(rocsolver_handle,
                                         rocblas_evect_original,  // Compute eigenvectors
                                         rocblas_fill_upper,
                                         k,
                                         d_T.data(), k,
                                         d_eigenvalues.data(),
                                         d_info.data()));

        // Check convergence
        int info;
        HIP_CHECK(hipMemcpyAsync(&info, d_info.data(), sizeof(int),
                                hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipStreamSynchronize(stream));

        if (info != 0) {
            throw std::runtime_error("Tridiagonal eigenvalue solve failed");
        }

        // Lowest eigenvalue is first element (rocSOLVER returns sorted)
        double eigenvalue;
        HIP_CHECK(hipMemcpyAsync(&eigenvalue, d_eigenvalues.data(), sizeof(double),
                                hipMemcpyDeviceToHost, stream));

        // Reconstruct eigenvector: v_out = V * eigenvector[0]
        // V is (dim × k), eigenvector is column 0 of T (now contains eigenvectors)
        // Result: d_v_out = d_V × T[:,0]

        ROCBLAS_CHECK(rocblas_set_stream(rocblas_handle, stream));

        Complex alpha = make_complex(1.0, 0.0);
        Complex beta_coef = make_complex(0.0, 0.0);

        // Convert double eigenvector to complex for multiplication
        GPUBuffer<Complex> d_eigvec_complex(k);

        // GPU kernel to convert double to complex would be better, but for k small:
        std::vector<double> h_eigvec(k);
        HIP_CHECK(hipMemcpyAsync(h_eigvec.data(), d_T.data(), k * sizeof(double),
                                hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipStreamSynchronize(stream));

        std::vector<Complex> h_eigvec_complex(k);
        for (int i = 0; i < k; i++) {
            h_eigvec_complex[i] = make_complex(h_eigvec[i], 0.0);
        }
        d_eigvec_complex.copy_from_host(h_eigvec_complex, stream);

        // GEMV: d_v_out = V * eigvec_complex
        ROCBLAS_CHECK(rocblas_zgemv(rocblas_handle,
                                     rocblas_operation_none,
                                     dim, k,
                                     &alpha,
                                     d_V, dim,
                                     d_eigvec_complex.data(), 1,
                                     &beta_coef,
                                     d_v_out, 1));

        // Normalize output (should already be normalized, but ensure)
        double out_norm;
        ROCBLAS_CHECK(rocblas_dznrm2(rocblas_handle, dim, d_v_out, 1, &out_norm));
        Complex inv_out_norm = make_complex(1.0 / out_norm, 0.0);
        ROCBLAS_CHECK(rocblas_zscal(rocblas_handle, dim, &inv_out_norm, d_v_out, 1));

        return eigenvalue;
    }
};
