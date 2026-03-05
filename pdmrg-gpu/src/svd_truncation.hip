// SVD Truncation for DMRG using rocSOLVER
// Priority: 1. Accuracy (complex128), 2. Performance
//
// SVD is critical for DMRG: truncates the Schmidt decomposition to keep bond dimension manageable

#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include <rocsolver/rocsolver.h>
#include <iostream>
#include <vector>
#include <complex>
#include <algorithm>
#include <cmath>

using Complex = hipDoubleComplex;

class SVDTruncator {
private:
    rocblas_handle handle;
    int max_bond_dim;
    double truncation_error;
    
public:
    SVDTruncator(int max_D = 100, double trunc_err = 1e-10)
        : max_bond_dim(max_D), truncation_error(trunc_err)
    {
        rocblas_create_handle(&handle);
    }
    
    ~SVDTruncator() {
        rocblas_destroy_handle(handle);
    }
    
    // Perform SVD: M = U * S * V^†
    // For DMRG, M is the wavefunction reshaped as a matrix
    // Returns: actual bond dimension kept after truncation
    int truncate_svd(
        Complex* d_M, int m, int n,  // Input matrix (will be destroyed)
        Complex* d_U, double* d_S, Complex* d_Vt,  // Output: U, S, V^dagger
        int& D_kept)  // Output: bond dimension kept
    {
        // rocSOLVER uses: rocsolver_zgesvd for complex128
        // 'S' = compute singular vectors
        
        char jobu = 'S';   // Compute first min(m,n) columns of U
        char jobvt = 'S';  // Compute first min(m,n) rows of V^†
        
        int lda = m;
        int ldu = m;
        int ldvt = std::min(m, n);
        int min_mn = std::min(m, n);
        
        // Allocate singular values on device
        double* d_rwork;  // Real workspace for complex SVD
        int* d_info;
        
        hipMalloc(&d_rwork, 5 * min_mn * sizeof(double));
        hipMalloc(&d_info, sizeof(int));
        
        // Query workspace size
        int lwork = 0;
        rocsolver_zgesvd_strided_batched(
            handle, jobu, jobvt,
            m, n,
            (rocblas_double_complex*)d_M, lda, 0,
            d_S, 0,
            (rocblas_double_complex*)d_U, ldu, 0,
            (rocblas_double_complex*)d_Vt, ldvt, 0,
            (rocblas_double_complex*)nullptr, lwork,
            d_rwork,
            d_info,
            1  // batch_count = 1
        );
        
        // Actually compute SVD
        // Note: This destroys the input matrix d_M
        rocblas_int info_h = 0;
        rocsolver_zgesvd(
            handle, jobu, jobvt,
            m, n,
            (rocblas_double_complex*)d_M, lda,
            d_S,
            (rocblas_double_complex*)d_U, ldu,
            (rocblas_double_complex*)d_Vt, ldvt,
            (rocblas_double_complex*)nullptr, // work (auto-allocated)
            lwork,
            d_rwork,
            d_info
        );
        
        // Check for convergence
        hipMemcpy(&info_h, d_info, sizeof(int), hipMemcpyDeviceToHost);
        if (info_h != 0) {
            std::cerr << "SVD did not converge, info = " << info_h << std::endl;
        }
        
        // Copy singular values to host to determine truncation
        std::vector<double> h_S(min_mn);
        hipMemcpy(h_S.data(), d_S, min_mn * sizeof(double), hipMemcpyDeviceToHost);
        
        // Determine how many singular values to keep
        // Strategy 1: Keep until truncation error threshold
        double total_weight = 0.0;
        for (double s : h_S) {
            total_weight += s * s;
        }
        
        double kept_weight = 0.0;
        D_kept = 0;
        for (int i = 0; i < min_mn && D_kept < max_bond_dim; i++) {
            kept_weight += h_S[i] * h_S[i];
            D_kept++;
            
            double discarded_weight = total_weight - kept_weight;
            if (discarded_weight / total_weight < truncation_error) {
                break;
            }
        }
        
        std::cout << "SVD: kept " << D_kept << " / " << min_mn 
                  << " singular values (max_D=" << max_bond_dim << ")" << std::endl;
        
        // Compute truncation error
        double actual_trunc_error = 0.0;
        for (int i = D_kept; i < min_mn; i++) {
            actual_trunc_error += h_S[i] * h_S[i];
        }
        actual_trunc_error = std::sqrt(actual_trunc_error / total_weight);
        
        std::cout << "  Truncation error: " << actual_trunc_error << std::endl;
        
        hipFree(d_rwork);
        hipFree(d_info);
        
        return D_kept;
    }
    
    // Apply SVD truncation in DMRG: reshape wavefunction, truncate, update MPS tensors
    void truncate_wavefunction(
        const Complex* d_psi,  // Input: 2-site wavefunction [D_left, d, d, D_right]
        int D_left, int d, int D_right,
        Complex* d_A_left,     // Output: left tensor [D_left, d, D_new]
        Complex* d_A_right,    // Output: right tensor [D_new, d, D_right]
        int& D_new)            // Output: new bond dimension
    {
        int m = D_left * d;
        int n = d * D_right;
        int min_mn = std::min(m, n);
        
        // Allocate workspace
        Complex* d_M;  // Matrix form of wavefunction
        Complex* d_U;
        double* d_S;
        Complex* d_Vt;
        
        hipMalloc(&d_M, m * n * sizeof(Complex));
        hipMalloc(&d_U, m * min_mn * sizeof(Complex));
        hipMalloc(&d_S, min_mn * sizeof(double));
        hipMalloc(&d_Vt, min_mn * n * sizeof(Complex));
        
        // Reshape wavefunction to matrix (this might require a kernel or hipMemcpy2D)
        reshape_to_matrix(d_psi, d_M, D_left, d, D_right);
        
        // Perform SVD with truncation
        truncate_svd(d_M, m, n, d_U, d_S, d_Vt, D_new);
        
        // Construct left tensor: A_left[i,s,a] = U[i*d+s, a] * sqrt(S[a])
        construct_left_tensor(d_U, d_S, d_A_left, D_left, d, D_new);
        
        // Construct right tensor: A_right[a,s,j] = sqrt(S[a]) * Vt[a, s*D_right+j]
        construct_right_tensor(d_Vt, d_S, d_A_right, D_new, d, D_right);
        
        hipFree(d_M);
        hipFree(d_U);
        hipFree(d_S);
        hipFree(d_Vt);
    }
    
private:
    void reshape_to_matrix(const Complex* d_psi, Complex* d_M,
                          int D_left, int d, int D_right) {
        // Reshape [D_left, d, d, D_right] -> [D_left*d, d*D_right]
        // This requires a custom HIP kernel for efficiency
        // For now, placeholder
        int m = D_left * d;
        int n = d * D_right;
        hipMemcpy(d_M, d_psi, m * n * sizeof(Complex), hipMemcpyDeviceToDevice);
    }
    
    void construct_left_tensor(const Complex* d_U, const double* d_S,
                               Complex* d_A_left, 
                               int D_left, int d, int D_new) {
        // A_left[i,s,a] = U[i*d+s, a] * sqrt(S[a])
        // Requires custom kernel or series of rocBLAS operations
    }
    
    void construct_right_tensor(const Complex* d_Vt, const double* d_S,
                                Complex* d_A_right,
                                int D_new, int d, int D_right) {
        // A_right[a,s,j] = sqrt(S[a]) * Vt[a, s*D_right+j]
        // Requires custom kernel or series of rocBLAS operations
    }
};

