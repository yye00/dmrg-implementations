// Tensor operations for GPU DMRG - exact contractions for 100% accuracy
// Priority: 1. Accuracy (exact match with Quimb), 2. Performance

#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include <rocblas/rocblas.h>
#include <iostream>
#include <vector>
#include <stdexcept>
#include "gpu_memory.hpp"

using Complex = hipDoubleComplex;

// Tensor contraction helper using rocBLAS GEMM
// For exact accuracy, we carefully reshape tensors into matrices and use GEMM
class TensorOpsGPU {
private:
    rocblas_handle handle;

public:
    TensorOpsGPU() {
        ROCBLAS_CHECK(rocblas_create_handle(&handle));
    }

    ~TensorOpsGPU() {
        rocblas_destroy_handle(handle);
    }

    // Contract 2-site tensor: theta[i,j,k,l] from A[i,j,a] * B[a,k,l]
    // Einsum: 'ija,akl->ijkl'
    // Dimension: (D_L, d, D_mid) * (D_mid, d, D_R) -> (D_L, d, d, D_R)
    void contract_2site_forward(
        const Complex* d_A, int D_L, int d1, int D_mid,
        const Complex* d_B, int D_mid2, int d2, int D_R,
        Complex* d_theta,
        hipStream_t stream = 0) {

        if (D_mid != D_mid2) {
            throw std::runtime_error("Dimension mismatch in 2-site contraction");
        }

        // Reshape: A[D_L, d1, D_mid] -> A_mat[D_L*d1, D_mid]
        // Reshape: B[D_mid, d2, D_R] -> B_mat[D_mid, d2*D_R]
        // GEMM: C = A_mat * B_mat -> C[D_L*d1, d2*D_R]
        // Reshape: C -> theta[D_L, d1, d2, D_R]

        ROCBLAS_CHECK(rocblas_set_stream(handle, stream));

        int m = D_L * d1;
        int n = d2 * D_R;
        int k = D_mid;

        Complex alpha = make_complex(1.0, 0.0);
        Complex beta = make_complex(0.0, 0.0);

        // rocBLAS uses column-major, so we need to transpose our row-major tensors
        // C = A * B becomes C^T = B^T * A^T in column-major
        ROCBLAS_CHECK(rocblas_zgemm(handle,
                                     rocblas_operation_none,
                                     rocblas_operation_none,
                                     n, m, k,
                                     &alpha,
                                     d_B, n,  // B^T in col-major = B in row-major
                                     d_A, k,  // A^T in col-major = A in row-major
                                     &beta,
                                     d_theta, n));
    }

    // Contract with MPO: result[i,j',k,l] = sum_j theta[i,j,k,l] * W[j',j]
    // For 2-site with MPO: more complex, needs multiple contractions
    // Einsum: 'ijkl,abij,cdkl->abcd' (simplified)
    void contract_with_mpo_2site(
        const Complex* d_theta, int D_L, int d, int D_R,
        const Complex* d_W1, int DL1, int d_in1, int d_out1, int DR1,
        const Complex* d_W2, int DL2, int d_in2, int d_out2, int DR2,
        Complex* d_result,
        hipStream_t stream = 0) {

        // This is a complex multi-stage contraction
        // For accuracy, we do this carefully in steps
        // TODO: Implement full contraction chain
        // For now, placeholder that uses GEMM-based approach

        throw std::runtime_error("contract_with_mpo_2site: Implementation needed");
    }

    // Build effective Hamiltonian for 2-site DMRG
    // H_eff contracts: L[i] - A[i]† - M[i]† - A[i+1]† - M[i+1]† - R[i+2]
    //                  with bottom layer for matrix elements
    void build_H_eff_2site(
        const Complex* d_L, int D_L_mps, int D_L_mpo,
        const Complex* d_R, int D_R_mpo, int D_R_mps,
        const Complex* d_M1, int DL_M1, int d1_in, int d1_out, int DR_M1,
        const Complex* d_M2, int DL_M2, int d2_in, int d2_out, int DR_M2,
        Complex* d_H_eff, int& H_dim,
        hipStream_t stream = 0) {

        // Build effective Hamiltonian as dense matrix
        // Dimension: (D_L_mps * d1 * d2 * D_R_mps)^2
        H_dim = D_L_mps * d1_in * d2_in * D_R_mps;

        // Complex contraction - needs careful implementation
        // For exact accuracy, contract step by step

        // TODO: Implement exact contraction sequence
        throw std::runtime_error("build_H_eff_2site: Implementation needed");
    }

    // Update left environment: L[i+1] = contract(L[i], A[i], M[i], A[i]†)
    void update_left_env(
        const Complex* d_L_old, int D_L_old_mps, int D_L_old_mpo,
        const Complex* d_A, int D_L, int d, int D_R,
        const Complex* d_M, int DL_M, int d_in, int d_out, int DR_M,
        Complex* d_L_new, int D_R_mps, int DR_mpo,
        hipStream_t stream = 0) {

        // Contract: L_new[b,beta] = sum_{a,alpha,s} L_old[a,alpha] * A[a,s,b] * M[alpha,s,s',beta] * A†[a,s',b]
        // This is a complex 4-tensor contraction

        // TODO: Implement exact contraction
        throw std::runtime_error("update_left_env: Implementation needed");
    }

    // Update right environment: R[i-1] = contract(A[i]†, M[i], A[i], R[i])
    void update_right_env(
        const Complex* d_A, int D_L, int d, int D_R,
        const Complex* d_M, int DL_M, int d_in, int d_out, int DR_M,
        const Complex* d_R_old, int D_R_old_mpo, int D_R_old_mps,
        Complex* d_R_new, int DL_mpo, int D_L_mps,
        hipStream_t stream = 0) {

        // Contract: R_new[alpha,a] = sum_{b,beta,s} A†[a,s,b] * M[alpha,s,s',beta] * A[a,s',b] * R_old[beta,b]

        // TODO: Implement exact contraction
        throw std::runtime_error("update_right_env: Implementation needed");
    }

    // Reshape 2-site tensor for SVD: theta[D_L,d1,d2,D_R] -> matrix[D_L*d1, d2*D_R]
    void reshape_for_svd(
        const Complex* d_theta, int D_L, int d1, int d2, int D_R,
        Complex* d_matrix,
        hipStream_t stream = 0) {

        // Simple memory copy since our tensor is already in the right layout
        int size = D_L * d1 * d2 * D_R;
        HIP_CHECK(hipMemcpyAsync(d_matrix, d_theta, size * sizeof(Complex),
                                hipMemcpyDeviceToDevice, stream));
    }

    // Reshape SVD results back to MPS tensors
    void reshape_svd_to_mps(
        const Complex* d_U, const double* d_S, const Complex* d_Vh,
        int D_L, int d1, int k, int d2, int D_R,
        Complex* d_A_left, Complex* d_A_right,
        hipStream_t stream = 0) {

        // A_left[D_L, d1, k] = U[D_L*d1, k] * sqrt(S[k])
        // A_right[k, d2, D_R] = sqrt(S[k]) * Vh[k, d2*D_R]

        // Apply sqrt(S) using rocBLAS diagonal scaling
        GPUBuffer<Complex> d_sqrt_S(k);
        std::vector<Complex> h_sqrt_S(k);
        std::vector<double> h_S(k);

        HIP_CHECK(hipMemcpyAsync(h_S.data(), d_S, k * sizeof(double),
                                hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipStreamSynchronize(stream));

        for (int i = 0; i < k; i++) {
            double s_sqrt = std::sqrt(h_S[i]);
            h_sqrt_S[i] = make_complex(s_sqrt, 0.0);
        }

        d_sqrt_S.copy_from_host(h_sqrt_S, stream);

        // Scale U columns by sqrt(S)
        ROCBLAS_CHECK(rocblas_set_stream(handle, stream));
        for (int j = 0; j < k; j++) {
            const Complex* d_U_col = d_U + j * (D_L * d1);
            Complex* d_A_col = d_A_left + j * (D_L * d1);

            HIP_CHECK(hipMemcpyAsync(d_A_col, d_U_col, (D_L * d1) * sizeof(Complex),
                                    hipMemcpyDeviceToDevice, stream));

            ROCBLAS_CHECK(rocblas_zscal(handle, D_L * d1,
                                        d_sqrt_S.data() + j,
                                        d_A_col, 1));
        }

        // Scale Vh rows by sqrt(S)
        for (int i = 0; i < k; i++) {
            const Complex* d_Vh_row = d_Vh + i * (d2 * D_R);
            Complex* d_A_row = d_A_right + i * (d2 * D_R);

            HIP_CHECK(hipMemcpyAsync(d_A_row, d_Vh_row, (d2 * D_R) * sizeof(Complex),
                                    hipMemcpyDeviceToDevice, stream));

            ROCBLAS_CHECK(rocblas_zscal(handle, d2 * D_R,
                                        d_sqrt_S.data() + i,
                                        d_A_row, 1));
        }
    }
};
