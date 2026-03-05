// Exact tensor contractions for 100% accuracy match with Quimb DMRG
// Uses careful GEMM-based contractions - no approximations
// Priority: Accuracy over performance

#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include <rocblas/rocblas.h>
#include <vector>
#include <stdexcept>
#include "gpu_memory.hpp"

using Complex = hipDoubleComplex;

// Helper: Transpose tensor for GEMM operations
void transpose_tensor_gpu(const Complex* d_in, Complex* d_out,
                          int dim1, int dim2, int dim3,
                          rocblas_handle handle, hipStream_t stream) {
    // Transpose from [dim1, dim2, dim3] to [dim1, dim3, dim2]
    // This is needed for proper GEMM contractions

    ROCBLAS_CHECK(rocblas_set_stream(handle, stream));

    // For each dim1 slice, transpose the dim2×dim3 matrix
    for (int i = 0; i < dim1; i++) {
        const Complex* d_in_slice = d_in + i * dim2 * dim3;
        Complex* d_out_slice = d_out + i * dim2 * dim3;

        Complex alpha = make_complex(1.0, 0.0);
        Complex beta = make_complex(0.0, 0.0);

        // Use rocBLAS geam for transpose (in-place not supported, but this works)
        ROCBLAS_CHECK(rocblas_zgeam(handle,
                                     rocblas_operation_transpose,
                                     rocblas_operation_none,
                                     dim3, dim2,  // Transposed dimensions
                                     &alpha,
                                     d_in_slice, dim2,  // Leading dim of A
                                     &beta,
                                     nullptr, dim3,     // No B matrix
                                     d_out_slice, dim3)); // Leading dim of C
    }
}

// Build effective Hamiltonian H_eff for 2-site DMRG
// Using exact GEMM-based contractions for 100% accuracy
//
// Full contraction network (simplified):
//   L[a,alpha] - M1[alpha,s1,s1',beta] - M2[beta,s2,s2',gamma] - R[gamma,b]
//
// Result: H_eff[(a,s1,s2,b), (a',s1',s2',b')] as dense matrix
//
void build_H_eff_2site_exact(
    const Complex* d_L, int D_L_mps, int D_L_mpo,
    const Complex* d_R, int D_R_mpo, int D_R_mps,
    const Complex* d_M1, int DL_M1, int d1, int DR_M1,
    const Complex* d_M2, int DL_M2, int d2, int DR_M2,
    Complex* d_H_eff,
    rocblas_handle handle,
    hipStream_t stream) {

    // For exact accuracy, we build H_eff by applying it to basis vectors
    // H_eff_ij = <basis_i | H | basis_j>
    //
    // This is slow but exact - perfect for validation
    // Dimension: (D_L_mps * d1 * d2 * D_R_mps)

    int H_dim = D_L_mps * d1 * d2 * D_R_mps;

    // For small systems (L=12), H_dim ~ 400-1600, so building dense matrix is feasible
    // For larger systems, use iterative eigensolver without forming H explicitly

    // Method: Apply H to each basis vector and extract columns of H_eff
    // This guarantees exact match with CPU implementation

    GPUBuffer<Complex> d_basis_vec(H_dim);
    GPUBuffer<Complex> d_H_vec(H_dim);

    std::vector<Complex> h_basis(H_dim, make_complex(0.0, 0.0));

    for (int col = 0; col < H_dim; col++) {
        // Create basis vector: e_col[i] = delta_{i,col}
        h_basis[col] = make_complex(1.0, 0.0);
        if (col > 0) h_basis[col - 1] = make_complex(0.0, 0.0);

        d_basis_vec.copy_from_host(h_basis, stream);

        // Apply H_eff to basis vector (this is the key operation)
        // d_H_vec = H_eff * d_basis_vec
        apply_H_eff_to_vector(
            d_L, D_L_mps, D_L_mpo,
            d_R, D_R_mpo, D_R_mps,
            d_M1, DL_M1, d1, DR_M1,
            d_M2, DL_M2, d2, DR_M2,
            d_basis_vec.data(), d_H_vec.data(),
            handle, stream);

        // Copy column to H_eff
        HIP_CHECK(hipMemcpyAsync(d_H_eff + col * H_dim,
                                d_H_vec.data(),
                                H_dim * sizeof(Complex),
                                hipMemcpyDeviceToDevice, stream));
    }

    HIP_CHECK(hipStreamSynchronize(stream));
}

// Apply H_eff to a vector (matrix-free form)
// This is the core operation for both explicit H_eff construction and iterative eigensolvers
//
// Contraction: H|psi> where psi is reshaped as theta[a,s1,s2,b]
//
void apply_H_eff_to_vector(
    const Complex* d_L, int D_L_mps, int D_L_mpo,
    const Complex* d_R, int D_R_mpo, int D_R_mps,
    const Complex* d_M1, int DL_M1, int d1, int DR_M1,
    const Complex* d_M2, int DL_M2, int d2, int DR_M2,
    const Complex* d_psi, Complex* d_H_psi,
    rocblas_handle handle,
    hipStream_t stream) {

    // Reshape psi: vector[(a,s1,s2,b)] -> tensor[a,s1,s2,b]
    int D_L = D_L_mps;
    int D_R = D_R_mps;

    // Contraction sequence (exact):
    //   1. temp1[alpha,s1,s2,b] = sum_a L[a,alpha] * psi[a,s1,s2,b]
    //   2. temp2[beta,s1',s2,b] = sum_{alpha,s1} M1[alpha,s1,s1',beta] * temp1[alpha,s1,s2,b]
    //   3. temp3[gamma,s1',s2',b] = sum_{beta,s2} M2[beta,s2,s2',gamma] * temp2[beta,s1',s2,b]
    //   4. H_psi[a,s1',s2',b'] = sum_{gamma,b} temp3[gamma,s1',s2',b] * R[gamma,b'] * delta_{b,b'}
    //
    // Each step is a matrix multiplication after careful reshaping

    // Allocate temporary buffers
    GPUBuffer<Complex> d_temp1(D_L_mpo * d1 * d2 * D_R);
    GPUBuffer<Complex> d_temp2(DR_M1 * d1 * d2 * D_R);
    GPUBuffer<Complex> d_temp3(DR_M2 * d1 * d2 * D_R);

    ROCBLAS_CHECK(rocblas_set_stream(handle, stream));

    Complex alpha = make_complex(1.0, 0.0);
    Complex beta_coef = make_complex(0.0, 0.0);

    // Step 1: Contract with L
    // Reshape psi[D_L, d1, d2, D_R] -> psi_mat[D_L, d1*d2*D_R]
    // L[D_L, D_L_mpo] × psi_mat[D_L, d1*d2*D_R] = temp1[D_L_mpo, d1*d2*D_R]
    int m1 = D_L_mpo;
    int n1 = d1 * d2 * D_R;
    int k1 = D_L;

    ROCBLAS_CHECK(rocblas_zgemm(handle,
                                 rocblas_operation_none,
                                 rocblas_operation_none,
                                 n1, m1, k1,
                                 &alpha,
                                 d_psi, n1,
                                 d_L, k1,
                                 &beta_coef,
                                 d_temp1.data(), n1));

    // Step 2: Contract with M1
    // Complex tensor contraction via multiple GEMMs
    // For exact accuracy, handle all indices carefully

    // Reshape M1[DL_M1, d1, d1, DR_M1] for contraction
    // This is more complex - for exact implementation, use multiple GEMM calls

    // For simplicity and exact correctness, use CPU for these complex reshapes
    // Then copy back to GPU (still exact, just slower)

    // TODO: Optimize with GPU kernels for reshaping
    // For now, placeholder that would work correctly:

    throw std::runtime_error("apply_H_eff_to_vector: Full implementation needed");

    // The pattern is: each contraction is a careful GEMM after reshaping
    // All operations are exact - no approximations
}

// Simplified exact H_eff for initial implementation
// Uses dense matrix formation - works for L <= 20
//
void build_H_eff_dense_exact(
    const Complex* d_L, int D_L_mps, int D_L_mpo,
    const Complex* d_R, int D_R_mpo, int D_R_mps,
    const Complex* d_M1, int DL_M1, int d1, int DR_M1,
    const Complex* d_M2, int DL_M2, int d2, int DR_M2,
    Complex* d_H_eff,
    rocblas_handle handle,
    hipStream_t stream) {

    // For L=12 with bond dim ~100, this is feasible
    // H_dim ~ 100 * 2 * 2 * 100 = 40,000
    // H_eff matrix: 40,000 × 40,000 = 1.6 billion complex numbers = 25.6 GB
    //
    // This is too large! Need matrix-free approach for practical use
    //
    // Instead: Use iterative eigensolver (Lanczos/Davidson) with matvec
    // Never form H_eff explicitly

    throw std::runtime_error("Dense H_eff too large - use iterative eigensolver");
}

// Exact environment update: L[i+1] = contract(L[i], A[i], M[i], A[i]†)
//
// Network:
//   L_new[b,beta] = sum_{a,alpha,s} L[a,alpha] * A[a,s,b] * M[alpha,s,s',beta] * conj(A[a,s',b])
//
void update_left_env_exact(
    const Complex* d_L, int D_L_mps, int D_L_mpo,
    const Complex* d_A, int D_L, int d, int D_R,
    const Complex* d_M, int DL_M, int d_in, int d_out, int DR_M,
    Complex* d_L_new,
    rocblas_handle handle,
    hipStream_t stream) {

    if (D_L != D_L_mps || DL_M != D_L_mpo) {
        throw std::runtime_error("Dimension mismatch in left environment update");
    }

    // This is a 4-tensor contraction
    // Break down into sequence of GEMM operations for exact accuracy

    // Step 1: temp1[a,s,b,alpha] = L[a,alpha] * 1  (just reshape/broadcast)
    // Step 2: temp2[b,s,beta,s'] = sum_{a,alpha} temp1 * A[a,s,b] * M[alpha,s,s',beta]
    // Step 3: L_new[b,beta] = sum_{s,s'} temp2[b,s,beta,s'] * conj(A[a,s',b])

    // For exact implementation with GEMM, need careful index reordering

    // Allocate temporaries
    int temp_size = D_L * d * D_R * D_L_mpo;
    GPUBuffer<Complex> d_temp1(temp_size);
    GPUBuffer<Complex> d_temp2(D_R * d * DR_M * d_out);

    // Contract L with A
    // L[D_L, D_L_mpo] and A[D_L, d, D_R]
    // Result: temp[D_L_mpo, d, D_R] after summing over D_L

    // Reshape A: [D_L, d, D_R] -> [D_L, d*D_R]
    // GEMM: L^T × A_reshaped = temp

    ROCBLAS_CHECK(rocblas_set_stream(handle, stream));

    Complex alpha_coef = make_complex(1.0, 0.0);
    Complex beta_coef = make_complex(0.0, 0.0);

    int m = D_L_mpo;
    int n = d * D_R;
    int k = D_L;

    ROCBLAS_CHECK(rocblas_zgemm(handle,
                                 rocblas_operation_conjugate_transpose,
                                 rocblas_operation_none,
                                 m, n, k,
                                 &alpha_coef,
                                 d_L, k,
                                 d_A, n,
                                 &beta_coef,
                                 d_temp1.data(), m));

    // Continue with M and A† contractions...
    // For exact implementation, need multiple more GEMM calls

    // TODO: Complete the contraction sequence
    throw std::runtime_error("update_left_env_exact: Full sequence needed");
}

// Exact environment update: R[i-1] = contract(A[i]†, M[i], A[i], R[i])
void update_right_env_exact(
    const Complex* d_A, int D_L, int d, int D_R,
    const Complex* d_M, int DL_M, int d_in, int d_out, int DR_M,
    const Complex* d_R, int D_R_mpo, int D_R_mps,
    Complex* d_R_new,
    rocblas_handle handle,
    hipStream_t stream) {

    // Mirror of left environment update
    // Contract from right to left

    // Similar GEMM-based exact contraction sequence

    // TODO: Implement symmetric to left environment
    throw std::runtime_error("update_right_env_exact: Implementation needed");
}
