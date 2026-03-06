#include "../include/tensor_ops.hpp"
#include <hip/hip_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>

namespace dmrg_gpu {

// Tensor contraction using hipTensor
void tensor_contract(
    const GPUTensor<Complex>& A,
    const GPUTensor<Complex>& B,
    GPUTensor<Complex>& C,
    const std::vector<int>& modes_a,
    const std::vector<int>& modes_b,
    const std::vector<int>& modes_c,
    hiptensorHandle_t handle
) {
    // Get dimensions
    const auto& shape_a = A.shape();
    const auto& shape_b = B.shape();
    const auto& shape_c = C.shape();
    
    int ndim_a = shape_a.size();
    int ndim_b = shape_b.size();
    int ndim_c = shape_c.size();
    
    // Convert to int64_t for hipTensor API
    std::vector<int64_t> extent_a(ndim_a), extent_b(ndim_b), extent_c(ndim_c);
    for (int i = 0; i < ndim_a; ++i) extent_a[i] = shape_a[i];
    for (int i = 0; i < ndim_b; ++i) extent_b[i] = shape_b[i];
    for (int i = 0; i < ndim_c; ++i) extent_c[i] = shape_c[i];
    
    // Create tensor descriptors
    hiptensorTensorDescriptor_t desc_a, desc_b, desc_c;
    HIPTENSOR_CHECK(hiptensorInitTensorDescriptor(
        handle, &desc_a, ndim_a, extent_a.data(), nullptr,
        HIP_C_64F, HIPTENSOR_OP_IDENTITY));
    HIPTENSOR_CHECK(hiptensorInitTensorDescriptor(
        handle, &desc_b, ndim_b, extent_b.data(), nullptr,
        HIP_C_64F, HIPTENSOR_OP_IDENTITY));
    HIPTENSOR_CHECK(hiptensorInitTensorDescriptor(
        handle, &desc_c, ndim_c, extent_c.data(), nullptr,
        HIP_C_64F, HIPTENSOR_OP_IDENTITY));
    
    // Contraction descriptor
    hiptensorContractionDescriptor_t desc_contraction;
    uint32_t alignment_req_a, alignment_req_b, alignment_req_c;
    
    HIPTENSOR_CHECK(hiptensorInitContractionDescriptor(
        handle, &desc_contraction,
        &desc_a, modes_a.data(),alignment_req_a,
        &desc_b, modes_b.data(), alignment_req_b,
        &desc_c, modes_c.data(), alignment_req_c,
        &desc_c, modes_c.data(), alignment_req_c,
        HIPTENSOR_COMPUTE_64F));
    
    // Find contraction plan
    hiptensorContractionFind_t find;
    HIPTENSOR_CHECK(hiptensorInitContractionFind(handle, &find,
                    HIPTENSOR_ALGO_DEFAULT));
    
    // Get workspace size
    uint64_t workspace_size = 0;
    HIPTENSOR_CHECK(hiptensorContractionGetWorkspaceSize(
        handle, &desc_contraction, &find, HIPTENSOR_WORKSPACE_RECOMMENDED,
        &workspace_size));
    
    // Allocate workspace
    void* workspace = nullptr;
    if (workspace_size > 0) {
        HIP_CHECK(hipMalloc(&workspace, workspace_size));
    }
    
    // Create contraction plan
    hiptensorContractionPlan_t plan;
    HIPTENSOR_CHECK(hiptensorInitContractionPlan(handle, &plan,
                    &desc_contraction, &find, workspace_size));
    
    // Execute contraction: C = alpha * A * B + beta * C
    hipDoubleComplex alpha = make_hipDoubleComplex(1.0, 0.0);
    hipDoubleComplex beta = make_hipDoubleComplex(0.0, 0.0);
    
    HIPTENSOR_CHECK(hiptensorContraction(
        handle, &plan,
        &alpha, A.data(), B.data(),
        &beta, C.data(), C.data(),
        workspace, workspace_size, 0 /* stream */));
    
    // Cleanup
    if (workspace) hipFree(workspace);
}

// Apply H_eff to a vector (for Lanczos)
// This is the core DMRG operation: contracts MPO with environments
void apply_hamiltonian_eff(
    const GPUTensor<Complex>& x,
    GPUTensor<Complex>& y,
    const MPO& H,
    const MPS& left_env,
    const MPS& right_env,
    int site,
    hiptensorHandle_t handle
) {
    // TODO: Implement full contraction sequence
    // For now, placeholder that copies x to y
    // Full implementation requires:
    // 1. Reshape x to MPS tensor shape
    // 2. Contract with left environment
    // 3. Contract with MPO at site
    // 4. Contract with right environment
    // 5. Reshape result back to vector y
    
    std::vector<Complex> x_host(x.size());
    x.copy_to_host(x_host.data());
    y.copy_from_host(x_host.data());
}

// Truncated SVD using rocSOLVER
void truncated_svd(
    const GPUTensor<Complex>& M,
    GPUTensor<Complex>& U,
    GPUTensor<Real>& S,
    GPUTensor<Complex>& Vt,
    int max_bond_dim,
    rocblas_handle rbhandle
) {
    // Get matrix dimensions
    const auto& shape = M.shape();
    if (shape.size() != 2) {
        std::cerr << "SVD requires 2D tensor" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    int m = shape[0];
    int n = shape[1];
    int k = std::min({m, n, max_bond_dim});
    
    // Allocate for full SVD first
    std::vector<Complex> M_host(m * n);
    M.copy_to_host(M_host.data());
    
    // Copy to device buffer for rocSOLVER (may need workspace)
    Complex* d_M_work;
    Real* d_S;
    Complex* d_U;
    Complex* d_Vt;
    
    HIP_CHECK(hipMalloc(&d_M_work, m * n * sizeof(Complex)));
    HIP_CHECK(hipMalloc(&d_S, k * sizeof(Real)));
    HIP_CHECK(hipMalloc(&d_U, m * k * sizeof(Complex)));
    HIP_CHECK(hipMalloc(&d_Vt, k * n * sizeof(Complex)));
    
    HIP_CHECK(hipMemcpy(d_M_work, M_host.data(), m * n * sizeof(Complex),
                        hipMemcpyHostToDevice));
    
    // Workspace query
    size_t lwork;
    rocsolver_zgesvd_strided_batched(rbhandle, rocsolver_svect_singular,
                                      rocsolver_svect_singular,
                                      m, n, nullptr, m, 0,
                                      nullptr, 0, nullptr, m, 0,
                                      nullptr, n, 0,
                                      nullptr, lwork, nullptr, 1);
    
    // Allocate workspace
    Complex* d_work;
    int* d_info;
    HIP_CHECK(hipMalloc(&d_work, lwork * sizeof(Complex)));
    HIP_CHECK(hipMalloc(&d_info, sizeof(int)));
    
    // Compute SVD: M = U * S * Vt (row-major Vt = column-major V^H)
    rocsolver_zgesvd_strided_batched(
        rbhandle,
        rocsolver_svect_singular,  // Compute thin U
        rocsolver_svect_singular,  // Compute thin Vt
        m, n,
        reinterpret_cast<rocblas_double_complex*>(d_M_work), m, 0,
        d_S, 0,
        reinterpret_cast<rocblas_double_complex*>(d_U), m, 0,
        reinterpret_cast<rocblas_double_complex*>(d_Vt), n, 0,
        reinterpret_cast<rocblas_double_complex*>(d_work), lwork,
        d_info, 1);
    
    // Check for errors
    int info;
    HIP_CHECK(hipMemcpy(&info, d_info, sizeof(int), hipMemcpyDeviceToHost));
    if (info != 0) {
        std::cerr << "rocSOLVER SVD failed with info = " << info << std::endl;
        exit(EXIT_FAILURE);
    }
    
    // Copy results to output tensors (truncated to k)
    U.copy_from_device(d_U, m * k);
    S.copy_from_device(d_S, k);
    Vt.copy_from_device(d_Vt, k * n);
    
    // Cleanup
    hipFree(d_M_work);
    hipFree(d_S);
    hipFree(d_U);
    hipFree(d_Vt);
    hipFree(d_work);
    hipFree(d_info);
}
// Test hipTensor with complex128 (hipDoubleComplex)
// Priority: Accuracy over performance
// This validates hipTensor can handle complex numbers correctly

#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include <hiptensor/hiptensor.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <complex>

// Helper to convert std::complex to hipDoubleComplex
__host__ __device__ hipDoubleComplex to_hip_complex(const std::complex<double>& z) {
}

// Helper to convert hipDoubleComplex to std::complex
std::complex<double> from_hip_complex(const hipDoubleComplex& z) {
}

int main() {

    // Test parameters (small for accuracy verification)
    const int D = 4;  // Bond dimension
    const int d = 2;  // Physical dimension (spin-1/2)


    // Allocate host memory and initialize with test data

    // Initialize with known values (including complex components)
    for (int i = 0; i < D; i++) {
        for (int j = 0; j < d; j++) {
            for (int k = 0; k < D; k++) {
                // Use non-trivial complex values
            }
        }
    }

    // Allocate device memory

    // Copy to device

    // Create hipTensor handle
    if (status != HIPTENSOR_STATUS_SUCCESS) {
    }

    // TODO: Implement tensor contraction using hipTensor API
    // For now, we'll do a manual verification using CPU calculation


    // CPU reference calculation: C[i,j,l,m] = sum_k A[i,j,k] * B[k,l,m]
    for (int i = 0; i < D; i++) {
        for (int j = 0; j < d; j++) {
            for (int l = 0; l < d; l++) {
                for (int m = 0; m < D; m++) {
                    for (int k = 0; k < D; k++) {
                    }
                }
            }
        }
    }

    // Verify result (comparing first few elements)
    for (int i = 0; i < 4; i++) {
        std::cout << "  C[" << i << "] = "
    }

    // Check that we got complex results (imaginary parts non-zero)
    for (const auto& c : ref_C) {
        if (std::abs(c.imag()) > 1e-14) {
        }
    }

    if (has_complex) {
    } else {
    }

    // Cleanup

ls -la ~/dmrg-implementations/gpu-port/include/ 2>/dev/null || echo 'Directory does not exist yet'

}
