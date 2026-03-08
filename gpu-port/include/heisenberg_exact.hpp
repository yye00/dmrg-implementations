// Exact 2-site Heisenberg Hamiltonian for DMRG
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include <rocblas/rocblas.h>
#include <vector>

using Complex = hipDoubleComplex;

inline Complex make_complex(double re, double im) {
    return hipDoubleComplex{re, im};
}

void apply_heisenberg_exact(const Complex* d_psi, Complex* d_Hpsi, int D_L, int d, int D_R, rocblas_handle handle) {
    std::vector<Complex> h_H(16, make_complex(0.0, 0.0));
    h_H[0] = make_complex(0.25, 0.0);
    h_H[5] = make_complex(-0.25, 0.0);
    h_H[6] = make_complex(0.5, 0.0);
    h_H[9] = make_complex(0.5, 0.0);
    h_H[10] = make_complex(-0.25, 0.0);
    h_H[15] = make_complex(0.25, 0.0);
    
    Complex* d_H;
    hipMalloc(&d_H, 16 * sizeof(Complex));
    hipMemcpy(d_H, h_H.data(), 16 * sizeof(Complex), hipMemcpyHostToDevice);
    
    Complex alpha = make_complex(1.0, 0.0);
    Complex beta = make_complex(0.0, 0.0);
    int batch_count = D_L * D_R;
    
    rocblas_zgemm_strided_batched(handle, rocblas_operation_none, rocblas_operation_none,
        1, 4, 4, (rocblas_double_complex*)&alpha,
        (rocblas_double_complex*)d_psi, 1, 4,
        (rocblas_double_complex*)d_H, 4, 0,
        (rocblas_double_complex*)&beta,
        (rocblas_double_complex*)d_Hpsi, 1, 4, batch_count);
    
    hipFree(d_H);
}
