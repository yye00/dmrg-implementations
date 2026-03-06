// ============================================================================
// LoadedMPO - Wrapper for loaded MPO tensors compatible with MPOBase interface
// ============================================================================
//
// This class wraps MPO tensors loaded from binary files and makes them
// compatible with the PDMRG_GPU MPOBase interface.
//
// Usage:
//   auto mpo_host = MPOLoader::load("path/to/mpo.bin");
//   auto* loaded_mpo = new LoadedMPO(mpo_host);
//   PDMRG_GPU dmrg(loaded_mpo, ...);
//   double energy = dmrg.run();

#pragma once

#include "mps_mpo_loader.hpp"
#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include <vector>
#include <complex>
#include <iostream>

using Complex = hipDoubleComplex;

#define HIP_CHECK(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ \
                  << ": " << hipGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

#ifndef MAKE_COMPLEX_DEFINED
#define MAKE_COMPLEX_DEFINED
__host__ __device__ inline Complex make_complex(double re, double im) {
    return hipDoubleComplex{re, im};
}
#endif

// ============================================================================
// MPOBase Interface (must match pdmrg_gpu.cpp definition)
// ============================================================================
class MPOBase {
public:
    virtual ~MPOBase() {}
    virtual Complex* get_mpo(int site) = 0;
    virtual int get_left_dim(int site) = 0;
    virtual int get_right_dim(int site) = 0;
    virtual int get_phys_dim() const = 0;
    virtual int get_length() const = 0;
};

// ============================================================================
// LoadedMPO - Implements MPOBase for loaded tensors
// ============================================================================
class LoadedMPO : public MPOBase {
private:
    int L, d;
    std::vector<Complex*> d_mpo;
    std::vector<int> left_dims, right_dims;

public:
    // Constructor from loaded MPOTensor vector
    LoadedMPO(const std::vector<MPOTensor>& mpo_host) : L(mpo_host.size()) {
        if (L == 0) {
            std::cerr << "Error: Cannot create LoadedMPO from empty MPO vector\n";
            exit(1);
        }

        d = mpo_host[0].d_bra;  // Assume d_bra == d_ket for physical dimension

        d_mpo.resize(L);
        left_dims.resize(L);
        right_dims.resize(L);

        std::cout << "Loading MPO to GPU device memory...\n";
        std::cout << "  Length: " << L << "\n";
        std::cout << "  Physical dim: " << d << "\n";

        for (int site = 0; site < L; site++) {
            const auto& tensor = mpo_host[site];
            left_dims[site] = tensor.D_mpo_left;
            right_dims[site] = tensor.D_mpo_right;

            // Verify physical dimensions are consistent
            if (tensor.d_bra != d || tensor.d_ket != d) {
                std::cerr << "Error: MPO physical dimension mismatch at site " << site << "\n";
                std::cerr << "  Expected: d_bra = d_ket = " << d << "\n";
                std::cerr << "  Got: d_bra = " << tensor.d_bra << ", d_ket = " << tensor.d_ket << "\n";
                exit(1);
            }

            // Allocate GPU memory and copy
            int size = tensor.D_mpo_left * tensor.d_bra * tensor.d_ket * tensor.D_mpo_right;
            HIP_CHECK(hipMalloc(&d_mpo[site], size * sizeof(Complex)));

            // Convert std::complex<double> to hipDoubleComplex
            std::vector<Complex> h_mpo(size);
            for (int i = 0; i < size; i++) {
                h_mpo[i] = make_complex(
                    tensor.data[i].real(),
                    tensor.data[i].imag()
                );
            }
            HIP_CHECK(hipMemcpy(d_mpo[site], h_mpo.data(),
                               size * sizeof(Complex),
                               hipMemcpyHostToDevice));
        }

        std::cout << "✓ MPO loaded to GPU\n";
        std::cout << "  Left dims:  [" << left_dims[0];
        for (int i = 1; i < std::min(5, L); i++) std::cout << ", " << left_dims[i];
        if (L > 5) std::cout << ", ...";
        std::cout << "]\n";

        std::cout << "  Right dims: [" << right_dims[0];
        for (int i = 1; i < std::min(5, L); i++) std::cout << ", " << right_dims[i];
        if (L > 5) std::cout << ", ...";
        std::cout << "]\n\n";
    }

    ~LoadedMPO() override {
        for (auto& p : d_mpo) {
            if (p) HIP_CHECK(hipFree(p));
        }
    }

    // MPOBase interface implementation
    Complex* get_mpo(int site) override {
        return d_mpo[site];
    }

    int get_left_dim(int site) override {
        return left_dims[site];
    }

    int get_right_dim(int site) override {
        return right_dims[site];
    }

    int get_phys_dim() const override {
        return d;
    }

    int get_length() const override {
        return L;
    }
};
