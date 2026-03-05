#pragma once

#include <vector>
#include <complex>

// Type aliases
using Complex = std::complex<double>;

// HIP-specific types and conversions (only when using HIP)
#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>

using hipComplex = hipDoubleComplex;

// Convert std::complex to hipDoubleComplex
inline hipDoubleComplex to_hip_complex(const Complex& c) {
    return make_hipDoubleComplex(c.real(), c.imag());
}

// Convert hipDoubleComplex to std::complex
inline Complex from_hip_complex(const hipDoubleComplex& c) {
    return Complex(c.x, c.y);
}
#endif  // __HIP_PLATFORM_AMD__

// Tensor types (CPU)
template<typename T>
using Tensor5D = std::vector<std::vector<std::vector<std::vector<std::vector<T>>>>>;

template<typename T>
using Tensor4D = std::vector<std::vector<std::vector<std::vector<T>>>>;

template<typename T>
using Tensor3D = std::vector<std::vector<std::vector<T>>>;

template<typename T>
using Tensor2D = std::vector<std::vector<T>>;
