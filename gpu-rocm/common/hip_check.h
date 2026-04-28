// Canonical HIP / rocBLAS error-check macros. Promote what was 12+
// byte-equal copies (one per variant impl.h) to a single source so a
// future change (e.g., logging format, error category) propagates
// everywhere instead of drifting per-variant.
#ifndef PMRG_HIP_CHECK_H
#define PMRG_HIP_CHECK_H

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <iostream>
#include <stdexcept>

#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "HIP error in " << __FILE__ << ":" << __LINE__ \
                      << " - " << hipGetErrorString(err) << std::endl; \
            throw std::runtime_error("HIP error"); \
        } \
    } while(0)

#define ROCBLAS_CHECK(call) \
    do { \
        rocblas_status status = call; \
        if (status != rocblas_status_success) { \
            std::cerr << "rocBLAS error in " << __FILE__ << ":" << __LINE__ \
                      << " - status " << status << std::endl; \
            throw std::runtime_error("rocBLAS error"); \
        } \
    } while(0)

#endif // PMRG_HIP_CHECK_H
