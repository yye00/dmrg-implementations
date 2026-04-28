// Canonical RAII guard for rocBLAS pointer-mode toggles. Promote the
// pattern previously living in dmrg-gpu (DmrgPointerModeGuard) and
// accurate_svd_gpu.h (AsvdPointerModeGuard) so every variant uses the
// same guard. Without this, paired set_pointer_mode(device)/...(host)
// inline calls leak the device mode into the caller's handle if any
// rocBLAS call between them throws (ROCBLAS_CHECK).
#ifndef PMRG_POINTER_MODE_GUARD_H
#define PMRG_POINTER_MODE_GUARD_H

#include <rocblas/rocblas.h>

struct PointerModeGuard {
    rocblas_handle handle;
    rocblas_pointer_mode prev_mode;
    PointerModeGuard(rocblas_handle h, rocblas_pointer_mode new_mode) : handle(h) {
        rocblas_get_pointer_mode(h, &prev_mode);
        rocblas_set_pointer_mode(h, new_mode);
    }
    ~PointerModeGuard() {
        // Best-effort restore. Swallow errors — throwing from a destructor
        // during stack unwinding would call std::terminate.
        rocblas_set_pointer_mode(handle, prev_mode);
    }
    PointerModeGuard(const PointerModeGuard&) = delete;
    PointerModeGuard& operator=(const PointerModeGuard&) = delete;
};

#endif // PMRG_POINTER_MODE_GUARD_H
