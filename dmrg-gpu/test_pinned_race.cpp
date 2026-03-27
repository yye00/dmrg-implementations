/**
 * Reproducer: race condition with pinned host memory in batched GEMM loop.
 *
 * When pointer arrays are allocated with hipHostMalloc (pinned), hipMemcpyAsync
 * is truly async. If the CPU overwrites the pinned arrays before the DMA engine
 * reads them, the GPU gets stale/mixed pointer values.
 *
 * Build: hipcc -o test_pinned_race test_pinned_race.cpp -lrocblas -I/opt/rocm/include
 */

#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define HIP_CHECK(x) do { hipError_t e = (x); if(e) { printf("HIP error %d at %s:%d\n", e, __FILE__, __LINE__); exit(1); } } while(0)
#define ROCBLAS_CHECK(x) do { rocblas_status s = (x); if(s) { printf("rocBLAS error %d at %s:%d\n", s, __FILE__, __LINE__); exit(1); } } while(0)

using Z = hipDoubleComplex;
using RZ = rocblas_double_complex;

double max_err(const Z* a, const Z* b, int n) {
    double mx = 0;
    for (int i = 0; i < n; i++) {
        mx = fmax(mx, fmax(fabs(hipCreal(a[i])-hipCreal(b[i])), fabs(hipCimag(a[i])-hipCimag(b[i]))));
    }
    return mx;
}

int main() {
    const int D = 4, d = 3, cL = 3, cR = 3;

    printf("=== Pinned memory race condition reproducer ===\n");
    printf("D=%d, d=%d, cL=%d, cR=%d\n\n", D, d, cL, cR);

    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    ROCBLAS_CHECK(rocblas_set_stream(handle, stream));

    srand(42);
    int U_size = D * d * cL * cR;
    int R_size = cR * D * cR;
    int buf_size = cL * d * cR;

    std::vector<Z> h_U(U_size), h_R(R_size);
    for (auto& v : h_U) v = make_hipDoubleComplex((double)rand()/RAND_MAX-0.5, (double)rand()/RAND_MAX-0.5);
    for (auto& v : h_R) v = make_hipDoubleComplex((double)rand()/RAND_MAX-0.5, (double)rand()/RAND_MAX-0.5);

    Z *d_U, *d_R, *d_result_seq, *d_result_pinned, *d_result_unpinned;
    HIP_CHECK(hipMalloc(&d_U, U_size * sizeof(Z)));
    HIP_CHECK(hipMalloc(&d_R, R_size * sizeof(Z)));
    HIP_CHECK(hipMalloc(&d_result_seq, buf_size * sizeof(Z)));
    HIP_CHECK(hipMalloc(&d_result_pinned, buf_size * sizeof(Z)));
    HIP_CHECK(hipMalloc(&d_result_unpinned, buf_size * sizeof(Z)));

    HIP_CHECK(hipMemcpy(d_U, h_U.data(), U_size * sizeof(Z), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_R, h_R.data(), R_size * sizeof(Z), hipMemcpyHostToDevice));

    RZ one = {1,0}, zero = {0,0};

    Z** d_A_ptrs; Z** d_B_ptrs; Z** d_C_ptrs;
    HIP_CHECK(hipMalloc(&d_A_ptrs, d * sizeof(Z*)));
    HIP_CHECK(hipMalloc(&d_B_ptrs, d * sizeof(Z*)));
    HIP_CHECK(hipMalloc(&d_C_ptrs, d * sizeof(Z*)));

    // === 1. Reference: sequential individual zgemm ===
    HIP_CHECK(hipMemset(d_result_seq, 0, buf_size * sizeof(Z)));
    for (int sp = 0; sp < d; sp++) {
        for (int wp = 0; wp < D; wp++) {
            RZ beta = (wp == 0) ? zero : one;
            int ws_out = wp * d + sp;
            ROCBLAS_CHECK(rocblas_zgemm(handle,
                rocblas_operation_none, rocblas_operation_none,
                cL, cR, cR, &one,
                (const RZ*)(d_U + ws_out*cL*cR), cL,
                (const RZ*)(d_R + wp*cR), cR*D,
                &beta, (RZ*)(d_result_seq + sp*cL), cL*d));
        }
    }
    HIP_CHECK(hipDeviceSynchronize());

    // === 2. Batched with PINNED host arrays (race condition expected) ===
    Z** h_pin_A; Z** h_pin_B; Z** h_pin_C;
    HIP_CHECK(hipHostMalloc(&h_pin_A, d * sizeof(Z*)));
    HIP_CHECK(hipHostMalloc(&h_pin_B, d * sizeof(Z*)));
    HIP_CHECK(hipHostMalloc(&h_pin_C, d * sizeof(Z*)));

    HIP_CHECK(hipMemset(d_result_pinned, 0, buf_size * sizeof(Z)));
    for (int wp = 0; wp < D; wp++) {
        RZ beta = (wp == 0) ? zero : one;
        for (int sp = 0; sp < d; sp++) {
            int ws_out = wp * d + sp;
            h_pin_A[sp] = d_U + ws_out*cL*cR;
            h_pin_B[sp] = d_R + wp*cR;
            h_pin_C[sp] = d_result_pinned + sp*cL;
        }
        HIP_CHECK(hipMemcpyAsync(d_A_ptrs, h_pin_A, d*sizeof(Z*), hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(d_B_ptrs, h_pin_B, d*sizeof(Z*), hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(d_C_ptrs, h_pin_C, d*sizeof(Z*), hipMemcpyHostToDevice, stream));
        ROCBLAS_CHECK(rocblas_zgemm_batched(handle,
            rocblas_operation_none, rocblas_operation_none,
            cL, cR, cR, &one,
            (const RZ**)d_A_ptrs, cL,
            (const RZ**)d_B_ptrs, cR*D,
            &beta, (RZ**)d_C_ptrs, cL*d, d));
    }
    HIP_CHECK(hipDeviceSynchronize());

    // === 3. Batched with NON-PINNED host arrays (sync copy, no race) ===
    std::vector<Z*> h_A(d), h_B(d), h_C(d);

    HIP_CHECK(hipMemset(d_result_unpinned, 0, buf_size * sizeof(Z)));
    for (int wp = 0; wp < D; wp++) {
        RZ beta = (wp == 0) ? zero : one;
        for (int sp = 0; sp < d; sp++) {
            int ws_out = wp * d + sp;
            h_A[sp] = d_U + ws_out*cL*cR;
            h_B[sp] = d_R + wp*cR;
            h_C[sp] = d_result_unpinned + sp*cL;
        }
        HIP_CHECK(hipMemcpyAsync(d_A_ptrs, h_A.data(), d*sizeof(Z*), hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(d_B_ptrs, h_B.data(), d*sizeof(Z*), hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(d_C_ptrs, h_C.data(), d*sizeof(Z*), hipMemcpyHostToDevice, stream));
        ROCBLAS_CHECK(rocblas_zgemm_batched(handle,
            rocblas_operation_none, rocblas_operation_none,
            cL, cR, cR, &one,
            (const RZ**)d_A_ptrs, cL,
            (const RZ**)d_B_ptrs, cR*D,
            &beta, (RZ**)d_C_ptrs, cL*d, d));
    }
    HIP_CHECK(hipDeviceSynchronize());

    // === 4. Batched with PINNED + hipStreamSynchronize between iterations ===
    Z** d_A_ptrs2; Z** d_B_ptrs2; Z** d_C_ptrs2;
    HIP_CHECK(hipMalloc(&d_A_ptrs2, d * sizeof(Z*)));
    HIP_CHECK(hipMalloc(&d_B_ptrs2, d * sizeof(Z*)));
    HIP_CHECK(hipMalloc(&d_C_ptrs2, d * sizeof(Z*)));

    Z* d_result_sync;
    HIP_CHECK(hipMalloc(&d_result_sync, buf_size * sizeof(Z)));
    HIP_CHECK(hipMemset(d_result_sync, 0, buf_size * sizeof(Z)));
    for (int wp = 0; wp < D; wp++) {
        RZ beta = (wp == 0) ? zero : one;
        for (int sp = 0; sp < d; sp++) {
            int ws_out = wp * d + sp;
            h_pin_A[sp] = d_U + ws_out*cL*cR;
            h_pin_B[sp] = d_R + wp*cR;
            h_pin_C[sp] = d_result_sync + sp*cL;
        }
        HIP_CHECK(hipMemcpyAsync(d_A_ptrs2, h_pin_A, d*sizeof(Z*), hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(d_B_ptrs2, h_pin_B, d*sizeof(Z*), hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(d_C_ptrs2, h_pin_C, d*sizeof(Z*), hipMemcpyHostToDevice, stream));
        ROCBLAS_CHECK(rocblas_zgemm_batched(handle,
            rocblas_operation_none, rocblas_operation_none,
            cL, cR, cR, &one,
            (const RZ**)d_A_ptrs2, cL,
            (const RZ**)d_B_ptrs2, cR*D,
            &beta, (RZ**)d_C_ptrs2, cL*d, d));
        HIP_CHECK(hipStreamSynchronize(stream));  // Force completion before overwriting pinned
    }
    HIP_CHECK(hipDeviceSynchronize());

    // Compare results
    std::vector<Z> ref(buf_size), res_pin(buf_size), res_unpin(buf_size), res_sync(buf_size);
    HIP_CHECK(hipMemcpy(ref.data(), d_result_seq, buf_size*sizeof(Z), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(res_pin.data(), d_result_pinned, buf_size*sizeof(Z), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(res_unpin.data(), d_result_unpinned, buf_size*sizeof(Z), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(res_sync.data(), d_result_sync, buf_size*sizeof(Z), hipMemcpyDeviceToHost));

    double e_pin = max_err(ref.data(), res_pin.data(), buf_size);
    double e_unpin = max_err(ref.data(), res_unpin.data(), buf_size);
    double e_sync = max_err(ref.data(), res_sync.data(), buf_size);

    printf("Test 1 - Batched + PINNED host (no sync):     error = %.3e  %s\n", e_pin, e_pin < 1e-12 ? "PASS" : "FAIL (race!)");
    printf("Test 2 - Batched + NON-PINNED host:           error = %.3e  %s\n", e_unpin, e_unpin < 1e-12 ? "PASS" : "FAIL");
    printf("Test 3 - Batched + PINNED + streamSync:       error = %.3e  %s\n", e_sync, e_sync < 1e-12 ? "PASS" : "FAIL");

    if (e_pin > 1e-12 && e_sync < 1e-12) {
        printf("\n*** CONFIRMED: Race condition on pinned host memory. ***\n");
        printf("*** hipMemcpyAsync from pinned memory is truly async; CPU overwrites ***\n");
        printf("*** the source buffer before the DMA engine finishes reading it.     ***\n");
    } else if (e_pin < 1e-12) {
        printf("\nNo race detected at this problem size (timing-dependent).\n");
        printf("Try larger sizes or repeated runs.\n");
    }

    // Cleanup
    HIP_CHECK(hipHostFree(h_pin_A)); HIP_CHECK(hipHostFree(h_pin_B)); HIP_CHECK(hipHostFree(h_pin_C));
    HIP_CHECK(hipFree(d_U)); HIP_CHECK(hipFree(d_R));
    HIP_CHECK(hipFree(d_result_seq)); HIP_CHECK(hipFree(d_result_pinned)); HIP_CHECK(hipFree(d_result_unpinned));
    HIP_CHECK(hipFree(d_result_sync));
    HIP_CHECK(hipFree(d_A_ptrs)); HIP_CHECK(hipFree(d_B_ptrs)); HIP_CHECK(hipFree(d_C_ptrs));
    HIP_CHECK(hipFree(d_A_ptrs2)); HIP_CHECK(hipFree(d_B_ptrs2)); HIP_CHECK(hipFree(d_C_ptrs2));
    ROCBLAS_CHECK(rocblas_destroy_handle(handle));
    HIP_CHECK(hipStreamDestroy(stream));

    return 0;
}
