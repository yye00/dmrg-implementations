/**
 * Test if rocBLAS calls can be captured in a hipGraph via stream capture.
 * This determines whether hipGraph is viable for the Lanczos loop.
 */
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#define HIP_CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("HIP error %d at %s:%d\n", e, __FILE__, __LINE__); return 1; } } while(0)
#define ROCBLAS_CHECK(x) do { rocblas_status s = (x); if (s != rocblas_status_success) { printf("rocBLAS error %d at %s:%d\n", s, __FILE__, __LINE__); return 1; } } while(0)

int main() {
    int N = 256;

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));
    ROCBLAS_CHECK(rocblas_set_stream(handle, stream));

    // Allocate GPU buffers
    double *d_A, *d_B, *d_C, *d_C_ref;
    HIP_CHECK(hipMalloc(&d_A, N * N * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_B, N * N * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_C, N * N * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_C_ref, N * N * sizeof(double)));

    // Initialize
    std::vector<double> h_A(N * N), h_B(N * N);
    for (int i = 0; i < N * N; i++) {
        h_A[i] = (double)rand() / RAND_MAX;
        h_B[i] = (double)rand() / RAND_MAX;
    }
    HIP_CHECK(hipMemcpy(d_A, h_A.data(), N * N * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), N * N * sizeof(double), hipMemcpyHostToDevice));

    // Reference: direct GEMM
    double one = 1.0, zero = 0.0;
    HIP_CHECK(hipMemset(d_C_ref, 0, N * N * sizeof(double)));
    ROCBLAS_CHECK(rocblas_dgemm(handle, rocblas_operation_none, rocblas_operation_none,
                                N, N, N, &one, d_A, N, d_B, N, &zero, d_C_ref, N));
    HIP_CHECK(hipStreamSynchronize(stream));

    // Attempt hipGraph stream capture
    printf("Attempting hipGraph stream capture with rocBLAS dgemm...\n");

    hipGraph_t graph;
    hipGraphExec_t graphExec;

    HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));

    // Capture a GEMM
    HIP_CHECK(hipMemsetAsync(d_C, 0, N * N * sizeof(double), stream));
    ROCBLAS_CHECK(rocblas_dgemm(handle, rocblas_operation_none, rocblas_operation_none,
                                N, N, N, &one, d_A, N, d_B, N, &zero, d_C, N));

    HIP_CHECK(hipStreamEndCapture(stream, &graph));
    printf("  Stream capture succeeded!\n");

    // Instantiate and launch the graph
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    printf("  Graph instantiation succeeded!\n");

    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    printf("  Graph launch succeeded!\n");

    // Verify result
    std::vector<double> h_C(N * N), h_C_ref(N * N);
    HIP_CHECK(hipMemcpy(h_C.data(), d_C, N * N * sizeof(double), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_C_ref.data(), d_C_ref, N * N * sizeof(double), hipMemcpyDeviceToHost));

    double max_err = 0;
    for (int i = 0; i < N * N; i++) {
        double err = fabs(h_C[i] - h_C_ref[i]);
        if (err > max_err) max_err = err;
    }
    printf("  Max error vs reference: %.4e\n", max_err);

    // Benchmark: graph launch vs direct call using hipEvents
    int n_reps = 100;
    hipEvent_t ev_start, ev_stop;
    HIP_CHECK(hipEventCreate(&ev_start));
    HIP_CHECK(hipEventCreate(&ev_stop));

    // Warmup
    for (int i = 0; i < 10; i++) {
        ROCBLAS_CHECK(rocblas_dgemm(handle, rocblas_operation_none, rocblas_operation_none,
                                    N, N, N, &one, d_A, N, d_B, N, &zero, d_C, N));
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    // Direct calls
    HIP_CHECK(hipEventRecord(ev_start, stream));
    for (int i = 0; i < n_reps; i++) {
        ROCBLAS_CHECK(rocblas_dgemm(handle, rocblas_operation_none, rocblas_operation_none,
                                    N, N, N, &one, d_A, N, d_B, N, &zero, d_C, N));
    }
    HIP_CHECK(hipEventRecord(ev_stop, stream));
    HIP_CHECK(hipEventSynchronize(ev_stop));
    float direct_ms;
    HIP_CHECK(hipEventElapsedTime(&direct_ms, ev_start, ev_stop));

    // Graph launches
    HIP_CHECK(hipEventRecord(ev_start, stream));
    for (int i = 0; i < n_reps; i++) {
        HIP_CHECK(hipGraphLaunch(graphExec, stream));
    }
    HIP_CHECK(hipEventRecord(ev_stop, stream));
    HIP_CHECK(hipEventSynchronize(ev_stop));
    float graph_ms;
    HIP_CHECK(hipEventElapsedTime(&graph_ms, ev_start, ev_stop));

    double direct_us = (double)direct_ms * 1000.0 / n_reps;
    double graph_us = (double)graph_ms * 1000.0 / n_reps;
    printf("\n  Direct GEMM: %.1f us/call\n", direct_us);
    printf("  Graph GEMM:  %.1f us/call\n", graph_us);
    printf("  Speedup:     %.2fx\n", direct_us / graph_us);

    HIP_CHECK(hipEventDestroy(ev_start));
    HIP_CHECK(hipEventDestroy(ev_stop));

    // Cleanup
    hipGraphExecDestroy(graphExec);
    hipGraphDestroy(graph);
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    hipFree(d_C_ref);
    rocblas_destroy_handle(handle);
    hipStreamDestroy(stream);

    printf("\nDone.\n");
    return 0;
}
