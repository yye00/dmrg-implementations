/**
 * Test hipGraph for multi-kernel sequences resembling apply_heff.
 * Measures kernel launch overhead: direct calls vs hipGraph replay.
 *
 * apply_heff pattern (D=5 MPO):
 *   1 batched GEMM (D batches) + 1 dense GEMM + 1 memset + D dense GEMMs
 *   = ~8 kernel launches total (batched=1 launch, dense=1 each, memset=1)
 *
 * Also tests: strided batched GEMM to fuse Step 3's D launches → 1 launch.
 */
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define HIP_CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("HIP error %d (%s) at %s:%d\n", e, hipGetErrorString(e), __FILE__, __LINE__); return 1; } } while(0)
#define ROCBLAS_CHECK(x) do { rocblas_status s = (x); if (s != rocblas_status_success) { printf("rocBLAS error %d at %s:%d\n", s, __FILE__, __LINE__); return 1; } } while(0)

int main(int argc, char** argv) {
    int chi = 128;
    int D = 5;
    if (argc > 1) chi = atoi(argv[1]);
    if (argc > 2) D = atoi(argv[2]);

    int d2 = 4;  // d*d for two-site
    printf("Testing hipGraph for apply_heff pattern: chi=%d, D=%d, d²=%d\n\n", chi, D, d2);

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));
    ROCBLAS_CHECK(rocblas_set_stream(handle, stream));

    // Allocate buffers sized for the operations
    // Step 1: D GEMMs of (chi x chi) @ (chi x d2*chi) -> (chi x d2*chi) each
    // Step 2: (d2 x D*d2) @ (D*d2 x chi*chi) -> (d2 x chi*chi)  [dense GEMM]
    // Step 3: D GEMMs of (chi*d2 x chi) @ (chi x chi) -> (chi*d2 x chi) each

    double *d_A1, *d_B1, *d_C1;  // Step 1 buffers
    double *d_A2, *d_B2, *d_C2;  // Step 2 buffers
    double *d_A3, *d_B3, *d_C3;  // Step 3 buffers

    size_t s1_A = D * chi * chi;
    size_t s1_B = chi * d2 * chi;
    size_t s1_C = D * chi * d2 * chi;
    size_t s2_A = d2 * D * d2;       // WW
    size_t s2_B = D * d2 * chi * chi; // reshaped temp1
    size_t s2_C = d2 * chi * chi;     // doesn't need D since we overwrite
    size_t s3_A = D * chi * d2 * chi;
    size_t s3_B = D * chi * chi;
    size_t s3_C = chi * d2 * chi;     // output vector

    HIP_CHECK(hipMalloc(&d_A1, s1_A * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_B1, s1_B * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_C1, s1_C * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_A2, s2_A * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_B2, s2_B * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_C2, s2_C * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_A3, s3_A * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_B3, s3_B * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_C3, s3_C * sizeof(double)));

    // Initialize with random data
    auto init_buf = [&](double* d_buf, size_t n) {
        std::vector<double> h(n);
        for (auto& v : h) v = (double)rand() / RAND_MAX;
        HIP_CHECK(hipMemcpy(d_buf, h.data(), n * sizeof(double), hipMemcpyHostToDevice));
    };
    init_buf(d_A1, s1_A); init_buf(d_B1, s1_B);
    init_buf(d_A2, s2_A); init_buf(d_B2, s2_B);
    init_buf(d_A3, s3_A); init_buf(d_B3, s3_B);

    double one = 1.0, zero = 0.0;

    // Set up batched GEMM pointer arrays for Step 1
    std::vector<double*> h_A1ptrs(D), h_B1ptrs(D), h_C1ptrs(D);
    for (int w = 0; w < D; w++) {
        h_A1ptrs[w] = d_A1 + w * chi * chi;
        h_B1ptrs[w] = d_B1;  // same input vector for all
        h_C1ptrs[w] = d_C1 + w * chi * d2 * chi;
    }
    double **d_A1ptrs, **d_B1ptrs, **d_C1ptrs;
    HIP_CHECK(hipMalloc(&d_A1ptrs, D * sizeof(double*)));
    HIP_CHECK(hipMalloc(&d_B1ptrs, D * sizeof(double*)));
    HIP_CHECK(hipMalloc(&d_C1ptrs, D * sizeof(double*)));
    HIP_CHECK(hipMemcpy(d_A1ptrs, h_A1ptrs.data(), D * sizeof(double*), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B1ptrs, h_B1ptrs.data(), D * sizeof(double*), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_C1ptrs, h_C1ptrs.data(), D * sizeof(double*), hipMemcpyHostToDevice));

    HIP_CHECK(hipStreamSynchronize(stream));

    // =========================================================================
    // Direct execution: 1 batched + 1 dense + 1 memset + D dense = 3+D launches
    // =========================================================================
    auto run_direct = [&]() {
        // Step 1: batched GEMM
        ROCBLAS_CHECK(rocblas_dgemm_batched(handle,
            rocblas_operation_none, rocblas_operation_none,
            chi, d2 * chi, chi, &one,
            (const double**)d_A1ptrs, chi,
            (const double**)d_B1ptrs, chi,
            &zero, d_C1ptrs, chi, D));

        // Step 2: dense GEMM
        ROCBLAS_CHECK(rocblas_dgemm(handle,
            rocblas_operation_none, rocblas_operation_none,
            d2, chi * chi, D * d2,
            &one, d_A2, d2,
            d_B2, D * d2,
            &zero, d_C2, d2));

        // Step 3: memset + D dense GEMMs with accumulation
        HIP_CHECK(hipMemsetAsync(d_C3, 0, s3_C * sizeof(double), stream));
        for (int w = 0; w < D; w++) {
            ROCBLAS_CHECK(rocblas_dgemm(handle,
                rocblas_operation_none, rocblas_operation_none,
                chi * d2, chi, chi,
                &one, d_A3 + w * chi * d2 * chi, chi * d2,
                d_B3 + w * chi * chi, chi,
                &one, d_C3, chi * d2));
        }
    };

    // Warmup
    for (int i = 0; i < 10; i++) run_direct();
    HIP_CHECK(hipStreamSynchronize(stream));

    // Save reference
    std::vector<double> h_ref(s3_C);
    HIP_CHECK(hipMemcpy(h_ref.data(), d_C3, s3_C * sizeof(double), hipMemcpyDeviceToHost));

    // Benchmark direct
    hipEvent_t ev0, ev1;
    HIP_CHECK(hipEventCreate(&ev0));
    HIP_CHECK(hipEventCreate(&ev1));

    int n_reps = 500;
    HIP_CHECK(hipEventRecord(ev0, stream));
    for (int i = 0; i < n_reps; i++) run_direct();
    HIP_CHECK(hipEventRecord(ev1, stream));
    HIP_CHECK(hipEventSynchronize(ev1));
    float direct_ms;
    HIP_CHECK(hipEventElapsedTime(&direct_ms, ev0, ev1));
    double direct_us = (double)direct_ms * 1000.0 / n_reps;
    printf("Direct: %.1f us/call  (%d kernel launches: 1 batched + 1 dense + 1 memset + %d dense)\n",
           direct_us, 3 + D, D);

    // =========================================================================
    // hipGraph capture of the direct pattern
    // =========================================================================
    printf("\nCapturing direct pattern in hipGraph...\n");
    hipGraph_t graph;
    hipGraphExec_t graphExec;

    HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    run_direct();
    HIP_CHECK(hipStreamEndCapture(stream, &graph));
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    printf("  Capture + instantiate succeeded!\n");

    // Verify
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    std::vector<double> h_graph(s3_C);
    HIP_CHECK(hipMemcpy(h_graph.data(), d_C3, s3_C * sizeof(double), hipMemcpyDeviceToHost));
    double max_err = 0;
    for (size_t i = 0; i < s3_C; i++) {
        double err = fabs(h_graph[i] - h_ref[i]);
        if (err > max_err) max_err = err;
    }
    printf("  Max error: %.4e\n", max_err);

    // Benchmark graph
    for (int i = 0; i < 10; i++) HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    HIP_CHECK(hipEventRecord(ev0, stream));
    for (int i = 0; i < n_reps; i++) HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipEventRecord(ev1, stream));
    HIP_CHECK(hipEventSynchronize(ev1));
    float graph_ms;
    HIP_CHECK(hipEventElapsedTime(&graph_ms, ev0, ev1));
    double graph_us = (double)graph_ms * 1000.0 / n_reps;
    printf("  Graph: %.1f us/call\n", graph_us);
    printf("  Speedup vs direct: %.2fx\n", direct_us / graph_us);

    // =========================================================================
    // Step 3 fusion: strided batched GEMM (memset + 1 strided batched = 2 launches)
    // =========================================================================
    printf("\n--- Step 3 Fusion: strided batched GEMM ---\n");

    auto run_fused_step3 = [&]() {
        // Steps 1 & 2 same
        ROCBLAS_CHECK(rocblas_dgemm_batched(handle,
            rocblas_operation_none, rocblas_operation_none,
            chi, d2 * chi, chi, &one,
            (const double**)d_A1ptrs, chi,
            (const double**)d_B1ptrs, chi,
            &zero, d_C1ptrs, chi, D));

        ROCBLAS_CHECK(rocblas_dgemm(handle,
            rocblas_operation_none, rocblas_operation_none,
            d2, chi * chi, D * d2,
            &one, d_A2, d2,
            d_B2, D * d2,
            &zero, d_C2, d2));

        // Step 3: memset + strided batched with beta=1
        HIP_CHECK(hipMemsetAsync(d_C3, 0, s3_C * sizeof(double), stream));
        ROCBLAS_CHECK(rocblas_dgemm_strided_batched(handle,
            rocblas_operation_none, rocblas_operation_none,
            chi * d2, chi, chi,
            &one,
            d_A3, chi * d2, (rocblas_stride)(chi * d2 * chi),
            d_B3, chi, (rocblas_stride)(chi * chi),
            &one,
            d_C3, chi * d2, (rocblas_stride)0,  // stride_C=0: accumulate to same output
            D));
    };

    // Warmup
    for (int i = 0; i < 10; i++) run_fused_step3();
    HIP_CHECK(hipStreamSynchronize(stream));

    // Verify
    std::vector<double> h_fused(s3_C);
    HIP_CHECK(hipMemcpy(h_fused.data(), d_C3, s3_C * sizeof(double), hipMemcpyDeviceToHost));
    double max_err_fused = 0;
    for (size_t i = 0; i < s3_C; i++) {
        double err = fabs(h_fused[i] - h_ref[i]);
        if (err > max_err_fused) max_err_fused = err;
    }
    printf("  Max error vs direct: %.4e\n", max_err_fused);

    bool fused_ok = (max_err_fused < 1e-10);
    if (fused_ok) {
        printf("  PASS\n");

        // Benchmark fused
        HIP_CHECK(hipEventRecord(ev0, stream));
        for (int i = 0; i < n_reps; i++) run_fused_step3();
        HIP_CHECK(hipEventRecord(ev1, stream));
        HIP_CHECK(hipEventSynchronize(ev1));
        float fused_ms;
        HIP_CHECK(hipEventElapsedTime(&fused_ms, ev0, ev1));
        double fused_us = (double)fused_ms * 1000.0 / n_reps;
        printf("  Fused Step3: %.1f us/call (4 launches: 1 batched + 1 dense + 1 memset + 1 strided)\n", fused_us);
        printf("  Speedup vs direct: %.2fx\n", direct_us / fused_us);

        // Graph capture the fused version
        printf("\n  Capturing fused version in hipGraph...\n");
        hipGraph_t graph2;
        hipGraphExec_t graphExec2;
        HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
        run_fused_step3();
        HIP_CHECK(hipStreamEndCapture(stream, &graph2));
        HIP_CHECK(hipGraphInstantiate(&graphExec2, graph2, nullptr, nullptr, 0));

        for (int i = 0; i < 10; i++) HIP_CHECK(hipGraphLaunch(graphExec2, stream));
        HIP_CHECK(hipStreamSynchronize(stream));

        HIP_CHECK(hipEventRecord(ev0, stream));
        for (int i = 0; i < n_reps; i++) HIP_CHECK(hipGraphLaunch(graphExec2, stream));
        HIP_CHECK(hipEventRecord(ev1, stream));
        HIP_CHECK(hipEventSynchronize(ev1));
        float gf_ms;
        HIP_CHECK(hipEventElapsedTime(&gf_ms, ev0, ev1));
        double gf_us = (double)gf_ms * 1000.0 / n_reps;
        printf("  Graph+Fused: %.1f us/call\n", gf_us);
        printf("  Speedup vs direct: %.2fx\n", direct_us / gf_us);

        hipGraphExecDestroy(graphExec2);
        hipGraphDestroy(graph2);
    } else {
        printf("  FAIL - stride_C=0 accumulation doesn't work (error: %.4e)\n", max_err_fused);
        printf("  Falling back: Step 3 fusion needs explicit accumulation kernel\n");
    }

    // =========================================================================
    // Summary
    // =========================================================================
    printf("\n=== Summary (chi=%d, D=%d) ===\n", chi, D);
    printf("Direct (%d launches): %.1f us\n", 3 + D, direct_us);
    printf("hipGraph (%d launches): %.1f us (%.2fx)\n", 3 + D, graph_us, direct_us / graph_us);

    // Cleanup
    hipGraphExecDestroy(graphExec);
    hipGraphDestroy(graph);
    hipEventDestroy(ev0);
    hipEventDestroy(ev1);
    hipFree(d_A1); hipFree(d_B1); hipFree(d_C1);
    hipFree(d_A2); hipFree(d_B2); hipFree(d_C2);
    hipFree(d_A3); hipFree(d_B3); hipFree(d_C3);
    hipFree(d_A1ptrs); hipFree(d_B1ptrs); hipFree(d_C1ptrs);
    rocblas_destroy_handle(handle);
    hipStreamDestroy(stream);

    printf("\nDone.\n");
    return 0;
}
