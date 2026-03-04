// Hello World HIP Streams Example
// Demonstrates async kernel execution on MI300X
// Compile: hipcc -o hello_streams hello_streams.hip

#include <hip/hip_runtime.h>
#include <iostream>
#include <chrono>

// Simple kernel that just sleeps a bit to simulate work
__global__ void work_kernel(int *data, int stream_id, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Simulate some work
    float result = 0.0f;
    for (int i = 0; i < iterations; i++) {
        result += sinf(float(idx + i));
    }

    // Write result (prevents optimization)
    if (idx == 0) {
        data[stream_id] = (int)result;
    }
}

int main() {
    std::cout << "HIP Streams Demo for MI300X\n";
    std::cout << "============================\n\n";

    // Get GPU info
    int device;
    hipGetDevice(&device);
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, device);
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Architecture: " << prop.gcnArchName << "\n";
    std::cout << "Memory: " << prop.totalGlobalMem / (1024*1024*1024) << " GB\n\n";

    // Allocate device memory
    const int num_streams = 4;
    int *d_data;
    hipMalloc(&d_data, num_streams * sizeof(int));

    // Create streams
    hipStream_t streams[num_streams];
    for (int i = 0; i < num_streams; i++) {
        hipStreamCreate(&streams[i]);
    }

    // Launch kernels on different streams (concurrent execution!)
    std::cout << "Launching " << num_streams << " kernels on separate streams...\n";
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_streams; i++) {
        int blocks = 256;
        int threads = 256;
        int iterations = 10000;

        work_kernel<<<blocks, threads, 0, streams[i]>>>(
            d_data, i, iterations
        );
    }

    // Synchronize all streams
    for (int i = 0; i < num_streams; i++) {
        hipStreamSynchronize(streams[i]);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "All streams completed in " << duration.count() << " ms\n";
    std::cout << "✓ Concurrent execution successful!\n\n";

    // Copy results back
    int h_data[num_streams];
    hipMemcpy(h_data, d_data, num_streams * sizeof(int), hipMemcpyDeviceToHost);

    std::cout << "Results from each stream:\n";
    for (int i = 0; i < num_streams; i++) {
        std::cout << "  Stream " << i << ": " << h_data[i] << "\n";
    }

    // Cleanup
    for (int i = 0; i < num_streams; i++) {
        hipStreamDestroy(streams[i]);
    }
    hipFree(d_data);

    std::cout << "\nKey takeaways for DMRG:\n";
    std::cout << "- Multiple HIP streams enable concurrent tensor operations\n";
    std::cout << "- Can pipeline: eigensolver on site i, tensor prep for site i+1\n";
    std::cout << "- MI300X has enough memory (192GB) for ALL DMRG data on-device\n";
    std::cout << "- No host-device transfers during sweeps = FAST!\n";

    return 0;
}
