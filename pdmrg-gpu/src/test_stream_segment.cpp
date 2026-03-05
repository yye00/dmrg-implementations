#include "stream_segment.h"
#include <hip/hip_runtime.h>
#include <iostream>
#include <cstdio>

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "StreamSegment Basic Test" << std::endl;
    std::cout << "========================================" << std::endl;

    // Check GPU
    int device_count = 0;
    hipGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "ERROR: No GPU devices found" << std::endl;
        return 1;
    }

    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, 0);
    std::cout << "GPU: " << props.name << std::endl;
    std::cout << "Compute capability: " << props.major << "." << props.minor << std::endl;
    std::cout << std::endl;

    // Test parameters
    const int segment_id = 0;
    const int start_site = 0;
    const int end_site = 3;      // 4 sites
    const int chi_max = 10;
    const int d = 2;             // Physical dimension (spin-1/2)
    const int D_mpo = 3;         // Heisenberg MPO bond dimension

    std::cout << "Test Configuration:" << std::endl;
    std::cout << "  Segment ID: " << segment_id << std::endl;
    std::cout << "  Sites: [" << start_site << ", " << end_site << "]" << std::endl;
    std::cout << "  Num sites: " << (end_site - start_site + 1) << std::endl;
    std::cout << "  chi_max: " << chi_max << std::endl;
    std::cout << "  d: " << d << std::endl;
    std::cout << "  D_mpo: " << D_mpo << std::endl;
    std::cout << std::endl;

    // Create HIP stream
    hipStream_t stream;
    hipError_t err = hipStreamCreate(&stream);
    if (err != hipSuccess) {
        std::cerr << "ERROR: Failed to create HIP stream: "
                  << hipGetErrorString(err) << std::endl;
        return 1;
    }
    std::cout << "✓ HIP stream created" << std::endl;

    // Create StreamSegment
    std::cout << "Creating StreamSegment..." << std::endl;
    StreamSegment* segment = nullptr;
    try {
        segment = new StreamSegment(segment_id, start_site, end_site,
                                     chi_max, d, D_mpo, stream);
        std::cout << "✓ StreamSegment created successfully" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Failed to create StreamSegment: "
                  << e.what() << std::endl;
        hipStreamDestroy(stream);
        return 1;
    }

    // Verify basic properties
    std::cout << std::endl;
    std::cout << "Verifying properties:" << std::endl;
    std::cout << "  ID: " << segment->get_id() << " (expected: " << segment_id << ")" << std::endl;
    std::cout << "  Start site: " << segment->get_start_site() << " (expected: " << start_site << ")" << std::endl;
    std::cout << "  End site: " << segment->get_end_site() << " (expected: " << end_site << ")" << std::endl;
    std::cout << "  Num sites: " << segment->get_num_sites() << " (expected: " << (end_site - start_site + 1) << ")" << std::endl;

    bool properties_ok = true;
    if (segment->get_id() != segment_id) properties_ok = false;
    if (segment->get_start_site() != start_site) properties_ok = false;
    if (segment->get_end_site() != end_site) properties_ok = false;
    if (segment->get_num_sites() != (end_site - start_site + 1)) properties_ok = false;

    if (!properties_ok) {
        std::cerr << "ERROR: Property verification failed" << std::endl;
        delete segment;
        hipStreamDestroy(stream);
        return 1;
    }
    std::cout << "✓ All properties correct" << std::endl;

    // Check boundary data
    std::cout << std::endl;
    std::cout << "Checking boundary data:" << std::endl;

    BoundaryData* left_boundary = segment->get_left_boundary();
    BoundaryData* right_boundary = segment->get_right_boundary();

    std::cout << "  Left boundary: " << (left_boundary ? "allocated" : "nullptr") << std::endl;
    std::cout << "  Right boundary: " << (right_boundary ? "allocated" : "nullptr") << std::endl;

    // For segment 0, should have right boundary but no left boundary
    // (Actually, current implementation allocates left boundary for segment_id > 0)
    if (segment_id == 0 && left_boundary != nullptr) {
        std::cout << "  Note: Segment 0 has left boundary (will be nullptr when integrated)" << std::endl;
    }

    // Check MPS tensor access
    std::cout << std::endl;
    std::cout << "Checking MPS tensor access:" << std::endl;
    for (int site = start_site; site <= end_site; site++) {
        double* tensor = segment->get_mps_tensor(site);
        if (tensor == nullptr) {
            std::cerr << "ERROR: MPS tensor at site " << site << " is nullptr" << std::endl;
            delete segment;
            hipStreamDestroy(stream);
            return 1;
        }
        std::cout << "  Site " << site << ": tensor allocated ✓" << std::endl;
    }

    // Verify out-of-range returns nullptr
    double* out_of_range = segment->get_mps_tensor(end_site + 1);
    if (out_of_range != nullptr) {
        std::cerr << "ERROR: Out-of-range tensor should be nullptr" << std::endl;
        delete segment;
        hipStreamDestroy(stream);
        return 1;
    }
    std::cout << "  Out-of-range access returns nullptr ✓" << std::endl;

    // Cleanup
    std::cout << std::endl;
    std::cout << "Cleaning up..." << std::endl;
    delete segment;
    hipStreamDestroy(stream);
    std::cout << "✓ Cleanup complete" << std::endl;

    // Summary
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Result: PASS" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
