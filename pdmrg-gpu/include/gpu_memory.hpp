#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include <vector>
#include <stdexcept>
#include <iostream>

// GPU memory management with RAII and stream support
// Priority: 1. Accuracy (complex128), 2. Performance

using Complex = hipDoubleComplex;

// Error checking macros
#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            throw std::runtime_error(std::string("HIP error at ") + __FILE__ + ":" + \
                                     std::to_string(__LINE__) + " - " + hipGetErrorString(err)); \
        } \
    } while(0)

// GPU memory buffer with automatic cleanup
template<typename T>
class GPUBuffer {
private:
    T* d_ptr;
    size_t size;
    bool owns_memory;

public:
    GPUBuffer() : d_ptr(nullptr), size(0), owns_memory(false) {}

    explicit GPUBuffer(size_t n) : size(n), owns_memory(true) {
        HIP_CHECK(hipMalloc(&d_ptr, n * sizeof(T)));
    }

    // No copy (avoid accidental duplication)
    GPUBuffer(const GPUBuffer&) = delete;
    GPUBuffer& operator=(const GPUBuffer&) = delete;

    // Move semantics
    GPUBuffer(GPUBuffer&& other) noexcept
        : d_ptr(other.d_ptr), size(other.size), owns_memory(other.owns_memory) {
        other.d_ptr = nullptr;
        other.owns_memory = false;
    }

    GPUBuffer& operator=(GPUBuffer&& other) noexcept {
        if (this != &other) {
            if (owns_memory && d_ptr) {
                hipFree(d_ptr);
            }
            d_ptr = other.d_ptr;
            size = other.size;
            owns_memory = other.owns_memory;
            other.d_ptr = nullptr;
            other.owns_memory = false;
        }
        return *this;
    }

    ~GPUBuffer() {
        if (owns_memory && d_ptr) {
            hipFree(d_ptr);
        }
    }

    // Resize (preserves data if growing)
    void resize(size_t new_size) {
        if (new_size == size) return;

        T* new_ptr = nullptr;
        HIP_CHECK(hipMalloc(&new_ptr, new_size * sizeof(T)));

        if (d_ptr && size > 0) {
            size_t copy_size = std::min(size, new_size);
            HIP_CHECK(hipMemcpy(new_ptr, d_ptr, copy_size * sizeof(T),
                               hipMemcpyDeviceToDevice));
        }

        if (owns_memory && d_ptr) {
            HIP_CHECK(hipFree(d_ptr));
        }

        d_ptr = new_ptr;
        size = new_size;
        owns_memory = true;
    }

    // Host to device transfer
    void copy_from_host(const T* h_ptr, size_t n, hipStream_t stream = 0) {
        if (n > size) {
            throw std::runtime_error("copy_from_host: buffer too small");
        }
        if (stream == 0) {
            HIP_CHECK(hipMemcpy(d_ptr, h_ptr, n * sizeof(T), hipMemcpyHostToDevice));
        } else {
            HIP_CHECK(hipMemcpyAsync(d_ptr, h_ptr, n * sizeof(T),
                                    hipMemcpyHostToDevice, stream));
        }
    }

    void copy_from_host(const std::vector<T>& h_vec, hipStream_t stream = 0) {
        copy_from_host(h_vec.data(), h_vec.size(), stream);
    }

    // Device to host transfer
    void copy_to_host(T* h_ptr, size_t n, hipStream_t stream = 0) const {
        if (n > size) {
            throw std::runtime_error("copy_to_host: buffer too small");
        }
        if (stream == 0) {
            HIP_CHECK(hipMemcpy(h_ptr, d_ptr, n * sizeof(T), hipMemcpyDeviceToHost));
        } else {
            HIP_CHECK(hipMemcpyAsync(h_ptr, d_ptr, n * sizeof(T),
                                    hipMemcpyDeviceToHost, stream));
        }
    }

    void copy_to_host(std::vector<T>& h_vec, hipStream_t stream = 0) const {
        if (h_vec.size() != size) {
            h_vec.resize(size);
        }
        copy_to_host(h_vec.data(), size, stream);
    }

    // Zero the buffer
    void zero(hipStream_t stream = 0) {
        if (stream == 0) {
            HIP_CHECK(hipMemset(d_ptr, 0, size * sizeof(T)));
        } else {
            HIP_CHECK(hipMemsetAsync(d_ptr, 0, size * sizeof(T), stream));
        }
    }

    // Accessors
    T* data() { return d_ptr; }
    const T* data() const { return d_ptr; }
    size_t get_size() const { return size; }
    bool empty() const { return size == 0 || d_ptr == nullptr; }
};

// Stream manager for overlapping computation and communication
class StreamManager {
private:
    std::vector<hipStream_t> streams;
    std::vector<hipEvent_t> events;

public:
    explicit StreamManager(int n_streams = 4) {
        streams.resize(n_streams);
        events.resize(n_streams);

        for (int i = 0; i < n_streams; i++) {
            HIP_CHECK(hipStreamCreate(&streams[i]));
            HIP_CHECK(hipEventCreate(&events[i]));
        }
    }

    ~StreamManager() {
        for (auto stream : streams) {
            hipStreamDestroy(stream);
        }
        for (auto event : events) {
            hipEventDestroy(event);
        }
    }

    // Get stream by index
    hipStream_t get_stream(int idx) {
        return streams[idx % streams.size()];
    }

    // Get event by index
    hipEvent_t get_event(int idx) {
        return events[idx % events.size()];
    }

    // Record event on stream
    void record_event(int stream_idx, int event_idx) {
        HIP_CHECK(hipEventRecord(events[event_idx], streams[stream_idx]));
    }

    // Make stream wait for event
    void wait_event(int stream_idx, int event_idx) {
        HIP_CHECK(hipStreamWaitEvent(streams[stream_idx], events[event_idx], 0));
    }

    // Synchronize specific stream
    void sync_stream(int idx) {
        HIP_CHECK(hipStreamSynchronize(streams[idx]));
    }

    // Synchronize all streams
    void sync_all() {
        for (auto stream : streams) {
            HIP_CHECK(hipStreamSynchronize(stream));
        }
    }

    int num_streams() const { return streams.size(); }
};

// Complex type conversions
inline Complex make_complex(double real, double imag = 0.0) {
    return make_hipDoubleComplex(real, imag);
}

inline Complex to_hip_complex(const std::complex<double>& z) {
    return make_hipDoubleComplex(z.real(), z.imag());
}

inline std::complex<double> from_hip_complex(const Complex& z) {
    return std::complex<double>(z.x, z.y);
}
