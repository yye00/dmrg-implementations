cmake_minimum_required(VERSION 3.21)

# Set compiler BEFORE project() to avoid the hipcc wrapper issue
set(CMAKE_C_COMPILER /opt/rocm/llvm/bin/clang)
set(CMAKE_CXX_COMPILER /opt/rocm/llvm/bin/clang++)

project(PDMRG_GPU LANGUAGES CXX HIP)

# AMD MI300X Single-GPU DMRG Implementation
# Priority: 1. Accuracy (complex128), 2. Performance

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)

# ROCm setup
if(NOT DEFINED ROCM_PATH)
    set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to ROCm installation")
endif()

list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})

# Find ROCm packages
find_package(hip REQUIRED)
find_package(rocblas REQUIRED)
find_package(rocsolver REQUIRED)

# Find hipTensor
find_library(HIPTENSOR_LIBRARY
    NAMES hiptensor
    PATHS ${ROCM_PATH}/lib
    REQUIRED
)

message(STATUS "Found hipTensor: ${HIPTENSOR_LIBRARY}")

# GPU architecture
set(GPU_TARGETS "gfx942" CACHE STRING "GPU targets to compile for")
set(CMAKE_HIP_ARCHITECTURES ${GPU_TARGETS})

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${ROCM_PATH}/include
)

# Heisenberg benchmark (original)
add_executable(heisenberg_gpu src/heisenberg_dmrg_gpu.cpp)
set_source_files_properties(src/heisenberg_dmrg_gpu.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(heisenberg_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=${GPU_TARGETS}>
    -O3
)
target_link_libraries(heisenberg_gpu
    hip::device
    roc::rocblas
    roc::rocsolver
    ${HIPTENSOR_LIBRARY}
)

# DMRG Benchmark (PDMRG vs PDMRG2 comparison)
add_executable(dmrg_benchmark
    src/dmrg_benchmark.cpp
    src/heisenberg_mpo.cpp
)
set_source_files_properties(src/dmrg_benchmark.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(dmrg_benchmark PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=${GPU_TARGETS}>
    -O3
)
target_link_libraries(dmrg_benchmark
    hip::device
    roc::rocblas
    roc::rocsolver
    ${HIPTENSOR_LIBRARY}
)

# GPU-Native DMRG (minimal transfers: upload → compute → download)
add_executable(dmrg_gpu_native
    src/dmrg_gpu_native.cpp
    src/heisenberg_mpo.cpp
)
set_source_files_properties(src/dmrg_gpu_native.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(dmrg_gpu_native PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=${GPU_TARGETS}>
    -O3
)
target_link_libraries(dmrg_gpu_native
    hip::device
    roc::rocblas
    roc::rocsolver
    ${HIPTENSOR_LIBRARY}
)

# Complete DMRG with Lanczos (working version)
add_executable(heisenberg_complete
    src/heisenberg_dmrg_complete.cpp
)
set_source_files_properties(src/heisenberg_dmrg_complete.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(heisenberg_complete PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=${GPU_TARGETS}>
    -O3
)
target_link_libraries(heisenberg_complete
    hip::device
    roc::rocblas
    roc::rocsolver
)

# PDMRG - Stream parallelized DMRG (production)
add_executable(pdmrg_complete
    src/pdmrg_complete.cpp
)
set_source_files_properties(src/pdmrg_complete.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(pdmrg_complete PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=${GPU_TARGETS}>
    -O3
)
target_link_libraries(pdmrg_complete
    hip::device
    roc::rocblas
    roc::rocsolver
)

# PDMRG2 - GPU-optimized DMRG with Block-Davidson (production)
add_executable(pdmrg2_complete
    src/pdmrg2_complete.cpp
)
set_source_files_properties(src/pdmrg2_complete.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(pdmrg2_complete PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=${GPU_TARGETS}>
    -O3
)
target_link_libraries(pdmrg2_complete
    hip::device
    roc::rocblas
    roc::rocsolver
)

# Benchmark runner
add_executable(benchmark_runner
    src/benchmark_runner.cpp
)
set_source_files_properties(src/benchmark_runner.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(benchmark_runner PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=${GPU_TARGETS}>
    -O3
)
target_link_libraries(benchmark_runner
    hip::device
    roc::rocblas
    roc::rocsolver
)

# Production DMRG (complete implementation for paper)
add_executable(dmrg_production
    src/dmrg_production.cpp
)
set_source_files_properties(src/dmrg_production.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(dmrg_production PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=${GPU_TARGETS}>
    -O3
)
target_link_libraries(dmrg_production
    hip::device
    roc::rocblas
    roc::rocsolver
)

# Installation
install(TARGETS heisenberg_gpu dmrg_benchmark dmrg_gpu_native heisenberg_complete
        pdmrg_complete pdmrg2_complete benchmark_runner dmrg_production DESTINATION bin)

# Configuration summary
message(STATUS "==============================================")
message(STATUS "GPU DMRG Build Configuration")
message(STATUS "==============================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "ROCm path: ${ROCM_PATH}")
message(STATUS "GPU targets: ${GPU_TARGETS}")
message(STATUS "HIP compiler: ${CMAKE_HIP_COMPILER}")
message(STATUS "CXX compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "hipTensor: ${HIPTENSOR_LIBRARY}")
message(STATUS "Priority: 1. Accuracy, 2. Performance")
message(STATUS "Data type: complex128 (hipDoubleComplex)")
message(STATUS "==============================================")

# WORKING DMRG (simple test version)
add_executable(dmrg_working src/dmrg_working.cpp)
set_source_files_properties(src/dmrg_working.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(dmrg_working PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=${GPU_TARGETS}>
    -O3
)
target_link_libraries(dmrg_working
    hip::device
    roc::rocblas
    roc::rocsolver
)

# DMRG with Full Environment Tensors (MPO-based)
add_executable(dmrg_with_environments
    src/dmrg_with_environments.cpp
)
set_source_files_properties(src/dmrg_with_environments.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(dmrg_with_environments PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=${GPU_TARGETS}>
    -O3
)
target_link_libraries(dmrg_with_environments
    hip::device
    roc::rocblas
    roc::rocsolver
    ${HIPTENSOR_LIBRARY}
)

# Minimal GPU-Only DMRG (Production - Zero CPU Transfers)
add_executable(dmrg_minimal_gpu
    src/dmrg_minimal_gpu.cpp
)
set_source_files_properties(src/dmrg_minimal_gpu.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(dmrg_minimal_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=${GPU_TARGETS}>
    -O3
)
target_link_libraries(dmrg_minimal_gpu
    hip::device
    roc::rocblas
    roc::rocsolver
)

# PDMRG GPU - Production with hipTensor environments + BLAS-2 (stream-parallelized)
add_executable(pdmrg_gpu
    src/pdmrg_gpu.cpp
)
set_source_files_properties(src/pdmrg_gpu.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(pdmrg_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=${GPU_TARGETS}>
    -O3
)
target_link_libraries(pdmrg_gpu
    hip::device
    roc::rocblas
    roc::rocsolver
    ${HIPTENSOR_LIBRARY}
)

# PDMRG2 GPU - Production with hipTensor environments + BLAS-3 H_eff (GPU-optimized)
add_executable(pdmrg2_gpu
    src/pdmrg2_gpu.cpp
)
set_source_files_properties(src/pdmrg2_gpu.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(pdmrg2_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=${GPU_TARGETS}>
    -O3
)
target_link_libraries(pdmrg2_gpu
    hip::device
    roc::rocblas
    roc::rocsolver
    ${HIPTENSOR_LIBRARY}
)

# Phase 1 Tests - AccurateSVD_GPU and OptimizedHeff
add_executable(test_phase1
    src/test_phase1.cpp
    src/accurate_svd_gpu.cpp
    src/heff_optimized_gpu.cpp
)
set_source_files_properties(src/test_phase1.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(src/accurate_svd_gpu.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(src/heff_optimized_gpu.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(test_phase1 PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=${GPU_TARGETS}>
    -O3
)
target_link_libraries(test_phase1
    hip::device
    roc::rocblas
    roc::rocsolver
    ${HIPTENSOR_LIBRARY}
)

# Phase 2: StreamSegment test
add_executable(test_stream_segment
    src/test_stream_segment.cpp
    src/stream_segment.cpp
    src/accurate_svd_gpu.cpp
    src/heff_optimized_gpu.cpp
)
set_source_files_properties(src/test_stream_segment.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(src/stream_segment.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(src/accurate_svd_gpu.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(src/heff_optimized_gpu.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(test_stream_segment PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=${GPU_TARGETS}>
    -O3
)
target_link_libraries(test_stream_segment
    hip::device
    roc::rocblas
    roc::rocsolver
    ${HIPTENSOR_LIBRARY}
)

# Phase 2: BoundaryMergeGPU test
add_executable(test_boundary_merge
    src/test_boundary_merge.cpp
    src/boundary_merge_gpu.cpp
    src/stream_segment.cpp
    src/accurate_svd_gpu.cpp
    src/heff_optimized_gpu.cpp
)
set_source_files_properties(src/test_boundary_merge.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(src/boundary_merge_gpu.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(src/stream_segment.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(src/accurate_svd_gpu.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(src/heff_optimized_gpu.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(test_boundary_merge PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=${GPU_TARGETS}>
    -O3
)
target_link_libraries(test_boundary_merge
    hip::device
    roc::rocblas
    roc::rocsolver
    ${HIPTENSOR_LIBRARY}
    lapack
)

# Phase 2: StreamCoordinator test
add_executable(test_stream_coordinator
    src/test_stream_coordinator.cpp
    src/stream_coordinator.cpp
    src/boundary_merge_gpu.cpp
    src/stream_segment.cpp
    src/accurate_svd_gpu.cpp
    src/heff_optimized_gpu.cpp
)
set_source_files_properties(src/test_stream_coordinator.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(src/stream_coordinator.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(src/boundary_merge_gpu.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(src/stream_segment.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(src/accurate_svd_gpu.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(src/heff_optimized_gpu.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(test_stream_coordinator PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=${GPU_TARGETS}>
    -O3
)
target_link_libraries(test_stream_coordinator
    hip::device
    roc::rocblas
    roc::rocsolver
    ${HIPTENSOR_LIBRARY}
    lapack
)

# Phase 2: Multi-stream iterative DMRG test
add_executable(test_heisenberg_multistream
    src/test_heisenberg_multistream.cpp
    src/stream_coordinator.cpp
    src/boundary_merge_gpu.cpp
    src/stream_segment.cpp
    src/accurate_svd_gpu.cpp
    src/heff_optimized_gpu.cpp
    src/heisenberg_mpo_real.cpp
)
set_source_files_properties(src/test_heisenberg_multistream.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(src/stream_coordinator.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(src/boundary_merge_gpu.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(src/stream_segment.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(src/accurate_svd_gpu.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(src/heff_optimized_gpu.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(src/heisenberg_mpo_real.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(test_heisenberg_multistream PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=${GPU_TARGETS}>
    -O3
)
target_link_libraries(test_heisenberg_multistream
    hip::device
    roc::rocblas
    roc::rocsolver
    ${HIPTENSOR_LIBRARY}
    lapack
)

# Test MPS/MPO binary file loader (CPU-only, no HIP)
add_executable(test_mps_mpo_loader
    src/test_mps_mpo_loader.cpp
)
target_compile_options(test_mps_mpo_loader PRIVATE
    -O2
)
# No GPU libraries needed for this test

# PDMRG Benchmark with Loaded Data (Production benchmarking executable)
add_executable(pdmrg_benchmark_loaded
    src/pdmrg_benchmark_loaded.cpp
)
set_source_files_properties(src/pdmrg_benchmark_loaded.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(pdmrg_benchmark_loaded PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=${GPU_TARGETS}>
    -O3
)
target_link_libraries(pdmrg_benchmark_loaded
    hip::device
    roc::rocblas
    roc::rocsolver
    ${HIPTENSOR_LIBRARY}
    lapack
)
