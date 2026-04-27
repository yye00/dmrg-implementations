cmake_minimum_required(VERSION 3.21)

# Set compiler BEFORE project() for ROCm/HIP
set(CMAKE_C_COMPILER /opt/rocm/llvm/bin/clang)
set(CMAKE_CXX_COMPILER /opt/rocm/llvm/bin/clang++)

project(DMRG_GPU_BASE LANGUAGES CXX HIP)

# ==========================================
# NAIVE BASELINE — single-site DMRG on GPU
# ==========================================
# Competent first-pass GPU implementation. All linear algebra on device:
# rocBLAS in device-pointer mode for the Lanczos inner loop, rocSOLVER
# dsteqr for the tridiagonal eigensolve, rocSOLVER gesvd_auto for the SVD,
# device-side kernels for SVD truncation/scaling. No CPU LAPACK calls and
# no host roundtrips of device-resident data inside the sweep loop.
# Used as the speedup reference for dmrg-gpu.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)

set(GPU_TARGETS "gfx942" CACHE STRING "GPU architectures to compile for")

if(NOT DEFINED ROCM_PATH)
    set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to ROCm installation")
endif()

list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})

find_package(hip REQUIRED)
find_package(rocblas REQUIRED)
find_package(rocsolver REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

add_executable(dmrg_gpu_base
    src/test_dmrg_gpu_base.cpp
    src/dmrg_gpu_base.cpp
)

set_source_files_properties(
    src/test_dmrg_gpu_base.cpp
    src/dmrg_gpu_base.cpp
    PROPERTIES LANGUAGE HIP
)

target_compile_options(dmrg_gpu_base PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=${GPU_TARGETS}>
    -O3
)

target_link_libraries(dmrg_gpu_base
    hip::device
    roc::rocblas
    roc::rocsolver
)

message(STATUS "==============================================")
message(STATUS "DMRG-GPU-BASE (NAIVE BASELINE) Build Config")
message(STATUS "==============================================")
message(STATUS "GPU targets: ${GPU_TARGETS}")
message(STATUS "Backend: naive rocBLAS (no gemm_batched, no fused)")
message(STATUS "Lanczos: device-pointer mode + rocSOLVER dsteqr")
message(STATUS "SVD: rocSOLVER gesvd_auto + on-device truncation")
message(STATUS "==============================================")
