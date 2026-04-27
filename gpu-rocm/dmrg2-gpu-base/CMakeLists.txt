cmake_minimum_required(VERSION 3.21)

# Set compiler BEFORE project() for ROCm/HIP
set(CMAKE_C_COMPILER /opt/rocm/llvm/bin/clang)
set(CMAKE_CXX_COMPILER /opt/rocm/llvm/bin/clang++)

project(DMRG2_GPU_BASE LANGUAGES CXX HIP)

# ==========================================
# NAIVE BASELINE — two-site DMRG on GPU
# ==========================================
# Competent first-pass GPU implementation. Two-site fused MPO (WW) is
# precomputed once per bond at set_mpo() time and reused across all
# Lanczos iterations (no per-iteration host roundtrip). All linear
# algebra on device: rocBLAS in device-pointer mode for the Lanczos
# inner loop, rocSOLVER dsteqr for the tridiagonal eigensolve, rocSOLVER
# gesvd_auto for the SVD, device-side kernels for SVD truncation/scaling.
# No CPU LAPACK calls. Used as the speedup reference for dmrg2-gpu.

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

add_executable(dmrg2_gpu_base
    src/test_dmrg2_gpu_base.cpp
    src/dmrg2_gpu_base.cpp
)

set_source_files_properties(
    src/test_dmrg2_gpu_base.cpp
    src/dmrg2_gpu_base.cpp
    PROPERTIES LANGUAGE HIP
)

target_compile_options(dmrg2_gpu_base PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=${GPU_TARGETS}>
    -O3
)

target_link_libraries(dmrg2_gpu_base
    hip::device
    roc::rocblas
    roc::rocsolver
)

message(STATUS "==============================================")
message(STATUS "DMRG2-GPU-BASE (NAIVE BASELINE) Build Config")
message(STATUS "==============================================")
message(STATUS "GPU targets: ${GPU_TARGETS}")
message(STATUS "Backend: naive rocBLAS (no gemm_batched, no D_PAD/SPARSE_MPO)")
message(STATUS "WW: precomputed at set_mpo() (host build, single H2D)")
message(STATUS "Lanczos: device-pointer mode + rocSOLVER dsteqr")
message(STATUS "SVD: rocSOLVER gesvd_auto + on-device truncation")
message(STATUS "==============================================")
