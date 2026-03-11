cmake_minimum_required(VERSION 3.21)

# Set compiler BEFORE project() for ROCm/HIP
set(CMAKE_C_COMPILER /opt/rocm/llvm/bin/clang)
set(CMAKE_CXX_COMPILER /opt/rocm/llvm/bin/clang++)

project(DMRG2_GPU LANGUAGES CXX HIP)

# ==========================================
# GPU-Native Two-Site DMRG (rocBLAS/rocSOLVER)
# ==========================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)

# GPU target architecture
set(GPU_TARGETS "gfx942" CACHE STRING "GPU architectures to compile for")

# ROCm setup
if(NOT DEFINED ROCM_PATH)
    set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to ROCm installation")
endif()

list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})

find_package(hip REQUIRED)
find_package(rocblas REQUIRED)
find_package(rocsolver REQUIRED)

# Find LAPACK (for Lanczos tridiagonal eigensolve)
find_package(LAPACK REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# ==========================================
# GPU-Native Two-Site DMRG executable
# ==========================================
add_executable(dmrg2_gpu
    src/test_dmrg2_gpu.cpp
    src/dmrg2_gpu.cpp
)

set_source_files_properties(
    src/test_dmrg2_gpu.cpp
    src/dmrg2_gpu.cpp
    PROPERTIES LANGUAGE HIP
)

target_compile_options(dmrg2_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=${GPU_TARGETS}>
    -O3
)

target_link_libraries(dmrg2_gpu
    hip::device
    roc::rocblas
    roc::rocsolver
    ${LAPACK_LIBRARIES}
)

# ==========================================
# Build info
# ==========================================
message(STATUS "==============================================")
message(STATUS "GPU-Native Two-Site DMRG Build Configuration")
message(STATUS "==============================================")
message(STATUS "GPU targets: ${GPU_TARGETS}")
message(STATUS "Backend: rocBLAS GEMM (two-site optimization)")
message(STATUS "==============================================")
