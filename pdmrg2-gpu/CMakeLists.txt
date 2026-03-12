cmake_minimum_required(VERSION 3.21)

# Set compiler BEFORE project() for ROCm/HIP
set(CMAKE_C_COMPILER /opt/rocm/llvm/bin/clang)
set(CMAKE_CXX_COMPILER /opt/rocm/llvm/bin/clang++)

project(PDMRG2_GPU LANGUAGES CXX HIP)

# ==========================================
# PDMRG2-GPU: Newton-Schulz + Block-Davidson
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

# Find LAPACK (for Block-Davidson eigensolve + CPU SVD fallback)
find_package(LAPACK REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# ==========================================
# PDMRG2-GPU executable
# ==========================================
add_executable(pdmrg2_gpu
    src/test_pdmrg2_gpu.cpp
    src/pdmrg2_gpu.cpp
)

set_source_files_properties(
    src/test_pdmrg2_gpu.cpp
    src/pdmrg2_gpu.cpp
    PROPERTIES LANGUAGE HIP
)

target_compile_options(pdmrg2_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=${GPU_TARGETS}>
    -O3
)

target_link_libraries(pdmrg2_gpu
    hip::device
    roc::rocblas
    roc::rocsolver
    ${LAPACK_LIBRARIES}
)

# ==========================================
# Build info
# ==========================================
message(STATUS "==============================================")
message(STATUS "PDMRG2-GPU: Newton-Schulz + Block-Davidson")
message(STATUS "==============================================")
message(STATUS "GPU targets: ${GPU_TARGETS}")
message(STATUS "Backend: rocBLAS GEMM + Newton-Schulz + Block-Davidson")
message(STATUS "==============================================")
