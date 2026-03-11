cmake_minimum_required(VERSION 3.21)

# Set compiler BEFORE project() for ROCm/HIP
set(CMAKE_C_COMPILER /opt/rocm/llvm/bin/clang)
set(CMAKE_CXX_COMPILER /opt/rocm/llvm/bin/clang++)

project(DMRG_GPU LANGUAGES CXX HIP)

# ==========================================
# GPU-Native DMRG (hipTensor + rocBLAS)
# ==========================================
# All tensor contractions on GPU via hipTensor
# All linear algebra on GPU via rocBLAS/rocSOLVER

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

# Find hipTensor
find_library(HIPTENSOR_LIBRARY NAMES hiptensor
    PATHS ${ROCM_PATH}/lib ${ROCM_PATH}/lib64)
find_path(HIPTENSOR_INCLUDE_DIR hiptensor/hiptensor.h
    PATHS ${ROCM_PATH}/include)
if(NOT HIPTENSOR_LIBRARY)
    message(FATAL_ERROR "hipTensor not found at ${ROCM_PATH}")
endif()
message(STATUS "hipTensor library: ${HIPTENSOR_LIBRARY}")
message(STATUS "hipTensor include: ${HIPTENSOR_INCLUDE_DIR}")
include_directories(${HIPTENSOR_INCLUDE_DIR})

# Find LAPACK (for Lanczos tridiagonal eigensolve)
find_package(LAPACK REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# ==========================================
# GPU-Native DMRG executable
# ==========================================
add_executable(dmrg_gpu
    src/test_dmrg_gpu.cpp
    src/dmrg_gpu.cpp
    src/accurate_svd_gpu.cpp
)

set_source_files_properties(
    src/test_dmrg_gpu.cpp
    src/dmrg_gpu.cpp
    src/accurate_svd_gpu.cpp
    PROPERTIES LANGUAGE HIP
)

target_compile_options(dmrg_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=${GPU_TARGETS}>
    -O3
)

target_link_libraries(dmrg_gpu
    hip::device
    roc::rocblas
    roc::rocsolver
    ${HIPTENSOR_LIBRARY}
    ${LAPACK_LIBRARIES}
)

# ==========================================
# Build info
# ==========================================
message(STATUS "==============================================")
message(STATUS "GPU-Native DMRG Build Configuration")
message(STATUS "==============================================")
message(STATUS "GPU targets: ${GPU_TARGETS}")
message(STATUS "hipTensor: ${HIPTENSOR_LIBRARY}")
message(STATUS "==============================================")
