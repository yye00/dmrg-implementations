cmake_minimum_required(VERSION 3.21)
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

# Compiler flags for accuracy and performance
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
set(CMAKE_HIP_FLAGS "-O3 --offload-arch=${GPU_TARGETS}")

# NO fast-math to ensure accuracy
# Complex128 precision is critical

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${ROCM_PATH}/include
)

# Heisenberg benchmark
add_executable(heisenberg_gpu src/heisenberg_dmrg_gpu.cpp)

target_link_libraries(heisenberg_gpu
    hip::device
    roc::rocblas
    roc::rocsolver
    ${HIPTENSOR_LIBRARY}
)

# Installation
install(TARGETS heisenberg_gpu DESTINATION bin)

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
