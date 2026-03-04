# PDMRG GPU Port - Minimal CMake Configuration
# Target: AMD MI300X with hipTensor

cmake_minimum_required(VERSION 3.20)
project(pdmrg_gpu LANGUAGES CXX)

# Enable HIP language support
enable_language(HIP)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ROCm/HIP setup
if(NOT DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to ROCm installation")
else()
    set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Path to ROCm installation")
endif()

list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})

# Find ROCm/HIP packages
find_package(hip REQUIRED)
find_package(hipblas REQUIRED)
find_package(rocsolver REQUIRED)

# Find hipTensor (may need manual path)
find_library(HIPTENSOR_LIBRARY
    NAMES hiptensor
    PATHS ${ROCM_PATH}/lib ${ROCM_PATH}/lib64
    REQUIRED
)
find_path(HIPTENSOR_INCLUDE_DIR
    NAMES hiptensor.h
    PATHS ${ROCM_PATH}/include
    REQUIRED
)

# Find MPI
find_package(MPI REQUIRED)

# Architecture target (MI300X = gfx942)
set(GPU_TARGETS "gfx942" CACHE STRING "GPU architectures to compile for")
set(CMAKE_HIP_ARCHITECTURES ${GPU_TARGETS})

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# HIP flags
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} --offload-arch=${GPU_TARGETS}")

# Optional: fast math (disable for initial correctness testing)
option(ENABLE_FAST_MATH "Enable fast math optimizations" OFF)
if(ENABLE_FAST_MATH)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -ffast-math")
endif()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${HIPTENSOR_INCLUDE_DIR}
    ${MPI_CXX_INCLUDE_DIRS}
)

# Source files
set(SOURCES
    src/main.cpp
    # Add more sources as developed
    # src/tensor_ops.cpp
    # src/mpo_builders.cpp
    # src/eigensolver.cpp
)

set(HIP_SOURCES
    # Add HIP kernel files
    # src/dmrg_kernel.hip
)

# Executable
add_executable(pdmrg_gpu ${SOURCES} ${HIP_SOURCES})

# Link libraries
target_link_libraries(pdmrg_gpu
    hip::host
    hip::device
    roc::hipblas
    roc::rocsolver
    ${HIPTENSOR_LIBRARY}
    MPI::MPI_CXX
)

# Install target
install(TARGETS pdmrg_gpu DESTINATION bin)

# Print configuration summary
message(STATUS "==============================================")
message(STATUS "PDMRG GPU Build Configuration")
message(STATUS "==============================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "ROCm path: ${ROCM_PATH}")
message(STATUS "GPU targets: ${GPU_TARGETS}")
message(STATUS "HIP compiler: ${CMAKE_HIP_COMPILER}")
message(STATUS "CXX compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "MPI found: ${MPI_FOUND}")
message(STATUS "hipTensor library: ${HIPTENSOR_LIBRARY}")
message(STATUS "Fast math: ${ENABLE_FAST_MATH}")
message(STATUS "==============================================")
