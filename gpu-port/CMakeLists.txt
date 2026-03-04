# PDMRG GPU Port - Single-GPU CMake Configuration
# Target: AMD MI300X (192GB) with hipTensor - NO MPI

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

# Find hipTensor (optional for now - simple_dmrg doesn't need it)
find_library(HIPTENSOR_LIBRARY
    NAMES hiptensor
    PATHS ${ROCM_PATH}/lib ${ROCM_PATH}/lib64
)
find_path(HIPTENSOR_INCLUDE_DIR
    NAMES hiptensor.h hiptensor/hiptensor.h
    PATHS ${ROCM_PATH}/include
)

if(HIPTENSOR_LIBRARY)
    message(STATUS "hipTensor found: ${HIPTENSOR_LIBRARY}")
else()
    message(WARNING "hipTensor not found - GPU implementation will need it")
endif()

# NO MPI - single GPU only!

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
    message(WARNING "Fast math enabled - verify DMRG convergence carefully!")
endif()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${HIPTENSOR_INCLUDE_DIR}
)

# Find Eigen (required for CPU test code)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Source files for simple CPU DMRG test
set(SIMPLE_DMRG_SOURCES
    src/simple_dmrg.cpp
    src/heisenberg_mpo.cpp
)

# Main GPU DMRG sources (to be developed)
set(SOURCES
    # src/main.cpp
    # Add more sources as developed
    # src/tensor_ops.cpp
    # src/mpo_builders.cpp
    # src/eigensolver.cpp
    # src/stream_manager.cpp
    # src/gpu_memory.cpp
)

set(HIP_SOURCES
    # Add HIP kernel files
    # src/dmrg_kernel.hip
)

# Build simple CPU DMRG test first
add_executable(simple_dmrg ${SIMPLE_DMRG_SOURCES})
target_link_libraries(simple_dmrg Eigen3::Eigen)

# Main GPU executable (commented out until implemented)
# add_executable(pdmrg_gpu ${SOURCES} ${HIP_SOURCES})

# Main GPU executable link libraries (when implemented)
# target_link_libraries(pdmrg_gpu
#     hip::host
#     hip::device
#     roc::hipblas
#     roc::rocsolver
#     ${HIPTENSOR_LIBRARY}
# )

# Install target
install(TARGETS simple_dmrg DESTINATION bin)
# install(TARGETS pdmrg_gpu DESTINATION bin)

# Print configuration summary
message(STATUS "==============================================")
message(STATUS "PDMRG GPU Build Configuration (Single-GPU)")
message(STATUS "==============================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "ROCm path: ${ROCM_PATH}")
message(STATUS "GPU targets: ${GPU_TARGETS}")
message(STATUS "HIP compiler: ${CMAKE_HIP_COMPILER}")
message(STATUS "CXX compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "MPI: NOT USED (single GPU)")
message(STATUS "hipTensor library: ${HIPTENSOR_LIBRARY}")
message(STATUS "Fast math: ${ENABLE_FAST_MATH}")
message(STATUS "==============================================")
