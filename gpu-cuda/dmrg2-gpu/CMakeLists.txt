cmake_minimum_required(VERSION 3.21)

project(DMRG2_GPU_CUDA LANGUAGES CXX CUDA)

# ==========================================
# GPU-Native Two-Site DMRG (cuBLAS/cuSOLVER)
# ==========================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# GPU target architecture (sm_89 for RTX 4090)
set(CMAKE_CUDA_ARCHITECTURES 89)

find_package(CUDAToolkit REQUIRED)

# Find LAPACK/BLAS (for CPU tridiagonal eigensolver in Lanczos)
find_package(LAPACK REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# ==========================================
# GPU-Native Two-Site DMRG executable
# ==========================================
add_executable(dmrg2_gpu
    src/test_dmrg2_gpu.cu
    src/dmrg2_gpu.cu
)

target_link_libraries(dmrg2_gpu
    CUDA::cublas
    CUDA::cusolver
    ${LAPACK_LIBRARIES}
)

target_compile_options(dmrg2_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --allow-unsupported-compiler>
)

# ==========================================
# Build info
# ==========================================
message(STATUS "==============================================")
message(STATUS "GPU-Native Two-Site DMRG Build Configuration")
message(STATUS "==============================================")
message(STATUS "GPU targets: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Backend: cuBLAS GEMM (two-site optimization)")
message(STATUS "==============================================")
