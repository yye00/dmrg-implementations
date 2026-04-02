cmake_minimum_required(VERSION 3.21)

project(DMRG_GPU_CUDA LANGUAGES CXX CUDA)

# ==========================================
# GPU-Native DMRG (cuBLAS/cuSOLVER GEMM)
# ==========================================
# All tensor contractions on GPU via cuBLAS dgemm
# All linear algebra on GPU via cuBLAS/cuSOLVER

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# GPU target architecture
set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures to compile for")

find_package(CUDAToolkit REQUIRED)

# Find LAPACK/BLAS
find_package(LAPACK REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# ==========================================
# GPU-Native DMRG executable
# ==========================================
add_executable(dmrg_gpu
    src/test_dmrg_gpu.cu
    src/dmrg_gpu.cu
    src/accurate_svd_gpu.cu
)

target_compile_options(dmrg_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --allow-unsupported-compiler>
)

target_link_libraries(dmrg_gpu
    CUDA::cublas
    CUDA::cusolver
    ${LAPACK_LIBRARIES}
)

# ==========================================
# Build info
# ==========================================
message(STATUS "==============================================")
message(STATUS "GPU-Native DMRG Build Configuration (CUDA)")
message(STATUS "==============================================")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Backend: cuBLAS GEMM")
message(STATUS "==============================================")
