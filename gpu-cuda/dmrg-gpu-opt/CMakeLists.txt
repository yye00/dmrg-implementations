cmake_minimum_required(VERSION 3.21)

project(DMRG_GPU_OPT_CUDA LANGUAGES CXX CUDA)

# ==========================================
# GPU-Native DMRG-OPT (Newton-Schulz + Block-Davidson) — CUDA port
# ==========================================
# Single-site optimization with NS polar decomposition
# and Block-Davidson eigensolver (BLAS-3 dominant)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

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
# GPU-Native DMRG-OPT executable
# ==========================================
add_executable(dmrg_gpu_opt
    src/test_dmrg_gpu_opt.cu
    src/dmrg_gpu_opt.cu
)

target_compile_options(dmrg_gpu_opt PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --allow-unsupported-compiler>
)

target_link_libraries(dmrg_gpu_opt
    CUDA::cublas
    CUDA::cusolver
    ${LAPACK_LIBRARIES}
)

# ==========================================
# Build info
# ==========================================
message(STATUS "==============================================")
message(STATUS "GPU-Native DMRG-OPT CUDA Build Configuration")
message(STATUS "==============================================")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Backend: cuBLAS GEMM + Newton-Schulz + Block-Davidson")
message(STATUS "==============================================")
