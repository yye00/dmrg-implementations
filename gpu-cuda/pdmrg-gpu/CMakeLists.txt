cmake_minimum_required(VERSION 3.21)

project(PDMRG_GPU_CUDA LANGUAGES CXX CUDA)

# ==========================================
# Stream-Parallel DMRG (PDMRG) on GPU
# ==========================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# GPU target architecture
set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures to compile for")

# Find CUDA toolkit
find_package(CUDAToolkit REQUIRED)

# Find LAPACK/BLAS
find_package(LAPACK REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# ==========================================
# PDMRG-GPU executable
# ==========================================
add_executable(pdmrg_gpu
    src/test_pdmrg_gpu.cu
    src/pdmrg_gpu.cu
)

target_compile_options(pdmrg_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --allow-unsupported-compiler>
)

target_link_libraries(pdmrg_gpu
    CUDA::cublas
    CUDA::cusolver
    ${LAPACK_LIBRARIES}
)

# ==========================================
# Build info
# ==========================================
message(STATUS "==============================================")
message(STATUS "Stream-Parallel DMRG (PDMRG) GPU Build (CUDA)")
message(STATUS "==============================================")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Backend: cuBLAS GEMM + CUDA streams")
message(STATUS "==============================================")
