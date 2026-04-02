cmake_minimum_required(VERSION 3.21)

# Set compiler BEFORE project() for ROCm/HIP
set(CMAKE_C_COMPILER /opt/rocm/llvm/bin/clang)
set(CMAKE_CXX_COMPILER /opt/rocm/llvm/bin/clang++)

project(PDMRG_GPU LANGUAGES CXX HIP)

# ==========================================
# Stream-Parallel DMRG (PDMRG) on GPU
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

# Find LAPACK/BLAS
# Prefer local OpenBLAS 0.3.28+ (system OpenBLAS 0.3.20 has broken dgesvd for tall-skinny matrices)
set(OPENBLAS_PATH "$ENV{HOME}/openblas-0.3.28" CACHE PATH "Path to local OpenBLAS installation")
if(EXISTS "/usr/local/openblas/lib/libopenblas.so")
    message(STATUS "Using system-local OpenBLAS: /usr/local/openblas")
    set(LAPACK_LIBRARIES "/usr/local/openblas/lib/libopenblas.so" pthread)
    include_directories(/usr/local/openblas/include)
elseif(EXISTS "${OPENBLAS_PATH}/lib/libopenblas.a")
    message(STATUS "Using local OpenBLAS: ${OPENBLAS_PATH}")
    set(LAPACK_LIBRARIES "${OPENBLAS_PATH}/lib/libopenblas.a" pthread)
    include_directories(${OPENBLAS_PATH}/include)
else()
    message(WARNING "Local OpenBLAS not found at ${OPENBLAS_PATH}, falling back to system LAPACK (may have SVD bugs)")
    find_package(LAPACK REQUIRED)
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# ==========================================
# PDMRG-GPU executable
# ==========================================
add_executable(pdmrg_gpu
    src/test_pdmrg_gpu.cpp
    src/pdmrg_gpu.cpp
)

set_source_files_properties(
    src/test_pdmrg_gpu.cpp
    src/pdmrg_gpu.cpp
    PROPERTIES LANGUAGE HIP
)

target_compile_options(pdmrg_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=${GPU_TARGETS}>
    -O3
)

target_link_libraries(pdmrg_gpu
    hip::device
    roc::rocblas
    roc::rocsolver
    ${LAPACK_LIBRARIES}
)

# ==========================================
# Build info
# ==========================================
message(STATUS "==============================================")
message(STATUS "Stream-Parallel DMRG (PDMRG) GPU Build")
message(STATUS "==============================================")
message(STATUS "GPU targets: ${GPU_TARGETS}")
message(STATUS "Backend: rocBLAS GEMM + HIP streams")
message(STATUS "==============================================")
