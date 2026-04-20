cmake_minimum_required(VERSION 3.21)

# Set compiler BEFORE project() for ROCm/HIP
set(CMAKE_C_COMPILER /opt/rocm/llvm/bin/clang)
set(CMAKE_CXX_COMPILER /opt/rocm/llvm/bin/clang++)

project(DMRG2_GPU_OPT LANGUAGES CXX HIP)

# ==========================================
# GPU-Native Two-Site DMRG-OPT (Block-Davidson)
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
# Detect full path to libgfortran (ld.lld does not search gcc default paths).
execute_process(COMMAND gfortran -print-file-name=libgfortran.so OUTPUT_VARIABLE GFORTRAN_LIB OUTPUT_STRIP_TRAILING_WHITESPACE)

# Prefer local OpenBLAS 0.3.28+ (system OpenBLAS 0.3.20 has broken dgesvd for tall-skinny matrices)
set(OPENBLAS_PATH "$ENV{HOME}/openblas-0.3.28" CACHE PATH "Path to local OpenBLAS installation")
if(EXISTS "/usr/local/openblas/lib/libopenblas.so")
    message(STATUS "Using system-local OpenBLAS: /usr/local/openblas")
    set(LAPACK_LIBRARIES "/usr/local/openblas/lib/libopenblas.so" pthread ${GFORTRAN_LIB} gomp)
    include_directories(/usr/local/openblas/include)
elseif(EXISTS "${OPENBLAS_PATH}/lib/libopenblas.a")
    message(STATUS "Using local OpenBLAS: ${OPENBLAS_PATH}")
    set(LAPACK_LIBRARIES "${OPENBLAS_PATH}/lib/libopenblas.a" pthread ${GFORTRAN_LIB} gomp)
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
# GPU-Native Two-Site DMRG-OPT executable
# ==========================================
add_executable(dmrg2_gpu_opt
    src/test_dmrg2_gpu_opt.cpp
    src/dmrg2_gpu_opt.cpp
)

set_source_files_properties(
    src/test_dmrg2_gpu_opt.cpp
    src/dmrg2_gpu_opt.cpp
    PROPERTIES LANGUAGE HIP
)

target_compile_options(dmrg2_gpu_opt PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=${GPU_TARGETS}>
    -O3
)

target_link_libraries(dmrg2_gpu_opt
    hip::device
    roc::rocblas
    roc::rocsolver
    ${LAPACK_LIBRARIES}
)

# ==========================================
# Build info
# ==========================================
message(STATUS "==============================================")
message(STATUS "GPU-Native Two-Site DMRG-OPT Build Configuration")
message(STATUS "==============================================")
message(STATUS "GPU targets: ${GPU_TARGETS}")
message(STATUS "Backend: rocBLAS GEMM (Block-Davidson)")
message(STATUS "==============================================")
