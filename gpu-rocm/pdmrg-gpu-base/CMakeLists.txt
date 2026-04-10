cmake_minimum_required(VERSION 3.21)

# Set compiler BEFORE project() for ROCm/HIP
set(CMAKE_C_COMPILER /opt/rocm/llvm/bin/clang)
set(CMAKE_CXX_COMPILER /opt/rocm/llvm/bin/clang++)

project(PDMRG_GPU_BASE LANGUAGES CXX HIP)

# ==========================================
# NAIVE BASELINE — stream-parallel PDMRG on GPU
# ==========================================
# Preserves PDMRG's defining feature — chain partitioned into N segments
# running concurrently on N HIP streams — but strips every per-stream
# optimization from `pdmrg-gpu`:
#   - WW two-site MPO rebuilt from scratch on host for every apply_heff call
#   - No gemm_batched: unbatched single rocBLAS gemm loops throughout
#   - Host-pointer-mode Lanczos + CPU LAPACK dstev for tridiagonal eigensolve
#   - rocSOLVER gesvd + host-side truncation for SVD split
#   - No randomized SVD, no on-device SVD truncation
#   - Naive host-side V-scaling at boundaries
# Hard-coded defaults: n_warmup=3, n_local=2, polish=10 sweeps.
# Used as the speedup reference for pdmrg-gpu.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)

set(GPU_TARGETS "gfx942" CACHE STRING "GPU architectures to compile for")

if(NOT DEFINED ROCM_PATH)
    set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to ROCm installation")
endif()

list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})

find_package(hip REQUIRED)
find_package(rocblas REQUIRED)
find_package(rocsolver REQUIRED)

# Find LAPACK/BLAS for CPU dstev + potential gesvd fallback
# Prefer local OpenBLAS 0.3.28+ (system OpenBLAS 0.3.20 has broken dgesvd)
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
    message(WARNING "Local OpenBLAS not found, falling back to system LAPACK")
    find_package(LAPACK REQUIRED)
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

add_executable(pdmrg_gpu_base
    src/test_pdmrg_gpu_base.cpp
    src/pdmrg_gpu_base.cpp
)

set_source_files_properties(
    src/test_pdmrg_gpu_base.cpp
    src/pdmrg_gpu_base.cpp
    PROPERTIES LANGUAGE HIP
)

target_compile_options(pdmrg_gpu_base PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=${GPU_TARGETS}>
    -O3
)

target_link_libraries(pdmrg_gpu_base
    hip::device
    roc::rocblas
    roc::rocsolver
    ${LAPACK_LIBRARIES}
    pthread
)

message(STATUS "==============================================")
message(STATUS "PDMRG-GPU-BASE (NAIVE BASELINE) Build Config")
message(STATUS "==============================================")
message(STATUS "GPU targets: ${GPU_TARGETS}")
message(STATUS "Backend: stream-parallel naive rocBLAS")
message(STATUS "  - N HIP streams + N rocBLAS handles")
message(STATUS "  - std::thread parallel segment sweeps")
message(STATUS "WW: rebuilt on host every apply_heff call")
message(STATUS "Lanczos: host-pointer mode + CPU dstev")
message(STATUS "SVD: rocSOLVER gesvd + host-side truncation")
message(STATUS "Hard-coded: n_warmup=3, n_local=2, polish=10")
message(STATUS "==============================================")
