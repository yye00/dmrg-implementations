cmake_minimum_required(VERSION 3.21)

# Set compiler BEFORE project() for ROCm/HIP
set(CMAKE_C_COMPILER /opt/rocm/llvm/bin/clang)
set(CMAKE_CXX_COMPILER /opt/rocm/llvm/bin/clang++)

project(PDMRG_GPU_BASE LANGUAGES CXX HIP)

# ==========================================
# NAIVE BASELINE — stream-parallel PDMRG on GPU
# ==========================================
# Competent first-pass GPU implementation. Preserves PDMRG's defining
# feature — chain partitioned into N segments running concurrently on N
# non-blocking HIP streams — and runs all linear algebra on device:
#   - Two-site fused MPO (WW) precomputed once per bond at set_mpo()
#   - rocBLAS in device-pointer mode for the per-stream Lanczos inner loops
#   - rocSOLVER dsteqr (per-stream) for the tridiagonal eigensolve
#   - rocSOLVER gesvd_auto + on-device kernels for SVD truncation/scaling
#   - Single-site warmup AND polish (CLAUDE.md compliant)
#   - Configurable n_warmup, n_polish, n_segments, n_local via CLI
# No CPU LAPACK calls. Used as the speedup reference for pdmrg-gpu.

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

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
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
    pthread
)

message(STATUS "==============================================")
message(STATUS "PDMRG-GPU-BASE (NAIVE BASELINE) Build Config")
message(STATUS "==============================================")
message(STATUS "GPU targets: ${GPU_TARGETS}")
message(STATUS "Backend: stream-parallel naive rocBLAS")
message(STATUS "  - N non-blocking HIP streams + N rocBLAS handles")
message(STATUS "  - std::thread parallel segment sweeps")
message(STATUS "WW: precomputed at set_mpo() (host build, single H2D)")
message(STATUS "Lanczos: device-pointer mode + rocSOLVER dsteqr (per-stream)")
message(STATUS "SVD: rocSOLVER gesvd_auto + on-device truncation")
message(STATUS "Warmup/polish: single-site (CLAUDE.md compliant)")
message(STATUS "==============================================")
