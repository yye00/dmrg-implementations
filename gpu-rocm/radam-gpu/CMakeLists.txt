cmake_minimum_required(VERSION 3.21)

set(CMAKE_C_COMPILER /opt/rocm/llvm/bin/clang)
set(CMAKE_CXX_COMPILER /opt/rocm/llvm/bin/clang++)

project(RADAM_GPU LANGUAGES CXX HIP)

# ==========================================
# RADAM-GPU: Riemannian Adam for MPS/TT ground state
# ==========================================

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
    message(WARNING "Local OpenBLAS not found; falling back to system LAPACK")
    find_package(LAPACK REQUIRED)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(radam_gpu
    src/test_radam_gpu.cpp
    src/radam_gpu.cpp
)

set_source_files_properties(
    src/test_radam_gpu.cpp
    src/radam_gpu.cpp
    PROPERTIES LANGUAGE HIP
)

target_compile_options(radam_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=${GPU_TARGETS}>
    -O3
)

target_link_libraries(radam_gpu
    hip::device
    roc::rocblas
    roc::rocsolver
    ${LAPACK_LIBRARIES}
)

message(STATUS "==============================================")
message(STATUS "RADAM-GPU: Riemannian Adam for MPS/TT")
message(STATUS "GPU targets: ${GPU_TARGETS}")
message(STATUS "==============================================")
