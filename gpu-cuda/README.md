# CUDA GPU Implementations

CUDA ports of the ROCm DMRG implementations for NVIDIA GPUs.
Each subdirectory is a fully independent codebase — no shared code with gpu-rocm/.

## Porting status

| Implementation | Algorithm | Status |
|---------------|-----------|--------|
| dmrg-gpu | Single-site, Lanczos + SVD | Ported (untested) |
| dmrg-gpu-opt | Single-site, Block-Davidson + Newton-Schulz | Ported (untested) |
| dmrg2-gpu | Two-site, Lanczos + SVD | Ported (untested) |
| dmrg2-gpu-opt | Two-site, Block-Davidson + Newton-Schulz | Ported (untested) |
| pdmrg-gpu | Parallel two-site, Lanczos + SVD | Ported (untested) |
| pdmrg-gpu-opt | Parallel two-site, Block-Davidson + Newton-Schulz | Ported (untested) |

## Building

Requires CUDA toolkit with nvcc, cuBLAS, cuSOLVER, and a host LAPACK (for CPU tridiagonal eigensolver).

```bash
cd gpu-cuda/<implementation>
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=90   # H100 = sm_90, RTX 4090 = sm_89
make -j
```

CMakeLists include `--allow-unsupported-compiler` for GCC 15 compatibility.

## Running tests (smallest cases)

```bash
# Single-site Heisenberg L=4 chi=4
./dmrg_gpu 4 4 10

# Two-site Heisenberg L=4 chi=4
./dmrg2_gpu 4 4 10

# Two-site Josephson L=4 chi=8 (complex128)
./dmrg2_gpu 4 8 10 --josephson

# Parallel two-site Heisenberg L=8 chi=8 np=2
./pdmrg_gpu 8 8 10 2
```

## Dependencies

- CUDA Toolkit (cublas, cusolver)
- LAPACK/BLAS (CPU eigensolver fallback)
- C++17 compiler
