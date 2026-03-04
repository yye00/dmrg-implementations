# PDMRG/PDMRG2 GPU Port Game Plan
## Target: AMD MI300X via HotAisle with hipTensor + C++

**Status:** Research & Planning Phase
**Date:** 2026-03-03
**Goal:** Port PDMRG/PDMRG2 to AMD MI300X using hipTensor for tensor contractions

---

## 1. Hardware & Platform Overview

### AMD MI300X Specifications
- **Architecture:** CDNA 3
- **Memory:** 192GB HBM3 (8TB/s bandwidth)
- **Compute:** ~1.3 PFLOPS FP16, ~650 TFLOPS FP32, ~325 TFLOPS FP64
- **Key Advantage:** Massive unified memory ideal for large tensor operations
- **Challenges:** Different programming model than CUDA

### HotAisle Provider
- **Need to verify:**
  - ROCm version available (ROCm 5.7+ or ROCm 6.x preferred)
  - Docker support vs bare metal access
  - MPI configuration for multi-GPU
  - Network topology (important for distributed DMRG)
  - Persistence (container restarts, volume mounts)

---

## 2. Technology Stack Decision Matrix

### Core Library: hipTensor
**Pros:**
- Native AMD tensor contraction library (equivalent to cuTENSOR)
- Optimized for MI300X architecture
- C++ API with similar semantics to cuTENSOR
- Supports batch operations and async execution

**Cons:**
- Less mature than cuTENSOR
- Documentation may be sparse
- Need to verify tensor network support vs simple GEMM

**Alternative to investigate:** Direct rocBLAS + manual tensor reshaping if hipTensor insufficient

### Language: C++ (Confirmed)
**Rationale:**
- hipTensor is C/C++ native
- Better performance control than Python bindings
- Direct access to HIP streams and async operations

### Parallelism Strategy: **Single-GPU with HIP Streams** (NO MPI)
**Key Decision:** Entire DMRG problem fits comfortably in 192GB HBM3

**Parallelism approach:**
1. **HIP Streams** for overlapping operations
   - Pipelined tensor contractions
   - Overlap computation with memory copies (if needed)
   - Multiple concurrent eigensolver iterations

2. **Kernel-level parallelism**
   - Each tensor contraction uses full GPU (thousands of threads)
   - Batch operations for multiple sites
   - Parallel sweep updates

3. **No MPI needed**
   - Simpler build (no ROCm-aware MPI complexity)
   - No inter-GPU communication overhead
   - Easier debugging and profiling

---

## 3. Deployment Strategy: Docker vs Native

### Recommendation: **Containerized with ROCm Docker Base**

#### Option A: ROCm Docker Container (RECOMMENDED)
**Pros:**
- Reproducible environment
- AMD provides official ROCm Docker images
- Easier dependency management
- Portable between HotAisle instances

**Cons:**
- Need `--device=/dev/kfd --device=/dev/dri` flags for GPU access
- Volume mounts for code/data
- ~~Slightly more complex MPI setup~~ (NOT NEEDED - single GPU only!)

**Base Image:**
```dockerfile
FROM rocm/dev-ubuntu-22.04:6.0
# or latest ROCm 6.x version
```

**Required Components:**
- ROCm runtime (rocminfo, rocm-smi)
- hipTensor library
- rocBLAS, rocSOLVER (for dense linear algebra)
- ~~MPI~~ (NOT NEEDED - single GPU only!)
- Build tools (CMake 3.20+, g++ 11+)

#### Option B: Native Installation
**Pros:**
- Direct hardware access
- ~~Simpler MPI configuration~~ (N/A - no MPI)
- Potentially lower latency

**Cons:**
- Dependency hell (ROCm versions, libraries)
- Less portable
- HotAisle may prefer containers for multi-tenancy

**Verdict:** Start with Docker for development, benchmark native if performance critical

**Single-GPU Simplification:** No MPI = much simpler deployment!

---

## 4. Compilation Strategy

### Build System: CMake
**Minimum CMakeLists.txt structure (Single-GPU, No MPI):**
```cmake
cmake_minimum_required(VERSION 3.20)
project(pdmrg_gpu LANGUAGES CXX HIP)

# Find ROCm/HIP
find_package(hip REQUIRED)
find_package(hipblas REQUIRED)
find_package(hiptensor REQUIRED)  # May need manual path
# NO MPI needed!

# Compiler flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_HIP_ARCHITECTURES gfx942)  # MI300X architecture

# Optimization flags
add_compile_options(
    -O3
    -march=native
    -ffast-math  # Use cautiously for numerics
    -DNDEBUG     # Release mode
)

# HIP-specific flags
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} --offload-arch=gfx942")

add_executable(pdmrg_gpu src/main.cpp src/dmrg_kernel.hip)
target_link_libraries(pdmrg_gpu hip::host hip::device hipblas hiptensor)
# Note: No MPI linking!
```

### Key Compilation Flags

#### For MI300X (gfx942 architecture):
```bash
-DCMAKE_HIP_ARCHITECTURES=gfx942
--offload-arch=gfx942
```

#### Optimization levels:
- `-O3` - Maximum optimization
- `-ffast-math` - Fast math (careful with DMRG convergence!)
- `-march=native` - CPU-side optimizations
- `-flto` - Link-time optimization (test carefully)

#### Debug vs Release:
```bash
# Debug build
cmake -DCMAKE_BUILD_TYPE=Debug -DHIP_DEBUG=ON ..

# Release build
cmake -DCMAKE_BUILD_TYPE=Release -DNDEBUG=ON ..
```

---

## 5. Code Architecture Plan

### Critical DMRG Operations to Port

Based on `pdmrg2_gpu.md` specification:

#### Phase 1: Core Tensor Operations (hipTensor)
1. **Two-site tensor contraction** (rank-3 × rank-3 → rank-4)
   - `H_eff = einsum('ijk,klm->ijlm', L, W)`
   - This is the bottleneck - ~60% of runtime in PDMRG

2. **MPS-MPO contraction**
   - `einsum('ab,bcd,de->ace', mps[i], mpo[i], mps[i+1])`

3. **Gauge propagation**
   - Large tensor reshaping and GEMM operations

#### Phase 2: Linear Algebra (rocBLAS/rocSOLVER)
1. **SVD for tensor decomposition**
   - `rocblas_dgesvd` or `rocsolver_dgesvd`
   - Critical for bond truncation

2. **QR factorization** (for gauge fixing)
   - `rocsolver_dgeqrf`

3. **Block Davidson eigensolver** (see pdmrg2_gpu.md §2.1)
   - Custom implementation using `rocblas_dgemm` (GEMM-heavy)

#### Phase 3: Stream-Based Parallelism (Single-GPU)
1. **HIP Streams for pipelining**
   - Create multiple streams for overlapping operations
   - Example: While one site is in eigensolver, prefetch next site's tensors

2. **Batch tensor operations**
   - Process multiple left/right contractions in parallel
   - Use hipTensor batch modes if available

3. **Async memory operations**
   - Overlap CPU-side bookkeeping with GPU computation
   - Only copy final results back to host

### Data Layout Strategy

**Key Decision: GPU-Resident Everything**

**Single Strategy: Keep everything on GPU** (STRONGLY RECOMMENDED)
- MI300X has 192GB HBM3 - problem fits easily!
- Store entire MPS, MPO, left/right environments on GPU
- No streaming needed (simplifies code enormously)
- Only copy final energy/state to CPU at end

**Memory estimate for L=40, D=100, complex128:**
- MPS: L × D² × 16 bytes ≈ 40 × 10,000 × 16 = 6.4 MB
- MPO: L × d² × D² × 16 bytes (d=5 for Josephson) ≈ 40 MB
- Environments: 2 × L × D² × 16 bytes ≈ 12.8 MB
- Working buffers (eigensolver, SVD): ~1 GB generous
- **Total: < 2 GB for huge problems!**

**For L=100, D=500, complex128:**
- Still only ~100 GB - fits comfortably in 192GB!

**Strategy:** Allocate all memory upfront, zero host-device transfers during sweeps

---

## 6. Development Workflow

### Phase 0: Environment Setup (Week 1)
- [ ] Provision HotAisle MI300X instance
- [ ] Verify ROCm version: `rocminfo` should show gfx942
- [ ] Test hipTensor installation: `hipcc --version`, check for libhiptensor.so
- [ ] Build hello-world HIP kernel with streams
- [ ] ~~Test MPI~~ (NOT NEEDED - single GPU only!)
- [ ] Build Docker image with all dependencies

### Phase 1: Single-GPU Prototype (Week 2-3)
- [ ] Implement Heisenberg MPO builder in C++
- [ ] Port single 2-site contraction to hipTensor
- [ ] Implement CPU-based eigensolver (scipy → Eigen or LAPACK wrapper)
- [ ] Run L=12 Heisenberg benchmark, verify correctness (ΔE < 1e-12)
- [ ] Profile with rocprof: identify hotspots

### Phase 2: Optimize Critical Path (Week 4-5)
- [ ] Replace eigensolver with Block Davidson (GEMM-heavy)
- [ ] Optimize tensor contractions (tuning hipTensor params)
- [ ] Implement GPU-resident SVD (rocSOLVER)
- [ ] Benchmark L=40 single-GPU: target >10x speedup vs CPU

### Phase 3: Stream-Based Optimization (Week 6-7)
- [ ] Implement HIP stream pipelining
- [ ] Batch tensor operations where possible
- [ ] Profile stream utilization with rocprof
- [ ] Optimize memory access patterns (coalescing)

### Phase 4: Production & Benchmarks (Week 8)
- [ ] Run Josephson junction benchmark (complex128)
- [ ] ~~Generate scaling plots~~ (single GPU - compare problem sizes instead)
- [ ] Compare vs PDMRG Python: target 50-100x speedup on single GPU
- [ ] Test large problems (L=100, D=500) that exploit 192GB memory
- [ ] Document deployment instructions

---

## 7. Docker Configuration

### Dockerfile Template
```dockerfile
FROM rocm/dev-ubuntu-22.04:6.0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    git \
    wget \
    libopenmpi-dev \
    openmpi-bin \
    rocm-dev \
    rocblas \
    rocsolver \
    hiptensor \
    && rm -rf /var/lib/apt/lists/*

# Set ROCm environment
ENV PATH="/opt/rocm/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH}"
ENV HIP_PLATFORM=amd

# Build PDMRG-GPU
WORKDIR /workspace
COPY . /workspace/pdmrg-gpu
RUN cd pdmrg-gpu && \
    mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && \
    make -j$(nproc)

# Entry point
CMD ["/bin/bash"]
```

### Docker Run Command (HotAisle)
```bash
docker run -it --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --security-opt seccomp=unconfined \
  -v $(pwd):/workspace \
  pdmrg-gpu:latest
```

### Single-GPU Launch (Simple!)
```bash
# Inside container - just run directly, no MPI!
./build/pdmrg_gpu \
  --model heisenberg \
  --length 40 \
  --bond-dim 100

# For Josephson junction
./build/pdmrg_gpu \
  --model josephson \
  --length 100 \
  --bond-dim 500 \
  --complex
```

---

## 8. Gotchas & Mitigation

### Issue 1: hipTensor Maturity
**Risk:** hipTensor may not support arbitrary tensor contractions
**Mitigation:**
- Test early with simple einsum: `'ijk,klm->ijlm'`
- Fallback: Manual einsum using rocBLAS GEMM + reshapes
- Reference: numpy.einsum logic can be translated to GEMM sequence

### Issue 2: Complex128 Support
**Risk:** Some AMD kernels optimized for FP32/FP64, not complex
**Mitigation:**
- Verify `hipblasZgemm` (complex double GEMM) performance
- May need to manually split real/imag parts
- Test Josephson benchmark early

### ~~Issue 3: ROCm-Aware MPI~~ (NOT APPLICABLE - Single GPU Only!)
**REMOVED:** No MPI needed, problem solved!

### Issue 4: Numerical Precision
**Risk:** GPU numerics differ from CPU (fused multiply-add, reduction order)
**Mitigation:**
- Use `--fno-fast-math` initially, enable only after validation
- Compare energy convergence vs CPU PDMRG (ΔE < 1e-10 tolerance)
- Test with exact L=12 Heisenberg reference

### Issue 5: Memory Bandwidth Limits
**Risk:** MI300X has massive bandwidth (8TB/s) but can still bottleneck
**Mitigation:**
- Profile with `rocprof --hip-trace`
- Optimize tensor layouts (row-major vs column-major)
- Use tensor core operations (MFMA instructions) where possible

---

## 9. Success Metrics

### Correctness (Required)
- [ ] Heisenberg L=12: |E - E_exact| < 1e-12
- [ ] Josephson L=20: |E_gpu - E_cpu| < 1e-10
- [ ] Reproducibility: 10 runs with same seed → identical results

### Performance (Target - Single GPU)
- [ ] Single MI300X > 50x faster than 48-core CPU (PDMRG Python) for L=40
- [ ] Single MI300X > 100x faster for L=100 (larger problems scale better on GPU)
- [ ] Demonstrate problems impossible on CPU: L=200, D=1000 (needs ~100GB)

### Deployment (Required)
- [ ] Docker image builds in <10 minutes
- [ ] Single command to run benchmark
- [ ] Clear documentation for HotAisle setup

---

## 10. Open Questions (Research Needed)

### HotAisle-Specific
1. What ROCm version is available? (Need 5.7+ for hipTensor)
2. ~~ROCm-aware MPI~~ (NOT NEEDED!)
3. ~~Network topology~~ (NOT NEEDED - single GPU!)
4. Persistent storage setup for Docker volumes?
5. Full MI300X allocation (192GB)?

### hipTensor API
1. Does hipTensor support einsum directly or only GEMM-like operations?
2. What's the optimal tensor layout (NCHW vs NHWC equivalent)?
3. Batch tensor contraction support?
4. Performance vs manual rocBLAS GEMM?
5. Stream support for async operations?
6. Workspace memory management (pre-allocate vs dynamic)?

### Numerical
1. Does Block Davidson converge faster on GPU than Lanczos on CPU?
2. What's the optimal block size for Block Davidson (pdmrg2_gpu.md suggests 3-5)?
3. Can we use FP32 for some operations without losing DMRG accuracy?

---

## 11. Recommended First Steps

### Before deploying Claude to HotAisle:

1. **Get HotAisle access and verify:**
   ```bash
   rocminfo | grep gfx942
   hipcc --version
   ls /opt/rocm/lib/libhiptensor.so
   mpirun --version
   ```

2. **Clone reference implementations:**
   ```bash
   git clone https://github.com/ROCmSoftwarePlatform/hipTensor
   cd hipTensor/samples
   mkdir build && cd build
   cmake .. && make
   ./contraction_sample  # Test basic operation
   ```

3. **Build minimal DMRG kernel:**
   - Single 2-site Heisenberg contraction
   - CPU eigensolver (Eigen library)
   - Compare result to Python PDMRG output

4. **Profile baseline:**
   ```bash
   rocprof --hip-trace ./minimal_dmrg
   # Analyze rocprof output for memory bandwidth, kernel launch overhead
   ```

5. **Document findings in `GPU_RESEARCH_NOTES.md`**
   - What works, what doesn't
   - Performance numbers
   - API quirks

---

## 12. Deliverables

### Code Repository Structure (Single-GPU)
```
pdmrg-gpu/
├── CMakeLists.txt
├── Dockerfile
├── README.md
├── src/
│   ├── main.cpp
│   ├── dmrg_kernel.hip         # GPU kernels
│   ├── tensor_ops.cpp          # hipTensor wrappers
│   ├── mpo_builders.cpp        # Heisenberg, Josephson MPOs
│   ├── eigensolver.cpp         # Block Davidson
│   ├── stream_manager.cpp      # HIP stream management
│   └── gpu_memory.cpp          # Device memory pool
├── include/
│   ├── dmrg_types.hpp          # Tensor, MPS, MPO types
│   ├── config.hpp              # Build configuration
│   └── stream_manager.hpp      # Stream utilities
├── benchmarks/
│   ├── heisenberg_L12.cpp
│   ├── heisenberg_L40.cpp
│   ├── heisenberg_L100.cpp     # Large problem demo
│   └── josephson_L20.cpp
└── scripts/
    ├── build_docker.sh
    └── run_benchmark.sh
```

### Documentation
1. **SETUP.md** - HotAisle deployment instructions
2. **PERFORMANCE.md** - Benchmark results, scaling plots
3. **API_NOTES.md** - hipTensor API quirks and gotchas
4. **COMPARISON.md** - GPU vs CPU PDMRG speedup analysis

---

## 13. Timeline Estimate

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Environment setup | 1 week | Docker image, hello-world GPU kernel with streams |
| Single-GPU prototype | 2 weeks | L=12 Heisenberg correctness |
| Optimization | 2 weeks | L=40 benchmark, 50x speedup |
| Stream pipelining | 1 week | Optimized async operations |
| Large problems | 1 week | L=100, D=500 demonstrations |
| Production & docs | 1 week | Final benchmarks, documentation |
| **Total** | **8 weeks** | Production-ready single-GPU DMRG |

---

## Next Steps

1. **Provision HotAisle MI300X instance** - get access, verify ROCm setup
2. **Run diagnostic script** (to be created) - check ROCm, hipTensor, MPI
3. **Deploy Claude to HotAisle** - point Claude at this game plan
4. **Start Phase 0** - environment setup and validation

---

## References for Claude on HotAisle

When you deploy Claude to work on this:
- Start with Phase 0 checklist
- Use this document as the master plan
- Document all findings in GPU_RESEARCH_NOTES.md
- Ask questions when hipTensor API is unclear
- Prioritize correctness over performance initially
- Profile early and often (rocprof is your friend)

**Key principle:** DMRG correctness is non-negotiable. Every optimization must be validated against reference results.

