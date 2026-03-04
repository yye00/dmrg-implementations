# PDMRG GPU Port for AMD MI300X

This directory contains the game plan, configuration, and supporting files for porting PDMRG/PDMRG2 to AMD MI300X GPUs using hipTensor and C++.

## Quick Start on HotAisle

### 1. Run Diagnostics
```bash
./diagnostic.sh > hotaisle_diagnostic.txt
cat hotaisle_diagnostic.txt
```

This will check:
- AMD GPU detection (MI300X = gfx942)
- ROCm version and libraries
- hipTensor availability
- MPI configuration
- HIP compiler functionality

### 2. Review Game Plan
Read `../GPU_PORT_GAMEPLAN.md` for the complete development strategy, including:
- Technology decisions (hipTensor, C++, MPI)
- Docker vs native deployment analysis
- Phase-by-phase development plan (8 weeks)
- Code architecture and critical operations to port
- Success metrics and gotchas

### 3. Choose Deployment Method

#### Option A: Docker (Recommended for Development)
```bash
# Build image
cd ..
docker build -f gpu-port/Dockerfile.rocm -t pdmrg-gpu:rocm6 .

# Run with GPU access
docker run -it --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  -v $(pwd):/workspace \
  pdmrg-gpu:rocm6

# Inside container, run diagnostics
cd /workspace/gpu-port
./diagnostic.sh
```

#### Option B: Native Installation
```bash
# Ensure ROCm 6.x is installed
rocminfo

# Install dependencies
sudo apt-get install cmake libopenmpi-dev hipblas rocsolver hiptensor

# Build
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### 4. Test Minimal Build
```bash
# From build directory
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
./pdmrg_gpu --help  # Once implemented
```

## Files in This Directory

- **GPU_PORT_GAMEPLAN.md** - Comprehensive development plan (main document)
- **diagnostic.sh** - Environment validation script
- **CMakeLists.txt** - Starter CMake configuration for hipTensor + MPI
- **Dockerfile.rocm** - ROCm 6.0 development container
- **README.md** - This file

## Development Workflow

Follow the phases in `GPU_PORT_GAMEPLAN.md`:

1. **Phase 0** (Week 1): Environment setup ← START HERE
   - Run diagnostics
   - Build Docker image
   - Test hello-world HIP kernel
   - Verify MPI multi-GPU support

2. **Phase 1** (Week 2-3): Single-GPU prototype
   - Implement Heisenberg MPO in C++
   - Port 2-site tensor contraction to hipTensor
   - Verify L=12 correctness (ΔE < 1e-12)

3. **Phase 2** (Week 4-5): Optimize critical path
   - Block Davidson eigensolver
   - GPU-resident SVD
   - Profile and optimize

4. **Phase 3** (Week 6-7): Multi-GPU with MPI
   - ROCm-aware MPI integration
   - Domain decomposition
   - Scaling tests (2/4/8 GPUs)

5. **Phase 4** (Week 8): Production benchmarks
   - Josephson junction (complex128)
   - Scaling plots
   - Documentation

## Critical Decisions to Make

Before starting, verify with HotAisle:

1. **ROCm Version**: Need 5.7+ for hipTensor (6.0+ preferred)
2. **MPI GPU Support**: Is OpenMPI ROCm-aware? Check with diagnostic.sh
3. **Network Topology**: How are MI300X GPUs connected? (Affects multi-GPU performance)
4. **Persistence**: Can we use Docker volumes or need native install?

## Next Steps

1. **Provision HotAisle MI300X instance**
2. **Run `./diagnostic.sh` and save output**
3. **Review GPU_PORT_GAMEPLAN.md sections 1-6** (decisions and architecture)
4. **Deploy Claude to HotAisle** with this gameplan as context
5. **Start Phase 0 checklist**

## Resources for Claude

When Claude is deployed to HotAisle:
- Use `GPU_PORT_GAMEPLAN.md` as the master plan
- Document findings in `GPU_RESEARCH_NOTES.md` (create as needed)
- Prioritize correctness over speed initially
- Profile early with `rocprof`
- Validate every optimization against CPU PDMRG results

## Questions?

Refer to the "Open Questions" section in `GPU_PORT_GAMEPLAN.md` or open an issue in the main repository.

---

**Key Principle:** DMRG correctness is non-negotiable. Machine precision (ΔE < 1e-12) must be maintained.
