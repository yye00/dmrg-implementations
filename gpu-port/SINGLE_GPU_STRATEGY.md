# Single-GPU Strategy for MI300X

## Key Decision: NO MPI, Everything on One GPU

### Why Single-GPU Works Perfectly

**MI300X Specs:**
- **192GB HBM3** unified memory
- **8 TB/s** memory bandwidth
- **325 TFLOPS FP64** (relevant for DMRG)

**Typical DMRG Memory Requirements:**

| Problem Size | MPS | MPO | Environments | Buffers | Total |
|--------------|-----|-----|--------------|---------|-------|
| L=40, D=100 | 6 MB | 40 MB | 13 MB | 1 GB | **~1 GB** |
| L=100, D=500 | 400 MB | 2.5 GB | 1.6 GB | 5 GB | **~10 GB** |
| L=200, D=1000 | 6.4 GB | 80 GB | 51 GB | 20 GB | **~150 GB** |

**Conclusion:** Even massive problems (L=200, D=1000) fit in 192GB with room to spare!

---

## Parallelism Strategy: HIP Streams, Not MPI

### HIP Streams for Async Execution

Instead of distributing across GPUs (MPI approach), we use:

1. **Multiple HIP Streams** for concurrent operations
   ```cpp
   hipStream_t stream_left, stream_right, stream_solver;

   // Launch concurrent operations
   contract_left<<<..., stream_left>>>(...)   // Prepare left environment
   contract_right<<<..., stream_right>>>(...) // Prepare right environment
   eigensolver<<<..., stream_solver>>>(...)   // Solve on previous site
   ```

2. **Pipelined DMRG Sweeps**
   ```
   Site i:   [ Eigensolver running    ]
   Site i+1:              [ Tensor prep ]
   Site i+2:                           [ Memory prefetch ]

   Timeline: |-------|-------|-------|-------|
             Overlap = Higher throughput!
   ```

3. **Kernel-Level Parallelism**
   - Each tensor contraction uses ALL 304 compute units
   - 65,536 threads per kernel launch (typical)
   - MI300X saturated with work = maximum performance

### No MPI Benefits

✅ **Simpler code** - No domain decomposition logic
✅ **Easier debugging** - Single address space
✅ **No communication** - Zero inter-GPU overhead
✅ **Faster development** - Skip ROCm-aware MPI complexity
✅ **Better profiling** - rocprof shows everything clearly

---

## Memory Management Strategy

### Pre-Allocate Everything

```cpp
// Allocate all memory at startup (NOT during sweeps!)
struct DMRGWorkspace {
    complex<double>* mps;           // L × D²
    complex<double>* mpo;           // L × d² × D²
    complex<double>* left_env;      // L × D²
    complex<double>* right_env;     // L × D²
    complex<double>* work_buffer;   // Eigensolver, SVD scratch

    // HIP streams for async ops
    hipStream_t stream_contract;
    hipStream_t stream_solve;
    hipStream_t stream_update;
};

void initialize_gpu_workspace(DMRGWorkspace& ws, int L, int D, int d) {
    size_t mps_size = L * D * D * sizeof(complex<double>);
    size_t mpo_size = L * d * d * D * D * sizeof(complex<double>);
    // ... allocate everything upfront

    hipMalloc(&ws.mps, mps_size);
    hipMalloc(&ws.mpo, mpo_size);
    // ... etc

    // Create streams
    hipStreamCreate(&ws.stream_contract);
    hipStreamCreate(&ws.stream_solve);
    hipStreamCreate(&ws.stream_update);
}
```

### Zero Host-Device Transfers During Sweeps

**Traditional CPU-GPU approach:**
```
For each site:
    1. Copy tensors CPU → GPU (SLOW!)
    2. Compute on GPU (fast)
    3. Copy result GPU → CPU (SLOW!)
    4. Repeat...
```

**Our approach:**
```
Startup:
    1. Copy MPO CPU → GPU (once)
    2. Initialize MPS on GPU (random or warmup)

DMRG Sweeps:
    - Everything stays on GPU
    - No host-device transfers!

Completion:
    1. Copy final energy GPU → CPU
    2. Optionally copy final MPS for analysis
```

**Speedup from this alone:** 10-20x for small problems!

---

## Optimization Opportunities (Beyond MPI)

### 1. Tensor Fusion
Combine multiple small tensor operations into one kernel:
```cpp
// Instead of 3 separate launches:
contract_left();
contract_right();
merge();

// Fuse into one:
contract_and_merge();  // One kernel, less overhead
```

### 2. Batch Operations
Use hipTensor batch modes for multiple sites:
```cpp
// Process 4 sites in parallel (if independent)
hipTensorBatchContract(
    tensors,      // Array of 4 tensor pairs
    batch_size=4,
    stream
);
```

### 3. Persistent Kernels
Keep kernels resident for the entire sweep:
```cpp
// Launch once, process all sites
persistent_dmrg_kernel<<<blocks, threads>>>(
    all_sites,  // Process L sites without relaunching
    L
);
```

### 4. Shared Memory for Small Tensors
For bond dimension D < 32, use shared memory:
```cpp
__global__ void small_contraction(...) {
    __shared__ complex<double> local_tensor[32*32];
    // Much faster than global memory access
}
```

---

## Performance Targets

### vs CPU PDMRG (48 cores)

| Problem | CPU Time | GPU Target | Speedup |
|---------|----------|------------|---------|
| L=12, D=50 | 1 min | 1 sec | 60x |
| L=40, D=100 | 30 min | 30 sec | 60x |
| L=100, D=500 | 10 hours | 6 min | 100x |
| L=200, D=1000 | Impossible (OOM) | 1 hour | ∞ (enables new science!) |

### Bottleneck Analysis

**CPU PDMRG bottlenecks:**
1. Memory bandwidth (40 GB/s typical)
2. Cache misses on large tensors
3. BLAS not optimized for tensor shapes

**MI300X advantages:**
1. **8 TB/s** memory bandwidth (200x faster!)
2. 192GB on-package (no cache misses)
3. hipTensor optimized for tensor contractions

**Expected:** 50-100x speedup is CONSERVATIVE

---

## Development Checklist

### Phase 1: Basic Single-GPU Implementation
- [ ] Allocate all memory on GPU
- [ ] Implement 2-site tensor contraction (hipTensor)
- [ ] Implement GPU-resident eigensolver (rocSOLVER)
- [ ] Implement SVD decomposition (rocSOLVER)
- [ ] Verify L=12 Heisenberg (ΔE < 1e-12)

### Phase 2: Stream Optimization
- [ ] Create 3 HIP streams (contract, solve, update)
- [ ] Pipeline operations across sites
- [ ] Profile with rocprof --hip-trace
- [ ] Eliminate all hipDeviceSynchronize() except at sweep end

### Phase 3: Advanced Optimizations
- [ ] Tensor fusion where beneficial
- [ ] Batch operations for multiple sites
- [ ] Shared memory for small D
- [ ] Custom kernels for critical paths

### Phase 4: Large-Scale Validation
- [ ] L=100, D=500 benchmark
- [ ] L=200, D=1000 demonstration
- [ ] Complex128 Josephson junction
- [ ] Scaling study vs problem size

---

## Code Structure (Single-GPU)

```cpp
// Main DMRG loop - entire sweep on GPU
void dmrg_sweep_gpu(DMRGWorkspace& ws, int L, int D) {
    // Left-to-right sweep
    for (int site = 0; site < L-1; site++) {
        // Async tensor prep for next site (stream 1)
        if (site < L-2) {
            prepare_tensors<<<..., ws.stream_contract>>>(site+1);
        }

        // Solve current site (stream 2)
        block_davidson<<<..., ws.stream_solve>>>(
            ws.left_env[site],
            ws.right_env[site],
            ws.mpo[site],
            ws.mps[site]
        );

        // Update environments (stream 3)
        update_environments<<<..., ws.stream_update>>>(site);

        // Only sync at critical points, not every iteration!
    }

    // Sync all streams at end of sweep
    hipDeviceSynchronize();
}
```

---

## Why This Is Better Than Multi-GPU

**Multi-GPU with MPI would require:**
- Domain decomposition (complex)
- Boundary tensor communication (slow)
- Load balancing (tricky)
- Debugging across processes (hard)
- ROCm-aware MPI setup (fragile)

**Efficiency loss:**
- Communication overhead: 10-30%
- Load imbalance: 5-15%
- Synchronization: 5-10%

**Net scaling:** 2 GPUs = 1.5-1.7x speedup (not 2x!)

**Our approach:**
- 1 GPU, 0% communication overhead
- 100% of GPU utilized
- Simpler code, faster development

**For DMRG:** Single MI300X is the sweet spot!

---

## Next Steps on HotAisle

1. **Run diagnostic.sh** - Verify ROCm, hipTensor, streams
2. **Compile hello_streams.hip** - Test async execution
3. **Start with single tensor contraction** - Get hipTensor working
4. **Add rocSOLVER eigensolver** - Complete basic DMRG cycle
5. **Optimize with streams** - Pipeline for performance

**Timeline:** 6-8 weeks to production-ready single-GPU DMRG

**Outcome:** 50-100x faster than CPU, enables L=200+ DMRG simulations!
