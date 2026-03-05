# Phase 2 Checkpoint: Multi-Stream Domain Decomposition

**Date**: 2026-03-05
**Status**: Architecture Complete, Implementation 80% Done
**GPU System**: AMD MI300X (enc1-gpuvm015 via tmux session `test_remote`)

---

## Executive Summary

Phase 2 implements domain-decomposed parallel DMRG on GPU using streams (analogous to CPU PDMRG's MPI ranks). The complete architecture is in place and tested on MI300X. Core algorithms work correctly; remaining work is filling in sweep implementation details.

### What Works ✅

1. **Phase 1** (Foundation):
   - `AccurateSVD_GPU`: Exact SVD, reconstruction error 5.57e-16 ✓
   - `OptimizedHeff`: hipTensor contractions, identity test passes ✓

2. **Phase 2 - BoundaryMergeGPU** (V = Lambda^-1 Reconciliation):
   - ✅ Form theta = psi_left . diag(V) . psi_right
   - ✅ Lanczos eigensolver (30 iterations, convergence checking)
   - ✅ Exact SVD split using AccurateSVD_GPU
   - ✅ V = 1/S computation with regularization (1e-12)
   - ✅ Memory reallocation for dynamic bond dimensions
   - **Test Result**: PASS on MI300X (V values: 7.57 to 441.17)

3. **Phase 2 - StreamSegment** (Segment Management):
   - ✅ Memory management (MPS, environments, MPO, boundaries)
   - ✅ Site range ownership [start, end]
   - ✅ BoundaryData structures
   - **Test Result**: PASS on MI300X

4. **Phase 2 - StreamCoordinator** (Orchestration):
   - ✅ Site distribution (load-balanced)
   - ✅ Multi-stream creation
   - ✅ Even/odd boundary merge pattern
   - ✅ Sweep/merge cycle orchestration
   - **Test Result**: Compiles and runs, encounters NaN due to placeholder sweeps

### What Needs Implementation ⚠️

**Estimated Time**: 2-3 hours focused work

1. **QR/LQ Decompositions** (~200 lines):
   - `StreamSegment::sweep_left_to_right()`: QR factorization
   - `StreamSegment::sweep_right_to_left()`: LQ factorization
   - Use rocSOLVER `rocsolver_dgeqrf` and `rocsolver_dgelqf`

2. **Boundary Tensor Extraction** (~100 lines):
   - `StreamSegment::extract_boundary_tensors()`: Copy edge MPS tensors to BoundaryData
   - Update after sweeps and merges

3. **Environment Contractions** (~300 lines):
   - `StreamSegment::rebuild_right_boundary_env()`: Contract MPS + MPO for L_env
   - `StreamSegment::rebuild_left_boundary_env()`: Contract MPS + MPO for R_env

---

## Remote GPU System Access

### System Information
- **Hostname**: `enc1-gpuvm015` (HotAisle cluster)
- **GPU**: AMD Instinct MI300X VF (gfx942)
- **ROCm**: 7.2.0
- **Access**: Via tmux session `test_remote`

### Working with tmux Session

```bash
# Send command to remote session:
tmux send-keys -t test_remote "command here" Enter

# Wait and capture output:
sleep 3
tmux capture-pane -t test_remote -p | tail -20
```

---

## Building and Testing on MI300X

### Quick Start

```bash
# 1. Pull latest code
tmux send-keys -t test_remote "cd ~/dmrg-implementations && git pull" Enter
sleep 2

# 2. Configure and build
tmux send-keys -t test_remote "cd gpu-port/build && cmake .. -DGPU_TARGETS=gfx942 && make -j8" Enter
sleep 15

# 3. Run tests
tmux send-keys -t test_remote "./test_phase1" Enter
tmux send-keys -t test_remote "./test_boundary_merge" Enter
```

### Expected Test Results

**test_phase1**: ✅ PASS (error 5.57e-16)
**test_stream_segment**: ✅ PASS
**test_boundary_merge**: ✅ PASS (V = 7.57 to 441.17)
**test_stream_coordinator**: ⚠️ NaN (expected - sweeps are placeholders)

---

## Next Steps

### Priority 1: Implement QR/LQ Sweeps
File: `gpu-port/src/stream_segment.cpp`
Use: `rocsolver_dgeqrf` and `rocsolver_dgelqf`

### Priority 2: Boundary Extraction
Implement `extract_boundary_tensors()` to copy MPS edges to BoundaryData

### Priority 3: Environment Contractions
Implement `rebuild_*_boundary_env()` methods

---

## Quick Reference

```bash
# Pull and rebuild
tmux send-keys -t test_remote "cd ~/dmrg-implementations && git pull && cd gpu-port/build && make -j8" Enter

# Run all tests
tmux send-keys -t test_remote "./test_phase1 && ./test_boundary_merge" Enter

# Check output
tmux capture-pane -t test_remote -p | tail -20
```
