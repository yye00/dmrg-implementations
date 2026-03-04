# GPU Port Executive Summary

**Target:** Single AMD MI300X GPU via HotAisle
**Goal:** 50-100x speedup over 48-core CPU DMRG
**Timeline:** 6-8 weeks
**Date:** 2026-03-03

---

## What We're Building

**Port PDMRG/PDMRG2 to AMD MI300X GPU using:**
- C++ for full control
- hipTensor (or rocBLAS fallback) for tensor contractions
- rocSOLVER for SVD decomposition
- Custom Lanczos eigensolver
- HIP streams for async pipelining
- **Single GPU only** - NO MPI (problem fits in 192GB)

---

## Documentation Suite (Read in Order)

### 1. **CONFIDENCE_ANALYSIS.md** (START HERE)
- **Purpose:** Honest assessment of risks and unknowns
- **Confidence ratings:** 🟢 High / 🟡 Medium / 🔴 Low / ⚫ Unknown
- **Key findings:**
  - ✅ rocBLAS/rocSOLVER: 🟢 HIGH confidence (will work)
  - ⚠️ hipTensor availability: 🔴 LOW confidence (may not be installed)
  - ⚠️ Sparse eigensolvers: MUST implement ourselves
  - ✅ Fallback path: 🟢 HIGH confidence (rocBLAS always works)

**Action:** Read sections 1-7, 14-16 for critical gaps

### 2. **RESEARCH_CHECKLIST.md** (RUN ON DAY 1)
- **Purpose:** Systematic discovery of unknowns
- **Contains:** Executable commands and test programs
- **Validates:**
  - ROCm version (need ≥5.7)
  - hipTensor availability
  - rocSOLVER capabilities
  - GPU architecture (gfx942)
  - Memory (192GB)

**Action:** Run all checks, fill in summary template

### 3. **GPU_PORT_GAMEPLAN.md** (MASTER PLAN)
- **Purpose:** Complete 8-week development roadmap
- **Contains:** 13 sections covering every aspect
- **Highlights:**
  - Phase-by-phase development plan
  - Code architecture
  - Compilation strategy
  - Performance targets

**Action:** Use as reference during development

### 4. **SINGLE_GPU_STRATEGY.md** (ARCHITECTURE)
- **Purpose:** Justification for single-GPU approach
- **Contains:**
  - Memory analysis (why 192GB >> DMRG needs)
  - HIP streams parallelism strategy
  - Performance estimates
  - Code structure examples

**Action:** Read sections on streams and memory layout

### 5. **This File (EXECUTIVE_SUMMARY.md)**
- Quick reference and decision matrix

---

## Critical Unknowns (Verify Day 1)

| Unknown | Impact | How to Validate | Fallback |
|---------|--------|-----------------|----------|
| **ROCm version** | ⚫ BLOCKER | `rocminfo \| grep Runtime` | Request upgrade |
| **hipTensor availability** | 🔴 HIGH | See RESEARCH_CHECKLIST | Use rocBLAS |
| **MI300X (gfx942)** | ⚫ BLOCKER | `rocminfo \| grep gfx` | Cannot proceed |
| **192GB memory** | 🟡 MEDIUM | `rocm-smi --showmeminfo` | Use smaller problems |
| **Sudo access** | 🟡 MEDIUM | `sudo -n true` | Use Docker or request |

---

## Decision Matrix

### Scenario 1: Everything Works (Optimistic)
```
✓ ROCm 6.0+
✓ hipTensor installed
✓ MI300X with 192GB
✓ Sudo access
```
**Action:** Full steam ahead with hipTensor implementation
**Timeline:** 8 weeks
**Expected:** 50-100x speedup
**Probability:** 🟡 40%

### Scenario 2: hipTensor Missing (Likely)
```
✓ ROCm 6.0+
✗ hipTensor not installed
✓ MI300X with 192GB
? Sudo access varies
```
**Action:**
1. Try: `sudo apt-get install hiptensor`
2. If fails: Use rocBLAS fallback
3. Implement tensor contractions as GEMM sequences

**Timeline:** 8 weeks (same)
**Expected:** 40-80x speedup (10% slower than hipTensor)
**Probability:** 🟢 50%

### Scenario 3: Old ROCm (Problem)
```
✗ ROCm <5.7
? Everything else unknown
```
**Action:**
1. Request ROCm upgrade
2. If impossible: Build hipTensor from source (risky)
3. If that fails: Reconsider GPU approach

**Timeline:** +2 weeks for upgrade process
**Probability:** 🔴 10%

### Scenario 4: Wrong Hardware (Blocker)
```
✗ Not MI300X (different GPU)
✗ Not gfx942
✗ <100GB memory
```
**Action:** STOP - GPU port not feasible
**Alternative:** Optimize CPU version or request different instance
**Probability:** 🟢 <5%

---

## Guaranteed Working Path (Worst Case)

Even if EVERYTHING goes wrong except basic ROCm:

### Minimal Requirements:
- Any AMD GPU with ROCm 5.0+
- rocBLAS (always included)
- rocSOLVER (always included)
- hipcc compiler

### Implementation:
```cpp
Stack:
├── rocBLAS GEMM for all tensor operations
├── rocSOLVER SVD for bond truncation
├── Custom Lanczos eigensolver (we write)
└── No hipTensor, no streams (simple serial)
```

### Expected Performance:
- **Speedup:** 10-25x (conservative)
- **Works for:** L=40, D=100 easily
- **Limitations:** Won't enable L=200+ problems

### Confidence: 🟢 85%

This is the **fallback floor** - we can always achieve this.

---

## Development Phases (8 Weeks)

### Week 1: Validation & Setup
- **Day 1:** Run RESEARCH_CHECKLIST.md, make decisions
- **Day 2-3:** Set up build system, compile hello-world
- **Day 4-5:** Test rocBLAS GEMM, rocSOLVER SVD

**Deliverable:** Environment validated, basic HIP kernels compile

### Week 2-3: MVP (Minimum Viable Product)
- Heisenberg MPO in C++
- Tensor contractions (hipTensor OR rocBLAS)
- Basic eigensolver
- Full DMRG sweep for L=12

**Deliverable:** L=12 Heisenberg with ΔE < 1e-12

### Week 4-5: Optimization
- Custom Lanczos eigensolver
- Optimize tensor contractions
- GPU-resident SVD
- L=40 benchmarks

**Deliverable:** 25-40x speedup on L=40

### Week 6-7: Advanced Features
- HIP stream pipelining (if beneficial)
- Complex128 Josephson junction
- Profile and optimize hotspots

**Deliverable:** 50-100x speedup (if possible)

### Week 8: Polish & Documentation
- Large problem demos (L=100+)
- Complete documentation
- Deployment instructions

**Deliverable:** Production-ready GPU DMRG

---

## Success Criteria (Realistic)

### MVP Success (Achievable, Week 3)
- ✅ L=12 Heisenberg correct to machine precision
- ✅ Any measurable speedup vs CPU
- ✅ Code compiles and runs without crashes

**Probability: 🟢 80%**

### Good Success (Target, Week 6)
- ✅ L=40 with 25x speedup
- ✅ Complex128 working (Josephson junction)
- ✅ Documented and reproducible

**Probability: 🟡 60%**

### Excellent Success (Stretch Goal, Week 8)
- ✅ L=100, D=500 demonstration
- ✅ 50-100x speedup on L=40
- ✅ Stream-optimized, production-ready

**Probability: 🟡 40%**

---

## Key Technical Decisions

### 1. Single GPU vs Multi-GPU
**Decision:** Single MI300X
**Rationale:**
- 192GB >> DMRG memory needs (~10-100GB)
- 0% communication overhead
- Simpler code, faster development
- Same or better performance than 4-GPU MPI

**Confidence: 🟢 100% - This is the right call**

### 2. hipTensor vs rocBLAS
**Decision:** TBD after Day 1 validation
**Default:** Try hipTensor first
**Fallback:** rocBLAS GEMM (always works)

**Impact:** 10% performance difference

### 3. Eigensolver Strategy
**Decision:** Implement Lanczos ourselves
**Rationale:**
- No AMD sparse eigensolver exists
- Standard DMRG approach (apply H_eff via contractions)
- rocBLAS provides all building blocks

**Confidence: 🟢 HIGH - This is required work**

### 4. Docker vs Native
**Decision:** Start with Docker, benchmark native if needed
**Rationale:**
- Reproducible environment
- Easier dependency management
- Can switch later

---

## Files Ready on HotAisle

```
dmrg-implementations/gpu-port/
├── CONFIDENCE_ANALYSIS.md    # Read first - honest risk assessment
├── RESEARCH_CHECKLIST.md     # Run on Day 1 - validates unknowns
├── GPU_PORT_GAMEPLAN.md      # Master plan - 13-section roadmap
├── SINGLE_GPU_STRATEGY.md    # Architecture - why single GPU wins
├── EXECUTIVE_SUMMARY.md      # This file - quick reference
│
├── diagnostic.sh             # Automated environment check
├── CMakeLists.txt            # Build configuration (hipTensor optional)
├── Dockerfile.rocm           # ROCm 6.0 container
│
└── examples/
    └── hello_streams.hip     # Test HIP streams
```

---

## Day 1 Action Plan (When You Get Access)

### Morning (2 hours)
```bash
# 1. Clone repository
git clone https://github.com/yye00/dmrg-implementations.git
cd dmrg-implementations/gpu-port

# 2. Run automated diagnostic
./diagnostic.sh > day1_diagnostic.txt

# 3. Run manual research checklist
# Follow RESEARCH_CHECKLIST.md step-by-step
# Fill in summary template
```

### Afternoon (4 hours)
```bash
# 4. Make critical decisions
# - hipTensor: available / install / fallback?
# - ROCm version: acceptable / need upgrade?
# - Architecture: confirmed gfx942?

# 5. Test compilation
cd examples
hipcc -o hello_streams hello_streams.hip --offload-arch=gfx942
./hello_streams

# 6. Test basic math libraries
# Compile and run test programs from RESEARCH_CHECKLIST:
# - test_rocsolver_svd.cpp
# - test_large_allocation.cpp
# - test_hiptensor_basic.cpp (if available)
```

### Evening (2 hours)
```bash
# 7. Document findings
# - Update day1_diagnostic.txt with all results
# - Fill in RESEARCH_CHECKLIST summary
# - Make go/no-go decision

# 8. Set up development environment
# - Build Docker image OR configure native
# - Clone additional dependencies if needed
# - Create initial CMake build directory
```

### Day 1 Deliverable:
```
Complete validation report with:
- ✓/✗ for each critical component
- Decisions made (hipTensor vs rocBLAS, etc.)
- Confidence update
- GO / NO-GO / CONDITIONAL decision
```

---

## What Could Cause Failure?

### Technical Blockers (Can We Work Around?)
1. **ROCm <5.0** - 🔴 BLOCKER - Request upgrade
2. **Not AMD GPU** - 🔴 BLOCKER - Wrong hardware
3. **<50GB memory** - 🟡 LIMITING - Reduce problem size
4. **No rocBLAS/rocSOLVER** - 🔴 BLOCKER - Very unlikely

### Development Risks (Can We Mitigate?)
1. **hipTensor missing** - ✅ Use rocBLAS fallback
2. **Poor performance** - ✅ Profile and optimize
3. **Numerical issues** - ✅ Disable fast-math, validate
4. **Complex128 problems** - ✅ Test early, debug

### Timeline Risks (Can We Adjust?)
1. **Learning curve** - ✅ Start simple, iterate
2. **Debugging GPU code** - ✅ Use rocgdb, printf debugging
3. **Unexpected bugs** - ✅ Budget extra time

**Overall Failure Probability: 🟢 <15%**

---

## Communication Protocol

### Daily Updates (During Development)
- What worked today
- What's blocked
- What's next

### Weekly Milestones
- Week 1: Environment validated
- Week 2: L=12 working
- Week 3: L=40 benchmark
- Week 4-8: Optimize, document

### Red Flags (Escalate Immediately)
- Cannot compile basic HIP code
- ROCm version incompatible
- Wrong GPU hardware
- Performance <5x vs CPU (investigate why)

---

## Final Recommendation

### Most Likely Path Forward:

1. **HotAisle has:** MI300X, ROCm 6.0, no hipTensor pre-installed
2. **We install:** hipTensor via apt-get OR use rocBLAS fallback
3. **We implement:** Lanczos eigensolver, tensor contractions, DMRG sweep
4. **We achieve:** 40-60x speedup on L=40, enable L=100+ problems
5. **Timeline:** 8 weeks

### Confidence in Success:
- **MVP (L=12 working):** 🟢 80%
- **Good (25x speedup):** 🟡 60%
- **Excellent (50-100x):** 🟡 40%

### Recommendation:
**PROCEED** with GPU port. Even conservative estimates (25x speedup) justify the effort, and we have multiple fallback options if optimal path fails.

---

## References

- AMD ROCm Documentation: https://rocmdocs.amd.com/
- hipTensor GitHub: https://github.com/ROCmSoftwarePlatform/hipTensor
- rocSOLVER API: https://rocm.docs.amd.com/projects/rocSOLVER/
- Our CPU PDMRG: `/home/captain/clawd/work/dmrg-implementations/pdmrg/`

---

## Contact / Questions

When Claude is deployed to HotAisle:
1. Read CONFIDENCE_ANALYSIS.md (sections 1-7, 14-16)
2. Run RESEARCH_CHECKLIST.md (all tests)
3. Reference GPU_PORT_GAMEPLAN.md (as needed)
4. Document findings and decisions
5. Begin Week 1 development

**Key Principle:** Validate early, iterate quickly, fall back gracefully.
