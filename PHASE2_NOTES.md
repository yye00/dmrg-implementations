# Phase 2 Josephson CPU Benchmarks - Partial Results

## Completed Successfully ✓
- Quimb DMRG1 (reference): E = -2.843801043133
- Quimb DMRG2 (reference): E = -2.843801043139
- PDMRG np=1: E = -2.843801043139, time ~1.1s (validated ✓)
- PDMRG np=2: E = -2.843801043139, time ~1.2s (validated ✓)

## Performance Issue - Deferred for Investigation ⚠️
- PDMRG np=4: time ~310s per run (300x slower than np=1,2)
- PDMRG np=8: not tested (expected similar slowdown)
- PDMRG2 (np=4,8): not tested

**Issue:** Josephson (complex128, d=5) with MPI np>=4 shows massive slowdown
**Action:** Deferred - needs investigation
**Note:** Heisenberg PDMRG np=4 was fast (~2s), issue is specific to Josephson + higher np

## Decision
Skipped slow configurations and proceeded to GPU benchmarks to maintain progress.
