# HEARTBEAT.md

## Heisenberg Benchmark - Complete ✅

### Test Results (2025-02-19)
All 10 tests PASS with machine precision accuracy

| Method | Energy | ΔE | Time | Status |
|--------|--------|-----|------|--------|
| quimb DMRG1 | -5.142090628199386 | -2.12e-11 | 0.14s | ✓ |
| quimb DMRG2 | -5.142090628178156 | [REF] | 0.09s | ✓ |
| PDMRG np=1 | -5.142090628178245 | -8.88e-14 | 0.17s | ✓ |
| PDMRG np=2 | -5.142090628178137 | +1.95e-14 | 0.11s | ✓ |
| PDMRG np=4 | -5.142090628178135 | +2.13e-14 | 0.12s | ✓ |
| PDMRG np=8 | -5.142090628178137 | +1.95e-14 | 0.12s | ✓ |
| A2DMRG np=1 | -5.142090628177471 | +6.85e-13 | 17.98s | ✓ |
| A2DMRG np=2 | -5.142090628178101 | +5.51e-14 | 29.96s | ✓ |
| A2DMRG np=4 | -5.142090628177506 | +6.50e-13 | 15.16s | ✓ |
| A2DMRG np=8 | -5.142090628175779 | +2.38e-12 | 20.13s | ✓ |

### Observations
1. **Accuracy**: All errors < 1e-10 (machine precision) ✅
2. **PDMRG timing**: ~0.1s for all np (overhead dominates at L=12)
3. **A2DMRG timing**: 15-30s (cotengra tensor contractions)
4. **A2DMRG scaling**: Non-monotonic (np=2 slowest)
5. **DMRG1 vs DMRG2**: Two-site DMRG2 more accurate

### Configuration
- L=12, bond_dim=20
- max_sweeps=30, tol=1e-12, cutoff=1e-14

### Next Steps
1. ✅ Heisenberg benchmark complete
2. → Josephson junction benchmark
3. → Larger system scaling tests

---

## Heisenberg Benchmark - Post-Approach-A Improvements ✅

### Test Results (2026-02-22)
All 10 tests PASS with machine precision accuracy. Streaming decompositions
and `coarse_reduction_tol` exposure confirmed to not regress accuracy.

| Method | Energy | ΔE | Time | Status |
|--------|--------|-----|------|--------|
| quimb DMRG1 | -5.142090628195758 | -1.76e-11 | 0.12s | ✓ |
| quimb DMRG2 | -5.142090628178138 | [REF] | 0.07s | ✓ |
| PDMRG np=1 | -5.142090628178230 | -9.24e-14 | 0.15s | ✓ |
| PDMRG np=2 | -5.142090628178131 | +6.22e-15 | 0.12s | ✓ |
| PDMRG np=4 | -5.142090628178136 | +1.78e-15 | 0.11s | ✓ |
| PDMRG np=8 | -5.142090628178130 | +7.99e-15 | 0.13s | ✓ |
| A2DMRG np=1 | -5.142090628178138 | 0.00e+00 | 5.90s | ✓ |
| A2DMRG np=2 | -5.142090628178134 | +3.55e-15 | 3.12s | ✓ |
| A2DMRG np=4 | -5.142090628178135 | +2.66e-15 | 1.93s | ✓ |
| A2DMRG np=8 | -5.142090628178134 | +3.55e-15 | 1.43s | ✓ |

### Speedup vs PDMRG np=1 (t=0.15s)

| Method | Time | Speedup vs PDMRG np=1 |
|--------|------|----------------------|
| quimb DMRG2 | 0.07s | 2.10x |
| PDMRG np=4 | 0.11s | 1.36x |
| PDMRG np=2 | 0.12s | 1.34x |
| quimb DMRG1 | 0.12s | 1.32x |
| PDMRG np=8 | 0.13s | 1.19x |
| PDMRG np=1 | 0.15s | 1.00x (ref) |
| A2DMRG np=8 | 1.43s | 0.11x |
| A2DMRG np=4 | 1.93s | 0.08x |
| A2DMRG np=2 | 3.12s | 0.05x |
| A2DMRG np=1 | 5.90s | 0.03x |

### A2DMRG Internal Scaling
| np | Time | Speedup | Efficiency |
|----|------|---------|-----------|
| 1 | 5.90s | 1.00x | 100.0% |
| 2 | 3.12s | 1.89x | 94.6% |
| 4 | 1.93s | 3.05x | 76.4% |
| 8 | 1.43s | 4.12x | 51.5% |

### Observations
1. **Accuracy**: All errors << 1e-10 (well within threshold) ✅
2. **A2DMRG accuracy improved**: np=1 now hits 0.00e+00 error (exact match to DMRG2 float). Previous run had errors up to 6.85e-13.
3. **A2DMRG timing improved**: np=1 dropped from 17.98s to 5.90s (-67%); np=2 from 29.96s to 3.12s (-90%); np=4 from 15.16s to 1.93s (-87%).
4. **A2DMRG scaling**: Near-linear scaling 1→2 (94.6% efficiency), good scaling to np=4 (76.4%).
5. **PDMRG**: Overhead-dominated at L=12; all np complete in ~0.11-0.15s.
6. **Note**: At L=12 A2DMRG is slower than PDMRG in wall time — this is expected as cotengra contraction overhead dominates at small system size. Advantage expected at larger L.

### Configuration
- L=12, bond_dim=20, max_sweeps=30, tol=1e-10, cutoff=1e-14
- Approach-A streaming decompositions active
