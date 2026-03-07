# Stable Merge Fix Attempt: Failure Report

**Date**: 2026-03-06
**Status**: ❌ **FAILED VALIDATION** - Implementation is mathematically incorrect

---

## Executive Summary

An attempt was made to fix PDMRG's boundary merge numerical instability by replacing `V = Lambda^-1` with `Lambda` directly. **This approach is fundamentally wrong** and produces completely incorrect results.

**Validation Results**:
- Energy difference vs correct merge: **9.860** (order of magnitude error)
- Tensor norm difference: **O(10^4)** (massive)
- Verdict: **DO NOT USE**

---

## The Attempted Fix

### What Was Tried

Created `merge_stable.py` that changed the wavefunction formation from:
```python
# Original (correct):
theta = psi_left · diag(V) · psi_right  # where V = 1/Lambda
```

To:
```python
# Attempted "fix" (WRONG):
theta = psi_left · diag(Lambda) · psi_right  # where Lambda = S from SVD
```

### Why This Was Attempted

**Root Cause (Real)**:
- When singular values Lambda → 0, V = 1/Lambda → ∞
- Large V values amplify numerical errors
- This causes spurious eigenvalues in H_eff optimization

**Naive Solution (Wrong)**:
- "Just use Lambda instead of V to avoid division"
- Seemed logical: small Lambda won't amplify errors
- **But**: Completely changes the wavefunction being formed

---

## Why This Failed

### Mathematical Error

**V = Lambda^-1 is not optional** - it's part of the MPS canonical form:

1. At boundaries, ranks have independently evolved tensors in specific gauges
2. The V matrix bridges these gauges correctly
3. Changing V → Lambda means multiplying by Lambda instead of 1/Lambda
4. **These are inverses** - you get a completely different wavefunction

### Numerical Evidence

**Direct Merge Test** (synthetic boundary data):
```
Configuration:
- Lambda = [1.0, 0.8, 0.5, 0.2, 0.1, 0.05, 0.01, 0.001]
- V condition number: 1e3

Results:
  Original merge:  E = +3.226827...
  Stable merge:    E = -6.632697...
  DIFFERENCE:      ΔE = 9.860

Tensor differences:
  ||A_left_new - A_left_old||  = 3.801
  ||A_right_new - A_right_old|| = 4.659e4  ← MASSIVE!
```

The energies have **opposite signs** and differ by nearly 10. This is not a numerical precision issue - it's a fundamental algorithmic error.

---

## What We Learned

### The Problem is Real

The V = 1/Lambda conditioning issue is **real and documented**:
- Line 807 of `pdmrg/pdmrg/dmrg.py`: `skip_opt = True  # Always skip until H_eff bug is fixed`
- Comment line 732-734: "H_eff eigensolver that produces spurious eigenvalues in certain gauge configurations"

### But the Naive Fix is Wrong

You **cannot** simply remove or change the V multiplication:
1. V is mathematically required by the MPS representation
2. The boundary tensors are in specific gauges
3. V = Lambda^-1 is the correct gauge bridge

### Current Safe Path

**PDMRG with `skip_opt=True` works correctly**:
- Produces correct energies (validated vs quimb DMRG2)
- No spurious eigenvalues (bypasses eigensolver)
- Trade-off: Slower convergence (more sweeps needed)

---

## What Might Actually Work

### Option 1: Tikhonov Regularization (Conservative)

Instead of hard clipping:
```python
# Current (unstable):
V = 1.0 / np.clip(S, 1e-12, None)

# Tikhonov (smoother):
lambda_reg = 1e-10
V = S / (S**2 + lambda_reg**2)
```

**Pros**: Smooth handling of small S, no sharp cutoff
**Cons**: Still changes V values, needs tuning
**Risk**: Medium - preserves general behavior

### Option 2: Gauge Synchronization (Moderate)

Before merge:
1. Explicitly canonize both boundary tensors to consistent gauge
2. Recompute V from the canonized tensors
3. Merge with corrected V

**Pros**: Addresses gauge inconsistency directly
**Cons**: Requires cross-rank communication, complex
**Risk**: Medium-High - changes algorithm flow

### Option 3: Alternative Formulation (Aggressive)

Study the DMRG literature for:
- Alternative boundary merge schemes
- Gauge-independent formulations
- Exact SVD approaches that don't use V at all

**Pros**: Could fundamentally solve the issue
**Cons**: Requires deep theory, major rewrite
**Risk**: High - unknown unknowns

### Option 4: Accept Current State (Pragmatic)

Keep `skip_opt=True` as the production path:
- Results are **correct** (validated)
- Performance is **acceptable** (just slower convergence)
- Focus on other improvements (benchmarking, A2DMRG, etc.)

**Pros**: No risk, move forward on known-good code
**Cons**: Leaves potential performance on table
**Risk**: Zero

---

## Recommendations

### Immediate (Do Now)

1. ✅ **Revert stable merge code** - Already done
2. ✅ **Keep benchmark infrastructure** - Valid and working
3. ✅ **Document failure** - This document
4. ⏳ **Update PDMRG_STATUS.md** - Note attempted fix and why it failed

### Short-Term (Next Steps)

1. **Complete benchmark data generation** - Use current PDMRG with `skip_opt=True`
2. **Create correctness benchmarks** - Validate all implementations vs golden references
3. **Measure performance** - Quantify actual impact of `skip_opt=True`
4. **Continue with A2DMRG** - Work on metadata and benchmarking

### Long-Term (Research)

1. **Study Stoudenmire et al. papers** - Understand boundary merge theory deeply
2. **Test Tikhonov regularization** - Conservative improvement worth trying
3. **Consult DMRG experts** - If available, get expert guidance
4. **Consider alternatives** - Is there a better parallel DMRG algorithm?

---

## Validation Results (Preserved)

### Golden References (Valid)

**Heisenberg L=12, D=20**:
- quimb DMRG2: E = -5.142090628178122 ✓

**Josephson L=20, D=50, n_max=2**:
- quimb DMRG2: E = -7.839066448948966 ✓

These golden references are **correct and validated** (reproduced to machine precision).

### Test Files (Valid)

**`tests/validate_stable_merge_serial.py`**:
- Correctly caught the algorithmic error
- Useful for future merge validation
- Keep for testing any future fixes

---

## Files Status

### Removed (Invalid)
- `pdmrg/pdmrg/parallel/merge_stable.py` ❌
- `pdmrg2/pdmrg/parallel/merge_stable.py` ❌

### Reverted (Back to Working State)
- `pdmrg/pdmrg/dmrg.py` ✓ (removed `use_stable_merge` flag)
- `pdmrg2/pdmrg/dmrg.py` ✓ (reverted to Phase 3)

### Kept (Valid)
- `benchmark_data_loader.py` ✓
- `scripts/generate_benchmark_data.py` ✓
- `benchmark_data/heisenberg/L12_D20/` ✓
- `benchmark_data/josephson/L20_D50_nmax2/` ✓
- `tests/validate_stable_merge_serial.py` ✓ (useful for future)

---

## Conclusion

**The stable merge fix attempt failed validation.** The naive approach of replacing V with Lambda is mathematically incorrect and produces wrong results.

**Current PDMRG with `skip_opt=True` remains the correct, safe path.** It works, produces correct results, and should be used for all production benchmarking until a correct fix is found.

**Future work** on boundary merge stability requires deeper theoretical understanding than currently available. Conservative approaches like Tikhonov regularization may be worth exploring, but fundamental algorithm changes should not be attempted without expert guidance.

**Lesson learned**: Numerical stability fixes must preserve mathematical correctness. Testing caught this error before it was deployed - validation works!

---

**Report Status**: Final
**Recommendation**: Proceed with current PDMRG (skip_opt=True) for benchmark suite completion
