# A2DMRG Implementation Fix Prompt

**Task**: Fix the A2DMRG (Additive Two-Level DMRG) implementation to correctly follow the algorithm described in `refs/a2dmrg.pdf` paper by Laura Grigori and Muhammad Hassan.

## Current Status: BROKEN

The A2DMRG implementation is currently non-functional with critical algorithm deviations:

### Critical Issues Identified:

1. **MISSING i-ORTHOGONAL TRANSFORMATION** (CRITICAL - Core Algorithm Requirement)
   - Location: `a2dmrg/a2dmrg/numerics/local_microstep.py` lines 70 and 253
   - Current code: `mps_updated = mps.copy()` with TODO comment
   - Impact: The algorithm is not actually A2DMRG without proper gauge transformation

2. **Performance Symptoms** (Evidence of broken implementation):
   - Negative parallel scaling (gets SLOWER with more processes)
   - Poor accuracy: 5-7e-09 errors on L=48 (fails acceptance threshold of 5e-10)
   - All A2DMRG tests timeout or produce unacceptable results
   - Expected: Competitive with PDMRG/PDMRG2 on correctness, better parallel scaling

3. **Test Results** (from `benchmarks/correctness_results.json`):
   - A2DMRG np=2,4,8: All timeout or fail accuracy thresholds
   - PDMRG/PDMRG2: 37/52 tests pass at tol=1e-11 (9 machine precision, 15 acceptance)
   - Conclusion: A2DMRG fundamentally broken, not just poorly tuned

---

## Reference Paper: `refs/a2dmrg.pdf`

**Full Title**: "An Additive Two-Level Parallel Variant of the DMRG Algorithm with Coarse-Space Correction"

**Authors**: Laura Grigori and Muhammad Hassan

**Key Sections**:
- **Section 2**: Problem Formulation and Setting
  - Definition 6 (page 6): **i-orthogonal tensor train decomposition**
  - Definition 7 (page 7): Retraction operators for DMRG
  - Lemma 8 (page 7): Orthogonality of retraction operators

- **Section 3**: Additive Two-Level DMRG Algorithm
  - **Algorithm 2 (page 10)**: THE CANONICAL A2DMRG ALGORITHM
  - Section 3.1 (page 13): Second-level minimization details
  - Section 3.2 (page 15): Application to quantum chemistry

---

## Algorithm 2 Breakdown (Page 10 of PDF)

The A2DMRG algorithm consists of 4 core steps per global iteration:

### Step 1: Orthogonalization Sweep (RIGHT-TO-LEFT)
```
for i = d, d-1, ..., 2 do
    Given an i-orthogonal TT decomposition U^(n),i
    Perform orthogonalization step:
        Compute W_i^{<i>} = L_i Q_i^{<i>} with L_i, Q_i
    Save (i-1)-orthogonal TT decomposition U^(n),i-1
end for
```

**CRITICAL**: This creates a sequence of i-orthogonal gauges. Each site must be in the proper gauge before optimization.

**Definition 6 (i-orthogonal, page 6)**:
- A TT decomposition U = (U_1, ..., U_d) is **i-orthogonal** if:
  - `(U_j^{<2>})^T U_j^{<2>} = I` for all j ∈ {1, ..., i-1} (left-orthogonal)
  - `U_k^{<1>} (U_k^{<1>})^T = I` for all k ∈ {i+1, ..., d} (right-orthogonal)
- The site i is the **orthogonality center** where gauge freedom resides

### Step 2: Parallel Local Micro-Steps (INDEPENDENT)
```
for i = 1, 2, ..., d (or i = 1, 2, ..., d-1 for two-site) IN PARALLEL do
    Perform one-site (or two-site) DMRG micro-step on i-orthogonal form
    Compute V_i = S_i(U^(n0),i) or W_{i,i+1} = S_{i,i+1}(U^(n0),i)
    Define updated TT decomposition U^(n+1),i using V_i or W_{i,i+1}
end for
```

**Key point**: Each processor works on a different orthogonality center SIMULTANEOUSLY. This is why proper i-orthogonal form is essential.

### Step 3: Second-Level Minimization (COARSE SPACE)
```
Compute minimizer {c_j*}_{j=0}^d ∈ R^{d+1} of

    min_{c_j} g(∑_{j=0}^d c_j τ(Ũ^(n+1),j))    (One-Site)

or for two-site:

    min_{d_j} g(∑_{j=0}^{d-1} d_j τ(Ũ^(n+1),j))    (Two-Site)
```

This step combines the d parallel local solutions using optimal linear coefficients.

### Step 4: Compression to Low-Rank Manifold
```
Apply compression to obtain left-orthogonal approximation:
    U_r̃ ∋ U^(n+1),d ≈ ∑_{j=0}^d c_j* Ũ^(n+1),j    (One-Site)

or:
    U_r̃ ∋ U^(n+1),d ≈ ∑_{j=0}^{d-1} d_j* Ũ^(n+1),j    (Two-Site)

for rank parameter r̃ ≥ r.
```

Uses TT-rounding (Oseledets) to project back to target rank.

---

## Files Requiring Fixes

### PRIMARY TARGET: `a2dmrg/a2dmrg/numerics/local_microstep.py`

**Function 1**: `local_microstep_1site` (lines 22-192)
- **Line 70**: Replace stub with proper i-orthogonal transformation
- **Current code**:
  ```python
  # TODO: Implement proper i-orthogonal transformation without bond compression
  mps_updated = mps.copy()
  ```
- **Required**: Transform MPS to i-orthogonal form centered at `site`
- **Constraint**: Do NOT change bond dimensions (no compression during gauge transformation)

**Function 2**: `local_microstep_2site` (lines 195-392)
- **Line 253**: Same issue - missing i-orthogonal transformation
- **Required**: Transform MPS to i-orthogonal form centered at sites `(site, site+1)`

### SECONDARY FILES (May need updates):

1. **`a2dmrg/a2dmrg/mps/canonical.py`**
   - May already have gauge transformation utilities
   - Check for existing `move_orthogonality_center`, `canonicalize`, etc.
   - Ensure no bond compression during gauge moves

2. **`a2dmrg/a2dmrg/dmrg.py`**
   - Lines 1-100: Main A2DMRG loop implementation
   - Verify Step 1 orthogonalization sweep is correct
   - Verify Step 3 second-level minimization logic

3. **`a2dmrg/a2dmrg/numerics/observables.py`**
   - Energy computation should work on any gauge
   - Verify normalization handling

---

## Implementation Requirements

### 1. i-Orthogonal Gauge Transformation (CRITICAL)

**What to implement**:
```python
def transform_to_i_orthogonal(mps: qtn.MatrixProductState, center: int) -> qtn.MatrixProductState:
    """
    Transform MPS to i-orthogonal canonical form with orthogonality center at `center`.

    Definition (from paper, page 6):
    - Sites j < center: left-orthogonal (U_j^{<2>})^T U_j^{<2>} = I
    - Sites k > center: right-orthogonal U_k^{<1>} (U_k^{<1>})^T = I
    - Site i = center: non-orthogonal (gauge freedom)

    CRITICAL: Must NOT change bond dimensions (no truncation/compression).
    Use QR decomposition for left-orthogonalization, LQ for right-orthogonalization.

    Parameters
    ----------
    mps : quimb.MatrixProductState
        Input MPS in any gauge
    center : int
        Site index to place orthogonality center (0-indexed)

    Returns
    -------
    mps_i_orth : quimb.MatrixProductState
        MPS in i-orthogonal form centered at `center`
    """
```

**Algorithm**:
1. Sweep LEFT (from site 0 to center-1):
   - Apply QR decomposition to make each site left-orthogonal
   - Contract R matrix into next site to the right
   - Preserve bond dimensions exactly

2. Sweep RIGHT (from site L-1 to center+1):
   - Apply LQ decomposition to make each site right-orthogonal
   - Contract L matrix into next site to the left
   - Preserve bond dimensions exactly

3. Result: Site `center` has full gauge freedom, all others orthogonal

**Reference implementation** (check if quimb has built-in):
- quimb may have `mps.canonicalize(where=center)` or similar
- Verify it doesn't truncate bonds (set max_bond very high or disable)
- If quimb's method truncates, implement custom QR/LQ sweeps

### 2. Two-Site i-Orthogonal Form

For two-site updates, the orthogonality center spans sites `(i, i+1)`:
- Sites j < i: left-orthogonal
- Sites k > i+1: right-orthogonal
- Sites i and i+1: form a two-site tensor with gauge freedom

**Note**: This is similar to standard two-site DMRG gauge, but must be verified.

### 3. Parallel Correctness

Each parallel worker in Step 2 must:
- Start with the SAME base MPS `U^(n),d` from Step 1
- Transform to DIFFERENT i-orthogonal centers (worker i → center at site i)
- Optimize independently (no communication)
- Return updated TT core(s) for second-level minimization

Current implementation may already handle this correctly in `dmrg.py` main loop.

### 4. Validation Checks

Add assertions to verify i-orthogonal form:
```python
# Check left-orthogonality for j < center
for j in range(center):
    assert np.allclose(U_j.T @ U_j, np.eye(bond_dim)), f"Site {j} not left-orthogonal"

# Check right-orthogonality for k > center
for k in range(center+1, L):
    assert np.allclose(U_k @ U_k.T, np.eye(bond_dim)), f"Site {k} not right-orthogonal"
```

---

## Expected Outcomes After Fix

### Correctness:
- A2DMRG tests should achieve similar accuracy to PDMRG/PDMRG2
- Target: <5e-10 error on Heisenberg L=12,32,48 (acceptance threshold)
- No timeouts on standard test suite (max_sweeps=40, tol=1e-11)

### Performance:
- **Positive parallel scaling**: np=4 should be faster than np=2
- Expected speedup: 1.5-3× at np=4, 2-5× at np=8 (depends on system size)
- Competitive cost per iteration vs classical DMRG (Table 1, page 12 of paper)

### Verification:
1. Run `benchmarks/correctness_suite.py` with A2DMRG enabled
2. Compare results to `benchmarks/correctness_results.json` baseline
3. Check for:
   - `"success": true` for A2DMRG runs
   - Energy errors < 5e-10 (acceptance) or < 1e-12 (machine precision)
   - Reasonable iteration counts (should converge in <40 sweeps)
   - Speedup metrics show positive scaling

---

## Artifacts to Review

After implementing fixes, provide:

1. **Updated Files**:
   - `a2dmrg/a2dmrg/numerics/local_microstep.py` (with i-orthogonal transformation)
   - Any helper functions added to `mps/canonical.py`
   - Updated `dmrg.py` if main loop needed changes

2. **Test Results**:
   - New `correctness_results_a2dmrg_fixed.json` showing passing tests
   - Performance comparison table (A2DMRG vs PDMRG/PDMRG2 vs quimb)
   - Parallel scaling data (np=1,2,4,8)

3. **Validation Report**:
   - Document showing i-orthogonal form is correctly implemented
   - Energy accuracy comparison with reference (quimb DMRG1)
   - Confirmation that paper's Algorithm 2 is faithfully implemented

4. **Documentation**:
   - Updated docstrings explaining i-orthogonal gauge transformation
   - Comments linking to specific equations/sections in refs/a2dmrg.pdf
   - Any deviations from paper (if needed) must be justified

---

## Testing Strategy

### Unit Tests (add if missing):
```python
def test_i_orthogonal_transformation():
    """Verify transform_to_i_orthogonal produces correct gauge."""
    mps = random_mps(L=10, chi=20, d=2)
    for center in range(L):
        mps_i = transform_to_i_orthogonal(mps, center)
        # Check left-orthogonality for j < center
        # Check right-orthogonality for k > center
        # Verify energy is preserved (gauge invariant)
```

### Integration Tests:
1. Run single A2DMRG iteration on Heisenberg L=12, verify:
   - Step 1 produces proper i-orthogonal sequence
   - Step 2 parallel updates are independent
   - Step 3 coarse-space combination reduces energy
   - Step 4 compression maintains accuracy

2. Full convergence test on small system (L=8, chi=32)

3. Scaling test: Compare np=1,2,4,8 on Heisenberg L=32

---

## Additional Context

### Related Issues (from CPU_AUDIT_REPORT.md):
- ✅ FIXED: V matrix shape bug (commit 0baebc6)
- ✅ FIXED: Hardcoded mpirun path (commit 0baebc6)
- ✅ FIXED: Josephson placeholder → NotImplementedError (commit 0baebc6)
- ⚠️ REMAINING: i-orthogonal transformation (THIS TASK)

### Branch: `cpu-audit`
All fixes should be committed to the `cpu-audit` branch (already checked out).

### Commit Message Template:
```
Implement i-orthogonal gauge transformation for A2DMRG

Fixes the core algorithmic deficiency in A2DMRG implementation.
Adds proper i-orthogonal canonical form transformation as required
by Algorithm 2 in refs/a2dmrg.pdf (Grigori & Hassan).

Changes:
- local_microstep.py: Replace MPS copy stub with i-orthogonal transformation
- [Any other files modified]

This enables correct parallel local micro-steps with proper gauge
conditions, fixing negative scaling and accuracy issues.

Refs: CPU_AUDIT_REPORT.md Critical Issue #4
Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```

---

## Questions to Resolve During Implementation

1. **Does quimb provide gauge transformation utilities?**
   - Check `quimb.tensor.MatrixProductState` methods
   - Look for `canonicalize`, `left_canonize`, `right_canonize`, etc.
   - Verify these don't truncate bonds

2. **How to handle boundary sites (i=0, i=L-1)?**
   - Site 0: Only right-orthogonalize sites to the right
   - Site L-1: Only left-orthogonalize sites to the left
   - Verify definitions in paper apply correctly

3. **Is the second-level minimization (Step 3) implemented correctly?**
   - Review `dmrg.py` main loop
   - Check if eigenvalue problem in Section 3.1 (page 13) is solved
   - Verify linear combination coefficients are computed

4. **Should we support particle number conservation?**
   - Paper Section 3.2 discusses this for quantum chemistry
   - Current code may have basic support - verify it works with i-orthogonal form

---

## Success Criteria

✅ **Implementation complete when**:
1. All TODOs removed from `local_microstep.py` lines 70, 253
2. i-orthogonal transformation correctly implemented (verified by unit tests)
3. A2DMRG tests pass with errors < 5e-10 (acceptance threshold)
4. Positive parallel scaling observed (np=4 faster than np=2)
5. No timeouts on correctness suite (converges within max_sweeps=40)
6. Code has clear docstrings linking to paper sections/equations
7. Commit message documents the fix with before/after comparison

✅ **Validation complete when**:
1. `correctness_suite.py` runs successfully for A2DMRG np=1,2,4,8
2. Performance table shows competitive accuracy vs PDMRG/PDMRG2
3. Memory usage reasonable (no unexpected bond dimension growth)
4. CPU_AUDIT_REPORT.md Critical Issue #4 can be marked as RESOLVED

---

## IMPORTANT NOTES

⚠️ **Do NOT**:
- Truncate/compress bonds during gauge transformation (violates algorithm)
- Change the public API of `local_microstep_1site` / `local_microstep_2site`
- Skip the i-orthogonal transformation "to make it work faster"
- Modify the reference paper PDF

⚠️ **DO**:
- Follow Algorithm 2 from the paper EXACTLY
- Add comprehensive comments explaining each step
- Link to specific equations/definitions from the paper
- Test thoroughly on small systems first (L=8, chi=16)
- Verify gauge invariance (energy should not change during transformation)

---

## Get Started

1. Read `refs/a2dmrg.pdf` Section 2 (pages 3-8) for tensor train background
2. Study Algorithm 2 (page 10) and Section 3.1 (pages 13-15) carefully
3. Review current `local_microstep.py` to understand existing structure
4. Check if quimb provides gauge transformation utilities
5. Implement `transform_to_i_orthogonal` function with unit tests
6. Update `local_microstep_1site` and `local_microstep_2site` to use it
7. Run tests and verify correctness
8. Document and commit

**Primary reference**: Algorithm 2 (page 10), Definition 6 (page 6)

Good luck! This is the critical fix that will make A2DMRG actually work.
