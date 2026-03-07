# Fix A2DMRG Bond Dimension Collapse Bug

**Task**: Fix the critical bond dimension collapse bug in A2DMRG's i-orthogonal transformation.

**Priority**: 🔴 P0 - BLOCKS CORRECTNESS

---

## Current Situation

A comprehensive adversarial audit against `refs/a2dmrg.pdf` revealed that the recent Opus agent's i-orthogonal transformation fix is **INCOMPLETE**. The implementation correctly adds the transformation but **fails to preserve bond dimensions**, which breaks the algorithm's additive structure.

### Audit Results:
- **80 tests total**: 47 passed, 33 failed (41% failure rate)
- **Critical finding**: Bond dimensions collapse during `canonize()`
- **Status**: Algorithm is BROKEN but FIXABLE

---

## The Bug: Bond Dimension Collapse

### Location:
`a2dmrg/a2dmrg/numerics/local_microstep.py:31-94`

### Current Implementation:
```python
def _transform_to_i_orthogonal(mps, center_site, normalize=True):
    """Transform MPS to i-orthogonal form..."""
    mps.canonize(where=center_site)  # ← BUG: This truncates bonds!
    if normalize:
        mps /= mps.norm()
    return mps
```

### Evidence of Bug:

Test case: `L=8, bond_dim=20`

```
BEFORE canonize(where=4):
  Site 0: (2, 2)
  Site 1: (2, 4, 2)
  Site 2: (4, 8, 2)
  Site 3: (8, 16, 2)
  Site 4: (16, 20, 2)  ← Target bond_dim = 20
  Site 5: (20, 20, 2)
  Site 6: (20, 20, 2)
  Site 7: (20, 2)

AFTER canonize(where=4):
  Site 0: (2, 2)
  Site 1: (2, 4, 2)
  Site 2: (4, 8, 2)
  Site 3: (8, 16, 2)
  Site 4: (16, 8, 2)   ← COLLAPSED: 20 → 8
  Site 5: (8, 4, 2)    ← COLLAPSED: 20 → 8 → 4
  Site 6: (4, 2, 2)    ← COLLAPSED: 20 → 4 → 2
  Site 7: (2, 2)       ← COLLAPSED: 20 → 2
```

**Bond dimensions cascaded from 20 down to 2!**

### Why This Happens:

quimb's `canonize()` performs exact QR/LQ decompositions, which truncate near-zero singular values for random or low-rank MPSs. This reduces bond dimensions to the numerical rank.

### Why This Breaks A2DMRG:

From **Algorithm 2** (page 10 of `refs/a2dmrg.pdf`):

1. **Step 2**: Each parallel worker creates U^{(n+1),i} with **same rank r**
2. **Step 3**: Coarse matrices H[i,j] = ⟨Y^(i), H Y^(j)⟩ assume **uniform bond structure**
3. **Step 4**: Linear combination Σ c_j U^{(n+1),j} requires **matching bond dimensions**

**With bond collapse**:
- Worker 1 produces MPS with bonds (2, 4, 8, ...)
- Worker 2 produces MPS with bonds (2, 4, 8, 16, ...)
- Worker 3 produces MPS with bonds (2, 4, 8, 16, 20, ...)

**CANNOT form linear combination - dimensions don't match!**

### Paper Requirement:

**Algorithm 2, Step 1 (page 10):**
> "Note that this is a gauge transformation **without bond compression**"

**Definition 6 (page 6):**
> The transformation is performed by QR and LQ decompositions which **preserve the bond dimensions exactly**.

---

## The Fix: Zero-Padding After Canonization

### Good News:

The codebase **already has** the necessary padding functions in `a2dmrg/a2dmrg/mps/canonical.py:185-271`:

```python
def pad_to_uniform_bond_dimension(mps, target_bond_dim):
    """
    Pad all bond dimensions to target_bond_dim using zero-padding.

    Zero-padding preserves the state mathematically (the padded dimensions
    represent empty subspace). This is needed for A2DMRG to ensure
    all candidate MPS have the same bond structure.
    """
    # ... implementation exists
```

**The Opus agent just didn't use it!**

### Required Implementation:

Replace `_transform_to_i_orthogonal()` in `a2dmrg/a2dmrg/numerics/local_microstep.py` with:

```python
def _transform_to_i_orthogonal(mps, center_site, normalize=True):
    """
    Transform MPS to i-orthogonal canonical form WITHOUT changing bond dimensions.

    This implements Definition 6 (page 6) from Grigori & Hassan (2025).

    CRITICAL: Bond dimensions must be preserved for A2DMRG's additive structure.
    Algorithm 2, Step 1 states this is a "gauge transformation without bond
    compression". We achieve this by:
    1. Storing original bond dimensions before canonization
    2. Calling quimb's canonize() (which may reduce bonds to numerical rank)
    3. Zero-padding back to original dimensions

    Zero-padding is mathematically valid because the padded dimensions
    represent empty subspace that doesn't affect the physical state.
    """
    L = mps.L
    if not (0 <= center_site < L):
        raise ValueError(f"center_site={center_site} out of range for MPS with L={L}")

    # Store original bond dimensions BEFORE canonization
    original_shapes = []
    for i in range(L):
        original_shapes.append(mps[i].data.shape)

    # Perform gauge transformation (may reduce bonds to numerical rank)
    mps.canonize(where=center_site)

    # CRITICAL FIX: Pad back to original bond dimensions
    # This ensures all candidate MPS have the same bond structure
    for i in range(L):
        current_shape = mps[i].data.shape
        target_shape = original_shapes[i]

        if current_shape != target_shape:
            # Bond dimensions changed - need to pad with zeros
            padded = np.zeros(target_shape, dtype=mps[i].data.dtype)

            # Copy actual data into top-left corner of padded array
            slices = tuple(slice(0, s) for s in current_shape)
            padded[slices] = mps[i].data

            # Replace tensor data
            mps[i].modify(data=padded)

    # Normalize if requested
    if normalize:
        norm = mps.norm()
        if abs(norm) > 1e-15:
            mps /= norm

    return mps
```

**Key points:**
1. Store shapes BEFORE `canonize()`
2. Call `canonize()` (performs exact QR/LQ)
3. Detect any shape changes
4. Pad with zeros to restore original dimensions
5. Zero-padding is valid because it represents empty subspace

---

## Validation Tests

### Test 1: Bond Dimensions Preserved

```python
def test_bond_dimensions_preserved():
    """Verify _transform_to_i_orthogonal preserves bond dimensions."""
    import sys
    sys.path.insert(0, 'a2dmrg')

    from a2dmrg.mps.mps_utils import create_random_mps
    from a2dmrg.numerics.local_microstep import _transform_to_i_orthogonal

    L = 8
    bond_dim = 20

    mps = create_random_mps(L, bond_dim=bond_dim, phys_dim=2)

    # Store original bond dimensions
    bonds_before = [mps[i].data.shape for i in range(L)]

    # Transform to i-orthogonal
    for center in range(L):
        mps_copy = mps.copy()
        _transform_to_i_orthogonal(mps_copy, center_site=center, normalize=True)

        bonds_after = [mps_copy[i].data.shape for i in range(L)]

        # CRITICAL: Bond dimensions must be EXACTLY the same
        assert bonds_before == bonds_after, \
            f"center={center}: bonds changed from {bonds_before} to {bonds_after}"

    print("✅ Bond dimensions preserved for all centers")
```

### Test 2: I-Orthogonal Form Achieved

```python
def test_i_orthogonal_correctness():
    """Verify the transformation creates valid i-orthogonal form."""
    import numpy as np
    from a2dmrg.mps.mps_utils import create_random_mps
    from a2dmrg.numerics.local_microstep import _transform_to_i_orthogonal

    def check_left_orthogonal(tensor, site, L):
        """Check if tensor is left-orthogonal: (U^{<2>})^T U^{<2>} = I"""
        if tensor.ndim == 2 and site == 0:
            # First site: (phys, right) or (right, phys)
            # Find which is which and reshape to (something, right_bond)
            mat = tensor.T if tensor.shape[0] == 2 else tensor
        elif tensor.ndim == 3:
            # Middle site: (left, phys, right) -> (left, phys*right)
            mat = tensor.reshape(tensor.shape[0], -1)
        else:
            return True  # Last site, not left-orthogonal

        gram = mat.T @ mat
        identity = np.eye(gram.shape[0])
        error = np.linalg.norm(gram - identity)
        return error < 1e-10

    def check_right_orthogonal(tensor, site, L):
        """Check if tensor is right-orthogonal: U^{<1>} (U^{<1>})^T = I"""
        if tensor.ndim == 2 and site == L-1:
            # Last site: (left, phys) or (phys, left)
            mat = tensor.T if tensor.shape[1] == 2 else tensor
        elif tensor.ndim == 3:
            # Middle site: (left, phys, right) -> (left*phys, right)
            mat = tensor.reshape(-1, tensor.shape[-1])
        else:
            return True  # First site, not right-orthogonal

        gram = mat @ mat.T
        identity = np.eye(gram.shape[0])
        error = np.linalg.norm(gram - identity)
        return error < 1e-10

    L = 8
    mps = create_random_mps(L, bond_dim=16, phys_dim=2)

    for center in range(L):
        mps_copy = mps.copy()
        _transform_to_i_orthogonal(mps_copy, center_site=center, normalize=True)

        # Check left-orthogonal sites (j < center)
        for j in range(center):
            tensor = mps_copy[j].data
            assert check_left_orthogonal(tensor, j, L), \
                f"Site {j} not left-orthogonal (center={center})"

        # Check right-orthogonal sites (k > center)
        for k in range(center+1, L):
            tensor = mps_copy[k].data
            assert check_right_orthogonal(tensor, k, L), \
                f"Site {k} not right-orthogonal (center={center})"

    print("✅ I-orthogonal form correctly achieved")
```

### Test 3: Energy Preserved (Gauge Invariance)

```python
def test_energy_preserved():
    """Verify gauge transformation doesn't change energy (unitary)."""
    from a2dmrg.mps.mps_utils import create_random_mps
    from a2dmrg.hamiltonians.heisenberg import heisenberg_mpo
    from a2dmrg.numerics.observables import compute_energy
    from a2dmrg.numerics.local_microstep import _transform_to_i_orthogonal

    L = 8
    mps = create_random_mps(L, bond_dim=16, phys_dim=2)
    mpo = heisenberg_mpo(L, J=1.0, cyclic=False)

    E_original = compute_energy(mps, mpo, normalize=True)

    for center in range(L):
        mps_copy = mps.copy()
        _transform_to_i_orthogonal(mps_copy, center_site=center, normalize=True)
        E_after = compute_energy(mps_copy, mpo, normalize=True)

        error = abs(E_after - E_original)
        assert error < 1e-12, \
            f"Energy changed by {error:.2e} at center={center}"

    print("✅ Energy preserved (gauge invariance)")
```

### Test 4: Full Algorithm Integration

```python
def test_full_algorithm_integration():
    """Run full A2DMRG and verify results."""
    from a2dmrg.dmrg import a2dmrg_main
    from a2dmrg.hamiltonians.heisenberg import heisenberg_mpo
    from a2dmrg.mpi_compat import MPI
    from quimb.tensor import DMRG2

    L = 12
    bond_dim = 64
    mpo = heisenberg_mpo(L, J=1.0, cyclic=False)

    # Run quimb reference
    dmrg = DMRG2(mpo, bond_dims=bond_dim)
    dmrg.solve(tol=1e-10, verbosity=0, max_sweeps=20)
    E_quimb = dmrg.energy

    # Run A2DMRG with fixed implementation
    E_a2dmrg, mps = a2dmrg_main(
        L, mpo, bond_dim=bond_dim, tol=1e-10,
        max_sweeps=10, warmup_sweeps=2, verbose=False,
        comm=MPI.COMM_WORLD
    )

    error = abs(E_a2dmrg - E_quimb)

    # Should achieve machine precision or at least acceptance
    assert error < 5e-10, f"Error {error:.2e} exceeds acceptance threshold"

    if error < 1e-12:
        print(f"✅ Full algorithm test: MACHINE PRECISION (error={error:.2e})")
    else:
        print(f"✅ Full algorithm test: ACCEPTANCE (error={error:.2e})")

    return error
```

---

## Implementation Steps

### Step 1: Update `_transform_to_i_orthogonal()`

File: `a2dmrg/a2dmrg/numerics/local_microstep.py:31-94`

1. Add `import numpy as np` if not already present
2. Store original shapes before `canonize()`
3. Add padding loop after `canonize()`
4. Update docstring to explain the fix

### Step 2: Add Validation Tests

Create file: `a2dmrg/a2dmrg/tests/test_bond_dimension_preservation.py`

Add all four test functions above plus:
- Edge cases: L=2, L=3
- Different bond dimensions: 8, 16, 32, 64
- Verify padding with zeros doesn't affect energy

### Step 3: Run Comprehensive Audit

```bash
# Set single-threaded BLAS for reproducibility
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Run the fixed audit
python3 audit_a2dmrg_comprehensive.py
```

**Expected results after fix:**
- Step 1 tests: 90%+ pass rate (up from 51%)
- Step 2 tests: 90%+ pass rate (up from 67%)
- Step 3 tests: 100% pass (already passing)
- Overall: 75+ tests passing out of 80

### Step 4: Performance Benchmark

Compare before/after on L=8, L=12 systems:
- Accuracy should improve to <1e-12
- Time should be similar or slightly faster

---

## Acceptance Criteria

✅ **Implementation complete when:**

1. Bond dimensions preserved: All sites maintain original shapes
2. I-orthogonal form achieved: Left/right orthogonality conditions satisfied
3. Energy preserved: Gauge transformation is unitary (error < 1e-12)
4. Full algorithm works: A2DMRG achieves <5e-10 error vs quimb
5. All tests pass: At least 75/80 tests in comprehensive audit
6. No performance regression: Time within 20% of original

✅ **Documentation complete when:**

1. Docstring explains why padding is needed (Algorithm 2 requirement)
2. Comments link to Definition 6 and Algorithm 2, Step 1
3. Test file has clear descriptions of what each test validates

✅ **Validation complete when:**

1. `test_bond_dimensions_preserved()` passes for L=4,8,12
2. `test_i_orthogonal_correctness()` passes for all centers
3. `test_energy_preserved()` shows <1e-12 error
4. `test_full_algorithm_integration()` shows <5e-10 error
5. Comprehensive audit passes with 75+ tests

---

## Additional Context

### Existing Code to Leverage:

**Option 1**: Inline padding (recommended for simplicity)
```python
# Just do it directly in _transform_to_i_orthogonal()
```

**Option 2**: Use existing function
```python
from ..mps.canonical import pad_to_uniform_bond_dimension

# After canonize()
max_bond = max(original_shapes[i][...] for i in range(L))
pad_to_uniform_bond_dimension(mps, max_bond)
```

**Recommendation**: Use Option 1 (inline) because:
- Simpler and more explicit
- No need to compute max bond dim
- Restores EXACT original shapes, not just uniform padding

### Why Zero-Padding is Valid:

From quantum mechanics: The MPS represents a state |ψ⟩ in Hilbert space. Adding zero-filled dimensions is equivalent to embedding |ψ⟩ in a larger space |ψ⟩ ⊗ |0⟩, which doesn't change expectation values:

⟨ψ ⊗ 0 | H | ψ ⊗ 0⟩ = ⟨ψ | H | ψ⟩

The padding gets removed during compression (Step 4), so it's purely a bookkeeping device to maintain uniform structure.

### Known Issues to Avoid:

1. **Don't use `pad_to_bond_dimensions()` with wrong dimensions**
   - Must restore EXACT original shapes
   - Not just uniform padding to max bond dim

2. **Don't skip normalization**
   - The norm accumulates at the center site
   - Must normalize after transformation

3. **Don't forget edge sites**
   - Sites 0 and L-1 have 2D tensors
   - Padding logic must handle this correctly

---

## References

- **refs/a2dmrg.pdf**: Algorithm 2 (page 10), Definition 6 (page 6), Lemma 10 (page 8)
- **A2DMRG_AUDIT_REPORT.md**: Full audit findings and evidence
- **audit_a2dmrg_comprehensive.py**: Test suite exposing the bug
- **a2dmrg/a2dmrg/mps/canonical.py:185-271**: Existing padding functions

---

## Success Metrics

After implementing this fix, the audit should show:

| Metric | Before Fix | After Fix | Target |
|--------|------------|-----------|---------|
| Total tests passing | 47/80 (59%) | 75+/80 | 90%+ |
| Step 1 (orthogonalization) | 26/51 (51%) | 45+/51 | 90%+ |
| Step 2 (microsteps) | 16/24 (67%) | 22+/24 | 90%+ |
| Step 3 (coarse space) | 5/5 (100%) | 5/5 | 100% |
| L=12 accuracy vs quimb | 3.7e-08 | <1e-12 | <5e-10 |

---

## Timeline

**Estimated effort**: 2-4 hours

1. Implement fix: 30 minutes
2. Add tests: 1 hour
3. Run validation: 1 hour
4. Debug/iterate: 30-60 minutes
5. Documentation: 30 minutes

---

## Next Steps After Fix

Once this is working:

1. **Remove energy recomputation** (Bug #3): Return eigenvalue directly from H_eff
2. **Optimize MPI communication**: Send only updated TT cores, not full MPS
3. **Re-enable global canonicalization**: Now safe with bond preservation
4. **Full parallel scaling tests**: Test np=2,4,8 with MPI

---

**Start here**: Update `_transform_to_i_orthogonal()` in `local_microstep.py` to preserve bond dimensions by zero-padding after `canonize()`.

Good luck! This is a straightforward fix that will make A2DMRG algorithmically correct.
